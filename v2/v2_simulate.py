"""
v2_simulate.py

The experiment driver. Edit the SETTINGS block, then run from inside v2/:

    python v2_simulate.py --region Magallanes --year 2023 --draws 100 --cores 20

It reads ../R/intermediate_data/<Region>/<Year>/instance.txt and writes to
../results/v2/<Region>_<Year>_<timestamp>/.

The region is read once per worker process. Each task is one (draw, method) pair,
so every core stays busy even when a few IP solves are slow. Lotteries are
resampled per draw with seed = SEED + draw, so every method sees the same lottery
on the same draw and a rerun reproduces the same numbers.

The run is resumable. Each finished row is appended to rows.jsonl. If the job
dies, point --out-dir at the same folder and it skips the pairs already on disk.

Every solved matching is checked against absolute contingent stability. Each row
records acs (0 or 1), n_blocking (distinct students in a blocking pair, counted
once each) and blocking_pct. This is done for every method, including the soft,
hybrid and partial ones: those formulations quantify over z, but the matching
they return is a concrete matching the checker can test.


FINDING A FEASIBLE POINT IS THE BOTTLENECK
------------------------------------------
On the large regions the exact IP is not hard to optimize. It is hard to make
feasible. The root relaxation takes 40 to 110 seconds, returns a bound several
hundred rank units below the optimum, and leaves over two thousand fractional
integer variables, because the tie-breaking rows are logical constraints that are
nearly vacuous in the relaxation and only bite at integrality. Branch and bound
then spends hours dividing the tree without reaching any integer point.

Gurobi's NoRel heuristic runs before the root relaxation and does not use it. It
searches the integer space directly. On the Atacama draws that had produced
nothing in four hours of branch and bound, it found a stable matching in 236 to
526 seconds, with zero nodes explored and zero simplex iterations.

So the IP methods here are staged:

  1. RA-DA. If it converges, its matching is stable and no solver is needed.
  2. Otherwise the feasibility IP: the same formulation with a zero objective,
     stopped at the first solution, with NoRel enabled.
  3. Then the rank IP, warm-started from whichever matching step 1 or 2 produced.

Step 3 closes at a single branch-and-bound node on most draws. The warm start
must be a STABLE matching: RA-DA's matching on a cycling draw is not stable, so
it is not feasible for the model, and Gurobi discards it as a MIP start. That is
why step 2 exists and why it is what step 3 starts from.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from v2_inputs import load_region, resample_lottery
import v2_exact as exact
import v2_heuristics as heuristics
import v2_metrics as metrics
import v2_rada as rada
import v2_stability as stability
import v2_tables as tables


# ==========================================================================
# SETTINGS
# ==========================================================================
R_ROOT = "../R"                 # <R_ROOT>/intermediate_data/<Region>/<Year>/
RESULTS_ROOT = "../results/v2"

REGION = "Magallanes"
YEAR = 2023
DRAWS = 100
TIE_BREAK = "mtbf"              # individual | family | mtbf
SEED = 0

METHODS = [
    "SOSM", "Descending", "Ascending", "LSDA", "SLDA",
    "RA-DA", "RADA-Paper", "RADA-Portfolio", "RADA-IP",
    "IP-Warm", "ACS-Feasible",
    "Absolute-Hard", "Absolute-Soft",
    "Partial-Hard", "Partial-Soft",
]

# Grids for the hybrid floor (sum of z >= zeta), per region. The floor scales
# with the number of students who have siblings, so each region needs its own
# range. To calibrate a new region, run once with an empty grid, read
# Absolute-Soft's eff_providers, and refine near the feasibility boundary.
# Students with siblings, 2023: Magallanes 1382, Atacama 3252, Lagos 6917,
# OHiggins 7736.
REGION_ZETAS = {
    "Magallanes": list(range(0, 401, 25)),
    "Atacama":    list(range(0, 1001, 50)),
    "Lagos":      list(range(0, 2001, 100)),
    "OHiggins":   list(range(0, 2201, 100)),
}
HYBRID_ZETAS_OVERRIDE: list = []   # non-empty overrides the per-region grid

IP_TIME_LIMIT = 3600            # seconds per IP solve
IP_MIP_GAP = 1e-3
IP_THREADS = 1                  # per solve; total cores = NUM_CORES
NUM_CORES = 20                  # 1 or less runs serially

# NoRelHeurTime for the feasibility solves, in seconds. This is the setting that
# makes the large regions tractable; see the note at the top of this file. Set it
# to 0 to disable, which is appropriate on small instances where it only delays
# the root relaxation.
FEAS_NO_REL_HEUR = 600.0

VERIFY_STABILITY = True
REQUIRE_COMMON = True           # separated-column semantics; see v2_metrics.py
ONE_PREF_BY = "assigned"

OUT_DIR = None                  # default: a fresh timestamped folder
# ==========================================================================


# ==========================================================================
# the staged IP
# ==========================================================================
def _feasible_ip(inst, no_rel=FEAS_NO_REL_HEUR):
    """The same formulation with a zero objective, stopped at the first solution.
    Returns some stable matching, not the rank-optimal one. This is what NoRel
    solves quickly, and its matching is a valid warm start for the rank IP."""
    return exact.solve(inst, "absolute", "hard", objective="feasible",
                       time_limit=IP_TIME_LIMIT, threads=IP_THREADS,
                       no_rel_heur_time=no_rel)


def _rank_ip(inst, warm=None):
    return exact.solve(inst, "absolute", "hard", time_limit=IP_TIME_LIMIT,
                       mip_gap=IP_MIP_GAP, threads=IP_THREADS, warm_start=warm)


def _ip_warm(inst, no_rel=FEAS_NO_REL_HEUR):
    """The exact IP, staged. info['via'] records the path taken:

      rada             RA-DA converged; its matching is stable and no solver ran
      ip               the rank IP returned a matching, warm-started or not
      feasible_only    the rank IP found nothing; the feasibility IP did. The
                       matching is stable but not proven rank-optimal
                       (rank_optimal is False)
      infeasible       the IP proved that no stable matching exists. The row is
                       left empty: filling it with a non-stable matching would
                       misreport the method's ACS rate
      no_solution      neither solve found anything inside the budget
    """
    # 1. RA-DA, which needs no solver at all when it converges
    mu, info = heuristics.rada(inst, "absolute")
    if info.get("converged"):
        return mu, {**info, "via": "rada", "rank_optimal": False}

    # 2. RA-DA's matching is NOT stable here, so it is infeasible for the model
    #    and Gurobi would discard it as a MIP start. Get a stable one from the
    #    feasibility IP instead.
    warm, feas = _feasible_ip(inst, no_rel)
    if warm is None and feas.get("status_str") == "INFEASIBLE":
        return None, {**feas, "via": "infeasible"}

    # 3. The rank IP, warm-started from a matching that is actually feasible for
    #    it. This closes at a single node on most draws.
    mu, ip = _rank_ip(inst, warm=warm)
    if mu is not None:
        return mu, {**ip, "via": "ip",
                    "rank_optimal": ip.get("status_str") == "OPTIMAL"}
    if ip.get("status_str") == "INFEASIBLE":
        return None, {**ip, "via": "infeasible"}
    if warm is not None:
        return warm, {**feas, "via": "feasible_only", "rank_optimal": False}
    return None, {**ip, "via": "no_solution"}


def _rada_then_ip(inst, no_rel=FEAS_NO_REL_HEUR):
    """RA-DA, escalating to the staged IP only on draws where it does not
    converge. Identical to IP-Warm on those draws; the difference is only that
    RADA-IP reports RA-DA's own iteration count when it converges."""
    return _ip_warm(inst, no_rel)


def _ip(priority, enforcement, zeta=None):
    def run(inst):
        return exact.solve(inst, priority, enforcement, zeta_min=zeta,
                           time_limit=IP_TIME_LIMIT, mip_gap=IP_MIP_GAP,
                           threads=IP_THREADS)
    return run


def method_registry(no_rel: float = FEAS_NO_REL_HEUR):
    return {
        "SOSM": lambda inst: (heuristics.deferred_acceptance(inst), None),
        "Descending": lambda inst: (heuristics.descending(inst, "absolute"), None),
        "Ascending": lambda inst: (heuristics.ascending(inst, "absolute"), None),
        "LSDA": lambda inst: (heuristics.lsda(inst), None),
        "SLDA": lambda inst: (heuristics.slda(inst), None),

        # RA-DA. RADA-Paper reaches the same fixed points and differs only in
        # what it reports on a cycle: the repeated matching, as the pseudocode
        # does, rather than the lowest-rank iterate seen.
        "RA-DA": lambda inst: heuristics.rada(inst, "absolute"),
        "RADA-Paper": lambda inst: rada.rada(inst),

        # Convergence-boosting variants. Portfolio restarts from four seeds and
        # is cheap. Sequential commits providers one at a time and costs one
        # deferred-acceptance run per provider per iteration. Search is a
        # depth-first search over provider subsequences, worst-case exponential,
        # bounded by node, sequence and time limits. Lottery raises the
        # tie-breaker instead of the priority group.
        "RADA-Portfolio": lambda inst: rada.rada_portfolio(inst),
        "RADA-Sequential": lambda inst: rada.rada_sequential(inst),
        "RADA-Search": _rada_search,
        "RADA-Lottery": lambda inst: rada.rada_lottery(inst),

        # RADA-IP escalates to the IP on draws that cycle. IP-Warm always runs
        # the IP. Both use the staged path above.
        "RADA-IP": lambda inst: _rada_then_ip(inst, no_rel),
        "IP-Warm": lambda inst: _ip_warm(inst, no_rel),

        # The feasibility IP alone: some stable matching, not the rank-optimal
        # one. Cheap, and it succeeds whenever a stable matching exists.
        "ACS-Feasible": lambda inst: _feasible_ip(inst, no_rel),

        "Absolute-Hard": _ip("absolute", "hard"),
        "Absolute-Soft": _ip("absolute", "soft"),
        "Partial-Hard": _ip("partial", "hard"),
        "Partial-Soft": _ip("partial", "soft"),
    }


def _rada_search(inst):
    """RADA with sequential search. If it finds a fixed point, use it. If it
    proves that every reachable sequence cycles, which is a real result, fall
    back to the DA matching for scoring. If it only ran out of budget, report no
    result rather than silently returning DA, so a timed-out row is not mistaken
    for RADA-Search agreeing with DA."""
    mu, info = rada.rada_sequential_search(inst)
    info["iters"] = info.get("nodes")
    if mu is None:
        if info.get("capped"):
            return None, info
        mu = heuristics.deferred_acceptance(inst)
        info["fell_back_to_da"] = True
    return mu, info


def _zeta_grid(region: str) -> list:
    if HYBRID_ZETAS_OVERRIDE:
        return HYBRID_ZETAS_OVERRIDE
    return REGION_ZETAS.get(region, [])


# Every solved matching is checked against ABSOLUTE contingent stability, so the
# ACS rate and the blocking count are defined for every method.
ACS_TARGET = "absolute"


def _is_heavy(method: str) -> bool:
    """IP methods run long, so they are scheduled first and the fast heuristics
    fill the gaps at the end. Hybrid is an IP, so the whole zeta grid counts."""
    return method.startswith(("Absolute", "Partial", "Hybrid")) \
        or method in ("RADA-Search", "RADA-IP", "RADA-Sequential", "IP-Warm",
                      "ACS-Feasible")


def _load_checkpoint(ckpt):
    """Read an existing rows.jsonl. Deduplicates by (draw, method), keeping the
    latest write, so a folder that accumulated duplicates from earlier crashed
    restarts is cleaned up on resume. Tolerates a half-written final line."""
    by_key = {}
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                by_key[(r.get("draw"), r.get("method"))] = r
    return list(by_key.values()), set(by_key)


# ==========================================================================
# workers
# ==========================================================================
_G: dict = {}


def _init_worker(cfg):
    _G["cfg"] = cfg
    _G["base"] = load_region(cfg["region"], cfg["year"], cfg["r_root"],
                             tie_break=cfg["tie_break"], seed=cfg["seed"],
                             verbose=False)
    _G["registry"] = method_registry(cfg["no_rel"])


def _run_one(task):
    draw, method = task
    cfg = _G["cfg"]
    inst = resample_lottery(_G["base"], seed=cfg["seed"] + draw,
                            tie_break=cfg["tie_break"])
    t0 = time.perf_counter()
    fn = _G["registry"].get(method)
    if fn is None and method.startswith("Hybrid-"):
        fn = _ip("absolute", "hybrid", int(method.split("-")[1]))
    mu, info = fn(inst)
    wall = time.perf_counter() - t0

    priority = ("partial" if method.startswith("Partial")
                else "absolute" if method.startswith(("Absolute", "Hybrid"))
                else None)
    row = metrics.evaluate(inst, mu, info, priority=priority,
                           require_common=cfg["require_common"],
                           one_pref_by=cfg["one_pref_by"])
    row["draw"], row["method"] = draw, method
    row.setdefault("runtime", wall)

    if cfg["verify"] and mu is not None:
        ok, blk = stability.is_contingent_stable(
            inst, mu, ACS_TARGET, return_blocking=True)
        row["acs"] = int(ok)
        # Each student that participates in a blocking pair is counted once,
        # however many schools they would block with. The incumbent they envy is
        # not counted.
        blockers = {t[1] for t in blk if t[0] in ("waste", "envy")}
        row["n_blocking"] = len(blockers)
        row["blocking_pct"] = 100.0 * len(blockers) / len(inst.students)
    return row


# ==========================================================================
# driver
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--year", default=YEAR)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--tie-break", default=TIE_BREAK)
    ap.add_argument("--cores", type=int, default=NUM_CORES)
    ap.add_argument("--r-root", default=R_ROOT)
    ap.add_argument("--results-root", default=RESULTS_ROOT)
    ap.add_argument("--no-rel-heur-time", type=float, default=FEAS_NO_REL_HEUR,
                    help="Gurobi NoRelHeurTime for the feasibility solves. This "
                         "is what makes the large regions tractable. 0 disables "
                         "it (default: %(default)s)")
    ap.add_argument("--out-dir", default=None,
                    help="output folder. If it already holds a rows.jsonl the "
                         "run RESUMES: finished (draw, method) pairs are skipped "
                         "and new rows appended. Default: a fresh timestamped "
                         "folder under --results-root.")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir or OUT_DIR
               or os.path.join(args.results_root,
                               f"{args.region}_{args.year}_{ts}"))
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "rows.jsonl")

    cfg = {"region": args.region, "year": args.year, "r_root": args.r_root,
           "tie_break": args.tie_break, "seed": SEED,
           "verify": VERIFY_STABILITY, "require_common": REQUIRE_COMMON,
           "one_pref_by": ONE_PREF_BY, "no_rel": args.no_rel_heur_time}

    zetas = _zeta_grid(args.region)
    print(f"[simulate] {args.region} {args.year}: {args.draws} draws x "
          f"{len(METHODS) + len(zetas)} methods, tie_break={args.tie_break}, "
          f"cores={args.cores}, NoRelHeurTime={args.no_rel_heur_time:.0f}s")
    print(f"[simulate] output: {os.path.abspath(out_dir)}")

    # read the instance before spawning workers, so a bad path fails immediately
    load_region(args.region, args.year, args.r_root,
                tie_break=args.tie_break, seed=SEED, verbose=True)

    tasks = [(d, m) for m in METHODS + [f"Hybrid-{z}" for z in zetas]
             for d in range(args.draws)]
    rows, done = _load_checkpoint(ckpt)
    if done:
        print(f"[simulate] resuming: {len(done)} rows already on disk, skipping")
        tasks = [t for t in tasks if t not in done]
    tasks.sort(key=lambda t: (not _is_heavy(t[1]), t[0]))     # heavy IPs first
    print(f"[simulate] {len(tasks)} (draw, method) pairs to run")

    t0 = time.perf_counter()
    with open(ckpt, "a") as ck:
        def record(i, row):
            rows.append(row)
            ck.write(json.dumps(row, default=str) + "\n")
            ck.flush()
            print(f"[{i}/{len(tasks)}] draw {row['draw']:>3} "
                  f"{row['method']:<16} solved={row['solved']} "
                  f"({time.perf_counter() - t0:7.1f}s)", flush=True)

        if args.cores <= 1:
            _init_worker(cfg)
            for i, task in enumerate(tasks, 1):
                record(i, _run_one(task))
        else:
            with ProcessPoolExecutor(max_workers=args.cores,
                                     initializer=_init_worker,
                                     initargs=(cfg,)) as ex:
                futs = {ex.submit(_run_one, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futs), 1):
                    record(i, fut.result())

    order = METHODS + [f"Hybrid-{z}" for z in zetas]
    tables.write_rows_csv(rows, os.path.join(out_dir, "rows.csv"))
    agg = metrics.aggregate_by_method(rows)
    tables.write_agg_csv(agg, os.path.join(out_dir, "aggregate.csv"), order)
    cap = f"{args.region} {args.year}, {args.draws} draws, {args.tie_break}"
    with open(os.path.join(out_dir, "table_main.tex"), "w") as f:
        f.write(tables.render_main(agg, order, cap, "tab:main"))
    with open(os.path.join(out_dir, "table_split.tex"), "w") as f:
        f.write(tables.render_split(agg, order, cap + " (sibling split)",
                                    "tab:split"))
    print(f"[simulate] wrote rows.csv, aggregate.csv, table_main.tex and "
          f"table_split.tex to {out_dir} "
          f"({time.perf_counter() - t0:.1f}s total)")

    if zetas:
        print(tables.hybrid_frontier(agg, zetas))

    if VERIFY_STABILITY:
        for m in order:
            a = agg.get(m, {})
            if "acs_pct" not in a:
                continue
            line = (f"[simulate] {m:<16} ACS {a['acs_pct']:5.0f}%   "
                    f"blocking {a.get('blocking_pct_mean', float('nan')):.2f}% "
                    f"of students")
            if "n_converged" in a:
                avg = a.get("iters_converged_mean")
                loops = f" (avg {avg:.1f} loops)" if avg is not None else ""
                line += f"   converged {a['n_converged']}/{a['solved']}{loops}"
            print(line)
        hard = agg.get("Absolute-Hard", {})
        if hard.get("acs_pct") not in (None, 100.0) and hard.get("solved"):
            print(f"[simulate] WARNING: Absolute-Hard reached "
                  f"{hard['acs_pct']:.0f}% ACS. Every matching it returns is "
                  f"stable by construction, so this means the formulation and "
                  f"the checker disagree.")


if __name__ == "__main__":
    main()
