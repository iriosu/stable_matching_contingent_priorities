"""
v5_simulate.py

The experiment driver. From inside v5/:

    nohup python v5_simulate.py --region OHiggins --draws 100 --cores 15 \
        > OHiggins.log 2>&1 &

It reads ../R/intermediate_data/<Region>/<Year>/instance.txt and writes to
../results/v5/<Region>_<Year>_<timestamp>/.


METHODS
-------
  SOSM        Deferred acceptance under the initial priority order. One DA pass,
              essentially free, and the benchmark the other rows are read against.
  Descending  Levels from the highest grade down, recomputing the contingent order
              from the placements already fixed above. Algorithm 1, the rule
              currently used in Chile.
  Ascending   The same, from the lowest grade up.
  LSDA        DA stratified by family size, largest first. Algorithm 2.
              Strategy-proof, but it need not return a contingent stable matching.
  RADA        DA run repeatedly, recomputing the contingent order from the previous
              matching. Algorithm 3. Contingent stable when it converges.
  Absolute    Formulation (6). The rank-optimal absolute contingent stable
              matching, or a proof that none exists.
  FOSM        Formulation (17). The initially stable matching that maximizes the
              number of family members placed together.


THE WARM-START PASS
-------------------
On the large regions the exact formulation is not hard to optimize. It is hard to
make feasible. The root relaxation leaves thousands of fractional variables and a
bound hundreds of rank units short of the optimum, so branch and bound cannot
reach an integer point. Gurobi's NoRel heuristic runs before the relaxation and
does not use it; on the hardest Atacama draw it found a stable matching in 236
seconds, with zero nodes explored, where four hours of branch and bound had found
nothing.

The warm start is computed ONCE per draw, before any method runs:

    1. RADA. If it converges the matching is stable and no solver is needed.
    2. Otherwise the feasibility IP: the same formulation, zero objective, stopped
       at the first solution, with NoRelHeurTime on.

It is written to <out>/warm/<draw>.json and read by the Absolute task on
that draw. FOSM warm-starts from its own DA matching, which is always feasible
for it. Where no stable matching exists the warm start is None and Absolute
proves infeasibility.

The pass is resumable: it skips draws already on disk.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from v5_inputs import load_region, resample_lottery
import v5_exact as exact
import v5_heuristics as heuristics
import v5_metrics as metrics
import v5_stability as stability
import v5_tables as tables


# ==========================================================================
# SETTINGS
# ==========================================================================
R_ROOT = "../R"
RESULTS_ROOT = "../results/v5"

REGION = "Magallanes"
YEAR = 2023
DRAWS = 100
TIE_BREAK = "mtbf"              # individual | family | mtbf
SEED = 0

METHODS = [
    "SOSM",
    "Descending",
    "Ascending",
    "LSDA",
    "RADA",
    "Absolute",
    "FOSM",
]

# One time limit governs every solve in a run: the IP methods and the warm-start
# feasibility pass all use it. It is a single number so that
# every method and every region is held to the same budget; nothing is tuned per
# region. Override it with --time-limit; the per-component flags below default to
# this value and exist only for the rare case of deliberately probing one
# component in isolation.
TIME_LIMIT = 3600               # seconds, applied to every solve
IP_MIP_GAP = 1e-3
IP_THREADS = 1                  # threads per IP
NUM_CORES = 15                  # tasks in parallel

FEAS_NO_REL_HEUR = 600.0        # NoRelHeurTime for the warm-start pass. 0 disables
                                # it, which is right on Magallanes, where it only
                                # delays the root relaxation.

VERIFY_STABILITY = True
REQUIRE_COMMON = True           # separated-column semantics; see v5_metrics.py
ONE_PREF_BY = "assigned"
# ==========================================================================


# ==========================================================================
# workers
# ==========================================================================
_G: dict = {}


def _init(cfg):
    _G["cfg"] = cfg
    _G["base"] = load_region(cfg["region"], cfg["year"], cfg["r_root"],
                             tie_break=cfg["tie_break"], seed=cfg["seed"],
                             verbose=False)


def _init2(cfg, warm):
    _init(cfg)
    _G["warm"] = warm


def _instance(draw):
    cfg = _G["cfg"]
    return resample_lottery(_G["base"], seed=cfg["seed"] + draw,
                            tie_break=cfg["tie_break"])


# ==========================================================================
# phase 1: one stable matching per draw
# ==========================================================================
def _warm_one(draw: int) -> dict:
    cfg = _G["cfg"]
    inst = _instance(draw)
    t0 = time.perf_counter()

    # RADA converges only at a fixed point, which is contingent stable by
    # Proposition 2; the flag is checked again below on the matching itself
    mu, info = heuristics.rada(inst, "absolute")
    via = "rada" if info.get("converged") else None

    if via is None:
        mu, ip = exact.solve(inst, "absolute", "hard", objective="feasible",
                             time_limit=cfg["feas_time_limit"],
                             threads=cfg["ip_threads"],
                             no_rel_heur_time=cfg["no_rel"])
        st = ip.get("status_str")
        via = ("feasibility_ip" if mu is not None
               else "infeasible" if st == "INFEASIBLE" else "not_found")

    rec = {"draw": draw, "via": via,
           "seconds": round(time.perf_counter() - t0, 1)}
    if mu is not None:
        # This costs nothing and is worth it: a warm start that is not stable is
        # not feasible for the model, and Gurobi silently discards it. That is
        # exactly the failure that made the IP look intractable.
        rec["acs"] = bool(stability.is_contingent_stable(inst, mu, "absolute"))
        rec["n_providers"] = len(stability.providers(inst, mu)["eff"])
        rec["matching"] = mu
    else:
        rec["acs"] = False
        rec["n_providers"] = 0
        rec["matching"] = None
    return rec


def warm_pass(draws, cfg, warm_dir):
    os.makedirs(warm_dir, exist_ok=True)
    todo = [d for d in draws
            if not os.path.exists(os.path.join(warm_dir, f"{d}.json"))]
    print(f"\n[warm] one stable matching per draw: {len(draws)} draw(s), "
          f"{len(draws) - len(todo)} on disk, {len(todo)} to compute")

    if todo:
        t0 = time.perf_counter()
        nw = max(1, min(cfg["cores"], len(todo)))
        with ProcessPoolExecutor(max_workers=nw, initializer=_init,
                                 initargs=(cfg,)) as ex:
            futs = {ex.submit(_warm_one, d): d for d in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                with open(os.path.join(warm_dir, f"{r['draw']}.json"), "w") as f:
                    json.dump(r, f)
                if i % 10 == 0 or r["via"] in ("infeasible", "not_found"):
                    print(f"[warm] {i}/{len(todo)}  draw {r['draw']:>3}  "
                          f"{r['via']:<15} providers={r['n_providers']:<5} "
                          f"({r['seconds']:.0f}s)  "
                          f"[{time.perf_counter() - t0:.0f}s]", flush=True)

    warm = {}
    for d in draws:
        with open(os.path.join(warm_dir, f"{d}.json")) as f:
            warm[d] = json.load(f)

    ok = sum(1 for r in warm.values() if r["acs"])
    inf = [d for d, r in warm.items() if r["via"] == "infeasible"]
    unk = [d for d, r in warm.items() if r["via"] == "not_found"]
    provs = [r["n_providers"] for r in warm.values() if r["acs"]]
    print(f"[warm] {ok}/{len(draws)} have a stable matching, "
          f"{len(inf)} proven to have none {inf if inf else ''}, "
          f"{len(unk)} unresolved {unk if unk else ''}")
    if provs:
        print(f"[warm] effective providers: {min(provs)} to {max(provs)}, "
              f"mean {sum(provs) / len(provs):.0f}.")
    if unk:
        print(f"[warm] WARNING: {len(unk)} draw(s) neither solved nor proven "
              f"infeasible. Their IP rows run without a warm start and will "
              f"probably time out.")
    return warm


# ==========================================================================
# phase 2: the methods
# ==========================================================================
def method_registry(warm, cfg):
    """warm is this draw's record from the warm-start pass."""
    acs = warm.get("matching")          # a stable matching, or None

    def hard(inst):
        mu, ip = exact.solve(inst, "absolute", "hard",
                             time_limit=cfg["ip_time_limit"],
                             mip_gap=cfg["ip_mip_gap"],
                             threads=cfg["ip_threads"], warm_start=acs)
        via = ("ip" if mu is not None
               else "infeasible" if ip.get("status_str") == "INFEASIBLE"
               else "no_solution")
        return mu, {**ip, "via": via, "warm": acs is not None,
                    "rank_optimal": ip.get("status_str") == "OPTIMAL"}

    def fosm(inst):
        # FOSM: the paper's benchmark FOSM. The matching that maximizes the
        # number of family members placed in the same school among matchings
        # satisfying STANDARD (initial) stability, with no contingent
        # priorities. This is the object defined in the appendix formulation
        # with the initial stability constraint (3a). SOSM is initially stable,
        # so the DA matching is always a feasible warm start and the model is
        # never infeasible. That start is its own DA pass, not the shared
        # feasibility cache, so it reports warm=False and no warm_seconds: the
        # cache time must not be charged to a method that never reads it.
        w = heuristics.deferred_acceptance(inst)
        mu, ip = exact.solve(inst, "none", "hard", objective="together",
                             time_limit=cfg["ip_time_limit"],
                             mip_gap=cfg["ip_mip_gap"], threads=cfg["ip_threads"],
                             warm_start=w)
        via = "ip" if mu is not None else "no_solution"
        return mu, {**ip, "via": via, "warm": False, "rank_optimal": False}

    return {
        "SOSM": lambda inst: (heuristics.deferred_acceptance(inst), None),
        "Descending": lambda inst: (heuristics.descending(inst, "absolute"), None),
        "Ascending": lambda inst: (heuristics.ascending(inst, "absolute"), None),
        "LSDA": lambda inst: (heuristics.lsda(inst), None),
        "RADA": lambda inst: heuristics.rada(inst, "absolute"),
        "Absolute": hard,
        "FOSM": fosm,
    }


ACS_TARGET = "absolute"


def _is_heavy(method):
    """Long jobs first, so the fast heuristics fill the tail and no core idles."""
    return method.startswith("Absolute") or method == "FOSM"


def _run_one(task):
    draw, method = task
    cfg = _G["cfg"]
    inst = _instance(draw)
    warm = _G["warm"][draw]
    reg = method_registry(warm, cfg)

    t0 = time.perf_counter()
    mu, info = reg[method](inst)
    wall = time.perf_counter() - t0

    pr = "absolute" if method.startswith("Absolute") or method == "FOSM" else None
    row = metrics.evaluate(inst, mu, info, priority=pr,
                           require_common=cfg["require_common"],
                           one_pref_by=cfg["one_pref_by"])
    row["draw"], row["method"] = draw, method
    row.setdefault("runtime", wall)
    row["warm_via"] = warm["via"]
    # The warm-start feasibility solve is computed once per draw and consumed by
    # Absolute. Record its cost on the rows that actually used it, so the
    # tables can report the solve time and, in parentheses, the warm-start time
    # behind it. The heuristics and FOSM (which warm-starts from its own DA
    # matching) get no warm time.
    if info is not None and info.get("warm"):
        row["warm_seconds"] = warm.get("seconds")

    if cfg["verify"] and mu is not None:
        ok, blk = stability.is_contingent_stable(inst, mu, ACS_TARGET,
                                                 return_blocking=True)
        row["acs"] = int(ok)
        # each student in a blocking pair counted once, however many schools they
        # would block with; the incumbent they envy is not counted
        blockers = {t[1] for t in blk if t[0] in ("waste", "envy")}
        row["n_blocking"] = len(blockers)
        row["blocking_pct"] = 100.0 * len(blockers) / len(inst.students)
    return row


def _load_rows(ckpt):
    by = {}
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
                by[(r.get("draw"), r.get("method"))] = r
    return list(by.values()), set(by)


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--year", default=YEAR)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--warm-only", action="store_true",
                    help="compute and persist the warm-start cache, print the "
                         "per-draw warm-start time statistics, and stop without "
                         "running any method. Use to repopulate warm/ when a "
                         "previous run did not keep it.")
    ap.add_argument("--methods", default=None,
                    help="comma list restricting which methods run, using the "
                         "names in METHODS (e.g. SOSM,Descending,LSDA,RADA,"
                         "Absolute). Default runs all of METHODS.")
    ap.add_argument("--tie-break", default=TIE_BREAK)
    ap.add_argument("--cores", type=int, default=NUM_CORES)
    ap.add_argument("--ip-threads", type=int, default=IP_THREADS)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT,
                    help="the single time budget, in seconds, applied to EVERY "
                         "solve (IP methods and the warm-start pass) and "
                         "identical across regions. Default 3600.")
    ap.add_argument("--ip-time-limit", type=float, default=None,
                    help="override the IP budget only (defaults to --time-limit)")
    ap.add_argument("--feas-time-limit", type=float, default=None,
                    help="override the warm-start budget (defaults to --time-limit)")
    ap.add_argument("--mip-gap", type=float, default=IP_MIP_GAP)
    ap.add_argument("--no-rel-heur-time", type=float, default=FEAS_NO_REL_HEUR,
                    help="NoRelHeurTime for the warm-start pass. It is what makes "
                         "the large regions tractable. Use 0 on Magallanes "
                         "(default: %(default)s)")
    ap.add_argument("--r-root", default=R_ROOT)
    ap.add_argument("--results-root", default=RESULTS_ROOT)
    ap.add_argument("--out-dir", default=None,
                    help="output folder. If it already holds rows.jsonl the run "
                         "RESUMES: it skips the (draw, method) pairs on disk and "
                         "reuses the warm-start pass. Default: a fresh timestamped "
                         "folder.")
    a = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = a.out_dir or os.path.join(a.results_root,
                                        f"{a.region}_{a.year}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "rows.jsonl")

    # every solve shares one time limit; the per-component flags override only if
    # the user set them explicitly
    tl = a.time_limit
    ip_tl = a.ip_time_limit if a.ip_time_limit is not None else tl
    feas_tl = a.feas_time_limit if a.feas_time_limit is not None else tl
    cfg = {"region": a.region, "year": a.year, "r_root": a.r_root,
           "tie_break": a.tie_break, "seed": SEED, "cores": a.cores,
           "ip_threads": a.ip_threads, "ip_time_limit": ip_tl,
           "ip_mip_gap": a.mip_gap, "feas_time_limit": feas_tl,
           "no_rel": a.no_rel_heur_time, "verify": VERIFY_STABILITY,
           "require_common": REQUIRE_COMMON, "one_pref_by": ONE_PREF_BY}

    if a.methods:
        chosen = [m.strip() for m in a.methods.split(",") if m.strip()]
        unknown = [m for m in chosen if m not in METHODS]
        if unknown:
            raise SystemExit(f"unknown method(s) {unknown}; choose from {METHODS}")
        base = [m for m in METHODS if m in chosen]   # preserve METHODS order
    else:
        base = list(METHODS)
    order = base
    print(f"[simulate] {a.region} {a.year}: {a.draws} draws x {len(order)} methods")
    print(f"[simulate] {', '.join(base)}")
    print(f"[simulate] {a.cores} tasks in parallel, {a.ip_threads} thread(s) per "
          f"IP, NoRelHeurTime {a.no_rel_heur_time:.0f}s")
    print(f"[simulate] time limit {tl:.0f}s, applied to every solve "
          f"(IP={ip_tl:.0f}s, warm={feas_tl:.0f}s)")
    print(f"[simulate] output: {os.path.abspath(out_dir)}")

    load_region(a.region, a.year, a.r_root, tie_break=a.tie_break, seed=SEED,
                verbose=True)

    draws = list(range(a.draws))
    warm = warm_pass(draws, cfg, os.path.join(out_dir, "warm"))

    if a.warm_only:
        # Recompute and persist the warm-start cache only, then stop. Useful when
        # a previous run did not keep warm/ and the per-draw warm-start times are
        # needed for the time tables. The cache is written to <out_dir>/warm.
        secs = [r["seconds"] for r in warm.values()
                if r.get("seconds") is not None
                and r.get("via") not in ("infeasible", "not_found")]
        if secs:
            import statistics as _st
            print(f"[warm-only] cache written to {os.path.join(out_dir, 'warm')}")
            print(f"[warm-only] per-draw warm-start seconds over "
                  f"{len(secs)} solved draw(s): mean {_st.mean(secs):.1f}, "
                  f"min {min(secs):.0f}, max {max(secs):.0f}")
        return

    tasks = [(d, m) for m in order for d in draws]
    rows, done = _load_rows(ckpt)
    if done:
        print(f"\n[simulate] resuming: {len(done)} row(s) on disk, skipping them")
        tasks = [t for t in tasks if t not in done]
    tasks.sort(key=lambda t: (not _is_heavy(t[1]), t[0]))
    print(f"[simulate] {len(tasks)} (draw, method) pair(s) to run\n")

    t0 = time.perf_counter()
    with open(ckpt, "a") as ck:
        with ProcessPoolExecutor(max_workers=a.cores, initializer=_init2,
                                 initargs=(cfg, warm)) as ex:
            futs = {ex.submit(_run_one, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                rows.append(row)
                ck.write(json.dumps(row, default=str) + "\n")
                ck.flush()
                print(f"[{i}/{len(tasks)}] draw {row['draw']:>3} "
                      f"{row['method']:<17} solved={row['solved']} "
                      f"({time.perf_counter() - t0:7.0f}s)", flush=True)

    tables.write_rows_csv(rows, os.path.join(out_dir, "rows.csv"))
    agg = metrics.aggregate_by_method(rows)
    tables.write_agg_csv(agg, os.path.join(out_dir, "aggregate.csv"), order)
    tb_label = {"mtbf": "MTB-F", "family": "STB-F",
                "individual": "STB"}.get(a.tie_break, a.tie_break)
    cap = f"{a.region} {a.year}, {a.draws} draws, {tb_label}."
    # every table in two coherent unit variants: _abs (counts) and _pct
    # (percentages). The two are never mixed inside one table.
    written = []
    for units, suf in (("absolute", "abs"), ("percent", "pct")):
        with open(os.path.join(out_dir, f"table_main_{suf}.tex"), "w") as f:
            f.write(tables.render_main(agg, order, cap, f"tab:main_{suf}",
                                       units=units))
        fn = f"table_separated_{suf}.tex"
        with open(os.path.join(out_dir, fn), "w") as f:
            f.write(tables.render_separated(
                agg, order, cap + " (separated)",
                f"tab:separated_{suf}", units=units, level="student"))
        written.append(fn)
        with open(os.path.join(out_dir, f"table_split_{suf}.tex"), "w") as f:
            f.write(tables.render_split(agg, order, cap + " (sibling split)",
                                        f"tab:split_{suf}", computation=False,
                                        units=units))
        written += [f"table_main_{suf}.tex", f"table_split_{suf}.tex"]
    with open(os.path.join(out_dir, "table_computation.tex"), "w") as f:
        f.write(tables.render_computation(agg, order, cap + " (by computation)",
                                          "tab:computation_time"))
    written.append("table_computation.tex")
    print(f"\n[simulate] wrote rows.csv, aggregate.csv, "
          f"{', '.join(written)} to {out_dir} "
          f"({time.perf_counter() - t0:.0f}s)")

    print()
    for m in order:
        x = agg.get(m, {})
        if "acs_pct" not in x:
            continue
        line = (f"[simulate] {m:<17} ACS {x['acs_pct']:5.0f}%   "
                f"blocking {x.get('blocking_pct_mean', float('nan')):5.2f}%   "
                f"solved {int(x.get('solved', 0)):>3}/{a.draws}")
        if "n_converged" in x:
            line += f"   converged {x['n_converged']}/{int(x['solved'])}"
        print(line)

    h = agg.get("Absolute", {})
    if h.get("acs_pct") not in (None, 100.0) and h.get("solved"):
        print(f"[simulate] WARNING: Absolute reached {h['acs_pct']:.0f}% ACS. "
              f"Every matching it returns is stable by construction, so the "
              f"formulation and the checker disagree. Run v1_certify.py.")


if __name__ == "__main__":
    main()
