"""
simulations_acs.py
==================
Unified runner in the spirit of simulations_new.py + simulations_checker.py.

For each lottery draw and each method it:
  1. runs the method to get a matching,
  2. scores it on Table-1 statistics (compute_stats), and
  3. judges Absolute Contingent Stability with the VALIDATED new verifier
     (acs_verifier.check_acs) -- NOT the old stability_check.

Methods compared
----------------
  heuristics (heuristics, absolute boost, basic_tb):
    fsda_single, simultaneous, descending_da, descending_fsda,
    ascending_da, ascending_fsda, LS, LS_DA, LS_nd, SL, SL_DA, SL_nd
  benchmarks:
    DA          -> algorithms.DA (SOSM, no sibling priority)
    FOSM        -> solve_opt.MaxSiblings
    Soft        -> solve_opt.AbsoluteSoft(control=0)
    Hard        -> solve_opt.AbsoluteHard
    Hybrid-310  -> solve_opt.AbsoluteSoft(control=310)
    Hybrid-320  -> solve_opt.AbsoluteSoft(control=320)
    Hard-NTB    -> solve_opt.AbsoluteHardNTB

Output: a per-(sim,method) CSV and a summary table ranking methods by ACS rate.

Usage
-----
  python simulations_acs.py             # real run (needs codebase + data + Gurobi)
  python simulations_acs.py --selftest  # wiring test (no data / Gurobi needed)

Config at the bottom: REGION / YEAR / TIE_BREAKER / NUM_SIMS / METHODS / NUM_CORES.
"""

import os
import sys
import copy
import time
import random

import acs_verifier
import acs_priority as P
import heuristics as hv2

# ---- codebase modules: guarded so --selftest runs without Gurobi/data ----
try:
    import numpy as np
    import pandas as pd
    import generate_inputs as genin
    import algorithms as alg
    import solve_opt as opt
    _HAVE_CODEBASE = True
except Exception as _e:
    _HAVE_CODEBASE = False
    _IMPORT_ERR = _e


# ============================================================
# Benchmark runners 
# ============================================================

def run_da(inputs_basic):
    students, colleges, pref, cap, siblings, levels, spl = inputs_basic
    t0 = time.time()
    match = hv2._run_DA(list(students), pref, cap)   # same alg.DA, import-safe
    x_opt = {s: {c: 1} for s, c in match.items() if c is not None}
    return {"status": "completed", "x_opt": x_opt, "runtime": time.time() - t0}


def run_fosm(inputs_full):
    return opt.MaxSiblings(inputs_full, penalty_unassigned="last_pref")


def run_soft(inputs_full):
    return opt.AbsoluteSoft(inputs_full, "last_pref", "SOSM", None, 0)


def run_hard(inputs_full):
    return opt.AbsoluteHard(inputs_full, penalty_unassigned="last_pref", objective="SOSM")


def run_hybrid310(inputs_full):
    return opt.AbsoluteSoft(inputs_full, "last_pref", "SOSM", None, 310)


def run_hybrid320(inputs_full):
    return opt.AbsoluteSoft(inputs_full, "last_pref", "SOSM", None, 320)


def run_hard_ntb(inputs_full):
    return opt.AbsoluteHardNTB(inputs_full, penalty_unassigned="last_pref", objective="SOSM")


# ============================================================
# Method registry: name -> (kind, runner)
#   kind in {"full" (inputs_full), "basic" (inputs_basic), "basic_tb"}
# ============================================================

_V2 = {
    "fsda_single":     hv2.fsda_single,
    "simultaneous":    hv2.simultaneous,
    "descending_da":   hv2.descending_da,
    "descending_fsda": hv2.descending_fsda,
    "ascending_da":    hv2.ascending_da,
    "ascending_fsda":  hv2.ascending_fsda,
    "LS":    hv2.LS,   "LS_DA":  hv2.LS_DA,  "LS_nd":  hv2.LS_nd,
    "SL":    hv2.SL,   "SL_DA":  hv2.SL_DA,  "SL_nd":  hv2.SL_nd,
}

_BENCH = {
    "DA":         ("basic", run_da),
    "FOSM":       ("full",  run_fosm),
    "Soft":       ("full",  run_soft),
    "Hard":       ("full",  run_hard),
    "Hybrid-310": ("full",  run_hybrid310),
    "Hybrid-320": ("full",  run_hybrid320),
    "Hard-NTB":   ("full",  run_hard_ntb),
}

METHODS = {name: ("basic_tb", fn) for name, fn in _V2.items()}
METHODS.update(_BENCH)


def run_one_method(name, inputs_basic, inputs_full, tb):
    kind, fn = METHODS[name]
    if kind == "full":
        return fn(inputs_full)
    if kind == "basic":
        return fn(inputs_basic)
    if kind == "basic_tb":
        return fn(inputs_basic, tb=tb)
    raise ValueError(kind)


# ============================================================
# Glue
# ============================================================

def x_opt_to_mu(x_opt, students):
    mu = {}
    for s in students:
        d = x_opt.get(s)
        if not d:
            mu[s] = None
            continue
        chosen = None
        for c, v in d.items():
            if v == 1:
                chosen = c
                break
        mu[s] = chosen if chosen is not None else next(iter(d.keys()))
    return mu


# ============================================================
# compute_stats (verbatim from simulations_new.py)
# ============================================================

def compute_stats(x_opt, students, pref, siblings):
    out = {
        "solved": 1, "num_students": len(students), "num_assigned": 0,
        "num_unassigned": 0, "sum_pref_all": 0.0, "sum_pref_sin": 0.0,
        "num_matched_sin": 0, "num_matched_sib": 0, "num_sin_total": 0,
        "num_sib_total": 0, "num_together": 0, "sep_none": 0, "sep_one": 0,
        "sep_both": 0, "top_pref": 0,
    }
    is_sib = {s: (len(siblings.get(s, [])) > 0) for s in students}
    out["num_sin_total"] = sum(1 for s in students if not is_sib[s])
    out["num_sib_total"] = sum(1 for s in students if is_sib[s])

    def rbd_rank(s, c):
        if s not in pref:
            return None
        seen = []
        for p in sorted(pref[s]):
            rbd = pref[s][p].split("_")[0]
            if rbd not in seen:
                seen.append(rbd)
            if rbd == c.split("_")[0]:
                return len(seen)
        return None

    assigned_rbd = {}
    for s in x_opt:
        for c in x_opt[s]:
            if x_opt[s][c] > 1 - 1e-3:
                assigned_rbd[s] = c.split("_")[0]
                out["num_assigned"] += 1
                r = rbd_rank(s, c)
                if r is not None:
                    out["sum_pref_all"] += r
                    if r == 1:
                        out["top_pref"] += 1
                    if not is_sib[s]:
                        out["sum_pref_sin"] += r
                        out["num_matched_sin"] += 1
                    else:
                        out["num_matched_sib"] += 1
    out["num_unassigned"] = len(students) - out["num_assigned"]

    for s in assigned_rbd:
        rbd = assigned_rbd[s]
        for sib in siblings.get(s, []):
            if sib in assigned_rbd and assigned_rbd[sib] == rbd:
                out["num_together"] += 1
                break

    sep_none_set, sep_one_set, sep_both_set = set(), set(), set()

    def rbds_listed(s):
        if s not in pref:
            return set()
        return {pref[s][p].split("_")[0] for p in pref[s]}

    def rbds_up_to_assigned(s):
        if s not in assigned_rbd or s not in pref:
            return set()
        assigned = assigned_rbd[s]
        out_set = []
        for p in sorted(pref[s]):
            rbd = pref[s][p].split("_")[0]
            if rbd not in out_set:
                out_set.append(rbd)
            if rbd == assigned:
                break
        return set(out_set)

    for id_s in siblings:
        if id_s not in students:
            continue
        for sib in siblings[id_s]:
            if sib not in students:
                continue
            s_in = id_s in assigned_rbd
            sib_in = sib in assigned_rbd
            if not s_in and not sib_in:
                if rbds_listed(id_s) & rbds_listed(sib):
                    sep_none_set.add(id_s); sep_none_set.add(sib)
            elif s_in and not sib_in:
                if rbds_up_to_assigned(id_s) & rbds_listed(sib):
                    sep_one_set.add(id_s); sep_one_set.add(sib)
            elif not s_in and sib_in:
                if rbds_listed(id_s) & rbds_up_to_assigned(sib):
                    sep_one_set.add(id_s); sep_one_set.add(sib)
            else:
                if assigned_rbd[id_s] != assigned_rbd[sib]:
                    if rbds_up_to_assigned(id_s) & rbds_up_to_assigned(sib):
                        sep_both_set.add(id_s); sep_both_set.add(sib)

    out["sep_none"] = len(sep_none_set)
    out["sep_one"] = len(sep_one_set)
    out["sep_both"] = len(sep_both_set)
    out["avg_pref"] = (out["sum_pref_all"] / out["num_assigned"]
                       if out["num_assigned"] > 0 else 0.0)
    out["avg_pref_sin"] = (out["sum_pref_sin"] / out["num_matched_sin"]
                           if out["num_matched_sin"] > 0 else 0.0)
    out["pct_sin_matched"] = (100.0 * out["num_matched_sin"] / out["num_sin_total"]
                              if out["num_sin_total"] > 0 else 0.0)
    out["pct_sib_matched"] = (100.0 * out["num_matched_sib"] / out["num_sib_total"]
                              if out["num_sib_total"] > 0 else 0.0)
    out["pct_sib_together"] = (100.0 * out["num_together"] / out["num_sib_total"]
                               if out["num_sib_total"] > 0 else 0.0)
    return out


# ============================================================
# One simulation: load instance, draw lottery, run all methods
# ============================================================

def run_one_sim(args):
    region_indir, year, tie_breaker, sim_idx, methods = args
    np.random.seed(sim_idx + 1)
    random.seed(sim_idx + 1)

    instance_file = os.path.join(region_indir, str(year), "instance.txt")
    (students, colleges, pref, cap, siblings, levels,
     students_per_level, Tp, Tn, Sp, Sn) = genin.read_instance(instance_file)

    pref_with_tb, tb = genin.modify_school_loterries(
        copy.deepcopy(pref), students, colleges, siblings, tie_breaker)
    _s2, _c2, Tp2, Tn2, Sp2, Sn2 = \
        genin.create_additional_inputs_from_instance(pref_with_tb, cap)

    inputs_basic = (students, colleges, pref_with_tb, cap, siblings,
                    levels, students_per_level)
    inputs_full = (students, colleges, pref_with_tb, cap, siblings,
                   levels, students_per_level, Tp2, Tn2, Sp2, Sn2, tb)
    levels_of = P.build_levels_of(students, pref_with_tb)

    rows = []
    for name in methods:
        t0 = time.time()
        try:
            out = run_one_method(name, inputs_basic, inputs_full, tb)
        except Exception as e:
            print(f"  sim={sim_idx:3d} {name:16s} [ERROR] {e}")
            rows.append({"sim": sim_idx, "method": name, "solved": 0,
                         "runtime": time.time() - t0})
            continue
        elapsed = time.time() - t0
        if not out or not out.get("x_opt"):
            print(f"  sim={sim_idx:3d} {name:16s} [UNSOLVED] t={elapsed:.1f}s")
            rows.append({"sim": sim_idx, "method": name, "solved": 0,
                         "runtime": elapsed})
            continue

        x_opt = out["x_opt"]
        mu = x_opt_to_mu(x_opt, students)
        stats = compute_stats(x_opt, students, pref_with_tb, siblings)
        acs = acs_verifier.check_acs(mu, students, colleges, pref_with_tb,
                                     cap, siblings, levels_of, tb=tb, exclude_sibling_envy=True)

        row = {
            "sim": sim_idx, "method": name, "solved": 1,
            "status": out.get("status", "completed"),
            "runtime": out.get("runtime", elapsed),
            "iterations": out.get("iterations", None),
            "is_acs": int(acs["is_acs"]),
            "n_blocking": acs["n_blocking_pairs"],
            "avg_pref": stats["avg_pref"],
            "top_pref": stats["top_pref"],
            "num_unassigned": stats["num_unassigned"],
            "num_together": stats["num_together"],
            "pct_sib_together": stats["pct_sib_together"],
            "sep_none": stats["sep_none"],
            "sep_one": stats["sep_one"],
            "sep_both": stats["sep_both"],
        }
        rows.append(row)
        print(f"  sim={sim_idx:3d} {name:16s} "
              f"{out.get('status','completed'):16s} "
              f"acs={int(acs['is_acs'])} n={acs['n_blocking_pairs']:<5} "
              f"together={stats['num_together']:<5} t={row['runtime']:.1f}s")
    return rows


# ============================================================
# Driver + aggregation
# ============================================================

def main(region, year, tie_breaker, num_sims, methods, region_root, num_cores=1,
         table_name="acs_table", out_root="outputs"):
    if not _HAVE_CODEBASE:
        print("ERROR: codebase modules not importable here:")
        print(f"  {_IMPORT_ERR}")
        print("Run on your machine with the full project + Gurobi on PYTHONPATH.")
        sys.exit(1)

    import datetime
    region_indir = os.path.join(region_root, region)
    out_dir = os.path.join(out_root, "new_simulations", datetime.date.today().isoformat())
    os.makedirs(out_dir, exist_ok=True)
    print(f"Region={region} Year={year} TB={tie_breaker} Sims={num_sims} "
          f"Cores={num_cores}  Table={table_name}")
    print(f"Output dir: {out_dir}")
    print(f"Methods: {methods}")
    print("=" * 90)

    tasks = [(region_indir, year, tie_breaker, s, methods) for s in range(num_sims)]
    all_rows = []
    if num_cores > 1:
        import multiprocessing
        with multiprocessing.Pool(num_cores) as pool:
            for rows in pool.map(run_one_sim, tasks):
                all_rows.extend(rows)
    else:
        for t in tasks:
            all_rows.extend(run_one_sim(t))

    if not all_rows:
        print("No rows.")
        return

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(out_dir, f"{table_name}_rows_{num_sims}sims.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {os.path.abspath(out_csv)}")

    # generate the three LaTeX tables (ACS summary + two descriptive tables)
    try:
        import analysis_acs
        stats = analysis_acs.summarize(df, methods)
        t1 = analysis_acs.table_acs_summary(stats, methods, table_name, region)
        t2 = analysis_acs._descriptive(stats, methods, table_name, region,
                                       "avg_pref", "Avg.\\ Pref.", first_nd=3)
        t3 = analysis_acs._descriptive(stats, methods, table_name, region,
                                       "top_pref", "Top Pref.", first_nd=1)
        for fn, content in [(f"{table_name}_acs_summary.tex", t1),
                            (f"{table_name}_descriptive_avgpref.tex", t2),
                            (f"{table_name}_descriptive_toppref.tex", t3)]:
            p = os.path.join(out_dir, fn)
            with open(p, "w") as fh:
                fh.write(content)
            print(f"Wrote {os.path.abspath(p)}")
        print("\n" + t1)
    except Exception:
        import traceback
        print("[WARN] table generation failed; CSV is saved. Traceback:")
        traceback.print_exc()
        print(f"You can regenerate tables with:\n"
              f"  python analysis_acs.py {os.path.abspath(out_csv)} "
              f"--name {table_name} --region {region} "
              f"--out-root {os.path.abspath(out_root)}")


# ============================================================
# Wiring self-test (no codebase IPs / data needed)
# ============================================================

def _selftest():
    """Run the v2 heuristics + DA through the full pipeline on a synthetic
    instance and confirm rows + ACS verdicts come out. IP benchmarks skipped."""
    # Micro-test D embedded in the codebase data shapes
    students = ["a1", "a2", "t"]
    colleges = ["30_1", "30_2"]
    pref = {"a1": {1: "30_1"}, "a2": {1: "30_2"}, "t": {1: "30_2"},
            "30_1": {1: "a1"}, "30_2": {1: "t", 2: "a2"}}
    cap = {"30_1": 1, "30_2": 1}
    siblings = {"a1": ["a2"], "a2": ["a1"], "t": []}
    tb = {"t": {"30": 9.}, "a2": {"30": 1.}, "a1": {"30": 5.}}
    inputs_basic = (students, colleges, pref, cap, siblings,
                    {"1": ["30_1"], "2": ["30_2"]}, {"1": ["a1"], "2": ["a2", "t"]})
    levels_of = P.build_levels_of(students, pref)

    test_methods = list(_V2.keys()) + ["DA"]
    print(f"{'method':16s} {'status':16s} acs  n  together")
    for name in test_methods:
        kind, fn = METHODS[name]
        out = fn(inputs_basic, tb=tb) if kind == "basic_tb" else fn(inputs_basic)
        mu = x_opt_to_mu(out["x_opt"], students)
        stats = compute_stats(out["x_opt"], students, pref, siblings)
        acs = acs_verifier.check_acs(mu, students, colleges, pref, cap,
                                     siblings, levels_of, tb=tb, exclude_sibling_envy=True)
        print(f"{name:16s} {out.get('status','completed'):16s} "
              f"{int(acs['is_acs'])}    {acs['n_blocking_pairs']}  "
              f"{stats['num_together']}")
    print("\nWIRING SELF-TEST OK (pipeline runs v2 methods + DA, scores stats + ACS)")


# ============================================================
# Config + entry
# ============================================================

REGION      = "Magallanes"
YEAR        = 2023
TIE_BREAKER = "mtbf"
NUM_SIMS    = 100
NUM_CORES   = 12
REGION_ROOT = "../R/intermediate_data"
TABLE_NAME  = "magallanes_acs"     # <-- name for the output tables/files
# outputs/ as a SIBLING of the python/ folder (script-relative, CWD-independent)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT    = os.path.join(_SCRIPT_DIR, "..", "outputs")

# Order: benchmarks first, then v2 heuristics (edit freely)
METHODS_TO_RUN = [
    "DA", "FOSM", "Soft", "Hard", "Hybrid-310", "Hybrid-320", "Hard-NTB",
    "fsda_single", "simultaneous",
    "descending_da", "descending_fsda", "ascending_da", "ascending_fsda",
    "LS", "LS_DA", "LS_nd", "SL", "SL_DA", "SL_nd",
]


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main(REGION, YEAR, TIE_BREAKER, NUM_SIMS, METHODS_TO_RUN,
             REGION_ROOT, num_cores=NUM_CORES,
             table_name=TABLE_NAME, out_root=OUT_ROOT)