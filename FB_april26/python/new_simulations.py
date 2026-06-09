"""
new_simulations.py
==================
Runs the 8 heuristics (DA, FSD-U, SF-Reserve, SF-SD, SF-DA, SF-AP-DA, FSD,
FSD-A) and compares them against the main paper's benchmarks (Hard, Soft,
FOSM, SOSM, Descending) across many lottery simulations.

Produces a LaTeX / CSV table in the same format as Table 1 of the paper,
augmented with:
  - Avg. pref. (singletons)
  - % matched singletons
  - % siblings matched
  - % siblings matched together

Usage
-----
  python new_simulations.py

Adjust REGION, YEAR, TIE_BREAKER, NUM_SIMS, NUM_CORES at the bottom.
"""
import os
import sys
import time
import pickle
import copy
import random
import multiprocessing
from collections import defaultdict

import numpy as np
import pandas as pd

np.random.seed(1)
random.seed(1)

import generate_inputs as genin
import algorithms as alg
import solve_opt as opt
import mechanisms as mech

from functools import partial



# ============================================================
# Benchmark wrappers (to unify output format with mechanisms.py)
# ============================================================

# ============================================================
# Benchmark wrappers (to unify output format with mechanisms.py)
# ============================================================

def run_hard(inputs_full):
    """Absolute-Hard IP (Formulation 21, Table 1)."""
    return opt.AbsoluteHard(inputs_full, penalty_unassigned="last_pref", objective="SOSM")


def run_soft(inputs_full, k=0):
    """Absolute-Soft IP (vanilla, k=0) or Absolute-Hybrid (k=ζ).
    With k>0 imposes Σz ≥ ζ (Hybrid constraint, §5.3.1, Table 2)."""
    return opt.AbsoluteSoft(inputs_full, "last_pref", "SOSM", None, k)


def run_abs_ntb(inputs_full):
    """Absolute-Hard No Tie-Breakers (Formulation in §E.2.2, Table 3 Panel 2 'Rank').
    Removes initial lotteries; SOSM objective minimizes sum of ranks."""
    return opt.AbsoluteHardNTB(inputs_full, penalty_unassigned="last_pref", objective="SOSM")


def run_descending(inputs_basic):
    """Descending heuristic: sequential DA from 12th grade down to Pre-K
    (current Chilean practice, §5.2.2)."""
    return alg.Descending(inputs_basic)


def run_ascending(inputs_basic):
    """Ascending heuristic: sequential DA from Pre-K up to 12th grade
    (Appendix D.3, Table 7 — empirically beats Descending on siblings together)."""
    return alg.Ascending(inputs_basic)


def run_sosm(inputs_basic):
    """SOSM = standard DA (no sibling priorities)."""
    students, colleges, pref, cap, siblings, levels, students_per_level = inputs_basic
    start = time.time()
    match = alg.DA(students, pref, cap)
    x_opt = {s: {c: 1} for s, c in match.items() if c is not None}
    return {
        "status": "completed",
        "x_opt": x_opt,
        "runtime": time.time() - start,
        "num_vars": 0, "num_cols": 0, "mipgap": 0, "nodes": 0,
    }


def run_fosm(inputs_full):
    """FOSM = Family-Optimal Stable Matching via MaxSiblings IP."""
    return opt.MaxSiblings(inputs_full, penalty_unassigned="last_pref")

# ============================================================
# Method registry
# ============================================================
#
# Each entry: name -> (runner, needs_full_inputs)
#   full_inputs = (students, colleges, pref, cap, siblings, levels,
#                  students_per_level, Tp, Tn, Sp, Sn, tb)
#   basic_inputs = (students, colleges, pref, cap, siblings, levels,
#                   students_per_level)

BENCHMARK_METHODS = {
    # No contingent priorities (paper's "best stable" baselines)
    "SOSM":        ("basic", run_sosm),
    "FOSM":        ("full",  run_fosm),

    # Sequential-by-level heuristics
    "Descending":  ("basic", run_descending),    # current Chilean practice
    "Ascending":   ("basic", run_ascending),     # paper Appendix D.3

    # Absolute contingent priorities (the paper's main contribution)
    "Hard":        ("full",  run_hard),                       # Table 1: 664.43 together, 86/100 solved
    "Soft":        ("full",  run_soft),                       # Table 1: 569.39 together, 100/100 solved
    "Hybrid-310":  ("full",  partial(run_soft, k=310)),       # Table 2: 646.93 together, 100/100 solved
    "Hybrid-320":  ("full",  partial(run_soft, k=320)),       # Table 2: 667.21 together, 80/100 solved
    "Hard-NTB":    ("full",  run_abs_ntb),                    # Table 3 Panel 2: 1073 together (no lotteries)
}

HEURISTIC_METHODS = {
    "DA":         ("basic_tb", mech.plain_DA),
    "FSD-U":      ("basic_tb", mech.FSD_U),
    "SF-Reserve": ("basic_tb", mech.SF_Reserve),
    "SF-SD":      ("basic_tb", mech.SF_SD),
    "SF-DA":      ("basic_tb", mech.SF_DA),
    "SF-AP-DA":   ("basic_tb", mech.SF_AP_DA),
    "FSD":        ("basic_tb", mech.FSD),
    "FSD-A":      ("basic_tb", mech.FSD_A),
}

ALL_METHODS = {**BENCHMARK_METHODS, **HEURISTIC_METHODS}


# ============================================================
# Statistics computation for a single simulation
# ============================================================

def compute_stats(x_opt, students, pref, siblings):
    """
    Computes the statistics needed to populate the Table-1-style output.

    Returns a dict with:
      - solved         : 1 if matching produced, else 0
      - avg_pref       : average preference of matched students (all)
      - avg_pref_sin   : average preference of matched singletons
      - num_unassigned : number of unassigned students (all)
      - num_together   : number of students matched with ≥1 sibling
      - sep_none       : separated siblings, neither assigned
      - sep_one        : separated siblings, exactly one assigned
      - sep_both       : separated siblings, both assigned
      - pct_sin_matched : % of singleton students matched
      - pct_sib_matched : % of sibling students matched
      - pct_sib_together: % of sibling students matched with ≥1 sibling
      - top_pref       : number of students who got their top preference
    """
    out = {
        "solved": 1,
        "num_students": len(students),
        "num_assigned": 0,
        "num_unassigned": 0,
        "sum_pref_all": 0.0,
        "sum_pref_sin": 0.0,
        "num_matched_sin": 0,
        "num_matched_sib": 0,
        "num_sin_total": 0,
        "num_sib_total": 0,
        "num_together": 0,
        "sep_none": 0,
        "sep_one": 0,
        "sep_both": 0,
        "top_pref": 0,
    }

    # Identify singletons vs siblings
    is_sib = {s: (len(siblings.get(s, [])) > 0) for s in students}
    out["num_sin_total"] = sum(1 for s in students if not is_sib[s])
    out["num_sib_total"] = sum(1 for s in students if is_sib[s])

    # Rank per student based on collapsed RBD pref
    def rbd_rank(s, c):
        """Rank of RBD(c) in s's preference list (collapsed RBDs)."""
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

    assigned_rbd = {}  # student -> rbd
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

    # Siblings together: student has ≥1 sibling at same RBD
    for s in assigned_rbd:
        rbd = assigned_rbd[s]
        for sib in siblings.get(s, []):
            if sib in assigned_rbd and assigned_rbd[sib] == rbd:
                out["num_together"] += 1
                break

    # Separated siblings breakdown (over families of size 2)
    # Uses the exact same definitions as ComputeOutputs in simulations.py.
    sep_none_set = set()
    sep_one_set = set()
    sep_both_set = set()

    def rbds_listed(s):
        if s not in pref:
            return set()
        return {pref[s][p].split("_")[0] for p in pref[s]}

    def rbds_up_to_assigned(s):
        """RBDs at or above s's assigned rank."""
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
                # both unassigned
                overlap = rbds_listed(id_s) & rbds_listed(sib)
                if overlap:
                    sep_none_set.add(id_s)
                    sep_none_set.add(sib)
            elif s_in and not sib_in:
                overlap = rbds_up_to_assigned(id_s) & rbds_listed(sib)
                if overlap:
                    sep_one_set.add(id_s)
                    sep_one_set.add(sib)
            elif not s_in and sib_in:
                overlap = rbds_listed(id_s) & rbds_up_to_assigned(sib)
                if overlap:
                    sep_one_set.add(id_s)
                    sep_one_set.add(sib)
            else:
                # both assigned
                if assigned_rbd[id_s] != assigned_rbd[sib]:
                    overlap = rbds_up_to_assigned(id_s) & rbds_up_to_assigned(sib)
                    if overlap:
                        sep_both_set.add(id_s)
                        sep_both_set.add(sib)

    out["sep_none"] = len(sep_none_set)
    out["sep_one"] = len(sep_one_set)
    out["sep_both"] = len(sep_both_set)

    # Derived: averages and percentages
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
# Run one method on one instance/lottery
# ============================================================

def run_one_method(method_name, inputs_basic, inputs_full, tb):
    """Returns outputs dict (status, x_opt, runtime, ...) or None on failure."""
    kind, runner = ALL_METHODS[method_name]

    try:
        if kind == "full":
            outputs = runner(inputs_full)
        elif kind == "basic":
            outputs = runner(inputs_basic)
        elif kind == "basic_tb":
            outputs = runner(inputs_basic, tb=tb)
        else:
            raise ValueError(f"Unknown input kind: {kind}")
    except Exception as e:
        print(f"  [ERROR] {method_name}: {e}")
        return None

    if outputs is None or outputs.get("status") != "completed":
        return None

    return outputs


# ============================================================
# Single simulation: load instance, draw lottery, run all methods
# ============================================================

def run_one_simulation(indat):
    """
    Executes one simulation for one region.
    indat = (region_indir, year, tie_breaker, sim_idx, methods_to_run)
    Returns: dict method_name -> stats dict
    """
    region_indir, year, tie_breaker, sim_idx, methods_to_run = indat

    # Set seed per sim for reproducibility
    np.random.seed(sim_idx + 1)
    random.seed(sim_idx + 1)

    instance_file = os.path.join(region_indir, str(year), "instance.txt")
    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn = \
        genin.read_instance(instance_file)

    # Draw fresh lottery
    pref_with_tb, tb = genin.modify_school_loterries(
        copy.deepcopy(pref), students, colleges, siblings, tie_breaker
    )

    # Recompute additional inputs after lottery
    students2, colleges2, Tp2, Tn2, Sp2, Sn2 = \
        genin.create_additional_inputs_from_instance(pref_with_tb, cap)

    inputs_basic = (students, colleges, pref_with_tb, cap, siblings,
                    levels, students_per_level)
    inputs_full = (students, colleges, pref_with_tb, cap, siblings,
                   levels, students_per_level, Tp2, Tn2, Sp2, Sn2, tb)

    sim_results = {}
    for method in methods_to_run:
        t0 = time.time()
        outputs = run_one_method(method, inputs_basic, inputs_full, tb)
        elapsed = time.time() - t0

        if outputs is None:
            sim_results[method] = {"solved": 0, "runtime": elapsed}
            print(f"  sim={sim_idx:3d}  {method:12s}  UNSOLVED    t={elapsed:.1f}s")
            continue

        stats = compute_stats(outputs["x_opt"], students, pref_with_tb, siblings)
        stats["runtime"] = outputs.get("runtime", elapsed)
        sim_results[method] = stats
        print(f"  sim={sim_idx:3d}  {method:12s}  "
              f"avg_pref={stats['avg_pref']:.2f}  "
              f"together={stats['num_together']:4d}  "
              f"t={stats['runtime']:.1f}s")

    return sim_results


# ============================================================
# Aggregate results across simulations
# ============================================================

STATS_FIELDS = [
    "avg_pref",        # Avg. Pref. (all)
    "top_pref",        # Top Pref.
    "num_unassigned",  # Unassigned
    "num_together",    # Together
    "sep_none",        # Separated - None
    "sep_one",         # Separated - One
    "sep_both",        # Separated - Both
    "avg_pref_sin",    # Avg. Pref. singletons (extra)
    "pct_sin_matched", # % singletons matched (extra)
    "pct_sib_matched", # % siblings matched (extra)
    "pct_sib_together",# % siblings matched together (extra)
    "runtime",
]


def aggregate(all_sim_results, methods):
    """
    all_sim_results : list of dict (one per sim) mapping method -> stats
    Returns DataFrame with one row per method: mean and SE over solved sims.
    """
    rows = []
    for method in methods:
        vals = {f: [] for f in STATS_FIELDS}
        solved_count = 0
        for sim_res in all_sim_results:
            if method not in sim_res:
                continue
            r = sim_res[method]
            if r.get("solved", 0) == 0:
                continue
            solved_count += 1
            for f in STATS_FIELDS:
                if f in r:
                    vals[f].append(r[f])

        row = {"Method": method, "Solved": solved_count}
        for f in STATS_FIELDS:
            arr = np.array(vals[f]) if len(vals[f]) > 0 else np.array([np.nan])
            row[f + "_mean"] = np.nanmean(arr)
            row[f + "_se"] = (np.nanstd(arr, ddof=1) / np.sqrt(len(arr))
                              if len(arr) > 1 else 0.0)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Output formatting
# ============================================================

def print_table(df, num_sims):
    """Print in a format close to the paper's Table 1."""
    cols = [
        ("Method",        "{:<12s}"),
        ("Solved",        "{:>6d}/{:d}"),
        ("Avg Pref",      "{:>6.3f}"),
        ("Top Pref Mean", "{:>10.2f}"),
        ("Top Pref SE",   "{:>6.2f}"),
        ("Unassigned M",  "{:>10.2f}"),
        ("Unassigned SE", "{:>6.2f}"),
        ("Together M",    "{:>10.2f}"),
        ("Together SE",   "{:>6.2f}"),
        ("Sep None M",    "{:>10.2f}"),
        ("Sep None SE",   "{:>6.2f}"),
        ("Sep One M",     "{:>10.2f}"),
        ("Sep One SE",    "{:>6.2f}"),
        ("Sep Both M",    "{:>10.2f}"),
        ("Sep Both SE",   "{:>6.2f}"),
        ("AvgPrefSin",    "{:>10.3f}"),
        ("%SinMatch",     "{:>10.2f}"),
        ("%SibMatch",     "{:>10.2f}"),
        ("%SibTogether",  "{:>10.2f}"),
    ]

    # Header
    header = (f"{'Method':<12s} {'Solved':>8s} {'AvgPref':>8s} "
              f"{'TopPrefM':>10s} {'TopPrefSE':>9s} "
              f"{'UnassgnM':>10s} {'UnassgnSE':>9s} "
              f"{'TogetherM':>10s} {'TogetherSE':>10s} "
              f"{'SepNoneM':>10s} {'SepNoneSE':>9s} "
              f"{'SepOneM':>9s} {'SepOneSE':>9s} "
              f"{'SepBothM':>10s} {'SepBothSE':>9s} "
              f"{'AvgPrefSin':>11s} "
              f"{'%SinMatch':>10s} {'%SibMatch':>10s} {'%SibTog':>10s}")
    print(header)
    print("-" * len(header))

    for _, r in df.iterrows():
        line = (f"{r['Method']:<12s} "
                f"{int(r['Solved']):>4d}/{num_sims:<3d} "
                f"{r['avg_pref_mean']:>8.3f} "
                f"{r['top_pref_mean']:>10.2f} {r['top_pref_se']:>9.2f} "
                f"{r['num_unassigned_mean']:>10.2f} {r['num_unassigned_se']:>9.2f} "
                f"{r['num_together_mean']:>10.2f} {r['num_together_se']:>10.2f} "
                f"{r['sep_none_mean']:>10.2f} {r['sep_none_se']:>9.2f} "
                f"{r['sep_one_mean']:>9.2f} {r['sep_one_se']:>9.2f} "
                f"{r['sep_both_mean']:>10.2f} {r['sep_both_se']:>9.2f} "
                f"{r['avg_pref_sin_mean']:>11.3f} "
                f"{r['pct_sin_matched_mean']:>10.2f} "
                f"{r['pct_sib_matched_mean']:>10.2f} "
                f"{r['pct_sib_together_mean']:>10.2f}")
        print(line)


def save_latex(df, outfile, num_sims, caption="Comparison of Benchmarks and Heuristics",
               benchmark_methods=None):
    """
    Save LaTeX output with TWO tables:
      1. Main comparison table (rotated 90 degrees as a sideways table).
      2. Additional metrics table (normal orientation).

    A midrule is inserted between benchmark methods and heuristic methods.
    Requires LaTeX packages: booktabs, rotating, float.
    """
    if benchmark_methods is None:
        benchmark_methods = set()
    else:
        benchmark_methods = set(benchmark_methods)

    def iter_with_midrule(df):
        """Yield rows, inserting a separator flag between benchmarks and heuristics."""
        prev_is_benchmark = None
        for _, r in df.iterrows():
            is_benchmark = r["Method"] in benchmark_methods
            need_midrule = (prev_is_benchmark is True and not is_benchmark)
            prev_is_benchmark = is_benchmark
            yield r, need_midrule

    with open(outfile, "w") as f:
        # ============================================================
        # Table 1: Main comparison (rotated / sideways)
        # ============================================================
        f.write("% Requires: \\usepackage{booktabs, rotating, float}\n\n")
        f.write("\\begin{sidewaystable}\n\\centering\n")
        f.write(f"\\caption{{{caption} ({num_sims} simulations)}}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lrrrrrrrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write(" & & & \\multicolumn{2}{c}{Top Pref.} & "
                "\\multicolumn{2}{c}{Unassigned} & "
                "\\multicolumn{2}{c}{Together} & "
                "\\multicolumn{2}{c}{Sep-None} & "
                "\\multicolumn{2}{c}{Sep-One} & "
                "\\multicolumn{2}{c}{Sep-Both} \\\\\n")
        f.write("\\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9} "
                "\\cmidrule(lr){10-11} \\cmidrule(lr){12-13} \\cmidrule(lr){14-15}\n")
        f.write("Method & Solved & AvgPref & Mean & SE & Mean & SE & Mean & SE "
                "& Mean & SE & Mean & SE & Mean & SE \\\\\n")
        f.write("\\midrule\n")
        for r, need_midrule in iter_with_midrule(df):
            if need_midrule:
                f.write("\\midrule\n")
            f.write(
                f"{r['Method']} & "
                f"{int(r['Solved'])} & "
                f"{r['avg_pref_mean']:.3f} & "
                f"{r['top_pref_mean']:.2f} & {r['top_pref_se']:.2f} & "
                f"{r['num_unassigned_mean']:.2f} & {r['num_unassigned_se']:.2f} & "
                f"{r['num_together_mean']:.2f} & {r['num_together_se']:.2f} & "
                f"{r['sep_none_mean']:.2f} & {r['sep_none_se']:.2f} & "
                f"{r['sep_one_mean']:.2f} & {r['sep_one_se']:.2f} & "
                f"{r['sep_both_mean']:.2f} & {r['sep_both_se']:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\\end{sidewaystable}\n\n")

        # ============================================================
        # Table 2: Additional metrics (normal orientation)
        # ============================================================
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write(f"\\caption{{Additional metrics -- {caption} ({num_sims} simulations)}}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Method & AvgPrefSin & \\% Sin Matched & "
                "\\% Sib Matched & \\% Sib Together \\\\\n")
        f.write("\\midrule\n")
        for r, need_midrule in iter_with_midrule(df):
            if need_midrule:
                f.write("\\midrule\n")
            f.write(
                f"{r['Method']} & "
                f"{r['avg_pref_sin_mean']:.3f} & "
                f"{r['pct_sin_matched_mean']:.2f} & "
                f"{r['pct_sib_matched_mean']:.2f} & "
                f"{r['pct_sib_together_mean']:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\\end{table}\n")


# ============================================================
# Main driver
# ============================================================

def run_simulations(
    region="Magallanes",
    year=2023,
    tie_breaker="mtbf",
    num_sims=10,
    num_cores=1,
    methods=None,
    base_indir=None,
    outdir=None,
):
    if methods is None:
        methods = list(ALL_METHODS.keys())

    if base_indir is None:
        base_indir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "R", "intermediate_data")
    if outdir is None:
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "outputs", "new_simulations")
    os.makedirs(outdir, exist_ok=True)

    region_indir = os.path.join(base_indir, region)

    # Build per-sim tasks
    tasks = [(region_indir, year, tie_breaker, sim, methods)
             for sim in range(num_sims)]

    print(f"\n=== Running {num_sims} simulations on {region} {year} ===")
    print(f"Tie-breaker: {tie_breaker}")
    print(f"Methods: {methods}")
    print(f"Cores: {num_cores}\n")

    if num_cores == 1:
        all_sim_results = [run_one_simulation(t) for t in tasks]
    else:
        nproc = min(num_cores, multiprocessing.cpu_count())
        pool = multiprocessing.Pool(processes=min(nproc, len(tasks)))
        all_sim_results = pool.map(run_one_simulation, tasks)
        pool.close()
        pool.join()

    # Aggregate
    df = aggregate(all_sim_results, methods)

    # Save raw CSV of all sim results
    raw_rows = []
    for sim_idx, sim_res in enumerate(all_sim_results):
        for method, stats in sim_res.items():
            raw_rows.append({"sim": sim_idx, "method": method, **stats})
    raw_df = pd.DataFrame(raw_rows)

    raw_csv = os.path.join(outdir,
        f"{region}_{year}_{tie_breaker}_raw_{num_sims}sims.csv")
    summary_csv = os.path.join(outdir,
        f"{region}_{year}_{tie_breaker}_summary_{num_sims}sims.csv")
    latex_out = os.path.join(outdir,
        f"{region}_{year}_{tie_breaker}_table_{num_sims}sims.tex")

    raw_df.to_csv(raw_csv, index=False)
    df.to_csv(summary_csv, index=False)
    save_latex(df, latex_out, num_sims,
               caption=f"Comparison on {region} {year} (tie-breaker: {tie_breaker})",
               benchmark_methods=set(BENCHMARK_METHODS.keys()))

    print("\n=== Results ===\n")
    print_table(df, num_sims)
    print(f"\nRaw data:  {raw_csv}")
    print(f"Summary:   {summary_csv}")
    print(f"LaTeX:     {latex_out}\n")

    return df, raw_df


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    REGION = "OHiggins" # "Magallanes"
    YEAR = 2023
    TIE_BREAKER = "mtbf"
    NUM_SIMS = 2
    NUM_CORES = 1

    # Subset of methods (leave as None to run all 13)
    METHODS = [
        # --- Paper benchmarks ---
        "SOSM",
        "FOSM",
        "Descending",
        "Ascending",
        "Soft",
        "Hard",
        "Hybrid-310",
        "Hybrid-320",
        "Hard-NTB",
        # --- New heuristics from mechanisms.py ---
        "DA",
        "FSD-U",
        "SF-Reserve",
        "SF-SD",
        "SF-DA",
        "SF-AP-DA",
        "FSD",
        "FSD-A",
    ]
    # e.g. METHODS = ["SOSM", "DA", "FSD-U", "SF-Reserve", "SF-SD", "SF-DA"]

    run_simulations(
        region=REGION,
        year=YEAR,
        tie_breaker=TIE_BREAKER,
        num_sims=NUM_SIMS,
        num_cores=NUM_CORES,
        methods=METHODS,
    )