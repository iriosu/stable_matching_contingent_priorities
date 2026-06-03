"""
compare_heuristics.py
=====================
Run Fede's heuristics (heuristics_v2) and Ignacio's algorithms
(algorithms_ignacio) on the same Magallanes seeds, verify every output with
acs_verifier.check_acs (exclude_sibling_envy=True), and write a grouped LaTeX
table in which each pair of "equivalent" mechanisms is adjacent, separated
from the next pair by a double midrule.

Setup
-----
  * Save Ignacio's algorithms-2.py to  python/algorithms_ignacio.py
    (he imports generate_inputs and solve_opt, which you already have).
  * From python/ run:
        python compare_heuristics.py

Outputs land in
        <python>/../outputs/comparison/<date>/
            rows.csv         per-sim per-method records
            summary.csv      aggregated per method
            comparison.tex   the grouped LaTeX table
"""

import os
import sys
import time
import copy
import random
import datetime
import math
import numpy as np
import pandas as pd

import heuristics as hv2
import algorithms_ignacio as alg_ig
import acs_verifier
import acs_priority as P
import generate_inputs as genin
import simulations_acs as S         # reuse compute_stats, x_opt_to_mu


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_BASE   = os.path.join(SCRIPT_DIR, "..", "outputs", "comparison",
                          datetime.date.today().isoformat())

REGION       = "Magallanes"
YEAR         = 2023
TIE_BREAKER  = "mtbf"
REGION_ROOT  = "../R/intermediate_data"
NUM_SIMS     = 100
EXCLUDE_SIB  = True        # ACS convention: family-as-unit


# ============================================================
# Adapters — every adapter returns {status, mu, runtime}
# ============================================================

def _mu_from(out, students):
    """Accept either {'mu': ...} or {'x_opt': ...} and return a uniform mu."""
    if "mu" in out and out["mu"] is not None:
        return {s: out["mu"].get(s) for s in students}
    return S.x_opt_to_mu(out.get("x_opt", {}), students)


def _wrap_ig(out, students, t0):
    return {"status":   out.get("status", "completed"),
            "mu":       _mu_from(out, students),
            "x_opt":    out.get("x_opt", {}),
            "runtime":  out.get("runtime", time.time() - t0)}


def _wrap_fede(out, students, t0):
    return {"status":   out.get("status", "completed"),
            "mu":       _mu_from(out, students),
            "x_opt":    out.get("x_opt", {}),
            "runtime":  out.get("runtime", time.time() - t0)}


# ----- Fede's heuristics ------------------------------------

def fede_desc_da   (inp, tb=None): return _wrap_fede(hv2.descending_da(inp,  tb=tb), inp[0], 0)
def fede_desc_fsda (inp, tb=None): return _wrap_fede(hv2.descending_fsda(inp, tb=tb), inp[0], 0)
def fede_asc_da    (inp, tb=None): return _wrap_fede(hv2.ascending_da(inp,   tb=tb), inp[0], 0)
def fede_asc_fsda  (inp, tb=None): return _wrap_fede(hv2.ascending_fsda(inp, tb=tb), inp[0], 0)
def fede_sim       (inp, tb=None): return _wrap_fede(hv2.simultaneous(inp,    tb=tb), inp[0], 0)
def fede_fp        (inp, tb=None): return _wrap_fede(hv2.fsda_single(inp,     tb=tb), inp[0], 0)
def fede_LS_DA     (inp, tb=None): return _wrap_fede(hv2.LS_DA(inp,           tb=tb), inp[0], 0)
def fede_LS_nd     (inp, tb=None): return _wrap_fede(hv2.LS_nd(inp,           tb=tb), inp[0], 0)
def fede_SL_DA     (inp, tb=None): return _wrap_fede(hv2.SL_DA(inp,           tb=tb), inp[0], 0)
def fede_SL_nd     (inp, tb=None): return _wrap_fede(hv2.SL_nd(inp,           tb=tb), inp[0], 0)


# ----- Ignacio's algorithms ---------------------------------

def _ig_seq(inp, reverse):
    students, colleges, pref, cap, siblings, levels, spl = inp
    t0 = time.time()
    out = alg_ig.Sequential(
        (students, colleges, pref, cap, siblings, levels, spl),
        [str(i) for i in sorted(range(-1, 13), reverse=reverse)])
    return _wrap_ig(out, students, t0)


def _ig_seqblock(inp, reverse):
    students, colleges, pref, cap, siblings, levels, spl = inp
    t0 = time.time()
    out = alg_ig.SequentialBlock(
        (students, colleges, pref, cap, siblings, levels, spl),
        [str(i) for i in sorted(range(-1, 13), reverse=reverse)])
    return _wrap_ig(out, students, t0)


def ig_seq_desc      (inp, tb=None): return _ig_seq(inp, reverse=True)
def ig_seq_asc       (inp, tb=None): return _ig_seq(inp, reverse=False)
def ig_seqblock_desc (inp, tb=None): return _ig_seqblock(inp, reverse=True)
def ig_seqblock_asc  (inp, tb=None): return _ig_seqblock(inp, reverse=False)


def _ig_sim(inp, decay):
    students, colleges, pref, cap, siblings, _, _ = inp
    t0 = time.time()
    if decay is None:
        out = alg_ig.Simultaneous((students, colleges, pref, cap, siblings))
    else:
        out = alg_ig.Simultaneous((students, colleges, pref, cap, siblings), decay=decay)
    return _wrap_ig(out, students, t0)


def ig_sim   (inp, tb=None): return _ig_sim(inp, decay=None)
def ig_rada  (inp, tb=None): return _ig_sim(inp, decay=1)        # non-monotone


def _ig_size(inp, direction, fix):
    students, colleges, pref, cap, siblings, _, _ = inp
    t0 = time.time()
    out = alg_ig.SizeSequential(
        (students, colleges, pref, cap, siblings),
        direction=direction, fix=fix)
    return _wrap_ig(out, students, t0)


def ig_lsda      (inp, tb=None): return _ig_size(inp, "decreasing", fix=True)
def ig_size_desc (inp, tb=None): return _ig_size(inp, "decreasing", fix=False)
def ig_slda      (inp, tb=None): return _ig_size(inp, "increasing", fix=True)
def ig_size_asc  (inp, tb=None): return _ig_size(inp, "increasing", fix=False)


# ============================================================
# Pairings  (group_label, fede_fn, ignacio_fn)
# ============================================================

PAIRS = [
    ("Descending (DA inner)",           fede_desc_da,    ig_seq_desc),
    ("Ascending (DA inner)",            fede_asc_da,     ig_seq_asc),
    ("Descending (FSD-A inner)",        fede_desc_fsda,  ig_seqblock_desc),
    ("Ascending (FSD-A inner)",         fede_asc_fsda,   ig_seqblock_asc),
    ("Simultaneous (monotone)",         fede_sim,        ig_sim),
    ("Fixed point (non-monotone)",      fede_fp,         ig_rada),
    ("LS (size $\\downarrow$, decrement)",       fede_LS_DA,      ig_lsda),
    ("LS-nd (size $\\downarrow$, no decrement)", fede_LS_nd,      ig_size_desc),
    ("SL (size $\\uparrow$, decrement)",         fede_SL_DA,      ig_slda),
    ("SL-nd (size $\\uparrow$, no decrement)",   fede_SL_nd,      ig_size_asc),
]


# ============================================================
# Per-sim setup
# ============================================================

def load_sim(sim):
    np.random.seed(sim + 1)
    random.seed(sim + 1)
    region_indir  = os.path.join(REGION_ROOT, REGION)
    instance_file = os.path.join(region_indir, str(YEAR), "instance.txt")
    (students, colleges, pref, cap, siblings, levels,
     students_per_level, Tp, Tn, Sp, Sn) = genin.read_instance(instance_file)
    pref_tb, tb = genin.modify_school_loterries(
        copy.deepcopy(pref), students, colleges, siblings, TIE_BREAKER)
    inputs_basic = (students, colleges, pref_tb, cap, siblings, levels,
                    students_per_level)
    levels_of = P.build_levels_of(students, pref_tb)
    return inputs_basic, tb, levels_of


def evaluate(mu, x_opt, inputs_basic, tb, levels_of):
    students, colleges, pref_tb, cap, siblings, _, _ = inputs_basic
    res = acs_verifier.check_acs(
        mu, students, colleges, pref_tb, cap, siblings, levels_of,
        tb=tb, exclude_sibling_envy=EXCLUDE_SIB)
    stats = S.compute_stats(x_opt, students, pref_tb, siblings)
    return res, stats


# ============================================================
# Main loop
# ============================================================

import traceback

def main(num_sims=NUM_SIMS):
    os.makedirs(OUT_BASE, exist_ok=True)
    print(f"Output dir: {os.path.abspath(OUT_BASE)}")
    print(f"Sims: {num_sims}  ACS convention: exclude_sibling_envy={EXCLUDE_SIB}\n")
    err_log = open(os.path.join(OUT_BASE, "errors.log"), "w")

    rows = []
    for sim in range(num_sims):
        try:
            inputs_basic, tb, levels_of = load_sim(sim)
        except Exception as e:
            print(f"sim {sim}: load failed — {e}")
            continue
        students = inputs_basic[0]

        for group, fede_fn, ig_fn in PAIRS:
            for who, fn in [("Fede", fede_fn), ("Ignacio", ig_fn)]:
                t0 = time.time()
                try:
                    out    = fn(inputs_basic, tb=tb)
                    mu     = out["mu"]
                    x_opt  = out["x_opt"]
                    rt     = out.get("runtime", time.time() - t0)
                    res, stats = evaluate(mu, x_opt, inputs_basic, tb, levels_of)
                    rows.append({
                        "sim": sim, "group": group, "who": who,
                        "method": fn.__name__, "solved": 1,
                        "is_acs":  int(res["is_acs"]),
                        "n_blocking":      res["n_blocking_pairs"],
                        "avg_pref":        stats["avg_pref"],
                        "top_pref":        stats["top_pref"],
                        "num_unassigned":  stats["num_unassigned"],
                        "num_together":    stats["num_together"],
                        "pct_sib_together":stats["pct_sib_together"],
                        "sep_none":        stats["sep_none"],
                        "sep_one":         stats["sep_one"],
                        "sep_both":        stats["sep_both"],
                        "runtime":         rt,
                    })
                    print(f"sim {sim:3d}  {who:7s}  {fn.__name__:18s}  "
                          f"is_acs={res['is_acs']!s:5s}  "
                          f"bp={res['n_blocking_pairs']:>4d}  "
                          f"tog={stats['num_together']:>6.0f}  rt={rt:.2f}s")
                except Exception as e:
                    err_log.write(f"\n\n=== sim {sim}  {who}:{fn.__name__} ===\n")
                    err_log.write(traceback.format_exc())
                    err_log.flush()
                    rows.append({"sim": sim, "group": group, "who": who,
                                 "method": fn.__name__, "solved": 0,
                                 "runtime": time.time() - t0,
                                 "error": f"{type(e).__name__}: {e}"})
                    print(f"sim {sim:3d}  {who:7s}  {fn.__name__:18s}  "
                          f"FAILED: {type(e).__name__}: {e}")

    err_log.close()
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_BASE, "rows.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {os.path.abspath(csv_path)}")

    write_summary_and_latex(df)


# ============================================================
# Aggregation + grouped LaTeX
# ============================================================

def write_summary_and_latex(df):
    df_ok = df[df["solved"] == 1].copy()
    summary_rows = []
    for group, fede_fn, ig_fn in PAIRS:
        for who, fn in [("Fede", fede_fn), ("Ignacio", ig_fn)]:
            sub = df_ok[df_ok["method"] == fn.__name__]
            if len(sub) == 0:
                summary_rows.append({"group": group, "who": who,
                                     "method": fn.__name__, "solved": 0})
                continue
            non_acs = sub[sub["is_acs"] == 0]
            summary_rows.append({
                "group":   group, "who": who, "method": fn.__name__,
                "solved":  len(sub),
                "pct_acs": 100.0 * sub["is_acs"].mean(),
                "avg_block_nonacs":
                    non_acs["n_blocking"].mean() if len(non_acs) else float("nan"),
                "avg_pref":         sub["avg_pref"].mean(),
                "num_together":     sub["num_together"].mean(),
                "pct_sib_together": sub["pct_sib_together"].mean(),
                "runtime":          sub["runtime"].mean(),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_BASE, "summary.csv"), index=False)
    print(f"Wrote {os.path.abspath(os.path.join(OUT_BASE, 'summary.csv'))}")

    tex_path = os.path.join(OUT_BASE, "comparison.tex")
    with open(tex_path, "w") as fh:
        fh.write(r"\begin{table}[t]" + "\n")
        fh.write(r"  \centering" + "\n")
        fh.write(r"  \caption{Algos comparison by Federico and Ignacio: paired heuristics, Magallanes."
                 r" \%ACS and Avg.\ Block. use exclude\_sibling\_envy=True.}"
                 + "\n")
        fh.write(r"  \label{tab:federico_ignacio}" + "\n")
        fh.write(r"  \begin{tabular}{llccccccc}" + "\n")
        fh.write(r"    \toprule" + "\n")
        fh.write(r"    \midrule" + "\n")
        fh.write(r"    Mechanism & Code & Solved & \%ACS & Avg.\ Block. "
                 r"& Avg.\ Pref. & Together & \%Tog. & Time \\" + "\n")
        fh.write(r"              &      &        &       & (non-ACS)    "
                 r"&             &          &              &    (s)       \\" + "\n")

        for group, fede_fn, ig_fn in PAIRS:
            fh.write(r"    \midrule\midrule" + "\n")
            for who, fn in [("Fede", fede_fn), ("Ignacio", ig_fn)]:
                rec = next((r for r in summary_rows
                            if r["method"] == fn.__name__), None)
                grp = group if who == "Fede" else ""
                if rec is None or rec.get("solved", 0) == 0:
                    fh.write(rf"    {grp} & {who} & 0 & -- & -- & -- & -- & -- & -- \\"
                             + "\n")
                    continue
                block = ("--" if (rec.get("avg_block_nonacs") is None or
                                  math.isnan(rec["avg_block_nonacs"]))
                         else f"{rec['avg_block_nonacs']:.1f}")
                fh.write(
                    f"    {grp} & {who} & {rec['solved']} & "
                    f"{rec['pct_acs']:.1f} & {block} & "
                    f"{rec['avg_pref']:.3f} & "
                    f"{rec['num_together']:.1f} & "
                    f"{rec['pct_sib_together']:.1f} & "
                    f"{rec['runtime']:.1f} \\\\" + "\n")

        fh.write(r"    \midrule\bottomrule" + "\n")
        fh.write(r"  \end{tabular}" + "\n")
        fh.write(r"\end{table}" + "\n")
    print(f"Wrote {os.path.abspath(tex_path)}")


if __name__ == "__main__":
    main()
