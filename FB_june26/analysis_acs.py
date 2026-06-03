"""
analysis_acs.py
===============
Reads the per-(sim,method) CSV produced by simulations_acs.py and writes three
LaTeX tables to  <out_root>/new_simulations/<date>/ :

  1. <name>_acs_summary.tex   : ACS rate, mean blocking pairs (over NON-ACS
                                instances only), avg pref, together, runtime.
  2. <name>_descriptive_avgpref.tex : Mean/SE of Avg.Pref, Unassigned, Together,
                                      Separated (None/One/Both).
  3. <name>_descriptive_toppref.tex : same, with Top Pref. instead of Avg.Pref.

The table NAME (used in filenames, captions, labels) and the region are inputs.

Usage:
  python analysis_acs.py rows.csv --name magallanes_acs --region Magallanes
"""

import os
import sys
import math
import datetime
import argparse
import pandas as pd


# pretty labels + display order; the two BENCHMARKS are Absolute-Hard and Descending
LABELS = {
    "Hard":            "Absolute - Hard",
    "descending_da":   "Descending",
    "DA":              "SOSM (DA)",
    "FOSM":            "FOSM",
    "Soft":            "Absolute - Soft",
    "Hybrid-310":      "Hybrid ($\\zeta{=}310$)",
    "Hybrid-320":      "Hybrid ($\\zeta{=}320$)",
    "Hard-NTB":        "Absolute - Hard (NTB)",
    "fsda_single":     "FSD-A (single)",
    "simultaneous":    "Simultaneous",
    "descending_fsda": "Descending (FSD-A)",
    "ascending_da":    "Ascending",
    "ascending_fsda":  "Ascending (FSD-A)",
    "LS":              "LS",
    "LS_DA":           "LS-DA",
    "LS_nd":           "LS-nd",
    "SL":              "SL",
    "SL_DA":           "SL-DA",
    "SL_nd":           "SL-nd",
}

DEFAULT_ORDER = [
    "Hard", "descending_da",                 # the two benchmarks, first
    "DA", "FOSM", "Soft", "Hybrid-310", "Hybrid-320", "Hard-NTB",
    "fsda_single", "simultaneous",
    "ascending_da", "ascending_fsda", "descending_fsda",
    "LS", "LS_DA", "LS_nd", "SL", "SL_DA", "SL_nd",
]


def _mean_se(series):
    s = series.dropna()
    n = len(s)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = s.mean()
    se = (s.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return (mean, se)


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def _label(m):
    return LABELS.get(m, m.replace("_", "\\_"))


def summarize(df, methods):
    """Return a dict method -> stats dict."""
    out = {}
    for m in methods:
        sub = df[df["method"] == m]
        solved = int((sub["solved"] == 1).sum())
        sub_ok = sub[sub["solved"] == 1]
        if solved == 0:
            out[m] = {"solved": 0}
            continue
        n_sims = len(sub_ok)
        acs_rate = 100.0 * sub_ok["is_acs"].mean() if "is_acs" in sub_ok else float("nan")
        # mean blocking over NON-ACS instances only
        non_acs = sub_ok[sub_ok["is_acs"] == 0]
        mean_block_nonacs = non_acs["n_blocking"].mean() if len(non_acs) else float("nan")
        n_nonacs = len(non_acs)

        rec = {
            "solved": solved,
            "n_sims": n_sims,
            "acs_rate": acs_rate,
            "mean_block_nonacs": mean_block_nonacs,
            "n_nonacs": n_nonacs,
            "runtime": sub_ok["runtime"].mean(),
        }
        for col in ["avg_pref", "top_pref", "num_unassigned", "num_together",
                    "pct_sib_together", "sep_none", "sep_one", "sep_both"]:
            if col in sub_ok.columns:
                rec[col] = _mean_se(sub_ok[col])
            else:
                rec[col] = (float("nan"), float("nan"))
        out[m] = rec
    return out


# ============================================================
# LaTeX table writers
# ============================================================

def table_acs_summary(stats, methods, name, region):
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"    \centering")
    L.append(rf"    \caption{{Absolute Contingent Stability: {region}.}}\label{{tab:{name}_acs}}")
    L.append(r"    \begin{tabular}{lccccccc}")
    L.append(r"        \toprule")
    L.append(r"        Method & Solved & \%ACS & Avg.\ Block. & Avg.\ Pref. & Together & \%Sib.\ Tog. & Time (s) \\")
    L.append(r"               &        &       & (non-ACS)    &             &          &             &          \\")
    L.append(r"        \midrule")
    for m in methods:
        st = stats.get(m, {})
        if st.get("solved", 0) == 0:
            L.append(rf"        {_label(m)} & 0 & -- & -- & -- & -- & -- & -- \\")
            continue
        block = ("--" if st["n_nonacs"] == 0 else _fmt(st["mean_block_nonacs"], 1))
        row = (f"        {_label(m)} & {st['solved']} & "
               f"{_fmt(st['acs_rate'],1)} & {block} & "
               f"{_fmt(st['avg_pref'][0],3)} & "
               f"{_fmt(st['num_together'][0],1)} & "
               f"{_fmt(st['pct_sib_together'][0],1)} & "
               f"{_fmt(st['runtime'],1)} \\\\")
        L.append(row)
    L.append(r"        \bottomrule")
    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


def _descriptive(stats, methods, name, region, first_col_key, first_col_head,
                 first_nd=2):
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"    \centering")
    L.append(rf"    \caption{{Assignment outcomes: {region} ({first_col_head}).}}"
             rf"\label{{tab:{name}_{first_col_key}}}")
    L.append(r"    \scalebox{0.85}{\begin{tabular}{lccccccccccccc}")
    L.append(r"        \toprule")
    L.append(r"        & & \multicolumn{6}{c}{} & \multicolumn{6}{c}{Separated} \\")
    L.append(r"        \cmidrule(lr){9-14}")
    L.append(rf"        & Solved & \multicolumn{{2}}{{c}}{{{first_col_head}}} & "
             r"\multicolumn{2}{c}{Unassigned} & \multicolumn{2}{c}{Together} & "
             r"\multicolumn{2}{c}{None} & \multicolumn{2}{c}{One} & \multicolumn{2}{c}{Both} \\")
    L.append(r"        \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}"
             r"\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}")
    L.append(r"        & & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE \\")
    L.append(r"        \midrule")
    for m in methods:
        st = stats.get(m, {})
        if st.get("solved", 0) == 0:
            L.append(rf"        {_label(m)} & 0 & " + " & ".join(["--"] * 12) + r" \\")
            continue
        cells = []
        fc = st[first_col_key]
        cells += [_fmt(fc[0], first_nd), _fmt(fc[1], first_nd)]
        for key in ["num_unassigned", "num_together", "sep_none", "sep_one", "sep_both"]:
            mse = st[key]
            cells += [_fmt(mse[0], 1), _fmt(mse[1], 1)]
        L.append(rf"        {_label(m)} & {st['solved']} & " + " & ".join(cells) + r" \\")
    L.append(r"        \bottomrule")
    L.append(r"    \end{tabular}}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_out = os.path.join(_script_dir, "..", "outputs")
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="per-(sim,method) CSV from simulations_acs.py")
    ap.add_argument("--name", default="acs_table",
                    help="table name: used in filenames, captions, labels")
    ap.add_argument("--region", default="Magallanes")
    ap.add_argument("--out-root", default=_default_out,
                    help="root; tables go to <out-root>/new_simulations/<date>/ "
                         "(default: sibling 'outputs' of the python folder)")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    present = [m for m in DEFAULT_ORDER if m in set(df["method"])]
    extra = [m for m in df["method"].unique() if m not in DEFAULT_ORDER]
    methods = present + sorted(extra)

    stats = summarize(df, methods)

    out_dir = os.path.join(args.out_root, "new_simulations", args.date)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {os.path.abspath(out_dir)}")

    t1 = table_acs_summary(stats, methods, args.name, args.region)
    t2 = _descriptive(stats, methods, args.name, args.region,
                      "avg_pref", "Avg.\\ Pref.", first_nd=3)
    t3 = _descriptive(stats, methods, args.name, args.region,
                      "top_pref", "Top Pref.", first_nd=1)

    for fn, content in [(f"{args.name}_acs_summary.tex", t1),
                        (f"{args.name}_descriptive_avgpref.tex", t2),
                        (f"{args.name}_descriptive_toppref.tex", t3)]:
        path = os.path.join(out_dir, fn)
        with open(path, "w") as fh:
            fh.write(content)
        print(f"wrote {path}")

    # echo the headline table to stdout
    print("\n" + "=" * 70)
    print("ACS SUMMARY (also written to .tex):")
    print("=" * 70)
    print(t1)
    return out_dir


if __name__ == "__main__":
    main()