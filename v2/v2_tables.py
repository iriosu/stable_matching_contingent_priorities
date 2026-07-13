r"""
v2_tables.py

Render aggregated metrics as the paper's tables and write the CSVs.

  render_main       main table: row | Solved | ACS % | Block % | Avg. Pref. |
                    Top Pref. | Unassigned | Together | Separated None/One/Both
                    (Mean, SE for the paired columns). Uses \scriptsize and a
                    tight \tabcolsep so the widened table still fits; a
                    \resizebox alternative is noted inline.
  render_split      sibling split with Siblings and No-Siblings side by side as
                    COLUMN groups (one row per method), not stacked rows.
  hybrid_frontier   text summary of the zeta grid search for hybrid, plus a
                    suggested best zeta.
  write_agg_csv     one row per method with every aggregated column.
  write_rows_csv    the raw per-(draw, method) rows (union of keys).

ACS % is agg["acs_pct"] (percentage of solved matchings that pass the absolute
ACS checker). Block % is agg["blocking_pct_mean"] (mean over solved matchings
of the number of blocking pairs as a percentage of the number of students).
"""

from __future__ import annotations

import csv
from typing import Dict, List, Optional


def _f(v, nd=2):
    if v is None or v != v:
        return "--"
    return f"{v:.{nd}f}"


def _pct(v, nd=1):
    """v is a fraction in [0, 1]; render as a percentage."""
    if v is None or v != v:
        return "--"
    return f"{100 * v:.{nd}f}"


# --------------------------------------------------------------------------
def render_main(agg: Dict[str, Dict[str, float]], order: List[str],
                caption: str, label: str,
                display: Optional[Dict[str, str]] = None) -> str:
    display = display or {}
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        # If it still overflows, wrap the tabular:
        #   \resizebox{\textwidth}{!}{ ... }   (needs \usepackage{graphicx})
        r"\begin{tabular}{l r r r r rr rr rr rr rr rr}",
        r"\toprule",
        r" & & & & & \multicolumn{2}{c}{Top Pref.} & \multicolumn{2}{c}{Unassigned}"
        r" & \multicolumn{2}{c}{Together} & \multicolumn{6}{c}{Separated} \\",
        r"\cmidrule(lr){12-17}",
        r" & & & & & & & & & & & \multicolumn{2}{c}{None} & \multicolumn{2}{c}{One}"
        r" & \multicolumn{2}{c}{Both} \\",
        r" & Solved & ACS \% & Block \% & Avg.\ Pref. & Mean & SE & Mean & SE"
        r" & Mean & SE & Mean & SE & Mean & SE & Mean & SE \\",
        r"\midrule",
    ]
    for name in order:
        a = agg.get(name)
        if a is None:
            continue
        cells = [display.get(name, name), str(int(a.get("solved", 0))),
                 _f(a.get("acs_pct"), 0), _f(a.get("blocking_pct_mean"), 2),
                 _f(a.get("avg_pref_mean"), 3)]
        for k in ("top_pref", "unassigned", "together",
                  "sep_none", "sep_one", "sep_both"):
            cells += [_f(a.get(f"{k}_mean")), _f(a.get(f"{k}_se"))]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
def render_split(agg: Dict[str, Dict[str, float]], order: List[str],
                 caption: str, label: str,
                 display: Optional[Dict[str, str]] = None) -> str:
    """Table 13: Siblings and No-Siblings side by side, plus an Overall group
    giving total matched students and how that compares to DA (SOSM) and
    Descending (positive = matching MORE students), plus mean solve time."""
    display = display or {}
    da = agg.get("SOSM", {}).get("matched_mean")
    desc = agg.get("Descending", {}).get("matched_mean")

    def delta(v, base):
        if v is None or base is None:
            return "--"
        return f"{v - base:+.1f}"

    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{l rrr rrr rrr r r}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Siblings} & \multicolumn{3}{c}{No-Siblings}"
        r" & \multicolumn{3}{c}{Overall} & & \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        r" & Avg.\ Pref. & Top [\%] & Unassg.\ [\%]"
        r" & Avg.\ Pref. & Top [\%] & Unassg.\ [\%]"
        r" & Matched & vs DA & vs Desc & Iters & Time [s] \\",
        r"\midrule",
    ]
    for name in order:
        a = agg.get(name)
        if a is None:
            continue
        m = a.get("matched_mean")
        cells = [display.get(name, name),
                 _f(a.get("avg_pref_sib_mean"), 3),
                 _pct(a.get("top_frac_sib_mean")),
                 _pct(a.get("unassg_frac_sib_mean")),
                 _f(a.get("avg_pref_nosib_mean"), 3),
                 _pct(a.get("top_frac_nosib_mean")),
                 _pct(a.get("unassg_frac_nosib_mean")),
                 _f(m, 1), delta(m, da), delta(m, desc),
                 _f(a.get("iters_converged_mean"), 1),
                 _f(a.get("runtime_all_mean", a.get("runtime_mean")), 2)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
def hybrid_frontier(agg: Dict[str, Dict[str, float]], zetas: List[int]) -> str:
    """Text summary of the hybrid grid search over zeta (sum z >= zeta): for
    each zeta the solved count, the proven-OPTIMAL count (a time-limited
    incumbent counts as solved but NOT optimal), mean rank, mean together,
    ACS %, and mean blocking %. The suggested best zeta is the largest zeta
    proven optimal on every draw, ties broken toward more siblings together;
    if none is optimal on every draw it falls back to feasible-on-every-draw
    and says so. Printed to the run log; edit the criterion if you want a
    different operating point."""
    frontier = []
    n_draws = None
    for z in zetas:
        a = agg.get(f"Hybrid-{z}")
        if a is None:
            continue
        n_draws = a.get("n_draws", n_draws)
        frontier.append((z, a.get("solved", 0), a.get("optimal", 0),
                         a.get("avg_pref_mean"), a.get("together_mean"),
                         a.get("acs_pct"), a.get("blocking_pct_mean")))
    if not frontier:
        return "[hybrid] no Hybrid-<zeta> rows found"
    out = ["[hybrid] grid search over zeta (constraint: sum z >= zeta):",
           f"  {'zeta':>6} {'solved':>7} {'optimal':>8} {'avg.pref':>9} "
           f"{'together':>9} {'ACS%':>6} {'block%':>7}"]

    def cell(v, w, nd):
        return f"{'--':>{w}}" if v is None else f"{v:>{w}.{nd}f}"

    for z, solved, opt, ap, tog, acs, blk in frontier:
        out.append(f"  {z:>6} {solved:>7} {opt:>8} {cell(ap, 9, 3)} "
                   f"{cell(tog, 9, 2)} {cell(acs, 6, 0)} {cell(blk, 7, 2)}")
    optimal_all = [f for f in frontier if n_draws and f[2] == n_draws]
    feasible_all = [f for f in frontier if n_draws and f[1] == n_draws]
    if optimal_all:
        pool, note = optimal_all, "proven optimal on all draws"
    elif feasible_all:
        pool, note = feasible_all, ("feasible on all draws, but some only as a "
                                    "time-limited incumbent")
    else:
        pool, note = frontier, "taken over all solved (none feasible on all draws)"
    best = max(pool, key=lambda f: (f[4] if f[4] is not None else -1.0, f[0]))
    out.append(f"  suggested best zeta = {best[0]} "
               f"(max siblings together among those {note})")
    return "\n".join(out)


# --------------------------------------------------------------------------
def write_agg_csv(agg: Dict[str, Dict[str, float]], path: str,
                  order: Optional[List[str]] = None) -> None:
    names = order or sorted(agg)
    keys: List[str] = []
    for n in names:
        for k in agg[n]:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + keys)
        for n in names:
            w.writerow([n] + [agg[n].get(k, "") for k in keys])


def write_rows_csv(rows: List[dict], path: str) -> None:
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
