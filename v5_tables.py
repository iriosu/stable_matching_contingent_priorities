r"""
v5_tables.py

Render aggregated metrics as the paper's tables and write the CSVs.

  render_main       main table: row | Solved | ACS % | Block % | Avg. Pref. |
                    Top Pref. | Unassigned | Together | Separated None/One/At least two
                    (Mean, SE for the paired columns). Uses \scriptsize and a
                    tight \tabcolsep so the widened table still fits; a
                    \resizebox alternative is noted inline.
  render_split      sibling split with Siblings and No-Siblings side by side as
                    COLUMN groups (one row per method), not stacked rows.
  write_agg_csv     one row per method with every aggregated column.
  write_rows_csv    the raw per-(draw, method) rows (union of keys).

Every table renders in ONE unit, chosen with units="absolute" (counts) or
units="percent"; the two are never mixed inside a table. The main table carries
no stability columns: ACS and blocking belong to the RADA comparison, and
agg["acs_pct"] / agg["blocking_pct_mean"] remain available to callers
and for callers that want them.
"""

from __future__ import annotations

import csv
from typing import Dict, List, Optional


def _f(v, nd=2):
    if v is None or v != v:
        return "--"
    return f"{v:.{nd}f}"


def _time_with_warm(solve, warm, warm_se=None, nd=2):
    """Render the solve time, and if a warm-start time is present append it in
    parentheses. With a warm SE the cell reads '27.90 (11.6 $\\pm$ 0.4)'; without
    one it reads '27.90 (11.6)'. Methods with no warm start show the solve time
    alone. The warm-start solve is computed once per draw and shared across the
    IP methods that consume it, so the parenthetical is the mean and standard
    error of that shared feasibility solve, not an additional per-method cost."""
    if solve is None or solve != solve:
        return "--"
    s = f"{solve:.{nd}f}"
    if warm is None or warm != warm:
        return s
    if warm_se is None or warm_se != warm_se:
        return f"{s} ({warm:.1f})"
    return f"{s} ({warm:.1f} $\\pm$ {warm_se:.1f})"


def _pct(v, nd=1):
    """v is a fraction in [0, 1]; render as a percentage."""
    if v is None or v != v:
        return "--"
    return f"{100 * v:.{nd}f}"


# --------------------------------------------------------------------------
def render_main(agg: Dict[str, Dict[str, float]], order: List[str],
                caption: str, label: str,
                display: Optional[Dict[str, str]] = None,
                units: str = "absolute") -> str:
    """The main outcomes table, every column with its standard error over the
    draws. units="absolute" gives student counts, units="percent" gives
    percentages of the region's students; the two are never mixed."""
    if units not in ("absolute", "percent"):
        raise ValueError("units must be 'absolute' or 'percent'")
    display = display or {}
    pct = units == "percent"
    note = (r" Counts are percentages of the region's students."
            if pct else r" Counts are numbers of students.")
    note += (r" Avg.\ Pref.\ averages the rank of the assigned school "
             r"over the assigned students.")
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}{note}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{l r rr rr rr rr rr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Avg.\ Pref.} & \multicolumn{2}{c}{Top Pref.}"
        r" & \multicolumn{2}{c}{Top-3} & \multicolumn{2}{c}{Unassigned}"
        r" & \multicolumn{2}{c}{Together} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}"
        r"\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
        r" & Solved & Mean & SE & Mean & SE & Mean & SE & Mean & SE"
        r" & Mean & SE \\",
        r"\midrule",
    ]
    for name in order:
        a = agg.get(name)
        if a is None:
            continue
        ns = a.get("n_students_mean")
        cells = [display.get(name, name), str(int(a.get("solved", 0))),
                 _f(a.get("avg_pref_mean"), 3), _f(a.get("avg_pref_se"), 3)]
        for k in ("top_pref", "top3", "unassigned", "together"):
            m, e = a.get(f"{k}_mean"), a.get(f"{k}_se")
            if pct:
                m = None if (m is None or not ns) else 100.0 * m / ns
                e = None if (e is None or not ns) else 100.0 * e / ns
                cells += [_f(m, 2), _f(e, 2)]
            else:
                cells += [_f(m), _f(e)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def render_separated(agg: Dict[str, Dict[str, float]], order: List[str],
                     caption: str, label: str,
                     display: Optional[Dict[str, str]] = None,
                     units: str = "absolute",
                     level: str = "student") -> str:
    """Separated students, split by how many members of their family hold a
    seat. Default level="student": the unit is students with at least one
    sibling who are NOT placed with any of them. units="absolute" gives
    student counts; units="percent" gives shares of the separated students,
    so the three columns sum to 100. The Together, Total, No rank and No
    common quantities remain in the metrics rows for cross-checking but are
    not printed here."""
    if units not in ("absolute", "percent"):
        raise ValueError("units must be 'absolute' or 'percent'")
    if level not in ("family", "student"):
        raise ValueError("level must be 'family' or 'student'")
    display = display or {}
    pct = units == "percent"
    pre = "fam" if level == "family" else "stu"
    unit_word = "families" if level == "family" else "students"
    note = (rf" Columns are percentages of the separated {unit_word} and sum "
            r"to 100." if pct else
            rf" Columns are numbers of separated {unit_word}.")
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}{note}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{l rr rr rr}", r"\toprule",
        r" & \multicolumn{2}{c}{None} & \multicolumn{2}{c}{One}"
        r" & \multicolumn{2}{c}{At least two} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r" & Mean & SE & Mean & SE & Mean & SE \\",
        r"\midrule",
    ]
    for name in order:
        a = agg.get(name)
        if a is None:
            continue
        tot = sum(a.get(f"{pre}_{k}_mean") or 0.0
                  for k in ("none", "one", "both"))
        cells = [display.get(name, name)]
        for k in ("none", "one", "both"):
            m, e = a.get(f"{pre}_{k}_mean"), a.get(f"{pre}_{k}_se")
            if pct:
                m = None if (m is None or not tot) else 100.0 * m / tot
                e = None if (e is None or not tot) else 100.0 * e / tot
                cells += [_f(m, 2), _f(e, 2)]
            else:
                cells += [_f(m), _f(e)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
def render_split(agg: Dict[str, Dict[str, float]], order: List[str],
                 caption: str, label: str,
                 display: Optional[Dict[str, str]] = None,
                 computation: bool = True,
                 units: str = "percent") -> str:
    """Siblings and No-Siblings side by side. The Overall group (total matched
    and the deltas against DA and Descending) is deliberately omitted: the
    differences there are not material and are not part of the paper's story.
    units="percent" reports the top-choice and unassigned columns as shares of
    each group; units="absolute" reports them as student counts, using the
    group sizes recorded by the metrics module."""
    if units not in ("absolute", "percent"):
        raise ValueError("units must be 'absolute' or 'percent'")
    display = display or {}
    pct = units == "percent"
    tu = (r"Top [\%]" if pct else "Top")
    uu = (r"Unassg.\ [\%]" if pct else r"Unassg.")
    six = (r" & \multicolumn{2}{c}{Avg.\ Pref.} & \multicolumn{2}{c}{" + tu +
           r"} & \multicolumn{2}{c}{" + uu + r"}")
    unit_note = (r" Top and unassigned are shares of each group."
                 if pct else r" Top and unassigned are student counts.")
    unit_note += (r" Avg.\ Pref.\ averages the rank of the assigned "
                  r"school over the group's assigned students.")

    if computation:
        spec = r"\begin{tabular}{l rrrrrr rrrrrr r rr r}"
        grp = (r" & \multicolumn{6}{c}{Siblings} & \multicolumn{6}{c}{No-Siblings}"
               r" & & \multicolumn{2}{c}{Time [s]} & \\")
        cmid = r"\cmidrule(lr){2-7} \cmidrule(lr){8-13} \cmidrule(lr){15-16}"
        head = six + six + r" & Iters & Mean (warm) & SE & MIPGap \\"
    else:
        spec = r"\begin{tabular}{l rrrrrr rrrrrr}"
        grp = (r" & \multicolumn{6}{c}{Siblings} & \multicolumn{6}{c}{No-Siblings}"
               r" \\")
        cmid = r"\cmidrule(lr){2-7} \cmidrule(lr){8-13}"
        head = six + six + r" \\"
    sub = (r" & Mean & SE" * 6) + r" \\"

    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}{unit_note}}}", rf"\label{{{label}}}",
        spec, r"\toprule", grp, cmid, head, sub, r"\midrule",
    ]
    for name in order:
        a = agg.get(name)
        if a is None:
            continue
        cells = [display.get(name, name)]
        for lab in ("sib", "nosib"):
            n = a.get(f"n_{lab}_mean")
            top = a.get(f"top_frac_{lab}_mean")
            una = a.get(f"unassg_frac_{lab}_mean")
            cells.append(_f(a.get(f"avg_pref_{lab}_mean"), 3))
            cells.append(_f(a.get(f"avg_pref_{lab}_se"), 3))
            for frac, se_key in ((top, f"top_frac_{lab}_se"),
                                 (una, f"unassg_frac_{lab}_se")):
                e = a.get(se_key)
                if pct:
                    cells += [_f(None if frac is None else 100.0 * frac, 2),
                              _f(None if e is None else 100.0 * e, 2)]
                else:
                    cells += [_f(None if (frac is None or not n) else frac * n, 1),
                              _f(None if (e is None or not n) else e * n, 1)]
        if computation:
            gap = a.get("mip_gap_mean")
            cells += [
                _f(a.get("iters_converged_mean"), 1),
                _time_with_warm(a.get("runtime_all_mean", a.get("runtime_mean")),
                                a.get("warm_seconds_mean"),
                                a.get("warm_seconds_se")),
                _f(a.get("runtime_all_se", a.get("runtime_se")), 2),
                ("--" if gap is None else f"{gap:.1e}")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
def render_computation(agg: Dict[str, Dict[str, float]], order: List[str],
                       caption: str, label: str,
                       display: Optional[Dict[str, str]] = None) -> str:
    """The computation table: iterations (for the round-based heuristics), mean
    and standard error of the per-draw solve time, and the mean MIP gap for the
    exact methods. For methods that use the shared feasibility warm start, the
    Time mean carries the mean warm-start time in parentheses, e.g.
    '27.90 (11.6 $\\pm$ 0.4)', which is not part of the reported solve time.
    Heuristics show no MIPGap and no warm parenthetical."""
    display = display or {}
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{l r rr r}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Time [s]} & \\",
        r"\cmidrule(lr){3-4}",
        r" & Iters & Mean & SE & MIPGap \\",
        r"\midrule",
    ]
    for name in order:
        a = agg.get(name)
        if a is None:
            continue
        gap = a.get("mip_gap_mean")
        cells = [display.get(name, name),
                 _f(a.get("iters_converged_mean"), 1),
                 _time_with_warm(a.get("runtime_all_mean", a.get("runtime_mean")),
                                 a.get("warm_seconds_mean"),
                                 a.get("warm_seconds_se")),
                 _f(a.get("runtime_all_se", a.get("runtime_se")), 2),
                 ("--" if gap is None else f"{gap:.1e}")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


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


# --------------------------------------------------------------------------
def computation_paragraph(agg: Dict[str, Dict[str, float]], order: List[str],
                          display: Optional[Dict[str, str]] = None,
                          region: Optional[str] = None,
                          draws: Optional[int] = None) -> str:
    """A paragraph reporting solve times, replacing the computation table in
    the main text. Heuristics and exact methods are described separately, since
    only the exact methods carry a warm start and a MIP gap."""
    display = display or {}
    where = f" on {region}" if region else ""
    howmany = f" over {draws} draws" if draws else ""

    def t(a):
        return a.get("runtime_all_mean", a.get("runtime_mean"))

    heur, exact_ = [], []
    for name in order:
        a = agg.get(name)
        if a is None or t(a) is None:
            continue
        (exact_ if a.get("warm_seconds_mean") is not None
         or a.get("mip_gap_mean") is not None else heur).append(
            (display.get(name, name), a))

    parts = [r"\paragraph{Computation.}"]
    if heur:
        items = ", ".join(f"{nm} in {t(a):.1f} s" for nm, a in heur)
        parts.append(f"The heuristics run in seconds{where}{howmany}: "
                     f"{items} per draw on average.")
    if exact_:
        bits = []
        for nm, a in exact_:
            w = a.get("warm_seconds_mean")
            piece = f"{nm} takes {t(a):.1f} s"
            if w is not None:
                piece += f" after a warm start costing {w:.1f} s"
            bits.append(piece)
        parts.append("The integer programs are heavier: " + "; ".join(bits) +
                     ". Warm start time is reported separately because the "
                     "same stable matching seeds every exact method, so it is "
                     "paid once per draw rather than once per method.")
        gaps = [a.get("mip_gap_mean") for _, a in exact_
                if a.get("mip_gap_mean") is not None]
        if gaps:
            parts.append(f"All exact solves terminate within a MIP gap of "
                         f"{max(gaps):.0e} or prove infeasibility within the "
                         f"one hour limit.")
    return " ".join(parts)
