"""
v2_metrics.py

Statistics on a single matching (the paper's table columns) and their
aggregation across draws.

Main tables, per (draw, method) row: solved, avg_pref (mean rank among the
ASSIGNED), top_pref, unassigned, together, sep_none / sep_one / sep_both,
total_rank. Table 13 split: avg_pref / top fraction / unassigned fraction for
students with and without siblings.

Separated semantics carry FOUR knobs because the published One/Both columns
have not yet been pinned to a rule (validate.py compares the variants against
the published Descending and SOSM rows):
  require_common : restrict the None column to sibling pairs with a common
                   listed school (the One/Both branches always require it).
  one_pref_by    : in One, who must prefer the common school ("assigned" or
                   "both"; the unassigned sibling counts any listed school).
  pref_test      : "strict" (paper text as written), "weak" (>=), or "none"
                   (any common listed school qualifies).
  count          : "students" (flag per student; footnote 23's double
                   counting across columns) or "pairs" (each unordered
                   sibling pair contributes once).
Defaults reproduce the strict per-student rule.

Provider counts: the IP's n_providers is a solver CERTIFICATE count. Passing
priority="absolute" or "partial" to evaluate() adds eff_providers, the
Definition-1 effective-provider count of the matching computed through
stability.py, which is the definitional quantity.
"""

from __future__ import annotations

import math
import statistics as pystats
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from v2_model import (Instance, Matching, total_rank,
                   members_with_a_sibling_at_same_school)
import v2_stability as stability


# ==========================================================================
# separated columns
# ==========================================================================
def _common_schools(inst: Instance, s, t) -> List[str]:
    ps = set(inst.prefs[s])
    return [c for c in inst.prefs[t] if c in ps]


def separated_counts(inst: Instance, mu: Matching, *,
                     require_common: bool = True,
                     one_pref_by: str = "assigned",
                     pref_test: str = "strict",
                     count: str = "students") -> Tuple[int, int, int]:
    """(#None, #One, #Both) over students with >=1 sibling that are not
    Together."""
    n_none = n_one = n_both = 0
    seen_pairs = set()
    for s in inst.students:
        sibs = inst.siblings(s)
        if not sibs:
            continue
        cs = mu.get(s)
        if cs is not None and any(mu.get(t) == cs for t in sibs):
            continue                                        # Together
        f_none = f_one = f_both = False
        for t in sibs:
            ct = mu.get(t)
            if cs is not None and ct == cs:
                continue
            common = _common_schools(inst, s, t)
            if require_common and not common:
                continue
            cat = None
            if cs is None and ct is None:
                cat = "none"
            elif (cs is None) != (ct is None):
                if not common:
                    continue
                assigned, ca = (t, ct) if cs is None else (s, cs)
                other = s if cs is None else t
                if pref_test == "none":
                    ok = bool(common)
                else:
                    ok = False
                    for c in common:
                        r = inst.rank(assigned, c)
                        ok = (r < inst.rank(assigned, ca) if pref_test == "strict"
                              else r <= inst.rank(assigned, ca))
                        if ok and one_pref_by == "both":
                            ok = inst.rank(other, c) <= len(inst.prefs[other])
                        if ok:
                            break
                if ok:
                    cat = "one"
            else:                                           # both assigned
                if not common:
                    continue
                if pref_test == "none":
                    ok = bool(common)
                else:
                    ok = False
                    for c in common:
                        if pref_test == "strict":
                            ok = (inst.rank(s, c) < inst.rank(s, cs) and
                                  inst.rank(t, c) < inst.rank(t, ct))
                        else:
                            ok = (inst.rank(s, c) <= inst.rank(s, cs) and
                                  inst.rank(t, c) <= inst.rank(t, ct))
                        if ok:
                            break
                if ok:
                    cat = "both"
            if cat is None:
                continue
            if count == "pairs":
                pk = (cat, frozenset((s, t)))
                if pk in seen_pairs:
                    continue
                seen_pairs.add(pk)
                n_none += cat == "none"
                n_one += cat == "one"
                n_both += cat == "both"
            else:
                f_none |= cat == "none"
                f_one |= cat == "one"
                f_both |= cat == "both"
        if count == "students":
            n_none += int(f_none)
            n_one += int(f_one)
            n_both += int(f_both)
    return n_none, n_one, n_both


# ==========================================================================
# per-matching row
# ==========================================================================
def evaluate(inst: Instance, mu: Optional[Matching],
             solve_info: Optional[dict] = None, *,
             priority: Optional[str] = None,
             require_common: bool = True,
             one_pref_by: str = "assigned") -> Dict[str, float]:
    """One row for one (draw, method). mu=None yields solved=0 only."""
    row: Dict[str, float] = {"solved": int(mu is not None)}
    if solve_info:
        for k in ("runtime", "mip_gap", "objective", "n_providers",
                  "n_receivers", "n_cuts", "status_str", "converged",
                  "cycle", "iters", "cycle_len", "hit_max",
                  "cycle_families", "cycle_students", "cycle_schools", "via",
                  "rank_optimal"):
            if solve_info.get(k) is not None:
                row[k] = solve_info[k]
    if mu is None:
        return row

    assigned = [s for s in inst.students if mu.get(s) is not None]
    ranks = [inst.rank(s, mu[s]) for s in assigned]
    row["avg_pref"] = pystats.mean(ranks) if ranks else 0.0
    row["top_pref"] = sum(1 for r in ranks if r == 1)
    row["top3"] = sum(1 for r in ranks if r <= 3)     # in one of top-3 choices
    row["unassigned"] = len(inst.students) - len(assigned)
    row["matched"] = len(assigned)
    row["together"] = members_with_a_sibling_at_same_school(inst, mu)
    n0, n1, n2 = separated_counts(inst, mu, require_common=require_common,
                                  one_pref_by=one_pref_by)
    row["sep_none"], row["sep_one"], row["sep_both"] = n0, n1, n2
    row["total_rank"] = total_rank(inst, mu)

    if priority in ("absolute", "partial"):
        row["eff_providers"] = len(stability.providers(inst, mu)["eff"])

    for label, keep in (("sib", True), ("nosib", False)):     # Table 13
        grp = [s for s in inst.students if bool(inst.siblings(s)) == keep]
        if not grp:
            continue
        a = [s for s in grp if mu.get(s) is not None]
        row[f"avg_pref_{label}"] = (pystats.mean(
            inst.rank(s, mu[s]) for s in a) if a else 0.0)
        row[f"top_frac_{label}"] = (sum(1 for s in a
                                        if inst.rank(s, mu[s]) == 1) / len(grp))
        row[f"top3_frac_{label}"] = (sum(1 for s in a
                                         if inst.rank(s, mu[s]) <= 3) / len(grp))
        row[f"unassg_frac_{label}"] = 1 - len(a) / len(grp)
    return row


# ==========================================================================
# aggregation across draws
# ==========================================================================
MAIN_COLS = ["avg_pref", "top_pref", "top3", "unassigned", "together",
             "sep_none", "sep_one", "sep_both"]
SPLIT_COLS = ["avg_pref_sib", "top_frac_sib", "top3_frac_sib", "unassg_frac_sib",
              "avg_pref_nosib", "top_frac_nosib", "top3_frac_nosib",
              "unassg_frac_nosib"]


def _mean_se(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    mean = math.fsum(vals) / n
    if n <= 1:
        return mean, 0.0
    var = math.fsum((float(x) - mean) ** 2 for x in vals) / (n - 1)
    return mean, math.sqrt(var) / math.sqrt(n)


def aggregate(rows: List[Dict[str, float]],
              cols: Optional[List[str]] = None) -> Dict[str, float]:
    """Aggregate ONE method's rows across draws: solved count, then mean/SE
    per column over solved draws only (the paper's convention). Adds acs_pct
    (percentage of solved matchings that are ACS) and acs_n (how many were
    checked), when the rows carry an `acs` flag."""
    cols = cols or (MAIN_COLS + SPLIT_COLS + ["runtime", "mip_gap",
                                              "eff_providers", "matched"])
    ok = [r for r in rows if r.get("solved")]
    out: Dict[str, float] = {"n_draws": len(rows), "solved": len(ok)}
    for k in cols:
        vals = [r[k] for r in ok if k in r]
        if vals:
            out[f"{k}_mean"], out[f"{k}_se"] = _mean_se(vals)
    # Blocking is reported CONDITIONAL on failing to reach ACS: averaged over
    # the non-ACS draws only (a draw that is ACS has zero blocking by
    # definition and would only dilute the number). If every solved draw is ACS,
    # there is no blocking anywhere, so report 0.
    non_acs = [r for r in ok if r.get("acs") == 0]
    for k in ("blocking_pct", "n_blocking"):
        vals = [r[k] for r in non_acs if k in r]
        if vals:
            out[f"{k}_mean"], out[f"{k}_se"] = _mean_se(vals)
        elif any(k in r for r in ok):
            out[f"{k}_mean"], out[f"{k}_se"] = 0.0, 0.0
    out["non_acs_n"] = len(non_acs)
    acs_vals = [r["acs"] for r in ok if "acs" in r]
    if acs_vals:
        out["acs_pct"] = 100.0 * sum(acs_vals) / len(acs_vals)
        out["acs_n"] = len(acs_vals)
    opt_rows = [r for r in ok if "status_str" in r]
    if opt_rows:
        # proven optimal, as opposed to a time-limited incumbent that is merely
        # feasible; only IP methods carry status_str
        out["optimal"] = sum(1 for r in opt_rows if r["status_str"] == "OPTIMAL")
    conv_rows = [r for r in ok if r.get("converged") is True]
    if any("converged" in r for r in ok):
        out["n_converged"] = len(conv_rows)
        it = [r["iters"] for r in conv_rows if "iters" in r]
        if it:
            out["iters_converged_mean"] = pystats.mean(it)
    # runtime over ALL attempted rows (solved or infeasible), so a zeta that is
    # infeasible on every draw still reports how long it took to prove that
    rt_all = [r["runtime"] for r in rows if r.get("runtime") is not None]
    if rt_all:
        out["runtime_all_mean"], out["runtime_all_se"] = _mean_se(rt_all)
    return out


def aggregate_by_method(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    by: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    return {mth: aggregate(rs) for mth, rs in by.items()}
