"""
v5_metrics.py

Statistics on a single matching (the paper's table columns) and their
aggregation across draws.

Main results table, per (draw, method) row: solved, avg_pref (mean rank among
the ASSIGNED), top_pref, top3, unassigned, together, and the separated
partition sep_none / sep_one / sep_both. The by-sibling-status table adds
avg_pref, top fraction and unassigned fraction separately for students with
and without siblings.

The separated columns admit several defensible readings of "separated", so the
rule is a parameter rather than a hard-coded choice:
  require_common : restrict the None column to sibling pairs with a common
                   listed school (the One and At-least-two branches always
                   require it).
  one_pref_by    : in One, who must prefer the common school ("assigned" or
                   "both"; the unassigned sibling counts any listed school).
  pref_test      : "strict" (the paper's text as written), "weak" (>=), or
                   "none" (any common listed school qualifies).
  count          : "students" (a flag per student, which is what the paper's
                   student-level partition reports) or "pairs" (each unordered
                   sibling pair contributes once).
The defaults reproduce the paper's numbers: require_common=True,
one_pref_by="assigned", pref_test="strict", count="students".

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

from v5_model import (Instance, Matching, total_rank,
                   members_with_a_sibling_at_same_school)
import v5_stability as stability


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
    # Avg. Pref. averages the rank of the assigned seat over the assigned
    # students only; the unassigned enter the Unassigned column, not this one.
    row["avg_pref"] = pystats.mean(ranks) if ranks else 0.0
    row["top_pref"] = sum(1 for r in ranks if r == 1)
    row["top3"] = sum(1 for r in ranks if r <= 3)     # in one of top-3 choices
    row["unassigned"] = len(inst.students) - len(assigned)
    row["n_students"] = len(inst.students)
    row["matched"] = len(assigned)
    row["together"] = members_with_a_sibling_at_same_school(inst, mu)
    n0, n1, n2 = separated_counts(inst, mu, require_common=require_common,
                                  one_pref_by=one_pref_by)
    row["sep_none"], row["sep_one"], row["sep_both"] = n0, n1, n2
    # The triple above is FILTERED: a separated student is counted only if the
    # pair shares a school and, in the One case, the assigned sibling strictly
    # prefers that shared school to their own seat. That makes the three
    # columns "separated and plausibly fixable" rather than "separated", so
    # their total moves with the mechanism and is not comparable across rows.
    # The unfiltered partition below does not move: for a given instance,
    # sepall_none + sepall_one + sepall_both is the same for every mechanism,
    # which is the accounting a reader expects from a Separated table.
    m0, m1, m2 = separated_counts(inst, mu, require_common=False,
                                  pref_test="none", one_pref_by=one_pref_by)
    row["sepall_none"], row["sepall_one"], row["sepall_both"] = m0, m1, m2
    row["sepall_total"] = m0 + m1 + m2
    row.update(family_separation(inst, mu))
    row.update(student_separation(inst, mu))
    row.update(no_ranking_counts(inst))
    row.update(no_common_school_counts(inst))
    row["total_rank"] = total_rank(inst, mu)

    if priority in ("absolute", "partial"):
        row["eff_providers"] = len(stability.providers(inst, mu)["eff"])

    for label, keep in (("sib", True), ("nosib", False)):  # by sibling status
        grp = [s for s in inst.students if bool(inst.siblings(s)) == keep]
        if not grp:
            continue
        row[f"n_{label}"] = len(grp)          # denominator for absolute tables
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
def family_separation(inst: Instance, mu: Matching) -> Dict[str, int]:
    """A true partition of the families with at least two members.

    A family is Together when at least two of its members share a school.
    Otherwise it is Separated, and separated families are split by how many
    members hold a seat at all: None, One, or At least two. By construction

        together + none + one + at-least-two = multi-member families,

    a quantity that does not depend on the mechanism, so the columns can be
    compared across rows and the Separated block accounts for exactly the
    families that Together does not.

    This is the family-level counterpart of separated_counts, which works at
    the student level, excludes Together students, and applies a shared-school
    and preference filter. Those choices make the student-level triple useful
    as a diagnostic but unsuitable for a table whose rows a reader will add up.
    """
    out = {"fam_total": 0, "fam_together": 0,
           "fam_none": 0, "fam_one": 0, "fam_both": 0}
    for fid, members in inst.families.items():
        if len(members) < 2:
            continue
        out["fam_total"] += 1
        seats = [mu.get(s) for s in members]
        placed = [c for c in seats if c is not None]
        shared = any(c is not None and seats.count(c) > 1 for c in seats)
        if shared:
            out["fam_together"] += 1
        elif len(placed) == 0:
            out["fam_none"] += 1
        elif len(placed) == 1:
            out["fam_one"] += 1
        else:
            out["fam_both"] += 1
    return out


def no_ranking_counts(inst: Instance) -> Dict[str, int]:
    """Students who rank no school at all, and the families they belong to.

    These students can never be assigned under any mechanism, so they sit
    inside Unassigned in the main table and inside the None and One columns of
    the separated tables. They depend only on the instance, not on the
    matching, which is why their standard error over draws is zero: reporting
    them makes the rest of the accounting checkable.
    """
    out = {"norank_all": 0, "norank_sib": 0, "norank_fam": 0}
    for s in inst.students:
        if not inst.prefs[s]:
            out["norank_all"] += 1
            if inst.siblings(s):
                out["norank_sib"] += 1
    for fid, members in inst.families.items():
        if len(members) >= 2 and all(not inst.prefs[s] for s in members):
            out["norank_fam"] += 1
    return out


def no_common_school_counts(inst: Instance) -> Dict[str, int]:
    """Families, and students with siblings, that could never be placed
    together whatever the mechanism does, because no school appears on the
    lists of two members.

    The separated columns do NOT condition on this: a family with no shared
    school is still counted under None, One or At least two according to how
    many of its members hold a seat. That is deliberate. Conditioning on a
    shared school is what the earlier student-level metric did, and it made the
    column totals move with the mechanism, so rows could not be compared or
    added. Reporting the impossible cases as a separate constant instead keeps
    the partition intact and still lets the reader net them out.
    """
    out = {"nocommon_fam": 0, "nocommon_sib": 0}
    for fid, members in inst.families.items():
        if len(members) < 2:
            continue
        seen, shared = set(), False
        for s in members:
            for c in inst.prefs[s]:
                if c in seen:
                    shared = True
                    break
                seen.add(c)
            if shared:
                break
        if not shared:
            out["nocommon_fam"] += 1
    for s in inst.students:
        sibs = inst.siblings(s)
        if not sibs:
            continue
        mine = set(inst.prefs[s])
        if not any(c in mine for t in sibs for c in inst.prefs[t]):
            out["nocommon_sib"] += 1
    return out


def student_separation(inst: Instance, mu: Matching) -> Dict[str, int]:
    """The student-level counterpart of family_separation, using the same
    categories so the two tables can be read side by side.

    A student with at least one sibling is Together when they share a school
    with one of them. Otherwise they are Separated, and separated students are
    split by how many members of their family hold a seat at all: None, One, or
    At least two. By construction

        together + none + one + at-least-two = students with a sibling,

    which does not depend on the mechanism. stu_together is the same quantity
    as the Together column of the main table.
    """
    out = {"stu_total": 0, "stu_together": 0,
           "stu_none": 0, "stu_one": 0, "stu_both": 0}
    for s in inst.students:
        sibs = inst.siblings(s)
        if not sibs:
            continue
        out["stu_total"] += 1
        cs = mu.get(s)
        if cs is not None and any(mu.get(t) == cs for t in sibs):
            out["stu_together"] += 1
            continue
        placed = sum(1 for t in list(sibs) + [s] if mu.get(t) is not None)
        if placed == 0:
            out["stu_none"] += 1
        elif placed == 1:
            out["stu_one"] += 1
        else:
            out["stu_both"] += 1
    return out


MAIN_COLS = ["avg_pref", "top_pref", "top3", "unassigned", "together",
             "sep_none", "sep_one", "sep_both",
             "sepall_none", "sepall_one", "sepall_both", "sepall_total",
             "fam_total", "fam_together", "fam_none", "fam_one", "fam_both",
             "stu_total", "stu_together", "stu_none", "stu_one", "stu_both",
             "norank_all", "norank_sib", "norank_fam",
             "nocommon_fam", "nocommon_sib"]
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
                                              "eff_providers", "matched",
                                              "n_students", "warm_seconds",
                                              "n_sib", "n_nosib"])
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
