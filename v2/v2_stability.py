"""
v2_stability.py

The definition. Every definitional primitive of the paper lives here and only
here. Nothing else in the package implements a stability test:

  Definition 1  provider / effective provider     -> is_provider, providers
  Definition 2  absolute contingent priority      -> absolute_group
  Definition 3  partial contingent priority       -> order_key(..., "partial")
  Definition 4  contingent justified-envy and
                stability (+ non-wastefulness)    -> is_contingent_stable
  initial stability                               -> priority = "none"

The integer programs in v2_exact.py carve out exactly the set these predicates
describe. That claim is certified by exhaustive enumeration in the v1 archive,
which contains the test suite; this package contains only the production code.

The checker implements HARD priorities: z = 1 for every effective provider,
Definitions 2 and 3 as written. Soft and hybrid quantify over z and have no
standalone checker; they are handled in v2_exact.py. A matching returned by the
soft or hybrid IP is still a concrete matching, so this checker can test it, and
v2_simulate.py does.
V-membership matters twice: the Definition 1 (ii) witness and the receiver set
R^mu both require the sibling to list c AND to have seats at their level.
[Definition 1 (ii) as printed reads mu(s') >=_s mu(s); per the R^mu display
and the surrounding prose the intended reading, implemented here, is
mu(s) >=_{s'} mu(s').]
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from v2_model import (Instance, Matching, School, Student, is_feasible,
                   occupants_by_slot)

PRIORITIES = ("absolute", "partial", "none")


# ==========================================================================
# Definition 1: provider and effective provider
# ==========================================================================
def count_above(inst: Instance, mu: Matching, s: Student, c: School) -> int:
    """|{s'' in S^{l(s)} : s'' >_c s, c >=_{s''} mu(s'')}| (condition (iii)).
    Non-listers cannot weakly prefer c, so listers(c, l(s)) is the universe."""
    cnt = 0
    for s2 in inst.listers(c, inst.level_of[s]):
        if s2 != s and inst.weakly_prefers(s2, c, mu) and inst.succ(s2, s, c):
            cnt += 1
    return cnt


def is_provider(inst: Instance, mu: Matching, s: Student, c: School) -> bool:
    """Definition 1: (i) mu(s) = c; (ii) some sibling t with (t, c) in V weakly
    prefers c to mu(t); (iii) count_above < q(c, l(s))."""
    if mu.get(s) != c:
        return False
    if count_above(inst, mu, s, c) > inst.q(c, inst.level_of[s]) - 1:
        return False
    for t in inst.siblings(s):
        if inst.q(c, inst.level_of[t]) > 0 and inst.weakly_prefers(t, c, mu):
            return True
    return False


def providers(inst: Instance, mu: Matching) -> dict:
    """qual: all (s, c) satisfying Definition 1; eff: per (school, family) the
    qualifying member with the best lottery (the effective provider)."""
    qual: Set[Tuple[Student, School]] = set()
    by_fam: Dict[Tuple[School, str], List[Student]] = defaultdict(list)
    for s in inst.students:
        c = mu.get(s)
        if c is not None and is_provider(inst, mu, s, c):
            qual.add((s, c))
            by_fam[(c, inst.family_of[s])].append(s)
    eff = {key: min(mem, key=lambda t: inst.lottery[(t, key[0])])
           for key, mem in by_fam.items()}
    return {"qual": qual, "eff": eff}


def receivers(inst: Instance, mu: Matching, s: Student) -> List[Student]:
    """R^mu(s): siblings t with (t, mu(s)) in V and mu(s) >=_t mu(t)."""
    c = mu.get(s)
    if c is None:
        return []
    return [t for t in inst.siblings(s)
            if inst.q(c, inst.level_of[t]) > 0 and inst.weakly_prefers(t, c, mu)]


def precompute(inst: Instance, mu: Matching, priority: str):
    """Per-matching provider state consumed by order_key. None for 'none'."""
    if priority == "none":
        return None
    ann = providers(inst, mu)
    if priority == "absolute":
        ann["recv"] = {s: receivers(inst, mu, s)
                       for s in inst.students if mu.get(s) is not None}
    return ann


# ==========================================================================
# Definition 2: absolute (group update)
# ==========================================================================
def absolute_group(inst: Instance, mu: Matching, ann, s: Student, c: School) -> int:
    """g^mu(s, c): keep g if (P) effective provider with standing (co-assigned
    or same-level receiver) or (R) receiver of the family's effective provider
    matched to c; otherwise |G| + 1. Higher initial groups are untouched."""
    g = inst.group[(s, c)]
    if g < inst.num_groups:
        return g
    eff = ann["eff"].get((c, inst.family_of[s]))
    if eff == s and (s, c) in ann["qual"]:
        coassigned = any(mu.get(t) == c for t in inst.siblings(s))
        same_recv = any(inst.level_of[r] == inst.level_of[s]
                        for r in ann["recv"].get(s, []))
        return g if (coassigned or same_recv) else inst.num_groups + 1
    if eff is not None and eff != s and mu.get(eff) == c:
        if inst.q(c, inst.level_of[s]) > 0 and inst.weakly_prefers(s, c, mu):
            return g                                  # s in R^mu(eff)
    return inst.num_groups + 1


# ==========================================================================
# contingent order (Definitions 2 and 3)
# ==========================================================================
def order_key(inst: Instance, mu: Matching, ann, s: Student, c: School,
              priority: str):
    """Lexicographic key for >^mu_c: lower is better. 'absolute' orders by the
    updated group then the untouched lottery. 'partial' keeps groups and uses
    the Definition 3 tie-breaker as (anchor, tier, own): the effective provider
    sits at (p*, 0), inheriting receivers at (p*, 1, own) preserving their
    initial order inside (p* - eps, p*], everyone else at their own lottery.
    'none' is the initial order."""
    if priority == "absolute":
        return (absolute_group(inst, mu, ann, s, c),
                inst.lottery[(s, c)], 0.0, 0.0)
    if priority == "partial":
        g = inst.group[(s, c)]
        own = inst.lottery[(s, c)]
        star = ann["eff"].get((c, inst.family_of[s]))
        if star is None:
            return (g, own, 1.0, own)
        p_star = inst.lottery[(star, c)]
        if star == s:
            return (g, p_star, 0.0, own)
        if p_star < own:                                # inherit if beneficial
            return (g, p_star, 1.0, own)
        return (g, own, 1.0, own)
    if priority == "none":
        return (inst.group[(s, c)], inst.lottery[(s, c)], 0.0, 0.0)
    raise ValueError(f"priority must be one of {PRIORITIES}, got {priority!r}")


# ==========================================================================
# Definition 4: contingent justified-envy and stability
# ==========================================================================
def is_contingent_stable(inst: Instance, mu: Matching, priority: str = "absolute",
                         return_blocking: bool = False):
    """Non-wasteful and no contingent justified-envy under `priority` (hard
    semantics). Blocking entries: ('waste', s, c), ('envy', s, c, s2), or
    ('infeasible', msg). Sibling-vs-sibling envy IS counted: Definition 4 does
    not exempt family members."""
    ok, problems = is_feasible(inst, mu)
    if not ok:
        return (False, [("infeasible", p) for p in problems]) if return_blocking else False

    ann = precompute(inst, mu, priority)
    occ = occupants_by_slot(inst, mu)
    blocking: List[Tuple] = []

    for s in inst.students:
        own_rank = inst.rank(s, mu.get(s))
        ell = inst.level_of[s]
        for c in inst.prefs[s]:
            if inst.rank(s, c) >= own_rank:
                continue                                # c not preferred
            seats = occ.get((c, ell), [])
            if len(seats) < inst.q(c, ell):             # non-wastefulness
                blocking.append(("waste", s, c))
                if not return_blocking:
                    return False
                continue
            ks = order_key(inst, mu, ann, s, c, priority)
            for s2 in seats:
                if ks < order_key(inst, mu, ann, s2, c, priority):
                    blocking.append(("envy", s, c, s2))
                    if not return_blocking:
                        return False
                    break
    if return_blocking:
        return (len(blocking) == 0), blocking
    return True


def is_initially_stable(inst: Instance, mu: Matching) -> bool:
    return is_contingent_stable(inst, mu, "none")
