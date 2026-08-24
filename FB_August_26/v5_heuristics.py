"""
v5_heuristics.py

The paper's non-exact mechanisms. All contingent orders come from
stability.order_key, so this file contains no stability logic of its own.

  deferred_acceptance  SOSM: student-proposing DA under the initial order.
  descending           Algorithm 1: levels top-down; at each level the
                       contingent order is recomputed from the already-fixed
                       higher-level placements, then one DA pass runs among
                       that level's students. `priority` selects the order
                       ("absolute" is the Chilean policy).
  ascending            descending with increasing level order.
  lsda                 Algorithm 2: family-SIZE-stratified DA. Process sizes
                       k_bar, ..., 1; each round runs DA among the students of
                       size-k families only, with INITIAL priorities and the
                       residual capacities; then fix and decrement. Strategy-
                       proof at the student and family levels, at the cost of
                       contingent stability.
  rada                 Algorithm 3: iterate DA, recomputing the contingent
                       order from the previous matching. The ACS check runs at
                       every iterate, not only at a fixed point, so an ACS
                       matching the sequence steps over is still caught. If the
                       sequence revisits a matching without ever producing an ACS
                       one, the lowest-total-rank iterate seen is returned; that
                       matching is NOT ACS, since every iterate was tested and
                       rejected.

The level-sequential variant of Correa et al. (boost when a sibling is seated
at an earlier-processed level) is deliberately NOT included: under the
absolute definition with |G| = 1 it coincides exactly with descending (the
family's earliest-processed seated member at c is unboosted, hence satisfies
Definition 1 (iii) and provides), verified on 350/350 random instances and on
all Magallanes draws, so it would duplicate a column.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from v5_model import Instance, Matching, School, Student, Level, total_rank
import v5_stability as stability

PriorityKey = Callable[[Student, School], tuple]


# ==========================================================================
# DA engines
# ==========================================================================
def da_with_key(inst: Instance, key: PriorityKey) -> Matching:
    """Student-proposing DA over the whole market (levels have separate seats).

    NOTE. An earlier version accepted a `warm` matching to seed each student's
    first proposal and skip proposals to schools they previously lost. That is
    only sound when the warm matching is itself DA-stable under THIS key. Across
    RADA iterations the priority key changes, so the previous matching is not
    stable under the new key, and seeding from it skips proposals a student is
    entitled to make, producing a different and wrong matching (it failed on
    ~7% of random instances). Warm-starting the proposal pointer is therefore
    not used; the sound speed-up is to commit providers in batches, which cuts
    the NUMBER of DA calls without changing any individual DA.
    """
    next_idx: Dict[Student, int] = {s: 0 for s in inst.students}
    held: Dict[Tuple[School, Level], List[Student]] = defaultdict(list)
    free: List[Student] = list(inst.students)
    while free:
        s = free.pop()
        while next_idx[s] < len(inst.prefs[s]):
            c = inst.prefs[s][next_idx[s]]
            next_idx[s] += 1
            ell = inst.level_of[s]
            if inst.q(c, ell) == 0:
                continue
            slot = (c, ell)
            held[slot].append(s)
            if len(held[slot]) <= inst.q(c, ell):
                break
            held[slot].sort(key=lambda t: key(t, c))
            worst = held[slot].pop()
            if worst != s:
                free.append(worst)
                break
    mu: Matching = {s: None for s in inst.students}
    for (c, _), lst in held.items():
        for s in lst:
            mu[s] = c
    return mu


def _da_single_level(inst: Instance, students: List[Student], ell: Level,
                     key: PriorityKey, cap=None) -> Dict[Student, Optional[School]]:
    """DA among `students` (one level). `cap` optionally supplies residual
    capacities {(school, level): seats}; default is inst.q."""
    q = (lambda c: cap.get((c, ell), 0)) if cap is not None \
        else (lambda c: inst.q(c, ell))
    next_idx = {s: 0 for s in students}
    held: Dict[School, List[Student]] = defaultdict(list)
    free = list(students)
    while free:
        s = free.pop()
        while next_idx[s] < len(inst.prefs[s]):
            c = inst.prefs[s][next_idx[s]]
            next_idx[s] += 1
            if q(c) == 0:
                continue
            held[c].append(s)
            if len(held[c]) <= q(c):
                break
            held[c].sort(key=lambda t: key(t, c))
            worst = held[c].pop()
            if worst != s:
                free.append(worst)
                break
    res = {s: None for s in students}
    for c, lst in held.items():
        for s in lst:
            res[s] = c
    return res


# ==========================================================================
# SOSM
# ==========================================================================
def deferred_acceptance(inst: Instance) -> Matching:
    return da_with_key(inst, lambda s, c: inst.order_key(s, c))


# ==========================================================================
# Descending / Ascending (Algorithm 1)
# ==========================================================================
def descending(inst: Instance, priority: str = "absolute",
               level_order: Optional[List[Level]] = None) -> Matching:
    if level_order is None:
        level_order = sorted(inst.levels, reverse=True)
    mu: Matching = {s: None for s in inst.students}
    for ell in level_order:
        ann = stability.precompute(inst, mu, priority)

        def key(s: Student, c: School, _a=ann) -> tuple:
            return stability.order_key(inst, mu, _a, s, c, priority)

        level_students = inst.students_at_level(ell)
        sub = _da_single_level(inst, level_students, ell, key)
        for s in level_students:
            mu[s] = sub[s]
    return mu


def ascending(inst: Instance, priority: str = "absolute") -> Matching:
    return descending(inst, priority, level_order=sorted(inst.levels))


# ==========================================================================
# LSDA (Algorithm 2): family-size-stratified DA
# ==========================================================================
def lsda(inst: Instance, order: str = "desc") -> Matching:
    """order='desc' processes the largest families first (footnote 16: 'asc'
    for the increasing variant)."""
    sizes = sorted({len(m) for m in inst.families.values()},
                   reverse=(order == "desc"))
    cap = dict(inst.capacity)
    mu: Matching = {s: None for s in inst.students}

    def key(s: Student, c: School) -> tuple:
        return (inst.group[(s, c)], inst.lottery[(s, c)])

    for k in sizes:
        kids = [s for s in inst.students if len(inst.family_members(s)) == k]
        by_level: Dict[Level, List[Student]] = defaultdict(list)
        for s in kids:
            by_level[inst.level_of[s]].append(s)
        for ell, studs in by_level.items():
            sub = _da_single_level(inst, studs, ell, key, cap=cap)
            for s in studs:
                mu[s] = sub[s]
                if sub[s] is not None:
                    cap[(sub[s], ell)] -= 1
    return mu


def slda(inst: Instance) -> Matching:
    """SLDA: the opposite size order to lsda (Algorithm 2). Process families
    from smallest (size 1) up to the largest, running standard DA within each
    size class against the residual capacities and fixing before moving on
    (footnote 16). Distinct mechanism from lsda: the processing order changes
    which families see the residual seats, so SLDA and LSDA differ on most
    instances."""
    return lsda(inst, order="asc")


# ==========================================================================
# RADA (Algorithm 3)
# ==========================================================================
def rada(inst: Instance, priority: str = "absolute", max_iter: int = 2000
         ) -> Tuple[Matching, dict]:
    """Algorithm 3. Run DA, recompute the contingent order from the resulting
    matching, and repeat.

    Two exits. If a pass reproduces the previous matching, the sequence is at a
    fixed point: that matching is DA-stable under its own contingent order, so
    it is contingent stable (Proposition 2), and it is returned with
    converged=True. Otherwise the sequence either revisits an earlier matching
    (a cycle) or exhausts max_iter, and the matching returned is the one of
    lowest total rank among those visited, with ties going to the earliest.
    That matching need not be contingent stable, so callers that need stability
    must check converged, or verify with v5_stability.is_contingent_stable.
    """
    seq: List[Matching] = []
    seen: Dict[frozenset, int] = {}
    mu = deferred_acceptance(inst)
    for t in range(max_iter):
        if t > 0 and mu == seq[-1]:                        # mu^t = mu^{t-1}
            return mu, {"converged": True, "cycle": False, "iters": t + 1}
        sig = frozenset(mu.items())
        if sig in seen:                                    # revisited: a cycle
            return min(seq, key=lambda m: total_rank(inst, m)), {
                "converged": False, "cycle": True, "iters": t + 1,
                "cycle_len": len(seq) - seen[sig]}
        seen[sig] = len(seq)
        seq.append(mu)
        ann = stability.precompute(inst, mu, priority)
        cur = mu
        mu = da_with_key(inst, lambda s, c, a=ann, m=cur: stability.order_key(
            inst, m, a, s, c, priority))
    return min(seq + [mu], key=lambda m: total_rank(inst, m)), {
        "converged": False, "cycle": False, "iters": max_iter, "hit_max": True}
