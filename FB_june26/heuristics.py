"""
heuristics.py
================
ACS-targeting rewrite of the sibling-priority heuristics. Every heuristic
augments school priorities with the ABSOLUTE contingent priority of
Definitions 1-3 via the shared `acs_priority` primitive (the same code the
verifier uses), so at convergence the output is ACS by construction.

Implemented so far:
  - fsda_single                 : student-side DA fixed point, non-monotone boost
  - simultaneous(mono=True)     : student-side, monotone (cumulative) boost
  - descending_da / descending_fsda
  - ascending_da  / ascending_fsda

To come: LS, LS_DA, LS_nd, SL, SL_DA, SL_nd.

Common return dict:
  {"status", "x_opt", "mu", "iterations", "runtime",
   "num_vars":0,"num_cols":0,"mipgap":0,"nodes":0}   # last 4 for table compat

status in {"completed", "cycle_detected", "max_iter_reached"}.

inputs_basic = (students, colleges, pref, cap, siblings, levels, students_per_level)
tb           = dict student -> {rbd: lottery}, higher = better (from
               generate_inputs.modify_school_loterries).
"""

import time
import copy

import acs_priority as P
import algorithms as alg          # uses the codebase DA


# ============================================================
# small shared utilities
# ============================================================

def _run_DA(students, pref, cap):
    return alg.DA(list(students), pref, cap)


def _mu_key(mu):
    return frozenset((s, c) for s, c in mu.items() if c is not None)


def _wrap(mu, students, status, iterations, t0):
    mu = {s: mu.get(s) for s in students}
    x_opt = {s: {c: 1} for s, c in mu.items() if c is not None}
    return {
        "status": status,
        "x_opt": x_opt,
        "mu": mu,
        "iterations": iterations,
        "runtime": time.time() - t0,
        "num_vars": 0, "num_cols": 0, "mipgap": 0, "nodes": 0,
    }


# ============================================================
# Core fixed-point iteration (fsda_single + simultaneous share this)
# ============================================================

def _fixed_point(inputs_basic, tb=None, mono=False, max_iter=1000,
                 initial="da", verbose=False, tag="FSD-A"):
    """
    Iterate  mu -> DA(augment(mu))  with the absolute boost.

    mono=False : recompute the upper-tier set from the *current* matching each
                 iteration (non-monotone). A fixed point is ACS. May cycle.
    mono=True  : accumulate the upper-tier set across iterations (never revoke).
                 Always converges; the output is ACS only if the accumulated
                 boost matches the final matching's induced boost (checked by
                 the verifier).
    """
    t0 = time.time()
    (students, colleges, pref, cap, siblings, levels, students_per_level) = inputs_basic
    students = list(students)
    colleges = list(colleges)
    levels_of = P.build_levels_of(students, pref)

    if initial == "empty":
        mu = {s: None for s in students}
    else:
        mu = _run_DA(students, pref, cap)
        mu = {s: mu.get(s) for s in students}

    U_cum = {}                      # cumulative upper set (mono only)
    seen = {}
    for it in range(1, max_iter + 1):
        U_t = P.upper_set_from_mu(mu, students, colleges, pref, cap,
                                  siblings, levels_of, tb=tb)
        if mono:
            for c, U in U_t.items():
                U_cum.setdefault(c, set()).update(U)
            U_use = U_cum
        else:
            U_use = U_t

        pref_aug = P.augmented_pref_from_upper(U_use, students, colleges, pref)
        mu_new = _run_DA(students, pref_aug, cap)
        mu_new = {s: mu_new.get(s) for s in students}

        if _mu_key(mu_new) == _mu_key(mu):
            if verbose:
                print(f"  {tag}: fixed point at iter {it}")
            return _wrap(mu_new, students, "completed", it, t0)

        if not mono:
            k = _mu_key(mu_new)
            if k in seen:
                if verbose:
                    print(f"  {tag}: cycle at iter {it} (repeat of {seen[k]})")
                return _wrap(mu_new, students, "cycle_detected", it, t0)
            seen[k] = it
        mu = mu_new

    if verbose:
        print(f"  {tag}: max_iter {max_iter} reached")
    return _wrap(mu, students, "max_iter_reached", max_iter, t0)


def fsda_single(inputs_basic, tb=None, max_iter=1000, initial="da", verbose=False):
    """Canonical FSD-A, single (student-side) variant: non-monotone fixed point.
    ACS at convergence; may cycle on co-assignment-requiring instances."""
    return _fixed_point(inputs_basic, tb=tb, mono=False, max_iter=max_iter,
                        initial=initial, verbose=verbose, tag="FSD-A(single)")


def simultaneous(inputs_basic, tb=None, mono=True, max_iter=1000, verbose=False):
    """Simultaneous heuristic with the absolute boost.
    mono=True (default): monotone cumulative boost -> always converges.
    mono=False: equivalent to fsda_single (non-monotone)."""
    tag = "Simultaneous(mono)" if mono else "Simultaneous(nonmono)"
    return _fixed_point(inputs_basic, tb=tb, mono=mono, max_iter=max_iter,
                        initial="da", verbose=verbose, tag=tag)


# ============================================================
# Level-stratified heuristics (descending / ascending; DA / fsda inner)
# ============================================================

def _sequential_by_level(inputs_basic, order, tb=None, inner="da",
                         inner_max_iter=200, verbose=False):
    """
    Process levels in `order`. The boost for a level's students comes from
    siblings in already-processed levels (carried in the global matching `mu`),
    computed with the absolute rule via acs_priority.

    inner="da"   : a single DA pass per level using the boost from earlier levels.
    inner="fsda" : a fixed-point iteration *within* the level (handles same-level
                   twin providers); reduces to the DA pass when a level cohort
                   shares no siblings, which is the usual case.
    """
    t0 = time.time()
    (students, colleges, pref, cap, siblings, levels, students_per_level) = inputs_basic
    students = list(students)
    colleges = list(colleges)
    levels_of = P.build_levels_of(students, pref)

    # PreK/K aliasing used in some instance files
    if '0' not in students_per_level or '-1' not in students_per_level:
        order = ['PreK' if l == '-1' else 'K' if l == '0' else l for l in order]

    mu = {s: None for s in students}     # global matching, fills as we go
    total_iters = 0

    for idx in order:
        if idx not in students_per_level or idx not in levels:
            continue
        schools_in_level = [c for c in levels[idx] if c in cap]
        students_in_level = [s for s in students_per_level[idx] if s in pref]
        cap_level = {c: cap[c] for c in schools_in_level}

        if inner == "da":
            # boost from earlier levels only (current level empty in mu)
            upper = P.upper_set_from_mu(
                mu, students, colleges, pref, cap, siblings, levels_of,
                tb=tb, colleges_subset=schools_in_level)
            pref_aug = P.augmented_pref_from_upper(
                upper, students, colleges, pref, colleges_subset=schools_in_level)
            pref_level = {i: pref_aug[i] for i in students_in_level if i in pref_aug}
            for c in schools_in_level:
                if c in pref_aug:
                    pref_level[c] = pref_aug[c]
            level_match = _run_DA(students_in_level, pref_level, cap_level)
            total_iters += 1
        else:  # inner == "fsda": fixed point within the level
            level_match = {s: None for s in students_in_level}
            level_seen = {}
            for jt in range(1, inner_max_iter + 1):
                combined = dict(mu)
                combined.update({s: level_match.get(s) for s in students_in_level})
                upper = P.upper_set_from_mu(
                    combined, students, colleges, pref, cap, siblings, levels_of,
                    tb=tb, colleges_subset=schools_in_level)
                pref_aug = P.augmented_pref_from_upper(
                    upper, students, colleges, pref, colleges_subset=schools_in_level)
                pref_level = {i: pref_aug[i] for i in students_in_level if i in pref_aug}
                for c in schools_in_level:
                    if c in pref_aug:
                        pref_level[c] = pref_aug[c]
                new_match = _run_DA(students_in_level, pref_level, cap_level)
                new_match = {s: new_match.get(s) for s in students_in_level}
                total_iters += 1
                if _mu_key(new_match) == _mu_key(level_match):
                    break
                k = _mu_key(new_match)
                if k in level_seen:
                    level_match = new_match
                    break
                level_seen[k] = jt
                level_match = new_match

        for s, c in level_match.items():
            mu[s] = c

    if verbose:
        print(f"  sequential[{inner}]: {len(order)} levels, {total_iters} DA runs")
    return _wrap(mu, students, "completed", total_iters, t0)


_DESC_ORDER = [str(i) for i in sorted(range(-1, 13), reverse=True)]
_ASC_ORDER = [str(i) for i in sorted(range(-1, 13))]


def descending_da(inputs_basic, tb=None, verbose=False):
    """Descending (12 -> PreK), one DA pass per level, absolute between-level boost."""
    return _sequential_by_level(inputs_basic, _DESC_ORDER, tb=tb, inner="da",
                                verbose=verbose)


def descending_fsda(inputs_basic, tb=None, verbose=False):
    """Descending, FSD-A fixed point per level (handles same-level twins)."""
    return _sequential_by_level(inputs_basic, _DESC_ORDER, tb=tb, inner="fsda",
                                verbose=verbose)


def ascending_da(inputs_basic, tb=None, verbose=False):
    """Ascending (PreK -> 12), one DA pass per level, absolute between-level boost."""
    return _sequential_by_level(inputs_basic, _ASC_ORDER, tb=tb, inner="da",
                                verbose=verbose)


def ascending_fsda(inputs_basic, tb=None, verbose=False):
    """Ascending, FSD-A fixed point per level (handles same-level twins)."""
    return _sequential_by_level(inputs_basic, _ASC_ORDER, tb=tb, inner="fsda",
                                verbose=verbose)


# ============================================================
# Size-stratified heuristics: LS / SL family
# ============================================================
# Families are grouped by SIZE (number of siblings). Unlike level-stratified
# heuristics, a family stays together in one cohort, so the inner solver can
# achieve within-cohort co-assignment. Cross-cohort sibling priority is carried
# through the global matching `mu` and computed with the absolute boost.
#
#   LS    : large-to-small, capacity DECREMENT, FSD-A inner   (greedy; not ACS)
#   LS_DA : large-to-small, capacity DECREMENT, DA inner      (greedy; not ACS)
#   LS_nd : large-to-small, FULL capacity + override, FSD-A inner (-> can be ACS)
#   SL    : small-to-large, capacity DECREMENT, FSD-A inner
#   SL_DA : small-to-large, capacity DECREMENT, DA inner
#   SL_nd : small-to-large, FULL capacity + override, FSD-A inner
#
# The decrement variants lock earlier cohorts' seats, so they are NOT ACS in
# general (a later, higher-priority student cannot reclaim a consumed seat).
# The _nd variants re-solve the cumulative set with full capacity; their final
# iteration is FSD-A on ALL students, so they are ACS exactly when that final
# FSD-A converges.

def _build_families(siblings, students):
    """Connected components of the sibling graph; each a sorted list."""
    visited = set()
    families = []
    student_set = set(students)
    for s in students:
        if s in visited:
            continue
        stack = [s]
        fam = set()
        while stack:
            u = stack.pop()
            if u in fam:
                continue
            fam.add(u)
            for v in siblings.get(u, []):
                if v in student_set and v not in fam:
                    stack.append(v)
        visited.update(fam)
        families.append(sorted(fam))
    return families


def _family_size_map(families):
    out = {}
    for fam in families:
        for s in fam:
            out[s] = len(fam)
    return out


def _restrict_pref(pref_aug, Sk, colleges):
    """Keep only the Sk students' preference lists plus all school lists."""
    out = {s: pref_aug[s] for s in Sk if s in pref_aug}
    for c in colleges:
        if c in pref_aug:
            out[c] = pref_aug[c]
    return out


def _augment_for_cohort(combined_mu, students_all, colleges, pref, cap_boost,
                        siblings, levels_of, tb):
    """Absolute-boost school priorities induced by `combined_mu`. Base
    admissibility is computed over ALL students (the verifier's notion)."""
    upper = P.upper_set_from_mu(combined_mu, students_all, colleges, pref,
                                cap_boost, siblings, levels_of, tb=tb)
    return P.augmented_pref_from_upper(upper, students_all, colleges, pref)


def _cohort_solve(Sk, students_all, colleges, pref, sub_cap, siblings,
                  levels_of, tb, mu_context, inner, max_iter=200):
    """Solve the cohort Sk against `sub_cap`, with the absolute boost computed
    from (mu_context updated by the cohort's own tentative matching).

    inner="da"   : single DA pass (boost from mu_context / warm start).
    inner="fsda" : fixed-point iteration within the cohort (captures within-
                   cohort co-assignment); warm-started from mu_context.
    Returns {s: school_id or None} for s in Sk.
    """
    Sk = list(Sk)
    if inner == "da":
        combined = dict(mu_context)
        for s in Sk:
            combined.setdefault(s, None)
        pref_aug = _augment_for_cohort(combined, students_all, colleges, pref,
                                       sub_cap, siblings, levels_of, tb)
        pref_sub = _restrict_pref(pref_aug, Sk, colleges)
        sub = _run_DA(Sk, pref_sub, sub_cap)
        return {s: sub.get(s) for s in Sk}

    # inner == "fsda": warm-started fixed point within the cohort
    sub_match = {s: mu_context.get(s) for s in Sk}
    seen = {}
    for it in range(1, max_iter + 1):
        combined = dict(mu_context)
        combined.update(sub_match)
        pref_aug = _augment_for_cohort(combined, students_all, colleges, pref,
                                       sub_cap, siblings, levels_of, tb)
        pref_sub = _restrict_pref(pref_aug, Sk, colleges)
        new = _run_DA(Sk, pref_sub, sub_cap)
        new = {s: new.get(s) for s in Sk}
        if _mu_key(new) == _mu_key(sub_match):
            break
        k = _mu_key(new)
        if k in seen:
            sub_match = new
            break
        seen[k] = it
        sub_match = new
    return sub_match


def _size_stratified(inputs_basic, tb, order, mode, inner, max_iter=200,
                     verbose=False):
    """
    order : "large_to_small" or "small_to_large"
    mode  : "decrement" (Sk = exactly size k; decrement capacity) or
            "nd"        (Sk = cumulative; full capacity; override)
    inner : "da" or "fsda"
    """
    import copy
    t0 = time.time()
    (students, colleges, pref, cap, siblings, levels, students_per_level) = inputs_basic
    students = list(students)
    colleges = list(colleges)
    levels_of = P.build_levels_of(students, pref)

    families = _build_families(siblings, students)
    if not families:
        return _wrap({}, students, "completed", 0, t0)
    Kmax = max(len(f) for f in families)
    size_of = _family_size_map(families)

    mu = {}
    cap_remaining = copy.deepcopy(cap)
    iters = 0
    ks = range(Kmax, 0, -1) if order == "large_to_small" else range(1, Kmax + 1)

    for k in ks:
        if mode == "decrement":
            Sk = [s for s in students if size_of.get(s, 1) == k]
        else:  # nd cumulative
            if order == "large_to_small":
                Sk = [s for s in students if size_of.get(s, 1) >= k]
            else:
                Sk = [s for s in students if size_of.get(s, 1) <= k]
        if not Sk:
            continue
        iters += 1

        sub_cap = copy.deepcopy(cap_remaining) if mode == "decrement" else copy.deepcopy(cap)
        sub = _cohort_solve(Sk, students, colleges, pref, sub_cap, siblings,
                            levels_of, tb, mu_context=mu, inner=inner,
                            max_iter=max_iter)

        if mode == "decrement":
            for s, c in sub.items():
                if c is not None:
                    mu[s] = c
                    if c in cap_remaining:
                        cap_remaining[c] -= 1
        else:  # nd override
            for s in Sk:
                mu.pop(s, None)
            for s, c in sub.items():
                if c is not None:
                    mu[s] = c

    if verbose:
        print(f"  size-stratified[{order},{mode},{inner}]: {iters} cohorts")
    return _wrap(mu, students, "completed", iters, t0)


def LS(inputs_basic, tb=None, verbose=False):
    """Large-to-Small, capacity decrement, FSD-A inner (greedy; not ACS)."""
    return _size_stratified(inputs_basic, tb, "large_to_small", "decrement",
                            "fsda", verbose=verbose)


def LS_DA(inputs_basic, tb=None, verbose=False):
    """Large-to-Small, capacity decrement, DA inner (greedy; not ACS)."""
    return _size_stratified(inputs_basic, tb, "large_to_small", "decrement",
                            "da", verbose=verbose)


def LS_nd(inputs_basic, tb=None, verbose=False):
    """Large-to-Small, full capacity + override, FSD-A inner (ACS iff final
    full FSD-A converges)."""
    return _size_stratified(inputs_basic, tb, "large_to_small", "nd",
                            "fsda", verbose=verbose)


def SL(inputs_basic, tb=None, verbose=False):
    """Small-to-Large, capacity decrement, FSD-A inner (greedy; not ACS)."""
    return _size_stratified(inputs_basic, tb, "small_to_large", "decrement",
                            "fsda", verbose=verbose)


def SL_DA(inputs_basic, tb=None, verbose=False):
    """Small-to-Large, capacity decrement, DA inner (greedy; not ACS)."""
    return _size_stratified(inputs_basic, tb, "small_to_large", "decrement",
                            "da", verbose=verbose)


def SL_nd(inputs_basic, tb=None, verbose=False):
    """Small-to-Large, full capacity + override, FSD-A inner (ACS iff final
    full FSD-A converges)."""
    return _size_stratified(inputs_basic, tb, "small_to_large", "nd",
                            "fsda", verbose=verbose)