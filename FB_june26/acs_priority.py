"""
acs_priority.py
===============
Single source of truth for ABSOLUTE contingent priority, following the main
contingent-priorities paper, Definitions 1-3. Used by BOTH:
  - acs_verifier.check_acs   (to *check* a matching is ACS), and
  - the heuristics           (to *augment* school priorities during matching).

Because the verifier and the heuristics share this exact code, a heuristic that
runs DA to a fixed point on `augmented_pref(...)` produces, at convergence, a
matching the verifier certifies as ACS — the property becomes near-definitional.

Definitions implemented
-----------------------
Provider at RBD r (Def 2, refined, strict):
    s provides contingent priority at RBD r iff
      (i)   mu(s) is in RBD r
      (ii)  some sibling s' has mu(s') ⪯_{s'} (r, level(s'))   [wants r; ∅ counts]
      (iii) s is base-admissible at (r, level(s)), i.e. with sid = f"{r}_{level(s)}"
              |{ s'' in S^{level(s)} : s'' ≻_sid s,  sid ⪰_{s''} mu(s'') }| <= q_sid - 1

Effective provider z^mu(s, r):
    among providers in a family at RBD r, the one with the best lottery
    tb[s][r] (higher = better) gets z=1; ties/fallback by base rank.

Upper tier at school c (RBD r = rbd_of(c)), all students initially in group |G|:
    s is upper-tier at c iff
      (a) z^mu(s, r) = 1 AND |f(s) ∩ {family matched to RBD r}| >= 2, OR
      (b) some sibling s' of s has z^mu(s', r) = 1.
Within a tier, ties break by base priority ≻_c.

Data conventions (identical to generate_inputs.py)
--------------------------------------------------
  - school ids are "RBD_grade", e.g. "15690_9".
  - mu        : dict student -> school_id or None
  - pref      : pref[s] = {rank: school_id} ; pref[c] = {rank: student}
                lower rank = more preferred / higher base priority.
  - cap       : dict school_id -> int
  - siblings  : dict student -> list of sibling ids (self excluded)
  - levels_of : dict student -> level string (the grade suffix)
  - tb        : dict student -> {rbd: lottery}, HIGHER = better. Optional;
                falls back to base rank at matched school if absent.
"""

from collections import defaultdict


# ============================================================
# School-id parsing
# ============================================================

def rbd_of(school_id):
    if school_id is None:
        return None
    return str(school_id).split("_")[0]


def level_of_school(school_id):
    if school_id is None:
        return None
    parts = str(school_id).split("_")
    return parts[1] if len(parts) > 1 else None


def school_id_at(rbd, level):
    return f"{rbd}_{level}"


# ============================================================
# Preference / priority lookups
# ============================================================

def build_rank(pref, ids):
    """Invert pref[id] = {rank: target} into rank_of[id][target] = rank."""
    rank_of = {}
    for k in ids:
        if k not in pref:
            continue
        rank_of[k] = {}
        for r, target in pref[k].items():
            rank_of[k][target] = r
    return rank_of


def strictly_prefers(s, c1, c2, student_rank):
    """c1 ≻_s c2 ?  c2=None is ∅; any listed school beats ∅."""
    if c1 == c2:
        return False
    if c2 is None:
        return c1 in student_rank.get(s, {})
    if c1 is None:
        return False
    r1 = student_rank.get(s, {}).get(c1)
    r2 = student_rank.get(s, {}).get(c2)
    if r1 is None:
        return False
    if r2 is None:
        return True
    return r1 < r2


def weakly_prefers(s, c1, c2, student_rank):
    if c1 == c2:
        return True
    return strictly_prefers(s, c1, c2, student_rank)


def base_outranks(s, s_prime, c, school_rank):
    """s ≻_c s' under base priority? lower rank in pref[c] = higher priority."""
    rs = school_rank.get(c, {}).get(s, float('inf'))
    rsp = school_rank.get(c, {}).get(s_prime, float('inf'))
    return rs < rsp


def family_of(s, siblings):
    return {s} | set(siblings.get(s, []))


# ============================================================
# Base-admissibility (Def 2(iii), strict) at (rbd, level_of_provider)
# ============================================================

def is_base_admissible_at(s, rbd, mu, students, school_rank, student_rank,
                          levels_of, cap):
    """
    Is candidate provider s base-admissible at RBD rbd?
    Uses the slot s competes for: sid = f"{rbd}_{level(s)}".
        |{ s'' in S^{level(s)} : s'' ≻_sid s,  sid ⪰_{s''} mu(s'') }|  <=  q_sid - 1
    """
    level_s = levels_of.get(s)
    sid = school_id_at(rbd, level_s)
    q = cap.get(sid, 0)
    if q <= 0:
        return False
    count_above = 0
    for s_pp in students:
        if s_pp == s:
            continue
        if levels_of.get(s_pp) != level_s:
            continue
        if not base_outranks(s_pp, s, sid, school_rank):
            continue
        if not weakly_prefers(s_pp, sid, mu.get(s_pp), student_rank):
            continue
        count_above += 1
        if count_above > q - 1:
            return False
    return True


# ============================================================
# Provider / effective provider at RBD level
# ============================================================

def is_provider_at_rbd(s, rbd, mu, siblings, students, school_rank,
                       student_rank, levels_of, cap):
    """Definition 2: does s provide contingent priority at RBD rbd?"""
    m = mu.get(s)
    if m is None or rbd_of(m) != rbd:
        return False
    any_sib_wants = False
    for sib in siblings.get(s, []):
        if sib == s:
            continue
        sib_level = levels_of.get(sib)
        sib_school_at_r = school_id_at(rbd, sib_level)
        if weakly_prefers(sib, sib_school_at_r, mu.get(sib), student_rank):
            any_sib_wants = True
            break
    if not any_sib_wants:
        return False
    return is_base_admissible_at(s, rbd, mu, students, school_rank,
                                 student_rank, levels_of, cap)


def compute_z(mu, colleges, siblings, students, school_rank, student_rank,
              levels_of, cap, tb=None):
    """z[rbd][s] = 1 if s is the effective provider in her family at RBD rbd."""
    rbds = sorted(set(rbd_of(c) for c in colleges))
    fam_of = {s: frozenset(family_of(s, siblings)) for s in students}
    distinct_families = list({fam: None for fam in fam_of.values()}.keys())

    def provider_key(s, rbd):
        if tb is not None and s in tb and rbd in tb[s]:
            return (-tb[s][rbd],)          # higher tb = better -> negate for min()
        m = mu.get(s)
        return (school_rank.get(m, {}).get(s, float('inf')),)

    z = {r: {} for r in rbds}
    for r in rbds:
        for fam in distinct_families:
            providers = [
                s for s in fam
                if is_provider_at_rbd(s, r, mu, siblings, students,
                                      school_rank, student_rank, levels_of, cap)
            ]
            if not providers:
                continue
            best = min(providers, key=lambda s: provider_key(s, r))
            z[r][best] = 1
    return z


# ============================================================
# Contingent priority (≻^mu_c)
# ============================================================

def in_upper_tier(s, c, mu, z, siblings):
    """Is s upper-tier at school c (RBD r = rbd_of(c))?"""
    r = rbd_of(c)
    for sib in siblings.get(s, []):
        if sib == s:
            continue
        if z.get(r, {}).get(sib, 0) == 1:
            return True
    if z.get(r, {}).get(s, 0) == 1:
        fam = family_of(s, siblings)
        n_at_rbd = sum(1 for fm in fam
                       if mu.get(fm) is not None and rbd_of(mu[fm]) == r)
        if n_at_rbd >= 2:
            return True
    return False


def contingent_outranks(s, s_prime, c, mu, z, siblings, school_rank):
    """s ≻^mu_c s' ?  Upper tier beats lower; within a tier, base priority."""
    ts = in_upper_tier(s, c, mu, z, siblings)
    tsp = in_upper_tier(s_prime, c, mu, z, siblings)
    if ts != tsp:
        return ts
    return base_outranks(s, s_prime, c, school_rank)


# ============================================================
# Augmentation: reorder school priorities by ≻^mu_c (for heuristics)
# ============================================================

def upper_set_from_mu(mu, students, colleges, pref, cap, siblings, levels_of,
                      tb=None, z=None, colleges_subset=None):
    """
    Return {c: set(students who are upper-tier at c under mu)} for the schools
    in `colleges_subset` (default: all of `colleges`).

    z (effective-provider table) is computed over the FULL `colleges` set so
    that providers at any RBD are found, even when only a subset of schools is
    being augmented (used by the level-stratified heuristics).
    """
    student_rank = build_rank(pref, students)
    school_rank = build_rank(pref, colleges)
    if z is None:
        z = compute_z(mu, colleges, siblings, students, school_rank,
                      student_rank, levels_of, cap, tb=tb)
    target = colleges_subset if colleges_subset is not None else colleges
    upper = {}
    for c in target:
        if c not in pref:
            continue
        upper[c] = set(s for s in pref[c].values()
                       if in_upper_tier(s, c, mu, z, siblings))
    return upper


def augmented_pref_from_upper(upper, students, colleges, pref,
                              colleges_subset=None):
    """
    Return a NEW pref dict where each school's priority list is reordered so
    that students in upper[c] come first, with base priority breaking ties
    within each tier. Student lists are copied unchanged. Schools not in
    `colleges_subset` (default: all) keep their original priority order.
    """
    school_rank = build_rank(pref, colleges)
    pref_aug = {}
    for s in students:
        if s in pref:
            pref_aug[s] = dict(pref[s])
    target = set(colleges_subset) if colleges_subset is not None else set(colleges)
    for c in colleges:
        if c not in pref:
            continue
        if c not in target:
            pref_aug[c] = dict(pref[c])
            continue
        U = upper.get(c, set())
        applicants = list(pref[c].values())

        def key(s, c=c, U=U):
            base_r = school_rank.get(c, {}).get(s, float('inf'))
            return (0 if s in U else 1, base_r)

        applicants_sorted = sorted(applicants, key=key)
        pref_aug[c] = {i + 1: applicants_sorted[i]
                       for i in range(len(applicants_sorted))}
    return pref_aug


def augmented_pref(mu, students, colleges, pref, cap, siblings, levels_of,
                   tb=None, z=None):
    """
    Reorder every school's priority by the absolute contingent priority ≻^mu_c
    induced by `mu`: upper-tier first, base priority within a tier. Thin wrapper
    over upper_set_from_mu + augmented_pref_from_upper.
    """
    upper = upper_set_from_mu(mu, students, colleges, pref, cap, siblings,
                              levels_of, tb=tb, z=z)
    return augmented_pref_from_upper(upper, students, colleges, pref)


# ============================================================
# build levels_of from preferences (grade suffix of top choice)
# ============================================================

def build_levels_of(students, pref):
    """student -> level = grade suffix of the student's top preference."""
    levels_of = {}
    for s in students:
        plist = pref.get(s, {})
        if plist:
            first = plist[min(plist)]
            levels_of[s] = level_of_school(first)
    return levels_of