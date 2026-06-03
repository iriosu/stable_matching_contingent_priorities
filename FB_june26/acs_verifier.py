"""
acs_verifier.py
================
Verifier for Absolute Contingent Stability (ACS).

This is the SELF-CONTAINED version (no acs_priority dependency), patched to
add `exclude_sibling_envy` to check_acs and `witness_is_sibling` to each
blocking pair record. 
"""

from collections import defaultdict


# ============================================================
# School ID parsing
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
    """Reconstruct the school_id for (rbd, level)."""
    return f"{rbd}_{level}"


# ============================================================
# Preference / priority lookups
# ============================================================

def _build_rank(pref, ids):
    """Invert pref[id] = {rank: target} into rank_of[id][target] = rank."""
    rank_of = {}
    for k in ids:
        if k not in pref:
            continue
        rank_of[k] = {}
        for r, target in pref[k].items():
            rank_of[k][target] = r
    return rank_of


def _strictly_prefers(s, c1, c2, student_rank):
    """c1 ≻_s c2 ?  c2=None is the outside option; any listed school beats ∅."""
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


def _weakly_prefers(s, c1, c2, student_rank):
    """c1 ⪰_s c2 ?"""
    if c1 == c2:
        return True
    return _strictly_prefers(s, c1, c2, student_rank)


def _base_outranks(s, s_prime, c, school_rank):
    """s ≻_c s' under base priority?  lower rank in pref[c] = higher priority."""
    rs = school_rank.get(c, {}).get(s, float('inf'))
    rsp = school_rank.get(c, {}).get(s_prime, float('inf'))
    return rs < rsp


def _family_of(s, siblings):
    return {s} | set(siblings.get(s, []))


# ============================================================
# Base-admissibility (Def 2(iii), strict) at (rbd, level_of_provider)
# ============================================================

def _is_base_admissible_at(s, rbd, mu, students, school_rank, student_rank,
                            levels_of, cap):
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
        if not _base_outranks(s_pp, s, sid, school_rank):
            continue
        if not _weakly_prefers(s_pp, sid, mu.get(s_pp), student_rank):
            continue
        count_above += 1
        if count_above > q - 1:
            return False
    return True


# ============================================================
# Provider / effective-provider at RBD level
# ============================================================

def _is_provider_at_rbd(s, rbd, mu, siblings, students, school_rank,
                         student_rank, levels_of, cap):
    m = mu.get(s)
    if m is None or rbd_of(m) != rbd:
        return False
    any_sib_wants = False
    for sib in siblings.get(s, []):
        if sib == s:
            continue
        sib_level = levels_of.get(sib)
        sib_school_at_r = school_id_at(rbd, sib_level)
        if _weakly_prefers(sib, sib_school_at_r, mu.get(sib), student_rank):
            any_sib_wants = True
            break
    if not any_sib_wants:
        return False
    return _is_base_admissible_at(s, rbd, mu, students, school_rank,
                                   student_rank, levels_of, cap)


def _compute_z(mu, colleges, siblings, students, school_rank, student_rank,
                levels_of, cap, tb=None):
    rbds = sorted(set(rbd_of(c) for c in colleges))
    fam_of = {s: frozenset(_family_of(s, siblings)) for s in students}
    distinct_families = list({fam: None for fam in fam_of.values()}.keys())

    def provider_key(s, rbd):
        if tb is not None and s in tb and rbd in tb[s]:
            return (-tb[s][rbd],)
        m = mu.get(s)
        return (school_rank.get(m, {}).get(s, float('inf')),)

    z = {r: {} for r in rbds}
    for r in rbds:
        for fam in distinct_families:
            providers = [
                s for s in fam
                if _is_provider_at_rbd(s, r, mu, siblings, students,
                                        school_rank, student_rank, levels_of, cap)
            ]
            if not providers:
                continue
            best = min(providers, key=lambda s: provider_key(s, r))
            z[r][best] = 1
    return z


# ============================================================
# Contingent priority (≻^µ_c)
# ============================================================

def _in_upper_tier(s, c, mu, z, siblings):
    r = rbd_of(c)
    for sib in siblings.get(s, []):
        if sib == s:
            continue
        if z.get(r, {}).get(sib, 0) == 1:
            return True
    if z.get(r, {}).get(s, 0) == 1:
        fam = _family_of(s, siblings)
        n_at_rbd = sum(
            1 for fm in fam
            if mu.get(fm) is not None and rbd_of(mu[fm]) == r
        )
        if n_at_rbd >= 2:
            return True
    return False


def _contingent_outranks(s, s_prime, c, mu, z, siblings, school_rank):
    ts = _in_upper_tier(s, c, mu, z, siblings)
    tsp = _in_upper_tier(s_prime, c, mu, z, siblings)
    if ts != tsp:
        return ts
    return _base_outranks(s, s_prime, c, school_rank)


# ============================================================
# Main entry point  (patched: + exclude_sibling_envy, + witness_is_sibling)
# ============================================================

def check_acs(mu, students, colleges, pref, cap, siblings, levels_of,
              tb=None, verbose=False, max_blocking_to_report=None,
              exclude_sibling_envy=False):
    """
    Verify whether µ is ACS.

    exclude_sibling_envy : if True, a student is NOT considered to have justified
        envy toward a same-level SIBLING. This matches the convention the
        Absolute-Hard IP and the old verifier use (a family does not envy
        itself). Default False = literal Definition 4 (no sibling exclusion).

    Returns a dict with is_acs, n_blocking_pairs, blocking_pairs
    (each carrying witness_is_sibling), n_providers_total, truncated.
    """
    students = list(students)
    colleges = list(colleges)
    student_rank = _build_rank(pref, students)
    school_rank = _build_rank(pref, colleges)

    z = _compute_z(mu, colleges, siblings, students, school_rank,
                    student_rank, levels_of, cap, tb=tb)
    n_providers_total = sum(1 for r in z for s in z[r] if z[r][s] == 1)

    matched_by_school = defaultdict(list)
    for s in students:
        c = mu.get(s)
        if c is not None:
            matched_by_school[c].append(s)

    blocking = []
    truncated = False

    for s in students:
        s_match = mu.get(s)
        fam_s = set(siblings.get(s, [])) if exclude_sibling_envy else set()
        student_pref = pref.get(s, {})
        for r in sorted(student_pref):
            c = student_pref[r]
            if c == s_match:
                break
            if not _strictly_prefers(s, c, s_match, student_rank):
                continue
            level_s = levels_of.get(s)
            q = cap.get(c, 0)
            matched_at_c = matched_by_school.get(c, [])
            count_above = 0
            for s_prime in matched_at_c:
                if levels_of.get(s_prime) != level_s:
                    continue
                if _contingent_outranks(s_prime, s, c, mu, z, siblings, school_rank):
                    count_above += 1
                elif exclude_sibling_envy and s_prime in fam_s:
                    # a sibling legitimately holds the seat; s does not envy it
                    count_above += 1
            if count_above >= q:
                continue
            if len([x for x in matched_at_c if levels_of.get(x) == level_s]) < q:
                btype = "wasteful"
                witness = None
            else:
                btype = "envy"
                witness = None
                for s_prime in matched_at_c:
                    if levels_of.get(s_prime) != level_s:
                        continue
                    if exclude_sibling_envy and s_prime in fam_s:
                        continue
                    if not _contingent_outranks(s_prime, s, c, mu, z, siblings, school_rank):
                        witness = s_prime
                        break
            base_adm_s = _is_base_admissible_at(
                s, rbd_of(c), mu, students, school_rank, student_rank, levels_of, cap
            )
            witness_is_sibling = (witness in set(siblings.get(s, []))) if witness else False
            blocking.append({
                "s": s, "c": c, "level": level_s, "q": q,
                "count_above": count_above, "type": btype,
                "witness_s_prime": witness, "base_admissible_s": base_adm_s,
                "witness_is_sibling": witness_is_sibling,
            })
            if verbose:
                print(f"  BLOCK s={s} c={c} type={btype} "
                      f"count={count_above}/q={q} witness={witness} "
                      f"base_adm_s={base_adm_s}")
            if max_blocking_to_report is not None and len(blocking) >= max_blocking_to_report:
                truncated = True
                break
        if truncated:
            break

    return {
        "is_acs": (len(blocking) == 0 and not truncated),
        "n_blocking_pairs": len(blocking),
        "blocking_pairs": blocking,
        "n_providers_total": n_providers_total,
        "truncated": truncated,
    }


def summarize(result):
    if result["truncated"]:
        return f"NOT ACS (truncated at >= {result['n_blocking_pairs']} blocking pairs)"
    if result["is_acs"]:
        return f"ACS (0 blocking pairs, {result['n_providers_total']} effective-provider slots)"
    n_w = sum(1 for b in result["blocking_pairs"] if b["type"] == "wasteful")
    n_e = sum(1 for b in result["blocking_pairs"] if b["type"] == "envy")
    return (f"NOT ACS ({result['n_blocking_pairs']} blocking: "
            f"{n_w} wasteful, {n_e} envy; "
            f"{result['n_providers_total']} provider slots)")