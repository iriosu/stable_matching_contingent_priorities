"""
mechanisms.py
=============
Implementations of the 8 mechanisms from the two-stage sibling paper.
All functions follow the codebase convention:
  - inputs = (students, colleges, pref, cap, siblings, levels, students_per_level)
  - tb     = dict  tb[student][rbd] -> float  (higher = better priority)
  - output = {'status': 'completed', 'x_opt': {s: {c: 1}}, 'runtime': ..., ...}

Course format: "RBD_level"   e.g. "12345_3"
School (RBD):  "12345"       = course.split("_")[0]

Mechanisms
----------
1. plain_DA           – standard DA (no sibling coordination)
2. FSD_U              – unconditional family boost, single DA
3. SF_Reserve         – reserve seats for siblings, single DA
4. SF_SD              – siblings-first serial dictatorship, then DA
5. SF_DA              – siblings-first family DA, then DA
6. SF_AP_DA           – siblings-first DA with absolute priorities, then DA
7. FSD                – DA first, freeze singletons, family SD
8. FSD_A              – DA first, absolute boost, re-run DA
"""

import copy
import time
import math
from collections import defaultdict, deque
from itertools import combinations

import algorithms as alg


# ============================================================
# Helpers
# ============================================================

def school_of(course_id):
    """RBD from course id."""
    return course_id.split("_")[0]


def level_of(course_id):
    """Level string from course id."""
    return course_id.split("_")[1]


def build_families(students, siblings):
    """Connected components of the sibling graph."""
    student_set = set(students)
    visited = set()
    families = []
    for s in students:
        if s in visited:
            continue
        q = deque([s])
        fam = []
        while q:
            u = q.popleft()
            if u in visited:
                continue
            visited.add(u)
            fam.append(u)
            for v in siblings.get(u, []):
                if v in student_set and v not in visited:
                    q.append(v)
        families.append(sorted(fam))
    return families


def match_to_xopt(match):
    return {s: {c: 1} for s, c in match.items() if c is not None}


def make_output(match, start_time, **extra):
    out = {
        "status": "completed",
        "x_opt": match_to_xopt(match),
        "runtime": time.time() - start_time,
        "num_vars": 0,
        "num_cols": 0,
        "mipgap": 0,
        "nodes": 0,
    }
    out.update(extra)
    return out


def student_ranks(pref, students):
    """ranks[s][course] = 1-based rank."""
    ranks = {}
    for s in students:
        ranks[s] = {}
        if s not in pref:
            continue
        for k in sorted(pref[s]):
            ranks[s][pref[s][k]] = k
    return ranks


def best_available(s, pref, residual):
    """Best available course for student s."""
    if s not in pref:
        return None
    for k in sorted(pref[s]):
        c = pref[s][k]
        if residual.get(c, 0) > 0:
            return c
    return None


def family_lottery(family, rbd, tb, rule="min"):
    """Aggregate individual lotteries to a family priority at rbd."""
    vals = []
    for s in family:
        if s in tb and rbd in tb[s]:
            vals.append(tb[s][rbd])
    if not vals:
        return -1e9  # worst possible
    if rule == "min":
        return min(vals)
    elif rule == "max":
        return max(vals)
    elif rule == "avg":
        return sum(vals) / len(vals)
    return min(vals)


# ============================================================
# 1. plain_DA
# ============================================================

def plain_DA(inputs, **kwargs):
    """Standard DA – no sibling coordination."""
    students, colleges, pref, cap, siblings, levels, students_per_level = inputs
    start = time.time()
    match = alg.DA(students, pref, cap)
    return make_output(match, start)


# ============================================================
# 2. FSD_U  (Unconditional Family Boost)
# ============================================================

def FSD_U(inputs, **kwargs):
    """
    All sibling students get a blanket group promotion at every school.
    Implemented by placing siblings first in each school's priority list
    (preserving relative order within siblings and within non-siblings).
    Single DA run with modified priorities.
    """
    students, colleges, pref, cap, siblings, levels, students_per_level = inputs
    start = time.time()

    has_sib = {s for s in students if len(siblings.get(s, [])) > 0}

    boosted_pref = copy.deepcopy(pref)
    for c in colleges:
        if c not in boosted_pref:
            continue
        current = [boosted_pref[c][k] for k in sorted(boosted_pref[c])]
        sibs = [s for s in current if s in has_sib]
        non_sibs = [s for s in current if s not in has_sib]
        new_order = sibs + non_sibs
        boosted_pref[c] = {i + 1: s for i, s in enumerate(new_order)}

    match = alg.DA(students, boosted_pref, cap)
    return make_output(match, start)


# ============================================================
# 3. SF_Reserve  (Sibling Seat Reserves)
# ============================================================

def SF_Reserve(inputs, alpha=0.2, **kwargs):
    """
    Reserve fraction alpha of seats at each course for sibling students.
    Each course c is split into two slot types:
      - c##COM  (common seats):   accessible by ALL students
      - c##RES  (reserved seats): accessible ONLY by sibling students

    A sibling student's preference list is expanded: for every course c in
    their original list, they list c##COM first, then c##RES immediately
    after.  If rejected from the common slot, DA automatically proposes to
    the reserved slot next.

    A singleton student's preference list only contains c##COM entries.

    Both slot types use the SAME priority ordering over students (same as
    the original school ordering).  This ensures strategy-proofness
    (Kamada-Kojima 2015; no slot-specific priority concern from
    Dur-Faenza-Gupta-Saban-Sethuraman 2023).

    A single DA is run on this expanded market.
    """
    students, colleges, pref, cap, siblings, levels, students_per_level = inputs
    start = time.time()

    COM = "##COM"
    RES = "##RES"

    has_sib = {s for s in students if len(siblings.get(s, [])) > 0}

    # ---- Build expanded capacities ----
    exp_cap = {}
    for c in cap:
        n_res = max(1, int(math.floor(alpha * cap[c]))) if cap[c] >= 2 else 0
        n_com = cap[c] - n_res
        exp_cap[c + COM] = n_com
        if n_res > 0:
            exp_cap[c + RES] = n_res

    # ---- Build expanded student preferences ----
    exp_pref = {}
    for s in students:
        if s not in pref:
            continue
        exp_pref[s] = {}
        idx = 1
        for k in sorted(pref[s]):
            c = pref[s][k]
            # Common slot (everyone)
            c_com = c + COM
            if c_com in exp_cap:
                exp_pref[s][idx] = c_com
                idx += 1
            # Reserved slot (siblings only, immediately after common)
            if s in has_sib:
                c_res = c + RES
                if c_res in exp_cap:
                    exp_pref[s][idx] = c_res
                    idx += 1

    # ---- Build expanded school priorities ----
    # Both COM and RES slots at course c use the same student ordering
    # as the original course c.
    exp_colleges = list(exp_cap.keys())
    for c in colleges:
        if c not in pref:
            continue
        # Original ordering of applicants at c
        original_order = [pref[c][k] for k in sorted(pref[c])]

        # COM slot: all applicants (everyone lists c##COM)
        c_com = c + COM
        if c_com in exp_cap:
            exp_pref[c_com] = {i + 1: s for i, s in enumerate(original_order)}

        # RES slot: only sibling applicants (only siblings list c##RES)
        c_res = c + RES
        if c_res in exp_cap:
            sib_order = [s for s in original_order if s in has_sib]
            if sib_order:
                exp_pref[c_res] = {i + 1: s for i, s in enumerate(sib_order)}
            else:
                # No sibling applicants: remove this slot
                del exp_cap[c_res]

    # ---- Run single DA on expanded market ----
    exp_colleges = list(exp_cap.keys())
    match_exp = alg.DA(students, exp_pref, exp_cap)

    # ---- Map back to original courses ----
    final = {}
    for s in students:
        slot = match_exp.get(s)
        if slot is None:
            final[s] = None
        else:
            # Strip the ##COM or ##RES suffix to get the original course
            original_course = slot.replace(RES, "").replace(COM, "")
            final[s] = original_course

    return make_output(final, start)


# ============================================================
# 4. SF_SD  (Siblings First – Serial Dictatorship)
# ============================================================

def _can_place_subset_at_school(subset, rbd, pref, ranks, residual, colleges):
    """
    Check whether ALL members of subset can be simultaneously assigned to
    courses at school rbd, respecting residual capacities.

    Returns (feasible, assignment_dict, score) where:
      - assignment_dict maps student -> course
      - score = sum of ranks (lower is better)

    The assignment is computed greedily: for each student, pick the
    best-ranked course at rbd that still has residual capacity after
    accounting for seats already claimed by earlier members in this call.
    """
    local_demand = defaultdict(int)   # course -> seats claimed so far
    assignment = {}
    score = 0

    for s in subset:
        # All courses at this school that s has listed, sorted by s's rank
        candidates = sorted(
            [c for c in ranks.get(s, {}) if school_of(c) == rbd],
            key=lambda c: ranks[s][c]
        )
        placed = False
        for c in candidates:
            if local_demand[c] + 1 <= residual.get(c, 0):
                assignment[s] = c
                local_demand[c] += 1
                score += ranks[s][c]
                placed = True
                break
        if not placed:
            return False, {}, float('inf')

    return True, assignment, score


def _find_best_joint(family, pref, ranks, residual):
    """
    Iterative co-assignment for a family.

    Algorithm
    ---------
    Let U = unmatched family members.

    For k = |U|, |U|-1, ..., 2:
        For each subset P ⊆ U with |P| = k:
            1. common_schools = {rbd : every s ∈ P lists at least one course
               at rbd in Pref(s)}.
            2. For each rbd ∈ common_schools, check whether rbd has enough
               residual capacity to place ALL members of P simultaneously
               (each at some course at rbd).  Remove infeasible schools.
            3. Rank the remaining feasible schools by
               W(f, rbd, P) = −Σ_{s∈P} r_{s,c_s}  (sum of assigned ranks,
               lower is better).
            4. Track the best (rbd, P, assignment) across all subsets of
               size k.
        If a feasible assignment was found at size k:
            commit it, update residual, remove assigned members from U,
            and restart the outer loop (try to match remaining members).
        If no subset of size k works: decrease k.

    When k = 1: remaining members are unmatched and go to Stage 2.

    Returns
    -------
    assignments : list of dict
        Each dict maps student -> course for one co-assigned group.
    remaining : list
        Students in U that could not be co-assigned with any sibling.
    """
    U = list(family)
    assignments = []

    # Precompute: for each student, the set of RBDs they list
    student_rbds = {}
    for s in U:
        student_rbds[s] = {school_of(c) for c in ranks.get(s, {})}

    # All courses indexed by RBD (needed for capacity checks)
    all_colleges = set(residual.keys())

    while len(U) >= 2:
        best_score = float('inf')
        best_assignment = None
        best_k = 0
        found = False

        for k in range(len(U), 1, -1):
            for subset in combinations(U, k):
                # Step 1: common schools = RBDs listed by every member of subset
                common_schools = student_rbds[subset[0]].copy()
                for s in subset[1:]:
                    common_schools = common_schools & student_rbds[s]

                if not common_schools:
                    continue

                # Step 2 & 3: for each common school, check capacity and compute W
                for rbd in common_schools:
                    feasible, asgn, score = _can_place_subset_at_school(
                        subset, rbd, pref, ranks, residual, all_colleges
                    )
                    if not feasible:
                        continue

                    # Track the best: prefer larger k, then lower score
                    if (k > best_k) or (k == best_k and score < best_score):
                        best_score = score
                        best_assignment = asgn
                        best_k = k

            # If we found any feasible assignment at this k, commit it
            # (don't try smaller k for this round)
            if best_assignment is not None:
                found = True
                break

        if not found:
            break  # no subset of size ≥ 2 can be co-assigned; done

        # Commit the best assignment
        assignments.append(best_assignment)
        for s in best_assignment:
            residual[best_assignment[s]] -= 1
            U.remove(s)
            # Update student_rbds (no longer needed for removed students,
            # but remove to keep U and student_rbds consistent)
            if s in student_rbds:
                del student_rbds[s]

    return assignments, U


def SF_SD(inputs, tb=None, family_order_rule="min", **kwargs):
    """
    Siblings-First Serial Dictatorship:
    Stage 1: process sibling families in order pi (from aggregated lottery).
             For each family, iteratively co-assign (k=|f|,...,2) at best
             school under W = -sum(ranks). Unmatched siblings -> Stage 2.
    Stage 2: DA for singletons + unmatched siblings over residual capacity.
    """
    students, colleges, pref, cap, siblings, levels, students_per_level = inputs
    start = time.time()

    families = build_families(students, siblings)
    sib_families = [f for f in families if len(f) >= 2]
    sin_students = [s for s in students if len(siblings.get(s, [])) == 0]

    ranks = student_ranks(pref, students)
    residual = {c: cap[c] for c in cap}

    # Family ordering: by aggregated lottery (or random if no tb)
    if tb is not None:
        # Aggregate across all RBDs listed by family members
        def fam_key(fam):
            vals = []
            for s in fam:
                if s in tb:
                    vals.extend(tb[s].values())
            if not vals:
                return -1e9
            if family_order_rule == "avg":
                return -(sum(vals) / len(vals))
            return -max(vals)  # "min" rule: best individual lottery (highest tb)
        sib_families.sort(key=fam_key)
    # else: process in whatever order they were built

    # Stage 1: family serial dictatorship
    final = {}
    unmatched_sibs = []

    for fam in sib_families:
        assignments, remaining = _find_best_joint(fam, pref, ranks, residual)
        for asgn in assignments:
            for s, c in asgn.items():
                final[s] = c
        unmatched_sibs.extend(remaining)

    # Stage 2: DA for singletons + unmatched siblings
    stage2_students = sin_students + unmatched_sibs
    stage2_set = set(stage2_students)

    stage2_pref = {}
    for s in stage2_students:
        if s in pref:
            stage2_pref[s] = pref[s]
    for c in colleges:
        if c not in pref:
            continue
        new_p, idx = {}, 1
        for k in sorted(pref[c]):
            if pref[c][k] in stage2_set:
                new_p[idx] = pref[c][k]
                idx += 1
        if new_p:
            stage2_pref[c] = new_p

    match2 = alg.DA(stage2_students, stage2_pref, residual)
    for s in stage2_students:
        if s not in final:
            final[s] = match2.get(s)

    return make_output(final, start)


# ============================================================
# 5. SF_DA  (Siblings First – Family DA)
# ============================================================

def SF_DA(inputs, tb=None, **kwargs):
    """
    Siblings-First Family DA (revised):

    Stage 1 — Build family preferences (same ranking as SF-SD):
      For each family f, enumerate schools (RBDs) by how many siblings
      share them: first schools listed by |f| siblings (ranked by W),
      then |f|-1, ..., down to 2.  Each entry records which subset of
      siblings can be placed there and the W-score.

    Family priority at each school = max individual lottery among family
    members at that school.  This privileges larger families (more draws).

    Run a family-proposing DA where each family may demand a variable
    number of seats (2, 3, or |f|) depending on which school it proposes to.
    Schools accept/reject families by family priority, respecting per-level
    capacity.

    Stage 2 — Fix only siblings co-assigned with another sibling.
    Siblings matched alone + singletons enter Stage-2 DA over residual.
    """
    students, colleges, pref, cap, siblings, levels, students_per_level = inputs
    start = time.time()

    families_all = build_families(students, siblings)
    sib_families = [f for f in families_all if len(f) >= 2]
    sin_students = [s for s in students if len(siblings.get(s, [])) == 0]
    ranks = student_ranks(pref, students)

    # ---- Courses indexed by RBD ----
    courses_by_rbd = defaultdict(list)
    for c in colleges:
        courses_by_rbd[school_of(c)].append(c)

    # ---- Student RBDs ----
    student_rbds = {}
    for s in students:
        student_rbds[s] = {school_of(c) for c in ranks.get(s, {})}

    # ================================================================
    # Build family preference lists  (same logic as SF-SD)
    # ================================================================
    # For each family, produce an ordered list of (rbd, subset, score).
    # Order: all k=|f| entries first (by score), then k=|f|-1, ..., k=2.
    #
    # IMPORTANT: the same school (RBD) can appear at MULTIPLE k-levels.
    # If a family of 3 gets rejected from school A asking for 3 seats,
    # they should be able to try school A again asking for 2 seats.
    #
    # Within the same k, for each school we keep only the best subset
    # (lowest W-score), since if the school rejects the best subset of
    # size k, it would also reject worse subsets of the same size
    # (same or higher level demands, worse score).

    fam_pref_list = {}   # fam_index -> [(rbd, subset_tuple, score, assignment), ...]
    fam_members = {}     # fam_index -> list of students

    for fi, fam in enumerate(sib_families):
        fam_members[fi] = fam

        # Collect all feasible (k, rbd, subset, score) entries
        # For each (k, rbd), keep only the best subset
        best_by_k_rbd = {}  # (neg_k, rbd) -> (score, subset, assignment)

        for k in range(len(fam), 1, -1):
            for subset in combinations(fam, k):
                # Common RBDs for this subset
                common = student_rbds.get(subset[0], set()).copy()
                for s in subset[1:]:
                    common = common & student_rbds.get(s, set())

                for rbd in common:
                    feasible, assignment, score = _can_place_subset_at_school(
                        subset, rbd, pref, ranks,
                        {c: cap[c] for c in cap},  # full capacity for ranking
                        set(cap.keys())
                    )
                    if not feasible:
                        continue

                    key = (-k, rbd)
                    if key not in best_by_k_rbd or score < best_by_k_rbd[key][0]:
                        best_by_k_rbd[key] = (score, subset, assignment)

        # Build ordered list: sort by (-k asc, score asc)
        # This gives: k=|f| entries first (by score), then k=|f|-1, etc.
        sorted_keys = sorted(best_by_k_rbd.keys(), key=lambda x: (x[0], best_by_k_rbd[x][0]))
        ordered = []
        for key in sorted_keys:
            neg_k, rbd = key
            score, subset, assignment = best_by_k_rbd[key]
            ordered.append((rbd, subset, score, assignment))

        fam_pref_list[fi] = ordered

    # ================================================================
    # Family priority at each school = max individual lottery
    # ================================================================
    # fam_priority[fi][rbd] = max tb[s][rbd] over s in family
    # Higher = better priority (same convention as tb)

    fam_priority = {}
    for fi, fam in fam_members.items():
        fam_priority[fi] = {}
        for rbd in set().union(*(student_rbds.get(s, set()) for s in fam)):
            best = -1e9
            for s in fam:
                if tb is not None and s in tb and rbd in tb[s]:
                    best = max(best, tb[s][rbd])
            fam_priority[fi][rbd] = best

    # ================================================================
    # Family-proposing DA with variable demand
    # ================================================================
    # Each family proposes to schools in order of fam_pref_list.
    # A school (RBD) tentatively holds families by priority (max lottery).
    # When a family proposes, the school checks if per-level capacity
    # can accommodate the family's demand alongside currently held families.
    # If not enough room, the lowest-priority held family is rejected.

    n_fams = len(sib_families)
    fam_pointer = {fi: 0 for fi in range(n_fams)}          # next school to propose to
    fam_matched = {fi: False for fi in range(n_fams)}       # done flag
    proposals = defaultdict(list)  # rbd -> list of (fi, subset, assignment)

    rejected = list(range(n_fams))  # initially all families propose

    for _round in range(5000):  # safety limit
        if not rejected:
            break

        new_rejected = []

        for fi in rejected:
            if fam_matched[fi]:
                continue
            plist = fam_pref_list[fi]
            ptr = fam_pointer[fi]
            if ptr >= len(plist):
                fam_matched[fi] = True  # exhausted list
                continue
            rbd, subset, score, assignment = plist[ptr]
            proposals[rbd].append((fi, subset, assignment))

        rejected = []

        # Each school processes proposals
        for rbd in list(proposals.keys()):
            held = proposals[rbd]
            if not held:
                continue

            # Sort by family priority (higher = better), descending
            held.sort(key=lambda x: -fam_priority[x[0]][rbd])

            # Greedily accept families checking per-level capacity
            level_used = defaultdict(int)  # level -> seats used
            level_cap = defaultdict(int)   # level -> total capacity at this rbd
            for c in courses_by_rbd.get(rbd, []):
                lev = level_of(c)
                level_cap[lev] += cap[c]

            accepted = []
            for fi, subset, assignment in held:
                # Compute demand by level for this family's assignment
                demand_by_level = defaultdict(int)
                for s, course in assignment.items():
                    demand_by_level[level_of(course)] += 1

                # Check if adding this family is feasible
                fits = True
                for lev, need in demand_by_level.items():
                    if level_used[lev] + need > level_cap[lev]:
                        fits = False
                        break

                if fits:
                    accepted.append((fi, subset, assignment))
                    for lev, need in demand_by_level.items():
                        level_used[lev] += need
                else:
                    # Rejected: advance pointer, add to rejected list
                    fam_pointer[fi] += 1
                    rejected.append(fi)

            proposals[rbd] = accepted

    # ================================================================
    # Collect results: assign students, fix only co-assigned
    # ================================================================
    final = {}
    residual = {c: cap[c] for c in cap}
    unmatched_sibs = []

    # For each school, get the accepted families
    for rbd, held_list in proposals.items():
        for fi, subset, assignment in held_list:
            fam = fam_members[fi]
            placed_at_rbd = set()

            for s, course in assignment.items():
                if residual.get(course, 0) > 0:
                    final[s] = course
                    residual[course] -= 1
                    placed_at_rbd.add(s)

            # Members not in this assignment
            for s in fam:
                if s not in placed_at_rbd and s not in final:
                    pass  # will be handled below

    # Identify who is co-assigned (at least 2 family members at same RBD)
    # and who is matched alone
    co_assigned = set()
    for fi, fam in fam_members.items():
        # Check which members are in final and at which RBD
        rbd_groups = defaultdict(list)
        for s in fam:
            if s in final:
                rbd_groups[school_of(final[s])].append(s)

        for rbd, group in rbd_groups.items():
            if len(group) >= 2:
                co_assigned.update(group)

    # Unfix siblings matched alone: release their seat, send to Stage 2
    for fi, fam in fam_members.items():
        for s in fam:
            if s in final and s not in co_assigned:
                # Matched alone -> release seat, go to Stage 2
                c = final[s]
                residual[c] += 1
                del final[s]
                unmatched_sibs.append(s)
            elif s not in final:
                unmatched_sibs.append(s)

    # ================================================================
    # Stage 2: DA for singletons + unmatched/unfixed siblings
    # ================================================================
    stage2_students = sin_students + unmatched_sibs
    stage2_set = set(stage2_students)

    stage2_pref = {}
    for s in stage2_students:
        if s in pref:
            stage2_pref[s] = pref[s]
    for c in colleges:
        if c not in pref:
            continue
        new_p, idx = {}, 1
        for k in sorted(pref[c]):
            if pref[c][k] in stage2_set:
                new_p[idx] = pref[c][k]
                idx += 1
        if new_p:
            stage2_pref[c] = new_p

    match2 = alg.DA(stage2_students, stage2_pref, residual)
    for s in stage2_students:
        if s not in final:
            final[s] = match2.get(s)

    return make_output(final, start)


# ============================================================
# 6. SF_AP_DA  (Siblings First – Absolute Priority DA)
# ============================================================

def SF_AP_DA(inputs, tb=None, max_iter=5, **kwargs):
    """
    Stage 1: Run DA restricted to sibling students.  After each DA round,
             check for co-assigned siblings; grant absolute priority (move to
             top of school list) at schools where siblings are co-assigned.
             Iterate until convergence or max_iter.
             Fix co-assigned siblings.
    Stage 2: DA for singletons + unmatched siblings over residual.
    """
    students, colleges, pref, cap, siblings, levels, students_per_level = inputs
    start = time.time()

    has_sib = {s for s in students if len(siblings.get(s, [])) > 0}
    sib_students = [s for s in students if s in has_sib]
    sin_students = [s for s in students if s not in has_sib]

    # Restrict to sibling students
    sib_set = set(sib_students)
    sib_pref = {}
    for s in sib_students:
        if s in pref:
            sib_pref[s] = pref[s]
    for c in colleges:
        if c not in pref:
            continue
        new_p, idx = {}, 1
        for k in sorted(pref[c]):
            if pref[c][k] in sib_set:
                new_p[idx] = pref[c][k]
                idx += 1
        if new_p:
            sib_pref[c] = new_p

    sib_cap = {c: cap[c] for c in cap}  # full capacity for siblings

    # Iterative DA with absolute priorities
    current_pref = copy.deepcopy(sib_pref)
    match = None
    prev_hash = None

    for it in range(max_iter):
        match = alg.DA(sib_students, current_pref, sib_cap)

        # Check convergence
        h = tuple(sorted((s, match.get(s)) for s in sib_students))
        if h == prev_hash:
            break
        prev_hash = h

        # Find co-assigned siblings and grant boosts
        boosted = set()  # (student, course) pairs
        for s in sib_students:
            c = match.get(s)
            if c is None:
                continue
            rbd = school_of(c)
            for sib in siblings.get(s, []):
                c_sib = match.get(sib)
                if c_sib is not None and school_of(c_sib) == rbd:
                    # Both s and sib at same school -> boost both at all courses at this school
                    for course in [cc for cc in colleges if school_of(cc) == rbd]:
                        boosted.add((s, course))
                        boosted.add((sib, course))

        if not boosted:
            break

        # Update priorities: move boosted students to top
        current_pref = copy.deepcopy(sib_pref)
        for c in colleges:
            if c not in current_pref:
                continue
            curr = [current_pref[c][k] for k in sorted(current_pref[c])]
            top = [s for s in curr if (s, c) in boosted]
            rest = [s for s in curr if (s, c) not in boosted]
            new_order = top + rest
            current_pref[c] = {i + 1: s for i, s in enumerate(new_order)}

    # Identify co-assigned
    fixed = {}
    residual = {c: cap[c] for c in cap}
    unmatched_sibs = []

    for s in sib_students:
        c = match.get(s)
        if c is None:
            unmatched_sibs.append(s)
            continue
        rbd = school_of(c)
        is_coassigned = False
        for sib in siblings.get(s, []):
            c_sib = match.get(sib)
            if c_sib is not None and school_of(c_sib) == rbd:
                is_coassigned = True
                break
        if is_coassigned:
            fixed[s] = c
            residual[c] -= 1
        else:
            unmatched_sibs.append(s)

    # Stage 2
    stage2_students = sin_students + unmatched_sibs
    stage2_set = set(stage2_students)
    stage2_pref = {}
    for s in stage2_students:
        if s in pref:
            stage2_pref[s] = pref[s]
    for c in colleges:
        if c not in pref:
            continue
        new_p, idx = {}, 1
        for k in sorted(pref[c]):
            if pref[c][k] in stage2_set:
                new_p[idx] = pref[c][k]
                idx += 1
        if new_p:
            stage2_pref[c] = new_p

    match2 = alg.DA(stage2_students, stage2_pref, residual)

    final = dict(fixed)
    for s in stage2_students:
        final[s] = match2.get(s)

    return make_output(final, start)


# ============================================================
# 7. FSD  (DA First, Freeze Singletons, Family SD)
# ============================================================

def FSD(inputs, tb=None, family_order_rule="min", **kwargs):
    """
    Stage 1: DA for all students -> mu0.  Fix singletons at mu0.
    Stage 2: Family serial dictatorship for sibling families over
             residual capacity (after removing singleton seats).
    """
    students, colleges, pref, cap, siblings, levels, students_per_level = inputs
    start = time.time()

    # Stage 1: full DA
    match0 = alg.DA(students, pref, cap)

    has_sib = {s for s in students if len(siblings.get(s, [])) > 0}
    sin_students = [s for s in students if s not in has_sib]

    # Fix singletons
    final = {}
    residual = {c: cap[c] for c in cap}
    for s in sin_students:
        c = match0.get(s)
        final[s] = c
        if c is not None:
            residual[c] -= 1

    # Build families
    families = build_families(students, siblings)
    sib_families = [f for f in families if len(f) >= 2]
    ranks = student_ranks(pref, students)

    # Family ordering
    if tb is not None:
        def fam_key(fam):
            vals = []
            for s in fam:
                if s in tb:
                    vals.extend(tb[s].values())
            if not vals:
                return -1e9
            if family_order_rule == "avg":
                return -(sum(vals) / len(vals))
            return -max(vals)
        sib_families.sort(key=fam_key)

    # Stage 2: family SD over residual
    for fam in sib_families:
        assignments, remaining = _find_best_joint(fam, pref, ranks, residual)
        for asgn in assignments:
            for s, c in asgn.items():
                final[s] = c
        # Fallback: greedy for unmatched
        for s in remaining:
            c = best_available(s, pref, residual)
            final[s] = c
            if c is not None:
                residual[c] -= 1

    return make_output(final, start)


# ============================================================
# 8. FSD_A  (DA First, Absolute Boost, Re-run DA)
# ============================================================

def FSD_A(inputs, tb=None, **kwargs):
    """
    Stage 1: DA for all -> mu0.
    Stage 2: For each family f with a member at school c in mu0,
             give all other members of f absolute priority at c.
             Re-run DA for all students with boosted priorities.
    """
    students, colleges, pref, cap, siblings, levels, students_per_level = inputs
    start = time.time()

    # Stage 1
    match0 = alg.DA(students, pref, cap)

    # Find boosts: for each family, find schools where members are assigned
    families = build_families(students, siblings)
    boosted = set()  # (student, course)

    for fam in families:
        if len(fam) < 2:
            continue
        # Schools where family has members
        assigned_rbds = defaultdict(list)
        for s in fam:
            c = match0.get(s)
            if c is not None:
                assigned_rbds[school_of(c)].append(s)

        for rbd, providers in assigned_rbds.items():
            for s in fam:
                if s in providers:
                    continue
                # s is not at this school -> boost s at all courses at this school
                for c in colleges:
                    if school_of(c) == rbd:
                        boosted.add((s, c))

    # Stage 2: re-run DA with boosted priorities
    boosted_pref = copy.deepcopy(pref)
    for c in colleges:
        if c not in boosted_pref:
            continue
        curr = [boosted_pref[c][k] for k in sorted(boosted_pref[c])]
        top = [s for s in curr if (s, c) in boosted]
        rest = [s for s in curr if (s, c) not in boosted]
        new_order = top + rest
        boosted_pref[c] = {i + 1: s for i, s in enumerate(new_order)}

    match1 = alg.DA(students, boosted_pref, cap)
    return make_output(match1, start)


# ============================================================
# Registry for simulations.py integration
# ============================================================

MECHANISMS = {
    "plain_da":   plain_DA,
    "fsd_u":      FSD_U,
    "sf_reserve": SF_Reserve,
    "sf_sd":      SF_SD,
    "sf_da":      SF_DA,
    "sf_ap_da":   SF_AP_DA,
    "fsd_v2":     FSD,
    "fsd_a":      FSD_A,
}


def run_mechanism(name, inputs, tb=None, **kwargs):
    """Dispatch to a mechanism by name."""
    if name not in MECHANISMS:
        raise ValueError(f"Unknown mechanism: {name}. Choose from {list(MECHANISMS.keys())}")
    return MECHANISMS[name](inputs, tb=tb, **kwargs)


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    import os
    import generate_inputs as genin
    import pickle

    base_dir = os.path.dirname(os.path.abspath(__file__))
    region = "OHiggins" # "Magallanes" 
    year = 2023
    tie_breaker = "mtbf"

    indir = os.path.join(base_dir, "..", "R", "intermediate_data", region, str(year))
    instance_file = os.path.join(indir, f"instance_{tie_breaker}.txt")

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn = \
        genin.read_instance(instance_file)

    tb_file = os.path.join(indir, f"tb_{tie_breaker}.pck")
    with open(tb_file, "rb") as f:
        tb = pickle.load(f)

    inputs = (students, colleges, pref, cap, siblings, levels, students_per_level)

    print(f"Instance: {region} {year} ({len(students)} students, {len(colleges)} courses)")
    print(f"Families with siblings: {len([f for f in build_families(students, siblings) if len(f) >= 2])}")
    print()

    for name in MECHANISMS:
        try:
            out = run_mechanism(name, inputs, tb=tb)
            x = out["x_opt"]
            n_assigned = len(x)
            # Count siblings together
            sib_together = 0
            for s in x:
                rbd = list(x[s].keys())[0].split("_")[0]
                for sib in siblings.get(s, []):
                    if sib in x and list(x[sib].keys())[0].split("_")[0] == rbd:
                        sib_together += 1
                        break
            print(f"{name:15s}  assigned={n_assigned:5d}  sib_together={sib_together:4d}  "
                  f"time={out['runtime']:.2f}s")
        except Exception as e:
            print(f"{name:15s}  ERROR: {e}")