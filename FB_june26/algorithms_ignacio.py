"""
algorithms_ignacio.py
=====================
Ignacio's heuristics, repackaged for
compare_heuristics.py.

This is taken verbatim from his algorithms-2.py with two small,
non-substantive changes that keep the algorithms identical but make the file
safe to import alongside Federico's code:

  1. The two module-level `random.seed(1); np.random.seed(1)` calls are
     dropped. The per-sim seeding in
     compare_heuristics.py is the only randomness controller.

  2. `import solve_opt` is dropped: none of the algorithms below call it
"""

import sys
import copy
import time


# ============================================================
# DA
# ============================================================

def DA(students, pref, cap):
    """
    Student-proposing Gale-Shapley deferred acceptance with strict preferences.
    Inputs:
      students : iterable of student ids
      pref     : pref[s] = {rank: school_id}; pref[c] = {rank: student}
                 lower rank = more preferred
      cap      : dict school_id -> int
    Returns:
      match    : dict student -> school_id or None
    """
    cp = {i: 1 for i in students}
    match, is_matched = {i: None for i in students}, {i: False for i in students}

    proposals = {c: [] for c in cap}
    rejected = copy.copy(list(students))
    it = 0
    while True:
        it += 1
        for m in rejected:
            if is_matched[m]:
                continue
            try:
                proposals[pref[m][cp[m]]].append(m)
            except Exception:
                print(m, cp[m])
                sys.exit(1)

        rejected = []
        for c in cap:
            if len(proposals[c]) <= cap[c]:
                continue
            top = [pref[c][k] for k in sorted(pref[c]) if pref[c][k] in proposals[c]]
            for i in range(cap[c], len(proposals[c])):
                proposals[c].remove(top[i])
                rejected.append(top[i])

        for m in rejected:
            cp[m] += 1
            if cp[m] not in pref[m]:
                is_matched[m] = True   # exhausted preferences -> match to self / None

        stop = all(is_matched[i] for i in students)
        if stop or len(rejected) == 0:
            break

    for m in students:
        if is_matched[m]:
            continue
        match[m], is_matched[m] = pref[m][cp[m]], True

    return match


# ============================================================
# Sequential (level-stratified, school-priority update only)
# ============================================================

def Sequential(inputs, levels_to_process=None):
    """
    Process levels in the given order. After each level's DA, mark for every
    sibling of a matched student the schools-at-the-same-RBD as sibling-priority,
    then reorder school priorities accordingly before the next level.
    NOTE: the "provider" condition is the naive one -- no base-admissibility
    check, no lone-provider demotion, no lottery-based effective provider.
    """

    def UpdatePriorities(in_match, colleges, pref, siblings, siblings_priority):
        for id_s in in_match:
            if len(siblings[id_s]) == 0 or in_match[id_s] is None:
                continue
            rbd = in_match[id_s].split("_")[0]
            for sib in siblings[id_s]:
                for p in pref[sib]:
                    if pref[sib][p].split("_")[0] == rbd:
                        siblings_priority[sib][pref[sib][p]] = 1

        out_pref = copy.copy(pref)
        for c in colleges:
            if c not in out_pref:
                continue
            sorted_values = sorted(
                out_pref[c].items(),
                key=lambda item: (-siblings_priority[item[1]][c], item[0]))
            sorted_values_only = [value for key, value in sorted_values]
            out_pref[c] = {p + 1: sorted_values_only[p]
                           for p in range(len(sorted_values_only))}

        return out_pref, siblings_priority

    students, colleges, pref, cap, siblings, levels, students_per_level = inputs

    if levels_to_process is None:
        levels_to_process = [str(idx) for idx in sorted(range(-1, 13), reverse=True)]

    if "0" not in students_per_level or "-1" not in students_per_level:
        levels_to_process = ["PreK" if lev == "-1" else lev for lev in levels_to_process]
        levels_to_process = ["K" if lev == "0" else lev for lev in levels_to_process]

    stime = time.time()
    siblings_priority = {s: {pref[s][p]: 0 for p in pref[s]} for s in students}
    pref_updated = copy.copy(pref)
    match = {}
    for idx in levels_to_process:
        schools_in_level = levels[idx]
        students_in_level = students_per_level[idx]
        students_and_schools_in_level = list(
            set(schools_in_level).union(set(students_in_level)))
        cap_in_level = {i: cap[i] for i in schools_in_level if i in cap}
        pref_in_level = {
            i: pref_updated[i] for i in students_and_schools_in_level if i in pref_updated
        }
        match[idx] = DA(students_in_level, pref_in_level, cap_in_level)

        pref_updated, siblings_priority = UpdatePriorities(
            match[idx], colleges, pref, siblings, siblings_priority)

    x_opt = {
        id_s: {match[idx][id_s]: 1}
        for idx in match
        for id_s in match[idx]
        if match[idx][id_s] is not None
    }
    runtime = time.time() - stime
    return {
        "status": "completed", "x_opt": x_opt, "runtime": runtime,
        "num_vars": 0, "num_cols": 0, "mipgap": 0, "nodes": 0,
    }


# ============================================================
# SequentialBlock (level-stratified, also reorders STUDENT preferences)
# ============================================================

def SequentialBlock(inputs, levels_to_process=None):
    """
    Same as Sequential, but additionally reorders each student's own
    preference list so that schools at which she has sibling priority appear
    first. This pushes students toward sibling-RBD schools even if not their
    truthful top choice.
    """

    def UpdatePriorities(in_match, students, colleges, pref, siblings, siblings_priority):
        for id_s in in_match:
            if len(siblings[id_s]) == 0 or in_match[id_s] is None:
                continue
            rbd = in_match[id_s].split("_")[0]
            for sib in siblings[id_s]:
                for p in pref[sib]:
                    if pref[sib][p].split("_")[0] == rbd:
                        siblings_priority[sib][pref[sib][p]] = 1

        out_pref = copy.copy(pref)
        for c in colleges:
            if c not in out_pref:
                continue
            sorted_values = sorted(
                out_pref[c].items(),
                key=lambda item: (-siblings_priority[item[1]][c], item[0]))
            sorted_values_only = [value for key, value in sorted_values]
            out_pref[c] = {p + 1: sorted_values_only[p]
                           for p in range(len(sorted_values_only))}

        for s in students:
            if s not in out_pref:
                continue
            sorted_values = sorted(
                out_pref[s].items(),
                key=lambda item: (-siblings_priority[s][item[1]], item[0]))
            sorted_values_only = [value for key, value in sorted_values]
            out_pref[s] = {p + 1: sorted_values_only[p]
                           for p in range(len(sorted_values_only))}

        return out_pref, siblings_priority

    students, colleges, pref, cap, siblings, levels, students_per_level = inputs

    if levels_to_process is None:
        levels_to_process = [str(idx) for idx in sorted(range(-1, 13), reverse=True)]

    if "0" not in students_per_level or "-1" not in students_per_level:
        levels_to_process = ["PreK" if lev == "-1" else lev for lev in levels_to_process]
        levels_to_process = ["K" if lev == "0" else lev for lev in levels_to_process]

    stime = time.time()
    siblings_priority = {s: {pref[s][p]: 0 for p in pref[s]} for s in students}
    pref_updated = copy.copy(pref)
    match = {}
    for idx in levels_to_process:
        schools_in_level = levels[idx]
        students_in_level = students_per_level[idx]
        students_and_schools_in_level = list(
            set(schools_in_level).union(set(students_in_level)))
        cap_in_level = {i: cap[i] for i in schools_in_level if i in cap}
        pref_in_level = {
            i: pref_updated[i] for i in students_and_schools_in_level if i in pref_updated
        }
        match[idx] = DA(students_in_level, pref_in_level, cap_in_level)

        pref_updated, siblings_priority = UpdatePriorities(
            match[idx], students, colleges, pref, siblings, siblings_priority)

    x_opt = {
        id_s: {match[idx][id_s]: 1}
        for idx in match
        for id_s in match[idx]
        if match[idx][id_s] is not None
    }
    runtime = time.time() - stime
    return {
        "status": "completed", "x_opt": x_opt, "runtime": runtime,
        "num_vars": 0, "num_cols": 0, "mipgap": 0, "nodes": 0,
    }


# ============================================================
# Simultaneous (all levels, with optional decay)
# ============================================================

def Simultaneous(inputs, decay=None):
    """
    Iterate DA on the full instance; after each pass, update sibling priorities
    from the resulting matching, then re-run DA. Two regimes:
      decay = None  -> monotone cumulative: a granted boost is never revoked.
      decay = 1     -> boost wiped to 0 at the top of every iteration =
                       non-monotone fixed-point dynamics (his "rada").
      decay in (0,1) -> exponential decay each iteration.
    """

    def UpdatePriorities(in_match, colleges, pref, siblings, siblings_priority, decay):
        if decay is not None:
            siblings_priority = {
                sib: {k: v * (1 - decay) for k, v in inner.items()}
                for sib, inner in siblings_priority.items()
            }

        for id_s in in_match:
            if len(siblings[id_s]) == 0 or in_match[id_s] is None:
                continue
            rbd = in_match[id_s].split("_")[0]
            for sib in siblings[id_s]:
                for p in pref[sib]:
                    if pref[sib][p].split("_")[0] == rbd:
                        siblings_priority[sib][pref[sib][p]] = 1            # receiver
                        siblings_priority[id_s][in_match[id_s]] = 1         # provider self-prot.

        out_pref = copy.copy(pref)
        for c in colleges:
            if c not in out_pref:
                continue
            sorted_values = sorted(
                out_pref[c].items(),
                key=lambda item: (-siblings_priority[item[1]][c], item[0]))
            sorted_values_only = [value for key, value in sorted_values]
            out_pref[c] = {p + 1: sorted_values_only[p]
                           for p in range(len(sorted_values_only))}

        return out_pref, siblings_priority

    students, colleges, pref, cap, siblings = inputs

    stime = time.time()
    siblings_priority = {s: {pref[s][p]: 0 for p in pref[s]} for s in students}
    pref_updated = copy.copy(pref)
    match = {}
    idx = 0
    while True:
        match[idx] = DA(students, pref_updated, cap)

        if idx > 0 and all(match[idx][s] == match[idx - 1][s] for s in match[idx]):
            break
        pref_updated, siblings_priority = UpdatePriorities(
            match[idx], colleges, pref, siblings, siblings_priority, decay)
        idx += 1

    x_opt = {
        id_s: {match[idx][id_s]: 1}
        for idx in match
        for id_s in match[idx]
        if match[idx][id_s] is not None
    }
    runtime = time.time() - stime
    return {
        "status": "completed", "x_opt": x_opt, "runtime": runtime,
        "num_vars": 0, "num_cols": 0, "mipgap": 0, "nodes": 0,
    }


# ============================================================
# SizeSequential (process families by size)
# ============================================================

def SizeSequential(inputs, direction="decreasing", fix=False):
    """
    Process students cohort by cohort, where a cohort is defined by family size.
      direction = "decreasing": cohort sizes max .. 1
      direction = "increasing": cohort sizes 1 .. max
      fix = True : capacities decrement after each cohort (greedy);
                   cohort = students with |f(s)| EXACTLY equal to size.
      fix = False: no decrement; cohort = students with |f(s)| >= size (dec)
                                          or |f(s)| <= size (inc).
                   each iteration's match overwrites the previous in x_opt;
                   the last iteration (size=1 or max) includes ALL students.
    """

    def UpdatePriorities(in_match, colleges, pref, siblings, siblings_priority):
        for id_s in in_match:
            if len(siblings[id_s]) == 0 or in_match[id_s] is None:
                continue
            rbd = in_match[id_s].split("_")[0]
            for sib in siblings[id_s]:
                for p in pref[sib]:
                    if pref[sib][p].split("_")[0] == rbd:
                        siblings_priority[sib][pref[sib][p]] = 1
                        siblings_priority[id_s][in_match[id_s]] = 1

        out_pref = copy.copy(pref)
        for c in colleges:
            if c not in out_pref:
                continue
            sorted_values = sorted(
                out_pref[c].items(),
                key=lambda item: (-siblings_priority[item[1]][c], item[0]))
            sorted_values_only = [value for key, value in sorted_values]
            out_pref[c] = {p + 1: sorted_values_only[p]
                           for p in range(len(sorted_values_only))}

        return out_pref, siblings_priority

    def UpdateCapacities(in_match, in_cap, fix):
        out_cap = copy.copy(in_cap)
        if not fix:
            return out_cap
        for id_s in in_match:
            if in_match[id_s] is not None:
                out_cap[in_match[id_s]] -= 1
        return out_cap

    students, colleges, pref, cap, siblings = inputs

    if direction == "decreasing":
        sizes_to_process = [
            idx for idx in sorted(
                range(max([len(siblings[id_s]) for id_s in siblings]) + 1),
                reverse=True)
        ]
    else:
        sizes_to_process = [
            idx for idx in sorted(
                range(max([len(siblings[id_s]) for id_s in siblings]) + 1))
        ]

    stime = time.time()
    siblings_priority = {s: {pref[s][p]: 0 for p in pref[s]} for s in students}
    pref_updated = copy.copy(pref)
    cap_updated = copy.copy(cap)
    match = {}
    for size in sizes_to_process:
        students_in_size = (
            {id_s for id_s in students if len(siblings[id_s]) >= size}
            if not fix
            else {id_s for id_s in students if len(siblings[id_s]) == size}
        )
        students_and_schools_in_size = list(set(colleges).union(set(students_in_size)))
        pref_in_size = {
            i: pref_updated[i] for i in students_and_schools_in_size if i in pref_updated
        }
        match[size] = DA(students_in_size, pref_in_size, cap_updated)

        pref_updated, siblings_priority = UpdatePriorities(
            match[size], colleges, pref, siblings, siblings_priority)
        cap_updated = UpdateCapacities(match[size], cap_updated, fix)

    x_opt = {
        id_s: {match[size][id_s]: 1}
        for size in match
        for id_s in match[size]
        if match[size][id_s] is not None
    }
    runtime = time.time() - stime
    return {
        "status": "completed", "x_opt": x_opt, "runtime": runtime,
        "num_vars": 0, "num_cols": 0, "mipgap": 0, "nodes": 0,
    }