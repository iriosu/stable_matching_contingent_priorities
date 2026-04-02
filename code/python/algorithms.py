import sys, os
import numpy as np
import copy, random, time, math

import generate_inputs as genin
import solve_opt
import pickle


random.seed(1)
np.random.seed(1)

"""
---------------------------
ALGORITHMS
---------------------------
"""


def DA(students, pref, cap):
    """
    New implementation of DA student optimal with strict preferences
    """
    ########################
    # Initialization
    ########################
    cp = {i: 1 for i in students}  # this stores the current preference of each student
    match, is_matched = {i: None for i in students}, {i: False for i in students}

    # Each student proposes to its current top school
    proposals = {c: [] for c in cap}
    rejected = copy.copy(students)
    it = 0
    while True:
        it += 1
        for (
            m
        ) in (
            rejected
        ):  # in the first iteration we consider all students; later we only consider those previously rejected
            if is_matched[m]:
                continue
            try:
                proposals[pref[m][cp[m]]].append(m)
            except:
                print(m, cp[m])
                sys.exit(1)

        rejected = []
        # Each school rejects those student who are not in the top
        for c in cap:
            # if the number of proposals received is less than the vacants
            if len(proposals[c]) <= cap[c]:
                continue
            # otherwise, we order the proposals according to school preferences and reject those students who are not in the top
            top = [pref[c][k] for k in sorted(pref[c]) if pref[c][k] in proposals[c]]
            for i in range(cap[c], len(proposals[c])):
                proposals[c].remove(top[i])
                rejected.append(top[i])

        # The rejected students proposed to their next top choice
        for m in rejected:
            cp[m] += 1
            if (
                cp[m] not in pref[m]
            ):  # i.e. the current preference is not in the preference list, and therefore m has no more preferences
                is_matched[m] = True  # we assume he is matched to himself

        stop = all(
            is_matched[i] for i in students
        )  # this boolean tell us if all agents are matched; if so we stop
        if stop or len(rejected) == 0:
            break

    for m in students:
        if is_matched[m]:
            continue
        match[m], is_matched[m] = pref[m][cp[m]], True

    return match


"""
---------------------------
HEURISTICS
---------------------------
"""


def Sequential(inputs, levels_to_process=None):
    def UpdatePriorities(in_match, colleges, pref, siblings, siblings_priority):
        """
        1. For each program in match, find assigned students
        2. Check if the assigned students have siblings applying to that same RBD.
        3. If there is such a siblings, put them in the top of the list.

        Note: in its current implementation, this uses the lottery of the student to break ties among students with siblings priority (i.e., independent rule)
        """
        for id_s in in_match:
            # print(id_s, in_match[id_s], siblings[id_s])
            if len(siblings[id_s]) == 0 or in_match[id_s] is None:
                continue
            rbd = in_match[id_s].split("_")[0]
            for sib in siblings[id_s]:
                for p in pref[sib]:
                    if pref[sib][p].split("_")[0] == rbd:  # student gets siblings priority
                        siblings_priority[sib][pref[sib][p]] = 1
                    # print(sib, p, pref[sib][p], pref[sib][p].split('_')[0], rbd, siblings_priority[sib][pref[sib][p]])

        out_pref = copy.copy(pref)
        for c in colleges:
            if c not in out_pref:
                continue
            sorted_values = sorted(
                out_pref[c].items(), key=lambda item: (-siblings_priority[item[1]][c], item[0])
            )
            sorted_values_only = [value for key, value in sorted_values]
            out_pref[c] = {p + 1: sorted_values_only[p] for p in range(len(sorted_values_only))}

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
        students_and_schools_in_level = list(set(schools_in_level).union(set(students_in_level)))
        # subset instance and solve match
        cap_in_level = {idx: cap[idx] for idx in schools_in_level if idx in cap}
        pref_in_level = {
            idx: pref_updated[idx] for idx in students_and_schools_in_level if idx in pref_updated
        }
        match[idx] = DA(students_in_level, pref_in_level, cap_in_level)

        # update priorities of all siblings in coming levels
        pref_updated, siblings_priority = UpdatePriorities(
            match[idx], colleges, pref, siblings, siblings_priority
        )

    x_opt = {
        id_s: {match[idx][id_s]: 1}
        for idx in match
        for id_s in match[idx]
        if match[idx][id_s] is not None
    }
    runtime = time.time() - stime
    outputs = {
        "status": "completed",
        "x_opt": x_opt,
        "runtime": runtime,
        "num_vars": 0,
        "num_cols": 0,
        "mipgap": 0,
        "nodes": 0,
    }
    return outputs


def SequentialBlock(inputs, levels_to_process=None):
    def UpdatePriorities(in_match, students, colleges, pref, siblings, siblings_priority):
        """
        1. For each program in match, find assigned students
        2. Check if the assigned students have siblings applying to that same RBD.
        3. If there is such a siblings, put them in the top of the list.

        Note: in its current implementation, this uses the lottery of the student to break ties among students with siblings priority (i.e., independent rule)
        """
        for id_s in in_match:
            # print(id_s, in_match[id_s], siblings[id_s])
            if len(siblings[id_s]) == 0 or in_match[id_s] is None:
                continue
            rbd = in_match[id_s].split("_")[0]
            for sib in siblings[id_s]:
                for p in pref[sib]:
                    if pref[sib][p].split("_")[0] == rbd:  # student gets siblings priority
                        siblings_priority[sib][pref[sib][p]] = 1
                    # print(sib, p, pref[sib][p], pref[sib][p].split('_')[0], rbd, siblings_priority[sib][pref[sib][p]])

        out_pref = copy.copy(pref)
        for c in colleges:
            if c not in out_pref:
                continue
            sorted_values = sorted(
                out_pref[c].items(), key=lambda item: (-siblings_priority[item[1]][c], item[0])
            )
            sorted_values_only = [value for key, value in sorted_values]
            out_pref[c] = {p + 1: sorted_values_only[p] for p in range(len(sorted_values_only))}

        for s in students:
            if s not in out_pref:
                continue
            sorted_values = sorted(
                out_pref[s].items(), key=lambda item: (-siblings_priority[s][item[1]], item[0])
            )
            sorted_values_only = [value for key, value in sorted_values]
            out_pref[s] = {p + 1: sorted_values_only[p] for p in range(len(sorted_values_only))}

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
        students_and_schools_in_level = list(set(schools_in_level).union(set(students_in_level)))
        # subset instance and solve match
        cap_in_level = {idx: cap[idx] for idx in schools_in_level if idx in cap}
        pref_in_level = {
            idx: pref_updated[idx] for idx in students_and_schools_in_level if idx in pref_updated
        }
        match[idx] = DA(students_in_level, pref_in_level, cap_in_level)

        # update priorities of all siblings in coming levels
        pref_updated, siblings_priority = UpdatePriorities(
            match[idx], students, colleges, pref, siblings, siblings_priority
        )

    x_opt = {
        id_s: {match[idx][id_s]: 1}
        for idx in match
        for id_s in match[idx]
        if match[idx][id_s] is not None
    }
    runtime = time.time() - stime
    outputs = {
        "status": "completed",
        "x_opt": x_opt,
        "runtime": runtime,
        "num_vars": 0,
        "num_cols": 0,
        "mipgap": 0,
        "nodes": 0,
    }
    return outputs


def Simultaneous(inputs):
    def UpdatePriorities(in_match, colleges, pref, siblings, siblings_priority):
        """
        1. For each program in match, find assigned students
        2. Check if the assigned students have siblings applying to that same RBD.
        3. If there is such a siblings, put them in the top of the list.

        Note: in its current implementation, this uses the lottery of the student to break ties among students with siblings priority (i.e., independent rule)
        """
        for id_s in in_match:
            # print(id_s, in_match[id_s], siblings[id_s])
            if len(siblings[id_s]) == 0 or in_match[id_s] is None:
                continue
            rbd = in_match[id_s].split("_")[0]
            for sib in siblings[id_s]:
                for p in pref[sib]:
                    # TODO: update priorities only if the school is more preferred than the current match of the sibling
                    if pref[sib][p].split("_")[0] == rbd:  # student gets siblings priority
                        siblings_priority[sib][pref[sib][p]] = 1

        out_pref = copy.copy(pref)
        for c in colleges:
            if c not in out_pref:
                continue
            sorted_values = sorted(
                out_pref[c].items(), key=lambda item: (-siblings_priority[item[1]][c], item[0])
            )
            sorted_values_only = [value for key, value in sorted_values]
            out_pref[c] = {p + 1: sorted_values_only[p] for p in range(len(sorted_values_only))}

        return out_pref, siblings_priority

    students, colleges, pref, cap, siblings = inputs

    stime = time.time()
    siblings_priority = {s: {pref[s][p]: 0 for p in pref[s]} for s in students}
    pref_updated = copy.copy(pref)
    match = {}
    idx = 0
    while True:
        print("Iteration", idx)
        # we process all levels at the same time, and update priorities after processing all levels
        match[idx] = DA(students, pref_updated, cap)

        # if there is no change in the match, we stop
        if idx > 0 and all(match[idx][s] == match[idx - 1][s] for s in match[idx]):
            break
        # update priorities of all siblings
        pref_updated, siblings_priority = UpdatePriorities(
            match[idx], colleges, pref, siblings, siblings_priority
        )
        idx += 1

    x_opt = {
        id_s: {match[idx][id_s]: 1}
        for idx in match
        for id_s in match[idx]
        if match[idx][id_s] is not None
    }
    runtime = time.time() - stime
    outputs = {
        "status": "completed",
        "x_opt": x_opt,
        "runtime": runtime,
        "num_vars": 0,
        "num_cols": 0,
        "mipgap": 0,
        "nodes": 0,
    }
    return outputs


"""
---------------------------
PREPROCESSING
---------------------------
"""


def RemoveDomination(pref, cap, B, verbose=False):
    print("")
    """
    A node (s,c) is student dominated if there are q_c + B students above s that prefer c in top pref => (s,c) cannot be part of any stable assignment, for any t_c in [0,B]
    A node (s,c) is university dominated if there is a school c' >_s c in which s is within the q_c most preferred students => (s,c) cannot be part of any stable assignment, because s is assigned to something preferred for sure.
    """

    def UpdatePreferences():
        for s in students:
            ks = sorted(pref[s].keys())
            vs = [pref[s][c] for c in ks]
            pref[s] = {it + 1: vs[it] for it in range(len(ks))}
        for c in colleges:
            if c not in pref:
                continue
            ks = sorted(pref[c].keys())
            vs = [pref[c][s] for s in ks]
            pref[c] = {it + 1: vs[it] for it in range(len(ks))}

        pref_map = {id: {pref[id][p]: p for p in pref[id]} for id in pref}

    def RemoveSchoolDominated():
        erased = 0
        for s in students:
            boo = False
            lpref = len(pref[s])
            for p in list(sorted(pref[s].keys())):
                c = pref[s][p]
                if pref_map[c][s] <= cap[c]:
                    # student s is guaranteed to be admitted to school c or better => erase all lower ranked schools
                    for pp in range(p + 1, lpref + 1):
                        # remove preference pp from s list
                        del pref[pref[s][pp]][pref_map[pref[s][pp]][s]]
                        del pref[s][pp]
                        erased += 1
                        boo = True
                if boo:
                    break
        UpdatePreferences()
        return erased

    def RemoveStudentDominated():
        erased = 0
        for c in colleges:
            # if q_c + B students prefer c in top pref, then remove everyone else
            if c not in pref:
                continue
            lpref = len(pref[c])
            boo = False
            in_top = 0
            for p in list(sorted(pref[c].keys())):
                in_top += pref_map[pref[c][p]][c] == 1
                if in_top >= cap[c] + B:
                    # delete everyone in the row
                    for pp in range(p + 1, lpref + 1):
                        del pref[pref[c][pp]][pref_map[pref[c][pp]][c]]
                        del pref[c][pp]
                        erased += 1
                        boo = True
                if boo:
                    break
        UpdatePreferences()
        return erased

    colleges = list(set(cap.keys()).intersection(set(pref.keys())))
    students = list(set(pref.keys()) - set(colleges))

    total_erased = 0
    pref_map = {id: {pref[id][p]: p for p in pref[id]} for id in pref}
    while True:
        erased = 0
        erased_c = RemoveSchoolDominated()
        erased_s = RemoveStudentDominated()
        erased = erased_c + erased_s
        total_erased += erased
        if verbose:
            print("Nodes erased by student:", erased_s)
            print("Nodes erased by school:", erased_c)
        if erased == 0:
            break
    # _,_,S,T = genin.create_additional_inputs_from_instance(pref, cap)
    # return (pref, S, T)

    for c in colleges:
        if len(pref[c]) == 0:
            del pref[c]
            del cap[c]

    for s in students:
        if len(pref[s]) == 0:
            del pref[s]

    colleges = list(set(cap.keys()).intersection(set(pref.keys())))
    students = list(set(pref.keys()) - set(colleges))

    return students, colleges, pref, cap


def EvaluateObjective(x, opref, penalty="last_ref"):
    out = sum([x[s][c] * opref[s][c] for s in x for c in x[s]])
    if isinstance(penalty, (int, float, complex)):
        out += penalty * sum([1 - sum([x[s][c] for c in x[s]]) for s in x])
    elif penalty == "last_pref":
        out += sum(
            [(max(opref[s].values()) + 1) * (1 - sum([x[s][c] for c in x[s]])) for s in x]
        ) + sum([(max(opref[s].values()) + 1) for s in opref if s not in x])
    else:
        print("***ERROR: Unkonwn penalty")
        sys.exit(1)
    return out


if __name__ == "__main__":

    home_dir = os.path.expanduser("~")
    if "riosigna" in home_dir:
        dropbox_dir = home_dir + os.sep + "Code"
        outdir = dropbox_dir + os.sep + "outputs"
    else:
        dropbox_dir = home_dir + os.sep + "Dropbox/Dynamic priorities in stable matching"
        outdir = dropbox_dir + os.sep + "outputs"

    indir = dropbox_dir + os.sep + "Data" + os.sep + "Magallanes" + os.sep + "2023"
    plotdir = dropbox_dir + os.sep + "plots"
    tabdir = dropbox_dir + os.sep + "tables"

    tie_breaker = "mtbf"

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(plotdir):
        os.makedirs(plotdir, exist_ok=True)
    if not os.path.exists(tabdir):
        os.makedirs(tabdir, exist_ok=True)

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn = (
        genin.read_instance(indir + os.sep + "instance_" + tie_breaker + ".txt")
    )

    out_seq = Sequential((students, colleges, pref, cap, siblings, levels, students_per_level))
    out_sim = Simultaneous((students, colleges, pref, cap, siblings))
    # compare differences in the match between the two algorithms
    match_seq = {
        s: list(out_seq["x_opt"][s].keys())[0] if s in out_seq["x_opt"] else None for s in students
    }
    match_sim = {
        s: list(out_sim["x_opt"][s].keys())[0] if s in out_sim["x_opt"] else None for s in students
    }
    differences = sum([match_seq[s] != match_sim[s] for s in students])
    print("Number of differences in the match between sequential and simultaneous:", differences)

    for s in students:
        if match_seq[s] != match_sim[s]:
            print(
                "Student",
                s,
                "is matched to",
                match_seq[s],
                "in sequential and to",
                match_sim[s],
                "in simultaneous.",
            )
