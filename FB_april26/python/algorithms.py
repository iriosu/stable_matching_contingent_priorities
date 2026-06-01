import sys, os
import numpy as np
import copy, random, time, math

import generate_inputs as genin
import solve_opt
import pickle


random.seed(1)
np.random.seed(1)

'''
---------------------------
ALGORITHMS
---------------------------
'''
def DA(students,pref,cap):
    '''
    New implementation of DA student optimal with strict preferences
    '''
    ########################
    # Initialization
    ########################
    cp = {i:1 for i in students} # this stores the current preference of each student
    match, is_matched = {i:None for i in students}, {i:False for i in students}

    # Each student proposes to its current top school
    proposals = {c:[] for c in cap}
    rejected = copy.copy(students)
    it = 0
    while True:
        it+=1
        for m in rejected: # in the first iteration we consider all students; later we only consider those previously rejected
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
            cp[m]+=1
            if cp[m] not in pref[m]: # i.e. the current preference is not in the preference list, and therefore m has no more preferences
                is_matched[m] = True # we assume he is matched to himself

        stop = all(is_matched[i] for i in students) # this boolean tell us if all agents are matched; if so we stop
        if stop or len(rejected) == 0:
            break

    for m in students:
        if is_matched[m]:
            continue
        match[m], is_matched[m] = pref[m][cp[m]], True

    return match


'''
---------------------------
HEURISTICS
---------------------------
'''
def _update_school_priorities(in_match, colleges, pref, siblings, siblings_priority):
    """After a level is processed, boost sibling-priority for schools where
    a sibling is now assigned. Updates ONLY school preference orders ≻_c.
    Student preferences ≻_s are left untouched (per the paper)."""
    for id_s, m_s in in_match.items():
        if m_s is None or len(siblings[id_s]) == 0:
            continue
        rbd = m_s.split('_')[0]
        for sib in siblings[id_s]:
            for p in pref[sib]:
                if pref[sib][p].split('_')[0] == rbd:
                    siblings_priority[sib][pref[sib][p]] = 1

    out_pref = copy.copy(pref)
    for c in colleges:
        if c not in out_pref:
            continue
        ranked = sorted(
            out_pref[c].items(),
            key=lambda item: (-siblings_priority[item[1]][c], item[0])  # MTB-F: original order breaks ties
        )
        out_pref[c] = {p + 1: v for p, (_, v) in enumerate(ranked)}
    return out_pref, siblings_priority


def _sequential_by_level(inputs, levels_to_process):
    students, colleges, pref, cap, siblings, levels, students_per_level = inputs

    # handle the PreK / K aliasing used in some instance files
    if '0' not in students_per_level or '-1' not in students_per_level:
        levels_to_process = ['PreK' if l == '-1' else 'K' if l == '0' else l for l in levels_to_process]

    stime = time.time()
    siblings_priority = {s: {pref[s][p]: 0 for p in pref[s]} for s in students}
    pref_updated = copy.copy(pref)
    match = {}

    for idx in levels_to_process:
        schools_in_level = levels[idx]
        students_in_level = students_per_level[idx]
        ids_in_level = list(set(schools_in_level).union(set(students_in_level)))

        cap_in_level = {i: cap[i] for i in schools_in_level if i in cap}
        pref_in_level = {i: pref_updated[i] for i in ids_in_level if i in pref_updated}

        match[idx] = DA(students_in_level, pref_in_level, cap_in_level)

        pref_updated, siblings_priority = _update_school_priorities(
            match[idx], colleges, pref, siblings, siblings_priority
        )

    x_opt = {s: {match[idx][s]: 1}
             for idx in match for s in match[idx] if match[idx][s] is not None}
    return {'status': 'completed', 'x_opt': x_opt,
            'runtime': time.time() - stime,
            'num_vars': 0, 'num_cols': 0, 'mipgap': 0, 'nodes': 0}


def Descending(inputs):
    """Current Chilean practice: process levels from 12th grade down to Pre-K
    (Section 5.2.2 of the paper). Only older siblings can grant priority."""
    order = [str(i) for i in sorted(range(-1, 13), reverse=True)]
    return _sequential_by_level(inputs, order)


def Ascending(inputs):
    """Process levels from Pre-K up to 12th grade (Appendix D.3, Table 7).
    Empirically produces more siblings together than Descending under MTB-F."""
    order = [str(i) for i in sorted(range(-1, 13))]
    return _sequential_by_level(inputs, order)





'''
---------------------------
PREPROCESSING
---------------------------
'''
def RemoveDomination(pref, cap, B, verbose=False):
    print("")
    '''
    A node (s,c) is student dominated if there are q_c + B students above s that prefer c in top pref => (s,c) cannot be part of any stable assignment, for any t_c in [0,B]
    A node (s,c) is university dominated if there is a school c' >_s c in which s is within the q_c most preferred students => (s,c) cannot be part of any stable assignment, because s is assigned to something preferred for sure.
    '''
    def UpdatePreferences():
        for s in students:
            ks = sorted(pref[s].keys())
            vs = [pref[s][c] for c in ks]
            pref[s] = {it+1:vs[it] for it in range(len(ks))}
        for c in colleges:
            if c not in pref:
                continue
            ks = sorted(pref[c].keys())
            vs = [pref[c][s] for s in ks]
            pref[c] = {it+1:vs[it] for it in range(len(ks))}

        pref_map = {id:{pref[id][p]:p for p in pref[id]} for id in pref}

    def RemoveSchoolDominated():
        erased = 0
        for s in students:
            boo = False
            lpref = len(pref[s])
            for p in list(sorted(pref[s].keys())):
                c = pref[s][p]
                if pref_map[c][s] <= cap[c]:
                    # student s is guaranteed to be admitted to school c or better => erase all lower ranked schools
                    for pp in range(p+1, lpref+1):
                        # remove preference pp from s list
                        del pref[pref[s][pp]][pref_map[pref[s][pp]][s]]
                        del pref[s][pp]
                        erased+=1
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
                in_top += (pref_map[pref[c][p]][c] == 1)
                if in_top >= cap[c] + B:
                    # delete everyone in the row
                    for pp in range(p+1, lpref+1):
                        del pref[pref[c][pp]][pref_map[pref[c][pp]][c]]
                        del pref[c][pp]
                        erased+=1
                        boo = True
                if boo:
                    break
        UpdatePreferences()
        return erased

    colleges = list(set(cap.keys()).intersection(set(pref.keys())))
    students = list(set(pref.keys()) - set(colleges))

    total_erased = 0
    pref_map = {id:{pref[id][p]:p for p in pref[id]} for id in pref}
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

def EvaluateObjective(x, opref, penalty='last_ref'):
    out = sum([x[s][c]*opref[s][c] for s in x for c in x[s] ])
    if isinstance(penalty, (int, float, complex)):
        out += penalty * sum([1-sum([x[s][c] for c in x[s]]) for s in x])
    elif penalty == "last_pref":
        out += sum([ (max(opref[s].values())+1) * (1-sum([x[s][c] for c in x[s]])) for s in x]) + sum([ (max(opref[s].values())+1) for s in opref if s not in x])
    else:
        print("***ERROR: Unkonwn penalty")
        sys.exit(1)
    return out

if __name__ == '__main__':

    region = "OHiggins"
    year = 2023
    tie_breaker = "mtbf"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    indir = os.path.join(base_dir, "..", "R", "intermediate_data", region, str(year))

    instance_file = os.path.join(indir, f"instance_{tie_breaker}.txt")

    if not os.path.exists(instance_file):
        raise FileNotFoundError(f"Could not find: {instance_file}")

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn = \
        genin.read_instance(instance_file)

    outputs = 'test'

    print(outputs["status"])
    print(f"Runtime: {outputs['runtime']:.4f} seconds")
    print(f"Assigned students: {len(outputs['x_opt'])}")