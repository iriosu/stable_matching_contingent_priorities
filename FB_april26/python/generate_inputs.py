import os, sys
import numpy as np
import pickle
import random
import itertools

np.random.seed(1)
random.seed(1)


def create_additional_inputs_from_instance(pref, cap):
    """
    Generates random instance based the number of students (n1) and schools (n2)
    Input: n1, n2 and total capacity
    Output: preferences, and matrices T and S
    """
    colleges = list(set(list(cap.keys())).intersection(set(list(pref.keys()))))
    students = [k for k in pref if k not in colleges]
    map = {c: {pref[c][p]: p for p in pref[c]} for c in colleges}

    Tp, Tn, Sp, Sn = (
        {s: {} for s in students},
        {s: {} for s in students},
        {s: {} for s in students},
        {s: {} for s in students},
    )
    for s in students:
        Tp[s] = {pref[s][p]: [pref[s][k] for k in pref[s] if k <= p] for p in pref[s]}
        Tn[s] = {pref[s][p]: [pref[s][k] for k in pref[s] if k >= p] for p in pref[s]}
        Sp[s] = {
            pref[s][p]: [pref[pref[s][p]][k] for k in pref[pref[s][p]] if k < map[pref[s][p]][s]]
            for p in pref[s]
        }
        Sn[s] = {
            pref[s][p]: [pref[pref[s][p]][k] for k in pref[pref[s][p]] if k > map[pref[s][p]][s]]
            for p in pref[s]
        }
    return students, colleges, Tp, Tn, Sp, Sn


def read_instance(filename):
    f = open(filename, "r")
    lines = f.readlines()
    cap, pref, siblings, levels, students_per_level = {}, {}, {}, {}, {}
    for i in range(len(lines)):
        line = lines[i].strip().rstrip("\n")
        if "# Capacities:" in line:
            ct = 1
            while True:
                line = lines[i + ct].strip().rstrip("\n")
                if "#" in line:
                    break
                pieces = line.split(" ")
                school = pieces[0]
                seats = int(pieces[1])
                cap[school] = seats
                ct += 1
        elif "# Student preferences:" in line:
            ct = 1
            while True:
                line = lines[i + ct].strip().rstrip("\n")
                if "#" in line:
                    break

                pieces = line.split(" ")
                student = pieces[0]
                if student not in pref:
                    pref[student] = {}
                for j in range(1, len(pieces)):
                    aux = pieces[j][1:-1].split(",")
                    k = int(aux[0])
                    pref[student][k] = aux[1]
                ct += 1
        elif "# College priorities:" in line:
            ct = 1
            while True:
                line = lines[i + ct].strip().rstrip("\n")
                if "#" in line:
                    break

                pieces = line.split(" ")
                school = pieces[0]
                if school not in pref:
                    pref[school] = {}
                for j in range(1, len(pieces)):
                    aux = pieces[j][1:-1].split(",")
                    k = int(aux[0])
                    pref[school][k] = aux[1]
                ct += 1
        elif "# Siblings:" in line:
            ct = 1
            while True:
                line = lines[i + ct].strip().rstrip("\n")
                if "#" in line:
                    break

                pieces = line.split(" ")
                student = pieces[0]
                if len(pieces) == 1:
                    siblings[student] = []
                else:
                    siblings[student] = [pieces[j] for j in range(1, len(pieces))]
                ct += 1
        elif "# Levels:" in line:
            ct = 1
            while True:
                line = lines[i + ct].strip().rstrip("\n")
                if "#" in line:
                    break
                pieces = line.split(" ")
                idx = pieces[0]
                levels[idx] = [pieces[j] for j in range(1, len(pieces))]
                ct += 1
        elif "# Students per Level:" in line:
            ct = 1
            while i + ct < len(lines):
                line = lines[i + ct].strip().rstrip("\n")
                if "#" in line:
                    break
                pieces = line.split(" ")
                idx = pieces[0]
                students_per_level[idx] = [pieces[j] for j in range(1, len(pieces))]
                ct += 1
        else:
            pass

    students, colleges, Tp, Tn, Sp, Sn = create_additional_inputs_from_instance(pref, cap)
    return students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn


def subset_instance(students, pref, cap, siblings=None):
    # Remove students from pref
    if siblings is not None:
        while True:
            new_students = set(students).union(set([sib for s in students for sib in siblings[s]]))
            if len(set(new_students).difference(set(students))) == 0:
                break
            else:
                students = new_students
        students = new_students
        siblings = {s: siblings[s] for s in students}

    colleges = list(cap.keys())
    for s in list(pref.keys()):
        if s in colleges or s in students:
            continue
        del pref[s]

    # Remove students from schools lists.
    for c in colleges:
        for p in list(pref[c].keys()):
            if pref[c][p] not in students:
                del pref[c][p]
        new_pref, ct = {}, 1
        for p in sorted(pref[c]):
            new_pref[ct] = pref[c][p]
            ct += 1
        pref[c] = new_pref

    students, colleges, Tp, Tn, Sp, Sn = create_additional_inputs_from_instance(pref, cap)
    if siblings is None:
        return students, colleges, pref, Tp, Tn, Sp, Sn
    else:
        return students, colleges, pref, siblings, Tp, Tn, Sp, Sn


def read_students(infile):
    f = open(infile, "r")
    students = []
    for line in f.readlines():
        line = line.strip().rstrip("\n")
        students.append(line)
    f.close()
    return students


def read_siblings(infile, students):
    f = open(infile, "r")
    siblings = {s: [] for s in students}
    for line in f.readlines():
        line = line.strip().rstrip("\n")
        line = line.replace('"', "")
        if "id_post" in line:
            continue
        pieces = line.split(",")
        if pieces[0] in siblings and pieces[1] in siblings:
            siblings[pieces[0]].append(pieces[1])
            siblings[pieces[1]].append(pieces[0])
    f.close()
    return siblings


def write_instance(students, colleges, pref, cap, siblings, levels, students_per_level, infile):
    f = open(infile, "w")
    f.write("# Num. students:" + str(len(students)) + "\n")
    f.write("# Num. colleges:" + str(len(colleges)) + "\n")
    f.write("# Students:" + ",".join([s for s in students]) + "\n")
    f.write("# Colleges:" + ",".join([c for c in colleges]) + "\n")
    f.write("# Capacities:\n")
    for c in cap:
        f.write(c + " " + str(cap[c]) + "\n")
    f.write("# Student preferences:\n")
    for s in students:
        f.write(
            s
            + " "
            + " ".join(["(" + str(p) + "," + str(pref[s][p]) + ")" for p in sorted(pref[s])])
            + "\n"
        )
    f.write("# College priorities:\n")
    for c in colleges:
        f.write(
            c
            + " "
            + " ".join(["(" + str(p) + "," + str(pref[c][p]) + ")" for p in sorted(pref[c])])
            + "\n"
        )
    f.write("# Siblings:\n")
    for s in students:
        if len(siblings[s]) == 0:
            f.write(s + "\n")
        else:
            f.write(s + " " + " ".join([str(sib) for sib in siblings[s]]) + "\n")
    f.write("# Levels:\n")
    for idx in levels:
        f.write(idx + " " + " ".join([str(cc) for cc in levels[idx]]) + "\n")
    f.write("# Students per Level:\n")
    for idx in students_per_level:
        f.write(idx + " " + " ".join([str(id_s) for id_s in students_per_level[idx]]) + "\n")
    f.close()


def create_levels(colleges):
    aux = ["PreK", "K"]
    aux.extend([str(k) for k in range(1, 13)])
    niveles = {k: [] for k in aux}
    for c in colleges:
        pieces = c.split("_")
        rbd, curso = pieces[0], pieces[1]
        if curso[1:3] == "11":
            niveles[curso[:1]].append(c)
        elif curso[:1] in ["4", "5"] and curso[1:3] == "01":
            if curso[:1] == "4":
                niveles["PreK"].append(c)
            elif curso[:1] == "5":
                niveles["K"].append(c)
            else:
                pass
        else:
            idx = str(int(curso[:1]) + 8)
            niveles[idx].append(c)
    return niveles


def create_levels_from_courses(colleges):
    out = {}
    for c in colleges:
        pieces = c.split("_")
        rbd, curso = pieces[0], pieces[1]
        if curso[1:3] == "11":
            out[c] = curso[:1]
        elif curso[:1] in ["4", "5"] and curso[1:3] == "01":
            if curso[:1] == "4":
                out[c] = "PreK"
            elif curso[:1] == "5":
                out[c] = "K"
            else:
                pass
        else:
            idx = str(int(curso[:1]) + 8)
            out[c] = idx

    out2 = {}
    for c in colleges:
        rbd = c.split("_")[0]
        level = out[c]
        if rbd not in out2:
            out2[rbd] = {}
        if level not in out2[rbd]:
            out2[rbd][level] = []
        out2[rbd][level].append(c)

    return out, out2


def create_students_per_level(students, pref):
    aux = ["PreK", "K"]
    aux.extend([str(k) for k in range(1, 13)])
    niveles = {k: [] for k in aux}
    for s in students:
        pieces = pref[s][1].split("_")
        rbd, curso = pieces[0], pieces[1]
        if curso[1:3] == "11":
            niveles[curso[:1]].append(s)
        elif curso[:1] in ["4", "5"] and curso[1:3] == "01":
            if curso[:1] == "4":
                niveles["PreK"].append(s)
            elif curso[:1] == "5":
                niveles["K"].append(s)
            else:
                pass
        else:
            idx = str(int(curso[:1]) + 8)
            niveles[idx].append(s)
    return niveles


def modify_school_loterries(pref, students, colleges, siblings, tie_breaker="mtb"):
    def dfs(dictionary, start_key, visited=None):
        if visited is None:
            visited = set()

        if start_key not in visited:
            visited.add(start_key)
            for neighbor_key in dictionary.get(start_key, []):
                dfs(dictionary, neighbor_key, visited)

        return list(visited)

    """
    Inputs: preferences and type of tie_breaker: stb, mtb, stb-f, mtb-f
    """
    out_tb = {}
    if tie_breaker == "mtb":
        out_tb = {s: {pref[s][p].split("_")[0]: random.random() for p in pref[s]} for s in students}
        for c in colleges:
            rbd = c.split("_")[0]
            applicants = list(pref[c].values())
            applicants = sorted(applicants, key=lambda id: -out_tb[id][rbd])
            pref[c] = {p + 1: applicants[p] for p in range(len(applicants))}
    elif tie_breaker == "mtbf":
        tb = {s: {pref[s][p].split("_")[0]: random.random() for p in pref[s]} for s in students}
        out_tb = {}
        for s in tb:
            if s not in out_tb:
                out_tb[s] = {}
            if len(siblings[s]) == 0:
                out_tb[s] = tb[s]
                continue

            for rbd in tb[s]:
                if rbd in out_tb[s]:
                    continue
                out_tb[s][rbd] = tb[s][rbd]
                sibs = dfs(siblings, s, visited=None)
                for sib in sibs:
                    if sib not in out_tb:
                        out_tb[sib] = {}
                    for rbdp in tb[sib]:
                        if rbdp in out_tb[sib]:
                            continue
                        if rbdp == rbd:
                            out_tb[sib][rbdp] = tb[s][rbd]
        for c in colleges:
            rbd = c.split("_")[0]
            applicants = list(pref[c].values())
            applicants = sorted(applicants, key=lambda id: -out_tb[id][rbd])
            pref[c] = {p + 1: applicants[p] for p in range(len(applicants))}
    elif tie_breaker == "stb":
        tb = {s: random.random() for s in students}
        out_tb = {s: {pref[s][p].split("_")[0]: tb[s] for p in pref[s]} for s in students}
        for c in colleges:
            applicants = list(pref[c].values())
            applicants = sorted(applicants, key=lambda id: -tb[id])
            pref[c] = {p + 1: applicants[p] for p in range(len(applicants))}
    elif tie_breaker == "stbf":
        tb = {s: random.random() for s in students}
        ftb = {}
        for s in tb:
            if s in ftb:
                continue
            if len(siblings[s]) == 0:
                ftb[s] = tb[s]
            else:
                ftb[s] = tb[s]
                # replace lottery for all siblings and siblings of siblings and etc.
                sibs = dfs(siblings, s, visited=None)
                for sib in sibs:
                    ftb[sib] = tb[s]
        out_tb = {s: {pref[s][p].split("_")[0]: ftb[s] for p in pref[s]} for s in students}
        for c in colleges:
            applicants = list(pref[c].values())
            applicants = sorted(applicants, key=lambda id: -ftb[id])
            pref[c] = {p + 1: applicants[p] for p in range(len(applicants))}
    else:
        print("***ERROR: Unknown tie breaking method")
        sys.exit(1)
    return pref, out_tb


def clean_reserves(students, pref, colleges, cap, siblings):
    out_pref = {}
    for c in colleges:
        if c not in pref:
            continue
        if "REG" not in c:
            continue
        out_pref[c.split("_")[0] + "_" + c.split("_")[1]] = pref[c]
    out_cap = {}
    for c in cap:
        course = c.split("_")[0] + "_" + c.split("_")[1]
        if course not in out_cap:
            out_cap[course] = 0
        out_cap[course] += cap[c]

    out_colleges = list(set(list(out_cap.keys())).intersection(set(list(out_pref.keys()))))
    out_cap = {c: out_cap[c] for c in out_colleges}
    out_pref = {c: out_pref[c] for c in out_colleges}

    for s in students:
        if s not in pref:
            continue
        if pref[s] == {}:
            continue
        out_pref[s], ct = {}, 1
        for p in sorted(pref[s]):
            course = pref[s][p].split("_")[0] + "_" + pref[s][p].split("_")[1]
            if course not in out_colleges:
                continue
            if course not in out_pref[s].values():
                out_pref[s][ct] = course
                ct += 1
        if out_pref[s] == {}:
            del out_pref[s]

    out_students = list(set(out_pref.keys()).difference(set(out_colleges)))
    out_siblings = {s: list(set(siblings[s]).intersection(set(out_students))) for s in out_students}

    return out_students, out_pref, out_colleges, out_cap, out_siblings


def clean_reserves_and_merge_schools(students, pref, colleges, cap, siblings):
    course_levels, courses_per_school_and_level = create_levels_from_courses(colleges)

    out_cap = {}
    for rbd in courses_per_school_and_level:
        for level in courses_per_school_and_level[rbd]:
            course = rbd + "_" + level
            if course not in out_cap:
                out_cap[course] = 0
            for c in courses_per_school_and_level[rbd][level]:
                out_cap[course] += cap[c]

    out_pref = {}
    for rbd in courses_per_school_and_level:
        for level in courses_per_school_and_level[rbd]:
            course = rbd + "_" + level
            applicants = []
            for c in courses_per_school_and_level[rbd][level]:
                if c in pref:
                    applicants.extend(list(pref[c].values()))
            applicants = list(set(applicants))
            out_pref[course] = {idx + 1: applicants[idx] for idx in range(len(applicants))}

    out_colleges = list(set(list(out_cap.keys())).intersection(set(list(out_pref.keys()))))
    out_cap = {c: out_cap[c] for c in out_colleges}
    out_pref = {c: out_pref[c] for c in out_colleges}

    for s in students:
        if s not in pref:
            continue
        if pref[s] == {}:
            continue
        out_pref[s], ct = {}, 1
        for p in sorted(pref[s]):
            rbd = pref[s][p].split("_")[0]
            level = course_levels[pref[s][p]]
            course = rbd + "_" + level
            if course not in out_colleges:
                continue
            if course not in out_pref[s].values():
                out_pref[s][ct] = course
                ct += 1
        if out_pref[s] == {}:
            del out_pref[s]

    out_students = list(set(out_pref.keys()).difference(set(out_colleges)))
    out_siblings = {s: list(set(siblings[s]).intersection(set(out_students))) for s in out_students}

    out_levels = {}
    for c in out_colleges:
        level = c.split("_")[1]
        if level not in out_levels:
            out_levels[level] = []
        out_levels[level].append(c)

    out_students_per_level = {}
    for s in out_students:
        if s not in out_pref:
            continue
        if out_pref[s] == {}:
            continue
        level = out_pref[s][1].split("_")[1]
        if level not in out_students_per_level:
            out_students_per_level[level] = []
        out_students_per_level[level].append(s)

    return (
        out_students,
        out_pref,
        out_colleges,
        out_cap,
        out_siblings,
        out_levels,
        out_students_per_level,
    )


def create_families(siblings):
    def dfs(dictionary, start_key, visited=None):
        if visited is None:
            visited = set()

        if start_key not in visited:
            visited.add(start_key)
            for neighbor_key in dictionary.get(start_key, []):
                dfs(dictionary, neighbor_key, visited)

        return list(visited)

    ct, families = 0, {}
    aux = list(siblings.keys())
    while len(aux) > 0:
        id_s = aux.pop()
        fam = dfs(siblings, id_s, visited=None)
        families[ct] = fam
        aux = set(aux).difference(fam)
        ct += 1
    return families


def write_instance_from_csv(indir, outdir, tie_breaker="mtbf", yr=2023):
    def read_capacities(indir):
        f = open(indir + os.sep + "capacities.csv", "r")
        cap = {}
        for line in f.readlines():
            if "vacantes" in line:
                continue
            pieces = line.strip().rstrip("\n").split(",")
            rbd = pieces[0].replace('"', "")
            cap[rbd] = int(pieces[1])
        f.close()
        return cap

    def read_preferences(indir):
        f = open(indir + os.sep + "pref_students.csv", "r")
        pref = {}
        for line in f.readlines():
            if "mrun" in line:
                continue
            pieces = line.strip().rstrip("\n").split(",")
            mrun = pieces[0]
            rbd = pieces[1].replace('"', "")
            ord = int(pieces[2])
            if mrun not in pref:
                pref[mrun] = {}
            pref[mrun][ord] = rbd
        f.close()

        f = open(indir + os.sep + "pref_schools.csv", "r")
        for line in f.readlines():
            if "rbd" in line:
                continue
            pieces = line.strip().rstrip("\n").split(",")
            rbd = pieces[0].replace('"', "")
            mrun = pieces[1]
            ord = int(pieces[2])
            if rbd not in pref:
                pref[rbd] = {}
            pref[rbd][ord] = mrun
        f.close()

        return pref

    def read_siblings(infile, students):
        f = open(infile + os.sep + "siblings.csv", "r")
        siblings = {s: [] for s in students}
        for line in f.readlines():
            line = line.strip().rstrip("\n")
            line = line.replace('"', "")
            if "id_post" in line:
                continue
            pieces = line.split(",")
            if pieces[0] in siblings and pieces[1] in siblings:
                siblings[pieces[0]].append(pieces[1])
                siblings[pieces[1]].append(pieces[0])
        f.close()
        return siblings

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    cap = read_capacities(indir + os.sep + str(yr))
    pref = read_preferences(indir + os.sep + str(yr))

    colleges = list(cap.keys())
    students = list(set(pref.keys()) - set(colleges))
    siblings = read_siblings(indir + os.sep + str(yr), students)

    students, pref, colleges, cap, siblings, levels, students_per_level = (
        clean_reserves_and_merge_schools(students, pref, colleges, cap, siblings)
    )

    write_instance(
        students,
        colleges,
        pref,
        cap,
        siblings,
        levels,
        students_per_level,
        indir + os.sep + str(yr) + os.sep + "instance.txt",
    )

    pref, tb = modify_school_loterries(pref, students, colleges, siblings, tie_breaker)

    with open(indir + os.sep + str(yr) + os.sep + "tb_" + tie_breaker + ".pck", "wb") as f:
        pickle.dump(tb, f)

    write_instance(
        students,
        colleges,
        pref,
        cap,
        siblings,
        levels,
        students_per_level,
        indir + os.sep + str(yr) + os.sep + "instance_" + tie_breaker + ".txt",
    )


if __name__ == "__main__":

    year = 2023
    region = "OHiggins"
    indir = "../R/intermediate_data" + os.sep + region

    for region in ["Magallanes", "Arica", "Tarapaca", "Atacama", "Lagos", "Coquimbo", "Antofagasta", "Rios"]:
        indir = f"../R/intermediate_data/{region}"
        write_instance_from_csv(indir, indir, "mtbf", yr=2023)


