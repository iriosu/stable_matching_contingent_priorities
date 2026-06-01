import sys, os, time, copy
import numpy as np
import pickle, math, random
import matplotlib.pyplot as plt
import seaborn as sns
import generate_inputs as genin

home_dir = os.path.expanduser("~")
if "riosigna" in home_dir:
    dropbox_dir = home_dir + os.sep + "Code"
    outdir = dropbox_dir + os.sep + "outputs"
    indir = dropbox_dir + os.sep + "Data"
else:
    dropbox_dir = home_dir + os.sep + "Dropbox/Dynamic priorities in stable matching"
    outdir = dropbox_dir + os.sep + "outputs"
    indir = dropbox_dir + os.sep + "Data"

plotdir = dropbox_dir + os.sep + "plots"
tabdir = dropbox_dir + os.sep + "tables"

if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)
if not os.path.exists(plotdir):
    os.makedirs(plotdir, exist_ok=True)
if not os.path.exists(tabdir):
    os.makedirs(tabdir, exist_ok=True)


# --------------------------------
# Auxiliary methods for handling simulations
# --------------------------------
def ReadOptimalSolution(infile):
    obj, x_opt, t_opt, runtime, mipgap = 0, {}, {}, 0, 0
    f = open(infile, "r")
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip().rstrip("\n")
        if "# Objective" in line:
            obj = float(line.split(":")[1])
        if "# Run time" in line:
            runtime = float(line.split(":")[1])
        if "# MipGap" in line:
            mipgap = float(line.split(":")[1])
        if "# Optimal x" in line:
            idx = i + 1
            while "#" not in lines[idx]:
                pieces = lines[idx].strip().rstrip("\n").split(" ")
                id_s, id_c, val = pieces[0], pieces[1], float(pieces[2])
                if id_s not in x_opt:
                    x_opt[id_s] = {}
                x_opt[id_s][id_c] = val
                idx += 1
    return {"x_opt": x_opt, "runtime": runtime, "mipgap": mipgap, "obj": obj}


def ReadOptimalSiblingsPriority(infile):
    y_opt = {}
    f = open(infile, "r")
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip().rstrip("\n")
        if "# Optimal y" in line:
            idx = i + 1
            while idx < len(lines):
                pieces = lines[idx].strip().rstrip("\n").split(" ")
                id_s, id_sib, id_c, val = pieces[0], pieces[1], pieces[2], float(pieces[3])
                if id_s not in y_opt:
                    y_opt[id_s] = {}
                if id_sib not in y_opt[id_s]:
                    y_opt[id_s][id_sib] = {}
                y_opt[id_s][id_sib][id_c] = val
                idx += 1
    return y_opt


def ReadSolutionDetails(infile):
    x_opt, t_opt, runtime, mipgap = {}, {}, 0, 0
    enter, leave, better, worst, obj = 0, 0, 0, 0, 0
    distr = {}
    f = open(infile, "r")
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip().rstrip("\n")
        if "# PROBLEM INFEASIBLE" in line:
            return {}
        if "# Objective" in line:
            obj = float(line.split(":")[1])
        if "# Run time" in line:
            runtime = float(line.split(":")[1])
        if "# MipGap" in line:
            mipgap = float(line.split(":")[1])
        if "# Enter" in line:
            enter = float(line.split(":")[1])
        if "# Leave" in line:
            leave = float(line.split(":")[1])
        if "# Better" in line:
            better = float(line.split(":")[1])
        if "# Worst" in line:
            worst = float(line.split(":")[1])
        if "# Num unassigned" in line:
            unassigned = float(line.split(":")[1])
        if "# Num siblings together" in line:
            together = float(line.split(":")[1])
        if "# Num siblings unassigned with more preferred overlap" in line:
            unassigned_separated = float(line.split(":")[1])
        if "# Num siblings separated with one unassigned and more preferred overlap" in line:
            separated_one_unassigned = float(line.split(":")[1])
        if "# Num siblings separated with both assigned and more preferred overlap" in line:
            assigned_separated = float(line.split(":")[1])
        if "# Distribution preference of assignment" in line:
            idx = i + 1
            while idx < len(lines):
                if "#" in lines[idx]:
                    break
                pieces = lines[idx].strip().rstrip("\n").split(" ")
                pref, count = int(pieces[0]), int(pieces[1])
                distr[pref] = count
                idx += 1
        if "# Optimal x" in line:
            idx = i + 1
            while idx < len(lines):
                if "#" in lines[idx]:
                    break
                pieces = lines[idx].strip().rstrip("\n").split(" ")
                id_s, id_c, val = pieces[0], pieces[1], float(pieces[2])
                if id_s not in x_opt:
                    x_opt[id_s] = {}
                x_opt[id_s][id_c] = val
                idx += 1
    return {
        "obj": obj,
        "x_opt": x_opt,
        "runtime": runtime,
        "mipgap": mipgap,
        "better": better,
        "worst": worst,
        "enter": enter,
        "leave": leave,
        "unassigned": unassigned,
        "together": together,
        "unassigned_separated": unassigned_separated,
        "separated_one_unassigned": separated_one_unassigned,
        "assigned_separated": assigned_separated,
        "distribution": distr,
    }


def ReadSolutions(indir):
    outs = {"comparison": {}, "sensitivity": {}}
    for name in os.listdir(indir):
        if ".DS_Store" in name:
            continue
        if "_k=" in name:
            sim = name.split("_")[1].split("=")[1]
            k = name.split("_")[2].split("=")[1].split(".")[0]
            filename = os.path.join(indir, name)
            if ".pck" in filename:
                continue
            if k not in outs["sensitivity"]:
                outs["sensitivity"][k] = {}
            outs["sensitivity"][k][sim] = ReadSolutionDetails(filename)
        else:
            sim = name.split("=")[1].split(".")[0]
            filename = os.path.join(indir, name)
            if ".pck" in filename:
                continue
            outs["comparison"][sim] = ReadSolutionDetails(filename)
    return outs


def ComputeAssignmentDistribution(x_opt, students, pref, siblings):
    def FindPreference(id_s, id_c):
        opref, px = {}, 1
        for p in sorted(pref[id_s]):
            if pref[id_s][p] not in opref:
                opref[pref[id_s][p]] = px
                px += 1
        return opref[id_c]

    opref = {}
    for s in pref:
        if s not in students:
            continue
        opref[s], idx, aux = {}, 0, []
        for p in sorted(pref[s]):
            if "_" in pref[s][p]:
                c = pref[s][p][:-4]
            else:
                c = pref[s][p]
            if c not in aux:
                idx += 1
                aux += [c]
            opref[s][p] = idx

    pref_assignment = {"sib": {}, "nosib": {}}

    assigned_students = [s for s in x_opt if max(x_opt[s].values(), default=0) > 1 - 1e-3]
    unassigned_students_sib = len(
        [s for s in students if s not in assigned_students and len(siblings[s]) > 0]
    )
    unassigned_students_nosib = len(
        [s for s in students if s not in assigned_students and len(siblings[s]) == 0]
    )

    for id_s in x_opt:
        gr = "nosib"
        if len(siblings[id_s]) > 0:
            gr = "sib"
        for id_c in x_opt[id_s]:
            if x_opt[id_s][id_c] > 1 - 1e-3:
                pr = FindPreference(id_s, id_c)
                if pr not in pref_assignment[gr]:
                    pref_assignment[gr][pr] = 0
                pref_assignment[gr][pr] += 1

    pref_assignment["sib"]["unassigned"] = unassigned_students_sib
    pref_assignment["nosib"]["unassigned"] = unassigned_students_nosib

    return pref_assignment


# --------------------------------
# Methods for descriptives
# --------------------------------
def DescriptivesSimulations(datdir, tabdir, methods=[], tie_breakers=[]):
    datdir = outdir + os.sep + "Comparison"
    outputs = {}
    for name in os.listdir(datdir):
        if "." in name:
            continue
        filename = os.path.join(datdir, name)
        if name in ["par_indep_mtb", "par_indep_stb"]:
            continue
        print(name)
        outputs[name] = ReadSolutions(filename)

    # Compute statistics about each method
    stats = {key: {} for key in outputs}
    outcomes = [
        "together",
        "unassigned_separated",
        "separated_one_unassigned",
        "assigned_separated",
    ]
    for key in outputs:
        for outcome in outcomes:
            stats[key][outcome] = {
                "mean": np.mean([outputs[key][sim][outcome] for sim in outputs[key]]),
                "se": np.std([outputs[key][sim][outcome] for sim in outputs[key]])
                / math.sqrt(len(outputs[key])),
            }

    methods = [
        "abs_indep",
        "par_indep",
        "nosib",
        "max_sib",
        "asc",
        "asc_block",
        "desc",
        "desc_block",
    ]
    tie_breakers = ["stbf", "mtbf", "stb", "mtb"]

    # Create plots or tables summarizing these summary stats
    f = open(tabdir + os.sep + "comparison.tex", "w")
    f.write(r"\begin{table}" + "\n")
    f.write(r"    \begin{tabular}{lcccccc}" + "\n")
    f.write(r"        \toprule" + "\n")
    f.write(r"        & & & \multicolumn{3}{c}{Separated} \\" + "\n")
    f.write(r"        \cmidrule(lr){4-6}" + "\n")
    f.write(r"        & & Together & None & One & Both \\" + "\n")
    f.write(r"        \midrule" + "\n")
    for tie_breaker in tie_breakers:
        tb_label = "STB"
        if tie_breaker == "stbf":
            tb_label = "STB-F"
        elif tie_breaker == "mtbf":
            tb_label = "MTB-F"
        elif tie_breaker == "mtb":
            tb_label = "MTB"
        else:
            pass

        f.write("        " + r"\multirow{" + str(len(methods)) + r"}{*}{" + tb_label + r"}" + "\n")
        for method in methods:
            m_label = "DA"
            if method == "asc":
                m_label = "Ascending"
            elif method == "desc":
                m_label = "Descending"
            elif method == "asc_block":
                m_label = "Ascending with Blocks"
            elif method == "desc_block":
                m_label = "Descending with Blocks"
            elif method == "abs_indep":
                m_label = "Absolute"
            elif method == "par_indep":
                m_label = "Partial"
            elif method == "max_sib":
                m_label = "Max. Siblings"
            else:
                pass
            key = method + "_" + tie_breaker

            if key in stats:
                if key not in stats:
                    continue
                f.write(
                    "        & "
                    + m_label
                    + " & "
                    + " & ".join(
                        [str(round(stats[key][outcome]["mean"], 2)) for outcome in outcomes]
                    )
                    + r"\\"
                    + "\n"
                )
        f.write(r"        \midrule" + "\n")
    f.write(r"        \bottomrule" + "\n")
    f.write(r"    \end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")
    f.close()


def DistributionAssignment(datdir, tabdir, methods=[], tie_breakers=[]):
    datdir = outdir + os.sep + "penalty=last_pref"
    outputs = {}
    for name in os.listdir(datdir):
        if "." in name:
            continue
        if name in ["par_indep_mtb", "par_indep_stb"]:
            continue
        filename = os.path.join(datdir, name)
        outputs[name] = ReadSolutions(filename)

    # Compute statistics about each method
    distr = {key: {p: {} for p in range(1, 6)} for key in outputs}
    for key in distr:
        for p in distr[key]:
            distr[key][p] = {
                "mean": np.mean([outputs[key][sim]["distribution"][p] for sim in outputs[key]]),
                "se": np.std([outputs[key][sim]["distribution"][p] for sim in outputs[key]])
                / math.sqrt(len(outputs[key])),
            }
        distr[key][6] = {
            "mean": np.mean([outputs[key][sim]["unassigned"] for sim in outputs[key]]),
            "se": np.std([outputs[key][sim]["unassigned"] for sim in outputs[key]])
            / math.sqrt(len(outputs[key])),
        }

    methods = [
        "abs_indep_mtbf",
        "par_indep_mtbf",
        "nosib_mtbf",
        "max_sib_mtbf",
        "desc_block_mtbf",
        "desc_mtbf",
    ]

    width = 0.1
    plt.figure()
    plt.bar(
        [p - 2 * width for p in sorted(distr["abs_indep_mtbf"])],
        [distr["abs_indep_mtbf"][p]["mean"] for p in sorted(distr["abs_indep_mtbf"])],
        width,
        label="Absolute",
        color="black",
    )
    plt.bar(
        [p - width for p in sorted(distr["par_indep_mtbf"])],
        [distr["par_indep_mtbf"][p]["mean"] for p in sorted(distr["par_indep_mtbf"])],
        width,
        label="Patial",
        color="dimgrey",
    )
    plt.bar(
        [p for p in sorted(distr["nosib_mtbf"])],
        [distr["nosib_mtbf"][p]["mean"] for p in sorted(distr["nosib_mtbf"])],
        width,
        label="DA",
        color="darkgrey",
    )
    plt.bar(
        [p + width for p in sorted(distr["desc_mtbf"])],
        [distr["desc_mtbf"][p]["mean"] for p in sorted(distr["desc_mtbf"])],
        width,
        label="Desc.",
        color="silver",
    )
    plt.bar(
        [p + 2 * width for p in sorted(distr["desc_block_mtbf"])],
        [distr["desc_block_mtbf"][p]["mean"] for p in sorted(distr["desc_block_mtbf"])],
        width,
        label="Desc. Block",
        color="gainsboro",
    )

    plt.legend()
    plt.xlabel("Preference of Assignment")
    plt.xlabel("Number of Students")
    plt.xticks(
        sorted(distr["desc_block_mtbf"]), ["1", "2", "3", "4", "5", "Unassigned"], ha="center"
    )  # Adjust rotation and alignment as needed
    plt.savefig(plotdir + os.sep + "comparison_preference_of_assignment.pdf")
    plt.show()

    n = {
        "sib": len([s for s in students if len(siblings[s]) > 0]),
        "nosib": len([s for s in students if len(siblings[s]) == 0]),
    }
    pref_ass = {key: {sim: {} for sim in outputs[key]} for key in methods}
    for key in methods:
        for sim in outputs[key]:
            print(key, sim)
            pref_ass[key][sim] = ComputeAssignmentDistribution(
                outputs[key][sim]["x_opt"], students, pref, siblings
            )

    prefs = [1, 2, 3, 4, 5, "unassigned"]
    distr_gr = {key: {gr: {} for gr in ["sib", "nosib"]} for key in methods}
    for key in distr_gr:
        for gr in ["sib", "nosib"]:
            for pref in prefs:
                if pref == "unassigned":
                    idx = 6
                else:
                    idx = pref
                distr_gr[key][gr][idx] = {
                    "mean": np.mean(
                        [
                            pref_ass[key][sim][gr][pref] / n[gr]
                            for sim in pref_ass[key]
                            if pref in pref_ass[key][sim][gr]
                        ]
                    ),
                    "se": np.std(
                        [
                            pref_ass[key][sim][gr][pref] / n[gr]
                            for sim in pref_ass[key]
                            if pref in pref_ass[key][sim][gr]
                        ]
                    )
                    / math.sqrt(len(pref_ass[key])),
                }

    width = 0.15
    plt.figure()
    plt.bar(
        [p - 2 * width for p in sorted(distr_gr["abs_indep_mtbf"]["sib"])],
        [
            distr_gr["abs_indep_mtbf"]["sib"][p]["mean"]
            for p in sorted(distr_gr["abs_indep_mtbf"]["sib"])
        ],
        width,
        label="Absolute",
        color="black",
    )
    # plt.bar([p - width for p in sorted(distr_gr['par_indep_mtbf']['sib'])], [distr_gr['par_indep_mtbf']['sib'][p]['mean'] for p in sorted(distr_gr['par_indep_mtbf']['sib'])], width, label='Patial', color='dimgrey')
    plt.bar(
        [p - width for p in sorted(distr_gr["nosib_mtbf"]["sib"])],
        [distr_gr["nosib_mtbf"]["sib"][p]["mean"] for p in sorted(distr_gr["nosib_mtbf"]["sib"])],
        width,
        label="SOSM",
        color="dimgrey",
    )
    plt.bar(
        [p for p in sorted(distr_gr["max_sib_mtbf"]["sib"])],
        [
            distr_gr["max_sib_mtbf"]["sib"][p]["mean"]
            for p in sorted(distr_gr["max_sib_mtbf"]["sib"])
        ],
        width,
        label="FOSM",
        color="darkgrey",
    )
    plt.bar(
        [p + width for p in sorted(distr_gr["desc_mtbf"]["sib"])],
        [distr_gr["desc_mtbf"]["sib"][p]["mean"] for p in sorted(distr_gr["desc_mtbf"]["sib"])],
        width,
        label="Desc.",
        color="silver",
    )
    plt.bar(
        [p + 2 * width for p in sorted(distr_gr["desc_block_mtbf"]["sib"])],
        [
            distr_gr["desc_block_mtbf"]["sib"][p]["mean"]
            for p in sorted(distr_gr["desc_block_mtbf"]["sib"])
        ],
        width,
        label="Desc. FA",
        color="gainsboro",
    )
    plt.legend()
    plt.xlabel("Preference of Assignment")
    plt.ylabel("Percentage of Students")
    plt.xticks(
        sorted(distr_gr["desc_block_mtbf"]["sib"]),
        ["1", "2", "3", "4", "5", "Unassigned"],
        ha="center",
    )  # Adjust rotation and alignment as needed
    plt.savefig(plotdir + os.sep + "comparison_preference_of_assignment_sib.pdf")
    plt.show()

    plt.figure()
    plt.bar(
        [p - 2 * width for p in sorted(distr_gr["abs_indep_mtbf"]["nosib"])],
        [
            distr_gr["abs_indep_mtbf"]["nosib"][p]["mean"]
            for p in sorted(distr_gr["abs_indep_mtbf"]["nosib"])
        ],
        width,
        label="Absolute",
        color="black",
    )
    # plt.bar([p - width for p in sorted(distr_gr['par_indep_mtbf']['nosib'])], [distr_gr['par_indep_mtbf']['nosib'][p]['mean'] for p in sorted(distr_gr['par_indep_mtbf']['nosib'])], width, label='Patial', color='dimgrey')
    plt.bar(
        [p - width for p in sorted(distr_gr["nosib_mtbf"]["nosib"])],
        [
            distr_gr["nosib_mtbf"]["nosib"][p]["mean"]
            for p in sorted(distr_gr["nosib_mtbf"]["nosib"])
        ],
        width,
        label="SOSM",
        color="dimgrey",
    )
    plt.bar(
        [p for p in sorted(distr_gr["max_sib_mtbf"]["nosib"])],
        [
            distr_gr["max_sib_mtbf"]["nosib"][p]["mean"]
            for p in sorted(distr_gr["max_sib_mtbf"]["nosib"])
        ],
        width,
        label="FOSM",
        color="darkgrey",
    )
    plt.bar(
        [p + width for p in sorted(distr_gr["desc_mtbf"]["nosib"])],
        [distr_gr["desc_mtbf"]["nosib"][p]["mean"] for p in sorted(distr_gr["desc_mtbf"]["nosib"])],
        width,
        label="Desc.",
        color="silver",
    )
    plt.bar(
        [p + 2 * width for p in sorted(distr_gr["desc_block_mtbf"]["nosib"])],
        [
            distr_gr["desc_block_mtbf"]["nosib"][p]["mean"]
            for p in sorted(distr_gr["desc_block_mtbf"]["nosib"])
        ],
        width,
        label="Desc. FA",
        color="gainsboro",
    )
    plt.legend()
    plt.xlabel("Preference of Assignment")
    plt.ylabel("Percentage of Students")
    plt.xticks(
        sorted(distr_gr["desc_block_mtbf"]["sib"]),
        ["1", "2", "3", "4", "5", "Unassigned"],
        ha="center",
    )  # Adjust rotation and alignment as needed
    plt.savefig(plotdir + os.sep + "comparison_preference_of_assignment_nosib.pdf")
    plt.show()

    distr_gr["abs_indep_mtbf"]["nosib"]
    distr_gr["desc_block_mtbf"]["nosib"]


def DistributionDirectionPriorities(students_per_level):
    level = {s: lev for lev in students_per_level for s in students_per_level[lev]}
    y_opt = ReadOptimalSiblingsPriority(
        "/Users/iriosu/Dropbox/Dynamic priorities in stable matching/outputs/outputs_s=0.txt"
    )
    out = {}
    for id_s in y_opt:
        for id_sib in y_opt[id_s]:
            g_s, g_sib = level[id_s], level[id_sib]
            if g_s not in out:
                out[g_s] = {}
            if g_sib not in out[g_s]:
                out[g_s][g_sib] = 0
            out[g_s][g_sib] += 1

    students_with_siblings_per_level = {
        lev: len([s for s in students_per_level[lev] if siblings[s] != []])
        for lev in students_per_level
    }

    entry_levels = ["PreK", "K", "1", "7", "9"]
    non_entry_levels = ["2", "3", "4", "5", "6", "8", "10", "11", "12"]

    gout = {"e": {"e": 0, "n": 0}, "n": {"e": 0, "n": 0}}
    for g1 in out:
        for g2 in out[g1]:
            id1 = "e" if g1 in entry_levels else "n"
            id2 = "e" if g2 in entry_levels else "n"
            gout[id1][id2] += out[g1][g2]

    l1s = [l1 for l1 in out for l2 in out[l1]]
    l2s = [l2 for l1 in out for l2 in out[l1]]
    vals = [
        1000 * out[l1][l2] / students_with_siblings_per_level[l1] for l1 in out for l2 in out[l1]
    ]

    plt.scatter(l1s, l2s, s=vals)
    ax = plt.gca()
    plt.xlabel("sepal_width")
    plt.ylabel("sepal_length")


def DistributionStudentsPerLevel():
    num_students_per_level = {lev: len(students_per_level[lev]) for lev in students_per_level}
    num_students_with_siblings_per_level = {
        lev: len([s for s in students_per_level[lev] if siblings[s] != []])
        for lev in students_per_level
    }
    order = ["PreK", "K", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

    # Create a sorted list of tuples based on the order
    sorted_data1 = sorted(num_students_per_level.items(), key=lambda x: order.index(x[0]))
    sorted_data2 = sorted(
        num_students_with_siblings_per_level.items(), key=lambda x: order.index(x[0])
    )

    # Extract the keys and values from the sorted lists
    keys1, values1 = zip(*sorted_data1)
    keys2, values2 = zip(*sorted_data2)

    # Create the bar plot with stacked bars
    fig, ax = plt.subplots()
    bottom = np.zeros(len(keys1))

    for i, key in enumerate(keys1):
        bar1 = plt.bar(key, values1[i], color="k", bottom=bottom[i], alpha=0.5)
        bar2 = plt.bar(key, values2[i], color="red", bottom=bottom[i], alpha=0.5)
        bottom[i] += values1[i] + values2[i]
    plt.xlabel("Level")
    plt.ylabel("Number of Students")
    plt.legend([bar1, bar2], ["All", "With Siblings"])
    plt.savefig(plotdir + os.sep + "students_per_level.pdf")
    plt.show()


def RecomputeOutputs(datdir, tabdir, methods=[], tie_breakers=[]):
    datdir = outdir + os.sep + "penalty=last_pref"
    outputs = {}
    for name in os.listdir(datdir):
        if "." in name:
            continue
        filename = os.path.join(datdir, name)
        if name in ["par_indep_mtb", "par_indep_stb"]:
            continue
        print(name)
        outputs[name] = ReadSolutions(filename)
        for s in outputs[name]:
            outputs[name][s] = simulations.ComputeOutputs(
                outputs[name][s]["x_opt"], students, pref, siblings, penalty="last_pref"
            )

    stats = {key: {} for key in outputs}
    outcomes = [
        "num_siblings_together",
        "num_siblings_unassigned_with_overlap",
        "num_siblings_separated_with_one_unassigned",
        "num_siblings_separated_with_both_assigned",
    ]
    for key in outputs:
        for outcome in outcomes:
            stats[key][outcome] = {
                "mean": np.mean([outputs[key][sim][0][outcome] for sim in outputs[key]]),
                "se": np.std([outputs[key][sim][0][outcome] for sim in outputs[key]])
                / math.sqrt(len(outputs[key])),
            }

    # methods = [ 'abs_indep', 'par_indep', 'nosib', 'max_sib',  'asc', 'asc_block', 'desc','desc_block']
    methods = ["abs_indep"]
    tie_breakers = ["stbf", "mtbf", "stb", "mtb"]

    # Create plots or tables summarizing these summary stats
    f = open(tabdir + os.sep + "comparison_fixed_v2.tex", "w")
    f.write(r"\begin{table}" + "\n")
    f.write(r"    \begin{tabular}{lcccccc}" + "\n")
    f.write(r"        \toprule" + "\n")
    f.write(r"        & & & \multicolumn{3}{c}{Separated} \\" + "\n")
    f.write(r"        \cmidrule(lr){4-6}" + "\n")
    f.write(r"        & & Together & None & One & Both \\" + "\n")
    f.write(r"        \midrule" + "\n")
    for tie_breaker in tie_breakers:
        tb_label = "STB"
        if tie_breaker == "stbf":
            tb_label = "STB-F"
        elif tie_breaker == "mtbf":
            tb_label = "MTB-F"
        elif tie_breaker == "mtb":
            tb_label = "MTB"
        else:
            pass

        f.write("        " + r"\multirow{" + str(len(methods)) + r"}{*}{" + tb_label + r"}" + "\n")
        for method in methods:
            m_label = "SOSM"
            if method == "asc":
                m_label = "Ascending"
            elif method == "desc":
                m_label = "Descending"
            elif method == "asc_block":
                m_label = "Ascending FA"
            elif method == "desc_block":
                m_label = "Descending FA"
            elif method == "abs_indep":
                m_label = "Absolute"
            elif method == "par_indep":
                m_label = "Partial"
            elif method == "max_sib":
                m_label = "FOSM"
            else:
                pass
            key = method + "_" + tie_breaker

            if key in stats:
                if key not in stats:
                    continue
                f.write(
                    "        & "
                    + m_label
                    + " & "
                    + " & ".join(
                        [str(round(stats[key][outcome]["mean"], 2)) for outcome in outcomes]
                    )
                    + r"\\"
                    + "\n"
                )
        f.write(r"        \midrule" + "\n")
    f.write(r"        \bottomrule" + "\n")
    f.write(r"    \end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")
    f.close()


# --------------------------------
# New Methods for descriptives
# --------------------------------
def DescriptivesSimulations(datdir, tabdir, methods=[], tie_breakers=[], region="Magallanes"):
    datdir = outdir + os.sep + "comparison"
    outputs = {}
    for name in os.listdir(datdir):
        if "." in name:
            continue
        filedir = os.path.join(datdir, name)
        try:
            outputs[name] = ReadSolutions(filedir)
        except:
            print(f"Error reading {filedir}. Skipping this file.")
            sys.exit(1)

    plot_distributions = False
    table_siblings = True
    stats_feasibility = True

    if plot_distributions:
        distr = {key: {p: {} for p in range(1, 6)} for key in outputs}
        for key in distr:
            for p in distr[key]:
                distr[key][p] = {
                    "mean": np.mean(
                        [
                            outputs[key][sim]["distribution"][p]
                            for sim in outputs[key]
                            if "distribution" in outputs[key][sim]
                        ]
                    ),
                    "se": np.std(
                        [
                            outputs[key][sim]["distribution"][p]
                            for sim in outputs[key]
                            if "distribution" in outputs[key][sim]
                        ]
                    )
                    / math.sqrt(
                        len([sim for sim in outputs[key] if "distribution" in outputs[key][sim]])
                    ),
                }
            distr[key][6] = {
                "mean": np.mean(
                    [
                        outputs[key][sim]["unassigned"]
                        for sim in outputs[key]
                        if "distribution" in outputs[key][sim]
                    ]
                ),
                "se": np.std(
                    [
                        outputs[key][sim]["unassigned"]
                        for sim in outputs[key]
                        if "distribution" in outputs[key][sim]
                    ]
                )
                / math.sqrt(
                    len([sim for sim in outputs[key] if "distribution" in outputs[key][sim]])
                ),
            }

        methods = [
            "abs_hard_mtbf",
            "abs_soft_mtbf",
            "nosib_mtbf",
            "max_sib_mtbf",
            "desc_mtbf",
            "desc_mtbf",
        ]

        width = 0.1
        plt.figure()
        plt.bar(
            [p - 2 * width for p in sorted(distr["abs_hard_mtbf"])],
            [distr["abs_hard_mtbf"][p]["mean"] for p in sorted(distr["abs_hard_mtbf"])],
            width,
            label="Absolute - Hard",
            color="black",
        )
        plt.bar(
            [p - width for p in sorted(distr["abs_soft_mtbf"])],
            [distr["abs_soft_mtbf"][p]["mean"] for p in sorted(distr["abs_soft_mtbf"])],
            width,
            label="Absolute - Soft",
            color="dimgrey",
        )
        plt.bar(
            [p for p in sorted(distr["nosib_mtbf"])],
            [distr["nosib_mtbf"][p]["mean"] for p in sorted(distr["nosib_mtbf"])],
            width,
            label="DA",
            color="darkgrey",
        )
        plt.bar(
            [p + width for p in sorted(distr["desc_mtbf"])],
            [distr["desc_mtbf"][p]["mean"] for p in sorted(distr["desc_mtbf"])],
            width,
            label="Desc.",
            color="silver",
        )
        plt.bar(
            [p + 2 * width for p in sorted(distr["desc_block_mtbf"])],
            [distr["desc_block_mtbf"][p]["mean"] for p in sorted(distr["desc_block_mtbf"])],
            width,
            label="Desc. Block",
            color="gainsboro",
        )

        plt.legend()
        plt.xlabel("Preference of Assignment")
        plt.xlabel("Number of Students")
        plt.xticks(
            sorted(distr["desc_block_mtbf"]), ["1", "2", "3", "4", "5", "Unassigned"], ha="center"
        )  # Adjust rotation and alignment as needed
        plt.savefig(plotdir + os.sep + "comparison_preference_of_assignment.pdf")
        plt.show()

        n = {
            "sib": len([s for s in students if len(siblings[s]) > 0]),
            "nosib": len([s for s in students if len(siblings[s]) == 0]),
        }
        pref_ass = {key: {sim: {} for sim in outputs[key]} for key in methods}
        for key in methods:
            for sim in outputs[key]:
                if "x_opt" not in outputs[key][sim]:
                    continue
                pref_ass[key][sim] = ComputeAssignmentDistribution(
                    outputs[key][sim]["x_opt"], students, pref, siblings
                )

        prefs = [1, 2, 3, 4, 5, "unassigned"]
        distr_gr = {key: {gr: {} for gr in ["sib", "nosib"]} for key in methods}
        for key in distr_gr:
            for gr in ["sib", "nosib"]:
                for pref in prefs:
                    if pref == "unassigned":
                        idx = 6
                    else:
                        idx = pref
                    distr_gr[key][gr][idx] = {
                        "mean": np.mean(
                            [
                                pref_ass[key][sim][gr][pref] / n[gr]
                                for sim in pref_ass[key]
                                if gr in pref_ass[key][sim] and pref in pref_ass[key][sim][gr]
                            ]
                        ),
                        "se": np.std(
                            [
                                pref_ass[key][sim][gr][pref] / n[gr]
                                for sim in pref_ass[key]
                                if gr in pref_ass[key][sim] and pref in pref_ass[key][sim][gr]
                            ]
                        )
                        / math.sqrt(
                            len([sim for sim in pref_ass[key] if gr in pref_ass[key][sim]])
                        ),
                    }

        width = 0.15
        plt.figure()
        plt.bar(
            [p - 2 * width for p in sorted(distr_gr["abs_hard_mtbf"]["sib"])],
            [
                distr_gr["abs_hard_mtbf"]["sib"][p]["mean"]
                for p in sorted(distr_gr["abs_hard_mtbf"]["sib"])
            ],
            width,
            label="Absolute - Hard",
            color="black",
        )
        plt.bar(
            [p - width for p in sorted(distr_gr["abs_soft_mtbf"]["sib"])],
            [
                distr_gr["abs_soft_mtbf"]["sib"][p]["mean"]
                for p in sorted(distr_gr["abs_soft_mtbf"]["sib"])
            ],
            width,
            label="Absolute - Soft",
            color="dimgrey",
        )
        plt.bar(
            [p for p in sorted(distr_gr["max_sib_mtbf"]["sib"])],
            [
                distr_gr["max_sib_mtbf"]["sib"][p]["mean"]
                for p in sorted(distr_gr["max_sib_mtbf"]["sib"])
            ],
            width,
            label="FOSM",
            color="darkgrey",
        )
        plt.bar(
            [p + width for p in sorted(distr_gr["nosib_mtbf"]["sib"])],
            [
                distr_gr["nosib_mtbf"]["sib"][p]["mean"]
                for p in sorted(distr_gr["nosib_mtbf"]["sib"])
            ],
            width,
            label="SOSM",
            color="silver",
        )
        plt.bar(
            [p + 2 * width for p in sorted(distr_gr["desc_mtbf"]["sib"])],
            [distr_gr["desc_mtbf"]["sib"][p]["mean"] for p in sorted(distr_gr["desc_mtbf"]["sib"])],
            width,
            label="Desc.",
            color="gainsboro",
        )
        plt.legend()
        plt.xlabel("Preference of Assignment")
        plt.ylabel("Percentage of Students")
        plt.xticks(
            sorted(distr_gr["desc_mtbf"]["sib"]),
            ["1", "2", "3", "4", "5", "Unassigned"],
            ha="center",
        )  # Adjust rotation and alignment as needed
        plt.savefig(plotdir + os.sep + "comparison_preference_of_assignment_sib.pdf")
        plt.show()

        plt.figure()
        plt.bar(
            [p - 2 * width for p in sorted(distr_gr["abs_hard_mtbf"]["nosib"])],
            [
                distr_gr["abs_hard_mtbf"]["nosib"][p]["mean"]
                for p in sorted(distr_gr["abs_hard_mtbf"]["nosib"])
            ],
            width,
            label="Absolute - Hard",
            color="black",
        )
        plt.bar(
            [p - width for p in sorted(distr_gr["abs_soft_mtbf"]["nosib"])],
            [
                distr_gr["abs_soft_mtbf"]["nosib"][p]["mean"]
                for p in sorted(distr_gr["abs_soft_mtbf"]["nosib"])
            ],
            width,
            label="Absolute - Soft",
            color="dimgrey",
        )
        plt.bar(
            [p for p in sorted(distr_gr["max_sib_mtbf"]["nosib"])],
            [
                distr_gr["max_sib_mtbf"]["nosib"][p]["mean"]
                for p in sorted(distr_gr["max_sib_mtbf"]["nosib"])
            ],
            width,
            label="FOSM",
            color="darkgrey",
        )
        plt.bar(
            [p + width for p in sorted(distr_gr["nosib_mtbf"]["nosib"])],
            [
                distr_gr["nosib_mtbf"]["nosib"][p]["mean"]
                for p in sorted(distr_gr["nosib_mtbf"]["nosib"])
            ],
            width,
            label="SOSM",
            color="silver",
        )
        plt.bar(
            [p + 2 * width for p in sorted(distr_gr["desc_mtbf"]["nosib"])],
            [
                distr_gr["desc_mtbf"]["nosib"][p]["mean"]
                for p in sorted(distr_gr["desc_mtbf"]["nosib"])
            ],
            width,
            label="Desc.",
            color="gainsboro",
        )
        plt.legend()
        plt.xlabel("Preference of Assignment")
        plt.ylabel("Percentage of Students")
        plt.xticks(
            sorted(distr_gr["desc_mtbf"]["sib"]),
            ["1", "2", "3", "4", "5", "Unassigned"],
            ha="center",
        )  # Adjust rotation and alignment as needed
        plt.savefig(plotdir + os.sep + "comparison_preference_of_assignment_nosib.pdf")
        plt.show()
    if table_siblings:
        stats = {key: {} for key in outputs}
        outcomes = [
            "together",
            "unassigned_separated",
            "separated_one_unassigned",
            "assigned_separated",
        ]
        for key in outputs:
            for outcome in outcomes:
                # stats[key][outcome] = {'mean':np.mean([outputs[key][sim][outcome] for sim in outputs[key] ]),\
                #                         'se':np.std([outputs[key][sim][outcome] for sim in outputs[key]])/math.sqrt(len(outputs[key])) }
                stats[key][outcome] = {
                    "mean": np.mean(
                        [
                            outputs[key]["comparison"][sim][outcome]
                            for sim in outputs[key]["comparison"]
                            if outcome in outputs[key]["comparison"][sim]
                        ]
                    ),
                    "se": np.std(
                        [
                            outputs[key]["comparison"][sim][outcome]
                            for sim in outputs[key]["comparison"]
                            if outcome in outputs[key]["comparison"][sim]
                        ]
                    )
                    / math.sqrt(
                        len(
                            [
                                sim
                                for sim in outputs[key]["comparison"]
                                if outcome in outputs[key]["comparison"][sim]
                            ]
                        )
                    ),
                }

        methods = ["abs_hard", "abs_soft", "max_sib", "nosib", "desc"]
        tie_breakers = ["stbf", "mtbf", "stb", "mtb"]

        # Create plots or tables summarizing these summary stats
        f = open(tabdir + os.sep + "comparison_new.tex", "w")
        f.write(r"\begin{table}" + "\n")
        f.write(r"    \begin{tabular}{lcccccc}" + "\n")
        f.write(r"        \toprule" + "\n")
        f.write(r"        & & & \multicolumn{3}{c}{Separated} \\" + "\n")
        f.write(r"        \cmidrule(lr){4-6}" + "\n")
        f.write(r"        & & Together & None & One & Both \\" + "\n")
        f.write(r"        \midrule" + "\n")
        for tie_breaker in tie_breakers:
            tb_label = "STB"
            if tie_breaker == "stbf":
                tb_label = "STB-F"
            elif tie_breaker == "mtbf":
                tb_label = "MTB-F"
            elif tie_breaker == "mtb":
                tb_label = "MTB"
            else:
                pass

            f.write(
                "        " + r"\multirow{" + str(len(methods)) + r"}{*}{" + tb_label + r"}" + "\n"
            )
            for method in methods:
                m_label = "SOSM"
                if method == "asc":
                    m_label = "Ascending"
                elif method == "desc":
                    m_label = "Descending"
                elif method == "asc_block":
                    m_label = "Ascending FA"
                elif method == "desc_block":
                    m_label = "Descending FA"
                elif method == "abs_hard":
                    m_label = "Absolute - Hard"
                elif method == "abs_soft":
                    m_label = "Absolute - Soft"
                elif method == "par_hard":
                    m_label = "Partial - Hard"
                elif method == "par_soft":
                    m_label = "Partial - Soft"
                elif method == "max_sib":
                    m_label = "FOSM"
                else:
                    pass
                key = method + "_" + tie_breaker

                if key in stats:
                    f.write(
                        "        & "
                        + m_label
                        + " & "
                        + " & ".join(
                            [str(round(stats[key][outcome]["mean"], 2)) for outcome in outcomes]
                        )
                        + r"\\"
                        + "\n"
                    )
            f.write(r"        \midrule" + "\n")
        f.write(r"        \bottomrule" + "\n")
        f.write(r"    \end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
        f.close()
    if stats_feasibility:
        methods = ["abs_hard", "par_hard"]
        tie_breakers = ["stb", "mtb", "stbf", "mtbf"]
        for method in methods:
            for tie_breaker in tie_breakers:
                key = method + "_" + tie_breaker
                if key in outputs:
                    print(
                        key,
                        "Infeasible instances (out of 100):",
                        len([s for s in outputs[key] if outputs[key][s] == {}]),
                    )


def DescriptivesComparison(datdir, tabdir, methods=[], tie_breakers=[], region="Magallanes"):
    region = "Magallanes"
    datdir = dropbox_dir + os.sep + "outputs" + os.sep + region + os.sep + "comparison"
    outputs = {}
    for name in os.listdir(datdir):
        if "." in name:
            continue
        filedir = os.path.join(datdir, name)
        # try:
        outputs[name] = ReadSolutions(filedir)
        # except:
        #     print(f"Error reading {filedir}. Skipping this file.")
        #     sys.exit(1)

    stats = {method: {} for method in outputs}
    outcomes = [
        "avg_pref",
        "top",
        "unassigned",
        "together",
        "unassigned_separated",
        "separated_one_unassigned",
        "assigned_separated",
    ]
    for method in outputs:
        stats[method]["solved"] = len(
            [
                sim
                for sim in outputs[method]["comparison"]
                if "together" in outputs[method]["comparison"][sim]
            ]
        )

        for outcome in outcomes:
            stats[method][outcome] = {
                "mean": np.mean(
                    [
                        (
                            outputs[method]["comparison"][sim]["distribution"][1]
                            if outcome == "top"
                            else (
                                sum(
                                    key * val
                                    for key, val in outputs[method]["comparison"][sim][
                                        "distribution"
                                    ].items()
                                )
                                / sum(outputs[method]["comparison"][sim]["distribution"].values())
                                if outcome == "avg_pref"
                                else (
                                    sum(
                                        key * val
                                        for key, val in outputs[method]["comparison"][sim][
                                            "distribution"
                                        ].items()
                                    )
                                    if outcome == "assigned_pref"
                                    else outputs[method]["comparison"][sim][outcome]
                                )
                            )
                        )
                        for sim in outputs[method]["comparison"]
                        if "obj" in outputs[method]["comparison"][sim]
                    ]
                ),
                "se": np.std(
                    [
                        (
                            outputs[method]["comparison"][sim]["distribution"][1]
                            if outcome == "top"
                            else (
                                sum(
                                    key * val
                                    for key, val in outputs[method]["comparison"][sim][
                                        "distribution"
                                    ].items()
                                )
                                / sum(outputs[method]["comparison"][sim]["distribution"].values())
                                if outcome == "avg_pref"
                                else (
                                    sum(
                                        key * val
                                        for key, val in outputs[method]["comparison"][sim][
                                            "distribution"
                                        ].items()
                                    )
                                    if outcome == "assigned_pref"
                                    else outputs[method]["comparison"][sim][outcome]
                                )
                            )
                        )
                        for sim in outputs[method]["comparison"]
                        if "obj" in outputs[method]["comparison"][sim]
                    ]
                )
                / math.sqrt(
                    len(
                        [
                            sim
                            for sim in outputs[method]["comparison"]
                            if "obj" in outputs[method]["comparison"][sim]
                        ]
                    )
                ),
            }

    if summarized_table_with_avg_preference_wo_se:
        methods = ["abs_hard_mtbf", "abs_soft_mtbf", "max_sib_mtbf", "nosib_mtbf", "desc_mtbf"]
        # Create plots or tables summarizing these summary stats
        f = open(tabdir + os.sep + region + "_comparison.tex", "w")
        f.write(r"\begin{table}" + "\n")
        f.write(r"    \caption{Sensitivity to Softness}\label{tab: sensitivity to softness}" + "\n")
        f.write(r"    \centerline{\scalebox{0.85}{\begin{tabular}{lccccccccccccccc}" + "\n")
        f.write(r"        \toprule" + "\n")
        f.write(
            r"        \multicolumn{3}{c}{} & \multicolumn{6}{c}{} & \multicolumn{6}{c}{Separated}\\"
            + "\n"
        )
        f.write(r"        \cmidrule(lr){10-15}" + "\n")
        f.write(
            r"                &  &  & \multicolumn{2}{c}{Top Pref.} & \multicolumn{2}{c}{Unassigned} & \multicolumn{2}{c}{Together} & \multicolumn{2}{c}{None} & \multicolumn{2}{c}{One} & \multicolumn{2}{c}{Both}  \\"
            + "\n"
        )
        f.write(
            r"        \cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}\cmidrule(lr){14-15}"
            + "\n"
        )
        f.write(
            r"        & Solved & Avg. Pref. & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE \\"
            + "\n"
        )
        f.write(r"        \midrule" + "\n")
        for method in methods:
            label = ""
            if method == "abs_hard_mtbf":
                label = "Absolute - Hard"
            elif method == "abs_soft_mtbf":
                label = "Absolute - Soft"
            elif method == "max_sib_mtbf":
                label = "FOSM"
            elif method == "nosib_mtbf":
                label = "SOSM"
            elif method == "desc_mtbf":
                label = "Descending"
            else:
                pass
            f.write(
                "         "
                + label
                + " & "
                + str(stats[method]["solved"])
                + " & "
                + f"{stats[method]['avg_pref']['mean']:.3f}"
                + " & "
                + " & ".join(
                    [
                        f"{stats[method][outcome]['mean']:.2f}"
                        + " & "
                        + f"{stats[method][outcome]['se']:.2f}"
                        for outcome in outcomes
                        if outcome != "avg_pref"
                    ]
                )
                + r"\\"
                + "\n"
            )
        f.write(r"        \bottomrule" + "\n")
        f.write(r"    \end{tabular}}}" + "\n")
        f.write(r"\end{table}" + "\n")
        f.close()

    if summarized_table_with_avg_preference:
        methods = ["abs_hard_mtbf", "abs_soft_mtbf", "max_sib_mtbf", "nosib_mtbf", "desc_mtbf"]
        # Create plots or tables summarizing these summary stats
        f = open(tabdir + os.sep + region + "_comparison.tex", "w")
        f.write(r"\begin{table}" + "\n")
        f.write(r"    \caption{Sensitivity to Softness}\label{tab: sensitivity to softness}" + "\n")
        f.write(r"    \centerline{\scalebox{0.85}{\begin{tabular}{lcccccccccccccc}" + "\n")
        f.write(r"        \toprule" + "\n")
        f.write(r"        & \multicolumn{7}{c}{} & \multicolumn{6}{c}{Separated} \\" + "\n")
        f.write(r"        \cmidrule(lr){9-14}" + "\n")
        f.write(
            r"        &  & \multicolumn{2}{c}{Assigned} & \multicolumn{2}{c}{Unassigned} & \multicolumn{2}{c}{Together} & \multicolumn{2}{c}{None} & \multicolumn{2}{c}{One} & \multicolumn{2}{c}{Both} \\"
            + "\n"
        )
        f.write(
            r"        \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}"
            + "\n"
        )
        f.write(
            r"        & Solved & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE \\"
            + "\n"
        )
        f.write(r"        \midrule" + "\n")
        for method in methods:
            label = ""
            if method == "abs_hard_mtbf":
                label = "Absolute - Hard"
            elif method == "abs_soft_mtbf":
                label = "Absolute - Soft"
            elif method == "max_sib_mtbf":
                label = "FOSM"
            elif method == "nosib_mtbf":
                label = "SOSM"
            elif method == "desc_mtbf":
                label = "Descending"
            else:
                pass
            f.write(
                "         "
                + label
                + " & "
                + str(stats[method]["solved"])
                + " & "
                + " & ".join(
                    [
                        str(round(stats[method][outcome]["mean"], 2))
                        + " & "
                        + str(round(stats[method][outcome]["se"], 2))
                        for outcome in outcomes
                    ]
                )
                + r"\\"
                + "\n"
            )
        f.write(r"        \bottomrule" + "\n")
        f.write(r"    \end{tabular}}}" + "\n")
        f.write(r"\end{table}" + "\n")
        f.close()

    if summarized_table_with_top_preference:
        methods = ["abs_hard_mtbf", "abs_soft_mtbf", "max_sib_mtbf", "nosib_mtbf", "desc_mtbf"]
        # Create plots or tables summarizing these summary stats
        f = open(tabdir + os.sep + region + "_comparison.tex", "w")
        f.write(r"\begin{table}" + "\n")
        f.write(r"    \caption{Sensitivity to Softness}\label{tab: sensitivity to softness}" + "\n")
        f.write(r"    \centerline{\scalebox{0.85}{\begin{tabular}{lcccccccccccccc}" + "\n")
        f.write(r"        \toprule" + "\n")
        f.write(r"        & \multicolumn{7}{c}{} & \multicolumn{6}{c}{Separated} \\" + "\n")
        f.write(r"        \cmidrule(lr){9-14}" + "\n")
        f.write(
            r"        &  & \multicolumn{2}{c}{Top Pref.} & \multicolumn{2}{c}{Unassigned} & \multicolumn{2}{c}{Together} & \multicolumn{2}{c}{None} & \multicolumn{2}{c}{One} & \multicolumn{2}{c}{Both} \\"
            + "\n"
        )
        f.write(
            r"        \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}"
            + "\n"
        )
        f.write(
            r"        & Solved & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE \\"
            + "\n"
        )
        f.write(r"        \midrule" + "\n")
        for method in methods:
            label = ""
            if method == "abs_hard_mtbf":
                label = "Absolute - Hard"
            elif method == "abs_soft_mtbf":
                label = "Absolute - Soft"
            elif method == "max_sib_mtbf":
                label = "FOSM"
            elif method == "nosib_mtbf":
                label = "SOSM"
            elif method == "desc_mtbf":
                label = "Descending"
            else:
                pass
            f.write(
                "         "
                + label
                + " & "
                + str(stats[method]["solved"])
                + " & "
                + " & ".join(
                    [
                        str(round(stats[method][outcome]["mean"], 2))
                        + " & "
                        + str(round(stats[method][outcome]["se"], 2))
                        for outcome in outcomes
                    ]
                )
                + r"\\"
                + "\n"
            )
        f.write(r"        \bottomrule" + "\n")
        f.write(r"    \end{tabular}}}" + "\n")
        f.write(r"\end{table}" + "\n")
        f.close()

    if all_tables:
        methods = [
            "abs_hard",
            "abs_soft",
            "par_hard",
            "par_soft",
            "max_sib",
            "nosib",
            "desc",
            "asc",
            "sim",
            "size_desc",
            "size_asc",
        ]
        tie_breakers = ["stb", "stbf", "mtb", "mtbf"]
        for tb in tie_breakers:
            f = open(tabdir + os.sep + region + "_comparison_" + tb + ".tex", "w")
            f.write(r"\begin{table}" + "\n")
            f.write(
                r"    \caption{Sensitivity to Softness}\label{tab: sensitivity to softness}" + "\n"
            )
            f.write(r"    \centerline{\scalebox{0.85}{\begin{tabular}{lccccccccccccccc}" + "\n")
            f.write(r"        \toprule" + "\n")
            f.write(
                r"        \multicolumn{3}{c}{} & \multicolumn{6}{c}{} & \multicolumn{6}{c}{Separated}\\"
                + "\n"
            )
            f.write(r"        \cmidrule(lr){10-15}" + "\n")
            f.write(
                r"                &  &  & \multicolumn{2}{c}{Top Pref.} & \multicolumn{2}{c}{Unassigned} & \multicolumn{2}{c}{Together} & \multicolumn{2}{c}{None} & \multicolumn{2}{c}{One} & \multicolumn{2}{c}{Both}  \\"
                + "\n"
            )
            f.write(
                r"        \cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}\cmidrule(lr){14-15}"
                + "\n"
            )
            f.write(
                r"        & Solved & Avg. Pref. & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE \\"
                + "\n"
            )
            f.write(r"        \midrule" + "\n")
            for method in methods:
                label = ""
                if method == "abs_hard":
                    label = "Absolute - Hard"
                elif method == "abs_soft":
                    label = "Absolute - Soft"
                elif method == "par_hard":
                    label = "Partial - Hard"
                elif method == "par_soft":
                    label = "Partial - Soft"
                elif method == "max_sib":
                    label = "FOSM"
                elif method == "nosib":
                    label = "SOSM"
                elif method == "desc":
                    label = "Descending"
                elif method == "asc":
                    label = "Ascending"
                elif method == "sim":
                    label = "Simultaneous"
                elif method == "size_desc":
                    label = "Size - Descending"
                elif method == "size_asc":
                    label = "Size - Ascending"
                else:
                    pass
                f.write(
                    "         "
                    + label
                    + " & "
                    + str(stats[method + "_" + tb]["solved"])
                    + " & "
                    + f"{stats[method + "_" + tb]['avg_pref']['mean']:.3f}"
                    + " & "
                    + " & ".join(
                        [
                            f"{stats[method + "_" + tb][outcome]['mean']:.2f}"
                            + " & "
                            + f"{stats[method + "_" + tb][outcome]['se']:.2f}"
                            for outcome in outcomes
                            if outcome != "avg_pref"
                        ]
                    )
                    + r"\\"
                    + "\n"
                )
            f.write(r"        \bottomrule" + "\n")
            f.write(r"    \end{tabular}}}" + "\n")
            f.write(r"\end{table}" + "\n")
            f.close()

    if performance_stats:
        stats = {method: {} for method in outputs}
        outcomes = [
            "runtime",
            "mipgap",
        ]
        outputs.keys()
        for method in outputs:
            for outcome in outcomes:
                stats[method][outcome] = {
                    "mean": np.mean(
                        [
                            (
                                outputs[method]["comparison"][sim]["distribution"][1]
                                if outcome == "top"
                                else (
                                    sum(
                                        key * val
                                        for key, val in outputs[method]["comparison"][sim][
                                            "distribution"
                                        ].items()
                                    )
                                    / sum(
                                        outputs[method]["comparison"][sim]["distribution"].values()
                                    )
                                    if outcome == "avg_pref"
                                    else (
                                        sum(
                                            key * val
                                            for key, val in outputs[method]["comparison"][sim][
                                                "distribution"
                                            ].items()
                                        )
                                        if outcome == "assigned_pref"
                                        else outputs[method]["comparison"][sim][outcome]
                                    )
                                )
                            )
                            for sim in outputs[method]["comparison"]
                            if "obj" in outputs[method]["comparison"][sim]
                        ]
                    ),
                    "se": np.std(
                        [
                            (
                                outputs[method]["comparison"][sim]["distribution"][1]
                                if outcome == "top"
                                else (
                                    sum(
                                        key * val
                                        for key, val in outputs[method]["comparison"][sim][
                                            "distribution"
                                        ].items()
                                    )
                                    / sum(
                                        outputs[method]["comparison"][sim]["distribution"].values()
                                    )
                                    if outcome == "avg_pref"
                                    else (
                                        sum(
                                            key * val
                                            for key, val in outputs[method]["comparison"][sim][
                                                "distribution"
                                            ].items()
                                        )
                                        if outcome == "assigned_pref"
                                        else outputs[method]["comparison"][sim][outcome]
                                    )
                                )
                            )
                            for sim in outputs[method]["comparison"]
                            if "obj" in outputs[method]["comparison"][sim]
                        ]
                    )
                    / math.sqrt(
                        len(
                            [
                                sim
                                for sim in outputs[method]["comparison"]
                                if "obj" in outputs[method]["comparison"][sim]
                            ]
                        )
                    ),
                }
            if method in ["abs_hard_mtbf", "abs_soft_mtbf"]:
                print(
                    method,
                    "Mean Runtime:",
                    stats[method]["runtime"]["mean"],
                    "SE:",
                    stats[method]["runtime"]["se"],
                )
                print(
                    method,
                    "Mean MIP Gap:",
                    stats[method]["mipgap"]["mean"],
                    "SE:",
                    stats[method]["mipgap"]["se"],
                )


def DescriptivesSensitivity(datdir, tabdir, methods=[], tie_breakers=[], region="Magallanes"):
    region = "Atacama"
    datdir = dropbox_dir + os.sep + "outputs" + os.sep + region + os.sep + "comparison"
    outputs = {}
    for name in os.listdir(datdir):
        if "." in name:
            continue
        filedir = os.path.join(datdir, name)
        outputs[name] = ReadSolutions(filedir)

    if region == "Magallanes":
        Ks = ["280", "290", "300", "310", "320"]
    else:
        Ks = list(outputs["abs_soft_mtbf"]["sensitivity"].keys())

    stats = {key: {k: {} for k in Ks} for key in outputs}
    outcomes = [
        "avg_pref",
        "top",
        "unassigned",
        "together",
        "unassigned_separated",
        "separated_one_unassigned",
        "assigned_separated",
    ]

    for method in outputs:
        for k in outputs[method]["sensitivity"]:
            stats[method][k]["solved"] = len(
                [
                    sim
                    for sim in outputs[method]["sensitivity"][k]
                    if "together" in outputs[method]["sensitivity"][k][sim]
                ]
            )
            for outcome in outcomes:
                stats[method][k][outcome] = {
                    "mean": np.mean(
                        [
                            (
                                outputs[method]["sensitivity"][k][sim]["distribution"][1]
                                if outcome == "top"
                                else (
                                    sum(
                                        key * val
                                        for key, val in outputs[method]["sensitivity"][k][sim][
                                            "distribution"
                                        ].items()
                                    )
                                    / sum(
                                        outputs[method]["sensitivity"][k][sim][
                                            "distribution"
                                        ].values()
                                    )
                                    if outcome == "avg_pref"
                                    else (
                                        sum(
                                            key * val
                                            for key, val in outputs[method]["sensitivity"][k][sim][
                                                "distribution"
                                            ].items()
                                        )
                                        if outcome == "assigned_pref"
                                        else outputs[method]["sensitivity"][k][sim][outcome]
                                    )
                                )
                            )
                            for sim in outputs[method]["sensitivity"][k]
                            if "obj" in outputs[method]["sensitivity"][k][sim]
                        ]
                    ),
                    "se": np.std(
                        [
                            (
                                outputs[method]["sensitivity"][k][sim]["distribution"][1]
                                if outcome == "top"
                                else (
                                    sum(
                                        key * val
                                        for key, val in outputs[method]["sensitivity"][k][sim][
                                            "distribution"
                                        ].items()
                                    )
                                    / sum(
                                        outputs[method]["sensitivity"][k][sim][
                                            "distribution"
                                        ].values()
                                    )
                                    if outcome == "avg_pref"
                                    else (
                                        sum(
                                            key * val
                                            for key, val in outputs[method]["sensitivity"][k][sim][
                                                "distribution"
                                            ].items()
                                        )
                                        if outcome == "assigned_pref"
                                        else outputs[method]["sensitivity"][k][sim][outcome]
                                    )
                                )
                            )
                            for sim in outputs[method]["sensitivity"][k]
                            if "obj" in outputs[method]["sensitivity"][k][sim]
                        ]
                    )
                    / math.sqrt(
                        len(
                            [
                                sim
                                for sim in outputs[method]["sensitivity"][k]
                                if "obj" in outputs[method]["sensitivity"][k][sim]
                            ]
                        )
                    ),
                }

    # Create plots or tables summarizing these summary stats
    including_top_without_se = True
    if including_top_without_se:
        f = open(tabdir + os.sep + region + "_sensitivity.tex", "w")
        f.write(r"\begin{table}" + "\n")
        f.write(r"    \caption{Sensitivity to Softness}\label{tab: sensitivity to softness}" + "\n")
        f.write(r"    \centerline{\scalebox{0.85}{\begin{tabular}{lccccccccccccccc}" + "\n")
        f.write(r"        \toprule" + "\n")
        f.write(
            r"        \multicolumn{3}{c}{} & \multicolumn{6}{c}{} & \multicolumn{6}{c}{Separated}\\"
            + "\n"
        )
        f.write(r"        \cmidrule(lr){10-15}" + "\n")
        f.write(
            r"                &  &  & \multicolumn{2}{c}{Top Pref.} & \multicolumn{2}{c}{Unassigned} & \multicolumn{2}{c}{Together} & \multicolumn{2}{c}{None} & \multicolumn{2}{c}{One} & \multicolumn{2}{c}{Both}  \\"
            + "\n"
        )
        f.write(
            r"        \cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}\cmidrule(lr){14-15}"
            + "\n"
        )
        f.write(
            r"        & Solved & Avg. Pref. & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE \\"
            + "\n"
        )
        f.write(r"        \midrule" + "\n")
        for k in sorted([str(kp) for kp in Ks]):
            f.write(
                "         "
                + str(k)
                + " & "
                + str(stats["abs_soft_mtbf"][k]["solved"])
                + " & "
                + str(round(stats["abs_soft_mtbf"][k]["avg_pref"]["mean"], 3))
                + " & "
                + " & ".join(
                    [
                        str(round(stats["abs_soft_mtbf"][k][outcome]["mean"], 2))
                        + " & "
                        + str(round(stats["abs_soft_mtbf"][k][outcome]["se"], 2))
                        for outcome in outcomes
                        if outcome != "avg_pref"
                    ]
                )
                + r"\\"
                + "\n"
            )
        f.write(r"        \midrule" + "\n")
        f.write(r"        \bottomrule" + "\n")
        f.write(r"    \end{tabular}}}" + "\n")
        f.write(r"\end{table}" + "\n")
        f.close()

    if including_top:
        f = open(tabdir + os.sep + region + "_sensitivity.tex", "w")
        f.write(r"\begin{table}" + "\n")
        f.write(r"    \caption{Sensitivity to Softness}\label{tab: sensitivity to softness}" + "\n")
        f.write(r"    \centerline{\scalebox{0.85}{\begin{tabular}{lcccccccccccccccc}" + "\n")
        f.write(r"        \toprule" + "\n")
        f.write(
            r"        & & \multicolumn{4}{c}{Assigned} & \multicolumn{4}{c}{} & \multicolumn{6}{c}{Separated} \\"
            + "\n"
        )
        f.write(r"        \cmidrule(lr){3-6}\cmidrule(lr){11-16}" + "\n")
        f.write(
            r"        &  & \multicolumn{2}{c}{Top Pref.} & \multicolumn{2}{c}{Avg. Pref.} & \multicolumn{2}{c}{Unassigned} & \multicolumn{2}{c}{Together} & \multicolumn{2}{c}{None} & \multicolumn{2}{c}{One} & \multicolumn{2}{c}{Both} \\"
            + "\n"
        )
        f.write(
            r"        \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}\cmidrule(lr){15-16}"
            + "\n"
        )
        f.write(
            r"        & Solved & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE \\"
            + "\n"
        )
        f.write(r"        \midrule" + "\n")
        for k in sorted([str(kp) for kp in Ks]):
            f.write(
                "         "
                + str(k)
                + " & "
                + str(stats["abs_soft_mtbf"][k]["solved"])
                + " & "
                + " & ".join(
                    [
                        str(round(stats["abs_soft_mtbf"][k][outcome]["mean"], 2))
                        + " & "
                        + str(round(stats["abs_soft_mtbf"][k][outcome]["se"], 2))
                        for outcome in outcomes
                    ]
                )
                + r"\\"
                + "\n"
            )
        f.write(r"        \midrule" + "\n")
        f.write(
            "         Hard & "
            + str(stats["abs_hard_mtbf"]["0"]["solved"])
            + " & "
            + " & ".join(
                [
                    str(round(stats["abs_hard_mtbf"]["0"][outcome]["mean"], 2))
                    + " & "
                    + str(round(stats["abs_hard_mtbf"]["0"][outcome]["se"], 2))
                    for outcome in outcomes
                ]
            )
            + r"\\"
            + "\n"
        )
        f.write(r"        \bottomrule" + "\n")
        f.write(r"    \end{tabular}}}" + "\n")
        f.write(r"\end{table}" + "\n")
        f.close()

    if excluding_top:
        f = open(tabdir + os.sep + region + "_sensitivity.tex", "w")
        f.write(r"\begin{table}" + "\n")
        f.write(r"    \caption{Sensitivity to Softness}\label{tab: sensitivity to softness}" + "\n")
        f.write(r"    \centerline{\scalebox{0.85}{\begin{tabular}{lcccccccccccccc}" + "\n")
        f.write(r"        \toprule" + "\n")
        f.write(
            r"        & & \multicolumn{4}{c}{Assigned} & \multicolumn{4}{c}{} & \multicolumn{6}{c}{Separated} \\"
            + "\n"
        )
        f.write(r"        \cmidrule(lr){3-6}\cmidrule(lr){11-16}" + "\n")
        f.write(
            r"        &  &  &  \multicolumn{4}{c}{Assigned} & \multicolumn{2}{c}{Unassigned} & \multicolumn{2}{c}{Together} & \multicolumn{2}{c}{None} & \multicolumn{2}{c}{One} & \multicolumn{2}{c}{Both} \\"
            + "\n"
        )
        f.write(
            r"        \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}"
            + "\n"
        )
        f.write(
            r"        & Solved & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE \\"
            + "\n"
        )
        f.write(r"        \midrule" + "\n")
        for k in sorted([str(kp) for kp in Ks]):
            f.write(
                "         "
                + str(k)
                + " & "
                + str(stats["abs_soft_mtbf"][k]["solved"])
                + " & "
                + " & ".join(
                    [
                        str(round(stats["abs_soft_mtbf"][k][outcome]["mean"], 2))
                        + " & "
                        + str(round(stats["abs_soft_mtbf"][k][outcome]["se"], 2))
                        for outcome in outcomes
                    ]
                )
                + r"\\"
                + "\n"
            )
        f.write(r"        \midrule" + "\n")
        f.write(
            "         Hard & "
            + str(stats["abs_hard_mtbf"]["0"]["solved"])
            + " & "
            + " & ".join(
                [
                    str(round(stats["abs_hard_mtbf"]["0"][outcome]["mean"], 2))
                    + " & "
                    + str(round(stats["abs_hard_mtbf"]["0"][outcome]["se"], 2))
                    for outcome in outcomes
                ]
            )
            + r"\\"
            + "\n"
        )
        f.write(r"        \bottomrule" + "\n")
        f.write(r"    \end{tabular}}}" + "\n")
        f.write(r"\end{table}" + "\n")
        f.close()


def DescriptivesSensitivityLotteries(datdir, tabdir, methods=[], tie_breakers=[]):
    datdir = outdir + os.sep + "lotteries_v4"
    outputs = {}
    for name in os.listdir(datdir):
        if "." in name:
            continue
        filedir = os.path.join(datdir, name)
        outputs[name] = ReadSolutions(filedir)

    Ks = list(outputs["abs_soft_mtbf"].keys())

    stats = {key: {k: {} for k in Ks} for key in outputs}
    outcomes = [
        "top",
        "unassigned",
        "together",
        "unassigned_separated",
        "separated_one_unassigned",
        "assigned_separated",
    ]
    for method in outputs:
        for k in outputs[method]:
            stats[method][k]["solved"] = len(
                [sim for sim in outputs[method][k] if "together" in outputs[method][k][sim]]
            )
            for outcome in outcomes:
                stats[method][k][outcome] = {
                    "mean": np.mean(
                        [
                            (
                                outputs[method][k][sim]["distribution"][1]
                                if outcome == "top"
                                else outputs[method][k][sim][outcome]
                            )
                            for sim in outputs[method][k]
                            if "obj" in outputs[method][k][sim]
                        ]
                    ),
                    "se": np.std(
                        [
                            (
                                outputs[method][k][sim]["distribution"][1]
                                if outcome == "top"
                                else outputs[method][k][sim][outcome]
                            )
                            for sim in outputs[method][k]
                            if "obj" in outputs[method][k][sim]
                        ]
                    )
                    / math.sqrt(
                        len([sim for sim in outputs[method][k] if "obj" in outputs[method][k][sim]])
                    ),
                }

    width = 0.1
    tbs = ["stb", "stbf", "mtb", "mtbf"]

    plt.figure()
    plt.bar(
        [p - 2 * width for p in range(len(tbs))],
        [stats["abs_hard_" + tb]["0"]["together"]["mean"] for tb in tbs],
        width,
        label="Absolute - Hard",
        color="black",
    )
    plt.bar(
        [p - width for p in range(len(tbs))],
        [stats["abs_soft_" + tb]["280"]["together"]["mean"] for tb in tbs],
        width,
        label="Absolute - Hybrid",
        color="dimgrey",
    )
    plt.bar(
        [p for p in range(len(tbs))],
        [stats["abs_soft_" + tb]["0"]["together"]["mean"] for tb in tbs],
        width,
        label="Absolute - Soft",
        color="darkgrey",
    )
    plt.bar(
        [p + width for p in range(len(tbs))],
        [stats["nosib_" + tb]["0"]["together"]["mean"] for tb in tbs],
        width,
        label="SOSM",
        color="silver",
    )
    plt.bar(
        [p + 2 * width for p in range(len(tbs))],
        [stats["desc_" + tb]["0"]["together"]["mean"] for tb in tbs],
        width,
        label="Desc.",
        color="gainsboro",
    )

    plt.legend()
    plt.ylabel("Together")
    plt.xlabel("Tie-Breaking Rule")
    plt.xticks(
        range(len(tbs)), ["STB", "STB-F", "MTB", "MTB-F"], ha="center"
    )  # Adjust rotation and alignment as needed
    # plt.savefig(plotdir + os.sep + 'comparison_preference_of_assignment.pdf')
    plt.ylim([300, 650])
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=3, frameon=False)
    plt.savefig(plotdir + os.sep + "together_across_tbs.pdf", bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.bar(
        [p - 2 * width for p in range(len(tbs))],
        [stats["abs_hard_" + tb]["0"]["top"]["mean"] for tb in tbs],
        width,
        label="Absolute - Hard",
        color="black",
    )
    plt.bar(
        [p - width for p in range(len(tbs))],
        [stats["abs_soft_" + tb]["280"]["top"]["mean"] for tb in tbs],
        width,
        label="Absolute - Hybrid",
        color="dimgrey",
    )
    plt.bar(
        [p for p in range(len(tbs))],
        [stats["abs_soft_" + tb]["0"]["top"]["mean"] for tb in tbs],
        width,
        label="Absolute - Soft",
        color="darkgrey",
    )
    plt.bar(
        [p + width for p in range(len(tbs))],
        [stats["nosib_" + tb]["0"]["top"]["mean"] for tb in tbs],
        width,
        label="SOSM",
        color="silver",
    )
    plt.bar(
        [p + 2 * width for p in range(len(tbs))],
        [stats["desc_" + tb]["0"]["top"]["mean"] for tb in tbs],
        width,
        label="Desc.",
        color="gainsboro",
    )

    plt.legend()
    plt.ylabel("Top Preference")
    plt.xlabel("Tie-Breaking Rule")
    plt.xticks(
        range(len(tbs)), ["STB", "STB-F", "MTB", "MTB-F"], ha="center"
    )  # Adjust rotation and alignment as needed
    # plt.savefig(plotdir + os.sep + 'comparison_preference_of_assignment.pdf')
    plt.ylim([2450, 2750])
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=3, frameon=False)
    plt.savefig(plotdir + os.sep + "top_preference_across_tbs.pdf", bbox_inches="tight")
    plt.show()


def DescriptivesReceiverOptimization(datdir, tabdir, objective="ROSM"):
    region = "Magallanes"
    datdir = (
        dropbox_dir
        + os.sep
        + "outputs"
        + os.sep
        + objective
        + os.sep
        + region
        + os.sep
        + "comparison"
    )
    outputs = {}
    for name in os.listdir(datdir):
        if "." in name:
            continue
        filedir = os.path.join(datdir, name)
        # try:
        outputs[name] = ReadSolutions(filedir)
        # except:
        #     print(f"Error reading {filedir}. Skipping this file.")
        #     sys.exit(1)

    stats = {method: {} for method in outputs}
    outcomes = [
        "avg_pref",
        "top",
        "unassigned",
        "together",
        "unassigned_separated",
        "separated_one_unassigned",
        "assigned_separated",
    ]
    for method in outputs:
        if outputs[method]["comparison"] == {}:
            continue
        stats[method]["solved"] = len(
            [
                sim
                for sim in outputs[method]["comparison"]
                if "together" in outputs[method]["comparison"][sim]
            ]
        )
        for outcome in outcomes:
            stats[method][outcome] = {
                "mean": np.mean(
                    [
                        (
                            outputs[method]["comparison"][sim]["distribution"][1]
                            if outcome == "top"
                            else (
                                sum(
                                    key * val
                                    for key, val in outputs[method]["comparison"][sim][
                                        "distribution"
                                    ].items()
                                )
                                / sum(outputs[method]["comparison"][sim]["distribution"].values())
                                if outcome == "avg_pref"
                                else outputs[method]["comparison"][sim][outcome]
                            )
                        )
                        for sim in outputs[method]["comparison"]
                        if "obj" in outputs[method]["comparison"][sim]
                    ]
                ),
                "se": np.std(
                    [
                        (
                            outputs[method]["comparison"][sim]["distribution"][1]
                            if outcome == "top"
                            else (
                                sum(
                                    key * val
                                    for key, val in outputs[method]["comparison"][sim][
                                        "distribution"
                                    ].items()
                                )
                                / sum(outputs[method]["comparison"][sim]["distribution"].values())
                                if outcome == "avg_pref"
                                else outputs[method]["comparison"][sim][outcome]
                            )
                        )
                        for sim in outputs[method]["comparison"]
                        if "obj" in outputs[method]["comparison"][sim]
                    ]
                )
                / math.sqrt(
                    len(
                        [
                            sim
                            for sim in outputs[method]["comparison"]
                            if "obj" in outputs[method]["comparison"][sim]
                        ]
                    )
                ),
            }

    for method in outputs:
        if outputs[method]["sensitivity"] == {}:
            continue
        for k in outputs[method]["sensitivity"]:
            if k not in stats[method]:
                stats[method][k] = {}

            stats[method][k]["solved"] = len(
                [
                    sim
                    for sim in outputs[method]["sensitivity"][k]
                    if "together" in outputs[method]["sensitivity"][k][sim]
                ]
            )
            for outcome in outcomes:
                stats[method][k][outcome] = {
                    "mean": np.mean(
                        [
                            (
                                outputs[method]["sensitivity"][k][sim]["distribution"][1]
                                if outcome == "top"
                                else (
                                    sum(
                                        key * val
                                        for key, val in outputs[method]["sensitivity"][k][sim][
                                            "distribution"
                                        ].items()
                                    )
                                    / sum(
                                        outputs[method]["sensitivity"][k][sim][
                                            "distribution"
                                        ].values()
                                    )
                                    if outcome == "avg_pref"
                                    else (
                                        sum(
                                            key * val
                                            for key, val in outputs[method]["sensitivity"][k][sim][
                                                "distribution"
                                            ].items()
                                        )
                                        if outcome == "assigned_pref"
                                        else outputs[method]["sensitivity"][k][sim][outcome]
                                    )
                                )
                            )
                            for sim in outputs[method]["sensitivity"][k]
                            if "obj" in outputs[method]["sensitivity"][k][sim]
                        ]
                    ),
                    "se": np.std(
                        [
                            (
                                outputs[method]["sensitivity"][k][sim]["distribution"][1]
                                if outcome == "top"
                                else (
                                    sum(
                                        key * val
                                        for key, val in outputs[method]["sensitivity"][k][sim][
                                            "distribution"
                                        ].items()
                                    )
                                    / sum(
                                        outputs[method]["sensitivity"][k][sim][
                                            "distribution"
                                        ].values()
                                    )
                                    if outcome == "avg_pref"
                                    else (
                                        sum(
                                            key * val
                                            for key, val in outputs[method]["sensitivity"][k][sim][
                                                "distribution"
                                            ].items()
                                        )
                                        if outcome == "assigned_pref"
                                        else outputs[method]["sensitivity"][k][sim][outcome]
                                    )
                                )
                            )
                            for sim in outputs[method]["sensitivity"][k]
                            if "obj" in outputs[method]["sensitivity"][k][sim]
                        ]
                    )
                    / math.sqrt(
                        len(
                            [
                                sim
                                for sim in outputs[method]["sensitivity"][k]
                                if "obj" in outputs[method]["sensitivity"][k][sim]
                            ]
                        )
                    ),
                }
    # Create plots or tables summarizing these summary stats
    Ks = [
        "0",
        "280",
        "290",
        "300",
        "310",
        "320",
    ]  # list(outputs["abs_soft_mtbf"]["sensitivity"].keys())
    f = open(tabdir + os.sep + region + "_analysis_" + objective + ".tex", "w")
    f.write(r"\begin{table}" + "\n")
    f.write(r"    \caption{Sensitivity to Softness}\label{tab: sensitivity to softness}" + "\n")
    f.write(r"    \centerline{\scalebox{0.85}{\begin{tabular}{lccccccccccccccc}" + "\n")
    f.write(r"        \toprule" + "\n")
    f.write(
        r"        \multicolumn{3}{c}{} & \multicolumn{6}{c}{} & \multicolumn{6}{c}{Separated}\\"
        + "\n"
    )
    f.write(r"        \cmidrule(lr){10-15}" + "\n")
    f.write(
        r"                &  &  & \multicolumn{2}{c}{Top Pref.} & \multicolumn{2}{c}{Unassigned} & \multicolumn{2}{c}{Together} & \multicolumn{2}{c}{None} & \multicolumn{2}{c}{One} & \multicolumn{2}{c}{Both}  \\"
        + "\n"
    )
    f.write(
        r"        \cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}\cmidrule(lr){14-15}"
        + "\n"
    )
    f.write(
        r"        & Solved & Avg. Pref. & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE \\"
        + "\n"
    )
    f.write(r"        \midrule" + "\n")
    for k in sorted([str(kp) for kp in Ks]):
        f.write(
            "         "
            + str(k)
            + " & "
            + str(stats["abs_soft_mtbf"][k]["solved"])
            + " & "
            + f"{stats["abs_soft_mtbf"][k]["avg_pref"]["mean"]:.3f}"
            + " & "
            + " & ".join(
                [
                    f"{stats["abs_soft_mtbf"][k][outcome]["mean"]:.2f}"
                    + " & "
                    + f"{stats["abs_soft_mtbf"][k][outcome]["se"]:.2f}"
                    for outcome in outcomes
                    if outcome != "avg_pref"
                ]
            )
            + r"\\"
            + "\n"
        )
    f.write(r"        \midrule" + "\n")
    f.write(
        "         Hard & "
        + str(stats["abs_hard_mtbf"]["0"]["solved"])
        + " & "
        + f"{stats["abs_hard_mtbf"]["0"]["avg_pref"]["mean"]:.3f}"
        + " & "
        + " & ".join(
            [
                f"{stats["abs_hard_mtbf"]["0"][outcome]["mean"]:.2f}"
                + " & "
                + f"{stats["abs_hard_mtbf"]["0"][outcome]["se"]:.2f}"
                for outcome in outcomes
                if outcome != "avg_pref"
            ]
        )
        + r"\\"
        + "\n"
    )
    f.write(
        "         NTB & "
        + str(stats["abs_ntb_mtbf"]["solved"])
        + " & "
        + f"{stats["abs_ntb_mtbf"]["avg_pref"]["mean"]:.3f}"
        + " & "
        + " & ".join(
            [
                f"{stats["abs_ntb_mtbf"][outcome]["mean"]:.2f}"
                + " & "
                + f"{stats["abs_ntb_mtbf"][outcome]["se"]:.2f}"
                for outcome in outcomes
                if outcome != "avg_pref"
            ]
        )
        + r"\\"
        + "\n"
    )
    f.write(r"        \bottomrule" + "\n")
    f.write(r"    \end{tabular}}}" + "\n")
    f.write(r"\end{table}" + "\n")
    f.close()


def ComparisonWithvsWOSiblings(datdir, tabdir, methods=[], tie_breakers=[], region="Magallanes"):
    region = "Magallanes"
    datdir = dropbox_dir + os.sep + "outputs" + os.sep + region + os.sep + "comparison"
    outputs = {}
    for name in os.listdir(datdir):
        if "." in name:
            continue
        filedir = os.path.join(datdir, name)
        outputs[name] = ReadSolutions(filedir)

    methods = [
        "abs_hard_mtbf",
        "abs_soft_mtbf",
        "max_sib_mtbf",
        "nosib_mtbf",
        "desc_mtbf",
        "asc_mtbf",
    ]

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn = (
        genin.read_instance(indir + os.sep + region + os.sep + "2023" + os.sep + "instance.txt")
    )

    n = {
        "sib": len([s for s in students if len(siblings[s]) > 0]),
        "nosib": len([s for s in students if len(siblings[s]) == 0]),
    }

    pref_ass = {key: {sim: {} for sim in outputs[key]} for key in methods}
    for key in methods:
        for sim in outputs[key]["comparison"]:
            print(key, sim)
            if "x_opt" not in outputs[key]["comparison"][sim]:
                continue
            pref_ass[key][sim] = ComputeAssignmentDistribution(
                outputs[key]["comparison"][sim]["x_opt"], students, pref, siblings
            )

    # compute average preference of assignment for students with and without siblings for each method
    stats = {method: {"sib": {}, "nosib": {}} for method in methods}
    for method in stats:
        for tp in stats[method]:
            stats[method][tp]["assigned"] = {
                "mean": np.mean(
                    [
                        sum(
                            num
                            for pos, num in pref_ass[method][sim][tp].items()
                            if pos != "unassigned"
                        )
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]
                    ]
                ),
                "se": np.std(
                    [
                        sum(
                            num
                            for pos, num in pref_ass[method][sim][tp].items()
                            if pos != "unassigned"
                        )
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]
                    ]
                )
                / math.sqrt(len(pref_ass[method])),
            }

            stats[method][tp]["unassigned"] = {
                "mean": np.mean(
                    [
                        pref_ass[method][sim][tp]["unassigned"]
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]
                    ]
                ),
                "se": np.std(
                    [
                        pref_ass[method][sim][tp]["unassigned"]
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]
                    ]
                )
                / math.sqrt(len(pref_ass[method])),
            }

            stats[method][tp]["pct_unassigned"] = {
                "mean": np.mean(
                    [
                        pref_ass[method][sim][tp]["unassigned"]
                        / sum(num for pos, num in pref_ass[method][sim][tp].items())
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]
                    ]
                ),
                "se": np.std(
                    [
                        pref_ass[method][sim][tp]["unassigned"]
                        / sum(num for pos, num in pref_ass[method][sim][tp].items())
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]
                    ]
                )
                / math.sqrt(len(pref_ass[method])),
            }

            stats[method][tp]["pct_assigned"] = {
                "mean": np.mean(
                    [
                        1
                        - (
                            pref_ass[method][sim][tp]["unassigned"]
                            / sum(num for pos, num in pref_ass[method][sim][tp].items())
                        )
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]
                    ]
                ),
                "se": np.std(
                    [
                        1
                        - (
                            pref_ass[method][sim][tp]["unassigned"]
                            / sum(num for pos, num in pref_ass[method][sim][tp].items())
                        )
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]
                    ]
                )
                / math.sqrt(len(pref_ass[method])),
            }

            stats[method][tp]["top"] = {
                "mean": np.mean(
                    [
                        pref_ass[method][sim][tp][1]
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]
                    ]
                ),
                "se": np.std(
                    [
                        pref_ass[method][sim][tp][1]
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]
                    ]
                )
                / math.sqrt(len(pref_ass[method])),
            }

            stats[method][tp]["pct_top"] = {
                "mean": np.mean(
                    [
                        pref_ass[method][sim][tp][1]
                        / sum(num for pos, num in pref_ass[method][sim][tp].items())
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]
                    ]
                ),
                "se": np.std(
                    [
                        pref_ass[method][sim][tp][1]
                        / sum(num for pos, num in pref_ass[method][sim][tp].items())
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]
                    ]
                )
                / math.sqrt(len(pref_ass[method])),
            }

            stats[method][tp]["avg_pref"] = {
                "mean": np.mean(
                    [
                        sum(
                            pos * num
                            for pos, num in pref_ass[method][sim][tp].items()
                            if pos != "unassigned"
                        )
                        / sum(
                            num
                            for pos, num in pref_ass[method][sim][tp].items()
                            if pos != "unassigned"
                        )
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]  # <-- moved outside to prevent KeyError
                    ]
                ),
                "se": np.std(
                    [
                        sum(
                            pos * num
                            for pos, num in pref_ass[method][sim][tp].items()
                            if pos != "unassigned"
                        )
                        / sum(
                            num
                            for pos, num in pref_ass[method][sim][tp].items()
                            if pos != "unassigned"
                        )
                        for sim in pref_ass[method]
                        if tp in pref_ass[method][sim]  # <-- moved outside to prevent KeyError
                    ]
                )
                / math.sqrt(len(pref_ass[method])),
            }

    outcomes = ["avg_pref", "pct_top", "pct_unassigned"]
    str_cols = "ll" + "".join("c" for i in range(2 * len(outcomes)))
    f = open(tabdir + os.sep + region + "_sib_vs_nosib.tex", "w")
    f.write(r"\begin{table}" + "\n")
    f.write(r"    \caption{Comparison Siblings vs. No-Siblings}\label{tab: sib vs no sib}" + "\n")
    f.write(r"    \centerline{\scalebox{0.85}{\begin{tabular}{" + str_cols + "}" + "\n")
    f.write(r"        \toprule" + "\n")
    f.write(
        r"        \multicolumn{2}{c}{} & \multicolumn{2}{c}{Avg. Pref.} & \multicolumn{2}{c}{Assigned - Top [\%]} & \multicolumn{2}{c}{Unassigned [\%]}\\"
        + "\n"
    )
    f.write(r"      \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}" + "\n")
    f.write(r"        &  &  Mean & SE & Mean & SE & Mean & SE  \\" + "\n")
    for method in stats:
        m_label = "SOSM"
        if method == "asc_mtbf":
            m_label = "Ascending"
        elif method == "desc_mtbf":
            m_label = "Descending"
        elif method == "asc_block":
            m_label = "Ascending FA"
        elif method == "desc_block":
            m_label = "Descending FA"
        elif method == "abs_hard_mtbf":
            m_label = "Absolute - Hard"
        elif method == "abs_soft_mtbf":
            m_label = "Absolute - Soft"
        elif method == "par_hard_mtbf":
            m_label = "Partial - Hard"
        elif method == "par_soft_mtbf":
            m_label = "Partial - Soft"
        elif method == "max_sib_mtbf":
            m_label = "FOSM"
        else:
            pass

        f.write(r"        \midrule" + "\n")
        f.write(r"        \multirow{2}{*}{" + m_label + "} \n")
        for tp in stats[method]:
            tp_label = "Siblings" if tp == "sib" else "No-Siblings"
            f.write(
                "         "
                + " & "
                + tp_label
                + " & "
                + " & ".join(
                    [
                        str(round(stats[method][tp][outcome]["mean"], 3))
                        + " & "
                        + str(round(stats[method][tp][outcome]["se"], 3))
                        for outcome in outcomes
                    ]
                )
                + r"\\"
                + "\n"
            )
    f.write(r"        \bottomrule" + "\n")
    f.write(r"    \end{tabular}}}" + "\n")
    f.write(r"\end{table}" + "\n")
    f.close()


def DescriptivesNTB(datdir, tabdir):
    region = "Magallanes"
    method = "abs_ntb_mtbf"
    objectives = ["SOSM", "ROSM", "MXSM"]
    outputs = {objective: {} for objective in objectives}
    for objective in outputs:
        datdir = (
            dropbox_dir
            + os.sep
            + "outputs"
            + os.sep
            + objective
            + os.sep
            + region
            + os.sep
            + "comparison"
        )
        outputs[objective] = ReadSolutions(os.path.join(datdir, method))

    stats = {method: {} for method in outputs}
    outcomes = [
        "avg_pref",
        "top",
        "unassigned",
        "together",
        "unassigned_separated",
        "separated_one_unassigned",
        "assigned_separated",
    ]
    for method in outputs:
        if outputs[method]["comparison"] == {}:
            continue
        stats[method]["solved"] = len(
            [
                sim
                for sim in outputs[method]["comparison"]
                if "together" in outputs[method]["comparison"][sim]
            ]
        )
        for outcome in outcomes:
            stats[method][outcome] = {
                "mean": np.mean(
                    [
                        (
                            outputs[method]["comparison"][sim]["distribution"][1]
                            if outcome == "top"
                            else (
                                sum(
                                    key * val
                                    for key, val in outputs[method]["comparison"][sim][
                                        "distribution"
                                    ].items()
                                )
                                / sum(outputs[method]["comparison"][sim]["distribution"].values())
                                if outcome == "avg_pref"
                                else outputs[method]["comparison"][sim][outcome]
                            )
                        )
                        for sim in outputs[method]["comparison"]
                        if "obj" in outputs[method]["comparison"][sim]
                    ]
                ),
                "se": np.std(
                    [
                        (
                            outputs[method]["comparison"][sim]["distribution"][1]
                            if outcome == "top"
                            else (
                                sum(
                                    key * val
                                    for key, val in outputs[method]["comparison"][sim][
                                        "distribution"
                                    ].items()
                                )
                                / sum(outputs[method]["comparison"][sim]["distribution"].values())
                                if outcome == "avg_pref"
                                else outputs[method]["comparison"][sim][outcome]
                            )
                        )
                        for sim in outputs[method]["comparison"]
                        if "obj" in outputs[method]["comparison"][sim]
                    ]
                )
                / math.sqrt(
                    len(
                        [
                            sim
                            for sim in outputs[method]["comparison"]
                            if "obj" in outputs[method]["comparison"][sim]
                        ]
                    )
                ),
            }

    f = open(tabdir + os.sep + region + "_analysis_NTB.tex", "w")
    f.write(r"\begin{table}" + "\n")
    f.write(r"    \caption{Sensitivity to Softness}\label{tab: sensitivity to softness}" + "\n")
    f.write(r"    \centerline{\scalebox{0.85}{\begin{tabular}{lccccccccccccccc}" + "\n")
    f.write(r"        \toprule" + "\n")
    f.write(
        r"        \multicolumn{3}{c}{} & \multicolumn{6}{c}{} & \multicolumn{6}{c}{Separated}\\"
        + "\n"
    )
    f.write(r"        \cmidrule(lr){10-15}" + "\n")
    f.write(
        r"                &  &  & \multicolumn{2}{c}{Top Pref.} & \multicolumn{2}{c}{Unassigned} & \multicolumn{2}{c}{Together} & \multicolumn{2}{c}{None} & \multicolumn{2}{c}{One} & \multicolumn{2}{c}{Both}  \\"
        + "\n"
    )
    f.write(
        r"        \cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}\cmidrule(lr){14-15}"
        + "\n"
    )
    f.write(
        r"        & Solved & Avg. Pref. & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE & Mean & SE \\"
        + "\n"
    )
    f.write(r"        \midrule" + "\n")
    for objective in objectives:
        f.write(
            "         "
            + str(objective)
            + " & "
            + str(stats[objective]["solved"])
            + " & "
            + f"{stats[objective]["avg_pref"]["mean"]:.3f}"
            + " & "
            + " & ".join(
                [
                    f"{stats[objective][outcome]["mean"]:.2f}"
                    + " & "
                    + f"{stats[objective][outcome]["se"]:.2f}"
                    for outcome in outcomes
                    if outcome != "avg_pref"
                ]
            )
            + r"\\"
            + "\n"
        )
    f.write(r"        \bottomrule" + "\n")
    f.write(r"    \end{tabular}}}" + "\n")
    f.write(r"\end{table}" + "\n")
    f.close()


if __name__ == "__main__":
    print()
