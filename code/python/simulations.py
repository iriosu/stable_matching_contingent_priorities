from fileinput import filename
import sys, os, time, copy
from gurobipy import *

__gurobi_threads = 1
import numpy as np
import pickle, math, random
import multiprocessing

np.random.seed(1)
random.seed(1)

import generate_inputs as genin
import algorithms as alg
import solve_opt as opt

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------
# Auxiliary methods for handling simulations
# --------------------------------
def ComputeOutputs(x_opt, students, pref, siblings, penalty=0, x_base=None):
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

    num_assigned = sum(x_opt[s][c] for s in x_opt for c in x_opt[s] if x_opt[s][c] > 1 - 1e-3)
    num_unassigned = len(students) - num_assigned
    pr_opt = {id_s: None for id_s in students}
    pref_assignment = {}

    assigned_students = [s for s in x_opt if max(x_opt[s].values(), default=0) > 1 - 1e-3]
    unassigned_students = [s for s in students if s not in assigned_students]

    for id_s in x_opt:
        for id_c in x_opt[id_s]:
            if x_opt[id_s][id_c] > 1 - 1e-3:
                pr = FindPreference(id_s, id_c)
                if pr not in pref_assignment:
                    pref_assignment[pr] = 0
                pref_assignment[pr] += 1
                pr_opt[id_s] = pr

    mean_pref = sum([pr * pref_assignment[pr] for pr in pref_assignment]) / sum(
        [pref_assignment[pr] for pr in pref_assignment]
    )

    obj = 0
    if isinstance(penalty, (int, float, complex)):
        obj = sum([pr * pref_assignment[pr] for pr in pref_assignment]) + sum(
            [penalty for s in unassigned_students]
        )
    elif penalty == "last_pref":
        obj = sum([pr * pref_assignment[pr] for pr in pref_assignment]) + sum(
            [(max(opref[s].values()) + 1) for s in unassigned_students]
        )
    else:
        print("***ERROR: Unkonwn penalty")
        sys.exit(1)

    # siblings assigned together
    siblings_together = []
    for id_s in x_opt:
        rbd = list(x_opt[id_s].keys())[0].split("_")[0]
        for sib in siblings[id_s]:
            if sib not in x_opt:
                continue
            rbd_sib = list(x_opt[sib].keys())[0].split("_")[0]
            if rbd == rbd_sib:
                siblings_together.append(id_s)
                siblings_together.append(sib)
    siblings_together = list(set(siblings_together))

    # siblings separated with room for improvement
    case1, case2, case3 = [], [], []
    for id_s in siblings:
        for sib in siblings[id_s]:
            if id_s not in x_opt and sib not in x_opt:
                if (
                    len(
                        set([pref[id_s][p].split("_")[0] for p in pref[id_s]]).intersection(
                            [pref[sib][p].split("_")[0] for p in pref[sib]]
                        )
                    )
                    > 0
                ):
                    case1.append(id_s)
                    case1.append(sib)
            elif id_s in x_opt and sib not in x_opt:
                if (
                    len(
                        set(
                            [
                                pref[id_s][p].split("_")[0]
                                for p in pref[id_s]
                                if p <= FindPreference(id_s, list(x_opt[id_s].keys())[0])
                            ]
                        ).intersection([pref[sib][p].split("_")[0] for p in pref[sib]])
                    )
                    > 0
                ):
                    case2.append(id_s)
                    case2.append(sib)
            elif id_s not in x_opt and sib in x_opt:
                if (
                    len(
                        set([pref[id_s][p].split("_")[0] for p in pref[id_s]]).intersection(
                            [
                                pref[sib][p].split("_")[0]
                                for p in pref[sib]
                                if p <= FindPreference(sib, list(x_opt[sib].keys())[0])
                            ]
                        )
                    )
                    > 0
                ):
                    case2.append(id_s)
                    case2.append(sib)
            else:
                rbd, rbd_sib = (
                    list(x_opt[id_s].keys())[0].split("_")[0],
                    list(x_opt[sib].keys())[0].split("_")[0],
                )
                if (
                    rbd != rbd_sib
                    and len(
                        set(
                            [
                                pref[id_s][p].split("_")[0]
                                for p in pref[id_s]
                                if p <= FindPreference(id_s, list(x_opt[id_s].keys())[0])
                            ]
                        ).intersection(
                            [
                                pref[sib][p].split("_")[0]
                                for p in pref[sib]
                                if p <= FindPreference(sib, list(x_opt[sib].keys())[0])
                            ]
                        )
                    )
                    > 0
                ):
                    case3.append(id_s)
                    case3.append(sib)

    case1 = list(set(case1))
    case2 = list(set(case2))
    case3 = list(set(case3))

    outs = {
        "mean_pref": mean_pref,
        "num_unassigned": num_unassigned,
        "pref_assignment": pref_assignment,
        "num_siblings_together": len(siblings_together),
        "num_siblings_unassigned_with_overlap": len(case1),
        "num_siblings_separated_with_one_unassigned": len(case2),
        "num_siblings_separated_with_both_assigned": len(case3),
    }

    if x_base is not None:
        enter, leave, better, worst = 0, 0, 0, 0
        pr_base = {id_s: None for id_s in students}
        for id_s in x_base:
            for id_c in x_base[id_s]:
                if x_base[id_s][id_c] > 1 - 1e-3:
                    pr_base[id_s] = FindPreference(id_s, id_c)
        for id_s in students:
            if pr_base[id_s] == None and pr_opt[id_s] == None:
                continue
            elif pr_base[id_s] != None and pr_opt[id_s] == None:
                leave += 1
            elif pr_base[id_s] == None and pr_opt[id_s] != None:
                enter += 1
            else:
                if pr_base[id_s] == pr_opt[id_s]:
                    continue
                elif pr_base[id_s] > pr_opt[id_s]:
                    better += 1
                elif pr_base[id_s] < pr_opt[id_s]:
                    worst += 1
                else:
                    print("***Error: unknown outcome")
                    sys.exit(1)
        outs.update({"enter": enter, "leave": leave, "better": better, "worst": worst})

    return outs, obj


def WriteOutput(outputs, outfile, instance=None):
    def write_instance(students, colleges, pref, cap, siblings, levels, students_per_level, out):
        out.write("# Num. students:" + str(len(students)) + "\n")
        out.write("# Num. colleges:" + str(len(colleges)) + "\n")
        out.write("# Students:" + ",".join([s for s in students]) + "\n")
        out.write("# Colleges:" + ",".join([c for c in colleges]) + "\n")
        out.write("# Capacities:\n")
        for c in cap:
            out.write(c + " " + str(cap[c]) + "\n")
        out.write("# Student preferences:\n")
        for s in students:
            out.write(
                s
                + " "
                + " ".join(["(" + str(p) + "," + str(pref[s][p]) + ")" for p in sorted(pref[s])])
                + "\n"
            )
        out.write("# College priorities:\n")
        for c in colleges:
            out.write(
                c
                + " "
                + " ".join(["(" + str(p) + "," + str(pref[c][p]) + ")" for p in sorted(pref[c])])
                + "\n"
            )
        out.write("# Siblings:\n")
        for s in students:
            if len(siblings[s]) == 0:
                out.write(s + "\n")
            else:
                out.write(s + " " + " ".join([str(sib) for sib in siblings[s]]) + "\n")
        out.write("# Levels:\n")
        for idx in levels:
            out.write(idx + " " + " ".join([str(cc) for cc in levels[idx]]) + "\n")
        out.write("# Students per Level:\n")
        for idx in students_per_level:
            out.write(idx + " " + " ".join([str(id_s) for id_s in students_per_level[idx]]) + "\n")

    f = open(outfile, "w")
    if outputs["status"] == "infeasible":
        f.write("# PROBLEM INFEASIBLE\n")
        outdir = os.path.dirname(outfile)
        # if instance is not None:
        #     students, colleges, pref, cap, siblings, levels, students_per_level,tb = instance
        #     write_instance(students, colleges, pref, cap, siblings, levels, students_per_level, f)
        #     f.write('\n')
        #
        # if os.path.exists(outdir + os.sep + 'model.ilp'):
        #     f.write('# IIS \n' )
        #     g = open(outdir + os.sep + 'model.ilp', 'r')
        #     for line in g.readlines():
        #         f.write(line)
        #     g.close()
    elif outputs["status"] == "stopped":
        f.write("# TIME OUT\n")
        outdir = os.path.dirname(outfile)
        # if instance is not None:
        #     students, colleges, pref, cap, siblings, levels, students_per_level,tb = instance
        #     write_instance(students, colleges, pref, cap, siblings, levels, students_per_level, f)
        #     f.write('\n')
    else:
        f.write("# Objective:" + str(outputs["obj"]) + "\n")
        f.write("# Num Vars:" + str(outputs["num_vars"]) + "\n")
        f.write("# Num Cols:" + str(outputs["num_cols"]) + "\n")
        f.write("# Mean preference of assignment:" + str(outputs["mean_pref"]) + "\n")
        f.write("# Num unassigned:" + str(outputs["num_unassigned"]) + "\n")
        f.write("# Num siblings together:" + str(outputs["num_siblings_together"]) + "\n")
        f.write(
            "# Num siblings unassigned with more preferred overlap:"
            + str(outputs["num_siblings_unassigned_with_overlap"])
            + "\n"
        )
        f.write(
            "# Num siblings separated with one unassigned and more preferred overlap:"
            + str(outputs["num_siblings_separated_with_one_unassigned"])
            + "\n"
        )
        f.write(
            "# Num siblings separated with both assigned and more preferred overlap:"
            + str(outputs["num_siblings_separated_with_both_assigned"])
            + "\n"
        )
        f.write("# Run time:" + str(outputs["runtime"]) + "\n")
        f.write("# MipGap:" + str(outputs["mipgap"]) + "\n")

        # extras
        if "nodes" in outputs:
            f.write("# Nodes:" + str(outputs["nodes"]) + "\n")
        if "obj_root" in outputs:
            f.write("# Objective Root:" + str(outputs["obj_root"]) + "\n")
        if "obj_lr" in outputs:
            f.write("# Objective LR:" + str(outputs["obj_lr"]) + "\n")
        if "gap_root" in outputs:
            f.write("# Gap Root:" + str(outputs["gap_root"]) + "\n")
        if "gap_lr" in outputs:
            f.write("# Gap LR:" + str(outputs["gap_lr"]) + "\n")
        if "enter" in outputs:
            f.write("# Enter:" + str(outputs["enter"]) + "\n")
        if "leave" in outputs:
            f.write("# Leave:" + str(outputs["leave"]) + "\n")
        if "better" in outputs:
            f.write("# Better:" + str(outputs["better"]) + "\n")
        if "worst" in outputs:
            f.write("# Worst:" + str(outputs["worst"]) + "\n")

        f.write("# Distribution preference of assignment:\n")
        for p in sorted(outputs["pref_assignment"]):
            f.write(str(p) + " " + str(outputs["pref_assignment"][p]) + "\n")
        f.write("# Optimal x:\n")
        for id_s in outputs["x_opt"]:
            for id_c in outputs["x_opt"][id_s]:
                f.write(
                    str(id_s) + " " + str(id_c) + " " + str(outputs["x_opt"][id_s][id_c]) + "\n"
                )
        if "y_opt" in outputs:
            f.write("# Optimal y:\n")
            for id_s in outputs["y_opt"]:
                for id_sib in outputs["y_opt"][id_s]:
                    for id_c in outputs["y_opt"][id_s][id_sib]:
                        f.write(
                            str(id_s)
                            + " "
                            + str(id_sib)
                            + " "
                            + str(id_c)
                            + " "
                            + str(outputs["y_opt"][id_s][id_sib][id_c])
                            + "\n"
                        )
    f.close()


# --------------------------------
# Methods to solve problem
# --------------------------------
def SubRoutine(indat):
    # read instance
    indir, outdir, methods, tie_breakers, regions, penalty_unassigned, sim, extras, objective = (
        indat
    )

    if extras is None:
        extras = [None]

    # randomize lotteries according to tie-breaker
    for region in regions:
        students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn = (
            genin.read_instance(indir + os.sep + region + os.sep + "2023" + os.sep + "instance.txt")
        )

        if not os.path.exists(outdir + os.sep + region):
            os.makedirs(outdir + os.sep + region, exist_ok=True)

        for tie_breaker in tie_breakers:
            if not os.path.exists(outdir + os.sep + region + os.sep + tie_breaker):
                os.makedirs(
                    outdir + os.sep + region + os.sep + "comparison" + os.sep + tie_breaker,
                    exist_ok=True,
                )

            if os.path.exists(
                outdir
                + os.sep
                + region
                + os.sep
                + "comparison"
                + os.sep
                + tie_breaker
                + os.sep
                + "tb_s="
                + str(sim)
                + ".pck"
            ):
                tb = pickle.load(
                    open(
                        outdir
                        + os.sep
                        + region
                        + os.sep
                        + "comparison"
                        + os.sep
                        + tie_breaker
                        + os.sep
                        + "tb_s="
                        + str(sim)
                        + ".pck",
                        "rb",
                    )
                )
                for c in colleges:
                    rbd = c.split("_")[0]
                    applicants = list(pref[c].values())
                    applicants = sorted(applicants, key=lambda id: -tb[id][rbd])
                    pref[c] = {p + 1: applicants[p] for p in range(len(applicants))}
            else:
                pref, tb = genin.modify_school_loterries(
                    pref, students, colleges, siblings, tie_breaker
                )
                if not os.path.exists(
                    outdir
                    + os.sep
                    + region
                    + os.sep
                    + "comparison"
                    + os.sep
                    + tie_breaker
                    + os.sep
                    + "tb_s="
                    + str(sim)
                    + ".pck"
                ):
                    with open(
                        outdir
                        + os.sep
                        + region
                        + os.sep
                        + "comparison"
                        + os.sep
                        + tie_breaker
                        + os.sep
                        + "tb_s="
                        + str(sim)
                        + ".pck",
                        "wb",
                    ) as pckf:
                        pickle.dump(tb, pckf)

            students, colleges, Tp, Tn, Sp, Sn = genin.create_additional_inputs_from_instance(
                pref, cap
            )

            for method in methods:
                output_dir = (
                    outdir
                    + os.sep
                    + region
                    + os.sep
                    + "comparison"
                    + os.sep
                    + method
                    + "_"
                    + tie_breaker
                )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                for extra in extras:
                    if "soft" not in method and extra != extras[0]:
                        continue

                    if extra is None:
                        outfile = output_dir + os.sep + "outputs_s=" + str(sim) + ".txt"
                    else:
                        outfile = (
                            output_dir
                            + os.sep
                            + "outputs_s="
                            + str(sim)
                            + "_k="
                            + str(extra)
                            + ".txt"
                        )

                    skip_sim = False
                    if os.path.exists(outfile):
                        if "# Objective" in open(outfile).read():
                            skip_sim = True

                    if skip_sim:
                        continue

                    outputs = {}
                    if method == "abs_hard":
                        outputs = opt.AbsoluteHard(
                            (
                                students,
                                colleges,
                                pref,
                                cap,
                                siblings,
                                levels,
                                students_per_level,
                                Tp,
                                Tn,
                                Sp,
                                Sn,
                                tb,
                            ),
                            "last_pref",
                            objective,
                        )
                    elif method == "abs_soft":
                        outputs = opt.AbsoluteSoft(
                            (
                                students,
                                colleges,
                                pref,
                                cap,
                                siblings,
                                levels,
                                students_per_level,
                                Tp,
                                Tn,
                                Sp,
                                Sn,
                                tb,
                            ),
                            "last_pref",
                            objective,
                            None,
                            extra,
                        )
                    elif method == "abs_ntb":
                        outputs = opt.AbsoluteHardNTB(
                            (
                                students,
                                colleges,
                                pref,
                                cap,
                                siblings,
                                levels,
                                students_per_level,
                                Tp,
                                Tn,
                                Sp,
                                Sn,
                                tb,
                            ),
                            "last_pref",
                            objective,
                        )
                    elif method == "par_hard":
                        outputs = opt.PartialHard(
                            (
                                students,
                                colleges,
                                pref,
                                cap,
                                siblings,
                                levels,
                                students_per_level,
                                Tp,
                                Tn,
                                Sp,
                                Sn,
                                tb,
                            ),
                            "last_pref",
                            objective,
                        )
                    elif method == "par_soft":
                        outputs = opt.PartialSoft(
                            (
                                students,
                                colleges,
                                pref,
                                cap,
                                siblings,
                                levels,
                                students_per_level,
                                Tp,
                                Tn,
                                Sp,
                                Sn,
                                tb,
                            ),
                            "last_pref",
                            objective,
                            None,
                            extra,
                        )
                    elif method == "desc":
                        outputs = alg.Sequential(
                            (students, colleges, pref, cap, siblings, levels, students_per_level),
                            [str(idx) for idx in sorted(range(-1, 13), reverse=True)],
                        )
                    elif method == "desc_ntb":
                        outputs = opt.DescendingNTB(
                            (
                                students,
                                colleges,
                                pref,
                                cap,
                                siblings,
                                levels,
                                students_per_level,
                                Tp,
                                Tn,
                                Sp,
                                Sn,
                                tb,
                            ),
                            "last_pref",
                            objective,
                        )
                    elif method == "asc":
                        outputs = alg.Sequential(
                            (students, colleges, pref, cap, siblings, levels, students_per_level),
                            [str(idx) for idx in sorted(range(-1, 13))],
                        )
                    elif method == "desc_block":
                        outputs = alg.SequentialBlock(
                            (students, colleges, pref, cap, siblings, levels, students_per_level),
                            [str(idx) for idx in sorted(range(-1, 13), reverse=True)],
                        )
                    elif method == "asc_block":
                        outputs = alg.SequentialBlock(
                            (students, colleges, pref, cap, siblings, levels, students_per_level),
                            [str(idx) for idx in sorted(range(-1, 13))],
                        )
                    elif method == "nosib":
                        match = alg.DA(students, pref, cap)
                        x_opt = {
                            id_s: {match[id_s]: 1} for id_s in match if match[id_s] is not None
                        }
                        outputs = {
                            "status": "completed",
                            "x_opt": x_opt,
                            "runtime": 0,
                            "num_vars": 0,
                            "num_cols": 0,
                            "mipgap": 0,
                            "nodes": 0,
                        }
                    elif method == "max_sib":
                        outputs = opt.MaxSiblings(
                            (
                                students,
                                colleges,
                                pref,
                                cap,
                                siblings,
                                levels,
                                students_per_level,
                                Tp,
                                Tn,
                                Sp,
                                Sn,
                                tb,
                            ),
                            penalty_unassigned="last_pref",
                            outdir=outdir,
                        )
                    elif method == "sim":
                        outputs = alg.Simultaneous(
                            (students, colleges, pref, cap, siblings),
                        )
                    elif method == "rada":
                        outputs = alg.Simultaneous(
                            (students, colleges, pref, cap, siblings), decay=1
                        )
                    elif method == "size_desc":
                        outputs = alg.SizeSequential(
                            (students, colleges, pref, cap, siblings), direction="decreasing"
                        )
                    elif method == "size_asc":
                        outputs = alg.SizeSequential(
                            (students, colleges, pref, cap, siblings), direction="increasing"
                        )
                    else:
                        print("***Error: unknown solving method.")
                        sys.exit(1)

                    if outputs != {}:
                        if outputs["status"] == "completed":
                            adds, obj = ComputeOutputs(outputs["x_opt"], students, pref, siblings)
                            outputs.update(adds)
                            if "obj" not in outputs:
                                outputs["obj"] = obj

                        WriteOutput(
                            outputs,
                            outfile,
                            (
                                students,
                                colleges,
                                pref,
                                cap,
                                siblings,
                                levels,
                                students_per_level,
                                tb,
                            ),
                        )


# --------------------------------
# Methods to simulate
# --------------------------------
def RunSimulations(
    indir,
    outdir,
    methods=["asc", "desc"],
    tie_breakers=["stb", "mtb"],
    regions=["Magallanes"],
    penalty_unassigned="last_pref",
    num_sims=10,
    num_cores=1,
    extras=None,
    objective="SOSM",
):

    if num_cores == 1:
        for sim in range(num_sims):
            print("Simulation:", sim)
            SubRoutine(
                (
                    indir,
                    outdir,
                    methods,
                    tie_breakers,
                    regions,
                    penalty_unassigned,
                    sim,
                    extras,
                    objective,
                )
            )
    else:
        outdata = []
        for sim in range(num_sims):
            outdata.append(
                (
                    indir,
                    outdir,
                    methods,
                    tie_breakers,
                    regions,
                    penalty_unassigned,
                    sim,
                    extras,
                    objective,
                )
            )

        if len(outdata) > 0:
            nproc = min(num_cores, multiprocessing.cpu_count())
            pool = multiprocessing.Pool(processes=min(nproc, len(outdata)))
            results = pool.map(SubRoutine, outdata)
            pool.close()
            pool.join()


def RunTests(
    indir,
    outdir,
    methods=["asc", "desc"],
    tie_breakers=["stb", "mtb"],
    penalty_unassigned="last_pref",
    num_sims=10,
    num_cores=1,
    extras=None,
):

    dir = outdir + os.sep + "tests"
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    if num_cores == 1:
        for sim in range(num_sims):
            print("Simulation:", sim)
            SubRoutine((indir, dir, methods, tie_breakers, penalty_unassigned, sim, extras))
    else:
        outdata = []
        for sim in range(num_sims):
            outdata.append((indir, dir, methods, tie_breakers, penalty_unassigned, sim, extras))

        if len(outdata) > 0:
            nproc = min(num_cores, multiprocessing.cpu_count())
            pool = multiprocessing.Pool(processes=min(nproc, len(outdata)))
            results = pool.map(SubRoutine, outdata)
            pool.close()
            pool.join()


if __name__ == "__main__":

    home_dir = os.path.expanduser("~")
    if "riosigna" in home_dir:
        dropbox_dir = home_dir + os.sep + "Dynamic"
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

    # Testing
    # region = "Atacama"
    # tie_breaker = "mtbf"
    # objective = "SOSM"
    # method = "abs_soft"
    # indir = dropbox_dir + os.sep + "Data"
    # outdir = dropbox_dir + os.sep + "outputs"

    # Sensitivity to tie breaking ----------- #
    # methods = ["abs_hard", "abs_soft", "max_sib", "nosib", "desc", "asc","par_hard", "par_soft"]
    # methods = ["abs_hard"]
    # tie_breakers = ["stb", "stbf", "mtb", "mtbf"]
    # regions = ["Magallanes"]
    # extras = None
    # objective = "SOSM"

    # Hybrid method ------------------------- # [running in screen -r hybrid]
    # methods = ["abs_soft"]
    # tie_breakers = ["mtbf", "mtb", "stb", "stbf"]
    # regions = ["Magallanes"]
    # extras = np.arange(280, 350 + 10, 10)
    # objective = "SOSM"

    # methods = ["abs_soft"]
    # tie_breakers = ["mtbf"]
    # regions = ["Atacama"]
    # # extras = np.arange(580, 670 + 10, 10)
    # objective = "SOSM"

    # methods = ["abs_soft"]
    # tie_breakers = ["mtbf"]
    # regions = ["OHiggins"]
    # extras = np.arange(1500, 1900 + 100, 100)
    # objective = "SOSM"

    # methods = ["abs_soft"]
    # tie_breakers = ["mtbf"]
    # regions = ["Lagos"]
    # # extras = np.arange(1200, 1600 + 100, 100)
    # extras = np.arange(1700, 1900 + 100, 100)
    # objective = "SOSM"

    # Additional - No Lotteries ------------- # [done]
    # methods = ["abs_ntb"]
    # tie_breakers = ["mtbf"]
    # regions = ["Magallanes"]
    # extras = None
    # objective="MXSM"

    # Sensitivity to regions ---------------- [done]
    # methods = ["abs_hard", "abs_soft", "max_sib", "nosib", "desc", "asc", "par_hard", "par_soft"]
    # regions = ["Atacama", "Lagos", "OHiggins"]
    # regions = ["Coquimbo"]
    # methods = ["abs_soft"]
    # tie_breakers = ["mtbf"]
    # regions = ["Atacama"]
    # extras = None
    # objective = "SOSM"

    # Extension: maximizing siblings together #
    # methods = ["abs_hard", "abs_soft", "desc_ntb"]
    # tie_breakers = ["mtbf"]
    # regions = ["Magallanes"]
    # extras = np.arange(280, 320 + 10, 10)
    # extras = np.append([0], extras)
    # objective = "ROSM"

    # methods = ["abs_ntb"]  # -- running in screen -r hard
    # tie_breakers = ["mtbf"]
    # regions = ["Magallanes"]
    # extras = None
    # objective = "ROSM"

    # methods = ["sim"]  # -- running local
    # tie_breakers = ["stb", "stbf", "mtb", "mtbf"]
    # regions = ["Magallanes"]
    # extras = None
    # objective = "SOSM"

    # RunSimulations(
    #     indir,
    #     outdir,
    #     methods,
    #     tie_breakers,
    #     regions,
    #     penalty_unassigned="last_pref",
    #     num_sims=100,
    #     extras=extras,
    #     num_cores=10,
    #     objective=objective,  # "SOSM", "ROSM", "MXSM"
    # )

    # methods = ["rada","size_desc", "size_asc"]  # -- running local
    methods = ["rada"]  # -- running local
    tie_breakers = ["stb", "stbf", "mtb", "mtbf"]
    regions = ["Magallanes"]
    extras = None
    objective = "SOSM"

    RunSimulations(
        indir,
        outdir,
        methods,
        tie_breakers,
        regions,
        penalty_unassigned="last_pref",
        num_sims=100,
        extras=extras,
        num_cores=20,
        objective=objective,  # "SOSM", "ROSM", "MXSM"
    )
