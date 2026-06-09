import sys, os, time, copy
from gurobipy import *

__gurobi_threads = 1
import numpy as np
import pickle, math, random

np.random.seed(1)
random.seed(1)
import copy
import generate_inputs as genin
import algorithms as alg
from itertools import combinations


def NoSiblings(inputs, penalty_unassigned="last_pref", outdir=None):

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn, tb = inputs

    # compute original position in list for objective and penalty
    opref = {}
    for s in pref:
        if s not in students:
            continue
        if pref[s] == {}:
            continue
        opref[s], idx = {}, 0
        for p in sorted(pref[s]):
            idx += 1
            opref[s][pref[s][p]] = idx

    model = Model("agg")
    model.setParam("OutputFlag", 1)
    model.setParam("OptimalityTol", 0.01)
    model.setParam("Method", 2)
    model.setParam("Threads", 1)
    # model.setParam('TimeLimit', 7200)

    x = {s: {} for s in students if pref[s] != {}}
    for s in x:
        for p in pref[s]:
            x[s][pref[s][p]] = model.addVar(
                vtype=GRB.BINARY,
                lb=0,
                ub=1,
                obj=opref[s][pref[s][p]],
                name="x" + "-" + s + "-" + pref[s][p],
            )

    # each student assigned to at most one school
    for s in x:
        model.addConstr(sum([x[s][c] for c in x[s]]) <= 1, "assignment_student_" + s)

    # each school receives at most capacity
    for c in colleges:
        model.addConstr(sum([x[s][c] for s in pref[c].values()]) <= cap[c], "capacity_college_" + c)

    # stability
    for s in x:
        for c in x[s]:
            model.addConstr(
                cap[c] * (1 - sum([x[s][j] for j in Tp[s][c]])) <= sum([x[i][c] for i in Sp[s][c]]),
                "stability_3b_" + s + "-" + c,
            )

    # adding penalty for unassignment
    penalty = model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=1, name="penalty_unassigned")
    if isinstance(penalty_unassigned, (int, float, complex)):
        model.addConstr(
            penalty_unassigned * sum([1 - sum([x[s][c] for c in x[s]]) for s in x]) <= penalty
        )
    elif penalty_unassigned == "last_pref":
        model.addConstr(
            sum([(max(opref[s].values()) + 1) * (1 - sum([x[s][c] for c in x[s]])) for s in x])
            + sum([(max(opref[s].values()) + 1) for s in students if s not in x])
            <= penalty
        )
    else:
        print("***ERROR: Unkonwn penalty")
        sys.exit(1)

    # Set objective
    model.Params.MIPGap = 1e-3
    model.Params.TimeLimit = 10000
    model.Params.MIPFocus = 1
    model.Params.Presolve = 2
    model.modelSense = GRB.MINIMIZE

    model.optimize()
    status = model.status

    if status == GRB.INF_OR_UNBD or status == GRB.INFEASIBLE or status == GRB.UNBOUNDED:
        print("The model cannot be solved because it is infeasible or unbounded")
        if outdir is not None:
            model.computeIIS()
            outfile = outdir + os.sep + "model.ilp"
            model.write(outfile)
        return {"status": "infeasible"}

    elif status != GRB.OPTIMAL:
        print("Optimization was stopped with status %d" % status)
        return {"status": "stopped"}
    else:
        obj = model.objVal
        x_opt = {}
        for v in model.getVars():
            if "penalty" in v.varName:
                continue
            if "x" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if v.x > 0:
                    if key_i not in x_opt:
                        x_opt[key_i] = {}
                    x_opt[key_i][key_j] = v.x
        outputs = {
            "status": "completed",
            "obj": obj,
            "x_opt": x_opt,
            "y_opt": {},
            "runtime": model.Runtime,
            "mipgap": model.MIPGap,
            "num_vars": model.NumVars,
            "num_cols": model.NumConstrs,
            "nodes": model.NodeCount,
        }
        return outputs


def MaxSiblings(inputs, penalty_unassigned="last_pref", outdir=None):

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn, tb = inputs
    families = genin.create_families(siblings)
    # compute original position in list for objective and penalty
    opref = {}
    for s in pref:
        if s not in students:
            continue
        if pref[s] == {}:
            continue
        opref[s], idx = {}, 0
        for p in sorted(pref[s]):
            idx += 1
            opref[s][pref[s][p]] = idx

    model = Model("agg")
    model.setParam("OutputFlag", 1)
    model.setParam("OptimalityTol", 0.01)
    model.setParam("Method", 2)
    model.setParam("Threads", 1)
    # model.setParam('TimeLimit', 7200)

    x, t = {s: {} for s in students if pref[s] != {}}, {f: {} for f in families}
    for s in x:
        for p in pref[s]:
            x[s][pref[s][p]] = model.addVar(
                vtype=GRB.BINARY, lb=0, ub=1, obj=1, name="x" + "-" + s + "-" + pref[s][p]
            )

    for f in families:
        for s in families[f]:
            for p in pref[s]:
                rbd = pref[s][p].split("_")[0]
                if rbd not in t[f]:
                    t[f][rbd] = model.addVar(
                        vtype=GRB.BINARY,
                        lb=0,
                        ub=1,
                        obj=-len(families[f]),
                        name="t" + "-" + str(f) + "-" + rbd,
                    )

    # each student assigned to at most one school
    for s in x:
        model.addConstr(sum([x[s][c] for c in x[s]]) <= 1, "assignment_student_" + s)

    # each school receives at most capacity
    for c in colleges:
        model.addConstr(sum([x[s][c] for s in pref[c].values()]) <= cap[c], "capacity_college_" + c)

    # stability
    for s in x:
        for c in x[s]:
            model.addConstr(
                cap[c] * (1 - sum([x[s][j] for j in Tp[s][c]])) <= sum([x[i][c] for i in Sp[s][c]]),
                "stability_3b_" + s + "-" + c,
            )

    # family definition
    for f in t:
        for rbd in t[f]:
            model.addConstr(
                t[f][rbd]
                <= sum([x[s][c] for s in families[f] for c in x[s] if c.split("_")[0] == rbd]),
                "family_1_" + str(f) + "-" + rbd,
            )
            model.addConstr(
                t[f][rbd]
                >= (1 / len(families[f]))
                * sum([x[s][c] for s in families[f] for c in x[s] if c.split("_")[0] == rbd]),
                "family_2_" + str(f) + "-" + rbd,
            )

    # adding penalty for unassignment
    penalty = model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=1, name="penalty_unassigned")
    if isinstance(penalty_unassigned, (int, float, complex)):
        model.addConstr(
            penalty_unassigned * sum([1 - sum([x[s][c] for c in x[s]]) for s in x]) <= penalty
        )
    elif penalty_unassigned == "last_pref":
        model.addConstr(
            sum([(max(opref[s].values()) + 1) * (1 - sum([x[s][c] for c in x[s]])) for s in x])
            + sum([(max(opref[s].values()) + 1) for s in students if s not in x])
            <= penalty
        )
    else:
        print("***ERROR: Unkonwn penalty")
        sys.exit(1)

    # Set objective
    model.Params.MIPGap = 1e-4
    model.Params.TimeLimit = 10000
    model.Params.MIPFocus = 1
    model.Params.Presolve = 2
    model.modelSense = GRB.MINIMIZE

    model.optimize()
    status = model.status

    if status == GRB.INF_OR_UNBD or status == GRB.INFEASIBLE or status == GRB.UNBOUNDED:
        print("The model cannot be solved because it is infeasible or unbounded")
        if outdir is not None:
            model.computeIIS()
            outfile = outdir + os.sep + "model.ilp"
            model.write(outfile)
        return {"status": "infeasible"}

    elif status != GRB.OPTIMAL:
        print("Optimization was stopped with status %d" % status)
        return {"status": "stopped"}
    else:
        obj = model.objVal
        x_opt = {}
        for v in model.getVars():
            if "penalty" in v.varName:
                continue
            if "x" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if v.x > 0:
                    if key_i not in x_opt:
                        x_opt[key_i] = {}
                    x_opt[key_i][key_j] = v.x
        outputs = {
            "status": "completed",
            "obj": obj,
            "x_opt": x_opt,
            "y_opt": {},
            "runtime": model.Runtime,
            "mipgap": model.MIPGap,
            "num_vars": model.NumVars,
            "num_cols": model.NumConstrs,
            "nodes": model.NodeCount,
        }
        return outputs


# -----------------------
# New implementations
# -----------------------
def AbsoluteHard(inputs, penalty_unassigned="last_pref", objective="SOSM", outdir=None):

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn, tb = inputs
    level = {s: lev for lev in students_per_level for s in students_per_level[lev]}
    # compute original position in list for objective and penalty
    opref = {}
    for s in pref:
        if s not in students:
            continue
        if pref[s] == {}:
            continue
        opref[s], idx = {}, 0
        for p in sorted(pref[s]):
            idx += 1
            opref[s][pref[s][p]] = idx

    model = Model("agg")

    siblings_in_school = {
        s: {
            pref[s][p].split("_")[0]: [
                sib
                for sib in siblings[s]
                if pref[s][p].split("_")[0] in [cp.split("_")[0] for cp in pref[sib].values()]
            ]
            for p in pref[s]
        }
        for s in siblings
    }

    x, y, z = (
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
    )
    # NOTE: x is defined at the course level (i.e., school + level); y,z are defined at the school level
    for s in x:
        for p in pref[s]:
            x[s][pref[s][p]] = model.addVar(
                vtype=GRB.BINARY,
                lb=0,
                ub=1,
                obj=opref[s][pref[s][p]] if objective in ["SOSM", "MXSM"] else 0,
                name="x" + "-" + s + "-" + pref[s][p],
            )
            rbd = pref[s][p].split("_")[0]
            if len(siblings_in_school[s][rbd]) > 0:
                z[s][rbd] = model.addVar(
                    vtype=GRB.BINARY, lb=0, ub=1, name="z" + "-" + s + "-" + rbd
                )
                for sib in siblings_in_school[s][rbd]:
                    if sib not in y[s]:
                        y[s][sib] = {}
                    y[s][sib][rbd] = model.addVar(
                        vtype=GRB.BINARY,
                        lb=0,
                        ub=1,
                        obj=-1 if objective in ["MXSM", "ROSM"] else 0,
                        name="y" + "-" + s + "-" + sib + "-" + rbd,
                    )

    # each student assigned to at most one school
    for s in x:
        model.addConstr(sum([x[s][c] for c in x[s]]) <= 1, "assignment_student_" + s)

    # each school receives at most capacity
    for c in colleges:
        model.addConstr(sum([x[s][c] for s in pref[c].values()]) <= cap[c], "capacity_college_" + c)

    # definition of providing priority
    for s in z:
        for rbd in z[s]:
            model.addConstr(
                z[s][rbd] <= sum([x[s][c] for c in x[s] if c.split("_")[0] == rbd]),
                "definition_z_1_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd] + sum(z[sib][rbd] for sib in siblings_in_school[s][rbd]) <= 1,
                "definition_z_3_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd]
                <= sum(
                    [
                        x[sib][c]
                        for sib in siblings_in_school[s][rbd]
                        for c in x[sib]
                        if c.split("_")[0] == rbd
                    ]
                ),
                "definition_z_4_" + s + "_" + rbd,
            )
        for c in x[s]:
            rbd = c.split("_")[0]
            if rbd in z[s]:
                model.addConstr(
                    len(Sp[s][c]) - sum([x[i][j] for i in Sp[s][c] for j in Tp[i][c] if j != c])
                    <= (cap[c] - 1) + len(students_per_level[level[s]]) * (1 - z[s][rbd]),
                    "definition_z_2_" + s + "_" + rbd,
                )

    # definition of receiving priority
    for s in y:
        for sib in y[s]:
            for rbd in y[s][sib]:
                model.addConstr(
                    y[s][sib][rbd] <= z[s][rbd], "6a-" + s + "-" + sib + "-" + rbd
                )  # cannot receive priority from a non-provider
                model.addConstr(
                    y[s][sib][rbd] <= 1 - z[sib][rbd], "6b-" + s + "-" + sib + "-" + rbd
                )  # cannot be provider and receiver of priority
                model.addConstr(
                    y[s][sib][rbd] <= sum([x[sib][c] for c in x[sib] if c.split("_")[0] == rbd]),
                    "6c-" + s + "-" + rbd,
                )  # receives siblings priority if gets assigned to school

    # stability
    for s in x:
        for c in x[s]:
            rbd = c.split("_")[0]
            model.addConstr(
                cap[c] * (1 - sum([x[s][j] for j in Tp[s][c]]))
                <= sum([x[i][c] for i in Sp[s][c]])
                + sum([y[i][j][rbd] for j in Sn[s][c] for i in siblings_in_school[j][rbd]])
                + sum([z[i][rbd] for i in z if rbd in z[i] and i in Sn[s][c]]),
                "stability_1_" + s + "-" + c,
            )

            for sib in siblings_in_school[s][rbd]:
                for l in pref[c].values():
                    if l in siblings_in_school[s][rbd] or l == s:
                        continue
                    if siblings_in_school[l][rbd] != []:
                        model.addConstr(
                            sum([x[sib][c] for c in x[sib] if c.split("_")[0] == rbd])
                            + (1 - sum([x[s][j] for j in Tp[s][c]]))
                            <= 2
                            - x[l][c]
                            + (z[l][rbd] + sum([y[k][l][rbd] for k in siblings_in_school[l][rbd]]))
                            * (tb[l][rbd] > tb[s][rbd]),
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )
                    else:
                        model.addConstr(
                            sum([x[sib][c] for c in x[sib] if c.split("_")[0] == rbd])
                            + (1 - sum([x[s][j] for j in Tp[s][c]]))
                            <= 2 - x[l][c],
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )

    # adding penalty for unassignment
    penalty = model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=1, name="penalty_unassigned")
    if isinstance(penalty_unassigned, (int, float, complex)):
        model.addConstr(
            penalty_unassigned * sum([1 - sum([x[s][c] for c in x[s]]) for s in x]) <= penalty
        )
    elif penalty_unassigned == "last_pref":
        model.addConstr(
            sum([(max(opref[s].values()) + 1) * (1 - sum([x[s][c] for c in x[s]])) for s in x])
            + sum([(max(opref[s].values()) + 1) for s in students if s not in x])
            <= penalty
        )
    else:
        print("***ERROR: Unkonwn penalty")
        sys.exit(1)

    # Set objective
    model.Params.OutputFlag = 1
    model.Params.Method = 3
    model.Params.Threads = 1
    model.Params.MIPFocus = 2
    model.Params.Presolve = 2
    model.Params.Cuts = 3
    model.Params.VarBranch = 1
    model.modelSense = GRB.MINIMIZE

    if objective in ["SOSM"]:
        model.Params.TimeLimit = 10000
        model.Params.MIPGap = 1e-2
    elif objective in ["MXSM", "ROSM"]:
        model.Params.TimeLimit = 36000
        model.Params.MIPGap = 1e-2
    else:
        print("***ERROR: Unrecognized objective function")
        sys.exit(1)

    # model.Params.PoolSearchMode = 2
    # model.Params.PoolSolutions = 10

    model.optimize()
    status = model.status

    if status == GRB.INF_OR_UNBD or status == GRB.INFEASIBLE or status == GRB.UNBOUNDED:
        print("The model cannot be solved because it is infeasible or unbounded")
        if outdir is not None:
            model.computeIIS()
            outfile = outdir + os.sep + "model.ilp"
            model.write(outfile)
        return {"status": "infeasible"}

    elif status != GRB.OPTIMAL:
        print("Optimization was stopped with status %d" % status)
        return {"status": "stopped"}
    else:
        obj = model.objVal
        x_opt, y_opt, z_opt = {}, {}, {}
        for v in model.getVars():
            if "penalty" in v.varName:
                continue
            if "x" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if v.x > 0:
                    if key_i not in x_opt:
                        x_opt[key_i] = {}
                    x_opt[key_i][key_j] = v.x
            if "y" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                key_k = v.varName[2:].split("-")[2]
                if v.x > 0:
                    if key_i not in y_opt:
                        y_opt[key_i] = {}
                    if key_j not in y_opt[key_i]:
                        y_opt[key_i][key_j] = {}
                    y_opt[key_i][key_j][key_k] = v.x
            if "z" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if v.x > 0:
                    if key_i not in z_opt:
                        z_opt[key_i] = {}
                    z_opt[key_i][key_j] = v.x

        outputs = {
            "status": "completed",
            "obj": obj,
            "x_opt": x_opt,
            "y_opt": y_opt,
            "z_opt": z_opt,
            "runtime": model.Runtime,
            "mipgap": model.MIPGap,
            "num_vars": model.NumVars,
            "num_cols": model.NumConstrs,
            "nodes": model.NodeCount,
        }
        return outputs


def AbsoluteSoft(
    inputs, penalty_unassigned="last_pref", objective="SOSM", outdir=None, control=None
):

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn, tb = inputs
    level = {s: lev for lev in students_per_level for s in students_per_level[lev]}
    # compute original position in list for objective and penalty
    opref = {}
    for s in pref:
        if s not in students:
            continue
        if pref[s] == {}:
            continue
        opref[s], idx = {}, 0
        for p in sorted(pref[s]):
            idx += 1
            opref[s][pref[s][p]] = idx

    model = Model("agg")

    siblings_in_school = {
        s: {
            pref[s][p].split("_")[0]: [
                sib
                for sib in siblings[s]
                if pref[s][p].split("_")[0] in [cp.split("_")[0] for cp in pref[sib].values()]
            ]
            for p in pref[s]
        }
        for s in siblings
    }

    x, y, z = (
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
    )
    # NOTE: x is defined at the course level (i.e., school + level); y,z are defined at the school level
    for s in x:
        for p in pref[s]:
            x[s][pref[s][p]] = model.addVar(
                vtype=GRB.BINARY,
                lb=0,
                ub=1,
                obj=opref[s][pref[s][p]] if objective in ["SOSM", "MXSM"] else 0,
                name="x" + "-" + s + "-" + pref[s][p],
            )
            rbd = pref[s][p].split("_")[0]
            if len(siblings_in_school[s][rbd]) > 0:
                z[s][rbd] = model.addVar(
                    vtype=GRB.BINARY, lb=0, ub=1, name="z" + "-" + s + "-" + rbd
                )
                for sib in siblings_in_school[s][rbd]:
                    if sib not in y[s]:
                        y[s][sib] = {}
                    y[s][sib][rbd] = model.addVar(
                        vtype=GRB.BINARY,
                        lb=0,
                        ub=1,
                        obj=-1 if objective in ["MXSM", "ROSM"] else 0,
                        name="y" + "-" + s + "-" + sib + "-" + rbd,
                    )

    # each student assigned to at most one school
    for s in x:
        model.addConstr(sum([x[s][c] for c in x[s]]) <= 1, "assignment_student_" + s)

    # each school receives at most capacity
    for c in colleges:
        model.addConstr(sum([x[s][c] for s in pref[c].values()]) <= cap[c], "capacity_college_" + c)

    # definition of providing priority
    for s in z:
        for rbd in z[s]:
            model.addConstr(
                z[s][rbd] <= sum([x[s][c] for c in x[s] if c.split("_")[0] == rbd]),
                "definition_z_1_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd] + sum(z[sib][rbd] for sib in siblings_in_school[s][rbd]) <= 1,
                "definition_z_3_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd]
                <= sum(
                    [
                        x[sib][c]
                        for sib in siblings_in_school[s][rbd]
                        for c in x[sib]
                        if c.split("_")[0] == rbd
                    ]
                ),
                "definition_z_4_" + s + "_" + rbd,
            )
        for c in x[s]:
            rbd = c.split("_")[0]
            if rbd in z[s]:
                model.addConstr(
                    len(Sp[s][c]) - sum([x[i][j] for i in Sp[s][c] for j in Tp[i][c] if j != c])
                    <= (cap[c] - 1) + len(students_per_level[level[s]]) * (1 - z[s][rbd]),
                    "definition_z_2_" + s + "_" + rbd,
                )

    # definition of receiving priority
    for s in y:
        for sib in y[s]:
            for rbd in y[s][sib]:
                model.addConstr(
                    y[s][sib][rbd] <= z[s][rbd], "6a-" + s + "-" + sib + "-" + rbd
                )  # cannot receive priority from a non-provider
                model.addConstr(
                    y[s][sib][rbd] <= 1 - z[sib][rbd], "6b-" + s + "-" + sib + "-" + rbd
                )  # cannot be provider and receiver of priority
                model.addConstr(
                    y[s][sib][rbd] <= sum([x[sib][c] for c in x[sib] if c.split("_")[0] == rbd]),
                    "6c-" + s + "-" + rbd,
                )  # receives siblings priority if gets assigned to school

    # stability
    for s in x:
        for c in x[s]:
            rbd = c.split("_")[0]
            model.addConstr(
                cap[c] * (1 - sum([x[s][j] for j in Tp[s][c]]))
                <= sum([x[i][c] for i in Sp[s][c]])
                + sum([y[i][j][rbd] for j in Sn[s][c] for i in siblings_in_school[j][rbd]])
                + sum([z[i][rbd] for i in z if rbd in z[i] and i in Sn[s][c]]),
                "stability_1_" + s + "-" + c,
            )

            for sib in siblings_in_school[s][rbd]:
                for l in pref[c].values():
                    if l in siblings_in_school[s][rbd] or l == s:
                        continue
                    if siblings_in_school[l][rbd] != []:
                        model.addConstr(
                            z[sib][rbd] + (1 - sum([x[s][j] for j in Tp[s][c]]))
                            <= 2
                            - x[l][c]
                            + (z[l][rbd] + sum([y[k][l][rbd] for k in siblings_in_school[l][rbd]]))
                            * (tb[l][rbd] > tb[s][rbd]),
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )
                    else:
                        model.addConstr(
                            z[sib][rbd] + (1 - sum([x[s][j] for j in Tp[s][c]])) <= 2 - x[l][c],
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )

    if control is not None:
        model.addConstr(sum([z[s][rbd] for s in z for rbd in z[s]]) >= control)
        # for s in z:
        #     for rbd in z[s]:
        #         model.addConstr(z[s][rbd] <= sum([y[s][sib][rbd] for sib in siblings_in_school[s][rbd]]) )

    # adding penalty for unassignment
    penalty = model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=1, name="penalty_unassigned")
    if isinstance(penalty_unassigned, (int, float, complex)):
        model.addConstr(
            penalty_unassigned * sum([1 - sum([x[s][c] for c in x[s]]) for s in x]) <= penalty
        )
    elif penalty_unassigned == "last_pref":
        model.addConstr(
            sum([(max(opref[s].values()) + 1) * (1 - sum([x[s][c] for c in x[s]])) for s in x])
            + sum([(max(opref[s].values()) + 1) for s in students if s not in x])
            <= penalty
        )
    else:
        print("***ERROR: Unkonwn penalty")
        sys.exit(1)

    # Set objective
    model.Params.OutputFlag = 1
    model.Params.Method = 3
    model.Params.Threads = 1
    model.Params.MIPFocus = 2
    model.Params.Presolve = 2
    model.Params.Cuts = 3
    model.Params.VarBranch = 1
    model.Params.FeasibilityTol = 1e-4
    model.modelSense = GRB.MINIMIZE

    if objective in ["SOSM"]:
        model.Params.TimeLimit = 10000
        model.Params.MIPGap = 1e-2
    elif objective in ["MXSM", "ROSM"]:
        model.Params.TimeLimit = 36000
        model.Params.MIPGap = 1e-2
    else:
        print("***ERROR: Unrecognized objective function")
        sys.exit(1)

    # model.Params.PoolSearchMode = 2
    # model.Params.PoolSolutions = 10

    model.optimize()
    status = model.status

    if status == GRB.INF_OR_UNBD or status == GRB.INFEASIBLE or status == GRB.UNBOUNDED:
        print("The model cannot be solved because it is infeasible or unbounded")
        if outdir is not None:
            model.computeIIS()
            outfile = outdir + os.sep + "model.ilp"
            model.write(outfile)
        return {"status": "infeasible"}

    elif status != GRB.OPTIMAL:
        print("Optimization was stopped with status %d" % status)
        return {"status": "stopped"}
    else:
        obj = model.objVal
        x_opt, y_opt, z_opt = {}, {}, {}
        for v in model.getVars():
            if "penalty" in v.varName:
                continue
            if "x" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if v.x > 0:
                    if key_i not in x_opt:
                        x_opt[key_i] = {}
                    x_opt[key_i][key_j] = v.x
            if "y" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                key_k = v.varName[2:].split("-")[2]
                if v.x > 0:
                    if key_i not in y_opt:
                        y_opt[key_i] = {}
                    if key_j not in y_opt[key_i]:
                        y_opt[key_i][key_j] = {}
                    y_opt[key_i][key_j][key_k] = v.x
            if "z" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if v.x > 0:
                    if key_i not in z_opt:
                        z_opt[key_i] = {}
                    z_opt[key_i][key_j] = v.x

        outputs = {
            "status": "completed",
            "obj": obj,
            "x_opt": x_opt,
            "y_opt": y_opt,
            "z_opt": z_opt,
            "runtime": model.Runtime,
            "mipgap": model.MIPGap,
            "num_vars": model.NumVars,
            "num_cols": model.NumConstrs,
            "nodes": model.NodeCount,
        }
        return outputs


def PartialHard(inputs, penalty_unassigned="last_pref", objective="SOSM", outdir=None):

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn, tb = inputs
    level = {s: lev for lev in students_per_level for s in students_per_level[lev]}
    # compute original position in list for objective and penalty
    opref = {}
    for s in pref:
        if s not in students:
            continue
        if pref[s] == {}:
            continue
        opref[s], idx = {}, 0
        for p in sorted(pref[s]):
            idx += 1
            opref[s][pref[s][p]] = idx

    model = Model("agg")

    siblings_in_school = {
        s: {
            pref[s][p].split("_")[0]: [
                sib
                for sib in siblings[s]
                if pref[s][p].split("_")[0] in [cp.split("_")[0] for cp in pref[sib].values()]
            ]
            for p in pref[s]
        }
        for s in siblings
    }

    x, y, z = (
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
    )
    # NOTE: x is defined at the course level (i.e., school + level); y,z are defined at the school level
    for s in x:
        for p in pref[s]:
            x[s][pref[s][p]] = model.addVar(
                vtype=GRB.BINARY,
                lb=0,
                ub=1,
                obj=opref[s][pref[s][p]] if objective in ["SOSM", "MXSM"] else 0,
                name="x" + "-" + s + "-" + pref[s][p],
            )
            rbd = pref[s][p].split("_")[0]
            if len(siblings_in_school[s][rbd]) > 0:
                z[s][rbd] = model.addVar(
                    vtype=GRB.BINARY, lb=0, ub=1, name="z" + "-" + s + "-" + rbd
                )
                for sib in siblings_in_school[s][rbd]:
                    if sib not in y[s]:
                        y[s][sib] = {}
                    y[s][sib][rbd] = model.addVar(
                        vtype=GRB.BINARY,
                        lb=0,
                        ub=1,
                        obj=-1 if objective in ["MXSM", "ROSM"] else 0,
                        name="y" + "-" + s + "-" + sib + "-" + rbd,
                    )

    # each student assigned to at most one school
    for s in x:
        model.addConstr(sum([x[s][c] for c in x[s]]) <= 1, "assignment_student_" + s)

    # each school receives at most capacity
    for c in colleges:
        model.addConstr(sum([x[s][c] for s in pref[c].values()]) <= cap[c], "capacity_college_" + c)

    # definition of providing priority
    for s in z:
        for rbd in z[s]:
            model.addConstr(
                z[s][rbd] <= sum([x[s][c] for c in x[s] if c.split("_")[0] == rbd]),
                "definition_z_1_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd] + sum(z[sib][rbd] for sib in siblings_in_school[s][rbd]) <= 1,
                "definition_z_3_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd]
                <= sum(
                    [
                        x[sib][c]
                        for sib in siblings_in_school[s][rbd]
                        for c in x[sib]
                        if c.split("_")[0] == rbd
                    ]
                ),
                "definition_z_4_" + s + "_" + rbd,
            )
        for c in x[s]:
            rbd = c.split("_")[0]
            if rbd in z[s]:
                model.addConstr(
                    len(Sp[s][c]) - sum([x[i][j] for i in Sp[s][c] for j in Tp[i][c] if j != c])
                    <= (cap[c] - 1) + len(students_per_level[level[s]]) * (1 - z[s][rbd]),
                    "definition_z_2_" + s + "_" + rbd,
                )

    # definition of receiving priority
    for s in y:
        for sib in y[s]:
            for rbd in y[s][sib]:
                model.addConstr(
                    y[s][sib][rbd] <= z[s][rbd], "6a-" + s + "-" + sib + "-" + rbd
                )  # cannot receive priority from a non-provider
                model.addConstr(
                    y[s][sib][rbd] <= 1 - z[sib][rbd], "6b-" + s + "-" + sib + "-" + rbd
                )  # cannot be provider and receiver of priority
                model.addConstr(
                    y[s][sib][rbd] <= sum([x[sib][c] for c in x[sib] if c.split("_")[0] == rbd]),
                    "6c-" + s + "-" + rbd,
                )  # receives siblings priority if gets assigned to school

    # stability
    for s in x:
        for c in x[s]:
            rbd = c.split("_")[0]
            model.addConstr(
                cap[c] * (1 - sum([x[s][j] for j in Tp[s][c]]))
                <= sum([x[i][c] for i in Sp[s][c]])
                + sum(
                    [
                        y[i][j][rbd]
                        for j in Sn[s][c]
                        for i in siblings_in_school[j][rbd]
                        if tb[i][rbd] > tb[s][rbd]
                    ]
                ),
                "stability_1_" + s + "-" + c,
            )

            for sib in siblings_in_school[s][rbd]:
                for l in pref[c].values():
                    if l in siblings_in_school[s][rbd] or l == s:
                        continue
                    if siblings_in_school[l][rbd] != []:
                        model.addConstr(
                            sum([x[sib][c] for c in x[sib] if c.split("_")[0] == rbd])
                            + (1 - sum([x[s][j] for j in Tp[s][c]]))
                            <= 2
                            - x[l][c] * int(tb[l][rbd] < max(tb[s][rbd], tb[sib][rbd]))
                            + sum(
                                [
                                    y[k][l][rbd]
                                    * (max(tb[l][rbd], tb[k][rbd]) > max(tb[s][rbd], tb[sib][rbd]))
                                    for k in siblings_in_school[l][rbd]
                                ]
                            ),
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )
                    else:
                        model.addConstr(
                            sum([x[sib][c] for c in x[sib] if c.split("_")[0] == rbd])
                            + (1 - sum([x[s][j] for j in Tp[s][c]]))
                            <= 2 - x[l][c] * int(tb[l][rbd] < max(tb[s][rbd], tb[sib][rbd])),
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )

    # adding penalty for unassignment
    penalty = model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=1, name="penalty_unassigned")
    if isinstance(penalty_unassigned, (int, float, complex)):
        model.addConstr(
            penalty_unassigned * sum([1 - sum([x[s][c] for c in x[s]]) for s in x]) <= penalty
        )
    elif penalty_unassigned == "last_pref":
        model.addConstr(
            sum([(max(opref[s].values()) + 1) * (1 - sum([x[s][c] for c in x[s]])) for s in x])
            + sum([(max(opref[s].values()) + 1) for s in students if s not in x])
            <= penalty
        )
    else:
        print("***ERROR: Unkonwn penalty")
        sys.exit(1)

    # Set objective
    model.Params.OutputFlag = 1
    model.Params.Method = 3
    model.Params.Threads = 1
    model.Params.MIPFocus = 2
    model.Params.Presolve = 2
    model.Params.Cuts = 3
    model.Params.VarBranch = 1
    model.modelSense = GRB.MINIMIZE

    if objective in ["SOSM"]:
        model.Params.TimeLimit = 10000
        model.Params.MIPGap = 1e-2
    elif objective in ["MXSM", "ROSM"]:
        model.Params.TimeLimit = 36000
        model.Params.MIPGap = 1e-2
    else:
        print("***ERROR: Unrecognized objective function")
        sys.exit(1)

    # model.Params.PoolSearchMode = 2
    # model.Params.PoolSolutions = 10

    model.optimize()
    status = model.status

    if status == GRB.INF_OR_UNBD or status == GRB.INFEASIBLE or status == GRB.UNBOUNDED:
        print("The model cannot be solved because it is infeasible or unbounded")
        if outdir is not None:
            model.computeIIS()
            outfile = outdir + os.sep + "model.ilp"
            model.write(outfile)
        return {"status": "infeasible"}

    elif status != GRB.OPTIMAL:
        print("Optimization was stopped with status %d" % status)
        return {"status": "stopped"}
    else:
        obj = model.objVal
        x_opt, y_opt, z_opt = {}, {}, {}
        for v in model.getVars():
            if "penalty" in v.varName:
                continue
            if "x" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if v.x > 0:
                    if key_i not in x_opt:
                        x_opt[key_i] = {}
                    x_opt[key_i][key_j] = v.x
            if "y" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                key_k = v.varName[2:].split("-")[2]
                if v.x > 0:
                    if key_i not in y_opt:
                        y_opt[key_i] = {}
                    if key_j not in y_opt[key_i]:
                        y_opt[key_i][key_j] = {}
                    y_opt[key_i][key_j][key_k] = v.x
            if "z" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if v.x > 0:
                    if key_i not in z_opt:
                        z_opt[key_i] = {}
                    z_opt[key_i][key_j] = v.x

        outputs = {
            "status": "completed",
            "obj": obj,
            "x_opt": x_opt,
            "y_opt": y_opt,
            "z_opt": z_opt,
            "runtime": model.Runtime,
            "mipgap": model.MIPGap,
            "num_vars": model.NumVars,
            "num_cols": model.NumConstrs,
            "nodes": model.NodeCount,
        }
        return outputs


def PartialSoft(
    inputs, penalty_unassigned="last_pref", objective="SOSM", outdir=None, control=None
):

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn, tb = inputs
    level = {s: lev for lev in students_per_level for s in students_per_level[lev]}
    # compute original position in list for objective and penalty
    opref = {}
    for s in pref:
        if s not in students:
            continue
        if pref[s] == {}:
            continue
        opref[s], idx = {}, 0
        for p in sorted(pref[s]):
            idx += 1
            opref[s][pref[s][p]] = idx

    model = Model("agg")

    siblings_in_school = {
        s: {
            pref[s][p].split("_")[0]: [
                sib
                for sib in siblings[s]
                if pref[s][p].split("_")[0] in [cp.split("_")[0] for cp in pref[sib].values()]
            ]
            for p in pref[s]
        }
        for s in siblings
    }

    x, y, z = (
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
    )
    # NOTE: x is defined at the course level (i.e., school + level); y,z are defined at the school level
    for s in x:
        for p in pref[s]:
            x[s][pref[s][p]] = model.addVar(
                vtype=GRB.BINARY,
                lb=0,
                ub=1,
                obj=opref[s][pref[s][p]] if objective in ["SOSM", "MXSM"] else 0,
                name="x" + "-" + s + "-" + pref[s][p],
            )
            rbd = pref[s][p].split("_")[0]
            if len(siblings_in_school[s][rbd]) > 0:
                z[s][rbd] = model.addVar(
                    vtype=GRB.BINARY, lb=0, ub=1, name="z" + "-" + s + "-" + rbd
                )
                for sib in siblings_in_school[s][rbd]:
                    if sib not in y[s]:
                        y[s][sib] = {}
                    y[s][sib][rbd] = model.addVar(
                        vtype=GRB.BINARY,
                        lb=0,
                        ub=1,
                        obj=-1 if objective in ["MXSM", "ROSM"] else 0,
                        name="y" + "-" + s + "-" + sib + "-" + rbd,
                    )

    # each student assigned to at most one school
    for s in x:
        model.addConstr(sum([x[s][c] for c in x[s]]) <= 1, "assignment_student_" + s)

    # each school receives at most capacity
    for c in colleges:
        model.addConstr(sum([x[s][c] for s in pref[c].values()]) <= cap[c], "capacity_college_" + c)

    # definition of providing priority
    for s in z:
        for rbd in z[s]:
            model.addConstr(
                z[s][rbd] <= sum([x[s][c] for c in x[s] if c.split("_")[0] == rbd]),
                "definition_z_1_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd] + sum(z[sib][rbd] for sib in siblings_in_school[s][rbd]) <= 1,
                "definition_z_3_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd]
                <= sum(
                    [
                        x[sib][c]
                        for sib in siblings_in_school[s][rbd]
                        for c in x[sib]
                        if c.split("_")[0] == rbd
                    ]
                ),
                "definition_z_4_" + s + "_" + rbd,
            )
        for c in x[s]:
            rbd = c.split("_")[0]
            if rbd in z[s]:
                model.addConstr(
                    len(Sp[s][c]) - sum([x[i][j] for i in Sp[s][c] for j in Tp[i][c] if j != c])
                    <= (cap[c] - 1) + len(students_per_level[level[s]]) * (1 - z[s][rbd]),
                    "definition_z_2_" + s + "_" + rbd,
                )

    # definition of receiving priority
    for s in y:
        for sib in y[s]:
            for rbd in y[s][sib]:
                model.addConstr(
                    y[s][sib][rbd] <= z[s][rbd], "6a-" + s + "-" + sib + "-" + rbd
                )  # cannot receive priority from a non-provider
                model.addConstr(
                    y[s][sib][rbd] <= 1 - z[sib][rbd], "6b-" + s + "-" + sib + "-" + rbd
                )  # cannot be provider and receiver of priority
                model.addConstr(
                    y[s][sib][rbd] <= sum([x[sib][c] for c in x[sib] if c.split("_")[0] == rbd]),
                    "6c-" + s + "-" + rbd,
                )  # receives siblings priority if gets assigned to school

    # stability
    for s in x:
        for c in x[s]:
            rbd = c.split("_")[0]
            model.addConstr(
                cap[c] * (1 - sum([x[s][j] for j in Tp[s][c]]))
                <= sum([x[i][c] for i in Sp[s][c]])
                + sum(
                    [
                        y[i][j][rbd]
                        for j in Sn[s][c]
                        for i in siblings_in_school[j][rbd]
                        if tb[i][rbd] > tb[s][rbd]
                    ]
                ),
                "stability_1_" + s + "-" + c,
            )

            for sib in siblings_in_school[s][rbd]:
                for l in pref[c].values():
                    if l in siblings_in_school[s][rbd] or l == s:
                        continue
                    if siblings_in_school[l][rbd] != []:
                        model.addConstr(
                            z[sib][rbd] + (1 - sum([x[s][j] for j in Tp[s][c]]))
                            <= 2
                            - x[l][c] * int(tb[l][rbd] < max(tb[s][rbd], tb[sib][rbd]))
                            + sum(
                                [
                                    y[k][l][rbd]
                                    * (max(tb[l][rbd], tb[k][rbd]) > max(tb[s][rbd], tb[sib][rbd]))
                                    for k in siblings_in_school[l][rbd]
                                ]
                            ),
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )
                    else:
                        model.addConstr(
                            z[sib][rbd] + (1 - sum([x[s][j] for j in Tp[s][c]]))
                            <= 2 - x[l][c] * int(tb[l][rbd] < max(tb[s][rbd], tb[sib][rbd])),
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )

    if control is not None:
        model.addConstr(sum([z[s][rbd] for s in z for rbd in z[s]]) >= control)
        # for s in z:
        #     for rbd in z[s]:
        #         model.addConstr(z[s][rbd] <= sum([y[s][sib][rbd] for sib in siblings_in_school[s][rbd]]) )

    # adding penalty for unassignment
    penalty = model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=1, name="penalty_unassigned")
    if isinstance(penalty_unassigned, (int, float, complex)):
        model.addConstr(
            penalty_unassigned * sum([1 - sum([x[s][c] for c in x[s]]) for s in x]) <= penalty
        )
    elif penalty_unassigned == "last_pref":
        model.addConstr(
            sum([(max(opref[s].values()) + 1) * (1 - sum([x[s][c] for c in x[s]])) for s in x])
            + sum([(max(opref[s].values()) + 1) for s in students if s not in x])
            <= penalty
        )
    else:
        print("***ERROR: Unkonwn penalty")
        sys.exit(1)

    # Set objective
    model.Params.OutputFlag = 1
    model.Params.Method = 3
    model.Params.Threads = 1
    model.Params.MIPFocus = 2
    model.Params.Presolve = 2
    model.Params.Cuts = 3
    model.Params.VarBranch = 1
    model.modelSense = GRB.MINIMIZE

    if objective in ["SOSM"]:
        model.Params.TimeLimit = 10000
        model.Params.MIPGap = 1e-2
    elif objective in ["MXSM", "ROSM"]:
        model.Params.TimeLimit = 36000
        model.Params.MIPGap = 1e-2
    else:
        print("***ERROR: Unrecognized objective function")
        sys.exit(1)

    # model.Params.PoolSearchMode = 2
    # model.Params.PoolSolutions = 10

    model.optimize()
    status = model.status

    if status == GRB.INF_OR_UNBD or status == GRB.INFEASIBLE or status == GRB.UNBOUNDED:
        print("The model cannot be solved because it is infeasible or unbounded")
        if outdir is not None:
            model.computeIIS()
            outfile = outdir + os.sep + "model.ilp"
            model.write(outfile)
        return {"status": "infeasible"}

    elif status != GRB.OPTIMAL:
        print("Optimization was stopped with status %d" % status)
        return {"status": "stopped"}
    else:
        obj = model.objVal
        x_opt, y_opt, z_opt = {}, {}, {}
        for v in model.getVars():
            if "penalty" in v.varName:
                continue
            if "x" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if v.x > 0:
                    if key_i not in x_opt:
                        x_opt[key_i] = {}
                    x_opt[key_i][key_j] = v.x
            if "y" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                key_k = v.varName[2:].split("-")[2]
                if v.x > 0:
                    if key_i not in y_opt:
                        y_opt[key_i] = {}
                    if key_j not in y_opt[key_i]:
                        y_opt[key_i][key_j] = {}
                    y_opt[key_i][key_j][key_k] = v.x
            if "z" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if v.x > 0:
                    if key_i not in z_opt:
                        z_opt[key_i] = {}
                    z_opt[key_i][key_j] = v.x

        outputs = {
            "status": "completed",
            "obj": obj,
            "x_opt": x_opt,
            "y_opt": y_opt,
            "z_opt": z_opt,
            "runtime": model.Runtime,
            "mipgap": model.MIPGap,
            "num_vars": model.NumVars,
            "num_cols": model.NumConstrs,
            "nodes": model.NodeCount,
        }
        return outputs


# -----------------------
# New implementations
# -----------------------
def Descending(inputs, penalty_unassigned="last_pref", objective="SOSM", outdir=None):

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn, tb = inputs
    level = {s: lev for lev in students_per_level for s in students_per_level[lev]}
    level_int = {
        s: -1 if level[s] == "PreK" else 0 if level[s] == "K" else int(level[s]) for s in level
    }
    # compute original position in list for objective and penalty
    opref = {}
    for s in pref:
        if s not in students:
            continue
        if pref[s] == {}:
            continue
        opref[s], idx = {}, 0
        for p in sorted(pref[s]):
            idx += 1
            opref[s][pref[s][p]] = idx

    model = Model("agg")

    siblings_in_school = {
        s: {
            pref[s][p].split("_")[0]: [
                sib
                for sib in siblings[s]
                if pref[s][p].split("_")[0] in [cp.split("_")[0] for cp in pref[sib].values()]
            ]
            for p in pref[s]
        }
        for s in siblings
    }

    x, y, z = (
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
    )
    # NOTE: x is defined at the course level (i.e., school + level); y,z are defined at the school level
    for s in x:
        for p in pref[s]:
            x[s][pref[s][p]] = model.addVar(
                vtype=GRB.BINARY,
                lb=0,
                ub=1,
                obj=opref[s][pref[s][p]],
                name="x" + "-" + s + "-" + pref[s][p],
            )
            rbd = pref[s][p].split("_")[0]
            if len(siblings_in_school[s][rbd]) > 0:
                z[s][rbd] = model.addVar(
                    vtype=GRB.BINARY,
                    lb=0,
                    ub=1,
                    name="z" + "-" + s + "-" + rbd,
                )
                for sib in siblings_in_school[s][rbd]:
                    if sib not in y[s]:
                        y[s][sib] = {}

                    y[s][sib][rbd] = model.addVar(
                        vtype=GRB.BINARY,
                        lb=0,
                        ub=int(level_int[s] > level_int[sib]),
                        name="y" + "-" + s + "-" + sib + "-" + rbd,
                    )

    # each student assigned to at most one school
    for s in x:
        model.addConstr(sum([x[s][c] for c in x[s]]) <= 1, "assignment_student_" + s)

    # each school receives at most capacity
    for c in colleges:
        model.addConstr(sum([x[s][c] for s in pref[c].values()]) <= cap[c], "capacity_college_" + c)

    # definition of providing priority
    for s in z:
        for rbd in z[s]:
            model.addConstr(
                z[s][rbd] <= sum([x[s][c] for c in x[s] if c.split("_")[0] == rbd]),
                "definition_z_1_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd] + sum(z[sib][rbd] for sib in siblings_in_school[s][rbd]) <= 1,
                "definition_z_3_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd]
                <= sum(
                    [
                        x[sib][c]
                        for sib in siblings_in_school[s][rbd]
                        for c in x[sib]
                        if c.split("_")[0] == rbd
                    ]
                ),
                "definition_z_4_" + s + "_" + rbd,
            )
        for c in x[s]:
            rbd = c.split("_")[0]
            if rbd in z[s]:
                model.addConstr(
                    len(Sp[s][c]) - sum([x[i][j] for i in Sp[s][c] for j in Tp[i][c] if j != c])
                    <= (cap[c] - 1) + len(students_per_level[level[s]]) * (1 - z[s][rbd]),
                    "definition_z_2_" + s + "_" + rbd,
                )

    # definition of receiving priority
    for s in y:
        for sib in y[s]:
            for rbd in y[s][sib]:
                model.addConstr(
                    y[s][sib][rbd] <= z[s][rbd], "6a-" + s + "-" + sib + "-" + rbd
                )  # cannot receive priority from a non-provider
                model.addConstr(
                    y[s][sib][rbd] <= 1 - z[sib][rbd], "6b-" + s + "-" + sib + "-" + rbd
                )  # cannot be provider and receiver of priority
                model.addConstr(
                    y[s][sib][rbd] <= sum([x[sib][c] for c in x[sib] if c.split("_")[0] == rbd]),
                    "6c-" + s + "-" + rbd,
                )  # receives siblings priority if gets assigned to school

    # stability
    for s in x:
        for c in x[s]:
            rbd = c.split("_")[0]
            model.addConstr(
                cap[c] * (1 - sum([x[s][j] for j in Tp[s][c]]))
                <= sum([x[i][c] for i in Sp[s][c]])
                + sum(
                    [
                        y[i][j][rbd]
                        for j in Sn[s][c]
                        for i in siblings_in_school[j][rbd]
                        if j in y[i]
                    ]
                ),
                "stability_1_" + s + "-" + c,
            )

            for sib in siblings_in_school[s][rbd]:
                for l in pref[c].values():
                    if l in siblings_in_school[s][rbd] or l == s:
                        continue
                    if siblings_in_school[l][rbd] != []:
                        model.addConstr(
                            sum(
                                [
                                    x[sib][c] * int(level_int[sib] > level_int[s])
                                    for c in x[sib]
                                    if c.split("_")[0] == rbd
                                ]
                            )
                            + (1 - sum([x[s][j] for j in Tp[s][c]]))
                            <= 2
                            - x[l][c]
                            + (sum([y[k][l][rbd] for k in siblings_in_school[l][rbd]]))
                            * (tb[l][rbd] > tb[s][rbd]),
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )
                    else:
                        model.addConstr(
                            sum(
                                [
                                    x[sib][c] * int(level_int[sib] > level_int[s])
                                    for c in x[sib]
                                    if c.split("_")[0] == rbd
                                ]
                            )
                            + (1 - sum([x[s][j] for j in Tp[s][c]]))
                            <= 2 - x[l][c],
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )

    # adding penalty for unassignment
    penalty = model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=1, name="penalty_unassigned")
    if isinstance(penalty_unassigned, (int, float, complex)):
        model.addConstr(
            penalty_unassigned * sum([1 - sum([x[s][c] for c in x[s]]) for s in x]) <= penalty
        )
    elif penalty_unassigned == "last_pref":
        model.addConstr(
            sum([(max(opref[s].values()) + 1) * (1 - sum([x[s][c] for c in x[s]])) for s in x])
            + sum([(max(opref[s].values()) + 1) for s in students if s not in x])
            <= penalty
        )
    else:
        print("***ERROR: Unkonwn penalty")
        sys.exit(1)

    # Set objective
    model.Params.OutputFlag = 1
    model.Params.Method = 3
    model.Params.Threads = 1
    model.Params.MIPFocus = 2
    model.Params.Presolve = 2
    model.Params.Cuts = 3
    model.Params.VarBranch = 1
    model.modelSense = GRB.MINIMIZE

    if objective in ["SOSM"]:
        model.Params.TimeLimit = 10000
        model.Params.MIPGap = 1e-2
    elif objective in ["MXSM", "ROSM"]:
        model.Params.TimeLimit = 36000
        model.Params.MIPGap = 1e-2
    else:
        print("***ERROR: Unrecognized objective function")
        sys.exit(1)

    # model.Params.PoolSearchMode = 2
    # model.Params.PoolSolutions = 10

    model.optimize()
    status = model.status

    if status == GRB.INF_OR_UNBD or status == GRB.INFEASIBLE or status == GRB.UNBOUNDED:
        print("The model cannot be solved because it is infeasible or unbounded")
        if outdir is not None:
            model.computeIIS()
            outfile = outdir + os.sep + "model.ilp"
            model.write(outfile)
        return {"status": "infeasible"}

    elif status != GRB.OPTIMAL:
        print("Optimization was stopped with status %d" % status)
        return {"status": "stopped"}
    else:
        obj = model.objVal
        x_opt, y_opt, z_opt = {}, {}, {}
        for v in model.getVars():
            if "penalty" in v.varName:
                continue
            if "x" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if key_i not in x_opt:
                    x_opt[key_i] = {}
                x_opt[key_i][key_j] = v.x
            if "y" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                key_k = v.varName[2:].split("-")[2]
                if key_i not in y_opt:
                    y_opt[key_i] = {}
                if key_j not in y_opt[key_i]:
                    y_opt[key_i][key_j] = {}
                y_opt[key_i][key_j][key_k] = v.x
            if "z" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if key_i not in z_opt:
                    z_opt[key_i] = {}
                z_opt[key_i][key_j] = v.x

        outputs = {
            "status": "completed",
            "obj": obj,
            "x_opt": x_opt,
            "y_opt": y_opt,
            "z_opt": z_opt,
            "runtime": model.Runtime,
            "mipgap": model.MIPGap,
            "num_vars": model.NumVars,
            "num_cols": model.NumConstrs,
            "nodes": model.NodeCount,
        }
        return outputs


def AbsoluteHardNTB(inputs, penalty_unassigned="last_pref", objective="SOSM", outdir=None):

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn, tb = inputs
    level = {s: lev for lev in students_per_level for s in students_per_level[lev]}
    # compute original position in list for objective and penalty
    opref = {}
    for s in pref:
        if s not in students:
            continue
        if pref[s] == {}:
            continue
        opref[s], idx = {}, 0
        for p in sorted(pref[s]):
            idx += 1
            opref[s][pref[s][p]] = idx

    model = Model("agg")

    siblings_in_school = {
        s: {
            pref[s][p].split("_")[0]: [
                sib
                for sib in siblings[s]
                if pref[s][p].split("_")[0] in [cp.split("_")[0] for cp in pref[sib].values()]
            ]
            for p in pref[s]
        }
        for s in siblings
    }

    x, y, z = (
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
    )
    # NOTE: x is defined at the course level (i.e., school + level); y,z are defined at the school level
    for s in x:
        for p in pref[s]:
            x[s][pref[s][p]] = model.addVar(
                vtype=GRB.BINARY,
                lb=0,
                ub=1,
                obj=opref[s][pref[s][p]] if objective in ["SOSM", "MXSM"] else 0,
                name="x" + "-" + s + "-" + pref[s][p],
            )
            rbd = pref[s][p].split("_")[0]
            if len(siblings_in_school[s][rbd]) > 0:
                z[s][rbd] = model.addVar(
                    vtype=GRB.BINARY, lb=0, ub=1, name="z" + "-" + s + "-" + rbd
                )
                for sib in siblings_in_school[s][rbd]:
                    if sib not in y[s]:
                        y[s][sib] = {}
                    y[s][sib][rbd] = model.addVar(
                        vtype=GRB.BINARY,
                        lb=0,
                        ub=1,
                        obj=-1 if objective in ["MXSM", "ROSM"] else 0,
                        name="y" + "-" + s + "-" + sib + "-" + rbd,
                    )
    # NOTE: modifying objective to prioritize siblings
    # each student assigned to at most one school
    for s in x:
        model.addConstr(sum([x[s][c] for c in x[s]]) <= 1, "assignment_student_" + s)

    # each school receives at most capacity
    for c in colleges:
        model.addConstr(sum([x[s][c] for s in pref[c].values()]) <= cap[c], "capacity_college_" + c)

    # definition of providing priority
    for s in z:
        for rbd in z[s]:
            model.addConstr(
                z[s][rbd] <= sum([x[s][c] for c in x[s] if c.split("_")[0] == rbd]),
                "definition_z_1_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd] + sum(z[sib][rbd] for sib in siblings_in_school[s][rbd]) <= 1,
                "definition_z_3_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd]
                <= sum(
                    [
                        x[sib][c]
                        for sib in siblings_in_school[s][rbd]
                        for c in x[sib]
                        if c.split("_")[0] == rbd
                    ]
                ),
                "definition_z_4_" + s + "_" + rbd,
            )
        # since everyone starts from the same priority group, the left hand side is zero so this constraint is not needed. Note that this would not be the case if we consider multiple priority groups.
        # for c in x[s]:
        #     rbd = c.split("_")[0]
        #     if rbd in z[s]:
        #         model.addConstr(
        #             len(Sp[s][c]) - sum([x[i][j] for i in Sp[s][c] for j in Tp[i][c] if j != c])
        #             <= (cap[c] - 1) + len(students_per_level[level[s]]) * (1 - z[s][rbd]),
        #             "definition_z_2_" + s + "_" + rbd,
        #         )

    # definition of receiving priority
    for s in y:
        for sib in y[s]:
            for rbd in y[s][sib]:
                model.addConstr(
                    y[s][sib][rbd] <= z[s][rbd], "6a-" + s + "-" + sib + "-" + rbd
                )  # cannot receive priority from a non-provider
                model.addConstr(
                    y[s][sib][rbd] <= 1 - z[sib][rbd], "6b-" + s + "-" + sib + "-" + rbd
                )  # cannot be provider and receiver of priority
                model.addConstr(
                    y[s][sib][rbd] <= sum([x[sib][c] for c in x[sib] if c.split("_")[0] == rbd]),
                    "6c-" + s + "-" + rbd,
                )  # receives siblings priority if gets assigned to school

    # stability
    for s in x:
        for c in x[s]:
            rbd = c.split("_")[0]
            model.addConstr(
                cap[c] * (1 - sum([x[s][j] for j in Tp[s][c]]))
                <= sum([x[i][c] for i in list(pref[c].values()) if i != s])
                + sum(
                    [
                        y[i][j][rbd]
                        for j in list(pref[c].values())
                        for i in siblings_in_school[j][rbd]
                        if i != s
                    ]
                )
                + sum(
                    [z[i][rbd] for i in z if rbd in z[i] and i in list(pref[c].values()) if i != s]
                ),
                "stability_1_" + s + "-" + c,
            )

            for sib in siblings_in_school[s][rbd]:
                for l in pref[c].values():
                    if l in siblings_in_school[s][rbd] or l == s:
                        continue
                    if siblings_in_school[l][rbd] != []:
                        model.addConstr(
                            sum([x[sib][c] for c in x[sib] if c.split("_")[0] == rbd])
                            + (1 - sum([x[s][j] for j in Tp[s][c]]))
                            <= 2
                            - x[l][c]
                            + z[l][rbd]
                            + sum([y[k][l][rbd] for k in siblings_in_school[l][rbd]]),
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )
                    else:
                        model.addConstr(
                            sum([x[sib][c] for c in x[sib] if c.split("_")[0] == rbd])
                            + (1 - sum([x[s][j] for j in Tp[s][c]]))
                            <= 2 - x[l][c],
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )

    # adding penalty for unassignment
    penalty = model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=1, name="penalty_unassigned")
    if isinstance(penalty_unassigned, (int, float, complex)):
        model.addConstr(
            penalty_unassigned * sum([1 - sum([x[s][c] for c in x[s]]) for s in x]) <= penalty
        )
    elif penalty_unassigned == "last_pref":
        model.addConstr(
            sum([(max(opref[s].values()) + 1) * (1 - sum([x[s][c] for c in x[s]])) for s in x])
            + sum([(max(opref[s].values()) + 1) for s in students if s not in x])
            <= penalty
        )
    else:
        print("***ERROR: Unkonwn penalty")
        sys.exit(1)

    # Set objective
    model.Params.OutputFlag = 1
    model.Params.Method = 3
    model.Params.Threads = 1
    model.Params.MIPFocus = 2
    model.Params.Presolve = 2
    model.Params.Cuts = 3
    model.Params.VarBranch = 1
    model.modelSense = GRB.MINIMIZE

    if objective in ["SOSM"]:
        model.Params.TimeLimit = 10000
        model.Params.MIPGap = 1e-2
    elif objective in ["MXSM", "ROSM"]:
        model.Params.TimeLimit = 36000
        model.Params.MIPGap = 1e-2
    else:
        print("***ERROR: Unrecognized objective function")
        sys.exit(1)

    # model.Params.PoolSearchMode = 2
    # model.Params.PoolSolutions = 10

    model.optimize()
    status = model.status

    if status == GRB.INF_OR_UNBD or status == GRB.INFEASIBLE or status == GRB.UNBOUNDED:
        print("The model cannot be solved because it is infeasible or unbounded")
        if outdir is not None:
            model.computeIIS()
            outfile = outdir + os.sep + "model.ilp"
            model.write(outfile)
        return {"status": "infeasible"}

    elif status != GRB.OPTIMAL:
        print("Optimization was stopped with status %d" % status)
        return {"status": "stopped"}
    else:
        obj = model.objVal
        x_opt, y_opt, z_opt = {}, {}, {}
        for v in model.getVars():
            if "penalty" in v.varName:
                continue
            if "x" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if key_i not in x_opt:
                    x_opt[key_i] = {}
                x_opt[key_i][key_j] = v.x
            if "y" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                key_k = v.varName[2:].split("-")[2]
                if key_i not in y_opt:
                    y_opt[key_i] = {}
                if key_j not in y_opt[key_i]:
                    y_opt[key_i][key_j] = {}
                y_opt[key_i][key_j][key_k] = v.x
            if "z" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if key_i not in z_opt:
                    z_opt[key_i] = {}
                z_opt[key_i][key_j] = v.x

        outputs = {
            "status": "completed",
            "obj": obj,
            "x_opt": x_opt,
            "y_opt": y_opt,
            "z_opt": z_opt,
            "runtime": model.Runtime,
            "mipgap": model.MIPGap,
            "num_vars": model.NumVars,
            "num_cols": model.NumConstrs,
            "nodes": model.NodeCount,
        }
        return outputs


def DescendingNTB(inputs, penalty_unassigned="last_pref", objective="SOSM", outdir=None):

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn, tb = inputs
    level = {s: lev for lev in students_per_level for s in students_per_level[lev]}
    level_int = {
        s: -1 if level[s] == "PreK" else 0 if level[s] == "K" else int(level[s]) for s in level
    }
    # compute original position in list for objective and penalty
    opref = {}
    for s in pref:
        if s not in students:
            continue
        if pref[s] == {}:
            continue
        opref[s], idx = {}, 0
        for p in sorted(pref[s]):
            idx += 1
            opref[s][pref[s][p]] = idx

    model = Model("agg")

    siblings_in_school = {
        s: {
            pref[s][p].split("_")[0]: [
                sib
                for sib in siblings[s]
                if pref[s][p].split("_")[0] in [cp.split("_")[0] for cp in pref[sib].values()]
            ]
            for p in pref[s]
        }
        for s in siblings
    }

    x, y, z = (
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
        {s: {} for s in students if pref[s] != {}},
    )
    # NOTE: x is defined at the course level (i.e., school + level); y,z are defined at the school level
    for s in x:
        for p in pref[s]:
            x[s][pref[s][p]] = model.addVar(
                vtype=GRB.BINARY,
                lb=0,
                ub=1,
                obj=opref[s][pref[s][p]] if objective in ["SOSM", "MXSM"] else 0,
                name="x" + "-" + s + "-" + pref[s][p],
            )
            rbd = pref[s][p].split("_")[0]
            if len(siblings_in_school[s][rbd]) > 0:
                z[s][rbd] = model.addVar(
                    vtype=GRB.BINARY, lb=0, ub=1, name="z" + "-" + s + "-" + rbd
                )
                for sib in siblings_in_school[s][rbd]:
                    if sib not in y[s]:
                        y[s][sib] = {}
                    y[s][sib][rbd] = model.addVar(
                        vtype=GRB.BINARY,
                        lb=0,
                        ub=int(level_int[s] > level_int[sib]),
                        obj=-1 if objective in ["SOSM", "MXSM", "ROSM"] else 0,
                        name="y" + "-" + s + "-" + sib + "-" + rbd,
                    )
    # NOTE: modifying objective to prioritize siblings
    # each student assigned to at most one school
    for s in x:
        model.addConstr(sum([x[s][c] for c in x[s]]) <= 1, "assignment_student_" + s)

    # each school receives at most capacity
    for c in colleges:
        model.addConstr(sum([x[s][c] for s in pref[c].values()]) <= cap[c], "capacity_college_" + c)

    # definition of providing priority
    for s in z:
        for rbd in z[s]:
            model.addConstr(
                z[s][rbd] <= sum([x[s][c] for c in x[s] if c.split("_")[0] == rbd]),
                "definition_z_1_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd] + sum(z[sib][rbd] for sib in siblings_in_school[s][rbd]) <= 1,
                "definition_z_3_" + s + "_" + rbd,
            )
            model.addConstr(
                z[s][rbd]
                <= sum(
                    [
                        x[sib][c]
                        for sib in siblings_in_school[s][rbd]
                        for c in x[sib]
                        if c.split("_")[0] == rbd
                    ]
                ),
                "definition_z_4_" + s + "_" + rbd,
            )
        # since everyone starts from the same priority group, the left hand side is zero so this constraint is not needed. Note that this would not be the case if we consider multiple priority groups.
        # for c in x[s]:
        #     rbd = c.split("_")[0]
        #     if rbd in z[s]:
        #         model.addConstr(
        #             len(Sp[s][c]) - sum([x[i][j] for i in Sp[s][c] for j in Tp[i][c] if j != c])
        #             <= (cap[c] - 1) + len(students_per_level[level[s]]) * (1 - z[s][rbd]),
        #             "definition_z_2_" + s + "_" + rbd,
        #         )

    # definition of receiving priority
    for s in y:
        for sib in y[s]:
            for rbd in y[s][sib]:
                model.addConstr(
                    y[s][sib][rbd] <= z[s][rbd], "6a-" + s + "-" + sib + "-" + rbd
                )  # cannot receive priority from a non-provider
                model.addConstr(
                    y[s][sib][rbd] <= 1 - z[sib][rbd], "6b-" + s + "-" + sib + "-" + rbd
                )  # cannot be provider and receiver of priority
                model.addConstr(
                    y[s][sib][rbd] <= sum([x[sib][c] for c in x[sib] if c.split("_")[0] == rbd]),
                    "6c-" + s + "-" + rbd,
                )  # receives siblings priority if gets assigned to school

    # stability
    for s in x:
        for c in x[s]:
            rbd = c.split("_")[0]
            model.addConstr(
                cap[c] * (1 - sum([x[s][j] for j in Tp[s][c]]))
                <= sum([x[i][c] for i in list(pref[c].values()) if i != s])
                + sum(
                    [
                        y[i][j][rbd]
                        for j in list(pref[c].values())
                        for i in siblings_in_school[j][rbd]
                        if i != s
                    ]
                ),
                "stability_1_" + s + "-" + c,
            )

            for sib in siblings_in_school[s][rbd]:
                for l in pref[c].values():
                    if l in siblings_in_school[s][rbd] or l == s:
                        continue
                    if siblings_in_school[l][rbd] != []:
                        model.addConstr(
                            sum(
                                [
                                    x[sib][c] * int(level_int[sib] > level_int[s])
                                    for c in x[sib]
                                    if c.split("_")[0] == rbd
                                ]
                            )
                            + (1 - sum([x[s][j] for j in Tp[s][c]]))
                            <= 2
                            - x[l][c]
                            + sum([y[k][l][rbd] for k in siblings_in_school[l][rbd]]),
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )
                    else:
                        model.addConstr(
                            sum(
                                [
                                    x[sib][c] * int(level_int[sib] > level_int[s])
                                    for c in x[sib]
                                    if c.split("_")[0] == rbd
                                ]
                            )
                            + (1 - sum([x[s][j] for j in Tp[s][c]]))
                            <= 2 - x[l][c],
                            "stability_2_" + s + "-" + sib + "-" + l + "-" + c,
                        )

    # adding penalty for unassignment
    penalty = model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=1, name="penalty_unassigned")
    if isinstance(penalty_unassigned, (int, float, complex)):
        model.addConstr(
            penalty_unassigned * sum([1 - sum([x[s][c] for c in x[s]]) for s in x]) <= penalty
        )
    elif penalty_unassigned == "last_pref":
        model.addConstr(
            sum([(max(opref[s].values()) + 1) * (1 - sum([x[s][c] for c in x[s]])) for s in x])
            + sum([(max(opref[s].values()) + 1) for s in students if s not in x])
            <= penalty
        )
    else:
        print("***ERROR: Unkonwn penalty")
        sys.exit(1)

    # Set objective
    model.Params.OutputFlag = 1
    model.Params.Method = 3
    model.Params.Threads = 1
    model.Params.MIPFocus = 2
    model.Params.Presolve = 2
    model.Params.Cuts = 3
    model.Params.VarBranch = 1
    model.modelSense = GRB.MINIMIZE

    if objective in ["SOSM"]:
        model.Params.TimeLimit = 10000
        model.Params.MIPGap = 1e-2
    elif objective in ["MXSM", "ROSM"]:
        model.Params.TimeLimit = 36000
        model.Params.MIPGap = 1e-2
    else:
        print("***ERROR: Unrecognized objective function")
        sys.exit(1)

    # model.Params.PoolSearchMode = 2
    # model.Params.PoolSolutions = 10

    model.optimize()
    status = model.status

    if status == GRB.INF_OR_UNBD or status == GRB.INFEASIBLE or status == GRB.UNBOUNDED:
        print("The model cannot be solved because it is infeasible or unbounded")
        if outdir is not None:
            model.computeIIS()
            outfile = outdir + os.sep + "model.ilp"
            model.write(outfile)
        return {"status": "infeasible"}

    elif status != GRB.OPTIMAL:
        print("Optimization was stopped with status %d" % status)
        return {"status": "stopped"}
    else:
        obj = model.objVal
        x_opt, y_opt, z_opt = {}, {}, {}
        for v in model.getVars():
            if "penalty" in v.varName:
                continue
            if "x" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if key_i not in x_opt:
                    x_opt[key_i] = {}
                x_opt[key_i][key_j] = v.x
            if "y" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                key_k = v.varName[2:].split("-")[2]
                if key_i not in y_opt:
                    y_opt[key_i] = {}
                if key_j not in y_opt[key_i]:
                    y_opt[key_i][key_j] = {}
                y_opt[key_i][key_j][key_k] = v.x
            if "z" in v.varName:
                key_i = v.varName[2:].split("-")[0]
                key_j = v.varName[2:].split("-")[1]
                if key_i not in z_opt:
                    z_opt[key_i] = {}
                z_opt[key_i][key_j] = v.x

        outputs = {
            "status": "completed",
            "obj": obj,
            "x_opt": x_opt,
            "y_opt": y_opt,
            "z_opt": z_opt,
            "runtime": model.Runtime,
            "mipgap": model.MIPGap,
            "num_vars": model.NumVars,
            "num_cols": model.NumConstrs,
            "nodes": model.NodeCount,
        }
        return outputs


if __name__ == "__main__":
    
    region = "OHiggins"
    year = 2023
    tie_breaker = "mtbf"
    objective = "SOSM"
    sim = 44

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Optional output folders
    outdir = os.path.join(base_dir, "..", "outputs")
    plotdir = os.path.join(base_dir, "..", "plots")
    tabdir = os.path.join(base_dir, "..", "tables")

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(plotdir, exist_ok=True)
    os.makedirs(tabdir, exist_ok=True)

    # Actual input folder where your instance files live
    indir = os.path.join(base_dir, "..", "R", "intermediate_data", region, str(year))

    tb_file = os.path.join(indir, f"tb_{tie_breaker}.pck")
    instance_file = os.path.join(indir, f"instance_{tie_breaker}.txt")

    if not os.path.exists(tb_file):
        raise FileNotFoundError(f"Could not find tie-breaker file: {tb_file}")

    if not os.path.exists(instance_file):
        raise FileNotFoundError(f"Could not find instance file: {instance_file}")

    with open(tb_file, "rb") as f:
        tb = pickle.load(f)

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn = (
        genin.read_instance(instance_file)
    )

    out_abs_n = AbsoluteHardNTB(
        (students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn, tb),
        "last_pref",
        objective,
    )

    print(out_abs_n)
    

    