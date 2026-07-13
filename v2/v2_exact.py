"""
v2_exact.py

Integer programming formulations for stable matching with contingent priorities,
plus a second exact solver used to cross-check them.

solve() is the only entry point. Every constraint block maps to a numbered
equation in the paper; the table below gives the mapping. The feasible set of
each formulation is exactly the stable set defined in v2_stability.py. That claim
is certified by exhaustive enumeration in the v1 archive, which contains the test
suite.

Priorities
  absolute   A student who provides or receives contingent priority outranks any
             student who does not, whatever the initial lottery.
  partial    A prioritized student outranks only students with a worse contingent
             tie-breaker, where a student's contingent tie-breaker is the better
             of their own lottery and their provider's.
  none       No contingent priority. Initial stability. The rank-optimal solution
             is the student-optimal stable matching.

Enforcement
  hard       Every effective priority provider must be honored.
  soft       The clearinghouse may ignore any provider. Always feasible: the set
             of initially stable matchings is contained in the feasible set.
  hybrid     Soft, plus a floor: at least zeta_min providers must be honored.
             Feasibility is monotone decreasing in zeta_min, so bisecting on it
             gives the largest number of providers the market can support.

Objectives
  rank       Minimize the sum of assigned ranks. An unassigned student costs
             len(prefs[s]) + 1. This is the paper's default.
  feasible   Constant objective, stop at the first solution. Answers "does a
             stable matching exist" without paying for optimality. On the large
             regions this is much cheaper than the rank objective, and its
             solution is a valid warm start for it.
  receivers  Maximize the number of students who receive contingent priority.


Constraint names and where they are in the paper
------------------------------------------------
Gurobi constraint names appear in solver logs and in infeasibility certificates,
so they carry the paper's equation numbers. Two constraints have no number
because they are not in the current draft; both are on the authors' list of
additions and are named for what they do.

    assign[s]           each student takes at most one seat        set X
    cap[c,l]            capacity, per school and level             set X
    4a[s,c]             z <= x                                     (4a)
    RE[s,c]             a sibling weakly prefers c to their match  NOT IN THE DRAFT
    4b[s,c]             the rightful-claim count                   (4b)
    4c[f,c]             at most one provider per family and school (4c)
    4d[s,c]             standing: co-assigned, or a same-level
                        sibling assigned strictly worse than c     (4d)
    5a[s,t,c]           y <= x                                     (5a)
    5b[s,t,c]           y <= z                                     (5b)
    5c[s,t,c]           y <= 1 - z                                 (5c)
    6a[s,c]             absolute contingent stability              (6a)
    6b[c,s,s',a]        absolute tie-breaking, hard                (6b)
    7d[c,s,s',a]        absolute tie-breaking, soft                (7d)
    hybrid              sum of z >= zeta_min                       Section 5.3.1
    16a[s,c]            partial contingent stability               (16a)
    16b[c,s,s',a]       partial tie-breaking, hard and soft        (16b)
    SC[c,s,a]           sibling competition                        NOT IN THE DRAFT

RE[s,c] is redundant under absolute priorities: (4d) already forces a sibling to
be co-assigned or assigned strictly worse than c, and either implies that some
sibling weakly prefers c to their match. It is not redundant under partial, where
there is no (4d), so it is imposed in both cases and costs nothing in the first.

SC[c,s,a] rules out sibling justified envy under partial priorities, which (16a)
and (16b) do not. Without it the formulation admits matchings that
v2_stability.py rejects.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Dict, Optional, Tuple

from v2_model import Instance, Matching, Student, School


def _grb():
    import gurobipy as gp
    from gurobipy import GRB
    return gp, GRB


def _status(GRB, code: int) -> str:
    for name in ("OPTIMAL", "INFEASIBLE", "TIME_LIMIT", "SOLUTION_LIMIT",
                 "INTERRUPTED", "SUBOPTIMAL", "INF_OR_UNBD"):
        if code == getattr(GRB, name):
            return name
    return str(code)


# ==========================================================================
# the explicit formulations
# ==========================================================================
def solve(inst: Instance,
          priority: str = "absolute",          # absolute | partial | none
          enforcement: str = "hard",           # hard | soft | hybrid
          *,
          objective: str = "rank",             # rank | feasible | receivers
          zeta_min: Optional[int] = None,      # required when enforcement=hybrid
          time_limit: Optional[float] = None,
          mip_gap: Optional[float] = None,
          threads: Optional[int] = None,
          warm_start: Optional[Matching] = None,
          no_rel_heur_time: float = 0.0,
          extra_params: Optional[dict] = None,
          callback=None,
          compute_iis: bool = False,
          iis_time_limit: Optional[float] = None,
          verbose: bool = False) -> Tuple[Optional[Matching], dict]:
    """Return (matching, info). The matching is None if the solver found no
    feasible point, either because none exists or because it ran out of time;
    info["status_str"] distinguishes the two.

    no_rel_heur_time sets Gurobi's NoRelHeurTime, which runs a heuristic before
    the root relaxation. It is only useful when finding any feasible point is the
    bottleneck, which is the case on the large regions: on one instance that had
    never produced a feasible point in four hours of branch and bound, this
    heuristic found one in 235 seconds. It is off by default because it delays
    the root relaxation and is wasted on small instances.
    """
    if inst.num_groups != 1:
        raise NotImplementedError("the formulations assume a single initial "
                                  "priority group (Remark 2)")
    if enforcement == "hybrid":
        if zeta_min is None:
            raise ValueError("enforcement='hybrid' requires zeta_min")
        enforcement = "soft"                   # hybrid is soft plus the floor
    elif enforcement not in ("hard", "soft"):
        raise ValueError("enforcement must be hard, soft or hybrid")
    if priority not in ("absolute", "partial", "none"):
        raise ValueError("priority must be absolute, partial or none")

    gp, GRB = _grb()
    t0 = time.perf_counter()
    m = gp.Model("contingent")
    m.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap
    if threads is not None:
        m.Params.Threads = threads
    if no_rel_heur_time > 0:
        m.Params.NoRelHeurTime = no_rel_heur_time

    S, lot, q = inst.students, inst.lottery, inst.q

    # (s, c) is a variable only if s lists c and c has seats at s's level
    pairs = [(s, c) for s in S for c in inst.prefs[s]
             if q(c, inst.level_of[s]) > 0]
    unassigned_cost = {s: len(inst.prefs[s]) + 1 for s in S}

    # ---------------- x: the assignment ----------------
    x = m.addVars(pairs, vtype=GRB.BINARY, name="x")
    m.addConstrs((x.sum(s, "*") <= 1 for s in S), name="assign")
    for c in inst.schools:
        for ell in inst.levels:
            seats = [(s, c) for s in inst.listers(c, ell) if (s, c) in x]
            if seats:
                m.addConstr(gp.quicksum(x[k] for k in seats) <= q(c, ell),
                            name=f"cap[{c},{ell}]")

    if warm_start is not None:
        for (s, c) in pairs:
            x[s, c].Start = 1.0 if warm_start.get(s) == c else 0.0

    def matched_at_or_better(s, c):
        """1 if s is matched to c or to a school they prefer to c. The comparison
        is WEAK: c itself counts."""
        r = inst.rank(s, c)
        return gp.quicksum(x[s, cp] for cp in inst.prefs[s]
                           if inst.rank(s, cp) <= r and (s, cp) in x)

    def missed(s, c):
        """1 if s got neither c nor anything better. This is the left-hand side
        of the stability rows: 1 - (matched to c or better).

        It is NOT the same as weakly_prefers below, which is 1 when s IS matched
        to c. Substituting one for the other makes the stability row demand that c
        be full of students who outrank s even for the student sitting in it, and
        the formulation becomes infeasible on almost every instance."""
        return 1 - matched_at_or_better(s, c)

    def weakly_prefers(s, c):
        """1 if s is matched to c, to a worse school, or to nothing. The
        comparison is STRICT: this is 1 minus the sum over schools s prefers to c.
        Used only in the provider rows RE, 4b and 4d."""
        r = inst.rank(s, c)
        return 1 - gp.quicksum(x[s, cp] for cp in inst.prefs[s]
                               if inst.rank(s, cp) < r and (s, cp) in x)

    contingent = priority in ("absolute", "partial")
    z: Dict[Tuple[Student, School], "gp.Var"] = {}
    y: Dict[Tuple[Student, Student, School], "gp.Var"] = {}

    # ---------------- z: effective priority providers, set Z(x) ----------------
    if contingent:
        def eligible(t, c):
            """t counts toward c only if t lists c and c has seats at t's level.
            This is the set V of the paper."""
            return inst.acceptable(t, c) and q(c, inst.level_of[t]) > 0

        # z exists only where at least two members of the family are eligible at
        # c, since a lone applicant can neither provide nor receive priority
        zpairs = [(s, c) for (s, c) in pairs
                  if sum(1 for t in inst.family_members(s) if eligible(t, c)) >= 2]
        z = m.addVars(zpairs, vtype=GRB.BINARY, name="z")

        for (s, c) in zpairs:
            ell = inst.level_of[s]
            fam = [t for t in inst.siblings(s) if eligible(t, c)]

            # (4a) a provider is assigned to the school
            m.addConstr(z[s, c] <= x[s, c], name=f"4a[{s},{c}]")

            # RE: some sibling weakly prefers c to their match, so there is
            # someone for s to provide priority to. Not in the draft; redundant
            # under absolute, required under partial.
            m.addConstr(z[s, c] <= gp.quicksum(weakly_prefers(t, c) for t in fam),
                        name=f"RE[{s},{c}]")

            # (4b) s can rightfully claim a seat: at most q - 1 same-level
            # students with a better initial order weakly prefer c to their
            # match. big_m only has to make the row vacuous when z = 0, and the
            # left side is at most len(better), so len(better) - (q - 1) would
            # do. The paper uses the size of the level. This is between the two.
            better = [t for t in inst.listers(c, ell)
                      if t != s and lot[(t, c)] < lot[(s, c)]]
            big_m = len(better) + 1
            m.addConstr(gp.quicksum(weakly_prefers(t, c) for t in better)
                        <= (q(c, ell) - 1) + big_m * (1 - z[s, c]),
                        name=f"4b[{s},{c}]")

            # (4d) standing: a provider keeps their initial group only if a
            # sibling is co-assigned to c, or a same-level sibling is assigned
            # strictly worse than c. Absolute only; under partial a provider
            # keeps their group unconditionally.
            #
            # weakly_prefers includes the case t is assigned to c, which the
            # paper's version excludes. That is harmless: the two agree whenever
            # the first sum is zero, which is the only case where the row binds.
            if priority == "absolute":
                same_level = [t for t in fam if inst.level_of[t] == ell]
                m.addConstr(z[s, c]
                            <= gp.quicksum(x[t, c] for t in fam if (t, c) in x)
                            + gp.quicksum(weakly_prefers(t, c) for t in same_level),
                            name=f"4d[{s},{c}]")

        # (4c) at most one provider per family and school
        done = set()
        for (s, c) in zpairs:
            key = (inst.family_of[s], c)
            if key in done:
                continue
            done.add(key)
            members = [(t, c) for t in inst.families[key[0]] if (t, c) in z]
            m.addConstr(gp.quicksum(z[k] for k in members) <= 1, name=f"4c[{key}]")

        # ---------------- y: receivers, set Y(x, z) ----------------
        # y[s, t, c] = 1 if t receives priority at c from their sibling s
        ykeys = [(s, t, c) for (s, c) in zpairs
                 for t in inst.siblings(s) if (t, c) in x]
        y = m.addVars(ykeys, vtype=GRB.BINARY, name="y")
        for (s, t, c) in ykeys:
            m.addConstr(y[s, t, c] <= x[t, c], name=f"5a[{s},{t},{c}]")  # (5a)
            m.addConstr(y[s, t, c] <= z[s, c], name=f"5b[{s},{t},{c}]")  # (5b)
            if (t, c) in z:                                              # (5c)
                m.addConstr(y[s, t, c] <= 1 - z[t, c], name=f"5c[{s},{t},{c}]")

    def prioritized(a, c):
        """z[a,c] + sum of y[.,a,c]: a is prioritized at c, as a provider with
        standing or as a receiver."""
        terms = []
        if (a, c) in z:
            terms.append(z[a, c])
        terms += [y[ap, a, c] for ap in inst.siblings(a) if (ap, a, c) in y]
        return gp.quicksum(terms) if terms else gp.LinExpr(0)

    # ---------------- stability ----------------
    # If s is not matched to c or better, then c is full at s's level with
    # students who outrank s. Who outranks s depends on the priority type.
    for (s, c) in pairs:
        ell = inst.level_of[s]
        peers = [a for a in inst.listers(c, ell) if a != s and (a, c) in x]

        if priority in ("absolute", "none"):
            # Anyone with a better initial lottery, siblings included.
            rhs = gp.quicksum(x[a, c] for a in peers
                              if lot[(a, c)] < lot[(s, c)])
            if priority == "absolute":
                # Plus anyone with a worse lottery who is prioritized. Members
                # of s's own family are excluded from this second sum, which is
                # what forces the within-family order: a seat held by a sibling
                # with a worse lottery does not count against s.
                fid = inst.family_of[s]
                rhs += gp.quicksum(prioritized(a, c) for a in peers
                                   if lot[(a, c)] > lot[(s, c)]
                                   and inst.family_of[a] != fid)
            name = f"6a[{s},{c}]" if priority == "absolute" else f"3a[{s},{c}]"
        else:
            # partial: a prioritized student only outranks s if the student who
            # provided that priority also has a better lottery than s
            rhs = gp.quicksum(x[a, c] for a in peers
                              if lot[(a, c)] < lot[(s, c)])
            for a in peers:
                if lot[(a, c)] > lot[(s, c)]:
                    rhs += gp.quicksum(
                        y[ap, a, c] for ap in inst.siblings(a)
                        if (ap, a, c) in y and lot[(ap, c)] < lot[(s, c)])
            name = f"16a[{s},{c}]"

        m.addConstr(q(c, ell) * missed(s, c) <= rhs, name=name)

    # ---------------- tie-breaking over students outside the family ----------
    # If s has a sibling s' at c and s is not matched to c or better, then any
    # outsider a holding a seat at c must outrank s.
    if contingent:
        for fid, members in inst.families.items():
            if len(members) < 2:
                continue
            for c in inst.schools:
                in_family = [t for t in members if (t, c) in x]
                for s in members:
                    if (s, c) not in x:
                        continue
                    ell = inst.level_of[s]
                    outsiders = [a for a in inst.listers(c, ell)
                                 if inst.family_of[a] != fid and (a, c) in x]
                    for sp in in_family:
                        if sp == s:
                            continue
                        # Under soft priorities the family only displaces an
                        # outsider if the sibling is an honored provider, so the
                        # left side is z rather than x. This is the only change,
                        # and it is what makes soft always feasible.
                        if enforcement == "soft":
                            if (sp, c) not in z:
                                continue
                            anchor = z[sp, c]
                        else:
                            anchor = x[sp, c]

                        for a in outsiders:
                            if priority == "absolute":
                                outranks = 1 if lot[(a, c)] < lot[(s, c)] else 0
                                rhs = 2 - x[a, c] + outranks * prioritized(a, c)
                                name = (f"7d[{c},{s},{sp},{a}]"
                                        if enforcement == "soft"
                                        else f"6b[{c},{s},{sp},{a}]")
                            else:
                                # partial: compare against the better lottery of
                                # s and s', and only count receivers whose own
                                # provider beats that
                                best = min(lot[(s, c)], lot.get((sp, c),
                                                                lot[(s, c)]))
                                outranks = 1 if lot[(a, c)] > best else 0
                                recv = gp.LinExpr(0)
                                for ap in inst.siblings(a):
                                    if (ap, a, c) in y and \
                                            min(lot[(a, c)], lot[(ap, c)]) < best:
                                        recv += y[ap, a, c]
                                rhs = 2 - outranks * x[a, c] + recv
                                name = f"16b[{c},{s},{sp},{a}]"

                            m.addConstr(anchor + missed(s, c) <= rhs,
                                        name=name)

        # SC: sibling competition. Under partial, a sibling with a worse lottery
        # cannot hold a seat at c while a sibling with a better lottery is left
        # out. (16a) and (16b) do not rule this out. Not in the draft.
        if priority == "partial":
            for fid, members in inst.families.items():
                if len(members) < 2:
                    continue
                for c in inst.schools:
                    listers = [t for t in members if (t, c) in x]
                    for s in listers:
                        for a in listers:
                            if a == s or inst.level_of[a] != inst.level_of[s]:
                                continue
                            if lot[(s, c)] < lot[(a, c)]:
                                m.addConstr(x[a, c] <= matched_at_or_better(s, c),
                                            name=f"SC[{c},{s},{a}]")

        # hybrid: honor at least zeta_min providers. Feasibility is monotone
        # decreasing in zeta_min.
        if zeta_min is not None:
            m.addConstr(gp.quicksum(z.values()) >= zeta_min, name="hybrid")

    # ---------------- objective ----------------
    rank_expr = (gp.quicksum(inst.rank(s, c) * x[s, c] for (s, c) in pairs)
                 + gp.quicksum(unassigned_cost[s] * (1 - x.sum(s, "*"))
                               for s in S))

    if objective == "rank":
        m.setObjective(rank_expr, GRB.MINIMIZE)
    elif objective == "feasible":
        # A constant objective plus a solution limit: Gurobi stops at the first
        # feasible point. This answers existence without paying for optimality,
        # and its solution is a valid warm start for the rank objective.
        m.setObjective(gp.LinExpr(0.0), GRB.MINIMIZE)
        m.Params.SolutionLimit = 1
        m.Params.MIPFocus = 1
    elif objective == "receivers":
        if not contingent:
            raise ValueError("objective='receivers' needs contingent priorities")
        m.setObjective(-gp.quicksum(y.values()), GRB.MINIMIZE)
    else:
        raise ValueError("objective must be rank, feasible or receivers")

    if extra_params:
        for k, v in extra_params.items():
            m.setParam(k, v)

    if callback is not None:
        m.optimize(callback)
    else:
        m.optimize()

    info = {"status": int(m.Status), "status_str": _status(GRB, m.Status),
            "runtime": time.perf_counter() - t0, "backend": "gurobi",
            "priority": priority, "enforcement": enforcement,
            "objective_kind": objective}
    for attr, key in (("ObjBound", "obj_bound"), ("ObjBoundC", "obj_bound_c"),
                      ("NodeCount", "nodes_explored"),
                      ("OpenNodeCount", "nodes_open"),
                      ("SolCount", "sol_count"), ("IterCount", "simplex_iters")):
        try:
            info[key] = getattr(m, attr)
        except Exception:
            pass

    # An irreducible infeasible subsystem is a minimal certificate that no stable
    # matching exists. Computed on the model that just proved infeasibility, so
    # it costs one solve rather than two. computeIIS inherits the model's
    # TimeLimit, which is why it is capped separately.
    if m.Status == GRB.INFEASIBLE and compute_iis:
        try:
            if iis_time_limit is not None:
                m.Params.TimeLimit = iis_time_limit
            m.computeIIS()
            rows = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
            info["iis_constraints"] = rows
            info["iis_size"] = len(rows)
            info["iis_by_tag"] = dict(Counter(r.split("[")[0] for r in rows))
            # a truncated IIS is still infeasible, it is just not minimal
            info["iis_truncated"] = (m.Status == GRB.TIME_LIMIT)
        except Exception as e:
            info["iis_error"] = str(e)

    if m.SolCount == 0:
        info["objective"] = None
        info["mip_gap"] = None
        return None, info

    info["objective"] = rank_expr.getValue()
    info["mip_gap"] = getattr(m, "MIPGap", None)
    if contingent:
        # how many providers and receivers the solver raised. This is a lower
        # bound on the count implied by the definition; use v2_metrics.evaluate
        # for the definitional numbers.
        info["n_providers"] = int(round(sum(v.X for v in z.values())))
        info["n_receivers"] = int(round(sum(v.X for v in y.values())))

    mu: Matching = {s: None for s in S}
    for (s, c), var in x.items():
        if var.X > 0.5:
            mu[s] = c
    return mu, info
