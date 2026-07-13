"""
v2_rada.py

The RADA variants. v2_v2_heuristics.rada is the production RA-DA; the routines
here follow the paper's pseudocode and the convergence-boosting alternatives, so
the two can be run side by side.

  rada(inst)                     Algorithm RADA (alg:rada-sequential): iterate
                                 DA, recomputing priorities with UpdatePriorities
                                 from the current matching; stop at a fixed point
                                 or a repeated matching (cycle). Returns the
                                 matching that triggered the stop with the status
                                 label, exactly as the pseudocode does.

  update_priorities_key(inst,mu[,providers])   Algorithm UpdatePriorities:
                                 start everyone at group |G|+1, keep initial
                                 better-than-default groups, then for each
                                 effective provider (lottery order) keep its own
                                 group iff co-assigned, and lift every
                                 weakly-preferring sibling to the provider's
                                 group. `providers` restricts P (search variant).

  rada_sequential_search(inst)   Algorithm RADA with Sequential Search: DFS over
                                 maximal provider subsequences
                                 (GenerateProviderSequences), returning a
                                 reachable fixed point or reporting that every
                                 reachable sequence cycles.

HOW THE THREE RADA FORMS RELATE (all now use the current ACS definition):
  RA-DA (v2_heuristics.rada)        Definition-2 order; on a cycle returns the best
                                 stable iterate found in the cycle.
  RADA-Paper (rada here)         Definition-2 order (same update as RA-DA); on a
                                 cycle returns the repeated matching itself, as
                                 the pseudocode does. So RADA-Paper and RA-DA
                                 reach the same fixed points and differ only in
                                 what they report on a cycle.
  RADA-Search (rada_sequential_search)  Definition-2 order; depth-first search
                                 over maximal provider subsequences, recovering a
                                 reachable fixed point that the single-pass forms
                                 miss, or reporting that every reachable sequence
                                 cycles.

The provider-retention rule matches v2_v2_stability.absolute_group: a provider
keeps its group when co-assigned, or when it has a same-level receiver. That is
the standing condition, constraint (4d) of the paper. On the paper's three worked
examples every sibling pair is at a different level, so this rule and the older
co-assigned-only rule agree there and the examples still cycle. The two differ
only on instances with same-level siblings.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from v2_model import Instance, Matching, School, Student, total_rank
from v2_heuristics import da_with_key, deferred_acceptance
import v2_stability as stability


def effective_providers(inst: Instance, mu: Matching) -> List[Tuple[Student, School]]:
    """P = {(s, c) in mu : z^mu(s,c) = 1}, ordered by increasing lottery."""
    prov = stability.providers(inst, mu)
    P = [(s, c) for (c, _f), s in prov["eff"].items()]
    P.sort(key=lambda sc: inst.lottery[sc])
    return P


def update_priorities_key(inst: Instance, mu: Matching,
                          providers: Optional[List[Tuple[Student, School]]] = None
                          ) -> Callable[[Student, School], tuple]:
    """DA comparison key for succ' = UpdatePriorities(mu), using the CURRENT ACS
    definition (Definition 2, v2_stability.absolute_group): a provider keeps its
    group when co-assigned OR when it has a same-level receiver, and receivers
    of the family's effective provider matched to c keep their group. `providers`
    restricts the effective-provider set to a chosen subsequence (search
    variant); None uses all effective providers. Lower key is better."""
    ann = stability.precompute(inst, mu, "absolute")        # eff, qual, recv
    if providers is not None:
        allowed = set(providers)
        ann = dict(ann)
        ann["eff"] = {k: s for k, s in ann["eff"].items() if (s, k[0]) in allowed}

    def key(s: Student, c: School) -> tuple:
        return stability.order_key(inst, mu, ann, s, c, "absolute")

    return key


def rada(inst: Instance, max_iter: int = 10000) -> Tuple[Matching, dict]:
    """Algorithm RADA. info['status'] in {'FixedPoint','Cycle','MaxIter'};
    info['iters'] = t at return. On Cycle the returned matching is the repeated
    one, per the pseudocode, and info also carries the cycle shape:
    cycle_len, and the number of families / students / schools that move across
    the cycle (for reporting the cycle-shape distribution)."""
    key = lambda s, c: inst.order_key(s, c)            # succ^0 = initial
    seq: List[Matching] = []                           # matchings mu^0..mu^{t-1}
    sigs: List[frozenset] = []
    t = 0
    while True:
        mu_t = da_with_key(inst, key)                  # mu^t = DA(succ^t)
        sig = frozenset(mu_t.items())
        if t > 0 and sig == sigs[-1]:
            return mu_t, {"status": "FixedPoint", "iters": t, "converged": True}
        if sig in sigs:
            cyc = seq[sigs.index(sig):]                # matchings in the cycle
            fam, stud, sch = set(), set(), set()
            for s in inst.students:
                vals = {m.get(s) for m in cyc}
                if len(vals) > 1:
                    stud.add(s); fam.add(inst.family_of[s])
                    sch |= {v for v in vals if v is not None}
            return mu_t, {"status": "Cycle", "iters": t, "converged": False,
                          "cycle": True, "cycle_len": len(cyc),
                          "cycle_families": len(fam), "cycle_students": len(stud),
                          "cycle_schools": len(sch)}
        key = update_priorities_key(inst, mu_t)        # succ^{t+1}
        seq.append(mu_t); sigs.append(sig)
        t += 1
        if t > max_iter:
            return mu_t, {"status": "MaxIter", "iters": t, "converged": False,
                          "hit_max": True}


def _rada_from(inst: Instance, mu0: Matching, update, max_iter: int = 2000):
    """RADA loop from an arbitrary seed mu0 with a given update-key builder."""
    sigs: List[frozenset] = []
    mu = mu0
    t = 0
    while True:
        sig = frozenset(mu.items())
        if t > 0 and sig == sigs[-1]:
            return mu, {"status": "FixedPoint", "iters": t}
        if sig in sigs:
            return mu, {"status": "Cycle", "iters": t}
        key = update(inst, mu)
        sigs.append(sig)
        mu = da_with_key(inst, key)
        t += 1
        if t > max_iter:
            return mu, {"status": "MaxIter", "iters": t}


def update_priorities_key_sequential(inst: Instance, mu: Matching,
                                    deadline: Optional[float] = None):
    """Sequential-commitment update: process effective providers in lottery
    order and keep a provider's boost only if, after re-running DA, every
    committed provider still holds her seat. This suppresses the simultaneous
    provision that produces 2-cycles. Costs one DA per provider, so it is heavy
    at large scale; `deadline` (a perf_counter timestamp) stops adding providers
    once the budget is spent and returns the update built so far."""
    import time as _time
    committed: List[Tuple[Student, School]] = []
    for p in effective_providers(inst, mu):
        if deadline is not None and _time.perf_counter() > deadline:
            break
        trial = committed + [p]
        mu2 = da_with_key(inst, update_priorities_key(inst, mu, providers=trial))
        if all(mu2.get(q[0]) == mu.get(q[0]) for q in trial):
            committed = trial
    return update_priorities_key(inst, mu, providers=committed)


def rada_portfolio(inst: Instance, use_sequential: bool = False,
                   max_iter: int = 2000) -> Tuple[Optional[Matching], dict]:
    """Run RADA from several seeds (SOSM, Descending, Ascending, LSDA), optionally
    crossed with the sequential update, and return the first ACS fixed point.
    Cheap when use_sequential is False (four fast RADA runs)."""
    import v2_heuristics as heuristics
    seeds = [heuristics.deferred_acceptance(inst), heuristics.descending(inst),
             heuristics.ascending(inst), heuristics.lsda(inst)]
    updates = [update_priorities_key]
    if use_sequential:
        updates.append(update_priorities_key_sequential)
    last = None
    tries = 0
    for mu0 in seeds:
        for upd in updates:
            tries += 1
            mu, info = _rada_from(inst, mu0, upd, max_iter)
            last = mu
            if info["status"] == "FixedPoint" \
                    and stability.is_contingent_stable(inst, mu, "absolute"):
                return mu, {"status": "FixedPoint", "converged": True,
                            "iters": info["iters"], "tries": tries}
    return last, {"status": "Cycle", "converged": False, "cycle": True,
                  "tries": tries}


def generate_provider_sequences(inst: Instance, mu: Matching, max_seq: int = 20000,
                                deadline: Optional[float] = None
                                ) -> List[List[Tuple[Student, School]]]:
    """Sigma(mu): maximal subsequences sigma of the effective providers
    (lottery order) such that applying only sigma's boosts and running DA keeps
    every provider in sigma at her original assignment mu(provider). Bounded by
    max_seq and by `deadline` (a perf_counter timestamp); if hit, the sequences
    found so far are returned and the search becomes heuristic."""
    import time as _time
    order = effective_providers(inst, mu)              # (q_1, ..., q_m)
    m = len(order)
    Sigma: List[List[Tuple[Student, School]]] = []
    stack: List[Tuple[List[Tuple[Student, School]], int]] = [([], 0)]
    while stack:
        if len(Sigma) >= max_seq:
            break
        if deadline is not None and _time.perf_counter() > deadline:
            break
        sigma, j = stack.pop()
        extended = False
        for t in range(j, m):
            if deadline is not None and _time.perf_counter() > deadline:
                break                                  # stop between DA solves
            sigma2 = sigma + [order[t]]
            key = update_priorities_key(inst, mu, providers=sigma2)
            mu2 = da_with_key(inst, key)
            if all(mu2.get(q[0]) == mu.get(q[0]) for q in sigma2):
                stack.append((sigma2, t + 1))
                extended = True
        if not extended:
            Sigma.append(sigma)
    return Sigma


def rada_sequential_search(inst: Instance, max_nodes: int = 50000,
                           max_seq: int = 20000, time_limit: float = 120.0
                           ) -> Tuple[Optional[Matching], dict]:
    """Algorithm RADA with Sequential Search. Returns (mu, {'status':
    'FixedPoint'}) for a reachable fixed point, or (None, {'status':'Cycle'})
    if every reachable sequence cycles. Because the search is worst-case
    exponential in the number of providers, it is bounded by max_nodes, max_seq
    (per sequence-generation call), and time_limit seconds; if a bound is hit it
    returns status 'MaxNodes'/'TimeLimit' with capped=True and the search is then
    only a heuristic (it may report a cycle where a fixed point is reachable)."""
    import time as _time
    t0 = _time.perf_counter()
    mu0 = deferred_acceptance(inst)

    def sig(mu):
        return frozenset(mu.items())

    E = set()
    R = set()
    Sigma0 = generate_provider_sequences(inst, mu0, max_seq, t0 + time_limit)
    K: List[list] = [[mu0, Sigma0, 0, sig(mu0)]]
    K_sigs = {sig(mu0)}
    nodes = 0
    while K:
        nodes += 1
        if nodes > max_nodes:
            return None, {"status": "MaxNodes", "nodes": nodes, "capped": True}
        if _time.perf_counter() - t0 > time_limit:
            return None, {"status": "TimeLimit", "nodes": nodes, "capped": True}
        mu, Sigma, k, musig = K[-1]
        if k >= len(Sigma):
            E.add(musig)
            K.pop()
            K_sigs.discard(musig)
            continue
        sigma_k = Sigma[k]
        K[-1][2] = k + 1
        rkey = (musig, tuple(sigma_k))
        if rkey in R:
            continue
        R.add(rkey)
        key = update_priorities_key(inst, mu, providers=sigma_k)
        mu2 = da_with_key(inst, key)
        s2 = sig(mu2)
        if s2 == musig:
            full = da_with_key(inst, update_priorities_key(inst, mu))
            if sig(full) == musig:
                return mu, {"status": "FixedPoint", "nodes": nodes,
                            "converged": True}
            continue
        if s2 in E or s2 in K_sigs:
            continue
        Sigma2 = generate_provider_sequences(inst, mu2, max_seq, t0 + time_limit)
        K.append([mu2, Sigma2, 0, s2])
        K_sigs.add(s2)
    return None, {"status": "Cycle", "nodes": nodes, "converged": False,
                  "cycle": True}


# ==========================================================================
# Ignacio's variant: contingent priority updates the LOTTERY, not the group
# ==========================================================================
def update_priorities_lottery_key(inst: Instance, mu: Matching, epsilon: float = 1e-7):
    """Update rule that boosts the tie-break (lottery) instead of the group: each
    effective provider's receivers inherit the provider's lottery value (plus a
    small epsilon so they sit just behind the provider), and DA runs on a single
    priority order given by the updated lotteries. The group / standing machinery
    is dropped. This is the fixed-epsilon inheritance scheme (the commented-out
    lines of UpdatePriorities). Provider and receiver identification is kept as
    in Definition 1; confirm with Ignacio whether his version also simplifies
    that."""
    plot = dict(inst.lottery)
    for (s, c) in sorted(effective_providers(inst, mu), key=lambda sc: inst.lottery[sc]):
        base = inst.lottery[(s, c)]
        recs = sorted((sp for sp in inst.siblings(s) if inst.weakly_prefers(sp, c, mu)),
                      key=lambda sp: inst.lottery[(sp, c)])
        for i, sp in enumerate(recs):
            plot[(sp, c)] = min(plot[(sp, c)], base) + (i + 1) * epsilon

    def key(s: Student, c: School) -> tuple:
        return (plot.get((s, c), inst.lottery[(s, c)]),)

    return key


def rada_lottery(inst: Instance, max_iter: int = 2000) -> Tuple[Matching, dict]:
    """RADA with the lottery-inheritance update (Ignacio's variant)."""
    mu, info = _rada_from(inst, deferred_acceptance(inst),
                          update_priorities_lottery_key, max_iter)
    info["converged"] = info["status"] == "FixedPoint"
    if info["status"] == "Cycle":
        info["cycle"] = True
    return mu, info


# ==========================================================================
# RADA-sequential: sequential-commitment update; lowest-avg-rank on a cycle
# ==========================================================================
def rada_sequential(inst: Instance, max_iter: int = 2000,
                    time_limit: float = 300.0) -> Tuple[Matching, dict]:
    """RADA using the sequential-commitment update. It checks ACS at every
    iterate and returns the first ACS matching found; if the sequence revisits a
    matching without finding one, it returns the lowest-average-rank matching
    among those seen. The sequential update does one DA per provider per
    iteration, so it is heavy at region scale; time_limit bounds the whole run
    and, on hitting it, returns the lowest-rank matching seen so far (capped)."""
    import time as _time
    t0 = _time.perf_counter()
    seq: List[Matching] = []
    sigs: List[frozenset] = []
    mu = deferred_acceptance(inst)
    for t in range(max_iter):
        if stability.is_contingent_stable(inst, mu, "absolute"):
            return mu, {"status": "FixedPoint", "iters": t + 1, "converged": True}
        sig = frozenset(mu.items())
        if sig in sigs:
            seen = seq + [mu]
            return min(seen, key=lambda m: total_rank(inst, m)), {
                "status": "Cycle", "iters": t + 1, "converged": False,
                "cycle": True, "cycle_len": len(seen) - sigs.index(sig) - 1,
                "picked": "min_total_rank"}
        seq.append(mu); sigs.append(sig)
        if _time.perf_counter() - t0 > time_limit:
            return min(seq, key=lambda m: total_rank(inst, m)), {
                "status": "TimeLimit", "iters": t + 1, "converged": False,
                "capped": True, "picked": "min_total_rank"}
        key = update_priorities_key_sequential(inst, mu, deadline=t0 + time_limit)
        mu = da_with_key(inst, key)
    return min(seq + [mu], key=lambda m: total_rank(inst, m)), {
        "status": "MaxIter", "iters": max_iter, "converged": False}
