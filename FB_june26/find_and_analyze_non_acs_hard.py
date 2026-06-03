"""
find_and_analyze_non_acs_hard.py
================================
Search Magallanes sims (seeds 0, 1, 2, ...) until we find one where
Absolute-Hard solves but the ACS verifier reports a non-ACS matching.
For that sim:
  - save the full instance (students, colleges, pref, cap, siblings,
    levels_of, tb) and the matching (mu, x_opt) to pickle on disk;
  - save the verifier's blocking pairs as JSON;
  - run a step-by-step analysis that walks the verifier's logic against
    Definitions 1-4 of the paper, line by line, to determine whether
    each ACS condition is computed correctly on this concrete witness.

Outputs land at
   <python>/../outputs/non_acs_hard/<date>/sim_<NNN>/
       instance.pkl
       matching.pkl
       blocking_pairs.json
       analysis.txt

Run:  python find_and_analyze_non_acs_hard.py
"""

import os
import sys
import pickle
import json
import copy
import random
import datetime
import numpy as np

import simulations_acs as S
import acs_verifier
import acs_priority as P


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_BASE   = os.path.join(SCRIPT_DIR, "..", "outputs", "non_acs_hard",
                          datetime.date.today().isoformat())
MAX_SEEDS  = 100        # search horizon


# ----------------------------------------------------------------------
# input loading (replicates simulations_acs.run_one_sim's construction)
# ----------------------------------------------------------------------

def load_sim(sim_idx):
    np.random.seed(sim_idx + 1)
    random.seed(sim_idx + 1)
    region_indir = os.path.join(S.REGION_ROOT, S.REGION)
    instance_file = os.path.join(region_indir, str(S.YEAR), "instance.txt")
    (students, colleges, pref, cap, siblings, levels,
     students_per_level, Tp, Tn, Sp, Sn) = S.genin.read_instance(instance_file)
    pref_tb, tb = S.genin.modify_school_loterries(
        copy.deepcopy(pref), students, colleges, siblings, S.TIE_BREAKER)
    _, _, Tp2, Tn2, Sp2, Sn2 = \
        S.genin.create_additional_inputs_from_instance(pref_tb, cap)
    inputs_full = (students, colleges, pref_tb, cap, siblings,
                   levels, students_per_level, Tp2, Tn2, Sp2, Sn2, tb)
    levels_of = P.build_levels_of(students, pref_tb)
    return students, colleges, pref_tb, cap, siblings, levels_of, tb, inputs_full


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _weakly_wants_rbd(s, r, pref, mu, student_rank, levels_of):
    """Does s weakly prefer (r, ℓ(s)) to µ(s)? (Definition 2(ii))."""
    sid = P.school_id_at(r, levels_of.get(s, ""))
    if sid not in set(pref.get(s, {}).values()):
        return False
    mu_s = mu.get(s)
    if mu_s is None:
        return True
    return P.weakly_prefers(s, sid, mu_s, student_rank)


# ----------------------------------------------------------------------
# Deep, step-by-step analysis of ONE blocking pair against Defs 1-4
# ----------------------------------------------------------------------

def deep_analyze(sim, mu, bp, ctx, fh):
    students, colleges, pref, cap, siblings, levels_of, tb = ctx
    s, c       = bp["s"], bp["c"]
    q, level_s = bp["q"], bp["level"]
    r          = P.rbd_of(c)

    student_rank = P.build_rank(pref, students)
    school_rank  = P.build_rank(pref, colleges)
    z = P.compute_z(mu, colleges, siblings, students, school_rank,
                    student_rank, levels_of, cap, tb=tb)

    def p(*a): print(*a, file=fh)

    p("\n" + "=" * 72)
    p(f"BLOCKING PAIR  s={s}   c={c}   (sim {sim})")
    p("=" * 72)

    # ---- 1. Preference: is c strictly preferred to µ(s)? ----
    mu_s   = mu.get(s)
    rk_c   = student_rank.get(s, {}).get(c)
    rk_mu  = student_rank.get(s, {}).get(mu_s) if mu_s is not None else None
    p("\n[1] Preference  (c ≻_s µ(s))")
    p(f"    µ(s) = {mu_s}   rank_s(µ(s)) = {rk_mu}")
    p(f"    c    = {c}   rank_s(c)    = {rk_c}")
    pref_ok = (rk_c is not None) and (rk_mu is None or rk_c < rk_mu)
    p(f"    c ≻_s µ(s)?  {pref_ok}")

    # ---- 2. s's family ----
    fam_s = list(siblings.get(s, []))
    p(f"\n[2] s's family  (|fam| = {len(fam_s)+1})")
    for s_i in [s] + fam_s:
        p(f"    {s_i:>14}  ℓ={levels_of.get(s_i)}   µ={mu.get(s_i)}   "
          f"wants RBD {r}? {_weakly_wants_rbd(s_i, r, pref, mu, student_rank, levels_of)}")

    # ---- 3. Occupants of c at level ℓ(s) ----  (Definition 1)
    occ = [t for t, c_t in mu.items() if c_t == c and levels_of.get(t) == level_s]
    p(f"\n[3] Def 1 — same-level occupants of c at ℓ(s)={level_s}")
    p(f"    occ = {occ}   |occ| = {len(occ)}   q_c^{{ℓ}} = {q}")
    p(f"    slot is {'FULL' if len(occ) >= q else 'NOT FULL'} "
      f"→ {'envy candidate' if len(occ) >= q else 'wasteful candidate'}")

    # ---- 4. Defs 2-4 broken down per occupant ----
    p(f"\n[4] Defs 2-4 — per-occupant tier breakdown at c (rbd r={r})")
    p( "    s' upper iff (z[r][s']=1 AND ≥2 family members co-assigned at r) "
       "OR (some sibling has z=1)")
    p( "    lone-provider with no co-assigned sibling is DEMOTED (lower tier).")
    p( "")
    p(f"    {'s_prime':>14}  {'base_rk@c':>9} {'prov(2)':>8} "
      f"{'z[r][s_p]':>9} {'|fam@r|':>8} {'sib_z':>6} "
      f"{'upper(4)':>8} {'sib_of_s':>9}  outranks(s_p,s)?")
    p( "    " + "-" * 90)
    for sp in occ:
        z_sp    = z[r].get(sp, 0)
        is_prov = P.is_provider_at_rbd(sp, r, mu, siblings, students,
                                       school_rank, student_rank, levels_of, cap)
        fam_at_r = [t for t in [sp] + list(siblings.get(sp, []))
                    if mu.get(t) is not None and P.rbd_of(mu[t]) == r]
        sib_has_z = any(z[r].get(t, 0) == 1 for t in siblings.get(sp, []))
        upper_sp  = P.in_upper_tier(sp, c, mu, z, siblings)
        outr_sp   = P.contingent_outranks(sp, s, c, mu, z, siblings, school_rank)
        base_rk   = school_rank.get(c, {}).get(sp, "--")
        is_sib    = sp in set(fam_s)
        p(f"    {sp:>14}  {str(base_rk):>9} {str(is_prov):>8} "
          f"{z_sp:>9} {len(fam_at_r):>8} {str(sib_has_z):>6} "
          f"{str(upper_sp):>8} {str(is_sib):>9}  {outr_sp}")

    # ---- 5. s's own tier at c ----
    p(f"\n[5] s's own status at c")
    p(f"    z[r][s] = {z[r].get(s, 0)}")
    p(f"    in_upper_tier(s, c) = {P.in_upper_tier(s, c, mu, z, siblings)}")
    p(f"    s's base rank at c = {school_rank.get(c, {}).get(s, '--')}")

    # ---- 6. Base-admissibility of s at (r, ℓ(s))  (Def 2(iii)) ----
    sid = P.school_id_at(r, level_s)
    base_adm = P.is_base_admissible_at(s, r, mu, students, school_rank,
                                       student_rank, levels_of, cap)
    p(f"\n[6] Def 2(iii) — base-admissibility of s at "
      f"(r={r}, ℓ={level_s}) → school {sid}: {base_adm}")
    if sid in cap:
        rk_s   = school_rank.get(sid, {}).get(s)
        above  = []
        if rk_s is not None:
            for t in students:
                if t == s or levels_of.get(t) != level_s:
                    continue
                rk_t = school_rank.get(sid, {}).get(t)
                if rk_t is None or rk_t >= rk_s:
                    continue
                mu_t = mu.get(t)
                if mu_t is not None and not P.weakly_prefers(t, mu_t, sid, student_rank):
                    continue
                above.append(t)
        p(f"    |{{t ≻ s at {sid} with µ(t) ⪰_t {sid}}}| = {len(above)}   "
          f"q_{sid} = {cap[sid]}   need ≤ q-1: "
          f"{rk_s is not None and len(above) <= cap[sid] - 1}")

    # ---- 7. Verifier verdict ----
    p(f"\n[7] Verifier verdict on this pair")
    p(f"    count_above = {bp['count_above']}   q = {q}   "
      f"count < q? {bp['count_above'] < q}  →  blocking ({bp['type']})")
    p(f"    witness s' = {bp['witness_s_prime']}   "
      f"witness_is_sibling = {bp['witness_is_sibling']}")

    # ---- 8. Verifier-vs-IP gap explanation ----
    p(f"\n[8] What this implies")
    if bp["witness_is_sibling"]:
        p( "    The witness is a SIBLING of s. The Absolute-Hard IP does not")
        p( "    impose no-justified-envy constraints between two same-level")
        p( "    siblings of the same family (a family does not envy itself).")
        p( "    The verifier here follows the literal Def 4, which has no such")
        p( "    carve-out. Conclusion: each verifier step above is internally")
        p( "    correct; the disagreement is a DEFINITIONAL choice about")
        p( "    whether twin envy counts. exclude_sibling_envy=True aligns the")
        p( "    verifier with the IP/model intent.")
    else:
        p( "    The witness is NOT a sibling of s. This is a genuine ACS")
        p( "    violation under any reading of Def 4: the IP produced a")
        p( "    matching where a non-family student of equal or lesser")
        p( "    contingent priority sits at c while s prefers c and has")
        p( "    higher contingent priority. ⚠ This is a real verifier-vs-IP")
        p( "    gap worth investigating further (IP encoding bug, MIPGap, or")
        p( "    a missing constraint).")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main(max_seeds=MAX_SEEDS):
    os.makedirs(OUT_BASE, exist_ok=True)
    print(f"Output base: {os.path.abspath(OUT_BASE)}")
    print(f"Sweeping {max_seeds} seeds; logging every non-ACS Hard.\n")

    per_sim = []
    n_unsolved = 0
    n_acs = 0
    n_nonacs = 0
    n_all_sibling = 0
    n_fixed_by_exclusion = 0

    for sim in range(max_seeds):
        try:
            (students, colleges, pref, cap, siblings, levels_of, tb,
             inputs_full) = load_sim(sim)
        except Exception as e:
            print(f"sim {sim:3d}: load failed ({e})")
            continue

        out = S.run_hard(inputs_full)
        if not out or not out.get("x_opt"):
            print(f"sim {sim:3d}: Hard UNSOLVED")
            n_unsolved += 1
            continue
        mu = S.x_opt_to_mu(out["x_opt"], students)

        r_lit = acs_verifier.check_acs(mu, students, colleges, pref, cap,
                                       siblings, levels_of, tb=tb,
                                       exclude_sibling_envy=False)
        r_excl = acs_verifier.check_acs(mu, students, colleges, pref, cap,
                                        siblings, levels_of, tb=tb,
                                        exclude_sibling_envy=True)

        if r_lit["is_acs"]:
            print(f"sim {sim:3d}: ACS ✓")
            n_acs += 1
            continue

        n_nonacs += 1
        sib_w = sum(1 for b in r_lit["blocking_pairs"]
                    if b["witness_is_sibling"])
        all_sib = (sib_w == r_lit["n_blocking_pairs"])
        if all_sib:
            n_all_sibling += 1

        # ---------- CONVENTION-ONLY: ACS under exclusion → don't save ----------
        if r_excl["is_acs"]:
            n_fixed_by_exclusion += 1
            print(f"sim {sim:3d}: convention-only  bp={r_lit['n_blocking_pairs']:2d} "
                  f"(all sibling-witness) — not saved")
            per_sim.append({
                "sim": sim,
                "n_blocking_literal": r_lit["n_blocking_pairs"],
                "n_blocking_exclude": r_excl["n_blocking_pairs"],
                "n_sibling_witness": sib_w,
                "all_sibling_witness": all_sib,
                "fixed_by_exclusion": True,
            })
            continue

        # ---------- TRUE ERROR: non-ACS even under exclusion → save and analyze ----------
        sim_dir = os.path.join(OUT_BASE, f"sim_{sim:03d}")
        os.makedirs(sim_dir, exist_ok=True)

        with open(os.path.join(sim_dir, "instance.pkl"), "wb") as fh:
            pickle.dump({
                "sim": sim, "students": students, "colleges": colleges,
                "pref": pref, "cap": cap, "siblings": siblings,
                "levels_of": levels_of, "tb": tb,
            }, fh)
        with open(os.path.join(sim_dir, "matching.pkl"), "wb") as fh:
            pickle.dump({"mu": mu, "x_opt": out["x_opt"]}, fh)
        with open(os.path.join(sim_dir, "blocking_pairs.json"), "w") as fh:
            json.dump(r_lit["blocking_pairs"], fh, indent=2, default=str)

        ctx = (students, colleges, pref, cap, siblings, levels_of, tb)
        ana_path = os.path.join(sim_dir, "analysis.txt")
        with open(ana_path, "w") as fh:
            print(f"TRUE ERROR — sim {sim} "
                  f"(non-ACS even under exclude_sibling_envy=True)", file=fh)
            print(f"  blocking pairs (literal Def 4)        : "
                  f"{r_lit['n_blocking_pairs']}", file=fh)
            print(f"  is_acs under exclude_sibling_envy=True: "
                  f"{r_excl['is_acs']}", file=fh)
            print(f"  blocking pairs under that convention  : "
                  f"{r_excl['n_blocking_pairs']}", file=fh)
            print(f"  blocking pairs with sibling witness   : "
                  f"{sib_w}/{r_lit['n_blocking_pairs']}", file=fh)
            for bp in r_lit["blocking_pairs"]:
                deep_analyze(sim, mu, bp, ctx, fh)

        print(f"sim {sim:3d}: *** TRUE ERROR — bp={r_lit['n_blocking_pairs']:2d}, "
              f"{sib_w}/{r_lit['n_blocking_pairs']} sibling, "
              f"still NON-ACS w/excl ({r_excl['n_blocking_pairs']} bp) ***")

        per_sim.append({
            "sim": sim,
            "n_blocking_literal": r_lit["n_blocking_pairs"],
            "n_blocking_exclude": r_excl["n_blocking_pairs"],
            "n_sibling_witness": sib_w,
            "all_sibling_witness": all_sib,
            "fixed_by_exclusion": False,
        })

    # ----- aggregate summary -----
    n_solved = n_acs + n_nonacs
    n_unexplained = n_nonacs - n_fixed_by_exclusion
    summary_obj = {
        "n_seeds": max_seeds,
        "n_unsolved": n_unsolved,
        "n_solved": n_solved,
        "n_acs_literal": n_acs,
        "n_nonacs_literal": n_nonacs,
        "n_all_sibling_witness": n_all_sibling,
        "n_fixed_by_exclusion": n_fixed_by_exclusion,
        "n_unexplained": n_unexplained,
        "per_sim": per_sim,
    }
    with open(os.path.join(OUT_BASE, "summary.json"), "w") as fh:
        json.dump(summary_obj, fh, indent=2)

    summary_path = os.path.join(OUT_BASE, "summary.txt")
    with open(summary_path, "w") as fh:
        p = lambda *a: print(*a, file=fh)
        p(f"Non-ACS Hard sweep — seeds 0..{max_seeds - 1}")
        p(f"  Hard unsolved              : {n_unsolved}")
        p(f"  Hard solved                : {n_solved}")
        p(f"      ACS (literal Def 4)    : {n_acs}")
        p(f"      NON-ACS (literal Def 4): {n_nonacs}")
        if n_solved:
            p(f"  %ACS literal               : {100*n_acs/n_solved:.1f}")
            p(f"  %ACS exclude-sibling-envy  : "
              f"{100*(n_acs + n_fixed_by_exclusion)/n_solved:.1f}")
        p(f"  All-sibling-witness        : {n_all_sibling}/{n_nonacs}")
        p(f"  Fixed by exclusion         : {n_fixed_by_exclusion}/{n_nonacs}")
        p(f"  UNEXPLAINED (NOT a twin-envy case): {n_unexplained}")
        if n_unexplained:
            p( "  ⚠ The unexplained sims may indicate a real verifier-vs-IP gap.")
            p( "    Inspect their analysis.txt for non-sibling witnesses.")
        p("")
        p("Per non-ACS sim:")
        p(f"  {'sim':>4}  {'bp':>3}  {'sib/bp':>7}  {'all_sib':>7}  "
          f"{'fixed':>5}  {'bp_after_excl':>13}")
        for s in per_sim:
            p(f"  {s['sim']:>4}  {s['n_blocking_literal']:>3}  "
              f"{s['n_sibling_witness']:>3}/{s['n_blocking_literal']:<3}  "
              f"{str(s['all_sibling_witness']):>7}  "
              f"{str(s['fixed_by_exclusion']):>5}  "
              f"{s['n_blocking_exclude']:>13}")

    print("\n" + "=" * 60)
    with open(summary_path) as fh:
        print(fh.read())
    print(f"Summary:   {os.path.abspath(summary_path)}")
    print(f"Per-sim:   {os.path.abspath(OUT_BASE)}/sim_NNN/analysis.txt")
    return summary_obj


if __name__ == "__main__":
    main()