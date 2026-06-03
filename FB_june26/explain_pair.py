"""
explain_pair.py
===============
Drill-down diagnostic for a single (student, school) pair, showing exactly why
the OLD verifier (stability_check.check_absolute_stability) and the NEW verifier
(acs_verifier.check_acs) disagree on whether it is a blocking pair.

For the school c and the blocking student s, it prints, for s and for every
occupant of c at s's level:
  - tb (lottery) at RBD(c)
  - OLD: provides? receives? -> contingent group (1=prioritized, 2=not)
  - NEW: provider? base-admissible? effective-provider z? lone-provider? -> tier
  - whether each occupant outranks s under OLD and under NEW

Then it states each verifier's verdict at (s, c) and names the WITNESS
occupants — those the old verifier keeps above s but the new verifier (correctly)
puts below s, i.e. the ones whose contingent priority is invalid under
Definitions 1-3.

----------------------------------------------------------
    from explain_pair import explain_pair
    # mu : dict student -> school_id or None
    explain_pair(mu, s="15359262", c="11678_11",
                 students=students, colleges=colleges, pref=pref_with_tb,
                 cap=cap, siblings=siblings, levels_of=levels_of, tb=tb)

To rebuild mu from a method's x_opt:
    from simulations_checker import x_opt_to_mu, build_levels_of
    mu = x_opt_to_mu(out["x_opt"], students)
    levels_of = build_levels_of(students, pref_with_tb)
"""

import acs_verifier as NEW

# Old verifier: prefer the project's stability_check; fall back to local copy.
try:
    import stability_check as OLD
except Exception:
    try:
        import old_stability_check as OLD
    except Exception:
        OLD = None


def _fmt(x):
    return "yes" if x else " . "


def explain_pair(mu, s, c, students, colleges, pref, cap, siblings, levels_of, tb,
                 old_mod=None):
    old = old_mod if old_mod is not None else OLD
    rbd = NEW.rbd_of(c)
    level_s = levels_of.get(s)
    q = cap.get(c, 0)

    student_rank = NEW._build_rank(pref, students)
    school_rank = NEW._build_rank(pref, colleges)

    # NEW: compute z (effective providers) once for the whole instance
    z = NEW._compute_z(mu, colleges, siblings, students, school_rank,
                       student_rank, levels_of, cap, tb=tb)

    occupants = [o for o in students
                 if mu.get(o) == c and levels_of.get(o) == level_s]

    print("=" * 88)
    print(f"EXPLAIN  s={s}  ->  c={c}   (RBD={rbd}, level={level_s}, capacity q={q})")
    print(f"  s currently matched to: {mu.get(s)}")
    sr = student_rank.get(s, {})
    print(f"  s prefers c to own match: "
          f"{NEW._strictly_prefers(s, c, mu.get(s), student_rank)}  "
          f"(rank_s(c)={sr.get(c)}, rank_s(own)={sr.get(mu.get(s))})")
    print(f"  occupants of c at level {level_s}: {occupants}  "
          f"({len(occupants)} of q={q})")
    print("-" * 88)

    def new_provider(x):
        return NEW._is_provider_at_rbd(x, rbd, mu, siblings, students,
                                       school_rank, student_rank, levels_of, cap)

    def new_base_adm(x):
        return NEW._is_base_admissible_at(x, rbd, mu, students, school_rank,
                                          student_rank, levels_of, cap)

    def new_tier(x):
        return "UPPER" if NEW._in_upper_tier(x, c, mu, z, siblings) else "lower"

    def new_z(x):
        return z.get(rbd, {}).get(x, 0) == 1

    def new_lone(x):
        # effective provider but no co-assigned family member at RBD
        if not new_z(x):
            return False
        fam = NEW._family_of(x, siblings)
        n_at_rbd = sum(1 for fm in fam
                       if mu.get(fm) is not None and NEW.rbd_of(mu[fm]) == rbd)
        return n_at_rbd < 2

    def row(x, is_s=False):
        tbv = tb.get(x, {}).get(rbd)
        tbv_s = f"{tbv:.4f}" if isinstance(tbv, (int, float)) else str(tbv)
        if old is not None:
            o_prov = old._provides_contingent_priority(x, c, mu, siblings, pref)
            o_recv = old._receives_contingent_priority(x, c, mu, siblings)
            o_grp = old._absolute_contingent_group(x, c, mu, siblings, pref)
        else:
            o_prov = o_recv = o_grp = None
        n_prov = new_provider(x)
        n_badm = new_base_adm(x)
        n_zz = new_z(x)
        n_lone = new_lone(x)
        n_tier = new_tier(x)

        tag = "  <-- s" if is_s else ""
        print(f"  {str(x):>10}{tag:8} tb={tbv_s:>9}   "
              f"OLD[prov={_fmt(o_prov)} recv={_fmt(o_recv)} grp={o_grp}]   "
              f"NEW[prov={_fmt(n_prov)} baseadm={_fmt(n_badm)} "
              f"z={_fmt(n_zz)} lone={_fmt(n_lone)} tier={n_tier}]")

    print("  student        lottery       OLD group rule            NEW tier rule")
    row(s, is_s=True)
    for o in occupants:
        row(o)
    print("-" * 88)

    # Pairwise: does each occupant outrank s, under each verifier?
    print("  occupant      OLD: o>s?   NEW: o>s?   note")
    old_above = 0
    new_above = 0
    witnesses = []
    fam_s = set(siblings.get(s, []))
    for o in occupants:
        if old is not None:
            o_out_old = old._outranks_absolute(o, s, c, mu, siblings, pref, tb)
        else:
            o_out_old = None
        o_out_new = NEW._contingent_outranks(o, s, c, mu, z, siblings, school_rank)
        # OLD's envy loop SKIPS siblings of s, so effectively a sibling never
        # counts as "blocking-relevant"; mark that.
        is_sib = (o in fam_s)
        if o_out_old:
            old_above += 1
        if o_out_new:
            new_above += 1
        note = ""
        if (o_out_old and not o_out_new):
            witnesses.append(o)
            note = "WITNESS (old keeps above s; new demotes below s)"
        elif (not o_out_old and o_out_new):
            note = "new keeps above s; old below"
        if is_sib:
            note = (note + "  [sibling of s: OLD skips in envy loop]").strip()
        print(f"  {str(o):>10}     {_fmt(o_out_old):>6}      {_fmt(o_out_new):>6}      {note}")
    print("-" * 88)

    # Verdicts
    # OLD: blocking iff s outranks some NON-sibling occupant (i.e. some non-sib
    #      occupant does NOT outrank s). Equivalent restatement for display:
    old_nonsib_below = [o for o in occupants
                        if o not in fam_s and old is not None
                        and not old._outranks_absolute(o, s, c, mu, siblings, pref, tb)]
    old_block = len(old_nonsib_below) > 0
    # NEW: blocking iff count of occupants outranking s < q
    new_block = (new_above < q)

    print(f"  OLD verdict at (s,c): "
          f"{'BLOCK' if old_block else 'no block'}  "
          f"(s outranks non-sibling occupants {old_nonsib_below or '[]'})")
    print(f"  NEW verdict at (s,c): "
          f"{'BLOCK' if new_block else 'no block'}  "
          f"(occupants outranking s: {new_above} < q={q}? "
          f"{new_above < q})")
    if witnesses:
        print(f"  >>> WITNESS occupant(s): {witnesses}")
        print(f"      These hold a seat via contingent priority that is INVALID under")
        print(f"      Definitions 1-3 (not base-admissible, or a lone provider, or")
        print(f"      receiving from a non-admissible sibling). The old verifier")
        print(f"      promotes them to group 1; the new verifier correctly demotes them.")
    print("=" * 88)

    return {
        "old_block": old_block,
        "new_block": new_block,
        "witnesses": witnesses,
        "old_above": old_above,
        "new_above": new_above,
        "q": q,
    }