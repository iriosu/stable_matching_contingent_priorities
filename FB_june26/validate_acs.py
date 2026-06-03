"""
validate_acs.py
===============
Hand-computed validation of acs_verifier.py against:
  - Example 1 from the paper ("unlawful siblings priority"): the three
    matchings µ, µ', µ'' with known ACS status.
  - Targeted micro-tests for: trivial stability, wastefulness, simple envy,
    sibling priority resolving envy, cross-level (RBD) sibling priority, and
    base-admissibility (iii) flipping a case.
  - Example from a companion document (split vs co-assignment under DA).

Each test prints PASS/FAIL against the hand-computed expected result.
"""

from acs_verifier import check_acs, summarize

PASS = "PASS"
FAIL = "FAIL"


def run(name, expected_is_acs, mu, students, colleges, pref, cap, siblings,
        levels_of, tb=None, show_blocks=False):
    res = check_acs(mu, students, colleges, pref, cap, siblings, levels_of, tb=tb)
    ok = (res["is_acs"] == expected_is_acs)
    tag = PASS if ok else FAIL
    print(f"[{tag}] {name}")
    print(f"       expected is_acs={expected_is_acs}, got is_acs={res['is_acs']}")
    print(f"       {summarize(res)}")
    if show_blocks or not ok:
        for b in res["blocking_pairs"][:8]:
            print(f"         block: s={b['s']} c={b['c']} type={b['type']} "
                  f"count={b['count_above']}/q={b['q']} witness={b['witness_s_prime']} "
                  f"base_adm_s={b['base_admissible_s']}")
    print()
    return ok


results = []


# ============================================================
# Example 1 (main paper): single school c, capacity 4, single level.
#   Students: s1,s2,s3 (singletons); f1,f2 (family f); F1,F2 (family f').
#   Base priority at c: s1 ≻ s2 ≻ s3 ≻ f1 ≻ F1 ≻ f2 ≻ F2
#   (lottery: p_s1<p_s2<p_s3<p_f1<p_F1<p_f2<p_F2; lower p = better).
#   All students list only c (and prefer it to ∅).
# ============================================================

C = "100_1"           # RBD 100, level 1
LVL = "1"
EX1_students = ["s1", "s2", "s3", "f1", "f2", "F1", "F2"]
EX1_colleges = [C]
EX1_cap = {C: 4}
EX1_levels = {s: LVL for s in EX1_students}
EX1_siblings = {
    "s1": [], "s2": [], "s3": [],
    "f1": ["f2"], "f2": ["f1"],
    "F1": ["F2"], "F2": ["F1"],
}
# student preferences: everyone lists only C
EX1_pref = {s: {1: C} for s in EX1_students}
# school base priority at C
EX1_pref[C] = {1: "s1", 2: "s2", 3: "s3", 4: "f1", 5: "F1", 6: "f2", 7: "F2"}
# tb at RBD 100 consistent with base order (higher = better)
EX1_tb = {
    "s1": {"100": 7.0}, "s2": {"100": 6.0}, "s3": {"100": 5.0},
    "f1": {"100": 4.0}, "F1": {"100": 3.0}, "f2": {"100": 2.0}, "F2": {"100": 1.0},
}

# µ  (partial-priority outcome): s1,s2,s3,f1 in; f2,F1,F2 out  -> NOT ACS
mu_1 = {"s1": C, "s2": C, "s3": C, "f1": C, "f2": None, "F1": None, "F2": None}
# µ' (undesirable): f1,f2,F1,F2 in; s1,s2,s3 out               -> NOT ACS
mu_2 = {"s1": None, "s2": None, "s3": None, "f1": C, "f2": C, "F1": C, "F2": C}
# µ''(absolute-priority outcome): s1,s2,f1,f2 in; s3,F1,F2 out -> ACS
mu_3 = {"s1": C, "s2": C, "s3": None, "f1": C, "f2": C, "F1": None, "F2": None}

results.append(run("Ex1 µ  (partial outcome; f2 has contingent envy)", False,
                    mu_1, EX1_students, EX1_colleges, EX1_pref, EX1_cap,
                    EX1_siblings, EX1_levels, tb=EX1_tb, show_blocks=True))
results.append(run("Ex1 µ' (undesirable; s1,s2,s3 envy F1,F2)", False,
                    mu_2, EX1_students, EX1_colleges, EX1_pref, EX1_cap,
                    EX1_siblings, EX1_levels, tb=EX1_tb, show_blocks=True))
results.append(run("Ex1 µ''(absolute outcome; stable)", True,
                    mu_3, EX1_students, EX1_colleges, EX1_pref, EX1_cap,
                    EX1_siblings, EX1_levels, tb=EX1_tb, show_blocks=True))


# ============================================================
# Micro-test A: trivial stable. 1 school cap 1, 1 student matched.
# ============================================================
A_students = ["s"]
A_colleges = ["10_1"]
A_pref = {"s": {1: "10_1"}, "10_1": {1: "s"}}
A_cap = {"10_1": 1}
A_sib = {"s": []}
A_lvl = {"s": "1"}
results.append(run("A trivial stable (s matched to only school)", True,
                    {"s": "10_1"}, A_students, A_colleges, A_pref, A_cap,
                    A_sib, A_lvl))


# ============================================================
# Micro-test B: trivial wasteful. Same as A but s unmatched.
# ============================================================
results.append(run("B trivial wasteful (s lists school but unmatched)", False,
                    {"s": None}, A_students, A_colleges, A_pref, A_cap,
                    A_sib, A_lvl, show_blocks=True))


# ============================================================
# Micro-test C: simple envy, no siblings.
#   s1,s2 singletons level 1. School cap 1. Base s1 ≻ s2.
#   µ: s2 in, s1 out -> s1 envies s2 -> NOT ACS.
# ============================================================
C_students = ["s1", "s2"]
C_colleges = ["20_1"]
C_pref = {"s1": {1: "20_1"}, "s2": {1: "20_1"}, "20_1": {1: "s1", 2: "s2"}}
C_cap = {"20_1": 1}
C_sib = {"s1": [], "s2": []}
C_lvl = {"s1": "1", "s2": "1"}
results.append(run("C simple envy (s1 outranks s2 but s2 is in)", False,
                    {"s1": None, "s2": "20_1"}, C_students, C_colleges,
                    C_pref, C_cap, C_sib, C_lvl, show_blocks=True))
# and the stable version: s1 in, s2 out -> ACS
results.append(run("C' simple stable (s1 in, s2 out)", True,
                    {"s1": "20_1", "s2": None}, C_students, C_colleges,
                    C_pref, C_cap, C_sib, C_lvl))


# ============================================================
# Micro-test D: cross-level sibling priority resolves envy.
#   Family {a1 (level 1), a2 (level 2)}. Singleton t (level 2).
#   RBD 30: school 30_1 (cap 1), 30_2 (cap 1).
#   Base priority at 30_1: a1 top. At 30_2: t ≻ a2.
#   µ: a1->30_1, a2->30_2, t->None.
#   Without sibling priority t would envy a2 (t ≻ a2 at 30_2).
#   With it: a1 at RBD 30 provides priority to a2 -> a2 upper-tier at 30_2,
#   t lower-tier -> a2 ≻^µ_{30_2} t -> no justified envy -> ACS.
# ============================================================
D_students = ["a1", "a2", "t"]
D_colleges = ["30_1", "30_2"]
D_pref = {
    "a1": {1: "30_1"},
    "a2": {1: "30_2"},
    "t":  {1: "30_2"},
    "30_1": {1: "a1"},
    "30_2": {1: "t", 2: "a2"},   # t outranks a2 in base priority
}
D_cap = {"30_1": 1, "30_2": 1}
D_sib = {"a1": ["a2"], "a2": ["a1"], "t": []}
D_lvl = {"a1": "1", "a2": "2", "t": "2"}
D_tb = {  # tb at RBD 30
    "t":  {"30": 9.0},
    "a2": {"30": 1.0},
    "a1": {"30": 5.0},
}
results.append(run("D cross-level sibling priority resolves envy", True,
                    {"a1": "30_1", "a2": "30_2", "t": None},
                    D_students, D_colleges, D_pref, D_cap, D_sib, D_lvl,
                    tb=D_tb, show_blocks=True))

# D-control: if a1 is NOT base-admissible at RBD 30 (so cannot be a provider),
#   then a2 stays lower-tier and t's envy is justified -> NOT ACS.
#   Make a1 not base-admissible at 30_1 by adding a higher-priority student x
#   at level 1 who fills the slot and wants 30_1.
D2_students = ["a1", "a2", "t", "x"]
D2_colleges = ["30_1", "30_2"]
D2_pref = {
    "a1": {1: "30_1"},
    "a2": {1: "30_2"},
    "t":  {1: "30_2"},
    "x":  {1: "30_1"},
    "30_1": {1: "x", 2: "a1"},   # x outranks a1; cap 1, so a1 not base-admissible
    "30_2": {1: "t", 2: "a2"},
}
D2_cap = {"30_1": 1, "30_2": 1}
D2_sib = {"a1": ["a2"], "a2": ["a1"], "t": [], "x": []}
D2_lvl = {"a1": "1", "a2": "2", "t": "2", "x": "1"}
D2_tb = {"t": {"30": 9.0}, "a2": {"30": 1.0}, "a1": {"30": 5.0}, "x": {"30": 8.0}}
# µ: a1->30_1 (but x outranks a1 and wants it -> a1 itself blocks? no: a1 is at 30_1,
#    x is unmatched and wants 30_1 with higher priority -> x envies a1).
# To isolate the a2/t question, put x in: x->30_1, a1->None, a2->30_2, t->None.
# Now a1 is unmatched, so a1 is not a provider (not at RBD 30). a2 has no provider.
# t (base ≻ a2) envies a2 at 30_2 -> NOT ACS.
results.append(run("D2 control: no valid provider, t's envy stands", False,
                    {"a1": None, "a2": "30_2", "t": None, "x": "30_1"},
                    D2_students, D2_colleges, D2_pref, D2_cap, D2_sib, D2_lvl,
                    tb=D2_tb, show_blocks=True))


# ============================================================
# Micro-test E: base-admissibility (iii) is what flips Ex1 µ'.
#   This is already covered by another example µ' (F1,F2 are NOT base-admissible, so they
#   cannot be providers, so s1,s2,s3 retain justified envy). Re-stated as a
#   focused 1-family test:
#   RBD 40 cap 1. Students: hi (good lottery, singleton), b1,b2 (family, bad lottery).
#   Base priority at 40_1: hi ≻ b1 ≻ b2.
#   µ: b1->40_1, b2->None, hi->None.
#   b1 wants to provide priority to b2, but is b1 base-admissible? Students ≻ b1
#   who weakly prefer 40_1: {hi}. |{hi}| = 1 > q-1 = 0. NOT base-admissible.
#   So b1 is not a provider; b2 gets no priority. hi (base ≻ b1) envies b1 -> NOT ACS.
# ============================================================
E_students = ["hi", "b1", "b2"]
E_colleges = ["40_1"]
E_pref = {
    "hi": {1: "40_1"}, "b1": {1: "40_1"}, "b2": {1: "40_1"},
    "40_1": {1: "hi", 2: "b1", 3: "b2"},
}
E_cap = {"40_1": 1}
E_sib = {"hi": [], "b1": ["b2"], "b2": ["b1"]}
E_lvl = {"hi": "1", "b1": "1", "b2": "1"}
E_tb = {"hi": {"40": 9.0}, "b1": {"40": 2.0}, "b2": {"40": 1.0}}
results.append(run("E base-admissibility blocks bad provider (hi envies b1)", False,
                    {"hi": None, "b1": "40_1", "b2": None},
                    E_students, E_colleges, E_pref, E_cap, E_sib, E_lvl,
                    tb=E_tb, show_blocks=True))
# E': stable version under absolute priority: hi in, b1,b2 out -> ACS
results.append(run("E' stable (hi takes the only slot)", True,
                    {"hi": "40_1", "b1": None, "b2": None},
                    E_students, E_colleges, E_pref, E_cap, E_sib, E_lvl,
                    tb=E_tb))


# ============================================================
# Summary
# ============================================================
n_pass = sum(results)
n_total = len(results)
print("=" * 60)
print(f"VALIDATION SUMMARY: {n_pass}/{n_total} tests passed")
print("=" * 60)
if n_pass != n_total:
    raise SystemExit(1)