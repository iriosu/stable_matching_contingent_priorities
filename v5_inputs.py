"""
v5_inputs.py

Inputs for the pipeline, self-contained. Three layers:

1. read_instance / write_instance -- the instance.txt format in which the
   Chilean admissions data is distributed with this archive. The reader is
   self-contained, so the package needs nothing outside this folder. The same
   schema is used in every region: sections
   "# Capacities:", "# Student preferences:", "# College priorities:",
   "# Siblings:", "# Levels:", "# Students per Level:", where a "college" is
   an "RBD_LABEL" string (school RBD at level label PreK/K/1/...), student
   preferences and college priorities are "(rank,target)" tokens, and the
   sibling section lists each student followed by their siblings.

2. load_region(region, year, ...) -- read
       <r_root>/intermediate_data/<Region>/<Year>/instance.txt
   once and convert it into a model.Instance:
       college "RBD_L"   -> school RBD, level id from the grade-ordered labels
       cap["RBD_L"]      -> capacity[(RBD, level_id)]
       student pref      -> prefs[s] over RBDs (order preserved, dedup)
       sibling graph     -> families = connected components (transitive
                            reading; the few non-transitive chains merge)
   The base priority in the file is NOT used as a lottery: the simulation
   draws its own via make_lottery and resamples per draw.

3. make_lottery / resample_lottery -- the tie-breaking rules. Each returns a
   lottery value p_{s,c} for every listed (student, school) pair; the school's
   initial order is the priority group first, then this value ascending.
       "individual" -> one draw per student, reused at every school. Single
                       tie-breaking at the individual level (STB).
       "family"     -> one draw per family, reused at every school, with the
                       within-family offsets of footnote 4. Single
                       tie-breaking at the family level (STB-F).
       "mtbf"       -> one draw per (family, school), shared by all members of
                       the family at that school across levels, with the same
                       within-family offsets. Multiple tie-breaking at the
                       family level (MTB-F): the rule used in Chile and behind
                       every table in the paper.
   The offsets are the sufficiently small perturbations of footnote 4: they
   order siblings within a family without disturbing the order across
   families.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

from v5_model import Instance, School, Level, Student


# ==========================================================================
# instance.txt format
# ==========================================================================
_SECTIONS = ("# Capacities:", "# Student preferences:", "# College priorities:",
             "# Siblings:", "# Levels:", "# Students per Level:")


def read_instance(filename: str):
    """Parse instance.txt. Returns
        (students, colleges, pref, cap, siblings, levels_map, students_per_level)
    with pref holding BOTH student and college keys as {rank: target} dicts,
    as written by write_instance below. Blank lines are skipped; sections end at
    the next '#' line or EOF."""
    with open(filename) as f:
        lines = [ln.strip() for ln in f.readlines()]

    cap: Dict[str, int] = {}
    pref: Dict[str, Dict[int, str]] = {}
    siblings: Dict[str, List[str]] = {}
    levels_map: Dict[str, List[str]] = {}
    students_per_level: Dict[str, List[str]] = {}

    section = None
    for line in lines:
        if not line:
            continue
        if line.startswith("#"):
            section = None
            for name in _SECTIONS:
                if name in line:
                    section = name
            continue
        pieces = line.split(" ")
        if section == "# Capacities:":
            cap[pieces[0]] = int(pieces[1])
        elif section in ("# Student preferences:", "# College priorities:"):
            who = pieces[0]
            d = pref.setdefault(who, {})
            for tok in pieces[1:]:
                r, target = tok[1:-1].split(",")
                d[int(r)] = target
        elif section == "# Siblings:":
            siblings[pieces[0]] = pieces[1:]
        elif section == "# Levels:":
            levels_map[pieces[0]] = pieces[1:]
        elif section == "# Students per Level:":
            students_per_level[pieces[0]] = pieces[1:]

    colleges = sorted(set(cap) & set(pref))
    students = [k for k in pref if k not in set(colleges)]
    return students, colleges, pref, cap, siblings, levels_map, students_per_level


def write_instance(students, colleges, pref, cap, siblings, levels_map,
                   students_per_level, path: str) -> None:
    """Write an instance in the format read_instance expects (round-trippable)."""
    with open(path, "w") as f:
        f.write(f"# Num. students:{len(students)}\n")
        f.write(f"# Num. colleges:{len(colleges)}\n")
        f.write("# Students:" + ",".join(students) + "\n")
        f.write("# Colleges:" + ",".join(colleges) + "\n")
        f.write("# Capacities:\n")
        for c in cap:
            f.write(f"{c} {cap[c]}\n")
        f.write("# Student preferences:\n")
        for s in students:
            f.write(s + " " + " ".join(f"({p},{pref[s][p]})"
                                       for p in sorted(pref[s])) + "\n")
        f.write("# College priorities:\n")
        for c in colleges:
            f.write(c + " " + " ".join(f"({p},{pref[c][p]})"
                                       for p in sorted(pref[c])) + "\n")
        f.write("# Siblings:\n")
        for s in students:
            sibs = siblings.get(s, [])
            f.write(s + ("" if not sibs else " " + " ".join(sibs)) + "\n")
        f.write("# Levels:\n")
        for lab in levels_map:
            f.write(lab + " " + " ".join(levels_map[lab]) + "\n")
        f.write("# Students per Level:\n")
        for lab in students_per_level:
            f.write(lab + " " + " ".join(students_per_level[lab]) + "\n")


# ==========================================================================
# conversion: codebase structures -> Instance
# ==========================================================================
def _connected_components(students, siblings):
    """Families = connected components of the sibling graph (undirected)."""
    parent = {s: s for s in students}

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for s in students:
        for t in siblings.get(s, []):
            if t in parent:
                ra, rb = find(s), find(t)
                if ra != rb:
                    parent[ra] = rb
    comps: Dict[str, List[str]] = {}
    for s in students:
        comps.setdefault(find(s), []).append(s)
    return {f"fam{i}": mem for i, mem in enumerate(comps.values())}


def _grade_rank(label):
    """PreK < K < 1 < 2 < ...; unknown labels last, alphabetically."""
    s = str(label).strip()
    u = s.upper()
    if u == "PREK":
        return (-2, "")
    if u == "K":
        return (-1, "")
    try:
        return (int(s), "")
    except ValueError:
        return (10 ** 6, s)


def structures_to_instance(students, colleges, pref, cap, siblings, levels_map,
                           students_per_level, *, tie_break="mtbf",
                           lottery_seed=0, verbose=True) -> Instance:
    """Convert read_instance output into a model.Instance. Level labels map to
    grade-ordered integer ids (the descending heuristic relies on the order)."""
    college_label = {}
    for lab, clist in levels_map.items():
        for c in clist:
            college_label[c] = lab
    for c in colleges:
        college_label.setdefault(c, c.split("_", 1)[1] if "_" in c else "?")

    student_label = {}
    for lab, slist in students_per_level.items():
        for s in slist:
            student_label[s] = lab

    labels = set(college_label.values()) | set(student_label.values())
    ordered = sorted(labels, key=_grade_rank)
    level_id = {lab: i for i, lab in enumerate(ordered)}

    schools = sorted({c.split("_", 1)[0] for c in colleges})
    level_set = list(range(len(ordered)))

    capacity: Dict[Tuple[School, Level], int] = {}
    for c in colleges:
        capacity[(c.split("_", 1)[0], level_id[college_label[c]])] = int(cap[c])

    prefs: Dict[Student, List[School]] = {}
    level_of: Dict[Student, Level] = {}
    for s in students:
        ranked = sorted(pref[s].items())                  # [(rank, college)]
        rbds: List[str] = []
        for _, c in ranked:
            rbd = c.split("_", 1)[0]
            if rbd not in rbds:
                rbds.append(rbd)
        prefs[s] = rbds
        if s in student_label:
            level_of[s] = level_id[student_label[s]]
        elif ranked:
            level_of[s] = level_id[college_label[ranked[0][1]]]
    for s in students:
        level_of.setdefault(s, 0)

    families = _connected_components(students, siblings)

    if verbose:
        print(f"[inputs] level labels -> ids: "
              f"{', '.join(f'{lab}->{level_id[lab]}' for lab in ordered)}")
        bad_cap = sum(1 for s in students for rbd in prefs[s]
                      if capacity.get((rbd, level_of[s]), 0) == 0)
        if bad_cap:
            print(f"[inputs] WARNING: {bad_cap} (student, listed-school) pairs "
                  f"have 0 capacity at the student's level")

    inst = build_instance(schools, level_set, capacity, students, level_of,
                          prefs, families, seed=lottery_seed,
                          tie_break=tie_break, num_groups=1)
    if verbose:
        _report(inst)
    return inst


def _report(inst: Instance) -> None:
    sizes = Counter(len(m) for m in inst.families.values())
    multi = sum(1 for m in inst.families.values() if len(m) >= 2)
    print(f"[inputs] students : {len(inst.students)}")
    print(f"[inputs] families : {len(inst.families)}  (with >=2 members: {multi})")
    print(f"[inputs] fam sizes: {dict(sorted(sizes.items()))}")
    print(f"[inputs] schools  : {len(inst.schools)} RBDs   levels: {inst.levels}")
    print(f"[inputs] capacity : {sum(inst.capacity.values())} seats over "
          f"{len(inst.capacity)} (school,level) slots")


# ==========================================================================
# region loader (call ONCE per run; resample the lottery per draw)
# ==========================================================================
def load_region(region: str, year, r_root: str = "../R", *,
                tie_break: str = "mtbf", seed: int = 0, verbose: bool = True,
                instance_path: Optional[str] = None) -> Instance:
    """Read <r_root>/intermediate_data/<region>/<year>/instance.txt and return
    the Instance. `instance_path` overrides the path join entirely, e.g. to
    point at a synthetic file."""
    path = instance_path or os.path.join(
        r_root, "intermediate_data", str(region), str(year), "instance.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"instance.txt not found: {os.path.abspath(path)} "
            f"(region={region!r}, year={year!r}, r_root={r_root!r})")
    out = read_instance(path)
    if verbose:
        print(f"[inputs] read {os.path.abspath(path)}")
    return structures_to_instance(*out, tie_break=tie_break,
                                  lottery_seed=seed, verbose=verbose)


# ==========================================================================
# lotteries
# ==========================================================================
def make_lottery(students: List[Student], prefs: Dict[Student, List[School]],
                 families: Dict[str, List[Student]], family_of: Dict[Student, str],
                 seed: int, tie_break: str = "mtbf"
                 ) -> Dict[Tuple[Student, School], float]:
    rng = random.Random(seed)
    lottery: Dict[Tuple[Student, School], float] = {}
    if tie_break == "individual":
        for s in students:
            p = rng.random()
            for c in prefs[s]:
                lottery[(s, c)] = p
    elif tie_break == "family":
        fam_p = {fid: rng.random() for fid in families}
        member_index = {s: i for fid, mem in families.items()
                        for i, s in enumerate(mem)}
        for s in students:
            base = fam_p[family_of[s]] + (member_index[s] + 1) * 1e-9
            for c in prefs[s]:
                lottery[(s, c)] = base
    elif tie_break == "mtbf":
        member_index = {s: i for fid, mem in families.items()
                        for i, s in enumerate(mem)}
        fam_schools = {fid: sorted({c for s in mem for c in prefs[s]})
                       for fid, mem in families.items()}
        fam_p = {}
        for fid in sorted(families):
            for c in fam_schools[fid]:
                fam_p[(fid, c)] = rng.random()
        for s in students:
            fid = family_of[s]
            for c in prefs[s]:
                lottery[(s, c)] = fam_p[(fid, c)] + (member_index[s] + 1) * 1e-9
    else:
        raise ValueError(f"tie_break must be 'individual', 'family' or 'mtbf', "
                         f"got {tie_break!r}")
    return lottery


def resample_lottery(inst: Instance, seed: int, tie_break: str = "mtbf") -> Instance:
    """New Instance on the same fixed data with a fresh lottery (cheap clone)."""
    return inst.with_lottery(make_lottery(inst.students, inst.prefs, inst.families,
                                          inst.family_of, seed, tie_break))


def build_instance(schools, levels, capacity, students, level_of, prefs,
                   families, *, seed: int = 0, tie_break: str = "mtbf",
                   num_groups: int = 1, group=None) -> Instance:
    family_of = {s: fid for fid, mem in families.items() for s in mem}
    lottery = make_lottery(students, prefs, families, family_of, seed, tie_break)
    return Instance(schools=list(schools), levels=list(levels),
                    capacity=dict(capacity), students=list(students),
                    level_of=dict(level_of),
                    prefs={s: list(prefs[s]) for s in students},
                    families=dict(families),
                    group=dict(group) if group else {}, lottery=lottery,
                    num_groups=num_groups)
