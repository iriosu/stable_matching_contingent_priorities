"""
v5_model.py

Data structures for the contingent-priorities pipeline. Pure: imports nothing
from the package. Everything downstream (stability, heuristics, exact IPs,
metrics) consumes the Instance defined here.

|G| = 1 is the paper's standing assumption (Remark 2); the fields carry a
general group map so the group-aware branches stay in place.

Performance notes: students-per-level and the per-(school, level) lister lists
are cached at construction, so the stability checker and DA stay near-linear
at Magallanes scale (~5,100 students). `with_lottery` returns a shallow clone
that shares those caches and only swaps the lottery, which is what the
repeated-draw simulations use.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

School = str
Level = int
Student = str
Matching = Dict[Student, Optional[School]]


@dataclass
class Instance:
    schools: List[School]
    levels: List[Level]
    capacity: Dict[Tuple[School, Level], int]      # q^ell_c, missing -> 0
    students: List[Student]
    level_of: Dict[Student, Level]                 # ell(s)
    prefs: Dict[Student, List[School]]             # best -> worst, acceptable only
    families: Dict[str, List[Student]]             # family_id -> members (partition)
    group: Dict[Tuple[Student, School], int]       # g(s,c); defaults to 1
    lottery: Dict[Tuple[Student, School], float]   # p_{s,c}; strict per (school,level)
    num_groups: int = 1                            # |G|

    family_of: Dict[Student, str] = field(default_factory=dict)
    _rank: Dict[Tuple[Student, School], int] = field(default_factory=dict)
    _level_students: Dict[Level, List[Student]] = field(default_factory=dict)
    _listers: Dict[Tuple[School, Level], List[Student]] = field(default_factory=dict)

    def __post_init__(self):
        self.family_of = {}
        for fid, members in self.families.items():
            for s in members:
                self.family_of[s] = fid
        for s in self.students:
            for c in self.prefs[s]:
                self.group.setdefault((s, c), 1)
        self._rank = {}
        self._level_students = defaultdict(list)
        self._listers = defaultdict(list)
        for s in self.students:
            ell = self.level_of[s]
            self._level_students[ell].append(s)
            for i, c in enumerate(self.prefs[s]):
                self._rank[(s, c)] = i + 1
                self._listers[(c, ell)].append(s)

    # ----- cheap clone with a fresh lottery (per-draw resampling) ----------
    def with_lottery(self, lottery: Dict[Tuple[Student, School], float]) -> "Instance":
        new = copy.copy(self)          # shares prefs/families/caches
        new.lottery = lottery
        return new

    # ----- basic accessors --------------------------------------------------
    def q(self, c: School, ell: Level) -> int:
        return self.capacity.get((c, ell), 0)

    def acceptable(self, s: Student, c: School) -> bool:
        return (s, c) in self._rank

    def rank(self, s: Student, c: Optional[School]) -> int:
        """1-based rank; being unassigned ranks len+1; unlisted schools len+2."""
        if c is None:
            return len(self.prefs[s]) + 1
        r = self._rank.get((s, c))
        return r if r is not None else len(self.prefs[s]) + 2

    def family_members(self, s: Student) -> List[Student]:
        return self.families[self.family_of[s]]

    def siblings(self, s: Student) -> List[Student]:
        return [t for t in self.family_members(s) if t != s]

    def students_at_level(self, ell: Level) -> List[Student]:
        return self._level_students.get(ell, [])

    def listers(self, c: School, ell: Level) -> List[Student]:
        """Students at level ell that list c (Definition 1 (iii) universe)."""
        return self._listers.get((c, ell), [])

    def in_V(self, s: Student, c: School) -> bool:
        """(s, c) in V: s lists c and c has seats at s's level."""
        return self.acceptable(s, c) and self.q(c, self.level_of[s]) > 0

    # ----- initial priority order -------------------------------------------
    def order_key(self, s: Student, c: School) -> Tuple[int, float]:
        return (self.group[(s, c)], self.lottery[(s, c)])

    def succ(self, s: Student, t: Student, c: School) -> bool:
        """s >_c t under the INITIAL priority order (groups, then lottery)."""
        return self.order_key(s, c) < self.order_key(t, c)

    def weakly_prefers(self, s: Student, c: School, mu: Matching) -> bool:
        """c >=_s mu(s). Requires c acceptable to s."""
        if not self.acceptable(s, c):
            return False
        return self.rank(s, c) <= self.rank(s, mu.get(s))


# ----- matching helpers ------------------------------------------------------
def empty_matching(inst: Instance) -> Matching:
    return {s: None for s in inst.students}


def occupants_by_slot(inst: Instance, mu: Matching) -> Dict[Tuple[School, Level], List[Student]]:
    out: Dict[Tuple[School, Level], List[Student]] = defaultdict(list)
    for s, c in mu.items():
        if c is not None:
            out[(c, inst.level_of[s])].append(s)
    return out


def is_feasible(inst: Instance, mu: Matching) -> Tuple[bool, List[str]]:
    problems: List[str] = []
    for s, c in mu.items():
        if c is None:
            continue
        if not inst.acceptable(s, c):
            problems.append(f"{s} assigned to unacceptable {c}")
        if inst.q(c, inst.level_of[s]) == 0:
            problems.append(f"{s} -> {c} has no level-{inst.level_of[s]} seat")
    for (c, ell), occ in occupants_by_slot(inst, mu).items():
        if len(occ) > inst.q(c, ell):
            problems.append(f"capacity exceeded at ({c}, L{ell})")
    return (len(problems) == 0, problems)


def total_rank(inst: Instance, mu: Matching) -> int:
    """Sum of assigned ranks, penalty |prefs|+1 per unassigned (paper objective)."""
    tot = 0
    for s in inst.students:
        c = mu.get(s)
        tot += (len(inst.prefs[s]) + 1) if c is None else inst.rank(s, c)
    return tot


def num_unassigned(inst: Instance, mu: Matching) -> int:
    return sum(1 for s in inst.students if mu.get(s) is None)


def families_together(inst: Instance, mu: Matching, min_size: int = 2) -> Tuple[int, int]:
    """(# multi-member families entirely at one school, # multi-member families)."""
    tog = total = 0
    for members in inst.families.values():
        if len(members) < min_size:
            continue
        total += 1
        schools = {mu.get(s) for s in members}
        if len(schools) == 1 and None not in schools:
            tog += 1
    return tog, total


def members_with_a_sibling_at_same_school(inst: Instance, mu: Matching) -> int:
    """The paper's Together column: students matched with >=1 sibling."""
    cnt = 0
    for s in inst.students:
        c = mu.get(s)
        if c is not None and any(mu.get(t) == c for t in inst.siblings(s)):
            cnt += 1
    return cnt
