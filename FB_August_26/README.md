# Software and Data for "Stable Matching with Contingent Priorities"

This archive is distributed in association with
[Operations Research](https://pubsonline.informs.org/journal/opre). It contains
the code and data behind every table and figure in the paper, and it is
self-contained: given the data in `R/` and a Gurobi installation, the commands
in the Replication section regenerate the reported numbers.

The archive is prepared for double-blind review. It carries no author names,
affiliations, acknowledgments, or repository links, and no license file; both
will be added on acceptance. Questions during review should go through the
journal's editorial system.

## Cite

To cite this archive, please cite both the paper and the archive, using their
respective DOIs.

```
@article{contingent,
  title   = {Stable Matching with Contingent Priorities},
  journal = {Operations Research},
  year    = {2026},
  doi     = {10.1287/opre.XXXX.XXXX}
}

@misc{contingentcode,
  title     = {{Software and Data for ``Stable Matching with Contingent Priorities''}},
  publisher = {Operations Research},
  year      = {2026},
  doi       = {10.1287/opre.XXXX.XXXX.cd}
}
```

## What the code does

School choice systems commonly give a child priority at a school where a
sibling is enrolled. When several children of one family apply in the same
round, the priority a child holds depends on where the siblings are assigned,
which is not known until the assignment is computed. Priorities are then
*contingent on the matching*, and standard stability no longer applies.

The paper defines who may provide such a priority, what receiving it does to a
school's order (absolute or partial), and when a matching is contingent stable.
This archive implements those definitions directly, in three layers that can be
used independently:

- the definitions as executable predicates, so any matching from any source can
  be tested for contingent stability;
- integer programming formulations that return a rank-optimal contingent stable
  matching or prove that none exists;
- deferred-acceptance mechanisms (the rule currently used in Chile, and the two
  proposed in the paper) that are fast and explainable but carry weaker
  guarantees.

Readers who want only the stability test need `v5_model.py` and
`v5_stability.py`. Readers who want to run the mechanisms on their own data
need those two plus `v5_inputs.py` and `v5_heuristics.py`; no solver is
required for that path.

## Quickstart

```
pip install gurobipy==9.5.1
cd v5
python -c "
from v5_inputs import load_region
from v5_heuristics import deferred_acceptance, rada
from v5_stability import is_contingent_stable
inst = load_region('Magallanes', 2023, '../R')
mu, info = rada(inst, 'absolute')
print('converged:', info['converged'],
      'contingent stable:', is_contingent_stable(inst, mu, 'absolute'))
"
```

The full campaign behind the paper is three commands; see Replication below.

## Contents

```
v5/
  v5_model.py               instances, matchings, and the basic quantities
  v5_inputs.py              the instance.txt reader and the tie-breaking rules
  v5_stability.py           Definitions 1 to 4 as code, and the stability test
  v5_heuristics.py          DA, Descending, Ascending, LSDA, RADA
  v5_exact.py               all integer programming formulations
  v5_metrics.py             the outcome columns of the tables
  v5_tables.py              LaTeX renderers for those tables
  v5_simulate.py            the per-region experiment driver
  v5_final_experiments.py   the four-region campaign
  v5_policy_alternatives.py the two policy panels
  v5_descriptives.py        the descriptive table and figures
  README.md
R/
  intermediate_data/<Region>/<Year>/instance.txt
results/
  v5/...                    created by the drivers
```

Table and figure numbers below refer to the submitted version of the paper.

## The model in code

`v5_model.Instance` is the market. Its fields are the objects of Section 3:
`schools`, `levels`, `capacity[(school, level)]`, `students`,
`level_of[student]`, `prefs[student]` (best to worst, acceptable schools only),
`families[family_id]`, `group[(student, school)]`, and
`lottery[(student, school)]`. A matching is a plain dictionary from student to
school or `None`.

`v5_stability.py` turns the definitions into functions. `providers` returns the
providers and the effective provider of each family at each school
(Definition 1). `order_key` returns the contingent priority order induced by a
matching, absolute (Definition 2) or partial (Definition 3).
`is_contingent_stable` tests non-wastefulness and the absence of contingent
justified envy (Definition 4), and with `return_blocking=True` it returns the
blocking pairs, which is the fastest way to see why a candidate matching fails.

## Data

Each region and year is one text file, `instance.txt`, with sections for
capacities, student preferences, school priorities, siblings, levels, and
students per level. A "college" token is `RBD_LABEL`: a school identifier and a
grade label such as `PreK`, `K`, `1`, ..., `12`. `v5_inputs.read_instance`
parses it and `v5_inputs.write_instance` writes it, so a reader can put their
own market in the same format and use everything here unchanged.

The data are built from the public records of the Chilean school choice system:
applicants, their reported preference lists, the sibling links they declare,
and the seats offered at each school and grade.

A school and a school-level program are different objects and the counts
differ by an order of magnitude, so `v5_descriptives.py` reports both: the
number of schools, the number of school-level programs, and the number of
programs offering at least one seat. A student may list a school that offers
no seat at their grade; such a pair is not a feasible assignment and does not
enter the model, but it does count toward the student's list length and hence
toward the penalty for going unassigned, following the paper's convention.

Lotteries are not read from the file; each simulation draws its own with
`make_lottery`, which implements three rules. `mtbf` draws one number per
(family, school), shared by the members of a family, and is the rule used in
Chile and in every table of the paper. `family` draws one number per family
reused at every school. `individual` draws one number per student reused at
every school. In all cases siblings are separated by the sufficiently small
offsets of the paper's footnote on family lotteries, which order siblings
within a family without disturbing the order across families.

## Mechanisms and formulations

| Name in the code | In the paper | What it is |
|---|---|---|
| `deferred_acceptance` | SOSM | student-proposing DA under the initial order |
| `descending` | Descending | levels from the top grade down, Algorithm 1, the rule used in Chile |
| `ascending` | Ascending | levels from the bottom grade up |
| `lsda` | LSDA | DA by family size, largest first, Algorithm 2 |
| `rada` | RADA | repeated DA with contingent updates, Algorithm 3 |
| `exact.solve(..., "absolute", "hard")` | Absolute | Formulation (6) |
| `exact.solve(..., "partial", "hard")` | | Formulation (14), Appendix D |
| `exact.solve(..., enforcement="soft")` | | (7d) for absolute, (8d) for partial |
| `exact.solve(..., enforcement="hybrid", zeta_min=z)` | | the floor (12) |
| `exact.solve(..., "none", "hard", objective="together")` | FOSM | Formulation (17) |
| `exact.solve(..., objective="together_members")` | Co-assignments | objective (19) with constraints (20) |
| `exact.solve(..., tie_breakers=False)` | no lotteries | Formulation (21) |

`rada` returns a matching and a dictionary. It converges when a DA pass
reproduces the previous matching, which is the fixed point of Algorithm 3; that
matching is contingent stable by Proposition 2 and `converged` is `True`.
Otherwise the sequence cycles or hits the iteration limit, `converged` is
`False`, and the matching returned is the one of lowest total rank among those
visited, which need not be contingent stable. Callers that need stability
should check the flag or call `is_contingent_stable`.

The soft, hybrid and partial formulations are implemented as printed but no
table in the paper reports them; they are here because the paper states them
and because they are the natural starting points for extending the framework.

## Replication

From inside `v5/`:

```
# main results and the four-region tables
nohup python v5_final_experiments.py --cores 15 > final.log 2>&1 &

# the two policy panels; reuses the Magallanes warm-start cache from above
nohup python v5_policy_alternatives.py > policy.log 2>&1 &

# the descriptive table and figures
python v5_descriptives.py
```

`v5_final_experiments.py` calls `v5_simulate.py` once per region, which writes
into `../results/v5/final_experiments/<Region>_<Year>_<timestamp>/`:
`rows.jsonl` and `rows.csv` with one line per (draw, method), `aggregate.csv`
with the means and standard errors, and the rendered `.tex` tables, including
`table_computation.tex` with the running times.

Single regions and shorter runs are one flag away, which is the sensible way to
start:

```
python v5_simulate.py --region Magallanes --draws 5 --cores 4
python v5_simulate.py --region Magallanes --draws 100 --methods SOSM,Descending,RADA
```

Runs are resumable. Pointed at an existing folder with `--out-dir`, a driver
skips the (draw, method) pairs already on disk and reuses the warm-start cache.
Lottery draws are deterministic given `--seed` (default 0), so a rerun
reproduces the same numbers, and draw `d` is the same market for every method.

## Performance

On the large regions the exact formulation is not hard to optimize. It is hard
to make feasible. The root relaxation returns a bound several hundred rank
units below the optimum and leaves thousands of fractional variables, because
the tie-breaking rows are logical constraints that are nearly vacuous in the
relaxation and only bite at integrality. Branch and bound can then spend hours
without reaching any integer point.

Gurobi's `NoRelHeurTime` heuristic runs before the root relaxation and searches
the integer space directly. On instances that had produced nothing in four
hours of branch and bound it found a stable matching in 236 to 526 seconds,
with zero nodes explored.

The exact method is therefore staged, once per draw:

1. RADA. If it converges, its matching is stable and no solver is needed.
2. Otherwise a feasibility solve: the same formulation, zero objective, stopped
   at the first solution, with `NoRelHeurTime` enabled.
3. The rank objective, warm-started from whichever matching step 1 or 2
   produced.

Step 3 closes at a single branch-and-bound node on most instances. The warm
start must be a stable matching: a matching that is not stable is not feasible
for the model, and Gurobi discards it silently, which is exactly the failure
that makes the formulation look intractable. `NoRelHeurTime` is set by
`--no-rel-heur-time`; use 0 on small regions, where it only delays the root
relaxation.

## Constraint names

Constraint names appear in solver logs and in infeasibility certificates, and
they carry the paper's equation numbers, so a certificate can be read directly
against the paper.

| Name in the code | Paper |
|---|---|
| `assign[s]`, `cap[c,l]` | feasible assignments |
| `3a` | initial stability |
| `4a`, `4b`, `4c`, `4d`, `RE` | effective priority providers, absolute |
| `13a`, `13b`, `13c`, `13d` | effective priority providers, partial |
| `5a`, `5b`, `5c` | receivers of priority |
| `6a`, `6b` | absolute contingent stability, hard |
| `7d` | absolute contingent stability, soft |
| `12` (`hybrid`) | the floor on the number of honored providers |
| `14a`, `14b` | partial contingent stability |
| `tlo`, `thi` | the two inequalities of (17b) |
| `wtog_a`, `wtog_b` | (20a) and (20b), the co-assignment objective |
| `21a`, `21b` | the group-order formulation without lotteries |

`RE[s,c]` requires that some sibling weakly prefer school `c` to their own
assignment, so that a provider has somebody to give priority to. Under absolute
priorities it is implied by `4d` and has no equation number of its own; under
partial priorities it is (13d).

The pair rows `6b`, `7d` and `14b` are instantiated for both orderings of every
sibling pair. Their body is role-asymmetric, one sibling co-assigned and the
other rejected, so a single row per unordered pair would drop the rejected
student's contingent claim.

## Environment

Python 3.8.10 and `gurobipy` 9.5.1, with a MIPGap tolerance of 0.1%, a one hour
limit per solve, and every other Gurobi parameter at its default. The reported
runs used a dual-socket Intel Xeon Gold 6226 (2x12 cores at 2.70 GHz, 24
threads) with 384 GB of RAM under AlmaLinux 9.8. The drivers default to these
settings. Nothing outside the standard library is required for the parts that
do not call the solver.
