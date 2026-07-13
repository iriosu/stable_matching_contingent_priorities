This archive is distributed in association with [Operations Research](https://pubsonline.informs.org/journal/opre)
under the [MIT License](LICENSE).

The software and data in this archive are a snapshot of the software and data
that were used in the research reported in the paper *Stable Matching with
Contingent Priorities*.

This archive is a snapshot, not a development branch. The results reported in the
paper were produced by the code exactly as it appears here.

## Cite

To cite this archive, please cite both the paper and this archive, using their
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

## Description

Most school choice systems give priority to a student whose sibling attends the
same school. When several children of one family apply at the same time, the
priority a child holds depends on where their sibling is assigned, which is not
known until the assignment has been computed. Priorities are therefore contingent
on the matching, and the standard theory of stable matching does not cover them.

This archive contains the methods introduced in the paper and the code that
reproduces the computational results.

A stable matching under absolute contingent priorities need not exist, and
deciding whether one exists is NP-complete. Every method therefore has three
possible outcomes on any instance: it returns a stable matching, it proves that
none exists, or it exhausts its time budget. The code distinguishes all three.

### Methods

| Name | Description |
|---|---|
| `SOSM` | Deferred acceptance under the initial priority order. Ignores contingent priorities. |
| `Descending` | Levels processed from the highest grade down. At each level the contingent order is recomputed from the placements already fixed at higher levels. This is the rule the Chilean system uses. |
| `Ascending` | The same, from the lowest grade up. |
| `LSDA` | Deferred acceptance stratified by family size, largest families first. Strategy-proof for students and for families, but it may not return a stable matching. |
| `SLDA` | The same, smallest families first. |
| `RA-DA` | Deferred acceptance run repeatedly, recomputing the contingent order from the previous matching. If it converges, the matching it returns is stable. It need not converge. |
| `RADA-Paper` | The algorithm as written in the paper. It reaches the same fixed points as `RA-DA` and differs only in what it reports on a cycle. |
| `RADA-Portfolio` | `RA-DA` restarted from four seeds. |
| `RADA-Sequential` | `RA-DA` with providers committed one at a time. |
| `RADA-Search` | Depth-first search over subsequences of providers. Worst-case exponential, so it is bounded by node, sequence and time limits. |
| `RADA-Lottery` | A variant that raises the tie-breaker rather than the priority group. |
| `RADA-IP` | `RA-DA`, escalating to the exact formulation only where it fails to converge. |
| `IP-Warm` | The exact formulation, staged. See below. |
| `ACS-Feasible` | The exact formulation with a constant objective, stopped at the first solution. Returns some stable matching, not the rank-optimal one, and is much cheaper. |
| `Absolute-Hard` | The exact formulation. Returns the rank-optimal stable matching under absolute contingent priorities whenever one exists, and proves infeasibility otherwise. |
| `Absolute-Soft` | The same, with contingent priorities the clearinghouse may decline to honor. Always feasible. |
| `Hybrid-<z>` | Soft, with a floor: at least `z` priority providers must be honored. Feasibility decreases as `z` grows. |
| `Partial-Hard`, `Partial-Soft` | The exact formulation under partial contingent priorities. |

## Contents

```
.
├── R/                                    the instance data
│   └── intermediate_data/
│       └── <Region>/<Year>/instance.txt
├── v2/                                   all the code
│   ├── README.md                         this file
│   ├── v2_model.py                       data structures; imports nothing
│   ├── v2_stability.py                   the definitions and the checker
│   ├── v2_inputs.py                      the instance.txt reader and the lotteries
│   ├── v2_heuristics.py                  DA, Descending, Ascending, LSDA, SLDA, RA-DA
│   ├── v2_rada.py                        the RADA variants
│   ├── v2_exact.py                       the integer programs
│   ├── v2_metrics.py                     the table columns
│   ├── v2_tables.py                      CSV and LaTeX output
│   ├── v2_simulate.py                    the experiment driver
│   
└── results/
    └── v2/
        └── <Region>_<Year>_<timestamp>/
            ├── rows.jsonl                one line per (draw, method), written as it finishes
            ├── rows.csv
            ├── aggregate.csv
            ├── table_main.tex
            └── table_split.tex
```

`v2/` reads `../R` and writes `../results/v2`. It touches nothing else. Every
module imports only the standard library, `gurobipy`, and other `v2_` modules.

Every definition in the paper is implemented in `v2_stability.py` and nowhere
else. `v2_heuristics.py`, `v2_rada.py`, `v2_metrics.py` and `v2_simulate.py` all
call it rather than reimplementing it.

## Building

Python 3.9 or later, and [Gurobi](https://www.gurobi.com) 9.5 or later with a
license for the integer programs. There are no other dependencies.

```
pip install gurobipy
python -c "import gurobipy; print(gurobipy.gurobi.version())"
```

Every heuristic runs without a solver. Only `v2_exact.py` and the integer-program
methods in `v2_simulate.py` need one.

## Replicating

From inside `v2/`:

```
cd v2
python v2_simulate.py --region Magallanes --year 2023 --draws 100 --cores 20
```

Every other setting is in the `SETTINGS` block at the top of `v2_simulate.py`,
and each has a command-line override. Output goes to
`../results/v2/<Region>_<Year>_<timestamp>/`.

The run is resumable. Each `(draw, method)` row is appended to `rows.jsonl` as it
finishes, so if the job dies you can point `--out-dir` at the same folder and it
will skip the pairs already on disk.

| Output | Contents |
|---|---|
| `table_main.tex` | Solved, ACS rate, blocking rate, average preference, top preference, unassigned, together, and the separated columns, by method |
| `table_split.tex` | The same, split into students with and without siblings |
| `aggregate.csv` | Every aggregated column, by method |
| `rows.csv` | The raw per-`(draw, method)` rows |

The ACS rate is the percentage of solved instances whose matching passes the
absolute stability checker. The blocking rate is averaged over the instances that
fail it, since an instance that passes has no blocking pairs by definition and
would only dilute the number.

## Finding a feasible point is the bottleneck

On the large regions the exact formulation is not hard to optimize. It is hard to
make feasible.

The root relaxation takes 40 to 110 seconds, returns a bound several hundred rank
units below the optimum, and leaves more than two thousand fractional integer
variables. The tie-breaking rows are logical constraints that are nearly vacuous
in the relaxation and only bite at integrality, so branch and bound spends hours
dividing the tree without reaching any integer point.

Gurobi's `NoRelHeurTime` heuristic runs before the root relaxation and does not
use it. It searches the integer space directly. On the Atacama instances that had
produced nothing in four hours of branch and bound, it found a stable matching in
236 to 526 seconds, with zero nodes explored and zero simplex iterations.

The IP methods are therefore staged:

1. `RA-DA`. If it converges, its matching is stable and no solver is needed.
2. Otherwise the feasibility IP: the same formulation, zero objective, stopped at
   the first solution, with `NoRelHeurTime` enabled.
3. The rank IP, warm-started from whichever matching step 1 or 2 produced.

Step 3 closes at a single branch-and-bound node on most instances.

The warm start must be a **stable** matching. `RA-DA`'s matching on a cycling
instance is not stable, so it is not feasible for the model and Gurobi discards
it as a MIP start. That is why step 2 exists and why step 3 starts from it.

`NoRelHeurTime` is set by `FEAS_NO_REL_HEUR` in `v2_simulate.py` and by
`--no-rel-heur-time` on the command line. It is worth disabling on small
instances, where it only delays the root relaxation.

## The instances the IP does not close

```
python v2_analyze_hard.py --regions Atacama,Lagos,OHiggins
```

This re-solves, one at a time, exactly the instances on which the exact
formulation hit its time limit or returned infeasible during a run, and reports
for each whether a stable matching was exhibited, proven not to exist, or left
open. For the instances with no stable matching it computes an irreducible
infeasible subsystem, which is a minimal certificate of that fact.

The three outcomes must not be conflated. An instance on which the solver ran out
of time is not an instance with no stable matching.

## Constraint names

Constraint names appear in solver logs and in infeasibility certificates, so they
carry the paper's equation numbers.

| Name in the code | Paper |
|---|---|
| `assign[s]`, `cap[c,l]` | the set of feasible assignments |
| `4a`, `4b`, `4c`, `4d` | the set of effective priority providers |
| `5a`, `5b`, `5c` | the set of receivers |
| `6a`, `6b` | absolute contingent stability, hard |
| `7d` | absolute contingent stability, soft |
| `16a`, `16b` | partial contingent stability |
| `hybrid` | the floor on the number of honored providers |
| `RE`, `SC` | see below |

Two constraints have no equation number because they are not in the paper.

`RE[s,c]` requires that some sibling weakly prefers school `c` to their own
assignment, so that the provider has somebody to give priority to. It is implied
by `4d` under absolute priorities. Under partial priorities there is no `4d`, so
nothing implies it, and without `RE` a student could be a provider with no
receiver. It is imposed in both cases and costs nothing in the first.

`SC[c,s,a]` rules out sibling justified envy under partial priorities: a sibling
with a worse tie-breaker cannot hold a seat while a sibling with a better one is
left out. Constraints `16a` and `16b` do not rule this out on their own, and
without `SC` the formulation admits matchings the checker rejects.

