# Stable Matching with Contingent Priorities — Replication Package

This archive contains the software and data used to produce the simulation results in the paper *Stable Matching with Contingent Priorities*. It implements the family of school-choice mechanisms studied in the paper (deferred-acceptance variants, simultaneous- and fixed-point iterations on contingent priorities, size-stratified heuristics, and an integer-programming benchmark), the Absolute Contingent Stability (ACS) verifier, and the comparison harness used in the companion note. The simulations are run on the 2023 Magallanes school-choice dataset.


## Description

The goal of this software is to evaluate and compare heuristic and exact mechanisms for the school-choice problem with sibling priorities under the framework of Absolute Contingent Stability (ACS). Concretely, the package implements:

* The ACS definition (Definitions 1–4 of the paper) and a self-contained verifier that, given a matching, returns whether it is ACS and, if not, every blocking pair witnessing the violation.
* A family of mechanisms that compose Deferred Acceptance with a contingent-priority boost:
  - Sequential by level, descending and ascending (level-DA inner).
  - Sequential by level with the RA-DA inner.
  - Simultaneous (monotone) and Simultaneous (non-monotone, fixed-point).
  - Size-stratified mechanisms (LS and SL, with and without capacity decrement).
* An integer-programming benchmark (Hard) that searches for an ACS matching that maximizes sibling co-assignment, solved with Gurobi.
* A diagnostic harness that, for every Hard solve that is reported as non-ACS by the literal Def-4 verifier, sweeps the seeds, isolates blocking pairs whose witness is a sibling, and confirms that the residual gap is fully explained by the sibling-envy convention.
* A paired comparison harness (`compare_heuristics.py`) that runs every mechanism on identical seeded instances under two boost definitions — the absolute-contingent boost of this paper and the naive sibling-priority boost in the main paper — and emits a side-by-side LaTeX table.

## Building

### Hardware and software requirements

The simulations were run on a Linux workstation with 64 GB of RAM and a single CPU core. A full 100-simulation sweep of the heuristic mechanisms takes roughly 100 minutes on this hardware; the Hard IP adds approximately 20–60 seconds per simulation depending on the seed.

* Python 3.10 or later.
* The packages listed in `requirements.txt`. Install with

  ```
  pip install -r requirements.txt
  ```

* [Gurobi](https://www.gurobi.com/) 10.0 or later, with a valid academic or commercial license, is required only for the integer-programming benchmark (`Hard`, `Hard-NTB`, `FOSM`, `Soft`, `Hybrid-310`, `Hybrid-320`). The heuristics, the ACS verifier, and `compare_heuristics.py` run without Gurobi.

### Data

The simulations expect Magallanes 2023 instance files at the path

```
<DATA_ROOT>/Magallanes/2023/instance.txt
```

where `<DATA_ROOT>` is configurable in `src/simulations_acs.py` (variable `REGION_INDIR`). The instance file is parsed by `generate_inputs.read_instance`, which is part of the project codebase and is not redistributed here. Collaborators with access to the codebase should place that module on their `PYTHONPATH` alongside the `src/` directory.

### Quick self-test

To confirm that the verifier is correctly installed, run

```
cd src
python validate_acs.py
```

This runs 11 unit tests for `check_acs` and prints `11/11 passed` on success. The self-test does not require Gurobi or the Magallanes data.

## Results

The package reproduces the following tables from the paper and the companion note:

* **Table: ACS summary.** Per-mechanism `%ACS`, average blocking pairs on non-ACS instances, average preference rank, sibling co-assignment count, and runtime, averaged over 100 simulations on Magallanes 2023.
* **Table: descriptive statistics (average preference).** Per-mechanism mean and standard deviation of the average preference rank.
* **Table: descriptive statistics (top preference).** Per-mechanism mean and standard deviation of the count of students matched to their first preference.
* **Table: Algos of Federico and Ignacio paired comparison.** Side-by-side comparison of every mechanism under the two boost definitions, on identical seeded instances.
* **Diagnostic summary.** For each Hard solve, a sim-by-sim report of any literal ACS violation, the witness blocking pair, and whether the witness is a sibling of the envying student.

The LaTeX source for each table is written to `results/<date>/`; the supporting per-row CSVs are written to the same directory.

## Replicating

All commands assume that the working directory is `src/` and that `generate_inputs.py`, `algorithms.py`, and `solve_opt.py` are on `PYTHONPATH`.

* To reproduce the main heuristics-vs-Hard table, run

  ```
  python simulations_acs.py --sims 100 --name main
  ```

  The script writes `results/<date>/main_summary.tex` (the LaTeX table), `results/<date>/main_avgpref.tex` and `results/<date>/main_toppref.tex` (the descriptive tables), and `results/<date>/main_rows_100sims.csv` (the per-row data).

* To reproduce the paired Federico-Ignacio comparison, run

  ```
  python compare_heuristics.py
  ```

  The script writes `results/<date>/comparison.tex` and `results/<date>/rows.csv`. With `NUM_SIMS=100` this takes roughly 100 minutes; reduce to a smaller value for a quick sanity check.

* To reproduce the Hard-ACS diagnostic (the convention check that resolves the residual non-ACS rate of the Hard benchmark), run

  ```
  python find_and_analyze_non_acs_hard.py
  ```

  The script writes `results/non_acs_hard/<date>/summary.txt`, `summary.json`, and a per-sim folder for every simulation that is non-ACS under the literal Def-4 convention. The summary distinguishes, for each non-ACS instance, whether the blocking pair is convention-only (i.e., resolved by `exclude_sibling_envy=True`) or a true error.

* To inspect an individual blocking pair on a single simulation, run

  ```
  python diagnose_hard.py --sim <seed>
  ```

## Files

The `src/` directory contains the following Python modules. Each is self-documenting; the high-level role is given here.

| File                                | Role                                                                                                                                |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `simulations_acs.py`                | Top-level runner. Loops over simulations and mechanisms, calls the ACS verifier, calls Gurobi for the IP benchmarks, writes CSVs.   |
| `heuristics.py`                     | Mechanisms studied in the paper: descending/ascending DA, RA-DA inner variants, simultaneous, fixed-point, LS, SL, LS-nd, SL-nd.    |
| `acs_priority.py`                   | Helper functions for the absolute-contingent priority: effective providers, levels-of, base-admissibility, lone-provider demotion.  |
| `acs_verifier.py`                   | The ACS verifier. Implements Definitions 1–4 literally and supports an optional `exclude_sibling_envy` convention.                  |
| `validate_acs.py`                   | 11 unit tests for `check_acs`. Runs without Gurobi or the Magallanes data.                                                          |
| `analysis_acs.py`                   | Aggregation and LaTeX-table generation from the CSV output of `simulations_acs.py`.                                                 |
| `algorithms_ignacio.py`             | Drop-in clean version of the algorithms module of Bobbio, Carvalho, Lodi, Rios, Torrico (2025). Used for paired comparison only.    |
| `compare_heuristics.py`             | Paired comparison runner: runs every mechanism under both boost definitions on identical seeded instances.                          |
| `find_and_analyze_non_acs_hard.py`  | Diagnostic for the Hard IP: sweeps every simulation, isolates blocking pairs, classifies each as convention-only or true error.     |
| `explain_pair.py`                   | Walks one blocking pair through Definitions 1–4 step by step and prints the verdict.                                                |
| `diagnose_hard.py`                  | Pre-flight: runs the Hard IP on a single simulation and reports the verifier output with an `explain_pair` trace for any witness.   |

The `data/` directory is empty in this snapshot; collaborators must populate it with the Magallanes 2023 instance file as described under **Building**. The `results/` directory holds the LaTeX tables and CSVs produced by the scripts and is the only directory written to.

## Ongoing Development

This code is being developed on an on-going basis. Please contact the maintainer before forking or branching to avoid divergence.

## Support

For support in using this software, contact the maintainer directly. When reporting a problem, include:

1. The exact command that was run.
2. The full stack trace (the comparison and diagnostic scripts also write the trace to `errors.log` in the output directory).
3. The Python version, the Gurobi version, and the operating system.
