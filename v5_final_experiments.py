"""
v5_final_experiments.py

The final experiment campaign for the main paper: seven methods on all four
regions, from one command.

    cd v5
    nohup python v5_final_experiments.py --cores 6 > final.log 2>&1 &

Methods, in the order they appear in the tables:

    SOSM            deferred acceptance (DA), the baseline
    Descending      the size stratified descending heuristic
    LSDA            largest size first deferred acceptance
    RADA           RADA, the recomputed simultaneous update
    Absolute   the exact formulation (IP)
    FOSM-ACS        family optimal stable matching over the CONTINGENT set
    FOSM            family optimal stable matching over the STANDARD set

Note the FOSM naming. FOSM-ACS maximizes togetherness over matchings that are
absolutely contingent stable; FOSM does the same over ordinary stable
matchings and is the paper's FOSM. Rows in results files written before this
naming was fixed carry the two labels the other way round.

Regions run in order: Magallanes, Atacama, Lagos, OHiggins. Output goes under
results/v5/final_experiments/<Region>_<year>_<timestamp>/ so it never collides
with earlier campaigns, and each region writes the three tables in both unit
variants plus the computation table.

This is a thin wrapper. It shells out to v5_simulate once per region, so every
number is produced by the same certified code path as a single region run, and
a failure in one region does not abort the others. Everything is resumable per
(draw, method).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

REGIONS = ["Magallanes", "Atacama", "Lagos", "OHiggins"]
FINAL_METHODS = "SOSM,Descending,Ascending,LSDA,RADA,Absolute,FOSM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=",".join(REGIONS),
                    help="comma list, run in order")
    ap.add_argument("--year", default="2023")
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--cores", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=3600.0)
    ap.add_argument("--methods", default=FINAL_METHODS)
    ap.add_argument("--tie-break", default="mtbf",
                    help="mtbf for the family-level rule (Tables 1-3 and 7-10); "
                         "individual for the individual-level rule (Tables 11-14)")
    ap.add_argument("--results-root",
                    default="../results/v5/final_experiments")
    ap.add_argument("--r-root", default="../R")
    a = ap.parse_args()

    regions = [r.strip() for r in a.regions.split(",") if r.strip()]
    print("[final] FINAL EXPERIMENTS")
    print(f"[final] regions in order: {regions}")
    print(f"[final] methods: {a.methods}")
    print(f"[final] {a.draws} draws, {a.cores} cores, "
          f"{a.time_limit:.0f}s per solve, tie-break {a.tie_break}")
    print(f"[final] results root: {a.results_root}")

    t0 = time.perf_counter()
    failed = []
    for region in regions:
        print(f"\n{'=' * 60}\n[final] === {region} ===\n{'=' * 60}", flush=True)
        cmd = [sys.executable, "v5_simulate.py",
               "--region", region,
               "--year", str(a.year),
               "--draws", str(a.draws),
               "--cores", str(a.cores),
               "--time-limit", str(a.time_limit),
               "--methods", a.methods,
               "--tie-break", a.tie_break,
               "--results-root", a.results_root,
               "--r-root", a.r_root]
        rc = subprocess.call(cmd)
        if rc != 0:
            failed.append(region)
            print(f"[final] WARNING: {region} exited with code {rc}; "
                  f"continuing to the next region", flush=True)
        else:
            print(f"[final] {region} done  "
                  f"[{time.perf_counter() - t0:.0f}s elapsed]", flush=True)

    print(f"\n[final] all regions finished "
          f"[{time.perf_counter() - t0:.0f}s total]")
    if failed:
        print(f"[final] regions that exited non-zero: {failed}")


if __name__ == "__main__":
    main()
