"""
v5_descriptives.py

The descriptive statistics of Appendix F.2, computed from the instance files.
From inside v5/:

    python v5_descriptives.py                # all regions under ../R
    python v5_descriptives.py --regions Magallanes,Atacama

For every region directory <r_root>/intermediate_data/<Region>/<year>/ that
holds an instance.txt, it writes CSVs to ../results/v5/descriptives/:

  table6.csv        one row per region plus an Overall row: number of schools,
                    number of programs (school-level pairs) and of programs
                    that offer at least one seat, total seats, number of
                    students, number of applications, students with a sibling
                    concurrently applying to any level (count and share), and
                    students with a sibling applying to the same level (count
                    and share). The numbers behind Table 6. Three counts are
                    reported where the table has one column, because a school
                    and a school-level program are different objects and the
                    distinction matters: a region with 61 schools offers
                    several hundred programs.
  fig2_<region>.csv per level: students with a same-level sibling, students
                    with siblings only at other levels, students with no
                    siblings. The numbers behind Figure 2(a) for Magallanes.
  fig2_all.csv      the same, aggregated over every region processed. The
                    numbers behind Figure 2(b).
  fig3.csv          the distribution, over pairs of siblings, of the number of
                    schools the two preference lists share, in buckets 0 to 10
                    and 11+, with the percent of pairs in each bucket. The
                    numbers behind Figure 3.

Levels are reported as the loader's integer ranks, which order the grades from
Pre-K upward (see _grade_rank in v5_inputs.py). The lottery draw plays no role
in any of these numbers, so a single load per region suffices.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from itertools import combinations

from v5_inputs import load_region


def region_stats(inst):
    """The Table 6 row and the per-level Figure 2 counts for one instance."""
    students = inst.students
    seats = sum(inst.capacity.values())
    programs = len(inst.capacity)
    programs_with_seats = sum(1 for q in inst.capacity.values() if q > 0)
    applications = sum(len(inst.prefs.get(s, [])) for s in students)

    members = {fid: fam for fid, fam in inst.families.items()}
    sib_any = [s for s in students
               if len(members.get(inst.family_of.get(s), [s])) >= 2]
    sib_same = [s for s in sib_any
                if any(t != s and inst.level_of[t] == inst.level_of[s]
                       for t in members[inst.family_of[s]])]

    per_level = {}
    same, anyset = set(sib_same), set(sib_any)
    for s in students:
        lv = inst.level_of[s]
        d = per_level.setdefault(lv, {"same_level": 0, "other_siblings": 0,
                                      "no_siblings": 0})
        if s in same:
            d["same_level"] += 1
        elif s in anyset:
            d["other_siblings"] += 1
        else:
            d["no_siblings"] += 1

    row = {"schools": len(inst.schools), "programs": programs,
           "programs_with_seats": programs_with_seats, "seats": seats,
           "students": len(students), "applications": applications,
           "sib_any": len(sib_any),
           "sib_any_pct": len(sib_any) / len(students) if students else 0.0,
           "sib_same": len(sib_same),
           "sib_same_pct": len(sib_same) / len(students) if students else 0.0}
    return row, per_level


def shared_schools(inst, counter: Counter):
    """Add this region's sibling pairs to the shared-schools distribution."""
    for fam in inst.families.values():
        if len(fam) < 2:
            continue
        for a, b in combinations(sorted(fam), 2):
            shared = len(set(inst.prefs.get(a, [])) & set(inst.prefs.get(b, [])))
            counter[min(shared, 11)] += 1        # 11 collects the 11+ bucket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-root", default="../R")
    ap.add_argument("--year", default=2023)
    ap.add_argument("--regions", default=None,
                    help="comma list; default is every directory under "
                         "<r_root>/intermediate_data that has an instance.txt "
                         "for the year")
    ap.add_argument("--out-dir", default=os.path.join("..", "results", "v5",
                                                      "descriptives"))
    a = ap.parse_args()

    root = os.path.join(a.r_root, "intermediate_data")
    if a.regions:
        regions = [r.strip() for r in a.regions.split(",") if r.strip()]
    else:
        regions = sorted(
            d for d in os.listdir(root)
            if os.path.exists(os.path.join(root, d, str(a.year),
                                           "instance.txt")))
    if not regions:
        raise SystemExit(f"no region with an instance.txt for {a.year} "
                         f"under {os.path.abspath(root)}")
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[descriptives] {len(regions)} region(s): {', '.join(regions)}")

    table6, fig2_all, pairs = [], {}, Counter()
    for region in regions:
        inst = load_region(region, a.year, a.r_root, verbose=False)
        row, per_level = region_stats(inst)
        row["region"] = region
        table6.append(row)
        shared_schools(inst, pairs)

        with open(os.path.join(a.out_dir, f"fig2_{region}.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["level", "same_level_siblings", "other_siblings",
                        "no_siblings"])
            for lv in sorted(per_level):
                d = per_level[lv]
                w.writerow([lv, d["same_level"], d["other_siblings"],
                            d["no_siblings"]])
        for lv, d in per_level.items():
            t = fig2_all.setdefault(lv, {"same_level": 0, "other_siblings": 0,
                                         "no_siblings": 0})
            for k in t:
                t[k] += d[k]
        print(f"[descriptives] {region}: {row['students']} students, "
              f"{row['schools']} schools, {row['programs']} programs "
              f"({row['programs_with_seats']} with seats), {row['seats']} seats, "
              f"{row['sib_any']} with siblings, {row['sib_same']} same level")

    cols = ["region", "schools", "programs", "programs_with_seats", "seats",
            "students", "applications", "sib_any", "sib_any_pct",
            "sib_same", "sib_same_pct"]
    overall = {"region": "Overall"}
    for k in cols[1:]:
        if k.endswith("_pct"):
            continue
        overall[k] = sum(r[k] for r in table6)
    overall["sib_any_pct"] = overall["sib_any"] / overall["students"]
    overall["sib_same_pct"] = overall["sib_same"] / overall["students"]
    with open(os.path.join(a.out_dir, "table6.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in table6 + [overall]:
            r = dict(r)
            r["sib_any_pct"] = f"{r['sib_any_pct']:.3f}"
            r["sib_same_pct"] = f"{r['sib_same_pct']:.3f}"
            w.writerow(r)

    with open(os.path.join(a.out_dir, "fig2_all.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "same_level_siblings", "other_siblings",
                    "no_siblings"])
        for lv in sorted(fig2_all):
            d = fig2_all[lv]
            w.writerow([lv, d["same_level"], d["other_siblings"],
                        d["no_siblings"]])

    total_pairs = sum(pairs.values())
    with open(os.path.join(a.out_dir, "fig3.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shared_schools", "pairs", "pct"])
        for k in range(12):
            lbl = "11+" if k == 11 else str(k)
            n = pairs.get(k, 0)
            w.writerow([lbl, n,
                        f"{100.0 * n / total_pairs:.1f}" if total_pairs else 0])

    print(f"[descriptives] wrote table6.csv, fig2_*.csv, fig3.csv to "
          f"{os.path.abspath(a.out_dir)} ({total_pairs} sibling pairs)")


if __name__ == "__main__":
    main()
