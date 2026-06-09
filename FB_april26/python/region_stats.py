import os
import math
import pandas as pd
import generate_inputs as genin


def count_families(siblings):
    visited = set()
    family_sizes = []

    for s in siblings:
        if s in visited:
            continue

        # DFS / stack over sibling graph
        stack = [s]
        comp = set()

        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            for v in siblings.get(u, []):
                if v not in comp:
                    stack.append(v)

        visited.update(comp)
        family_sizes.append(len(comp))

    return family_sizes


def summarize_region(base_indir, region, year=2023):
    instance_file = os.path.join(base_indir, region, str(year), "instance.txt")
    if not os.path.exists(instance_file):
        raise FileNotFoundError(f"Missing instance file: {instance_file}")

    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn = genin.read_instance(instance_file)

    # Families from sibling graph
    family_sizes = count_families(siblings)
    num_families = len(family_sizes)
    avg_family_size = sum(family_sizes) / num_families if num_families > 0 else 0.0

    # Students with siblings
    students_with_siblings = [s for s in students if len(siblings.get(s, [])) > 0]
    pct_students_with_siblings = (
        len(students_with_siblings) / len(students) if len(students) > 0 else 0.0
    )

    # Preference lengths
    student_pref_lengths = [len(pref[s]) for s in students if s in pref]
    avg_student_pref_len = (
        sum(student_pref_lengths) / len(student_pref_lengths)
        if len(student_pref_lengths) > 0 else 0.0
    )

    college_priority_lengths = [len(pref[c]) for c in colleges if c in pref]
    avg_college_priority_len = (
        sum(college_priority_lengths) / len(college_priority_lengths)
        if len(college_priority_lengths) > 0 else 0.0
    )

    # Levels
    nonempty_levels_students = {lev: sts for lev, sts in students_per_level.items() if len(sts) > 0}
    nonempty_levels_colleges = {lev: cols for lev, cols in levels.items() if len(cols) > 0}

    avg_students_per_level = (
        sum(len(sts) for sts in nonempty_levels_students.values()) / len(nonempty_levels_students)
        if len(nonempty_levels_students) > 0 else 0.0
    )

    avg_colleges_per_level = (
        sum(len(cols) for cols in nonempty_levels_colleges.values()) / len(nonempty_levels_colleges)
        if len(nonempty_levels_colleges) > 0 else 0.0
    )

    total_capacity = sum(cap.values())

    return {
        "region": region,
        "students": len(students),
        "colleges": len(colleges),
        "total_capacity": total_capacity,
        "capacity_minus_students": total_capacity - len(students),
        "students_with_siblings": len(students_with_siblings),
        "pct_students_with_siblings": pct_students_with_siblings,
        "families": num_families,
        "avg_family_size": avg_family_size,
        "levels_with_students": len(nonempty_levels_students),
        "levels_with_colleges": len(nonempty_levels_colleges),
        "avg_students_per_level": avg_students_per_level,
        "avg_colleges_per_level": avg_colleges_per_level,
        "avg_student_pref_len": avg_student_pref_len,
        "avg_college_priority_len": avg_college_priority_len,
    }


def build_region_stats_table(base_indir, regions, year=2023, out_csv=None, out_tex=None):
    rows = []
    for region in regions:
        rows.append(summarize_region(base_indir, region, year=year))

    df = pd.DataFrame(rows)

    # Nice formatting
    pct_cols = ["pct_students_with_siblings"]
    for col in pct_cols:
        df[col] = 100 * df[col]

    # Sort by size if you want:
    # df = df.sort_values("students", ascending=False).reset_index(drop=True)

    if out_csv is not None:
        df.to_csv(out_csv, index=False)

    if out_tex is not None:
        latex_df = df.copy()
        for col in [
            "pct_students_with_siblings",
            "avg_family_size",
            "avg_students_per_level",
            "avg_colleges_per_level",
            "avg_student_pref_len",
            "avg_college_priority_len",
        ]:
            latex_df[col] = latex_df[col].map(lambda x: f"{x:.2f}")
        latex_df.to_latex(out_tex, index=False, escape=False)

    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_indir = os.path.join(base_dir, "..", "R", "intermediate_data")

    regions = [
        "OHiggins",
        "Magallanes",
        "Arica",
        "Antofagasta",
        "Atacama",
        "Coquimbo",
        "Tarapaca",
        "Lagos",
        "Rios"
    ]

    outdir = os.path.join(base_dir, "..", "tables")
    os.makedirs(outdir, exist_ok=True)

    df = build_region_stats_table(
        base_indir=base_indir,
        regions=regions,
        year=2023,
        out_csv=os.path.join(outdir, "region_stats_2023.csv"),
        out_tex=os.path.join(outdir, "region_stats_2023.tex"),
    )

    print(df.to_string(index=False))