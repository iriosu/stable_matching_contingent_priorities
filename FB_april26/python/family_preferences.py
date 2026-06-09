import os
import generate_inputs as genin


def build_families_from_siblings(siblings):
    visited = set()
    families = []

    for student in siblings:
        if student in visited:
            continue

        stack = [student]
        family = set()

        while stack:
            u = stack.pop()
            if u in family:
                continue
            family.add(u)
            for v in siblings.get(u, []):
                if v not in family:
                    stack.append(v)

        visited.update(family)
        families.append(sorted(family))

    return families


def preference_list(pref, student):
    if student not in pref:
        return tuple()
    return tuple(pref[student][k] for k in sorted(pref[student]))


def summarize_region(instance_file):
    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn = genin.read_instance(instance_file)

    families = build_families_from_siblings(siblings)
    sibling_families = [fam for fam in families if len(fam) >= 2]

    num_same = 0
    num_diff = 0

    for family in sibling_families:
        fam_prefs = {s: preference_list(pref, s) for s in family}
        if len(set(fam_prefs.values())) == 1:
            num_same += 1
        else:
            num_diff += 1

    return {
        "families_with_siblings": len(sibling_families),
        "same_preferences": num_same,
        "different_preferences": num_diff,
    }


def build_families(siblings):
    visited = set()
    families = []

    for s in siblings:
        if s in visited:
            continue
        stack = [s]
        fam = set()
        while stack:
            u = stack.pop()
            if u in fam:
                continue
            fam.add(u)
            for v in siblings.get(u, []):
                if v not in fam:
                    stack.append(v)
        visited.update(fam)
        families.append(sorted(fam))
    return families


def pref_list(pref, s):
    if s not in pref:
        return []
    return [pref[s][k] for k in sorted(pref[s])]


def is_subsequence(short, long):
    it = iter(long)
    return all(x in it for x in short)


def has_common_master_order(pref_lists):
    """
    Check if all lists are consistent with a single master ordering.
    """
    # candidate master = longest list
    master = max(pref_lists, key=len)

    # check if all are subsequences of this master
    for p in pref_lists:
        if not is_subsequence(p, master):
            return False

    return True


def check_families(instance_file):
    students, colleges, pref, cap, siblings, levels, students_per_level, Tp, Tn, Sp, Sn = \
        genin.read_instance(instance_file)

    families = build_families(siblings)

    ok = 0
    bad = 0

    for fam in families:
        if len(fam) <= 1:
            continue

        pref_lists = [pref_list(pref, s) for s in fam]

        if has_common_master_order(pref_lists):
            ok += 1
        else:
            bad += 1
            print("\nInconsistent family:", fam)
            for s, p in zip(fam, pref_lists):
                print(f"  {s}: {p}")

    print("\nSummary")
    print("Consistent families:", ok)
    print("Inconsistent families:", bad)



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
        "Rios"
    ]

    # check if siblings have the same preference lists
    for region in regions:
        instance_file = os.path.join(base_indir, region, "2023", "instance.txt")
        if not os.path.exists(instance_file):
            print(f"{region}: missing instance.txt")
            continue

        stats = summarize_region(instance_file)
        print(
            f"{region}: "
            f"families={stats['families_with_siblings']}, "
            f"same={stats['same_preferences']}, "
            f"different={stats['different_preferences']}"
        )

        
    # check if family siblings have preferences derived from the same master list
    base_dir = os.path.dirname(os.path.abspath(__file__))
    instance_file = os.path.join(
        base_dir, "..", "R", "intermediate_data", "OHiggins", "2023", "instance.txt"
    )

    check_families(instance_file)