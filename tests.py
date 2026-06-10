from pathlib import Path
import sys


DEFAULT_FILES = (
    Path(
        "/Users/iar200000/Dropbox/Dynamic priorities in stable matching/outputs/"
        "Magallanes/comparison/rada_mtbf/outputs_s=2.txt"
    ),
    Path(
        "/Users/iar200000/Dropbox/Dynamic priorities in stable matching/outputs/"
        "Magallanes/comparison/abs_hard_mtbf/outputs_s=2.txt"
    ),
)


def read_optimal_x(path):
    """Return {student_id: assigned_school_grade} from the '# Optimal x:' section."""
    assignments = {}
    in_optimal_x = False

    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("#"):
            if line.lower().startswith("# optimal x"):
                in_optimal_x = True
                continue
            if in_optimal_x:
                break
            continue

        if not in_optimal_x:
            continue

        student_id, school_grade, value = line.split()[:3]
        if float(value) != 0:
            assignments[student_id] = school_grade

    if not in_optimal_x:
        raise ValueError(f"No '# Optimal x:' section found in {path}")

    return assignments


def compare_assignments(path_a, path_b):
    assignments_a = read_optimal_x(path_a)
    assignments_b = read_optimal_x(path_b)

    all_students = sorted(set(assignments_a) | set(assignments_b), key=int)
    differences = [
        (student_id, assignments_a.get(student_id), assignments_b.get(student_id))
        for student_id in all_students
        if assignments_a.get(student_id) != assignments_b.get(student_id)
    ]

    print(f"File A: {path_a}")
    print(f"File B: {path_b}")
    print(f"Assignments in A: {len(assignments_a)}")
    print(f"Assignments in B: {len(assignments_b)}")
    print(f"Differences: {len(differences)}")

    if differences:
        print("\nstudent_id | A assignment | B assignment")
        print("-" * 42)
        for student_id, a_assignment, b_assignment in differences:
            print(f"{student_id} | {a_assignment} | {b_assignment}")


if __name__ == "__main__":
    files = tuple(Path(arg) for arg in sys.argv[1:]) or DEFAULT_FILES
    if len(files) != 2:
        raise SystemExit("Usage: python tests.py [output_file_a output_file_b]")

    compare_assignments(*files)
