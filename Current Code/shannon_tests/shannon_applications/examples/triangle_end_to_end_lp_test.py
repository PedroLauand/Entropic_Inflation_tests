"""End-to-end DAG-based Shannon LP test with an optional Farkas certificate."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shannon_tests.shannon_applications import ShannonSourceModel


if __name__ == "__main__":
    triangle = ShannonSourceModel(
        dag={
            "a": ["B", "C"],
            "b": ["A", "C"],
            "c": ["A", "B"],
        }
    )

    result = triangle.solve(
        candidate={
            "A": 1.0,
            "B": 1.0,
            "C": 1.0,
            "A,B": 1.0,
            "A,C": 1.0,
            "B,C": 1.0,
            "A,B,C": 1.0,
        },
        basic_inequalities="elemental",
        include_farkas_certificate=True,
    )

    print("sources:", triangle.sources)
    print("observed:", triangle.observed)
    print("problem_status:", result.problem_status)
    print("solution_status:", result.solution_status)
    print("is_feasible:", result.is_feasible)

    if result.farkas_certificate is not None:
        print("farkas_objective:", result.farkas_certificate.objective_value)
        print("farkas_expression:", result.farkas_certificate.expression)
