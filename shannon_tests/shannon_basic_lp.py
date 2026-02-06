"""Basic Shannon LP test for variables A,B,C."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mosek.fusion import Expr, ObjectiveSense

from shannon_tests.shannon_utils import LP_test


def shannon_basic_lp(candidate: list[float] | None = None):
    names = ["A", "B", "C"]
    model, *_ = LP_test(
        names,
        candidate=candidate,
        candidate_names=names,
    )
    model.objective("feas", ObjectiveSense.Minimize, Expr.constTerm(0.0))
    model.solve()
    return model


if __name__ == "__main__":
    # Candidate with all entries equal to 1.
    model = shannon_basic_lp(candidate=[1.0] * 7)
    print("problem_status:", model.getProblemStatus())
    print("solution_status:", model.getPrimalSolutionStatus())
