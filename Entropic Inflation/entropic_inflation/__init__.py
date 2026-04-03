"""Public API for the entropic inflation package."""

from .InflationProblem import InflationProblem
from ._about import about
from ._version import __version__
from .lp.InflationLP import EntropicSolveResult, FarkasCertificate, InflationLP
from .scenarios import (
    TRIANGLE_CDD_ALL,
    TRIANGLE_CDD_REPRESENTATIVES,
    TRIANGLE_CDD_RAYS,
    TRIANGLE_COMPLEMENTARY_SPIRAL_INFLATION_DAG,
    TRIANGLE_DAG,
    TRIANGLE_SPIRAL_INFLATION_DAG,
    TRIANGLE_SPIRAL_OBSERVED_NODES,
    TRIANGLE_SPIRAL_ROW_LABELS,
    triangle_cdd_inequalities,
    triangle_cdd_representatives,
    triangle_complementary_spiral_equalities,
    triangle_complementary_spiral_problem,
    triangle_spiral_candidate_rays,
    triangle_lp_objective_from_certificate,
    triangle_problem,
    triangle_spiral_equalities,
    triangle_spiral_problem,
    triangle_violation_objective_from_certificate,
)

__all__ = [
    "InflationProblem",
    "InflationLP",
    "EntropicSolveResult",
    "FarkasCertificate",
    "about",
    "__version__",
    "TRIANGLE_DAG",
    "TRIANGLE_CDD_ALL",
    "TRIANGLE_CDD_REPRESENTATIVES",
    "TRIANGLE_CDD_RAYS",
    "TRIANGLE_COMPLEMENTARY_SPIRAL_INFLATION_DAG",
    "TRIANGLE_SPIRAL_INFLATION_DAG",
    "TRIANGLE_SPIRAL_OBSERVED_NODES",
    "TRIANGLE_SPIRAL_ROW_LABELS",
    "triangle_problem",
    "triangle_complementary_spiral_problem",
    "triangle_complementary_spiral_equalities",
    "triangle_spiral_problem",
    "triangle_spiral_equalities",
    "triangle_spiral_candidate_rays",
    "triangle_cdd_inequalities",
    "triangle_cdd_representatives",
    "triangle_lp_objective_from_certificate",
    "triangle_violation_objective_from_certificate",
]
