"""Scenario presets and helper constructors."""

from .triangle import (
    TRIANGLE_CDD_REPRESENTATIVES,
    TRIANGLE_CDD_RAYS,
    TRIANGLE_COMPLEMENTARY_SPIRAL_INFLATION_DAG,
    TRIANGLE_DAG,
    TRIANGLE_SPIRAL_INFLATION_DAG,
    TRIANGLE_SPIRAL_OBSERVED_NODES,
    TRIANGLE_SPIRAL_ROW_LABELS,
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
    "TRIANGLE_DAG",
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
    "triangle_cdd_representatives",
    "triangle_lp_objective_from_certificate",
    "triangle_violation_objective_from_certificate",
]
