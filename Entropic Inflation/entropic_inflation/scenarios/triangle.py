"""Triangle helpers for readable entropic examples."""

from __future__ import annotations

from typing import Dict, List

from ..InflationProblem import InflationProblem


TRIANGLE_DAG = {
    "a": ["B", "C"],
    "b": ["A", "C"],
    "c": ["A", "B"],
}

TRIANGLE_SPIRAL_INFLATION_DAG = {
    "a0": ["B0", "C0", "C1"],
    "a1": ["B1"],
    "b0": ["A0", "A1", "C0"],
    "b1": ["C1"],
    "c0": ["A0", "B0", "B1"],
    "c1": ["A1"],
}

TRIANGLE_COMPLEMENTARY_SPIRAL_INFLATION_DAG = {
    "a0": ["B0", "B1", "C0"],
    "a1": ["C1"],
    "b0": ["A0", "C0", "C1"],
    "b1": ["A1"],
    "c0": ["A0", "A1", "B0"],
    "c1": ["B1"],
}

TRIANGLE_SPIRAL_OBSERVED_NODES = ["A0", "A1", "B0", "B1", "C0", "C1"]

TRIANGLE_SPIRAL_ROW_LABELS = [
    "H(A0)",
    "H(B0)",
    "H(C0)",
    "H(A0,B0)",
    "H(A0,C0)",
    "H(B0,C0)",
    "H(A0,B0,C0)",
]

TRIANGLE_CDD_RAYS = [
    [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 3.0],
    [1.5, 1.0, 1.5, 2.0, 2.5, 2.0, 3.0],
    [1.5, 1.5, 1.0, 2.5, 2.0, 2.0, 3.0],
    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
    [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
]

TRIANGLE_CDD_REPRESENTATIVES = [
    {
        "name": "type_1",
        "description": "H(A,B) + H(A,C) - H(A) - H(B) - H(C) >= 0",
        "certificate": {
            "A": -1.0,
            "B": -1.0,
            "C": -1.0,
            "A,B": 1.0,
            "A,C": 1.0,
        },
    },
    {
        "name": "type_2",
        "description": "-5 H(A) - 5 H(B) - 5 H(C) + 4 H(A,B) + 4 H(A,C) + 4 H(B,C) - 2 H(A,B,C) >= 0",
        "certificate": {
            "A": -5.0,
            "B": -5.0,
            "C": -5.0,
            "A,B": 4.0,
            "A,C": 4.0,
            "B,C": 4.0,
            "A,B,C": -2.0,
        },
    },
    {
        "name": "type_3",
        "description": "-3 H(A) - 3 H(B) - 3 H(C) + 3 H(A,B) + 2 H(A,C) + 2 H(B,C) - H(A,B,C) >= 0",
        "certificate": {
            "A": -3.0,
            "B": -3.0,
            "C": -3.0,
            "A,B": 3.0,
            "A,C": 2.0,
            "B,C": 2.0,
            "A,B,C": -1.0,
        },
    },
]


def triangle_problem() -> InflationProblem:
    """Return the no-inflation triangle problem used in the first examples."""
    return InflationProblem(
        dag=TRIANGLE_DAG,
        outcomes_per_party=(2, 2, 2),
        settings_per_party=(1, 1, 1),
        inflation_level_per_source=(1, 1, 1),
    )


def triangle_spiral_problem() -> InflationProblem:
    """Return the manual-mode triangle spiral inflation problem."""
    return InflationProblem(
        dag=TRIANGLE_DAG,
        outcomes_per_party=(2, 2, 2),
        settings_per_party=(1, 1, 1),
        inflation_mode="manual",
        inflation_dag=TRIANGLE_SPIRAL_INFLATION_DAG,
        order=("A", "B", "C"),
    )


def triangle_complementary_spiral_problem() -> InflationProblem:
    """Return the complementary manual-mode triangle spiral inflation problem."""
    return InflationProblem(
        dag=TRIANGLE_DAG,
        outcomes_per_party=(2, 2, 2),
        settings_per_party=(1, 1, 1),
        inflation_mode="manual",
        inflation_dag=TRIANGLE_COMPLEMENTARY_SPIRAL_INFLATION_DAG,
        order=("A", "B", "C"),
    )


def triangle_observed_name_map(problem: InflationProblem | None = None) -> Dict[str, str]:
    """Map archived observed labels to the public entropic labels."""
    if problem is None:
        problem = triangle_problem()
    obs = problem.public_node_labels(include_latents=False)
    return {"A": obs[0], "B": obs[1], "C": obs[2]}


def triangle_spiral_equalities() -> List[Dict[str, float]]:
    """Return the observed-variable linear equalities used in the old spiral LP."""
    return [
        {"A1,B0,B1,C1": 1.0, "A1": -1.0, "B0,B1,C1": -1.0},
        {"B1,C0,C1,A1": 1.0, "B1": -1.0, "C0,C1,A1": -1.0},
        {"C1,A0,A1,B1": 1.0, "C1": -1.0, "A0,A1,B1": -1.0},
        {"A1,B1,C1": 1.0, "A1": -1.0, "B1,C1": -1.0},
        {"B1,C1": 1.0, "B1": -1.0, "C1": -1.0},
        {"B1,C0,A1": 1.0, "B1": -1.0, "C0,A1": -1.0},
        {"C1,A0,B1": 1.0, "C1": -1.0, "A0,B1": -1.0},
        {"A0,C1": 1.0, "A0": -1.0, "C1": -1.0},
        {"B0,A1": 1.0, "B0": -1.0, "A1": -1.0},
        {"C0,B1": 1.0, "C0": -1.0, "B1": -1.0},
        {"A0": 1.0, "A1": -1.0},
        {"B0": 1.0, "B1": -1.0},
        {"C0": 1.0, "C1": -1.0},
        {"A0,B0": 1.0, "A0,B1": -1.0},
        {"B0,C0": 1.0, "B0,C1": -1.0},
        {"A0,C0": 1.0, "A1,C0": -1.0},
    ]


def triangle_complementary_spiral_equalities() -> List[Dict[str, float]]:
    """Return the hand-added equalities for the complementary spiral LP."""
    return [
        {"A1,C0,C1,B1": 1.0, "A1": -1.0, "C0,C1,B1": -1.0},
        {"B1,A0,A1,C1": 1.0, "B1": -1.0, "A0,A1,C1": -1.0},
        {"C1,B0,B1,A1": 1.0, "C1": -1.0, "B0,B1,A1": -1.0},
        {"A1,B1,C1": 1.0, "A1": -1.0, "B1,C1": -1.0},
        {"B1,C1": 1.0, "B1": -1.0, "C1": -1.0},
        {"A1,C0,B1": 1.0, "A1": -1.0, "C0,B1": -1.0},
        {"B1,A0,C1": 1.0, "B1": -1.0, "A0,C1": -1.0},
        {"C1,B0,A1": 1.0, "C1": -1.0, "B0,A1": -1.0},
        {"A0,B1": 1.0, "A0": -1.0, "B1": -1.0},
        {"B0,C1": 1.0, "B0": -1.0, "C1": -1.0},
        {"C0,A1": 1.0, "C0": -1.0, "A1": -1.0},
        {"A0": 1.0, "A1": -1.0},
        {"B0": 1.0, "B1": -1.0},
        {"C0": 1.0, "C1": -1.0},
        {"A0,B0": 1.0, "A1,B0": -1.0},
        {"B0,C0": 1.0, "B1,C0": -1.0},
        {"A0,C0": 1.0, "A0,C1": -1.0},
    ]


def triangle_spiral_candidate_rays() -> List[Dict[str, object]]:
    """Return the historical 10 candidate rays on the ``A0,B0,C0`` marginal."""
    return [
        {
            "index": index,
            "values": {
                label[2:-1]: float(value)
                for label, value in zip(TRIANGLE_SPIRAL_ROW_LABELS, row)
            },
        }
        for index, row in enumerate(TRIANGLE_CDD_RAYS)
    ]


def triangle_cdd_representatives(
    problem: InflationProblem | None = None,
) -> List[Dict[str, object]]:
    """Return the three symmetry-class representatives in readable public labels."""
    if problem is None:
        problem = triangle_problem()
    name_map = triangle_observed_name_map(problem)

    translated: List[Dict[str, object]] = []
    for item in TRIANGLE_CDD_REPRESENTATIVES:
        certificate = {
            ",".join(name_map[token] for token in key.split(",")): value
            for key, value in item["certificate"].items()
        }
        translated.append(
            {
                "name": item["name"],
                "description": item["description"],
                "certificate": certificate,
            }
        )
    return translated


def triangle_lp_objective_from_certificate(
    certificate: Dict[str, float],
) -> Dict[str, float]:
    """Convert a certificate `expression >= 0` into a minimization objective."""
    return dict(certificate)


def triangle_violation_objective_from_certificate(
    certificate: Dict[str, float],
) -> Dict[str, float]:
    """Convert a certificate `expression >= 0` into a maximized violation objective."""
    return {key: -value for key, value in certificate.items()}
