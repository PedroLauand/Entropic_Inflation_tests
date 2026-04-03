"""Helpers for the no-inflation triangle source scenario.

This module keeps together the two pieces that appear repeatedly in the repo:

- the 7 additional observed-variable inequalities used in the triangle CDD
  computation,
- the source/observed DAG description from which the LP equalities
  ``I(A ; B,C,a | b,c)=0``, ``I(B ; A,C,b | a,c)=0``,
  ``I(C ; A,B,c | a,b)=0``, ``I(a ; b,c)=0``, ``I(b ; a,c)=0``,
  ``I(c ; a,b)=0`` are derived automatically.
"""

from __future__ import annotations

from typing import Dict, List

from .dag import ShannonSourceModel
from shannon_tests.shannon_utils import build_shannon_inequality_matrix


TRIANGLE_OBSERVED_VARIABLES = ["A", "B", "C"]
TRIANGLE_SOURCE_NAMES = ["a", "b", "c"]
TRIANGLE_SOURCE_DAG = {
    "a": ["B", "C"],
    "b": ["A", "C"],
    "c": ["A", "B"],
}
TRIANGLE_SOURCE_MODEL = ShannonSourceModel(
    dag=TRIANGLE_SOURCE_DAG,
)
TRIANGLE_SOURCE_VARIABLES = TRIANGLE_SOURCE_MODEL.variable_names

STANDARD_OBSERVED_LABELS = [
    "H(A)",
    "H(B)",
    "H(C)",
    "H(A,B)",
    "H(A,C)",
    "H(B,C)",
    "H(A,B,C)",
]

# These are the equalities imposed on the LP entropy vector h by the triangle
# DAG. They are exposed for transparency, but applications can use
# TRIANGLE_SOURCE_MODEL directly instead of handling them manually.
TRIANGLE_SOURCE_INDEPENDENCIES = TRIANGLE_SOURCE_MODEL.implied_independencies()

TRIANGLE_ARRAY7 = [
    [0.0, -1.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, -1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, -5.0, -5.0, -5.0, 4.0, 4.0, 4.0, -2.0],
    [0.0, -3.0, -3.0, -3.0, 3.0, 2.0, 2.0, -1.0],
    [0.0, -3.0, -3.0, -3.0, 2.0, 3.0, 2.0, -1.0],
    [0.0, -3.0, -3.0, -3.0, 2.0, 2.0, 3.0, -1.0],
]

TRIANGLE_ARRAY7_DESCRIPTIONS = [
    "I(A:B) + I(A:C) <= H(A)",
    "I(A:B) + I(B:C) <= H(B)",
    "I(A:C) + I(B:C) <= H(C)",
    "I(A:B:C) + I(A:B) + I(B:C) + I(A:C) <= 1/2*(H(A)+H(B)+H(C))  [scaled by 2]",
    "I(A:B:C) + I(A:B) + I(B:C) + I(A:C) <= H(A,B)",
    "I(A:B:C) + I(A:B) + I(B:C) + I(A:C) <= H(A,C)",
    "I(A:B:C) + I(A:B) + I(B:C) + I(A:C) <= H(B,C)",
]

TRIANGLE_RAYS_FALLBACK = [
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


def triangle_cdd_rays_with_array7() -> List[Dict[str, float]]:
    """Return the CDD rays for the observed triangle cone with the 7 rows added.

    Each output is a sparse candidate dictionary on the observed entropy
    coordinates ``H(A)``, ``H(B)``, ``H(C)``, ``H(A,B)``, ``H(A,C)``,
    ``H(B,C)``, ``H(A,B,C)``.

    When ``cdd`` is not available locally, this falls back to the 10-ray list
    already committed in ``lucas_version_test.py`` and annotated there as
    coming from ``shannon_cdd``.
    """

    try:
        from shannon_tests.shannon_cdd import polytope
    except ModuleNotFoundError:
        return [_row_to_candidate(row) for row in TRIANGLE_RAYS_FALLBACK]

    M, _b, _bcap, x_caption, _idx, _vars, _meta = build_shannon_inequality_matrix(
        ["A", "B", "C"]
    )
    label_to_idx = {label: i for i, label in enumerate(x_caption)}
    reorder = [label_to_idx[label] for label in STANDARD_OBSERVED_LABELS]
    shannon_rows = [[0.0] + [-row[i] for i in reorder] for row in M]

    generators = polytope(TRIANGLE_ARRAY7 + shannon_rows)
    rays = []
    for row in generators:
        if round(float(row[0]), 12) != 0.0:
            continue
        rays.append(_row_to_candidate(row[1:]))
    return rays


def triangle_all_ones_candidate() -> Dict[str, float]:
    """Return the distinguished observed candidate h=(1,1,1,1,1,1,1)."""

    return {
        "A": 1.0,
        "B": 1.0,
        "C": 1.0,
        "A,B": 1.0,
        "A,C": 1.0,
        "B,C": 1.0,
        "A,B,C": 1.0,
    }


def triangle_array7_objectives() -> List[Dict[str, object]]:
    """Return the 7 observed-variable inequalities as objective dictionaries.

    Each objective acts only on the observed entropy coordinates
    ``A, B, C, A,B, A,C, B,C, A,B,C``.
    """

    objectives: List[Dict[str, object]] = []
    for index, (description, row) in enumerate(
        zip(TRIANGLE_ARRAY7_DESCRIPTIONS, TRIANGLE_ARRAY7)
    ):
        objectives.append(
            {
                "index": index,
                "description": description,
                "objective": _row_to_candidate(row[1:]),
            }
        )
    return objectives


def _label_to_spec(label: str) -> str:
    if not label.startswith("H(") or not label.endswith(")"):
        raise ValueError(f"unexpected entropy label: {label}")
    return label[2:-1]


def _row_to_candidate(row: List[float] | tuple[float, ...]) -> Dict[str, float]:
    return {
        _label_to_spec(label): float(value)
        for label, value in zip(STANDARD_OBSERVED_LABELS, row)
    }
