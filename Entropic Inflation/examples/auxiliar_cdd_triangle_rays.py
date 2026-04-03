"""Recreate the old triangle CDD ray computation with the new package layout.

This example:

1. builds the 3-variable elemental Shannon cone for ``A,B,C``,
2. appends the seven observed triangle inequalities by hand,
3. computes the generators of the resulting cone with ``pycddlib``,
4. prints the extreme rays in the standard observed entropy coordinates.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cdd

from entropic_inflation.lp import build_elemental_shannon_cone


STANDARD_LABELS = [
    "H(A)",
    "H(B)",
    "H(C)",
    "H(A,B)",
    "H(A,C)",
    "H(B,C)",
    "H(A,B,C)",
]

# Each row encodes b + a.h >= 0 with h ordered as:
# [H(A), H(B), H(C), H(A,B), H(A,C), H(B,C), H(A,B,C)].
TRIANGLE_ROWS = [
    [0.0, -1.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, -1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, -5.0, -5.0, -5.0, 4.0, 4.0, 4.0, -2.0],
    [0.0, -3.0, -3.0, -3.0, 3.0, 2.0, 2.0, -1.0],
    [0.0, -3.0, -3.0, -3.0, 2.0, 3.0, 2.0, -1.0],
    [0.0, -3.0, -3.0, -3.0, 2.0, 2.0, 3.0, -1.0],
]


def polytope(array: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return the cdd generators for an H-representation array."""
    if not array:
        raise ValueError("array must be non-empty")

    if hasattr(cdd, "matrix_from_array"):
        mat = cdd.matrix_from_array(array, rep_type=cdd.RepType.INEQUALITY)
    else:
        mat = cdd.Matrix(array, number_type="fraction")
        mat.rep_type = cdd.RepType.INEQUALITY

    poly = cdd.polyhedron_from_matrix(mat)
    generators = cdd.copy_generators(poly)
    return generators.array


def build_observed_shannon_rows() -> list[list[float]]:
    """Return the 3-variable Shannon cone rows in cdd H-representation form."""
    matrix, tuple_caption = build_elemental_shannon_cone(3)
    index_by_subset = {tuple(subset): i for i, subset in enumerate(tuple_caption)}
    reorder = [
        index_by_subset[(0,)],
        index_by_subset[(1,)],
        index_by_subset[(2,)],
        index_by_subset[(0, 1)],
        index_by_subset[(0, 2)],
        index_by_subset[(1, 2)],
        index_by_subset[(0, 1, 2)],
    ]
    return [[0.0] + [-float(row[i]) for i in reorder] for row in matrix]


def triangle_cone_rays() -> list[list[float]]:
    """Return the extreme rays in observed entropy coordinates."""
    constraints = TRIANGLE_ROWS + build_observed_shannon_rows()
    generators = polytope(constraints)
    return [list(map(float, row[1:])) for row in generators if round(float(row[0]), 12) == 0.0]


def format_ray(ray: Iterable[float]) -> str:
    return ", ".join(f"{label}={value:g}" for label, value in zip(STANDARD_LABELS, ray))


if __name__ == "__main__":
    rays = triangle_cone_rays()
    print(f"Rays from triangle rows + 3-variable Shannon cone: {len(rays)}")
    for ray in rays:
        print("ray:", format_ray(ray))
