"""Calculate extreme points/rays using pycddlib (cddlib sampleh1.ine)."""
from pathlib import Path
import sys

import cdd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from shannon_tests.shannon_utils import build_shannon_inequality_matrix
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "shannon_tests"))
    from shannon_utils import build_shannon_inequality_matrix

STANDARD_LABELS = [
    "H(A)",
    "H(B)",
    "H(C)",
    "H(A,B)",
    "H(A,C)",
    "H(B,C)",
    "H(A,B,C)",
]


def polytope(array: list[list[float]]) -> list[list[float]]:
    # H-representation (inequalities): b + A x >= 0.
    # Each row is [b, a1, a2, ...].
    if not array:
        raise ValueError("array must be a non-empty list of rows")

    # Build the matrix in inequality (H) representation.
    # Older/alternate builds may not expose matrix_from_array; fall back to Matrix.
    if hasattr(cdd, "matrix_from_array"):
        mat = cdd.matrix_from_array(array, rep_type=cdd.RepType.INEQUALITY)
    else:
        mat = cdd.Matrix(array, number_type="fraction")
        mat.rep_type = cdd.RepType.INEQUALITY

    # Create the polyhedron and compute its generators (V-representation).
    poly = cdd.polyhedron_from_matrix(mat)
    ext = cdd.copy_generators(poly)

    # ext.array rows are [t, x1, x2, ...]
    # t=1 => vertex (point), t=0 => ray (direction).
    return ext.array


if __name__ == "__main__":
    # Entropic-vector caption for this problem (8 slots):
    # h = ["1", "H(A)", "H(B)", "H(C)", "H(A,B)", "H(A,C)", "H(B,C)", "H(A,B,C)"]
    #
    # Inequalities encoded below (each row · h >= 0):
    # 1) I(A:B) + I(A:C) <= H(A)
    # 2) I(A:B) + I(B:C) <= H(B)
    # 3) I(A:C) + I(B:C) <= H(C)
    # 4) I(A:B:C) + I(A:B) + I(B:C) + I(A:C) <= 1/2 (H(A)+H(B)+H(C))  (scaled by 2)
    # 5) I(A:B:C) + I(A:B) + I(B:C) + I(A:C) <= H(A,B)
    # 6) I(A:B:C) + I(A:B) + I(B:C) + I(A:C) <= H(A,C)
    # 7) I(A:B:C) + I(A:B) + I(B:C) + I(A:C) <= H(B,C)
    array = [
        [0, -1, -1, -1, 1, 1, 0, 0],
        [0, -1, -1, -1, 1, 0, 1, 0],
        [0, -1, -1, -1, 0, 1, 1, 0],
        [0, -5, -5, -5, 4, 4, 4, -2],
        [0, -3, -3, -3, 3, 2, 2, -1],
        [0, -3, -3, -3, 2, 3, 2, -1],
        [0, -3, -3, -3, 2, 2, 3, -1],
    ]
    # Full Shannon inequalities from the builder: M h <= 0.
    M, _b, _bcap, x_caption, _idx, _vars, _meta = build_shannon_inequality_matrix(
        ["A", "B", "C"]
    )
    label_to_idx = {label: i for i, label in enumerate(x_caption)}
    reorder = [label_to_idx[label] for label in STANDARD_LABELS]
    shannon_rows = [[0.0] + [-row[i] for i in reorder] for row in M]
    # Generators caption (same order as h above):
    # [t, H(A), H(B), H(C), H(A,B), H(A,C), H(B,C), H(A,B,C)]
    # t = 0 -> ray (direction), t = 1 -> vertex (point)
    constraints = array + shannon_rows
    generators_constraints = polytope(constraints)
    def _rounded(row: list[float]) -> tuple[float, ...]:
        return tuple(round(x, 5) for x in row)

    caption = STANDARD_LABELS
    rays = [row for row in generators_constraints if _rounded(row)[0] == 0]

    print(f"Rays from constraints (rounded to 5 decimals): {len(rays)}")
    for row in rays:
        r = _rounded(row)
        values = [f"{name}={value}" for name, value in zip(caption, r[1:])]
        print("ray: " + ", ".join(values))
