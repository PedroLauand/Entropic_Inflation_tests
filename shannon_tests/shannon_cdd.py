"""Calculate extreme points/rays using pycddlib (cddlib sampleh1.ine)."""
from pprint import pprint

import cdd


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
    # Basic Shannon-type inequalities (each row · h >= 0).
    basic_shannon_array = [
        # Nonnegativity of single entropies.
        [0, 1, 0, 0, 0, 0, 0, 0],  # H(A) >= 0
        [0, 0, 1, 0, 0, 0, 0, 0],  # H(B) >= 0
        [0, 0, 0, 1, 0, 0, 0, 0],  # H(C) >= 0
        # Conditional entropies >= 0.
        [0, 0, -1, 0, 1, 0, 0, 0],  # H(A|B) = H(AB) - H(B) >= 0
        [0, 0, 0, -1, 0, 1, 0, 0],  # H(A|C) = H(AC) - H(C) >= 0
        [0, 0, 0, -1, 0, 0, 1, 0],  # H(B|C) = H(BC) - H(C) >= 0
        [0, 0, 0, 0, 0, 0, -1, 1],  # H(A|BC) = H(ABC) - H(BC) >= 0
        [0, 0, 0, 0, 0, -1, 0, 1],  # H(B|AC) = H(ABC) - H(AC) >= 0
        [0, 0, 0, 0, -1, 0, 0, 1],  # H(C|AB) = H(ABC) - H(AB) >= 0
        # Conditional mutual informations >= 0 (submodularity).
        [0, 0, -1, 0, 1, 0, 1, -1],  # I(A;C|B) >= 0
        [0, -1, 0, 0, 1, 1, 0, -1],  # I(B;C|A) >= 0
        [0, 0, 0, -1, 0, 1, 1, -1],  # I(A;B|C) >= 0
    ]
    # Generators caption (same order as h above):
    # [t, H(A), H(B), H(C), H(A,B), H(A,C), H(B,C), H(A,B,C)]
    # t = 0 -> ray (direction), t = 1 -> vertex (point)
    constraints = array + basic_shannon_array
    generators_basic = polytope(basic_shannon_array)
    generators_constraints = polytope(constraints)
    def _rounded(row: list[float]) -> tuple[float, ...]:
        return tuple(round(x, 5) for x in row)

    # Compare after rounding each entry to 5 decimal places.
    basic_set = {_rounded(row) for row in generators_basic}
    constraints_set = {_rounded(row) for row in generators_constraints}
    generators = [row for row in generators_constraints if _rounded(row) not in basic_set]
    caption = ["H(A)", "H(B)", "H(C)", "H(A,B)", "H(A,C)", "H(B,C)", "H(A,B,C)"]

    only_basic = sorted(basic_set - constraints_set)
    if only_basic:
        print("Only in basic_shannon_array (rounded to 5 decimals):")
        for row in only_basic:
            t = row[0]
            kind = "ray" if t == 0 else "extreme point"
            values = [f"{name}={value}" for name, value in zip(caption, row[1:])]
            print(f"{kind}: " + ", ".join(values))
        print("")

    print("All generators from constraints (rounded to 5 decimals):")
    for row in generators_constraints:
        r = _rounded(row)
        t = r[0]
        kind = "ray" if t == 0 else "extreme point"
        values = [f"{name}={value}" for name, value in zip(caption, r[1:])]
        print(f"{kind}: " + ", ".join(values))
