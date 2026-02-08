"""Compute cdd rays for A_1 h <= 0 with extra Shannon inequalities and non-negativity."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import cdd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DESC_PATH = ROOT / "shannon_tests" / "feasibility_test" / "desc_entro"
if str(DESC_PATH) not in sys.path:
    sys.path.insert(0, str(DESC_PATH))

# Stub IPython.display if not installed (desc_entro imports it but doesn't need it here).
if "IPython" not in sys.modules:
    ipy = types.ModuleType("IPython")
    display_mod = types.ModuleType("IPython.display")

    def _display(*_args, **_kwargs):
        return None

    display_mod.display = _display
    ipy.display = display_mod
    sys.modules["IPython"] = ipy
    sys.modules["IPython.display"] = display_mod

import desc_entro  # type: ignore


def polytope(array: list[list[float]]) -> list[list[float]]:
    if not array:
        raise ValueError("array must be non-empty")
    if hasattr(cdd, "matrix_from_array"):
        mat = cdd.matrix_from_array(array, rep_type=cdd.RepType.INEQUALITY)
    else:
        mat = cdd.Matrix(array, number_type="fraction")
        mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.polyhedron_from_matrix(mat)
    ext = cdd.copy_generators(poly)
    return ext.array


def main() -> None:
    # A_1 from Lucas version (basic Shannon) then impose A_1 h <= 0.
    A_1, _caption_1_tuples = desc_entro.desigualdades_basicas(3)
    A_1 = np.asarray(A_1, dtype=float)
    A_leq = -A_1

    # The 7 extra Shannon inequalities (same as shannon_cdd.py array).
    array7 = np.asarray(
        [
            [0, -1, -1, -1, 1, 1, 0, 0],
            [0, -1, -1, -1, 1, 0, 1, 0],
            [0, -1, -1, -1, 0, 1, 1, 0],
            [0, -5, -5, -5, 4, 4, 4, -2],
            [0, -3, -3, -3, 3, 2, 2, -1],
            [0, -3, -3, -3, 2, 3, 2, -1],
            [0, -3, -3, -3, 2, 2, 3, -1],
        ],
        dtype=float,
    )

    # Non-negativity: H(S) >= 0 for all entries (7 vars).
    nonneg = np.eye(A_1.shape[1], dtype=float)
    nonneg_rows = [[0.0] + row.tolist() for row in nonneg]

    # Convert A_leq into cdd rows with b=0 and add array7 (already in [b, a...] format).
    rows = [[0.0] + row.tolist() for row in A_leq]
    rows += array7.tolist()
    rows += nonneg_rows

    generators = polytope(rows)
    rays = [row for row in generators if row[0] == 0]

    print("A_1 rows:", A_1.shape[0])
    print("array7 rows:", array7.shape[0])
    print("nonneg rows:", nonneg.shape[0])
    print("total constraints:", len(rows))
    print("generators:", len(generators))
    print("rays:", len(rays))


if __name__ == "__main__":
    main()
