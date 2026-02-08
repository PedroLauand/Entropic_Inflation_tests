"""Compare Shannon-utility rays against A_1 conditions and report violations."""

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
from shannon_tests.shannon_utils import build_shannon_inequality_matrix


def polytope(array: list[list[float]]) -> list[list[float]]:
    if hasattr(cdd, "matrix_from_array"):
        mat = cdd.matrix_from_array(array, rep_type=cdd.RepType.INEQUALITY)
    else:
        mat = cdd.Matrix(array, number_type="fraction")
        mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.polyhedron_from_matrix(mat)
    ext = cdd.copy_generators(poly)
    return ext.array


standard_labels = [
    "H(A)",
    "H(B)",
    "H(C)",
    "H(A,B)",
    "H(A,C)",
    "H(B,C)",
    "H(A,B,C)",
]


def expr_from_row(row: np.ndarray) -> str:
    terms = []
    for coef, lab in zip(row, standard_labels):
        if abs(coef) < 1e-12:
            continue
        terms.append(f"{coef:g}*{lab}")
    return " + ".join(terms) if terms else "0"


def fmt_ray(ray: np.ndarray) -> str:
    return ", ".join(f"{lab}={val:g}" for lab, val in zip(standard_labels, ray))


def main() -> None:
    # Rays from Shannon utilities (full Shannon constraints).
    M, _b, _bcap, x_caption, _idx, _vars, _meta = build_shannon_inequality_matrix(
        ["A", "B", "C"]
    )
    M = np.asarray(M, dtype=float)

    # M h <= 0 -> (-M) h >= 0 for cdd H-representation.
    rows = [[0.0] + (-row).tolist() for row in M]

    ext = polytope(rows)
    rays = [row for row in ext if row[0] == 0]

    # Map Shannon-utils caption to standard A_1 order.
    label_to_idx = {label: i for i, label in enumerate(x_caption)}
    reorder = [label_to_idx[label] for label in standard_labels]

    # A_1 inequalities (Lucas) interpreted as A_1 h <= 0.
    A_1, _ = desc_entro.desigualdades_basicas(3)
    A_1 = np.asarray(A_1, dtype=float)

    violations = []
    for i, row in enumerate(rays):
        h = np.array(row[1:], dtype=float)[reorder]
        vals = A_1 @ h
        bad = [(j, vals[j]) for j in range(len(vals)) if vals[j] > 1e-9]
        if bad:
            violations.append((i, h, bad))

    print(f"rays from shannon_utils (full Shannon): {len(rays)}")
    print(f"rays violating A_1 h<=0: {len(violations)}")

    for i, h, bad in violations:
        print(f"\nray {i}: {fmt_ray(h)}")
        for j, val in bad:
            expr = expr_from_row(A_1[j])
            print(f"  violates A_1 row {j}: ({expr}) <= 0  with value {val:g}")


if __name__ == "__main__":
    main()
