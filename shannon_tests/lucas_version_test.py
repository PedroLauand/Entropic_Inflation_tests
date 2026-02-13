"""Compare basic Shannon inequalities from feasibility_test vs shannon_utils (n=3)."""

from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Make feasibility_test importable.
FEAS_PATH = ROOT / "shannon_tests" / "feasibility_test"
if str(FEAS_PATH) not in sys.path:
    sys.path.insert(0, str(FEAS_PATH))

# Make feasibility_test/desc_entro importable.
DESC_PATH = FEAS_PATH / "desc_entro"
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

from shannon_tests.shannon_utils import build_shannon_inequality_matrix
from Feas_problem import (  # type: ignore
    Equalities_spiral_inflation,
    Feasibility_Entropic_vector,
)
import desc_entro  # type: ignore


def tuple_caption_to_labels(
    caption: Sequence[Tuple[int, ...]],
    names: Sequence[str],
) -> List[str]:
    labels = []
    for item in caption:
        vars_ = [names[i] for i in item]
        labels.append("H(" + ",".join(vars_) + ")")
    return labels


def normalize_row(row: Sequence[float], *, tol: float = 1e-12) -> Tuple[float, ...]:
    if len(row) == 0:
        return tuple()
    max_abs = max(abs(v) for v in row)
    if max_abs < tol:
        return tuple(0.0 for _ in row)
    scaled = [v / max_abs for v in row]
    return tuple(round(v, 12) for v in scaled)


def unique_normalized_rows(matrix: np.ndarray) -> set[Tuple[float, ...]]:
    return {normalize_row(row.tolist()) for row in matrix}


def main() -> None:
    names = ["A", "B", "C"]

    # A_1 from feasibility_test (Lucas version).
    A_1, caption_1_tuples = desc_entro.desigualdades_basicas(3)
    caption_1 = tuple_caption_to_labels(caption_1_tuples, names)

    # A_2 from shannon_utils (current Shannon test code).
    M, _b, _bcap, caption_2, _idx, _vars, _meta = build_shannon_inequality_matrix(names)
    A_2 = np.asarray(M, dtype=float)

    # Translate caption_1 -> caption_2 order so columns match.
    label_to_idx = {label: i for i, label in enumerate(caption_2)}
    reorder = [label_to_idx[label] for label in caption_1]
    A_2_reordered = A_2[:, reorder]

    print("Caption translation (A_1 -> A_2 index):")
    for label in caption_1:
        print(f"  {label} -> col {label_to_idx[label]} in A_2")

    print("\nCounts:")
    print(f"  A_1 rows: {A_1.shape[0]}")
    print(f"  A_2 rows: {A_2.shape[0]}")

    # Compare normalized rows (scale-invariant).
    set_1 = unique_normalized_rows(np.asarray(A_1, dtype=float))
    set_2 = unique_normalized_rows(A_2_reordered)

    print("\nUnique rows after normalization:")
    print(f"  A_1 unique: {len(set_1)}")
    print(f"  A_2 unique: {len(set_2)}")
    print(f"  Only in A_1: {len(set_1 - set_2)}")
    print(f"  Only in A_2: {len(set_2 - set_1)}")

    print("\nHow this comparison is done:")
    print("1) A_1 is built by desc_entro.desigualdades_basicas(3) from feasibility_test.")
    print("2) A_2 is built by shannon_utils.build_shannon_inequality_matrix([A,B,C]).")
    print("3) Columns of A_2 are reordered to match A_1's caption order.")
    print("4) Rows are normalized by their max-abs coefficient to ignore scaling.")
    print("5) We compare sets of normalized rows for overlap and differences.")

    # Rays from shannon_cdd (basic Shannon + array, 10 rays).
    rays = [
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

    # Check each ray against A_1 h <= 0 (Lucas basic inequalities).
    ray_labels = [
        "H(A)",
        "H(B)",
        "H(C)",
        "H(A,B)",
        "H(A,C)",
        "H(B,C)",
        "H(A,B,C)",
    ]
    label_to_idx_1 = {label: i for i, label in enumerate(caption_1)}
    tol = 1e-9

    print("\nRay check against A_1 (Lucas basic inequalities):")
    for i, ray in enumerate(rays):
        h = np.zeros(len(caption_1), dtype=float)
        for lbl, val in zip(ray_labels, ray):
            h[label_to_idx_1[lbl]] = float(val)
        residual = A_1 @ h
        max_violation = float(np.max(residual))
        status = "feasible" if max_violation <= tol else f"violates (max={max_violation:.3g})"
        print(f"  ray {i}: {status}")

    # LP feasibility test (Lucas feasibility_test with fixed entries).
    # Uses 6-variable entropic vector (A0,B0,C0,A1,B1,C1) and fixes A0/B0/C0 marginals.
    A_basic_6, vetor_H = desc_entro.desigualdades_basicas(6)
    A_basic_6 = np.asarray(A_basic_6, dtype=float)
    A_eq = Equalities_spiral_inflation(vetor_H)
    A = np.vstack([A_basic_6, A_eq])

    label_to_tuple = {
        "H(A0)": (0,),
        "H(B0)": (1,),
        "H(C0)": (2,),
        "H(A0,B0)": (0, 1),
        "H(A0,C0)": (0, 2),
        "H(B0,C0)": (1, 2),
        "H(A0,B0,C0)": (0, 1, 2),
    }
    row_labels = list(label_to_tuple.keys())

    # Map A,B,C ray entries into the A0,B0,C0 slots.
    print("\nRay check against Lucas feasibility LP (A h <= 0 with fixed entries):")
    for i, ray in enumerate(rays):
        C = np.zeros((len(row_labels), len(vetor_H)), dtype=float)
        d = np.zeros(len(row_labels), dtype=float)
        for r, (lbl, val) in enumerate(zip(row_labels, ray)):
            tup = label_to_tuple[lbl]
            idx = vetor_H.index(tup)
            C[r, idx] = 1.0
            d[r] = float(val)
        status = Feasibility_Entropic_vector(A, C, d)
        ok = status == 2  # gp.GRB.OPTIMAL
        print(f"  ray {i}: {'feasible' if ok else 'infeasible'}")


if __name__ == "__main__":
    main()
