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

# Make feasibility_test/desc_entro importable.
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

from shannon_tests.shannon_utils import build_shannon_inequality_matrix
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


if __name__ == "__main__":
    main()
