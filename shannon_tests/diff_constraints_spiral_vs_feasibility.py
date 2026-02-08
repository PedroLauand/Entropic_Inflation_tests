"""List constraint rows present in feasibility_test but not in spiral LP (same variables)."""

from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEAS_PATH = ROOT / "shannon_tests" / "feasibility_test"
if str(FEAS_PATH) not in sys.path:
    sys.path.insert(0, str(FEAS_PATH))

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

from Feas_problem import Equalities_spiral_inflation  # type: ignore
from desc_entro.desc_entro import desigualdades_basicas  # type: ignore
from shannon_tests.shannon_utils import build_shannon_inequality_matrix


NAMES = ["A0", "B0", "C0", "A1", "B1", "C1"]
NAME_TO_IDX = {name: i for i, name in enumerate(NAMES)}


def tuple_label(tup: Tuple[int, ...]) -> str:
    names = [NAMES[i] for i in tup]
    return "H(" + ",".join(names) + ")"


def normalize_row(row: Sequence[float], *, tol: float = 1e-12) -> Tuple[float, ...]:
    if len(row) == 0:
        return tuple()
    max_abs = max(abs(v) for v in row)
    if max_abs < tol:
        return tuple(0.0 for _ in row)
    scaled = [v / max_abs for v in row]
    return tuple(round(v, 12) for v in scaled)


def row_to_expr(row: Sequence[float], labels: Sequence[str]) -> str:
    terms = []
    for coef, lab in zip(row, labels):
        if abs(coef) < 1e-12:
            continue
        terms.append(f"{coef:g}*{lab}")
    return " + ".join(terms) if terms else "0"


def build_spiral_constraints_in_vetorH_order(
    vetor_H: Sequence[Tuple[int, ...]],
) -> np.ndarray:
    # Full Shannon constraints from shannon_utils: M h <= 0
    M, _b, _bcap, x_caption, _idx, _vars, _meta = build_shannon_inequality_matrix(NAMES)
    M = np.asarray(M, dtype=float)

    # Map shannon_utils labels -> vetor_H index
    label_to_vetor_idx: Dict[str, int] = {
        tuple_label(tup): i for i, tup in enumerate(vetor_H)
    }

    def label_to_index(label: str) -> int:
        if label not in label_to_vetor_idx:
            raise ValueError(f"label not found in vetor_H: {label}")
        return label_to_vetor_idx[label]

    # Expand M rows into vetor_H order
    expanded_rows = []
    for row in M:
        full = np.zeros(len(vetor_H), dtype=float)
        for coef, label in zip(row, x_caption):
            if abs(coef) < 1e-12:
                continue
            full[label_to_index(label)] = coef
        expanded_rows.append(full)

    # Independence constraints (equalities -> two inequalities)
    indep_input = [
        ["A1", "B0,B1,C1"],
        ["B1", "C0,C1,A1"],
        ["C1", "A0,A1,B1"],
        ["A1", "B1,C1"],
        ["B1", "C1"],
        ["B1", "C0,A1"],
        ["C1", "A0,B1"],
        ["A0", "C1"],
        ["B0", "A1"],
        ["C0", "B1"],
    ]

    def parse_vars(item: str) -> List[str]:
        return [t.strip() for t in item.split(",") if t.strip()]

    def label_for_vars(vars_: Sequence[str]) -> str:
        idxs = sorted(NAME_TO_IDX[v] for v in vars_)
        return tuple_label(tuple(idxs))

    for x_str, y_str in indep_input:
        x_vars = parse_vars(x_str)
        y_vars = parse_vars(y_str)
        xy_vars = x_vars + [v for v in y_vars if v not in x_vars]
        x_label = label_for_vars(x_vars)
        y_label = label_for_vars(y_vars)
        xy_label = label_for_vars(xy_vars)
        full = np.zeros(len(vetor_H), dtype=float)
        full[label_to_index(x_label)] += 1.0
        full[label_to_index(y_label)] += 1.0
        full[label_to_index(xy_label)] -= 1.0
        expanded_rows.append(full)
        expanded_rows.append(-full)

    # Symmetry constraints (equalities -> two inequalities)
    symmetry_input = [
        ["A0", "A1"],
        ["B0", "B1"],
        ["C0", "C1"],
        ["A0,B0", "A0,B1"],
        ["B0,C0", "B0,C1"],
        ["A0,C0", "A1,C0"],
    ]

    for a_str, b_str in symmetry_input:
        a_label = label_for_vars(parse_vars(a_str))
        b_label = label_for_vars(parse_vars(b_str))
        full = np.zeros(len(vetor_H), dtype=float)
        full[label_to_index(a_label)] += 1.0
        full[label_to_index(b_label)] -= 1.0
        expanded_rows.append(full)
        expanded_rows.append(-full)

    return np.asarray(expanded_rows, dtype=float)


def main() -> None:
    # Feasibility constraints: A_basic + A_eq, both as <= 0
    A_basic, vetor_H = desigualdades_basicas(6)
    A_basic = np.asarray(A_basic, dtype=float)
    A_eq = Equalities_spiral_inflation(vetor_H)
    A_feas = np.vstack([A_basic, A_eq])

    # Spiral constraints expressed on the same vetor_H basis
    A_spiral = build_spiral_constraints_in_vetorH_order(vetor_H)

    labels = [tuple_label(tup) for tup in vetor_H]

    set_feas = {normalize_row(row) for row in A_feas}
    set_spiral = {normalize_row(row) for row in A_spiral}

    only_feas = [row for row in A_feas if normalize_row(row) not in set_spiral]
    only_spiral = [row for row in A_spiral if normalize_row(row) not in set_feas]

    print("feasibility rows:", len(A_feas))
    print("spiral rows:", len(A_spiral))
    print("only in feasibility:", len(only_feas))
    print("only in spiral:", len(only_spiral))

    print("\nConditions in feasibility but not in spiral:")
    for i, row in enumerate(only_feas):
        print(f"[{i}] {row_to_expr(row, labels)} <= 0")


if __name__ == "__main__":
    main()
