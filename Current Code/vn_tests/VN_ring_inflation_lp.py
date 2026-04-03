"""LP feasibility tests for von-Neumann entropies using MOSEK."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import sys
import mosek
from mosek.fusion import (
    AccSolutionStatus,
    Domain,
    Expr,
    Model,
    ObjectiveSense,
    ProblemStatus,
    SolutionStatus,
    Variable,
)

from entropy_utils import (
    LP_test,
    build_farkas_model,
    entropic_caption,
    solve_farkas_model,
)


def add_linear_equality_constraint(
    model: Model,
    x: Variable,
    label_to_index: Dict[str, int],
    coeffs: Dict[str, float],
    *,
    constraints_meta: List[Dict[str, object]] | None = None,
) -> None:
    """Add linear equality sum_i coeffs[i] * S(i) = 0."""
    expr = Expr.constTerm(0.0)
    vec = [0.0] * len(label_to_index)
    for label, coef in coeffs.items():
        idx = label_to_index[label]
        expr = Expr.add(expr, Expr.mul(coef, x.index(idx)))
        vec[idx] += coef
    constr = model.constraint(expr, Domain.equalsTo(0.0))
    if constraints_meta is not None:
        constraints_meta.append(
            {"constraint": constr, "coeffs": vec, "const": 0.0, "sense": "eq"}
        )


def extract_infeasibility_certificate(
    model: Model,
    constraints_meta: Sequence[Dict[str, object]],
    label_to_index: Dict[str, int],
    row_labels: Sequence[str],
    *,
    tol: float = 1e-8,
) -> Tuple[Dict[str, float], float, Dict[str, float], Dict[str, float]] | None:
    """Return (coeffs, const, extras) for a separating inequality if infeasible.

    - coeffs: coefficients for row_labels
    - const: constant term
    - extras: non-negligible coefficients on other labels
    """
    if model.getProblemStatus() != ProblemStatus.PrimalInfeasible:
        return None

    task = model.getTask()
    solsta = task.getsolsta(mosek.soltype.itr)
    if solsta != mosek.solsta.prim_infeas_cer:
        return None

    y = task.gety(mosek.soltype.itr)
    if len(y) != len(constraints_meta):
        raise RuntimeError(
            f"constraint count mismatch: task has {len(y)} duals, "
            f"but constraints_meta has {len(constraints_meta)} entries"
        )

    n = len(label_to_index)
    full_coeffs = [0.0] * n
    const = 0.0
    for meta, y_i in zip(constraints_meta, y):
        y_val = float(y_i)
        coeffs = meta["coeffs"]
        for i, a in enumerate(coeffs):
            if a != 0.0:
                full_coeffs[i] += y_val * a
        const += y_val * float(meta["const"])

    row_set = set(row_labels)
    coeffs_out: Dict[str, float] = {}
    extras: Dict[str, float] = {}
    for label, idx in label_to_index.items():
        val = full_coeffs[idx]
        if abs(val) < tol:
            continue
        if label in row_set:
            coeffs_out[label] = val
        else:
            extras[label] = val

    return coeffs_out, const, extras, {
        label: full_coeffs[idx]
        for label, idx in label_to_index.items()
        if abs(full_coeffs[idx]) >= tol
    }


def build_ax_leq_b_from_meta(
    constraints_meta: Sequence[Dict[str, object]],
    *,
    signed_labels: bool = False,
) -> Tuple[List[List[float]], List[float], List[str | None]]:
    """Return (A, b, labels) for inequalities A x <= b from constraints_meta."""
    A: List[List[float]] = []
    b: List[float] = []
    labels: List[str | None] = []

    def add_ineq(r: List[float], rhs: float, lbl: str | None) -> None:
        A.append(r)
        b.append(rhs)
        labels.append(lbl)

    for meta in constraints_meta:
        coeffs = [float(v) for v in meta["coeffs"]]
        const = float(meta["const"])
        label = meta.get("label")
        sense = meta.get("sense")
        if sense == "ge":
            # a x + const >= 0  ->  -a x <= const
            add_ineq([-v for v in coeffs], const, label)
        elif sense == "eq":
            # a x + const = 0  ->  a x <= -const  and  -a x <= const
            add_ineq(coeffs, -const, label)
            neg_label = f"-{label}" if (signed_labels and label is not None) else label
            add_ineq([-v for v in coeffs], const, neg_label)
        else:
            raise RuntimeError(f"unknown constraint sense: {sense}")

    return A, b, labels


if __name__ == "__main__":
    # Example: named variables with symmetry + separability + fixed row constraints.
    names = [
        ["A0", "B0", "C0", "A1", "B1", "C1"],
    ]
    indep_input = [
        ["A0,B0", "A1,B1"],
        ["B0,C0", "B1,C1"],
        ["A0,C0", "A1,C1"],
    ]
    separability_input: list[list[str]] = []
    symmetry_input = [
        ["A0", "A1"],
        ["B0", "B1"],
        ["C0", "C1"],
        ["A0,B0", "A1,B1"],
        ["A0,C0", "A1,C1"],
        ["B0,C0", "B1,C1"],
        ["A0,B0,C0", "A1,B1,C1"],
        ["A0,B0,C1", "A1,B1,C0"],
        ["A0,C1", "A1,C0"],
        ["B0,C1", "B1,C0"],
    ]

    row_labels_A0 = [
        "S(A0)",
        "S(B0)",
        "S(C0)",
        "S(A0,B0)",
        "S(A0,C0)",
        "S(B0,C0)",
    ]
    # Rays from the basic von-Neumann entropy polytope for ABC.
    # Caption order: [S(A), S(B), S(AB), S(C), S(AC), S(BC), S(ABC)]
    rays = [
        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    ]
    # Drop S(A,B,C) (last col) and reorder to match row_labels_A0:
    # [S(A), S(B), S(C), S(AB), S(AC), S(BC)]
    rays_a0 = [[row[0], row[1], row[3], row[2], row[4], row[5]] for row in rays]

    feasible_rows = []
    infeasible_rows = []
    tol = 1e-8

    candidate_names = ["A0", "B0", "C0"]
    candidate_caption = entropic_caption(candidate_names)
    for i, row in enumerate(rays):
        label_to_value = dict(zip(row_labels_A0, rays_a0[i]))
        candidate = []
        for label in candidate_caption:
            if label in label_to_value:
                candidate.append(label_to_value[label])
            else:
                candidate.append("")
        (
            model,
            x,
            label_to_index,
            labels,
            var_names,
            constraints_meta,
            matrix,
        ) = LP_test(
            names,
            indep_input=indep_input,
            separability_input=separability_input,
            symmetry_input=symmetry_input,
            candidate=candidate,
            candidate_names=candidate_names,
            return_matrix=True,
        )
        farkas_model, y = build_farkas_model(matrix["M"], matrix["b"])
        obj, expr, y_vals = solve_farkas_model(
            farkas_model, y, b_caption=matrix["b_caption"]
        )
        is_feasible = obj >= -tol
        status = "feasible" if is_feasible else "infeasible"
        print(f"ray {i} b^T y: {obj:g} -> {status}")
        if is_feasible:
            feasible_rows.append((i, row))
        else:
            infeasible_rows.append((i, row))

    print("feasible_rows:", feasible_rows)
    print("infeasible_rows:", infeasible_rows)
    print("feasible_count:", len(feasible_rows))
    print("infeasible_count:", len(infeasible_rows))
