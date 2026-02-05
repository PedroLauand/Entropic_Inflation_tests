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

from entropy_utils import LP_test


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
        ["A0", "B0", "C0"],
    ]
    indep_input = []
    separability_input = [
        ["A0", "C0"],
    ]

    row_labels = [
        "S(A0)",
        "S(B0)",
        "S(C0)",
        "S(A0,B0)",
        "S(B0,C0)",
    ]
    # Rays from the basic von-Neumann entropy polytope for ABC.
    rays = [[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0], [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]
    # Drop S(A,C) (5th col) and S(A,B,C) (last col).
    rays_full = [[row[0], row[1], row[2], row[3], row[5]] for row in rays]

    model, x, label_to_index, labels, var_names, constraints_meta = LP_test(
        names,
        indep_input=indep_input,
        separability_input=separability_input,
    )
    model.acceptedSolutionStatus(AccSolutionStatus.Certificate)
    model.setLogHandler(sys.stdout)
    model.objective("feas_base", ObjectiveSense.Minimize, Expr.constTerm(0.0))
    model.solve()
    base_problem_status = model.getProblemStatus()
    base_solution_status = model.getPrimalSolutionStatus()
    print("base problem_status:", base_problem_status)
    print("base solution_status:", base_solution_status)
    if str(base_problem_status) == "ProblemStatus.PrimalInfeasible":
        cert = extract_infeasibility_certificate(
            model, constraints_meta, label_to_index, row_labels
        )
        if cert is None:
            print("base no certificate available")
        else:
            coeffs, const, extras, full = cert
            print("base certificate coeffs:", {lbl: coeffs.get(lbl, 0.0) for lbl in row_labels})
            print("base certificate const:", const)
            if extras:
                print("base certificate extras:", extras)
            print("base certificate full:", full)

    feasible_rows = []
    infeasible_rows = []
    certificate_summaries: List[str] = []
    for i, row in enumerate(rays_full):
        model, x, label_to_index, labels, var_names, constraints_meta = LP_test(
            names,
            indep_input=indep_input,
            separability_input=separability_input,
            row=row,
            row_labels=row_labels,
        )
        model.acceptedSolutionStatus(AccSolutionStatus.Certificate)
        model.setLogHandler(sys.stdout)
        model.objective("feas", ObjectiveSense.Minimize, Expr.constTerm(0.0))
        model.solve()
        problem_status = model.getProblemStatus()
        solution_status = model.getPrimalSolutionStatus()
        print(f"ray {i} problem_status:", problem_status)
        print(f"ray {i} solution_status:", solution_status)
        if str(problem_status) == "ProblemStatus.PrimalAndDualFeasible":
            feasible_rows.append((i, row))
        else:
            infeasible_rows.append((i, row))
            from certificate_lp import solve_farkas_lp

            A2, b2, lab2 = build_ax_leq_b_from_meta(constraints_meta, signed_labels=True)
            cert_lp2 = solve_farkas_lp(A2, b2, labels=lab2)
            inequality = f"{cert_lp2.expression} >= 0"
            print(f"ray {i} signed-caption inequality: {inequality}")
            certificate_summaries.append(f"ray {i}: {inequality}")

    print("feasible_rows:", feasible_rows)
    print("infeasible_rows:", infeasible_rows)
    print("feasible_count:", len(feasible_rows))
    print("infeasible_count:", len(infeasible_rows))
    print("certificate_count:", len(certificate_summaries))
    if certificate_summaries:
        print("certificates:")
        for entry in certificate_summaries:
            print(entry)
