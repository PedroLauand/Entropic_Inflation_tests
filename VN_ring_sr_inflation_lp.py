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


def _normalize_label(label: str) -> str:
    # Strip copy indices 0/1 from variable names inside S(...)
    if not label.startswith("S(") or not label.endswith(")"):
        return label
    inner = label[2:-1]
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    norm_parts = []
    for p in parts:
        if p and p[-1] in ("0", "1"):
            norm_parts.append(p[:-1])
        else:
            norm_parts.append(p)
    return "S(" + ",".join(norm_parts) + ")"


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
    separability_input = []
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
    row_labels_A1 = [
        "S(A1)",
        "S(B1)",
        "S(C1)",
        "S(A1,B1)",
        "S(A1,C1)",
        "S(B1,C1)",
    ]
    # Rays from the basic von-Neumann entropy polytope for ABC.
    rays = [[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0], [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]
    # Drop S(A,B,C) (last col) for each copy.
    rays_a0 = [row[:6] for row in rays]

    model, x, label_to_index, labels, var_names, constraints_meta = LP_test(
        names,
        indep_input=indep_input,
        separability_input=separability_input,
        symmetry_input=symmetry_input,
    )
    model.acceptedSolutionStatus(AccSolutionStatus.Certificate)
    model.setLogHandler(sys.stdout)
    model.objective("feas_base", ObjectiveSense.Minimize, Expr.constTerm(0.0))
    model.solve()
    base_problem_status = model.getProblemStatus()
    base_solution_status = model.getPrimalSolutionStatus()
    print("base problem_status:", base_problem_status)
    print("base solution_status:", base_solution_status)

    feasible_rows = []
    infeasible_rows = []
    certificate_summaries: List[str] = []
    distinct_certificates: Dict[str, int] = {}
    for i, row in enumerate(rays):
        value_constraints = [
            f"{lbl}={val}" for lbl, val in zip(row_labels_A0, rays_a0[i])
        ] + [
            f"{lbl}={val}" for lbl, val in zip(row_labels_A1, rays_a0[i])
        ]
        model, x, label_to_index, labels, var_names, constraints_meta = LP_test(
            names,
            indep_input=indep_input,
            separability_input=separability_input,
            symmetry_input=symmetry_input,
            value_constraints=value_constraints,
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
            # Normalize labels by dropping 0/1 indices and combine coefficients.
            coeffs: Dict[str, float] = {}
            for coef, lbl in zip(cert_lp2.y, lab2):
                if lbl is None or abs(coef) < 1e-12:
                    continue
                sign = 1.0
                if lbl.startswith("-"):
                    sign = -1.0
                    lbl = lbl[1:]
                norm = _normalize_label(lbl)
                coeffs[norm] = coeffs.get(norm, 0.0) + sign * coef
            # Normalize coefficients for deduping.
            pruned = {k: v for k, v in coeffs.items() if abs(v) >= 1e-10}
            if pruned:
                max_abs = max(abs(v) for v in pruned.values())
                norm = {k: v / max_abs for k, v in pruned.items()}
            else:
                norm = {}
            # Round to stabilize floating noise
            norm_rounded = {k: round(v, 6) for k, v in norm.items() if abs(v) >= 1e-8}
            # Canonical sign: make first nonzero positive
            sign = 1.0
            for k in sorted(norm_rounded.keys()):
                if norm_rounded[k] != 0:
                    sign = 1.0 if norm_rounded[k] > 0 else -1.0
                    break
            if sign < 0:
                norm_rounded = {k: -v for k, v in norm_rounded.items()}
            # Build display from normalized coefficients
            terms = []
            for lbl in sorted(norm_rounded.keys()):
                coef = norm_rounded[lbl]
                if abs(coef) < 1e-8:
                    continue
                terms.append(f"{coef:g}*{lbl}")
            lhs = " + ".join(terms) if terms else "0"
            inequality = f"{lhs} >= 0"
            print(f"ray {i} signed-caption inequality: {inequality}")
            certificate_summaries.append(f"ray {i}: {inequality}")
            distinct_certificates[inequality] = distinct_certificates.get(inequality, 0) + 1

    print("feasible_rows:", feasible_rows)
    print("infeasible_rows:", infeasible_rows)
    print("feasible_count:", len(feasible_rows))
    print("infeasible_count:", len(infeasible_rows))
    print("certificate_count:", len(certificate_summaries))
    if certificate_summaries:
        print("certificates:")
        for entry in certificate_summaries:
            print(entry)
    print("distinct_certificate_count:", len(distinct_certificates))
    if distinct_certificates:
        print("distinct_certificates:")
        for ineq, count in distinct_certificates.items():
            print(f"{ineq}  (count={count})")
