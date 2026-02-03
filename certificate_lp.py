"""Solve a Farkas-style certificate LP for Ax + c >= 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from mosek.fusion import Domain, Expr, Model, ObjectiveSense


@dataclass(frozen=True)
class CertificateResult:
    y: List[float]
    objective: float
    expression: str
    bound: float | None = None


def solve_certificate_lp(
    A: Sequence[Sequence[float]],
    c: Sequence[float],
    labels: Sequence[Optional[str]] | None = None,
    *,
    bound: float = 1.0,
) -> CertificateResult:
    """Solve max y^T c s.t. A^T y = 0 and -bound <= y <= bound.

    labels: optional list aligned with c; use None for slots without meaning.
    """
    A_np = np.asarray(A, dtype=float)
    c_np = np.asarray(c, dtype=float)
    if A_np.ndim != 2:
        raise ValueError("A must be a 2D matrix")
    if c_np.ndim != 1:
        raise ValueError("c must be a 1D vector")
    if A_np.shape[0] != c_np.shape[0]:
        raise ValueError("A and c must have the same number of rows")
    if labels is not None and len(labels) != len(c):
        raise ValueError("labels must match length of c")

    m, n = A_np.shape
    with Model("certificate_lp") as model:
        y = model.variable("y", m, Domain.inRange(-bound, bound))
        # A^T y = 0
        model.constraint(Expr.mul(A_np.T, y), Domain.equalsTo(0.0))
        # Maximize c^T y
        model.objective("obj", ObjectiveSense.Maximize, Expr.dot(c_np, y))
        model.solve()
        y_val = y.level().tolist()
        obj = float(model.primalObjValue())

    expr = _format_expression(c_np.tolist(), labels)
    return CertificateResult(y=y_val, objective=obj, expression=expr)


def solve_farkas_lp(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    labels: Sequence[Optional[str]] | None = None,
    *,
    bound: float = 1.0,
) -> CertificateResult:
    """Solve min y^T b s.t. A^T y >= 0 and 0 <= y <= bound."""
    A_np = np.asarray(A, dtype=float)
    b_np = np.asarray(b, dtype=float)
    if A_np.ndim != 2:
        raise ValueError("A must be a 2D matrix")
    if b_np.ndim != 1:
        raise ValueError("b must be a 1D vector")
    if A_np.shape[0] != b_np.shape[0]:
        raise ValueError("A and b must have the same number of rows")
    if labels is not None and len(labels) != len(b):
        raise ValueError("labels must match length of b")

    m, _ = A_np.shape
    with Model("farkas_lp") as model:
        y = model.variable("y", m, Domain.inRange(0.0, bound))
        model.constraint(Expr.mul(A_np.T, y), Domain.greaterThan(0.0))
        model.objective("obj", ObjectiveSense.Minimize, Expr.dot(b_np, y))
        model.solve()
        y_val = y.level().tolist()
        obj = float(model.primalObjValue())

    expr = _format_expression_from_y(y_val, labels)
    return CertificateResult(y=y_val, objective=obj, expression=expr)


def check_certificate_bound(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    coeffs: Sequence[float],
    labels: Sequence[Optional[str]] | None = None,
    *,
    drop_labeled_rows: bool = False,
) -> float:
    """Maximize coeffs^T x subject to A x <= b. Returns the optimal value."""
    A_np = np.asarray(A, dtype=float)
    b_np = np.asarray(b, dtype=float)
    c_np = np.asarray(coeffs, dtype=float)
    if A_np.ndim != 2:
        raise ValueError("A must be a 2D matrix")
    if b_np.ndim != 1 or c_np.ndim != 1:
        raise ValueError("b and coeffs must be 1D vectors")
    if A_np.shape[0] != b_np.shape[0]:
        raise ValueError("A and b must have the same number of rows")
    if A_np.shape[1] != c_np.shape[0]:
        raise ValueError("coeffs must match number of columns in A")

    if drop_labeled_rows and labels is not None:
        keep = [i for i, lbl in enumerate(labels) if lbl is None]
        A_np = A_np[keep, :]
        b_np = b_np[keep]

    with Model("certificate_check") as model:
        x = model.variable("x", A_np.shape[1], Domain.unbounded())
        if A_np.shape[0] > 0:
            model.constraint(Expr.mul(A_np, x), Domain.lessThan(b_np))
        model.objective("obj", ObjectiveSense.Maximize, Expr.dot(c_np, x))
        model.solve()
        status = model.getProblemStatus()
        if str(status) != "ProblemStatus.PrimalAndDualFeasible":
            return float("nan")
        return float(model.primalObjValue())


def _format_expression_from_y(
    y: Sequence[float], labels: Sequence[Optional[str]] | None
) -> str:
    if labels is None:
        return " + ".join(f"{v:g}*y[{i}]" for i, v in enumerate(y) if v != 0.0) or "0"
    terms: dict[str, float] = {}
    for coef, label in zip(y, labels):
        if label is None or abs(coef) < 1e-12:
            continue
        sign = 1.0
        if label.startswith("-"):
            sign = -1.0
            label = label[1:]
        terms[label] = terms.get(label, 0.0) + sign * coef
    out = []
    for label, coef in terms.items():
        if abs(coef) < 1e-12:
            continue
        out.append(f"{coef:g}*{label}")
    return " + ".join(out) if out else "0"


def _format_expression(
    c: Sequence[float], labels: Sequence[Optional[str]] | None
) -> str:
    if labels is None:
        return " + ".join(f"{v:g}*c[{i}]" for i, v in enumerate(c) if v != 0.0) or "0"
    terms = []
    for coef, label in zip(c, labels):
        if label is None or abs(coef) < 1e-12:
            continue
        terms.append(f"{coef:g}*{label}")
    return " + ".join(terms) if terms else "0"


def build_symbolic_inequality(
    y: Iterable[float],
    c: Sequence[float],
    labels: Sequence[Optional[str]] | None,
) -> str:
    """Return the inequality y^T c <= 0 as a symbolic expression."""
    y_list = list(y)
    if len(y_list) != len(c):
        raise ValueError("y and c must have same length")
    value_terms = []
    for yi, ci, label in zip(y_list, c, labels or [None] * len(c)):
        if label is None or abs(yi * ci) < 1e-12:
            continue
        value_terms.append(f"{(yi*ci):g}*{label}")
    lhs = " + ".join(value_terms) if value_terms else "0"
    return f"{lhs} <= 0"
