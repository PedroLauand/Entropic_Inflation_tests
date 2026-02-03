"""LP feasibility tests for von-Neumann entropies using MOSEK."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Union

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

from constraints.vn_constraints import build_vn_inequalities


def _var_names_from_input(n_or_names: Union[int, Sequence[str]]) -> List[str]:
    if isinstance(n_or_names, int):
        if n_or_names < 1:
            raise ValueError("n must be >= 1")
        return [f"X{i}" for i in range(n_or_names)]
    names = list(n_or_names)
    if not names:
        raise ValueError("names must be a non-empty list")
    return names


def _is_context_list(names: Sequence[object]) -> bool:
    return bool(names) and all(isinstance(x, (list, tuple)) for x in names)


def _flatten_contexts(contexts: Sequence[Sequence[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for ctx in contexts:
        for name in ctx:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _parse_var_set(item: Union[str, Iterable[str]], var_names: Sequence[str]) -> List[str]:
    name_set = set(var_names)
    tokens: List[str] = []
    if isinstance(item, str):
        tokens = [t.strip() for t in item.split(",") if t.strip()]
    else:
        for elem in item:
            if isinstance(elem, str) and "," in elem:
                tokens.extend(t.strip() for t in elem.split(",") if t.strip())
            else:
                tokens.append(str(elem).strip())
        tokens = [t for t in tokens if t]
    if not tokens:
        raise ValueError("empty variable set")
    for name in tokens:
        if name not in name_set:
            raise ValueError(f"unknown variable name: {name}")
    return tokens


def _label_for_vars(vars_: Sequence[str], var_order: Sequence[str]) -> str:
    order_index = {name: i for i, name in enumerate(var_order)}
    names = sorted(vars_, key=lambda n: order_index[n])
    return "S(" + ",".join(names) + ")"


def build_vn_model(
    n_or_names: Union[int, Sequence[str], Sequence[Sequence[str]]],
) -> Tuple[Model, Variable, Dict[str, int], List[str], List[str], List[Dict[str, object]]]:
    if _is_context_list(n_or_names):  # type: ignore[arg-type]
        contexts = list(n_or_names)  # type: ignore[assignment]
        var_names = _flatten_contexts(contexts)
    else:
        var_names = _var_names_from_input(n_or_names)  # type: ignore[arg-type]
        contexts = [var_names]

    # Build global label index across all contexts.
    label_to_index: Dict[str, int] = {}
    for ctx in contexts:
        # all non-empty subsets of ctx
        ctx_vars = list(ctx)
        m = len(ctx_vars)
        for mask in range(1, 1 << m):
            subset = [ctx_vars[i] for i in range(m) if (mask >> i) & 1]
            label = _label_for_vars(subset, var_names)
            if label not in label_to_index:
                label_to_index[label] = len(label_to_index)

    labels = ["1"] + list(label_to_index.keys())
    model = Model(f"vn_lp_{len(var_names)}")
    x = model.variable("s", len(label_to_index), Domain.unbounded())

    constraints_meta: List[Dict[str, object]] = []

    # Add VN inequalities for each context, mapped into the global variable set.
    for ctx in contexts:
        caption, array = build_vn_inequalities(list(ctx))
        for row in array:
            coeffs = [0.0] * len(label_to_index)
            expr = Expr.constTerm(row[0])
            for i, coef in enumerate(row[1:]):
                if coef == 0:
                    continue
                if caption[i + 1] == "S()":
                    continue
                label = _label_for_vars(
                    caption[i + 1][2:-1].split(",") if caption[i + 1] != "1" else [],
                    var_names,
                )
                idx = label_to_index[label]
                coeffs[idx] += coef
                expr = Expr.add(expr, Expr.mul(coef, x.index(idx)))
            constr = model.constraint(expr, Domain.greaterThan(0.0))
            constraints_meta.append(
                {
                    "constraint": constr,
                    "coeffs": coeffs,
                    "const": float(row[0]),
                    "sense": "ge",
                }
            )

    return model, x, label_to_index, labels, var_names, constraints_meta


def add_symmetry_constraints(
    model: Model,
    x: Variable,
    label_to_index: Dict[str, int],
    var_names: Sequence[str],
    symmetry_input: Sequence[Sequence[str]],
) -> None:
    """Add constraints H(S1) = H(S2) for each pair."""
    for pair in symmetry_input:
        if len(pair) != 2:
            raise ValueError(f"symmetry entry must have 2 items: {pair}")
        a_vars = _parse_var_set(pair[0], var_names)
        b_vars = _parse_var_set(pair[1], var_names)
        a_label = _label_for_vars(a_vars, var_names)
        b_label = _label_for_vars(b_vars, var_names)
        model.constraint(
            Expr.sub(x.index(label_to_index[a_label]), x.index(label_to_index[b_label])),
            Domain.equalsTo(0.0),
        )


def add_separability_constraints(
    model: Model,
    x: Variable,
    label_to_index: Dict[str, int],
    var_names: Sequence[str],
    separability_input: Sequence[Sequence[Union[str, Iterable[str]]]],
    constraints_meta: List[Dict[str, object]] | None = None,
) -> None:
    """Impose separability: S(X|Y)>=0 and S(Y|X)>=0 for each (X,Y)."""
    for item in separability_input:
        if len(item) != 2:
            raise ValueError(f"separability entry must have 2 items: {item}")
        x_vars = _parse_var_set(item[0], var_names)
        y_vars = _parse_var_set(item[1], var_names)
        xy_vars = list(dict.fromkeys(x_vars + y_vars))
        x_label = _label_for_vars(x_vars, var_names)
        y_label = _label_for_vars(y_vars, var_names)
        xy_label = _label_for_vars(xy_vars, var_names)
        # S(X|Y) = S(XY) - S(Y) >= 0
        constr = model.constraint(
            Expr.sub(x.index(label_to_index[xy_label]), x.index(label_to_index[y_label])),
            Domain.greaterThan(0.0),
        )
        if constraints_meta is not None:
            coeffs = [0.0] * len(label_to_index)
            coeffs[label_to_index[xy_label]] += 1.0
            coeffs[label_to_index[y_label]] -= 1.0
            constraints_meta.append(
                {"constraint": constr, "coeffs": coeffs, "const": 0.0, "sense": "ge"}
            )
        # S(Y|X) = S(XY) - S(X) >= 0
        constr = model.constraint(
            Expr.sub(x.index(label_to_index[xy_label]), x.index(label_to_index[x_label])),
            Domain.greaterThan(0.0),
        )
        if constraints_meta is not None:
            coeffs = [0.0] * len(label_to_index)
            coeffs[label_to_index[xy_label]] += 1.0
            coeffs[label_to_index[x_label]] -= 1.0
            constraints_meta.append(
                {"constraint": constr, "coeffs": coeffs, "const": 0.0, "sense": "ge"}
            )


def add_independence_constraints(
    model: Model,
    x: Variable,
    label_to_index: Dict[str, int],
    var_names: Sequence[str],
    indep_input: Sequence[Sequence[Union[str, Iterable[str]]]],
    constraints_meta: List[Dict[str, object]] | None = None,
) -> None:
    """Impose independence constraints.

    Each entry is either:
      - (X, Y) for I(X;Y)=0, or
      - (X, Y, Z) for I(X;Y|Z)=0.
    """
    for item in indep_input:
        if len(item) == 2:
            x_vars = _parse_var_set(item[0], var_names)
            y_vars = _parse_var_set(item[1], var_names)
            xy_vars = list(dict.fromkeys(x_vars + y_vars))
            x_label = _label_for_vars(x_vars, var_names)
            y_label = _label_for_vars(y_vars, var_names)
            xy_label = _label_for_vars(xy_vars, var_names)
            expr = Expr.sub(
                Expr.add(x.index(label_to_index[x_label]), x.index(label_to_index[y_label])),
                x.index(label_to_index[xy_label]),
            )
            constr = model.constraint(expr, Domain.equalsTo(0.0))
            if constraints_meta is not None:
                coeffs = [0.0] * len(label_to_index)
                coeffs[label_to_index[x_label]] += 1.0
                coeffs[label_to_index[y_label]] += 1.0
                coeffs[label_to_index[xy_label]] -= 1.0
                constraints_meta.append(
                    {"constraint": constr, "coeffs": coeffs, "const": 0.0, "sense": "eq"}
                )
            continue
        if len(item) == 3:
            x_vars = _parse_var_set(item[0], var_names)
            y_vars = _parse_var_set(item[1], var_names)
            z_vars = _parse_var_set(item[2], var_names)
            xz_vars = list(dict.fromkeys(x_vars + z_vars))
            yz_vars = list(dict.fromkeys(y_vars + z_vars))
            xyz_vars = list(dict.fromkeys(x_vars + y_vars + z_vars))
            z_label = _label_for_vars(z_vars, var_names)
            xz_label = _label_for_vars(xz_vars, var_names)
            yz_label = _label_for_vars(yz_vars, var_names)
            xyz_label = _label_for_vars(xyz_vars, var_names)
            expr = Expr.sub(
                Expr.add(x.index(label_to_index[xz_label]), x.index(label_to_index[yz_label])),
                Expr.add(x.index(label_to_index[z_label]), x.index(label_to_index[xyz_label])),
            )
            constr = model.constraint(expr, Domain.equalsTo(0.0))
            if constraints_meta is not None:
                coeffs = [0.0] * len(label_to_index)
                coeffs[label_to_index[xz_label]] += 1.0
                coeffs[label_to_index[yz_label]] += 1.0
                coeffs[label_to_index[z_label]] -= 1.0
                coeffs[label_to_index[xyz_label]] -= 1.0
                constraints_meta.append(
                    {"constraint": constr, "coeffs": coeffs, "const": 0.0, "sense": "eq"}
                )
            continue
        raise ValueError(f"independence entry must have 2 or 3 items: {item}")

def add_entropy_value_constraints(
    model: Model,
    x: Variable,
    label_to_index: Dict[str, int],
    var_names: Sequence[str],
    constraints: Sequence[str] | None = None,
    *,
    row: Sequence[float] | None = None,
    row_labels: Sequence[str] | None = None,
    constraints_meta: List[Dict[str, object]] | None = None,
) -> None:
    """Add constraints of the form S(S) = value.

    Provide either:
      - constraints: strings like "S(A,B)=1", or
      - row + row_labels: numeric row with matching labels.
    """

    def parse_constraint(s: str) -> Tuple[int, float]:
        if "=" not in s:
            raise ValueError(f"missing '=' in constraint: {s}")
        left, right = s.split("=", 1)
        left = left.strip()
        right = right.strip()
        if not left.startswith("S(") or not left.endswith(")"):
            raise ValueError(f"invalid entropy label: {left}")
        vars_part = left[2:-1]
        if not vars_part:
            raise ValueError("empty entropy set in constraint")
        names = [t.strip() for t in vars_part.split(",") if t.strip()]
        vars_ = _parse_var_set(names, var_names)
        label = _label_for_vars(vars_, var_names)
        try:
            value = float(right)
        except ValueError as exc:
            raise ValueError(f"invalid numeric value: {right}") from exc
        return label, value

    if constraints is None:
        if row is None or row_labels is None:
            raise ValueError("provide constraints or (row and row_labels)")
        if len(row) != len(row_labels):
            raise ValueError("row and row_labels must have same length")
        constraints = [f"{lbl}={val}" for lbl, val in zip(row_labels, row)]

    for s in constraints:
        label, value = parse_constraint(s)
        constr = model.constraint(x.index(label_to_index[label]), Domain.equalsTo(value))
        if constraints_meta is not None:
            coeffs = [0.0] * len(label_to_index)
            coeffs[label_to_index[label]] += 1.0
            constraints_meta.append(
                {
                    "constraint": constr,
                    "coeffs": coeffs,
                    "const": -float(value),
                    "label": label,
                    "sense": "eq",
                }
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
        ["A0", "B0", "C0"],
    ]
    indep_input = [
        ["A0", "C0"],
    ]
    separability_input = []

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

    model, x, label_to_index, labels, var_names, constraints_meta = build_vn_model(names)
    add_independence_constraints(
        model, x, label_to_index, var_names, indep_input, constraints_meta
    )
    add_separability_constraints(
        model, x, label_to_index, var_names, separability_input, constraints_meta
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
        model, x, label_to_index, labels, var_names, constraints_meta = build_vn_model(names)
        add_independence_constraints(
            model, x, label_to_index, var_names, indep_input, constraints_meta
        )
        add_separability_constraints(
            model, x, label_to_index, var_names, separability_input, constraints_meta
        )
        add_entropy_value_constraints(
            model,
            x,
            label_to_index,
            var_names,
            row=row,
            row_labels=row_labels,
            constraints_meta=constraints_meta,
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
