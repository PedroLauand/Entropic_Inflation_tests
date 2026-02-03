"""Build a Shannon-type LP for n variables in entropic coordinates."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import sys
from mosek.fusion import Domain, Expr, Model, ObjectiveSense, Variable


def _all_masks(n: int) -> List[int]:
    return list(range(1, 1 << n))


def _mask_to_label(mask: int, var_names: Sequence[str]) -> str:
    names = [var_names[i] for i in range(len(var_names)) if (mask >> i) & 1]
    return "H(" + ",".join(names) + ")"


def _var_names_from_input(n_or_names: Union[int, Sequence[str]]) -> List[str]:
    if isinstance(n_or_names, int):
        if n_or_names < 1:
            raise ValueError("n must be >= 1")
        return [f"X{i}" for i in range(n_or_names)]
    names = list(n_or_names)
    if not names:
        raise ValueError("names must be a non-empty list")
    return names


def _parse_var_set(item: Union[str, Iterable[str]], var_names: Sequence[str]) -> int:
    name_to_idx = {name: i for i, name in enumerate(var_names)}
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
        raise ValueError("empty variable set in independence specification")

    mask = 0
    for name in tokens:
        if name not in name_to_idx:
            raise ValueError(f"unknown variable name: {name}")
        mask |= 1 << name_to_idx[name]
    return mask


def add_independence_constraints(
    model: Model,
    x: Variable,
    index_of: Dict[int, int],
    var_names: Sequence[str],
    independences: Sequence[Sequence[Union[str, Iterable[str]]]],
) -> None:
    """Add constraints of the form I(A;B|C) = 0.

    Each entry in independences has length 2 or 3:
      - [A, B] means I(A;B) = 0
      - [A, B, C] means I(A;B|C) = 0
    Each of A, B, C can be a string like "X0,X1" or an iterable of names.
    """

    def H(mask: int):
        return 0.0 if mask == 0 else x.index(index_of[mask])

    for item in independences:
        if len(item) not in (2, 3):
            raise ValueError(f"independence entry must have 2 or 3 items: {item}")
        a_mask = _parse_var_set(item[0], var_names)
        b_mask = _parse_var_set(item[1], var_names)
        c_mask = _parse_var_set(item[2], var_names) if len(item) == 3 else 0

        abc_mask = a_mask | b_mask | c_mask
        expr = Expr.sub(
            Expr.add(H(a_mask | c_mask), H(b_mask | c_mask)),
            Expr.add(H(c_mask), H(abc_mask)),
        )
        model.constraint(expr, Domain.equalsTo(0.0))


def add_entropy_value_constraints(
    model: Model,
    x: Variable,
    index_of: Dict[int, int],
    var_names: Sequence[str],
    constraints: Sequence[str] | None = None,
    *,
    row: Sequence[float] | None = None,
    row_labels: Sequence[str] | None = None,
) -> None:
    """Add constraints of the form H(S) = value.

    Provide either:
      - constraints: strings like "H(A,B)=1", or
      - row + row_labels: numeric row with matching labels.
    """

    def parse_constraint(s: str) -> Tuple[int, float]:
        if "=" not in s:
            raise ValueError(f"missing '=' in constraint: {s}")
        left, right = s.split("=", 1)
        left = left.strip()
        right = right.strip()
        if not left.startswith("H(") or not left.endswith(")"):
            raise ValueError(f"invalid entropy label: {left}")
        vars_part = left[2:-1]
        if not vars_part:
            raise ValueError("empty entropy set in constraint")
        names = [t.strip() for t in vars_part.split(",") if t.strip()]
        mask = 0
        name_to_idx = {name: i for i, name in enumerate(var_names)}
        for name in names:
            if name not in name_to_idx:
                raise ValueError(f"unknown variable name: {name}")
            mask |= 1 << name_to_idx[name]
        try:
            value = float(right)
        except ValueError as exc:
            raise ValueError(f"invalid numeric value: {right}") from exc
        return mask, value

    if constraints is None:
        if row is None or row_labels is None:
            raise ValueError("provide constraints or (row and row_labels)")
        if len(row) != len(row_labels):
            raise ValueError("row and row_labels must have same length")
        constraints = [f"{lbl}={val}" for lbl, val in zip(row_labels, row)]

    for s in constraints:
        mask, value = parse_constraint(s)
        model.constraint(x.index(index_of[mask]), Domain.equalsTo(value))


def add_symmetry_constraints(
    model: Model,
    x: Variable,
    index_of: Dict[int, int],
    var_names: Sequence[str],
    symmetry_input: Sequence[Sequence[str]],
) -> None:
    """Add constraints of the form H(S1) = H(S2) for each pair in symmetry_input."""
    for pair in symmetry_input:
        if len(pair) != 2:
            raise ValueError(f"symmetry entry must have 2 items: {pair}")
        a_mask = _parse_var_set(pair[0], var_names)
        b_mask = _parse_var_set(pair[1], var_names)
        model.constraint(
            Expr.sub(x.index(index_of[a_mask]), x.index(index_of[b_mask])),
            Domain.equalsTo(0.0),
        )


def build_shannon_model(
    n_or_names: Union[int, Sequence[str]],
) -> Tuple[Model, Variable, Dict[int, int], List[str]]:
    """Return a model with Shannon-type constraints.

    Variables are entropies H(S) for all non-empty subsets S of the given variables.
    If n_or_names is an int, variables are named X0, X1, ...
    Enforced conditions: monotonicity H(Y) >= H(X) for X ⊆ Y, and
    strong subadditivity I(A;B|C) >= 0 for all disjoint A,B,C.
    H(∅) is treated as 0 implicitly.
    """
    var_names = _var_names_from_input(n_or_names)
    n = len(var_names)
    masks = _all_masks(n)
    index_of: Dict[int, int] = {m: i for i, m in enumerate(masks)}

    model = Model(f"shannon_n{n}")
    x = model.variable("h", len(masks), Domain.unbounded())

    def H(mask: int):
        """Entropy expression with H(∅)=0."""
        return 0.0 if mask == 0 else x.index(index_of[mask])

    # Monotonicity: H(Y) - H(X) >= 0 for all X ⊆ Y.
    # This includes H(S) >= 0 via X=∅.
    for y_mask in masks:
        sub = y_mask
        while True:
            x_mask = sub
            if x_mask != y_mask:
                model.constraint(
                    Expr.sub(H(y_mask), H(x_mask)),
                    Domain.greaterThan(0.0),
                )
            if sub == 0:
                break
            sub = (sub - 1) & y_mask

    # Conditional mutual information constraints:
    # I(A;B|C) = H(A∪C) + H(B∪C) - H(C) - H(A∪B∪C) >= 0
    for a_mask in masks:
        for b_mask in masks:
            if a_mask & b_mask:
                continue
            # require A and B non-empty, disjoint
            if a_mask == 0 or b_mask == 0:
                continue
            ab_mask = a_mask | b_mask
            remaining = [i for i in range(n) if not (ab_mask >> i) & 1]
            for r in range(len(remaining) + 1):
                for combo in combinations(remaining, r):
                    c_mask = 0
                    for i in combo:
                        c_mask |= 1 << i
                    ac_mask = a_mask | c_mask
                    bc_mask = b_mask | c_mask
                    abc_mask = ab_mask | c_mask
                    expr = Expr.sub(
                        Expr.add(H(ac_mask), H(bc_mask)),
                        Expr.add(H(c_mask), H(abc_mask)),
                    )
                    model.constraint(expr, Domain.greaterThan(0.0))

    return model, x, index_of, var_names


def caption(n_or_names: Union[int, Sequence[str]]) -> List[str]:
    var_names = _var_names_from_input(n_or_names)
    return [_mask_to_label(m, var_names) for m in _all_masks(len(var_names))]


if __name__ == "__main__":
    # Example: named variables with independence and symmetry constraints.
    names = ["A0", "A1", "B0", "B1", "C0", "C1"]
    indep_input = [
        ["A1", "B0,B1,C1"],  # I(A1:B0,B1,C1)=0
        ["B1", "C0,C1,A1"],  # I(B1:C0,C1,A1)=0
        ["C1", "A0,A1,B1"],  # I(C1:A0,A1,B1)=0
    ]
    symmetry_input = [
        ["A0", "A1"],        # H(A0)=H(A1)
        ["B0", "B1"],        # H(B0)=H(B1)
        ["C0", "C1"],        # H(C0)=H(C1)
        ["A0,B0", "A0,B1"],  # H(A0,B0)=H(A0,B1)
        ["B0,C0", "B0,C1"],  # H(B0,C0)=H(B0,C1)
        ["A0,C0", "A1,C0"],  # H(A0,C0)=H(A1,C0)
    ]
    row_labels = [
        "H(A0)",
        "H(B0)",
        "H(C0)",
        "H(A0,B0)",
        "H(A0,C0)",
        "H(B0,C0)",
        "H(A0,B0,C0)",
    ]
    # Rays matrix for testing (each row follows row_labels order above).
    rays = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [1.5, 1.5, 1.0, 2.5, 2.0, 2.0, 3.0],
        [1.5, 1.0, 1.5, 2.0, 2.5, 2.0, 3.0],
        [1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 3.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    ]

    feasible_rows = []
    infeasible_rows = []
    for i, row in enumerate(rays):
        model, x, index_of, names = build_shannon_model(names)
        add_independence_constraints(model, x, index_of, names, indep_input)
        add_symmetry_constraints(model, x, index_of, names, symmetry_input)
        add_entropy_value_constraints(
            model,
            x,
            index_of,
            names,
            row=row,
            row_labels=row_labels,
        )
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

    print("feasible_rows:", feasible_rows, len(feasible_rows))
    print("infeasible_rows:", infeasible_rows, len(infeasible_rows))
