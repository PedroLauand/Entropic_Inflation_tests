"""Utilities for building Shannon-entropy LP tests."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Union
import math

from mosek.fusion import Domain, Expr, Model, Variable


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
    return "H(" + ",".join(names) + ")"


def _entropic_caption_from_names(names: Sequence[str]) -> List[str]:
    var_names = _var_names_from_input(names)
    n = len(var_names)
    caption: List[str] = []
    for mask in range(1, 1 << n):
        subset = [var_names[i] for i in range(n) if (mask >> i) & 1]
        caption.append(_label_for_vars(subset, var_names))
    return caption


def _build_context_constraints(
    ctx_vars: Sequence[str],
    global_var_names: Sequence[str],
    label_to_index: Dict[str, int],
) -> List[List[float]]:
    n = len(ctx_vars)
    if n == 0:
        return []
    masks = list(range(1, 1 << n))
    n_cols = len(label_to_index)

    def add_term(coeffs: List[float], mask: int, value: float) -> None:
        if mask == 0:
            return
        subset = [ctx_vars[i] for i in range(n) if (mask >> i) & 1]
        label = _label_for_vars(subset, global_var_names)
        idx = label_to_index[label]
        coeffs[idx] += value

    rows: List[List[float]] = []

    # Monotonicity: H(Y) - H(X) >= 0 for all X ⊆ Y (including X=∅).
    for y_mask in masks:
        sub = y_mask
        while True:
            x_mask = sub
            if x_mask != y_mask:
                coeffs = [0.0] * n_cols
                add_term(coeffs, y_mask, -1.0)
                add_term(coeffs, x_mask, 1.0)
                rows.append(coeffs)
            if sub == 0:
                break
            sub = (sub - 1) & y_mask

    # Strong subadditivity: I(A;B|C) >= 0 for all disjoint A,B,C.
    for a_mask in masks:
        for b_mask in masks:
            if a_mask & b_mask:
                continue
            ab_mask = a_mask | b_mask
            remaining = [i for i in range(n) if not (ab_mask >> i) & 1]
            # C can be empty: build subsets of remaining via bitmasks.
            for c_bits in range(1 << len(remaining)):
                c_mask = 0
                for j, idx in enumerate(remaining):
                    if (c_bits >> j) & 1:
                        c_mask |= 1 << idx
                coeffs = [0.0] * n_cols
                add_term(coeffs, a_mask | c_mask, -1.0)
                add_term(coeffs, b_mask | c_mask, -1.0)
                add_term(coeffs, c_mask, 1.0)
                add_term(coeffs, a_mask | b_mask | c_mask, 1.0)
                rows.append(coeffs)

    return rows


def build_shannon_inequality_matrix(
    n_or_names: Union[int, Sequence[str], Sequence[Sequence[str]]],
) -> Tuple[
    List[List[float]],
    List[float],
    List[str],
    List[str],
    Dict[str, int],
    List[str],
    List[Dict[str, object]],
]:
    if _is_context_list(n_or_names):  # type: ignore[arg-type]
        contexts = list(n_or_names)  # type: ignore[assignment]
        var_names = _flatten_contexts(contexts)
    else:
        var_names = _var_names_from_input(n_or_names)  # type: ignore[arg-type]
        contexts = [var_names]

    label_to_index: Dict[str, int] = {}
    for ctx in contexts:
        ctx_vars = list(ctx)
        m = len(ctx_vars)
        for mask in range(1, 1 << m):
            subset = [ctx_vars[i] for i in range(m) if (mask >> i) & 1]
            label = _label_for_vars(subset, var_names)
            if label not in label_to_index:
                label_to_index[label] = len(label_to_index)

    x_caption = list(label_to_index.keys())
    M: List[List[float]] = []
    b: List[float] = []
    b_caption: List[str] = []
    constraints_meta: List[Dict[str, object]] = []

    for ctx in contexts:
        ctx_rows = _build_context_constraints(ctx, var_names, label_to_index)
        for coeffs in ctx_rows:
            M.append(coeffs)
            b.append(0.0)
            b_caption.append("0")
            constraints_meta.append(
                {
                    "constraint": None,
                    "coeffs": [-v for v in coeffs],
                    "const": 0.0,
                    "sense": "ge",
                }
            )

    return M, b, b_caption, x_caption, label_to_index, var_names, constraints_meta


def LP_test(
    names: Union[int, Sequence[str], Sequence[Sequence[str]]],
    *,
    indep_input: Sequence[Sequence[Union[str, Iterable[str]]]] | None = None,
    symmetry_input: Sequence[Sequence[str]] | None = None,
    candidate: Sequence[float | int | str | None] | None = None,
    candidate_caption: Sequence[str] | None = None,
    candidate_names: Sequence[str] | None = None,
    return_meta: bool = True,
    return_matrix: bool = False,
) -> Tuple[
    Model,
    Variable,
    Dict[str, int],
    List[str],
    List[str],
    List[Dict[str, object]] | None,
] | Tuple[
    Model,
    Variable,
    Dict[str, int],
    List[str],
    List[str],
    List[Dict[str, object]] | None,
    Dict[str, object],
]:
    """Build a Shannon LP with optional constraints.

    - names: int, list of variable names, or list of contexts (list of list).
    - indep_input/symmetry_input: entries are lists like ["A","B"] or
      ["A","C","B"], and elements may be comma-strings like "C,B".
    - candidate: partial entropic vector aligned with Shannon caption; use None or "" for free slots.
    - candidate_names: if provided, generates the candidate caption from these names.
    - return_matrix: when True, return a dict with M, b, b_caption, and x_caption.
    """
    if candidate_caption is not None and candidate_names is not None:
        raise ValueError("provide candidate_caption or candidate_names, not both")

    (
        M,
        b,
        b_caption,
        x_caption,
        label_to_index,
        var_names,
        constraints_meta,
    ) = build_shannon_inequality_matrix(names)

    n = len(x_caption)

    def append_leq(coeffs: List[float], rhs: float, caption: str) -> None:
        M.append(coeffs)
        b.append(rhs)
        b_caption.append(caption)
        constraints_meta.append(
            {
                "constraint": None,
                "coeffs": [-v for v in coeffs],
                "const": rhs,
                "label": None if caption == "0" else caption,
                "sense": "ge",
            }
        )

    def append_eq(coeffs: List[float]) -> None:
        append_leq(coeffs, 0.0, "0")
        append_leq([-v for v in coeffs], 0.0, "0")

    if indep_input:
        for item in indep_input:
            if len(item) == 2:
                x_vars = _parse_var_set(item[0], var_names)
                y_vars = _parse_var_set(item[1], var_names)
                xy_vars = list(dict.fromkeys(x_vars + y_vars))
                x_label = _label_for_vars(x_vars, var_names)
                y_label = _label_for_vars(y_vars, var_names)
                xy_label = _label_for_vars(xy_vars, var_names)
                coeffs = [0.0] * n
                coeffs[label_to_index[x_label]] += 1.0
                coeffs[label_to_index[y_label]] += 1.0
                coeffs[label_to_index[xy_label]] -= 1.0
                append_eq(coeffs)
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
                coeffs = [0.0] * n
                coeffs[label_to_index[xz_label]] += 1.0
                coeffs[label_to_index[yz_label]] += 1.0
                coeffs[label_to_index[z_label]] -= 1.0
                coeffs[label_to_index[xyz_label]] -= 1.0
                append_eq(coeffs)
                continue
            raise ValueError(f"independence entry must have 2 or 3 items: {item}")

    if symmetry_input:
        for pair in symmetry_input:
            if len(pair) != 2:
                raise ValueError(f"symmetry entry must have 2 items: {pair}")
            a_vars = _parse_var_set(pair[0], var_names)
            b_vars = _parse_var_set(pair[1], var_names)
            a_label = _label_for_vars(a_vars, var_names)
            b_label = _label_for_vars(b_vars, var_names)
            coeffs = [0.0] * n
            coeffs[label_to_index[a_label]] += 1.0
            coeffs[label_to_index[b_label]] -= 1.0
            append_eq(coeffs)

    if candidate is not None:
        if candidate_caption is None:
            if candidate_names is not None:
                candidate_caption = _entropic_caption_from_names(candidate_names)
            else:
                candidate_caption = x_caption

        if len(candidate) != len(candidate_caption):
            raise ValueError("candidate and candidate_caption must have same length")

        items: List[Tuple[int, float, str]] = []
        for label, val in zip(candidate_caption, candidate):
            if val is None:
                continue
            if isinstance(val, float) and math.isnan(val):
                continue
            if isinstance(val, str) and not val.strip():
                continue
            if not label.startswith("H(") or not label.endswith(")"):
                raise ValueError(f"invalid entropy label in candidate caption: {label}")
            value = float(val)
            vars_part = label[2:-1]
            names = [t.strip() for t in vars_part.split(",") if t.strip()]
            vars_ = _parse_var_set(names, var_names)
            norm_label = _label_for_vars(vars_, var_names)
            if norm_label not in label_to_index:
                raise ValueError(f"unknown entropy label in candidate caption: {label}")
            idx = label_to_index[norm_label]
            items.append((idx, value, label))

        for idx, value, label in items:
            row_pos = [0.0] * n
            row_pos[idx] = 1.0
            append_leq(row_pos, value, label)
        for idx, value, label in items:
            row_neg = [0.0] * n
            row_neg[idx] = -1.0
            append_leq(row_neg, -value, f"-{label}")

    labels = ["1"] + x_caption
    model = Model(f"shannon_lp_{len(var_names)}")
    x = model.variable("h", len(label_to_index), Domain.unbounded())

    if M:
        constr = model.constraint(Expr.mul(M, x), Domain.lessThan(b))
        for meta in constraints_meta:
            meta["constraint"] = constr

    meta = constraints_meta if return_meta else None

    if return_matrix:
        matrix = {
            "M": M,
            "b": b,
            "b_caption": b_caption,
            "x_caption": x_caption,
        }
        return model, x, label_to_index, labels, var_names, meta, matrix

    return model, x, label_to_index, labels, var_names, meta
