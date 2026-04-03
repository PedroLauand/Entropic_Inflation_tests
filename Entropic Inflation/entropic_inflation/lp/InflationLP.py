"""Entropic analogue of the reference inflation.lp.InflationLP."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from gc import collect
from typing import Dict, Iterable, List, Mapping, Sequence, Union

import numpy as np
from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense, Variable
from scipy.sparse import coo_array, issparse

from ..InflationProblem import InflationProblem
from .lp_utils import build_elemental_shannon_cone

SetSpec = Union[str, Iterable[str]]
NamedVector = Mapping[str, float | int]


@dataclass(slots=True)
class EntropicSolveResult:
    problem: "InflationLP"
    is_feasible: bool
    problem_status: str
    solution_status: str
    objective_value: float | None = None
    farkas_certificate: "FarkasCertificate | None" = None


@dataclass(slots=True)
class FarkasCertificate:
    objective_value: float
    expression: str
    multipliers: List[float]
    is_separating: bool


class InflationLP:
    """Elemental entropic LP generated from an inflated causal scenario."""

    def __init__(
        self,
        inflationproblem: InflationProblem,
        *,
        include_latents: bool = False,
        candidate: NamedVector | None = None,
        objective: NamedVector | None = None,
        objective_sense: str = "min",
        objective_constant: float = 0.0,
        objective_name: str = "objective",
    ) -> None:
        self.InflationProblem = inflationproblem
        self.include_latents = include_latents
        self.variable_names = inflationproblem.entropic_node_labels(
            include_latents=include_latents
        )
        self.public_variable_names = inflationproblem.public_node_labels(
            include_latents=include_latents
        )
        self.independencies = inflationproblem.inflated_independencies(
            include_latents=include_latents
        )
        self.factorization_equalities: List[tuple[tuple[str, ...], ...]] = []
        self.symmetry_generators = inflationproblem.inflation_permutation_generators(
            include_latents=include_latents
        )
        self.known_values: Dict[str, float] = {}
        self.lower_bounds: Dict[str, float] = {}
        self.upper_bounds: Dict[str, float] = {}
        self.extra_equalities: List[Dict[str, float]] = []
        self.extra_inequalities: List[Dict[str, float]] = []
        self.objective: Dict[str, float] = {}
        self.maximize = objective_sense.strip().lower() in {
            "max",
            "maximum",
            "maximize",
            "maximise",
        }
        self.objective_constant = float(objective_constant)
        self.objective_name = objective_name
        self.model: Model | None = None
        self.entropy_vector: Variable | None = None
        self.label_to_index: Dict[str, int] = {}
        self.entropy_labels: List[str] = []
        self.matrix: Dict[str, object] = {}
        self.status = "not_solved"
        self.success = False
        self.objective_value: float | str | None = None
        self.solution_object: Dict[str, object] = {}
        self._model_dirty = True

        self.set_values(candidate)
        self.set_objective(objective, objective_sense)
        self._rebuild_model()

    def _rebuild_model(self) -> None:
        if not self._model_dirty and self.model is not None and self.entropy_vector is not None:
            return
        self.factorization_equalities = []
        (
            self.model,
            self.entropy_vector,
            self.label_to_index,
            self.entropy_labels,
            self.matrix,
        ) = _build_elemental_problem(
            self.variable_names,
            independencies=self.independencies,
            factorization_equalities=None if self.include_latents else self.factorization_equalities,
            symmetry_generators=self.symmetry_generators,
            candidate=self.known_values,
            inflationproblem=self.InflationProblem,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            extra_equalities=self.extra_equalities,
            extra_inequalities=self.extra_inequalities,
        )
        self.model.objective(
            self.objective_name,
            ObjectiveSense.Maximize if self.maximize else ObjectiveSense.Minimize,
            _build_objective_expression(
                self.entropy_vector,
                self.label_to_index,
                self.variable_names,
                self.objective,
                constant=self.objective_constant,
            ),
        )
        self._model_dirty = False

    def _mark_model_dirty(self) -> None:
        self._model_dirty = True

    def set_values(self, values: NamedVector | None) -> None:
        self.known_values = _sanitise_named_vector(
            values,
            self.InflationProblem,
            self.include_latents,
            self.variable_names,
        )
        self._mark_model_dirty()

    def update_values(self, values: NamedVector | None) -> None:
        if not values:
            return
        self.known_values.update(
            _sanitise_named_vector(
                values,
                self.InflationProblem,
                self.include_latents,
                self.variable_names,
            )
        )
        self._mark_model_dirty()

    def set_distribution(self, values: NamedVector | None, **kwargs) -> None:
        """Alias for entropic value assignment.

        Unlike the reference probability LP, entropic_inflation expects a
        sparse mapping of entropy coordinates to values.
        """
        self.set_values(values)

    def set_bounds(self, bounds: NamedVector | None, bound_type: str = "up") -> None:
        assert bound_type in {"up", "lo"}
        target = self.upper_bounds if bound_type == "up" else self.lower_bounds
        target.clear()
        target.update(
            _sanitise_named_vector(
                bounds,
                self.InflationProblem,
                self.include_latents,
                self.variable_names,
            )
        )
        self._mark_model_dirty()

    def set_objective(
        self,
        objective: NamedVector | None,
        direction: str = "max",
        constant: float | None = None,
        name: str | None = None,
    ) -> None:
        self.objective = _sanitise_named_vector(
            objective,
            self.InflationProblem,
            self.include_latents,
            self.variable_names,
        )
        self.maximize = direction.strip().lower() in {
            "max",
            "maximum",
            "maximize",
            "maximise",
        }
        if constant is not None:
            self.objective_constant = float(constant)
        if name is not None:
            self.objective_name = name
        if self.model is not None and self.entropy_vector is not None and not self._model_dirty:
            self.model.objective(
                self.objective_name,
                ObjectiveSense.Maximize if self.maximize else ObjectiveSense.Minimize,
                _build_objective_expression(
                    self.entropy_vector,
                    self.label_to_index,
                    self.variable_names,
                    self.objective,
                    constant=self.objective_constant,
                ),
            )

    def set_extra_equalities(self, extra_equalities: Sequence[NamedVector] | None) -> None:
        self.extra_equalities = [
            _sanitise_named_vector(
                eq,
                self.InflationProblem,
                self.include_latents,
                self.variable_names,
            )
            for eq in (extra_equalities or [])
        ]
        self._mark_model_dirty()

    def set_extra_inequalities(self, extra_inequalities: Sequence[NamedVector] | None) -> None:
        self.extra_inequalities = [
            _sanitise_named_vector(
                ineq,
                self.InflationProblem,
                self.include_latents,
                self.variable_names,
            )
            for ineq in (extra_inequalities or [])
        ]
        self._mark_model_dirty()

    def reset(self, which: Union[str, List[str]] = "all") -> None:
        if isinstance(which, str):
            if which == "all":
                self.reset(
                    ["objective", "values", "equalities", "inequalities", "lowerbounds", "upperbounds"]
                )
            elif which == "objective":
                self.objective = {}
                self.objective_constant = 0.0
            elif which == "values":
                self.known_values = {}
            elif which == "bounds":
                self.lower_bounds = {}
                self.upper_bounds = {}
            elif which == "lowerbounds":
                self.lower_bounds = {}
            elif which == "upperbounds":
                self.upper_bounds = {}
            elif which == "equalities":
                self.extra_equalities = []
            elif which == "inequalities":
                self.extra_inequalities = []
            else:
                raise ValueError(f"Unknown reset target: {which}")
        else:
            for item in which:
                self.reset(item)
        if which != "objective":
            self._mark_model_dirty()
        collect()

    def solve(self) -> None:
        self._rebuild_model()
        self.model.solve()
        problem_status = str(self.model.getProblemStatus())
        solution_status = str(self.model.getPrimalSolutionStatus())
        self.success = problem_status in {
            "ProblemStatus.PrimalAndDualFeasible",
            "ProblemStatus.PrimalFeasible",
        }
        self.status = solution_status
        self.objective_value = float(self.model.primalObjValue()) if self.success else problem_status
        self.solution_object = {
            "problem_status": problem_status,
            "solution_status": solution_status,
            "objective_value": self.objective_value,
        }

    def solve_result(
        self,
        *,
        include_farkas_certificate: bool = False,
        farkas_bound: float = 1.0,
        certificate_tol: float = 1e-9,
    ) -> EntropicSolveResult:
        self.solve()
        problem_status = str(self.model.getProblemStatus())
        solution_status = str(self.model.getPrimalSolutionStatus())
        is_feasible = self.success
        objective_value = float(self.model.primalObjValue()) if is_feasible else None
        farkas_certificate = None
        if include_farkas_certificate and not is_feasible:
            farkas_certificate = self.farkas_certificate(
                bound=farkas_bound,
                tol=certificate_tol,
            )
        return EntropicSolveResult(
            problem=self,
            is_feasible=is_feasible,
            problem_status=problem_status,
            solution_status=solution_status,
            objective_value=objective_value,
            farkas_certificate=farkas_certificate,
        )

    @property
    def automatic_equalities(self) -> Dict[str, object]:
        return {
            "independencies": [
                tuple(self.InflationProblem.public_subset_label(part) for part in item)
                for item in self.independencies
            ],
            "factorization_equalities": [
                tuple(
                    tuple(self.InflationProblem.public_node_names[label] for label in block)
                    for block in blocks
                )
                for blocks in self.factorization_equalities
            ],
            "symmetry_generators": list(self.symmetry_generators),
        }

    @property
    def known_values_by_label(self) -> Dict[str, float]:
        return dict(self.known_values)

    def certificate_as_dict(
        self,
        *,
        chop_tol: float = 1e-10,
        round_decimals: int = 3,
        bound: float = 1.0,
    ) -> Dict[str, float]:
        cert = self.farkas_certificate(bound=bound, tol=chop_tol)
        if not cert.is_separating:
            return {}
        raw = _expression_to_dict(cert.expression)
        if not raw:
            return {}
        return _clean_coefficients(raw, chop_tol=chop_tol, round_decimals=round_decimals)

    def certificate_as_string(
        self,
        *,
        chop_tol: float = 1e-10,
        round_decimals: int = 3,
        bound: float = 1.0,
    ) -> str:
        cert = self.certificate_as_dict(
            chop_tol=chop_tol,
            round_decimals=round_decimals,
            bound=bound,
        )
        if not cert:
            return "0 >= 0"
        return _dict_to_string(cert) + " >= 0"

    def farkas_certificate(self, *, bound: float = 1.0, tol: float = 1e-9) -> FarkasCertificate:
        objective_value, expression, multipliers = _solve_farkas_certificate(
            self.matrix["M"],
            self.matrix["b"],
            self.matrix["b_caption"],
            bound=bound,
            tol=tol,
        )
        return FarkasCertificate(
            objective_value=objective_value,
            expression=expression,
            multipliers=multipliers,
            is_separating=objective_value < -tol,
        )

    def write_to_file(self, filename: str) -> None:
        self._rebuild_model()
        self.model.writeTask(filename)


def entropy_label(spec: SetSpec, var_names: Sequence[str] | None = None) -> str:
    names = _split_set_spec(spec)
    names = list(dict.fromkeys(names))
    if var_names is not None:
        order = {name: i for i, name in enumerate(var_names)}
        unknown = [name for name in names if name not in order]
        if unknown:
            raise ValueError(f"unknown variable names in entropy label: {unknown}")
        names = sorted(names, key=lambda name: order[name])
    return "H(" + ",".join(names) + ")"


def _build_elemental_problem(
    variable_names: Sequence[str],
    *,
    independencies: Sequence[Sequence[str]] | None,
    factorization_equalities: List[tuple[tuple[str, ...], ...]] | None,
    symmetry_generators: Sequence[Dict[str, str]] | None,
    candidate: NamedVector | None,
    inflationproblem: InflationProblem,
    lower_bounds: Mapping[str, float] | None,
    upper_bounds: Mapping[str, float] | None,
    extra_equalities: Sequence[NamedVector] | None,
    extra_inequalities: Sequence[NamedVector] | None,
) -> tuple[Model, Variable, Dict[str, int], List[str], Dict[str, object]]:
    (
        M,
        b,
        b_caption,
        tuple_caption,
        x_caption,
        label_to_index,
    ) = _build_elemental_inequality_system(variable_names)
    n = len(x_caption)
    seen_symmetries = set()
    base_matrix = M.tocoo(copy=False)
    row_entries = np.asarray(base_matrix.row, dtype=np.int32).tolist()
    col_entries = np.asarray(base_matrix.col, dtype=np.int32).tolist()
    data_entries = np.asarray(base_matrix.data, dtype=float).tolist()
    rhs_entries = list(b)
    caption_entries = list(b_caption)
    next_row = base_matrix.shape[0]

    def append_leq(entries: Mapping[int, float], rhs: float, caption: str) -> None:
        nonlocal next_row
        for col, value in entries.items():
            if abs(value) < 1e-15:
                continue
            row_entries.append(next_row)
            col_entries.append(int(col))
            data_entries.append(float(value))
        rhs_entries.append(float(rhs))
        caption_entries.append(caption)
        next_row += 1

    def append_eq(entries: Mapping[int, float], caption: str = "0") -> None:
        append_leq(entries, 0.0, caption)
        append_leq({col: -value for col, value in entries.items()}, 0.0, caption)

    if independencies:
        for item in independencies:
            entries = _independence_entries(item, variable_names, label_to_index)
            append_eq(entries)

    if factorization_equalities is not None:
        for subset in tuple_caption:
            subset_labels = tuple(variable_names[i] for i in subset)
            blocks = inflationproblem.factorization_blocks(subset_labels)
            if len(blocks) <= 1:
                continue
            factorization_equalities.append(blocks)
            whole_label = _tuple_to_entropy_label(subset, variable_names)
            entries = {label_to_index[whole_label]: -1.0}
            for block in blocks:
                block_label = entropy_label(block, variable_names)
                index = label_to_index[block_label]
                entries[index] = entries.get(index, 0.0) + 1.0
            append_eq(entries)

    if inflationproblem.inflation_mode == "manual":
        symmetry_classes: Dict[tuple, List[str]] = {}
        for subset in tuple_caption:
            subset_labels = tuple(variable_names[i] for i in subset)
            signature = inflationproblem.entropic_subset_signature(subset_labels)
            symmetry_classes.setdefault(signature, []).append(
                _tuple_to_entropy_label(subset, variable_names)
            )
        for labels_in_class in symmetry_classes.values():
            if len(labels_in_class) <= 1:
                continue
            representative = labels_in_class[0]
            for image_label in labels_in_class[1:]:
                key = tuple(sorted((representative, image_label)))
                if key in seen_symmetries:
                    continue
                seen_symmetries.add(key)
                append_eq(
                    {
                        label_to_index[representative]: 1.0,
                        label_to_index[image_label]: -1.0,
                    }
                )
    elif symmetry_generators:
        for mapping in symmetry_generators:
            for subset in tuple_caption:
                label = _tuple_to_entropy_label(subset, variable_names)
                try:
                    image_names = [mapping[variable_names[i]] for i in subset]
                except KeyError:
                    continue
                image_label = entropy_label(image_names, variable_names)
                if label == image_label:
                    continue
                key = tuple(sorted((label, image_label)))
                if key in seen_symmetries:
                    continue
                seen_symmetries.add(key)
                append_eq(
                    {
                        label_to_index[label]: 1.0,
                        label_to_index[image_label]: -1.0,
                    }
                )

    if candidate:
        for spec, value in candidate.items():
            label = str(spec)
            entries = {label_to_index[label]: 1.0}
            append_leq(entries, float(value), label)
            append_leq({label_to_index[label]: -1.0}, -float(value), f"-{label}")

    for label, value in (upper_bounds or {}).items():
        append_leq({label_to_index[str(label)]: 1.0}, float(value), str(label))

    for label, value in (lower_bounds or {}).items():
        append_leq({label_to_index[str(label)]: -1.0}, -float(value), f"-{label}")

    for equality in extra_equalities or []:
        append_eq(_linear_expression_entries(equality, label_to_index))

    for inequality in extra_inequalities or []:
        append_leq(
            {col: -value for col, value in _linear_expression_entries(inequality, label_to_index).items()},
            0.0,
            "0",
        )

    M = coo_array(
        (
            np.asarray(data_entries, dtype=float),
            (np.asarray(row_entries, dtype=np.int32), np.asarray(col_entries, dtype=np.int32)),
        ),
        shape=(next_row, n),
    )

    model = Model(f"entropic_inflation_lp_{len(variable_names)}")
    # Basic-solution recovery dominates runtime on large latent-inclusive LPs,
    # while the package only needs the interior-point optimum/status here.
    model.setSolverParam("intpntBasis", "never")
    entropy_vector = model.variable("h", len(x_caption), Domain.unbounded())
    if M.shape[0]:
        model.constraint(
            Expr.mul(_scipy_to_mosek(M), entropy_vector),
            Domain.lessThan(rhs_entries),
        )
    matrix = {"M": M, "b": rhs_entries, "b_caption": caption_entries, "x_caption": x_caption}
    return model, entropy_vector, label_to_index, x_caption, matrix


def _build_elemental_inequality_system(
    variable_names: Sequence[str],
) -> tuple[
    coo_array,
    List[float],
    List[str],
    List[tuple[int, ...]],
    List[str],
    Dict[str, int],
]:
    basic_matrix, tuple_caption, x_caption, label_to_index_items = _cached_elemental_system(
        tuple(variable_names)
    )
    tuple_caption = list(tuple_caption)
    x_caption = list(x_caption)
    label_to_index = dict(label_to_index_items)
    M = basic_matrix.astype(float, copy=False).tocoo(copy=False)
    b = [0.0] * M.shape[0]
    b_caption = ["0"] * M.shape[0]
    return M, b, b_caption, tuple_caption, x_caption, label_to_index


@lru_cache(maxsize=None)
def _cached_elemental_system(
    variable_names: tuple[str, ...],
) -> tuple[coo_array, tuple[tuple[int, ...], ...], tuple[str, ...], tuple[tuple[str, int], ...]]:
    basic_matrix, tuple_caption = build_elemental_shannon_cone(len(variable_names))
    tuple_caption_tuple = tuple(tuple(item) for item in tuple_caption)
    x_caption_tuple = tuple(_tuple_to_entropy_label(item, variable_names) for item in tuple_caption_tuple)
    label_to_index_items = tuple((label, i) for i, label in enumerate(x_caption_tuple))
    return basic_matrix, tuple_caption_tuple, x_caption_tuple, label_to_index_items


def _tuple_to_entropy_label(item: Sequence[int], variable_names: Sequence[str]) -> str:
    return "H(" + ",".join(variable_names[i] for i in item) + ")"


def _split_set_spec(spec: SetSpec) -> List[str]:
    if isinstance(spec, str):
        text = spec.strip()
        if text.startswith("H(") and text.endswith(")"):
            text = text[2:-1]
        return _split_top_level_commas(text)

    out: List[str] = []
    for item in spec:
        text = str(item).strip()
        if not text:
            continue
        if text.startswith("H(") and text.endswith(")"):
            text = text[2:-1]
        out.extend(_split_top_level_commas(text))
    return out


def _independence_entries(
    item: Sequence[str],
    variable_names: Sequence[str],
    label_to_index: Dict[str, int],
) -> Dict[int, float]:
    if len(item) not in {2, 3}:
        raise ValueError(f"independence entry must have 2 or 3 items: {item}")

    x_label = entropy_label(item[0], variable_names)
    y_label = entropy_label(item[1], variable_names)
    if len(item) == 2:
        xy_label = entropy_label(
            _split_set_spec(item[0]) + _split_set_spec(item[1]),
            variable_names,
        )
        return {
            label_to_index[x_label]: 1.0,
            label_to_index[y_label]: 1.0,
            label_to_index[xy_label]: -1.0,
        }

    z_label = entropy_label(item[2], variable_names)
    xz_label = entropy_label(
        _split_set_spec(item[0]) + _split_set_spec(item[2]),
        variable_names,
    )
    yz_label = entropy_label(
        _split_set_spec(item[1]) + _split_set_spec(item[2]),
        variable_names,
    )
    xyz_label = entropy_label(
        _split_set_spec(item[0]) + _split_set_spec(item[1]) + _split_set_spec(item[2]),
        variable_names,
    )
    return {
        label_to_index[xz_label]: 1.0,
        label_to_index[yz_label]: 1.0,
        label_to_index[xyz_label]: -1.0,
        label_to_index[z_label]: -1.0,
    }


def _build_objective_expression(
    entropy_vector: Variable,
    label_to_index: Dict[str, int],
    var_names: Sequence[str],
    objective: NamedVector | None,
    *,
    constant: float,
) -> Expr:
    expr = Expr.constTerm(float(constant))
    if not objective:
        return expr
    for spec, coefficient in objective.items():
        label = str(spec)
        expr = Expr.add(
            expr,
            Expr.mul(float(coefficient), entropy_vector.index(label_to_index[label])),
        )
    return expr


def _sanitise_named_vector(
    values: Mapping[str, float | int] | None,
    problem: InflationProblem,
    include_latents: bool,
    var_names: Sequence[str],
) -> Dict[str, float]:
    if not values:
        return {}
    return {
        entropy_label(problem.canonical_subset_label(spec), var_names): float(value)
        for spec, value in values.items()
    }


def _linear_expression_entries(
    expression: Mapping[str, float],
    label_to_index: Dict[str, int],
) -> Dict[int, float]:
    coeffs: Dict[int, float] = {}
    for label, coefficient in expression.items():
        index = label_to_index[str(label)]
        coeffs[index] = coeffs.get(index, 0.0) + float(coefficient)
    return coeffs


def _scipy_to_mosek(mat: coo_array) -> Matrix:
    internal_mat = mat.tocoo(copy=False)
    return Matrix.sparse(
        *internal_mat.shape,
        np.asarray(internal_mat.row, dtype=np.int32),
        np.asarray(internal_mat.col, dtype=np.int32),
        np.asarray(internal_mat.data, dtype=float),
    )


def _solve_farkas_certificate(
    A: Sequence[Sequence[float]] | coo_array,
    b: Sequence[float],
    b_caption: Sequence[str],
    *,
    bound: float,
    tol: float,
) -> tuple[float, str, List[float]]:
    if issparse(A):
        A_T = A.transpose().tocoo(copy=False)
    elif A:
        n_cols = len(A[0])
        A_T = [list(col) for col in zip(*A)]
        assert all(len(row) == n_cols for row in A)
    else:
        A_T = []

    model = Model("entropic_farkas_lp")
    y = model.variable("y", len(b), Domain.inRange(0.0, bound))
    if issparse(A_T):
        if A_T.shape[0]:
            model.constraint(Expr.mul(_scipy_to_mosek(A_T), y), Domain.greaterThan(0.0))
    elif A_T:
        model.constraint(Expr.mul(A_T, y), Domain.greaterThan(0.0))
    model.objective("obj", ObjectiveSense.Minimize, Expr.dot(b, y))
    model.solve()

    y_values = y.level().tolist()
    objective_value = float(model.primalObjValue())
    expression = _format_farkas_expression(y_values, b_caption, tol=tol)
    return objective_value, expression, y_values


def _format_farkas_expression(
    y: Sequence[float],
    b_caption: Sequence[str],
    *,
    tol: float,
) -> str:
    terms: Dict[str, float] = {}
    for coef, caption in zip(y, b_caption):
        if caption == "0" or abs(coef) < tol:
            continue
        sign = 1.0
        label = caption
        if label.startswith("-"):
            sign = -1.0
            label = label[1:]
        terms[label] = terms.get(label, 0.0) + sign * float(coef)
    return _dict_to_string(terms) if terms else "0"


def _expression_to_dict(expression: str) -> Dict[str, float]:
    expression = expression.strip()
    if expression == "0" or not expression:
        return {}
    result: Dict[str, float] = {}
    for chunk in expression.split(" + "):
        coef_text, label = chunk.split("*", 1)
        result[label] = result.get(label, 0.0) + float(coef_text)
    return result


def _clean_coefficients(
    cert: Dict[str, float],
    *,
    chop_tol: float = 1e-10,
    round_decimals: int = 3,
) -> Dict[str, float]:
    if not cert:
        return {}
    good = {k: v for k, v in cert.items() if abs(v) > abs(chop_tol)}
    if not good:
        return {}
    normalizer = min(abs(v) for v in good.values()) if chop_tol > 0 else max(abs(v) for v in good.values())
    return {k: round(v / normalizer, round_decimals) for k, v in good.items()}


def _dict_to_string(coeffs: Mapping[str, float]) -> str:
    if not coeffs:
        return "0"
    parts = []
    for label in sorted(coeffs):
        coef = float(coeffs[label])
        if abs(coef) < 1e-12:
            continue
        parts.append(f"{coef:g}*{label}")
    return " + ".join(parts) if parts else "0"


def _split_top_level_commas(text: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                parts.append(token)
            current = []
            continue
        current.append(char)
    token = "".join(current).strip()
    if token:
        parts.append(token)
    return parts


def _parse_objective_sense(sense: str) -> ObjectiveSense:
    normalized = sense.strip().lower()
    if normalized in {"min", "minimum", "minimize", "minimise"}:
        return ObjectiveSense.Minimize
    if normalized in {"max", "maximum", "maximize", "maximise"}:
        return ObjectiveSense.Maximize
    raise ValueError("objective_sense must be 'min' or 'max'")
