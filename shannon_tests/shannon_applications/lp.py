"""Minimal application API for Shannon inflation LPs.

This module is a thin wrapper around ``shannon_tests.shannon_utils.LP_test``.
It keeps the existing Shannon-cone machinery, but exposes a smaller API aimed
at concrete applications:

- declare the inflated variables (or contexts),
- add independence constraints ``I(X;Y|Z)=0``,
- add symmetry constraints ``H(S1)=H(S2)``,
- pin observed entropy coordinates ``H(S)=value``,
- optionally optimize a linear functional of the entropy vector.

Mathematically, the LP variable is the entropy vector ``h`` whose coordinates
are the non-empty subset entropies ``H(S)`` of the declared inflated
variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Union

from mosek.fusion import Domain, Expr, Model, ObjectiveSense, Variable

from shannon_tests.shannon_utils import LP_test

SetSpec = Union[str, Iterable[str]]
NamedVector = Mapping[str, float | int]
VariablesSpec = Union[Sequence[str], Sequence[Sequence[str]]]


@dataclass(slots=True)
class ShannonInflationLP:
    """Container returned by ``build_shannon_inflation_lp``.

    Attributes
    ----------
    model:
        MOSEK Fusion model containing all Shannon, inflation, and candidate
        constraints.
    entropy_vector:
        The LP decision variable ``h``. Each coordinate is one entropy
        component ``H(S)``.
    label_to_index:
        Maps an entropy label such as ``H(A0,B1)`` to the coordinate of ``h``
        representing that entropy.
    entropy_labels:
        Caption for the coordinates of ``h``, aligned with ``label_to_index``.
    variable_names:
        Flat list of declared variable names after context flattening.
    constraints_meta:
        Metadata returned by the underlying builder. This is useful if later
        applications want certificates or raw matrix access.
    matrix:
        Optional raw matrix data when ``return_matrix=True`` was requested.
    """

    model: Model
    entropy_vector: Variable
    label_to_index: Dict[str, int]
    entropy_labels: List[str]
    variable_names: List[str]
    constraints_meta: List[Dict[str, object]] | None
    matrix: Dict[str, object] | None = None

    def solve(self) -> None:
        """Solve the LP with the current objective."""
        self.model.solve()

    def farkas_certificate(
        self,
        *,
        bound: float = 1.0,
        tol: float = 1e-9,
    ) -> "FarkasCertificate":
        """Return a Farkas-style infeasibility certificate from the raw matrix.

        This uses the older certificate logic already present in
        ``entropy_utils.py``: solve ``min b^T y`` subject to ``A^T y >= 0`` and
        ``0 <= y <= bound`` for the stored matrix representation ``A h <= b``.
        """

        if self.matrix is None:
            raise ValueError(
                "matrix data is required for a Farkas certificate; "
                "build the problem with return_matrix=True"
            )

        matrix = self.matrix
        objective_value, expression, multipliers = _solve_farkas_certificate(
            matrix["M"],
            matrix["b"],
            matrix["b_caption"],
            bound=bound,
            tol=tol,
        )
        return FarkasCertificate(
            objective_value=objective_value,
            expression=expression,
            multipliers=multipliers,
            is_separating=objective_value < -tol,
        )

    def solve_result(
        self,
        *,
        include_farkas_certificate: bool = False,
        farkas_bound: float = 1.0,
        certificate_tol: float = 1e-9,
    ) -> "ShannonSolveResult":
        """Solve the LP and return statuses, objective value, and certificate."""

        self.solve()

        problem_status = str(self.model.getProblemStatus())
        solution_status = str(self.model.getPrimalSolutionStatus())
        is_feasible = problem_status in {
            "ProblemStatus.PrimalAndDualFeasible",
            "ProblemStatus.PrimalFeasible",
        }

        objective_value: float | None
        if is_feasible:
            objective_value = float(self.model.primalObjValue())
        else:
            objective_value = None

        farkas_certificate = None
        if include_farkas_certificate and not is_feasible:
            farkas_certificate = self.farkas_certificate(
                bound=farkas_bound,
                tol=certificate_tol,
            )

        return ShannonSolveResult(
            problem=self,
            is_feasible=is_feasible,
            problem_status=problem_status,
            solution_status=solution_status,
            objective_value=objective_value,
            farkas_certificate=farkas_certificate,
        )


@dataclass(slots=True)
class CandidateConeResult:
    """Result of a cone-membership test for one candidate.

    A candidate is considered valid when the LP obtained by pinning its
    entropy coordinates admits a feasible point in the Shannon cone plus the
    user-specified equality constraints.
    """

    index: int
    candidate: Dict[str, float]
    is_valid: bool
    problem_status: str
    solution_status: str
    objective_value: float | None = None


@dataclass(slots=True)
class FarkasCertificate:
    """Farkas-style infeasibility certificate built from the LP matrix."""

    objective_value: float
    expression: str | None
    multipliers: List[float]
    is_separating: bool


@dataclass(slots=True)
class ShannonSolveResult:
    """Solved end-to-end Shannon LP result."""

    problem: ShannonInflationLP
    is_feasible: bool
    problem_status: str
    solution_status: str
    objective_value: float | None = None
    farkas_certificate: FarkasCertificate | None = None


def build_shannon_inflation_lp(
    variables: VariablesSpec,
    *,
    independencies: Sequence[Sequence[SetSpec]] | None = None,
    symmetries: Sequence[Sequence[SetSpec]] | None = None,
    candidate: NamedVector | None = None,
    objective: NamedVector | None = None,
    basic_inequalities: str = "elemental",
    objective_sense: str = "min",
    objective_constant: float = 0.0,
    objective_name: str = "objective",
    return_matrix: bool = False,
) -> ShannonInflationLP:
    """Build a Shannon inflation LP with a small application-facing API.

    Parameters
    ----------
    variables:
        Either a flat list of inflated variable names, e.g.
        ``["A0", "A1", "B0", "B1"]``, or a list of jointly observable
        contexts, e.g. ``[["A0", "B0"], ["A1", "B1"]]``.

        These names determine the coordinates of the entropy vector ``h``.

    independencies:
        Each entry is either ``[X, Y]`` for ``I(X;Y)=0`` or ``[X, Y, Z]`` for
        ``I(X;Y|Z)=0``. These are equality constraints added on top of the
        Shannon cone.

    symmetries:
        Each entry is ``[S1, S2]`` and enforces ``H(S1)=H(S2)``. These are
        useful for copy symmetries in inflation problems.

    candidate:
        Sparse dictionary fixing selected entropy coordinates, e.g.
        ``{"A0": 1.0, "A0,B0": 1.5}``. Each item becomes the equality
        constraint ``H(S)=value``.

    objective:
        Sparse linear functional of the entropy vector. For example,
        ``{"A0": 1.0, "B1": 1.0, "A0,B1": -1.0}`` represents
        ``H(A0) + H(B1) - H(A0,B1)``.

    basic_inequalities:
        Choice of Shannon-cone outer approximation:

        - ``"elemental"`` uses Lucas's elemental inequalities for ``n``
          variables, namely the elemental conditional entropies and elemental
          conditional mutual informations.
        - ``"full"`` uses the existing builder from ``shannon_utils.py``,
          which enumerates a larger family of monotonicity and strong
          subadditivity constraints.

    objective_sense:
        Either ``"min"`` or ``"max"``.
    """

    variable_names = _all_variable_names(variables)
    candidate_values, candidate_caption = _named_vector_to_caption(
        candidate,
        variable_names,
    )

    if basic_inequalities.strip().lower() == "full":
        built = LP_test(
            variables,
            indep_input=independencies,
            symmetry_input=symmetries,
            candidate=candidate_values,
            candidate_caption=candidate_caption,
            return_matrix=return_matrix,
        )

        if return_matrix:
            (
                model,
                entropy_vector,
                label_to_index,
                _labels,
                _var_names,
                constraints_meta,
                matrix,
            ) = built
        else:
            (
                model,
                entropy_vector,
                label_to_index,
                _labels,
                _var_names,
                constraints_meta,
            ) = built
            matrix = None
    elif basic_inequalities.strip().lower() == "elemental":
        (
            model,
            entropy_vector,
            label_to_index,
            constraints_meta,
            matrix,
        ) = _build_elemental_problem(
            variable_names,
            independencies=independencies,
            symmetries=symmetries,
            candidate_values=candidate_values,
            candidate_caption=candidate_caption,
            return_matrix=return_matrix,
        )
    else:
        raise ValueError("basic_inequalities must be 'elemental' or 'full'")

    objective_expr = _build_objective_expression(
        entropy_vector,
        label_to_index,
        variable_names,
        objective,
        constant=objective_constant,
    )
    model.objective(
        objective_name,
        _parse_objective_sense(objective_sense),
        objective_expr,
    )

    entropy_labels = [None] * len(label_to_index)
    for label, index in label_to_index.items():
        entropy_labels[index] = label

    return ShannonInflationLP(
        model=model,
        entropy_vector=entropy_vector,
        label_to_index=label_to_index,
        entropy_labels=entropy_labels,  # type: ignore[arg-type]
        variable_names=variable_names,
        constraints_meta=constraints_meta,
        matrix=matrix,
    )


def test_candidates_in_cone(
    variables: VariablesSpec,
    candidates: Sequence[NamedVector],
    *,
    independencies: Sequence[Sequence[SetSpec]] | None = None,
    symmetries: Sequence[Sequence[SetSpec]] | None = None,
    basic_inequalities: str = "elemental",
) -> List[CandidateConeResult]:
    """Loop over candidates and test whether each lies inside the cone.

    For each candidate, this builds a feasibility LP whose decision variable is
    the entropy vector ``h``. The constraint system contains:

    - the chosen Shannon cone inequalities on ``h``,
    - the user-declared independence equalities on ``h``,
    - the user-declared symmetry equalities on ``h``,
    - the candidate equalities ``H(S)=value`` for the supplied coordinates.

    If the resulting LP is feasible, the candidate is declared valid.
    """

    results: List[CandidateConeResult] = []
    for index, candidate in enumerate(candidates):
        problem = build_shannon_inflation_lp(
            variables=variables,
            independencies=independencies,
            symmetries=symmetries,
            candidate=candidate,
            basic_inequalities=basic_inequalities,
        )
        problem.solve()

        problem_status = str(problem.model.getProblemStatus())
        solution_status = str(problem.model.getPrimalSolutionStatus())
        is_valid = problem_status in {
            "ProblemStatus.PrimalAndDualFeasible",
            "ProblemStatus.PrimalFeasible",
        }

        objective_value: float | None
        if is_valid:
            objective_value = float(problem.model.primalObjValue())
        else:
            objective_value = None

        results.append(
            CandidateConeResult(
                index=index,
                candidate={str(key): float(value) for key, value in candidate.items()},
                is_valid=is_valid,
                problem_status=problem_status,
                solution_status=solution_status,
                objective_value=objective_value,
            )
        )

    return results


def solve_shannon_inflation_lp(
    variables: VariablesSpec,
    *,
    independencies: Sequence[Sequence[SetSpec]] | None = None,
    symmetries: Sequence[Sequence[SetSpec]] | None = None,
    candidate: NamedVector | None = None,
    objective: NamedVector | None = None,
    basic_inequalities: str = "elemental",
    objective_sense: str = "min",
    objective_constant: float = 0.0,
    objective_name: str = "objective",
    include_farkas_certificate: bool = False,
    farkas_bound: float = 1.0,
    certificate_tol: float = 1e-9,
) -> ShannonSolveResult:
    """Build and solve a Shannon inflation LP in one call."""

    problem = build_shannon_inflation_lp(
        variables=variables,
        independencies=independencies,
        symmetries=symmetries,
        candidate=candidate,
        objective=objective,
        basic_inequalities=basic_inequalities,
        objective_sense=objective_sense,
        objective_constant=objective_constant,
        objective_name=objective_name,
        return_matrix=include_farkas_certificate,
    )
    return problem.solve_result(
        include_farkas_certificate=include_farkas_certificate,
        farkas_bound=farkas_bound,
        certificate_tol=certificate_tol,
    )


def entropy_label(spec: SetSpec, var_names: Sequence[str] | None = None) -> str:
    """Return the canonical entropy label for a subset specification.

    Accepted forms:
    - ``"A0,B1"``
    - ``"H(A0,B1)"``
    - ``("A0", "B1")``
    """

    names = _split_set_spec(spec)
    names = list(dict.fromkeys(names))
    if var_names is not None:
        order = {name: i for i, name in enumerate(var_names)}
        unknown = [name for name in names if name not in order]
        if unknown:
            raise ValueError(f"unknown variable names in entropy label: {unknown}")
        names = sorted(names, key=lambda name: order[name])
    return "H(" + ",".join(names) + ")"


def _all_variable_names(variables: VariablesSpec) -> List[str]:
    if variables and all(isinstance(item, (list, tuple)) for item in variables):
        flat: List[str] = []
        seen = set()
        for ctx in variables:  # type: ignore[assignment]
            for name in ctx:
                if name not in seen:
                    seen.add(name)
                    flat.append(str(name))
        return flat
    return [str(name) for name in variables]  # type: ignore[list-item]


def _split_set_spec(spec: SetSpec) -> List[str]:
    if isinstance(spec, str):
        text = spec.strip()
        if text.startswith("H(") and text.endswith(")"):
            text = text[2:-1]
        return [token.strip() for token in text.split(",") if token.strip()]

    out: List[str] = []
    for item in spec:
        text = str(item).strip()
        if not text:
            continue
        if text.startswith("H(") and text.endswith(")"):
            text = text[2:-1]
        if "," in text:
            out.extend(token.strip() for token in text.split(",") if token.strip())
        else:
            out.append(text)
    return out


def _named_vector_to_caption(
    values: NamedVector | None,
    var_names: Sequence[str],
) -> tuple[List[float] | None, List[str] | None]:
    if not values:
        return None, None

    caption: List[str] = []
    dense_values: List[float] = []
    for spec, value in values.items():
        caption.append(entropy_label(spec, var_names))
        dense_values.append(float(value))
    return dense_values, caption


def _build_objective_expression(
    entropy_vector: Variable,
    label_to_index: Dict[str, int],
    var_names: Sequence[str],
    objective: NamedVector | None,
    *,
    constant: float,
) -> Expr:
    # The LP variable is the entropy vector h, so each term below targets one
    # coordinate H(S) of h and adds the coefficient c_S * H(S) to the
    # objective functional.
    expr = Expr.constTerm(float(constant))
    if not objective:
        return expr

    for spec, coefficient in objective.items():
        label = entropy_label(spec, var_names)
        if label not in label_to_index:
            raise ValueError(f"objective refers to unknown entropy coordinate: {label}")
        expr = Expr.add(
            expr,
            Expr.mul(float(coefficient), entropy_vector.index(label_to_index[label])),
        )
    return expr


def _parse_objective_sense(sense: str) -> ObjectiveSense:
    normalized = sense.strip().lower()
    if normalized in {"min", "minimum", "minimize", "minimise"}:
        return ObjectiveSense.Minimize
    if normalized in {"max", "maximum", "maximize", "maximise"}:
        return ObjectiveSense.Maximize
    raise ValueError("objective_sense must be 'min' or 'max'")


def _solve_farkas_certificate(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    b_caption: Sequence[str],
    *,
    bound: float,
    tol: float,
) -> tuple[float, str, List[float]]:
    if len(A) != len(b):
        raise ValueError("A and b must have the same number of rows")

    if A:
        n_cols = len(A[0])
        for row in A:
            if len(row) != n_cols:
                raise ValueError("A must be a rectangular matrix")
        A_T = [list(col) for col in zip(*A)]
    else:
        A_T = []

    model = Model("shannon_farkas_lp")
    y = model.variable("y", len(b), Domain.inRange(0.0, bound))
    if A_T:
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
    if len(y) != len(b_caption):
        raise ValueError("y and b_caption must have the same length")

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

    out = []
    for label in sorted(terms.keys()):
        coef = terms[label]
        if abs(coef) < tol:
            continue
        out.append(f"{coef:g}*{label}")
    return " + ".join(out) if out else "0"


def _build_elemental_problem(
    variable_names: Sequence[str],
    *,
    independencies: Sequence[Sequence[SetSpec]] | None,
    symmetries: Sequence[Sequence[SetSpec]] | None,
    candidate_values: List[float] | None,
    candidate_caption: List[str] | None,
    return_matrix: bool,
) -> tuple[
    Model,
    Variable,
    Dict[str, int],
    List[Dict[str, object]] | None,
    Dict[str, object] | None,
]:
    (
        M,
        b,
        b_caption,
        x_caption,
        label_to_index,
        constraints_meta,
    ) = _build_elemental_inequality_system(variable_names)

    n = len(x_caption)

    def append_leq(coeffs: List[float], rhs: float, caption: str) -> None:
        # Each appended row is a linear inequality of the form coeffs * h <= rhs.
        M.append(coeffs)
        b.append(rhs)
        b_caption.append(caption)
        constraints_meta.append(
            {
                "constraint": None,
                "coeffs": [-value for value in coeffs],
                "const": rhs,
                "label": None if caption == "0" else caption,
                "sense": "ge",
            }
        )

    def append_eq(coeffs: List[float]) -> None:
        # An equality L(h)=0 is represented by the pair L(h)<=0 and -L(h)<=0.
        append_leq(coeffs, 0.0, "0")
        append_leq([-value for value in coeffs], 0.0, "0")

    if independencies:
        for item in independencies:
            coeffs = _independence_coefficients(item, variable_names, label_to_index, n)
            append_eq(coeffs)

    if symmetries:
        for pair in symmetries:
            if len(pair) != 2:
                raise ValueError(f"symmetry entry must have 2 items: {pair}")
            coeffs = [0.0] * n
            left = entropy_label(pair[0], variable_names)
            right = entropy_label(pair[1], variable_names)
            if left not in label_to_index or right not in label_to_index:
                raise ValueError(f"symmetry refers to unknown entropy coordinate: {pair}")
            coeffs[label_to_index[left]] += 1.0
            coeffs[label_to_index[right]] -= 1.0
            append_eq(coeffs)

    if candidate_values is not None and candidate_caption is not None:
        if len(candidate_values) != len(candidate_caption):
            raise ValueError("candidate_values and candidate_caption must have same length")

        for label, value in zip(candidate_caption, candidate_values):
            if label not in label_to_index:
                raise ValueError(f"candidate refers to unknown entropy coordinate: {label}")
            coeffs = [0.0] * n
            coeffs[label_to_index[label]] = 1.0
            append_leq(coeffs, float(value), label)
            append_leq([-entry for entry in coeffs], -float(value), f"-{label}")

    model = Model(f"shannon_elemental_lp_{len(variable_names)}")
    entropy_vector = model.variable("h", len(x_caption), Domain.unbounded())

    if M:
        constraint = model.constraint(Expr.mul(M, entropy_vector), Domain.lessThan(b))
        for meta in constraints_meta:
            meta["constraint"] = constraint

    matrix = None
    if return_matrix:
        matrix = {
            "M": M,
            "b": b,
            "b_caption": b_caption,
            "x_caption": x_caption,
        }

    return model, entropy_vector, label_to_index, constraints_meta, matrix


def _build_elemental_inequality_system(
    variable_names: Sequence[str],
) -> tuple[
    List[List[float]],
    List[float],
    List[str],
    List[str],
    Dict[str, int],
    List[Dict[str, object]],
]:
    from shannon_tests.feasibility_test.desc_entro.desc_entro import (
        desigualdades_basicas,
    )

    # Lucas's implementation returns the elemental Shannon inequalities for n
    # variables in matrix form A h <= 0, where h is the full entropy vector.
    basic_matrix, tuple_caption = desigualdades_basicas(len(variable_names))
    x_caption = [_tuple_to_entropy_label(item, variable_names) for item in tuple_caption]
    label_to_index = {label: i for i, label in enumerate(x_caption)}

    M = [[float(value) for value in row] for row in basic_matrix.tolist()]
    b = [0.0] * len(M)
    b_caption = ["0"] * len(M)
    constraints_meta = [
        {
            "constraint": None,
            "coeffs": [-value for value in row],
            "const": 0.0,
            "sense": "ge",
        }
        for row in M
    ]
    return M, b, b_caption, x_caption, label_to_index, constraints_meta


def _tuple_to_entropy_label(item: Sequence[int], variable_names: Sequence[str]) -> str:
    return "H(" + ",".join(variable_names[i] for i in item) + ")"


def _independence_coefficients(
    item: Sequence[SetSpec],
    variable_names: Sequence[str],
    label_to_index: Dict[str, int],
    n_coords: int,
) -> List[float]:
    if len(item) not in {2, 3}:
        raise ValueError(f"independence entry must have 2 or 3 items: {item}")

    x_label = entropy_label(item[0], variable_names)
    y_label = entropy_label(item[1], variable_names)
    if x_label not in label_to_index or y_label not in label_to_index:
        raise ValueError(f"independence refers to unknown entropy coordinate: {item}")

    coeffs = [0.0] * n_coords
    if len(item) == 2:
        xy_label = entropy_label(
            _split_set_spec(item[0]) + _split_set_spec(item[1]),
            variable_names,
        )
        if xy_label not in label_to_index:
            raise ValueError(f"independence refers to unknown entropy coordinate: {xy_label}")
        coeffs[label_to_index[x_label]] += 1.0
        coeffs[label_to_index[y_label]] += 1.0
        coeffs[label_to_index[xy_label]] -= 1.0
        return coeffs

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
    needed = [z_label, xz_label, yz_label, xyz_label]
    missing = [label for label in needed if label not in label_to_index]
    if missing:
        raise ValueError(f"independence refers to unknown entropy coordinates: {missing}")

    # I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z).
    coeffs[label_to_index[xz_label]] += 1.0
    coeffs[label_to_index[yz_label]] += 1.0
    coeffs[label_to_index[z_label]] -= 1.0
    coeffs[label_to_index[xyz_label]] -= 1.0
    return coeffs
