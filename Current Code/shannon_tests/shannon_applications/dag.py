"""High-level DAG API for Shannon source-model LP tests.

This module mirrors the DAG style used by the ``inflation`` package: the user
provides a child map, namely a dictionary whose keys are nodes and whose values
are the nodes they point to.

For example, the triangle source model is written as
``{"a": ["B", "C"], "b": ["A", "C"], "c": ["A", "B"]}``.

From this graph, the code infers:

- source nodes: the roots of the DAG,
- observed nodes: the non-root nodes,
- parent sets: by inverting the child map.

The LP decision variable is still the full entropy vector ``h`` on all nodes.
The DAG only determines which equality constraints are imposed:

- for each observed node ``X``, the local Markov equality
  ``I(X ; ND(X) \\ Pa(X) | Pa(X)) = 0``,
- for each source node ``s``, the source-independence equality
  ``I(s ; other sources) = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from .lp import (
    CandidateConeResult,
    NamedVector,
    SetSpec,
    ShannonInflationLP,
    ShannonSolveResult,
    build_shannon_inflation_lp,
    solve_shannon_inflation_lp,
    test_candidates_in_cone,
)


def build_shannon_dag_lp(
    *,
    dag: Mapping[str, Sequence[str]],
    symmetries: Sequence[Sequence[SetSpec]] | None = None,
    candidate: NamedVector | None = None,
    objective: NamedVector | None = None,
    basic_inequalities: str = "elemental",
    objective_sense: str = "min",
    objective_constant: float = 0.0,
    objective_name: str = "objective",
    extra_independencies: Sequence[Sequence[SetSpec]] | None = None,
    return_matrix: bool = False,
) -> ShannonInflationLP:
    """Build a Shannon LP from an inflation-style child-map DAG."""

    return ShannonSourceModel(
        dag=dag,
    ).build_lp(
        symmetries=symmetries,
        candidate=candidate,
        objective=objective,
        basic_inequalities=basic_inequalities,
        objective_sense=objective_sense,
        objective_constant=objective_constant,
        objective_name=objective_name,
        extra_independencies=extra_independencies,
        return_matrix=return_matrix,
    )


def test_dag_candidates_in_cone(
    *,
    dag: Mapping[str, Sequence[str]],
    candidates: Sequence[NamedVector],
    symmetries: Sequence[Sequence[SetSpec]] | None = None,
    basic_inequalities: str = "elemental",
    extra_independencies: Sequence[Sequence[SetSpec]] | None = None,
) -> List[CandidateConeResult]:
    """Test a list of observed candidates against the DAG-defined Shannon cone."""

    return ShannonSourceModel(
        dag=dag,
    ).test_candidates(
        candidates=candidates,
        symmetries=symmetries,
        basic_inequalities=basic_inequalities,
        extra_independencies=extra_independencies,
    )


def solve_shannon_dag_lp(
    *,
    dag: Mapping[str, Sequence[str]],
    symmetries: Sequence[Sequence[SetSpec]] | None = None,
    candidate: NamedVector | None = None,
    objective: NamedVector | None = None,
    basic_inequalities: str = "elemental",
    objective_sense: str = "min",
    objective_constant: float = 0.0,
    objective_name: str = "objective",
    extra_independencies: Sequence[Sequence[SetSpec]] | None = None,
    include_farkas_certificate: bool = False,
    farkas_bound: float = 1.0,
    certificate_tol: float = 1e-9,
) -> ShannonSolveResult:
    """Build and solve a DAG-defined Shannon LP in one call."""

    return ShannonSourceModel(dag=dag).solve(
        symmetries=symmetries,
        candidate=candidate,
        objective=objective,
        basic_inequalities=basic_inequalities,
        objective_sense=objective_sense,
        objective_constant=objective_constant,
        objective_name=objective_name,
        extra_independencies=extra_independencies,
        include_farkas_certificate=include_farkas_certificate,
        farkas_bound=farkas_bound,
        certificate_tol=certificate_tol,
    )


@dataclass(slots=True, init=False)
class ShannonSourceModel:
    """A causal source model with a DAG-driven LP interface.

    Parameters
    ----------
    dag:
        Child map in the same direction as the ``inflation`` package. Each key
        is a node and each value is the list of nodes it points to. Sources are
        inferred as the roots of this DAG, and observed variables are inferred
        as the non-root nodes.

    Notes
    -----
    The entropy LP is built on the variable order ``observed + sources``.
    Observed and source names are inferred from the graph; they do not need to
    be declared separately.
    """

    dag: Dict[str, List[str]]
    observed: List[str]
    sources: List[str]
    _parent_map: Dict[str, List[str]]

    def __init__(
        self,
        *,
        dag: Mapping[str, Sequence[str]],
    ) -> None:
        self.dag = _normalize_child_dag(dag)
        node_order = _topological_order(self.dag)
        self._parent_map = _invert_child_dag(self.dag, node_order)
        self.sources = sorted(node for node in node_order if not self._parent_map[node])
        self.observed = sorted(node for node in node_order if self._parent_map[node])

    @property
    def variable_names(self) -> List[str]:
        """LP variable order used to build the entropy vector ``h``."""

        return [*self.observed, *self.sources]

    @property
    def parent_map(self) -> Dict[str, List[str]]:
        return {node: list(parents) for node, parents in self._parent_map.items()}

    @property
    def children_map(self) -> Dict[str, List[str]]:
        return {node: list(children) for node, children in self.dag.items()}

    def descendants_of(self, node: str) -> List[str]:
        """Return descendants of ``node`` in the declared DAG order."""

        if node not in self.children_map:
            raise ValueError(f"unknown node in DAG: {node}")

        order = {name: index for index, name in enumerate(self.variable_names)}
        seen = set()
        stack = list(reversed(self.children_map[node]))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(reversed(self.children_map[current]))
        return sorted(seen, key=lambda name: order[name])

    def implied_independencies(self) -> List[List[str]]:
        """Translate the DAG into entropy equalities used by the LP.

        The output is a list in the same low-level format accepted by
        ``build_shannon_inflation_lp``.
        """

        independencies: List[List[str]] = []
        order = {name: index for index, name in enumerate(self.variable_names)}

        for node in self.observed:
            parents = self.parent_map[node]
            descendants = set(self.descendants_of(node))
            separated = [
                name
                for name in self.variable_names
                if name != node and name not in parents and name not in descendants
            ]
            if not separated:
                continue

            separated = sorted(separated, key=lambda name: order[name])
            if parents:
                independencies.append(
                    [node, _join_names(separated), _join_names(parents)]
                )
            else:
                independencies.append([node, _join_names(separated)])

        for source in self.sources:
            other_sources = [name for name in self.sources if name != source]
            if other_sources:
                independencies.append([source, _join_names(other_sources)])

        return independencies

    def build_lp(
        self,
        *,
        symmetries: Sequence[Sequence[SetSpec]] | None = None,
        candidate: NamedVector | None = None,
        objective: NamedVector | None = None,
        basic_inequalities: str = "elemental",
        objective_sense: str = "min",
        objective_constant: float = 0.0,
        objective_name: str = "objective",
        extra_independencies: Sequence[Sequence[SetSpec]] | None = None,
        return_matrix: bool = False,
    ) -> ShannonInflationLP:
        """Build the entropy LP attached to this source model.

        The DAG contributes equality constraints on the entropy vector ``h``;
        the Shannon cone contributes the inequality rows; candidates pin chosen
        observed coordinates of ``h``; and the optional objective is a linear
        functional on ``h``.
        """

        independencies = self.implied_independencies()
        if extra_independencies:
            independencies.extend([list(item) for item in extra_independencies])

        return build_shannon_inflation_lp(
            variables=self.variable_names,
            independencies=independencies,
            symmetries=symmetries,
            candidate=candidate,
            objective=objective,
            basic_inequalities=basic_inequalities,
            objective_sense=objective_sense,
            objective_constant=objective_constant,
            objective_name=objective_name,
            return_matrix=return_matrix,
        )

    def solve(
        self,
        *,
        symmetries: Sequence[Sequence[SetSpec]] | None = None,
        candidate: NamedVector | None = None,
        objective: NamedVector | None = None,
        basic_inequalities: str = "elemental",
        objective_sense: str = "min",
        objective_constant: float = 0.0,
        objective_name: str = "objective",
        extra_independencies: Sequence[Sequence[SetSpec]] | None = None,
        include_farkas_certificate: bool = False,
        farkas_bound: float = 1.0,
        certificate_tol: float = 1e-9,
    ) -> ShannonSolveResult:
        """Build and solve the Shannon LP induced by this DAG."""

        independencies = self.implied_independencies()
        if extra_independencies:
            independencies.extend([list(item) for item in extra_independencies])

        return solve_shannon_inflation_lp(
            variables=self.variable_names,
            independencies=independencies,
            symmetries=symmetries,
            candidate=candidate,
            objective=objective,
            basic_inequalities=basic_inequalities,
            objective_sense=objective_sense,
            objective_constant=objective_constant,
            objective_name=objective_name,
            include_farkas_certificate=include_farkas_certificate,
            farkas_bound=farkas_bound,
            certificate_tol=certificate_tol,
        )

    def test_candidates(
        self,
        *,
        candidates: Sequence[NamedVector],
        symmetries: Sequence[Sequence[SetSpec]] | None = None,
        basic_inequalities: str = "elemental",
        extra_independencies: Sequence[Sequence[SetSpec]] | None = None,
    ) -> List[CandidateConeResult]:
        """Check whether each candidate is feasible in this DAG-defined cone."""

        independencies = self.implied_independencies()
        if extra_independencies:
            independencies.extend([list(item) for item in extra_independencies])

        return test_candidates_in_cone(
            variables=self.variable_names,
            candidates=candidates,
            independencies=independencies,
            symmetries=symmetries,
            basic_inequalities=basic_inequalities,
        )


def _normalize_child_dag(dag: Mapping[str, Sequence[str]]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    all_nodes = set()

    for raw_parent, raw_children in dag.items():
        parent = str(raw_parent).strip()
        if not parent:
            raise ValueError("dag contains an empty node name")
        if parent in normalized:
            raise ValueError(f"dag contains duplicate node '{parent}'")

        children: List[str] = []
        seen_children = set()
        for raw_child in raw_children:
            child = str(raw_child).strip()
            if not child:
                raise ValueError(f"dag for '{parent}' contains an empty child name")
            if child == parent:
                raise ValueError(f"node '{parent}' cannot point to itself")
            if child in seen_children:
                raise ValueError(f"dag for '{parent}' contains duplicate children")
            seen_children.add(child)
            children.append(child)
            all_nodes.add(child)

        normalized[parent] = children
        all_nodes.add(parent)

    for node in sorted(all_nodes):
        normalized.setdefault(node, [])

    _check_acyclic(sorted(normalized), normalized)
    return normalized


def _invert_child_dag(
    dag: Mapping[str, Sequence[str]],
    node_order: Sequence[str],
) -> Dict[str, List[str]]:
    order = {name: index for index, name in enumerate(node_order)}
    parent_map = {node: [] for node in node_order}
    for parent, children in dag.items():
        for child in children:
            parent_map[child].append(parent)
    for node, parents in parent_map.items():
        parent_map[node] = sorted(parents, key=lambda name: order[name])
    return parent_map


def _topological_order(dag: Mapping[str, Sequence[str]]) -> List[str]:
    indegree = {node: 0 for node in dag}
    for children in dag.values():
        for child in children:
            indegree[child] += 1

    available = sorted(node for node, degree in indegree.items() if degree == 0)
    order: List[str] = []

    while available:
        node = available.pop(0)
        order.append(node)
        for child in sorted(dag[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                available.append(child)
                available.sort()

    if len(order) != len(dag):
        raise ValueError("dag must be acyclic")
    return order


def _check_acyclic(
    nodes: Sequence[str],
    children_map: Mapping[str, Sequence[str]],
) -> None:
    visit_state = {node: 0 for node in nodes}

    def visit(node: str) -> None:
        state = visit_state[node]
        if state == 1:
            raise ValueError("dag must be acyclic")
        if state == 2:
            return

        visit_state[node] = 1
        for child in children_map[node]:
            visit(child)
        visit_state[node] = 2

    for node in nodes:
        visit(node)


def _join_names(names: Sequence[str]) -> str:
    return ",".join(names)
