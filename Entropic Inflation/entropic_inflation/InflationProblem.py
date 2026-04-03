"""Entropic analogue of the reference inflation.InflationProblem.

This first implementation mirrors the structural scenario parsing logic of the
reference package while intentionally stopping short of probability/operator
specific internals. The class owns:

- DAG normalization
- visible party naming and ordering
- source classification
- hypergraph construction
- parent/setting bookkeeping
- inflation-level metadata
- inflation-copy index generation

The entropic LP semantics built on top of this object will be added later.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from functools import cached_property
from itertools import chain, combinations_with_replacement
from typing import Dict, List, Tuple, Union
from warnings import warn

import networkx as nx
import numpy as np


formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda msg, category, filename, lineno, line=None: (
    formatwarning_orig(msg, category, filename, lineno, line="")
)


class InflationProblem:
    """Class for encoding structural details of an inflated causal scenario."""

    def __init__(
        self,
        dag: Union[Dict, None] = None,
        outcomes_per_party: Union[Tuple[int, ...], List[int], np.ndarray] = tuple(),
        settings_per_party: Union[Tuple[int, ...], List[int], np.ndarray] = tuple(),
        inflation_level_per_source: Union[Tuple[int, ...], List[int], np.ndarray, int] = tuple(),
        inflation_mode: str = "automatic",
        inflation_dag: Union[Dict, None] = None,
        classical_sources: Union[str, Tuple[str, ...], List[str], None] = tuple(),
        nonclassical_intermediate_latents: Union[Tuple[str, ...], List[str]] = tuple(),
        classical_intermediate_latents: Union[Tuple[str, ...], List[str]] = tuple(),
        order: Union[Tuple[str, ...], List[str]] = tuple(),
        verbose: int = 0,
    ) -> None:
        self.verbose = verbose
        self.inflation_mode = str(inflation_mode).strip().lower()
        if self.inflation_mode not in {"automatic", "manual"}:
            raise ValueError("inflation_mode must be 'automatic' or 'manual'.")
        self.manual_inflation_dag = (
            {str(parent): set(map(str, children)) for parent, children in inflation_dag.items()}
            if inflation_dag
            else None
        )
        if self.inflation_mode == "manual" and not self.manual_inflation_dag:
            raise ValueError("Manual inflation mode requires a non-empty inflation_dag.")

        if not outcomes_per_party:
            raise ValueError("Please provide outcomes per party.")
        self.outcomes_per_party = np.asarray(outcomes_per_party, dtype=int)
        self.nr_parties = len(self.outcomes_per_party)

        if not settings_per_party:
            if self.verbose > 0:
                warn("No settings per party provided, assuming all parties have one setting.")
            self.private_settings_per_party = np.ones(self.nr_parties, dtype=int)
        else:
            self.private_settings_per_party = np.asarray(settings_per_party, dtype=int)
            if len(self.private_settings_per_party) != self.nr_parties:
                raise AssertionError(
                    f"You have specified a list of {len(outcomes_per_party)} outcomes and "
                    f"a list of {len(settings_per_party)} inputs. These lists must have the "
                    "same length and equal the number of visible variables in the scenario."
                )

        self.expected_distro_shape = tuple(
            np.hstack((self.outcomes_per_party, self.private_settings_per_party)).tolist()
        )

        self.classical_intermediate_latents = set(map(str, classical_intermediate_latents))
        self.nonclassical_intermediate_latents = set(
            map(str, nonclassical_intermediate_latents)
        )
        if not self.classical_intermediate_latents.isdisjoint(
            self.nonclassical_intermediate_latents
        ):
            raise AssertionError(
                "An intermediate latent cannot be both classical and nonclassical."
            )
        self.intermediate_latents = self.classical_intermediate_latents.union(
            self.nonclassical_intermediate_latents
        )

        names_have_been_set_yet = False
        if dag:
            implicit_names = set(map(str, chain.from_iterable(dag.values()))).difference(
                self.intermediate_latents
            )
            if len(implicit_names) != self.nr_parties:
                raise AssertionError(
                    "You must provide a number of outcomes for the following "
                    f"{len(implicit_names)} variables: {implicit_names}"
                )
            if order:
                sanity_check = implicit_names.issubset(order) and implicit_names.issuperset(order)
                if sanity_check:
                    self.names = list(map(str, order))
                    names_have_been_set_yet = True
                elif self.verbose > 0:
                    warn(
                        "The names read from the DAG do not match those read from the "
                        "`order` argument. The names used are those read from the DAG."
                    )
            if not names_have_been_set_yet:
                if len(implicit_names) > 1 and self.verbose > 0:
                    warn("The order of variables is inferred by the DAG according to lexicographic order.")
                self.names = sorted(implicit_names)
                names_have_been_set_yet = True

        if order and (not names_have_been_set_yet):
            if len(order) == self.nr_parties:
                self.names = list(map(str, order))
                names_have_been_set_yet = True
            elif self.verbose > 0:
                warn(
                    "The number of names provided does not match the number of variables "
                    "that need a name. The names used are determined by the list of "
                    "variables with outcomes."
                )

        if not names_have_been_set_yet:
            self.names = [chr(ord("A") + i) for i in range(self.nr_parties)]

        if not dag:
            if self.verbose > 0:
                warn(
                    "The DAG must be a non-empty dictionary with parent variables as keys "
                    "and lists of children as values. Defaulting to one global source."
                )
            self.dag = {"h_global": set(self.names)}
        else:
            self.dag = {str(parent): set(map(str, children)) for parent, children in dag.items()}

        nodes_with_children_as_list = list(self.dag.keys())
        nodes_with_children = set(nodes_with_children_as_list)
        self._actual_sources = np.asarray(
            sorted(
                nodes_with_children.difference(self.names, self.intermediate_latents),
                key=nodes_with_children_as_list.index,
            ),
            dtype=object,
        )
        self.nr_sources = len(self._actual_sources)

        if isinstance(classical_sources, str):
            if classical_sources.lower() == "all":
                self._classical_sources = np.ones(self.nr_sources, dtype=bool)
            else:
                raise ValueError(
                    f"The keyword argument classical_sources=`{classical_sources}` could not be parsed."
                )
        else:
            self._classical_sources = np.zeros(self.nr_sources, dtype=bool)

        if not isinstance(classical_sources, (str, type(None))):
            if not set(classical_sources).issubset(self._actual_sources):
                raise AssertionError(
                    "Some specified classical source cannot be found in the DAG."
                )
            for ii, source in enumerate(self._actual_sources):
                if source in classical_sources:
                    self._classical_sources[ii] = True

        self._nonclassical_sources = np.logical_not(self._classical_sources)

        self._inverse_dag = defaultdict(set)
        for parent, children in self.dag.items():
            for child in children:
                self._inverse_dag[child].add(parent)

        for il in self.intermediate_latents:
            parents = self._inverse_dag[il]
            if not parents.isdisjoint(self.names):
                raise NotImplementedError(
                    "InflationProblem cannot handle intermediate latents with observable parents at this time."
                )
            if parents.isdisjoint(self._actual_sources.flat):
                raise ValueError(
                    f"The intermediate latent {il} has no source parent. Please add a source parent to the DAG and re-initialize InflationProblem."
                )

        for ncil in self.nonclassical_intermediate_latents:
            parents = self._inverse_dag[ncil]
            if parents.isdisjoint(self._actual_sources[self._nonclassical_sources].flat):
                raise NotImplementedError(
                    f"The nonclassical intermediate latent {ncil} has no nonclassical source parent."
                )

        parties_with_children = nodes_with_children.intersection(self.names)
        self.has_children = np.zeros(self.nr_parties, dtype=bool)
        self.is_network = set(nodes_with_children).isdisjoint(self.names)
        names_to_positions = {party: position for position, party in enumerate(self.names)}
        adjacency_matrix = np.zeros((self.nr_parties, self.nr_parties), dtype=bool)

        for parent in parties_with_children:
            ii = names_to_positions[parent]
            self.has_children[ii] = True
            observable_children = set(self.dag[parent])
            if not observable_children.issubset(self.names):
                raise NotImplementedError(
                    "At this time InflationProblem does not accept DAGs with observed nodes pointing to intermediate latents."
                )
            latents_behind_this_node = self._inverse_dag[parent].difference(self.names)
            siblings_of_parent = set()
            for latent in latents_behind_this_node:
                siblings_of_parent.update(self.dag[latent])
            if not observable_children.issubset(siblings_of_parent):
                raise NotImplementedError(
                    "At this time InflationProblem does not accept DAGs with directed edges "
                    "between observed nodes lacking a common latent parent."
                )
            child_indices = [names_to_positions[child] for child in observable_children]
            adjacency_matrix[ii, child_indices] = True

        self.parents_per_party = list(map(np.flatnonzero, adjacency_matrix.T))
        settings_per_party_lst = [[s] for s in self.private_settings_per_party]
        for party_idx, party_parent_idxs in enumerate(self.parents_per_party):
            settings_per_party_lst[party_idx].extend(
                np.take(self.outcomes_per_party, party_parent_idxs)
            )
        self.settings_per_party = np.asarray(
            [np.prod(multisetting) for multisetting in settings_per_party_lst],
            dtype=int,
        )

        effective_to_parent_settings = []
        for i in range(self.nr_parties):
            effective_to_parent_settings.append(
                dict(zip(range(self.settings_per_party[i]), np.ndindex(*settings_per_party_lst[i])))
            )
        self.effective_to_parent_settings = effective_to_parent_settings

        self.hypergraph = np.zeros((self.nr_sources, self.nr_parties), dtype=np.uint8)
        self.sources_to_check_for_party_pair_commutation = np.zeros(
            (self.nr_parties, self.nr_parties, self.nr_sources), dtype=np.uint8
        )
        for source_idx, source in enumerate(self._actual_sources):
            quantum_source_bonus = self._nonclassical_sources[source_idx]
            children = self.dag[source]
            observable_children = children.intersection(self.names)
            latent_children = children.difference(self.names)
            if not latent_children.issubset(self.intermediate_latents):
                raise AssertionError(
                    f"{latent_children.difference(self.intermediate_latents)} are not all a recognized party or an intermediate latent."
                )
            for child in observable_children:
                child_idx = names_to_positions[child]
                self.hypergraph[source_idx, child_idx] = 1
                self.sources_to_check_for_party_pair_commutation[
                    child_idx, child_idx, source_idx
                ] = 1 + quantum_source_bonus
            for intermediate_latent in latent_children:
                observable_descendants = self.dag[intermediate_latent]
                if not observable_descendants.issubset(self.names):
                    raise AssertionError(
                        "At this time InflationProblem does not accept DAGs with intermediate latents pointing to other intermediate latents."
                    )
                quantum_connection_bonus = np.logical_and(
                    intermediate_latent in self.nonclassical_intermediate_latents,
                    quantum_source_bonus,
                ).astype(np.uint8)
                for desc in observable_descendants:
                    desc_idx = names_to_positions[desc]
                    self.hypergraph[source_idx, desc_idx] = 1
                    for desc2 in observable_descendants:
                        desc2_idx = names_to_positions[desc2]
                        self.sources_to_check_for_party_pair_commutation[
                            desc_idx, desc2_idx, source_idx
                        ] = max(
                            self.sources_to_check_for_party_pair_commutation[
                                desc_idx, desc2_idx, source_idx
                            ],
                            1 + quantum_connection_bonus,
                        )

        if not np.sum(self.hypergraph, axis=0).all():
            raise AssertionError(
                "There appears to be a party with no sources in its past. This is not allowed."
            )

        if not inflation_level_per_source:
            if self.verbose > 0:
                warn(
                    "The inflation level per source must be a non-empty list. "
                    "Defaulting to 1 (no inflation)."
                )
            self.inflation_level_per_source = np.array([1] * self.nr_sources)
        elif isinstance(inflation_level_per_source, int):
            self.inflation_level_per_source = np.array(
                [inflation_level_per_source] * self.nr_sources
            )
        else:
            self.inflation_level_per_source = np.array(inflation_level_per_source)
            if self.nr_sources != len(self.inflation_level_per_source):
                raise AssertionError(
                    "The number of sources and the number of inflation levels do not coincide."
                )

        shared_sources = [
            np.all(np.vstack(pair), axis=0)
            for pair in combinations_with_replacement(self.hypergraph.T, 2)
        ]
        just_one_copy = self.inflation_level_per_source == 1
        self.ever_factorizes = False
        for sources_are_shared in shared_sources:
            if (not np.any(sources_are_shared)) or (not np.all(just_one_copy[sources_are_shared])):
                self.ever_factorizes = True
                break

        self._np_dtype = np.result_type(
            *[
                np.min_scalar_type(np.max(self.settings_per_party)),
                np.min_scalar_type(np.max(self.outcomes_per_party)),
                np.min_scalar_type(self.nr_parties + 1),
                np.min_scalar_type(np.max(self.inflation_level_per_source) + 1),
            ]
        )

        self.observable_nodes = list(self.names)
        self.source_nodes = self._actual_sources.tolist()
        self.intermediate_latent_nodes = _topological_sort_subset(
            self.intermediate_latents,
            self.dag,
        )
        self.all_nodes = _topological_sort_subset(
            set(self.observable_nodes)
            .union(self.source_nodes)
            .union(self.intermediate_latent_nodes),
            self.dag,
        )
        self.node_kind = {node: "observable" for node in self.observable_nodes}
        self.node_kind.update({node: "source" for node in self.source_nodes})
        self.node_kind.update({node: "latent" for node in self.intermediate_latent_nodes})

        self.parent_map = {
            node: tuple(
                parent
                for parent in self.all_nodes
                if parent in self._inverse_dag.get(node, set())
            )
            for node in self.all_nodes
        }
        self.children_map = {
            node: tuple(
                child for child in self.all_nodes if child in self.dag.get(node, set())
            )
            for node in self.all_nodes
        }
        self.source_index = {source: i for i, source in enumerate(self.source_nodes)}

        self.source_ancestry_by_node: Dict[str, np.ndarray] = {}
        for node in self.all_nodes:
            ancestry = np.zeros(self.nr_sources, dtype=bool)
            if node in self.source_index:
                ancestry[self.source_index[node]] = True
            else:
                for parent in self.parent_map[node]:
                    ancestry |= self.source_ancestry_by_node[parent]
            self.source_ancestry_by_node[node] = ancestry

        if self.inflation_mode == "manual":
            self._initialize_manual_inflation()
            return

        self.inflation_indices_per_party = []
        for party in range(self.nr_parties):
            inflation_indices = []
            active_sources = self.hypergraph[:, party]
            num_copies = np.multiply(active_sources, self.inflation_level_per_source)
            num_copies = np.maximum(num_copies, 1)
            for increase_from_base in np.ndindex(*num_copies):
                inflation_indxs = active_sources + np.array(
                    increase_from_base, dtype=self._np_dtype
                )
                inflation_indices.append(inflation_indxs)
            self.inflation_indices_per_party.append(np.vstack(inflation_indices))

        self._all_unique_inflation_indices = np.unique(
            np.vstack(self.inflation_indices_per_party), axis=0
        ).astype(self._np_dtype)
        self._inflation_indices_hash = {
            op.tobytes(): i for i, op in enumerate(self._all_unique_inflation_indices)
        }
        self._inflation_indices_overlap = _overlap_matrix(self._all_unique_inflation_indices)

        # Placeholder for later entropic-symmetry implementation. The reference
        # package computes permutations on operator lexorder; the entropic package
        # will compute permutations on entropy coordinates once those are defined.
        self.symmetries = np.arange(len(self._all_unique_inflation_indices), dtype=np.intc)[
            np.newaxis
        ]

        self.inflation_indices_per_node: Dict[str, np.ndarray] = {}
        for node in self.all_nodes:
            self.inflation_indices_per_node[node] = self._inflation_indices_for_mask(
                self.source_ancestry_by_node[node]
            )

        self.inflated_node_labels_by_base: Dict[str, List[str]] = {}
        self.inflated_node_lookup: Dict[str, tuple[str, tuple[int, ...]]] = {}
        self.inflated_node_parents: Dict[str, tuple[str, ...]] = {}
        inflated_children = defaultdict(list)
        for node in self.all_nodes:
            labels = []
            for copy_vec in self.inflation_indices_per_node[node]:
                copy_tuple = tuple(int(x) for x in copy_vec.tolist())
                label = _inflated_node_label(node, copy_tuple)
                labels.append(label)
                self.inflated_node_lookup[label] = (node, copy_tuple)
            self.inflated_node_labels_by_base[node] = labels

        for label, (node, copy_tuple) in self.inflated_node_lookup.items():
            parents = []
            for parent in self.parent_map[node]:
                parent_copy = self._project_copy_to_node(copy_tuple, parent)
                parent_label = _inflated_node_label(parent, parent_copy)
                parents.append(parent_label)
                inflated_children[parent_label].append(label)
            self.inflated_node_parents[label] = tuple(parents)

        self.inflated_node_children = {
            label: tuple(inflated_children.get(label, []))
            for label in self.inflated_node_lookup
        }
        self.inflated_source_nodes = [
            label
            for source in self.source_nodes
            for label in self.inflated_node_labels_by_base[source]
        ]
        self.inflated_latent_nodes = [
            label
            for latent in self.intermediate_latent_nodes
            for label in self.inflated_node_labels_by_base[latent]
        ]
        self.inflated_observable_nodes = [
            label
            for obs in self.observable_nodes
            for label in self.inflated_node_labels_by_base[obs]
        ]
        self._inflated_label_by_base_and_copy = {
            (base, copy_tuple): label
            for label, (base, copy_tuple) in self.inflated_node_lookup.items()
        }
        self.inflated_node_order = self.entropic_node_labels(include_latents=True)

    def _initialize_manual_inflation(self) -> None:
        manual_dag = self.manual_inflation_dag or {}
        manual_nodes = set(manual_dag.keys()).union(chain.from_iterable(manual_dag.values()))
        if not manual_nodes:
            raise ValueError("Manual inflation DAG must contain at least one node.")

        base_nodes = sorted(self.all_nodes, key=len, reverse=True)
        self.inflated_node_lookup = {}
        manual_base_by_node: Dict[str, str] = {}
        manual_token_by_node: Dict[str, str] = {}
        manual_prefix_by_node: Dict[str, str] = {}
        for node in sorted(manual_nodes):
            base, token = _infer_manual_base_and_token(node, base_nodes)
            manual_base_by_node[node] = base
            manual_token_by_node[node] = token
            manual_prefix_by_node[node] = node[: len(node) - len(token)] if token else node
            self.inflated_node_lookup[node] = (base, tuple())
        self._manual_base_by_node = manual_base_by_node
        self._manual_token_by_node = manual_token_by_node
        self._manual_prefix_by_node = manual_prefix_by_node

        token_sets = {source: set() for source in self.source_nodes}
        for node, base in manual_base_by_node.items():
            if base in token_sets:
                token_sets[base].add(manual_token_by_node[node] or "0")
        token_maps = {
            source: {
                token: idx + 1
                for idx, token in enumerate(sorted(tokens, key=_manual_token_sort_key))
            }
            for source, tokens in token_sets.items()
        }
        inferred_levels = [
            max(len(token_maps[source]), 1)
            for source in self.source_nodes
        ]
        self.inflation_level_per_source = np.asarray(inferred_levels, dtype=int)
        self._any_inflation = bool(np.any(self.inflation_level_per_source > 1))

        graph = nx.DiGraph()
        graph.add_nodes_from(sorted(manual_nodes))
        for parent, children in manual_dag.items():
            for child in children:
                graph.add_edge(parent, child)
        self._manual_inflation_graph = graph
        manual_order = list(nx.topological_sort(graph))

        self.inflated_node_labels_by_base = {node: [] for node in self.all_nodes}
        self.inflated_node_parents = {}
        inflated_children = defaultdict(list)
        incoming = defaultdict(list)
        for parent, children in manual_dag.items():
            for child in children:
                incoming[child].append(parent)
        manual_copy_tuples: Dict[str, tuple[int, ...]] = {}

        for node in manual_order:
            base = manual_base_by_node[node]
            parent_labels = tuple(incoming.get(node, ()))
            self.inflated_node_parents[node] = parent_labels
            for parent in parent_labels:
                inflated_children[parent].append(node)
            expected_parents = set(self.parent_map[base])
            actual_parents = {manual_base_by_node[parent] for parent in parent_labels}
            if actual_parents != expected_parents:
                raise ValueError(
                    f"Manual inflation parents for {node} do not match the base DAG parents of {base}."
                )

            if base in self.source_index:
                token = manual_token_by_node[node] or "0"
                copy_vec = [0] * self.nr_sources
                copy_vec[self.source_index[base]] = token_maps[base][token]
                manual_copy_tuples[node] = tuple(copy_vec)
            else:
                copy_vec = [0] * self.nr_sources
                for parent in parent_labels:
                    for idx, value in enumerate(manual_copy_tuples[parent]):
                        if not value:
                            continue
                        if copy_vec[idx] and copy_vec[idx] != value:
                            raise ValueError(
                                f"Manual inflation DAG gives conflicting source copies for {node}."
                            )
                        copy_vec[idx] = value
                manual_copy_tuples[node] = tuple(copy_vec)

            self.inflated_node_lookup[node] = (base, manual_copy_tuples[node])
            self.inflated_node_labels_by_base[base].append(node)

        self.inflated_node_children = {
            label: tuple(inflated_children.get(label, ()))
            for label in manual_order
        }
        self.inflated_source_nodes = [
            label for label in manual_order if manual_base_by_node[label] in self.source_nodes
        ]
        self.inflated_latent_nodes = [
            label
            for label in manual_order
            if manual_base_by_node[label] in self.intermediate_latent_nodes
        ]
        self.inflated_observable_nodes = [
            label for label in manual_order if manual_base_by_node[label] in self.observable_nodes
        ]
        self._inflated_label_by_base_and_copy = {
            (base, copy_tuple): label
            for label, (base, copy_tuple) in self.inflated_node_lookup.items()
        }
        self.inflated_node_order = self.entropic_node_labels(include_latents=True)
        self.inflation_indices_per_node = {
            base: np.vstack(
                [
                    np.asarray(self.inflated_node_lookup[label][1], dtype=self._np_dtype)
                    for label in self.inflated_node_labels_by_base[base]
                ]
            )
            for base in self.all_nodes
            if self.inflated_node_labels_by_base[base]
        }
        self._all_unique_inflation_indices = np.unique(
            np.vstack(
                [
                    np.asarray(copy_tuple, dtype=self._np_dtype)
                    for _, copy_tuple in self.inflated_node_lookup.values()
                ]
            ),
            axis=0,
        ).astype(self._np_dtype)
        self._inflation_indices_hash = {
            op.tobytes(): i for i, op in enumerate(self._all_unique_inflation_indices)
        }
        self._inflation_indices_overlap = _overlap_matrix(self._all_unique_inflation_indices)
        self.symmetries = np.arange(len(self._all_unique_inflation_indices), dtype=np.intc)[
            np.newaxis
        ]

    def __repr__(self) -> str:
        if self._classical_sources.all():
            source_info = "All sources are classical."
        elif self._nonclassical_sources.all():
            source_info = "All sources are quantum."
        else:
            classical_sources = self._actual_sources[self._classical_sources.astype(bool)]
            quantum_sources = self._actual_sources[self._nonclassical_sources.astype(bool)]
            source_info = ""
            if len(classical_sources):
                plural = "s" if len(classical_sources) > 1 else ""
                verb = "are" if len(classical_sources) > 1 else "is"
                source_info += f"Source{plural} " + ", ".join(classical_sources) + f" {verb} classical"
            if len(classical_sources) and len(quantum_sources):
                source_info += ", and "
            if len(quantum_sources):
                plural = "s" if len(quantum_sources) > 1 else ""
                verb = "are" if len(quantum_sources) > 1 else "is"
                source_info += f"source{plural} " + ", ".join(quantum_sources) + f" {verb} quantum"
            source_info += "."
        return (
            "InflationProblem with "
            + str(self.dag)
            + " as DAG, "
            + str(self.outcomes_per_party)
            + " outcomes per party, "
            + str(self.settings_per_party)
            + " settings per party and "
            + str(self.inflation_level_per_source)
            + " inflation copies per source. "
            + source_info
        )

    @cached_property
    def _any_inflation(self) -> bool:
        """Whether any source has inflation level greater than one."""
        return bool(np.any(self.inflation_level_per_source > 1))

    @cached_property
    def names_to_ints(self) -> dict:
        """Map party names to one-based integer identifiers."""
        return {name: i + 1 for i, name in enumerate(self.names)}

    @cached_property
    def entropic_observable_nodes(self) -> List[str]:
        """Inflated observable nodes used when ``include_latents=False``."""
        return self.entropic_node_labels(include_latents=False)

    @cached_property
    def entropic_all_nodes(self) -> List[str]:
        """Inflated source, latent, and observable nodes for entropic LPs."""
        return self.entropic_node_labels(include_latents=True)

    @cached_property
    def public_node_names(self) -> Dict[str, str]:
        """Readable names for inflated entropic nodes.

        The canonical internal labels stay unchanged, but the public API can
        use short names such as ``A``, ``a``, ``A_11`` or ``b_2``.
        """
        if self.inflation_mode == "manual":
            return {canonical: canonical for canonical in self.inflated_node_lookup}
        mapping: Dict[str, str] = {}
        any_nontrivial_inflation = bool(np.any(self.inflation_level_per_source > 1))
        for canonical, (base, copy_tuple) in self.inflated_node_lookup.items():
            active_digits = [str(v) for v in copy_tuple if v]
            if (not any_nontrivial_inflation) and all(v == 1 for v in copy_tuple if v):
                public = base
            elif not active_digits:
                public = base
            else:
                public = f"{base}_{''.join(active_digits)}"
            mapping[canonical] = public
        return mapping

    @cached_property
    def public_to_canonical_node_names(self) -> Dict[str, str]:
        return {public: canonical for canonical, public in self.public_node_names.items()}

    def public_node_labels(self, include_latents: bool = False) -> List[str]:
        """Readable node labels aligned with ``entropic_node_labels``."""
        return [
            self.public_node_names[label]
            for label in self.entropic_node_labels(include_latents=include_latents)
        ]

    def canonical_node_name(self, name: str) -> str:
        """Translate a readable node name to the internal canonical node label."""
        return self.public_to_canonical_node_names.get(name, name)

    def public_subset_label(self, spec: Sequence[str] | str) -> str:
        names = _split_public_spec(spec)
        canonical = [self.canonical_node_name(name) for name in names]
        readable = [self.public_node_names[name] for name in canonical]
        return ",".join(readable)

    def canonical_subset_label(self, spec: Sequence[str] | str) -> str:
        names = _split_public_spec(spec)
        canonical = [self.canonical_node_name(name) for name in names]
        order = {name: i for i, name in enumerate(self.entropic_all_nodes)}
        canonical = sorted(dict.fromkeys(canonical), key=lambda name: order[name])
        return ",".join(canonical)

    @cached_property
    def observable_factorization_equalities(self) -> List[tuple[tuple[str, ...], ...]]:
        """Inflation-style observable factorization blocks.

        Each entry is a tuple of disconnected observable blocks. It represents
        the additive entropy identity
        ``H(union blocks) = sum_i H(block_i)``.
        """
        labels = self.entropic_observable_nodes
        out: List[tuple[tuple[str, ...], ...]] = []
        seen = set()
        for subset_size in range(2, len(labels) + 1):
            for subset in combinations_with_replacement(labels, subset_size):
                if len(set(subset)) != len(subset):
                    continue
                blocks = self.factorization_blocks(subset)
                if len(blocks) <= 1:
                    continue
                if blocks in seen:
                    continue
                seen.add(blocks)
                out.append(blocks)
        return out

    @cached_property
    def observable_independencies(self) -> List[tuple[str, ...]]:
        """Observable-side entropic independences used by ``include_latents=False``.

        These are the pairwise independences induced by two-block observable
        factorizations. Higher-order disconnected factorizations appear in
        ``observable_factorization_equalities``.
        """
        out: List[tuple[str, ...]] = []
        for blocks in self.observable_factorization_equalities:
            if len(blocks) == 2 and len(blocks[0]) == 1 and len(blocks[1]) == 1:
                out.append((blocks[0][0], blocks[1][0]))
        return out

    @cached_property
    def latent_dsep_independencies(self) -> List[tuple[str, ...]]:
        """Markov independences on the full inflated DAG."""
        return self.inflated_independencies(include_latents=True)

    @cached_property
    def latent_markov_independencies(self) -> List[tuple[str, ...]]:
        """Alias for the full-DAG Markov independences."""
        return self.latent_dsep_independencies

    @cached_property
    def public_observable_independencies(self) -> List[tuple[str, ...]]:
        return [
            tuple(self.public_subset_label(part) for part in indep)
            for indep in self.observable_independencies
        ]

    @cached_property
    def public_latent_dsep_independencies(self) -> List[tuple[str, ...]]:
        return [
            tuple(self.public_subset_label(part) for part in indep)
            for indep in self.latent_dsep_independencies
        ]

    @cached_property
    def public_observable_factorization_equalities(self) -> List[tuple[tuple[str, ...], ...]]:
        return [
            tuple(tuple(self.public_node_names[label] for label in block) for block in blocks)
            for blocks in self.observable_factorization_equalities
        ]

    def _inflation_indices_for_mask(self, active_sources: np.ndarray) -> np.ndarray:
        active_sources = np.asarray(active_sources, dtype=self._np_dtype)
        inflation_indices = []
        num_copies = np.multiply(active_sources, self.inflation_level_per_source)
        num_copies = np.maximum(num_copies, 1)
        for increase_from_base in np.ndindex(*num_copies):
            inflation_indxs = active_sources + np.array(
                increase_from_base,
                dtype=self._np_dtype,
            )
            inflation_indices.append(inflation_indxs)
        return np.vstack(inflation_indices)

    def _project_copy_to_node(
        self,
        copy_tuple: tuple[int, ...],
        node: str,
    ) -> tuple[int, ...]:
        mask = self.source_ancestry_by_node[node]
        return tuple(int(value if active else 0) for value, active in zip(copy_tuple, mask))

    def entropic_node_labels(self, include_latents: bool = False) -> List[str]:
        """Return inflated node labels to be used as entropy variables."""
        labels = []
        if include_latents:
            groups = (
                self.source_nodes,
                self.intermediate_latent_nodes,
                self.observable_nodes,
            )
        else:
            groups = (self.observable_nodes,)
        for group in groups:
            for node in group:
                labels.extend(self.inflated_node_labels_by_base[node])
        return labels

    def inflation_permutation_generators(
        self,
        include_latents: bool = False,
    ) -> List[Dict[str, str]]:
        """Adjacent-transposition generators of the inflation symmetry group."""
        if self.inflation_mode == "manual":
            return self._manual_inflation_permutation_generators(include_latents=include_latents)
        labels = self.entropic_node_labels(include_latents=include_latents)
        generators: List[Dict[str, str]] = []
        for source_idx, level in enumerate(self.inflation_level_per_source.tolist()):
            for left in range(1, int(level)):
                right = left + 1
                mapping: Dict[str, str] = {}
                valid = True
                for label in labels:
                    node, copy_tuple = self.inflated_node_lookup[label]
                    value = copy_tuple[source_idx]
                    if value == 0:
                        mapping[label] = label
                        continue
                    new_copy = list(copy_tuple)
                    if value == left:
                        new_copy[source_idx] = right
                    elif value == right:
                        new_copy[source_idx] = left
                    image = self._inflated_label_by_base_and_copy.get((node, tuple(new_copy)))
                    if image is None:
                        valid = False
                        break
                    mapping[label] = image
                if valid and any(k != v for k, v in mapping.items()):
                    generators.append(mapping)
        return generators

    def entropic_subset_signature(self, labels: Sequence[str]) -> tuple:
        """Canonical signature of an entropic subset under inflation-copy relabeling.

        For manual inflations, entropy equalities should compare marginals whose
        ancestral inflated sub-DAGs are the same up to source-copy renaming.
        """
        labels = tuple(labels)
        if not labels:
            return tuple()

        if self.inflation_mode != "manual":
            return tuple(labels)

        order = {label: i for i, label in enumerate(self.inflated_node_order)}
        ancestor_set = set(labels)
        stack = list(labels)
        while stack:
            current = stack.pop()
            for parent in self.inflated_node_parents.get(current, ()):
                if parent not in ancestor_set:
                    ancestor_set.add(parent)
                    stack.append(parent)

        ancestor_nodes = tuple(sorted(ancestor_set, key=order.get))
        renamers = [dict() for _ in range(self.nr_sources)]
        normalized = {}
        for node in ancestor_nodes:
            base, copy_tuple = self.inflated_node_lookup[node]
            norm_copy = []
            for source_idx, value in enumerate(copy_tuple):
                if value == 0:
                    norm_copy.append(0)
                    continue
                relabel = renamers[source_idx]
                if value not in relabel:
                    relabel[value] = len(relabel) + 1
                norm_copy.append(relabel[value])
            normalized[node] = (self.node_kind[base], base, tuple(norm_copy))

        edges = []
        for child in ancestor_nodes:
            for parent in self.inflated_node_parents.get(child, ()):
                if parent in ancestor_set:
                    edges.append((normalized[parent], normalized[child]))

        targets = tuple(normalized[label] for label in sorted(labels, key=order.get))
        return targets, tuple(sorted(edges))

    def _manual_inflation_permutation_generators(
        self,
        *,
        include_latents: bool = False,
    ) -> List[Dict[str, str]]:
        labels = self.entropic_node_labels(include_latents=include_latents)
        all_labels = self.entropic_node_labels(include_latents=True)
        generators: List[Dict[str, str]] = []
        seen = set()

        for source_idx, level in enumerate(self.inflation_level_per_source.tolist()):
            for left in range(1, int(level)):
                right = left + 1
                mapping: Dict[str, str] = {}
                for label in all_labels:
                    base, copy_tuple = self.inflated_node_lookup[label]
                    value = copy_tuple[source_idx]
                    if value == 0 or value not in {left, right}:
                        mapping[label] = label
                        continue
                    new_copy = list(copy_tuple)
                    new_copy[source_idx] = right if value == left else left
                    image = self._inflated_label_by_base_and_copy.get((base, tuple(new_copy)))
                    if image is None:
                        continue
                    mapping[label] = image
                restricted = {label: mapping[label] for label in labels if label in mapping}
                key = tuple(sorted(restricted.items()))
                if key in seen or not any(k != v for k, v in restricted.items()):
                    continue
                seen.add(key)
                generators.append(restricted)
        return generators

    def inflated_independencies(self, include_latents: bool = False) -> List[tuple[str, ...]]:
        """Infer mutual-information equalities from the inflated structure.

        With latents included this imposes the local Markov independences of the
        explicit inflated DAG together with source-independence equalities.
        Without latents, the observable-side factorization equalities are handled
        separately by the LP layer.
        """

        labels = self.entropic_node_labels(include_latents=include_latents)
        order = {label: i for i, label in enumerate(labels)}
        independencies: List[tuple[str, ...]] = []

        if include_latents:
            graph = nx.DiGraph()
            graph.add_nodes_from(labels)
            for child, parents in self.inflated_node_parents.items():
                if child not in order:
                    continue
                for parent in parents:
                    if parent in order:
                        graph.add_edge(parent, child)

            descendants_cache = {
                node: nx.descendants(graph, node)
                for node in labels
            }

            for node in labels:
                parents = tuple(
                    sorted(
                        [parent for parent in self.inflated_node_parents.get(node, ()) if parent in order],
                        key=order.get,
                    )
                )
                nondesc_not_parents = tuple(
                    sorted(
                        [
                            other
                            for other in labels
                            if other != node
                            and other not in parents
                            and other not in descendants_cache[node]
                        ],
                        key=order.get,
                    )
                )
                if nondesc_not_parents:
                    if parents:
                        independencies.append(
                            (
                                node,
                                ",".join(nondesc_not_parents),
                                ",".join(parents),
                            )
                        )
                    else:
                        independencies.append((node, ",".join(nondesc_not_parents)))
            return independencies

        return independencies

    def _inflated_descendants(
        self,
        label: str,
        *,
        allowed: dict[str, int],
    ) -> set[str]:
        seen: set[str] = set()
        stack = list(reversed(self.inflated_node_children.get(label, ())))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            if current in allowed:
                seen.add(current)
            stack.extend(reversed(self.inflated_node_children.get(current, ())))
        return seen

    def factorization_blocks(self, labels: Sequence[str]) -> tuple[tuple[str, ...], ...]:
        """Return disconnected inflation-support components for observable labels."""
        if len(labels) <= 1:
            return (tuple(labels),)
        positions = np.array(
            [
                self._inflation_indices_hash[
                    np.asarray(self.inflated_node_lookup[label][1], dtype=self._np_dtype).tobytes()
                ]
                for label in labels
            ],
            dtype=int,
        )
        subgraph = self._inflation_indices_overlap[np.ix_(positions, positions)]
        components = _connected_components(subgraph)
        return tuple(
            tuple(labels[index] for index in component)
            for component in components
        )


def _topological_sort_subset(nodes: set[str] | List[str] | tuple[str, ...], dag: Dict[str, set[str]]) -> List[str]:
    node_set = set(nodes)
    indegree = {node: 0 for node in node_set}
    for parent, children in dag.items():
        if parent not in node_set:
            continue
        for child in children:
            if child in node_set:
                indegree[child] += 1
    ready = sorted(node for node, deg in indegree.items() if deg == 0)
    ordered: List[str] = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for child in sorted(dag.get(node, set())):
            if child not in indegree:
                continue
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    remaining = [node for node in node_set if node not in ordered]
    return ordered + sorted(remaining)


def _inflated_node_label(base: str, copy_tuple: tuple[int, ...]) -> str:
    return f"{base}[{','.join(map(str, copy_tuple))}]"


def _infer_manual_base_and_token(node: str, base_nodes: Sequence[str]) -> tuple[str, str]:
    for base in base_nodes:
        if node == base:
            return base, ""
        if not node.startswith(base):
            continue
        suffix = node[len(base) :]
        if suffix.startswith("_"):
            suffix = suffix[1:]
        if suffix and suffix.replace("_", "").isalnum():
            return base, suffix
    raise ValueError(f"Could not infer a base variable name from manual inflation node {node!r}.")


def _manual_token_sort_key(token: str) -> tuple[int, object]:
    if token.isdigit():
        return (0, int(token))
    return (1, token)


def _split_public_spec(spec: Sequence[str] | str) -> List[str]:
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


def _overlap_matrix(inflation_indices: np.ndarray) -> np.ndarray:
    n = len(inflation_indices)
    out = np.zeros((n, n), dtype=bool)
    for i in range(n):
        out[i, i] = True
        for j in range(i + 1, n):
            overlap = bool(
                np.any(
                    (inflation_indices[i] > 0)
                    & (inflation_indices[j] > 0)
                    & (inflation_indices[i] == inflation_indices[j])
                )
            )
            out[i, j] = overlap
            out[j, i] = overlap
    return out


def _connected_components(adj_mat: np.ndarray) -> List[List[int]]:
    n = adj_mat.shape[0]
    seen = [False] * n
    components: List[List[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: List[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in np.flatnonzero(adj_mat[node]).tolist():
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components
