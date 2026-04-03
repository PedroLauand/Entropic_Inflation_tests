"""Smoke tests for the Shannon application API."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shannon_tests.shannon_applications import (
    ShannonSourceModel,
    build_shannon_dag_lp,
    build_shannon_inflation_lp,
    entropy_label,
    solve_shannon_dag_lp,
    test_dag_candidates_in_cone,
    test_candidates_in_cone,
)
from shannon_tests.shannon_applications.triangle import (
    triangle_all_ones_candidate,
    triangle_array7_objectives,
    triangle_cdd_rays_with_array7,
)


class ShannonInflationAPITest(unittest.TestCase):
    def test_entropy_label_normalizes_order(self) -> None:
        label = entropy_label("B1,A0", ["A0", "A1", "B0", "B1"])
        self.assertEqual(label, "H(A0,B1)")

    def test_builds_problem_with_sparse_candidate_and_objective(self) -> None:
        problem = build_shannon_inflation_lp(
            variables=["A0", "A1", "B0", "B1"],
            basic_inequalities="elemental",
            independencies=[["A0", "B0"]],
            symmetries=[["A0", "A1"]],
            candidate={
                "A0": 1.0,
                "B0": 1.0,
                "A0,B0": 1.5,
            },
            objective={
                "A0": 1.0,
                "B1": 1.0,
                "A0,B1": -1.0,
            },
            return_matrix=True,
        )

        self.assertIn("H(A0)", problem.label_to_index)
        self.assertIn("H(A0,B1)", problem.label_to_index)
        self.assertEqual(
            problem.entropy_labels[problem.label_to_index["H(A0,B1)"]],
            "H(A0,B1)",
        )
        self.assertIsNotNone(problem.matrix)
        self.assertGreater(len(problem.matrix["M"]), 0)  # type: ignore[index]

    def test_builds_triangle_source_example_constraints(self) -> None:
        problem = build_shannon_inflation_lp(
            variables=["A", "B", "C", "a", "b", "c"],
            basic_inequalities="elemental",
            independencies=[
                ["A", "B,C,a", "b,c"],
                ["B", "A,C,b", "a,c"],
                ["C", "A,B,c", "a,b"],
            ],
            return_matrix=True,
        )

        self.assertIn("H(A,B,C,a,b,c)", problem.label_to_index)
        self.assertGreater(len(problem.matrix["M"]), 0)  # type: ignore[index]

    def test_dag_api_derives_triangle_source_constraints(self) -> None:
        triangle = ShannonSourceModel(
            dag={
                "a": ["B", "C"],
                "b": ["A", "C"],
                "c": ["A", "B"],
            },
        )

        self.assertEqual(triangle.observed, ["A", "B", "C"])
        self.assertEqual(triangle.sources, ["a", "b", "c"])
        self.assertEqual(triangle.variable_names, ["A", "B", "C", "a", "b", "c"])
        self.assertEqual(
            triangle.implied_independencies(),
            [
                ["A", "B,C,a", "b,c"],
                ["B", "A,C,b", "a,c"],
                ["C", "A,B,c", "a,b"],
                ["a", "b,c"],
                ["b", "a,c"],
                ["c", "a,b"],
            ],
        )

        problem = triangle.build_lp(
            basic_inequalities="elemental",
            return_matrix=True,
        )
        self.assertIn("H(A,B,C,a,b,c)", problem.label_to_index)
        self.assertGreater(len(problem.matrix["M"]), 0)  # type: ignore[index]

    def test_candidate_loop_classifies_feasible_and_infeasible_examples(self) -> None:
        results = test_candidates_in_cone(
            variables=["A", "B"],
            candidates=[
                {"A": 1.0, "B": 1.0, "A,B": 1.0},
                {"A": 1.0, "B": 1.0, "A,B": 0.0},
            ],
            basic_inequalities="elemental",
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].is_valid)
        self.assertFalse(results[1].is_valid)

    def test_dag_candidate_loop_classifies_simple_examples(self) -> None:
        results = test_dag_candidates_in_cone(
            dag={
                "s": ["A", "B"],
            },
            candidates=[
                {"A": 1.0, "B": 1.0, "A,B": 1.0},
                {"A": 1.0, "B": 1.0, "A,B": 2.5},
            ],
            basic_inequalities="elemental",
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].is_valid)
        self.assertFalse(results[1].is_valid)

    def test_triangle_helpers_return_expected_shapes(self) -> None:
        rays = triangle_cdd_rays_with_array7()
        self.assertGreater(len(rays), 0)
        self.assertIn("A,B,C", rays[0])

        ones = triangle_all_ones_candidate()
        self.assertEqual(set(ones.keys()), {"A", "B", "C", "A,B", "A,C", "B,C", "A,B,C"})

        array7 = triangle_array7_objectives()
        self.assertEqual(len(array7), 7)
        self.assertIn("objective", array7[0])

    def test_build_shannon_dag_lp_matches_object_api(self) -> None:
        problem = build_shannon_dag_lp(
            dag={
                "a": ["B", "C"],
                "b": ["A", "C"],
                "c": ["A", "B"],
            },
            basic_inequalities="elemental",
            candidate={"A": 1.0},
            return_matrix=True,
        )

        self.assertIn("H(A)", problem.label_to_index)
        self.assertGreater(len(problem.matrix["M"]), 0)  # type: ignore[index]

    def test_dag_solve_returns_infeasibility_and_farkas_certificate(self) -> None:
        triangle = ShannonSourceModel(
            dag={
                "a": ["B", "C"],
                "b": ["A", "C"],
                "c": ["A", "B"],
            }
        )

        result = triangle.solve(
            candidate=triangle_all_ones_candidate(),
            basic_inequalities="elemental",
            include_farkas_certificate=True,
        )

        self.assertFalse(result.is_feasible)
        self.assertIsNotNone(result.farkas_certificate)
        self.assertIsNotNone(result.farkas_certificate.expression)  # type: ignore[union-attr]

    def test_top_level_dag_solve_function_works(self) -> None:
        result = solve_shannon_dag_lp(
            dag={
                "a": ["B", "C"],
                "b": ["A", "C"],
                "c": ["A", "B"],
            },
            candidate={"A": 1.0},
            basic_inequalities="elemental",
        )

        self.assertTrue(result.is_feasible)


if __name__ == "__main__":
    unittest.main()
