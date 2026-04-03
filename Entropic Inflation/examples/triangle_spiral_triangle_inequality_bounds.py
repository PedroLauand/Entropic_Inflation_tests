"""Compute all seven triangle-inequality lower bounds in the observed-only spiral LP.

This example loops over the full triangle CDD family:

- 3 inequalities of type 1
- 1 inequality of type 2
- 3 inequalities of type 3

1. build the manual spiral inflation,
2. keep only observed entropy coordinates,
3. add the historical hand-written spiral equalities,
4. minimize each triangle inequality in turn.
"""

from __future__ import annotations

from entropic_inflation import (
    InflationLP,
    triangle_cdd_inequalities,
    triangle_spiral_equalities,
    triangle_spiral_problem,
)


def spiral_marginal_certificate(certificate: dict[str, float]) -> dict[str, float]:
    """Translate triangle labels A,B,C onto the observed spiral marginal A0,B0,C0."""
    name_map = {"A": "A0", "B": "B0", "C": "C0"}
    return {
        ",".join(name_map[token] for token in subset.split(",")): value
        for subset, value in certificate.items()
    }


def solve_bound(name: str, description: str, certificate: dict[str, float]) -> dict[str, object]:
    """Minimize one triangle inequality over the observed-only spiral LP."""
    lp = InflationLP(triangle_spiral_problem(), include_latents=False)
    lp.set_extra_equalities(triangle_spiral_equalities())
    lp.set_objective(
        spiral_marginal_certificate(certificate),
        direction="min",
        name=name,
    )
    result = lp.solve_result()
    return {
        "name": name,
        "description": description,
        "problem_status": result.problem_status,
        "solution_status": result.solution_status,
        "minimum_value": result.objective_value,
        "valid": (
            result.is_feasible
            and result.objective_value is not None
            and result.objective_value >= -1e-9
        ),
    }


if __name__ == "__main__":
    spiral = triangle_spiral_problem()
    equalities = triangle_spiral_equalities()
    results = [
        solve_bound(item["name"], item["description"], item["certificate"])
        for item in triangle_cdd_inequalities()
    ]

    print("scenario:", "triangle spiral")
    print("include_latents:", False)
    print("observed nodes:", spiral.public_node_labels(include_latents=False))
    print("hand_added_equalities:", len(equalities))

    for item in results:
        print()
        print("inequality:", item["name"])
        print("description:", item["description"])
        print("problem_status:", item["problem_status"])
        print("solution_status:", item["solution_status"])
        print("minimum_value:", item["minimum_value"])
        print("valid:", item["valid"])
