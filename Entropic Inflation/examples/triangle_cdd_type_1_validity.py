"""Validity of the first triangle CDD inequality family.

This example mirrors the style of the ``inflation`` package examples:

1. declare the scenario with ``InflationProblem``,
2. build the entropic LP with ``InflationLP``,
3. set the inequality as an objective,
4. solve and report the certified lower bound.

The inequality is written in certificate form, namely ``expression >= 0``.
Validity is proved by minimizing the expression over the latent-inclusive
triangle entropic LP and checking that the optimum is nonnegative.
"""

from entropic_inflation import InflationLP, triangle_cdd_representatives, triangle_problem


if __name__ == "__main__":
    triangle = triangle_problem()
    inequality = triangle_cdd_representatives(triangle)[0]

    lp = InflationLP(triangle, include_latents=True)
    lp.set_objective(inequality["certificate"], direction="min", name=inequality["name"])
    result = lp.solve_result()

    print("triangle nodes:", triangle.public_node_labels(include_latents=True))
    print("inequality:", inequality["description"])
    print("certificate:", inequality["certificate"])
    print("problem_status:", result.problem_status)
    print("solution_status:", result.solution_status)
    print("minimum_value:", result.objective_value)
    print("valid:", result.is_feasible and result.objective_value is not None and result.objective_value >= -1e-9)
