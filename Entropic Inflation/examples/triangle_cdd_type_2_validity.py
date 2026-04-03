"""Check validity of the second triangle CDD inequality family.

This is the same workflow as the first example, just with a different
representative certificate.
"""

from entropic_inflation import InflationLP, triangle_cdd_representatives, triangle_problem


if __name__ == "__main__":
    triangle = triangle_problem()
    inequality = triangle_cdd_representatives(triangle)[1]

    lp = InflationLP(triangle, include_latents=True)
    # Minimize the certificate expression. A nonnegative minimum proves validity.
    lp.set_objective(inequality["certificate"], direction="min", name=inequality["name"])
    result = lp.solve_result()

    print("triangle nodes:", triangle.public_node_labels(include_latents=True))
    print("inequality:", inequality["description"])
    print("certificate:", inequality["certificate"])
    print("problem_status:", result.problem_status)
    print("solution_status:", result.solution_status)
    print("minimum_value:", result.objective_value)
    print("valid:", result.is_feasible and result.objective_value is not None and result.objective_value >= -1e-6)
