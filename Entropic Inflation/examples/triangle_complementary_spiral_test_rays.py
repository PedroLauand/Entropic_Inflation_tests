"""Test the historical 10 candidate rays in the complementary spiral LP.

This is the observed-only comparison for the complementary winding direction.
"""

from __future__ import annotations

from entropic_inflation import (
    InflationLP,
    triangle_complementary_spiral_equalities,
    triangle_complementary_spiral_problem,
    triangle_spiral_candidate_rays,
)


if __name__ == "__main__":
    spiral = triangle_complementary_spiral_problem()
    equalities = triangle_complementary_spiral_equalities()
    rays = triangle_spiral_candidate_rays()

    feasible = []
    infeasible = []

    for item in rays:
        # Each ray fixes the seven observed coordinates on the A0,B0,C0 marginal.
        lp = InflationLP(spiral, include_latents=False)
        lp.set_extra_equalities(equalities)
        lp.set_values(item["values"])
        result = lp.solve_result()

        status = "feasible" if result.is_feasible else "infeasible"
        print(
            f"ray {item['index']}: {status} | "
            f"problem_status={result.problem_status} | "
            f"solution_status={result.solution_status}"
        )

        if result.is_feasible:
            feasible.append(item["index"])
        else:
            infeasible.append(item["index"])

    print("feasible rays:", feasible)
    print("infeasible rays:", infeasible)
