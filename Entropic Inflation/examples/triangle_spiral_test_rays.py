"""Test the historical 10 candidate rays in the observed-only spiral LP.

This matches the old setup where the spiral equalities were added by hand and
the LP did not include latent entropy coordinates.
"""

from __future__ import annotations

from entropic_inflation import (
    InflationLP,
    triangle_spiral_candidate_rays,
    triangle_spiral_equalities,
    triangle_spiral_problem,
)


if __name__ == "__main__":
    spiral = triangle_spiral_problem()
    equalities = triangle_spiral_equalities()
    rays = triangle_spiral_candidate_rays()

    feasible = []
    infeasible = []

    for item in rays:
        # Each ray fixes the seven observed coordinates
        # [H(A0), H(B0), H(C0), H(A0,B0), H(A0,C0), H(B0,C0), H(A0,B0,C0)].
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
