"""Inspect the manual spiral setup used by the observed-only examples.

This script does not solve an LP. It shows the ingredients of the manual-mode
construction:

1. the inflated node labels,
2. the observed entropy coordinates,
3. the hand-added equalities used by the old observed-only spiral LP.
"""

from __future__ import annotations

from entropic_inflation import (
    InflationLP,
    TRIANGLE_SPIRAL_OBSERVED_NODES,
    triangle_spiral_equalities,
    triangle_spiral_problem,
)


if __name__ == "__main__":
    spiral = triangle_spiral_problem()
    lp = InflationLP(spiral, include_latents=False)
    equalities = triangle_spiral_equalities()
    lp.set_extra_equalities(equalities)

    print("manual inflation mode:", spiral.inflation_mode)
    print("inflated nodes (all):", spiral.public_node_labels(include_latents=True))
    print("observed LP nodes (internal order):", spiral.public_node_labels(include_latents=False))
    print("observed nodes (historical order):", TRIANGLE_SPIRAL_OBSERVED_NODES)
    print("number of hand-added spiral equalities:", len(equalities))
    print("first equality:", equalities[0])
    print("entropy coordinates:", lp.entropy_labels)
