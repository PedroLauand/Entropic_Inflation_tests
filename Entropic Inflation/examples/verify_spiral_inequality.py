"""Cross-check the spiral inequality, Eq. (4) of the notes, without the LP.

The spiral-inflation LP derives the fully symmetric witness

    7[H(AB) + H(AC) + H(BC)] >= 8[H(A) + H(B) + H(C)] + 5 H(ABC).      (4)

This script does three independent sanity checks that do *not* call MOSEK:

1. every one of the ten triangle candidate rays satisfies the seven known
   triangle (Chaves-Luft-Gross) inequalities;
2. Eq. (4) is violated exactly by rays 1, 2, 3 (the single cyclic orbit the
   spiral inflation excludes) and satisfied by the rest;
3. Eq. (4) is *not* a non-negative combination of the seven known triangle
   inequalities plus the elemental Shannon inequalities -- i.e. it is a
   genuinely new facet -- proved by the infeasibility of a small decomposition
   LP solved with SciPy.

The rays and the seven known inequalities are pulled straight from the package
so this stays in sync with the rest of the library; only Eq. (4) itself (the
claim under test) is written out explicitly here.

Run from the package root:

    PYTHONPATH=. python examples/verify_spiral_inequality.py
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import linprog

from entropic_inflation import (
    triangle_cdd_inequalities,
    triangle_spiral_candidate_rays,
)

# Observed-entropy basis used throughout: index i <-> coordinate below.
BASIS = ("A", "B", "C", "A,B", "A,C", "B,C", "A,B,C")
IDX = {name: i for i, name in enumerate(BASIS)}


def _coeff_vector(certificate: dict[str, float]) -> np.ndarray:
    """Turn a ``{subset: coefficient}`` certificate into a 7-vector."""
    v = np.zeros(7)
    for subset, coef in certificate.items():
        v[IDX[subset]] = coef
    return v


def _ray_vector(values: dict[str, float]) -> np.ndarray:
    """Turn a candidate-ray dict (labels ``A0``, ``A0,B0``, ...) into a 7-vector."""
    v = np.zeros(7)
    for label, value in values.items():
        # Strip the spiral copy index ('0') to land back in the A/B/C basis.
        key = ",".join(token[0] for token in label.split(","))
        v[IDX[key]] = value
    return v


# Eq. (4) as  coeffs . H >= 0  (the inequality under test).
SPIRAL = _coeff_vector(
    {"A": -8, "B": -8, "C": -8, "A,B": 7, "A,C": 7, "B,C": 7, "A,B,C": -5}
)


def elemental_shannon_rows() -> np.ndarray:
    """The nine elemental Shannon inequalities for three variables, as rows."""
    full = {0, 1, 2}
    table = {
        frozenset([0]): "A", frozenset([1]): "B", frozenset([2]): "C",
        frozenset([0, 1]): "A,B", frozenset([0, 2]): "A,C", frozenset([1, 2]): "B,C",
        frozenset([0, 1, 2]): "A,B,C",
    }
    rows = []
    # H(i | rest) = H(all) - H(rest) >= 0
    for i in range(3):
        row = np.zeros(7)
        row[IDX[table[frozenset(full)]]] += 1
        row[IDX[table[frozenset(full - {i})]]] -= 1
        rows.append(row)
    # I(i ; j | S) = H(iS) + H(jS) - H(ijS) - H(S) >= 0
    for i, j in combinations(range(3), 2):
        others = list(full - {i, j})
        for k in range(len(others) + 1):
            for S in combinations(others, k):
                S = set(S)
                row = np.zeros(7)
                row[IDX[table[frozenset({i} | S)]]] += 1
                row[IDX[table[frozenset({j} | S)]]] += 1
                row[IDX[table[frozenset({i, j} | S)]]] -= 1
                if S:
                    row[IDX[table[frozenset(S)]]] -= 1
                rows.append(row)
    return np.array(rows)


def main() -> None:
    rays = {item["index"]: _ray_vector(item["values"]) for item in triangle_spiral_candidate_rays()}
    triangle = {item["name"]: _coeff_vector(item["certificate"]) for item in triangle_cdd_inequalities()}

    print("Step 1 -- each ray vs the seven known triangle inequalities (want >= 0):")
    for idx, r in rays.items():
        worst = min(float(t @ r) for t in triangle.values())
        print(f"  ray {idx}: min slack = {worst:+.3f}  {'OK' if worst >= -1e-9 else 'VIOLATES'}")

    print("\nStep 2 -- each ray vs Eq. (4)  (value = LHS - RHS):")
    for idx, r in rays.items():
        val = float(SPIRAL @ r)
        tag = "VIOLATES (excluded)" if val < -1e-9 else ("tight" if abs(val) < 1e-9 else "satisfied")
        print(f"  ray {idx}: {val:+.3f}   {tag}")

    print("\nStep 3 -- is Eq. (4) a non-negative combination of the known ineqs?")
    gens = np.vstack([np.array(list(triangle.values())), elemental_shannon_rows()])
    res = linprog(
        c=np.zeros(gens.shape[0]),
        A_eq=gens.T,
        b_eq=SPIRAL,
        bounds=[(0, None)] * gens.shape[0],
        method="highs",
    )
    if res.status == 0:
        print("  decomposition FEASIBLE -> Eq. (4) is implied (NOT new)")
    else:
        print("  decomposition infeasible -> Eq. (4) is a strictly NEW facet")


if __name__ == "__main__":
    main()
