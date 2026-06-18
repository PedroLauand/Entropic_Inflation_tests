"""Search the measurement parameters for the smallest Eq. (4) slack.

We fix the maximally entangled qudit sources and a single (cyclic-symmetric)
projective measurement, block-diagonal in the diagonal / upper / lower sectors.
Each block carries a free real rotation SO(m) (Givens angles); the whole vector
of angles is the search variable. The objective is the spiral inequality slack

    slack = 7[H(AB)+H(AC)+H(BC)] - 8[H(A)+H(B)+H(C)] - 5 H(ABC),

which is >= 0 for every classical triangle distribution. We minimise it: a
negative value would be a quantum entropic violation of Eq. (4).

At d=2 there is a single angle and it recovers the qubit family (the diagonal
'u'); the off-diagonal blocks are then 1-dimensional and frozen.

Note: the search uses *real* SO(m) rotations, so its family does not contain the
*complex* DFT measurement of ``renou_offdiagonal_distribution``; the DFT slack is
reported only as a reference point, not as a member of the optimised family.

Run from the package root:

    PYTHONPATH=. python examples/quantum_nonclassicality/triangle_renou_optimize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.optimize import differential_evolution, minimize

from triangle_probability import (
    maximally_entangled_state,
    stack_povm,
    swap_povm_qubits,
    triangle_probability_distribution,
)
from triangle_token_counting import _givens_orthogonal, renou_offdiagonal_distribution
from entropic_vector import triangle_entropic_vector

# Eq. (4) coefficients in the order [H(A),H(B),H(C),H(AB),H(AC),H(BC),H(ABC)].
SPIRAL_COEFFS = np.array([-8, -8, -8, 7, 7, 7, -5], dtype=float)


def _block_indices(d: int) -> tuple[list[int], list[int], list[int]]:
    diag = [k * d + k for k in range(d)]
    upper = [i * d + j for i in range(d) for j in range(d) if i < j]
    lower = [i * d + j for i in range(d) for j in range(d) if i > j]
    return diag, upper, lower


def angle_counts(d: int) -> tuple[int, int, int]:
    """(#diag, #upper, #lower) Givens angles for the three blocks."""
    diag, upper, lower = _block_indices(d)
    so = lambda m: m * (m - 1) // 2
    return so(len(diag)), so(len(upper)), so(len(lower))


def n_params(d: int) -> int:
    return sum(angle_counts(d))


def parametrized_measurement(d: int, theta: np.ndarray) -> np.ndarray:
    """Block-diagonal projective measurement from a flat angle vector."""
    diag, upper, lower = _block_indices(d)
    nd, nu, nl = angle_counts(d)
    groups = [(diag, theta[:nd]), (upper, theta[nd:nd + nu]), (lower, theta[nd + nu:])]
    kets: list[np.ndarray] = []
    for block, ang in groups:
        m = len(block)
        O = _givens_orthogonal(m, ang) if m > 1 else np.eye(m)
        for r in range(m):
            ket = np.zeros(d * d, dtype=complex)
            for s, idx in enumerate(block):
                ket[idx] = O[s, r]
            kets.append(ket)
    return stack_povm([np.outer(k, k.conj()) for k in kets], d, d)


def slack_of(d: int, theta: np.ndarray, source: np.ndarray) -> float:
    M = parametrized_measurement(d, theta)
    p = triangle_probability_distribution(source, source, source, M, swap_povm_qubits(M), M)
    H = triangle_entropic_vector(p, label_suffix="")
    h = np.array([H["A"], H["B"], H["C"], H["A,B"], H["A,C"], H["B,C"], H["A,B,C"]])
    return float(SPIRAL_COEFFS @ h)


def dft_baseline_slack(d: int) -> float:
    p = renou_offdiagonal_distribution(d)
    H = triangle_entropic_vector(p, label_suffix="")
    h = np.array([H["A"], H["B"], H["C"], H["A,B"], H["A,C"], H["B,C"], H["A,B,C"]])
    return float(SPIRAL_COEFFS @ h)


def _scan_1d(d: int, source: np.ndarray, n: int = 600) -> tuple[float, np.ndarray]:
    """Exact 1-parameter scan (used for d=2: the single diagonal 'u' angle)."""
    best, best_x = np.inf, None
    for t in np.linspace(0.0, np.pi, n):
        s = slack_of(d, np.array([t]), source)
        if s < best:
            best, best_x = s, np.array([t])
    return best, best_x


def _random_search(d: int, source: np.ndarray, n_samples: int, seed: int) -> tuple[float, np.ndarray]:
    """Random angle sampling followed by a local Powell polish on the best."""
    rng = np.random.default_rng(seed)
    npar = n_params(d)
    best, best_x = np.inf, None
    for _ in range(n_samples):
        th = rng.uniform(0.0, np.pi, npar)
        s = slack_of(d, th, source)
        if s < best:
            best, best_x = s, th
    res = minimize(lambda th: slack_of(d, th, source), best_x, method="Powell",
                   bounds=[(0.0, np.pi)] * npar)
    if res.fun < best:
        best, best_x = float(res.fun), res.x
    return best, best_x


def optimise(d: int, *, method: str, seed: int = 7, **kw) -> dict:
    source = maximally_entangled_state(d)
    npar = n_params(d)
    if npar == 0:
        best, theta = dft_baseline_slack(d), np.array([])
    elif method == "scan":
        best, theta = _scan_1d(d, source, n=kw.get("n", 600))
    elif method == "de":
        res = differential_evolution(
            lambda th: slack_of(d, th, source), bounds=[(0.0, np.pi)] * npar,
            maxiter=kw.get("maxiter", 60), popsize=kw.get("popsize", 12),
            seed=seed, polish=True, tol=1e-7,
        )
        best, theta = float(res.fun), res.x
    elif method == "random":
        best, theta = _random_search(d, source, kw.get("n_samples", 800), seed)
    else:
        raise ValueError(method)
    return {"d": d, "n_params": npar, "best": best, "dft": dft_baseline_slack(d), "theta": theta}


if __name__ == "__main__":
    print("Minimising the Eq. (4) slack over the measurement angles")
    print("(maximally entangled source; cyclic-symmetric block-diagonal measurement)")
    print("=" * 80)
    print(f"{'d':>3}{'#angles':>9}{'method':>10}{'DFT slack':>13}{'best slack found':>18}{'violation?':>12}")
    print("-" * 80)
    plan = [(2, "scan", {}), (3, "de", {}), (4, "random", {"n_samples": 800})]
    for d, method, kw in plan:
        out = optimise(d, method=method, **kw)
        viol = "YES" if out["best"] < -1e-6 else "no"
        print(f"{d:>3}{out['n_params']:>9}{method:>10}{out['dft']:>+13.4f}"
              f"{out['best']:>+18.4f}{viol:>12}", flush=True)
    print("-" * 80)
    print("slack >= 0: Eq. (4) satisfied.  < 0: quantum entropic violation.")
    print("d=4 uses random search + local polish (not exhaustive).")
