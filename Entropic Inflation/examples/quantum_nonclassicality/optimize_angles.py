"""Search the measurement angles for the smallest spiral-inequality Eq. (4) slack.

For a fixed family ("TC" or "CM") and source dimension, the measurement is the
block-diagonal real-rotation POVM of :mod:`coherent_strategies`; the free Givens
angles of every multi-dimensional sector are the search variables.  We minimise

    slack = 7[H(AB)+H(AC)+H(BC)] - 8[H(A)+H(B)+H(C)] - 5 H(ABC),

which is >= 0 for every classical triangle distribution; a negative value would
be a quantum entropic violation of Eq. (4).  The source is fixed by the family
(uniform tokens / maximally entangled), so only the measurement is optimised.

Run from the package root:

    PYTHONPATH=. python examples/quantum_nonclassicality/optimize_angles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.optimize import differential_evolution, minimize

import coherent_strategies as cs

_BUILD = {"TC": cs.token_counting, "CM": cs.color_matching}


def _rotatable(family: str, dim: int) -> list[tuple[object, int]]:
    """Ordered ``(label, n_angles)`` for the rotatable sectors (``n_angles > 0``)."""
    return [(lab, na) for lab, _, na in cs.angle_layout(family, dim) if na > 0]


def _unpack(family: str, dim: int, theta: np.ndarray) -> dict[object, np.ndarray]:
    """Flat angle vector -> the ``angles`` dict keyed by sector label."""
    angles, i = {}, 0
    for lab, na in _rotatable(family, dim):
        angles[lab] = theta[i : i + na]
        i += na
    return angles


def n_params(family: str, dim: int) -> int:
    """Number of free measurement angles for the family at this dimension."""
    return cs.n_angles(family, dim)


def slack_of(family: str, dim: int, theta: np.ndarray) -> float:
    """Eq. (4) slack of the family's distribution at a flat angle vector."""
    return cs.eq4_slack(_BUILD[family](dim, _unpack(family, dim, theta)))


def optimise(family: str, dim: int, *, method: str = "de", seed: int = 7, **kw) -> dict:
    """Minimise the Eq. (4) slack over the measurement angles.

    ``method``: ``"scan"`` (exact 1-D, only when there is a single angle),
    ``"de"`` (differential evolution), or ``"random"`` (random sampling + Powell
    polish).  Returns ``{family, dim, n_params, best, theta}``.
    """
    npar = n_params(family, dim)
    if npar == 0:  # nothing to rotate -> the (decohered) baseline is forced
        return {"family": family, "dim": dim, "n_params": 0,
                "best": cs.eq4_slack(_BUILD[family](dim)), "theta": np.array([])}

    f = lambda th: slack_of(family, dim, th)
    bounds = [(0.0, np.pi)] * npar

    if method == "scan" and npar == 1:
        ts = np.linspace(0.0, np.pi, kw.get("n", 600))
        vals = [f(np.array([t])) for t in ts]
        i = int(np.argmin(vals))
        best, theta = float(vals[i]), np.array([ts[i]])
    elif method == "de":
        res = differential_evolution(f, bounds, maxiter=kw.get("maxiter", 60),
                                     popsize=kw.get("popsize", 12), seed=seed,
                                     polish=True, tol=1e-7)
        best, theta = float(res.fun), res.x
    elif method == "random":
        rng = np.random.default_rng(seed)
        best, theta = np.inf, None
        for _ in range(kw.get("n_samples", 800)):
            th = rng.uniform(0.0, np.pi, npar)
            s = f(th)
            if s < best:
                best, theta = s, th
        res = minimize(f, theta, method="Powell", bounds=bounds)
        if res.fun < best:
            best, theta = float(res.fun), res.x
    else:
        raise ValueError(f"unknown method {method!r} (or scan with npar != 1).")
    return {"family": family, "dim": dim, "n_params": npar, "best": best, "theta": theta}


if __name__ == "__main__":
    print("Minimising the Eq. (4) slack over the measurement angles")
    print("(coherent_strategies measurements; real block-diagonal rotations)")
    print("=" * 78)
    print(f"{'family':>7}{'dim':>5}{'#angles':>9}{'method':>9}"
          f"{'decohered':>12}{'best slack':>13}{'violation?':>12}")
    print("-" * 78)
    plan = [
        ("TC", 2, "scan", {}),
        ("TC", 3, "de", {}),
        ("CM", 3, "de", {}),
    ]
    for family, dim, method, kw in plan:
        out = optimise(family, dim, method=method, **kw)
        base = cs.eq4_slack(_BUILD[family](dim))  # angles = None
        viol = "YES" if out["best"] < -1e-6 else "no"
        print(f"{family:>7}{dim:>5}{out['n_params']:>9}{method:>9}"
              f"{base:>+12.4f}{out['best']:>+13.4f}{viol:>12}", flush=True)
    print("-" * 78)
    print("slack >= 0: Eq. (4) satisfied.  < 0: quantum entropic violation.")
    print("(The maximally-entangled / cyclic-symmetric / real-projective family")
    print(" does not violate Eq. (4): the minimum is the decohered facet at 0.)")
