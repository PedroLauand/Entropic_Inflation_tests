"""Entropic vector of the triangle from a joint probability tensor.

Given a probability tensor ``p[a, b, c]`` over the three observed
outcomes of the triangle network, this module computes the seven
observed Shannon entropies

.. math::
    H(A),\\ H(B),\\ H(C),\\ H(A,B),\\ H(A,C),\\ H(B,C),\\ H(A,B,C)

and returns them in the dictionary format expected by
``InflationLP(triangle_spiral_problem()).set_values``, ready to be used
as an entropic candidate for the spiral-inflation feasibility test.

It also provides a cheap pre-screen that evaluates the seven known
triangle Shannon-type inequalities and the symmetric spiral-derived
inequality without invoking the LP.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Shannon entropy and the entropic vector of the triangle.
# ---------------------------------------------------------------------------


def shannon_entropy(p: np.ndarray, *, base: float = 2.0) -> float:
    """Shannon entropy of a probability tensor of arbitrary rank.

    Uses base-2 logarithm by default (units: bits). Zero-probability
    outcomes are dropped (``0 * log 0 = 0``).
    """
    flat = np.asarray(p, dtype=float).ravel()
    nonzero = flat[flat > 0.0]
    return float(-np.sum(nonzero * (np.log(nonzero) / np.log(base))))


def triangle_entropic_vector(
    p: np.ndarray,
    *,
    label_suffix: str = "0",
) -> dict[str, float]:
    """Return the seven observed entropies of the triangle as a dict.

    The labels are ``A{s}``, ``B{s}``, ``C{s}``, ``A{s},B{s}``,
    ``A{s},C{s}``, ``B{s},C{s}``, ``A{s},B{s},C{s}`` with
    ``{s} = label_suffix``. The default matches the observed node names of
    ``triangle_spiral_problem()`` so the output can be passed directly to
    ``InflationLP.set_values``.
    """
    if p.ndim != 3:
        raise ValueError(
            f"Expected a 3-index probability tensor, got shape {p.shape}."
        )
    s = label_suffix
    return {
        f"A{s}": shannon_entropy(p.sum(axis=(1, 2))),
        f"B{s}": shannon_entropy(p.sum(axis=(0, 2))),
        f"C{s}": shannon_entropy(p.sum(axis=(0, 1))),
        f"A{s},B{s}": shannon_entropy(p.sum(axis=2)),
        f"A{s},C{s}": shannon_entropy(p.sum(axis=1)),
        f"B{s},C{s}": shannon_entropy(p.sum(axis=0)),
        f"A{s},B{s},C{s}": shannon_entropy(p),
    }


# ---------------------------------------------------------------------------
# Cheap pre-screen against the known triangle Shannon-type inequalities.
# ---------------------------------------------------------------------------

# Each row encodes ``coeffs . H >= 0`` in the basis
# ``(H(A), H(B), H(C), H(A,B), H(A,C), H(B,C), H(A,B,C))``.
TRIANGLE_CDD_INEQUALITIES: Sequence[tuple[str, np.ndarray]] = (
    ("Type 1a", np.array([-1, -1, -1, 1, 1, 0, 0])),
    ("Type 1b", np.array([-1, -1, -1, 1, 0, 1, 0])),
    ("Type 1c", np.array([-1, -1, -1, 0, 1, 1, 0])),
    ("Type 2 ", np.array([-5, -5, -5, 4, 4, 4, -2])),
    ("Type 3a", np.array([-3, -3, -3, 3, 2, 2, -1])),
    ("Type 3b", np.array([-3, -3, -3, 2, 3, 2, -1])),
    ("Type 3c", np.array([-3, -3, -3, 2, 2, 3, -1])),
)

# The fully-symmetric spiral-derived inequality:
#   7 [H(AB)+H(AC)+H(BC)] >= 8 [H(A)+H(B)+H(C)] + 5 H(ABC).
SPIRAL_INEQUALITY: tuple[str, np.ndarray] = (
    "Spiral",
    np.array([-8, -8, -8, 7, 7, 7, -5]),
)


def _vector_in_order(H: Mapping[str, float], *, suffix: str) -> np.ndarray:
    s = suffix
    return np.array(
        [
            H[f"A{s}"], H[f"B{s}"], H[f"C{s}"],
            H[f"A{s},B{s}"], H[f"A{s},C{s}"], H[f"B{s},C{s}"],
            H[f"A{s},B{s},C{s}"],
        ],
        dtype=float,
    )


def check_triangle_inequalities(
    H: Mapping[str, float],
    *,
    label_suffix: str = "0",
    tol: float = 1e-9,
) -> list[tuple[str, float, bool]]:
    """Evaluate the seven CDD triangle inequalities and the spiral one.

    Returns a list of ``(name, slack, satisfied)`` triples, where
    ``slack = coeffs . H`` and ``satisfied = slack >= -tol``.
    """
    h = _vector_in_order(H, suffix=label_suffix)
    report = []
    for name, coeffs in (*TRIANGLE_CDD_INEQUALITIES, SPIRAL_INEQUALITY):
        slack = float(coeffs @ h)
        report.append((name, slack, slack >= -tol))
    return report


def spiral_slack(H: Mapping[str, float], *, label_suffix: str = "0") -> float:
    """Slack of the spiral inequality Eq. (4) for an entropy vector ``H``.

        slack = 7[H(AB)+H(AC)+H(BC)] - 8[H(A)+H(B)+H(C)] - 5 H(ABC)

    ``slack >= 0`` means Eq. (4) is satisfied; ``< 0`` is a violation. This is the
    single source of truth for the Eq. (4) coefficients used across the examples.
    """
    return float(SPIRAL_INEQUALITY[1] @ _vector_in_order(H, suffix=label_suffix))
