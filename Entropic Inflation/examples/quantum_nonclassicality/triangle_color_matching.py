"""Color-Matching triangle distributions, as a function of the number of colours d.

One of two strategy families in this folder's generator of quantum triangle
distributions that are *nonlocal at the probability level* (see the folder
README); by rigidity, coherent members of this family admit no classical
(trilocal) model.

Color-Matching (CM), following Renou & Beigi [arXiv:2202.00905]: each source
carries one of ``d`` colours; a party checks whether the colours of its two
sources match. Quantum version: each source is the maximally entangled qudit

    |Phi_d> = (sum_c |c,c>) / sqrt(d)

so the two connected parties always receive the SAME colour. A party holds two
colour registers and measures:

* coarse CM (Remark 2 of arXiv:2202.00905): output the matched colour ``c`` if
  the two registers agree (the ``d`` diagonal states ``|c,c>``), otherwise a
  single "no-match" outcome (the rank ``d^2-d`` off-diagonal subspace). That is
  ``d + 1`` outcomes, and the distribution is classically simulable.

* refined CM: additionally measure the off-diagonal "no-match" subspace
  coherently (upper ``{|ij>: i<j}`` and lower ``{|ij>: i>j}`` blocks in a chosen
  basis) -> ``d^2`` outcomes. This is the construction that can be nonlocal; it
  is the same object as ``renou_offdiagonal_distribution`` in
  ``triangle_token_counting`` (the d>=3 Renou construction is really CM).

This module mirrors ``triangle_token_counting``: a dimension parameter, a
``describe_tuning(d)`` helper, and a ``__main__`` that scans d = 2, 3, 4 and
reports the entropy vector and the spiral inequality Eq. (4).

Quickstart (exploration)
------------------------
>>> p = color_matching_distribution(3)            # simplest CM strategy: coarse, 3 colours
>>> H = entropy_vector(p)                          # the 7 observed entropies
>>> eq4_slack(p)                                   # spiral inequality Eq. (4) slack
>>> color_matching_entropy_vector(4, refined=True)  # refined CM, in one call

Run from the package root:

    PYTHONPATH=. python examples/quantum_nonclassicality/triangle_color_matching.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from triangle_probability import (
    maximally_entangled_state,
    stack_povm,
    triangle_probability_distribution,
)
from triangle_token_counting import renou_offdiagonal_distribution
from entropic_vector import spiral_slack, triangle_entropic_vector


def color_matching_source(d: int) -> np.ndarray:
    """Maximally entangled qudit source |Phi_d> (both parties get the same colour)."""
    return maximally_entangled_state(d)


def coarse_color_matching_povm(d: int) -> np.ndarray:
    """Coarse CM measurement: d colour-match projectors + 1 'no-match' element."""
    mats = []
    no_match = np.eye(d * d, dtype=complex)
    for c in range(d):
        proj = np.zeros((d * d, d * d), dtype=complex)
        proj[c * d + c, c * d + c] = 1.0
        mats.append(proj)
        no_match = no_match - proj
    mats.append(no_match)  # rank d^2 - d "no-match" POVM element
    return stack_povm(mats, d, d)  # d + 1 outcomes


def color_matching_distribution(d: int, *, refined: bool = False) -> np.ndarray:
    """p(a,b,c) for the CM strategy with ``d`` colours.

    ``refined=False`` gives the coarse CM (d+1 outcomes, classically simulable);
    ``refined=True`` delegates to the off-diagonal (coherent) construction
    (d^2 outcomes), which can be nonlocal.
    """
    if refined:
        # Refined CM == the Renou off-diagonal construction (DFT off-diagonal basis).
        return renou_offdiagonal_distribution(d)
    # Coarse CM: match projectors + no-match are swap-symmetric, so no B-swap.
    rho = color_matching_source(d)
    M = coarse_color_matching_povm(d)
    return triangle_probability_distribution(rho, rho, rho, M, M, M)


def entropy_vector(p: np.ndarray) -> dict[str, float]:
    """The seven observed Shannon entropies of ``p(a,b,c)`` (plain A,B,C labels)."""
    return triangle_entropic_vector(p, label_suffix="")


def eq4_slack(p: np.ndarray) -> float:
    """Spiral inequality Eq. (4) slack for a distribution (>= 0 means satisfied)."""
    return spiral_slack(entropy_vector(p), label_suffix="")


def color_matching_entropy_vector(d: int, *, refined: bool = False) -> dict[str, float]:
    """Build the d-colour CM distribution and return its observed entropy vector."""
    return entropy_vector(color_matching_distribution(d, refined=refined))


def describe_tuning(d: int) -> None:
    """Print the source/measurement tuning available for the d-colour CM strategy."""
    m = d * (d - 1) // 2  # size of each off-diagonal (upper/lower) block
    off_angles = 2 * (m * (m - 1) // 2)
    print(f"d = {d} colours")
    print(f"  SOURCE     : maximally entangled |Phi_{d}> = (sum_c |cc>)/sqrt({d})  [fixed]")
    print(f"  coarse  CM : {d + 1} outcomes ({d} matched colours + 1 no-match); no free parameters")
    print(f"  refined CM : {d * d} outcomes ({d} matched colours + 2 off-diagonal blocks of size {m});")
    print(f"               free measurement angles: {off_angles}  (SO({m}) on each off-diagonal block)")


if __name__ == "__main__":
    print("Color-Matching tuning by dimension")
    print("=" * 72)
    for d in (2, 3, 4):
        describe_tuning(d)
        print()

    print("Entropy vector & Eq. (4) across colours and CM modes")
    print("=" * 72)
    header = f"{'d':>3}{'mode':>10}{'#out':>7}{'H(A)':>9}{'H(ABC)':>10}{'Eq4 slack':>12}{'slack/H(ABC)':>14}"
    print(header)
    print("-" * 72)
    for d in (2, 3, 4):
        for refined in (False, True):
            p = color_matching_distribution(d, refined=refined)
            H = triangle_entropic_vector(p, label_suffix="")
            slack = eq4_slack(p)
            mode = "refined" if refined else "coarse"
            print(f"{d:>3}{mode:>10}{p.shape[0]:>7}{H['A']:>9.4f}{H['A,B,C']:>10.4f}"
                  f"{slack:>+12.4f}{slack / H['A,B,C']:>14.4f}")
    print("-" * 72)
    print("slack >= 0: Eq. (4) satisfied. Coarse CM is classically simulable;")
    print("refined CM (= Renou off-diagonal) carries the coherence.")
