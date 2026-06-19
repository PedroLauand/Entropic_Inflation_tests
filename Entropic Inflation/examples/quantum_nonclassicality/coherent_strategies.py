"""Generic coherent quantum strategies on the triangle network.

Token Counting (TC) and Color Matching (CM) of Renou & Beigi [arXiv:2202.00905],
as a single dimension- and angle-parametrized construction.

Common skeleton
---------------
Each party receives two ``dim``-dimensional registers (one from each of its two
sources) and measures them.  The ``dim**2``-dimensional two-register space splits
into SECTORS fixed by a classical invariant:

    TC :  total token count   n = k1 + k2          (n = 0 .. 2*(dim-1))
    CM :  colour-match pattern (a matched colour, or an off-diagonal block)

A sector of dimension 1 is frozen (no freedom).  In every multi-dimensional
sector the measurement basis is a free REAL rotation ``O in SO(m)``, supplied as
``m*(m-1)/2`` Givens angles.  Identity rotations (angles = 0) give the decohered,
classically-simulable baseline; non-trivial angles inject the coherence that
makes the strategy network-nonlocal.  Real (not complex) rotations restrict CM
to the real off-diagonal bases of arXiv:1905.04902 / arXiv:2202.00905 (the paper
fixes the coefficients real; the basis within each block is otherwise free).

Only the SOURCE and the SECTOR PARTITION distinguish the two families:

    TC :  source = uniform token state (sum_k |k, eta-k>)/sqrt(dim),  eta = dim-1
          sectors grouped by total count n
    CM :  source = maximally entangled |Phi_d> = (sum_c |cc>)/sqrt(dim)
          sectors = the dim matched-colour singletons + the upper/lower
                    off-diagonal "mismatch" blocks

Interface
---------
    token_counting(dim, angles=None) -> p[a, b, c]
    color_matching(dim, angles=None) -> p[a, b, c]

``dim`` is the source local dimension (TC: tokens + 1; CM: number of colours).
``angles`` maps a sector label to its Givens angles; omitted sectors default to
the identity.  Inspect the rotatable sectors with ``describe(family, dim)`` or
``angle_layout(family, dim)``, and build random explorations with
``random_angles(family, dim, rng)``.

    PYTHONPATH=. python examples/quantum_nonclassicality/coherent_strategies.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from triangle_probability import (
    maximally_entangled_state,
    stack_povm,
    swap_povm_qubits,
    triangle_probability_distribution,
)
from entropic_vector import spiral_slack, triangle_entropic_vector


# ---------------------------------------------------------------------------
# Shared core: real SO(m) rotations and block-diagonal measurements.
# ---------------------------------------------------------------------------


def givens_rotation(m: int, angles: Sequence[float]) -> np.ndarray:
    """Real ``SO(m)`` matrix as a product of plane (Givens) rotations.

    Expects ``m*(m-1)/2`` angles, one per plane ``(i, j)`` with ``i < j``.
    ``m = 1`` takes no angles and returns ``[[1]]``.
    """
    angles = list(angles)
    needed = m * (m - 1) // 2
    if len(angles) != needed:
        raise ValueError(f"SO({m}) needs {needed} angles, got {len(angles)}.")
    O = np.eye(m)
    k = 0
    for i in range(m):
        for j in range(i + 1, m):
            c, s = np.cos(angles[k]), np.sin(angles[k])
            R = np.eye(m)
            R[i, i] = R[j, j] = c
            R[i, j], R[j, i] = -s, s
            O = O @ R
            k += 1
    return O


# A sector partition is an ordered list of (label, [computational indices]).
Blocks = Sequence[tuple[object, Sequence[int]]]


def block_measurement(
    blocks: Blocks, dim: int, angles: Mapping[object, Sequence[float]] | None = None
) -> tuple[np.ndarray, list[tuple[object, int]]]:
    """Projective POVM that is block-diagonal over ``blocks``.

    Each ``(label, indices)`` spans a sector of the ``dim**2`` space; inside it
    the basis is the real ``SO(len(indices))`` rotation ``angles[label]``
    (identity if absent).  Outcomes follow block order then in-block order.
    Returns the stacked POVM tensor and the list of ``(label, within-index)``
    outcome tags.  The blocks must partition ``range(dim**2)``.
    """
    angles = dict(angles or {})
    labels = {label for label, _ in blocks}
    unknown = [k for k in angles if k not in labels]
    if unknown:
        raise ValueError(
            f"angles has unknown sector labels {sorted(map(str, unknown))}; "
            f"valid labels are {sorted(map(str, labels))}."
        )
    kets: list[np.ndarray] = []
    tags: list[tuple[object, int]] = []
    for label, idx in blocks:
        m = len(idx)
        O = givens_rotation(m, angles.get(label, np.zeros(m * (m - 1) // 2)))
        for r in range(m):  # basis vector r = sum_s O[s, r] |idx_s>
            ket = np.zeros(dim * dim, dtype=complex)
            for s, i in enumerate(idx):
                ket[i] = O[s, r]
            kets.append(ket)
            tags.append((label, r))
    return stack_povm([np.outer(k, k.conj()) for k in kets], dim, dim), tags


def _triangle(source: np.ndarray, M: np.ndarray) -> np.ndarray:
    """p(a,b,c): identical sources, identical party POVM ``M``.

    Party B's two subsystems are stored in reverse order relative to A and C, so
    its POVM is swapped to restore the cyclic A->B->C orientation.  The swap is
    required -- the coherent measurements built here are not swap-symmetric -- and
    matches the reference TC / off-diagonal-CM constructions.
    """
    return triangle_probability_distribution(
        source, source, source, M, swap_povm_qubits(M), M
    )


# ---------------------------------------------------------------------------
# Token Counting.
# ---------------------------------------------------------------------------


def token_counting_blocks(dim: int) -> list[tuple[int, list[int]]]:
    """Two-register sectors grouped by total token count ``n = k1 + k2``.

    Labelled by the count ``n = 0 .. 2*(dim-1)``; the two end sectors are
    1-dimensional (frozen), the middle ones carry the provenance coherence.
    """
    return [
        (n, [k1 * dim + (n - k1) for k1 in range(dim) if 0 <= n - k1 < dim])
        for n in range(2 * dim - 1)
    ]


def uniform_token_source(dim: int) -> np.ndarray:
    """Uniform token source ``(sum_k |k, eta-k>)/sqrt(dim)`` with ``eta = dim-1``."""
    eta = dim - 1
    psi = np.zeros(dim * dim, dtype=complex)
    for k in range(dim):
        psi[k * dim + (eta - k)] = 1.0
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj()).reshape(dim, dim, dim, dim)


def token_counting(
    dim: int, angles: Mapping[object, Sequence[float]] | None = None
) -> np.ndarray:
    """``p(a,b,c)`` for the ``dim``-dimensional Token-Counting strategy.

    ``angles`` maps a count ``n`` to the Givens angles of that sector's ``SO(m)``
    rotation (``angles=None`` -> decohered/classical baseline).
    """
    M, _ = block_measurement(token_counting_blocks(dim), dim, angles)
    return _triangle(uniform_token_source(dim), M)


# ---------------------------------------------------------------------------
# Color Matching.
# ---------------------------------------------------------------------------


def color_matching_blocks(dim: int) -> list[tuple[object, list[int]]]:
    """Sectors of CM: ``dim`` frozen matched-colour singletons ``|cc>`` plus the
    upper ``{|ij>: i<j}`` and lower ``{|ij>: i>j}`` off-diagonal mismatch blocks.

    Each off-diagonal block has dimension ``dim*(dim-1)/2``; they are coherent
    only for ``dim >= 3`` (at ``dim = 2`` each is 1-dimensional, so CM reduces to
    the classical computational measurement -- this is intrinsic to the
    color-matching construction, not a limitation of the implementation).
    """
    diagonal = [(("match", c), [c * dim + c]) for c in range(dim)]
    upper = ("upper", [i * dim + j for i in range(dim) for j in range(dim) if i < j])
    lower = ("lower", [i * dim + j for i in range(dim) for j in range(dim) if i > j])
    return diagonal + [upper, lower]


def color_matching(
    dim: int, angles: Mapping[object, Sequence[float]] | None = None
) -> np.ndarray:
    """``p(a,b,c)`` for the ``dim``-colour Color-Matching strategy.

    The matched-colour outcomes ``|cc>`` are reported verbatim; ``angles`` maps
    ``"upper"`` / ``"lower"`` to the Givens angles resolving the mismatch blocks
    (``angles=None`` -> decohered/classical baseline).  Rotations are real: the
    paper fixes the off-diagonal coefficients real (a deliberate restriction; the
    basis within each block is otherwise free).
    """
    M, _ = block_measurement(color_matching_blocks(dim), dim, angles)
    return _triangle(maximally_entangled_state(dim), M)


# ---------------------------------------------------------------------------
# Exploration helpers: angle layout, random angles, entropy vector.
# ---------------------------------------------------------------------------

_FAMILIES = {"TC": token_counting_blocks, "CM": color_matching_blocks}


def _blocks(family: str, dim: int) -> Blocks:
    try:
        return _FAMILIES[family](dim)
    except KeyError:
        raise ValueError(f"family must be one of {sorted(_FAMILIES)}, got {family!r}.")


def angle_layout(family: str, dim: int) -> list[tuple[object, int, int]]:
    """List of ``(label, sector_dim, n_angles)`` for each sector of the family.

    ``n_angles = m*(m-1)/2`` is the number of free Givens angles in that sector;
    sectors with ``sector_dim == 1`` are frozen (``n_angles == 0``).
    """
    return [(lab, len(idx), len(idx) * (len(idx) - 1) // 2) for lab, idx in _blocks(family, dim)]


def n_angles(family: str, dim: int) -> int:
    """Total number of free measurement angles for the family at this dimension."""
    return sum(na for _, _, na in angle_layout(family, dim))


def random_angles(family: str, dim: int, rng: np.random.Generator) -> dict[object, list[float]]:
    """Random Givens angles for every rotatable sector (for exploration)."""
    return {
        lab: list(rng.uniform(0.0, 2.0 * np.pi, size=na))
        for lab, _, na in angle_layout(family, dim)
        if na > 0
    }


def describe(family: str, dim: int) -> None:
    """Print the source and the rotatable measurement sectors of the family."""
    src = (
        f"uniform token state (sum_k |k,{dim-1}-k>)/sqrt({dim})"
        if family == "TC"
        else f"maximally entangled |Phi_{dim}> = (sum_c |cc>)/sqrt({dim})"
    )
    print(f"{family}  dim = {dim}   ({dim*dim} outcomes/party)")
    print(f"  SOURCE: {src}")
    print("  MEASUREMENT sectors (label: dim -> #angles):")
    for lab, d, na in angle_layout(family, dim):
        kind = "frozen" if d == 1 else f"SO({d})"
        print(f"    {str(lab):>14}: {d:>3}  -> {na:>3}   [{kind}]")
    print(f"  TOTAL free angles: {n_angles(family, dim)}")


def entropy_vector(p: np.ndarray) -> dict[str, float]:
    """The seven observed Shannon entropies of ``p(a,b,c)`` (plain A,B,C labels)."""
    return triangle_entropic_vector(p, label_suffix="")


def eq4_slack(p: np.ndarray) -> float:
    """Spiral inequality Eq. (4) slack of ``p(a,b,c)`` (>= 0 means satisfied)."""
    return spiral_slack(entropy_vector(p), label_suffix="")


# ---------------------------------------------------------------------------
# Demo / sanity checks.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 70)
    print("Sector layouts")
    print("=" * 70)
    describe("TC", 2)
    print()
    describe("CM", 3)

    print()
    print("=" * 70)
    print("Token Counting: qubit (dim=2) reproduces the Renou construction")
    print("=" * 70)
    u = np.sqrt(0.8)
    theta = float(np.arccos(u))           # n=1 sector SO(2) angle with cos = u
    p_tc = token_counting(2, {1: [theta]})
    H = entropy_vector(p_tc)
    print(f"  u^2 = 0.8 -> n=1 angle = arccos(u) = {theta:.4f}")
    print(f"  H(A) = {H['A']:.4f}   H(AB) = {H['A,B']:.4f}   H(ABC) = {H['A,B,C']:.4f}")
    print(f"  Eq. (4) slack = {eq4_slack(p_tc):+.4f}   (expect +7.7688)")

    print()
    print("=" * 70)
    print("Color Matching: qutrit (dim=3), real coherent mismatch rotation")
    print("=" * 70)
    rng = np.random.default_rng(0)
    angles = random_angles("CM", 3, rng)
    p_cm = color_matching(3, angles)
    H = entropy_vector(p_cm)
    M, _ = block_measurement(color_matching_blocks(3), 3, angles)
    print(f"  random real angles: upper={np.round(angles['upper'],3)}")
    print(f"                      lower={np.round(angles['lower'],3)}")
    print(f"  measurement is real:        max|Im(M)| = {np.max(np.abs(M.imag)):.1e}")
    print(f"  POVM complete (sum = I):    max|sum M - I| = "
          f"{np.max(np.abs(M.sum(axis=0) - np.eye(9).reshape(3,3,3,3))):.1e}")
    print(f"  H(A) = {H['A']:.4f}  (= 2 log2 3 = {2*np.log2(3):.4f})")
    print(f"  H(ABC) = {H['A,B,C']:.4f}   Eq. (4) slack = {eq4_slack(p_cm):+.4f}")

    print()
    print("=" * 70)
    print("Decohered baselines (angles = None) are classical")
    print("=" * 70)
    for fam, dim, fn in (("TC", 3, token_counting), ("CM", 3, color_matching)):
        p = fn(dim)
        print(f"  {fam} dim={dim}: Eq. (4) slack = {eq4_slack(p):+.4f}  (decohered)")
