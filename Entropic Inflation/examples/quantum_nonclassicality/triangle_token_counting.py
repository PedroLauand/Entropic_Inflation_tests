"""Token-Counting triangle distributions, validated at low dimension.

One of two strategy families in this folder's generator of quantum triangle
distributions that are *nonlocal at the probability level* (see the folder
README). In Token-Counting each source distributes a fixed number of tokens and
each party counts the tokens it receives; by rigidity, coherent members of this
family admit no classical (trilocal) model.

This module builds the *Token-Counting* (TC) strategy of

    Renou & Beigi, "Network nonlocality via rigidity of token counting and
    colour matching" [arXiv:2202.00905],

and shows the qubit case coincides with the explicit triangle construction of

    Renou, Baeumer, Boreiri, Brunner, Gisin, Beigi,
    "Genuine quantum nonlocality in the triangle network",
    PRL 123, 140401 (2019)  [arXiv:1905.04902].

Qubit Token-Counting strategy (eta = 1)
---------------------------------------
Each source distributes exactly ONE token in the uniform superposition

    |psi> = (|01> + |10>)/sqrt(2)

i.e. precisely one of its two recipients holds the token (|1>), the other the
vacuum (|0>). Each party holds the two qubits sent by its two sources and reads
out the TOTAL token count n = k1 + k2 in {0, 1, 2} (one real parameter ``u``,
with ``u^2 + v^2 = 1`` and ``0 < v < u < 1``):

    |up>   = |00>                count n = 0   (unambiguous)
    |down> = |11>                count n = 2   (unambiguous)
    |chi0> = u|01> + v|10>       count n = 1   (degenerate sector, coherent)
    |chi1> = v|01> - u|10>       count n = 1

So every party has 4 outcomes, ordered here as ``[up, down, chi0, chi1]``. The
ambiguous n=1 sector encodes "which of my two sources sent the token?" -- the
*provenance* superposition that carries the coherence.

Equivalence to Renou Eq. (2) [arXiv:1905.04902]
-----------------------------------------------
Renou's original qubit basis instead uses the maximally entangled source
``(|00>+|11>)/sqrt(2)`` with deterministic outcomes ``|01>, |10>`` and coherent
outcomes ``u|00>+v|11>, v|00>-u|11>``. Flipping the second qubit each party
measures (the local unitary ``I (x) X`` on that wire) turns one description into
the other: the source becomes ``(|01>+|10>)/sqrt(2)`` and the basis becomes the
token-count basis above. A per-wire change of basis is absorbed between source
and measurement, so p(a,b,c) is *identical* -- verified bit-for-bit by
``confirm_qubit``. ``token_counting_qubit_distribution`` builds the TC form,
``renou_qubit_distribution`` the equivalent Renou form.

Generic eta-token construction
------------------------------
``token_counting_distribution(eta, sector_angles)`` generalises to any number of
tokens (source local dimension ``d = eta + 1``): the uniform source
``(sum_k |k, eta-k>)/sqrt(eta+1)`` and a party measurement BLOCK-DIAGONAL in the
total count ``n = k1 + k2``, with a free rotation ``SO(m_n)`` in each degenerate
count sector (the end sectors n=0 and n=2*eta are 1-dim and frozen). Identity
rotations = the decohered, classically simulable baseline; non-trivial angles
inject the coherence. ``describe_tuning(eta)`` lists the angles; the qubit
strategy above is recovered exactly at ``eta = 1``.

Quickstart (exploration)
------------------------
>>> import numpy as np
>>> p = renou_qubit_distribution(np.sqrt(0.8))   # simplest TC strategy, p(a,b,c)
>>> H = entropy_vector(p)                          # the 7 observed entropies
>>> eq4_slack(p)                                   # spiral inequality Eq. (4) slack
7.76878...
>>> token_counting_entropy_vector(2)              # qutrit-source TC, in one call
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from triangle_probability import (
    maximally_entangled_state,
    stack_povm,
    swap_povm_qubits,
    triangle_probability_distribution,
)
from entropic_vector import spiral_slack, triangle_entropic_vector

# Outcome ordering used for the qubit (d=2) construction.
QUBIT_OUTCOMES = ("up", "down", "chi0", "chi1")


def _projectors_from_kets(kets: np.ndarray) -> np.ndarray:
    """Stack rank-1 projectors |k><k| (rows = outcomes) into (n, 2, 2, 2, 2)."""
    mats = [np.outer(ket, ket.conj()) for ket in kets]
    return stack_povm(mats, 2, 2)


def renou_qubit_measurement(u: float) -> np.ndarray:
    """The four-outcome qubit basis of Eq. (2), as a stacked POVM tensor.

    Basis order on the two qubits is |00>, |01>, |10>, |11> (indices 0..3).
    Returns outcomes in the order ``QUBIT_OUTCOMES``.
    """
    if not 0.0 < u < 1.0:
        raise ValueError("u must lie in (0, 1).")
    v = np.sqrt(1.0 - u * u)
    up = np.array([0, 1, 0, 0], dtype=complex)          # |01>
    down = np.array([0, 0, 1, 0], dtype=complex)        # |10>
    chi0 = np.array([u, 0, 0, v], dtype=complex)        # u|00> + v|11>
    chi1 = np.array([v, 0, 0, -u], dtype=complex)       # v|00> - u|11>
    return _projectors_from_kets(np.stack([up, down, chi0, chi1]))


def token_counting_qubit_measurement(u: float) -> np.ndarray:
    """Eq. (2) basis after flipping the second qubit (the TC / 2202.00905 view).

    Outcomes, same order ``QUBIT_OUTCOMES``:
        up   -> |00|  (token count n = 0)
        down -> |11|  (token count n = 2)
        chi0 -> u|01> + v|10>   (coherent n = 1 sector)
        chi1 -> v|01> - u|10>
    """
    if not 0.0 < u < 1.0:
        raise ValueError("u must lie in (0, 1).")
    v = np.sqrt(1.0 - u * u)
    up = np.array([1, 0, 0, 0], dtype=complex)          # |00>  (n=0)
    down = np.array([0, 0, 0, 1], dtype=complex)        # |11>  (n=2)
    chi0 = np.array([0, u, v, 0], dtype=complex)        # u|01> + v|10>
    chi1 = np.array([0, v, -u, 0], dtype=complex)       # v|01> - u|10>
    return _projectors_from_kets(np.stack([up, down, chi0, chi1]))


def renou_qubit_distribution(u: float) -> np.ndarray:
    """p(a,b,c) for the maximally entangled qubit construction of Eq. (2).

    The basis vectors ``|up>=|01>`` and ``|down>=|10>`` are not symmetric under
    exchanging a party's two qubits, so the construction depends on a consistent
    cyclic orientation A->B->C->A. In this module's subsystem layout, party B's
    two subsystems (alpha_B, gamma_B) are stored in the reverse order relative to
    A and C, so B's POVM is swapped to restore the orientation (same fix as the
    EJM example).
    """
    rho = maximally_entangled_state(2)
    M = renou_qubit_measurement(u)
    return triangle_probability_distribution(rho, rho, rho, M, swap_povm_qubits(M), M)


def token_counting_qubit_distribution(u: float) -> np.ndarray:
    """p(a,b,c) for the bit-flipped (token-counting) qubit construction.

    Source is the one-token state ``(|01> + |10>)/sqrt(2)``; party basis is the
    flipped basis above. Must equal :func:`renou_qubit_distribution`.
    """
    psi = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2.0)   # (|01>+|10>)/sqrt2
    rho = np.outer(psi, psi.conj()).reshape(2, 2, 2, 2)
    M = token_counting_qubit_measurement(u)
    # Same cyclic-orientation swap on party B as renou_qubit_distribution.
    return triangle_probability_distribution(rho, rho, rho, M, swap_povm_qubits(M), M)


# ---------------------------------------------------------------------------
# Renou qutrit construction (arXiv:1905.04902, p. 4).
# ---------------------------------------------------------------------------
#
# All sources share the maximally entangled qutrit |phi_3> = (|00>+|11>+|22>)/sqrt3.
# Each party measures its two qutrits in a 9-outcome basis: the three diagonal
# (colour-match) states |00>,|11>,|22> in the computational basis, and the two
# off-diagonal triangular blocks {|01>,|02>,|12>} and {|10>,|20>,|21>} each in a
# chosen orthonormal basis. The paper leaves those off-diagonal bases free; here
# we use the canonical DFT (Fourier) basis as a concrete representative.


def _dft_block_kets(flat_indices: list[int], d: int) -> list[np.ndarray]:
    """DFT (Fourier) orthonormal basis supported on the given computational indices."""
    m = len(flat_indices)
    omega = np.exp(2j * np.pi / m)
    kets = []
    for r in range(m):
        ket = np.zeros(d * d, dtype=complex)
        for s, idx in enumerate(flat_indices):
            ket[idx] = omega ** (s * r) / np.sqrt(m)
        kets.append(ket)
    return kets


def renou_offdiagonal_distribution(d: int) -> np.ndarray:
    """Maximally entangled ``|Phi_d>`` source with the Renou off-diagonal measurement.

    Measurement (d*d outcomes per party): the ``d`` diagonal colour-match states
    ``|kk>`` in the computational basis, plus the upper-triangular
    ``{|ij> : i<j}`` and lower-triangular ``{|ij> : i>j}`` blocks each in the DFT
    basis. Generalises the qutrit construction of arXiv:1905.04902 p. 4 to any
    ``d``. Party B is swapped for the cyclic orientation.

    Note: at ``d=2`` the off-diagonal blocks are one-dimensional, so this reduces
    to the computational (decohered, classical) measurement -- the genuinely
    nonlocal qubit instead carries its coherence on the diagonal (see
    ``renou_qubit_distribution``).
    """
    diagonal = [k * d + k for k in range(d)]
    upper = [i * d + j for i in range(d) for j in range(d) if i < j]
    lower = [i * d + j for i in range(d) for j in range(d) if i > j]
    kets = [np.eye(d * d, dtype=complex)[i] for i in diagonal]
    kets += _dft_block_kets(upper, d)
    kets += _dft_block_kets(lower, d)
    M = stack_povm([np.outer(k, k.conj()) for k in kets], d, d)
    rho = maximally_entangled_state(d)
    return triangle_probability_distribution(rho, rho, rho, M, swap_povm_qubits(M), M)


def renou_qutrit_distribution() -> np.ndarray:
    """p(a,b,c) for the maximally entangled qutrit construction (9 outcomes/party)."""
    return renou_offdiagonal_distribution(3)


# ---------------------------------------------------------------------------
# Generic eta-token construction (source dimension d = eta + 1).
# ---------------------------------------------------------------------------
#
# Source : each source distributes eta tokens between its two parties in the
#          uniform superposition  |phi_eta> = (sum_k |k, eta-k>) / sqrt(eta+1),
#          so each party register has local dimension d = eta + 1.
# Party  : holds two registers (k1, k2), each in {0, ..., eta}. Its measurement
#          is a complete orthonormal basis of the d*d space that is
#          BLOCK-DIAGONAL in the total count n = k1 + k2. Within each count
#          sector H_n = span{|k1, n-k1>} the basis is a free rotation O_n in
#          SO(dim H_n), parametrised by Givens angles. The two end sectors
#          (n = 0 and n = 2*eta) are one-dimensional and therefore frozen.
#
# Outcomes per party: d*d (full refined). The bare count would give 2*eta+1.


def count_sector_provenances(eta: int) -> dict[int, list[tuple[int, int]]]:
    """Provenance configurations ``(k1, k2)`` grouped by total count ``n``.

    Returns a dict ``n -> [(k1, k2), ...]`` for ``n = 0 .. 2*eta``, each list
    ordered by ``k1`` ascending. ``len(list)`` is the dimension of sector ``H_n``.
    """
    sectors: dict[int, list[tuple[int, int]]] = {}
    for n in range(2 * eta + 1):
        sectors[n] = [(k1, n - k1) for k1 in range(eta + 1) if 0 <= n - k1 <= eta]
    return sectors


def n_angles_per_sector(eta: int) -> dict[int, int]:
    """Number of free Givens angles ``m(m-1)/2`` available in each count sector."""
    return {n: (len(p) * (len(p) - 1)) // 2 for n, p in count_sector_provenances(eta).items()}


def total_angles(eta: int) -> int:
    """Total number of free measurement angles for the eta-token strategy."""
    return sum(n_angles_per_sector(eta).values())


def _givens_orthogonal(m: int, angles) -> np.ndarray:
    """Build an ``SO(m)`` matrix as a product of plane (Givens) rotations.

    Expects ``m(m-1)/2`` angles, one per plane ``(i, j)`` with ``i < j``.
    """
    angles = list(angles)
    expected = m * (m - 1) // 2
    if len(angles) != expected:
        raise ValueError(f"SO({m}) needs {expected} angles, got {len(angles)}.")
    O = np.eye(m)
    idx = 0
    for i in range(m):
        for j in range(i + 1, m):
            c, s = np.cos(angles[idx]), np.sin(angles[idx])
            R = np.eye(m)
            R[i, i] = R[j, j] = c
            R[i, j], R[j, i] = -s, s
            O = O @ R
            idx += 1
    return O


def block_diagonal_basis(eta: int, sector_angles: dict[int, "list"] | None = None):
    """Measurement kets (rows) + their ``(count, within-sector index)`` labels.

    ``sector_angles`` maps a count ``n`` to the Givens angles of that sector's
    ``SO(m)`` rotation. Any omitted sector defaults to the identity (its
    computational/provenance basis). Identity everywhere = the decohered
    (classically simulable) measurement; non-trivial angles inject coherence.
    """
    d = eta + 1
    sectors = count_sector_provenances(eta)
    sector_angles = sector_angles or {}
    kets: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    for n in range(2 * eta + 1):
        prov = sectors[n]
        m = len(prov)
        O = np.eye(m) if m == 1 else _givens_orthogonal(m, sector_angles.get(n, np.zeros(m * (m - 1) // 2)))
        for r in range(m):  # basis vector r = sum_s O[s, r] |prov_s>
            ket = np.zeros(d * d, dtype=complex)
            for s, (k1, k2) in enumerate(prov):
                ket[k1 * d + k2] += O[s, r]
            kets.append(ket)
            labels.append((n, r))
    return np.array(kets), labels


def uniform_token_source(eta: int) -> np.ndarray:
    """Uniform eta-token source ``(sum_k |k, eta-k>)/sqrt(eta+1)`` as a tensor."""
    d = eta + 1
    psi = np.zeros(d * d, dtype=complex)
    for k in range(eta + 1):
        psi[k * d + (eta - k)] = 1.0
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj()).reshape(d, d, d, d)


def token_counting_distribution(eta: int, sector_angles: dict[int, "list"] | None = None):
    """``p(a,b,c)`` and outcome labels for the generic eta-token TC strategy.

    All three sources are the uniform eta-token state; all three parties use the
    same block-diagonal measurement, with party B's subsystems swapped to keep
    the cyclic A->B->C orientation (same fix as the qubit case).
    """
    d = eta + 1
    rho = uniform_token_source(eta)
    kets, labels = block_diagonal_basis(eta, sector_angles)
    M = stack_povm([np.outer(k, k.conj()) for k in kets], d, d)
    p = triangle_probability_distribution(rho, rho, rho, M, swap_povm_qubits(M), M)
    return p, labels


def describe_tuning(eta: int) -> None:
    """Print the available source/measurement tuning for the eta-token strategy."""
    d = eta + 1
    sectors = count_sector_provenances(eta)
    nang = n_angles_per_sector(eta)
    print(f"eta = {eta} tokens/source   ->   source local dimension d = {d}")
    print(f"  SOURCE   : uniform (sum_k |k,{eta}-k>)/sqrt({d})   [fixed here; the "
          f"{d} coefficients c_k are a further optional knob]")
    print(f"  MEASUREMENT: {d * d} outcomes, block-diagonal in total count n = k1 + k2")
    for n in range(2 * eta + 1):
        m = len(sectors[n])
        prov = ", ".join(f"|{k1},{k2}>" for k1, k2 in sectors[n])
        kind = "frozen (1-dim, no freedom)" if m == 1 else f"SO({m}) -> {nang[n]} angle(s)"
        print(f"    count n={n}: dim {m}  [{prov}]   {kind}")
    print(f"  TOTAL free measurement angles: {total_angles(eta)}"
          f"   (shared across the 3 parties by the cyclic symmetry)")


def qubit_sector_angles(u: float) -> dict[int, list]:
    """The eta=1 sector angles reproducing the calibrated qubit ``u`` (cos = u)."""
    return {1: [float(np.arccos(u))]}


# ---------------------------------------------------------------------------
# From a strategy to its observed entropy vector (for exploration).
# ---------------------------------------------------------------------------
#
# The entropic side of the project tests the *observed* triangle entropies, so
# the bridge from a strategy is: distribution p(a,b,c) -> the seven entropies
# H(A), H(B), H(C), H(A,B), H(A,C), H(B,C), H(A,B,C). The entropy computation
# itself lives in ``entropic_vector`` (the single source of truth); these are
# one-call conveniences so you can go straight from token-counting parameters to
# the entropy vector and the spiral inequality Eq. (4).


def entropy_vector(p: np.ndarray) -> dict[str, float]:
    """The seven observed Shannon entropies of ``p(a,b,c)`` (plain A,B,C labels)."""
    return triangle_entropic_vector(p, label_suffix="")


def eq4_slack(p: np.ndarray) -> float:
    """Spiral inequality Eq. (4) slack of ``p(a,b,c)`` (>= 0 means satisfied)."""
    return spiral_slack(entropy_vector(p), label_suffix="")


def token_counting_entropy_vector(
    eta: int, sector_angles: dict[int, "list"] | None = None
) -> dict[str, float]:
    """Build the eta-token TC distribution and return its observed entropy vector."""
    p, _labels = token_counting_distribution(eta, sector_angles)
    return entropy_vector(p)


# ---------------------------------------------------------------------------
# Confirmation against the explicit formulas of arXiv:1905.04902.
# ---------------------------------------------------------------------------

UP, DOWN, CHI0, CHI1 = 0, 1, 2, 3


def print_distribution(p: np.ndarray, labels=QUBIT_OUTCOMES, tol: float = 1e-12) -> None:
    """Print the full probability vector p(a,b,c), one labelled entry per line."""
    print(f"  probability vector: {p.shape} -> {p.size} entries, sum = {p.sum():.10f}")
    nonzero = 0
    for a in range(p.shape[0]):
        for b in range(p.shape[1]):
            for c in range(p.shape[2]):
                val = float(p[a, b, c])
                if val > tol:
                    nonzero += 1
                    print(f"    p({labels[a]:>4}, {labels[b]:>4}, {labels[c]:>4}) = {val:.6f}")
    print(f"    [{nonzero} nonzero entries; {p.size - nonzero} are zero]")


def confirm_qubit(u: float, tol: float = 1e-12) -> None:
    """Check normalisation and Eqs. (3),(4),(5) at the maximally entangled point."""
    v = np.sqrt(1.0 - u * u)
    p = renou_qubit_distribution(u)
    u_i = (u, v)          # |00> coefficients of (chi0, chi1)
    v_i = (v, -u)         # |11> coefficients of (chi0, chi1)

    print(f"u = {u:.6f}  v = {v:.6f}   (u^2 = {u*u:.4f})")
    print(f"  outcomes per party : {p.shape}  ->  {p.size} probabilities")
    print(f"  normalisation      : sum p = {p.sum():.12f}")

    # Eq. (3): P(a=up,b=up) = P(a=down,b=down) = 0 (and cyclic permutations).
    eq3 = [
        ("P(A=up,B=up)", p[UP, UP, :].sum()),
        ("P(A=down,B=down)", p[DOWN, DOWN, :].sum()),
        ("P(B=up,C=up)", p[:, UP, UP].sum()),
        ("P(A=down,C=down)", p[DOWN, :, DOWN].sum()),
    ]
    print("  Eq.(3) zeros:")
    for name, val in eq3:
        print(f"    {name:18s} = {val:+.3e}   {'OK' if abs(val) < 1e-9 else 'MISMATCH'}")

    # Eq. (4): P(chi_i, up, down) = u_i^2 / 8 ;  P(chi_i, down, up) = v_i^2 / 8.
    print("  Eq.(4)  P(chi_i,up,down)=u_i^2/8 , P(chi_i,down,up)=v_i^2/8:")
    for i in (0, 1):
        got1, exp1 = p[CHI0 + i, UP, DOWN], u_i[i] ** 2 / 8
        got2, exp2 = p[CHI0 + i, DOWN, UP], v_i[i] ** 2 / 8
        ok1 = abs(got1 - exp1) < tol
        ok2 = abs(got2 - exp2) < tol
        print(f"    i={i}: up/down got {got1:.8f} exp {exp1:.8f} {'OK' if ok1 else 'X'}"
              f" | down/up got {got2:.8f} exp {exp2:.8f} {'OK' if ok2 else 'X'}")

    # Eq. (5): P(chi_i, chi_j, chi_k) = (u_i u_j u_k + v_i v_j v_k)^2 / 8.
    print("  Eq.(5)  P(chi_i,chi_j,chi_k)=(u_i u_j u_k + v_i v_j v_k)^2/8:")
    worst = 0.0
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                got = p[CHI0 + i, CHI0 + j, CHI0 + k]
                exp = (u_i[i] * u_i[j] * u_i[k] + v_i[i] * v_i[j] * v_i[k]) ** 2 / 8
                worst = max(worst, abs(got - exp))
    print(f"    max |got - expected| over the 8 triples = {worst:.3e}"
          f"   {'OK' if worst < tol else 'MISMATCH'}")

    # Token-counting equivalence.
    p_tc = token_counting_qubit_distribution(u)
    diff = float(np.max(np.abs(p - p_tc)))
    print(f"  TC equivalence: max|p_renou - p_tokencount| = {diff:.3e}"
          f"   {'OK (identical)' if diff < 1e-12 else 'DIFFER'}")


def confirm_generic_matches_qubit(u: float, tol: float = 1e-12) -> None:
    """The generic eta=1 builder must reproduce the simple qubit example."""
    p_gen, labels = token_counting_distribution(1, qubit_sector_angles(u))
    # generic outcome order [n0, chi0, chi1, n2]; qubit example order [n0, n2, chi0, chi1]
    perm = [0, 3, 1, 2]
    p_gen_reordered = p_gen[np.ix_(perm, perm, perm)]
    diff = float(np.max(np.abs(p_gen_reordered - token_counting_qubit_distribution(u))))
    print(f"  generic(eta=1) vs qubit example: max|diff| = {diff:.3e}"
          f"   {'OK (identical)' if diff < tol else 'MISMATCH'}")


if __name__ == "__main__":
    # u^2 in (u_max^2, 1) ~ (0.785, 1) is the nonlocal regime; pick u^2 = 0.8.
    u = np.sqrt(0.8)
    print("=" * 70)
    print("Simple qubit example (sanity check vs arXiv:1905.04902)")
    print("=" * 70)
    confirm_qubit(u=u)
    print()
    print("Full probability vector p(a,b,c):")
    p_qubit = renou_qubit_distribution(u)
    print_distribution(p_qubit)
    print()
    print("Observed entropy vector (bits) and spiral inequality Eq. (4):")
    H_qubit = entropy_vector(p_qubit)
    for label in ("A", "B", "C", "A,B", "A,C", "B,C", "A,B,C"):
        print(f"    H({label:5s}) = {H_qubit[label]:.6f}")
    print(f"    Eq. (4) slack = {eq4_slack(p_qubit):+.6f}   (>= 0 means satisfied)")

    print()
    print("=" * 70)
    print("Generic eta-token construction")
    print("=" * 70)
    confirm_generic_matches_qubit(u)
    for eta in (1, 2, 3):
        print()
        describe_tuning(eta)
    print()
    # Show that the generic builder runs and normalises at higher dimension.
    for eta in (2, 3):
        p, labels = token_counting_distribution(eta)  # identity angles (decohered baseline)
        print(f"eta={eta}: p has shape {p.shape} -> {p.size} entries, sum = {p.sum():.10f}")
