"""Validate :mod:`coherent_strategies` against the published constructions, and
survey the entropy vectors across source dimensions.

Three parts:

1. PAPER CHECK.  The qubit Token-Counting strategy must reproduce the explicit
   Renou et al. construction (arXiv:1905.04902 Eq. (2)).  We build that
   construction *independently* here (its own source + four-outcome basis),
   verify it satisfies the paper's Eqs. (3),(4),(5), and confirm that
   ``coherent_strategies.token_counting(2, ...)`` matches it.

2. STRUCTURE CHECK.  The Color-Matching measurement is real, a complete
   projective POVM, and has ``H(A) = 2 log2(dim)`` exactly (maximally mixed
   single-party marginal), at every dimension.

3. SURVEY.  Eq. (4) slack and the entropy vector across dimensions for both
   families (a fixed representative coherent angle choice vs the decohered
   baseline), plus the output-size scaling fact.

Run from the package root:

    PYTHONPATH=. python examples/quantum_nonclassicality/validate_strategies.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import coherent_strategies as cs
from triangle_probability import (
    maximally_entangled_state,
    stack_povm,
    swap_povm_qubits,
    triangle_probability_distribution,
)
from entropic_vector import triangle_entropic_vector

# Outcome indices of the Renou qubit basis (up, down, chi0, chi1).
UP, DOWN, CHI0, CHI1 = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# 1. Independent Renou reference (arXiv:1905.04902 Eq. (2)) and its equations.
# ---------------------------------------------------------------------------


def renou_qubit_reference(u: float) -> np.ndarray:
    """Explicit Renou Eq. (2) qubit triangle distribution (independent of the core).

    Sources ``(|00>+|11>)/sqrt2``; party basis ``{|01>, |10>, u|00>+v|11>,
    v|00>-u|11>}`` with ``v = sqrt(1-u^2)``; party B swapped for cyclic orientation.
    """
    v = np.sqrt(1.0 - u * u)
    kets = [
        np.array([0, 1, 0, 0], dtype=complex),    # up   = |01>
        np.array([0, 0, 1, 0], dtype=complex),    # down = |10>
        np.array([u, 0, 0, v], dtype=complex),    # chi0 = u|00> + v|11>
        np.array([v, 0, 0, -u], dtype=complex),   # chi1 = v|00> - u|11>
    ]
    M = stack_povm([np.outer(k, k.conj()) for k in kets], 2, 2)
    rho = maximally_entangled_state(2)
    return triangle_probability_distribution(rho, rho, rho, M, swap_povm_qubits(M), M)


def check_renou_equations(u: float, tol: float = 1e-12) -> bool:
    """Verify the reference distribution obeys Renou Eqs. (3), (4), (5)."""
    v = np.sqrt(1.0 - u * u)
    p = renou_qubit_reference(u)
    ui, vi = (u, v), (v, -u)  # |00> and |11> coefficients of (chi0, chi1)
    ok = True
    # Eq. (3): perfect zeros P(A=up,B=up) = P(A=down,B=down) = 0 (and cyclic).
    ok &= abs(p[UP, UP, :].sum()) < 1e-9
    ok &= abs(p[DOWN, DOWN, :].sum()) < 1e-9
    ok &= abs(p[:, UP, UP].sum()) < 1e-9
    # Eq. (4): P(chi_i, up, down) = ui^2/8 ; P(chi_i, down, up) = vi^2/8.
    for i in (0, 1):
        ok &= abs(p[CHI0 + i, UP, DOWN] - ui[i] ** 2 / 8) < tol
        ok &= abs(p[CHI0 + i, DOWN, UP] - vi[i] ** 2 / 8) < tol
    # Eq. (5): P(chi_i, chi_j, chi_k) = (ui uj uk + vi vj vk)^2 / 8.
    for i, j, k in itertools.product((0, 1), repeat=3):
        exp = (ui[i] * ui[j] * ui[k] + vi[i] * vi[j] * vi[k]) ** 2 / 8
        ok &= abs(p[CHI0 + i, CHI0 + j, CHI0 + k] - exp) < tol
    return ok


def core_matches_renou(u: float) -> float:
    """Permutation-invariant distance between the core TC qubit and the reference."""
    p_core = cs.token_counting(2, {1: [float(np.arccos(u))]})  # n=1 SO(2) angle, cos = u
    p_ref = renou_qubit_reference(u)
    return float(np.linalg.norm(np.sort(p_core.ravel()) - np.sort(p_ref.ravel())))


# ---------------------------------------------------------------------------
# 2. Color-Matching structure checks.
# ---------------------------------------------------------------------------


def cm_structure_report(dim: int, rng: np.random.Generator) -> dict:
    """Reality, POVM completeness, and H(A) = 2 log2(dim) for a random CM measurement."""
    angles = cs.random_angles("CM", dim, rng)
    M, _ = cs.block_measurement(cs.color_matching_blocks(dim), dim, angles)
    eye = np.eye(dim * dim).reshape(dim, dim, dim, dim)
    H = triangle_entropic_vector(cs.color_matching(dim, angles), label_suffix="")
    return {
        "dim": dim,
        "max_imag": float(np.max(np.abs(M.imag))),
        "completeness": float(np.max(np.abs(M.sum(axis=0) - eye))),
        "HA": H["A"],
        "HA_expected": 2 * np.log2(dim),
    }


# ---------------------------------------------------------------------------
# 3. Survey across dimensions, and the output-scaling fact.
# ---------------------------------------------------------------------------


def survey_row(family: str, dim: int, rng: np.random.Generator) -> dict:
    """Eq. (4) slack at a representative coherent angle choice and decohered."""
    coherent = cs.random_angles(family, dim, rng)
    p_coh = (cs.token_counting if family == "TC" else cs.color_matching)(dim, coherent)
    p_dec = (cs.token_counting if family == "TC" else cs.color_matching)(dim)
    H = cs.entropy_vector(p_coh)
    return {
        "family": family, "dim": dim, "outcomes": p_coh.shape[0],
        "HA": H["A"], "HABC": H["A,B,C"],
        "slack_coherent": cs.eq4_slack(p_coh), "slack_decohered": cs.eq4_slack(p_dec),
    }


if __name__ == "__main__":
    print("=" * 74)
    print("1. PAPER CHECK -- qubit Token-Counting == Renou arXiv:1905.04902 Eq. (2)")
    print("=" * 74)
    for u2 in (0.8, 0.9):
        u = np.sqrt(u2)
        eqs = check_renou_equations(u)
        diff = core_matches_renou(u)
        slack = cs.eq4_slack(cs.token_counting(2, {1: [float(np.arccos(u))]}))
        print(f"  u^2={u2}:  Renou Eqs.(3,4,5) hold: {eqs}   "
              f"core==reference (perm-inv): {diff:.1e}   Eq.(4) slack: {slack:+.4f}")

    print()
    print("=" * 74)
    print("2. STRUCTURE CHECK -- Color-Matching is real, complete, H(A)=2 log2(dim)")
    print("=" * 74)
    rng = np.random.default_rng(0)
    for dim in (3, 4):
        r = cm_structure_report(dim, rng)
        print(f"  dim={dim}:  max|Im(M)|={r['max_imag']:.1e}   "
              f"|sum M - I|={r['completeness']:.1e}   "
              f"H(A)={r['HA']:.4f} (expect {r['HA_expected']:.4f})")
    print(f"  dim=2:  CM degenerate -> classical, Eq.(4) slack = "
          f"{cs.eq4_slack(cs.color_matching(2)):+.4f}")

    print()
    print("=" * 74)
    print("3. SURVEY -- Eq. (4) across dimensions (representative coherent angles)")
    print("=" * 74)
    print(f"{'family':>7}{'dim':>5}{'#out':>7}{'H(A)':>9}{'H(ABC)':>10}"
          f"{'coherent':>11}{'decohered':>11}")
    print("-" * 74)
    rng = np.random.default_rng(1)
    for family in ("TC", "CM"):
        for dim in (3, 4, 5):
            r = survey_row(family, dim, rng)
            print(f"{r['family']:>7}{r['dim']:>5}{r['outcomes']:>7}{r['HA']:>9.4f}"
                  f"{r['HABC']:>10.4f}{r['slack_coherent']:>+11.4f}{r['slack_decohered']:>+11.4f}")
    print("-" * 74)
    print("Output scaling: outcomes/party = dim^2, probability vector = dim^6,")
    print("entropy vector always 7-dim; H(A) = 2 log2(dim) for any full coherent CM.")
