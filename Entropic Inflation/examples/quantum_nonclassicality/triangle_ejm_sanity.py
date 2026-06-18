"""EJM sanity-check example for the triangle network.

The Elegant Joint Measurement (EJM) is the canonical 4-outcome triangle
protocol of Renou, Baeumer, Boreiri, Brunner, Gisin, and Beigi (PRL 123,
140401 (2019)). With each of the three sources prepared in the
maximally entangled state :math:`|\\Phi^+\\rangle` and each party
performing the EJM POVM, the joint probability has the closed form

.. math::
    p(a, b, c) = \\frac{|\\,\\mathrm{Tr}(m_a m_b m_c)\\,|^2}{16^3},

where :math:`m_x = N_x \\cdot \\Psi`, the :math:`N_x` are the four
"native" EJM matrices, and :math:`\\Psi = i\\sigma_y`.

This file serves two purposes.

1. *Sanity check.* It constructs the EJM POVM in the generic
   ``(rho, M_A, M_B, M_C)`` framework of :mod:`triangle_probability` and
   verifies that our :func:`triangle_probability_distribution` reproduces
   the upstream closed-form distribution at machine precision.
2. *Worked example.* It demonstrates the full entropic-nonclassicality
   pipeline using the generic helpers in :mod:`entropic_vector`:
   probability tensor :math:`\\to` entropic vector :math:`\\to` quick
   inequality pre-screen :math:`\\to` latent-inclusive spiral LP. The EJM
   distribution is the canonical *negative-control* for the entropic
   approach --- it is nonclassical at the probability level but classical
   at the Shannon-entropic level --- so the pipeline reports the
   spiral LP as feasible (no witness).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from triangle_probability import (
    maximally_entangled_state,
    swap_povm_qubits,
    triangle_probability_distribution,
)
from entropic_vector import (
    check_triangle_inequalities,
    triangle_entropic_vector,
)


# ---------------------------------------------------------------------------
# Upstream EJM matrices and closed-form probability formula.
# ---------------------------------------------------------------------------


EJM_NATIVE: np.ndarray = np.array(
    [
        [[-1 - 1j, 0 + 0j], [-2j, -1 + 1j]],
        [[1 - 1j, 2j], [0 + 0j, 1 + 1j]],
        [[-1 + 1j, 2j], [0 + 0j, -1 - 1j]],
        [[1 + 1j, 0 + 0j], [-2j, 1 - 1j]],
    ],
    dtype=complex,
)
PSI: np.ndarray = np.array([[0, 1], [-1, 0]], dtype=complex)
EJM_MATRICES: np.ndarray = np.einsum("xij,jk->xik", EJM_NATIVE, PSI)


def upstream_probability_tensor() -> np.ndarray:
    """Evaluate :math:`p(a, b, c) = |\\mathrm{Tr}(m_a m_b m_c)|^2 / 16^3`
    over all 64 outcomes.
    """
    p = np.empty((4, 4, 4), dtype=float)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                amp = np.trace(EJM_MATRICES[a] @ EJM_MATRICES[b] @ EJM_MATRICES[c])
                p[a, b, c] = float(np.abs(amp) ** 2 / 16**3)
    return p


# ---------------------------------------------------------------------------
# EJM POVM and the three triangle-aligned POVMs (M_A, M_B, M_C).
# ---------------------------------------------------------------------------


def ejm_povm() -> np.ndarray:
    """Return the EJM POVM as a 5-index tensor of shape ``(4, 2, 2, 2, 2)``.

    Each POVM element is the rank-1 projector
    ``|psi_x><psi_x| / ||psi_x||^2`` with
    ``|psi_x> = (m_x \\otimes I)|Phi^+>``, where the second qubit (the one
    the identity acts on) is the alphabetical-first subsystem of the
    party that holds this POVM.
    """
    psi = EJM_MATRICES.reshape(4, 4) / np.sqrt(2)
    norms = np.einsum("xi,xi->x", psi.conj(), psi).real
    projectors = np.einsum("xi,xj->xij", psi, psi.conj())
    M_2d = projectors / norms[:, None, None]
    return M_2d.reshape(4, 2, 2, 2, 2)


def ejm_triangle_povms() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(M_A, M_B, M_C)`` for the EJM protocol on the triangle.

    The upstream loop-trace formula ``Tr(m_a m_b m_c)`` implicitly
    orientates each ``m_x`` as a 2-qubit transfer matrix going around
    the cyclic loop ``A -> B -> C -> A``. Translating that orientation
    into our alphabetical-by-source convention requires swapping the two
    qubit subsystems of party B's POVM (and leaving A and C alone).
    """
    M = ejm_povm()
    return M, swap_povm_qubits(M), M


def ejm_triangle_inputs() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Return ``(rho_alpha, rho_beta, rho_gamma, M_A, M_B, M_C)`` ready
    to feed into :func:`triangle_probability_distribution`.
    """
    rho = maximally_entangled_state(2)
    M_A, M_B, M_C = ejm_triangle_povms()
    return rho, rho, rho, M_A, M_B, M_C


# ---------------------------------------------------------------------------
# Sanity check + entropic-pipeline demo.
# ---------------------------------------------------------------------------


def _verify_povm() -> None:
    flat = ejm_povm().reshape(4, 4, 4)
    summed = flat.sum(axis=0)
    err = float(np.max(np.abs(summed - np.eye(4))))
    assert err < 1e-10, f"EJM POVM does not sum to identity (err = {err:.3e})"
    print(f"POVM completeness:  |Sum_x M_x - I|_max = {err:.3e}")


def _sanity_check_probability() -> np.ndarray:
    p_upstream = upstream_probability_tensor()
    p_ours = triangle_probability_distribution(*ejm_triangle_inputs())
    err = float(np.max(np.abs(p_ours - p_upstream)))
    print(f"Probability match:  max|p_ours - p_upstream| = {err:.3e}")
    assert err < 1e-10, "Probability tensor disagrees with upstream EJM formula."
    return p_ours


def _entropic_pipeline(p: np.ndarray) -> None:
    H = triangle_entropic_vector(p)
    print()
    print("EJM entropic vector (bits):")
    for label, value in H.items():
        print(f"  H({label}) = {value:.8f}")

    print()
    print("Pre-screen against known Shannon-type inequalities:")
    for name, slack, ok in check_triangle_inequalities(H):
        flag = "OK     " if ok else "VIOLATED"
        print(f"  [{flag}] {name}: slack = {slack:+.6f}")

    print()
    print("Latent-inclusive spiral LP (this can take a minute or two)...")
    from entropic_inflation import InflationLP, triangle_spiral_problem

    lp = InflationLP(triangle_spiral_problem(), include_latents=True)
    lp.set_values(H)
    result = lp.solve_result(include_farkas_certificate=True)
    if result.is_feasible:
        print("  Result: feasible (no entropic-level witness).")
    else:
        print("  Result: INFEASIBLE -- entropic-level nonclassicality certified.")
        try:
            cert = lp.certificate_as_string(chop_tol=1e-9, round_decimals=4)
            print(f"  Farkas witness: {cert}")
        except Exception as exc:  # pragma: no cover
            print(f"  (could not format witness: {exc})")


if __name__ == "__main__":
    _verify_povm()
    p = _sanity_check_probability()
    _entropic_pipeline(p)
