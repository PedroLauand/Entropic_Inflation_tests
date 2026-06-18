"""Joint probability distribution :math:`p(a, b, c)` for the triangle network.

This is the contraction engine of the strategy generator in this folder: given
the three source states and the three party POVMs of a Token-Counting or
Color-Matching strategy (at any dimension), it returns the observed
distribution ``p(a, b, c)``. See the folder README for the strategy families.

Triangle network conventions
----------------------------
Three observed parties ``A, B, C`` and three latent sources ``alpha, beta,
gamma``. The connectivity is

* ``alpha`` is a common cause for ``B`` and ``C``,
* ``beta``  is a common cause for ``A`` and ``C``,
* ``gamma`` is a common cause for ``A`` and ``B``.

Each source distributes a bipartite quantum state to its two children;
each party performs a POVM on the two subsystems it receives.

Tensor layout
-------------
Each state and each POVM element is a 4-index tensor whose first two
indices are bra (row) and last two indices are ket (column), with the two
subsystems on which it acts listed in alphabetical order of source name:

* ``rho_alpha`` : ``(alpha_B^bra, alpha_C^bra, alpha_B^ket, alpha_C^ket)``
* ``rho_beta``  : ``(beta_A^bra,  beta_C^bra,  beta_A^ket,  beta_C^ket)``
* ``rho_gamma`` : ``(gamma_A^bra, gamma_B^bra, gamma_A^ket, gamma_B^ket)``
* ``M_A[x]``    : ``(beta_A^bra,  gamma_A^bra, beta_A^ket,  gamma_A^ket)``
* ``M_B[y]``    : ``(alpha_B^bra, gamma_B^bra, alpha_B^ket, gamma_B^ket)``
* ``M_C[z]``    : ``(alpha_C^bra, beta_C^bra,  alpha_C^ket, beta_C^ket)``

Probability formula
-------------------
The joint probability of outcomes ``(a, b, c)`` is

.. math::
    p(a, b, c)
    = \\operatorname{Tr}\\!\\bigl[(M_A^a \\otimes M_B^b \\otimes M_C^c)
        (\\rho_\\alpha \\otimes \\rho_\\beta \\otimes \\rho_\\gamma)\\bigr]

with the natural reordering of subsystems so that each measurement acts
on its two subsystems. The whole expression collapses to a single tensor
contraction performed below by :func:`numpy.einsum`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def triangle_probability_distribution(
    rho_alpha: np.ndarray,
    rho_beta: np.ndarray,
    rho_gamma: np.ndarray,
    M_A: np.ndarray,
    M_B: np.ndarray,
    M_C: np.ndarray,
) -> np.ndarray:
    """Return the joint probability tensor ``p[a, b, c]``.

    Parameters
    ----------
    rho_alpha, rho_beta, rho_gamma : ndarray
        Bipartite density matrices, each a 4-index tensor in the layout
        described in the module docstring.
    M_A, M_B, M_C : ndarray
        Stacked POVM elements; each is a 5-index tensor whose first axis
        indexes the outcome.

    Returns
    -------
    ndarray
        Real probability tensor of shape ``(n_a, n_b, n_c)`` where ``n_X``
        is the number of outcomes of party ``X``.
    """
    # Index letters (lowercase = ket, uppercase = bra; same letter on
    # the bra of one operator and the ket of the other = contraction):
    #   alpha_B: a / A          beta_A : c / C          gamma_A: e / E
    #   alpha_C: b / B          beta_C : d / D          gamma_B: f / F
    # optimize=True picks a good contraction path; essential at higher source
    # dimension, where the naive path cost grows like d^12 * (n_a n_b n_c).
    p = np.einsum(
        "ABab,CDcd,EFef,xceCE,yafAF,zbdBD->xyz",
        rho_alpha, rho_beta, rho_gamma,
        M_A, M_B, M_C,
        optimize=True,
    )
    if np.iscomplexobj(p):
        # Tr of (Hermitian) x (positive) is real up to numerical noise.
        if np.max(np.abs(p.imag)) > 1e-9:
            raise ValueError(
                f"Probability tensor has significant imaginary part "
                f"(max |imag| = {np.max(np.abs(p.imag)):.3e}); check that "
                "the inputs are Hermitian positive operators."
            )
        p = p.real
    return p


# ---------------------------------------------------------------------------
# Helpers for assembling the tensors from more conventional matrix inputs.
# ---------------------------------------------------------------------------


def reshape_state(rho_matrix: np.ndarray, d1: int, d2: int) -> np.ndarray:
    """Reshape a ``(d1*d2, d1*d2)`` bipartite density matrix into a
    ``(d1, d2, d1, d2)`` tensor with the (bra, bra, ket, ket) layout.
    """
    rho_matrix = np.asarray(rho_matrix)
    expected = (d1 * d2, d1 * d2)
    if rho_matrix.shape != expected:
        raise ValueError(
            f"rho_matrix shape {rho_matrix.shape} does not match expected {expected}."
        )
    return rho_matrix.reshape(d1, d2, d1, d2)


def stack_povm(elements: Sequence[np.ndarray], d1: int, d2: int) -> np.ndarray:
    """Stack a sequence of ``(d1*d2, d1*d2)`` POVM elements into a
    ``(n, d1, d2, d1, d2)`` tensor.
    """
    return np.stack([reshape_state(M, d1, d2) for M in elements])


def swap_povm_qubits(M_stack: np.ndarray) -> np.ndarray:
    """Swap the two qubit subsystems of a stacked POVM tensor.

    Convenience for re-orientating a POVM whose two subsystems are stored
    in the opposite order from this module's alphabetical convention.
    """
    return M_stack.transpose(0, 2, 1, 4, 3)


# ---------------------------------------------------------------------------
# Stock ingredients commonly useful in examples.
# ---------------------------------------------------------------------------


def maximally_entangled_state(d: int) -> np.ndarray:
    """Return :math:`|\\Phi^+\\rangle\\langle\\Phi^+|` on two ``d``-dim
    subsystems as a 4-index tensor of shape ``(d, d, d, d)``.
    """
    psi = np.eye(d).reshape(d * d) / np.sqrt(d)
    return np.outer(psi, psi.conj()).reshape(d, d, d, d)


def computational_basis_povm(d1: int, d2: int) -> np.ndarray:
    """Product computational-basis projective POVM with ``d1*d2`` outcomes."""
    M = np.zeros((d1 * d2, d1, d2, d1, d2))
    for i in range(d1):
        for j in range(d2):
            M[i * d2 + j, i, j, i, j] = 1.0
    return M
