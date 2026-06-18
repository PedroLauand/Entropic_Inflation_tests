"""Entropy vector of the token-counting triangle distributions, and Eq. (4).

Pipeline: a probability vector ``p(a,b,c)`` (from ``triangle_token_counting``)
-> the seven observed Shannon entropies via ``triangle_entropic_vector`` ->
evaluation of the seven known triangle inequalities and the new spiral
inequality, Eq. (4) of the notes:

    7[H(AB)+H(AC)+H(BC)] - 8[H(A)+H(B)+H(C)] - 5 H(ABC)  >= 0.

Key scaling fact: the *probability vector* grows fast with the source
dimension d = eta+1 (d^2 outcomes per party, d^6 entries in total), but the
*entropy vector* is always the same 7 numbers. The Shannon-entropic test
therefore compresses an ever-larger distribution into a fixed 7-dim check.

Run from the package root:

    PYTHONPATH=. python examples/quantum_nonclassicality/triangle_token_counting_entropy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from triangle_token_counting import (
    renou_qubit_distribution,
    renou_qutrit_distribution,
    token_counting_distribution,
)
from entropic_vector import (
    SPIRAL_INEQUALITY,
    check_triangle_inequalities,
    triangle_entropic_vector,
)


def entropy_vector(p: np.ndarray) -> dict[str, float]:
    """The seven observed entropies of p(a,b,c), with plain A,B,C labels."""
    return triangle_entropic_vector(p, label_suffix="")


def report_entropy(p: np.ndarray, name: str) -> None:
    """Print the entropy vector and every triangle inequality slack for p."""
    H = entropy_vector(p)
    print(f"--- {name} ---")
    print(f"  outcomes per party : {p.shape[0]}   (probability vector: {p.size} entries)")
    print("  entropy vector (bits):")
    for label in ("A", "B", "C", "A,B", "A,C", "B,C", "A,B,C"):
        print(f"    H({label:5s}) = {H[label]:.6f}")
    print("  inequality slacks (>= 0 means satisfied):")
    for ineq_name, slack, ok in check_triangle_inequalities(H, label_suffix=""):
        tag = "satisfied" if ok else "VIOLATED"
        marker = "  <-- Eq. (4)" if ineq_name == SPIRAL_INEQUALITY[0] else ""
        print(f"    {ineq_name:8s}: slack = {slack:+.6f}   {tag}{marker}")


def scaling_table(max_eta: int = 5) -> None:
    """Confirm how the output size scales with the source dimension d = eta+1."""
    print("Output scaling with source dimension (uniform source, decohered measurement):")
    print(f"  {'eta':>3} {'d':>3} {'outcomes/party':>15} {'prob-vector size':>17} "
          f"{'entropy-vec dim':>15} {'max H(X)=2log2 d':>17}")
    for eta in range(1, max_eta + 1):
        p, _ = token_counting_distribution(eta)  # identity angles = decohered baseline
        d = eta + 1
        print(f"  {eta:>3} {d:>3} {p.shape[0]:>15} {p.size:>17} {7:>15} "
              f"{2*np.log2(d):>17.4f}")
    print("  (outcomes/party = d^2, prob-vector size = d^6; the entropy vector stays 7-dim.)")


if __name__ == "__main__":
    print("=" * 72)
    scaling_table(max_eta=5)
    print()
    print("=" * 72)
    print("Renou strategy: qubit vs qutrit -- entropy vector & Eq.(4)")
    print("=" * 72)
    report_entropy(renou_qubit_distribution(np.sqrt(0.8)), "Renou qubit (d=2, u^2=0.8)")
    print()
    report_entropy(renou_qutrit_distribution(), "Renou qutrit (d=3, DFT off-diagonal)")
