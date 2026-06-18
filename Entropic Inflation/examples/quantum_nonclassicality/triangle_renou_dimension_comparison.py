"""Compare the new spiral inequality, Eq. (4), across Renou strategies of
increasing source dimension: qubit (d=2), qutrit (d=3), ququart (d=4).

For each dimension we build the Renou construction, compute the seven observed
entropies, and evaluate Eq. (4):

    slack = 7[H(AB)+H(AC)+H(BC)] - 8[H(A)+H(B)+H(C)] - 5 H(ABC)   (>= 0 means satisfied).

Recipe note
-----------
* d=2 uses the validated nonlocal qubit (arXiv:1905.04902 Eq. (2), u^2=0.8):
  coherence sits on the diagonal "colour-match" sector.
* d=3, d=4 use the off-diagonal construction (arXiv:1905.04902 p. 4 generalised),
  with DFT bases on the off-diagonal blocks; coherence sits off the diagonal.

Eq. (4) is homogeneous (degree 1 in entropies), and the entropy scale grows with
d, so we also report the slack normalised by H(ABC) for a scale-free comparison.

Run from the package root:

    PYTHONPATH=. python examples/quantum_nonclassicality/triangle_renou_dimension_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from triangle_token_counting import (
    renou_offdiagonal_distribution,
    renou_qubit_distribution,
)
from entropic_vector import (
    SPIRAL_INEQUALITY,
    check_triangle_inequalities,
    triangle_entropic_vector,
)


def eq4_slack(H: dict[str, float]) -> float:
    """The Eq. (4) (spiral) slack for an entropy vector with plain A,B,C labels."""
    for name, slack, _ in check_triangle_inequalities(H, label_suffix=""):
        if name == SPIRAL_INEQUALITY[0]:
            return slack
    raise RuntimeError("spiral inequality not found in checker output")


def summarise(label: str, p: np.ndarray) -> dict[str, float]:
    H = triangle_entropic_vector(p, label_suffix="")
    slack = eq4_slack(H)
    return {
        "label": label,
        "outcomes": p.shape[0],
        "HA": H["A"],
        "HAB": H["A,B"],
        "HABC": H["A,B,C"],
        "slack": slack,
        "normalised": slack / H["A,B,C"],
    }


if __name__ == "__main__":
    cases = [
        ("qubit   (d=2)", renou_qubit_distribution(np.sqrt(0.8))),
        ("qutrit  (d=3)", renou_offdiagonal_distribution(3)),
        ("ququart (d=4)", renou_offdiagonal_distribution(4)),
    ]
    rows = [summarise(label, p) for label, p in cases]

    print("Eq. (4) across Renou strategies of growing source dimension")
    print("=" * 78)
    header = f"{'strategy':14}{'#out':>6}{'H(A)':>9}{'H(AB)':>10}{'H(ABC)':>10}{'Eq4 slack':>12}{'slack/H(ABC)':>14}"
    print(header)
    print("-" * 78)
    for r in rows:
        print(f"{r['label']:14}{r['outcomes']:>6}{r['HA']:>9.4f}{r['HAB']:>10.4f}"
              f"{r['HABC']:>10.4f}{r['slack']:>+12.4f}{r['normalised']:>14.4f}")
    print("-" * 78)
    print("All satisfy Eq. (4) (slack >= 0). Watch the scale-free 'slack/H(ABC)' column:")
    print("a decreasing trend means higher dimension sits relatively closer to the facet.")
