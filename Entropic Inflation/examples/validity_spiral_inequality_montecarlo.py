"""Monte-Carlo validity test of the spiral inequality, Eq. (4) of the notes.

This script is deliberately self-contained: it does NOT use the inflation
package at all. It samples genuine classical triangle distributions from
scratch and checks that Eq. (4) is never violated. That makes it an
independent witness that the LP-derived cut is actually valid for the
triangle, rather than an artifact of the LP construction.

A classical triangle model has three INDEPENDENT latent sources and one local
response function per party, following the triangle DAG (a -> B,C; b -> A,C;
c -> A,B), so with (alpha, beta, gamma) = (a, b, c):

    A = f_A(beta, gamma),   B = f_B(alpha, gamma),   C = f_C(alpha, beta).

If Eq. (4) is valid then
    7[H(AB)+H(AC)+H(BC)] - 8[H(A)+H(B)+H(C)] - 5 H(ABC) >= 0
must hold for every such distribution.

Run with plain Python (no MOSEK needed):

    python examples/validity_spiral_inequality_montecarlo.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(12345)


def shannon(p: np.ndarray, base: float = 2.0) -> float:
    flat = np.asarray(p, float).ravel()
    nz = flat[flat > 1e-15]
    return float(-np.sum(nz * (np.log(nz) / np.log(base))))


def spiral_slack(P: np.ndarray) -> float:
    A = shannon(P.sum(axis=(1, 2)))
    B = shannon(P.sum(axis=(0, 2)))
    C = shannon(P.sum(axis=(0, 1)))
    AB = shannon(P.sum(axis=2))
    AC = shannon(P.sum(axis=1))
    BC = shannon(P.sum(axis=0))
    ABC = shannon(P)
    return 7 * (AB + AC + BC) - 8 * (A + B + C) - 5 * ABC


def random_dist(n: int) -> np.ndarray:
    x = rng.random(n)
    return x / x.sum()


def random_response(card_out: int, card_in1: int, card_in2: int, deterministic: bool) -> np.ndarray:
    """Response table R[o, i1, i2] = P(out = o | in1, in2)."""
    R = np.zeros((card_out, card_in1, card_in2))
    for i1 in range(card_in1):
        for i2 in range(card_in2):
            if deterministic:
                R[rng.integers(card_out), i1, i2] = 1.0
            else:
                R[:, i1, i2] = random_dist(card_out)
    return R


def sample_triangle(max_source_card: int = 3, max_out: int = 3, deterministic: bool = False) -> np.ndarray:
    ca, cb, cg = (rng.integers(1, max_source_card + 1) for _ in range(3))  # |alpha|,|beta|,|gamma|
    oa, ob, oc = (rng.integers(2, max_out + 1) for _ in range(3))          # |A|,|B|,|C|
    Pa, Pb, Pg = random_dist(ca), random_dist(cb), random_dist(cg)
    RA = random_response(oa, cb, cg, deterministic)   # A = f(beta, gamma)
    RB = random_response(ob, ca, cg, deterministic)   # B = f(alpha, gamma)
    RC = random_response(oc, ca, cb, deterministic)   # C = f(alpha, beta)
    P = np.zeros((oa, ob, oc))
    for al in range(ca):
        for be in range(cb):
            for ga in range(cg):
                w = Pa[al] * Pb[be] * Pg[ga]
                if w:
                    P += w * np.einsum("a,b,c->abc", RA[:, be, ga], RB[:, al, ga], RC[:, al, be])
    return P


def run(n_samples: int, deterministic: bool, label: str) -> float:
    worst = np.inf
    n_viol = 0
    for _ in range(n_samples):
        s = spiral_slack(sample_triangle(deterministic=deterministic))
        worst = min(worst, s)
        n_viol += s < -1e-9
    print(f"{label}: samples={n_samples}  min slack={worst:+.6f}  violations={n_viol}")
    return worst


if __name__ == "__main__":
    print("Eq. (4): 7[H(AB)+H(AC)+H(BC)] - 8[H(A)+H(B)+H(C)] - 5 H(ABC) >= 0")
    print("(slack must stay >= 0 for every genuine triangle distribution)\n")
    w1 = run(200_000, deterministic=True, label="deterministic responses")
    w2 = run(200_000, deterministic=False, label="stochastic responses   ")
    overall = min(w1, w2)
    print(f"\nOVERALL min slack = {overall:+.6f}")
    print("RESULT:", "VALID (no violation found)" if overall >= -1e-9 else "INVALID -- violation found!")
