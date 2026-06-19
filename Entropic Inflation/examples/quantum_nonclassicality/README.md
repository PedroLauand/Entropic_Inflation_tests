# Quantum strategy generator for the triangle

This folder is a **generic, dimension-scalable generator of quantum triangle
distributions that are known to be nonlocal at the probability level.** It
produces `p(a,b,c)` from two strategy families and feeds their entropy vectors
into the classical-triangle entropic tests of the parent package (the seven
triangle inequalities and the spiral inequality, Eq. (4)).

The two families come from
[Renou & Beigi, arXiv:2202.00905](https://arxiv.org/abs/2202.00905)
("Network Nonlocality via Rigidity of Token-Counting and Color-Matching"), and
the qubit case coincides with the triangle distribution of
[Renou et al., PRL 123, 140401 (2019), arXiv:1905.04902](https://arxiv.org/abs/1905.04902).

## Why these strategies

By the **rigidity** theorems of those papers, both families contain members that
**cannot be reproduced by any classical (trilocal) model** — they are nonlocal
*at the probability level*, for appropriate coherent measurements. (Their
decohered baselines are, by contrast, classically simulable.) That makes them a
clean, parameter-controlled supply of *known-nonlocal* quantum distributions to
test against entropic constraints.

The motivating question: probability-level nonlocality need not show up at the
Shannon-entropic level (the EJM is the canonical example — nonlocal yet
entropically classical). So we generate these strategies at growing dimension
and ask whether any leave an entropic signature (violate Eq. (4) / make the
spiral LP infeasible).

## One construction, two families

Both families are the **same** object, implemented once in
`coherent_strategies.py`. Each party holds two `dim`-dimensional registers; the
`dim²`-dimensional space splits into **sectors** fixed by a classical label, and
inside every multi-dimensional sector the measurement basis is a free **real**
`SO(m)` rotation (Givens angles). Frozen 1-dim sectors carry no freedom;
identity rotations (`angles=None`) give the decohered/classical baseline, and
non-trivial angles inject the coherence that makes the strategy nonlocal.

Only the **source** and the **sector partition** differ:

| | source | sectors |
|---|---|---|
| **Token-Counting (TC)** | uniform token state `Σₖ\|k, η−k⟩/√dim` (`η = dim−1`) | grouped by total count `n = k₁+k₂` |
| **Color-Matching (CM)** | maximally entangled `\|Φ_d⟩ = Σ_c\|cc⟩/√dim` | `dim` matched-colour singletons `\|cc⟩` + the upper/lower mismatch blocks |

```python
from coherent_strategies import token_counting, color_matching, describe, random_angles

p = token_counting(dim=2, angles={1: [0.4636]})        # qubit; reproduces Renou Eq. (2)
p = color_matching(dim=3, angles={'upper': [...], 'lower': [...]})   # qutrit, real rotations
describe('CM', 3)                                       # list the rotatable sectors + #angles
```

- **TC**: the qubit case (`dim=2`) reproduces Renou et al. Eqs. (3)–(5) exactly.
- **CM**: rotations are **real** (the papers fix the off-diagonal coefficients real; the basis is otherwise free). The matched-colour outcomes `|cc⟩` are reported verbatim; coherence lives in the off-diagonal mismatch blocks. CM is coherent only for `dim ≥ 3` — at `dim=2` the mismatch blocks are 1-dimensional, so CM reduces to the classical computational measurement.

## Pipeline

```
token_counting / color_matching (dim, angles)  ->  p(a,b,c)     [coherent_strategies]
                                               ->  entropy vector H(A),…,H(A,B,C)   [entropic_vector]
                                               ->  triangle inequalities + Eq. (4)  [entropic_vector / parent LP]
```

## Files

| File | Role |
|------|------|
| `triangle_probability.py` | contract sources + POVMs into `p(a,b,c)` (any dimension) |
| `entropic_vector.py` | `p(a,b,c)` → seven observed entropies + inequality pre-screen |
| `coherent_strategies.py` | **the generator** — `token_counting(dim, angles)` and `color_matching(dim, angles)` on one shared core, with `describe` / `angle_layout` / `random_angles` |
| `validate_strategies.py` | checks vs the papers (Renou Eqs. (3)–(5), CM real/complete, `H(A)=2log₂dim`) + survey across dimensions |
| `optimize_angles.py` | minimise the Eq. (4) slack over the measurement angles (either family) |
| `triangle_ejm_sanity.py` | EJM end-to-end check (entropically classical → feasible) |

## What we have found so far

For the constructions tried (maximally entangled sources, cyclic-symmetric
real-projective measurements), every generated distribution **satisfies** Eq. (4):
the minimum achievable slack is `0`, reached only at the classical (decohered)
point, with quantum coherence moving strictly inside. As source dimension grows,
the coherent CM entropy vector marches along the independent ray `(1,2,3)·H(A)`
with `H(A)=2log₂dim`, while the genuine correlations *saturate* to bounded
constants — so probability-level nonlocality here leaves *no* Eq. (4) signature,
consistent with the EJM lesson. Open levers: non-maximally-entangled / tunable
sources, party-dependent or POVM measurements, and the full `include_latents=True`
spiral LP.

## Running

```bash
cd "Entropic Inflation"
PYTHONPATH=. python examples/quantum_nonclassicality/coherent_strategies.py
PYTHONPATH=. python examples/quantum_nonclassicality/validate_strategies.py
```
