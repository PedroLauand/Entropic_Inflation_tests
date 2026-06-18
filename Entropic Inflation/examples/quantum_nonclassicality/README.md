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
*at the probability level*, for appropriate coherent measurements and at any
source dimension `d`. (Their decohered / "coarse" variants are, by contrast,
classically simulable.) That makes them a clean, parameter-controlled supply of
*known-nonlocal* quantum distributions to test against entropic constraints.

The motivating question: probability-level nonlocality need not show up at the
Shannon-entropic level (the EJM is the canonical example — nonlocal yet
entropically classical). So we generate these strategies at growing dimension
and ask whether any leave an entropic signature (violate Eq. (4) / make the
spiral LP infeasible).

## The two families (parametrised by source dimension `d`)

### Token-Counting (TC) — `triangle_token_counting.py`
Each source distributes `η = d−1` tokens between its two parties (uniform
superposition `Σₖ |k, η−k⟩/√(η+1)`, local dimension `d`). Each party sums the
tokens received from its two sources, `n = k₁+k₂ ∈ {0,…,2η}`, in a measurement
that is **block-diagonal in the total count** with a free rotation inside each
degenerate count sector. Outcomes: `d²` (full) or `2d−1` (bare count). The qubit
case (`η=1`) reproduces Renou et al. Eqs. (3)–(5) exactly.

### Color-Matching (CM) — `triangle_color_matching.py`
Each source is the maximally entangled qudit `|Φ_d⟩ = Σ_c|cc⟩/√d`, so both
connected parties receive the **same colour**. Each party checks whether its two
colours match:
- **coarse**: output the matched colour, else "no-match" → `d+1` outcomes (classically simulable);
- **refined**: also measure the off-diagonal (no-match) subspace coherently → `d²` outcomes (the nonlocal version; the `d≥3` Renou construction is exactly this).

## Pipeline

```
strategy(d)  ->  p(a,b,c)                         [triangle_probability]
             ->  entropy vector H(A),…,H(A,B,C)    [entropic_vector]
             ->  triangle inequalities + Eq. (4)   [entropic_vector / parent LP]
```

## Files

| File | Role |
|------|------|
| `triangle_probability.py` | contract sources + POVMs into `p(a,b,c)` (any dimension) |
| `entropic_vector.py` | `p(a,b,c)` → seven observed entropies + inequality pre-screen |
| `triangle_token_counting.py` | TC family by token number `η` (`describe_tuning(η)` lists knobs); qubit sanity vs arXiv:1905.04902 |
| `triangle_color_matching.py` | CM family by colour number `d` (coarse `d+1` / refined `d²`) |
| `triangle_token_counting_entropy.py` | entropy vector + Eq. (4) for TC; output scaling `d²`/`d⁶` |
| `triangle_renou_dimension_comparison.py` | Eq. (4) slack for qubit/qutrit/ququart |
| `triangle_renou_optimize.py` | minimise the Eq. (4) slack over measurement angles |
| `triangle_ejm_sanity.py` | EJM end-to-end check (entropically classical → feasible) |

## What we have found so far

For the constructions tried (maximally entangled sources, cyclic-symmetric
projective measurements), every generated distribution **satisfies** Eq. (4):
the minimum achievable slack is `0`, reached only at the classical (decohered)
point, with quantum coherence moving strictly inside. So probability-level
nonlocality here leaves *no* Eq. (4) signature — consistent with the EJM lesson.
Open levers: non-maximally-entangled / tunable sources, party-dependent or POVM
measurements, and the full `include_latents=True` spiral LP.

## Running

```bash
cd "Entropic Inflation"
PYTHONPATH=. python examples/quantum_nonclassicality/triangle_color_matching.py
```
