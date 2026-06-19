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

- **TC**: the qubit case (`dim=2`) reproduces Renou et al. Eqs. (3)–(5) exactly.
- **CM**: rotations are **real** (the papers fix the off-diagonal coefficients real; the basis is otherwise free). The matched-colour outcomes `|cc⟩` are reported verbatim; coherence lives in the off-diagonal mismatch blocks. CM is coherent only for `dim ≥ 3` — at `dim=2` the mismatch blocks are 1-dimensional, so CM reduces to the classical computational measurement.

## Quick start

```python
import numpy as np
from coherent_strategies import (
    token_counting, color_matching,   # strategy  -> p(a,b,c)
    entropy_vector, eq4_slack,        # p(a,b,c)  -> 7 entropies / Eq. (4) slack
    describe, angle_layout, n_angles, random_angles,
)

# Token-Counting qubit (reproduces Renou): one SO(2) angle in the count-1 sector
p = token_counting(dim=2, angles={1: [np.arccos(np.sqrt(0.8))]})
entropy_vector(p)        # {'A': 2.0, 'A,B': 3.5871, ..., 'A,B,C': 3.9119}
eq4_slack(p)             # +7.7688   (>= 0: Eq. (4) satisfied)

# Color-Matching qutrit with random real off-diagonal rotations
rng = np.random.default_rng(0)
p = color_matching(dim=3, angles=random_angles('CM', 3, rng))
eq4_slack(p)             # some value >= 0

describe('CM', 3)        # prints the source + rotatable sectors and #angles each
```

`angles` is a dict `{sector_label: [Givens angles]}`; omitted sectors default to
the identity (decohered baseline). The labels are the counts `n` for TC and
`'upper'` / `'lower'` for CM — use `angle_layout(family, dim)` to see them.

| call | returns |
|------|---------|
| `token_counting(dim, angles=None)` | `p[a,b,c]` (TC; `angles=None` → decohered) |
| `color_matching(dim, angles=None)` | `p[a,b,c]` (CM; real rotations) |
| `entropy_vector(p)` | dict of the seven entropies `H(A),…,H(A,B,C)` |
| `eq4_slack(p)` | spiral inequality Eq. (4) slack (`≥ 0` = satisfied) |
| `describe(family, dim)` | print the source + rotatable sectors |
| `angle_layout(family, dim)` | list of `(label, sector_dim, n_angles)` |
| `n_angles(family, dim)` | total number of free angles |
| `random_angles(family, dim, rng)` | a random `angles` dict for exploration |

The two core builders underneath are `givens_rotation(m, angles)` (real `SO(m)`)
and `block_measurement(blocks, dim, angles)` (the block-diagonal POVM); `family`
is `"TC"` or `"CM"`.

## Pipeline

```
token_counting / color_matching (dim, angles)  ->  p(a,b,c)     [coherent_strategies]
                                               ->  entropy vector H(A),…,H(A,B,C)   [entropic_vector]
                                               ->  triangle inequalities + Eq. (4)  [entropic_vector / parent LP]
```

## Walkthrough: building `p(a,b,c)` from scratch

This reconstructs every distribution from first principles and names the exact
function for each step, so you can audit the code rather than trust it.
Everything lives in four small files: `triangle_probability.py` (engine),
`coherent_strategies.py` (the strategies), `entropic_vector.py` (entropies),
`validate_strategies.py` (independent checks).

### Step 0 — the network and the Born rule
Three parties `A, B, C`, three sources `α, β, γ`, wired so each source feeds two
parties and each party is fed by two sources:

```
α → (B, C)        A measures (β, γ)
β → (A, C)        B measures (α, γ)
γ → (A, B)        C measures (α, β)
```

Source `s` emits a bipartite state `ρ_s` (one half to each child); party `X`
applies a POVM `{M_X^x}` to its two halves. The observed distribution is

```
p(a,b,c) = Tr[ (M_A^a ⊗ M_B^b ⊗ M_C^c) (ρ_α ⊗ ρ_β ⊗ ρ_γ) ].
```

### Step 1 — the contraction engine (`triangle_probability.py`)
There are six wires (source–party links): `α_B, α_C, β_A, β_C, γ_A, γ_B`. Each
carries a bra and a ket index (dimension `dim`), shared between the source that
emits it and the party that measures it. `triangle_probability_distribution`
does the whole trace in one `einsum`:

```
"ABab, CDcd, EFef, xceCE, yafAF, zbdBD -> xyz"
  ρ_α   ρ_β   ρ_γ   M_A    M_B    M_C
```

Index dictionary (lowercase = ket, uppercase = bra):

| wire | letters | appears in source | appears in party |
|---|---|---|---|
| α_B | a / A | ρ_α | M_B |
| α_C | b / B | ρ_α | M_C |
| β_A | c / C | ρ_β | M_A |
| β_C | d / D | ρ_β | M_C |
| γ_A | e / E | ρ_γ | M_A |
| γ_B | f / F | ρ_γ | M_B |

Each wire's two letters occur exactly twice — once in its source, once in its
party — so summing them *is* the trace above. You can verify the wiring against
the DAG by reading each operator: `M_A = xceCE` touches `β_A (c,C)` and
`γ_A (e,E)` — exactly A's two sources; `ρ_α = ABab` touches `α_B (A,a)` and
`α_C (B,b)` — exactly α's two children. `x,y,z` are the free outputs → `p[x,y,z]`.
Each `ρ_s`, `M_X^x` is a 4-index `(bra, bra, ket, ket)` tensor from
`reshape_state` / `stack_povm`, subsystems in alphabetical source order.

### Step 2 — the measurement (`coherent_strategies.py`)
Every measurement is built identically:

1. `givens_rotation(m, angles)` — a real `SO(m)` matrix, a product of plane
   rotations with one angle per `(i<j)` plane (`m(m-1)/2` total). For `m=2`,
   angle θ: `[[cosθ, −sinθ], [sinθ, cosθ]]`.
2. `block_measurement(blocks, dim, angles)` — `blocks` is an ordered list of
   `(label, [computational indices])` that **partition** `range(dim²)`. For each
   block of size `m`, take `O = givens_rotation(m, angles[label])` (identity if
   no angle supplied) and emit `m` kets = the columns of `O` placed on that
   block's indices; the POVM element is `|ket⟩⟨ket|`.

Completeness is automatic: the blocks partition the space and each `O` is
orthonormal, so `Σ M = I` (checked in `validate_strategies.py`).

### Step 3 — Token Counting
- **Source** `uniform_token_source(dim)`: `(Σₖ |k, η−k⟩)/√dim`, `η = dim−1`. For
  `dim=2` this is `(|01⟩+|10⟩)/√2` — one token shared between the two recipients.
- **Sectors** `token_counting_blocks(dim)`: the two-register indices grouped by
  total count `n = k₁+k₂`. For `dim=2`: `n=0→{|00⟩}`, `n=1→{|01⟩,|10⟩}`,
  `n=2→{|11⟩}`. The end sectors are frozen (1-dim); the middle ones carry coherence.
- **Qubit example** `token_counting(2, {1:[θ]})`: the `n=1` block gets `SO(2)`
  angle θ → outcomes `cosθ|01⟩+sinθ|10⟩` and `−sinθ|01⟩+cosθ|10⟩`. Setting
  `θ=arccos(u)` reproduces Renou et al. (arXiv:1905.04902 Eq. (2)) up to the
  single-qubit relabeling `I⊗X`. `validate_strategies.py` builds Renou's basis
  `{|01⟩,|10⟩, u|00⟩+v|11⟩, v|00⟩−u|11⟩}` independently and checks both give the
  same `p(a,b,c)` and that it obeys Renou's Eqs. (3),(4),(5).

*Checkable:* `token_counting(2, {1:[arccos√0.8]})` → `H(A)=2.0000,
H(AB)=3.5871, H(ABC)=3.9119`, Eq. (4) slack `+7.7688`; and Renou Eq. (4) gives
`P(χ₀, up, down) = u²/8 = 0.1`.

### Step 4 — Color Matching
- **Source** `maximally_entangled_state(dim)`: `|Φ_d⟩=(Σ_c|cc⟩)/√dim` — both
  connected parties receive the *same* colour.
- **Sectors** `color_matching_blocks(dim)`: `dim` frozen singletons `|cc⟩`
  (report the matched colour) + the upper `{|ij⟩:i<j}` and lower `{|ij⟩:i>j}`
  mismatch blocks, each of size `dim(dim−1)/2`, each given a real `SO(m)` rotation.
- **`H(A)=2log₂dim` exactly**: a party's two registers are each one half of a
  maximally entangled pair (reduced state `I/dim`) from *independent* sources, so
  its joint state is `I/dim²`; every POVM element is rank-1, so all `dim²`
  outcomes are equiprobable. (Independent of the angles — a robust check.)
- **`dim=2` is classical**: the mismatch blocks are 1-dimensional, so there is
  nothing to rotate and CM reduces to the computational measurement.

### Step 5 — the B-swap (do not skip)
The basis vectors are not symmetric under exchanging a party's two registers, so
the construction needs a consistent cyclic `A→B→C` orientation. Party B stores
its two subsystems in reverse order, so its POVM is transposed on those two legs
(`swap_povm_qubits`); `_triangle` contracts `M, swap(M), M`. This is
**load-bearing** — dropping it changes `p(a,b,c)` by up to ~0.075 and breaks the
match with Renou (same fix the EJM example uses).

### Step 6 — entropy vector and Eq. (4) (`entropic_vector.py`)
From `p[a,b,c]`: marginals are axis sums (`p.sum(axis=…)`), and `shannon_entropy`
is `−Σ p log₂ p` over nonzero entries. `triangle_entropic_vector` returns the
seven `H(A),…,H(A,B,C)`; `spiral_slack` evaluates

```
slack = 7[H(AB)+H(AC)+H(BC)] − 8[H(A)+H(B)+H(C)] − 5 H(ABC).
```

### Verify it yourself
```bash
PYTHONPATH=. python examples/quantum_nonclassicality/validate_strategies.py
```
confirms: Renou Eqs. (3),(4),(5) hold; the core TC qubit equals an independent
Renou build to ~1e-16; CM is real, a complete POVM, `H(A)=2log₂dim`. For a
*fully* independent re-derivation, build the six-wire global statevector by hand
(without `triangle_probability`), contract, and compare — that matches to <1e-14.

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
