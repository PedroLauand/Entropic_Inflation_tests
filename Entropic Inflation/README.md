# Entropic Inflation

`entropic_inflation` builds and solves the **entropic linear programs** that
arise when the [inflation technique](https://doi.org/10.1515/jci-2017-0020) is
applied to the **entropic** description of a causal network. It is the
companion code to the notes *"The entropic approach for quantum networks"*
(`../overleaf/notes.pdf`).

Where the reference [`inflation`](https://github.com/ecboghiu/inflation)
package works at the level of probabilities, this package works at the level of
**Shannon entropies**: the source-independence constraint that makes the
probability picture non-convex becomes a *linear* equality in entropy space, so
causal-compatibility tests reduce to linear programs (LPs). The package mirrors
the reference API (`InflationProblem` + `InflationLP`) so the two feel familiar.

The headline result it reproduces is a new, fully symmetric entropic inequality
for the triangle network, derived from the *spiral* inflation — Eq. (4) of the
notes:

```
7 [H(AB) + H(AC) + H(BC)]  >=  8 [H(A) + H(B) + H(C)] + 5 H(ABC).
```

---

## What it does

Given a causal DAG and an inflation of it, the package:

1. enumerates the **joint-entropy coordinates** of the inflated variables (one
   per non-empty subset);
2. assembles every linear constraint the inflated structure imposes on those
   entropies (the four families below);
3. builds the resulting LP as a [MOSEK](https://www.mosek.com/) Fusion model and
   solves it;
4. when a candidate entropy vector is **infeasible**, extracts the **Farkas
   dual** — a separating linear inequality (an entropic witness) expressed
   directly in the observed-entropy coordinates.

The four constraint families (see `lp/InflationLP.py::_build_elemental_problem`):

| # | Family | Where it comes from |
|---|--------|---------------------|
| 1 | Elemental Shannon inequalities (every conditional entropy and conditional mutual information `>= 0`) | `lp/lp_utils.py::build_elemental_shannon_cone` |
| 2 | Causal (in)dependencies of the inflated DAG, e.g. `I(A1 ; A2 \| b1,c1) = 0`; for root sources this gives joint source independence | `InflationProblem.inflated_independencies` |
| 3 | Observable factorization equalities `H(∪ blocks) = Σ H(block)` (observed-only mode) | `InflationProblem.factorization_blocks` |
| 4 | Copy-symmetry equalities for identically-distributed copies | `entropic_subset_signature` (manual) / `inflation_permutation_generators` (automatic) |

---

## Requirements

- Python ≥ 3.9
- `numpy`, `scipy`, `networkx`
- `mosek` (with a valid license — the LP backend)
- `sympy` *(optional; only for `about()` version reporting)*
- `pycddlib` *(optional; only for `examples/auxiliar_cdd_triangle_rays.py`)*

The repository ships a virtual environment at `../.venv` that already has these.

## Installation / running

There is no `pip install` step yet; the package is run in place. From **this
directory** (`Entropic Inflation/`), put it on `PYTHONPATH` and use a Python
that has the dependencies:

```bash
# from the repository root, activate the bundled environment …
source .venv/bin/activate
cd "Entropic Inflation"

# … then run any example with the package importable:
PYTHONPATH=. python examples/triangle_spiral_dual_witnesses.py
```

Without activating, point directly at the interpreter:

```bash
cd "Entropic Inflation"
PYTHONPATH=. ../.venv/bin/python examples/triangle_spiral_dual_witnesses.py
```

Check your environment with:

```bash
PYTHONPATH=. python -c "import entropic_inflation as ei; ei.about()"
```

---

## Quickstart

### Feasibility test + entropic witness (the main workflow)

```python
from entropic_inflation import InflationLP, triangle_spiral_problem

# A candidate observed entropy vector (here triangle ray r1, which the
# spiral inflation excludes). Keys are the observed marginal A0,B0,C0.
ray = {
    "A0": 1.0, "B0": 1.5, "C0": 1.5,
    "A0,B0": 2.0, "A0,C0": 2.0, "B0,C0": 2.5,
    "A0,B0,C0": 3.0,
}

lp = InflationLP(triangle_spiral_problem(), include_latents=True)
lp.set_values(ray)
result = lp.solve_result(include_farkas_certificate=True)

print("feasible?", result.is_feasible)          # -> False
print(lp.certificate_as_string())               # -> the separating inequality
```

`certificate_as_string()` normalises to the smallest non-zero coefficient:

```
-1.6*H(A0) + 1.4*H(A0,B0) - 1*H(A0,B0,C0) + 1.4*H(A0,C0) - 1.6*H(B0) + 1.4*H(B0,C0) - 1.6*H(C0) >= 0
```

Multiplying through by 5 gives the integer form, i.e. Eq. (4) (this is exactly
what `examples/triangle_spiral_dual_witnesses.py` reports):

```
-8*H(A0) + 7*H(A0,B0) - 5*H(A0,B0,C0) + 7*H(A0,C0) - 8*H(B0) + 7*H(B0,C0) - 8*H(C0) >= 0
```

### Optimising a linear functional over the cone

```python
from entropic_inflation import InflationLP, triangle_problem

lp = InflationLP(triangle_problem(), include_latents=True)
# minimise the LHS of triangle inequality "type 1": H(AB)+H(AC)-H(A)-H(B)-H(C)
lp.set_objective(
    {"A,B": 1.0, "A,C": 1.0, "A": -1.0, "B": -1.0, "C": -1.0},
    direction="min",
)
print(lp.solve_result().objective_value)   # -> ~0  (inequality is valid)
```

---

## User interface

### `InflationProblem`

Encodes the scenario DAG and its inflation. Two inflation modes:

- **automatic** — give an `inflation_level_per_source` and the inflated DAG /
  copy bookkeeping is generated for you (mirrors the reference package);
- **manual** — give the explicit `inflation_dag`; node names are parsed into a
  base variable + a source-copy token (this is how the spiral is defined).

```python
from entropic_inflation import InflationProblem

problem = InflationProblem(
    dag={"a": ["B", "C"], "b": ["A", "C"], "c": ["A", "B"]},  # sources lowercase
    outcomes_per_party=(2, 2, 2),
    settings_per_party=(1, 1, 1),
    inflation_mode="manual",
    inflation_dag={
        "a0": ["B0", "C0", "C1"], "a1": ["B1"],
        "b0": ["A0", "A1", "C0"], "b1": ["C1"],
        "c0": ["A0", "B0", "B1"], "c1": ["A1"],
    },
    order=("A", "B", "C"),
)
```

Useful members:

| Member | Meaning |
|--------|---------|
| `inflation_mode` | `"automatic"` or `"manual"` |
| `public_node_labels(include_latents=False)` | readable inflated node names |
| `entropic_node_labels(include_latents=False)` | the LP variable labels |
| `inflated_independencies(include_latents)` | family-2 constraints |
| `inflation_permutation_generators(include_latents)` | family-4 generators (automatic) |
| `entropic_subset_signature(labels)` | canonical key for family-4 (manual) |
| `factorization_blocks(labels)` | disconnected inflation supports (family 3) |

### `InflationLP`

```python
InflationLP(problem, *, include_latents=True,
            candidate=None, objective=None, objective_sense="min")
```

- `include_latents=True` (**the default**) keeps source/latent entropy
  coordinates in the LP — the authoritative spiral construction. Passing
  `include_latents=False` keeps only observed coordinates and relies on
  factorization equalities + hand-supplied equalities (weaker; see *Status*
  below), and is used only by the historical observed-only examples.

Building / querying:

| Method | Purpose |
|--------|---------|
| `set_values(dict)` / `update_values` / `set_distribution` | pin observed entropies to a candidate vector |
| `set_objective(dict, direction="max", constant=None)` | set the linear objective |
| `set_bounds(dict, bound_type="up" \| "lo")` | bound individual coordinates |
| `set_extra_equalities([dict, …])` / `set_extra_inequalities` | add custom linear constraints |
| `reset(which)` | clear objective / values / bounds / extras |
| `solve()` | solve; sets `.success`, `.status`, `.objective_value` |
| `solve_result(include_farkas_certificate=False)` | returns an `EntropicSolveResult` |
| `farkas_certificate(bound=1.0, tol=1e-9)` | returns a `FarkasCertificate` |
| `certificate_as_dict()` / `certificate_as_string()` | the witness, cleaned & integer-normalised |
| `write_to_file(path)` | dump the MOSEK task |
| `entropy_labels` / `variable_names` | the ordered LP coordinates |

Linear forms (objectives, equalities, certificates) are sparse dicts keyed by
subset labels, e.g. `{"A,B": 1.0, "A": -1.0}` ≡ `H(A,B) - H(A)`.

### Result objects

- `EntropicSolveResult`: `is_feasible`, `problem_status`, `solution_status`,
  `objective_value`, `farkas_certificate`.
- `FarkasCertificate`: `objective_value`, `expression`, `multipliers`,
  `is_separating`.

### Triangle scenario helpers (`entropic_inflation.scenarios.triangle`)

`triangle_problem()`, `triangle_spiral_problem()`,
`triangle_complementary_spiral_problem()`, `triangle_spiral_candidate_rays()`,
`triangle_cdd_inequalities()`, `triangle_cdd_representatives()`,
`triangle_spiral_equalities()`, and the raw constants `TRIANGLE_DAG`,
`TRIANGLE_SPIRAL_INFLATION_DAG`, `TRIANGLE_CDD_RAYS`, …

---

## Examples

Run each with `PYTHONPATH=. python examples/<file>` (see *Running* above).

**Spiral inflation — the main result (latent-inclusive, authoritative):**

| File | What it shows |
|------|----------------|
| `triangle_spiral_dual_witnesses.py` | excludes rays 1,2,3 and prints Eq. (4) as a Farkas witness |
| `triangle_spiral_test_rays_with_latents.py` | feasibility of all 10 rays → `infeasible=[1,2,3]` |
| `triangle_complementary_spiral_test_rays_with_latents.py` | same, other spiral orientation |
| `verify_spiral_inequality.py` | LP-free cross-check: Eq. (4) cuts rays 1,2,3 and is a strictly new facet |
| `validity_spiral_inequality_montecarlo.py` | LP-free validity check over 400k genuine triangle distributions |

**Observed-only spiral (historical / weaker relaxation):**

| File | What it shows |
|------|----------------|
| `triangle_spiral_manual_setup.py` | prints the inflated nodes + hand-added equalities (no solve) |
| `triangle_spiral_test_rays.py` | observed-only feasibility (finds all rays feasible) |
| `triangle_complementary_spiral_test_rays.py` | same, other orientation |
| `triangle_spiral_triangle_inequality_bounds.py` | minimises each known triangle inequality |
| `triangle_complementary_spiral_triangle_inequality_bounds.py` | same, other orientation |

**Known triangle inequalities (validity / ray enumeration):**

| File | What it shows |
|------|----------------|
| `triangle_cdd_type_1_validity.py` … `_type_3_validity.py` | each known triangle inequality is valid (min ≈ 0) |
| `auxiliar_cdd_triangle_rays.py` | re-derives the extreme rays with `pycddlib` |

**Probability → entropy pipeline (quantum).** The `quantum_nonclassicality/`
folder is a generic, dimension-scalable **generator of quantum triangle
strategies that are known to be nonlocal at the probability level** — the
Token-Counting and Color-Matching families of Renou & Beigi (arXiv:2202.00905)
and Renou et al. (arXiv:1905.04902). It produces `p(a,b,c)` at any source
dimension `d` and feeds the entropy vectors into the tests above, to probe
whether probability-level nonlocality leaves a Shannon-entropic signature. See
[`examples/quantum_nonclassicality/README.md`](examples/quantum_nonclassicality/README.md).

| File | What it shows |
|------|----------------|
| `quantum_nonclassicality/triangle_probability.py` | builds `p(a,b,c)` from states + POVMs (any dimension) |
| `quantum_nonclassicality/entropic_vector.py` | `p(a,b,c)` → entropy vector + cheap inequality pre-screen |
| `quantum_nonclassicality/triangle_ejm_sanity.py` | end-to-end EJM check (entropically classical → feasible) |
| `quantum_nonclassicality/triangle_token_counting.py` | Token-Counting strategy for any token number `η` (source dim `d=η+1`); qubit case reproduces arXiv:1905.04902 Eqs. (3)–(5). `describe_tuning(η)` lists the free angles |
| `quantum_nonclassicality/triangle_token_counting_entropy.py` | `p(a,b,c)` → entropy vector → triangle inequalities + Eq. (4); confirms output scaling `d²` outcomes/party, `d⁶` entries |
| `quantum_nonclassicality/triangle_renou_dimension_comparison.py` | Eq. (4) slack for qubit/qutrit/ququart side by side |
| `quantum_nonclassicality/triangle_renou_optimize.py` | minimise the Eq. (4) slack over the measurement angles (d=2,3,4) — finds the classical facet, no violation |
| `quantum_nonclassicality/triangle_color_matching.py` | Color-Matching strategy by colour number `d`: coarse (`d+1` outcomes) and refined (`d²`; the coherent/nonlocal version for `d≥3` — at `d=2` it reduces to the decohered, classical measurement) |

---

## Status & limitations

- The **latent-inclusive** spiral LP (`include_latents=True`) is the
  authoritative construction; both spiral orientations give
  `feasible = [0,4,5,6,7,8,9]`, `infeasible = [1,2,3]`.
- The **observed-only** examples are a strictly weaker relaxation kept for
  historical comparison: they find all ten rays feasible and cannot even bound
  the type-2/type-3 triangle inequalities. Prefer the latent-inclusive path.
- `InflationProblem.symmetries` is a placeholder (the entropic-symmetry layer is
  handled instead by the copy-symmetry equalities in the LP builder).
- Quantum/classical source flags are parsed but the entropic LP here treats all
  sources classically.

## Repository layout

```
Entropic Inflation/
├── entropic_inflation/
│   ├── InflationProblem.py     # DAG parsing, inflation bookkeeping
│   ├── lp/
│   │   ├── InflationLP.py       # LP assembly, solve, Farkas dual
│   │   └── lp_utils.py          # elemental Shannon cone builder
│   └── scenarios/triangle.py    # triangle DAGs, rays, known inequalities
└── examples/                    # the runnable scripts above
```

The repository root also contains `Current Code/` and `shannon_tests/`, which
are **older/legacy** code kept for reference and not part of this package.
