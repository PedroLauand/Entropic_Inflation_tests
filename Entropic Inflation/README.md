Entropic Inflation
==================

`entropic_inflation` is a compact package for building entropic inflation LPs.
The current tree keeps only:

- the package itself in `entropic_inflation/`
- a small triangle scenario helper module
- runnable examples in `examples/`

Main API
--------

- `InflationProblem`: define the scenario and its inflation
- `InflationLP`: build and solve the entropic LP
- `entropic_inflation.scenarios.triangle`: ready-made triangle helpers

Typical workflow
----------------

```python
from entropic_inflation import InflationLP, triangle_problem

problem = triangle_problem()
lp = InflationLP(problem, include_latents=True)
lp.set_objective({"A,B": 1.0, "A,C": 1.0, "A": -1.0, "B": -1.0, "C": -1.0})
result = lp.solve_result()
print(result.objective_value)
```

Examples
--------

- `triangle_cdd_type_1_validity.py`
- `triangle_cdd_type_2_validity.py`
- `triangle_cdd_type_3_validity.py`
- `auxiliar_cdd_triangle_rays.py`
- `triangle_spiral_manual_setup.py`
- `triangle_spiral_test_rays.py`
- `triangle_complementary_spiral_test_rays.py`
- `triangle_spiral_test_rays_with_latents.py`
- `triangle_complementary_spiral_test_rays_with_latents.py`

Run from this directory with:

```bash
PYTHONPATH=. /Users/Pedro/.venvs/entropic-inflation311/bin/python3 examples/triangle_cdd_type_1_validity.py
```

Current spiral benchmark
------------------------

With the current automatic latent-inclusive derivation, both spiral directions
classify the candidate triangle rays in the same way:

- feasible rays: `[0, 4, 5, 6, 7, 8, 9]`
- infeasible rays: `[1, 2, 3]`

Observed-only spiral examples remain available as historical comparisons.
