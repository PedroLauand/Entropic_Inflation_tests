"""Compare spiral LP infeasible rays with feasibility_test implementation."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEAS_PATH = ROOT / "shannon_tests" / "feasibility_test"
if str(FEAS_PATH) not in sys.path:
    sys.path.insert(0, str(FEAS_PATH))

# Stub IPython.display if not installed (desc_entro imports it but doesn't need it here).
if "IPython" not in sys.modules:
    ipy = types.ModuleType("IPython")
    display_mod = types.ModuleType("IPython.display")

    def _display(*_args, **_kwargs):
        return None

    display_mod.display = _display
    ipy.display = display_mod
    sys.modules["IPython"] = ipy
    sys.modules["IPython.display"] = display_mod

from Feas_problem import Feasibility_Entropic_vector, Equalities_spiral_inflation  # type: ignore
from desc_entro.desc_entro import desigualdades_basicas  # type: ignore

from shannon_tests.shannon_spiral_lp import (
    add_independence_constraints,
    add_symmetry_constraints,
    build_shannon_model,
)

import gurobipy as gp
from mosek.fusion import Domain, Expr, ObjectiveSense


def spiral_lp_infeasible_indices() -> List[int]:
    names = ["A0", "A1", "B0", "B1", "C0", "C1"]
    indep_input = [
        ["A1", "B0,B1,C1"],
        ["B1", "C0,C1,A1"],
        ["C1", "A0,A1,B1"],
    ]
    symmetry_input = [
        ["A0", "A1"],
        ["B0", "B1"],
        ["C0", "C1"],
        ["A0,B0", "A0,B1"],
        ["B0,C0", "B0,C1"],
        ["A0,C0", "A1,C0"],
    ]
    row_labels = [
        "H(A0)",
        "H(B0)",
        "H(C0)",
        "H(A0,B0)",
        "H(A0,C0)",
        "H(B0,C0)",
        "H(A0,B0,C0)",
    ]
    rays = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [1.5, 1.5, 1.0, 2.5, 2.0, 2.0, 3.0],
        [1.5, 1.0, 1.5, 2.0, 2.5, 2.0, 3.0],
        [1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 3.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    ]

    infeasible = []
    for i, row in enumerate(rays):
        model, x, index_of, names = build_shannon_model(names)
        add_independence_constraints(model, x, index_of, names, indep_input)
        add_symmetry_constraints(model, x, index_of, names, symmetry_input)
        for lbl, val in zip(row_labels, row):
            vars_part = lbl[2:-1]
            name_to_idx = {name: i for i, name in enumerate(names)}
            mask = 0
            for name in vars_part.split(","):
                name = name.strip()
                if not name:
                    continue
                mask |= 1 << name_to_idx[name]
            model.constraint(x.index(index_of[mask]), Domain.equalsTo(float(val)))
        model.objective("feas", ObjectiveSense.Minimize, Expr.constTerm(0.0))
        model.solve()
        if str(model.getProblemStatus()) != "ProblemStatus.PrimalAndDualFeasible":
            infeasible.append(i)
    return infeasible


def feasibility_test_infeasible_indices() -> List[int]:
    # Build A (basic Shannon inequalities) and vetor_H for 6 variables.
    A_basic, vetor_H = desigualdades_basicas(6)
    A_basic = np.asarray(A_basic, dtype=float)

    # Equalities for spiral inflation, encoded as inequalities (pairs).
    A_eq = Equalities_spiral_inflation(vetor_H)
    A = np.vstack([A_basic, A_eq])

    # Build C,d to fix A0,B0,C0 and their joint entropies from each ray.
    label_to_tuple = {
        "H(A0)": (0,),
        "H(B0)": (1,),
        "H(C0)": (2,),
        "H(A0,B0)": (0, 1),
        "H(A0,C0)": (0, 2),
        "H(B0,C0)": (1, 2),
        "H(A0,B0,C0)": (0, 1, 2),
    }
    row_labels = list(label_to_tuple.keys())

    rays = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [1.5, 1.5, 1.0, 2.5, 2.0, 2.0, 3.0],
        [1.5, 1.0, 1.5, 2.0, 2.5, 2.0, 3.0],
        [1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 3.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    ]

    infeasible = []
    for i, row in enumerate(rays):
        C = np.zeros((len(row_labels), len(vetor_H)), dtype=float)
        d = np.zeros(len(row_labels), dtype=float)
        for r, (lbl, val) in enumerate(zip(row_labels, row)):
            tup = label_to_tuple[lbl]
            idx = vetor_H.index(tup)
            C[r, idx] = 1.0
            d[r] = float(val)

        status = Feasibility_Entropic_vector(A, C, d)
        if status != gp.GRB.OPTIMAL:
            infeasible.append(i)
    return infeasible


def main() -> None:
    infeasible_spiral = spiral_lp_infeasible_indices()
    infeasible_feas = feasibility_test_infeasible_indices()

    print("infeasible (spiral lp):", infeasible_spiral)
    print("infeasible (feasibility_test):", infeasible_feas)

    only_spiral = sorted(set(infeasible_spiral) - set(infeasible_feas))
    only_feas = sorted(set(infeasible_feas) - set(infeasible_spiral))

    print("only in spiral lp:", only_spiral)
    print("only in feasibility_test:", only_feas)


if __name__ == "__main__":
    main()
