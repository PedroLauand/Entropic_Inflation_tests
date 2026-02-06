"""LP feasibility tests for von-Neumann entropies using MOSEK."""

from __future__ import annotations

import sys

from mosek.fusion import AccSolutionStatus, Expr, ObjectiveSense

from entropy_utils import (
    LP_test,
    build_farkas_model,
    entropic_caption,
    solve_farkas_model,
)


if __name__ == "__main__":
    # Inflation scenario
    names = [["A0", "B0", "C0"]]
    indep_input = [["A0", "C0"]]
    separability_input: list[list[str]] = []

    row_labels = [
        "S(A0)",
        "S(B0)",
        "S(C0)",
        "S(A0,B0)",
        "S(B0,C0)",
    ]

    # Rays from the basic von-Neumann entropy polytope for ABC.
    # Caption order: [S(A), S(B), S(AB), S(C), S(AC), S(BC), S(ABC)]
    rays = [
        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    ]
    # Drop S(A,C) and S(A,B,C). Reorder to match row_labels.
    rays_full = [[row[0], row[1], row[3], row[2], row[5]] for row in rays]

    feasible_rows = []
    infeasible_rows = []
    certificate_summaries = []
    certificate_y_summaries: dict[int, str] = {}
    tol = 1e-9

    candidate_names = ["A0", "B0", "C0"]
    candidate_caption = entropic_caption(candidate_names)

    for i, row in enumerate(rays_full):
        label_to_value = dict(zip(row_labels, row))
        candidate = []
        for label in candidate_caption:
            if label in label_to_value:
                candidate.append(label_to_value[label])
            else:
                candidate.append("")
        (
            model,
            x,
            label_to_index,
            labels,
            var_names,
            constraints_meta,
            matrix,
        ) = LP_test(
            names,
            indep_input=indep_input,
            separability_input=separability_input,
            candidate=candidate,
            candidate_names=candidate_names,
            return_matrix=True,
        )
        model.acceptedSolutionStatus(AccSolutionStatus.Certificate)
        model.setLogHandler(sys.stdout)
        model.objective("feas", ObjectiveSense.Minimize, Expr.constTerm(0.0))
        model.solve()

        problem_status = model.getProblemStatus()
        solution_status = model.getPrimalSolutionStatus()
        print(f"ray {i} problem_status:", problem_status)
        print(f"ray {i} solution_status:", solution_status)

        farkas_model, y = build_farkas_model(matrix["M"], matrix["b"])
        obj, expr, y_vals = solve_farkas_model(
            farkas_model, y, b_caption=matrix["b_caption"]
        )
        print(f"ray {i} b^T y expression: {expr}")
        print(f"ray {i} b^T y value: {obj:g}")

        is_feasible = obj >= -tol
        if is_feasible:
            feasible_rows.append((i, row))
        else:
            infeasible_rows.append((i, row))
            print(f"ray {i} y values:")
            y_terms = []
            for cap, y_val in zip(matrix["b_caption"], y_vals):
                if cap == "0":
                    continue
                if abs(y_val) < 1e-12:
                    continue
                print(f"  {cap}: {y_val:g}")
                y_terms.append(f"{cap}={y_val:g}")
            certificate_y_summaries[i] = ", ".join(y_terms)
            certificate_summaries.append(f"ray {i}: {expr} = {obj:g}")

    print("feasible_rows:", feasible_rows)
    print("infeasible_rows:", infeasible_rows)
    print("feasible_count:", len(feasible_rows))
    print("infeasible_count:", len(infeasible_rows))
    print("certificate_count:", len(certificate_summaries))
    if certificate_summaries:
        print("certificates:")
        for entry in certificate_summaries:
            print(entry)
            if entry.startswith("ray "):
                try:
                    ray_id = int(entry.split(":", 1)[0].split()[1])
                except (IndexError, ValueError):
                    ray_id = None
                if ray_id is not None and ray_id in certificate_y_summaries:
                    print(f"ray {ray_id} y: {certificate_y_summaries[ray_id]}")
