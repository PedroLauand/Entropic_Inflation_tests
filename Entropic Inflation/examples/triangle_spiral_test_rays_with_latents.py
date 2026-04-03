"""Test the historical 10 candidate rays in the latent-inclusive spiral LP.

This is the current pure automatic version:

- the manual inflated DAG comes from ``triangle_spiral_problem()``
- latent entropy coordinates are included
- structural constraints are derived automatically from the package
- each candidate ray only fixes the observed ``A0,B0,C0`` marginal
"""

from __future__ import annotations

from multiprocessing import Process, Queue

from entropic_inflation import (
    InflationLP,
    triangle_spiral_candidate_rays,
    triangle_spiral_problem,
)

PER_RAY_TIMEOUT_SECONDS = 30


def _solve_ray(queue: Queue, values: dict[str, float]) -> None:
    lp = InflationLP(triangle_spiral_problem(), include_latents=True)
    lp.set_values(values)
    result = lp.solve_result()
    queue.put(
        {
            "is_feasible": result.is_feasible,
            "problem_status": result.problem_status,
            "solution_status": result.solution_status,
        }
    )


def solve_with_timeout(values: dict[str, float], timeout_seconds: int) -> dict[str, object] | None:
    queue: Queue = Queue()
    process = Process(target=_solve_ray, args=(queue, values))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return None

    try:
        return queue.get_nowait()
    except Exception:
        return None


if __name__ == "__main__":
    rays = triangle_spiral_candidate_rays()

    feasible = []
    infeasible = []
    timed_out = []

    for item in rays:
        payload = solve_with_timeout(item["values"], PER_RAY_TIMEOUT_SECONDS)
        if payload is None:
            print(f"ray {item['index']}: timed out after {PER_RAY_TIMEOUT_SECONDS}s")
            timed_out.append(item["index"])
            continue

        status = "feasible" if payload["is_feasible"] else "infeasible"
        print(
            f"ray {item['index']}: {status} | "
            f"problem_status={payload['problem_status']} | "
            f"solution_status={payload['solution_status']}"
        )

        if payload["is_feasible"]:
            feasible.append(item["index"])
        else:
            infeasible.append(item["index"])

    print("feasible rays:", feasible)
    print("infeasible rays:", infeasible)
    print("timed out rays:", timed_out)
