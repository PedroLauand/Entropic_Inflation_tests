"""Extract clean dual entropic witnesses from rays excluded by the latent-inclusive spiral LP.

For each of the ten triangle candidate rays, the spiral inflation LP is solved
with Farkas certificates enabled. Infeasible rays produce a dual inequality on
the seven observed triangle entropy coordinates that every classical realisation
of the triangle satisfies and that the ray violates.

The projection onto observed coordinates happens automatically: the only
non-trivial captions in the LP's ``b``-vector are those of the entropies pinned
by ``lp.set_values(ray)``, so the Farkas expression lives natively in the basis
``{H(A0), H(B0), H(C0), H(A0,B0), H(A0,C0), H(B0,C0), H(A0,B0,C0)}``.
"""

from __future__ import annotations

from multiprocessing import Process, Queue
from typing import Any

from entropic_inflation import (
    InflationLP,
    triangle_spiral_candidate_rays,
    triangle_spiral_problem,
)

PER_RAY_TIMEOUT_SECONDS = 90


def _solve_ray(queue: Queue, values: dict[str, float]) -> None:
    lp = InflationLP(triangle_spiral_problem(), include_latents=True)
    lp.set_values(values)
    result = lp.solve_result(include_farkas_certificate=True)
    payload: dict[str, Any] = {
        "is_feasible": result.is_feasible,
        "problem_status": result.problem_status,
        "solution_status": result.solution_status,
    }
    if not result.is_feasible:
        payload["witness_string"] = lp.certificate_as_string(chop_tol=1e-9, round_decimals=3)
        payload["witness_coeffs"] = lp.certificate_as_dict(chop_tol=1e-9, round_decimals=3)
    queue.put(payload)


def solve_with_timeout(values: dict[str, float], timeout_seconds: int) -> dict[str, Any] | None:
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


def _normalise_to_integers(coeffs: dict[str, float]) -> dict[str, int]:
    """Scale to integers by trial common-denominator, then reduce by GCD."""
    from math import gcd as _gcd
    items = list(coeffs.items())
    if not items:
        return {}
    for scale in (1, 2, 5, 10, 20, 50, 100):
        if all(abs(v * scale - round(v * scale)) < 1e-6 for _, v in items):
            ints = {k: int(round(v * scale)) for k, v in items}
            g = 0
            for v in ints.values():
                g = _gcd(g, abs(v))
            if g > 1:
                ints = {k: v // g for k, v in ints.items()}
            return ints
    return {k: int(round(v)) for k, v in items}


def _format_inequality(coeffs: dict[str, float]) -> str:
    """Render the dual witness as ``c1*H(label1) + c2*H(label2) + ... >= 0``.

    Labels returned by ``certificate_as_dict`` already include the leading ``H``,
    so we emit them verbatim.
    """
    ints = _normalise_to_integers(coeffs)
    parts: list[str] = []
    for label, coef in sorted(ints.items()):
        if coef == 0:
            continue
        sign = "+" if coef > 0 else "-"
        mag = abs(coef)
        term = label if mag == 1 else f"{mag}*{label}"
        parts.append(f"{sign} {term}")
    if not parts:
        return "(trivial)"
    text = " ".join(parts)
    if text.startswith("+ "):
        text = text[2:]
    return text + " >= 0"


if __name__ == "__main__":
    rays = triangle_spiral_candidate_rays()
    witnesses: dict[int, dict[str, float]] = {}

    print("Testing rays against the latent-inclusive spiral LP...")
    print()

    for item in rays:
        idx = item["index"]
        payload = solve_with_timeout(item["values"], PER_RAY_TIMEOUT_SECONDS)
        if payload is None:
            print(f"ray {idx:>2}: TIMED OUT after {PER_RAY_TIMEOUT_SECONDS}s")
            continue
        if payload["is_feasible"]:
            print(f"ray {idx:>2}: feasible (no witness)")
            continue

        print(f"ray {idx:>2}: infeasible")
        print(f"        raw certificate: {payload['witness_string']}")
        witnesses[idx] = payload["witness_coeffs"]

    if witnesses:
        print()
        print("=" * 70)
        print("Dual witnesses on observed triangle entropies:")
        print("=" * 70)
        for idx, coeffs in witnesses.items():
            print(f"  ray {idx}: {_format_inequality(coeffs)}")
