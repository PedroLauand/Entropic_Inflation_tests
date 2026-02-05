"""Von-Neumann entropy inequality generator and rays via cdd."""

import cdd


def polytope(array: list[list[float]]) -> list[list[float]]:
    # H-representation: b + A x >= 0.
    if not array:
        raise ValueError("array must be non-empty")
    if hasattr(cdd, "matrix_from_array"):
        mat = cdd.matrix_from_array(array, rep_type=cdd.RepType.INEQUALITY)
    else:
        mat = cdd.Matrix(array, number_type="fraction")
        mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.polyhedron_from_matrix(mat)
    ext = cdd.copy_generators(poly)
    return ext.array


def build_vn_inequalities(var_names: list[str]) -> tuple[list[str], list[list[float]]]:
    n = len(var_names)
    masks = list(range(0, 1 << n))
    index_of = {m: i for i, m in enumerate(masks)}

    def mask_label(mask: int) -> str:
        names = [var_names[i] for i in range(n) if (mask >> i) & 1]
        return "S(" + ",".join(names) + ")"

    caption = ["1"] + [mask_label(m) for m in masks]

    def weak_monotonicity() -> list[list[float]]:
        rows: list[list[float]] = []
        for a in masks:
            for b in masks:
                if a & b:
                    continue
                for c in masks:
                    if (a & c) or (b & c):
                        continue
                    row = [0.0] * (1 + len(masks))
                    row[1 + index_of[a]] -= 1.0
                    row[1 + index_of[b]] -= 1.0
                    row[1 + index_of[a | c]] += 1.0
                    row[1 + index_of[b | c]] += 1.0
                    rows.append(row)
        return rows

    def strong_subadditivity() -> list[list[float]]:
        rows: list[list[float]] = []
        for a in masks:
            for b in masks:
                if a & b:
                    continue
                for c in masks:
                    if (a & c) or (b & c):
                        continue
                    row = [0.0] * (1 + len(masks))
                    row[1 + index_of[a | b]] += 1.0
                    row[1 + index_of[b | c]] += 1.0
                    row[1 + index_of[b]] -= 1.0
                    row[1 + index_of[a | b | c]] -= 1.0
                    rows.append(row)
        return rows

    array: list[list[float]] = []
    array.extend(weak_monotonicity())
    array.extend(strong_subadditivity())
    # Enforce S(empty)=0 by substituting S(empty)=0 in all rows.
    if 0 in index_of:
        empty_idx = 1 + index_of[0]
        for row in array:
            row[empty_idx] = 0.0

    # Remove trivial rows (all zeros).
    array = [row for row in array if any(abs(v) > 1e-12 for v in row)]

    # Drop S(empty) variable entirely after substitution.
    if 0 in index_of:
        empty_idx = 1 + index_of[0]
        for row in array:
            row.pop(empty_idx)
        # Remove S() from caption
        caption = [caption[0]] + [c for c in caption[1:] if c != "S()"]

    # Normalize rows to remove scalar duplicates (e.g., 2*S(X)>=0).
    def normalize_row(row: list[float]) -> tuple[float, ...]:
        # scale by max abs coefficient (excluding constant term which is always 0 here)
        coeffs = row[1:]
        max_abs = max(abs(v) for v in coeffs) if coeffs else 0.0
        if max_abs == 0.0:
            return tuple(row)
        scaled = [row[0]] + [v / max_abs for v in coeffs]
        # round to stabilize
        return tuple(round(v, 12) for v in scaled)

    deduped_map = {}
    for row in array:
        key = normalize_row(row)
        deduped_map[key] = key
    deduped = [list(row) for row in deduped_map.values()]
    return caption, deduped


def build_vn_matrix(var_names: list[str]) -> tuple[list[list[float]], list[str]]:
    """Return (M, h_caption) for basic VN inequalities M h <= 0.

    The inequalities are generated from weak monotonicity (WM) and
    strong subadditivity (SSA) over disjoint variable sets.
    """
    caption, array = build_vn_inequalities(var_names)
    h_caption = caption[1:]
    M: list[list[float]] = []
    for row in array:
        coeffs = row[1:]
        M.append([-float(c) for c in coeffs])
    return M, h_caption


if __name__ == "__main__":
    # Example: use A,B,C and build inequalities via functions.
    caption, array = build_vn_inequalities(["A", "B", "C"])
    generators = polytope(array)
    for row in generators:
        t = row[0]
        kind = "ray" if t == 0 else "extreme point"
        values = [f"{name}={value}" for name, value in zip(caption[1:], row[1:])]
        print(f"{kind}: " + ", ".join(values))
    print("constraints:", len(array))
    print("generators:", len(generators))
