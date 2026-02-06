"""Compute extreme rays for von-Neumann entropy inequalities using cdd."""

from constraints.vn_constraints import build_vn_inequalities, polytope


if __name__ == "__main__":
    caption, array = build_vn_inequalities(["A", "B", "C"])
    generators = polytope(array)
    for row in generators:
        t = row[0]
        kind = "ray" if t == 0 else "extreme point"
        values = [f"{name}={value}" for name, value in zip(caption[1:], row[1:])]
        print(f"{kind}: " + ", ".join(values))
    print("constraints:", len(array))
    print("generators:", len(generators))
