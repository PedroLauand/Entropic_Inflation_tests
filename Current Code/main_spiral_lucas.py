"""Compatibility wrapper for the relocated Lucas spiral script."""

import runpy


if __name__ == "__main__":
    runpy.run_module("legacy.lucas_gurobi.main_spiral_lucas", run_name="__main__")
