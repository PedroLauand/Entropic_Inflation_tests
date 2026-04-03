import platform

from numpy import __version__ as numpy_version
from scipy import __version__ as scipy_version
from sympy import __version__ as sympy_version

from ._version import __version__


def about() -> None:
    """Display package and dependency versions for entropic_inflation."""
    try:
        from mosek import Env

        major, minor, revision = Env.getversion()
        mosek_version = f"{major}.{minor}.{revision}"
    except ImportError:
        mosek_version = "Not installed"

    about_str = f"""
Entropic Inflation: Entropic LPs for inflated causal scenarios
==============================================================================

Entropic Inflation Version:\t{__version__}

Core Dependencies
-----------------
NumPy Version:\t{numpy_version}
SciPy Version:\t{scipy_version}
SymPy Version:\t{sympy_version}
Mosek Version:\t{mosek_version}

Python Version:\t{platform.python_version()}
Platform Info:\t{platform.system()} ({platform.machine()})
"""
    print(about_str)


if __name__ == "__main__":
    about()
