"""
PyContinuum: A pure Python library for polynomial homotopy continuation methods.

This library provides tools for solving polynomial systems using numerical
homotopy continuation, with a focus on ease of use and integration with
the Python scientific ecosystem.
"""

__version__ = "0.1.0"

from pycontinuum.polynomial import polyvar, Polynomial, PolynomialSystem, make_system
from pycontinuum.solver import solve

__all__ = [
    "polyvar",
    "Polynomial",
    "PolynomialSystem",
    "make_system",
    "solve",
]