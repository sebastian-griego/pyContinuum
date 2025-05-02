"""
PyContinuum: A pure Python library for polynomial homotopy continuation methods.

This library provides tools for solving polynomial systems using numerical
homotopy continuation, with a focus on ease of use and integration with
the Python scientific ecosystem.
"""

__version__ = "0.1.0"

from pycontinuum.polynomial import polyvar, Polynomial, PolynomialSystem, make_system
from pycontinuum.solver import solve

# Add new imports
from pycontinuum.witness_set import WitnessSet, generate_generic_slice, compute_witness_superset
from pycontinuum.parameter_homotopy import ParameterHomotopy, track_parameter_path
from pycontinuum.monodromy import (
    track_monodromy_loop, 
    numerical_irreducible_decomposition,
    compute_numerical_decomposition
)

__all__ = [
    "polyvar",
    "Polynomial",
    "PolynomialSystem",
    "make_system",
    "solve",
    # Add new functionality
    "WitnessSet",
    "generate_generic_slice",
    "compute_witness_superset",
    "ParameterHomotopy",
    "track_parameter_path",
    "track_monodromy_loop",
    "numerical_irreducible_decomposition",
    "compute_numerical_decomposition"
]