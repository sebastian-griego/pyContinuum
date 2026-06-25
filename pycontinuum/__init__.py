"""
PyContinuum: A pure Python library for polynomial homotopy continuation methods.

This library provides tools for solving polynomial systems using numerical
homotopy continuation, with a focus on ease of use and integration with
the Python scientific ecosystem.
"""

__version__ = "0.1.1"

# Import from polynomial module
from pycontinuum.polynomial import (
    polyvar,
    Variable,
    Monomial,
    Polynomial,
    PolynomialSystem,
)

# Import from solver module
from pycontinuum.solver import (
    solve,
    refine_solution,
    refine_solutions,
    Solution,
    SolutionSet,
)

# Import start-system and path-tracking helpers for custom homotopy workflows
from pycontinuum.start_systems import (
    generate_total_degree_solutions,
    generate_total_degree_start_system,
)
from pycontinuum.tracking import (
    compute_tangent,
    homotopy_function,
    homotopy_jacobian,
    track_paths,
    track_single_path,
)

# Import validation helpers
from pycontinuum.validation import (
    SolutionAudit,
    SolutionDiagnostics,
    diagnose_solution,
    diagnose_solutions,
    validate_solutions,
)

# Import from other modules as needed
from pycontinuum.witness_set import (
    WitnessSet,
    generate_generic_slice,
    compute_witness_superset,
)
from pycontinuum.parameter_homotopy import ParameterHomotopy, track_parameter_path

# Note: Monodromy functionality depends on optional sympy combinatorics.
# To keep core imports lightweight (and tests focused), we avoid importing
# monodromy symbols at package import time. Users can import from
# pycontinuum.monodromy directly when needed.

# Update __all__ to include all names you want to be importable
# directly from 'pycontinuum'
__all__ = [
    "polyvar",
    "Variable",
    "Monomial",
    "Polynomial",
    "PolynomialSystem",
    "solve",
    "refine_solution",
    "refine_solutions",
    "Solution",
    "SolutionSet",
    "generate_total_degree_solutions",
    "generate_total_degree_start_system",
    "compute_tangent",
    "homotopy_function",
    "homotopy_jacobian",
    "track_paths",
    "track_single_path",
    "SolutionAudit",
    "SolutionDiagnostics",
    "diagnose_solution",
    "diagnose_solutions",
    "validate_solutions",
    "WitnessSet",
    "generate_generic_slice",
    "compute_witness_superset",
    "ParameterHomotopy",
    "track_parameter_path",
    # Monodromy exports intentionally omitted from default namespace
]
