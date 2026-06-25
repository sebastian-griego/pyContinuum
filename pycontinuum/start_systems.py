"""
Start system generation module for PyContinuum.

This module provides functions to generate start systems for homotopy continuation,
including total-degree homotopies and other approaches.
"""

import cmath
from itertools import product
from numbers import Integral, Number
from typing import Any, List, Optional, Tuple

import numpy as np

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem, Monomial

def _coerce_rng(random_state: Any = None):
    """Return a NumPy-compatible random number generator."""
    if random_state is None:
        return np.random
    if isinstance(random_state, np.random.Generator):
        return random_state
    if hasattr(random_state, "uniform"):
        return random_state
    return np.random.default_rng(random_state)


def _rng_uniform_scalar(
    rng: Any,
    low: float,
    high: float,
    *,
    context: str,
) -> float:
    try:
        value = rng.uniform(low, high)
    except Exception as exc:
        raise ValueError(
            f"random_state.uniform failed while generating {context}"
        ) from exc
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(
            f"random_state.uniform must return a finite scalar for {context}"
        )
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(
            f"random_state.uniform must return a finite scalar for {context}"
        )
    try:
        numeric_value = float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"random_state.uniform must return a finite scalar for {context}"
        ) from exc
    if not np.isfinite(numeric_value):
        raise ValueError(
            f"random_state.uniform must return a finite scalar for {context}"
        )
    return numeric_value


def _rng_standard_normal(
    rng: Any,
    size: Any = None,
    *,
    context: str,
) -> Any:
    try:
        value = rng.standard_normal(size)
    except Exception as exc:
        raise ValueError(
            f"random_state.standard_normal failed while generating {context}"
        ) from exc
    expected_shape = _normal_sample_shape(size)
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "random_state.standard_normal must return finite numeric values "
            f"with shape {expected_shape} for {context}"
        ) from exc
    if array.shape != expected_shape:
        raise ValueError(
            "random_state.standard_normal must return finite numeric values "
            f"with shape {expected_shape} for {context}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            "random_state.standard_normal must return finite numeric values "
            f"with shape {expected_shape} for {context}"
        )
    if expected_shape == ():
        return float(array.item())
    return array


def _normal_sample_shape(size: Any) -> Tuple[int, ...]:
    if size is None:
        return ()
    if isinstance(size, (bool, np.bool_)):
        raise ValueError("standard_normal size must be an integer or tuple")
    if isinstance(size, Integral):
        if size < 0:
            raise ValueError("standard_normal size entries must be nonnegative")
        return (int(size),)
    try:
        dimensions = tuple(size)
    except TypeError as exc:
        raise ValueError("standard_normal size must be an integer or tuple") from exc
    shape = []
    for dimension in dimensions:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(
            dimension,
            Integral,
        ):
            raise ValueError("standard_normal size entries must be integers")
        if dimension < 0:
            raise ValueError("standard_normal size entries must be nonnegative")
        shape.append(int(dimension))
    return tuple(shape)


def _random_unit_complex(rng: Any, *, context: str = "unit complex") -> complex:
    angle = _rng_uniform_scalar(rng, 0.0, 2.0 * np.pi, context=context)
    return complex(np.cos(angle), np.sin(angle))


def generate_total_degree_start_system(
    target_system: PolynomialSystem,
    variables: List[Variable],
    allow_underdetermined: bool = False,
    random_state: Any = None,
    max_solutions: Optional[int] = None,
) -> Tuple[PolynomialSystem, List[List[complex]]]:
    """Generate a total-degree start system and its solutions.
    
    This creates a decoupled system where each equation has the form
    x_i^d - c_i = 0, where d is the degree of the corresponding target equation.
    
    Args:
        target_system: Target polynomial system to solve
        variables: List of variables in the system
        allow_underdetermined: If True, allow systems with fewer equations than variables
        random_state: Optional seed or NumPy random generator for reproducible starts
        max_solutions: Optional cap on the number of start solutions to allocate
        
    Returns:
        Tuple of (start_system, start_solutions)
    """
    if not isinstance(target_system, PolynomialSystem):
        raise TypeError("target_system must be a PolynomialSystem")
    if not isinstance(allow_underdetermined, bool):
        raise TypeError("allow_underdetermined must be a boolean")
    max_solutions = _validate_max_solutions(max_solutions)
    variables = _normalize_variable_list(variables)

    # Get the degrees of each equation in the target system
    degrees = target_system.degrees()
    n_eqs = len(degrees)
    n_vars = len(variables)
    rng = _coerce_rng(random_state)
    _validate_variables_cover_system(
        target_system,
        variables,
        allow_extra=allow_underdetermined,
    )

    zero_degree_indices = [
        index for index, degree in enumerate(degrees) if degree <= 0
    ]
    if zero_degree_indices:
        raise ValueError(
            "Total-degree start systems require positive-degree equations; "
            f"found constant equation(s) at index {zero_degree_indices}"
        )
    
    if n_eqs > n_vars:
        raise ValueError(
            "Total-degree start systems cannot be generated directly for "
            f"overdetermined systems; got {n_eqs} equations in {n_vars} "
            "variables. Square or reduce the system before generating starts."
        )

    # Ensure the system is square unless allow_underdetermined is True
    if n_eqs < n_vars and not allow_underdetermined:
        raise ValueError(f"Expected a square system, but got {n_eqs} equations in {n_vars} variables")
    _check_total_degree_solution_limit(
        _total_degree_solution_count(degrees),
        max_solutions,
        "total-degree start system",
    )
    
    # For underdetermined systems, we'll handle them specially
    working_variables = variables[:n_eqs] if n_eqs < n_vars else variables
    
    # Generate random complex coefficients for the start system
    # We use random values on the unit circle
    c_values = []
    for i in range(n_eqs):
        c_values.append(
            _random_unit_complex(
                rng,
                context=f"total-degree coefficient {i}",
            )
        )
    
    # Create the start system equations x_i^(d_i) - c_i = 0
    start_equations = []
    for i, (var, deg, c) in enumerate(zip(working_variables, degrees, c_values)):
        # Create x_i^(d_i) term using Monomial
        term1 = Monomial({var: deg})
        # Create c_i term using Monomial
        term2 = Monomial({}, coefficient=c)
        # Create x_i^(d_i) - c_i using Polynomial
        eq = Polynomial([term1, Monomial({}, coefficient=-c)])
        start_equations.append(eq)
    
    # Create the start system
    start_system = PolynomialSystem(start_equations)
    
    # Generate all solutions to the start system
    start_solutions = generate_total_degree_solutions(
        degrees,
        c_values,
        max_solutions=max_solutions,
    )
    
    # For underdetermined systems, extend each solution with zeros for remaining variables
    if n_eqs < n_vars and allow_underdetermined:
        extended_solutions = []
        for sol in start_solutions:
            # Extend with zeros for the remaining variables
            extended_sol = list(sol) + [0.0] * (n_vars - n_eqs)
            extended_solutions.append(extended_sol)
        start_solutions = extended_solutions
    
    return start_system, start_solutions


def _normalize_variable_list(variables: Any) -> List[Variable]:
    try:
        return list(variables)
    except TypeError as exc:
        raise TypeError(
            "variables must be an iterable of Variable objects"
        ) from exc


def generate_total_degree_solutions(
    degrees: List[int],
    c_values: List[complex],
    max_solutions: Optional[int] = None,
) -> List[List[complex]]:
    """Generate all solutions to a total-degree start system.
    
    For each equation x_i^(d_i) - c_i = 0, the solutions are the d_i-th roots of c_i.
    The total number of solutions is the product of the degrees.
    
    Args:
        degrees: List of polynomial degrees
        c_values: List of constant terms
        max_solutions: Optional cap on the number of solution vectors to allocate
        
    Returns:
        List of solution vectors
    """
    degrees = _validate_total_degree_list(degrees)
    c_values = _validate_total_degree_coefficients(c_values)
    max_solutions = _validate_max_solutions(max_solutions)
    if len(degrees) != len(c_values):
        raise ValueError("degrees and c_values must have the same length")
    _check_total_degree_solution_limit(
        _total_degree_solution_count(degrees),
        max_solutions,
        "total-degree solution generation",
    )

    # For each variable, compute all the d-th roots of the coefficient c
    roots_per_var = []
    for i, (degree, c) in enumerate(zip(degrees, c_values)):
        var_roots = []
        for k in range(degree):
            # Compute c^(1/d) * e^(2πik/d) for k = 0,1,...,d-1
            root = c**(1/degree) * cmath.exp(2j * np.pi * k / degree)
            var_roots.append(root)
        roots_per_var.append(var_roots)
    
    # Generate all combinations of roots (cartesian product)
    return [list(solution) for solution in product(*roots_per_var)]


def _validate_max_solutions(max_solutions: Any) -> Optional[int]:
    if max_solutions is None:
        return None
    if isinstance(max_solutions, bool) or not isinstance(max_solutions, Integral):
        raise TypeError("max_solutions must be an integer or None")
    if max_solutions < 0:
        raise ValueError("max_solutions must be nonnegative")
    return int(max_solutions)


def _total_degree_solution_count(degrees: List[int]) -> int:
    total = 1
    for degree in degrees:
        total *= int(degree)
    return total


def _check_total_degree_solution_limit(
    solution_count: int,
    max_solutions: Optional[int],
    source: str,
) -> None:
    if max_solutions is None:
        return
    if solution_count > max_solutions:
        raise ValueError(
            f"{source} would generate {solution_count} solution(s), exceeding "
            f"max_solutions={max_solutions}"
        )


def _validate_total_degree_list(degrees: Any) -> List[int]:
    try:
        degree_list = list(degrees)
    except TypeError as exc:
        raise TypeError("degrees must be an iterable of positive integers") from exc
    for index, degree in enumerate(degree_list):
        if isinstance(degree, bool) or not isinstance(degree, Integral):
            raise TypeError(f"degrees[{index}] must be an integer")
        if degree <= 0:
            raise ValueError(f"degrees[{index}] must be positive")
    return [int(degree) for degree in degree_list]


def _validate_total_degree_coefficients(c_values: Any) -> List[complex]:
    try:
        coefficient_list = list(c_values)
    except TypeError as exc:
        raise TypeError("c_values must be an iterable of numeric coefficients") from exc
    validated = []
    for index, coefficient in enumerate(coefficient_list):
        if isinstance(coefficient, bool) or not isinstance(coefficient, Number):
            raise TypeError(f"c_values[{index}] must be a numeric coefficient")
        value = complex(coefficient)
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError(f"c_values[{index}] must be finite")
        if value == 0:
            raise ValueError(
                f"c_values[{index}] must be nonzero to give distinct start roots"
            )
        validated.append(value)
    return validated


def _validate_variables_cover_system(
    system: PolynomialSystem,
    variables: List[Variable],
    *,
    allow_extra: bool = False,
) -> None:
    seen = set()
    duplicates = []
    for index, variable in enumerate(variables):
        if not isinstance(variable, Variable):
            raise TypeError(f"variables[{index}] must be a Variable")
        if variable in seen:
            duplicates.append(variable.name)
        seen.add(variable)
    if duplicates:
        raise ValueError(
            "Variable list contains duplicate variable(s): "
            + ", ".join(sorted(set(duplicates)))
        )

    system_variables = system.variables()
    missing = sorted(
        variable.name for variable in system_variables if variable not in seen
    )
    if missing:
        raise ValueError(
            "Variable list is missing system variable(s): " + ", ".join(missing)
        )
    if system_variables and not allow_extra:
        extra = sorted(
            variable.name for variable in seen if variable not in system_variables
        )
        if extra:
            raise ValueError(
                "Variable list contains variable(s) not used by system: "
                + ", ".join(extra)
            )
