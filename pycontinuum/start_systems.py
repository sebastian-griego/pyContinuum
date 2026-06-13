"""
Start system generation module for PyContinuum.

This module provides functions to generate start systems for homotopy continuation,
including total-degree homotopies and other approaches.
"""

import cmath
from typing import Any, List, Tuple

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


def generate_total_degree_start_system(target_system: PolynomialSystem,
                                      variables: List[Variable],
                                      allow_underdetermined: bool = False,
                                      random_state: Any = None) -> Tuple[PolynomialSystem, List[List[complex]]]:
    """Generate a total-degree start system and its solutions.
    
    This creates a decoupled system where each equation has the form
    x_i^d - c_i = 0, where d is the degree of the corresponding target equation.
    
    Args:
        target_system: Target polynomial system to solve
        variables: List of variables in the system
        allow_underdetermined: If True, allow systems with fewer equations than variables
        random_state: Optional seed or NumPy random generator for reproducible starts
        
    Returns:
        Tuple of (start_system, start_solutions)
    """
    # Get the degrees of each equation in the target system
    degrees = target_system.degrees()
    n_eqs = len(degrees)
    n_vars = len(variables)
    rng = _coerce_rng(random_state)
    
    # Ensure the system is square unless allow_underdetermined is True
    if n_eqs != n_vars and not allow_underdetermined:
        raise ValueError(f"Expected a square system, but got {n_eqs} equations in {n_vars} variables")
    
    # For underdetermined systems, we'll handle them specially
    working_variables = variables[:n_eqs] if n_eqs < n_vars else variables
    
    # Generate random complex coefficients for the start system
    # We use random values on the unit circle
    c_values = []
    for i in range(n_eqs):
        angle = rng.uniform(0, 2 * np.pi)
        c_values.append(complex(np.cos(angle), np.sin(angle)))
    
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
    start_solutions = generate_total_degree_solutions(degrees, c_values)
    
    # For underdetermined systems, extend each solution with zeros for remaining variables
    if n_eqs < n_vars and allow_underdetermined:
        extended_solutions = []
        for sol in start_solutions:
            # Extend with zeros for the remaining variables
            extended_sol = list(sol) + [0.0] * (n_vars - n_eqs)
            extended_solutions.append(extended_sol)
        start_solutions = extended_solutions
    
    return start_system, start_solutions

def generate_total_degree_solutions(degrees: List[int], 
                                   c_values: List[complex]) -> List[List[complex]]:
    """Generate all solutions to a total-degree start system.
    
    For each equation x_i^(d_i) - c_i = 0, the solutions are the d_i-th roots of c_i.
    The total number of solutions is the product of the degrees.
    
    Args:
        degrees: List of polynomial degrees
        c_values: List of constant terms
        
    Returns:
        List of solution vectors
    """
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
    solutions = []
    
    # Recursive helper to build combinations
    def build_solutions(current_sol, var_idx):
        if var_idx == len(degrees):
            solutions.append(current_sol.copy())
            return
        
        for root in roots_per_var[var_idx]:
            current_sol.append(root)
            build_solutions(current_sol, var_idx + 1)
            current_sol.pop()
    
    build_solutions([], 0)
    return solutions
