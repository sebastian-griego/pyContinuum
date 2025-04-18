"""
Start system generation module for PyContinuum.

This module provides functions to generate start systems for homotopy continuation,
including total-degree homotopies and other approaches.
"""

import numpy as np
import cmath
from typing import Dict, List, Tuple, Set, Any

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem


def generate_total_degree_start_system(target_system: PolynomialSystem, 
                                      variables: List[Variable]) -> Tuple[PolynomialSystem, List[List[complex]]]:
    """Generate a total-degree start system and its solutions.
    
    This creates a decoupled system where each equation has the form
    x_i^d - c_i = 0, where d is the degree of the corresponding target equation.
    
    Args:
        target_system: Target polynomial system to solve
        variables: List of variables in the system
        
    Returns:
        Tuple of (start_system, start_solutions)
    """
    # Get the degrees of each equation in the target system
    degrees = target_system.degrees()
    n_eqs = len(degrees)
    n_vars = len(variables)
    
    # Ensure the system is square (same number of equations as variables)
    if n_eqs != n_vars:
        raise ValueError(f"Expected a square system, but got {n_eqs} equations in {n_vars} variables")
    
    # Generate random complex coefficients for the start system
    # We use random values on the unit circle
    c_values = []
    for i in range(n_vars):
        angle = np.random.uniform(0, 2 * np.pi)
        c_values.append(complex(np.cos(angle), np.sin(angle)))
    
    # Create the start system equations x_i^(d_i) - c_i = 0
    start_equations = []
    for i, (var, deg, c) in enumerate(zip(variables, degrees, c_values)):
        # Create x_i^(d_i)
        term1 = Polynomial([Monomial({var: deg})])
        # Create c_i
        term2 = Polynomial([Monomial({}, coefficient=c)])
        # Create x_i^(d_i) - c_i
        eq = term1 - term2
        start_equations.append(eq)
    
    # Create the start system
    start_system = PolynomialSystem(start_equations)
    
    # Generate all solutions to the start system
    start_solutions = generate_total_degree_solutions(degrees, c_values)
    
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


class Monomial:
    """Representation of a monomial (term in a polynomial) for internal use."""
    
    def __init__(self, variables: Dict[Variable, int], coefficient: complex = 1):
        """Initialize a monomial with variables and their exponents.
        
        Args:
            variables: Dict mapping Variable objects to their exponents
            coefficient: Coefficient of the monomial (default: 1)
        """
        # Filter out zero exponents
        self.variables = {var: exp for var, exp in variables.items() if exp != 0}
        self.coefficient = coefficient