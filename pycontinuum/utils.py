"""
Utility functions for PyContinuum.

This module provides common utility functions used across the library.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
try:
    from scipy.linalg import lu_factor, lu_solve  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    lu_factor = None  # type: ignore
    lu_solve = None  # type: ignore
    _HAS_SCIPY = False

from pycontinuum.polynomial import Variable, PolynomialSystem

def evaluate_system_at_point(system: PolynomialSystem, 
                            point: List[complex], 
                            variables: List[Variable]) -> np.ndarray:
    """Evaluate a polynomial system at a point."""
    # Create a dictionary mapping variables to their values
    var_dict = {var: val for var, val in zip(variables, point)}
    
    # Evaluate the system
    values = system.evaluate(var_dict)
    
    # Convert to numpy array
    return np.array(values, dtype=complex)

def evaluate_jacobian_at_point(system: PolynomialSystem,
                              point: List[complex],
                              variables: List[Variable]) -> np.ndarray:
    """Evaluate the Jacobian of a polynomial system at a point."""
    # Create a dictionary mapping variables to their values
    var_dict = {var: val for var, val in zip(variables, point)}
    
    # Get the Jacobian polynomials
    jac_polys = system.jacobian(variables)
    
    # Evaluate each polynomial in the Jacobian
    jac_values = []
    for row in jac_polys:
        jac_row = []
        for poly in row:
            jac_row.append(poly.evaluate(var_dict))
        jac_values.append(jac_row)
    
    # Convert to numpy array
    return np.array(jac_values, dtype=complex)

def newton_corrector(system: PolynomialSystem,
                    point: np.ndarray,
                    variables: List[Variable],
                    max_iters: int = 10,
                    tol: float = 1e-10) -> Tuple[np.ndarray, bool, int]:
    """Apply Newton's method to correct a point to a solution.
    
    Args:
        system: The polynomial system
        point: Initial point for correction
        variables: The variables in the system
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence
        
    Returns:
        Tuple of (corrected point, success flag, number of iterations)
    """
    current = np.array(point, dtype=complex)
    
    for i in range(max_iters):
        # Evaluate the system and Jacobian
        f_val = evaluate_system_at_point(system, current, variables)
        jac = evaluate_jacobian_at_point(system, current, variables)
        
        # Check if we're already at a solution
        if np.linalg.norm(f_val) < tol:
            return current, True, i
        
        # Solve the linear system J * delta = -f
        try:
            delta = np.linalg.solve(jac, -f_val)
        except np.linalg.LinAlgError:
            # If the Jacobian is singular, try a pseudoinverse
            delta = np.linalg.lstsq(jac, -f_val, rcond=None)[0]
        
        # Update the point
        current = current + delta
        
        # Check for convergence
        if np.linalg.norm(delta) < tol:
            return current, True, i+1
    
    # If we got here, we didn't converge
    return current, False, max_iters


def solve_linear_system(jac: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a linear system robustly using LU with fallback to least squares.

    Args:
        jac: Jacobian matrix (m x n), typically square
        rhs: Right-hand side vector (m,)

    Returns:
        Solution vector x minimizing ||Jx - rhs||.
    """
    if _HAS_SCIPY:
        try:
            lu, piv = lu_factor(jac)
            return lu_solve((lu, piv), rhs)
        except Exception:
            pass
    # Fallback to least squares
    return np.linalg.lstsq(jac, rhs, rcond=None)[0]


def newton_corrector_numeric(
    f: Callable[[np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    max_iters: int = 10,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, bool, int]:
    """Newton's method for numeric function and Jacobian callables.

    Args:
        f: Function mapping x -> f(x)
        jac: Function mapping x -> J(x)
        point: Initial guess
        max_iters: Maximum iterations
        tol: Convergence tolerance on step and residual

    Returns:
        (x, success, iters)
    """
    current = np.array(point, dtype=complex)
    for i in range(max_iters):
        f_val = f(current)
        if np.linalg.norm(f_val) < tol:
            return current, True, i
        J = jac(current)
        delta = solve_linear_system(J, -f_val)
        current = current + delta
        if np.linalg.norm(delta) < tol:
            return current, True, i + 1
    return current, False, max_iters


def evaluate_jacobian_polynomials(jac_polys: List[List[Any]], var_dict: Dict[Variable, complex]) -> np.ndarray:
    """Evaluate a Jacobian represented as polynomials at a point.

    Args:
        jac_polys: List of rows, each a list of Polynomial
        var_dict: Mapping variable -> value

    Returns:
        Numeric Jacobian matrix.
    """
    rows: List[List[complex]] = []
    for row in jac_polys:
        rows.append([poly.evaluate(var_dict) for poly in row])
    return np.array(rows, dtype=complex)