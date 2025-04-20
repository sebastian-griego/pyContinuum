"""
Utility functions for PyContinuum.

This module provides common utility functions used across the library.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

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