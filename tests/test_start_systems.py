"""
Tests for the start systems module of PyContinuum.
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import pycontinuum
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem
from pycontinuum.start_systems import generate_total_degree_start_system, generate_total_degree_solutions


def test_total_degree_solutions():
    """Test generating solutions for a total-degree start system."""
    # Degrees and coefficients for a simple system
    degrees = [2, 3]
    c_values = [1.0, 1.0]  # Using 1.0 for simplicity in checking roots
    
    # Generate solutions
    solutions = generate_total_degree_solutions(degrees, c_values)
    
    # Should have 2*3 = 6 solutions
    assert len(solutions) == 6
    
    # For the first variable, we expect the 2nd roots of unity
    expected_x_roots = [1.0, -1.0]
    
    # For the second variable, we expect the 3rd roots of unity
    expected_y_roots = [1.0, -0.5 + 0.866j, -0.5 - 0.866j]  # Approximately
    
    # Verify all combinations appear
    for x_root in expected_x_roots:
        for y_root in expected_y_roots:
            # Look for this combination
            found = False
            for sol in solutions:
                if (abs(sol[0] - x_root) < 1e-10 and 
                    abs(sol[1] - y_root) < 1e-10):
                    found = True
                    break
            
            assert found, f"Solution ({x_root}, {y_root}) not found"


def test_total_degree_start_system():
    """Test generating a total-degree start system."""
    # Create a simple target system
    x = Variable('x')
    y = Variable('y')
    
    # System: x^2 + y = 0, x + y^3 = 0
    eq1 = x**2 + y
    eq2 = x + y**3
    target_system = PolynomialSystem([eq1, eq2])
    
    # Generate start system
    variables = [x, y]
    start_system, start_solutions = generate_total_degree_start_system(target_system, variables)
    
    # Check the structure of the start system
    assert len(start_system.equations) == 2
    
    # Degrees should match the target system
    degrees = target_system.degrees()
    assert degrees == [2, 3]
    
    # The start system should have the form x^2 - c1 = 0, y^3 - c2 = 0
    # Extract coefficients
    c1 = None
    c2 = None
    
    for eq in start_system.equations:
        if x in eq.variables() and y not in eq.variables():
            # This is the x equation
            # Should have the form x^2 - c1 = 0
            for term in eq.terms:
                if not term.variables:  # constant term
                    c1 = -term.coefficient
        elif y in eq.variables() and x not in eq.variables():
            # This is the y equation
            # Should have the form y^3 - c2 = 0
            for term in eq.terms:
                if not term.variables:  # constant term
                    c2 = -term.coefficient
    
    assert c1 is not None and c2 is not None
    
    # Number of start solutions should be 2*3 = 6
    assert len(start_solutions) == 6
    
    # Verify solutions satisfy the start system
    for sol in start_solutions:
        x_val, y_val = sol
        
        # Evaluate the start system equations
        eq1_val = x_val**2 - c1
        eq2_val = y_val**3 - c2
        
        # Should be close to zero
        assert abs(eq1_val) < 1e-10
        assert abs(eq2_val) < 1e-10