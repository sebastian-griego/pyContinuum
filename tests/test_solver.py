"""
Tests for the solver module of PyContinuum.
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import pycontinuum
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pycontinuum import polyvar, PolynomialSystem, solve


def test_simple_solver():
    """Test solving a simple system with known solutions."""
    # Define variables
    x, y = polyvar('x', 'y')
    
    # Define a simple system with known solutions:
    # x^2 + y^2 = 1
    # x = y
    # Solutions are (±1/√2, ±1/√2) where signs are the same
    
    f1 = x**2 + y**2 - 1
    f2 = x - y
    system = PolynomialSystem([f1, f2])
    
    # Solve the system
    solutions = solve(system, tol=1e-8)
    
    # Check that we found the right number of solutions
    assert len(solutions) == 2
    
    # Verify the solutions
    expected_solutions = [
        (1/np.sqrt(2), 1/np.sqrt(2)),
        (-1/np.sqrt(2), -1/np.sqrt(2))
    ]
    
    # Helper to check if two complex numbers are close
    def is_close(a, b, tol=1e-8):
        return abs(a - b) < tol
    
    # Check each solution
    for expected_sol in expected_solutions:
        # Look for a matching solution
        found_match = False
        for sol in solutions:
            x_val = sol.values[x]
            y_val = sol.values[y]
            
            # Compare with expected (allowing for numerical error)
            if (is_close(x_val.real, expected_sol[0]) and
                is_close(y_val.real, expected_sol[1]) and
                is_close(x_val.imag, 0) and
                is_close(y_val.imag, 0)):
                found_match = True
                break
        
        assert found_match, f"Solution {expected_sol} not found"


def test_circle_parabola():
    """Test the circle-parabola intersection problem."""
    # Define variables
    x, y = polyvar('x', 'y')
    
    # Define system:
    # x^2 + y^2 = 1 (circle)
    # x^2 = y (parabola)
    
    f1 = x**2 + y**2 - 1
    f2 = x**2 - y
    system = PolynomialSystem([f1, f2])
    
    # Solve the system
    solutions = solve(system, tol=1e-8)
    
    # This system should have 2 solutions
    assert len(solutions) == 2
    
    # Check if solutions satisfy both equations
    for sol in solutions:
        x_val = sol.values[x]
        y_val = sol.values[y]
        
        # Evaluate equations
        eq1_val = x_val**2 + y_val**2 - 1
        eq2_val = x_val**2 - y_val
        
        # Both should be close to zero
        assert abs(eq1_val) < 1e-6
        assert abs(eq2_val) < 1e-6
        
        # For this system, y should equal x^2
        assert abs(y_val - x_val**2) < 1e-6


def test_solution_filtering():
    """Test the solution filtering functionality."""
    # Define variables
    x, y = polyvar('x', 'y')
    
    # Define system with complex solutions:
    # x^2 + y^2 = -1
    # x = y
    
    f1 = x**2 + y**2 + 1  # No real solutions
    f2 = x - y
    system = PolynomialSystem([f1, f2])
    
    # Solve the system
    solutions = solve(system, tol=1e-8)
    
    # This system should have 2 solutions (both complex)
    assert len(solutions) == 2
    
    # Check that all solutions are complex
    for sol in solutions:
        x_val = sol.values[x]
        y_val = sol.values[y]
        
        # Both should have non-zero imaginary parts
        assert abs(x_val.imag) > 0.1
        assert abs(y_val.imag) > 0.1
    
    # Filter for real solutions (should be none)
    real_solutions = solutions.filter(real=True)
    assert len(real_solutions) == 0