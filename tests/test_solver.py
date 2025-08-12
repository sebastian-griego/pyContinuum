# tests/test_solver.py
import pytest
import numpy as np

from pycontinuum import solve, Solution, SolutionSet, polyvar, PolynomialSystem

def test_solve_very_simple_system_x_squared_minus_one():
    """
    Tests solving x^2 - 1 = 0. Solutions: x = 1, x = -1.
    """
    x = polyvar('x')
    
    # Create system: x^2 - 1 = 0
    system = PolynomialSystem([x**2 - 1])
    
    # Solve the system
    solution_set = solve(system, verbose=False)
    
    assert solution_set is not None
    assert isinstance(solution_set, SolutionSet)
    assert len(solution_set.solutions) == 2
    
    # Extract real solutions and sort them
    real_solutions = []
    for sol in solution_set.solutions:
        if np.isclose(sol.values[x].imag, 0, atol=1e-10):
            real_solutions.append(sol.values[x].real)
    
    real_solutions.sort()
    assert np.allclose(real_solutions, [-1.0, 1.0], atol=1e-5)
    
    # Check residuals
    for sol in solution_set.solutions:
        assert sol.residual < 1e-5

def test_solve_linear_system_two_vars():
    """
    Tests x - 1 = 0, y - 2 = 0. Solution: (x,y) = (1,2)
    """
    x, y = polyvar('x', 'y')
    
    # Create system
    system = PolynomialSystem([x - 1, y - 2])
    
    # Solve
    solution_set = solve(system, verbose=False)
    
    assert solution_set is not None
    assert isinstance(solution_set, SolutionSet)
    assert len(solution_set.solutions) == 1
    
    sol = solution_set.solutions[0]
    assert np.isclose(sol.values[x].real, 1.0, atol=1e-5)
    assert np.isclose(sol.values[y].real, 2.0, atol=1e-5)
    assert np.isclose(sol.values[x].imag, 0.0, atol=1e-5)
    assert np.isclose(sol.values[y].imag, 0.0, atol=1e-5)
    assert sol.residual < 1e-5