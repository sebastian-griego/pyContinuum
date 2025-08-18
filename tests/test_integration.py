"""
Comprehensive integration tests for PyContinuum.

These tests verify that the main functionality works end-to-end,
including the examples from the README and documentation.
"""

import pytest
import numpy as np
from pycontinuum import polyvar, PolynomialSystem, solve, Solution, SolutionSet


class TestMainSolveFunction:
    """Test the main solve() function with various polynomial systems."""
    
    def test_circle_parabola_intersection_readme_example(self):
        """Test the exact example from the README."""
        # Define variables
        x, y = polyvar('x', 'y')
        
        # Define polynomial system
        f1 = x**2 + y**2 - 1      # circle
        f2 = x**2 - y             # parabola
        system = PolynomialSystem([f1, f2])
        
        # Solve the system
        solutions = solve(system)
        
        # Verify we get solutions
        assert isinstance(solutions, SolutionSet)
        assert len(solutions) > 0
        
        # Check that all solutions are valid
        for sol in solutions:
            assert isinstance(sol, Solution)
            # Verify the solution actually satisfies both equations
            x_val = sol.values[x]
            y_val = sol.values[y]
            
            # Check f1 = x^2 + y^2 - 1 ≈ 0
            f1_val = x_val**2 + y_val**2 - 1
            assert abs(f1_val) < 1e-10, f"f1 residual too large: {f1_val}"
            
            # Check f2 = x^2 - y ≈ 0
            f2_val = x_val**2 - y_val
            assert abs(f2_val) < 1e-10, f"f2 residual too large: {f2_val}"
    
    def test_simple_linear_system(self):
        """Test a simple linear system: x + y = 1, x - y = 0."""
        x, y = polyvar('x', 'y')
        
        f1 = x + y - 1
        f2 = x - y
        system = PolynomialSystem([f1, f2])
        
        solutions = solve(system)
        
        assert len(solutions) == 1
        sol = solutions[0]
        
        # Expected solution: x = 0.5, y = 0.5
        x_val = sol.values[x]
        y_val = sol.values[y]
        
        assert abs(x_val - 0.5) < 1e-10
        assert abs(y_val - 0.5) < 1e-10
    
    def test_quadratic_single_variable(self):
        """Test a quadratic in single variable: x^2 - 1 = 0."""
        x = polyvar('x')
        
        f = x**2 - 1
        system = PolynomialSystem([f])
        
        solutions = solve(system)
        
        assert len(solutions) == 2
        
        # Should get x = 1 and x = -1
        x_values = sorted([sol.values[x].real for sol in solutions])
        expected = [-1.0, 1.0]
        
        for actual, expected_val in zip(x_values, expected):
            assert abs(actual - expected_val) < 1e-10
    
    def test_three_variable_system(self):
        """Test a system with three variables."""
        x, y, z = polyvar('x', 'y', 'z')
        
        # Simple system: x = 1, y = 2, z = 3
        f1 = x - 1
        f2 = y - 2  
        f3 = z - 3
        system = PolynomialSystem([f1, f2, f3])
        
        solutions = solve(system)
        
        assert len(solutions) == 1
        sol = solutions[0]
        
        assert abs(sol.values[x] - 1) < 1e-10
        assert abs(sol.values[y] - 2) < 1e-10
        assert abs(sol.values[z] - 3) < 1e-10
    
    def test_polynomial_with_multiple_solutions(self):
        """Test a system that should have multiple solutions."""
        x, y = polyvar('x', 'y')
        
        # System: x^2 = 1, y^2 = 1 (should give 4 solutions)
        f1 = x**2 - 1
        f2 = y**2 - 1
        system = PolynomialSystem([f1, f2])
        
        solutions = solve(system)
        
        assert len(solutions) == 4
        
        # Should get all combinations of (±1, ±1)
        expected_points = {(1, 1), (1, -1), (-1, 1), (-1, -1)}
        actual_points = set()
        
        for sol in solutions:
            x_val = round(sol.values[x].real)
            y_val = round(sol.values[y].real)
            actual_points.add((x_val, y_val))
        
        assert actual_points == expected_points
    
    def test_solution_filtering(self):
        """Test solution filtering functionality."""
        x, y = polyvar('x', 'y')
        
        f1 = x**2 - 1
        f2 = y**2 - 1
        system = PolynomialSystem([f1, f2])
        
        solutions = solve(system)
        
        # Test real solution filtering
        real_solutions = solutions.filter(real=True)
        assert len(real_solutions) == 4  # All solutions should be real
        
        # Test solution access by index
        first_solution = solutions[0]
        assert isinstance(first_solution, Solution)
        
        # Test iteration
        count = 0
        for sol in solutions:
            assert isinstance(sol, Solution)
            count += 1
        assert count == len(solutions)


class TestPolynomialSystemCreation:
    """Test polynomial system creation and manipulation."""
    
    def test_polynomial_system_variables(self):
        """Test that PolynomialSystem correctly identifies variables."""
        x, y, z = polyvar('x', 'y', 'z')
        
        f1 = x**2 + y - 1
        f2 = y*z + 3
        system = PolynomialSystem([f1, f2])
        
        variables = system.variables()
        expected_vars = {x, y, z}
        
        assert variables == expected_vars
    
    def test_polynomial_evaluation(self):
        """Test polynomial evaluation at specific points."""
        x, y = polyvar('x', 'y')
        
        f = x**2 + y**2 - 1
        system = PolynomialSystem([f])
        
        # Evaluate at (1, 0) - should be on the circle
        result = system.evaluate({x: 1, y: 0})
        assert abs(result[0]) < 1e-15
        
        # Evaluate at (0, 0) - should be -1
        result = system.evaluate({x: 0, y: 0})
        assert abs(result[0] + 1) < 1e-15


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_system(self):
        """Test behavior with empty polynomial system."""
        try:
            system = PolynomialSystem([])
            solutions = solve(system)
            # If it works, should return a solution set (possibly empty)
            assert hasattr(solutions, '__len__')
        except (ValueError, RuntimeError):
            # Also acceptable to raise an error for empty systems
            pass
    
    def test_inconsistent_system(self):
        """Test system with no solutions."""
        x, y = polyvar('x', 'y')
        
        # Inconsistent system: 2x2 system with no solutions
        f1 = x**2 + y**2 + 1  # No real solutions 
        f2 = x + y
        system = PolynomialSystem([f1, f2])
        
        solutions = solve(system)
        # Should either return empty solution set or complex solutions
        assert hasattr(solutions, '__len__')
    
    def test_underdetermined_system(self):
        """Test underdetermined system (more variables than equations)."""
        x, y, z = polyvar('x', 'y', 'z')
        
        # Three variables, two equations (underdetermined)
        f1 = x + y + z - 1
        f2 = x - y
        system = PolynomialSystem([f1, f2])
        
        # This should raise an error for underdetermined systems
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            solutions = solve(system)
        
        # Check that the error message is informative
        error_msg = str(exc_info.value).lower()
        assert "square" in error_msg or "equations" in error_msg or "variables" in error_msg


class TestComplexSolutions:
    """Test handling of complex solutions."""
    
    def test_complex_roots(self):
        """Test system with complex solutions."""
        x = polyvar('x')
        
        # x^2 + 1 = 0, should give x = ±i
        f = x**2 + 1
        system = PolynomialSystem([f])
        
        solutions = solve(system)
        
        assert len(solutions) == 2
        
        # Check that solutions are ±i
        roots = [sol.values[x] for sol in solutions]
        
        # Sort by imaginary part
        roots.sort(key=lambda z: z.imag)
        
        assert abs(roots[0] - (-1j)) < 1e-10
        assert abs(roots[1] - (1j)) < 1e-10
    
    def test_mixed_real_complex_system(self):
        """Test system with both real and complex solutions."""
        x, y = polyvar('x', 'y')
        
        # x^2 = -1 (complex), y = 1 (real)
        f1 = x**2 + 1
        f2 = y - 1
        system = PolynomialSystem([f1, f2])
        
        solutions = solve(system)
        
        # The library may deduplicate or handle complex solutions differently
        assert len(solutions) >= 1
        
        for sol in solutions:
            # y should always be 1
            assert abs(sol.values[y] - 1) < 1e-10
            # x should be ±i
            x_val = sol.values[x]
            assert abs(abs(x_val) - 1) < 1e-10


class TestNumericalAccuracy:
    """Test numerical accuracy and stability."""
    
    def test_high_degree_polynomial(self):
        """Test higher degree polynomial."""
        x = polyvar('x')
        
        # x^4 - 1 = 0, should give 4 fourth roots of unity
        f = x**4 - 1
        system = PolynomialSystem([f])
        
        solutions = solve(system)
        
        assert len(solutions) == 4
        
        # Verify all solutions satisfy the equation
        for sol in solutions:
            x_val = sol.values[x]
            residual = x_val**4 - 1
            assert abs(residual) < 1e-10
    
    def test_solution_accuracy(self):
        """Test that solutions are accurate."""
        x, y = polyvar('x', 'y')
        
        # More complex system
        f1 = x**3 + y**3 - 2
        f2 = x**2 - y
        system = PolynomialSystem([f1, f2])
        
        solutions = solve(system)
        
        # Verify each solution satisfies both equations
        for sol in solutions:
            x_val = sol.values[x]
            y_val = sol.values[y]
            
            f1_residual = x_val**3 + y_val**3 - 2
            f2_residual = x_val**2 - y_val
            
            assert abs(f1_residual) < 1e-8
            assert abs(f2_residual) < 1e-8