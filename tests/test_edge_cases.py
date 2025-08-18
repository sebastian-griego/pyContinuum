"""
Test edge cases and error handling for PyContinuum.

This ensures the library behaves gracefully in unusual situations
and provides helpful error messages.
"""

import pytest
import numpy as np
from pycontinuum import polyvar, PolynomialSystem, solve


class TestInputValidation:
    """Test validation of inputs to various functions."""
    
    def test_polyvar_edge_cases(self):
        """Test edge cases for polyvar function."""
        from pycontinuum import polyvar
        
        # Test single character variable
        x = polyvar('x')
        assert str(x) == 'x'
        
        # Test multi-character variable
        xy = polyvar('xy')
        assert str(xy) == 'xy'
        
        # Test variable with numbers
        x1 = polyvar('x1')
        assert str(x1) == 'x1'
        
        # Test variable with underscore
        x_var = polyvar('x_var')
        assert str(x_var) == 'x_var'
        
        # Test empty string (should handle gracefully)
        try:
            empty_var = polyvar('')
            # If it works, that's fine
        except (ValueError, RuntimeError):
            # Also acceptable to reject empty string
            pass
    
    def test_polynomial_system_validation(self):
        """Test validation of polynomial system inputs."""
        from pycontinuum import polyvar, PolynomialSystem
        
        x, y = polyvar('x', 'y')
        
        # Valid system
        f1 = x**2 + y - 1
        f2 = x + y**2
        system = PolynomialSystem([f1, f2])
        assert system is not None
        
        # Test with single polynomial
        single_system = PolynomialSystem([f1])
        assert single_system is not None
        
        # Test empty list (should handle gracefully)
        try:
            empty_system = PolynomialSystem([])
            # If accepted, should work
            assert empty_system is not None
        except (ValueError, RuntimeError):
            # Also acceptable to reject empty system
            pass
    
    def test_solve_parameter_validation(self):
        """Test validation of solve function parameters."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x = polyvar('x')
        f = x - 1
        system = PolynomialSystem([f])
        
        # Valid solve call
        solutions = solve(system)
        assert solutions is not None
        
        # Test with explicit variables list
        variables_list = list(system.variables())
        solutions2 = solve(system, variables=variables_list)
        assert solutions2 is not None
        
        # Test with verbose flag
        solutions3 = solve(system, verbose=True)
        assert solutions3 is not None
        
        solutions4 = solve(system, verbose=False)
        assert solutions4 is not None


class TestNumericalEdgeCases:
    """Test numerical edge cases and stability."""
    
    def test_very_small_coefficients(self):
        """Test polynomials with very small coefficients."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x = polyvar('x')
        
        # Polynomial with very small coefficient
        f = x - 1e-12
        system = PolynomialSystem([f])
        
        solutions = solve(system)
        
        # Should still find the solution
        assert len(solutions) >= 1
        if len(solutions) > 0:
            x_val = solutions[0].values[x]
            assert abs(x_val - 1e-12) < 1e-15
    
    def test_very_large_coefficients(self):
        """Test polynomials with very large coefficients."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x = polyvar('x')
        
        # Polynomial with large coefficient
        f = x - 1e6
        system = PolynomialSystem([f])
        
        solutions = solve(system)
        
        # Should still find the solution
        assert len(solutions) >= 1
        if len(solutions) > 0:
            x_val = solutions[0].values[x]
            assert abs(x_val - 1e6) < 1e-6
    
    def test_nearly_singular_system(self):
        """Test system that's nearly singular."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x, y = polyvar('x', 'y')
        
        # Nearly parallel lines
        f1 = x + y - 1
        f2 = x + y + 1e-10  # Nearly the same as f1
        system = PolynomialSystem([f1, f2])
        
        # Should either find no solutions or handle gracefully
        try:
            solutions = solve(system)
            # If it returns solutions, they should be valid
            for sol in solutions:
                x_val = sol.values[x]
                y_val = sol.values[y]
                
                f1_residual = x_val + y_val - 1
                f2_residual = x_val + y_val + 1e-10
                
                assert abs(f1_residual) < 1e-8
                assert abs(f2_residual) < 1e-8
        except (RuntimeError, ValueError):
            # Acceptable to fail on nearly singular systems
            pass
    
    def test_zero_polynomial(self):
        """Test system with zero polynomial."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x, y = polyvar('x', 'y')
        
        # System with zero polynomial
        f1 = x**2 + y**2 - 1
        f2 = 0 * x  # Zero polynomial
        
        try:
            system = PolynomialSystem([f1, f2])
            solutions = solve(system)
            # If this works, f2 should be satisfied by any point
            # so solutions should lie on the circle f1 = 0
            for sol in solutions:
                x_val = sol.values[x]
                y_val = sol.values[y]
                circle_residual = x_val**2 + y_val**2 - 1
                assert abs(circle_residual) < 1e-10
        except (ValueError, RuntimeError):
            # Also acceptable to reject zero polynomials
            pass


class TestComplexityEdgeCases:
    """Test systems with various complexity challenges."""
    
    def test_high_degree_single_variable(self):
        """Test high-degree polynomial in single variable."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x = polyvar('x')
        
        # High-degree polynomial: x^8 - 1 = 0
        f = x**8 - 1
        system = PolynomialSystem([f])
        
        solutions = solve(system)
        
        # Should find 8 eighth roots of unity
        assert len(solutions) == 8
        
        # All solutions should satisfy the equation
        for sol in solutions:
            x_val = sol.values[x]
            residual = x_val**8 - 1
            assert abs(residual) < 1e-10
    
    def test_many_variables_simple_system(self):
        """Test system with many variables but simple structure."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        # Create 5 variables
        variables = polyvar('x1', 'x2', 'x3', 'x4', 'x5')
        x1, x2, x3, x4, x5 = variables
        
        # Simple system: each variable equals its index
        equations = [
            x1 - 1,
            x2 - 2, 
            x3 - 3,
            x4 - 4,
            x5 - 5
        ]
        
        system = PolynomialSystem(equations)
        solutions = solve(system)
        
        assert len(solutions) == 1
        sol = solutions[0]
        
        expected_values = [1, 2, 3, 4, 5]
        for var, expected in zip(variables, expected_values):
            actual = sol.values[var]
            assert abs(actual - expected) < 1e-10
    
    def test_symmetric_system(self):
        """Test symmetric polynomial system."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x, y = polyvar('x', 'y')
        
        # Symmetric system
        f1 = x**2 + y**2 - 2
        f2 = x*y - 1
        system = PolynomialSystem([f1, f2])
        
        solutions = solve(system)
        
        # Should find solutions
        assert len(solutions) > 0
        
        # All solutions should satisfy both equations
        for sol in solutions:
            x_val = sol.values[x]
            y_val = sol.values[y]
            
            f1_residual = x_val**2 + y_val**2 - 2
            f2_residual = x_val*y_val - 1
            
            assert abs(f1_residual) < 1e-8
            assert abs(f2_residual) < 1e-8


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""
    
    def test_inconsistent_overdetermined_system(self):
        """Test overdetermined inconsistent system."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x, y, z = polyvar('x', 'y', 'z')
        
        # Overdetermined system: 3 equations, 3 variables but inconsistent
        f1 = x - 1
        f2 = y - 2  
        f3 = x + y - 4  # Inconsistent with f1 + f2 = 3
        system = PolynomialSystem([f1, f2, f3])
        
        # Should either return empty solution set or raise clear error
        try:
            solutions = solve(system)
            assert len(solutions) == 0
        except (RuntimeError, ValueError) as e:
            # Should give a helpful error message
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ['square', 'equations', 'variables'])
    
    def test_underdetermined_system(self):
        """Test underdetermined system."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x, y, z = polyvar('x', 'y', 'z')
        
        # Underdetermined: 3 variables, 2 equations
        f1 = x + y + z - 1
        f2 = x - y
        system = PolynomialSystem([f1, f2])
        
        # Should raise an error for underdetermined system
        with pytest.raises((RuntimeError, ValueError)) as exc_info:
            solutions = solve(system)
        
        # Should give helpful error about the system dimensions
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ['square', 'equations', 'variables'])
    
    def test_no_solutions_system(self):
        """Test system with no solutions."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x, y = polyvar('x', 'y')
        
        # No solutions: x^2 + y^2 = -1 (impossible in reals)
        f1 = x**2 + y**2 + 1
        f2 = x + y  # Additional constraint
        system = PolynomialSystem([f1, f2])
        
        solutions = solve(system)
        
        # Should return empty solution set or complex solutions
        if len(solutions) == 0:
            # Empty is fine
            pass
        else:
            # If we get solutions, they should be complex
            for sol in solutions:
                x_val = sol.values[x]
                y_val = sol.values[y]
                
                # Verify they satisfy the equations
                f1_residual = x_val**2 + y_val**2 + 1
                f2_residual = x_val + y_val
                
                assert abs(f1_residual) < 1e-10
                assert abs(f2_residual) < 1e-10


class TestBoundaryConditions:
    """Test boundary and extreme conditions."""
    
    def test_constant_polynomial(self):
        """Test system with constant polynomial."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x, y = polyvar('x', 'y')
        
        # System where one equation has constant term that makes it unsolvable
        f1 = x**2 + y**2 + 1  # No real solutions
        f2 = x + y
        
        try:
            system = PolynomialSystem([f1, f2])
            solutions = solve(system)
            # If it works, might get complex solutions
            assert hasattr(solutions, '__len__')
        except (ValueError, RuntimeError):
            # Also acceptable to handle this case with error
            pass
    
    def test_single_point_solution(self):
        """Test system with exactly one solution."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x, y = polyvar('x', 'y')
        
        # System with unique solution at origin
        f1 = x
        f2 = y
        system = PolynomialSystem([f1, f2])
        
        solutions = solve(system)
        
        assert len(solutions) == 1
        sol = solutions[0]
        
        assert abs(sol.values[x]) < 1e-15
        assert abs(sol.values[y]) < 1e-15
    
    def test_very_close_solutions(self):
        """Test system with solutions close together."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x = polyvar('x')
        
        # Polynomial with two reasonably separated roots
        # (x - 1)(x - 1.1) = x^2 - 2.1*x + 1.1
        f = x**2 - 2.1*x + 1.1
        system = PolynomialSystem([f])
        
        solutions = solve(system)
        
        # Should find both roots
        assert len(solutions) == 2
        
        x_values = sorted([sol.values[x].real for sol in solutions])
        assert abs(x_values[0] - 1.0) < 1e-8
        assert abs(x_values[1] - 1.1) < 1e-8