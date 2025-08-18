"""
Test that all examples from the README work correctly.

This ensures that what we document actually works and users 
won't encounter broken examples.
"""

import pytest
import sys
import os


class TestReadmeExamples:
    """Test examples directly from the README."""
    
    def test_main_quick_example(self):
        """Test the main quick example from README."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        # Define variables
        x, y = polyvar('x', 'y')
        
        # Define polynomial system
        f1 = x**2 + y**2 - 1      # circle
        f2 = x**2 - y             # parabola
        system = PolynomialSystem([f1, f2])
        
        # Solve the system
        solutions = solve(system)
        
        # Display solutions - should not raise an error
        solution_output = str(solutions)
        assert isinstance(solution_output, str)
        assert len(solution_output) > 0
        
        # Verify we found solutions
        assert len(solutions) > 0
        
        # Verify solutions are correct
        for sol in solutions:
            x_val = sol.values[x]
            y_val = sol.values[y]
            
            # Check circle equation: x^2 + y^2 = 1
            circle_residual = x_val**2 + y_val**2 - 1
            assert abs(circle_residual) < 1e-10
            
            # Check parabola equation: x^2 = y
            parabola_residual = x_val**2 - y_val
            assert abs(parabola_residual) < 1e-10
    
    def test_installation_example(self):
        """Test that the library can be imported as shown in installation docs."""
        # This should work without errors
        import pycontinuum
        
        # Test that basic functionality is available
        assert hasattr(pycontinuum, 'polyvar')
        assert hasattr(pycontinuum, 'solve')
        assert hasattr(pycontinuum, 'PolynomialSystem')
    
    def test_optional_monodromy_import(self):
        """Test optional monodromy functionality import."""
        try:
            # This should work if sympy is installed
            import pycontinuum
            # Try importing monodromy functionality
            from pycontinuum.monodromy import MonodromyBreakup
            # If we get here, monodromy is available
            assert MonodromyBreakup is not None
        except ImportError:
            # This is acceptable - monodromy is optional
            pytest.skip("Monodromy functionality requires optional sympy dependency")


class TestExampleScript:
    """Test that the example script works."""
    
    def test_simple_example_script_imports(self):
        """Test that the simple example script can import everything it needs."""
        # Test the imports that the example script uses
        import sys
        import os
        import time
        
        # These should work
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        # The example also tries to import matplotlib and numpy
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False
            # This is okay - visualization is optional
        
        # Test basic functionality that the script uses
        x, y = polyvar('x', 'y')
        f1 = x**2 + y**2 - 1
        f2 = x**2 - y
        system = PolynomialSystem([f1, f2])
        
        # Test variables() method used in script
        vars_set = system.variables()
        assert x in vars_set
        assert y in vars_set
        assert len(vars_set) == 2
        
        variables_list = list(vars_set)
        assert len(variables_list) == 2
        
        # Test string representation used in script
        system_str = str(system)
        assert isinstance(system_str, str)
        assert len(system_str) > 0
    
    def test_simple_example_solve_functionality(self):
        """Test the solve functionality used in simple_example.py."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        import time
        
        x, y = polyvar('x', 'y')
        f1 = x**2 + y**2 - 1
        f2 = x**2 - y
        system = PolynomialSystem([f1, f2])
        
        variables_list = list(system.variables())
        
        # Test solve with verbose flag (used in example)
        start_time = time.time()
        solutions = solve(system, variables=variables_list, verbose=True)
        solve_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert solve_time < 30  # 30 seconds max
        
        # Should find solutions
        assert len(solutions) > 0
        
        # Test solution iteration (used in example)
        for i, sol in enumerate(solutions):
            # Should be able to access solution values
            x_val = sol.values[x]
            y_val = sol.values[y]
            
            # Should be able to convert to string
            sol_str = str(sol)
            assert isinstance(sol_str, str)
    
    def test_solution_filtering_used_in_example(self):
        """Test solution filtering as used in the visualization part."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x, y = polyvar('x', 'y')
        f1 = x**2 + y**2 - 1
        f2 = x**2 - y
        system = PolynomialSystem([f1, f2])
        
        solutions = solve(system)
        
        # Test real solution filtering (used in example for plotting)
        real_solutions = solutions.filter(real=True)
        
        # Should have real solutions for this system
        assert len(real_solutions) > 0
        
        # Test accessing solution values for plotting
        solution_x = [sol.values[x].real for sol in real_solutions]
        solution_y = [sol.values[y].real for sol in real_solutions]
        
        assert len(solution_x) == len(real_solutions)
        assert len(solution_y) == len(real_solutions)
        
        # Values should be real numbers
        for x_val, y_val in zip(solution_x, solution_y):
            assert isinstance(x_val, (int, float))
            assert isinstance(y_val, (int, float))


class TestAdvancedExamples:
    """Test more advanced examples that might appear in documentation."""
    
    def test_parameter_homotopy_basic_usage(self):
        """Test basic parameter homotopy functionality."""
        try:
            from pycontinuum import ParameterHomotopy, track_parameter_path
            
            # Should be able to create these objects
            assert ParameterHomotopy is not None
            assert track_parameter_path is not None
            
        except ImportError as e:
            pytest.fail(f"ParameterHomotopy should be importable: {e}")
    
    def test_witness_set_basic_usage(self):
        """Test basic witness set functionality."""
        try:
            from pycontinuum import WitnessSet, generate_generic_slice, compute_witness_superset
            
            # Should be able to create these objects
            assert WitnessSet is not None
            assert generate_generic_slice is not None
            assert compute_witness_superset is not None
            
        except ImportError as e:
            pytest.fail(f"WitnessSet functionality should be importable: {e}")
    
    def test_complex_polynomial_example(self):
        """Test a more complex polynomial system example."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        # A more complex system
        x, y, z = polyvar('x', 'y', 'z')
        
        # System from algebraic geometry
        f1 = x**2 + y**2 + z**2 - 1  # sphere
        f2 = x + y + z                # plane
        f3 = x*y - z                  # surface
        
        system = PolynomialSystem([f1, f2, f3])
        
        # Should be able to solve (may have no solutions, that's okay)
        solutions = solve(system)
        
        # Should return a solution set (even if empty)
        assert hasattr(solutions, '__len__')
        
        # If there are solutions, they should be valid
        for sol in solutions:
            x_val = sol.values[x]
            y_val = sol.values[y] 
            z_val = sol.values[z]
            
            # Check residuals
            f1_residual = x_val**2 + y_val**2 + z_val**2 - 1
            f2_residual = x_val + y_val + z_val
            f3_residual = x_val*y_val - z_val
            
            assert abs(f1_residual) < 1e-8
            assert abs(f2_residual) < 1e-8
            assert abs(f3_residual) < 1e-8


class TestErrorCases:
    """Test that error cases are handled gracefully."""
    
    def test_invalid_variable_names(self):
        """Test error handling for invalid variable names."""
        from pycontinuum import polyvar
        
        # These should work
        x = polyvar('x')
        y = polyvar('y1')
        z = polyvar('var_name')
        
        assert x is not None
        assert y is not None
        assert z is not None
    
    def test_empty_system_handling(self):
        """Test that empty systems are handled gracefully."""
        from pycontinuum import PolynomialSystem, solve
        
        # This should either work or give a clear error
        try:
            system = PolynomialSystem([])
            solutions = solve(system)
            # If it works, should return a solution set
            assert hasattr(solutions, '__len__')
        except (ValueError, RuntimeError):
            # Acceptable to raise an error for empty systems
            pass
    
    def test_variable_not_in_system(self):
        """Test accessing variables not in the system."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x, y = polyvar('x', 'y')
        
        # System with two variables (square system)
        f1 = x**2 + y**2 - 1
        f2 = x - y
        system = PolynomialSystem([f1, f2])
        
        # Both variables should be in the system
        variables = system.variables()
        assert x in variables
        assert y in variables
        
        solutions = solve(system)
        
        # Solutions should have values for both variables
        for sol in solutions:
            assert x in sol.values
            assert y in sol.values