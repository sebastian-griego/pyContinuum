"""
Test the public API and import functionality of PyContinuum.

This ensures that all public functions and classes are properly exposed
and work as documented.
"""

import pytest


class TestImports:
    """Test that all public API elements can be imported correctly."""
    
    def test_main_imports(self):
        """Test importing main classes and functions."""
        from pycontinuum import (
            polyvar,
            Variable,
            Monomial, 
            Polynomial,
            PolynomialSystem,
            solve,
            Solution,
            SolutionSet
        )
        
        # Test that imports are not None
        assert polyvar is not None
        assert Variable is not None
        assert Monomial is not None
        assert Polynomial is not None
        assert PolynomialSystem is not None
        assert solve is not None
        assert Solution is not None
        assert SolutionSet is not None
    
    def test_witness_set_imports(self):
        """Test witness set related imports."""
        from pycontinuum import (
            WitnessSet,
            generate_generic_slice,
            compute_witness_superset
        )
        
        assert WitnessSet is not None
        assert generate_generic_slice is not None
        assert compute_witness_superset is not None
    
    def test_parameter_homotopy_imports(self):
        """Test parameter homotopy imports."""
        from pycontinuum import (
            ParameterHomotopy,
            track_parameter_path
        )
        
        assert ParameterHomotopy is not None
        assert track_parameter_path is not None
    
    def test_monodromy_imports(self):
        """Test monodromy imports (should work with optional dependency)."""
        try:
            from pycontinuum.monodromy import (
                MonodromyBreakup,
                trace_monodromy_loops
            )
            assert MonodromyBreakup is not None
            assert trace_monodromy_loops is not None
        except ImportError:
            # This is acceptable if sympy is not installed
            pytest.skip("Monodromy functionality requires optional sympy dependency")
    
    def test_star_import(self):
        """Test that star import works and includes expected symbols."""
        # Import everything from pycontinuum
        import pycontinuum
        
        # Check that __all__ is defined
        assert hasattr(pycontinuum, '__all__')
        
        # Check that key functions are in __all__
        expected_in_all = [
            'polyvar',
            'Variable', 
            'Polynomial',
            'PolynomialSystem',
            'solve',
            'Solution',
            'SolutionSet'
        ]
        
        for symbol in expected_in_all:
            assert symbol in pycontinuum.__all__
    
    def test_version_available(self):
        """Test that version information is available."""
        import pycontinuum
        
        assert hasattr(pycontinuum, '__version__')
        assert isinstance(pycontinuum.__version__, str)
        assert len(pycontinuum.__version__) > 0


class TestPolynomialCreation:
    """Test polynomial creation and basic operations."""
    
    def test_polyvar_single_variable(self):
        """Test creating a single variable."""
        from pycontinuum import polyvar
        
        x = polyvar('x')
        
        assert x is not None
        assert str(x) == 'x'
    
    def test_polyvar_multiple_variables(self):
        """Test creating multiple variables."""
        from pycontinuum import polyvar
        
        x, y, z = polyvar('x', 'y', 'z')
        
        assert str(x) == 'x'
        assert str(y) == 'y' 
        assert str(z) == 'z'
        
        # Variables should be different objects
        assert x is not y
        assert y is not z
        assert x is not z
    
    def test_polynomial_creation(self):
        """Test creating polynomials."""
        from pycontinuum import polyvar, Polynomial
        
        x, y = polyvar('x', 'y')
        
        # Test simple polynomial creation
        p1 = x**2 + y
        assert isinstance(p1, Polynomial)
        
        # Test more complex polynomial
        p2 = x**3 + 2*x*y + y**2 - 5
        assert isinstance(p2, Polynomial)
    
    def test_polynomial_system_creation(self):
        """Test creating polynomial systems."""
        from pycontinuum import polyvar, PolynomialSystem
        
        x, y = polyvar('x', 'y')
        
        f1 = x**2 + y**2 - 1
        f2 = x - y
        
        system = PolynomialSystem([f1, f2])
        assert isinstance(system, PolynomialSystem)
        
        # Test that system knows its variables
        variables = system.variables()
        assert x in variables
        assert y in variables


class TestSolutionObjects:
    """Test Solution and SolutionSet objects."""
    
    def test_solution_creation_and_access(self):
        """Test that solutions can be created and accessed."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x = polyvar('x')
        f = x - 1
        system = PolynomialSystem([f])
        
        solutions = solve(system)
        
        # Test SolutionSet properties
        assert len(solutions) == 1
        assert isinstance(solutions, (list, object))  # SolutionSet should be iterable
        
        # Test Solution properties
        sol = solutions[0]
        assert hasattr(sol, 'values')
        assert x in sol.values
        
        # Test solution value access
        x_val = sol.values[x]
        assert abs(x_val - 1) < 1e-10
    
    def test_solution_filtering(self):
        """Test solution filtering capabilities."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x = polyvar('x')
        f = x**2 + 1  # Has complex solutions
        system = PolynomialSystem([f])
        
        solutions = solve(system)
        
        # Test real filtering
        real_solutions = solutions.filter(real=True)
        assert len(real_solutions) == 0  # No real solutions for x^2 + 1 = 0
        
        # Test that we can still access all solutions
        assert len(solutions) == 2  # Two complex solutions


class TestExampleUsage:
    """Test typical usage patterns that users would encounter."""
    
    def test_readme_quick_example(self):
        """Test the exact quick example from README."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        # Define variables
        x, y = polyvar('x', 'y')
        
        # Define polynomial system
        f1 = x**2 + y**2 - 1      # circle
        f2 = x**2 - y             # parabola
        system = PolynomialSystem([f1, f2])
        
        # Solve the system
        solutions = solve(system)
        
        # Verify basic properties
        assert len(solutions) > 0
        
        # Should be able to print solutions (test __str__ method)
        solution_str = str(solutions)
        assert isinstance(solution_str, str)
        assert len(solution_str) > 0
    
    def test_single_equation_usage(self):
        """Test solving a single equation."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x = polyvar('x')
        f = x**2 - 4
        system = PolynomialSystem([f])
        
        solutions = solve(system)
        
        assert len(solutions) == 2
        
        # Solutions should be x = ±2
        x_values = sorted([sol.values[x].real for sol in solutions])
        assert abs(x_values[0] + 2) < 1e-10
        assert abs(x_values[1] - 2) < 1e-10
    
    def test_three_equations_three_unknowns(self):
        """Test a well-posed 3x3 system."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x, y, z = polyvar('x', 'y', 'z')
        
        # Linear system with unique solution
        f1 = x + y + z - 6
        f2 = x - y + z - 2  
        f3 = x + y - z - 2
        system = PolynomialSystem([f1, f2, f3])
        
        solutions = solve(system)
        
        assert len(solutions) == 1
        sol = solutions[0]
        
        # Expected solution: x=2, y=2, z=2
        assert abs(sol.values[x] - 2) < 1e-10
        assert abs(sol.values[y] - 2) < 1e-10  
        assert abs(sol.values[z] - 2) < 1e-10


class TestDocumentationExamples:
    """Test that examples that might appear in documentation work."""
    
    def test_polynomial_operations(self):
        """Test polynomial arithmetic operations."""
        from pycontinuum import polyvar
        
        x, y = polyvar('x', 'y')
        
        # Test basic operations
        p1 = x**2 + 1
        p2 = y**2 - 1
        
        # Test addition
        p3 = p1 + p2
        assert p3 is not None
        
        # Test multiplication  
        p4 = p1 * p2
        assert p4 is not None
        
        # Test scalar multiplication
        p5 = 3 * p1
        assert p5 is not None
        
        # Test subtraction
        p6 = p1 - p2
        assert p6 is not None
    
    def test_system_evaluation(self):
        """Test evaluating polynomial systems at points."""
        from pycontinuum import polyvar, PolynomialSystem
        
        x, y = polyvar('x', 'y')
        
        f1 = x**2 + y**2 - 1
        f2 = x - y
        system = PolynomialSystem([f1, f2])
        
        # Test evaluation at a point
        import numpy as np
        point = {x: 1/np.sqrt(2), y: 1/np.sqrt(2)}
        
        result = system.evaluate(point)
        assert isinstance(result, (list, tuple, np.ndarray))
        assert len(result) == 2  # Two equations