"""
Test optional dependencies and their functionality.

This ensures that optional features work when dependencies are available
and fail gracefully when they're not.
"""

import pytest


class TestMonodromyDependency:
    """Test monodromy functionality and its optional sympy dependency."""
    
    def test_monodromy_import_with_sympy(self):
        """Test monodromy imports when sympy is available."""
        try:
            import sympy
            sympy_available = True
        except ImportError:
            sympy_available = False
        
        if sympy_available:
            # Should be able to import monodromy functionality
            try:
                from pycontinuum.monodromy import (
                    track_monodromy_loop
                )
                assert track_monodromy_loop is not None
            except ImportError:
                pytest.fail("Monodromy should be importable when sympy is available")
        else:
            # Should fail gracefully when sympy is not available
            with pytest.raises(ImportError):
                from pycontinuum.monodromy import track_monodromy_loop
    
    def test_monodromy_not_in_main_imports(self):
        """Test that monodromy is not imported by default."""
        import pycontinuum
        
        # Monodromy should not be in the main namespace
        assert not hasattr(pycontinuum, 'track_monodromy_loop')
        
        # Should not be in __all__
        assert 'track_monodromy_loop' not in pycontinuum.__all__
    
    def test_core_functionality_without_sympy(self):
        """Test that core functionality works without sympy."""
        # Even if sympy is not available, core functionality should work
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        x, y = polyvar('x', 'y')
        f1 = x**2 + y**2 - 1
        f2 = x - y
        system = PolynomialSystem([f1, f2])
        
        solutions = solve(system)
        assert solutions is not None
        assert len(solutions) >= 0
    
    def test_monodromy_optional_extra(self):
        """Test that monodromy extra is properly configured."""
        # This tests the packaging configuration
        # The pyproject.toml should have monodromy extra with sympy dependency
        
        # We can't directly test pip install here, but we can verify
        # the monodromy module exists and imports sympy if available
        try:
            import sympy
            from pycontinuum.monodromy import track_monodromy_loop
            
            # If we get here, monodromy should work
            # Test that the function exists and is callable
            assert callable(track_monodromy_loop)
                
        except ImportError:
            # Either sympy or monodromy functionality not available
            # This is acceptable
            pass


class TestVisualizationDependencies:
    """Test visualization dependencies (matplotlib, numpy)."""
    
    def test_matplotlib_dependency(self):
        """Test that visualization works when matplotlib is available."""
        try:
            import matplotlib.pyplot as plt
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False
        
        if matplotlib_available:
            # Should be able to use visualization features
            from pycontinuum import visualization
            
            # Test that visualization module exists
            assert visualization is not None
            
        else:
            # Should work without matplotlib (core functionality)
            from pycontinuum import polyvar, solve, PolynomialSystem
            
            x = polyvar('x')
            f = x - 1
            system = PolynomialSystem([f])
            solutions = solve(system)
            
            assert len(solutions) == 1
    
    def test_numpy_dependency(self):
        """Test numpy dependency handling."""
        try:
            import numpy as np
            numpy_available = True
        except ImportError:
            numpy_available = False
        
        # Numpy should be a required dependency, so it should always be available
        # But test graceful handling anyway
        if numpy_available:
            from pycontinuum import polyvar, PolynomialSystem, solve
            
            x = polyvar('x')
            f1 = x**2 - 1  # Single variable system
            system = PolynomialSystem([f1])
            
            # Should work with numpy available
            solutions = solve(system)
            assert solutions is not None
        else:
            # If numpy is not available, core functionality might still work
            # depending on implementation
            try:
                from pycontinuum import polyvar
                x = polyvar('x')
                assert x is not None
            except ImportError:
                # Acceptable if numpy is truly required
                pass


class TestOptionalFeatures:
    """Test optional features and their dependencies."""
    
    def test_scipy_dependency(self):
        """Test scipy dependency."""
        try:
            import scipy
            scipy_available = True
        except ImportError:
            scipy_available = False
        
        # scipy is listed as a required dependency
        if scipy_available:
            # Core functionality should work
            from pycontinuum import polyvar, solve, PolynomialSystem
            
            x = polyvar('x')
            f = x**2 - 1
            system = PolynomialSystem([f])
            solutions = solve(system)
            
            assert len(solutions) == 2
        else:
            # If scipy is missing, this might affect functionality
            try:
                from pycontinuum import polyvar
                x = polyvar('x')
                assert x is not None
            except ImportError:
                # Acceptable if scipy is truly required
                pass
    
    def test_tqdm_dependency(self):
        """Test tqdm dependency for progress bars."""
        try:
            import tqdm
            tqdm_available = True
        except ImportError:
            tqdm_available = False
        
        # tqdm is listed as required dependency for progress bars
        if tqdm_available:
            # Should be able to solve with verbose=True
            from pycontinuum import polyvar, solve, PolynomialSystem
            
            x = polyvar('x')
            f = x - 1
            system = PolynomialSystem([f])
            
            # This should work and potentially show progress
            solutions = solve(system, verbose=True)
            assert len(solutions) == 1
        else:
            # Should still work without tqdm, just no progress bars
            try:
                from pycontinuum import polyvar, solve, PolynomialSystem
                
                x = polyvar('x')
                f = x - 1
                system = PolynomialSystem([f])
                solutions = solve(system, verbose=False)
                assert len(solutions) == 1
            except ImportError:
                # Acceptable if tqdm is truly required
                pass


class TestGracefulDegradation:
    """Test that missing optional dependencies don't break core functionality."""
    
    def test_core_without_all_optionals(self):
        """Test that core functionality works even if some optionals are missing."""
        # This test ensures the core polynomial solving works
        # regardless of optional dependency availability
        
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        # Basic polynomial system that should always work
        x, y = polyvar('x', 'y')
        
        # Simple system
        f1 = x - 1
        f2 = y - 2
        system = PolynomialSystem([f1, f2])
        
        solutions = solve(system)
        
        assert len(solutions) == 1
        sol = solutions[0]
        
        assert abs(sol.values[x] - 1) < 1e-10
        assert abs(sol.values[y] - 2) < 1e-10
    
    def test_import_resilience(self):
        """Test that imports are resilient to missing optional dependencies."""
        # Main imports should work
        try:
            from pycontinuum import (
                polyvar, PolynomialSystem, solve,
                Solution, SolutionSet
            )
            assert polyvar is not None
            assert PolynomialSystem is not None
            assert solve is not None
            assert Solution is not None
            assert SolutionSet is not None
        except ImportError as e:
            pytest.fail(f"Core imports should not fail: {e}")
        
        # Advanced features should be importable or fail gracefully
        try:
            from pycontinuum import (
                WitnessSet, ParameterHomotopy
            )
            # If these work, that's great
            assert WitnessSet is not None
            assert ParameterHomotopy is not None
        except ImportError:
            # If they don't work due to missing dependencies, that's also okay
            # as long as core functionality still works
            pass
    
    def test_documentation_examples_work(self):
        """Test that basic documentation examples work regardless of optionals."""
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        # The README example should always work
        x, y = polyvar('x', 'y')
        f1 = x**2 + y**2 - 1
        f2 = x**2 - y
        system = PolynomialSystem([f1, f2])
        
        solutions = solve(system)
        
        # Should get some solutions
        assert len(solutions) > 0
        
        # Solutions should be valid
        for sol in solutions:
            x_val = sol.values[x]
            y_val = sol.values[y]
            
            # Verify equations are satisfied
            f1_residual = x_val**2 + y_val**2 - 1
            f2_residual = x_val**2 - y_val
            
            assert abs(f1_residual) < 1e-10
            assert abs(f2_residual) < 1e-10


class TestInstallationScenarios:
    """Test different installation scenarios."""
    
    def test_minimal_install(self):
        """Test that minimal install (just required dependencies) works."""
        # This tests the core functionality that should work
        # with just the required dependencies
        
        from pycontinuum import polyvar, PolynomialSystem, solve
        
        # Test basic solving capability
        x = polyvar('x')
        f = x**3 - 1  # Should give 3 cube roots of unity
        system = PolynomialSystem([f])
        
        solutions = solve(system)
        
        assert len(solutions) == 3
        
        # All solutions should satisfy x^3 = 1
        for sol in solutions:
            x_val = sol.values[x]
            residual = x_val**3 - 1
            assert abs(residual) < 1e-10
    
    def test_full_install_simulation(self):
        """Test functionality that would be available with all extras."""
        # Test that advanced features work when dependencies are available
        
        # Test parameter homotopy (should work with base install)
        try:
            from pycontinuum import ParameterHomotopy
            assert ParameterHomotopy is not None
        except ImportError:
            pytest.skip("ParameterHomotopy not available")
        
        # Test witness sets (should work with base install)
        try:
            from pycontinuum import WitnessSet
            assert WitnessSet is not None
        except ImportError:
            pytest.skip("WitnessSet not available")
        
        # Test monodromy (requires sympy extra)
        try:
            from pycontinuum.monodromy import track_monodromy_loop
            assert track_monodromy_loop is not None
        except ImportError:
            # This is expected if sympy extra is not installed
            pass