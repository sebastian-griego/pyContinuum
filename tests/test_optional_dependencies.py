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
                    MonodromyBreakup,
                    trace_monodromy_loops,
                    track_monodromy_loop,
                )
                assert MonodromyBreakup is not None
                assert trace_monodromy_loops is not None
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

    def test_path_plot_accepts_generators_and_validates_records(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from pycontinuum.visualization import plot_path

        path_points = (
            sample
            for sample in [
                (1.0, [1.0 + 2.0j]),
                (0.0, [3.0 + 0.0j]),
            ]
        )

        fig = plot_path(path_points)
        try:
            assert len(fig.axes[0].collections) == 1
        finally:
            plt.close(fig)

        try:
            with pytest.raises(
                ValueError,
                match="at least one path point",
            ):
                plot_path([])
            with pytest.raises(
                TypeError,
                match="var_idx must be an integer",
            ):
                plot_path([(0.0, [1.0])], var_idx=True)
            with pytest.raises(
                ValueError,
                match=r"path_points\[0\] must be a \(t, point\) pair",
            ):
                plot_path([(0.0, [1.0], "extra")])
            with pytest.raises(
                ValueError,
                match=r"path_points\[0\]\[1\] must contain coordinate index 1",
            ):
                plot_path([(0.0, [1.0])], var_idx=1)
            with pytest.raises(
                ValueError,
                match=r"path_points\[0\]\[0\] must be finite",
            ):
                plot_path([(float("nan"), [1.0])])
        finally:
            plt.close("all")

    def test_all_paths_plot_accepts_empty_paths_and_validates_options(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from pycontinuum.visualization import plot_all_paths

        all_path_points = (
            path
            for path in [
                [(1.0, [1.0 + 1.0j]), (0.0, [2.0 + 0.0j])],
                [],
            ]
        )

        fig = plot_all_paths(all_path_points, alpha=0.25)
        empty_fig = plot_all_paths([], show_endpoints=True)
        try:
            assert len(fig.axes[0].lines) == 2
            assert fig.axes[0].get_legend() is not None
            assert empty_fig.axes[0].get_legend() is None
        finally:
            plt.close(fig)
            plt.close(empty_fig)

        try:
            with pytest.raises(
                ValueError,
                match=r"all_path_points\[0\]\[0\]\[1\] must contain coordinate index 1",
            ):
                plot_all_paths([[(0.0, [1.0])]], var_idx=1)
            with pytest.raises(
                ValueError,
                match="alpha must be between 0 and 1",
            ):
                plot_all_paths([], alpha=1.5)
        finally:
            plt.close("all")

    def test_visualization_2d_complex_marker_sizes_use_imaginary_parts(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from pycontinuum import PolynomialSystem, Solution, SolutionSet, polyvar
        from pycontinuum.visualization import plot_solutions_2d

        x, y = polyvar("x", "y")
        solutions = SolutionSet(
            [
                Solution({x: 1.0 + 0.0j, y: 2.0 + 0.0j}, residual=0.0),
                Solution({x: 1.0 + 2.0j, y: 2.0 + 4.0j}, residual=0.0),
            ],
            PolynomialSystem([x - y]),
        )

        fig = plot_solutions_2d(
            solutions,
            x,
            y,
            marker_size=10,
            real_only=False,
        )
        try:
            sizes = fig.axes[0].collections[0].get_sizes()
            assert sizes[0] == pytest.approx(10.0)
            assert sizes[1] == pytest.approx(40.0)
        finally:
            plt.close(fig)

    def test_visualization_rejects_missing_solution_coordinates(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from pycontinuum import PolynomialSystem, Solution, SolutionSet, polyvar
        from pycontinuum.visualization import plot_solutions_2d

        x, y = polyvar("x", "y")
        solutions = SolutionSet(
            [Solution({x: 1.0 + 0.0j}, residual=0.0)],
            PolynomialSystem([x - 1]),
        )
        conflicting_solutions = type(
            "SolutionSetLike",
            (),
            {
                "solutions": [
                    {x: 1.0 + 0.0j, "x": 2.0 + 0.0j, y: 1.0 + 0.0j}
                ]
            },
        )()

        try:
            with pytest.raises(ValueError, match="missing coordinate.*y"):
                plot_solutions_2d(solutions, x, y)
            with pytest.raises(ValueError, match="conflicting coordinates.*x"):
                plot_solutions_2d(conflicting_solutions, x, y)
        finally:
            plt.close("all")

    def test_parameter_continuation_accepts_coordinate_records(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from pycontinuum import Solution, polyvar
        from pycontinuum.visualization import plot_parameter_continuation

        class SolutionLike:
            def __init__(self, values):
                self.values = values

        a, x = polyvar("a", "x")
        results = [
            (0.0, [{"x": 1.0 + 0.0j}, Solution({x: 10.0 + 0.0j}, residual=0.0)]),
            (1.0, [{x: 2.0 + 1.0j}, SolutionLike({"x": 11.0 - 2.0j})]),
        ]

        fig = plot_parameter_continuation(results, a, x, plot_imag=True)
        try:
            assert len(fig.axes[0].lines) == 4
        finally:
            plt.close(fig)

    def test_parameter_continuation_matches_branches_one_to_one(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from pycontinuum import polyvar
        from pycontinuum.visualization import plot_parameter_continuation

        a, x = polyvar("a", "x")
        results = [
            (0.0, [{x: 1.0}, {x: 3.0}]),
            (1.0, [{x: 2.0}, {x: 100.0}]),
        ]

        fig = plot_parameter_continuation(results, a, x)
        try:
            y_data = [list(line.get_ydata()) for line in fig.axes[0].lines]
            assert y_data[0] == pytest.approx([1.0, 2.0])
            assert y_data[1] == pytest.approx([3.0, 100.0])
        finally:
            plt.close(fig)

    def test_parameter_continuation_validates_malformed_results(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from pycontinuum import polyvar
        from pycontinuum.visualization import plot_parameter_continuation

        a, x = polyvar("a", "x")

        try:
            with pytest.raises(
                ValueError,
                match="at least one parameter sample",
            ):
                plot_parameter_continuation([], a, x)
            with pytest.raises(
                ValueError,
                match=r"results\[0\] must be a \(parameter, solutions\) pair",
            ):
                plot_parameter_continuation([(0.0, [], "extra")], a, x)
            with pytest.raises(
                TypeError,
                match=r"results\[0\]\[0\] must be a number",
            ):
                plot_parameter_continuation([("zero", [])], a, x)
            with pytest.raises(
                TypeError,
                match=r"results\[0\]\[1\] must be an iterable",
            ):
                plot_parameter_continuation([(0.0, None)], a, x)
            with pytest.raises(
                ValueError,
                match=r"results\[0\]\[1\]\[0\].*missing coordinate.*x",
            ):
                plot_parameter_continuation([(0.0, [{"y": 1.0}])], a, x)
        finally:
            plt.close("all")
    
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

    def test_monodromy_compatibility_entry_points(self):
        """Test advertised monodromy compatibility entry points."""
        try:
            from pycontinuum import PolynomialSystem, SolutionSet, polyvar
            from pycontinuum.monodromy import (
                MonodromyBreakup,
                trace_monodromy_loops,
            )
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([x])
        slicing = PolynomialSystem([])
        witnesses = SolutionSet([], system)
        breakup = MonodromyBreakup(system, slicing, witnesses, [x], monodromy_options={"num_loops": 1})

        assert callable(trace_monodromy_loops)
        assert breakup.run() == []
        assert breakup.components == []

    def test_monodromy_match_tol_is_not_forwarded_to_parameter_tracker(self, monkeypatch):
        """Monodromy matching options should not trip strict path option checks."""
        try:
            from pycontinuum import PolynomialSystem, Solution, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        y = polyvar("y")
        system = PolynomialSystem([])
        slicing = PolynomialSystem([x - 1])
        witnesses = [Solution({x: 1.0 + 0j}, residual=0.0)]
        captured_options = []

        def fake_track_parameter_path(
            parameter_homotopy,
            start_point,
            start_t=0.0,
            end_t=1.0,
            options=None,
        ):
            captured_options.append(dict(options or {}))
            return start_point, {"success": True}

        monkeypatch.setattr(
            monodromy,
            "track_parameter_path",
            fake_track_parameter_path,
        )

        monodromy.track_monodromy_loop(
            system,
            slicing,
            witnesses,
            [x],
            num_loops=1,
            parameter_tracker_options={"match_tol": 0.25, "tol": 1e-9},
        )

        assert captured_options
        assert all("match_tol" not in options for options in captured_options)
        assert all(options["tol"] == 1e-9 for options in captured_options)

    def test_monodromy_loop_accepts_mapping_witness_points(self, monkeypatch):
        """Witness coordinates can be supplied as direct mappings."""
        try:
            from pycontinuum import PolynomialSystem, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x, y = polyvar("x", "y")
        system = PolynomialSystem([x - y])
        slicing = PolynomialSystem([x - 1])
        captured_start_points = []

        def fake_track_parameter_path(
            parameter_homotopy,
            start_point,
            start_t=0.0,
            end_t=1.0,
            options=None,
        ):
            captured_start_points.append(start_point.copy())
            return start_point, {"success": True}

        monkeypatch.setattr(
            monodromy,
            "track_parameter_path",
            fake_track_parameter_path,
        )

        permutations = monodromy.track_monodromy_loop(
            system,
            slicing,
            [{"y": 1.0 + 0j, x: 1.0 + 0j}],
            [x, y],
            num_loops=1,
            random_state=0,
        )

        assert permutations == []
        assert captured_start_points
        assert captured_start_points[0].tolist() == [1.0 + 0j, 1.0 + 0j]

    def test_monodromy_loop_accepts_vector_witness_points(self, monkeypatch):
        """Witness coordinates can be supplied as ordered vectors."""
        try:
            from pycontinuum import PolynomialSystem, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x, y = polyvar("x", "y")
        system = PolynomialSystem([x - y])
        slicing = PolynomialSystem([x - 1])
        captured_start_points = []

        def fake_track_parameter_path(
            parameter_homotopy,
            start_point,
            start_t=0.0,
            end_t=1.0,
            options=None,
        ):
            captured_start_points.append(start_point.copy())
            return start_point, {"success": True}

        monkeypatch.setattr(
            monodromy,
            "track_parameter_path",
            fake_track_parameter_path,
        )

        permutations = monodromy.track_monodromy_loop(
            system,
            slicing,
            [[1.0 + 0j, 1.0 + 0j]],
            [x, y],
            num_loops=1,
            random_state=0,
        )

        assert permutations == []
        assert captured_start_points
        assert captured_start_points[0].tolist() == [1.0 + 0j, 1.0 + 0j]

    def test_monodromy_rejects_undersliced_witness_setup(
        self,
    ):
        """Monodromy loops require enough constraints for isolated witnesses."""
        try:
            from pycontinuum import PolynomialSystem, Solution, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x, y = polyvar("x", "y")
        system = PolynomialSystem([])
        slicing = PolynomialSystem([x - 1])
        witnesses = [Solution({x: 1.0 + 0j, y: 0.0 + 0j}, residual=0.0)]

        with pytest.raises(ValueError, match="zero-dimensional witness set"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                witnesses,
                [x, y],
                num_loops=1,
                random_state=0,
            )

    def test_monodromy_random_state_controls_target_slices(self, monkeypatch):
        """Monodromy loops should be reproducible when seeded."""
        try:
            from pycontinuum import PolynomialSystem, Solution, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x, y = polyvar("x", "y")
        system = PolynomialSystem([])
        slicing = PolynomialSystem([x - 1])
        witnesses = [Solution({x: 1.0 + 0j}, residual=0.0)]
        real_generate_generic_slice = monodromy.generate_generic_slice

        def run(seed):
            generated_slices = []

            def recording_generate_generic_slice(
                dimension,
                variables,
                random_state=None,
            ):
                result = real_generate_generic_slice(
                    dimension,
                    variables,
                    random_state=random_state,
                )
                generated_slices.append(repr(result))
                return result

            def fake_track_parameter_path(
                parameter_homotopy,
                start_point,
                start_t=0.0,
                end_t=1.0,
                options=None,
            ):
                return start_point, {"success": True}

            monkeypatch.setattr(
                monodromy,
                "generate_generic_slice",
                recording_generate_generic_slice,
            )
            monkeypatch.setattr(
                monodromy,
                "track_parameter_path",
                fake_track_parameter_path,
            )
            monodromy.track_monodromy_loop(
                system,
                slicing,
                witnesses,
                [x],
                num_loops=2,
                random_state=seed,
            )
            return generated_slices

        first = run(42)
        second = run(42)
        third = run(43)

        assert first
        assert first == second
        assert first != third

    def test_monodromy_return_failure_does_not_skip_remaining_paths(self, monkeypatch):
        """A failed return leg should not skip the next tracked witness path."""
        try:
            from pycontinuum import PolynomialSystem, Solution, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([])
        slicing = PolynomialSystem([(x - 1) * (x - 2) * (x - 3)])
        witnesses = [
            Solution({x: complex(index + 1)}, residual=0.0)
            for index in range(3)
        ]
        calls = {"count": 0}
        return_attempts = []

        def fake_track_parameter_path(
            parameter_homotopy,
            start_point,
            start_t=0.0,
            end_t=1.0,
            options=None,
        ):
            calls["count"] += 1
            point_id = int(round(start_point[0].real))
            if calls["count"] <= len(witnesses):
                return start_point, {"success": True}

            return_attempts.append(point_id)
            return start_point, {"success": point_id != 1}

        monkeypatch.setattr(
            monodromy,
            "track_parameter_path",
            fake_track_parameter_path,
        )

        monodromy.track_monodromy_loop(
            system,
            slicing,
            witnesses,
            [x],
            num_loops=1,
            random_state=0,
        )

        assert return_attempts == [1, 2, 3]

    def test_monodromy_matches_affine_points_without_projective_normalization(
        self,
        monkeypatch,
    ):
        """Collinear affine witness points are distinct and must match by distance."""
        try:
            from pycontinuum import PolynomialSystem, Solution, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x, y = polyvar("x", "y")
        system = PolynomialSystem([])
        slicing = PolynomialSystem([x - y, (x - 1) * (x - 2)])
        witnesses = [
            Solution({x: 1.0 + 0j, y: 1.0 + 0j}, residual=0.0),
            Solution({x: 2.0 + 0j, y: 2.0 + 0j}, residual=0.0),
        ]
        calls = {"count": 0}

        def fake_track_parameter_path(
            parameter_homotopy,
            start_point,
            start_t=0.0,
            end_t=1.0,
            options=None,
        ):
            calls["count"] += 1
            if calls["count"] <= len(witnesses):
                return start_point, {"success": True}

            returned = start_point.copy()
            if abs(start_point[0] - 1.0) < 1e-12:
                returned[0] = 2.0 + 0j
                returned[1] = 2.0 + 0j
            else:
                returned[0] = 1.0 + 0j
                returned[1] = 1.0 + 0j
            return returned, {"success": True}

        monkeypatch.setattr(
            monodromy,
            "track_parameter_path",
            fake_track_parameter_path,
        )

        permutations = monodromy.track_monodromy_loop(
            system,
            slicing,
            witnesses,
            [x, y],
            num_loops=1,
            parameter_tracker_options={"match_tol": 1e-8},
            random_state=0,
        )

        assert len(permutations) == 1
        assert permutations[0].array_form == [1, 0]

    def test_monodromy_loop_is_quiet_by_default(self, monkeypatch, capsys):
        """Loop tracing should not print progress unless verbose=True."""
        try:
            from pycontinuum import PolynomialSystem, Solution, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([])
        slicing = PolynomialSystem([x - 1])
        witnesses = [Solution({x: 1.0 + 0j}, residual=0.0)]

        def fake_track_parameter_path(
            parameter_homotopy,
            start_point,
            start_t=0.0,
            end_t=1.0,
            options=None,
        ):
            return start_point, {"success": True}

        monkeypatch.setattr(
            monodromy,
            "track_parameter_path",
            fake_track_parameter_path,
        )

        monodromy.track_monodromy_loop(
            system,
            slicing,
            witnesses,
            [x],
            num_loops=1,
            random_state=0,
        )

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_monodromy_norm_helpers_keep_large_finite_values_finite(self):
        """Monodromy residual and affine matching norms should avoid overflow."""
        try:
            import numpy as np
            from pycontinuum import PolynomialSystem, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x, y = polyvar("x", "y")
        huge = 1e200
        point = np.array([huge + 0j, huge + 0j])
        system = PolynomialSystem([x, y])

        residual = monodromy._monodromy_system_residual(system, point, [x, y])
        factor_residual = monodromy._factor_residual_norm(x, point, [x, y])
        relative_distance = monodromy._scaled_affine_distance(
            np.array([0.0 + 0j, 0.0 + 0j]),
            point,
        )
        overflow_distance = monodromy._scaled_affine_distance(
            [10**400],
            [0.0 + 0j],
        )

        assert np.isfinite(residual)
        assert residual == pytest.approx(np.sqrt(2.0) * huge)
        assert np.isfinite(factor_residual)
        assert factor_residual == pytest.approx(huge)
        assert np.isfinite(relative_distance)
        assert relative_distance == pytest.approx(1.0)
        assert np.isinf(overflow_distance)

    def test_monodromy_loop_rejects_invalid_options(self):
        try:
            from pycontinuum import PolynomialSystem, Solution, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([])
        slicing = PolynomialSystem([x - 1])
        witnesses = [Solution({x: 1.0 + 0j}, residual=0.0)]

        with pytest.raises(ValueError, match="num_loops must be positive"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                witnesses,
                [x],
                num_loops=0,
            )
        with pytest.raises(TypeError, match="parameter_tracker_options"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                witnesses,
                [x],
                parameter_tracker_options="fast",
            )
        with pytest.raises(ValueError, match="match_tol must be positive"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                witnesses,
                [x],
                parameter_tracker_options={"match_tol": 0.0},
            )
        with pytest.raises(TypeError, match="match_tol must be a number"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                witnesses,
                [x],
                parameter_tracker_options={"match_tol": "0.25"},
            )
        with pytest.raises(TypeError, match="match_tol must be a number"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                witnesses,
                [x],
                parameter_tracker_options={"match_tol": True},
            )
        with pytest.raises(ValueError, match="Unknown monodromy tracker option"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                witnesses,
                [x],
                parameter_tracker_options={"stepz": 0.1},
            )
        with pytest.raises(TypeError, match="verbose must be a boolean"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                witnesses,
                [x],
                verbose="yes",
            )

    def test_monodromy_loop_rejects_invalid_variables_and_witnesses(self):
        try:
            from pycontinuum import PolynomialSystem, Solution, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x, y = polyvar("x", "y")
        system = PolynomialSystem([])
        slicing = PolynomialSystem([x - 1])
        witnesses = [Solution({x: 1.0 + 0j}, residual=0.0)]

        with pytest.raises(TypeError, match=r"variables\[1\] must be a Variable"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                witnesses,
                [x, "extra"],
                num_loops=1,
            )
        with pytest.raises(TypeError, match="start_witness_points"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                [object()],
                [x],
                num_loops=1,
            )
        with pytest.raises(ValueError, match="missing variable"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                [Solution({}, residual=0.0)],
                [x],
                num_loops=1,
            )
        with pytest.raises(ValueError, match="missing variable.*y"):
            monodromy.track_monodromy_loop(
                PolynomialSystem([x - y]),
                slicing,
                [{x: 1.0 + 0j}],
                [x, y],
                num_loops=1,
            )
        with pytest.raises(ValueError, match="outside the monodromy variable list"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                [Solution({x: 1.0 + 0j, y: 2.0 + 0j}, residual=0.0)],
                [x],
                num_loops=1,
            )
        with pytest.raises(ValueError, match=r"start_witness_points\[0\].*numeric"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                [{x: 10**400}],
                [x],
                num_loops=1,
            )

    def test_monodromy_loop_validates_start_witness_residuals(self):
        try:
            from pycontinuum import PolynomialSystem, Solution, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x, y = polyvar("x", "y")
        system = PolynomialSystem([x - y])
        slicing = PolynomialSystem([x - 1])

        with pytest.raises(ValueError, match="original_system"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                [Solution({x: 1.0 + 0j, y: 2.0 + 0j}, residual=0.0)],
                [x, y],
                num_loops=1,
            )
        with pytest.raises(ValueError, match="start_slice"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                [Solution({x: 2.0 + 0j, y: 2.0 + 0j}, residual=0.0)],
                [x, y],
                num_loops=1,
            )
        with pytest.raises(TypeError, match="tol must be a number"):
            monodromy.track_monodromy_loop(
                system,
                slicing,
                [Solution({x: 1.0 + 0j, y: 1.0 + 0j}, residual=0.0)],
                [x, y],
                num_loops=1,
                parameter_tracker_options={"tol": "1e-8"},
            )

    def test_numerical_irreducible_decomposition_is_quiet_for_empty_witnesses(
        self,
        capsys,
    ):
        try:
            from pycontinuum import PolynomialSystem, SolutionSet, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([x])
        slicing = PolynomialSystem([])
        witnesses = SolutionSet([], system)

        components = monodromy.numerical_irreducible_decomposition(
            system,
            slicing,
            witnesses,
            [x],
        )

        captured = capsys.readouterr()
        assert components == []
        assert captured.out == ""

    def test_numerical_irreducible_decomposition_validates_options(self):
        try:
            from pycontinuum import PolynomialSystem, SolutionSet, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([x])
        slicing = PolynomialSystem([])
        witnesses = SolutionSet([], system)

        with pytest.raises(TypeError, match="monodromy_options must be a dictionary"):
            monodromy.numerical_irreducible_decomposition(
                system,
                slicing,
                witnesses,
                [x],
                monodromy_options="fast",
            )
        with pytest.raises(ValueError, match="Unknown monodromy option"):
            monodromy.numerical_irreducible_decomposition(
                system,
                slicing,
                witnesses,
                [x],
                monodromy_options={"loops": 2},
            )
        with pytest.raises(TypeError, match="match_tol must be a number"):
            monodromy.numerical_irreducible_decomposition(
                system,
                slicing,
                witnesses,
                [x],
                monodromy_options={"tracker_options": {"match_tol": "0.25"}},
            )
        with pytest.raises(ValueError, match="Unknown monodromy tracker option"):
            monodromy.numerical_irreducible_decomposition(
                system,
                slicing,
                witnesses,
                [x],
                monodromy_options={"tracker_options": {"stepz": 0.1}},
            )
        with pytest.raises(TypeError, match="witness_superset must be"):
            monodromy.numerical_irreducible_decomposition(
                system,
                slicing,
                object(),
                [x],
            )

    def test_numerical_irreducible_decomposition_accepts_coordinate_records(
        self,
        monkeypatch,
    ):
        try:
            from pycontinuum import PolynomialSystem, Solution, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([])
        slicing = PolynomialSystem([x - 1])
        captured = {}

        def fake_track_monodromy_loop(
            original_system,
            start_slice,
            start_witness_points,
            variables,
            num_loops=3,
            parameter_tracker_options=None,
            random_state=None,
            verbose=False,
        ):
            captured["start_witness_points"] = list(start_witness_points)
            return []

        monkeypatch.setattr(
            monodromy,
            "track_monodromy_loop",
            fake_track_monodromy_loop,
        )

        components = monodromy.numerical_irreducible_decomposition(
            system,
            slicing,
            [[1.0 + 0j], {"x": 1.0 + 0j}],
            [x],
            monodromy_options={"num_loops": 1},
        )

        assert len(components) == 1
        assert components[0].degree == 2
        assert all(
            isinstance(point, Solution)
            for point in captured["start_witness_points"]
        )
        assert all(
            isinstance(point, Solution)
            for point in components[0].witness_points
        )

    def test_numerical_irreducible_decomposition_passes_validated_loop_options(
        self,
        monkeypatch,
    ):
        try:
            from pycontinuum import PolynomialSystem, Solution, SolutionSet, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([])
        slicing = PolynomialSystem([x - 1])
        witnesses = SolutionSet([Solution({x: 1.0 + 0j}, residual=0.0)], system)
        captured = {}

        def fake_track_monodromy_loop(
            original_system,
            start_slice,
            start_witness_points,
            variables,
            num_loops=3,
            parameter_tracker_options=None,
            random_state=None,
            verbose=False,
        ):
            captured["num_loops"] = num_loops
            captured["parameter_tracker_options"] = dict(
                parameter_tracker_options or {}
            )
            captured["verbose"] = verbose
            return []

        monkeypatch.setattr(
            monodromy,
            "track_monodromy_loop",
            fake_track_monodromy_loop,
        )

        components = monodromy.numerical_irreducible_decomposition(
            system,
            slicing,
            witnesses,
            [x],
            monodromy_options={
                "num_loops": 2,
                "tracker_options": {"match_tol": 0.25, "tol": 1e-9},
                "verbose": True,
            },
        )

        assert len(components) == 1
        assert captured["num_loops"] == 2
        assert captured["parameter_tracker_options"]["match_tol"] == 0.25
        assert captured["parameter_tracker_options"]["tol"] == 1e-9
        assert captured["verbose"] is True

    def test_numerical_irreducible_decomposition_uses_scaled_rank_fallback(
        self,
        monkeypatch,
        capsys,
    ):
        try:
            from pycontinuum import PolynomialSystem, Solution, SolutionSet, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x, y = polyvar("x", "y")
        system = PolynomialSystem([(10**400) * (x - 1)])
        slicing = PolynomialSystem([y])
        witnesses = SolutionSet(
            [Solution({x: 1.0 + 0j, y: 0.0 + 0j}, residual=0.0)],
            system,
        )

        monkeypatch.setattr(
            monodromy,
            "track_monodromy_loop",
            lambda *args, **kwargs: [],
        )

        components = monodromy.numerical_irreducible_decomposition(
            system,
            slicing,
            witnesses,
            [x, y],
            monodromy_options={"verbose": True},
        )

        captured = capsys.readouterr()
        assert len(components) == 1
        assert "consistent Jacobian rank 1" in captured.out
        assert "consistent Jacobian rank 0" not in captured.out

    def test_compute_numerical_decomposition_is_quiet_by_default(
        self,
        monkeypatch,
        capsys,
    ):
        try:
            from pycontinuum import PolynomialSystem, SolutionSet, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([x])

        def fake_compute_witness_superset(
            system,
            variables,
            dimension,
            options=None,
            random_state=None,
        ):
            return PolynomialSystem([]), SolutionSet([], system)

        monkeypatch.setattr(
            monodromy,
            "compute_witness_superset",
            fake_compute_witness_superset,
        )

        decomposition = monodromy.compute_numerical_decomposition(
            system,
            variables=[x],
            max_dimension=0,
        )

        captured = capsys.readouterr()
        assert decomposition == {}
        assert captured.out == ""

    def test_compute_numerical_decomposition_reraises_unexpected_dimension_errors(
        self,
        monkeypatch,
    ):
        try:
            from pycontinuum import PolynomialSystem, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([x])

        def failing_compute_witness_superset(*args, **kwargs):
            raise TypeError("bad internal call")

        monkeypatch.setattr(
            monodromy,
            "compute_witness_superset",
            failing_compute_witness_superset,
        )

        with pytest.raises(RuntimeError, match="Unexpected failure.*dimension 0"):
            monodromy.compute_numerical_decomposition(
                system,
                variables=[x],
                max_dimension=0,
            )

    def test_compute_numerical_decomposition_can_continue_after_unexpected_errors(
        self,
        monkeypatch,
        capsys,
    ):
        try:
            from pycontinuum import PolynomialSystem, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([x])

        def failing_compute_witness_superset(*args, **kwargs):
            raise TypeError("bad internal call")

        monkeypatch.setattr(
            monodromy,
            "compute_witness_superset",
            failing_compute_witness_superset,
        )

        decomposition = monodromy.compute_numerical_decomposition(
            system,
            variables=[x],
            max_dimension=0,
            monodromy_options={"continue_on_error": True, "verbose": True},
        )

        captured = capsys.readouterr()
        assert decomposition == {}
        assert "Error computing dimension 0: bad internal call" in captured.out

    def test_compute_numerical_decomposition_does_not_swallow_invalid_solver_options(
        self,
    ):
        try:
            from pycontinuum import PolynomialSystem, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([x])

        with pytest.raises(ValueError, match="tol must be positive and finite"):
            monodromy.compute_numerical_decomposition(
                system,
                variables=[x],
                max_dimension=0,
                solver_options={"tol": 0.0},
            )

    def test_compute_numerical_decomposition_skips_expected_dimension_errors(
        self,
        monkeypatch,
        capsys,
    ):
        try:
            from pycontinuum import PolynomialSystem, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([x])

        def no_witness_for_dimension(*args, **kwargs):
            raise ValueError(
                "The requested dimension leaves an underdetermined augmented "
                "system with 0 equations in 1 variables"
            )

        monkeypatch.setattr(
            monodromy,
            "compute_witness_superset",
            no_witness_for_dimension,
        )

        decomposition = monodromy.compute_numerical_decomposition(
            system,
            variables=[x],
            max_dimension=0,
            monodromy_options={"verbose": True},
        )

        captured = capsys.readouterr()
        assert decomposition == {}
        assert "Skipping dimension 0" in captured.out

    def test_compute_numerical_decomposition_scans_redundant_overdetermined_dimensions(
        self,
    ):
        try:
            from pycontinuum import PolynomialSystem, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x, y = polyvar("x", "y")
        system = PolynomialSystem([x, 2 * x])

        decomposition = monodromy.compute_numerical_decomposition(
            system,
            variables=[x, y],
            solver_options={"random_state": 123},
            monodromy_options={"num_loops": 1, "random_state": 123},
        )

        assert set(decomposition) == {1}
        assert len(decomposition[1]) == 1
        component = decomposition[1][0]
        assert component.dimension == 1
        assert component.degree == 1
        assert abs(component.witness_points[0].values[x]) < 1e-8

    def test_compute_numerical_decomposition_splits_reducible_hypersurface(
        self,
    ):
        try:
            from pycontinuum import PolynomialSystem, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x, y = polyvar("x", "y")
        system = PolynomialSystem([x * y])

        decomposition = monodromy.compute_numerical_decomposition(
            system,
            variables=[x, y],
            solver_options={"random_state": 0},
            monodromy_options={"num_loops": 1, "random_state": 0},
        )

        assert set(decomposition) == {1}
        components = decomposition[1]
        assert len(components) == 2
        assert sorted(component.degree for component in components) == [1, 1]
        axis_signatures = sorted(
            (
                abs(component.witness_points[0].values[x]) < 1e-8,
                abs(component.witness_points[0].values[y]) < 1e-8,
            )
            for component in components
        )
        assert axis_signatures == [(False, True), (True, False)]

    def test_compute_numerical_decomposition_random_state_controls_witness_slices(
        self,
        monkeypatch,
    ):
        try:
            from pycontinuum import PolynomialSystem, WitnessSet, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x, y = polyvar("x", "y")
        system = PolynomialSystem([x - y])
        captured_slices = []

        def fake_numerical_irreducible_decomposition(
            original_system,
            slicing_system,
            witness_superset,
            variables,
            monodromy_options=None,
        ):
            captured_slices.append(repr(slicing_system))
            return [
                WitnessSet(
                    original_system,
                    slicing_system,
                    witness_superset.solutions,
                    len(slicing_system.equations),
                )
            ]

        monkeypatch.setattr(
            monodromy,
            "numerical_irreducible_decomposition",
            fake_numerical_irreducible_decomposition,
        )

        def run(seed):
            captured_slices.clear()
            monodromy.compute_numerical_decomposition(
                system,
                variables=[x, y],
                max_dimension=1,
                monodromy_options={"random_state": seed},
            )
            return tuple(captured_slices)

        first = run(123)
        second = run(123)
        third = run(124)

        assert first
        assert first == second
        assert first != third

    def test_compute_numerical_decomposition_validates_options(self):
        try:
            from pycontinuum import PolynomialSystem, polyvar
            import pycontinuum.monodromy as monodromy
        except ImportError:
            pytest.skip("Monodromy functionality requires optional sympy dependency")

        x = polyvar("x")
        system = PolynomialSystem([x])

        with pytest.raises(TypeError, match="solver_options must be a dictionary"):
            monodromy.compute_numerical_decomposition(
                system,
                variables=[x],
                solver_options="fast",
            )
        with pytest.raises(ValueError, match="max_dimension must be non-negative"):
            monodromy.compute_numerical_decomposition(
                system,
                variables=[x],
                max_dimension=-1,
            )
        with pytest.raises(ValueError, match="Unknown monodromy option"):
            monodromy.compute_numerical_decomposition(
                system,
                variables=[x],
                monodromy_options={"loops": 2},
            )
        with pytest.raises(TypeError, match="continue_on_error must be a boolean"):
            monodromy.compute_numerical_decomposition(
                system,
                variables=[x],
                monodromy_options={"continue_on_error": "yes"},
            )
