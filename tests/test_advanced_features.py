# test_advanced_features.py
from pycontinuum import (
    ParameterHomotopy,
    PolynomialSystem,
    Solution,
    polyvar,
    solve,
    track_parameter_path,
)
from pycontinuum.endgame import (
    CauchyEndgame,
    Endgamer,
    EndgamerOptions,
    run_cauchy_endgame,
)
import pycontinuum.endgame as endgame_module
import pycontinuum.parameter_homotopy as parameter_homotopy_module
from pycontinuum.parameter_homotopy import _correct_parameter_prediction
import numpy as np
import pytest


def _parameter_dummy(index=0):
    return polyvar(f"_pc_parameter_dummy_{index}")


def test_witness_sets():
    """Test computation of witness sets for positive-dimensional components."""
    x, y, z = polyvar('x', 'y', 'z')
    # A line in 3D: x = y, z = 0
    system = PolynomialSystem([x - y, z])
    # Placeholder for future positive-dimensional tests

def test_monodromy():
    """Test monodromy loops for numerical irreducible decomposition."""
    # Placeholder: heavy functionality, skip for unit run

def test_parameter_homotopy_simple_linear():
    """Parameter homotopy should track a known linear solution path."""
    x = polyvar('x')
    # Fixed F(x) = 0 (none for this simple test)
    F = PolynomialSystem([])
    # L1(x) = x - 1, L2(x) = x + 1. Solution moves from x=1 to x=-1 as t goes 0->1
    L1 = PolynomialSystem([x - 1])
    L2 = PolynomialSystem([x + 1])
    ph = ParameterHomotopy(F, L1, L2, [x], square_fix=True)

    start = np.array([1.0 + 0j])
    end, info = track_parameter_path(ph, start_point=start, start_t=0.0, end_t=1.0, options={"tol": 1e-10})

    assert info.get('steps', 0) > 0
    assert np.allclose(end, np.array([-1.0 + 0j]), atol=1e-6)


def test_parameter_homotopy_accepts_solution_start_point():
    x = polyvar("x")
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x + 1]),
        [x],
        square_fix=True,
    )

    end, info = track_parameter_path(
        ph,
        start_point=Solution({x: 1.0 + 0j}, residual=0.0),
        start_t=0.0,
        end_t=1.0,
        options={"tol": 1e-10},
    )

    assert info["success"]
    assert info["start_residual"] == 0.0
    np.testing.assert_allclose(end, np.array([-1.0 + 0j]), atol=1e-8)


def test_parameter_homotopy_accepts_mapping_start_point_and_evaluator_points():
    x, y = polyvar("x", "y")
    ph = ParameterHomotopy(
        PolynomialSystem([x - y]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x - 2]),
        [x, y],
    )
    start = {"y": 1.0 + 0j, x: 1.0 + 0j}

    np.testing.assert_allclose(ph.evaluate(start, 0.0), np.array([0.0, 0.0]))
    np.testing.assert_allclose(
        ph.jacobian_x(start, 0.0),
        np.array([[1.0, -1.0], [1.0, 0.0]], dtype=complex),
    )
    np.testing.assert_allclose(ph.deriv_t(start, 0.0), np.array([0.0, -0.5]))

    with pytest.raises(ValueError, match="conflicting coordinates.*x"):
        ph.evaluate({x: 1.0 + 0j, "x": 2.0 + 0j, y: 1.0 + 0j}, 0.0)

    end, info = track_parameter_path(
        ph,
        start_point=start,
        start_t=0.0,
        end_t=1.0,
        options={"tol": 1e-8, "max_steps": 2000},
    )

    assert info["success"]
    assert info["start_residual"] == 0.0
    np.testing.assert_allclose(end, np.array([2.0 + 0j, 2.0 + 0j]), atol=1e-8)


def test_parameter_homotopy_system_at_materializes_intermediate_system():
    x, y = polyvar("x", "y")
    ph = ParameterHomotopy(
        PolynomialSystem([x - y]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x - 3]),
        [x, y],
    )

    system = ph.system_at(0.25)
    values = {x: 1.5 + 0j, y: 1.5 + 0j}

    assert len(system.equations) == 2
    np.testing.assert_allclose(
        np.array(system.evaluate(values), dtype=complex),
        ph.evaluate(values, 0.25),
    )


def test_parameter_homotopy_system_at_includes_square_fix_dummy_equations():
    x = polyvar("x")
    dummy = _parameter_dummy()
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x - 3]),
        [x, dummy],
        square_fix=True,
    )

    system = ph.system_at(0.5)
    values = {x: 2.0 + 0j, dummy: 0.25 + 0j}

    assert len(system.equations) == 2
    np.testing.assert_allclose(
        np.array(system.evaluate(values), dtype=complex),
        ph.evaluate(values, 0.5),
    )


def test_parameter_homotopy_system_at_rejects_nonfinite_t():
    x = polyvar("x")
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x - 3]),
        [x],
    )

    with pytest.raises(ValueError, match="t must be finite"):
        ph.system_at(float("nan"))


def test_parameter_homotopy_path_info_records_effective_tracking_options():
    x = polyvar("x")
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x + 1]),
        [x],
    )

    point, info = track_parameter_path(
        ph,
        start_point={"x": 0.5 + 0j},
        start_t=0.25,
        end_t=0.75,
        options={
            "tol": 1e-10,
            "min_step_size": 0.01,
            "max_step_size": 0.1,
            "max_corrections": 4,
            "max_steps": 50,
            "max_predictor_norm": 0.5,
            "predictor": "heun",
            "store_paths": True,
        },
    )

    assert info["success"]
    assert info["start_t"] == pytest.approx(0.25)
    assert info["end_t"] == pytest.approx(0.75)
    assert info["direction"] == 1
    assert info["tol"] == pytest.approx(1e-10)
    assert info["min_step_size"] == pytest.approx(0.01)
    assert info["max_step_size"] == pytest.approx(0.1)
    assert info["initial_step_size"] == pytest.approx(0.05)
    assert info["max_corrections"] == 4
    assert info["max_steps"] == 50
    assert info["max_predictor_norm"] == pytest.approx(0.5)
    assert info["predictor"] == "heun"
    assert info["store_paths"] is True
    assert info["normalize_tangent"] is False
    assert info["path_points"]
    np.testing.assert_allclose(point, np.array([-0.5 + 0j]), atol=1e-8)


def test_parameter_homotopy_zero_length_path_is_successful_noop():
    x = polyvar("x")
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x + 1]),
        [x],
    )

    point, info = track_parameter_path(
        ph,
        start_point={"x": 0.5 + 0j},
        start_t=0.25,
        end_t=0.25,
        options={"tol": 1e-12, "max_step_size": 0.1, "store_paths": True},
    )

    assert info["success"]
    assert info["steps"] == 0
    assert info["newton_iters"] == 0
    assert info["direction"] == 0
    assert info["initial_step_size"] == 0.0
    assert info["failure_reason"] is None
    assert info["t"] == pytest.approx(0.25)
    assert info["final_residual"] == pytest.approx(0.0)
    assert len(info["path_points"]) == 1
    assert info["path_points"][0][0] == pytest.approx(0.25)
    np.testing.assert_allclose(point, np.array([0.5 + 0j]))
    np.testing.assert_allclose(info["path_points"][0][1], point)


def test_parameter_homotopy_rejects_mapping_start_point_missing_coordinate():
    x, y = polyvar("x", "y")
    ph = ParameterHomotopy(
        PolynomialSystem([x - y]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x - 2]),
        [x, y],
    )

    with pytest.raises(
        ValueError,
        match=r"start_point is missing coordinate\(s\): y",
    ):
        track_parameter_path(
            ph,
            start_point={x: 1.0 + 0j},
            start_t=0.0,
            end_t=1.0,
            options={"tol": 1e-8},
        )


def test_parameter_homotopy_uses_scaled_fallback_for_extreme_fixed_equations():
    x, y = polyvar("x", "y")
    huge = 10**400
    ph = ParameterHomotopy(
        PolynomialSystem([huge * (x - y)]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x - 2]),
        [x, y],
    )
    start = np.array([1.0 + 0j, 1.0 + 0j])

    np.testing.assert_allclose(ph.evaluate(start, 0.0), np.array([0.0, 0.0]))
    np.testing.assert_allclose(
        ph.jacobian_x(start, 0.0),
        np.array([[1.0, -1.0], [1.0, 0.0]], dtype=complex),
    )

    end, info = track_parameter_path(
        ph,
        start_point=start,
        start_t=0.0,
        end_t=1.0,
        options={"tol": 1e-8, "max_steps": 2000},
    )

    assert info["success"]
    np.testing.assert_allclose(end, np.array([2.0 + 0j, 2.0 + 0j]), atol=1e-8)


@pytest.mark.parametrize("scale", [1e12, 1e-12])
def test_parameter_homotopy_is_invariant_to_finite_parameter_row_scaling(scale):
    x = polyvar("x")
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([scale * (x - 2)]),
        [x],
        square_fix=True,
    )

    assert ph.evaluate(np.array([1.0 + 0j]), 1.0)[0] == pytest.approx(
        -0.5 + 0j
    )
    assert ph.jacobian_x(np.array([2.0 + 0j]), 1.0)[0, 0] == pytest.approx(
        0.5 + 0j
    )

    end, info = track_parameter_path(
        ph,
        start_point=np.array([1.0 + 0j]),
        start_t=0.0,
        end_t=1.0,
        options={"tol": 1e-8, "max_steps": 2000},
    )

    assert info["success"]
    assert info["failure_reason"] is None
    assert info["final_residual"] < 1e-8
    np.testing.assert_allclose(end, np.array([2.0 + 0j]), atol=1e-8)


def test_parameter_homotopy_rejects_tiny_fixed_row_start_violation():
    x, y = polyvar("x", "y")
    ph = ParameterHomotopy(
        PolynomialSystem([1e-12 * (x - y)]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x - 2]),
        [x, y],
        square_fix=True,
    )

    with pytest.raises(ValueError, match="does not satisfy"):
        track_parameter_path(
            ph,
            start_point=np.array([1.0 + 0j, 2.0 + 0j]),
            start_t=0.0,
            end_t=1.0,
            options={"tol": 1e-8},
        )


def test_parameter_homotopy_uses_true_tangent_by_default():
    """The parameter predictor should use dx/dt, not only the unit direction."""
    x = polyvar('x')
    F = PolynomialSystem([])
    L1 = PolynomialSystem([x - 1])
    L2 = PolynomialSystem([x + 3])
    ph = ParameterHomotopy(F, L1, L2, [x], square_fix=True)

    end, info = track_parameter_path(
        ph,
        start_point=np.array([1.0 + 0j]),
        start_t=0.0,
        end_t=1.0,
        options={"tol": 1e-12, "max_step_size": 0.5},
    )

    assert info["success"]
    assert info["start_residual"] == 0.0
    assert info["start_residual_limit"] >= 1e-10
    assert info["max_observed_predictor_norm"] > 0.9
    assert np.allclose(end, np.array([-3.0 + 0j]), atol=1e-8)


def test_parameter_homotopy_predictor_cap_reduces_steps():
    x = polyvar('x')
    F = PolynomialSystem([])
    L1 = PolynomialSystem([x - 1])
    L2 = PolynomialSystem([x + 3])
    ph = ParameterHomotopy(F, L1, L2, [x], square_fix=True)

    end, info = track_parameter_path(
        ph,
        start_point=np.array([1.0 + 0j]),
        start_t=0.0,
        end_t=1.0,
        options={"tol": 1e-12, "max_step_size": 0.5, "max_predictor_norm": 0.2},
    )

    assert info["success"]
    assert info["step_reductions"] > 0
    assert np.allclose(end, np.array([-3.0 + 0j]), atol=1e-8)


def test_parameter_homotopy_corrector_damps_residual_increasing_step():
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 0.1]),
        PolynomialSystem([x**3 - 1]),
        [x],
        square_fix=True,
    )
    predicted = np.array([0.1 + 0j])
    initial_residual = abs(ph.evaluate(predicted, 1.0)[0])

    corrected, converged, steps, metadata = _correct_parameter_prediction(
        ph,
        predicted,
        t_target=1.0,
        tol=1e-12,
        max_corrections=1,
    )

    assert not converged
    assert steps == 1
    assert metadata["backtracks"] > 0
    assert metadata["min_step_scale"] < 1.0
    assert metadata["final_residual"] < initial_residual
    assert abs(corrected[0]) < 2.0


def test_parameter_homotopy_heun_predictor_records_metadata():
    x = polyvar('x')
    F = PolynomialSystem([])
    L1 = PolynomialSystem([x - 1])
    L2 = PolynomialSystem([x**2 - 4])
    ph = ParameterHomotopy(F, L1, L2, [x], square_fix=True)

    end, info = track_parameter_path(
        ph,
        start_point=np.array([1.0 + 0j]),
        start_t=0.0,
        end_t=1.0,
        options={"tol": 1e-12, "max_step_size": 0.2, "predictor": "heun"},
    )

    assert info["success"]
    assert info["predictor"] == "heun"
    assert info["predictor_fallbacks"] == 0
    assert info["max_predictor_correction_norm"] > 0
    assert np.allclose(end, np.array([2.0 + 0j]), atol=1e-8)


def test_parameter_homotopy_rk4_predictor_records_metadata():
    x = polyvar('x')
    F = PolynomialSystem([])
    L1 = PolynomialSystem([x - 1])
    L2 = PolynomialSystem([x**2 - 4])
    ph = ParameterHomotopy(F, L1, L2, [x], square_fix=True)

    end, info = track_parameter_path(
        ph,
        start_point=np.array([1.0 + 0j]),
        start_t=0.0,
        end_t=1.0,
        options={"tol": 1e-12, "max_step_size": 0.2, "predictor": "rk4"},
    )

    assert info["success"]
    assert info["predictor"] == "rk4"
    assert info["predictor_fallbacks"] == 0
    assert info["max_predictor_correction_norm"] > 0
    assert np.allclose(end, np.array([2.0 + 0j]), atol=1e-8)


def test_parameter_homotopy_rejects_unknown_predictor():
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x + 1]),
        [x],
    )

    try:
        track_parameter_path(
            ph,
            start_point=np.array([1.0 + 0j]),
            options={"predictor": "bogus"},
        )
    except ValueError as exc:
        assert "predictor must be" in str(exc)
    else:
        raise AssertionError("Expected invalid predictor to be rejected")


def test_parameter_homotopy_rejects_bad_start_point_dimension():
    x, y = polyvar('x', 'y')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1, y - 1]),
        PolynomialSystem([x + 1, y + 1]),
        [x, y],
    )

    try:
        track_parameter_path(ph, start_point=np.array([1.0 + 0j]))
    except ValueError as exc:
        assert "start_point" in str(exc)
    else:
        raise AssertionError("Expected bad start point dimension to be rejected")


def test_parameter_homotopy_rejects_start_point_off_start_system():
    x = polyvar('x')
    dummy = _parameter_dummy()
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x - 2]),
        [x, dummy],
        square_fix=True,
    )

    with pytest.raises(ValueError, match="start_point does not satisfy"):
        track_parameter_path(
            ph,
            start_point=np.array([1.0 + 0j, 1.0 + 0j]),
            options={"tol": 1e-10},
        )

    end, info = track_parameter_path(
        ph,
        start_point=np.array([1.0 + 0j, 0.0 + 0j]),
        options={"tol": 1e-10},
    )
    assert info["success"]
    assert info["start_residual"] == 0.0
    np.testing.assert_allclose(end, np.array([2.0 + 0j, 0.0 + 0j]), atol=1e-8)


def test_parameter_homotopy_start_residual_limit_honors_tolerance():
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x]),
        PolynomialSystem([x - 1]),
        [x],
    )

    with pytest.raises(ValueError, match="start_point does not satisfy"):
        track_parameter_path(
            ph,
            start_point=np.array([1e-9 + 0j]),
            options={"tol": 1e-12},
        )

    end, info = track_parameter_path(
        ph,
        start_point=np.array([1e-9 + 0j]),
        options={"tol": 1e-8},
    )

    assert info["success"]
    assert info["start_residual"] == pytest.approx(1e-9)
    assert info["start_residual_limit"] == pytest.approx(1e-6)
    np.testing.assert_allclose(end, np.array([1.0 + 0j]), atol=1e-8)


def test_parameter_homotopy_rejects_invalid_tracking_options():
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x + 1]),
        [x],
    )

    try:
        track_parameter_path(
            ph,
            start_point=np.array([1.0 + 0j]),
            options={"max_steps": 0},
        )
    except ValueError as exc:
        assert "max_steps" in str(exc)
    else:
        raise AssertionError("Expected invalid max_steps to be rejected")


def test_parameter_homotopy_rejects_unknown_tracking_options():
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x + 1]),
        [x],
    )

    try:
        track_parameter_path(
            ph,
            start_point=np.array([1.0 + 0j]),
            options={"max_stepz": 0.1},
        )
    except ValueError as exc:
        assert "Unknown parameter tracking option" in str(exc)
    else:
        raise AssertionError("Expected unknown parameter option to be rejected")


def test_parameter_homotopy_rejects_malformed_variable_list():
    x, y = polyvar('x', 'y')

    with pytest.raises(TypeError, match="variables must be an iterable"):
        ParameterHomotopy(
            PolynomialSystem([]),
            PolynomialSystem([x - 1]),
            PolynomialSystem([x + 1]),
            x,
        )
    with pytest.raises(TypeError, match=r"variables\[1\] must be a Variable"):
        ParameterHomotopy(
            PolynomialSystem([]),
            PolynomialSystem([x - 1]),
            PolynomialSystem([x + 1]),
            [x, "extra"],
        )
    with pytest.raises(ValueError, match="duplicate variable"):
        ParameterHomotopy(
            PolynomialSystem([]),
            PolynomialSystem([x - 1]),
            PolynomialSystem([x + 1]),
            [x, x],
        )
    with pytest.raises(ValueError, match="explicit parameter dummy.*y"):
        ParameterHomotopy(
            PolynomialSystem([]),
            PolynomialSystem([x - 1]),
            PolynomialSystem([x + 1]),
            [x, y],
        )


def test_parameter_homotopy_square_fix_requires_unused_explicit_dummy_variable():
    x, y = polyvar('x', 'y')
    dummy = _parameter_dummy()

    with pytest.raises(ValueError, match="final 1 variable"):
        ParameterHomotopy(
            PolynomialSystem([]),
            PolynomialSystem([x + y - 1]),
            PolynomialSystem([x + y - 2]),
            [x, y],
            square_fix=True,
        )

    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x - 2]),
        [x, dummy],
        square_fix=True,
    )
    assert ph.is_square
    assert ph.dummy_count == 1
    np.testing.assert_allclose(
        ph.evaluate(np.array([1.0 + 0j, 3.0 + 0j]), 0.0),
        np.array([0.0 + 0j, 3.0 + 0j]),
    )


def test_parameter_homotopy_rejects_unused_variables_without_square_fix():
    x, y = polyvar('x', 'y')

    with pytest.raises(ValueError, match="not used by parameter homotopy.*y"):
        ParameterHomotopy(
            PolynomialSystem([]),
            PolynomialSystem([x - 1]),
            PolynomialSystem([x - 2]),
            [x, y],
            square_fix=False,
        )


def test_parameter_homotopy_public_evaluators_validate_point_and_t():
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x + 1]),
        [x],
    )

    assert ph.evaluate([1.0 + 0j], 0.0)[0] == 0.0
    np.testing.assert_allclose(ph.deriv_t([1.0 + 0j], 0.0), np.array([2.0]))

    with pytest.raises(ValueError, match="point must have 1 coordinate"):
        ph.evaluate(np.array([1.0 + 0j, 2.0 + 0j]), 0.0)
    with pytest.raises(ValueError, match="point must have 1 coordinate"):
        ph.deriv_t(np.array([1.0 + 0j, 2.0 + 0j]), 0.0)
    with pytest.raises(ValueError, match="point contains nonfinite values"):
        ph.jacobian_x(np.array([np.nan + 0j]), 0.0)
    with pytest.raises(ValueError, match="t must be finite"):
        ph.evaluate(np.array([1.0 + 0j]), float("nan"))
    with pytest.raises(TypeError, match="t must be a number"):
        ph.jacobian_x(np.array([1.0 + 0j]), "0.0")


@pytest.mark.parametrize("option", ["square_fix", "verbose"])
def test_parameter_homotopy_rejects_nonboolean_constructor_flags(option):
    x = polyvar('x')
    kwargs = {option: "yes"}

    with pytest.raises(TypeError, match=f"{option} must be a boolean"):
        ParameterHomotopy(
            PolynomialSystem([]),
            PolynomialSystem([x - 1]),
            PolynomialSystem([x + 1]),
            [x],
            **kwargs,
        )


@pytest.mark.parametrize(
    "option",
    ["normalize_tangent", "verbose", "store_paths"],
)
def test_parameter_homotopy_rejects_nonboolean_tracking_flags(option):
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x + 1]),
        [x],
    )

    with pytest.raises(TypeError, match=f"{option} must be a boolean"):
        track_parameter_path(
            ph,
            start_point=np.array([1.0 + 0j]),
            options={option: "yes"},
        )


@pytest.mark.parametrize(
    ("options", "error_type", "message"),
    [
        ({"tol": "tight"}, TypeError, "tol must be a number"),
        ({"tol": "1e-8"}, TypeError, "tol must be a number"),
        ({"min_step_size": 0.0}, ValueError, "min_step_size must be positive"),
        ({"max_step_size": "0.1"}, TypeError, "max_step_size must be a number"),
        ({"max_corrections": 1.5}, TypeError, "max_corrections must be an integer"),
        ({"max_predictor_norm": "0.2"}, TypeError, "max_predictor_norm must be a number"),
        ({"max_predictor_norm": float("nan")}, ValueError, "max_predictor_norm"),
    ],
)
def test_parameter_homotopy_rejects_invalid_numeric_options(
    options, error_type, message
):
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x + 1]),
        [x],
    )

    with pytest.raises(error_type, match=message):
        track_parameter_path(
            ph,
            start_point=np.array([1.0 + 0j]),
            options=options,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start_t": "0.0"}, "start_t must be a number"),
        ({"end_t": "1.0"}, "end_t must be a number"),
    ],
)
def test_parameter_homotopy_rejects_nonnumeric_path_bounds(kwargs, message):
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x + 1]),
        [x],
    )

    with pytest.raises(TypeError, match=message):
        track_parameter_path(
            ph,
            start_point=np.array([1.0 + 0j]),
            **kwargs,
        )


def test_parameter_homotopy_max_steps_sets_failure_reason(capsys):
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x + 3]),
        [x],
    )

    end, info = track_parameter_path(
        ph,
        start_point=np.array([1.0 + 0j]),
        start_t=0.0,
        end_t=1.0,
        options={"max_steps": 1, "max_step_size": 0.05},
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert not info["success"]
    assert info["failure_reason"] == "max_steps_exceeded"
    assert info["start_t"] == pytest.approx(0.0)
    assert info["end_t"] == pytest.approx(1.0)
    assert info["direction"] == 1
    assert info["max_steps"] == 1
    assert info["max_step_size"] == pytest.approx(0.05)
    assert info["initial_step_size"] == pytest.approx(0.025)
    assert np.isfinite(info["final_residual"])
    assert np.all(np.isfinite(end))


def test_parameter_homotopy_newton_failure_reports_final_and_trial_state():
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x**2]),
        PolynomialSystem([x**2 + 1]),
        [x],
    )

    point, info = track_parameter_path(
        ph,
        start_point=np.array([0.0 + 0j]),
        options={
            "tol": 1e-10,
            "min_step_size": 0.05,
            "max_step_size": 0.05,
            "max_corrections": 1,
        },
    )

    assert not info["success"]
    assert info["failure_reason"] == "newton_failed"
    assert info["t"] == 0.0
    np.testing.assert_allclose(info["final_point"], point)
    assert info["final_residual"] == 0.0
    assert info["trial_t"] == 0.025
    np.testing.assert_allclose(info["trial_point"], np.array([0.0 + 0j]))
    assert info["trial_residual"] == 0.025


def test_parameter_homotopy_rejects_loose_min_step_newton_failure(monkeypatch):
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x - 2]),
        [x],
    )

    def loose_failed_corrector(
        parameter_homotopy,
        predicted,
        t_target,
        tol,
        max_corrections,
    ):
        return np.asarray(predicted, dtype=complex), False, 1, {
            "final_residual": 1e-6,
            "backtracks": 0,
            "min_step_scale": 1.0,
        }

    monkeypatch.setattr(
        parameter_homotopy_module,
        "_correct_parameter_prediction",
        loose_failed_corrector,
    )

    point, info = track_parameter_path(
        ph,
        start_point=np.array([1.0 + 0j]),
        options={
            "tol": 1e-10,
            "min_step_size": 0.05,
            "max_step_size": 0.05,
        },
    )

    assert not info["success"]
    assert info["failure_reason"] == "newton_failed"
    assert info["trial_residual"] == pytest.approx(1e-6)
    assert info["relaxed_correction_residual_limit"] < info["trial_residual"]
    assert info["relaxed_correction_acceptances"] == 0
    np.testing.assert_allclose(point, np.array([1.0 + 0j]))


@pytest.mark.parametrize("predictor", ["heun", "rk4"])
def test_parameter_homotopy_predictor_probe_exceptions_fallback_to_euler(
    monkeypatch,
    predictor,
):
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x - 2]),
        [x],
    )
    calls = {"count": 0}

    def tangent_with_failed_probe(parameter_homotopy, point, t, normalize_tangent):
        calls["count"] += 1
        if calls["count"] > 1:
            raise ValueError("forced predictor probe failure")
        return np.array([1.0 + 0j])

    monkeypatch.setattr(
        parameter_homotopy_module,
        "_parameter_tangent",
        tangent_with_failed_probe,
    )

    point, info = track_parameter_path(
        ph,
        start_point=np.array([1.0 + 0j]),
        start_t=0.0,
        end_t=0.05,
        options={
            "tol": 1e-10,
            "min_step_size": 0.05,
            "max_step_size": 0.1,
            "predictor": predictor,
        },
    )

    assert info["success"]
    assert info["predictor"] == predictor
    assert info["predictor_fallbacks"] == 1
    assert info["final_residual"] < 1e-10
    assert np.all(np.isfinite(point))
    assert point[0].real > 1.0


def test_parameter_homotopy_nonfinite_predictor_reports_trial_state(monkeypatch):
    x = polyvar('x')
    huge = np.finfo(float).max
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - huge]),
        PolynomialSystem([x]),
        [x],
    )

    def huge_tangent(parameter_homotopy, point, t, normalize_tangent):
        return np.array([huge + 0j])

    monkeypatch.setattr(
        parameter_homotopy_module,
        "_parameter_tangent",
        huge_tangent,
    )

    point, info = track_parameter_path(
        ph,
        start_point=np.array([huge + 0j]),
        start_t=0.0,
        end_t=1.0,
        options={
            "tol": 1e-10,
            "min_step_size": 1.0,
            "max_step_size": 2.0,
            "predictor": "euler",
        },
    )

    assert not info["success"]
    assert info["failure_reason"] == "nonfinite_predictor"
    assert info["t"] == 0.0
    assert info["trial_t"] == 1.0
    assert np.isinf(info["trial_residual"])
    assert not np.all(np.isfinite(info["trial_point"]))
    np.testing.assert_allclose(point, np.array([huge + 0j]))
    np.testing.assert_allclose(info["final_point"], point)
    assert info["final_residual"] < 1e-12


def test_parameter_homotopy_nonfinite_tangent_reports_current_residual(monkeypatch):
    x = polyvar('x')
    ph = ParameterHomotopy(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x + 1]),
        [x],
    )

    def nonfinite_tangent(parameter_homotopy, point, t, normalize_tangent):
        return np.array([np.nan + 0j])

    monkeypatch.setattr(
        parameter_homotopy_module,
        "_parameter_tangent",
        nonfinite_tangent,
    )

    point, info = track_parameter_path(
        ph,
        start_point=np.array([1.0 + 0j]),
        options={"tol": 1e-10},
    )

    assert not info["success"]
    assert info["failure_reason"] == "nonfinite_tangent"
    assert info["t"] == 0.0
    np.testing.assert_allclose(info["final_point"], point)
    assert info["final_residual"] == 0.0


def test_parameter_homotopy_nonsquare_is_quiet_by_default(capsys):
    x = polyvar('x')

    ParameterHomotopy(
        PolynomialSystem([x]),
        PolynomialSystem([x - 1]),
        PolynomialSystem([x + 1]),
        [x],
    )

    captured = capsys.readouterr()
    assert captured.out == ""


def test_cauchy_endgame_direct_polish_reports_failure_metadata():
    x = polyvar('x')
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    point, info = run_cauchy_endgame(
        start_system,
        target_system,
        np.array([0.0 + 0j]),
        0.0,
        [x],
        options={"newton_max_iters": 0},
    )

    assert not info["success"]
    assert info["status"] == "failed"
    assert info["failure_code"] == "newton_failed"
    assert np.isclose(info["final_residual"], 2.0)
    np.testing.assert_allclose(info["final_point"], point)


def test_cauchy_endgame_direct_polish_reports_success_metadata():
    x = polyvar('x')
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    point, info = run_cauchy_endgame(
        start_system,
        target_system,
        np.array([1.5 + 0j]),
        0.0,
        [x],
    )

    assert info["success"]
    assert info["status"] == "successful"
    assert info["failure_code"] is None
    assert info["final_residual"] < 1e-12
    np.testing.assert_allclose(point, np.array([2.0 + 0j]))
    np.testing.assert_allclose(info["final_point"], point)


def test_cauchy_endgame_accepts_solution_point():
    x = polyvar("x")
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    point, info = run_cauchy_endgame(
        start_system,
        target_system,
        Solution({x: 1.5 + 0j}, residual=0.5),
        0.0,
        [x],
    )

    assert info["success"]
    assert info["final_residual"] < 1e-12
    np.testing.assert_allclose(point, np.array([2.0 + 0j]))


def test_endgamer_setup_accepts_mapping_point_and_rejects_missing_coordinate():
    x, y = polyvar("x", "y")
    endgamer = Endgamer(
        PolynomialSystem([x - 1, y - 1]),
        PolynomialSystem([x - 2, y - 3]),
        [x, y],
    )

    endgamer.setup({"y": 1.0 + 0j, x: 1.0 + 0j}, 0.1)
    np.testing.assert_allclose(
        endgamer.current_point,
        np.array([1.0 + 0j, 1.0 + 0j]),
    )
    np.testing.assert_allclose(endgamer.xs[0], endgamer.current_point)

    with pytest.raises(ValueError, match=r"point is missing coordinate\(s\): y"):
        endgamer.setup({x: 1.0 + 0j}, 0.1)
    with pytest.raises(ValueError, match="conflicting coordinates.*x"):
        endgamer.setup({x: 1.0 + 0j, "x": 2.0 + 0j, y: 1.0 + 0j}, 0.1)


def test_cauchy_endgame_direct_polish_uses_scaled_extreme_residuals():
    x = polyvar("x")
    huge = 10**400
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([huge * (x - 2)])

    point, info = run_cauchy_endgame(
        start_system,
        target_system,
        np.array([2.0 + 0j]),
        0.0,
        [x],
        options={"newton_max_iters": 0, "abstol": 1e-10},
    )

    assert info["success"]
    assert info["status"] == "successful"
    assert info["failure_code"] is None
    assert info["final_residual"] == 0.0
    np.testing.assert_allclose(point, np.array([2.0 + 0j]))

    off_point, off_info = run_cauchy_endgame(
        start_system,
        target_system,
        np.array([1.0 + 0j]),
        0.0,
        [x],
        options={"newton_max_iters": 0, "abstol": 1e-10},
    )

    assert not off_info["success"]
    assert off_info["status"] == "failed"
    assert off_info["failure_code"] == "newton_failed"
    assert off_info["final_residual"] == 0.5
    np.testing.assert_allclose(off_point, np.array([1.0 + 0j]))


def test_cauchy_endgame_direct_polish_rejects_tiny_row_false_positive():
    x = polyvar("x")
    start_system = PolynomialSystem([x - 2])
    target_system = PolynomialSystem([1e-12 * (x - 1)])

    point, info = run_cauchy_endgame(
        start_system,
        target_system,
        np.array([2.0 + 0j]),
        0.0,
        [x],
        options={"newton_max_iters": 0, "abstol": 1e-10},
    )

    assert not info["success"]
    assert info["status"] == "failed"
    assert info["failure_code"] == "newton_failed"
    assert info["final_residual"] == pytest.approx(1.0)
    np.testing.assert_allclose(point, np.array([2.0 + 0j]))


def test_cauchy_endgame_direct_polish_rechecks_successful_corrector_residual(
    monkeypatch,
):
    x = polyvar("x")
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    def bad_successful_corrector(*args, **kwargs):
        return np.array([100.0 + 0j]), True, 1

    monkeypatch.setattr(
        endgame_module,
        "newton_corrector",
        bad_successful_corrector,
    )

    point, info = run_cauchy_endgame(
        start_system,
        target_system,
        np.array([1.0 + 0j]),
        0.0,
        [x],
        options={"abstol": 1e-12},
    )

    assert not info["success"]
    assert info["status"] == "failed"
    assert info["failure_code"] == "large_final_residual"
    assert info["final_residual"] > 1.0
    np.testing.assert_allclose(point, np.array([100.0 + 0j]))


def test_cauchy_endgame_rejects_invalid_options():
    x = polyvar('x')
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    with pytest.raises(TypeError, match="options must be a dictionary"):
        run_cauchy_endgame(
            start_system,
            target_system,
            np.array([1.5 + 0j]),
            0.0,
            [x],
            options="fast",
        )
    with pytest.raises(ValueError, match="Unknown endgame option"):
        run_cauchy_endgame(
            start_system,
            target_system,
            np.array([1.5 + 0j]),
            0.0,
            [x],
            options={"newton_iterations": 5},
        )
    with pytest.raises(ValueError, match="gamma must be finite and nonzero"):
        run_cauchy_endgame(
            start_system,
            target_system,
            np.array([1.5 + 0j]),
            0.0,
            [x],
            options={"gamma": 0.0},
        )
    with pytest.raises(TypeError, match="loopclosed_tolerance must be a number"):
        run_cauchy_endgame(
            start_system,
            target_system,
            np.array([1.5 + 0j]),
            0.1,
            [x],
            options={"loopclosed_tolerance": "tight"},
        )
    with pytest.raises(TypeError, match="newton_max_iters must be an integer"):
        run_cauchy_endgame(
            start_system,
            target_system,
            np.array([1.5 + 0j]),
            0.1,
            [x],
            options={"newton_max_iters": True},
        )
    with pytest.raises(ValueError, match="geometric_series_factor must be between 0 and 1"):
        run_cauchy_endgame(
            start_system,
            target_system,
            np.array([1.5 + 0j]),
            0.1,
            [x],
            options={"geometric_series_factor": 1.0},
        )


def test_endgame_option_classes_validate_numeric_inputs():
    with pytest.raises(TypeError, match="samples_per_loop must be an integer"):
        CauchyEndgame(samples_per_loop=True)
    with pytest.raises(ValueError, match="K must be between 0 and 1"):
        CauchyEndgame(K=0.0)
    with pytest.raises(TypeError, match="geometric_series_factor must be a number"):
        EndgamerOptions(geometric_series_factor="0.5")
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        EndgamerOptions(max_iterations=0)
    with pytest.raises(ValueError, match="max_winding_number must be positive"):
        EndgamerOptions(max_winding_number=0)


def test_endgamer_rejects_invalid_gamma():
    x = polyvar('x')
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    with pytest.raises(ValueError, match="gamma must be finite and nonzero"):
        Endgamer(
            start_system,
            target_system,
            [x],
            gamma=complex(float("nan"), 0.0),
        )


def test_endgamer_radius_failure_reports_metadata():
    x = polyvar('x')
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])
    endgamer = Endgamer(start_system, target_system, [x])
    endgamer.setup(np.array([0.0 + 0j]), 1e-15)

    point, info = endgamer.run()

    assert not info["success"]
    assert info["status"] == "failed"
    assert info["failure_code"] == "radius_too_small"
    assert np.isclose(info["final_residual"], 2.0)
    np.testing.assert_allclose(info["final_point"], point)


def test_endgamer_iteration_limit_reports_metadata():
    x = polyvar('x')
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])
    endgamer = Endgamer(
        start_system,
        target_system,
        [x],
        options=EndgamerOptions(max_iterations=1),
    )
    endgamer.setup(np.array([1.0 + 0j]), 0.1)

    point, info = endgamer.run()

    assert not info["success"]
    assert info["status"] == "failed"
    assert info["failure_code"] == "max_iterations_exceeded"
    assert info["max_iterations"] == 1
    np.testing.assert_allclose(info["final_point"], point)


def test_endgamer_converged_prediction_requires_target_residual(monkeypatch):
    x = polyvar('x')
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])
    endgamer = Endgamer(
        start_system,
        target_system,
        [x],
        options=EndgamerOptions(abstol=1e-12),
    )
    endgamer.setup(np.array([1.0 + 0j]), 0.1)
    endgamer.status = endgame_module.EndgameStatus.STARTED

    monkeypatch.setattr(
        endgamer,
        "track_loop",
        lambda: ([np.array([100.0 + 0j])], "success"),
    )
    monkeypatch.setattr(
        endgamer,
        "predict_endpoint",
        lambda samples: np.array([100.0 + 0j]),
    )
    monkeypatch.setattr(endgamer, "check_convergence", lambda: True)

    point, info = endgamer.run()

    assert not info["success"]
    assert info["status"] == "failed"
    assert info["failure_code"] == "large_final_residual"
    assert info["final_residual"] > 1.0
    np.testing.assert_allclose(info["final_point"], point)


def test_endgamer_first_heuristic_uses_configured_rng():
    class FixedRng:
        def __init__(self):
            self.standard_normal_calls = []

        def uniform(self, *args, **kwargs):
            return 0.0

        def standard_normal(self, size=None):
            self.standard_normal_calls.append(size)
            return np.array([1.0, 0.0])

    x, y = polyvar('x', 'y')
    rng = FixedRng()
    endgamer = Endgamer(
        PolynomialSystem([x - 1, y - 1]),
        PolynomialSystem([x - 2, y - 2]),
        [x, y],
        random_state=rng,
    )
    endgamer.R = 1.0

    accepted = endgamer.first_heuristic(
        np.array([1.0 + 0j, 0.0 + 0j]),
        np.array([0.5 + 0j, 0.0 + 0j]),
        np.array([0.25 + 0j, 0.0 + 0j]),
        np.array([0.125 + 0j, 0.0 + 0j]),
    )

    assert accepted
    assert rng.standard_normal_calls == [2, 2]


def test_endgamer_first_heuristic_rejects_malformed_rng_output():
    class WrongShapeRng:
        def uniform(self, *args, **kwargs):
            return 0.0

        def standard_normal(self, size=None):
            return np.array([1.0])

    x, y = polyvar('x', 'y')
    endgamer = Endgamer(
        PolynomialSystem([x - 1, y - 1]),
        PolynomialSystem([x - 2, y - 2]),
        [x, y],
        random_state=WrongShapeRng(),
    )
    endgamer.R = 1.0

    with pytest.raises(
        ValueError,
        match=r"standard_normal.*shape \(2,\).*endgame first heuristic direction real",
    ):
        endgamer.first_heuristic(
            np.array([1.0 + 0j, 0.0 + 0j]),
            np.array([0.5 + 0j, 0.0 + 0j]),
            np.array([0.25 + 0j, 0.0 + 0j]),
            np.array([0.125 + 0j, 0.0 + 0j]),
        )


def test_endgame_singular_corner_case():
    """Embedded positive-dimensional singular systems must not look finite."""
    x, y = polyvar('x', 'y')
    system = PolynomialSystem([x**2, x*y])

    with pytest.raises(ValueError, match="positive-dimensional.*witness-set"):
        solve(system, variables=[x, y], tol=1e-10, use_endgame=True, verbose=False)
