"""
Path tracking module for PyContinuum.

This module implements the predictor-corrector methods for tracking solution paths
along a homotopy.
"""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Integral, Real
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

import numpy as np
import time
from tqdm.auto import tqdm

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem
from pycontinuum.start_systems import _coerce_rng, _rng_uniform_scalar
# Import utility functions
from pycontinuum.utils import (
    evaluate_system_at_point,
    evaluate_scaled_system_at_point,
    evaluate_jacobian_at_point,
    evaluate_equation_scaled_jacobian_at_point,
    newton_corrector,
    newton_corrector_numeric,
    solve_linear_system,
    _mapping_coordinate_for_variable,
    _scaled_euclidean_norm,
)
from pycontinuum.endgame import run_cauchy_endgame, _validate_endgame_options

def check_singularity(target_system, current_point, variables, threshold, verbose=False, debug=False):
    """
    Check if the Jacobian at the current point is singular using a given threshold.
    
    Args:
        target_system: The target polynomial system.
        current_point: The current point on the path.
        variables: List of variables in the system.
        threshold: The condition number threshold for singularity.
        verbose: Whether to print verbose output.
        debug: Whether to print debug information.
        
    Returns:
        True if the Jacobian condition number exceeds the threshold, or if the Jacobian is singular.
        False otherwise.
    """
    threshold = _validate_positive_tracking_float("threshold", threshold)
    verbose = _validate_boolean_option("verbose", verbose)
    debug = _validate_boolean_option("debug", debug)
    jac = evaluate_equation_scaled_jacobian_at_point(
        target_system,
        current_point,
        variables,
    )
    if not np.all(np.isfinite(jac)):
        jac = evaluate_jacobian_at_point(target_system, current_point, variables)
    if not np.all(np.isfinite(jac)):
        if verbose:
            print("Nonfinite Jacobian detected!")
        return True
    try:
        if jac.ndim != 2:
            if verbose:
                print("Malformed Jacobian detected!")
            return True
        n_equations, n_variables = jac.shape
        if n_variables == 0:
            return False
        if n_equations < n_variables:
            if verbose:
                print("Underdetermined Jacobian detected!")
            return True
        rank = int(np.linalg.matrix_rank(jac))
        if rank < n_variables:
            if verbose:
                print("Rank-deficient Jacobian detected!")
            return True
        cond = float(np.linalg.cond(jac))
        if verbose and debug:
            print(f"Jacobian condition: {cond}")
        if not np.isfinite(cond):
            if verbose:
                print("Potential singularity detected!")
            return True
        if cond > threshold:
            if verbose:
                print("Potential singularity detected!")
            return True
        return False
    except (np.linalg.LinAlgError, ValueError, OverflowError, FloatingPointError):
        if verbose:
            print("Singular Jacobian detected!")
        return True


def _validate_positive_tracking_float(name: str, value: Any) -> float:
    numeric_value = _validate_tracking_float(name, value)
    if numeric_value <= 0:
        raise ValueError(f"{name} must be positive")
    return numeric_value


"""
NOTE: The evaluation helpers and Newton corrector are centralized in
`pycontinuum.utils`. This module imports and uses them directly to avoid
duplication and drift.
"""


def euler_predictor(t_current: float,
                   t_target: float,
                   point: np.ndarray,
                   tangent: np.ndarray) -> np.ndarray:
    """Euler predictor step.
    
    Args:
        t_current: Current parameter value
        t_target: Target parameter value
        point: Current point
        tangent: Direction to move (tangent to the path)
        
    Returns:
        Predicted point at t_target
    """
    # Simple linear extrapolation
    delta_t = t_target - t_current
    with np.errstate(over="ignore", invalid="ignore"):
        return point + delta_t * tangent


def heun_predictor(start_system: PolynomialSystem,
                   target_system: PolynomialSystem,
                   t_current: float,
                   t_target: float,
                   point: np.ndarray,
                   tangent: np.ndarray,
                   variables: List[Variable],
                   gamma: complex = 0.6+0.8j) -> Tuple[np.ndarray, bool]:
    """Second-order tangent predictor with Euler fallback.

    The first tangent predicts an Euler endpoint. A second tangent at that
    predicted endpoint corrects the slope average. If the second tangent is not
    finite, return the Euler endpoint and report the fallback.
    """
    euler_point = euler_predictor(t_current, t_target, point, tangent)
    if not np.all(np.isfinite(euler_point)):
        return euler_point, True

    try:
        target_tangent = compute_tangent(
            start_system,
            target_system,
            euler_point,
            t_target,
            variables,
            gamma,
        )
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return euler_point, True
    if not np.all(np.isfinite(target_tangent)):
        return euler_point, True

    delta_t = t_target - t_current
    with np.errstate(over="ignore", invalid="ignore"):
        predicted = point + 0.5 * delta_t * (tangent + target_tangent)
    if not np.all(np.isfinite(predicted)):
        return euler_point, True
    return predicted, False


def rk4_predictor(start_system: PolynomialSystem,
                  target_system: PolynomialSystem,
                  t_current: float,
                  t_target: float,
                  point: np.ndarray,
                  tangent: np.ndarray,
                  variables: List[Variable],
                  gamma: complex = 0.6+0.8j) -> Tuple[np.ndarray, bool]:
    """Fourth-order Runge-Kutta tangent predictor with Euler fallback."""
    delta_t = t_target - t_current
    euler_point = euler_predictor(t_current, t_target, point, tangent)
    if not np.all(np.isfinite(euler_point)):
        return euler_point, True

    half_t = t_current + 0.5 * delta_t
    k1 = tangent
    with np.errstate(over="ignore", invalid="ignore"):
        k2_point = point + 0.5 * delta_t * k1
    try:
        k2 = compute_tangent(
            start_system, target_system, k2_point, half_t, variables, gamma
        )
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return euler_point, True
    if not np.all(np.isfinite(k2)):
        return euler_point, True

    with np.errstate(over="ignore", invalid="ignore"):
        k3_point = point + 0.5 * delta_t * k2
    try:
        k3 = compute_tangent(
            start_system, target_system, k3_point, half_t, variables, gamma
        )
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return euler_point, True
    if not np.all(np.isfinite(k3)):
        return euler_point, True

    with np.errstate(over="ignore", invalid="ignore"):
        k4_point = point + delta_t * k3
    try:
        k4 = compute_tangent(
            start_system, target_system, k4_point, t_target, variables, gamma
        )
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return euler_point, True
    if not np.all(np.isfinite(k4)):
        return euler_point, True

    with np.errstate(over="ignore", invalid="ignore"):
        predicted = point + (delta_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if not np.all(np.isfinite(predicted)):
        return euler_point, True
    return predicted, False


def homotopy_function(start_system: PolynomialSystem,
                     target_system: PolynomialSystem,
                     point: Any,
                     t: float,
                     variables: List[Variable],
                     gamma: complex = 0.6+0.8j) -> np.ndarray:
    """Evaluate the homotopy function H(x, t) = (1-t)f(x) + t*gamma*g(x).
    
    Args:
        start_system: Start system g(x)
        target_system: Target system f(x)
        point: Point at which to evaluate, supplied as a coordinate vector,
            mapping keyed by variables or variable names, or an object with a
            ``values`` mapping.
        t: Homotopy parameter (1 = start, 0 = target)
        variables: System variables
        gamma: Random complex number for the homotopy
        
    Returns:
        Value of the homotopy at (point, t)
    """
    variables, t, gamma = _validate_homotopy_inputs(
        start_system,
        target_system,
        t,
        variables,
        gamma,
    )
    point = _coerce_start_point(point, variables, "point")
    # Use coefficient-scaled equations for the tracking homotopy. This makes
    # public path tracking invariant to harmless row scaling and prevents tiny
    # finite rows from accepting non-roots by raw absolute residual alone.
    f_val, g_val = _evaluate_homotopy_system_pair(
        start_system,
        target_system,
        point,
        variables,
    )
    
    # Compute the homotopy value
    with np.errstate(over="ignore", invalid="ignore"):
        return (1 - t) * f_val + t * gamma * g_val


def homotopy_jacobian(start_system: PolynomialSystem,
                     target_system: PolynomialSystem,
                     point: Any,
                     t: float,
                     variables: List[Variable],
                     gamma: complex = 0.6+0.8j) -> np.ndarray:
    """Evaluate the Jacobian of the homotopy function with respect to x.
    
    Args:
        start_system: Start system g(x)
        target_system: Target system f(x)
        point: Point at which to evaluate, supplied as a coordinate vector,
            mapping keyed by variables or variable names, or an object with a
            ``values`` mapping.
        t: Homotopy parameter (1 = start, 0 = target)
        variables: System variables
        gamma: Random complex number for the homotopy
        
    Returns:
        Jacobian of the homotopy at (point, t)
    """
    variables, t, gamma = _validate_homotopy_inputs(
        start_system,
        target_system,
        t,
        variables,
        gamma,
    )
    point = _coerce_start_point(point, variables, "point")
    # Use the Jacobian of the same coefficient-scaled equations used by
    # homotopy_function.
    jac_f, jac_g = _evaluate_homotopy_jacobian_pair(
        start_system,
        target_system,
        point,
        variables,
    )
    
    # Compute the homotopy Jacobian
    with np.errstate(over="ignore", invalid="ignore"):
        return (1 - t) * jac_f + t * gamma * jac_g


def compute_tangent(start_system: PolynomialSystem,
                   target_system: PolynomialSystem,
                   point: Any,
                   t: float,
                   variables: List[Variable],
                   gamma: complex = 0.6+0.8j,
                   normalize: bool = False) -> np.ndarray:
    """Compute the tangent to the path at a point.
    
    Args:
        start_system: Start system g(x)
        target_system: Target system f(x)
        point: Current point, supplied as a coordinate vector, mapping keyed by
            variables or variable names, or an object with a ``values`` mapping.
        t: Current t value
        variables: System variables
        gamma: Random complex number for the homotopy
        normalize: If True, return only the tangent direction. The default
            returns the actual derivative ``dx/dt`` for predictor accuracy.
        
    Returns:
        Tangent vector to the path at (point, t)
    """
    variables, t, gamma = _validate_homotopy_inputs(
        start_system,
        target_system,
        t,
        variables,
        gamma,
    )
    point = _coerce_start_point(point, variables, "point")
    # Get the Jacobian of H with respect to x
    jac = homotopy_jacobian(start_system, target_system, point, t, variables, gamma)
    
    # Compute dH/dt = -f(x) + gamma*g(x), using the same coefficient-scaled
    # equations as homotopy_function.
    f_val, g_val = _evaluate_homotopy_system_pair(
        start_system,
        target_system,
        point,
        variables,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        dH_dt = -f_val + gamma * g_val
    
    # Solve jac * dx/dt = -dH/dt to get the tangent vector. Homotopy
    # predictors need the actual derivative; normalizing it changes the
    # predicted path and can force unnecessary corrector work.
    tangent = solve_linear_system(jac, -dH_dt)

    if normalize:
        norm = _scaled_euclidean_norm(tangent)
        if norm > 1e-10:
            tangent = tangent / norm
        
    return tangent


def _evaluate_homotopy_system_pair(
    start_system: PolynomialSystem,
    target_system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        evaluate_scaled_system_at_point(target_system, point, variables),
        evaluate_scaled_system_at_point(start_system, point, variables),
    )


def _evaluate_homotopy_jacobian_pair(
    start_system: PolynomialSystem,
    target_system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        evaluate_equation_scaled_jacobian_at_point(target_system, point, variables),
        evaluate_equation_scaled_jacobian_at_point(start_system, point, variables),
    )


def _validate_homotopy_inputs(
    start_system: PolynomialSystem,
    target_system: PolynomialSystem,
    t: Any,
    variables: Any,
    gamma: Any,
) -> Tuple[List[Variable], float, complex]:
    _validate_homotopy_systems(start_system, target_system)
    variables = _normalize_tracking_variables(variables)
    _validate_variables_cover_homotopy_systems(
        start_system,
        target_system,
        variables,
    )
    t = _validate_tracking_float("t", t)
    gamma = _validate_gamma(gamma)
    return variables, t, gamma


def _validate_tracking_parameters(
    min_step_size: float,
    max_step_size: float,
    max_newton_iters: int,
    max_steps: int,
    max_predictor_norm: float,
    endgame_start: float,
    singularity_threshold: float,
    final_singularity_threshold: float,
) -> None:
    if isinstance(max_newton_iters, bool) or not isinstance(max_newton_iters, Integral):
        raise TypeError("max_newton_iters must be an integer")
    if isinstance(max_steps, bool) or not isinstance(max_steps, Integral):
        raise TypeError("max_steps must be an integer")

    min_step_size = _validate_tracking_float("min_step_size", min_step_size)
    max_step_size = _validate_tracking_float("max_step_size", max_step_size)
    max_predictor_norm = _validate_tracking_float(
        "max_predictor_norm",
        max_predictor_norm,
        allow_infinite=True,
    )
    endgame_start = _validate_tracking_float("endgame_start", endgame_start)
    singularity_threshold = _validate_tracking_float(
        "singularity_threshold",
        singularity_threshold,
    )
    final_singularity_threshold = _validate_tracking_float(
        "final_singularity_threshold",
        final_singularity_threshold,
    )

    if np.isnan(max_predictor_norm):
        raise ValueError("max_predictor_norm cannot be NaN")
    if min_step_size <= 0:
        raise ValueError("min_step_size must be positive")
    if max_step_size <= 0:
        raise ValueError("max_step_size must be positive")
    if min_step_size > max_step_size:
        raise ValueError("min_step_size cannot exceed max_step_size")
    if max_newton_iters <= 0:
        raise ValueError("max_newton_iters must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if max_predictor_norm <= 0:
        raise ValueError("max_predictor_norm must be positive")
    if not 0 <= endgame_start <= 1:
        raise ValueError("endgame_start must be between 0 and 1")
    if singularity_threshold <= 0:
        raise ValueError("singularity_threshold must be positive")
    if final_singularity_threshold <= 0:
        raise ValueError("final_singularity_threshold must be positive")


def _validate_tracking_float(
    name: str,
    value: Any,
    *,
    allow_infinite: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if np.isnan(numeric_value):
        if name == "max_predictor_norm":
            return numeric_value
        raise ValueError(f"{name} must be finite")
    if not allow_infinite and not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def _validate_worker_count(n_jobs: int) -> int:
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral):
        raise TypeError("n_jobs must be an integer")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")
    return int(n_jobs)


def _validate_predictor_method(predictor: str) -> str:
    if not isinstance(predictor, str):
        raise TypeError("predictor must be a string")
    normalized = predictor.lower()
    if normalized not in {"euler", "heun", "rk4"}:
        raise ValueError("predictor must be 'euler', 'heun', or 'rk4'")
    return normalized


def _validate_boolean_option(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _validate_positive_finite_float(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return numeric_value


def _validate_gamma(gamma: Any) -> complex:
    if isinstance(gamma, (bool, np.bool_)):
        raise TypeError("gamma must be a complex number")
    try:
        value = complex(gamma)
    except (TypeError, ValueError) as exc:
        raise TypeError("gamma must be a complex number") from exc
    if (
        not np.isfinite(value.real)
        or not np.isfinite(value.imag)
        or value == 0
    ):
        raise ValueError("gamma must be finite and nonzero")
    return value


def _validate_homotopy_systems(
    start_system: PolynomialSystem,
    target_system: PolynomialSystem,
) -> None:
    if not isinstance(start_system, PolynomialSystem):
        raise TypeError("start_system must be a PolynomialSystem")
    if not isinstance(target_system, PolynomialSystem):
        raise TypeError("target_system must be a PolynomialSystem")
    if len(start_system.equations) != len(target_system.equations):
        raise ValueError(
            "start_system and target_system must have the same number of equations"
        )


def _normalize_tracking_variables(variables: Any) -> List[Variable]:
    try:
        normalized = list(variables)
    except TypeError as exc:
        raise TypeError(
            "variables must be an iterable of Variable objects"
        ) from exc

    seen = set()
    duplicates = []
    for index, variable in enumerate(normalized):
        if not isinstance(variable, Variable):
            raise TypeError(f"variables[{index}] must be a Variable")
        if variable in seen:
            duplicates.append(variable.name)
        seen.add(variable)
    if duplicates:
        raise ValueError(
            "Variable list contains duplicate variable(s): "
            + ", ".join(sorted(set(duplicates)))
        )
    return normalized


def _validate_variables_cover_homotopy_systems(
    start_system: PolynomialSystem,
    target_system: PolynomialSystem,
    variables: List[Variable],
) -> None:
    seen = set(variables)
    homotopy_variables = start_system.variables().union(target_system.variables())
    missing = sorted(
        variable.name
        for variable in homotopy_variables
        if variable not in seen
    )
    if missing:
        raise ValueError(
            "Variable list is missing homotopy variable(s): "
            + ", ".join(missing)
        )
    extra = sorted(
        variable.name for variable in seen if variable not in homotopy_variables
    )
    if extra:
        raise ValueError(
            "Variable list contains variable(s) not used by homotopy: "
            + ", ".join(extra)
        )


def _coerce_start_point(
    point: Any,
    variables: List[Variable],
    label: str,
) -> np.ndarray:
    values = None
    if isinstance(point, Mapping):
        values = point
    elif isinstance(getattr(point, "values", None), Mapping):
        values = point.values

    if values is not None:
        array = _coerce_start_point_mapping(values, variables, label)
    else:
        try:
            array = np.asarray(point, dtype=complex)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{label} must be a numeric one-dimensional point"
            ) from exc
    if array.ndim != 1 or array.size != len(variables):
        raise ValueError(
            f"{label} must have {len(variables)} coordinate(s); "
            f"got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains nonfinite values")
    return array


def _coerce_start_point_mapping(
    values: Mapping,
    variables: List[Variable],
    label: str,
) -> np.ndarray:
    coordinates = []
    missing = []
    for variable in variables:
        found, coordinate = _mapping_coordinate_for_variable(
            values,
            variable,
            label,
        )
        if found:
            coordinates.append(coordinate)
        else:
            missing.append(variable.name)
    if missing:
        raise ValueError(
            f"{label} is missing coordinate(s): "
            + ", ".join(sorted(missing))
        )
    try:
        return np.asarray(coordinates, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} coordinate(s) must be numeric") from exc


def _validate_start_point_on_start_system(
    start_system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
    tol: float,
    label: str,
) -> Tuple[float, float, float]:
    residuals = evaluate_system_at_point(start_system, point, variables)
    scaled_residuals = evaluate_scaled_system_at_point(start_system, point, variables)
    residual = _scaled_euclidean_norm(residuals)
    scaled_residual = _scaled_euclidean_norm(scaled_residuals)
    residual_limit = 1000.0 * tol
    if scaled_residual > residual_limit:
        raise ValueError(
            f"{label} does not satisfy start_system within tolerance "
            f"({residual:.2e} raw, {scaled_residual:.2e} scaled > "
            f"{residual_limit:.2e})"
        )
    return residual, residual_limit, scaled_residual


def _spawn_endgame_random_states(
    endgame_options: Optional[Dict[str, Any]],
    n_paths: int,
) -> Optional[List[int]]:
    if not endgame_options or "random_state" not in endgame_options:
        return None

    random_state = endgame_options.get("random_state")
    if random_state is None:
        return None

    rng = _coerce_rng(random_state)
    return [_draw_endgame_seed(rng) for _ in range(n_paths)]


def _draw_endgame_seed(rng: Any) -> int:
    high = 2**32 - 1
    if hasattr(rng, "integers"):
        return _validated_endgame_seed(
            _call_seed_rng(rng.integers, 0, high, "random_state.integers"),
            high,
            "random_state.integers",
        )
    if hasattr(rng, "randint"):
        high = 2**31 - 1
        return _validated_endgame_seed(
            _call_seed_rng(rng.randint, 0, high, "random_state.randint"),
            high,
            "random_state.randint",
        )
    return _validated_endgame_seed(
        int(
            _rng_uniform_scalar(
                rng,
                0.0,
                float(high),
                context="endgame random seed",
            )
        ),
        high,
        "random_state.uniform",
    )


def _call_seed_rng(draw: Callable[[int, int], Any], low: int, high: int, name: str) -> Any:
    try:
        return draw(low, high)
    except Exception as exc:
        raise ValueError(f"{name} failed while generating endgame random seed") from exc


def _validated_endgame_seed(value: Any, high: int, name: str) -> int:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(
            f"{name} must return a scalar integer endgame random seed"
        )
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(
            f"{name} must return a scalar integer endgame random seed"
        )
    if not isinstance(scalar, Integral):
        raise ValueError(
            f"{name} must return a scalar integer endgame random seed"
        )
    seed = int(scalar)
    if seed < 0 or seed >= high:
        raise ValueError(
            f"{name} must return an endgame random seed in [0, {high})"
        )
    return seed


def _endgame_options_for_path(
    endgame_options: Optional[Dict[str, Any]],
    path_index: int,
    random_states: Optional[List[int]],
) -> Optional[Dict[str, Any]]:
    if not endgame_options and random_states is None:
        return None

    path_options = {} if endgame_options is None else dict(endgame_options)
    if random_states is not None:
        path_options["random_state"] = random_states[path_index]
    return path_options


def homotopy_residual_at(
    start_system: PolynomialSystem,
    target_system: PolynomialSystem,
    point: np.ndarray,
    t: float,
    variables: List[Variable],
    gamma: complex,
) -> float:
    variables = _normalize_tracking_variables(variables)
    try:
        point = _coerce_start_point(point, variables, "point")
    except ValueError:
        return float("inf")
    residuals = homotopy_function(
        start_system,
        target_system,
        point,
        t,
        variables,
        gamma,
    )
    if not np.all(np.isfinite(residuals)):
        return float("inf")
    return _scaled_euclidean_norm(residuals)


def target_residual_at(
    target_system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> float:
    variables = _normalize_tracking_variables(variables)
    try:
        point = _coerce_start_point(point, variables, "point")
    except ValueError:
        return float("inf")
    residuals = evaluate_scaled_system_at_point(target_system, point, variables)
    if not np.all(np.isfinite(residuals)):
        return float("inf")
    return _scaled_euclidean_norm(residuals)


def track_single_path(start_system: PolynomialSystem,
                     target_system: PolynomialSystem,
                     start_solution: Any,
                     variables: List[Variable],
                     tol: float = 1e-10,
                     min_step_size: float = 1e-6,
                     max_step_size: float = 0.05,
                     max_newton_iters: int = 10,
                     max_steps: int = 10000,
                     max_predictor_norm: float = float("inf"),
                     gamma: complex = 0.6+0.8j,
                     endgame_start: float = 0.1,
                     singularity_threshold: float = 1e3,
                      final_singularity_threshold: float = 1e8,
                      store_paths: bool = False,
                      use_endgame: bool = True,
                      endgame_options: Optional[Dict[str, Any]] = None,
                      predictor: str = "euler",
                      verbose: bool = False,
                      debug: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Track a single path from start_solution (t=1) to the target system (t=0).
    Args:
        start_system: Start system g(x)
        target_system: Target system f(x)
        start_solution: Solution of the start system, supplied as a coordinate
            vector, mapping keyed by variables or variable names, or an object
            with a ``values`` mapping.
        variables: System variables
        tol: Tolerance for numerical methods
        min_step_size: Minimum step size for t
        max_step_size: Maximum step size for t
        max_newton_iters: Maximum Newton iterations per corrector attempt
        max_steps: Maximum predictor-corrector steps before failing a path
        max_predictor_norm: Maximum allowed predictor displacement before
            adaptively shrinking the step
        gamma: Random complex number for the homotopy
        endgame_start: t-value at which to start the endgame
        singularity_threshold: Jacobian condition threshold for entering endgame
        final_singularity_threshold: Jacobian condition threshold at t=0
        store_paths: Whether to store path points
        use_endgame: Whether to use the endgame for singular endpoints
        endgame_options: Optional endgame configuration
        predictor: Predictor method, ``"euler"``, ``"heun"``, or ``"rk4"``
        
    Returns:
        Tuple of (end_solution, path_info)
    """
    _validate_homotopy_systems(start_system, target_system)
    variables = _normalize_tracking_variables(variables)
    _validate_variables_cover_homotopy_systems(
        start_system, target_system, variables
    )
    tol = _validate_positive_finite_float("tol", tol)
    _validate_tracking_parameters(
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        max_newton_iters=max_newton_iters,
        max_steps=max_steps,
        max_predictor_norm=max_predictor_norm,
        endgame_start=endgame_start,
        singularity_threshold=singularity_threshold,
        final_singularity_threshold=final_singularity_threshold,
    )
    min_step_size = float(min_step_size)
    max_step_size = float(max_step_size)
    max_newton_iters = int(max_newton_iters)
    max_steps = int(max_steps)
    max_predictor_norm = float(max_predictor_norm)
    endgame_start = float(endgame_start)
    singularity_threshold = float(singularity_threshold)
    final_singularity_threshold = float(final_singularity_threshold)
    predictor = _validate_predictor_method(predictor)
    gamma = _validate_gamma(gamma)
    store_paths = _validate_boolean_option("store_paths", store_paths)
    use_endgame = _validate_boolean_option("use_endgame", use_endgame)
    verbose = _validate_boolean_option("verbose", verbose)
    debug = _validate_boolean_option("debug", debug)
    endgame_options = _validate_endgame_options(
        endgame_options,
        name="endgame_options",
    )
    start_solution = _coerce_start_point(
        start_solution, variables, "start_solution"
    )
    (
        start_residual,
        start_residual_limit,
        start_scaled_residual,
    ) = _validate_start_point_on_start_system(
        start_system,
        start_solution,
        variables,
        tol,
        "start_solution",
    )

    # Initialize tracking
    t = 1.0
    current_point = np.array(start_solution, dtype=complex)
    step_size = max_step_size
    
    # Path data for return
    path_info = {
        'success': False,
        'singular': False,
        'steps': 0,
        'newton_iters': 0,
        'start_t': 1.0,
        'end_t': 0.0,
        'direction': -1,
        'tol': float(tol),
        'min_step_size': float(min_step_size),
        'max_step_size': float(max_step_size),
        'initial_step_size': float(step_size),
        'max_newton_iters': int(max_newton_iters),
        'max_steps': int(max_steps),
        'max_predictor_norm': float(max_predictor_norm),
        'gamma': gamma,
        'endgame_start': float(endgame_start),
        'singularity_threshold': float(singularity_threshold),
        'final_singularity_threshold': float(final_singularity_threshold),
        'store_paths': bool(store_paths),
        'use_endgame': bool(use_endgame),
        'step_reductions': 0,
        'max_observed_predictor_norm': 0.0,
        'max_predictor_correction_norm': 0.0,
        'predictor': predictor,
        'predictor_fallbacks': 0,
        'final_t': t,
        'final_residual': float("inf"),
        'start_residual': float(start_residual),
        'start_scaled_residual': float(start_scaled_residual),
        'start_residual_limit': float(start_residual_limit),
        'failure_reason': None,
        'path_points': [(t, current_point.copy())] if store_paths else []
    }
    
    # Use a simple continuation method to track the path
    while t > 0:
        if path_info['steps'] >= max_steps:
            path_info['failure_reason'] = 'max_steps_exceeded'
            path_info['final_t'] = t
            path_info['final_point'] = current_point.copy()
            path_info['final_residual'] = homotopy_residual_at(
                start_system,
                target_system,
                current_point,
                t,
                variables,
                gamma,
            )
            return current_point, path_info

        path_info['steps'] += 1
        
        # Check if we should switch to endgame
        if use_endgame and t <= endgame_start:
            # Import here to avoid circular import
            from pycontinuum.endgame import run_cauchy_endgame
            # Check if path might be approaching a singular point
            # by examining Jacobian condition number
            
            might_be_singular = check_singularity(
                target_system,
                current_point,
                variables,
                threshold=singularity_threshold,
                verbose=verbose,
                debug=debug
            )

            
            if might_be_singular:
                local_endgame_options = {
                    'abstol': tol,
                    'geometric_series_factor': 0.5,
                    'gamma': gamma
                }
                if endgame_options:
                    local_endgame_options.update(endgame_options)

                polish_max_iters = int(local_endgame_options.get('newton_max_iters', 50))
                if t < 1.0 and polish_max_iters > 0:
                    polished_point, polish_success, polish_iters = newton_corrector(
                        target_system,
                        current_point,
                        variables,
                        max_iters=polish_max_iters,
                        tol=tol,
                    )
                    polish_residual = target_residual_at(
                        target_system,
                        polished_point,
                        variables,
                    )
                    polish_accepted = bool(
                        np.isfinite(polish_residual)
                        and (polish_success or polish_residual < 100.0 * tol)
                    )
                    path_info['endgame_target_polish'] = {
                        'attempted': True,
                        'success': bool(polish_success),
                        'accepted': polish_accepted,
                        'iterations': int(polish_iters),
                        'residual': float(polish_residual),
                    }
                    if polish_accepted:
                        path_info['success'] = True
                        path_info['singular'] = True
                        path_info['endgame_used'] = True
                        path_info['winding_number'] = 1
                        path_info['final_point'] = np.array(
                            polished_point,
                            dtype=complex,
                        )
                        path_info['final_t'] = 0.0
                        path_info['final_residual'] = float(polish_residual)
                        path_info['failure_reason'] = None
                        return polished_point, path_info
                else:
                    path_info['endgame_target_polish'] = {
                        'attempted': False,
                        'success': False,
                        'accepted': False,
                        'iterations': 0,
                        'residual': float("inf"),
                    }

                # Switch to Cauchy endgame
                if verbose:
                    print(f"Switching to Cauchy endgame at t={t}")
                
                end_point, endgame_info = run_cauchy_endgame(
                    start_system, target_system, current_point, t, 
                    variables, local_endgame_options
                )
                
                # Update path info with endgame results. The target residual
                # below is authoritative for success.
                path_info['success'] = False
                path_info['singular'] = True
                path_info['endgame_used'] = True
                path_info['winding_number'] = endgame_info['winding_number']
                
                # Check residual of the endgame solution
                end_residual = target_residual_at(
                    target_system,
                    end_point,
                    variables,
                )
                path_info['final_point'] = np.array(end_point, dtype=complex)
                path_info['final_t'] = 0.0
                path_info['final_residual'] = float(end_residual)
                
                # If residual is small enough, mark as success
                if end_residual < 100 * tol:
                    path_info['success'] = True
                    path_info['failure_reason'] = None
                elif endgame_info.get('success', False):
                    path_info['failure_reason'] = 'large_final_residual'
                else:
                    path_info['failure_reason'] = (
                        endgame_info.get('failure_code')
                        or 'large_final_residual'
                    )
                
                # Store path points if requested
                if store_paths and endgame_info.get('predictions'):
                    for i, pred in enumerate(endgame_info['predictions']):
                        # Use a small decreasing t value for predictions
                        path_info['path_points'].append((t * (0.5 ** (i+1)), pred))
                
                return end_point, path_info
        
        # Reduce step size for the final approach
        if t < endgame_start and step_size > min_step_size:
            step_size = max(min_step_size, t / 10)  # Reduce step size for endpoint accuracy
            
        # Check for infinity or NaN
        if not np.all(np.isfinite(current_point)):
            if verbose:
                print("Path diverged to infinity or NaN")
            path_info['failure_reason'] = 'nonfinite_point'
            path_info['final_t'] = t
            path_info['final_point'] = current_point.copy()
            return current_point, path_info
        
        # Compute tangent at current point
        tangent = compute_tangent(start_system, target_system, current_point, t, variables, gamma)
        if not np.all(np.isfinite(tangent)):
            path_info['failure_reason'] = 'nonfinite_tangent'
            path_info['final_t'] = t
            path_info['final_point'] = current_point.copy()
            path_info['final_residual'] = homotopy_residual_at(
                start_system,
                target_system,
                current_point,
                t,
                variables,
                gamma,
            )
            return current_point, path_info

        def predict_at(target_t: float) -> np.ndarray:
            euler_point = euler_predictor(t, target_t, current_point, tangent)
            if predictor == "heun":
                predicted_point, fallback = heun_predictor(
                    start_system,
                    target_system,
                    t,
                    target_t,
                    current_point,
                    tangent,
                    variables,
                    gamma,
                )
                if fallback:
                    path_info['predictor_fallbacks'] += 1
                else:
                    correction_norm = _scaled_euclidean_norm(
                        predicted_point - euler_point
                    )
                    path_info['max_predictor_correction_norm'] = max(
                        path_info['max_predictor_correction_norm'],
                        correction_norm,
                    )
                return predicted_point
            if predictor == "rk4":
                predicted_point, fallback = rk4_predictor(
                    start_system,
                    target_system,
                    t,
                    target_t,
                    current_point,
                    tangent,
                    variables,
                    gamma,
                )
                if fallback:
                    path_info['predictor_fallbacks'] += 1
                else:
                    correction_norm = _scaled_euclidean_norm(
                        predicted_point - euler_point
                    )
                    path_info['max_predictor_correction_norm'] = max(
                        path_info['max_predictor_correction_norm'],
                        correction_norm,
                    )
                return predicted_point
            return euler_point

        # Keep true dx/dt but adapt the t-step when it predicts an excessive
        # displacement in x-space. This preserves predictor geometry while
        # avoiding jumps that would make Newton correction needlessly hard.
        trial_step_size = min(step_size, t)
        while True:
            t_target = max(0.0, t - trial_step_size)
            predicted = predict_at(t_target)
            if not np.all(np.isfinite(predicted)):
                path_info['max_observed_predictor_norm'] = float("inf")
                if trial_step_size <= min_step_size:
                    path_info['failure_reason'] = 'nonfinite_predictor'
                    path_info['final_t'] = t
                    path_info['final_point'] = current_point.copy()
                    path_info['final_residual'] = homotopy_residual_at(
                        start_system,
                        target_system,
                        current_point,
                        t,
                        variables,
                        gamma,
                    )
                    path_info['trial_t'] = t_target
                    path_info['trial_point'] = np.asarray(
                        predicted,
                        dtype=complex,
                    ).copy()
                    path_info['trial_residual'] = float("inf")
                    return current_point, path_info
                trial_step_size = max(min_step_size, trial_step_size / 2)
                path_info['step_reductions'] += 1
                continue
            predictor_norm = _scaled_euclidean_norm(predicted - current_point)
            path_info['max_observed_predictor_norm'] = max(
                path_info['max_observed_predictor_norm'], predictor_norm
            )
            if predictor_norm <= max_predictor_norm or trial_step_size <= min_step_size:
                break
            shrink = 0.8 * max_predictor_norm / max(predictor_norm, 1e-300)
            next_step_size = max(min_step_size, trial_step_size * shrink)
            if next_step_size >= trial_step_size:
                next_step_size = max(min_step_size, trial_step_size / 2)
            trial_step_size = next_step_size
            path_info['step_reductions'] += 1

        step_size = trial_step_size

        def correct_prediction(candidate: np.ndarray, target_t: float):
            def f_numeric(x: np.ndarray) -> np.ndarray:
                return homotopy_function(
                    start_system, target_system, x, target_t, variables, gamma
                )

            def j_numeric(x: np.ndarray) -> np.ndarray:
                return homotopy_jacobian(
                    start_system, target_system, x, target_t, variables, gamma
                )

            return newton_corrector_numeric(
                f_numeric, j_numeric, candidate, max_iters=max_newton_iters, tol=tol
            )

        def homotopy_residual(point: np.ndarray, target_t: float) -> float:
            return homotopy_residual_at(
                start_system,
                target_system,
                point,
                target_t,
                variables,
                gamma,
            )

        corrected, success, iters = correct_prediction(predicted, t_target)
        residual_after_correction = homotopy_residual(corrected, t_target)
        if not success and residual_after_correction <= 10.0 * tol:
            success = True
        
        path_info['newton_iters'] += iters
        
        # Adjust step size based on Newton convergence
        if not success or iters > 5:
            # Reduce step size and retry the correction with smaller step
            retry = True
            while retry and (step_size > min_step_size):
                step_size = max(min_step_size, step_size / 2)
                t_target = max(0.0, t - step_size)
                predicted = predict_at(t_target)
                if not np.all(np.isfinite(predicted)):
                    path_info['max_observed_predictor_norm'] = float("inf")
                    path_info['step_reductions'] += 1
                    if step_size <= min_step_size:
                        path_info['failure_reason'] = 'nonfinite_predictor'
                        path_info['final_t'] = t
                        path_info['final_point'] = current_point.copy()
                        path_info['final_residual'] = homotopy_residual_at(
                            start_system,
                            target_system,
                            current_point,
                            t,
                            variables,
                            gamma,
                        )
                        path_info['trial_t'] = t_target
                        path_info['trial_point'] = np.asarray(
                            predicted,
                            dtype=complex,
                        ).copy()
                        path_info['trial_residual'] = float("inf")
                        return current_point, path_info
                    continue
                corrected, success, iters_retry = correct_prediction(predicted, t_target)
                residual_after_correction = homotopy_residual(corrected, t_target)
                if not success and residual_after_correction <= 10.0 * tol:
                    success = True
                path_info['newton_iters'] += iters_retry
                path_info['step_reductions'] += 1
                retry = not success and (step_size > min_step_size)
            if not success:
                if verbose:
                    print(
                        f"Newton failed to converge at t={t_target}, path may be near a singularity"
                    )
                path_info['failure_reason'] = (
                    'nonfinite_corrector'
                    if not np.all(np.isfinite(corrected))
                    else 'newton_failed'
                )
                path_info['final_t'] = t_target
                path_info['final_point'] = np.array(corrected, dtype=complex)
                path_info['final_residual'] = float(residual_after_correction)
                return corrected, path_info
        elif iters <= 2 and step_size < max_step_size:
            # Increase step size if Newton converges quickly
            step_size = min(max_step_size, step_size * 1.5)
        
        # Update the current point and t
        current_point = corrected
        t = t_target
        
        # Save point if store_paths
        if store_paths:
            path_info['path_points'].append((t, current_point.copy()))
    
    # We've reached t=0, final check for target system
    final_residual = target_residual_at(target_system, current_point, variables)
    path_info['final_point'] = current_point.copy()
    path_info['final_t'] = 0.0
    path_info['final_residual'] = float(final_residual)
    if final_residual < 100 * tol:
        path_info['success'] = True
        
        path_info['singular'] = check_singularity(
            target_system,
            current_point,
            variables,
            threshold=final_singularity_threshold,
            verbose=verbose,
            debug=debug
        )
    else:
        path_info['failure_reason'] = 'large_final_residual'

            
    return current_point, path_info

def track_paths(start_system: PolynomialSystem,
                target_system: PolynomialSystem,
                start_solutions: Any,
                variables: List[Variable],
                tol: float = 1e-8,
                min_step_size: float = 1e-6,
                max_step_size: float = 0.05,
                max_newton_iters: int = 10,
                max_steps: int = 10000,
                max_predictor_norm: float = float("inf"),
                gamma: complex = 0.6+0.8j,
                endgame_start: float = 0.1,
                singularity_threshold: float = 1e3,
                final_singularity_threshold: float = 1e8,
                verbose: bool = False,
                store_paths: bool = False,
                use_endgame: bool = True,
                endgame_options: Optional[Dict[str, Any]] = None,
                predictor: str = "euler",
                n_jobs: int = 1) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Track solution paths from start solutions to the target system.
    
    Args:
        start_system: The start polynomial system.
        target_system: The target polynomial system.
        start_solutions: Iterable of start solution points. Each point may be
            a coordinate vector, mapping keyed by variables or variable names,
            or an object with a ``values`` mapping.
        variables: List of variables in the system.
        tol: Tolerance for convergence of the corrector.
        min_step_size: Minimum t-step size.
        max_step_size: Maximum t-step size.
        max_newton_iters: Maximum Newton iterations per corrector attempt.
        max_steps: Maximum predictor-corrector steps per path.
        max_predictor_norm: Maximum predictor displacement before shrinking.
        gamma: Gamma multiplier for the total-degree homotopy.
        endgame_start: t-value at which singular endgame can begin.
        singularity_threshold: Condition threshold for early singular detection.
        final_singularity_threshold: Condition threshold at the target endpoint.
        verbose: Whether to print progress information.
        store_paths: Whether to store all points along each path.
        use_endgame: Whether to use the endgame procedure near t=0.
        endgame_options: Optional dictionary of options for the endgame procedure.
        predictor: Predictor method, ``"euler"``, ``"heun"``, or ``"rk4"``.
        n_jobs: Number of worker threads for independent path tracking.
        
    Returns:
        Tuple of (end_solutions, path_results).
        end_solutions: List of final points for each path.
        path_results: List of dictionaries containing result info for each path.
    """
    _validate_homotopy_systems(start_system, target_system)
    variables = _normalize_tracking_variables(variables)
    _validate_variables_cover_homotopy_systems(
        start_system, target_system, variables
    )
    tol = _validate_positive_finite_float("tol", tol)
    _validate_tracking_parameters(
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        max_newton_iters=max_newton_iters,
        max_steps=max_steps,
        max_predictor_norm=max_predictor_norm,
        endgame_start=endgame_start,
        singularity_threshold=singularity_threshold,
        final_singularity_threshold=final_singularity_threshold,
    )
    predictor = _validate_predictor_method(predictor)
    gamma = _validate_gamma(gamma)
    n_jobs = _validate_worker_count(n_jobs)
    verbose = _validate_boolean_option("verbose", verbose)
    store_paths = _validate_boolean_option("store_paths", store_paths)
    use_endgame = _validate_boolean_option("use_endgame", use_endgame)
    endgame_options = _validate_endgame_options(
        endgame_options,
        name="endgame_options",
    )

    try:
        raw_start_solutions = list(start_solutions)
    except TypeError as exc:
        raise TypeError("start_solutions must be an iterable of points") from exc
    n_paths = len(raw_start_solutions)
    validated_start_solutions: List[np.ndarray] = []
    for index, start_solution in enumerate(raw_start_solutions):
        point = _coerce_start_point(
            start_solution,
            variables,
            f"start_solutions[{index}]",
        )
        _validate_start_point_on_start_system(
            start_system,
            point,
            variables,
            tol,
            f"start_solutions[{index}]",
        )
        validated_start_solutions.append(point)

    end_solutions: List[Optional[np.ndarray]] = [None] * n_paths
    path_results: List[Optional[Dict[str, Any]]] = [None] * n_paths
    endgame_random_states = _spawn_endgame_random_states(endgame_options, n_paths)
    
    if verbose:
        print(f"Tracking {n_paths} paths from t=1 to t=0...")
        pbar = tqdm(total=n_paths)
        
    start_time = time.time()
    
    def track_and_polish_path(
        path_index: int,
        start_sol: np.ndarray,
        worker_verbose: bool,
    ) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        path_endgame_options = _endgame_options_for_path(
            endgame_options,
            path_index,
            endgame_random_states,
        )
        end_sol, path_info = track_single_path(
            start_system=start_system,
            target_system=target_system,
            start_solution=start_sol,
            variables=variables,
            tol=tol,
            min_step_size=min_step_size,
            max_step_size=max_step_size,
            max_newton_iters=max_newton_iters,
            max_steps=max_steps,
            max_predictor_norm=max_predictor_norm,
            gamma=gamma,
            endgame_start=endgame_start,
            singularity_threshold=singularity_threshold,
            final_singularity_threshold=final_singularity_threshold,
            store_paths=store_paths,
            use_endgame=use_endgame,
            endgame_options=path_endgame_options,
            predictor=predictor,
            verbose=worker_verbose,
            debug=False  # Default to no debug output
        )
        
        # Polish successful endpoints at t=0. For regular paths this is just a
        # Newton cleanup; preserve the existing singular classification.
        if (
            use_endgame
            and path_info.get('success', False)
        ):
            was_singular = path_info.get('singular', False)
            used_endgame = path_info.get('endgame_used', False)
            original_residual = float(path_info.get('final_residual', float("inf")))
            polish_options = {'abstol': min(tol, 1e-12)}
            if path_endgame_options:
                polish_options.update(path_endgame_options)
            final_point, endgame_info = run_cauchy_endgame(
                start_system=start_system,
                target_system=target_system,
                point=end_sol,
                t=0.0,  # Pass the current t value
                variables=variables,
                options=polish_options
            )

            polished_point = np.array(final_point, dtype=complex)
            polished_residual = target_residual_at(
                target_system,
                polished_point,
                variables,
            )
            polish_success = bool(endgame_info.get('success', False)) or (
                np.isfinite(polished_residual)
                and polished_residual <= tol
            )
            accept_polish = (
                polish_success
                and np.isfinite(polished_residual)
                and (
                    not np.isfinite(original_residual)
                    or polished_residual <= max(original_residual, tol)
                )
            )

            path_info['polish'] = {
                'attempted': True,
                'success': polish_success,
                'accepted': accept_polish,
                'residual': polished_residual,
                'steps': endgame_info.get('steps', 0),
                'winding_number': endgame_info.get('winding_number'),
            }
            if accept_polish:
                end_sol = polished_point
                path_info['final_point'] = polished_point.copy()
                path_info['final_residual'] = polished_residual
            path_info['singular'] = was_singular
            if was_singular or used_endgame:
                path_info['endgame_used'] = True
            else:
                path_info['polished'] = accept_polish
        
        path_info['path_index'] = path_index
        return path_index, np.array(end_sol, dtype=complex), path_info

    if n_jobs == 1 or n_paths <= 1:
        for i, start_sol in enumerate(validated_start_solutions):
            if verbose and i > 0 and i % 10 == 0:
                elapsed = time.time() - start_time
                paths_per_sec = i / elapsed
                eta = (n_paths - i) / paths_per_sec if paths_per_sec > 0 else 0
                print(
                    f"Completed {i}/{n_paths} paths "
                    f"({paths_per_sec:.2f} paths/sec, ETA: {eta:.1f}s)"
                )

            path_index, end_sol, path_info = track_and_polish_path(
                i, start_sol, verbose
            )
            end_solutions[path_index] = end_sol
            path_results[path_index] = path_info

            if verbose:
                pbar.update(1)
    else:
        worker_count = min(n_jobs, n_paths)
        if verbose:
            print(f"Using {worker_count} path-tracking workers")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(track_and_polish_path, i, start_sol, False)
                for i, start_sol in enumerate(validated_start_solutions)
            ]
            completed = 0
            for future in as_completed(futures):
                path_index, end_sol, path_info = future.result()
                end_solutions[path_index] = end_sol
                path_results[path_index] = path_info
                completed += 1
                if verbose:
                    if completed % 10 == 0 or completed == n_paths:
                        elapsed = time.time() - start_time
                        paths_per_sec = completed / elapsed if elapsed > 0 else 0
                        eta = (
                            (n_paths - completed) / paths_per_sec
                            if paths_per_sec > 0 else 0
                        )
                        print(
                            f"Completed {completed}/{n_paths} paths "
                            f"({paths_per_sec:.2f} paths/sec, ETA: {eta:.1f}s)"
                        )
                    pbar.update(1)
    if verbose:
        pbar.close()
        success_count = sum(info['success'] for info in path_results if info)
        print(f"Path tracking complete: {success_count}/{n_paths} successful paths")

    return (
        [solution for solution in end_solutions if solution is not None],
        [info for info in path_results if info is not None],
    )
