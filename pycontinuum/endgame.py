"""
Endgame module for PyContinuum.

This module implements endgame techniques for handling singular solutions
in homotopy continuation, primarily using the Cauchy endgame algorithm.
"""

from collections.abc import Mapping
import numpy as np
import cmath
import time
from numbers import Integral, Real
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from enum import Enum

from pycontinuum.polynomial import Variable, PolynomialSystem
from pycontinuum.start_systems import _coerce_rng, _rng_standard_normal
# Import utility functions
from pycontinuum.utils import (
    evaluate_system_at_point,
    evaluate_jacobian_at_point,
    evaluate_scaled_system_at_point,
    evaluate_equation_scaled_jacobian_at_point,
    newton_corrector,
    solve_linear_system,
    _mapping_coordinate_for_variable,
    _scaled_euclidean_norm,
)

_ENDGAME_OPTION_KEYS = {
    "L",
    "K",
    "abstol",
    "gamma",
    "geometric_series_factor",
    "loopclosed_tolerance",
    "max_iterations",
    "max_winding_number",
    "newton_max_iters",
    "random_state",
    "samples_per_loop",
}


def _validate_endgame_options(
    options: Optional[Dict[str, Any]],
    *,
    name: str = "options",
) -> Optional[Dict[str, Any]]:
    if options is None:
        return None
    if not isinstance(options, dict):
        raise TypeError(f"{name} must be a dictionary")

    unknown = sorted(set(options) - _ENDGAME_OPTION_KEYS)
    if unknown:
        raise ValueError(
            "Unknown endgame option(s): " + ", ".join(unknown)
        )
    validated = options.copy()
    for option in ("samples_per_loop", "max_iterations", "max_winding_number"):
        if option in validated:
            validated[option] = _validate_positive_endgame_integer(
                option,
                validated[option],
            )
    if "newton_max_iters" in validated:
        validated["newton_max_iters"] = _validate_nonnegative_endgame_integer(
            "newton_max_iters",
            validated["newton_max_iters"],
        )

    for option in ("abstol", "loopclosed_tolerance"):
        if option in validated:
            validated[option] = _validate_positive_finite_endgame_float(
                option,
                validated[option],
            )
    for option in ("L", "K", "geometric_series_factor"):
        if option in validated:
            validated[option] = _validate_unit_interval_endgame_float(
                option,
                validated[option],
            )
    if "gamma" in validated:
        validated["gamma"] = _validate_endgame_gamma(validated["gamma"])
    return validated


def _validate_endgame_gamma(gamma: Any) -> complex:
    if isinstance(gamma, bool):
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


def _validate_positive_endgame_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _validate_nonnegative_endgame_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def _validate_positive_finite_endgame_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return numeric_value


def _validate_unit_interval_endgame_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or not 0 < numeric_value < 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return numeric_value


def _normalize_endgame_variables(variables: Any) -> List[Variable]:
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


def _coerce_endgame_point(
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
        array = _coerce_endgame_point_mapping(values, variables, label)
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


def _coerce_endgame_point_mapping(
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


class EndgameStatus(Enum):
    """Status of an endgame procedure."""
    NOT_STARTED = 0
    STARTED = 1
    SUCCESSFUL = 2
    FAILED = 3


class CauchyEndgame:
    """
    Cauchy endgame implementation for handling singular solutions.
    
    This tracks the solution around a circle in the complex plane and uses
    the Cauchy integral formula to predict the endpoint at t=0.
    """
    
    def __init__(self, 
                 samples_per_loop: int = 8,
                 loopclosed_tolerance: float = 1e-8,
                 max_winding_number: int = 16,
                 L: float = 0.75,  # Parameter for first heuristic
                 K: float = 0.5):  # Parameter for second heuristic
        """Initialize a Cauchy endgame.
        
        Args:
            samples_per_loop: Number of sample points to use per loop
            loopclosed_tolerance: Tolerance for determining if a loop is closed
            max_winding_number: Maximum allowed winding number
            L: Parameter for the first heuristic
            K: Parameter for the second heuristic
        """
        self.samples_per_loop = _validate_positive_endgame_integer(
            "samples_per_loop",
            samples_per_loop,
        )
        self.loopclosed_tolerance = _validate_positive_finite_endgame_float(
            "loopclosed_tolerance",
            loopclosed_tolerance,
        )
        self.max_winding_number = _validate_positive_endgame_integer(
            "max_winding_number",
            max_winding_number,
        )
        self.L = _validate_unit_interval_endgame_float("L", L)
        self.K = _validate_unit_interval_endgame_float("K", K)


class EndgamerOptions:
    """Configuration options for the endgame procedure."""
    
    def __init__(self,
                 geometric_series_factor: float = 0.5,
                 abstol: float = 1e-10,
                 max_iterations: int = 64,
                 max_winding_number: int = 16):
        """Initialize endgamer options.
        
        Args:
            geometric_series_factor: Factor for geometric series (typically 0.5)
            abstol: Absolute tolerance for convergence
            max_iterations: Maximum geometric-radius endgame iterations
            max_winding_number: Maximum allowed winding number
        """
        self.geometric_series_factor = _validate_unit_interval_endgame_float(
            "geometric_series_factor",
            geometric_series_factor,
        )
        self.abstol = _validate_positive_finite_endgame_float("abstol", abstol)
        self.max_iterations = _validate_positive_endgame_integer(
            "max_iterations",
            max_iterations,
        )
        self.max_winding_number = _validate_positive_endgame_integer(
            "max_winding_number",
            max_winding_number,
        )


class Endgamer:
    """
    Main class for managing the endgame process.
    
    This class coordinates the endgame process, including when to start
    the endgame, applying the algorithm, and checking for convergence.
    """
    
    def __init__(self,
                 start_system: PolynomialSystem,
                 target_system: PolynomialSystem,
                 variables: List[Variable],
                 alg: CauchyEndgame = None,
                 options: EndgamerOptions = None,
                 gamma: complex = 0.6+0.8j,
                 random_state: Any = None):
        """Initialize an endgamer.
        
        Args:
            start_system: Start system g(x)
            target_system: Target system f(x)
            variables: System variables
            alg: Endgame algorithm (default: CauchyEndgame with default params)
            options: Endgame options (default: EndgamerOptions with default params)
            gamma: Random complex number for the homotopy
            random_state: Optional seed or NumPy random generator.
        """
        self.start_system = start_system
        self.target_system = target_system
        self.variables = _normalize_endgame_variables(variables)
        self.alg = alg if alg is not None else CauchyEndgame()
        self.options = options if options is not None else EndgamerOptions()
        self.gamma = _validate_endgame_gamma(gamma)
        self.rng = _coerce_rng(random_state)
        
        # State
        self.R = 0.0  # Current radius
        self.iter = 0  # Iteration counter
        self.windingnumber = 1  # Estimated winding number
        self.status = EndgameStatus.NOT_STARTED
        self.failurecode = "default_failure_code"
        
        # Data
        self.current_point = None
        self.xs = []  # History of points during approach
        self.predictions = []  # Predictions from endgame
        self.samples = []  # Samples from circle
    
    def setup(self, point: Any, radius: float):
        """Set up the endgamer to start at the given point and radius.
        
        Args:
            point: Starting point, supplied as a coordinate vector, mapping
                keyed by variables or variable names, or an object with a
                ``values`` mapping.
            radius: Starting radius (t value)
        """
        point = _coerce_endgame_point(point, self.variables, "point")
        self.R = radius
        self.iter = 0
        self.windingnumber = 1
        self.status = EndgameStatus.NOT_STARTED
        self.failurecode = "default_failure_code"
        self.current_point = point.copy()
        self.xs = [point.copy()]
        self.predictions = []
        self.samples = []
    
    def first_heuristic(self, x_R, x_λR, x_λ2R, x_λ3R) -> bool:
        """
        First heuristic to determine when to start the endgame.
        
        This checks whether two consecutive approximations of the first 
        nonvanishing monomial of the Puiseux series expansion of x(t) at 0 agree.
        
        Args:
            x_R: Point at radius R
            x_λR: Point at radius λR
            x_λ2R: Point at radius λ²R
            x_λ3R: Point at radius λ³R
            
        Returns:
            True if the endgame should start, False otherwise
        """
        # Failsafe mechanism for very small radius
        if self.R < 1e-8:
            return True
        
        λ = self.options.geometric_series_factor
        L = self.alg.L
        
        # Generate a random complex direction
        v = (
            _rng_standard_normal(
                self.rng,
                len(x_R),
                context="endgame first heuristic direction real",
            )
            + 1j * _rng_standard_normal(
                self.rng,
                len(x_R),
                context="endgame first heuristic direction imaginary",
            )
        )
        v_norm = _scaled_euclidean_norm(v)
        if not np.isfinite(v_norm) or v_norm == 0:
            return False
        v = v / v_norm
        
        # Define the approximation function
        def g(x1, x2, x3):
            numer = np.abs(np.dot(v, x2) - np.dot(v, x3))**2
            denom = np.abs(np.dot(v, x1) - np.dot(v, x2))**2
            if numer <= 0 or denom <= 0:
                return float("nan")
            return np.log(numer / denom) / np.log(λ)
        
        g_R = g(x_R, x_λR, x_λ2R)
        g_λR = g(x_λR, x_λ2R, x_λ3R)
        if not np.isfinite(g_R) or not np.isfinite(g_λR) or g_λR == 0:
            return False
        
        # If g_R <= 0, the approximations disagree
        if g_R <= 0:
            return False
        
        # Check if the ratio is in the acceptable range
        return L < g_R / g_λR < 1 / L
    
    def second_heuristic(self, samples) -> bool:
        """
        Second heuristic to determine if the endgame should start.
        
        This ensures that the values around the loop do not differ radically.
        
        Args:
            samples: Sample points collected from a loop
            
        Returns:
            True if the endgame should start, False otherwise
        """
        # Failsafe for very small radius
        if self.R < 1e-14:
            return True
        
        K = self.alg.K
        β = self.options.abstol
        
        # Compute norms of all sample points
        norms = [_scaled_euclidean_norm(s) for s in samples]
        m = min(norms)
        M = max(norms)
        if not np.isfinite(m) or not np.isfinite(M):
            return False
        if M == 0:
            return True
        
        # Check if the variation is small or if the ratio is large enough
        return (M - m < β) or (m / M > K)
    
    def track_loop(self) -> Tuple[List[np.ndarray], str]:
        """
        Track the solution around a loop in the complex plane.
        
        This is the core of the Cauchy endgame, tracking the path around a
        circle of radius R centered at the origin in the t-plane.
        
        Returns:
            Tuple of (samples, status_code)
        """
        samples_per_loop = self.alg.samples_per_loop
        loopclosed_tolerance = self.alg.loopclosed_tolerance
        max_winding_number = self.options.max_winding_number
        
        # Initialize samples list
        samples = [self.current_point.copy()]
        
        # Generate points on the unit circle
        angles = np.linspace(0, 2*np.pi, samples_per_loop, endpoint=False)
        points_on_circle = [self.R * np.exp(1j * angle) for angle in angles]
        
        # Set starting point
        current_sample = self.current_point.copy()
        
        # Track around the circle
        winding_number = 1
        
        for loop_idx in range(max_winding_number):
            for i in range(samples_per_loop):
                # Get start and end points for this segment
                start_t = points_on_circle[i]
                end_t = points_on_circle[(i+1) % samples_per_loop]
                
                # Create a simple homotopy that keeps target and start systems fixed,
                # but varies the point on the circle
                def track_point_on_circle(start_point, start_t, end_t):
                    # Create interpolation parameter for this segment
                    steps = 10  # Fixed number of steps for simplicity
                    path_t = np.linspace(0, 1, steps)
                    current = start_point.copy()
                    
                    for s in path_t[1:]:  # Skip the first point (s=0)
                        # Interpolate t between start_t and end_t
                        t_current = (1-s) * start_t + s * end_t
                        
                        # Define the homotopy function H(x, t)
                        def H_at_point(point):
                            return _homotopy_values(
                                self.start_system,
                                self.target_system,
                                point,
                                abs(t_current),
                                self.variables,
                                self.gamma,
                            )
                        
                        # Use a simple Newton step
                        # In a more robust implementation, this would be a full Newton corrector
                        jac = _homotopy_jacobian(
                            self.start_system,
                            self.target_system,
                            current,
                            abs(t_current),
                            self.variables,
                            self.gamma,
                        )
                        
                        H_val = H_at_point(current)
                        if not np.all(np.isfinite(H_val)) or not np.all(np.isfinite(jac)):
                            return _failed_point(current)
                        
                        delta = solve_linear_system(jac, -H_val)
                        if not np.all(np.isfinite(delta)):
                            return _failed_point(current)
                        
                        current = current + delta
                        if not np.all(np.isfinite(current)):
                            return _failed_point(current)
                        
                        # Additional Newton corrections for accuracy
                        for _ in range(2):
                            H_val = H_at_point(current)
                            if not np.all(np.isfinite(H_val)):
                                return _failed_point(current)
                            if _scaled_euclidean_norm(H_val) < 1e-10:
                                break
                                
                            jac = _homotopy_jacobian(
                                self.start_system,
                                self.target_system,
                                current,
                                abs(t_current),
                                self.variables,
                                self.gamma,
                            )
                            if not np.all(np.isfinite(jac)):
                                return _failed_point(current)
                            
                            delta = solve_linear_system(jac, -H_val)
                            if not np.all(np.isfinite(delta)):
                                return _failed_point(current)
                            
                            current = current + delta
                            if not np.all(np.isfinite(current)):
                                return _failed_point(current)
                    
                    return current
                
                # Track around this segment
                next_sample = track_point_on_circle(current_sample, start_t, end_t)
                
                # Check if tracking failed
                if not np.all(np.isfinite(next_sample)):
                    return samples, "tracker_failed"
                
                # Add to samples
                samples.append(next_sample)
                current_sample = next_sample
                
                # Apply second heuristic after completing the first loop
                if loop_idx == 0 and i == samples_per_loop - 1 and self.status == EndgameStatus.NOT_STARTED:
                    if not self.second_heuristic(samples[:samples_per_loop+1]):
                        return samples, "heuristic_failed"
            
            # Check if the loop is closed (compare with first point)
            if (
                _scaled_euclidean_norm(samples[0] - current_sample)
                < loopclosed_tolerance
            ):
                break
            
            # Loop not closed, increment winding number
            winding_number += 1
            
            # Check if winding number is too high
            if winding_number > max_winding_number:
                return samples, "winding_number_too_high"
        
        # Successfully completed tracking around the loop
        self.windingnumber = winding_number
        return samples, "success"
    
    def predict_endpoint(self, samples: List[np.ndarray]) -> np.ndarray:
        """
        Predict the endpoint using the Cauchy integral formula.
        
        With uniformly spaced samples around a circle, the Cauchy integral
        formula simplifies to the mean of the samples.
        
        Args:
            samples: Sample points collected from the loop
            
        Returns:
            Predicted endpoint
        """
        # Since we're sampling uniformly around a circle, the Cauchy integral
        # is approximated by the mean of the samples
        return np.mean(samples, axis=0)
    
    def check_convergence(self) -> bool:
        """
        Check if predictions have converged.
        
        Returns:
            True if converged, False otherwise
        """
        if len(self.predictions) < 2:
            return False
        
        p, pprev = self.predictions[-1], self.predictions[-2]
        
        # Compute distance between predictions
        dist = _scaled_euclidean_norm(p - pprev)
        
        # Check if converged
        if dist < self.options.abstol:
            if len(self.predictions) > 2:
                pprev2 = self.predictions[-3]
                
                # Check if improvement is significant
                prev_dist = _scaled_euclidean_norm(pprev - pprev2)
                if prev_dist / dist < 2 and dist < self.options.abstol:
                    return True
            else:
                return True
                
        return False
    
    def move_forward(self) -> bool:
        """
        Move forward to the next radius in the geometric series.
        
        Updates the current point and radius.
        """
        λ = self.options.geometric_series_factor
        next_radius = λ * self.R
        
        # Create a simple system for tracking to the next radius
        def H_at_point(point, t):
            return _homotopy_values(
                self.start_system,
                self.target_system,
                point,
                t,
                self.variables,
                self.gamma,
            )
        
        # Track from current radius to next radius
        current = self.current_point.copy()
        
        # Simple straight-line tracking from R to λR
        steps = 10  # Fixed number of steps for simplicity
        path_t = np.linspace(self.R, next_radius, steps)
        
        for t in path_t[1:]:  # Skip the first point
            H_val = H_at_point(current, t)
            
            jac = _homotopy_jacobian(
                self.start_system,
                self.target_system,
                current,
                t,
                self.variables,
                self.gamma,
            )
            if not np.all(np.isfinite(H_val)) or not np.all(np.isfinite(jac)):
                return False
            
            delta = solve_linear_system(jac, -H_val)
            if not np.all(np.isfinite(delta)):
                return False
            
            current = current + delta
            if not np.all(np.isfinite(current)):
                return False
            
            # Additional Newton corrections
            for _ in range(2):
                H_val = H_at_point(current, t)
                if not np.all(np.isfinite(H_val)):
                    return False
                if _scaled_euclidean_norm(H_val) < 1e-10:
                    break
                    
                jac = _homotopy_jacobian(
                    self.start_system,
                    self.target_system,
                    current,
                    t,
                    self.variables,
                    self.gamma,
                )
                if not np.all(np.isfinite(jac)):
                    return False
                
                delta = solve_linear_system(jac, -H_val)
                if not np.all(np.isfinite(delta)):
                    return False
                
                current = current + delta
                if not np.all(np.isfinite(current)):
                    return False
        
        # Update state
        self.xs.append(current)
        self.current_point = current
        self.R = next_radius
        self.iter += 1
        return True
    
    def run(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run the endgame until convergence or failure.
        
        Returns:
            Tuple of (final_point, result_info)
        """
        result_info = {
            'success': False,
            'singular': True,  # Endgame is typically used for singular points
            'steps': 0,
            'max_iterations': self.options.max_iterations,
            'winding_number': self.windingnumber,
            'predictions': [],
            'status': self.status.name.lower(),
            'failure_code': None,
            'final_residual': float("inf"),
            'final_point': (
                self.current_point.copy() if self.current_point is not None else None
            ),
        }
        
        # Main loop
        while self.status != EndgameStatus.SUCCESSFUL and self.status != EndgameStatus.FAILED:
            result_info['steps'] += 1
            
            # If radius is too small, stop
            if self.R <= 1e-14:
                self.status = EndgameStatus.FAILED
                self.failurecode = "radius_too_small"
                break
            if result_info['steps'] > self.options.max_iterations:
                self.status = EndgameStatus.FAILED
                self.failurecode = "max_iterations_exceeded"
                break
            
            # Check if we have enough points for first heuristic
            if len(self.xs) >= 4 and self.status == EndgameStatus.NOT_STARTED:
                # Apply first heuristic
                if self.first_heuristic(
                    self.xs[-4], self.xs[-3], self.xs[-2], self.xs[-1]
                ):
                    self.status = EndgameStatus.STARTED
            
            # If endgame started or we have enough points, track around loop
            if self.status == EndgameStatus.STARTED or len(self.xs) >= 10:
                # Track solution around circle
                samples, code = self.track_loop()
                self.samples = samples
                
                if code == "success":
                    # Predict endpoint using Cauchy integral
                    prediction = self.predict_endpoint(samples)
                    self.predictions.append(prediction)
                    result_info['predictions'].append(prediction.tolist())
                    
                    # Check for convergence
                    if self.check_convergence():
                        self.status = EndgameStatus.SUCCESSFUL
                        break
                elif code == "tracker_failed":
                    self.status = EndgameStatus.FAILED
                    self.failurecode = code
                    break
                elif code == "winding_number_too_high":
                    # Try again with more samples per loop
                    self.alg.samples_per_loop *= 3
                    samples, code = self.track_loop()
                    
                    if code == "success":
                        prediction = self.predict_endpoint(samples)
                        self.predictions.append(prediction)
                        result_info['predictions'].append(prediction.tolist())
                    else:
                        self.status = EndgameStatus.FAILED
                        self.failurecode = code
                        break
            
            # Move forward to next radius
            if not self.move_forward():
                self.status = EndgameStatus.FAILED
                self.failurecode = "move_forward_failed"
                break
        
        # If we have predictions but they didn't meet convergence criteria,
        # use the last one anyway if it's good enough
        if self.predictions and self.status != EndgameStatus.SUCCESSFUL:
            last_prediction = self.predictions[-1]
            
            # Check residual
            residual = _system_residual_norm(
                self.target_system, last_prediction, self.variables
            )
            
            if residual < _endgame_success_residual_limit(self.options.abstol):
                self.status = EndgameStatus.SUCCESSFUL
        
        # Set result information
        # Final point is either the last prediction or the current point
        final_point = self.predictions[-1] if self.predictions else self.current_point
        final_residual = _system_residual_norm(
            self.target_system, final_point, self.variables
        )
        residual_limit = _endgame_success_residual_limit(self.options.abstol)
        success = bool(
            self.status == EndgameStatus.SUCCESSFUL
            and np.isfinite(final_residual)
            and final_residual < residual_limit
        )
        if self.status == EndgameStatus.SUCCESSFUL and not success:
            self.status = EndgameStatus.FAILED
            self.failurecode = "large_final_residual"

        result_info['success'] = success
        result_info['winding_number'] = self.windingnumber
        result_info['status'] = self.status.name.lower()
        result_info['failure_code'] = (
            None if result_info['success'] else self.failurecode
        )
        result_info['final_point'] = final_point.copy()
        result_info['final_residual'] = final_residual
        
        return final_point, result_info


def run_cauchy_endgame(start_system: PolynomialSystem,
                      target_system: PolynomialSystem,
                      point: Any,
                      t: float,
                      variables: List[Variable],
                      options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run the Cauchy endgame from a given point and time value.
    
    This is a convenience function for using the Cauchy endgame.
    
    Args:
        start_system: Start system g(x)
        target_system: Target system f(x)
        point: Current point on the path, supplied as a coordinate vector,
            mapping keyed by variables or variable names, or an object with a
            ``values`` mapping.
        t: Current t value
        variables: System variables
        options: Optional configuration parameters
            Supported keys include ``random_state`` for reproducible endgame
            heuristic projections and ``max_iterations`` to bound Cauchy
            endgame radius updates.
        
    Returns:
        Tuple of (end_point, result_info)
    """
    opts = _validate_endgame_options(options) or {}
    variables = _normalize_endgame_variables(variables)
    point = _coerce_endgame_point(point, variables, "point")
    
    # If t is extremely small, just use Newton's method directly
    if t < 1e-6:
        # At this point, we're so close to the solution, just polish with Newton
        tol = opts.get('abstol', 1e-10)
        corrected, success, iters = newton_corrector(
            target_system, point, variables, 
            max_iters=opts.get('newton_max_iters', 50),
            tol=tol
        )
        corrector_success = bool(success)
        final_residual, _ = _system_residual_norm_with_source(
            target_system,
            corrected,
            variables,
        )
        success = bool(
            np.isfinite(final_residual)
            and final_residual < _endgame_success_residual_limit(tol)
        )
        failure_code = None
        if not success:
            failure_code = (
                "large_final_residual"
                if corrector_success
                else "newton_failed"
            )
        
        result_info = {
            'success': success,
            'singular': True,
            'steps': iters,
            'winding_number': 1,
            'predictions': [corrected.tolist()],
            'status': 'successful' if success else 'failed',
            'failure_code': failure_code,
            'final_residual': final_residual,
            'final_point': np.array(corrected, dtype=complex),
        }
        
        return corrected, result_info
    
    # Create endgame algorithm
    alg = CauchyEndgame(
        samples_per_loop=opts.get('samples_per_loop', 8),
        loopclosed_tolerance=opts.get('loopclosed_tolerance', 1e-8),
        max_winding_number=opts.get('max_winding_number', 16),
        L=opts.get('L', 0.75),
        K=opts.get('K', 0.5)
    )
    
    # Create endgamer options
    endgamer_opts = EndgamerOptions(
        geometric_series_factor=opts.get('geometric_series_factor', 0.5),
        abstol=opts.get('abstol', 1e-10),
        max_iterations=opts.get('max_iterations', 64),
        max_winding_number=opts.get('max_winding_number', 16)
    )
    
    # Create the endgamer
    endgamer = Endgamer(
        start_system=start_system,
        target_system=target_system,
        variables=variables,
        alg=alg,
        options=endgamer_opts,
        gamma=opts.get('gamma', 0.6+0.8j),
        random_state=opts.get('random_state')
    )
    
    # Set up the endgamer
    endgamer.setup(point, t)
    
    # Run the endgame
    return endgamer.run()


def _failed_point(point: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=complex)
    return np.full(point.shape, np.nan + 0j, dtype=complex)


def _system_values_with_scaled_fallback(
    system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> np.ndarray:
    values = evaluate_system_at_point(system, point, variables)
    if np.all(np.isfinite(values)):
        return values
    return evaluate_scaled_system_at_point(system, point, variables)


def _system_value_pair_with_scaled_fallback(
    start_system: PolynomialSystem,
    target_system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> Tuple[np.ndarray, np.ndarray]:
    target_values = evaluate_system_at_point(target_system, point, variables)
    start_values = evaluate_system_at_point(start_system, point, variables)
    if np.all(np.isfinite(target_values)) and np.all(np.isfinite(start_values)):
        return target_values, start_values
    return (
        evaluate_scaled_system_at_point(target_system, point, variables),
        evaluate_scaled_system_at_point(start_system, point, variables),
    )


def _jacobian_pair_with_scaled_fallback(
    start_system: PolynomialSystem,
    target_system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> Tuple[np.ndarray, np.ndarray]:
    target_jacobian = evaluate_jacobian_at_point(target_system, point, variables)
    start_jacobian = evaluate_jacobian_at_point(start_system, point, variables)
    if (
        np.all(np.isfinite(target_jacobian))
        and np.all(np.isfinite(start_jacobian))
    ):
        return target_jacobian, start_jacobian
    return (
        evaluate_equation_scaled_jacobian_at_point(target_system, point, variables),
        evaluate_equation_scaled_jacobian_at_point(start_system, point, variables),
    )


def _homotopy_values(
    start_system: PolynomialSystem,
    target_system: PolynomialSystem,
    point: np.ndarray,
    t: float,
    variables: List[Variable],
    gamma: complex,
) -> np.ndarray:
    target_values, start_values = _system_value_pair_with_scaled_fallback(
        start_system,
        target_system,
        point,
        variables,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        return (1 - t) * target_values + t * gamma * start_values


def _homotopy_jacobian(
    start_system: PolynomialSystem,
    target_system: PolynomialSystem,
    point: np.ndarray,
    t: float,
    variables: List[Variable],
    gamma: complex,
) -> np.ndarray:
    target_jacobian, start_jacobian = _jacobian_pair_with_scaled_fallback(
        start_system,
        target_system,
        point,
        variables,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        return (1 - t) * target_jacobian + t * gamma * start_jacobian


def _endgame_success_residual_limit(abstol: float) -> float:
    return 10.0 * float(abstol)


def _system_residual_norm(
    system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> float:
    residual, _ = _system_residual_norm_with_source(system, point, variables)
    return residual


def _system_residual_norm_with_source(
    system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> Tuple[float, bool]:
    point = np.asarray(point, dtype=complex)
    if not np.all(np.isfinite(point)):
        return float("inf"), False
    raw_residuals = evaluate_system_at_point(system, point, variables)
    raw_is_finite = np.all(np.isfinite(raw_residuals))
    raw_residual = (
        _scaled_euclidean_norm(raw_residuals)
        if raw_is_finite else float("inf")
    )
    scaled_residuals = evaluate_scaled_system_at_point(system, point, variables)
    if not np.all(np.isfinite(scaled_residuals)):
        if raw_is_finite:
            return float("inf"), True
        return float("inf"), True
    scaled_residual = _scaled_euclidean_norm(scaled_residuals)
    if raw_is_finite:
        return max(raw_residual, scaled_residual), scaled_residual > raw_residual
    return scaled_residual, True
