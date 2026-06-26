"""
Parameter homotopy module for PyContinuum.

This module implements parameter homotopies, which track solutions
as parameters in the system change. Parameter homotopies are essential
for computing witness sets and numerical irreducible decomposition.
"""

from collections.abc import Mapping
from numbers import Integral, Real
from typing import List, Dict, Tuple, Any, Optional

import numpy as np

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem
from pycontinuum.utils import (
    evaluate_equation_scaled_jacobian_at_point,
    evaluate_scaled_system_at_point,
    solve_linear_system,
    _mapping_coordinate_for_variable,
    _polynomial_coefficient_scale,
    _scaled_euclidean_norm,
)


_EXPLICIT_PARAMETER_DUMMY_PREFIXES = (
    "_pc_parameter_dummy_",
    "_pc_monodromy_dummy_",
)


class ParameterHomotopy:
    """
    Represents a homotopy where parameters change with t.
    
    This is a homotopy of the form H(x, t) = (F(x), (1-t)L1(x) + tL2(x)) = 0
    where F is fixed and L1, L2 vary with the parameter t.
    """
    
    def __init__(self,
                 fixed_system: PolynomialSystem,  # System F(x)
                 start_param_system: PolynomialSystem,  # System L1(x)
                 end_param_system: PolynomialSystem,  # System L2(x)
                 variables: List[Variable],
                 square_fix: bool = True,
                 verbose: bool = False):
        """
        Initialize a parameter homotopy.
        
        Args:
            fixed_system: The fixed part of the system (F).
            start_param_system: The start parametric part (L1).
            end_param_system: The end parametric part (L2).
            variables: The variables in the system.
            square_fix: If True, add dummy equations to make the system square.
            verbose: Whether to print non-square system diagnostics.
        """
        if not isinstance(fixed_system, PolynomialSystem):
            raise TypeError("fixed_system must be a PolynomialSystem")
        if not isinstance(start_param_system, PolynomialSystem):
            raise TypeError("start_param_system must be a PolynomialSystem")
        if not isinstance(end_param_system, PolynomialSystem):
            raise TypeError("end_param_system must be a PolynomialSystem")
        variables = _normalize_parameter_variables(variables)
        square_fix = _validate_parameter_boolean_option("square_fix", square_fix)
        verbose = _validate_parameter_boolean_option("verbose", verbose)
        used_variables = _validate_variable_list_covers_systems(
            variables, fixed_system, start_param_system, end_param_system
        )

        self.fixed_system = fixed_system
        self.start_param_system = start_param_system
        self.end_param_system = end_param_system
        self.variables = variables
        
        # Combine all equations for easier indexing
        self.all_start_equations = fixed_system.equations + start_param_system.equations
        self.all_end_equations = fixed_system.equations + end_param_system.equations
        self.num_fixed = len(fixed_system.equations)
        self.num_param = len(start_param_system.equations)
        
        # Check that start and end parameter systems have the same number of equations
        if len(start_param_system.equations) != len(end_param_system.equations):
            raise ValueError("Start and end parameter systems must have the same number of equations")
            
        self.total_eqs = self.num_fixed + self.num_param
        _validate_parameter_variable_usage(
            variables,
            used_variables,
            self.total_eqs,
            square_fix,
        )
        
        # For non-square systems, we'll handle them specially
        self.is_square = self.total_eqs == len(variables)
        self.dummy_count = 0
        self.extended_variables = False
        
        if not self.is_square:
            if verbose:
                print(
                    "Warning: ParameterHomotopy created with "
                    f"{self.total_eqs} equations but {len(variables)} variables."
                )
            
            if square_fix and self.total_eqs < len(variables):
                # We'll add dummy "Lagrange multiplier" variables for the underdetermined case
                self.dummy_count = len(variables) - self.total_eqs
                if verbose:
                    print(
                        f"  Adding {self.dummy_count} dummy equations to make "
                        "the system square."
                    )
                self.extended_variables = True
                
                # The dummy equations will be of the form λ_i = 0
                # These will be handled in evaluate() and other methods
                self.total_eqs = len(variables)  # Now it's square
                self.is_square = True
                  
    def evaluate(self, point: np.ndarray, t: float) -> np.ndarray:
        """
        Evaluate the homotopy H(x, t) at a point x and parameter t.
        
        Args:
            point: The point x at which to evaluate.
            t: The parameter value t.
            
        Returns:
            The value of H(x, t).
        """
        point = _coerce_parameter_point(point, self.variables, "point")
        t = _validate_finite_parameter_float("t", t)

        # Handle the base (non-dummy) equations
        base_eqs = self.num_fixed + self.num_param
        vals = np.zeros(self.total_eqs, dtype=complex)
        
        # Evaluate fixed part F(x)
        fixed_vals = _evaluate_parameter_system_with_scaled_fallback(
            self.fixed_system,
            point,
            self.variables,
        )
        vals[:self.num_fixed] = fixed_vals
        
        # Evaluate parametric part (1-t)L1(x) + tL2(x)
        l1_vals, l2_vals = _evaluate_parameter_system_pair(
            self.start_param_system,
            self.end_param_system,
            point,
            self.variables,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            param_vals = (1 - t) * l1_vals + t * l2_vals
        vals[self.num_fixed:base_eqs] = param_vals
        
        # Handle dummy equations if we added them
        if self.extended_variables and self.dummy_count > 0:
            # The dummy equations are λ_i = 0, i.e., just extract the corresponding variables
            # We assume that the "dummy variables" are the last self.dummy_count variables
            dummy_start = len(self.variables) - self.dummy_count
            vals[base_eqs:] = point[dummy_start:]
        
        return vals
        
    def jacobian_x(self, point: np.ndarray, t: float) -> np.ndarray:
        """
        Evaluate the Jacobian of H(x, t) with respect to x.
        
        Args:
            point: The point x at which to evaluate.
            t: The parameter value t.
            
        Returns:
            The Jacobian matrix ∂H/∂x.
        """
        point = _coerce_parameter_point(point, self.variables, "point")
        t = _validate_finite_parameter_float("t", t)

        base_eqs = self.num_fixed + self.num_param
        jac = np.zeros((self.total_eqs, len(self.variables)), dtype=complex)
        
        # Jacobian of fixed part dF/dx (may be empty if no fixed equations)
        if self.num_fixed > 0:
            jac_F = _evaluate_parameter_jacobian_with_scaled_fallback(
                self.fixed_system,
                point,
                self.variables,
            )
            if jac_F.size:
                jac[:self.num_fixed, :] = jac_F
        
        # Jacobian of parametric part (1-t)dL1/dx + t*dL2/dx
        if self.num_param > 0:
            jac_L1, jac_L2 = _evaluate_parameter_jacobian_pair(
                self.start_param_system,
                self.end_param_system,
                point,
                self.variables,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                jac_param = (1 - t) * jac_L1 + t * jac_L2
            if jac_param.size:
                jac[self.num_fixed:base_eqs, :] = jac_param
        
        # Handle dummy equations if we added them
        if self.extended_variables and self.dummy_count > 0:
            # The Jacobian of λ_i = 0 is the identity matrix in the last dummy_count rows
            dummy_start = len(self.variables) - self.dummy_count
            for i in range(self.dummy_count):
                jac[base_eqs + i, dummy_start + i] = 1.0
        
        return jac
        
    def deriv_t(self, point: np.ndarray, t: float) -> np.ndarray:
        """
        Evaluate the derivative of H(x, t) with respect to t.
        
        Args:
            point: The point x at which to evaluate.
            t: The parameter value t.
            
        Returns:
            The derivative ∂H/∂t.
        """
        point = _coerce_parameter_point(point, self.variables, "point")
        t = _validate_finite_parameter_float("t", t)

        base_eqs = self.num_fixed + self.num_param
        deriv = np.zeros(self.total_eqs, dtype=complex)
        # Fixed part doesn't depend on t
        # deriv[:self.num_fixed] = 0
        
        # Parametric part derivative: d/dt[(1-t)L1(x) + tL2(x)] = -L1(x) + L2(x)
        l1_vals, l2_vals = _evaluate_parameter_system_pair(
            self.start_param_system,
            self.end_param_system,
            point,
            self.variables,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            param_deriv = -l1_vals + l2_vals
        deriv[self.num_fixed:base_eqs] = param_deriv
        
        # Dummy equations don't depend on t
        # deriv[base_eqs:] = 0
        
        return deriv

    def system_at(self, t: float) -> PolynomialSystem:
        """Return the algebraic homotopy system at parameter value ``t``.

        The returned system uses the same per-equation coefficient scaling as
        :meth:`evaluate`: fixed equations are scaled individually, and
        parameter equations interpolate the scaled start and end equations. If
        this homotopy was square-fixed with dummy variables, the dummy equations
        are appended so the materialized system has the same equation count as
        :meth:`evaluate`.
        """
        t = _validate_finite_parameter_float("t", t)
        equations = [
            _scaled_parameter_equation(equation)
            for equation in self.fixed_system.equations
        ]
        equations.extend(
            (1.0 - t) * _scaled_parameter_equation(start_equation)
            + t * _scaled_parameter_equation(end_equation)
            for start_equation, end_equation in zip(
                self.start_param_system.equations,
                self.end_param_system.equations,
            )
        )

        if self.extended_variables and self.dummy_count > 0:
            equations.extend(self.variables[-self.dummy_count:])

        return PolynomialSystem(equations)


def _scaled_parameter_equation(equation: Polynomial) -> Polynomial:
    return equation / _polynomial_coefficient_scale(equation)


def _evaluate_parameter_system_with_scaled_fallback(
    system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> np.ndarray:
    return evaluate_scaled_system_at_point(system, point, variables)


def _evaluate_parameter_system_pair(
    left_system: PolynomialSystem,
    right_system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        evaluate_scaled_system_at_point(left_system, point, variables),
        evaluate_scaled_system_at_point(right_system, point, variables),
    )


def _evaluate_parameter_jacobian_with_scaled_fallback(
    system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> np.ndarray:
    return evaluate_equation_scaled_jacobian_at_point(system, point, variables)


def _evaluate_parameter_jacobian_pair(
    left_system: PolynomialSystem,
    right_system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        evaluate_equation_scaled_jacobian_at_point(left_system, point, variables),
        evaluate_equation_scaled_jacobian_at_point(right_system, point, variables),
    )


def track_parameter_path(parameter_homotopy: ParameterHomotopy,
                         start_point: Any,
                         start_t: float = 0.0,
                         end_t: float = 1.0,
                         options: Dict = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Track a solution path for a parameter homotopy.
    
    This tracks a solution x(t) of H(x, t) = 0 from start_t to end_t.
    
    Args:
        parameter_homotopy: The parameter homotopy to follow.
        start_point: The starting solution x(start_t), supplied as a coordinate
            vector, mapping keyed by variables or variable names, or an object
            with a ``values`` mapping.
        start_t: The starting parameter value.
        end_t: The ending parameter value.
        options: Options for the tracking process.
        
    Returns:
        Tuple of (end_point, tracking_info).
    """
    if options is None:
        options = {}
    if not isinstance(parameter_homotopy, ParameterHomotopy):
        raise TypeError("parameter_homotopy must be a ParameterHomotopy")
    if not isinstance(options, dict):
        raise TypeError("options must be a dictionary")
    options = _validate_parameter_option_keys(options)
        
    # Default options for robustness
    tol = options.get('tol', 1e-8)
    min_step_size = options.get('min_step_size', 1e-6)
    max_step_size = options.get('max_step_size', 0.05)  # Smaller steps for parameter homotopy
    max_corrections = options.get('max_corrections', 10)
    max_steps = options.get('max_steps', 1000)
    max_predictor_norm = options.get('max_predictor_norm', float('inf'))
    normalize_tangent = _validate_parameter_boolean_option(
        "normalize_tangent", options.get('normalize_tangent', False)
    )
    predictor = _validate_parameter_predictor(options.get('predictor', 'euler'))
    verbose = _validate_parameter_boolean_option(
        "verbose", options.get('verbose', False)
    )
    store_paths = _validate_parameter_boolean_option(
        "store_paths", options.get('store_paths', False)
    )
    tracking_values = _validate_parameter_tracking_options(
        tol=tol,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        max_corrections=max_corrections,
        max_steps=max_steps,
        max_predictor_norm=max_predictor_norm,
        start_t=start_t,
        end_t=end_t,
    )
    tol = tracking_values["tol"]
    min_step_size = tracking_values["min_step_size"]
    max_step_size = tracking_values["max_step_size"]
    max_corrections = tracking_values["max_corrections"]
    max_steps = tracking_values["max_steps"]
    max_predictor_norm = tracking_values["max_predictor_norm"]
    start_t = tracking_values["start_t"]
    end_t = tracking_values["end_t"]
    
    t_current = start_t
    current_point = _coerce_parameter_point(
        start_point, parameter_homotopy.variables, "start_point"
    )
    start_residual = _parameter_residual_or_inf(
        parameter_homotopy,
        current_point,
        start_t,
    )
    start_residual_limit = 100.0 * tol
    if start_residual > start_residual_limit:
        raise ValueError(
            "start_point does not satisfy the parameter homotopy at start_t; "
            f"residual={start_residual:.3e} exceeds "
            f"{start_residual_limit:.3e}"
        )
    relaxed_correction_limit = max(
        100.0 * tol,
        1000.0 * float(np.finfo(float).eps),
    )
    direction = 1 if end_t > start_t else (-1 if end_t < start_t else 0)
    step_size = (
        max_step_size / 2
        if direction != 0
        else 0.0
    )  # Start with half max step size for caution when tracking.
    
    # Store path points
    path_points = [(t_current, current_point.copy())] if store_paths else []
    
    # Path data for return
    path_info = {
        'success': False,
        'steps': 0,
        'newton_iters': 0,
        't': t_current,
        'start_t': float(start_t),
        'end_t': float(end_t),
        'direction': direction,
        'tol': float(tol),
        'min_step_size': float(min_step_size),
        'max_step_size': float(max_step_size),
        'initial_step_size': float(step_size),
        'max_corrections': int(max_corrections),
        'max_steps': int(max_steps),
        'max_predictor_norm': float(max_predictor_norm),
        'normalize_tangent': bool(normalize_tangent),
        'store_paths': bool(store_paths),
        'step_reductions': 0,
        'newton_backtracks': 0,
        'min_newton_step_scale': 1.0,
        'max_observed_predictor_norm': 0.0,
        'max_predictor_correction_norm': 0.0,
        'predictor': predictor,
        'predictor_fallbacks': 0,
        'relaxed_correction_acceptances': 0,
        'relaxed_correction_residual_limit': float(relaxed_correction_limit),
        'failure_reason': None,
        'start_residual': float(start_residual),
        'start_residual_limit': float(start_residual_limit),
        'final_residual': float(start_residual),
        'final_point': current_point.copy(),
        'path_points': path_points
    }
    # Main tracking loop
    while direction * (end_t - t_current) > 1e-12:  # Stop when we reach end_t
        path_info['steps'] += 1
        
        # Check for too many steps
        if path_info['steps'] > max_steps:
            if verbose:
                print(f"Warning: Reached maximum number of steps ({max_steps})")
            _record_parameter_failure(
                path_info,
                parameter_homotopy,
                'max_steps_exceeded',
                current_point,
                t_current,
            )
            return current_point, path_info
        if not np.all(np.isfinite(current_point)):
            _record_parameter_failure(
                path_info,
                parameter_homotopy,
                'nonfinite_point',
                current_point,
                t_current,
            )
            return current_point, path_info

        # 1. Compute tangent dx/dt = -Hx^{-1} * Ht
        tangent = _parameter_tangent(
            parameter_homotopy,
            current_point,
            t_current,
            normalize_tangent,
        )

        if not np.all(np.isfinite(tangent)):
            _record_parameter_failure(
                path_info,
                parameter_homotopy,
                'nonfinite_tangent',
                current_point,
                t_current,
            )
            return current_point, path_info

        # 2. Determine step size, making sure we don't overshoot end_t
        trial_step_size = min(step_size, abs(end_t - t_current))

        def predict_at(target_t: float, dt: float) -> np.ndarray:
            with np.errstate(over="ignore", invalid="ignore"):
                euler_prediction = current_point + dt * tangent
            if predictor == "euler":
                return euler_prediction

            if predictor == "heun":
                predicted = _parameter_heun_prediction(
                    parameter_homotopy,
                    current_point,
                    t_current,
                    target_t,
                    tangent,
                    normalize_tangent,
                )
            else:
                predicted = _parameter_rk4_prediction(
                    parameter_homotopy,
                    current_point,
                    t_current,
                    target_t,
                    tangent,
                    normalize_tangent,
                )

            if predicted is None:
                path_info['predictor_fallbacks'] += 1
                return euler_prediction

            correction_norm = _scaled_euclidean_norm(predicted - euler_prediction)
            path_info['max_predictor_correction_norm'] = max(
                path_info['max_predictor_correction_norm'],
                correction_norm,
            )
            return predicted

        while True:
            dt = direction * trial_step_size
            t_target = t_current + dt
            predicted = predict_at(t_target, dt)
            if not np.all(np.isfinite(predicted)):
                path_info['max_observed_predictor_norm'] = float("inf")
                if trial_step_size <= min_step_size:
                    _record_parameter_failure(
                        path_info,
                        parameter_homotopy,
                        'nonfinite_predictor',
                        current_point,
                        t_current,
                        trial_point=predicted,
                        trial_t=t_target,
                        trial_residual=float("inf"),
                    )
                    return current_point, path_info
                trial_step_size = max(min_step_size, trial_step_size * 0.5)
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

        # 3. Predictor step (Euler predictor)
        step_size = trial_step_size

        # 4. Corrector steps (damped Newton's method)
        corrected, converged, newton_steps, correction_info = (
            _correct_parameter_prediction(
                parameter_homotopy,
                predicted,
                t_target,
                tol,
                max_corrections,
            )
        )
        residual = correction_info["final_residual"]
        path_info['newton_iters'] += newton_steps
        path_info['newton_backtracks'] += correction_info["backtracks"]
        path_info['min_newton_step_scale'] = min(
            path_info['min_newton_step_scale'],
            correction_info["min_step_scale"],
        )
        
        # 5. Adjust step size based on corrector performance
        if converged:
            # If Newton converged quickly, increase step size
            if newton_steps <= 3 and step_size < max_step_size:
                step_size = min(max_step_size, step_size * 1.5)
            # If Newton took many steps, decrease step size slightly
            elif newton_steps >= 7 and step_size > min_step_size * 2:
                step_size = max(min_step_size, step_size * 0.75)
                
            # Update current point and t
            current_point = corrected
            t_current = t_target
            
            # Store path point if requested
            if store_paths:
                path_points.append((t_current, current_point.copy()))
        else:
            # Newton failed to converge, reduce step size and try again
            step_size = max(min_step_size, step_size * 0.5)
            
            if step_size <= min_step_size:
                if verbose:
                    print(f"Warning: Failed to converge at t={t_target}, step size minimal.")
                # Last attempt with minimal step size
                # This could potentially be improved with adaptive precision
                if (
                    newton_steps > 0
                    and np.isfinite(residual)
                    and residual <= relaxed_correction_limit
                ):
                    # Use the last corrected point even though it didn't fully converge
                    path_info['relaxed_correction_acceptances'] += 1
                    current_point = corrected
                    t_current = t_target
                    if store_paths:
                        path_points.append((t_current, current_point.copy()))
                    continue
                else:
                    # Really failed
                    _record_parameter_failure(
                        path_info,
                        parameter_homotopy,
                        'newton_failed',
                        current_point,
                        t_current,
                        trial_point=corrected,
                        trial_t=t_target,
                        trial_residual=residual,
                    )
                    return current_point, path_info
                    
    # Reached end_t
    # Final residual check
    final_residual = _parameter_residual_or_inf(
        parameter_homotopy,
        current_point,
        end_t,
    )
    success = final_residual < tol * 10  # Slightly looser tolerance at end
    
    path_info['success'] = success
    path_info['t'] = t_current
    path_info['final_point'] = current_point.copy()
    path_info['final_residual'] = final_residual
    if not success:
        path_info['failure_reason'] = 'large_final_residual'
    
    return current_point, path_info


def _record_parameter_failure(
    path_info: Dict[str, Any],
    parameter_homotopy: ParameterHomotopy,
    reason: str,
    point: np.ndarray,
    t: float,
    *,
    trial_point: Optional[np.ndarray] = None,
    trial_t: Optional[float] = None,
    trial_residual: Optional[float] = None,
) -> None:
    path_info['failure_reason'] = reason
    path_info['t'] = t
    path_info['final_point'] = np.asarray(point, dtype=complex).copy()
    path_info['final_residual'] = _parameter_residual_or_inf(
        parameter_homotopy,
        path_info['final_point'],
        t,
    )
    if trial_point is not None:
        path_info['trial_point'] = np.asarray(trial_point, dtype=complex).copy()
    if trial_t is not None:
        path_info['trial_t'] = float(trial_t)
    if trial_residual is not None:
        path_info['trial_residual'] = float(trial_residual)


def _validate_variable_list_covers_systems(
    variables: List[Variable],
    *systems: PolynomialSystem,
) -> set:
    seen = set()
    duplicates = []
    for index, variable in enumerate(variables):
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

    system_variables = {
        variable
        for system in systems
        for variable in system.variables()
    }
    missing = sorted(
        variable.name
        for variable in system_variables
        if variable not in seen
    )
    if missing:
        raise ValueError(
            "Variable list is missing system variable(s): "
            + ", ".join(sorted(set(missing)))
        )
    return system_variables


def _validate_parameter_variable_usage(
    variables: List[Variable],
    used_variables: set,
    total_eqs: int,
    square_fix: bool,
) -> None:
    unused = [variable for variable in variables if variable not in used_variables]
    underdetermined = total_eqs < len(variables)
    if not underdetermined or not square_fix:
        if unused:
            raise ValueError(
                "Variable list contains variable(s) not used by parameter homotopy: "
                + ", ".join(_sorted_variable_names(unused))
            )
        return

    dummy_count = len(variables) - total_eqs
    dummy_variables = variables[-dummy_count:]
    if (
        len(unused) != dummy_count
        or set(unused) != set(dummy_variables)
    ):
        raise ValueError(
            "square_fix=True for an underdetermined parameter homotopy "
            f"requires the final {dummy_count} variable(s) to be unused "
            "explicit dummy variables"
        )

    invalid_dummy_names = [
        variable.name
        for variable in dummy_variables
        if not _is_explicit_parameter_dummy_variable(variable)
    ]
    if invalid_dummy_names:
        raise ValueError(
            "Dummy variable(s) must be explicit parameter dummy variables "
            "named with one of the prefixes "
            + ", ".join(_EXPLICIT_PARAMETER_DUMMY_PREFIXES)
            + ": "
            + ", ".join(sorted(invalid_dummy_names))
        )


def _sorted_variable_names(variables: List[Variable]) -> List[str]:
    return sorted(variable.name for variable in variables)


def _is_explicit_parameter_dummy_variable(variable: Variable) -> bool:
    return any(
        variable.name.startswith(prefix)
        for prefix in _EXPLICIT_PARAMETER_DUMMY_PREFIXES
    )


def _normalize_parameter_variables(variables: Any) -> List[Variable]:
    try:
        return list(variables)
    except TypeError as exc:
        raise TypeError(
            "variables must be an iterable of Variable objects"
        ) from exc


def _validate_parameter_boolean_option(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _validate_positive_finite_parameter_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return numeric_value


def _validate_finite_parameter_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def _validate_positive_or_infinite_parameter_float(
    name: str,
    value: Any,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if np.isnan(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be positive or infinity")
    return numeric_value


def _validate_positive_parameter_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _validate_parameter_tracking_options(
    *,
    tol: float,
    min_step_size: float,
    max_step_size: float,
    max_corrections: int,
    max_steps: int,
    max_predictor_norm: float,
    start_t: float,
    end_t: float,
) -> Dict[str, Any]:
    tol = _validate_positive_finite_parameter_float("tol", tol)
    min_step_size = _validate_positive_finite_parameter_float(
        "min_step_size", min_step_size
    )
    max_step_size = _validate_positive_finite_parameter_float(
        "max_step_size", max_step_size
    )
    max_corrections = _validate_positive_parameter_integer(
        "max_corrections", max_corrections
    )
    max_steps = _validate_positive_parameter_integer("max_steps", max_steps)
    max_predictor_norm = _validate_positive_or_infinite_parameter_float(
        "max_predictor_norm", max_predictor_norm
    )
    start_t = _validate_finite_parameter_float("start_t", start_t)
    end_t = _validate_finite_parameter_float("end_t", end_t)

    if min_step_size > max_step_size:
        raise ValueError("min_step_size cannot exceed max_step_size")
    return {
        "tol": tol,
        "min_step_size": min_step_size,
        "max_step_size": max_step_size,
        "max_corrections": max_corrections,
        "max_steps": max_steps,
        "max_predictor_norm": max_predictor_norm,
        "start_t": start_t,
        "end_t": end_t,
    }


def _validate_parameter_option_keys(options: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "tol",
        "min_step_size",
        "max_step_size",
        "max_corrections",
        "max_steps",
        "max_predictor_norm",
        "normalize_tangent",
        "predictor",
        "verbose",
        "store_paths",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(
            "Unknown parameter tracking option(s): " + ", ".join(unknown)
        )
    return options.copy()


def _validate_parameter_predictor(predictor: Any) -> str:
    if not isinstance(predictor, str):
        raise TypeError("predictor must be a string")
    normalized = predictor.lower()
    if normalized not in {"euler", "heun", "rk4"}:
        raise ValueError("predictor must be 'euler', 'heun', or 'rk4'")
    return normalized


def _parameter_heun_prediction(
    parameter_homotopy: ParameterHomotopy,
    point: np.ndarray,
    t_current: float,
    t_target: float,
    tangent: np.ndarray,
    normalize_tangent: bool,
) -> Optional[np.ndarray]:
    delta_t = t_target - t_current
    try:
        euler_prediction = point + delta_t * tangent
        target_tangent = _parameter_tangent(
            parameter_homotopy,
            euler_prediction,
            t_target,
            normalize_tangent,
        )
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return None
    if not np.all(np.isfinite(target_tangent)):
        return None

    with np.errstate(over="ignore", invalid="ignore"):
        prediction = point + 0.5 * delta_t * (tangent + target_tangent)
    if not np.all(np.isfinite(prediction)):
        return None
    return prediction


def _correct_parameter_prediction(
    parameter_homotopy: ParameterHomotopy,
    predicted: np.ndarray,
    t_target: float,
    tol: float,
    max_corrections: int,
) -> Tuple[np.ndarray, bool, int, Dict[str, Any]]:
    corrected = np.array(predicted, dtype=complex)
    metadata: Dict[str, Any] = {
        "backtracks": 0,
        "min_step_scale": 1.0,
        "final_residual": float("inf"),
    }

    for iteration in range(max_corrections):
        try:
            H_val = parameter_homotopy.evaluate(corrected, t_target)
        except (TypeError, ValueError):
            metadata["final_residual"] = float("inf")
            return corrected, False, iteration, metadata
        residual = _norm_or_inf(H_val)
        metadata["final_residual"] = residual
        if residual < tol:
            return corrected, True, iteration, metadata

        jac = parameter_homotopy.jacobian_x(corrected, t_target)
        delta = solve_linear_system(jac, -H_val)
        delta_norm = _scaled_euclidean_norm(delta)
        if not np.isfinite(delta_norm):
            return corrected, False, iteration + 1, metadata
        if delta_norm < tol:
            return corrected, False, iteration + 1, metadata

        (
            corrected,
            residual,
            accepted,
            step_scale,
            backtracks,
        ) = _parameter_damped_update(
            parameter_homotopy,
            corrected,
            t_target,
            delta,
            residual,
        )
        metadata["final_residual"] = residual
        metadata["backtracks"] += backtracks
        if accepted:
            metadata["min_step_scale"] = min(
                metadata["min_step_scale"],
                step_scale,
            )
        else:
            return corrected, False, iteration + 1, metadata

        if residual < tol:
            return corrected, True, iteration + 1, metadata

    return corrected, False, max_corrections, metadata


def _parameter_damped_update(
    parameter_homotopy: ParameterHomotopy,
    current: np.ndarray,
    t_target: float,
    delta: np.ndarray,
    current_residual: float,
    *,
    max_backtracks: int = 12,
) -> Tuple[np.ndarray, float, bool, float, int]:
    step_scale = 1.0
    for backtracks in range(max_backtracks + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = current + step_scale * delta
        if not np.all(np.isfinite(candidate)):
            candidate_residual = float("inf")
        else:
            candidate_residual = _parameter_residual_or_inf(
                parameter_homotopy,
                candidate,
                t_target,
            )
        if candidate_residual < current_residual:
            return candidate, candidate_residual, True, step_scale, backtracks
        step_scale *= 0.5
    return current, current_residual, False, 0.0, max_backtracks + 1


def _parameter_rk4_prediction(
    parameter_homotopy: ParameterHomotopy,
    point: np.ndarray,
    t_current: float,
    t_target: float,
    tangent: np.ndarray,
    normalize_tangent: bool,
) -> Optional[np.ndarray]:
    delta_t = t_target - t_current
    half_t = t_current + 0.5 * delta_t

    k1 = tangent
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            k2_point = point + 0.5 * delta_t * k1
        k2 = _parameter_tangent(
            parameter_homotopy,
            k2_point,
            half_t,
            normalize_tangent,
        )
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return None
    if not np.all(np.isfinite(k2)):
        return None

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            k3_point = point + 0.5 * delta_t * k2
        k3 = _parameter_tangent(
            parameter_homotopy,
            k3_point,
            half_t,
            normalize_tangent,
        )
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return None
    if not np.all(np.isfinite(k3)):
        return None

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            k4_point = point + delta_t * k3
        k4 = _parameter_tangent(
            parameter_homotopy,
            k4_point,
            t_target,
            normalize_tangent,
        )
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return None
    if not np.all(np.isfinite(k4)):
        return None

    with np.errstate(over="ignore", invalid="ignore"):
        prediction = point + (delta_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if not np.all(np.isfinite(prediction)):
        return None
    return prediction


def _parameter_tangent(
    parameter_homotopy: ParameterHomotopy,
    point: np.ndarray,
    t: float,
    normalize_tangent: bool,
) -> np.ndarray:
    jac_x = parameter_homotopy.jacobian_x(point, t)
    deriv_t = parameter_homotopy.deriv_t(point, t)
    tangent = solve_linear_system(jac_x, -deriv_t)

    # Predictors need dx/dt, not just the unit direction. Retain normalization
    # as an opt-in compatibility mode for older parameter-tracking runs.
    tangent_norm = _scaled_euclidean_norm(tangent)
    if normalize_tangent and tangent_norm > 1e-10:
        tangent = tangent / tangent_norm
    return tangent


def _coerce_parameter_point(
    point: Any,
    variables_or_dimension: Any,
    label: str,
) -> np.ndarray:
    if isinstance(variables_or_dimension, Integral):
        variables = None
        expected_dimension = int(variables_or_dimension)
    else:
        variables = _normalize_parameter_variables(variables_or_dimension)
        expected_dimension = len(variables)

    values = None
    if isinstance(point, Mapping):
        values = point
    elif isinstance(getattr(point, "values", None), Mapping):
        values = point.values

    if values is not None and variables is not None:
        array = _coerce_parameter_point_mapping(values, variables, label)
    else:
        try:
            array = np.asarray(point, dtype=complex)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{label} must be a numeric one-dimensional point"
            ) from exc
    if array.ndim != 1 or array.size != expected_dimension:
        raise ValueError(
            f"{label} must have {expected_dimension} coordinate(s); "
            f"got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains nonfinite values")
    return array


def _coerce_parameter_point_mapping(
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


def _norm_or_inf(values: np.ndarray) -> float:
    return _scaled_euclidean_norm(values)


def _parameter_residual_or_inf(
    parameter_homotopy: ParameterHomotopy,
    point: np.ndarray,
    t: float,
) -> float:
    try:
        return _norm_or_inf(parameter_homotopy.evaluate(point, t))
    except (TypeError, ValueError):
        return float("inf")
