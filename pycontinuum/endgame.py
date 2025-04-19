"""
Endgame strategies for handling singular solutions in PyContinuum.

This module implements endgame algorithms to handle the numerical challenges
that arise when solution paths approach singular points (where the Jacobian becomes
nearly singular).
"""

import numpy as np
import cmath
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

from pycontinuum.polynomial import Variable, PolynomialSystem


class EndgameResult:
    """Result of applying an endgame strategy."""
    
    def __init__(self, 
                 solution: np.ndarray,
                 success: bool = False,
                 winding_number: int = 1,
                 residual: float = float('inf'),
                 iterations: int = 0,
                 predictions: List[np.ndarray] = None,
                 ill_conditioned: bool = False):
        """Initialize an endgame result.
        
        Args:
            solution: The predicted solution at t=0
            success: Whether the endgame strategy successfully predicted a solution
            winding_number: Estimated winding number of the solution
            residual: The residual of the solution on the target system
            iterations: Number of iterations used in the endgame
            predictions: List of intermediate predictions
            ill_conditioned: Whether the path is in an ill-conditioned zone
        """
        self.solution = solution
        self.success = success
        self.winding_number = winding_number
        self.residual = residual
        self.iterations = iterations
        self.predictions = predictions if predictions else []
        self.ill_conditioned = ill_conditioned
        
    def __repr__(self) -> str:
        """String representation of the endgame result."""
        status = "success" if self.success else "failed"
        if self.ill_conditioned:
            status = "ill_conditioned"
            
        return (f"EndgameResult({status}, winding_number={self.winding_number}, "
                f"residual={self.residual:.2e}, iterations={self.iterations})")


class CauchyEndgame:
    """Implementation of the Cauchy endgame strategy.
    
    The Cauchy endgame uses Cauchy's integral formula to predict the solution at t=0
    by tracking loops in the complex t-plane.
    """
    
    def __init__(self, 
                 samples_per_loop: int = 8,
                 loopclosed_tolerance: float = 1e-5,
                 max_winding_number: int = 16,
                 geometric_series_factor: float = 0.5,
                 max_iterations: int = 10,
                 abs_tol: float = 1e-10):
        """Initialize the Cauchy endgame strategy.
        
        Args:
            samples_per_loop: Number of sample points per loop
            loopclosed_tolerance: Tolerance for considering a loop closed
            max_winding_number: Maximum allowed winding number
            geometric_series_factor: Factor for the geometric series of t values
            max_iterations: Maximum number of iterations for the endgame
            abs_tol: Absolute tolerance for convergence of predictions
        """
        self.samples_per_loop = samples_per_loop
        self.loopclosed_tolerance = loopclosed_tolerance
        self.max_winding_number = max_winding_number
        self.geometric_series_factor = geometric_series_factor
        self.max_iterations = max_iterations
        self.abs_tol = abs_tol
        
    def run(self, 
           start_system: PolynomialSystem,
           target_system: PolynomialSystem,
           current_solution: np.ndarray,
           current_t: float,
           variables: List[Variable],
           predictor_corrector_func: Callable,
           gamma: complex = 0.6+0.8j) -> EndgameResult:
        """Run the Cauchy endgame to predict the solution at t=0.
        
        Args:
            start_system: The start system
            target_system: The target system
            current_solution: The current solution at t=current_t
            current_t: The current value of t (should be in the endgame zone)
            variables: The system variables
            predictor_corrector_func: Function to perform one predictor-corrector step
            gamma: The gamma value for the homotopy
            
        Returns:
            EndgameResult containing the predicted solution
        """
        # Initialize result trackers
        current_point = np.array(current_solution, dtype=complex)
        predictions = []
        iterations = 0
        best_prediction = None
        
        # Check if the Jacobian is already too ill-conditioned to proceed
        jac = self._evaluate_jacobian(start_system, target_system, current_point, 
                                      current_t, variables, gamma)
        try:
            cond = np.linalg.cond(jac)
            if cond > 1e12:  # Extremely ill-conditioned
                return EndgameResult(
                    solution=current_point,
                    success=False,
                    winding_number=1,
                    ill_conditioned=True
                )
        except np.linalg.LinAlgError:
            # If we can't even compute condition number, definitely ill-conditioned
            return EndgameResult(
                solution=current_point,
                success=False,
                winding_number=1,
                ill_conditioned=True
            )
        
        # Start with winding number 1 and increase if needed
        winding_number = 1
        
        # Main endgame loop
        t = current_t
        while iterations < self.max_iterations and t > 1e-14:
            # Next t in the geometric series
            next_t = t * self.geometric_series_factor
            
            # Track a loop in the complex t-plane
            loop_points = self._track_loop(
                start_system, target_system, current_point, t, 
                winding_number, variables, predictor_corrector_func, gamma
            )
            
            # If the loop failed to close, try with a higher winding number
            if loop_points is None:
                winding_number *= 2
                if winding_number > self.max_winding_number:
                    return EndgameResult(
                        solution=current_point,
                        success=False,
                        winding_number=winding_number,
                        iterations=iterations,
                        predictions=predictions
                    )
                continue
            
            # Use the loop to predict the value at t=0
            prediction = self._predict_solution(loop_points, winding_number)
            predictions.append(prediction)
            
            # Update the current point for the next iteration
            current_point = loop_points[0]  # The point after completing the loop
            t = next_t
            iterations += 1
            
            # Check for convergence between consecutive predictions
            if len(predictions) >= 2:
                diff = np.linalg.norm(predictions[-1] - predictions[-2])
                if diff < self.abs_tol:
                    best_prediction = prediction
                    break
        
        # If we didn't find a best prediction, use the last one
        if best_prediction is None and predictions:
            best_prediction = predictions[-1]
        else:
            best_prediction = current_point  # Fallback
        
        # Compute residual on the target system
        if best_prediction is not None:
            # Create a dictionary mapping variables to their values
            var_dict = {var: val for var, val in zip(variables, best_prediction)}
            
            # Evaluate the target system at the prediction
            residual = np.linalg.norm(target_system.evaluate(var_dict))
        else:
            residual = float('inf')
        
        return EndgameResult(
            solution=best_prediction,
            success=best_prediction is not None,
            winding_number=winding_number,
            residual=residual,
            iterations=iterations,
            predictions=predictions
        )
    
    def _track_loop(self, 
                   start_system: PolynomialSystem,
                   target_system: PolynomialSystem,
                   start_point: np.ndarray,
                   t: float,
                   winding_number: int,
                   variables: List[Variable],
                   predictor_corrector_func: Callable,
                   gamma: complex) -> Optional[List[np.ndarray]]:
        """Track a loop in the complex t-plane.
        
        Args:
            start_system: The start system
            target_system: The target system
            start_point: The starting point for the loop
            t: The current t value
            winding_number: The winding number to use
            variables: The system variables
            predictor_corrector_func: Function to perform one predictor-corrector step
            gamma: The gamma value for the homotopy
            
        Returns:
            List of points along the loop, or None if the loop failed to close
        """
        # Number of sample points adjusted for the winding number
        n_samples = self.samples_per_loop * winding_number
        
        # Generate complex t values for the loop
        t_loop = [t * cmath.exp(2j * np.pi * k / n_samples) for k in range(n_samples + 1)]
        
        # Track the path around the loop
        loop_points = [start_point.copy()]
        current_point = start_point.copy()
        
        for i in range(1, len(t_loop)):
            t_from = t_loop[i-1]
            t_to = t_loop[i]
            
            # Use the provided predictor-corrector to track from t_from to t_to
            next_point, success = predictor_corrector_func(
                start_system, target_system, current_point, t_from, t_to, variables, gamma
            )
            
            if not success:
                return None  # Loop tracking failed
            
            current_point = next_point
            loop_points.append(current_point)
        
        # Check if the loop closed correctly
        loop_closure_error = np.linalg.norm(loop_points[-1] - loop_points[0])
        if loop_closure_error > self.loopclosed_tolerance:
            return None  # Loop didn't close properly
        
        return loop_points
    
    def _predict_solution(self, 
                         loop_points: List[np.ndarray],
                         winding_number: int) -> np.ndarray:
        """Predict the solution at t=0 using the loop points.
        
        Uses a weighted average of the points, which approximates Cauchy's integral formula.
        
        Args:
            loop_points: Points sampled along the loop
            winding_number: The winding number
            
        Returns:
            Predicted solution at t=0
        """
        # Remove the last point (duplicate of first)
        points = loop_points[:-1]
        n = len(points)
        
        # For winding number 1, a simple average is often effective
        if winding_number == 1:
            return np.mean(points, axis=0)
        
        # For higher winding numbers, we combine the points in groups
        # This is a simplified approximation of Cauchy's formula
        grouped_points = []
        group_size = n // winding_number
        
        for i in range(winding_number):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < winding_number - 1 else n
            group = points[start_idx:end_idx]
            grouped_points.append(np.mean(group, axis=0))
        
        return np.mean(grouped_points, axis=0)
    
    def _evaluate_jacobian(self,
                          start_system: PolynomialSystem,
                          target_system: PolynomialSystem,
                          point: np.ndarray,
                          t: float,
                          variables: List[Variable],
                          gamma: complex) -> np.ndarray:
        """Evaluate the Jacobian of the homotopy function at a point.
        
        Args:
            start_system: The start system
            target_system: The target system
            point: The point at which to evaluate
            t: The t value
            variables: The system variables
            gamma: The gamma value for the homotopy
            
        Returns:
            Jacobian matrix
        """
        # Create a dictionary mapping variables to their values
        var_dict = {var: val for var, val in zip(variables, point)}
        
        # Get the Jacobian polynomials
        jac_f = target_system.jacobian(variables)
        jac_g = start_system.jacobian(variables)
        
        # Evaluate each Jacobian
        jac_f_vals = []
        for row in jac_f:
            jac_row = []
            for poly in row:
                jac_row.append(poly.evaluate(var_dict))
            jac_f_vals.append(jac_row)
        
        jac_g_vals = []
        for row in jac_g:
            jac_row = []
            for poly in row:
                jac_row.append(poly.evaluate(var_dict))
            jac_g_vals.append(jac_row)
        
        # Compute the homotopy Jacobian
        jac_f_array = np.array(jac_f_vals, dtype=complex)
        jac_g_array = np.array(jac_g_vals, dtype=complex)
        
        return (1 - t) * jac_f_array + t * gamma * jac_g_array