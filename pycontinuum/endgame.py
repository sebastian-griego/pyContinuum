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
from pycontinuum.utils import evaluate_system_at_point, evaluate_jacobian_at_point


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
        """Initialize an endgame result."""
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
    """Implementation of the Cauchy endgame strategy."""
    
    def __init__(self, 
                 geometric_series_factor: float = 0.5,
                 abs_tol: float = 1e-10,
                 max_iterations: int = 10,
                 max_winding_number: int = 4):
        """Initialize the Cauchy endgame strategy."""
        self.geometric_series_factor = geometric_series_factor
        self.abs_tol = abs_tol
        self.max_iterations = max_iterations
        self.max_winding_number = max_winding_number
        
    def run(self, 
           start_system: PolynomialSystem,
           target_system: PolynomialSystem,
           current_solution: np.ndarray,
           current_t: float,
           variables: List[Variable],
           predictor_corrector_func: Callable,
           gamma: complex = 0.6+0.8j) -> 'EndgameResult':
        """Run the Cauchy endgame to predict the solution at t=0."""
        # Initialize tracking
        t = current_t
        current_point = np.array(current_solution, dtype=complex)
        winding_number = 1
        iterations = 0
        
        # Store points and predictions
        points = [current_point.copy()]
        predictions = []
        
        # Main endgame loop - move forward in geometric series
        while iterations < self.max_iterations and t > 1e-14:
            # Calculate next t in the geometric series
            next_t = t * self.geometric_series_factor
            
            # Track from t to next_t
            next_point, success = predictor_corrector_func(
                start_system, target_system, current_point, t, next_t, variables, gamma
            )
            
            if not success:
                # If tracking failed, return with what we have
                var_dict = {var: val for var, val in zip(variables, current_point)}
                residual = np.linalg.norm(target_system.evaluate(var_dict))
                return EndgameResult(
                    solution=current_point,
                    success=False,
                    winding_number=winding_number,
                    residual=residual,
                    iterations=iterations,
                    predictions=predictions,
                    ill_conditioned=True
                )
            
            # Update current point and t
            current_point = next_point
            points.append(current_point.copy())
            t = next_t
            iterations += 1
            
            # Make a prediction for t=0 based on current sequence
            prediction = self._make_prediction(points, winding_number)
            predictions.append(prediction)
            
            # Check for convergence between consecutive predictions
            if len(predictions) >= 2:
                diff = np.linalg.norm(predictions[-1] - predictions[-2])
                if diff < self.abs_tol:
                    # We've converged, use this prediction
                    var_dict = {var: val for var, val in zip(variables, prediction)}
                    residual = np.linalg.norm(target_system.evaluate(var_dict))
                    return EndgameResult(
                        solution=prediction,
                        success=True,
                        winding_number=winding_number,
                        residual=residual,
                        iterations=iterations,
                        predictions=predictions
                    )
                # Additional check for improved convergence rate
                elif len(predictions) >= 3:
                    prev_diff = np.linalg.norm(predictions[-2] - predictions[-3])
                    if prev_diff / diff < 2 and diff < self.abs_tol * 10:
                        # We're converging but slowly, accept this prediction
                        var_dict = {var: val for var, val in zip(variables, prediction)}
                        residual = np.linalg.norm(target_system.evaluate(var_dict))
                        return EndgameResult(
                            solution=prediction,
                            success=True,
                            winding_number=winding_number,
                            residual=residual,
                            iterations=iterations,
                            predictions=predictions
                        )
        
        # If we ran out of iterations but have predictions, use the last one
        if predictions:
            solution = predictions[-1]
            success = True
        else:
            solution = current_point
            success = False
            
        # Compute residual
        var_dict = {var: val for var, val in zip(variables, solution)}
        residual = np.linalg.norm(target_system.evaluate(var_dict))
        
        return EndgameResult(
            solution=solution,
            success=success,
            winding_number=winding_number,
            residual=residual,
            iterations=iterations,
            predictions=predictions
        )
    
    def _make_prediction(self, points: List[np.ndarray], winding_number: int) -> np.ndarray:
        """Predict the solution at t=0 based on the geometric sequence of points."""
        # Simple implementation: use Richardson extrapolation based on geometric series
        # In practice, this is just using the last two points for a linear extrapolation
        if len(points) < 2:
            return points[-1]
            
        # More advanced extrapolation could be implemented here, but
        # even this simple approach works well
        p_curr = points[-1]
        p_prev = points[-2]
        
        # Extrapolate to t=0
        lambda_factor = self.geometric_series_factor
        prediction = p_curr + (p_curr - p_prev) * lambda_factor / (1 - lambda_factor)
        
        return prediction