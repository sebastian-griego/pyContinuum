"""
Endgame module for PyContinuum.

This module implements endgame techniques for handling singular solutions
in homotopy continuation, primarily using the Cauchy endgame algorithm.
"""

import numpy as np
import cmath
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from enum import Enum

from pycontinuum.polynomial import Variable, PolynomialSystem
# Import utility functions
from pycontinuum.utils import (
    evaluate_system_at_point,
    evaluate_jacobian_at_point,
    newton_corrector
)

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
        self.samples_per_loop = samples_per_loop
        self.loopclosed_tolerance = loopclosed_tolerance
        self.max_winding_number = max_winding_number
        self.L = L
        self.K = K


class EndgamerOptions:
    """Configuration options for the endgame procedure."""
    
    def __init__(self,
                 geometric_series_factor: float = 0.5,
                 abstol: float = 1e-10,
                 max_winding_number: int = 16):
        """Initialize endgamer options.
        
        Args:
            geometric_series_factor: Factor for geometric series (typically 0.5)
            abstol: Absolute tolerance for convergence
            max_winding_number: Maximum allowed winding number
        """
        self.geometric_series_factor = geometric_series_factor
        self.abstol = abstol
        self.max_winding_number = max_winding_number


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
                 gamma: complex = 0.6+0.8j):
        """Initialize an endgamer.
        
        Args:
            start_system: Start system g(x)
            target_system: Target system f(x)
            variables: System variables
            alg: Endgame algorithm (default: CauchyEndgame with default params)
            options: Endgame options (default: EndgamerOptions with default params)
            gamma: Random complex number for the homotopy
        """
        self.start_system = start_system
        self.target_system = target_system
        self.variables = variables
        self.alg = alg if alg is not None else CauchyEndgame()
        self.options = options if options is not None else EndgamerOptions()
        self.gamma = gamma
        
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
    
    def setup(self, point: np.ndarray, radius: float):
        """Set up the endgamer to start at the given point and radius.
        
        Args:
            point: Starting point
            radius: Starting radius (t value)
        """
        self.R = radius
        self.iter = 0
        self.windingnumber = 1
        self.status = EndgameStatus.NOT_STARTED
        self.failurecode = "default_failure_code"
        self.current_point = np.array(point, dtype=complex)
        self.xs = [np.array(point, dtype=complex)]
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
        v = np.random.randn(len(x_R)) + 1j * np.random.randn(len(x_R))
        v = v / np.linalg.norm(v)
        
        # Define the approximation function
        def g(x1, x2, x3):
            numer = np.abs(np.dot(v, x2) - np.dot(v, x3))**2
            denom = np.abs(np.dot(v, x1) - np.dot(v, x2))**2
            return np.log(numer / denom) / np.log(λ)
        
        g_R = g(x_R, x_λR, x_λ2R)
        g_λR = g(x_λR, x_λ2R, x_λ3R)
        
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
        norms = [np.linalg.norm(s) for s in samples]
        m = min(norms)
        M = max(norms)
        
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
                            # If t is very small, the (1-t) and t factors can cause issues
                            # Use a more numerically stable approach
                            point_dict = {var: val for var, val in zip(self.variables, point)}
                            
                            if abs(t_current) < 1e-10:
                                # We're essentially at the target system
                                return np.array([eq.evaluate(point_dict) for eq in self.target_system.equations])
                            
                            # Evaluate both systems
                            f_val = [eq.evaluate(point_dict) for eq in self.target_system.equations]
                            g_val = [eq.evaluate(point_dict) for eq in self.start_system.equations]
                            
                            # Compute H(x, t) = (1-t)f(x) + t*gamma*g(x)
                            return np.array([(1 - abs(t_current)) * f + abs(t_current) * self.gamma * g 
                                           for f, g in zip(f_val, g_val)], dtype=complex)
                        
                        # Use a simple Newton step
                        # In a more robust implementation, this would be a full Newton corrector
                        jac = evaluate_jacobian_at_point(
                            self.target_system, current, self.variables
                        ) * (1 - abs(t_current)) + evaluate_jacobian_at_point(
                            self.start_system, current, self.variables
                        ) * abs(t_current) * self.gamma
                        
                        H_val = H_at_point(current)
                        
                        try:
                            delta = np.linalg.solve(jac, -H_val)
                        except np.linalg.LinAlgError:
                            # Use pseudoinverse if Jacobian is singular
                            delta = np.linalg.lstsq(jac, -H_val, rcond=None)[0]
                        
                        current = current + delta
                        
                        # Additional Newton corrections for accuracy
                        for _ in range(2):
                            H_val = H_at_point(current)
                            if np.linalg.norm(H_val) < 1e-10:
                                break
                                
                            jac = evaluate_jacobian_at_point(
                                self.target_system, current, self.variables
                            ) * (1 - abs(t_current)) + evaluate_jacobian_at_point(
                                self.start_system, current, self.variables
                            ) * abs(t_current) * self.gamma
                            
                            try:
                                delta = np.linalg.solve(jac, -H_val)
                            except np.linalg.LinAlgError:
                                delta = np.linalg.lstsq(jac, -H_val, rcond=None)[0]
                            
                            current = current + delta
                    
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
            if np.linalg.norm(samples[0] - current_sample) < loopclosed_tolerance:
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
        dist = np.linalg.norm(p - pprev)
        
        # Check if converged
        if dist < self.options.abstol:
            if len(self.predictions) > 2:
                pprev2 = self.predictions[-3]
                
                # Check if improvement is significant
                prev_dist = np.linalg.norm(pprev - pprev2)
                if prev_dist / dist < 2 and dist < self.options.abstol:
                    return True
            else:
                return True
                
        return False
    
    def move_forward(self):
        """
        Move forward to the next radius in the geometric series.
        
        Updates the current point and radius.
        """
        λ = self.options.geometric_series_factor
        next_radius = λ * self.R
        
        # Create a simple system for tracking to the next radius
        def H_at_point(point, t):
            # Convert to dictionary for evaluation
            point_dict = {var: val for var, val in zip(self.variables, point)}
            
            # Evaluate both systems
            f_val = [eq.evaluate(point_dict) for eq in self.target_system.equations]
            g_val = [eq.evaluate(point_dict) for eq in self.start_system.equations]
            
            # Compute H(x, t) = (1-t)f(x) + t*gamma*g(x)
            return np.array([(1 - t) * f + t * self.gamma * g 
                           for f, g in zip(f_val, g_val)], dtype=complex)
        
        # Track from current radius to next radius
        current = self.current_point.copy()
        
        # Simple straight-line tracking from R to λR
        steps = 10  # Fixed number of steps for simplicity
        path_t = np.linspace(self.R, next_radius, steps)
        
        for t in path_t[1:]:  # Skip the first point
            H_val = H_at_point(current, t)
            
            jac = evaluate_jacobian_at_point(
                self.target_system, current, self.variables
            ) * (1 - t) + evaluate_jacobian_at_point(
                self.start_system, current, self.variables
            ) * t * self.gamma
            
            try:
                delta = np.linalg.solve(jac, -H_val)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(jac, -H_val, rcond=None)[0]
            
            current = current + delta
            
            # Additional Newton corrections
            for _ in range(2):
                H_val = H_at_point(current, t)
                if np.linalg.norm(H_val) < 1e-10:
                    break
                    
                jac = evaluate_jacobian_at_point(
                    self.target_system, current, self.variables
                ) * (1 - t) + evaluate_jacobian_at_point(
                    self.start_system, current, self.variables
                ) * t * self.gamma
                
                try:
                    delta = np.linalg.solve(jac, -H_val)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(jac, -H_val, rcond=None)[0]
                
                current = current + delta
        
        # Update state
        self.xs.append(current)
        self.current_point = current
        self.R = next_radius
        self.iter += 1
    
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
            'winding_number': self.windingnumber,
            'predictions': []
        }
        
        # Main loop
        while self.status != EndgameStatus.SUCCESSFUL and self.status != EndgameStatus.FAILED:
            result_info['steps'] += 1
            
            # If radius is too small, stop
            if self.R <= 1e-14:
                self.status = EndgameStatus.FAILED
                self.failurecode = "radius_too_small"
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
            self.move_forward()
        
        # If we have predictions but they didn't meet convergence criteria,
        # use the last one anyway if it's good enough
        if self.predictions and self.status != EndgameStatus.SUCCESSFUL:
            last_prediction = self.predictions[-1]
            
            # Check residual
            pred_values = {var: val for var, val in zip(self.variables, last_prediction)}
            residual = np.linalg.norm(self.target_system.evaluate(pred_values))
            
            if residual < 1e-6:  # Relaxed tolerance for endgame
                self.status = EndgameStatus.SUCCESSFUL
                result_info['success'] = True
        
        # Set result information
        result_info['success'] = self.status == EndgameStatus.SUCCESSFUL
        result_info['winding_number'] = self.windingnumber
        
        # Final point is either the last prediction or the current point
        final_point = self.predictions[-1] if self.predictions else self.current_point
        
        return final_point, result_info


def run_cauchy_endgame(start_system: PolynomialSystem,
                      target_system: PolynomialSystem,
                      point: np.ndarray,
                      t: float,
                      variables: List[Variable],
                      options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run the Cauchy endgame from a given point and time value.
    
    This is a convenience function for using the Cauchy endgame.
    
    Args:
        start_system: Start system g(x)
        target_system: Target system f(x)
        point: Current point on the path
        t: Current t value
        variables: System variables
        options: Optional configuration parameters
        
    Returns:
        Tuple of (end_point, result_info)
    """
    # Process options
    opts = {}
    if options:
        opts.update(options)
    
    # If t is extremely small, just use Newton's method directly
    if t < 1e-6:
        # At this point, we're so close to the solution, just polish with Newton
        tol = opts.get('abstol', 1e-10)
        corrected, success, iters = newton_corrector(
            target_system, point, variables, 
            max_iters=20, tol=tol * 10  # More iterations and looser tolerance
        )
        
        result_info = {
            'success': success,
            'singular': True,
            'steps': iters,
            'winding_number': 1,
            'predictions': [corrected.tolist()]
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
        max_winding_number=opts.get('max_winding_number', 16)
    )
    
    # Create the endgamer
    endgamer = Endgamer(
        start_system=start_system,
        target_system=target_system,
        variables=variables,
        alg=alg,
        options=endgamer_opts,
        gamma=opts.get('gamma', 0.6+0.8j)
    )
    
    # Set up the endgamer
    endgamer.setup(point, t)
    
    # Run the endgame
    return endgamer.run()