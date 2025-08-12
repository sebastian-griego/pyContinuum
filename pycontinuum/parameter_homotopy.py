"""
Parameter homotopy module for PyContinuum.

This module implements parameter homotopies, which track solutions
as parameters in the system change. Parameter homotopies are essential
for computing witness sets and numerical irreducible decomposition.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem
from pycontinuum.utils import evaluate_system_at_point, evaluate_jacobian_at_point


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
                 square_fix: bool = True):  # Add option to make the system square
        """
        Initialize a parameter homotopy.
        
        Args:
            fixed_system: The fixed part of the system (F).
            start_param_system: The start parametric part (L1).
            end_param_system: The end parametric part (L2).
            variables: The variables in the system.
            square_fix: If True, add dummy equations to make the system square.
        """
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
        
        # For non-square systems, we'll handle them specially
        self.is_square = self.total_eqs == len(variables)
        self.dummy_count = 0
        self.extended_variables = False
        
        if not self.is_square:
            print(f"Warning: ParameterHomotopy created with {self.total_eqs} equations but {len(variables)} variables.")
            
            if square_fix and self.total_eqs < len(variables):
                # We'll add dummy "Lagrange multiplier" variables for the underdetermined case
                self.dummy_count = len(variables) - self.total_eqs
                print(f"  Adding {self.dummy_count} dummy equations to make the system square.")
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
        # Handle the base (non-dummy) equations
        base_eqs = self.num_fixed + self.num_param
        vals = np.zeros(self.total_eqs, dtype=complex)
        
        # Create dictionary for polynomial evaluation
        point_dict = {var: val for var, val in zip(self.variables, point)}
        
        # Evaluate fixed part F(x)
        fixed_vals = self.fixed_system.evaluate(point_dict)
        vals[:self.num_fixed] = fixed_vals
        
        # Evaluate parametric part (1-t)L1(x) + tL2(x)
        l1_vals = self.start_param_system.evaluate(point_dict)
        l2_vals = self.end_param_system.evaluate(point_dict)
        param_vals = (1 - t) * np.array(l1_vals) + t * np.array(l2_vals)
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
        base_eqs = self.num_fixed + self.num_param
        jac = np.zeros((self.total_eqs, len(self.variables)), dtype=complex)
        
        # Jacobian of fixed part dF/dx (may be empty if no fixed equations)
        if self.num_fixed > 0:
            jac_F = evaluate_jacobian_at_point(self.fixed_system, point, self.variables)
            if jac_F.size:
                jac[:self.num_fixed, :] = jac_F
        
        # Jacobian of parametric part (1-t)dL1/dx + t*dL2/dx
        if self.num_param > 0:
            jac_L1 = evaluate_jacobian_at_point(self.start_param_system, point, self.variables)
            jac_L2 = evaluate_jacobian_at_point(self.end_param_system, point, self.variables)
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
        base_eqs = self.num_fixed + self.num_param
        deriv = np.zeros(self.total_eqs, dtype=complex)
        point_dict = {var: val for var, val in zip(self.variables, point)}
        
        # Fixed part doesn't depend on t
        # deriv[:self.num_fixed] = 0
        
        # Parametric part derivative: d/dt[(1-t)L1(x) + tL2(x)] = -L1(x) + L2(x)
        l1_vals = self.start_param_system.evaluate(point_dict)
        l2_vals = self.end_param_system.evaluate(point_dict)
        param_deriv = -np.array(l1_vals) + np.array(l2_vals)
        deriv[self.num_fixed:base_eqs] = param_deriv
        
        # Dummy equations don't depend on t
        # deriv[base_eqs:] = 0
        
        return deriv


def track_parameter_path(parameter_homotopy: ParameterHomotopy,
                         start_point: np.ndarray,
                         start_t: float = 0.0,
                         end_t: float = 1.0,
                         options: Dict = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Track a solution path for a parameter homotopy.
    
    This tracks a solution x(t) of H(x, t) = 0 from start_t to end_t.
    
    Args:
        parameter_homotopy: The parameter homotopy to follow.
        start_point: The starting solution x(start_t).
        start_t: The starting parameter value.
        end_t: The ending parameter value.
        options: Options for the tracking process.
        
    Returns:
        Tuple of (end_point, tracking_info).
    """
    if options is None:
        options = {}
        
    # Default options for robustness
    tol = options.get('tol', 1e-8)
    min_step_size = options.get('min_step_size', 1e-6)
    max_step_size = options.get('max_step_size', 0.05)  # Smaller steps for parameter homotopy
    max_corrections = options.get('max_corrections', 10)
    verbose = options.get('verbose', False)
    store_paths = options.get('store_paths', False)
    
    t_current = start_t
    current_point = np.array(start_point, dtype=complex)
    step_size = max_step_size / 2  # Start with half max step size for caution
    
    # Store path points
    path_points = [(t_current, current_point.copy())] if store_paths else []
    
    # Path data for return
    path_info = {
        'success': False,
        'steps': 0,
        'newton_iters': 0,
        't': t_current,
        'path_points': path_points
    }
    
    # Determine direction
    direction = 1 if end_t > start_t else -1
    
    # Main tracking loop
    while direction * (end_t - t_current) > 1e-12:  # Stop when we reach end_t
        path_info['steps'] += 1
        
        # Check for too many steps
        if path_info['steps'] > options.get('max_steps', 1000):
            print(f"Warning: Reached maximum number of steps ({options.get('max_steps', 1000)})")
            return current_point, path_info
        
        # 1. Compute tangent dx/dt = -Hx^{-1} * Ht
        jac_x = parameter_homotopy.jacobian_x(current_point, t_current)
        deriv_t = parameter_homotopy.deriv_t(current_point, t_current)
        
        try:
            tangent = np.linalg.solve(jac_x, -deriv_t)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if Jacobian is singular
            if verbose:
                print(f"Warning: Singular Jacobian at t={t_current}, using pseudoinverse")
            tangent = np.linalg.lstsq(jac_x, -deriv_t, rcond=None)[0]
        
        # Normalize tangent for numerical stability
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-10:
            tangent = tangent / tangent_norm
        
        # 2. Determine step size, making sure we don't overshoot end_t
        dt = direction * min(step_size, abs(end_t - t_current))
        t_target = t_current + dt
        
        # 3. Predictor step (Euler predictor)
        predicted = current_point + dt * tangent
        
        # 4. Corrector steps (Newton's method)
        corrected = predicted.copy()
        converged = False
        newton_steps = 0
        
        for i in range(max_corrections):
            newton_steps += 1
            
            # Evaluate homotopy at current prediction
            H_val = parameter_homotopy.evaluate(corrected, t_target)
            residual = np.linalg.norm(H_val)
            
            # Check if we've already converged
            if residual < tol:
                converged = True
                break
                
            # Compute Jacobian for Newton step
            jac = parameter_homotopy.jacobian_x(corrected, t_target)
            
            # Newton step: solve J * delta = -H
            try:
                delta = np.linalg.solve(jac, -H_val)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if Jacobian is singular
                delta = np.linalg.lstsq(jac, -H_val, rcond=None)[0]
                
            # Update point
            corrected = corrected + delta
            
            # Check for convergence
            if np.linalg.norm(delta) < tol:
                # Double-check residual
                H_val = parameter_homotopy.evaluate(corrected, t_target)
                converged = np.linalg.norm(H_val) < tol
                break
                
        path_info['newton_iters'] += newton_steps
        
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
                if newton_steps > 0 and residual < 1e-4:  # Very loose tolerance
                    # Use the last corrected point even though it didn't fully converge
                    current_point = corrected
                    t_current = t_target
                    if store_paths:
                        path_points.append((t_current, current_point.copy()))
                    continue
                else:
                    # Really failed
                    path_info['t'] = t_current
                    return current_point, path_info
                    
    # Reached end_t
    # Final residual check
    final_H_val = parameter_homotopy.evaluate(current_point, end_t)
    success = np.linalg.norm(final_H_val) < tol * 10  # Slightly looser tolerance at end
    
    path_info['success'] = success
    path_info['t'] = t_current
    path_info['final_residual'] = float(np.linalg.norm(final_H_val))
    
    return current_point, path_info