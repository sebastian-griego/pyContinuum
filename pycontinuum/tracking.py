"""
Path tracking module for PyContinuum.

This module implements the predictor-corrector methods for tracking solution paths
along a homotopy.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import time
from tqdm.auto import tqdm

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem


def evaluate_system_at_point(system: PolynomialSystem, 
                             point: List[complex], 
                             variables: List[Variable]) -> np.ndarray:
    """Evaluate a polynomial system at a point.
    
    Args:
        system: The polynomial system to evaluate
        point: The point at which to evaluate the system
        variables: The variables in the system
        
    Returns:
        Array of values for each equation
    """
    # Create a dictionary mapping variables to their values
    var_dict = {var: val for var, val in zip(variables, point)}
    
    # Evaluate the system
    values = system.evaluate(var_dict)
    
    # Convert to numpy array
    return np.array(values, dtype=complex)


def evaluate_jacobian_at_point(system: PolynomialSystem,
                              point: List[complex],
                              variables: List[Variable]) -> np.ndarray:
    """Evaluate the Jacobian of a polynomial system at a point.
    
    Args:
        system: The polynomial system to evaluate
        point: The point at which to evaluate the Jacobian
        variables: The variables in the system
        
    Returns:
        Jacobian matrix as a numpy array
    """
    # Create a dictionary mapping variables to their values
    var_dict = {var: val for var, val in zip(variables, point)}
    
    # Get the Jacobian polynomials
    jac_polys = system.jacobian(variables)
    
    # Evaluate each polynomial in the Jacobian
    jac_values = []
    for row in jac_polys:
        jac_row = []
        for poly in row:
            jac_row.append(poly.evaluate(var_dict))
        jac_values.append(jac_row)
    
    # Convert to numpy array
    return np.array(jac_values, dtype=complex)


def newton_corrector(system: PolynomialSystem,
                    point: np.ndarray,
                    variables: List[Variable],
                    max_iters: int = 10,
                    tol: float = 1e-10) -> Tuple[np.ndarray, bool, int]:
    """Apply Newton's method to correct a point to a solution.
    
    Args:
        system: The polynomial system
        point: Initial point for correction
        variables: The variables in the system
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence
        
    Returns:
        Tuple of (corrected point, success flag, number of iterations)
    """
    current = np.array(point, dtype=complex)
    
    for i in range(max_iters):
        # Evaluate the system and Jacobian
        f_val = evaluate_system_at_point(system, current, variables)
        jac = evaluate_jacobian_at_point(system, current, variables)
        
        # Check if we're already at a solution
        if np.linalg.norm(f_val) < tol:
            return current, True, i
        
        # Solve the linear system J * delta = -f
        try:
            delta = np.linalg.solve(jac, -f_val)
        except np.linalg.LinAlgError:
            # If the Jacobian is singular, try a pseudoinverse
            delta = np.linalg.lstsq(jac, -f_val, rcond=None)[0]
        
        # Update the point
        current = current + delta
        
        # Check for convergence
        if np.linalg.norm(delta) < tol:
            return current, True, i+1
    
    # If we got here, we didn't converge
    return current, False, max_iters


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
    return point + delta_t * tangent


def homotopy_function(start_system: PolynomialSystem,
                     target_system: PolynomialSystem,
                     point: np.ndarray,
                     t: float,
                     variables: List[Variable],
                     gamma: complex = 0.6+0.8j) -> np.ndarray:
    """Evaluate the homotopy function H(x, t) = (1-t)f(x) + t*gamma*g(x).
    
    Args:
        start_system: Start system g(x)
        target_system: Target system f(x)
        point: Point at which to evaluate
        t: Homotopy parameter (1 = start, 0 = target)
        variables: System variables
        gamma: Random complex number for the homotopy
        
    Returns:
        Value of the homotopy at (point, t)
    """
    # Evaluate both systems
    f_val = evaluate_system_at_point(target_system, point, variables)
    g_val = evaluate_system_at_point(start_system, point, variables)
    
    # Compute the homotopy value
    return (1 - t) * f_val + t * gamma * g_val


def homotopy_jacobian(start_system: PolynomialSystem,
                     target_system: PolynomialSystem,
                     point: np.ndarray,
                     t: float,
                     variables: List[Variable],
                     gamma: complex = 0.6+0.8j) -> np.ndarray:
    """Evaluate the Jacobian of the homotopy function with respect to x.
    
    Args:
        start_system: Start system g(x)
        target_system: Target system f(x)
        point: Point at which to evaluate
        t: Homotopy parameter (1 = start, 0 = target)
        variables: System variables
        gamma: Random complex number for the homotopy
        
    Returns:
        Jacobian of the homotopy at (point, t)
    """
    # Evaluate both Jacobians
    jac_f = evaluate_jacobian_at_point(target_system, point, variables)
    jac_g = evaluate_jacobian_at_point(start_system, point, variables)
    
    # Compute the homotopy Jacobian
    return (1 - t) * jac_f + t * gamma * jac_g


def compute_tangent(start_system: PolynomialSystem,
                   target_system: PolynomialSystem,
                   point: np.ndarray,
                   t: float,
                   variables: List[Variable],
                   gamma: complex = 0.6+0.8j) -> np.ndarray:
    """Compute the tangent to the path at a point.
    
    Args:
        start_system: Start system g(x)
        target_system: Target system f(x)
        point: Current point
        t: Current t value
        variables: System variables
        gamma: Random complex number for the homotopy
        
    Returns:
        Tangent vector to the path at (point, t)
    """
    # Get the Jacobian of H with respect to x
    jac = homotopy_jacobian(start_system, target_system, point, t, variables, gamma)
    
    # Compute dH/dt = -f(x) + gamma*g(x)
    f_val = evaluate_system_at_point(target_system, point, variables)
    g_val = evaluate_system_at_point(start_system, point, variables)
    dH_dt = -f_val + gamma * g_val
    
    # Solve jac * dx/dt = -dH/dt to get the tangent vector
    try:
        tangent = np.linalg.solve(jac, -dH_dt)
    except np.linalg.LinAlgError:
        # If the Jacobian is singular, try a pseudoinverse
        tangent = np.linalg.lstsq(jac, -dH_dt, rcond=None)[0]
    
    # Normalize the tangent vector for numerical stability
    norm = np.linalg.norm(tangent)
    if norm > 1e-10:
        tangent = tangent / norm
        
    return tangent


def track_single_path(start_system: PolynomialSystem,
                     target_system: PolynomialSystem,
                     start_solution: np.ndarray,
                     variables: List[Variable],
                     tol: float = 1e-10,
                     min_step_size: float = 1e-6,
                     max_step_size: float = 0.1,
                     gamma: complex = 0.6+0.8j,
                     endgame_start: float = 0.1,
                     verbose: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Track a single path from start_solution (t=1) to the target system (t=0).
    
    Args:
        start_system: Start system g(x)
        target_system: Target system f(x)
        start_solution: Solution of the start system
        variables: System variables
        tol: Tolerance for numerical methods
        min_step_size: Minimum step size for t
        max_step_size: Maximum step size for t
        gamma: Random complex number for the homotopy
        endgame_start: t-value at which to start the endgame
        verbose: Whether to print detailed progress
        
    Returns:
        Tuple of (end_solution, path_info)
    """
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
        'path_points': [(t, current_point.copy())] if verbose else []
    }
    
    # Use a simple continuation method to track the path
    while t > 0:
        path_info['steps'] += 1
        
        # Reduce step size for the final approach
        if t < endgame_start and step_size > min_step_size:
            step_size = max(min_step_size, t / 10)  # Reduce step size for endpoint accuracy
            
        # Set target t for this step (don't go below 0)
        t_target = max(0.0, t - step_size)
        
        # Check for infinity or NaN
        if not np.all(np.isfinite(current_point)):
            if verbose:
                print("Path diverged to infinity or NaN")
            return current_point, path_info
        
        # Compute tangent at current point
        tangent = compute_tangent(start_system, target_system, current_point, t, variables, gamma)
        
        # Predict the next point using Euler's method
        predicted = euler_predictor(t, t_target, current_point, tangent)
        
        # Create the homotopy system at t_target
        def homotopy_t_target(x):
            return homotopy_function(start_system, target_system, x, t_target, variables, gamma)
            
        def homotopy_jacobian_t_target(x):
            return homotopy_jacobian(start_system, target_system, x, t_target, variables, gamma)
            
        # Correct the prediction using Newton's method
        homotopy_system = PolynomialSystem([])  # Placeholder
        homotopy_system.evaluate = lambda values: homotopy_t_target(
            [values[var] for var in variables]
        )
        homotopy_system.jacobian = lambda vars_list: homotopy_jacobian_t_target(
            [values[var] for var in variables]
        ).tolist()
        
        corrected, success, iters = newton_corrector(homotopy_system, predicted, variables, tol=tol)
        
        path_info['newton_iters'] += iters
        
        # Adjust step size based on Newton convergence
        if not success or iters > 5:
            # Reduce step size and try again
            step_size = max(min_step_size, step_size / 2)
            if step_size == min_step_size and not success:
                if verbose:
                    print(f"Newton failed to converge at t={t_target}, path may be near a singularity")
                # Do a final correction attempt at a larger tolerance
                corrected, success, _ = newton_corrector(homotopy_system, predicted, variables, tol=tol*10)
                if not success:
                    return corrected, path_info
        elif iters <= 2 and step_size < max_step_size:
            # Increase step size if Newton converges quickly
            step_size = min(max_step_size, step_size * 1.5)
        
        # Update the current point and t
        current_point = corrected
        t = t_target
        
        # Save point if verbose
        if verbose:
            path_info['path_points'].append((t, current_point.copy()))
    
    # We've reached t=0, final check for target system
    final_values = {var: val for var, val in zip(variables, current_point)}
    final_residual = np.linalg.norm(target_system.evaluate(final_values))
    
    if final_residual < 100 * tol:
        path_info['success'] = True
        
        # Check if the solution is singular by examining the Jacobian
        jac = evaluate_jacobian_at_point(target_system, current_point, variables)
        try:
            # Try to compute the condition number of the Jacobian
            cond = np.linalg.cond(jac)
            path_info['singular'] = cond > 1e8  # Condition number threshold for singularity
        except np.linalg.LinAlgError:
            # If LinAlgError occurs, the Jacobian is singular
            path_info['singular'] = True
            
    return current_point, path_info


def track_paths(start_system: PolynomialSystem,
               target_system: PolynomialSystem,
               start_solutions: List[List[complex]],
               variables: List[Variable],
               tol: float = 1e-10,
               gamma: Optional[complex] = None,
               verbose: bool = False) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Track multiple paths from start solutions to the target system.
    
    Args:
        start_system: Start system g(x)
        target_system: Target system f(x)
        start_solutions: List of solutions to the start system
        variables: System variables
        tol: Tolerance for numerical methods
        gamma: Random complex number for the homotopy (default: random)
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (end_solutions, path_results)
    """
    # Use a fixed random gamma if not provided
    if gamma is None:
        # Use a random complex number with unit magnitude for gamma
        rand_angle = np.random.uniform(0, 2 * np.pi)
        gamma = np.exp(1j * rand_angle)
    
    # Initialize result containers
    end_solutions = []
    path_results = {
        'success': [],
        'singular': [],
        'steps': [],
        'newton_iters': []
    }
    
    # Track each path in parallel (in the future this could be parallel)
    n_paths = len(start_solutions)
    
    if verbose:
        print(f"Tracking {n_paths} paths from t=1 to t=0...")
        pbar = tqdm(total=n_paths)
        
    start_time = time.time()
    
    for i, start_sol in enumerate(start_solutions):
        if verbose and i > 0 and i % 10 == 0:
            elapsed = time.time() - start_time
            paths_per_sec = i / elapsed
            eta = (n_paths - i) / paths_per_sec if paths_per_sec > 0 else 0
            print(f"Completed {i}/{n_paths} paths ({paths_per_sec:.2f} paths/sec, ETA: {eta:.1f}s)")
            
        # Track this path
        end_sol, path_info = track_single_path(
            start_system=start_system,
            target_system=target_system,
            start_solution=start_sol,
            variables=variables,
            tol=tol,
            gamma=gamma,
            verbose=False  # Don't store path points for all paths
        )
        
        # Store results
        end_solutions.append(end_sol)
        path_results['success'].append(path_info['success'])
        path_results['singular'].append(path_info.get('singular', False))
        path_results['steps'].append(path_info['steps'])
        path_results['newton_iters'].append(path_info['newton_iters'])
        
        if verbose:
            pbar.update(1)
            
    if verbose:
        pbar.close()
        success_count = sum(path_results['success'])
        print(f"Path tracking complete: {success_count}/{n_paths} successful paths")
        
    return end_solutions, path_results