"""
Path tracking module for PyContinuum.

This module implements the predictor-corrector methods for tracking solution paths
along a homotopy.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import time
# Import multiprocessing and cpu_count
import multiprocessing
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem
# Import utility functions
from pycontinuum.utils import (
    evaluate_system_at_point,
    evaluate_jacobian_at_point,
    newton_corrector
)
from pycontinuum.endgame import run_cauchy_endgame

# --- Helper function for parallel execution ---
# Needs to be defined at the top level or be pickleable
# We wrap the track_single_path call here
def _track_single_path_parallel_worker(args):
    """Worker function for parallel path tracking."""
    # Unpack arguments
    (start_system, target_system, start_solution, variables, tol,
     min_step_size, max_step_size, gamma, endgame_start, store_paths,
     use_endgame, endgame_options, verbose_single, debug_single) = args # Added endgame_options

    # Call the original single path tracker
    end_sol, path_info = track_single_path(
        start_system=start_system,
        target_system=target_system,
        start_solution=start_solution,
        variables=variables,
        tol=tol,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        gamma=gamma,
        endgame_start=endgame_start,
        store_paths=store_paths,
        use_endgame=use_endgame,
        verbose=verbose_single, # Pass verbosity for single path if needed
        debug=debug_single # Pass debug for single path if needed
    )

    # --- BEGIN MODIFICATION: Apply endgame within worker ---
    # Apply endgame immediately after tracking if requested and path was successful
    # This avoids needing to pass potentially large systems back and forth again
    # NOTE: This assumes run_cauchy_endgame is also pickle-friendly or stateless enough.
    #       If endgame needs complex shared state, this approach needs revision.
    if use_endgame and path_info.get('success', False):
        # Ensure t is close to 0 if path tracking finished normally
        # If endgame was already used inside track_single_path, path_info['endgame_used'] might be True
        # Let's apply endgame only if path tracking reached t=0 without internal endgame use OR
        # if we want to re-run/refine endgame at t=0 regardless.
        # For simplicity, let's just run it if use_endgame is True and path succeeded so far.
        # The t value passed here should ideally be the *final* t from tracking (close to 0)
        # Since track_single_path doesn't return final t, we pass 0.0

        # Check if endgame was already run internally to avoid double-running
        if not path_info.get('endgame_used', False):
            final_point_before_endgame = end_sol
            final_t_value = 0.0 # Assuming tracking reached t=0

            # Call endgame function
            try:
                final_point, endgame_info = run_cauchy_endgame(
                    start_system=start_system,
                    target_system=target_system,
                    point=final_point_before_endgame,
                    t=final_t_value,
                    variables=variables,
                    options=endgame_options
                )
                end_sol = final_point # Update the solution with the endgame result
                path_info.update(endgame_info) # Merge endgame info into path results
                path_info['endgame_applied_post_track'] = True # Mark that endgame ran here
            except Exception as e:
                # Handle potential errors during endgame execution within the worker
                if verbose_single:
                     print(f"Endgame failed for a path in worker: {e}")
                path_info['success'] = False # Mark path as failed if endgame crashes
                path_info['endgame_error'] = str(e)

    # --- END MODIFICATION ---

    # Return the endpoint and the path info dictionary
    return end_sol, path_info
# --- End Helper Function ---


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
    jac = evaluate_jacobian_at_point(target_system, current_point, variables)
    try:
        cond = np.linalg.cond(jac)
        if verbose and debug:
            print(f"Jacobian condition: {cond}")
        if cond > threshold:
            if verbose:
                print("Potential singularity detected!")
            return True
        return False
    except np.linalg.LinAlgError:
        if verbose:
            print("Singular Jacobian detected!")
        return True


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
            try:
                delta = np.linalg.lstsq(jac, -f_val, rcond=None)[0]
            except np.linalg.LinAlgError: # Handle cases where even lstsq fails
                 return current, False, i # Indicate failure

        # Update the point
        current = current + delta

        # Check for convergence
        if np.linalg.norm(delta) < tol:
            # Final check on residual after last step
            f_val_final = evaluate_system_at_point(system, current, variables)
            if np.linalg.norm(f_val_final) < tol:
                return current, True, i+1
            else:
                # Converged step size but not residual, likely near singularity or difficult point
                return current, False, i+1 # Indicate failure to meet residual tol

    # If we got here, we didn't converge in max_iters
    # Check final residual
    f_val_final = evaluate_system_at_point(system, current, variables)
    final_success = np.linalg.norm(f_val_final) < tol
    return current, final_success, max_iters


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
    # Ensure tangent has finite values before using
    if not np.all(np.isfinite(tangent)):
        # Handle cases where tangent might be inf/nan, return current point or raise error
        # For now, return current point to prevent crash, but signal issue upstream?
        # print("Warning: Non-finite tangent detected in Euler predictor.")
        return point
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
    # Use abs(t) if t can be complex during endgame loops? Assume t is real [0,1] here.
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
    # Use abs(t) if t can be complex? Assume t is real [0,1] here.
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
    tangent = np.zeros_like(point, dtype=complex) # Initialize tangent
    try:
        tangent = np.linalg.solve(jac, -dH_dt)
    except np.linalg.LinAlgError:
        # If the Jacobian is singular, try a pseudoinverse
        try:
            tangent = np.linalg.lstsq(jac, -dH_dt, rcond=None)[0]
        except np.linalg.LinAlgError:
             # Handle cases where even lstsq fails - return zero tangent?
             # This path might be stuck.
             pass # tangent remains zero

    # Normalize the tangent vector for numerical stability? Report suggests this helps.
    norm = np.linalg.norm(tangent)
    if norm > 1e-12: # Avoid division by zero
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
                     endgame_start: float = 0.01, # Changed default based on report suggestion
                     store_paths: bool = False,
                     use_endgame: bool = True,
                     verbose: bool = False,
                     debug: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]: # Removed endgame_options here
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
        endgame_start: t-value at which to start considering the endgame
        store_paths: Whether to store path points and print detailed progress
        use_endgame: Whether to use the endgame for singular endpoints
        verbose: If True, print status updates for this specific path.
        debug: If True, print detailed debug info for this path.

    Returns:
        Tuple of (end_solution, path_info)
        path_info dictionary contains success status, singularity info, etc.
    """
    # Initialize tracking
    t = 1.0
    current_point = np.array(start_solution, dtype=complex)
    step_size = max_step_size

    # Path data for return
    path_info = {
        'success': False,
        'singular': False, # Will be updated at the end or by endgame
        'steps': 0,
        'newton_iters': 0,
        'path_points': [(t, current_point.copy())] if store_paths else [],
        'status_message': 'Tracking started',
        'endgame_used': False, # Flag if endgame was triggered internally
        'final_t': t,
    }

    # Define the homotopy system for the corrector step
    # This structure helps avoid passing complex systems repeatedly in loops
    # Note: This captures t_target, gamma, systems, variables from the outer scope.
    # Ensure this works correctly with parallelization if track_single_path is called directly by workers.
    # (Our current approach uses a top-level worker function)
    class HomotopySystemForCorrection:
        def __init__(self, t_val):
            self.t = t_val

        def evaluate(self, point_dict):
            point_array = np.array([point_dict[var] for var in variables], dtype=complex)
            f_values = evaluate_system_at_point(target_system, point_array, variables)
            g_values = evaluate_system_at_point(start_system, point_array, variables)
            return (1 - self.t) * f_values + self.t * gamma * g_values

        def jacobian(self, vars_list):
            # Note: This evaluate Jacobian at the 'current_point' fixed for the corrector step
            # This might be slightly incorrect - Jacobian should be evaluated at the point passed to newton_corrector
            # Let's fix newton_corrector to handle this structure.
            # -- Correction: Let's make newton_corrector accept the system object directly --
            # The evaluate_jacobian_at_point utility needs the system object anyway.
            # So, we just need to ensure the 't' is correctly set for the evaluation.
            # Let's pass the t_target directly to newton_corrector or make the system adaptable.

            # -- Revised Approach: Pass t into corrector functions ---
            # Let's modify newton_corrector slightly or use utilities directly.
            # For simplicity here, let's assume `newton_corrector` internally calls
            # `homotopy_function` and `homotopy_jacobian` with the correct `t`.
            # We need to make `newton_corrector` aware of the homotopy structure.

            # Let's modify `newton_corrector` to accept the homotopy parameters.
            # (Modification needed in utils.py or redefine corrector logic here)
            # For now, assume `newton_corrector` works correctly with a standard system interface.
            # We create a dummy system object that uses the correct t internally.
            # This requires `newton_corrector` to call system.evaluate(point_dict) and system.jacobian(vars_list, point_dict)
            # Let's assume the current newton_corrector in utils.py is sufficient FOR NOW.
            # We will instantiate HomotopySystemForCorrection with t_target inside the loop.
            pass # Jacobian logic relies on the system passed to newton_corrector


    # Main tracking loop
    max_steps = int(2 / min_step_size) # Safety break for steps
    while t > tol / 10: # Stop condition adjusted slightly
        if path_info['steps'] > max_steps:
             path_info['status_message'] = 'Failed: Exceeded max steps'
             path_info['success'] = False
             break # Exit loop

        path_info['steps'] += 1

        # Check for internal endgame trigger
        if use_endgame and t <= endgame_start and not path_info['endgame_used']:
            # Check singularity condition before potentially switching to endgame
            # Use a reasonable threshold, e.g., 1/tol
            singularity_threshold = 1 / (tol * 10) # Example threshold
            might_be_singular = check_singularity(target_system, current_point, variables, singularity_threshold, verbose=verbose, debug=debug)

            if might_be_singular:
                if verbose:
                    print(f"Path {path_info.get('path_index', 'N/A')}: Potential singularity detected near t={t:.4e}. Considering endgame.")

                # --- Internal Endgame Call (Optional, if worker doesn't handle it) ---
                # If the parallel worker design applies endgame post-tracking, this section
                # might be redundant or only used for non-parallel runs.
                # For now, let's assume the main logic here *can* call endgame if needed.
                # NOTE: Endgame options need to be passed here if used.
                # endgame_options_dict = endgame_options or {} # Use passed options or default

                # try:
                #      end_point, endgame_info = run_cauchy_endgame(
                #           start_system, target_system, current_point, t,
                #           variables, options=endgame_options_dict
                #      )
                #      # Update path info and return immediately
                #      path_info.update(endgame_info)
                #      path_info['endgame_used'] = True
                #      path_info['final_t'] = t # Record t where endgame started
                #      # Ensure success status reflects endgame result
                #      path_info['success'] = endgame_info.get('success', False)
                #      path_info['status_message'] = f"Switched to endgame at t={t:.4e}, Result: {'Success' if path_info['success'] else 'Failure'}"
                #      if store_paths and endgame_info.get('predictions'):
                #           # Add endgame predictions to path_points if available
                #           for i, pred in enumerate(endgame_info['predictions']):
                #                pred_t = t * (endgame_options_dict.get('geometric_series_factor', 0.5) ** (i + 1))
                #                path_info['path_points'].append((pred_t, np.array(pred)))

                #      current_point = end_point # Update current point to endgame result
                #      return current_point, path_info # Return from function after endgame

                # except Exception as e:
                #      if verbose:
                #           print(f"Endgame failed during tracking for path {path_info.get('path_index', 'N/A')}: {e}")
                #      path_info['status_message'] = f"Endgame failed at t={t:.4e}: {e}"
                #      path_info['success'] = False
                #      # Decide whether to stop or continue tracking without endgame
                #      # Let's stop for now if endgame fails internally
                #      return current_point, path_info
                # ----------------------------------------------------------------------
                # If endgame is handled post-tracking by the worker, just note potential singularity
                path_info['status_message'] = f'Potential singularity near t={t:.4e}'
                # Continue standard tracking, endgame will be applied later if needed.

        # Adjust step size: smaller steps near t=0 or if previous step struggled
        current_max_step = min(max_step_size, t) # Don't overshoot t=0
        step_size = min(step_size, current_max_step) # Ensure step size respects max_step_size
        step_size = max(step_size, min_step_size)   # Ensure step size respects min_step_size

        # Determine target t for this step
        t_target = max(0.0, t - step_size)

        # Check for path divergence (infinity or NaN)
        if not np.all(np.isfinite(current_point)):
            path_info['status_message'] = 'Failed: Path diverged to infinity or NaN'
            path_info['success'] = False
            break

        # --- Predictor Step ---
        tangent = compute_tangent(start_system, target_system, current_point, t, variables, gamma)
        if not np.all(np.isfinite(tangent)):
             path_info['status_message'] = 'Failed: Tangent computation resulted in non-finite values'
             path_info['success'] = False
             break
        predicted = euler_predictor(t, t_target, current_point, tangent)
        if not np.all(np.isfinite(predicted)):
            path_info['status_message'] = 'Failed: Predictor step resulted in non-finite values'
            path_info['success'] = False
            break

        # --- Corrector Step (using Newton's method) ---
        # Create a system object representing H(x, t_target) = 0
        homotopy_system_at_t_target = PolynomialSystem([]) # Dummy object
        def eval_h_target(point_dict_eval):
             point_arr = np.array([point_dict_eval[var] for var in variables], dtype=complex)
             return homotopy_function(start_system, target_system, point_arr, t_target, variables, gamma)
        def jac_h_target(vars_list_jac, point_dict_jac):
             point_arr = np.array([point_dict_jac[var] for var in variables], dtype=complex)
             return homotopy_jacobian(start_system, target_system, point_arr, t_target, variables, gamma)

        # Modify the dummy system's methods - This feels hacky.
        # It's better if newton_corrector takes the functions/params directly.
        # Let's assume newton_corrector is adapted or reimplement here.
        # For now, we'll rely on evaluate_system_at_point and evaluate_jacobian_at_point
        # utilities which accept the system object. Let's create a *functional* system object.

        class FunctionalHomotopySystem:
             def __init__(self, t_val):
                  self.t = t_val

             # Need evaluate and jacobian methods that match PolynomialSystem interface somewhat
             # evaluate needs to accept dict, jacobian needs to accept list of vars
             def evaluate(self, point_dict_eval):
                  point_arr = np.array([point_dict_eval.get(var, 0j) for var in variables], dtype=complex)
                  return homotopy_function(start_system, target_system, point_arr, self.t, variables, gamma)

             def jacobian(self, vars_list_jac):
                  # This is tricky - the utility expects a PolynomialSystem to get equations
                  # Let's modify the newton_corrector call to use the utilities directly.

                  # Re-implementing Newton step here for clarity with homotopy:
                  point_corr = predicted.copy()
                  corrector_success = False
                  iters = 0
                  for i in range(10): # Max corrector iterations
                       iters = i + 1
                       f_val = homotopy_function(start_system, target_system, point_corr, t_target, variables, gamma)
                       if np.linalg.norm(f_val) < tol:
                            corrector_success = True
                            break

                       jac_val = homotopy_jacobian(start_system, target_system, point_corr, t_target, variables, gamma)

                       try:
                            delta = np.linalg.solve(jac_val, -f_val)
                       except np.linalg.LinAlgError:
                            try:
                                 delta = np.linalg.lstsq(jac_val, -f_val, rcond=None)[0]
                            except np.linalg.LinAlgError:
                                 # Corrector fails catastrophically
                                 corrector_success = False
                                 iters = i + 1 # Record iterations attempted
                                 break # Exit corrector loop

                       point_corr += delta
                       if np.linalg.norm(delta) < tol:
                            # Check residual after step converges
                            f_val_final = homotopy_function(start_system, target_system, point_corr, t_target, variables, gamma)
                            if np.linalg.norm(f_val_final) < tol:
                                 corrector_success = True
                            # else: corrector_success remains False
                            break

                  corrected = point_corr
                  success = corrector_success
                  path_info['newton_iters'] += iters


        # --- Step Size Adaptation ---
        if not success:
            # Newton failed: Reduce step size significantly and retry step if possible
            step_size = max(min_step_size, step_size / 4) # Reduce more aggressively
            path_info['status_message'] = f'Corrector failed at t={t_target:.4e}. Reducing step size.'
            # Don't update t or current_point, loop will retry with smaller step
            continue # Skip to next iteration without advancing t

        elif iters > 5: # Converged, but slowly
            # Reduce step size for next time
            step_size = max(min_step_size, step_size / 1.5)
            path_info['status_message'] = f'Corrector slow ({iters} iters) at t={t_target:.4e}. Reducing step size.'

        elif iters <= 2 and step_size < max_step_size: # Converged quickly
            # Increase step size for next time
            step_size = min(max_step_size, step_size * 1.2) # Increase less aggressively
            path_info['status_message'] = f'Step successful at t={t_target:.4e} ({iters} iters).'


        # --- Update State ---
        current_point = corrected
        t = t_target
        path_info['final_t'] = t # Update final t reached

        # Save point if storing paths
        if store_paths:
            path_info['path_points'].append((t, current_point.copy()))

    # --- Post-Loop Processing ---
    if path_info['success'] is False and path_info['status_message'] == 'Tracking started':
       # Loop finished without explicit failure, check final state
       if t <= tol / 10: # Reached end
          path_info['status_message'] = 'Reached target t=0'
       else: # Should have failed earlier if loop broke
          path_info['status_message'] = 'Loop finished unexpectedly'

    # Final check: Ensure the final point is a solution to the target system (F(x)=0)
    # This check happens *after* the loop, regardless of internal endgame use.
    # Endgame might refine the solution further in the worker function post-tracking.
    final_values = {var: val for var, val in zip(variables, current_point)}
    final_residual = np.linalg.norm(target_system.evaluate(final_values))

    # Success condition: Reached t=0 vicinity AND final residual is small
    if t <= tol / 10 and final_residual < 100 * tol: # Allow slightly larger tol for final check
        # If loop didn't explicitly fail, mark as success
        if path_info['success'] is False and 'Failed' not in path_info['status_message']:
             path_info['success'] = True
             if 'Endgame failed' not in path_info['status_message']: # Don't overwrite endgame failure message
                  path_info['status_message'] = f'Success: Reached t=0, residual={final_residual:.2e}'
        elif path_info['success'] is True: # If already marked success (e.g., by internal endgame)
             path_info['status_message'] += f', final residual={final_residual:.2e}'
        # Else: Keep existing failure message if loop failed before reaching end

    else:
        # Failed to converge or meet residual tolerance at the end
        if path_info['success'] is True: # If success was wrongly assumed earlier
             path_info['success'] = False
        if 'Failed' not in path_info['status_message']: # Add failure reason if not already set
             if t > tol/10:
                  path_info['status_message'] = f'Failed: Did not reach t=0 (final t={t:.4e})'
             else:
                  path_info['status_message'] = f'Failed: High residual at t=0 ({final_residual:.2e})'

    # Check singularity of the *final* point w.r.t the *target* system
    # Use a higher threshold for final classification
    final_singularity_threshold = 1 / tol # e.g., 1e8 for tol=1e-8
    path_info['singular'] = check_singularity(target_system, current_point, variables, final_singularity_threshold, verbose=False, debug=False) # Less verbose for final check


    return current_point, path_info


def track_paths(start_system: PolynomialSystem,
                target_system: PolynomialSystem,
                start_solutions: List[List[complex]],
                variables: List[Variable],
                tol: float = 1e-8,
                min_step_size: float = 1e-6, # Pass down tracking params
                max_step_size: float = 0.1,
                gamma: complex = 0.6+0.8j,
                endgame_start: float = 0.01,
                num_cores: int = 1, # Add num_cores argument
                verbose: bool = False,
                store_paths: bool = False,
                use_endgame: bool = True,
                endgame_options: Optional[Dict[str, Any]] = None) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Track solution paths from start solutions to the target system, potentially in parallel.

    Args:
        start_system: The start polynomial system.
        target_system: The target polynomial system.
        start_solutions: List of start solution vectors.
        variables: List of variables in the system.
        tol: Tolerance for convergence of the corrector.
        min_step_size: Minimum step size for t during tracking.
        max_step_size: Maximum step size for t during tracking.
        gamma: Random complex number for the homotopy.
        endgame_start: t-value at which to start considering the endgame.
        num_cores: Number of CPU cores to use for parallel tracking.
                   If 1, runs sequentially. If > 1, uses multiprocessing.
                   If None or 0, uses all available cores. (Default: 1)
        verbose: Whether to print overall progress information.
        store_paths: Whether to store all points along each path (passed to single tracker).
        use_endgame: Whether to use the endgame procedure near t=0.
        endgame_options: Optional dictionary of options for the endgame procedure.

    Returns:
        Tuple of (end_solutions, path_results).
        end_solutions: List of final points for each path.
        path_results: List of dictionaries containing result info for each path.
    """
    n_paths = len(start_solutions)
    results: List[Tuple[np.ndarray, Dict[str, Any]]] = [] # Store tuples of (end_sol, path_info)

    # --- BEGIN PARALLELIZATION MODIFICATION ---
    # Determine the number of cores to use
    if num_cores is None or num_cores == 0:
        actual_cores = cpu_count()
        if verbose: print(f"Using all available {actual_cores} cores.")
    elif num_cores > 1:
        actual_cores = min(num_cores, cpu_count())
        if verbose: print(f"Using {actual_cores} cores for parallel path tracking.")
    else:
        actual_cores = 1
        if verbose: print("Running path tracking sequentially.")

    start_time = time.time()

    # Prepare arguments for each task
    # Pass necessary parameters to the worker function
    tasks_args = []
    for i, start_sol in enumerate(start_solutions):
        # Arguments for _track_single_path_parallel_worker
        # Ensure endgame_options is passed
        args = (start_system, target_system, start_sol, variables, tol,
                min_step_size, max_step_size, gamma, endgame_start, store_paths,
                use_endgame, endgame_options, False, False) # verbose/debug for single path set to False
        tasks_args.append(args)


    if actual_cores > 1:
        # --- Parallel Execution ---
        if verbose:
            print(f"Tracking {n_paths} paths in parallel using {actual_cores} cores...")
        try:
            # Use multiprocessing Pool
            # Consider using imap_unordered for potentially better load balancing
            # if path tracking times vary significantly. map preserves order.
            with Pool(processes=actual_cores) as pool:
                 # Use tqdm for progress bar with multiprocessing
                 results = list(tqdm(pool.imap(_track_single_path_parallel_worker, tasks_args),
                                     total=n_paths, desc="Tracking Paths"))

        except Exception as e:
            print(f"\nError during parallel path tracking: {e}")
            print("Trying to run sequentially...")
            actual_cores = 1 # Fallback to sequential

    if actual_cores == 1:
        # --- Sequential Execution (or fallback) ---
        if verbose and num_cores <=1 : # Only print if originally sequential
             print(f"Tracking {n_paths} paths sequentially...")
        elif verbose and num_cores > 1: # Print if falling back
             print(f"Falling back to sequential tracking for {n_paths} paths...")

        # Use tqdm for progress bar in sequential mode
        pbar = tqdm(total=n_paths, desc="Tracking Paths (Sequential)")
        for i, args in enumerate(tasks_args):
             # Set verbosity/debug for single path if overall verbose is true
             # Unpack args, modify verbose/debug, call worker directly
             (start_sys, target_sys, start_sol_seq, var_seq, tol_seq,
              min_step_seq, max_step_seq, gamma_seq, endgame_start_seq, store_paths_seq,
              use_endgame_seq, endgame_options_seq, _, _) = args # Discard original verbose/debug flags

             # Call worker function directly
             single_result = _track_single_path_parallel_worker(
                  (start_sys, target_sys, start_sol_seq, var_seq, tol_seq,
                   min_step_seq, max_step_seq, gamma_seq, endgame_start_seq, store_paths_seq,
                   use_endgame_seq, endgame_options_seq, # Pass endgame options
                   verbose and (n_paths <= 10), # Maybe only verbose single path if few paths total?
                   False # Keep debug off by default for single paths
                  )
             )
             results.append(single_result)
             pbar.update(1)
        pbar.close()

    # --- END PARALLELIZATION MODIFICATION ---

    # Unpack results
    end_solutions = [res[0] for res in results]
    path_results_list = [res[1] for res in results] # Renamed to avoid conflict

    # Add path index to results AFTER parallel processing (since order is preserved by map/imap)
    for i, info in enumerate(path_results_list):
         info['path_index'] = i


    # --- POST-PROCESSING (Endgame application moved into worker) ---
    # The endgame logic is now applied within the _track_single_path_parallel_worker
    # This avoids passing large system objects back from the pool.
    # We just collect the final results here.

    if verbose:
        elapsed = time.time() - start_time
        success_count = sum(1 for info in path_results_list if info.get('success', False))
        endgame_internal_count = sum(1 for info in path_results_list if info.get('endgame_used', False))
        endgame_post_count = sum(1 for info in path_results_list if info.get('endgame_applied_post_track', False))

        print(f"\nPath tracking complete in {elapsed:.2f} seconds.")
        print(f" - Successfully tracked: {success_count}/{n_paths} paths")
        if use_endgame:
             print(f" - Endgame used internally during tracking: {endgame_internal_count} paths")
             print(f" - Endgame applied after tracking: {endgame_post_count} paths")


    return end_solutions, path_results_list