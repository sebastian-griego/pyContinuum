"""
Main solver module for PyContinuum.

This module provides functions to solve polynomial systems using
homotopy continuation methods.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem
from pycontinuum.start_systems import generate_total_degree_start_system
from pycontinuum.tracking import track_paths


class Solution:
    """Class representing a solution to a polynomial system."""

    def __init__(self, 
                 values: Dict[Variable, complex],
                 residual: float,
                 is_singular: bool = False,
                 path_index: Optional[int] = None):
        """Initialize a solution with its values and metadata.
        
        Args:
            values: Dictionary mapping variables to their values
            residual: Residual norm of the solution
            is_singular: Whether the solution is singular (Jacobian is rank-deficient)
            path_index: Index of the path that led to this solution (for tracking)
        """
        self.values = values
        self.residual = residual
        self.is_singular = is_singular
        self.path_index = path_index
        self.path_points = None
        self.winding_number = None

    def __repr__(self) -> str:
        """String representation of the single solution."""
        status = "singular" if self.is_singular else "regular"
        var_strs = []

        # Sort variables by name for consistent output
        if hasattr(self, 'values') and isinstance(self.values, dict):
            sorted_vars = sorted(self.values.keys(), key=lambda v: v.name if hasattr(v, 'name') else '')
        else:
            return f"Solution object (incomplete data)"

        for var in sorted_vars:
            val = self.values[var]
            if abs(val.imag) < 1e-10:
                var_strs.append(f"{var.name} = {val.real:.8g}")
            else:
                sign = '+' if val.imag >= 0 else '-'
                var_strs.append(f"{var.name} = {val.real:.8g} {sign} {abs(val.imag):.8g}j")

        # Ensure self.residual exists before formatting
        residual_str = f"{self.residual:.2e}" if hasattr(self, 'residual') else 'N/A'
        return f"Solution ({status}, residual={residual_str}):\n  " + "\n  ".join(var_strs)

    def is_real(self, tol: float = 1e-10) -> bool:
        """Check if the solution is real (all imaginary parts close to zero).

        Args:
            tol: Tolerance for imaginary parts

        Returns:
            True if all values have imaginary parts less than tol
        """
        # Check if self.values exists before iterating
        if not hasattr(self, 'values') or not isinstance(self.values, dict):
            return False
        return all(abs(val.imag) < tol for val in self.values.values())

    def is_positive(self, tol: float = 1e-10) -> bool:
        """Check if the solution is positive real.
        
        Args:
            tol: Tolerance for imaginary parts and negative reals
            
        Returns:
            True if solution is real and all values have positive real parts
        """
        # Check if self.values exists before iterating
        if not hasattr(self, 'values') or not isinstance(self.values, dict):
            return False
        # Ensure is_real is called correctly now
        return self.is_real(tol=tol) and all(val.real > -tol for val in self.values.values())

    def distance(self, other: 'Solution', variables: List[Variable]) -> float:
        """Compute the Euclidean distance between this solution and another."""
        dist_sq = 0
        # Check attributes exist before access
        if not hasattr(self, 'values') or not isinstance(self.values, dict) or \
           not hasattr(other, 'values') or not isinstance(other.values, dict):
            return float('inf')

        for var in variables:
            dist_sq += abs(self.values.get(var, 0) - other.values.get(var, 0))**2
        return np.sqrt(dist_sq)

class SolutionSet:
    """Class representing a set of solutions to a polynomial system."""
    
    def __init__(self, solutions: List[Solution], system: PolynomialSystem):
        """Initialize a solution set.
        Args:
            solutions: List of Solution objects
            system: The polynomial system that was solved
        """
        self.solutions = solutions
        self.system = system
        self._meta = {}  # For storing metadata about the solve process
        
    def __repr__(self) -> str:
        """String representation of the solution set."""
        # Use default tolerance for counts in representation
        real_count = sum(1 for sol in self.solutions if sol.is_real())
        singular_count = sum(1 for sol in self.solutions if sol.is_singular)

        # Check if this set resulted from filtering
        is_filtered = self._meta.get('is_filtered', False)
        set_type = "Filtered SolutionSet" if is_filtered else "SolutionSet"

        header = f"{set_type}: {len(self.solutions)} solutions ({real_count} real, {singular_count} singular)"

        # Display metadata about the original solve process if available
        # Avoid stating "found {len(self.solutions)}" if it's filtered, as that's confusing
        if not is_filtered and 'total_paths' in self._meta:
             # Only show "found N" for the original, unfiltered set
             header += f"\n  Result of tracking {self._meta['total_paths']} paths, found {self._meta.get('raw_solutions_found', '?')} raw, {len(self.solutions)} distinct solutions after deduplication."
        elif 'total_paths' in self._meta:
             # For filtered sets, just mention the original tracking stats
             header += f"\n  (Filtered from solve process that tracked {self._meta['total_paths']} paths)"

        if 'solve_time' in self._meta:
            header += f"\n  Solve time: {self._meta['solve_time']:.2f} seconds"
        if 'successful_paths' in self._meta:
             header += f"\n  Paths successfully tracked: {self._meta['successful_paths']}/{self._meta.get('total_paths', '?')}"

        # Display solutions
        solution_details = ""
        if not self.solutions:
            solution_details = "\n(No solutions in this set)"
        elif len(self.solutions) <= 5:
            solution_details = "\n\n" + "\n\n".join(str(sol) for sol in self.solutions)
        else:
            # Otherwise just print the first 3
            solution_details = "\n\n" + "\n\n".join(str(sol) for sol in self.solutions[:3]) + "\n\n... and {} more".format(len(self.solutions) - 3)

        return header + solution_details
    
    def __len__(self) -> int:
        """Get the number of solutions."""
        return len(self.solutions)
    
    def __getitem__(self, index) -> Solution:
        """Get a solution by index."""
        return self.solutions[index]
    
    def filter(self,
               real: Optional[bool] = None,
               positive: Optional[bool] = None,
               tol: float = 1e-10,
               max_residual: Optional[float] = None,
               custom_filter: Optional[Callable[[Solution], bool]] = None) -> 'SolutionSet':
        """Filter solutions based on criteria.

        Args:
            real: If True, only include real solutions. If False, only non-real. If None, no filter.
            positive: If True, only include positive real solutions. If False, only non-positive real. If None, no filter.
            tol: Tolerance used for real and positive checks (default: 1e-10).
            max_residual: Maximum residual threshold.
            custom_filter: Custom filter function taking a Solution and returning bool.

        Returns:
            A new SolutionSet with filtered solutions.
        """
        filtered_sols = self.solutions

        if real is True:
            filtered_sols = [sol for sol in filtered_sols if sol.is_real(tol=tol)]
        elif real is False:
            filtered_sols = [sol for sol in filtered_sols if not sol.is_real(tol=tol)]

        if positive is True:
            filtered_sols = [sol for sol in filtered_sols if sol.is_positive(tol=tol)]
        elif positive is False:
            filtered_sols = [sol for sol in filtered_sols if not sol.is_positive(tol=tol)]

        if max_residual is not None:
            filtered_sols = [sol for sol in filtered_sols if sol.residual <= max_residual]

        if custom_filter is not None:
            filtered_sols = [sol for sol in filtered_sols if custom_filter(sol)]

        result = SolutionSet(filtered_sols, self.system)
        result._meta = self._meta.copy()
        result._meta['is_filtered'] = True
        return result

def solve(system: PolynomialSystem,
          start_system=None,
          start_solutions=None,
          variables=None,
          tol: float = 1e-10,
          verbose: bool = False,
          store_paths: bool = False,
          use_endgame: bool = True,
          endgame_options: Optional[Dict[str, Any]] = None,
          deduplication_tol_factor: float = 10.0,
          singular_deduplication_tol: float = 1e-3,
          **extra_tracker_options: Any) -> SolutionSet:
    """Solve a polynomial system using homotopy continuation.
    Args:
        system: The polynomial system to solve
        start_system: Optional custom start system (default: total-degree homotopy)
        start_solutions: Optional known solutions of the start system
        variables: Optional list of variables to use (default: extracted from system)
        tol: Tolerance for path tracking and solution classification
        verbose: Whether to print progress information
        store_paths: Whether to store path tracking points
        use_endgame: Whether to use endgame methods for singular solutions
        endgame_options: Optional dictionary of options for the endgame procedure
        deduplication_tol_factor: Factor multiplied by `tol` for regular solution deduplication.
        singular_deduplication_tol: Absolute tolerance for singular solution deduplication.
        **extra_tracker_options: Additional keyword arguments passed directly to track_paths.

    Returns:
        A SolutionSet containing all found solutions
    """
    start_time = time.time()

    if variables is None:
        variables = list(system.variables())

    if verbose:
        print(f"Variables used for solving: {variables}")

    # Check if system is square
    n_eqs = len(system.equations)
    n_vars = len(variables)
    if n_eqs != n_vars:
        raise ValueError(f"solver.solve currently requires a square system "
                         f"({n_eqs} equations, {n_vars} variables). "
                         "Use witness set methods for non-square systems.")

    # If no start system provided, generate a total-degree system
    if start_system is None or start_solutions is None:
        if verbose:
            print("Generating total-degree start system...")
        start_system, start_solutions = generate_total_degree_start_system(
            system, variables
        )
        if verbose:
            degrees = system.degrees()
            total_paths = np.prod(degrees) if degrees else 0
            print(f"Using total-degree homotopy with {total_paths} start paths ({' * '.join(map(str, degrees))})")

    # Track the paths from start solutions to the target system
    if verbose:
        print(f"Tracking {len(start_solutions)} paths...")

    tracker_options = {
        'tol': tol,
        'verbose': verbose,
        'store_paths': store_paths,
        'use_endgame': use_endgame,
        'endgame_options': endgame_options,
        **extra_tracker_options
    }

    end_solutions, path_results = track_paths(
        start_system=start_system,
        target_system=system,
        start_solutions=start_solutions,
        variables=variables,
        **tracker_options
    )

    raw_solutions = []
    if verbose:
        print("Processing and classifying solutions...")

    def compute_residual(sol_values):
        eval_result = system.evaluate(sol_values)
        return np.linalg.norm(np.array(eval_result, dtype=complex))

    successful_paths = 0
    failed_paths = 0
    for i, path_info in enumerate(path_results):
        if not isinstance(path_info, dict) or 'success' not in path_info:
            print(f"Warning: Invalid path result format for path {i}. Skipping.")
            failed_paths += 1
            continue

        endpoint = end_solutions[i]

        if not path_info.get('success', False):
            failed_paths += 1
            continue

        successful_paths += 1

        if not isinstance(endpoint, np.ndarray) or not np.all(np.isfinite(endpoint)):
            print(f"Warning: Invalid endpoint for successful path {i}. Skipping.")
            successful_paths -= 1
            failed_paths += 1
            continue

        final_point = np.array(path_info.get('final_point', endpoint), dtype=complex)

        solution_dict = {var: val for var, val in zip(variables, final_point)}
        try:
            residual = compute_residual(solution_dict)
        except Exception as e:
            print(f"Warning: Error computing residual for path {i}: {e}. Skipping.")
            successful_paths -= 1
            failed_paths += 1
            continue

        if residual > 100 * tol:
            if verbose:
                print(f"Skipping solution from path {i} due to large residual ({residual:.2e} > {100*tol:.2e})")
            successful_paths -= 1
            failed_paths += 1
            continue

        is_singular = path_info.get('singular', False)
        winding_number = path_info.get('winding_number', None)

        solution = Solution(
            values=solution_dict,
            residual=residual,
            is_singular=is_singular,
            path_index=i
        )

        if winding_number is not None:
            solution.winding_number = winding_number

        if store_paths and path_info.get('path_points'):
            solution.path_points = path_info['path_points']

        raw_solutions.append(solution)

    unique_solutions = []
    unique_representatives: List[Solution] = []
    current_regular_tol = tol * deduplication_tol_factor
    current_singular_tol = singular_deduplication_tol

    if verbose:
        print(f"Attempting to deduplicate {len(raw_solutions)} raw solutions "
              f"(reg_tol={current_regular_tol:.2e}, sing_tol={current_singular_tol:.2e})...")

    for sol in raw_solutions:
        is_duplicate = False
        for existing_sol in unique_representatives:
            dist = sol.distance(existing_sol, variables)
            use_tol = current_singular_tol if sol.is_singular and existing_sol.is_singular else current_regular_tol
            if dist < use_tol:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_solutions.append(sol)
            unique_representatives.append(sol)

    result = SolutionSet(unique_solutions, system)

    result._meta['total_paths'] = len(start_solutions)
    result._meta['successful_paths'] = successful_paths
    result._meta['failed_paths'] = failed_paths
    result._meta['solve_time'] = time.time() - start_time
    result._meta['raw_solutions_found'] = len(raw_solutions)

    if verbose:
        print(f"Found {len(unique_solutions)} distinct solutions (from {len(raw_solutions)} raw).")
        print(f"Solve time: {result._meta['solve_time']:.2f}s. Successful paths: {successful_paths}/{len(start_solutions)}.")

    return result