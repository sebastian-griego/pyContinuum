"""
Main solver module for PyContinuum.

This module provides functions to solve polynomial systems using
homotopy continuation methods.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
# Import cpu_count to determine default cores
import multiprocessing
import os # For potentially better cpu_count

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem, polyvar, make_system # Added polyvar, make_system back
from pycontinuum.start_systems import generate_total_degree_start_system
from pycontinuum.tracking import track_paths


class Solution:
    """Class representing a solution to a polynomial system."""

    def __init__(self,
                 values: Dict[Variable, complex],
                 residual: float,
                 is_singular: bool = False,
                 path_index: Optional[int] = None,
                 path_info: Optional[Dict[str, Any]] = None): # Added path_info
        """Initialize a solution with its values and metadata.

        Args:
            values: Dictionary mapping variables to their values
            residual: Residual norm of the solution
            is_singular: Whether the solution is singular (Jacobian is rank-deficient)
            path_index: Index of the path that led to this solution (for tracking)
            path_info: Dictionary containing detailed results from the path tracker.
        """
        self.values = values
        self.residual = residual
        self.is_singular = is_singular # Note: Singularity is often determined by endgame/path tracker
        self.path_index = path_index
        # Store relevant info from path_results directly in the Solution object
        self.path_info = path_info if path_info else {}
        self.path_points = self.path_info.get('path_points', None) # Get path points if stored
        self.winding_number = self.path_info.get('winding_number', None) # Get winding number if available
        # Ensure is_singular reflects the final status from path_info
        self.is_singular = self.path_info.get('singular', is_singular)


    def __repr__(self) -> str:
        """String representation of the single solution."""
        # Update status based on final path_info if available
        status = "singular" if self.is_singular else "regular"
        if self.winding_number is not None:
             status += f" (w={self.winding_number})"

        var_strs = []

        # Sort variables by name for consistent output
        if hasattr(self, 'values') and isinstance(self.values, dict):
            # Ensure keys are Variable objects before sorting
            valid_keys = [k for k in self.values.keys() if isinstance(k, Variable)]
            sorted_vars = sorted(valid_keys, key=lambda v: v.name)
        else:
            return f"Solution object (incomplete data)"

        for var in sorted_vars:
            val = self.values.get(var, 0j) # Use .get for safety
            # Check if imaginary part is negligible
            imag_tol = 1e-9 # Use a slightly larger tolerance for display
            if abs(val.imag) < imag_tol:
                var_strs.append(f"{var.name} = {val.real:.8g}")
            else:
                sign = '+' if val.imag >= 0 else '-'
                var_strs.append(f"{var.name} = {val.real:.8g} {sign} {abs(val.imag):.8g}j")

        # Ensure self.residual exists before formatting
        residual_str = f"{self.residual:.2e}" if hasattr(self, 'residual') and self.residual is not None else 'N/A'

        # Add path status message if available
        status_msg = self.path_info.get('status_message', '')
        repr_str = f"Solution (Index: {self.path_index}, Status: {status}, Residual: {residual_str})"
        if status_msg:
            repr_str += f"\n  Tracker Status: {status_msg}"
        repr_str += "\n  Values:\n    " + "\n    ".join(var_strs)

        return repr_str


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
        try:
             # Ensure values are complex numbers before checking .imag
             return all(abs(complex(val).imag) < tol for val in self.values.values())
        except (TypeError, ValueError):
             return False # Handle non-numeric values gracefully

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
        if not self.is_real(tol=tol):
             return False
        try:
             # Ensure values are numeric before checking .real
             return all(complex(val).real > -tol for val in self.values.values())
        except (TypeError, ValueError):
             return False # Handle non-numeric values gracefully


    def distance(self, other: 'Solution', variables: List[Variable]) -> float:
        """Compute the Euclidean distance between this solution and another."""
        dist_sq = 0
        # Check attributes exist before access
        if not hasattr(self, 'values') or not isinstance(self.values, dict) or \
           not hasattr(other, 'values') or not isinstance(other.values, dict):
            return float('inf')

        for var in variables:
            # Use .get with default 0j for safety if a variable is missing in one solution
            val1 = complex(self.values.get(var, 0j))
            val2 = complex(other.values.get(var, 0j))
            dist_sq += abs(val1 - val2)**2
        return np.sqrt(dist_sq)

class SolutionSet:
    """Class representing a set of solutions to a polynomial system."""

    def __init__(self, solutions: List[Solution], system: PolynomialSystem, variables: List[Variable]): # Added variables
        """Initialize a solution set.
        Args:
            solutions: List of Solution objects
            system: The polynomial system that was solved
            variables: The list of variables used in solving
        """
        self.solutions = solutions
        self.system = system
        self.variables = variables # Store variables for potential later use
        self._meta = {}  # For storing metadata about the solve process

    def __repr__(self) -> str:
        """String representation of the solution set."""
        # Use default tolerance for counts in representation
        real_count = sum(1 for sol in self.solutions if sol.is_real())
        singular_count = sum(1 for sol in self.solutions if sol.is_singular)

        # Check if this set resulted from filtering
        is_filtered = self._meta.get('is_filtered', False)
        set_type = "Filtered SolutionSet" if is_filtered else "SolutionSet"

        header = f"{set_type}: {len(self.solutions)} solutions found"
        if len(self.solutions) > 0:
             header += f" ({real_count} real, {singular_count} singular)"

        # Display metadata about the original solve process if available
        if not is_filtered and 'total_paths' in self._meta:
             raw_found = self._meta.get('raw_solutions_found', len(self.solutions)) # Use raw count if available
             header += f"\n  Result of tracking {self._meta['total_paths']} paths."
             header += f"\n  Initially found {raw_found} potential solutions before deduplication."
             header += f"\n  {len(self.solutions)} distinct solutions remain."
        elif is_filtered and 'total_paths' in self._meta:
             # For filtered sets, just mention the original tracking stats
             header += f"\n  (Filtered from solve process that tracked {self._meta['total_paths']} paths)"


        if 'solve_time' in self._meta:
            header += f"\n  Solve time: {self._meta['solve_time']:.2f} seconds"
        if 'successful_paths' in self._meta:
             header += f"\n  Paths successfully tracked: {self._meta['successful_paths']}/{self._meta.get('total_paths', '?')}"
        if 'num_cores_used' in self._meta:
             cores_used = self._meta['num_cores_used']
             run_type = "sequentially" if cores_used == 1 else f"in parallel ({cores_used} cores)"
             header += f"\n  Run {run_type}"


        # Display solutions
        solution_details = ""
        max_solutions_to_print = 5 # Limit printed solutions
        if not self.solutions:
            solution_details = "\n(No solutions in this set)"
        else:
            solution_details = "\n\n--- Solutions ---"
            # Sort solutions by path index for consistency
            sorted_solutions = sorted(self.solutions, key=lambda s: s.path_index if s.path_index is not None else float('inf'))

            num_to_print = min(len(sorted_solutions), max_solutions_to_print)
            for i in range(num_to_print):
                 solution_details += f"\n\n{repr(sorted_solutions[i])}" # Use repr for detailed view

            if len(self.solutions) > num_to_print:
                 solution_details += f"\n\n... and {len(self.solutions) - num_to_print} more solutions."

        return header + solution_details

    def __len__(self) -> int:
        """Get the number of solutions."""
        return len(self.solutions)

    def __getitem__(self, index) -> Solution:
        """Get a solution by index."""
        # Should probably return from sorted list if sorting is standard?
        # For now, return from original list.
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

        # Apply positive filter *after* real filter if both are used
        if positive is True:
             # is_positive() implicitly checks is_real()
            filtered_sols = [sol for sol in filtered_sols if sol.is_positive(tol=tol)]
        elif positive is False:
            # Careful here: non-positive could mean complex or negative real.
            # Let's interpret as "real but not positive" OR "not real".
            # If real=True was also passed, it means "real and non-positive"
            if real is True:
                 # Already filtered for real, now filter out positive ones
                 filtered_sols = [sol for sol in filtered_sols if not sol.is_positive(tol=tol)]
            else:
                 # Include complex numbers and non-positive reals
                 filtered_sols = [sol for sol in filtered_sols if not sol.is_positive(tol=tol)]


        if max_residual is not None:
            filtered_sols = [sol for sol in filtered_sols if sol.residual is not None and sol.residual <= max_residual]

        if custom_filter is not None:
            # Add error handling for custom filter
            valid_filtered_sols = []
            for sol in filtered_sols:
                 try:
                      if custom_filter(sol):
                           valid_filtered_sols.append(sol)
                 except Exception as e:
                      print(f"Warning: Custom filter failed for solution {sol.path_index}: {e}")
            filtered_sols = valid_filtered_sols


        # Pass variables to the new SolutionSet
        result = SolutionSet(filtered_sols, self.system, self.variables)
        result._meta = self._meta.copy() # Copy metadata
        result._meta['is_filtered'] = True # Mark as filtered
        return result

    # --- Added Methods for SolutionSet ---
    def real_solutions(self, tol: float = 1e-10) -> 'SolutionSet':
         """Return a new SolutionSet containing only real solutions."""
         return self.filter(real=True, tol=tol)

    def positive_solutions(self, tol: float = 1e-10) -> 'SolutionSet':
         """Return a new SolutionSet containing only positive real solutions."""
         return self.filter(positive=True, tol=tol)
    # ----------------------------------


def solve(system: PolynomialSystem,
          start_system=None,
          start_solutions=None,
          variables=None,
          tol: float = 1e-8, # Adjusted default tol slightly
          num_cores: Optional[int] = 1, # Added num_cores option
          verbose: bool = False,
          store_paths: bool = False,
          use_endgame: bool = True,
          endgame_options: Optional[Dict[str, Any]] = None,
          deduplication_tol_factor: float = 100.0, # Increased factor for dedupe
          singular_deduplication_tol: float = 1e-4) -> SolutionSet: # Increased singular dedupe tol
    """Solve a polynomial system using homotopy continuation, with parallel path tracking.

    Args:
        system: The polynomial system to solve
        start_system: Optional custom start system (default: total-degree homotopy)
        start_solutions: Optional known solutions of the start system
        variables: Optional list of variables to use (default: extracted from system)
        tol: Tolerance for path tracking and solution classification (Default: 1e-8)
        num_cores: Number of CPU cores for parallel path tracking.
                   Set to 1 for sequential. Set to None or 0 to use all cores. (Default: 1)
        verbose: Whether to print progress information
        store_paths: Whether to store path tracking points (memory intensive)
        use_endgame: Whether to use endgame methods for singular solutions
        endgame_options: Optional dictionary of options for the endgame procedure
        deduplication_tol_factor: Factor multiplied by `tol` for regular solution deduplication.
        singular_deduplication_tol: Absolute tolerance for singular solution deduplication.

    Returns:
        A SolutionSet containing all found distinct solutions
    """
    start_time = time.time()

    # --- Determine number of cores ---
    if num_cores is None or num_cores == 0:
        actual_cores = multiprocessing.cpu_count()
    else:
        actual_cores = max(1, int(num_cores)) # Ensure at least 1 core

    # Get the variables in the system
    if variables is None:
        # Extract variables from the system, ensure consistent order
        vars_set = system.variables()
        # Sort variables by name for reproducibility
        variables = sorted(list(vars_set), key=lambda v: v.name)

    if not variables:
         raise ValueError("Could not extract variables from the polynomial system.")

    if verbose:
        print(f"--- Starting PyContinuum Solver ---")
        print(f"System variables: {', '.join(v.name for v in variables)}")
        print(f"Tolerance: {tol:.2e}")
        print(f"Parallel cores requested: {num_cores} (Actual cores to use: {actual_cores})")
        print(f"Use endgame: {use_endgame}")
        print(f"Store paths: {store_paths}")


    # If no start system provided, generate a total-degree system
    if start_system is None or start_solutions is None:
        if verbose:
            print("Generating total-degree start system...")
        try:
             start_system, start_solutions = generate_total_degree_start_system(system, variables)
             if verbose:
                  degrees = system.degrees()
                  total_paths = np.prod(degrees) if degrees else 0
                  print(f"Using total-degree homotopy.")
                  print(f" - System degrees: {degrees}")
                  print(f" - Expected paths (Bézout bound): {total_paths}")
        except ValueError as e:
             print(f"Error generating start system: {e}")
             # Return an empty solution set if start system fails
             return SolutionSet([], system, variables)
        except Exception as e:
             print(f"Unexpected error during start system generation: {e}")
             return SolutionSet([], system, variables)


    if not start_solutions:
         if verbose:
              print("Warning: No start solutions provided or generated. Cannot track paths.")
         return SolutionSet([], system, variables)

    # Track the paths from start solutions to the target system
    if verbose:
        print(f"Tracking {len(start_solutions)} paths...")

    # --- Call track_paths with parallelization ---
    # Define tracking parameters (can be customized)
    track_tol = tol
    min_step = tol * 1e-2 # Example: relate min step to tol
    max_step = 0.1
    endgame_start_t = 0.01 # t value to start considering endgame
    gamma_val = complex(np.random.rand(), np.random.rand()) # Generate random gamma

    # Default endgame options if none provided
    default_endgame_opts = {
         'abstol': track_tol,
         'geometric_series_factor': 0.5,
         'max_winding_number': 16, # From report
         'gamma': gamma_val # Pass the same gamma
    }
    final_endgame_options = default_endgame_opts.copy()
    if endgame_options:
         final_endgame_options.update(endgame_options)


    try:
        end_points, path_results_list = track_paths(
            start_system=start_system,
            target_system=system,
            start_solutions=start_solutions,
            variables=variables,
            tol=track_tol,
            min_step_size=min_step,
            max_step_size=max_step,
            gamma=gamma_val,
            endgame_start=endgame_start_t,
            num_cores=actual_cores, # Pass core count here
            verbose=verbose,
            store_paths=store_paths,
            use_endgame=use_endgame,
            endgame_options=final_endgame_options # Pass options
        )
    except Exception as e:
         print(f"\nCritical error during path tracking: {e}")
         # Return empty set or partial results? For now, return empty.
         result = SolutionSet([], system, variables)
         result._meta['error'] = str(e)
         return result
    # -----------------------------------------

    # Process the raw solutions
    raw_solutions: List[Solution] = []

    if verbose:
        print("Processing and classifying solutions...")

    # Function to compute residual (using the target system)
    def compute_residual(sol_values_dict):
        # Ensure all variables are present, default to 0 if not (shouldn't happen ideally)
        full_values = {var: complex(sol_values_dict.get(var, 0j)) for var in variables}
        try:
            eval_result = system.evaluate(full_values)
            return np.linalg.norm(eval_result)
        except Exception as e:
            # Handle potential evaluation errors
            if verbose: print(f"Warning: Error computing residual for a solution: {e}")
            return float('inf') # Assign infinite residual if evaluation fails


    successful_paths_count = 0
    failed_paths_count = 0

    for i, path_info in enumerate(path_results_list):
        endpoint = end_points[i] # Get the endpoint corresponding to this path result

        # Check path success based on the 'success' flag in path_info
        if not path_info.get('success', False):
             failed_paths_count += 1
             # Optionally log failed paths if verbose
             # if verbose: print(f"Path {i} failed: {path_info.get('status_message', 'No message')}")
             continue

        successful_paths_count += 1

        # Create solution dictionary from the final endpoint
        # Ensure endpoint has the correct dimension
        if len(endpoint) != len(variables):
             if verbose: print(f"Warning: Skipping path {i}. Endpoint dimension mismatch ({len(endpoint)}) vs variables ({len(variables)}).")
             failed_paths_count += 1
             successful_paths_count -= 1 # Decrement success count
             continue

        solution_dict = {var: complex(val) for var, val in zip(variables, endpoint)}

        # Compute residual
        residual = compute_residual(solution_dict)

        # Define residual tolerance (allow slightly higher than tracking tol)
        residual_tol = 1000 * tol # More generous residual tolerance

        if residual > residual_tol or not np.isfinite(residual):
             if verbose:
                 # Only print warning if path was marked as success initially
                 if path_info.get('success', False):
                      print(f"Warning: Path {i} marked success, but has high/invalid residual ({residual:.2e}). Skipping.")
             failed_paths_count += 1
             successful_paths_count -= 1 # Decrement success count
             # Update path info status if residual is the reason for failure
             path_info['success'] = False
             path_info['status_message'] = f'Failed residual check (Residual: {residual:.2e})'
             continue


        # Create the Solution object, passing the full path_info
        solution = Solution(
            values=solution_dict,
            residual=residual,
            path_index=i,
            path_info=path_info # Pass the detailed dictionary
        )
        # Singularity status is now handled within the Solution init using path_info

        raw_solutions.append(solution)


    # --- Deduplication ---
    unique_solutions: List[Solution] = []
    if not raw_solutions:
         if verbose: print("No valid raw solutions found after processing paths.")
    else:
        if verbose:
            print(f"Deduplicating {len(raw_solutions)} raw solutions...")

        # Sort raw solutions primarily by singularity status (regular first), then by residual
        raw_solutions.sort(key=lambda s: (s.is_singular, s.residual))

        # Use separate tolerances for regular and singular solutions
        regular_dedupe_tol = tol * deduplication_tol_factor
        singular_dedupe_tol_abs = singular_deduplication_tol # Absolute tolerance for singular

        # Keep track of representatives for deduplication
        representatives: List[Solution] = []

        for sol in raw_solutions:
            is_duplicate = False
            for existing_rep in representatives:
                # Calculate distance
                try:
                    dist = sol.distance(existing_rep, variables)
                except Exception as e:
                     if verbose: print(f"Warning: Error calculating distance for solution {sol.path_index}: {e}")
                     dist = float('inf') # Treat as non-duplicate if distance fails


                # Determine appropriate tolerance based on singularity
                # If either solution is singular, use the singular tolerance
                use_singular_tol = sol.is_singular or existing_rep.is_singular
                current_tol = singular_dedupe_tol_abs if use_singular_tol else regular_dedupe_tol

                if dist < current_tol:
                    is_duplicate = True
                    # Optional: Refine representative? (e.g., keep the one with lower residual)
                    # If current sol has lower residual, replace representative
                    if sol.residual < existing_rep.residual:
                         # Find index of existing_rep and replace it
                         try:
                              rep_index = representatives.index(existing_rep)
                              representatives[rep_index] = sol
                         except ValueError: # Should not happen if existing_rep is from the list
                              pass
                    # If singular solutions are merged, could potentially sum winding numbers, etc.
                    break # Stop checking against other representatives

            if not is_duplicate:
                unique_solutions.append(sol)
                representatives.append(sol) # Add as a new representative

    # Create the final result SolutionSet, passing variables
    result = SolutionSet(unique_solutions, system, variables)

    # Add metadata
    result._meta['total_paths'] = len(start_solutions)
    result._meta['successful_paths'] = successful_paths_count # Use count after residual check
    result._meta['failed_paths'] = failed_paths_count # Use count after residual check
    result._meta['raw_solutions_found'] = len(raw_solutions) # Count before deduplication
    result._meta['solve_time'] = time.time() - start_time
    result._meta['num_cores_used'] = actual_cores # Store cores used
    result._meta['tolerance'] = tol
    result._meta['deduplication_tolerance_regular'] = regular_dedupe_tol
    result._meta['deduplication_tolerance_singular'] = singular_dedupe_tol_abs


    if verbose:
        print(f"--- Solver Finished ---")
        print(f"Total time: {result._meta['solve_time']:.2f} seconds")
        print(f"Paths tracked: {result._meta['total_paths']}")
        print(f" - Successful: {result._meta['successful_paths']}")
        print(f" - Failed: {result._meta['failed_paths']}")
        print(f"Raw solutions found (passed residual check): {result._meta['raw_solutions_found']}")
        print(f"Distinct solutions found (after deduplication): {len(unique_solutions)}")
        print(f"------------------------")


    return result


# Keep polyvar definition if it's part of the public API exposed by solver.py
# Or ensure it's imported correctly if moved to __init__.py's __all__
# Re-adding polyvar and make_system here as they were present originally
def polyvar(*names: str) -> Union[Variable, Tuple[Variable, ...]]:
    """Create polynomial variables with the given names.

    Args:
        *names: Variable names

    Returns:
        A single Variable or a tuple of Variables
    """
    variables = tuple(Variable(name) for name in names)
    return variables[0] if len(variables) == 1 else variables

def make_system(*equations) -> "PolynomialSystem":
     """Create a polynomial system from various types of equations.

     Args:
         *equations: Polynomial equations to include in the system

     Returns:
         A PolynomialSystem object containing the processed equations

     Raises:
         TypeError: If an equation cannot be converted to a Polynomial
     """
     # This function was in polynomial.py originally, but maybe intended here?
     # Let's use the one from polynomial.py to avoid duplication.
     # Ensure it's imported: from pycontinuum.polynomial import make_system
     # No, it was defined here originally, let's keep it for now.
     processed_equations = []
     for eq in equations:
         if isinstance(eq, Polynomial):
             processed_equations.append(eq)
         elif isinstance(eq, (Monomial, Variable)):
             # Convert Monomial or Variable to Polynomial
             processed_equations.append(Polynomial([eq]))
         else:
             # Attempt conversion for numeric types or assume it's convertable
             try:
                 processed_equations.append(Polynomial([Monomial({}, coefficient=complex(eq))]))
             except TypeError:
                 raise TypeError(f"Cannot convert {type(eq)} to polynomial equation")

     return PolynomialSystem(processed_equations)