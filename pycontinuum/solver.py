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
        self.path_points = None  # Add this line
        self.winding_number = None  # Add this
        
    def __repr__(self) -> str:
        """String representation of the solution."""
        status = "singular" if self.is_singular else "regular"
        var_strs = []
        
        # Sort variables by name for consistent output
        sorted_vars = sorted(self.values.keys(), key=lambda v: v.name)
        
        for var in sorted_vars:
            val = self.values[var]
            # Format complex numbers nicely
            if abs(val.imag) < 1e-10:  # Small imaginary part, treat as real
                var_strs.append(f"{var.name} = {val.real:.8g}")
            else:
                var_strs.append(f"{var.name} = {val.real:.8g} + {val.imag:.8g}j")
                
        return f"Solution ({status}, residual={self.residual:.2e}):\n  " + "\n  ".join(var_strs)
    
    def is_real(self, tol: float = 1e-10) -> bool:
        """Check if the solution is real (all imaginary parts close to zero).
        
        Args:
            tol: Tolerance for imaginary parts
            
        Returns:
            True if all values have imaginary parts less than tol
        """
        return all(abs(val.imag) < tol for val in self.values.values())
    
    def is_positive(self, tol: float = 1e-10) -> bool:
        """Check if the solution is positive real.
        
        Args:
            tol: Tolerance for imaginary parts and negative reals
            
        Returns:
            True if solution is real and all values have positive real parts
        """
        return self.is_real(tol) and all(val.real > -tol for val in self.values.values())
    
    def distance(self, other: 'Solution', variables: List[Variable]) -> float:
        """Compute the Euclidean distance between this solution and another."""
        dist_sq = 0
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
        real_count = sum(1 for sol in self.solutions if sol.is_real())
        singular_count = sum(1 for sol in self.solutions if sol.is_singular)
        
        header = f"SolutionSet: {len(self.solutions)} solutions ({real_count} real, {singular_count} singular)"
        
        # If we have solve metadata, include it
        if self._meta:
            if 'total_paths' in self._meta:
                header += f"\nTracked {self._meta['total_paths']} paths, found {len(self.solutions)} distinct solutions"
            if 'solve_time' in self._meta:
                header += f"\nSolve time: {self._meta['solve_time']:.2f} seconds"
                
        # If few solutions, print them all
        if len(self.solutions) <= 5:
            return header + "\n\n" + "\n\n".join(str(sol) for sol in self.solutions)
        else:
            # Otherwise just print the first 3
            return header + "\n\n" + "\n\n".join(str(sol) for sol in self.solutions[:3]) + "\n\n... and more"
    
    def __len__(self) -> int:
        """Get the number of solutions."""
        return len(self.solutions)
    
    def __getitem__(self, index) -> Solution:
        """Get a solution by index."""
        return self.solutions[index]
    
    def filter(self, 
               real: bool = False, 
               positive: bool = False,
               max_residual: Optional[float] = None,
               custom_filter: Optional[Callable[[Solution], bool]] = None) -> 'SolutionSet':
        """Filter solutions based on criteria.
        
        Args:
            real: Only include real solutions
            positive: Only include positive real solutions
            max_residual: Maximum residual threshold
            custom_filter: Custom filter function taking a Solution and returning bool
            
        Returns:
            A new SolutionSet with filtered solutions
        """
        filtered = self.solutions.copy()
        
        if real:
            filtered = [sol for sol in filtered if sol.is_real()]
            
        if positive:
            filtered = [sol for sol in filtered if sol.is_positive()]
            
        if max_residual is not None:
            filtered = [sol for sol in filtered if sol.residual <= max_residual]
            
        if custom_filter is not None:
            filtered = [sol for sol in filtered if custom_filter(sol)]
            
        result = SolutionSet(filtered, self.system)
        result._meta = self._meta.copy()
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
          singular_deduplication_tol: float = 1e-3) -> SolutionSet:
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
        
    Returns:
        A SolutionSet containing all found solutions
    """
    start_time = time.time()
    
    # Get the variables in the system
    if variables is None:
        variables = list(system.variables())
        
    if verbose:
        print(f"Variables used for solving: {variables}")
    
    # If no start system provided, generate a total-degree system
    if start_system is None or start_solutions is None:
        if verbose:
            print("Generating total-degree start system...")
        start_system, start_solutions = generate_total_degree_start_system(system, variables)
        
        if verbose:
            degrees = system.degrees()
            total_paths = np.prod(degrees)
            print(f"Using total-degree homotopy with {total_paths} start paths ({' * '.join(map(str, degrees))})")
    
    # Track the paths from start solutions to the target system
    if verbose:
        print(f"Tracking {len(start_solutions)} paths...")
    
    end_solutions, path_results = track_paths(
        start_system=start_system,
        target_system=system,
        start_solutions=start_solutions,
        variables=variables,
        tol=tol,
        verbose=verbose,
        store_paths=store_paths,
        use_endgame=use_endgame,
        endgame_options=endgame_options
    )
    # Process the raw solutions
    raw_solutions = []
    
    if verbose:
        print("Processing and classifying solutions...")
    
    # Function to compute residual
    def compute_residual(sol_values):
        return np.linalg.norm(system.evaluate(sol_values))
    
    # Process each end solution and its path result info
    successful_paths = 0
    failed_paths = 0

    # path_results should now be a list of dictionaries, one for each path
    # Each dict contains keys like 'success', 'singular', 'endgame_used', 'winding_number', 'predictions', etc.
    for i, path_result_info in enumerate(path_results):
        endpoint = end_solutions[i] # Get the endpoint corresponding to this path result

        # Check if the path was successful (either tracker reached t=0 or endgame succeeded)
        if not path_result_info.get('success', False):
             failed_paths += 1
             continue

        successful_paths += 1

        # Use the final point from the path or endgame prediction
        final_point = np.array(path_result_info.get('final_point', endpoint), dtype=complex) # Assume track_paths adds 'final_point'

        # Create Solution object
        solution_dict = {var: val for var, val in zip(variables, final_point)}
        residual = compute_residual(solution_dict)

        # Skip solutions with large residuals (adjust tolerance if endgame was used?)
        # Maybe the endgame success check is sufficient, but keeping this as a safeguard
        if residual > 100 * tol: # Consider a different tolerance if endgame was used
             if verbose:
                 print(f"Skipping solution from path {i} due to large residual ({residual:.2e})")
             failed_paths += 1 # Count as failed path if residual is too high
             continue

        # Determine singularity status and winding number from path_result_info
        is_singular = path_result_info.get('singular', False)
        winding_number = path_result_info.get('winding_number', None)

        solution = Solution(
            values=solution_dict,
            residual=residual,
            is_singular=is_singular,
            path_index=i
        )

        # Store winding number if available
        if winding_number is not None:
             solution.winding_number = winding_number

        # Store path points if available
        if store_paths and path_result_info.get('path_points'): # Access path_points from the specific path's result info
            solution.path_points = path_result_info['path_points']

        raw_solutions.append(solution)

    # Remove duplicate solutions using a proximity-based grouping
    unique_solutions = []
    # Use a list to store representatives of unique solutions found so far
    unique_representatives: List[Solution] = []

    # Define tolerances based on input args
    regular_tol = tol * deduplication_tol_factor
    singular_tol = singular_deduplication_tol # Use the new specific tolerance

    if verbose:
        print(f"Attempting to deduplicate {len(raw_solutions)} raw solutions...")

    for sol in raw_solutions:
        is_duplicate = False
        # Check against existing unique representatives
        for existing_sol in unique_representatives:
            dist = sol.distance(existing_sol, variables)

            # Use different tolerances for singular vs. regular solutions
            current_tol = singular_tol if sol.is_singular and existing_sol.is_singular else regular_tol

            if dist < current_tol:
                is_duplicate = True
                # Optional: If singular, combine winding numbers or other properties
                # if sol.is_singular and existing_sol.is_singular and hasattr(existing_sol, 'winding_number'):
                #     existing_sol.winding_number = (existing_sol.winding_number or 0) + (sol.winding_number or 0)
                break

        if not is_duplicate:
            unique_solutions.append(sol)
            unique_representatives.append(sol) # Add this solution as a new representative

    # Create the result
    result = SolutionSet(unique_solutions, system)
    
    # Add metadata
    result._meta['total_paths'] = len(start_solutions)
    result._meta['successful_paths'] = successful_paths
    result._meta['failed_paths'] = failed_paths # This now includes paths skipped due to high residual
    result._meta['solve_time'] = time.time() - start_time
    result._meta['raw_solutions_found'] = len(raw_solutions) # Count of solutions *before* deduplication but *after* residual check
    
    if verbose:
        print(f"Found {len(unique_solutions)} distinct solutions (from {len(raw_solutions)} raw solutions)")
        print(f"Solution process completed in {result._meta['solve_time']:.2f} seconds")
    
    return result


def polyvar(*names: str) -> Union[Variable, Tuple[Variable, ...]]:
    """Create polynomial variables with the given names.
    
    Args:
        *names: Variable names
        
    Returns:
        A single Variable or a tuple of Variables
    """
    variables = tuple(Variable(name) for name in names)
    return variables[0] if len(variables) == 1 else variables