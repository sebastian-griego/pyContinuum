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
          store_paths: bool = False) -> SolutionSet:
    """Solve a polynomial system using homotopy continuation.
    Args:
        system: The polynomial system to solve
        start_system: Optional custom start system (default: total-degree homotopy)
        start_solutions: Optional known solutions of the start system
        variables: Optional list of variables to use (default: extracted from system)
        tol: Tolerance for path tracking and solution classification
        verbose: Whether to print progress information
        
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
        store_paths=store_paths
    )
    # Process the raw solutions
    solutions = []
    
    if verbose:
        print("Processing and classifying solutions...")
    
    # Function to compute residual
    def compute_residual(sol_values):
        return np.linalg.norm(system.evaluate(sol_values))
    
    # Process each end solution
    successful_paths = 0
    failed_paths = 0
    
    for i, (endpoint, success, is_singular) in enumerate(zip(end_solutions, path_results['success'], path_results['singular'])):
        if not success:
            failed_paths += 1
            continue
            
        successful_paths += 1
        
        # Create Solution object
        solution_dict = {var: val for var, val in zip(variables, endpoint)}
        residual = compute_residual(solution_dict)
        
        # Skip solutions with large residuals
        if residual > 100 * tol:
            failed_paths += 1
            continue
            
        solution = Solution(
            values=solution_dict,
            residual=residual,
            is_singular=is_singular,
            path_index=i
        )
        
        # Store path points if available
        if store_paths and path_results.get('path_points'):
            solution.path_points = path_results['path_points'][i]
        
        solutions.append(solution)
    
    # Remove duplicate solutions
    unique_solutions = []
    for sol in solutions:
        # Check if this solution is already in our list
        is_duplicate = False
        for existing_sol in unique_solutions:
            # Compute distance between solutions
            dist = 0
            for var in variables:
                dist += abs(sol.values[var] - existing_sol.values[var])**2
            dist = np.sqrt(dist)
            
            if dist < 10 * tol:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_solutions.append(sol)
    
    # Create the result
    result = SolutionSet(unique_solutions, system)
    
    # Add metadata
    result._meta['total_paths'] = len(start_solutions)
    result._meta['successful_paths'] = successful_paths
    result._meta['failed_paths'] = failed_paths
    result._meta['solve_time'] = time.time() - start_time
    
    if verbose:
        print(f"Found {len(unique_solutions)} distinct solutions")
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