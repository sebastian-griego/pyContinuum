"""
Witness set module for PyContinuum.

This module provides classes and functions for computing and manipulating
witness sets, which represent positive-dimensional components of algebraic varieties.
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Any, Optional

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem, Monomial, polyvar
from pycontinuum.solver import solve, Solution, SolutionSet


class WitnessSet:
    """
    Represents a witness set for a component of a variety.
    
    A witness set consists of:
    - F: The original polynomial system defining the variety
    - L: A set of generic linear equations (slicing system)
    - W: A set of points in the intersection of V(F) and V(L)
    
    The dimension is the number of linear slices needed, and the degree
    is the number of witness points (for an irreducible component).
    """
    
    def __init__(self,
                 original_system: PolynomialSystem,
                 slicing_system: PolynomialSystem,
                 witness_points: List[Solution],
                 dimension: int):
        """
        Initialize a witness set.
        
        Args:
            original_system: The polynomial system F defining the variety.
            slicing_system: The generic linear equations L.
            witness_points: The intersection points W = V(F) ∩ V(L).
            dimension: The dimension of the component.
        """
        self.original_system = original_system
        self.slicing_system = slicing_system
        self.witness_points = witness_points
        self.dimension = dimension
        # The degree of an irreducible component is the number of witness points
        self.degree = len(witness_points)
        
    def __repr__(self) -> str:
        """String representation of the witness set."""
        return (f"WitnessSet(dimension={self.dimension}, degree={self.degree}, "
                f"{len(self.witness_points)} points)")
    
    def sample_point(self, 
                    target_slice: Optional[PolynomialSystem] = None, 
                    variables: Optional[List[Variable]] = None,
                    options: Dict[str, Any] = None) -> Optional[np.ndarray]:
        """
        Sample a point on the component by moving to a different slice.
        
        Args:
            target_slice: A target slicing system different from the current one.
                          If None, a random slice will be generated.
            variables: The system variables. If None, extracted from original system.
            options: Options for parameter tracking.
            
        Returns:
            A point on the component at the target slice, or None if tracking fails.
        """
        # Import here to avoid circular imports
        from pycontinuum.parameter_homotopy import ParameterHomotopy, track_parameter_path
        
        if variables is None:
            variables = list(self.original_system.variables())
            
        if options is None:
            options = {}
            
        # If no target slice provided, generate a random one
        if target_slice is None:
            target_slice = generate_generic_slice(self.dimension, variables)
            
        # Choose a random witness point to track
        if not self.witness_points:
            return None
            
        source_point_sol = np.random.choice(self.witness_points)
        source_point = np.array([source_point_sol.values[var] for var in variables], dtype=complex)
        
        # Create parameter homotopy from current slice to target slice
        ph = ParameterHomotopy(
            self.original_system,
            self.slicing_system,
            target_slice,
            variables
        )
        
        # Track the point to the target slice
        target_point, info = track_parameter_path(
            ph, source_point, start_t=0.0, end_t=1.0, options=options
        )
        
        if info['success']:
            return target_point
        else:
            print("Warning: Failed to sample point on component.")
            return None
            
    def is_point_on_component(self, 
                             point: np.ndarray, 
                             variables: List[Variable],
                             tolerance: float = 1e-8) -> bool:
        """
        Check if a point lies on this component.
        
        This uses parameter homotopy-based membership test.
        
        Args:
            point: The point to test.
            variables: The system variables.
            tolerance: Tolerance for equality.
            
        Returns:
            True if the point is on this component, False otherwise.
        """
        # Import here to avoid circular imports
        from pycontinuum.parameter_homotopy import ParameterHomotopy, track_parameter_path
        
        # Evaluate the original system at the point
        point_dict = {var: val for var, val in zip(variables, point)}
        system_vals = np.array(self.original_system.evaluate(point_dict), dtype=complex)
        
        # Check if the point satisfies the original system
        if np.linalg.norm(system_vals) > tolerance:
            return False
            
        # Generate a random generic slice through the point
        # We need to create D linear equations that all pass through the test point
        target_slice_eqs = []
        for _ in range(self.dimension):
            # Generate random coefficients
            coeffs = np.random.randn(len(variables)) + 1j * np.random.randn(len(variables))
            
            # Compute constant term so the equation passes through the point
            const_term = -sum(coef * val for coef, val in zip(coeffs, point))
            
            # Build polynomial
            terms = [Monomial({}, coefficient=const_term)]
            for i, var in enumerate(variables):
                terms.append(Monomial({var: 1}, coefficient=coeffs[i]))
            
            poly = Polynomial(terms)
            target_slice_eqs.append(poly)
            
        target_slice = PolynomialSystem(target_slice_eqs)
        
        # Try to track a witness point to this new slice
        sample_pt = self.sample_point(target_slice, variables)
        if sample_pt is None:
            # Tracking failed, can't determine membership
            return False
            
        # Check if the sampled point is close to the test point
        dist = np.linalg.norm(sample_pt - point)
        return dist < tolerance


def generate_generic_slice(dimension: int, variables: List[Variable]) -> PolynomialSystem:
    """
    Generate D random linear equations in the given variables.
    
    These represent a generic codimension-D linear subspace of the ambient space.
    
    Args:
        dimension: Number of linear equations to generate (D).
        variables: The variables to use.
        
    Returns:
        A PolynomialSystem representing the slicing system L.
    """
    n_vars = len(variables)
    slice_eqs = []
    
    for _ in range(dimension):
        # Generate random complex coefficients for the linear equation
        # Using standard normal distribution for both real and imaginary parts
        coeffs = np.random.randn(n_vars) + 1j * np.random.randn(n_vars)
        const = np.random.randn() + 1j * np.random.randn()
        
        # Build the polynomial: a1*x1 + a2*x2 + ... + an*xn + c = 0
        # Start with the constant term
        poly_terms = [Monomial({}, coefficient=const)]
        
        # Add the variable terms
        for i, var in enumerate(variables):
            poly_terms.append(Monomial({var: 1}, coefficient=coeffs[i]))
            
        poly = Polynomial(poly_terms)
        slice_eqs.append(poly)
    
    return PolynomialSystem(slice_eqs)


def compute_witness_superset(original_system: PolynomialSystem,
                            variables: List[Variable],
                            dimension: int,
                            solver_options: Dict = None) -> Tuple[PolynomialSystem, SolutionSet]:
    """
    Compute a witness superset for components of a given dimension.
    
    This creates D generic linear equations, combines them with the original 
    system, and solves the resulting square system to find potential witness points.
    
    Args:
        original_system: The system defining the variety.
        variables: List of variables.
        dimension: The dimension of components to find witness points for.
        solver_options: Options passed to the solver.
        
    Returns:
        Tuple of (slicing_system, witness_superset).
    """
    if solver_options is None:
        solver_options = {}
        
    n_equations = len(original_system.equations)
    n_variables = len(variables)
    
    # Check if the requested dimension is valid
    expected_max_dim = n_variables - n_equations
    if expected_max_dim < 0:
        # System is overdetermined, should have no solutions unless special structure
        raise ValueError(f"System is overdetermined with {n_equations} equations in "
                         f"{n_variables} variables. Cannot compute witness set.")
                         
    if dimension > expected_max_dim:
        raise ValueError(f"Expected dimension at most {expected_max_dim}, "
                         f"cannot find witness set for dimension {dimension}")
    
    if dimension < 0:
        raise ValueError("Dimension must be non-negative")
        
    # Generate D generic linear slicing equations L
    slicing_system = generate_generic_slice(dimension, variables)
    
    # Create the augmented system F' = (F, L)
    augmented_equations = original_system.equations + slicing_system.equations
    augmented_system = PolynomialSystem(augmented_equations)
    
    # Check if the augmented system is square
    n_aug_equations = len(augmented_system.equations)
    if n_aug_equations == n_variables:
        # Use the regular solve function for square systems
        print(f"Solving augmented system with {n_aug_equations} equations "
              f"for witness superset of dimension {dimension}...")
        witness_superset = solve(
            augmented_system, 
            variables=variables, 
            allow_underdetermined=True,  # This is the key change
            **solver_options
        )
    else:
        # For non-square systems, we need a custom approach
        print(f"Working with non-square augmented system ({n_aug_equations} equations, "
              f"{n_variables} variables) for witness superset of dimension {dimension}...")
        
        # Import needed functions
        from pycontinuum.start_systems import generate_total_degree_solutions
        from pycontinuum.tracking import track_paths
        
        # Create a "fake" square system by adding dummy equations (x_i = 0)
        # for the extra variables
        dummy_equations = []
        for i in range(n_aug_equations, n_variables):
            # Create equation x_i = 0 using just the variable at index i
            dummy_equations.append(Polynomial([Monomial({variables[i]: 1})]))
        
        # Create the square augmented system with dummy equations
        square_system = PolynomialSystem(augmented_equations + dummy_equations)
        
        # Generate a total-degree start system manually
        degrees = square_system.degrees()
        
        # Generate random complex coefficients for the start system
        c_values = []
        for i in range(n_variables):
            angle = np.random.uniform(0, 2 * np.pi)
            c_values.append(complex(np.cos(angle), np.sin(angle)))
        
        # Create the start system equations x_i^(d_i) - c_i = 0
        start_equations = []
        for i, (var, deg, c) in enumerate(zip(variables, degrees, c_values)):
            # Create x_i^(d_i) term using Monomial
            term1 = Monomial({var: deg})
            # Create c_i term using Monomial
            term2 = Monomial({}, coefficient=c)
            # Create x_i^(d_i) - c_i using Polynomial
            eq = Polynomial([term1, Monomial({}, coefficient=-c)])
            start_equations.append(eq)
        
        # Create the start system
        start_system = PolynomialSystem(start_equations)
        
        # Generate all solutions to the start system
        start_solutions = generate_total_degree_solutions(degrees, c_values)
        
        # Track the paths
        print(f"Tracking {len(start_solutions)} paths for witness superset...")
        
        # Use modified solver options for better robustness
        track_options = solver_options.copy()
        # Add options specific to witness set computation if needed
        
        # Track paths directly using the tracking module
        end_solutions, path_results = track_paths(
            start_system=start_system,
            target_system=square_system,
            start_solutions=start_solutions,
            variables=variables,
            **track_options
        )
        
        # Filter the solutions to only include those where dummy equations are satisfied
        # (should be all of them since we made the dummy equations x_i = 0)
        
        # Convert the solutions to a SolutionSet
        from pycontinuum.solver import Solution, SolutionSet
        
        # Process solutions like in solver.py
        solutions = []
        for i, (end_point, path_info) in enumerate(zip(end_solutions, path_results)):
            if path_info.get('success', False):
                # Create solution dictionary
                solution_dict = {var: val for var, val in zip(variables, end_point)}
                
                # Check residual for the original augmented system (excluding dummy equations)
                residual = np.linalg.norm([eq.evaluate(solution_dict) for eq in augmented_equations])
                
                # Create Solution object
                solution = Solution(
                    values=solution_dict,
                    residual=residual,
                    is_singular=path_info.get('singular', False),
                    path_index=i
                )
                
                # Store winding number if available
                if 'winding_number' in path_info:
                    solution.winding_number = path_info['winding_number']
                
                # Store path points if available
                if 'path_points' in path_info:
                    solution.path_points = path_info['path_points']
                
                solutions.append(solution)
        
        # Create the SolutionSet
        witness_superset = SolutionSet(solutions, augmented_system)
        
        # Add metadata like in solve()
        witness_superset._meta['total_paths'] = len(start_solutions)
        witness_superset._meta['successful_paths'] = sum(1 for info in path_results if info.get('success', False))
        witness_superset._meta['failed_paths'] = len(start_solutions) - witness_superset._meta['successful_paths']
        witness_superset._meta['raw_solutions_found'] = len(solutions)
    
    print(f"Found {len(witness_superset)} potential witness points.")
    return slicing_system, witness_superset