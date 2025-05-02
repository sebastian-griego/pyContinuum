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
    Computes a witness superset for components of a given dimension D=N-n.
    Requires the augmented system F'=(F, L) to be square.
    """
    if solver_options is None:
        solver_options = {}

    n_equations = len(original_system.equations)
    n_variables = len(variables)

    # Check if the dimension D requested matches the expected dimension N-n
    expected_dimension = n_variables - n_equations
    if expected_dimension < 0:
         raise ValueError(f"Original system is overdetermined ({n_equations} eqs, "
                          f"{n_variables} vars). Cannot compute witness set.")
    if dimension != expected_dimension:
         raise ValueError(f"Witness set computation via slicing requires dimension "
                          f"D = N-n = {expected_dimension}, but got D = {dimension}.")
    if dimension < 0: # Should be covered by above, but keep for clarity
        raise ValueError("Dimension must be non-negative")

    # Generate D generic linear slicing equations L (D = N-n)
    slicing_system = generate_generic_slice(dimension, variables)

    # Create the augmented system F' = (F, L)
    augmented_equations = original_system.equations + slicing_system.equations
    augmented_system = PolynomialSystem(augmented_equations)

    # ASSERT: augmented_system should now be square (n_eqs + D = n_eqs + N - n_eqs = N equations)
    if len(augmented_system.equations) != n_variables:
        # This should ideally not happen if logic above is correct
        raise RuntimeError(f"Internal Error: Augmented system is not square! "
                           f"({len(augmented_system.equations)} eqs, {n_variables} vars)")

    # Solve the augmented SQUARE system using the existing solver
    print(f"Solving augmented square system ({n_variables}x{n_variables}) "
          f"for witness superset of dimension {dimension}...")

    # We assume 'solve' works correctly for square systems.
    # We pass the specific variables list.
    # We might need to ensure the start system used by 'solve' is appropriate
    # for this augmented system. The default total degree might work but
    # could be inefficient if F was sparse.
    witness_superset = solve(augmented_system, variables=variables, **solver_options)

    print(f"Found {len(witness_superset)} potential witness points.")
    return slicing_system, witness_superset
