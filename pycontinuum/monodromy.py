"""
Monodromy module for PyContinuum.

This module implements monodromy-based algorithms for numerical
irreducible decomposition of positive-dimensional solution components.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Set, Any, Optional
import time
# For permutation group operations
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup

from pycontinuum.polynomial import Variable, PolynomialSystem
from pycontinuum.witness_set import WitnessSet, generate_generic_slice, compute_witness_superset
from pycontinuum.parameter_homotopy import ParameterHomotopy, track_parameter_path
from pycontinuum.solver import SolutionSet, Solution
from pycontinuum.utils import evaluate_system_at_point, evaluate_jacobian_at_point


def track_monodromy_loop(original_system: PolynomialSystem,
                         start_slice: PolynomialSystem,
                         start_witness_points: List[Solution],
                         variables: List[Variable],
                         num_loops: int = 3,
                         parameter_tracker_options: Dict = None) -> List[Permutation]:
    """
    Track witness points around random loops in parameter space.
    
    This creates random loops in the space of linear slices and tracks
    witness points along these loops to discover permutations.
    
    Args:
        original_system: The original polynomial system F.
        start_slice: The starting slicing system L.
        start_witness_points: The list of witness points W.
        variables: System variables.
        num_loops: Number of random loops to perform.
        parameter_tracker_options: Options for parameter tracking.
        
    Returns:
        A list of permutations representing the action of monodromy.
    """
    if parameter_tracker_options is None:
        parameter_tracker_options = {}
        
    n_points = len(start_witness_points)
    if n_points == 0:
        return []
        
    # Convert Solution objects to numeric arrays for tracking
    start_w_numeric = np.array([
        [sol.values[var] for var in variables] 
        for sol in start_witness_points
    ], dtype=complex)
    
    # Initialize with identity permutation
    identity_perm = Permutation(list(range(n_points)))
    permutations = []
    
    # Use the starting slice as the current slice
    current_slice = start_slice
    current_w_numeric = start_w_numeric.copy()
    
    # IMPORTANT: Create extended variables for non-square systems
    n_equations = len(original_system.equations) + len(start_slice.equations)
    n_vars = len(variables)
    
    # If system is underdetermined, add Lagrange multiplier variables 
    # This is just for the monodromy tracking
    extended_vars = variables.copy()
    dummy_vars = []
    
    if n_equations < n_vars:
        # Create dummy variables λ_1, λ_2, etc.
        for i in range(n_vars - n_equations):
            dummy_var_name = f"λ_{i+1}"
            dummy_var = Variable(dummy_var_name)
            dummy_vars.append(dummy_var)
            extended_vars.append(dummy_var)
            
        # Extend the current points with zeros for the dummy variables
        dummy_zeros = np.zeros((n_points, len(dummy_vars)), dtype=complex)
        current_w_numeric = np.hstack((current_w_numeric, dummy_zeros))
    
    for loop_num in range(num_loops):
        print(f"Monodromy Loop {loop_num + 1}/{num_loops}")
        
        # 1. Generate a random target slice (same dimension as start_slice)
        dimension = len(current_slice.equations)
        target_slice = generate_generic_slice(dimension, variables)
        
        # 2. Create parameter homotopy from current to target slice
        ph_forward = ParameterHomotopy(
            original_system, 
            current_slice, 
            target_slice, 
            extended_vars if dummy_vars else variables,
            square_fix=True  # Use our new square_fix parameter
        )
        
        # 3. Track all points from current to target (t=0 to t=1)
        end_points_leg1 = np.zeros_like(current_w_numeric)
        success_leg1 = True
        tracked_indices = list(range(n_points))
        results_leg1 = {}
        
        print(f"  Tracking {n_points} points to intermediate slice...")
        for i in range(n_points):
            # Set tracking options with higher precision for better matching
            track_opts = parameter_tracker_options.copy()
            track_opts.setdefault('tol', 1e-10)  # Tighter tolerance
            
            end_pt, info = track_parameter_path(
                ph_forward, 
                current_w_numeric[i], 
                start_t=0.0, 
                end_t=1.0, 
                options=track_opts
            )
            
            if not info['success']:
                print(f"  Warning: Path {i} failed during forward tracking.")
                success_leg1 = False
                tracked_indices.remove(i)
                continue
                
            end_points_leg1[i] = end_pt
            results_leg1[i] = info
            
        # 4. Create return homotopy from target back to start
        ph_return = ParameterHomotopy(
            original_system,
            target_slice,
            start_slice,
            extended_vars if dummy_vars else variables,
            square_fix=True  # Use our new square_fix parameter
        )
        
        # If forward leg failed completely, skip this loop
        if not tracked_indices:
            print("  All paths failed on forward leg. Skipping this loop.")
            continue
            
        # 5. Track successful points back to start slice
        final_end_points = np.zeros_like(current_w_numeric)
        success_leg2 = True
        results_leg2 = {}
        
        print(f"  Tracking {len(tracked_indices)} points back to start slice...")
        for i in tracked_indices:
            # Use the same refined options
            track_opts = parameter_tracker_options.copy()
            track_opts.setdefault('tol', 1e-10)
            
            end_pt, info = track_parameter_path(
                ph_return,
                end_points_leg1[i],
                start_t=0.0,
                end_t=1.0,
                options=track_opts
            )
            
            if not info['success']:
                print(f"  Warning: Path {i} failed during return tracking.")
                success_leg2 = False
                tracked_indices.remove(i)
                continue
                
            final_end_points[i] = end_pt
            results_leg2[i] = info
            
        # If return leg failed completely, skip this loop
        if not tracked_indices:
            print("  All paths failed on return leg. Skipping this loop.")
            continue
            
        # 6. Match final points back to start points to find permutation
        # Only consider points that were successfully tracked both ways
        # Use only the original variables (not the dummy variables) for matching
        match_tol = parameter_tracker_options.get('match_tol', 1e-3)  # Increased tolerance
        
        # Create dictionaries of successfully tracked points, but only use original variables
        orig_var_count = len(variables)
        start_subset = {idx: start_w_numeric[idx, :orig_var_count] for idx in tracked_indices}
        final_subset = {idx: final_end_points[idx, :orig_var_count] for idx in tracked_indices}
        
        # Try normalized matching (less sensitive to scaling differences)
        def normalize_point(p):
            norm = np.linalg.norm(p)
            if norm > 1e-10:
                return p / norm
            return p
            
        # Normalize all points for better matching
        start_norm = {idx: normalize_point(p) for idx, p in start_subset.items()}
        final_norm = {idx: normalize_point(p) for idx, p in final_subset.items()}
        
        # Initialize mapping: final_idx -> start_idx
        mapping = [-1] * n_points
        used_start_indices = set()
        
        # Match points based on distance of normalized points
        for final_idx in tracked_indices:
            best_match_idx = -1
            min_dist = float('inf')
            
            for start_idx in tracked_indices:
                if start_idx not in used_start_indices:
                    # Try different distance metrics
                    # 1. Euclidean distance of normalized points
                    dist1 = np.linalg.norm(final_norm[final_idx] - start_norm[start_idx])
                    # 2. Angular distance (dot product of normalized vectors)
                    dot = np.abs(np.vdot(final_norm[final_idx], start_norm[start_idx]))
                    dist2 = 1.0 - dot  # Smaller is better (0 = parallel)
                    
                    # Use combined metric
                    dist = dist1 * 0.5 + dist2 * 0.5
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_match_idx = start_idx
                        
            # Looser matching tolerance
            if min_dist < match_tol and best_match_idx != -1:
                mapping[final_idx] = best_match_idx
                used_start_indices.add(best_match_idx)
            else:
                print(f"  Warning: Could not match final point {final_idx} (min_dist={min_dist:.2e}).")
                # Don't fail completely - just leave this point unmapped
        
        # If we've matched enough points, construct a permutation
        if len(used_start_indices) >= 2:  # Need at least 2 points for non-identity perm
            # Create a map: start_idx -> final_idx from mapping: final_idx -> start_idx
            start_to_final = {}
            for final_idx, start_idx in enumerate(mapping):
                if start_idx != -1:
                    start_to_final[start_idx] = final_idx
                    
            # Create the permutation array - use identity for unmapped points
            perm_array = list(range(n_points))
            for i in range(n_points):
                if i in start_to_final:
                    perm_array[i] = start_to_final[i]
                    
            # Create sympy Permutation object
            perm = Permutation(perm_array)
            
            # Only add non-identity permutations
            if perm != identity_perm:
                permutations.append(perm)
                print(f"  Found permutation: {perm}")
            else:
                print("  Loop produced identity permutation.")
        else:
            print(f"  Not enough matches ({len(used_start_indices)}) to determine permutation.")
            
        # Update current slice and points for next loop
        # This allows exploring more of the parameter space
        current_slice = target_slice
        current_w_numeric = end_points_leg1
        
    return permutations

def numerical_irreducible_decomposition(original_system: PolynomialSystem,
                                       slicing_system: PolynomialSystem,
                                       witness_superset: SolutionSet,
                                       variables: List[Variable],
                                       monodromy_options: Dict = None) -> List[WitnessSet]:
    """
    Perform numerical irreducible decomposition on a witness superset.
    
    This uses monodromy loops to identify the irreducible components
    within a witness superset.
    
    Args:
        original_system: The original polynomial system F.
        slicing_system: The slicing system L.
        witness_superset: The set of potential witness points W.
        variables: System variables.
        monodromy_options: Options for the monodromy computation.
        
    Returns:
        List of WitnessSet objects, one for each irreducible component.
    """
    if monodromy_options is None:
        monodromy_options = {}
        
    witness_points_list = witness_superset.solutions
    n_points = len(witness_points_list)
    
    if n_points == 0:
        print("No witness points found in the witness superset.")
        return []
        
    dimension = len(slicing_system.equations)
    print(f"Starting numerical irreducible decomposition for {n_points} "
          f"potential witness points of dimension {dimension}...")
          
    # Get number of loops to perform
    # Heuristic: max(5, n_points / 2) loops often work well in practice
    num_loops = monodromy_options.get('num_loops', max(5, n_points // 2))
    
    # 1. Track loops to get permutations
    permutations = track_monodromy_loop(
        original_system,
        slicing_system,
        witness_points_list,
        variables,
        num_loops=num_loops,
        parameter_tracker_options=monodromy_options.get('tracker_options', {})
    )
    
    # If monodromy failed to produce any permutations, check if we can make an educated guess
    if not permutations:
        print("Warning: Monodromy tracking failed to produce non-identity permutations.")
        
        # Special case: if we have a single line, curve, or smooth component, we can often detect it
        # by checking the Jacobian rank at the witness points
        if n_points > 0 and dimension == 1:
            print("Attempting to determine reducibility by analyzing Jacobian rank...")
            
            # Try to detect if all points are on a single component by checking rank consistency
            ranks = []
            for pt in witness_points_list:
                # Convert to point array
                pt_arr = np.array([pt.values[var] for var in variables], dtype=complex)
                
                # Evaluate Jacobian of original system at point
                jac = evaluate_jacobian_at_point(original_system, pt_arr, variables)
                
                # Compute rank (using SVD which is more robust numerically)
                # Use a tolerance appropriate for the working precision
                u, s, vh = np.linalg.svd(jac)
                rank = np.sum(s > 1e-8)
                ranks.append(rank)
                
            # If all points have the same rank, they might be on one component
            if all(r == ranks[0] for r in ranks):
                print(f"All points have consistent Jacobian rank {ranks[0]}.")
                print("This suggests they may be on a single irreducible component.")
                print("Returning a single component.")
                return [WitnessSet(original_system, slicing_system, witness_points_list, dimension)]
        
        # If we couldn't determine, return the superset as a single component
        print("Returning the superset as a single component.")
        return [WitnessSet(original_system, slicing_system, witness_points_list, dimension)]
        
    # 2. Compute permutation group and its orbits
    print(f"Computing orbits from {len(permutations)} permutations...")
    
    try:
        # Create permutation group and compute orbits
        group = PermutationGroup(permutations)
        orbits = group.orbits()
        print(f"Found {len(orbits)} orbits (potential irreducible components).")
    except Exception as e:
        print(f"Error computing permutation group: {e}")
        print("Returning the superset as a single component.")
        return [WitnessSet(original_system, slicing_system, witness_points_list, dimension)]
        
    # If orbits are empty, return all points as a single component
    if not orbits:
        print("No orbits found. Returning the superset as a single component.")
        return [WitnessSet(original_system, slicing_system, witness_points_list, dimension)]
        
    # 3. Create WitnessSet objects for each orbit
    irreducible_components = []
    
    # Keep track of which points have been assigned to components
    used_indices = set()
    
    for orbit_idx, orbit in enumerate(orbits):
        # Convert orbit (set of integers) to list and sort
        orbit_list = sorted(list(orbit))
        
        # Get witness points for this orbit
        component_points = [witness_points_list[i] for i in orbit_list]
        
        # Create WitnessSet
        ws = WitnessSet(original_system, slicing_system, component_points, dimension)
        irreducible_components.append(ws)
        
        print(f"  Component {orbit_idx+1}: dimension={ws.dimension}, degree={ws.degree}")
        
        # Add these indices to used_indices
        used_indices.update(orbit_list)
    
    # Check if we missed any points
    missed_indices = set(range(n_points)) - used_indices
    if missed_indices:
        print(f"Warning: {len(missed_indices)} points were not assigned to any component.")
        print("Creating an additional component for these points.")
        
        # Create an additional component for these points
        additional_points = [witness_points_list[i] for i in sorted(missed_indices)]
        ws = WitnessSet(original_system, slicing_system, additional_points, dimension)
        irreducible_components.append(ws)
        
        print(f"  Additional component: dimension={ws.dimension}, degree={ws.degree}")
        
    return irreducible_components


def compute_numerical_decomposition(system: PolynomialSystem,
                                   variables: List[Variable] = None,
                                   max_dimension: int = None,
                                   solver_options: Dict = None,
                                   monodromy_options: Dict = None) -> Dict[int, List[WitnessSet]]:
    """
    Compute the numerical irreducible decomposition of a variety V(system).
    
    This is the high-level function that finds components of all relevant
    dimensions and decomposes them into irreducible components.
    
    Args:
        system: The polynomial system F.
        variables: List of variables (if None, extracted from system).
        max_dimension: Highest dimension to check (default: N-n).
        solver_options: Options for the solver.
        monodromy_options: Options for the monodromy computation.
        
    Returns:
        A dictionary mapping dimension to a list of WitnessSet objects.
    """
    if variables is None:
        variables = list(system.variables())
        
    if solver_options is None:
        solver_options = {}
        
    if monodromy_options is None:
        monodromy_options = {}
        
    n_vars = len(variables)
    n_eqs = len(system.equations)
    
    # Determine the expected maximum dimension
    expected_max_dim = max(0, n_vars - n_eqs)
    
    if max_dimension is None:
        max_dimension = expected_max_dim
    else:
        max_dimension = min(max_dimension, expected_max_dim)
        
    print(f"Computing numerical decomposition of a system with {n_eqs} equations "
          f"in {n_vars} variables.")
    print(f"Expected maximum dimension: {expected_max_dim}, "
          f"checking dimensions up to {max_dimension}.")
          
    # Dictionary to store components by dimension
    decomposition = {}
    
    # Start from highest dimension and work down
    for D in range(max_dimension, -1, -1):
        print(f"\n--- Computing components of dimension {D} ---")
        
        try:
            # 1. Compute witness superset for dimension D
            slicing_system, witness_superset = compute_witness_superset(
                system, variables, D, solver_options
            )
            
            if len(witness_superset) == 0:
                print(f"No witness points found for dimension {D}.")
                continue
                
            # 2. Perform numerical irreducible decomposition
            components_D = numerical_irreducible_decomposition(
                system, slicing_system, witness_superset, variables, monodromy_options
            )
            
            if components_D:
                decomposition[D] = components_D
                
        except ValueError as e:
            print(f"Skipping dimension {D}: {e}")
        except Exception as e:
            print(f"Error computing dimension {D}: {e}")
            import traceback
            traceback.print_exc()
            
    return decomposition