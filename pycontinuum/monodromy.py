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
    (Removed dummy variable logic)
    """
    if parameter_tracker_options is None:
        parameter_tracker_options = {}

    n_points = len(start_witness_points)
    if n_points == 0:
        return []

    start_w_numeric = np.array([
        [sol.values[var] for var in variables]
        for sol in start_witness_points
    ], dtype=complex)

    identity_perm = Permutation(list(range(n_points)))
    permutations = []

    current_slice = start_slice
    current_w_numeric = start_w_numeric.copy()  # Keep track of current points numerically

    max_attempts = num_loops * 2  # Allow some retries if loops fail
    loop_count = 0
    successful_loops = 0

    while successful_loops < num_loops and loop_count < max_attempts:
        loop_count += 1
        print(f"Monodromy Loop Attempt {loop_count}/{max_attempts} (Target: {num_loops} successful)")

        # 1. Generate a random target slice
        dimension = len(current_slice.equations)
        target_slice = generate_generic_slice(dimension, variables)

        # 2. Create FORWARD parameter homotopy (Current -> Target)
        ph_forward = ParameterHomotopy(
            original_system, current_slice, target_slice, variables
        )

        # 3. Track points forward
        end_points_leg1 = np.zeros_like(current_w_numeric)
        success_flags_leg1 = [False] * n_points
        tracked_indices_leg1 = set()  # Indices that succeed leg 1

        print(f"  Tracking {n_points} points to intermediate slice...")
        for i in range(n_points):
            track_opts = parameter_tracker_options.copy()  # Pass options down
            end_pt, info = track_parameter_path(
                ph_forward, current_w_numeric[i], options=track_opts
            )
            if info.get('success', False):
                end_points_leg1[i] = end_pt
                success_flags_leg1[i] = True
                tracked_indices_leg1.add(i)
            else:
                print(f"  Warning: Path {i} failed during forward tracking (t={info.get('t', '?'):.3f}).")

        # Check if enough paths succeeded
        if len(tracked_indices_leg1) < 2:  # Need at least 2 points to define a non-trivial permutation
            print("  Not enough paths succeeded on forward leg. Trying new loop.")
            current_slice = start_slice  # Reset to original slice for next attempt
            current_w_numeric = start_w_numeric.copy()
            continue  # Go to next loop attempt

        # 4. Create RETURN parameter homotopy (Target -> Start)
        ph_return = ParameterHomotopy(
            original_system, target_slice, start_slice, variables
        )

        # 5. Track successful points back
        final_end_points = np.zeros_like(current_w_numeric)
        success_flags_leg2 = [False] * n_points
        tracked_indices_leg2 = set()  # Indices that succeed leg 2

        print(f"  Tracking {len(tracked_indices_leg1)} points back to start slice...")
        for i in tracked_indices_leg1:  # Only track points that succeeded leg 1
            track_opts = parameter_tracker_options.copy()
            end_pt, info = track_parameter_path(
                ph_return, end_points_leg1[i], options=track_opts
            )
            if info.get('success', False):
                final_end_points[i] = end_pt
                success_flags_leg2[i] = True
                tracked_indices_leg2.add(i)
            else:
                print(f"  Warning: Path {i} failed during return tracking (t={info.get('t', '?'):.3f}).")

        # Identify points tracked successfully both ways
        fully_tracked_indices = tracked_indices_leg1.intersection(tracked_indices_leg2)

        if len(fully_tracked_indices) < 2:
            print("  Not enough paths succeeded on return leg. Trying new loop.")
            current_slice = start_slice  # Reset
            current_w_numeric = start_w_numeric.copy()
            continue  # Go to next loop attempt

        # 6. Match final points back to start points for successfully tracked paths
        match_tol = parameter_tracker_options.get('match_tol', 1e-4)  # Adjust tolerance maybe

        start_subset = {idx: start_w_numeric[idx] for idx in fully_tracked_indices}
        final_subset = {idx: final_end_points[idx] for idx in fully_tracked_indices}

        # --- Consider using scipy.optimize.linear_sum_assignment for robust matching ---
        # Build cost matrix (distances)
        cost_matrix = np.full((len(fully_tracked_indices), len(fully_tracked_indices)), float('inf'))
        idx_list = sorted(list(fully_tracked_indices))
        idx_map = {original_idx: mapped_idx for mapped_idx, original_idx in enumerate(idx_list)}

        for r_idx, start_idx in enumerate(idx_list):
            for c_idx, final_idx in enumerate(idx_list):
                # Use norm for distance
                dist = np.linalg.norm(start_subset[start_idx] - final_subset[final_idx])
                cost_matrix[r_idx, c_idx] = dist

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Check if matches are within tolerance
        mapping = {}  # Stores start_idx -> final_idx
        valid_match = True
        for r, c in zip(row_ind, col_ind):
            start_idx = idx_list[r]
            final_idx = idx_list[c]
            if cost_matrix[r, c] < match_tol:
                mapping[start_idx] = final_idx
            else:
                print(f"  Warning: Match distance too high for pair "
                      f"({start_idx} -> {final_idx}, dist={cost_matrix[r, c]:.2e}).")
                valid_match = False
                break

        if valid_match:
            # Construct permutation array (identity for non-tracked/non-matched)
            perm_array = list(range(n_points))
            for start_idx, final_idx in mapping.items():
                perm_array[start_idx] = final_idx

            perm = Permutation(perm_array)
            if perm != identity_perm:
                permutations.append(perm)
                print(f"  Found permutation: {perm.array_form}")
                successful_loops += 1
            else:
                print("  Loop produced identity permutation.")
        else:
            print("  Matching failed due to high distance. Trying new loop.")
            current_slice = start_slice  # Reset
            current_w_numeric = start_w_numeric.copy()
            continue  # Go to next loop attempt

        # Update current points/slice for the next attempt (using target from successful loop)
        current_slice = target_slice
        # Update only the successfully tracked points for the next iteration start
        for idx in fully_tracked_indices:
            current_w_numeric[idx] = end_points_leg1[idx]  # Use points at target slice

    if successful_loops < num_loops:
        print(f"Warning: Only found {successful_loops}/{num_loops} successful monodromy loops after {max_attempts} attempts.")

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
                                   max_dimension: int = None,  # Keep parameter for potential future use
                                   solver_options: Dict = None,
                                   monodromy_options: Dict = None) -> Dict[int, List[WitnessSet]]:
    """
    Compute the numerical irreducible decomposition of the top-dimensional
    components of the variety V(system).

    Args:
        system: The polynomial system F.
        variables: List of variables (if None, extracted from system).
        max_dimension: (Optional) Can specify a dimension to check, but
                       typically determined as N-n.
        solver_options: Options for the solver.
        monodromy_options: Options for the monodromy computation.

    Returns:
        A dictionary mapping dimension D to a list of WitnessSet objects for
        irreducible components of that dimension.
    """
    if variables is None:
        variables = list(system.variables())

    if solver_options is None:
        solver_options = {}

    if monodromy_options is None:
        monodromy_options = {}

    n_vars = len(variables)
    n_eqs = len(system.equations)

    # Determine the expected dimension D = N-n
    # Only proceed if the system is not overdetermined (n <= N)
    if n_eqs > n_vars:
        print(f"System is overdetermined ({n_eqs} eqs, {n_vars} vars). "
              "Decomposition currently supports n <= N.")
        return {}

    # The dimension of the components we expect to find via slicing
    top_dimension = n_vars - n_eqs

    print(f"Computing numerical decomposition for dimension D = {top_dimension} "
          f"(System: {n_eqs} eqs, {n_vars} vars).")

    decomposition = {}
    D = top_dimension  # Only compute for this dimension

    print(f"\n--- Computing components of dimension {D} ---")

    try:
        # 1. Compute witness superset for dimension D
        current_solver_options = solver_options.copy()
        slicing_system, witness_superset = compute_witness_superset(
            system, variables, D, current_solver_options
        )

        if len(witness_superset) == 0:
            print(f"No witness points found for dimension {D}.")
        else:
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

    print("\nDecomposition computation finished for top dimension.")
    return decomposition