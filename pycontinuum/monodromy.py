"""
Monodromy module for PyContinuum.

This module implements monodromy-based algorithms for numerical
irreducible decomposition of positive-dimensional solution components.
"""

from collections.abc import Mapping
import numpy as np
import random
from numbers import Integral, Real
from typing import List, Dict, Tuple, Set, Any, Optional
import time
# For permutation group operations
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem
from pycontinuum.witness_set import WitnessSet, generate_generic_slice, compute_witness_superset
from pycontinuum.parameter_homotopy import ParameterHomotopy, track_parameter_path
from pycontinuum.solver import (
    SolutionSet,
    Solution,
    _polynomial_to_sympy_expr,
    _sympy_expr_to_polynomial,
)
from pycontinuum.start_systems import _coerce_rng
from pycontinuum.utils import (
    evaluate_scaled_jacobian_at_point,
    evaluate_scaled_system_at_point,
    evaluate_system_at_point,
    evaluate_jacobian_at_point,
    _mapping_coordinate_for_variable,
    _scaled_euclidean_norm,
)


def track_monodromy_loop(original_system: PolynomialSystem,
                         start_slice: PolynomialSystem,
                         start_witness_points: Any,
                         variables: List[Variable],
                         num_loops: int = 3,
                         parameter_tracker_options: Dict = None,
                         random_state: Any = None,
                         verbose: bool = False) -> List[Permutation]:
    """
    Track witness points around random loops in parameter space.
    
    This creates random loops in the space of linear slices and tracks
    witness points along these loops to discover permutations.
    
    Args:
        original_system: The original polynomial system F.
        start_slice: The starting slicing system L.
        start_witness_points: The list of witness points W. Entries may be
            coordinate vectors, Solution objects, solution-like objects with a
            ``values`` mapping, or mappings keyed by variables or variable
            names.
        variables: System variables.
        num_loops: Number of random loops to perform.
        parameter_tracker_options: Options for parameter tracking.
        random_state: Optional seed or NumPy random generator for reproducible
            loop target slices.
        verbose: Whether to print loop progress and matching diagnostics.
        
    Returns:
        A list of permutations representing the action of monodromy.
    """
    _validate_polynomial_system("original_system", original_system)
    _validate_polynomial_system("start_slice", start_slice)
    variables = _normalize_monodromy_variables(variables)
    _validate_variable_list_covers_systems(variables, original_system, start_slice)
    num_loops = _validate_positive_integer("num_loops", num_loops)
    parameter_tracker_options = _validate_monodromy_tracker_options(
        parameter_tracker_options
    )
    verbose = _validate_boolean_option("verbose", verbose)
    rng = _coerce_rng(random_state)

    start_witness_points = _normalize_witness_points(start_witness_points)
    n_points = len(start_witness_points)
    if n_points == 0:
        return []
        
    # Convert Solution objects to numeric arrays for tracking
    start_w_numeric = _witness_points_to_array(start_witness_points, variables)
    _validate_start_witness_points_on_systems(
        original_system,
        start_slice,
        start_w_numeric,
        variables,
        _monodromy_witness_tolerance(parameter_tracker_options),
    )
    
    # Initialize with identity permutation
    identity_perm = Permutation(list(range(n_points)))
    permutations = []
    
    # Use the starting slice as the current slice
    current_slice = start_slice
    current_w_numeric = start_w_numeric.copy()
    
    n_equations = len(original_system.equations) + len(start_slice.equations)
    n_vars = len(variables)
    if n_equations < n_vars:
        raise ValueError(
            "Monodromy loops require original_system plus start_slice to "
            "define a zero-dimensional witness set; got "
            f"{n_equations} equation(s) for {n_vars} variable(s)"
        )
    
    for loop_num in range(num_loops):
        _monodromy_log(verbose, f"Monodromy Loop {loop_num + 1}/{num_loops}")
        
        # 1. Generate a random target slice (same dimension as start_slice)
        dimension = len(current_slice.equations)
        target_slice = generate_generic_slice(
            dimension,
            variables,
            random_state=rng,
        )
        
        # 2. Create parameter homotopy from current to target slice
        ph_forward = ParameterHomotopy(
            original_system, 
            current_slice, 
            target_slice, 
            variables,
            square_fix=True  # Use our new square_fix parameter
        )
        
        # 3. Track all points from current to target (t=0 to t=1)
        end_points_leg1 = np.zeros_like(current_w_numeric)
        success_leg1 = True
        tracked_indices = []
        results_leg1 = {}
        
        _monodromy_log(
            verbose,
            f"  Tracking {n_points} points to intermediate slice...",
        )
        for i in range(n_points):
            # Set tracking options with higher precision for better matching
            track_opts = _parameter_path_options(parameter_tracker_options)
            track_opts.setdefault('tol', 1e-10)  # Tighter tolerance
            
            end_pt, info = track_parameter_path(
                ph_forward, 
                current_w_numeric[i], 
                start_t=0.0, 
                end_t=1.0, 
                options=track_opts
            )
            
            if not info['success']:
                _monodromy_log(
                    verbose,
                    f"  Warning: Path {i} failed during forward tracking.",
                )
                success_leg1 = False
                continue
                
            end_points_leg1[i] = end_pt
            results_leg1[i] = info
            tracked_indices.append(i)
            
        # 4. Create return homotopy from target back to start
        ph_return = ParameterHomotopy(
            original_system,
            target_slice,
            start_slice,
            variables,
            square_fix=True  # Use our new square_fix parameter
        )
        
        # If forward leg failed completely, skip this loop
        if not tracked_indices:
            _monodromy_log(
                verbose,
                "  All paths failed on forward leg. Skipping this loop.",
            )
            continue
            
        # 5. Track successful points back to start slice
        final_end_points = np.zeros_like(current_w_numeric)
        success_leg2 = True
        results_leg2 = {}
        returned_indices = []
        
        _monodromy_log(
            verbose,
            f"  Tracking {len(tracked_indices)} points back to start slice...",
        )
        for i in tracked_indices:
            # Use the same refined options
            track_opts = _parameter_path_options(parameter_tracker_options)
            track_opts.setdefault('tol', 1e-10)
            
            end_pt, info = track_parameter_path(
                ph_return,
                end_points_leg1[i],
                start_t=0.0,
                end_t=1.0,
                options=track_opts
            )
            
            if not info['success']:
                _monodromy_log(
                    verbose,
                    f"  Warning: Path {i} failed during return tracking.",
                )
                success_leg2 = False
                continue
                
            final_end_points[i] = end_pt
            results_leg2[i] = info
            returned_indices.append(i)

        tracked_indices = returned_indices
            
        # If return leg failed completely, skip this loop
        if not tracked_indices:
            _monodromy_log(
                verbose,
                "  All paths failed on return leg. Skipping this loop.",
            )
            continue
            
        # 6. Match final points back to start points to find permutation
        # Only consider points that were successfully tracked both ways.
        match_tol = parameter_tracker_options.get('match_tol', 1e-3)
        
        # Create dictionaries of successfully tracked points, but only use original variables
        orig_var_count = len(variables)
        start_subset = {idx: start_w_numeric[idx, :orig_var_count] for idx in tracked_indices}
        final_subset = {idx: final_end_points[idx, :orig_var_count] for idx in tracked_indices}

        mapping, used_start_indices, min_distances = _match_affine_points(
            start_subset,
            final_subset,
            tracked_indices,
            n_points,
            match_tol,
        )

        for final_idx in tracked_indices:
            if mapping[final_idx] == -1:
                _monodromy_log(
                    verbose,
                    "  Warning: Could not match final point "
                    f"{final_idx} (min_dist={min_distances[final_idx]:.2e}).",
                )
        
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
                _monodromy_log(verbose, f"  Found permutation: {perm}")
            else:
                _monodromy_log(verbose, "  Loop produced identity permutation.")
        else:
            _monodromy_log(
                verbose,
                "  Not enough matches "
                f"({len(used_start_indices)}) to determine permutation.",
            )
            
        # Update current slice and points for next loop
        # This allows exploring more of the parameter space
        current_slice = target_slice
        current_w_numeric = end_points_leg1
        
    return permutations


def _validate_polynomial_system(name: str, system: Any) -> None:
    if not isinstance(system, PolynomialSystem):
        raise TypeError(f"{name} must be a PolynomialSystem")


def _normalize_monodromy_variables(variables: Any) -> List[Variable]:
    try:
        return list(variables)
    except TypeError as exc:
        raise TypeError(
            "variables must be an iterable of Variable objects"
        ) from exc


def _validate_variable_list_covers_systems(
    variables: List[Variable],
    *systems: PolynomialSystem,
) -> None:
    seen = set()
    duplicates = []
    for index, variable in enumerate(variables):
        if not isinstance(variable, Variable):
            raise TypeError(f"variables[{index}] must be a Variable")
        if variable in seen:
            duplicates.append(variable.name)
        seen.add(variable)
    if duplicates:
        raise ValueError(
            "Variable list contains duplicate variable(s): "
            + ", ".join(sorted(set(duplicates)))
        )

    missing = sorted(
        variable.name
        for system in systems
        for variable in system.variables()
        if variable not in seen
    )
    if missing:
        raise ValueError(
            "Variable list is missing system variable(s): "
            + ", ".join(sorted(set(missing)))
        )


def _validate_positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _validate_positive_finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return numeric_value


def _validate_boolean_option(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _validate_monodromy_tracker_options(
    parameter_tracker_options: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if parameter_tracker_options is None:
        return {}
    if not isinstance(parameter_tracker_options, dict):
        raise TypeError("parameter_tracker_options must be a dictionary")

    options = parameter_tracker_options.copy()
    allowed = {
        "match_tol",
        "tol",
        "min_step_size",
        "max_step_size",
        "max_corrections",
        "max_steps",
        "max_predictor_norm",
        "normalize_tangent",
        "predictor",
        "verbose",
        "store_paths",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(
            "Unknown monodromy tracker option(s): " + ", ".join(unknown)
        )
    if "match_tol" in options:
        options["match_tol"] = _validate_positive_finite_float(
            "match_tol",
            options["match_tol"],
        )
    if "tol" in options:
        options["tol"] = _validate_positive_finite_float("tol", options["tol"])
    return options


def _validate_monodromy_options(
    monodromy_options: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if monodromy_options is None:
        return {}
    if not isinstance(monodromy_options, dict):
        raise TypeError("monodromy_options must be a dictionary")

    allowed = {
        "num_loops",
        "tracker_options",
        "random_state",
        "verbose",
        "continue_on_error",
    }
    unknown = sorted(set(monodromy_options) - allowed)
    if unknown:
        raise ValueError(
            "Unknown monodromy option(s): " + ", ".join(unknown)
        )

    options = monodromy_options.copy()
    if "num_loops" in options:
        options["num_loops"] = _validate_positive_integer(
            "num_loops",
            options["num_loops"],
        )
    if "tracker_options" in options:
        options["tracker_options"] = _validate_monodromy_tracker_options(
            options["tracker_options"]
        )
    if "verbose" in options:
        options["verbose"] = _validate_boolean_option(
            "verbose",
            options["verbose"],
        )
    if "continue_on_error" in options:
        options["continue_on_error"] = _validate_boolean_option(
            "continue_on_error",
            options["continue_on_error"],
        )
    return options


def _normalize_witness_points(start_witness_points: Any) -> List[Any]:
    try:
        witness_points = list(start_witness_points)
    except TypeError as exc:
        raise TypeError(
            "start_witness_points must be an iterable of witness points"
        ) from exc

    return witness_points


def _witness_point_values(witness_point: Any) -> Optional[Mapping]:
    if isinstance(witness_point, Mapping):
        return witness_point
    values = getattr(witness_point, "values", None)
    if isinstance(values, Mapping):
        return values
    return None


def _witness_points_to_array(
    witness_points: List[Any],
    variables: List[Variable],
    *,
    label: str = "start_witness_points",
) -> np.ndarray:
    rows = []
    ambient = set(variables)
    ambient_names = {variable.name for variable in variables}
    for index, witness_point in enumerate(witness_points):
        values = _witness_point_values(witness_point)
        if values is None:
            try:
                point = np.asarray(witness_point, dtype=complex)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(
                    f"{label}[{index}] must be a coordinate "
                    "vector, coordinate mapping, or expose a values mapping"
                ) from exc
            if point.ndim != 1 or point.size != len(variables):
                raise ValueError(
                    f"{label}[{index}] must contain exactly "
                    f"{len(variables)} coordinate(s)"
                )
            rows.append(point)
        else:
            row = []
            for variable in variables:
                found, coordinate = _mapping_coordinate_for_variable(
                    values,
                    variable,
                    f"{label}[{index}]",
                )
                if found:
                    row.append(coordinate)
                else:
                    raise ValueError(
                        f"{label}[{index}] is missing variable "
                        f"{variable.name}"
                    )
            extra = _extra_witness_coordinate_names(
                values,
                ambient,
                ambient_names,
            )
            if extra:
                raise ValueError(
                    f"{label}[{index}] contains variable(s) "
                    "outside the monodromy variable list: " + ", ".join(extra)
                )
            try:
                point = np.asarray(row, dtype=complex)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"{label}[{index}] coordinate(s) must be numeric"
                ) from exc
            rows.append(point)
    try:
        point_array = np.array(rows, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} coordinate rows must be numeric") from exc
    if not np.all(np.isfinite(point_array)):
        raise ValueError(f"{label} contain nonfinite coordinates")
    return point_array


def _extra_witness_coordinate_names(
    values: Mapping,
    ambient: Set[Variable],
    ambient_names: Set[str],
) -> List[str]:
    extra = []
    for key in values:
        if isinstance(key, Variable):
            if key not in ambient:
                extra.append(key.name)
        elif isinstance(key, str):
            if key not in ambient_names:
                extra.append(key)
        else:
            extra.append(repr(key))
    return sorted(set(extra))


def _monodromy_witness_tolerance(
    parameter_tracker_options: Dict[str, Any],
) -> float:
    return 100.0 * float(parameter_tracker_options.get("tol", 1e-10))


def _validate_start_witness_points_on_systems(
    original_system: PolynomialSystem,
    start_slice: PolynomialSystem,
    points: np.ndarray,
    variables: List[Variable],
    tolerance: float,
) -> None:
    for witness_index, point in enumerate(points):
        original_residual = _monodromy_system_residual(
            original_system,
            point,
            variables,
        )
        if original_residual > tolerance:
            raise ValueError(
                f"start_witness_points[{witness_index}] does not satisfy "
                "original_system within the monodromy start tolerance "
                f"({original_residual:.2e} > {tolerance:.2e})"
            )
        slice_residual = _monodromy_system_residual(
            start_slice,
            point,
            variables,
        )
        if slice_residual > tolerance:
            raise ValueError(
                f"start_witness_points[{witness_index}] does not satisfy "
                "start_slice within the monodromy start tolerance "
                f"({slice_residual:.2e} > {tolerance:.2e})"
            )


def _monodromy_system_residual(
    system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> float:
    values = evaluate_system_at_point(system, point, variables)
    if np.all(np.isfinite(values)):
        return _scaled_euclidean_norm(values)
    scaled_values = evaluate_scaled_system_at_point(system, point, variables)
    if np.all(np.isfinite(scaled_values)):
        return _scaled_euclidean_norm(scaled_values)
    return float("inf")


def _coerce_witness_superset(
    name: str,
    witness_superset: Any,
    original_system: PolynomialSystem,
    variables: List[Variable],
) -> SolutionSet:
    if isinstance(witness_superset, SolutionSet):
        return witness_superset
    if isinstance(witness_superset, (str, bytes)):
        raise TypeError(
            f"{name} must be a SolutionSet or iterable of coordinate records"
        )
    try:
        witness_records = list(witness_superset)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a SolutionSet or iterable of coordinate records"
        ) from exc

    point_array = _witness_points_to_array(
        witness_records,
        variables,
        label=name,
    )
    solutions = []
    for point in point_array:
        values = {
            variable: value for variable, value in zip(variables, point)
        }
        solutions.append(
            Solution(
                values,
                residual=_monodromy_system_residual(
                    original_system,
                    point,
                    variables,
                ),
            )
        )
    result = SolutionSet(solutions, original_system)
    result._meta["source"] = name
    result._meta["variables"] = [variable.name for variable in variables]
    return result


def _validate_options_dict(name: str, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise TypeError(f"{name} must be a dictionary")
    return options.copy()


def _validate_nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _monodromy_log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _parameter_path_options(options: Dict[str, Any]) -> Dict[str, Any]:
    path_options = dict(options)
    path_options.pop('match_tol', None)
    return path_options


def _is_dimension_skip_error(error: ValueError) -> bool:
    """Return True for expected no-witness conditions in dimension scans."""
    message = str(error).lower()
    return (
        "underdetermined augmented system" in message
        or "variable list contains variable(s) not used by system" in message
        or (
            "positive-dimensional" in message
            and "witness-set" in message
        )
    )


def _match_affine_points(
    start_points: Dict[int, np.ndarray],
    final_points: Dict[int, np.ndarray],
    tracked_indices: List[int],
    n_points: int,
    match_tol: float,
) -> Tuple[List[int], Set[int], Dict[int, float]]:
    """Match returned affine witness points to starts by scaled distance."""
    mapping = [-1] * n_points
    used_final_indices: Set[int] = set()
    used_start_indices: Set[int] = set()
    min_distances = {index: float("inf") for index in tracked_indices}
    candidates: List[Tuple[float, int, int]] = []

    for final_idx in tracked_indices:
        final_point = final_points[final_idx]
        for start_idx in tracked_indices:
            distance = _scaled_affine_distance(final_point, start_points[start_idx])
            min_distances[final_idx] = min(min_distances[final_idx], distance)
            if np.isfinite(distance):
                candidates.append((distance, final_idx, start_idx))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    for distance, final_idx, start_idx in candidates:
        if distance > match_tol:
            break
        if final_idx in used_final_indices or start_idx in used_start_indices:
            continue
        mapping[final_idx] = start_idx
        used_final_indices.add(final_idx)
        used_start_indices.add(start_idx)

    return mapping, used_start_indices, min_distances


def _scaled_affine_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return a finite relative affine distance, or infinity if invalid."""
    try:
        first = np.asarray(first, dtype=complex)
        second = np.asarray(second, dtype=complex)
    except (TypeError, ValueError, OverflowError):
        return float("inf")
    if first.shape != second.shape:
        return float("inf")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        return float("inf")

    with np.errstate(over="ignore", invalid="ignore"):
        delta = first - second
    if not np.all(np.isfinite(delta)):
        return float("inf")
    distance = _scaled_euclidean_norm(delta)
    first_norm = _scaled_euclidean_norm(first)
    second_norm = _scaled_euclidean_norm(second)
    if not all(np.isfinite(value) for value in (distance, first_norm, second_norm)):
        return float("inf")
    scale = max(1.0, first_norm, second_norm)
    return distance / scale


def trace_monodromy_loops(original_system: PolynomialSystem,
                          start_slice: PolynomialSystem,
                          start_witness_points: List[Solution],
                          variables: List[Variable],
                          num_loops: int = 3,
                          parameter_tracker_options: Dict = None,
                          random_state: Any = None,
                          verbose: bool = False) -> List[Permutation]:
    """Compatibility wrapper for tracing monodromy loop permutations."""
    return track_monodromy_loop(
        original_system=original_system,
        start_slice=start_slice,
        start_witness_points=start_witness_points,
        variables=variables,
        num_loops=num_loops,
        parameter_tracker_options=parameter_tracker_options,
        random_state=random_state,
        verbose=verbose,
    )


class MonodromyBreakup:
    """Convenience object for monodromy-based component breakup.

    This wraps :func:`numerical_irreducible_decomposition` so users can keep
    the system, slice, witness superset, variables, and options together while
    running repeated decompositions.
    """

    def __init__(self,
                 original_system: PolynomialSystem,
                 slicing_system: PolynomialSystem,
                 witness_superset: Any,
                 variables: List[Variable],
                 monodromy_options: Dict = None):
        self.original_system = original_system
        self.slicing_system = slicing_system
        self.witness_superset = witness_superset
        self.variables = list(variables)
        self.monodromy_options = {} if monodromy_options is None else dict(monodromy_options)
        self.components: Optional[List[WitnessSet]] = None

    def decompose(self) -> List[WitnessSet]:
        """Run monodromy breakup and return witness sets by component."""
        self.components = numerical_irreducible_decomposition(
            original_system=self.original_system,
            slicing_system=self.slicing_system,
            witness_superset=self.witness_superset,
            variables=self.variables,
            monodromy_options=self.monodromy_options,
        )
        return self.components

    def run(self) -> List[WitnessSet]:
        """Alias for :meth:`decompose`."""
        return self.decompose()

    def __call__(self) -> List[WitnessSet]:
        """Run the breakup when the object is called."""
        return self.decompose()

def numerical_irreducible_decomposition(original_system: PolynomialSystem,
                                       slicing_system: PolynomialSystem,
                                       witness_superset: Any,
                                       variables: List[Variable],
                                       monodromy_options: Dict = None) -> List[WitnessSet]:
    """
    Perform numerical irreducible decomposition on a witness superset.
    
    This uses monodromy loops to identify the irreducible components
    within a witness superset.
    
    Args:
        original_system: The original polynomial system F.
        slicing_system: The slicing system L.
        witness_superset: The set of potential witness points W, supplied as a
            ``SolutionSet`` or an iterable of coordinate records.
        variables: System variables.
        monodromy_options: Options for the monodromy computation.
        
    Returns:
        List of WitnessSet objects, one for each irreducible component.
    """
    _validate_polynomial_system("original_system", original_system)
    _validate_polynomial_system("slicing_system", slicing_system)
    variables = _normalize_monodromy_variables(variables)
    _validate_variable_list_covers_systems(
        variables,
        original_system,
        slicing_system,
    )
    witness_superset = _coerce_witness_superset(
        "witness_superset",
        witness_superset,
        original_system,
        variables,
    )
    monodromy_options = _validate_monodromy_options(monodromy_options)
    verbose = monodromy_options.get("verbose", False)
        
    witness_points_list = witness_superset.solutions
    n_points = len(witness_points_list)
    
    if n_points == 0:
        _monodromy_log(verbose, "No witness points found in the witness superset.")
        return []
        
    dimension = len(slicing_system.equations)
    _monodromy_log(
        verbose,
        "Starting numerical irreducible decomposition for "
        f"{n_points} potential witness points of dimension {dimension}...",
    )
          
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
        parameter_tracker_options=monodromy_options.get('tracker_options', {}),
        random_state=monodromy_options.get('random_state'),
        verbose=verbose,
    )
    
    # If monodromy failed to produce any permutations, check if we can make an educated guess
    if not permutations:
        _monodromy_log(
            verbose,
            "Warning: Monodromy tracking failed to produce non-identity permutations.",
        )

        factor_components = _factorized_witness_partition(
            original_system,
            slicing_system,
            witness_points_list,
            variables,
            dimension,
            monodromy_options,
        )
        if factor_components is not None:
            _monodromy_log(
                verbose,
                "Split witness superset by factorized equation fallback.",
            )
            return factor_components
        
        # Special case: if we have a single line, curve, or smooth component, we can often detect it
        # by checking the Jacobian rank at the witness points
        if n_points > 0 and dimension == 1:
            _monodromy_log(
                verbose,
                "Attempting to determine reducibility by analyzing Jacobian rank...",
            )
            
            # Try to detect if all points are on a single component by checking rank consistency
            ranks = []
            for pt in witness_points_list:
                pt_arr = _witness_points_to_array([pt], variables)[0]
                rank = _monodromy_jacobian_rank(
                    original_system,
                    pt_arr,
                    variables,
                )
                ranks.append(rank)
                
            # If all points have the same rank, they might be on one component
            if all(r == ranks[0] for r in ranks):
                _monodromy_log(
                    verbose,
                    f"All points have consistent Jacobian rank {ranks[0]}.",
                )
                _monodromy_log(
                    verbose,
                    "This suggests they may be on a single irreducible component.",
                )
                _monodromy_log(verbose, "Returning a single component.")
                return [WitnessSet(original_system, slicing_system, witness_points_list, dimension)]
        
        # If we couldn't determine, return the superset as a single component
        _monodromy_log(verbose, "Returning the superset as a single component.")
        return [WitnessSet(original_system, slicing_system, witness_points_list, dimension)]
        
    # 2. Compute permutation group and its orbits
    _monodromy_log(
        verbose,
        f"Computing orbits from {len(permutations)} permutations...",
    )
    
    try:
        # Create permutation group and compute orbits
        group = PermutationGroup(permutations)
        orbits = group.orbits()
        _monodromy_log(
            verbose,
            f"Found {len(orbits)} orbits (potential irreducible components).",
        )
    except Exception as e:
        _monodromy_log(verbose, f"Error computing permutation group: {e}")
        _monodromy_log(verbose, "Returning the superset as a single component.")
        return [WitnessSet(original_system, slicing_system, witness_points_list, dimension)]
        
    # If orbits are empty, return all points as a single component
    if not orbits:
        _monodromy_log(
            verbose,
            "No orbits found. Returning the superset as a single component.",
        )
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
        
        _monodromy_log(
            verbose,
            f"  Component {orbit_idx+1}: dimension={ws.dimension}, degree={ws.degree}",
        )
        
        # Add these indices to used_indices
        used_indices.update(orbit_list)
    
    # Check if we missed any points
    missed_indices = set(range(n_points)) - used_indices
    if missed_indices:
        _monodromy_log(
            verbose,
            f"Warning: {len(missed_indices)} points were not assigned to any component.",
        )
        _monodromy_log(verbose, "Creating an additional component for these points.")
        
        # Create an additional component for these points
        additional_points = [witness_points_list[i] for i in sorted(missed_indices)]
        ws = WitnessSet(original_system, slicing_system, additional_points, dimension)
        irreducible_components.append(ws)
        
        _monodromy_log(
            verbose,
            f"  Additional component: dimension={ws.dimension}, degree={ws.degree}",
        )
        
    return irreducible_components


def _monodromy_jacobian_rank(
    system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
    rank_tolerance: float = 1e-8,
) -> int:
    """Compute a finite Jacobian rank for monodromy fallback diagnostics."""
    jacobian = evaluate_jacobian_at_point(system, point, variables)
    if not np.all(np.isfinite(jacobian)):
        jacobian = evaluate_scaled_jacobian_at_point(system, point, variables)
    if jacobian.size == 0 or not np.all(np.isfinite(jacobian)):
        return 0
    try:
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
    except (np.linalg.LinAlgError, ValueError, OverflowError, FloatingPointError):
        return 0
    if not np.all(np.isfinite(singular_values)):
        return 0
    return int(np.sum(singular_values > rank_tolerance))


def _factorized_witness_partition(
    original_system: PolynomialSystem,
    slicing_system: PolynomialSystem,
    witness_points: List[Solution],
    variables: List[Variable],
    dimension: int,
    monodromy_options: Dict[str, Any],
) -> Optional[List[WitnessSet]]:
    """Split a witness superset using a square-free factorization fallback."""
    factor_data = _factorized_equation_data(original_system, variables)
    if factor_data is None:
        return None

    tracker_options = monodromy_options.get("tracker_options", {})
    factor_tolerance = max(
        100.0 * float(tracker_options.get("tol", 1e-8)),
        1e-7,
    )
    grouped_points: List[List[Solution]] = [
        [] for _ in factor_data["factors"]
    ]

    for witness_index, witness_point in enumerate(witness_points):
        try:
            point = _witness_points_to_array([witness_point], variables)[0]
        except (TypeError, ValueError):
            return None

        residuals = [
            _factor_residual_norm(factor["polynomial"], point, variables)
            for factor in factor_data["factors"]
        ]
        matches = [
            factor_index
            for factor_index, residual in enumerate(residuals)
            if residual <= factor_tolerance
        ]
        if len(matches) != 1:
            return None
        grouped_points[matches[0]].append(witness_point)

    nonempty_groups = [group for group in grouped_points if group]
    if len(nonempty_groups) <= 1 or len(nonempty_groups) != len(grouped_points):
        return None

    return [
        WitnessSet(original_system, slicing_system, group, dimension)
        for group in nonempty_groups
    ]


def _factorized_equation_data(
    system: PolynomialSystem,
    variables: List[Variable],
) -> Optional[Dict[str, Any]]:
    """Return nonconstant factors for the best factorizable equation."""
    try:
        import sympy as sp
    except Exception:
        return None

    candidates = []
    for equation_index, equation in enumerate(system.equations):
        if equation.degree() <= 1:
            continue
        try:
            symbols, expr = _polynomial_to_sympy_expr(equation, variables, sp)
        except Exception:
            continue
        if expr is None:
            continue
        try:
            _, raw_factors = sp.factor_list(expr, *symbols)
        except Exception:
            continue

        factors = []
        for factor_expr, multiplicity in raw_factors:
            factor_polynomial = _sympy_expr_to_polynomial(
                factor_expr,
                variables,
                symbols,
                sp,
            )
            if factor_polynomial is None or factor_polynomial.degree() <= 0:
                continue
            factors.append({
                "expression": str(sp.factor(factor_expr)),
                "multiplicity": int(multiplicity),
                "polynomial": factor_polynomial,
            })

        if len(factors) <= 1:
            continue
        original_degree = equation.degree()
        total_factor_degree = sum(
            factor["polynomial"].degree() * factor["multiplicity"]
            for factor in factors
        )
        if total_factor_degree != original_degree:
            continue
        distinct_factor_degree = sum(
            factor["polynomial"].degree() for factor in factors
        )
        candidates.append((
            distinct_factor_degree,
            len(factors),
            original_degree,
            equation_index,
            tuple(factors),
        ))

    if not candidates:
        return None

    (
        distinct_factor_degree,
        _,
        original_degree,
        equation_index,
        factors,
    ) = min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
    return {
        "equation_index": int(equation_index),
        "original_degree": int(original_degree),
        "distinct_factor_degree": int(distinct_factor_degree),
        "factors": factors,
    }


def _factor_residual_norm(
    factor: Polynomial,
    point: np.ndarray,
    variables: List[Variable],
) -> float:
    system = PolynomialSystem([factor])
    values = evaluate_system_at_point(system, point, variables)
    if np.all(np.isfinite(values)):
        return _scaled_euclidean_norm(values)
    scaled_values = evaluate_scaled_system_at_point(system, point, variables)
    if np.all(np.isfinite(scaled_values)):
        return _scaled_euclidean_norm(scaled_values)
    return float("inf")


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
        max_dimension: Highest dimension to check (default: ambient dimension).
        solver_options: Options for the solver.
        monodromy_options: Options for the monodromy computation. Set
            ``continue_on_error=True`` to skip unexpected per-dimension
            failures and return the components found in other dimensions.
        
    Returns:
        A dictionary mapping dimension to a list of WitnessSet objects.
    """
    _validate_polynomial_system("system", system)
    if variables is None:
        variables = system.ordered_variables()
    else:
        variables = _normalize_monodromy_variables(variables)
    _validate_variable_list_covers_systems(variables, system)

    solver_options = _validate_options_dict("solver_options", solver_options)
    monodromy_options = _validate_monodromy_options(monodromy_options)
    verbose = monodromy_options.get("verbose", False)
    continue_on_error = monodromy_options.get("continue_on_error", False)
    decomposition_random_state = (
        monodromy_options["random_state"]
        if "random_state" in monodromy_options
        else solver_options.get("random_state")
    )
    decomposition_rng = _coerce_rng(decomposition_random_state)

    n_vars = len(variables)
    n_eqs = len(system.equations)
    
    # A complete intersection has expected dimension n_vars - n_eqs, but
    # redundant or dependent equations can define higher-dimensional
    # components. Scan the ambient range by default so overdetermined
    # descriptions such as [x, 2*x] still reveal the line x = 0.
    complete_intersection_dim = max(0, n_vars - n_eqs)
    ambient_max_dim = n_vars

    if max_dimension is None:
        max_dimension = ambient_max_dim
    else:
        max_dimension = _validate_nonnegative_integer(
            "max_dimension",
            max_dimension,
        )
        max_dimension = min(max_dimension, ambient_max_dim)
        
    _monodromy_log(
        verbose,
        "Computing numerical decomposition of a system with "
        f"{n_eqs} equations in {n_vars} variables.",
    )
    _monodromy_log(
        verbose,
        "Complete-intersection dimension estimate: "
        f"{complete_intersection_dim}, checking dimensions up to "
        f"{max_dimension}.",
    )
          
    # Dictionary to store components by dimension
    decomposition = {}
    
    # Start from highest dimension and work down
    for D in range(max_dimension, -1, -1):
        _monodromy_log(verbose, f"\n--- Computing components of dimension {D} ---")
        
        try:
            # 1. Compute witness superset for dimension D
            slicing_system, witness_superset = compute_witness_superset(
                system,
                variables,
                D,
                solver_options,
                random_state=decomposition_rng,
            )
            
            if len(witness_superset) == 0:
                _monodromy_log(
                    verbose,
                    f"No witness points found for dimension {D}.",
                )
                continue
                
            # 2. Perform numerical irreducible decomposition
            components_D = numerical_irreducible_decomposition(
                system, slicing_system, witness_superset, variables, monodromy_options
            )
            
            if components_D:
                decomposition[D] = components_D
                
        except ValueError as e:
            if _is_dimension_skip_error(e):
                _monodromy_log(verbose, f"Skipping dimension {D}: {e}")
                continue
            if not continue_on_error:
                raise
            _monodromy_log(verbose, f"Error computing dimension {D}: {e}")
        except Exception as e:
            if not continue_on_error:
                raise RuntimeError(
                    f"Unexpected failure while computing numerical "
                    f"decomposition for dimension {D}"
                ) from e
            _monodromy_log(verbose, f"Error computing dimension {D}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            
    return decomposition


__all__ = [
    "MonodromyBreakup",
    "track_monodromy_loop",
    "trace_monodromy_loops",
    "numerical_irreducible_decomposition",
    "compute_numerical_decomposition",
]
