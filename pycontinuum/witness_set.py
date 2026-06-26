"""
Witness set module for PyContinuum.

This module provides classes and functions for computing and manipulating
witness sets, which represent positive-dimensional components of algebraic varieties.
"""

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem, Monomial, polyvar
from pycontinuum.solver import solve, Solution, SolutionSet
from pycontinuum.start_systems import (
    _coerce_rng,
    _rng_standard_normal,
    _rng_uniform_scalar,
)
from pycontinuum.utils import (
    evaluate_scaled_system_at_point,
    evaluate_system_at_point,
    _mapping_coordinate_for_variable,
    _scaled_euclidean_norm,
)


def _ordered_witness_variables(
    original_system: PolynomialSystem,
    slicing_system: PolynomialSystem,
) -> List[Variable]:
    """Infer deterministic ambient variables for a witness set."""
    variables: List[Variable] = []
    seen = set()
    for system in (original_system, slicing_system):
        for var in system.ordered_variables():
            if var not in seen:
                variables.append(var)
                seen.add(var)
    return variables


def _normalize_witness_variables(variables: Any) -> List[Variable]:
    try:
        normalized = list(variables)
    except TypeError as exc:
        raise TypeError("variables must be an iterable of Variable objects") from exc
    for index, variable in enumerate(normalized):
        if not isinstance(variable, Variable):
            raise TypeError(f"variables[{index}] must be a Variable")
    if len(set(normalized)) != len(normalized):
        raise ValueError("variables must not contain duplicates")
    return normalized


def _validate_polynomial_system(name: str, system: Any) -> None:
    if not isinstance(system, PolynomialSystem):
        raise TypeError(f"{name} must be a PolynomialSystem")


def _validate_witness_dimension(
    dimension: Any,
    *,
    max_dimension: Optional[int] = None,
) -> int:
    if isinstance(dimension, bool) or not isinstance(dimension, Integral):
        raise TypeError("dimension must be an integer")
    dimension = int(dimension)
    if dimension < 0:
        raise ValueError("dimension must be non-negative")
    if max_dimension is not None and dimension > max_dimension:
        raise ValueError(
            f"dimension cannot exceed the number of variables ({max_dimension})"
        )
    return dimension


def _validate_variables_cover_system(
    variables: List[Variable],
    system: PolynomialSystem,
) -> None:
    missing = sorted(
        variable.name
        for variable in system.variables()
        if variable not in set(variables)
    )
    if missing:
        raise ValueError(
            "variables is missing system variable(s): " + ", ".join(missing)
        )


def _coerce_witness_point(point: Any, variables_or_size: Any, name: str) -> np.ndarray:
    if isinstance(variables_or_size, Integral):
        variables = None
        size = int(variables_or_size)
    else:
        variables = _normalize_witness_variables(variables_or_size)
        size = len(variables)

    values = None
    if isinstance(point, Mapping):
        values = point
    elif isinstance(getattr(point, "values", None), Mapping):
        values = point.values

    if values is not None and variables is not None:
        point_array = _coerce_witness_point_mapping(values, variables, name)
    else:
        try:
            point_array = np.asarray(point, dtype=complex)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be an array-like point") from exc
    if point_array.ndim != 1 or point_array.shape[0] != size:
        raise ValueError(f"{name} must contain exactly {size} coordinate(s)")
    if not np.all(np.isfinite(point_array)):
        raise ValueError(f"{name} must contain finite coordinates")
    return point_array


def _coerce_witness_point_mapping(
    values: Mapping,
    variables: List[Variable],
    name: str,
) -> np.ndarray:
    coordinates = []
    missing = []
    for variable in variables:
        found, coordinate = _mapping_coordinate_for_variable(
            values,
            variable,
            name,
        )
        if found:
            coordinates.append(coordinate)
        else:
            missing.append(variable.name)
    if missing:
        raise ValueError(
            f"{name} is missing coordinate(s): " + ", ".join(sorted(missing))
        )
    try:
        return np.asarray(coordinates, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} coordinate(s) must be numeric") from exc


def _validate_positive_finite_witness_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return numeric_value


def _validate_witness_boolean_option(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _validate_witness_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise TypeError("options must be a dictionary")
    return options.copy()


def _validate_witness_solver_options(
    solver_options: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if solver_options is None:
        return {}
    if not isinstance(solver_options, dict):
        raise TypeError("solver_options must be a dictionary")
    options = solver_options.copy()
    controlled = {"system", "variables"}
    conflicts = sorted(controlled.intersection(options))
    if conflicts:
        raise ValueError(
            "solver_options cannot override compute_witness_superset "
            "argument(s): " + ", ".join(conflicts)
        )
    if "verbose" in options and not isinstance(options["verbose"], bool):
        raise TypeError("solver_options['verbose'] must be a boolean")
    return options


def _witness_point_coordinates(
    witness_point: Any,
    variables: List[Variable],
    witness_index: int,
) -> np.ndarray:
    name = f"witness_points[{witness_index}]"
    point = _coerce_witness_point(witness_point, variables, name)
    values = _witness_point_mapping(witness_point)
    if values is not None:
        extra = _extra_witness_coordinate_names(values, variables)
        if extra:
            raise ValueError(
                f"{name} contains variable(s) outside "
                "the witness ambient variables: " + ", ".join(extra)
            )
    return point


def _witness_point_mapping(witness_point: Any) -> Optional[Mapping]:
    if isinstance(witness_point, Mapping):
        return witness_point
    values = getattr(witness_point, "values", None)
    if isinstance(values, Mapping):
        return values
    return None


def _extra_witness_coordinate_names(
    values: Mapping,
    variables: List[Variable],
) -> List[str]:
    ambient_variables = set(variables)
    ambient_names = {variable.name for variable in variables}
    extra = []
    for key in values:
        if isinstance(key, Variable):
            if key not in ambient_variables:
                extra.append(key.name)
        elif isinstance(key, str):
            if key not in ambient_names:
                extra.append(key)
        else:
            extra.append(repr(key))
    return sorted(set(extra))


def _coerce_witness_solution(
    witness_point: Any,
    variables: List[Variable],
    witness_index: int,
    original_system: PolynomialSystem,
) -> Solution:
    point = _witness_point_coordinates(witness_point, variables, witness_index)
    if isinstance(witness_point, Solution):
        return witness_point
    values = {
        variable: value for variable, value in zip(variables, point)
    }
    return Solution(
        values,
        residual=_witness_system_residual(original_system, point, variables),
    )


def _validate_witness_points_on_systems(
    original_system: PolynomialSystem,
    slicing_system: PolynomialSystem,
    witness_points: List[Solution],
    variables: List[Variable],
    tolerance: float,
) -> None:
    for witness_index, witness_point in enumerate(witness_points):
        point = _witness_point_coordinates(
            witness_point,
            variables,
            witness_index,
        )
        original_residual = _witness_system_residual(
            original_system,
            point,
            variables,
        )
        if original_residual > tolerance:
            raise ValueError(
                f"witness_points[{witness_index}] does not satisfy "
                "original_system within validation_tolerance "
                f"({original_residual:.2e} > {tolerance:.2e})"
            )
        slice_residual = _witness_system_residual(
            slicing_system,
            point,
            variables,
        )
        if slice_residual > tolerance:
            raise ValueError(
                f"witness_points[{witness_index}] does not satisfy "
                "slicing_system within validation_tolerance "
                f"({slice_residual:.2e} > {tolerance:.2e})"
            )


def _validate_target_slice(
    target_slice: Optional[PolynomialSystem],
    dimension: int,
    variables: List[Variable],
    rng: Any,
) -> PolynomialSystem:
    if target_slice is None:
        return generate_generic_slice(dimension, variables, random_state=rng)
    if not isinstance(target_slice, PolynomialSystem):
        raise TypeError("target_slice must be a PolynomialSystem")
    if len(target_slice.equations) != dimension:
        raise ValueError(
            "target_slice must contain "
            f"{dimension} equation(s) for this witness set"
        )
    return target_slice


def _slice_through_point(
    dimension: int,
    variables: List[Variable],
    point: np.ndarray,
    rng: Any,
) -> PolynomialSystem:
    slice_equations = []
    for slice_index in range(dimension):
        coefficients = (
            _rng_standard_normal(
                rng,
                len(variables),
                context=f"point slice {slice_index} coefficients real",
            )
            + 1j * _rng_standard_normal(
                rng,
                len(variables),
                context=f"point slice {slice_index} coefficients imaginary",
            )
        )
        constant = -sum(
            coefficient * value
            for coefficient, value in zip(coefficients, point)
        )
        terms = [Monomial({}, coefficient=constant)]
        for variable, coefficient in zip(variables, coefficients):
            terms.append(Monomial({variable: 1}, coefficient=coefficient))
        slice_equations.append(Polynomial(terms))
    return PolynomialSystem(slice_equations)


def _draw_witness_index(rng: Any, count: int) -> int:
    if hasattr(rng, "integers"):
        return _validated_witness_index(
            _call_witness_index_rng(rng.integers, count, "random_state.integers"),
            count,
            "random_state.integers",
        )
    if hasattr(rng, "randint"):
        return _validated_witness_index(
            _call_witness_index_rng(rng.randint, count, "random_state.randint"),
            count,
            "random_state.randint",
        )
    return _validated_witness_index(
        int(
            _rng_uniform_scalar(
                rng,
                0.0,
                float(count),
                context="witness point index",
            )
        ),
        count,
        "random_state.uniform",
    )


def _call_witness_index_rng(draw: Any, count: int, name: str) -> Any:
    try:
        return draw(count)
    except Exception as exc:
        raise ValueError(
            f"{name} failed while selecting a witness point"
        ) from exc


def _validated_witness_index(value: Any, count: int, name: str) -> int:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must return a scalar witness point index")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, Integral):
        raise ValueError(f"{name} must return a scalar witness point index")
    index = int(scalar)
    if index < 0 or index >= count:
        raise ValueError(
            f"{name} must return a witness point index in [0, {count})"
        )
    return index


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
                 witness_points: List[Any],
                 dimension: int,
                 validation_tolerance: float = 1e-8):
        """
        Initialize a witness set.
        
        Args:
            original_system: The polynomial system F defining the variety.
            slicing_system: The generic linear equations L.
            witness_points: The intersection points W = V(F) ∩ V(L).
            Witness point inputs may be coordinate vectors, coordinate
            mappings, ``Solution`` objects, or solution-like objects with a
            ``values`` mapping.
            dimension: The dimension of the component.
            validation_tolerance: Residual tolerance for checking witness
                points against F and L.
        """
        _validate_polynomial_system("original_system", original_system)
        _validate_polynomial_system("slicing_system", slicing_system)
        variables = _ordered_witness_variables(original_system, slicing_system)
        dimension = _validate_witness_dimension(
            dimension,
            max_dimension=len(variables),
        )
        validation_tolerance = _validate_positive_finite_witness_float(
            "validation_tolerance",
            validation_tolerance,
        )
        if len(slicing_system.equations) != dimension:
            raise ValueError(
                "slicing_system must contain "
                f"{dimension} equation(s) for a dimension-{dimension} "
                "witness set"
            )
        try:
            witness_points = list(witness_points)
        except TypeError as exc:
            raise TypeError("witness_points must be an iterable") from exc
        witness_points = [
            _coerce_witness_solution(
                witness_point,
                variables,
                index,
                original_system,
            )
            for index, witness_point in enumerate(witness_points)
        ]
        _validate_witness_points_on_systems(
            original_system,
            slicing_system,
            witness_points,
            variables,
            validation_tolerance,
        )
        self.original_system = original_system
        self.slicing_system = slicing_system
        self.witness_points = witness_points
        self.dimension = dimension
        self.variables = variables
        self.validation_tolerance = validation_tolerance
        # The degree of an irreducible component is the number of witness points
        self.degree = len(witness_points)
        
    def __repr__(self) -> str:
        """String representation of the witness set."""
        return (f"WitnessSet(dimension={self.dimension}, degree={self.degree}, "
                f"{len(self.witness_points)} points)")
    
    def sample_point(self, 
                    target_slice: Optional[PolynomialSystem] = None, 
                    variables: Optional[List[Variable]] = None,
                    options: Dict[str, Any] = None,
                    random_state: Any = None,
                    return_info: bool = False):
        """
        Sample a point on the component by moving to a different slice.
        
        Args:
            target_slice: A target slicing system different from the current one.
                          If None, a random slice will be generated.
            variables: The system variables. If None, inferred from the witness set.
            options: Options for parameter tracking.
            random_state: Optional seed or NumPy random generator.
            return_info: If True, return ``(point_or_none, info)`` where
                ``info`` records the selected witness path and validation
                outcome.
            
        Returns:
            A point on the component at the target slice, or None if tracking fails.
        """
        # Import here to avoid circular imports
        from pycontinuum.parameter_homotopy import ParameterHomotopy, track_parameter_path
        
        variables = _normalize_witness_variables(
            self.variables if variables is None else variables
        )
        options = _validate_witness_options(options)
        return_info = _validate_witness_boolean_option(
            "return_info",
            return_info,
        )
        rng = _coerce_rng(random_state)
            
        # Choose a random witness point to track
        if not self.witness_points:
            info = _witness_sample_info(
                witness_index=None,
                tracking_info=None,
                validated=False,
                sample_tolerance=self.validation_tolerance,
                failure_reason="empty_witness_set",
            )
            return (None, info) if return_info else None

        target_slice = _validate_target_slice(
            target_slice,
            self.dimension,
            variables,
            rng,
        )
        sample_tolerance = _witness_sample_tolerance(
            options,
            self.validation_tolerance,
        )
            
        witness_index = _draw_witness_index(rng, len(self.witness_points))
        source_point = _witness_point_coordinates(
            self.witness_points[witness_index],
            variables,
            witness_index,
        )
        
        # Create parameter homotopy from current slice to target slice
        ph = ParameterHomotopy(
            self.original_system,
            self.slicing_system,
            target_slice,
            variables
        )
        
        # Track the point to the target slice
        target_point, info = track_parameter_path(
            ph, source_point, start_t=0.0, end_t=1.0, options=options.copy()
        )
        validated = info.get("success", False) and _tracked_witness_point_is_valid(
            self.original_system,
            target_slice,
            target_point,
            variables,
            sample_tolerance,
        )
        sample_info = _witness_sample_info(
            witness_index=witness_index,
            tracking_info=info,
            validated=validated,
            sample_tolerance=sample_tolerance,
        )
        sample = target_point if validated else None
        return (sample, sample_info) if return_info else sample

    def sample_points(
                    self,
                    target_slice: Optional[PolynomialSystem] = None,
                    variables: Optional[List[Variable]] = None,
                    options: Optional[Dict[str, Any]] = None,
                    random_state: Any = None,
                    return_info: bool = False):
        """
        Track all witness points to a target slice.

        Args:
            target_slice: Target slicing system. If None, a random slice is
                generated.
            variables: Coordinate order. If None, inferred from the witness set.
            options: Options for parameter tracking.
            random_state: Optional seed or NumPy random generator.
            return_info: If True, return ``(points, info_records)`` where each
                info record corresponds to one witness path.

        Returns:
            Successfully tracked points on the component at the target slice.
        """
        from pycontinuum.parameter_homotopy import ParameterHomotopy, track_parameter_path

        variables = _normalize_witness_variables(
            self.variables if variables is None else variables
        )
        options = _validate_witness_options(options)
        return_info = _validate_witness_boolean_option(
            "return_info",
            return_info,
        )
        rng = _coerce_rng(random_state)
        if not self.witness_points:
            return ([], []) if return_info else []

        target_slice = _validate_target_slice(
            target_slice,
            self.dimension,
            variables,
            rng,
        )
        sample_tolerance = _witness_sample_tolerance(
            options,
            self.validation_tolerance,
        )
        ph = ParameterHomotopy(
            self.original_system,
            self.slicing_system,
            target_slice,
            variables,
        )

        tracked_points = []
        info_records = []
        for witness_index, witness_point in enumerate(self.witness_points):
            source_point = _witness_point_coordinates(
                witness_point,
                variables,
                witness_index,
            )
            target_point, info = track_parameter_path(
                ph,
                source_point,
                start_t=0.0,
                end_t=1.0,
                options=options.copy(),
            )
            validated = info.get("success", False) and _tracked_witness_point_is_valid(
                self.original_system,
                target_slice,
                target_point,
                variables,
                sample_tolerance,
            )
            if validated:
                tracked_points.append(target_point)
            info_records.append(
                _witness_sample_info(
                    witness_index=witness_index,
                    tracking_info=info,
                    validated=validated,
                    sample_tolerance=sample_tolerance,
                )
            )
        return (tracked_points, info_records) if return_info else tracked_points
            
    def is_point_on_component(self,
                             point: Any,
                             variables: Optional[List[Variable]] = None,
                             tolerance: float = 1e-8,
                             random_state: Any = None,
                             options: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if a point lies on this component.
        
        This uses parameter homotopy-based membership test.
        
        Args:
            point: The point to test, supplied as a coordinate vector, mapping
                keyed by variables or variable names, or an object with a
                ``values`` mapping.
            variables: The system variables. If None, inferred from the witness set.
            tolerance: Tolerance for equality.
            random_state: Optional seed or NumPy random generator.
            options: Optional parameter tracking options.
            
        Returns:
            True if the point is on this component, False otherwise.
        """
        from pycontinuum.parameter_homotopy import ParameterHomotopy, track_parameter_path

        variables = _normalize_witness_variables(
            self.variables if variables is None else variables
        )
        tolerance = _validate_positive_finite_witness_float(
            "tolerance", tolerance
        )
        point = _coerce_witness_point(point, variables, "point")
        if options is None:
            options = {}
        tracking_options = _validate_witness_options(options)
        tracking_options.setdefault("tol", min(tolerance, 1e-8))
        
        # Check if the point satisfies the original system
        if _witness_system_residual(
            self.original_system,
            point,
            variables,
        ) > tolerance:
            return False

        rng = _coerce_rng(random_state)
            
        # Generate a random generic slice through the point
        # We need to create D linear equations that all pass through the test point
        target_slice = _slice_through_point(
            self.dimension,
            variables,
            point,
            rng,
        )
        parameter_homotopy = ParameterHomotopy(
            self.original_system,
            self.slicing_system,
            target_slice,
            variables,
        )

        # A degree-d component has d witness paths to the target slice. Testing
        # only one path can miss the queried point even when it lies on the
        # component, so track all available witnesses until one matches.
        for witness_index, witness_point in enumerate(self.witness_points):
            source_point = _witness_point_coordinates(
                witness_point,
                variables,
                witness_index,
            )
            target_point, info = track_parameter_path(
                parameter_homotopy,
                source_point,
                start_t=0.0,
                end_t=1.0,
                options=tracking_options.copy(),
            )
            if not info.get("success", False):
                continue
            if _scaled_euclidean_norm(target_point - point) < tolerance:
                return True
        return False


def _witness_sample_info(
    *,
    witness_index: Optional[int],
    tracking_info: Optional[Dict[str, Any]],
    validated: bool,
    sample_tolerance: float,
    failure_reason: Optional[str] = None,
) -> Dict[str, Any]:
    tracking_success = bool(
        tracking_info.get("success", False)
        if isinstance(tracking_info, dict)
        else False
    )
    if failure_reason is None and not validated:
        if tracking_success:
            failure_reason = "invalid_sample"
        elif isinstance(tracking_info, dict):
            failure_reason = tracking_info.get("failure_reason") or "tracking_failed"
        else:
            failure_reason = "tracking_failed"

    return {
        "success": bool(validated),
        "validated": bool(validated),
        "tracking_success": tracking_success,
        "witness_index": witness_index,
        "sample_tolerance": float(sample_tolerance),
        "failure_reason": failure_reason,
        "tracking_info": tracking_info,
    }


def _witness_system_residual(
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


def _witness_sample_tolerance(
    options: Dict[str, Any],
    validation_tolerance: float,
) -> float:
    if "tol" not in options:
        return validation_tolerance
    tol = _validate_positive_finite_witness_float("options['tol']", options["tol"])
    return max(validation_tolerance, 10.0 * tol)


def _tracked_witness_point_is_valid(
    original_system: PolynomialSystem,
    target_slice: PolynomialSystem,
    point: Any,
    variables: List[Variable],
    tolerance: float,
) -> bool:
    try:
        point = _coerce_witness_point(point, variables, "target_point")
    except (TypeError, ValueError):
        return False
    if _witness_system_residual(original_system, point, variables) > tolerance:
        return False
    if _witness_system_residual(target_slice, point, variables) > tolerance:
        return False
    return True


def generate_generic_slice(dimension: int,
                           variables: List[Variable],
                           random_state: Any = None) -> PolynomialSystem:
    """
    Generate D random linear equations in the given variables.
    
    These represent a generic codimension-D linear subspace of the ambient space.
    
    Args:
        dimension: Number of linear equations to generate (D).
        variables: The variables to use.
        random_state: Optional seed or NumPy random generator.
        
    Returns:
        A PolynomialSystem representing the slicing system L.
    """
    variables = _normalize_witness_variables(variables)
    dimension = _validate_witness_dimension(
        dimension,
        max_dimension=len(variables),
    )
    n_vars = len(variables)
    slice_eqs = []
    rng = _coerce_rng(random_state)
    
    for slice_index in range(dimension):
        # Generate random complex coefficients for the linear equation
        # Using standard normal distribution for both real and imaginary parts
        coeffs = (
            _rng_standard_normal(
                rng,
                n_vars,
                context=f"generic slice {slice_index} coefficients real",
            )
            + 1j * _rng_standard_normal(
                rng,
                n_vars,
                context=f"generic slice {slice_index} coefficients imaginary",
            )
        )
        const = (
            _rng_standard_normal(
                rng,
                context=f"generic slice {slice_index} constant real",
            )
            + 1j * _rng_standard_normal(
                rng,
                context=f"generic slice {slice_index} constant imaginary",
            )
        )
        
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
                            solver_options: Dict = None,
                            random_state: Any = None) -> Tuple[PolynomialSystem, SolutionSet]:
    """
    Compute a witness superset for components of a given dimension.
    
    This creates D generic linear equations, combines them with the original 
    system, and solves the resulting square system to find potential witness points.
    
    Args:
        original_system: The system defining the variety.
        variables: List of variables.
        dimension: The dimension of components to find witness points for.
        solver_options: Options passed to the solver.
        random_state: Optional seed or NumPy random generator.
        
    Returns:
        Tuple of (slicing_system, witness_superset).
    """
    solver_options = _validate_witness_solver_options(solver_options)
    _validate_polynomial_system("original_system", original_system)
    variables = _normalize_witness_variables(variables)
    _validate_variables_cover_system(variables, original_system)
    rng = _coerce_rng(random_state)
        
    n_equations = len(original_system.equations)
    n_variables = len(variables)
    dimension = _validate_witness_dimension(
        dimension,
        max_dimension=n_variables,
    )
        
    # Generate D generic linear slicing equations L
    slicing_system = generate_generic_slice(dimension, variables, random_state=rng)
    
    # Create the augmented system F' = (F, L)
    augmented_equations = original_system.equations + slicing_system.equations
    augmented_system = PolynomialSystem(augmented_equations)
    
    n_aug_equations = len(augmented_system.equations)
    if n_aug_equations < n_variables:
        raise ValueError(
            "The requested dimension leaves an underdetermined augmented "
            f"system with {n_aug_equations} equations in {n_variables} "
            "variables; add more slices or request a larger component dimension"
        )

    solver_options.setdefault("random_state", rng)
    solver_options.setdefault("allow_underdetermined", False)
    verbose = solver_options.get("verbose", False)
    if verbose:
        print(
            f"Solving augmented witness system with {n_aug_equations} "
            f"equations in {n_variables} variables..."
        )

    witness_superset = solve(
        augmented_system,
        variables=variables,
        **solver_options
    )
    witness_superset._meta["witness_set"] = {
        "dimension": dimension,
        "original_equations": n_equations,
        "slice_equations": len(slicing_system.equations),
        "augmented_equations": n_aug_equations,
        "variables": tuple(variable.name for variable in variables),
    }

    if verbose:
        print(f"Found {len(witness_superset)} potential witness points.")
    return slicing_system, witness_superset
