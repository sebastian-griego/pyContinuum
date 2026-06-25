"""
Main solver module for PyContinuum.

This module provides functions to solve polynomial systems using
homotopy continuation methods.
"""

from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from itertools import product
from math import gcd
import math
from numbers import Integral, Number, Rational, Real
import time

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable, Iterator

from pycontinuum.polynomial import Variable, Monomial, Polynomial, PolynomialSystem, polyvar as _polyvar
from pycontinuum.endgame import _validate_endgame_options
from pycontinuum.start_systems import (
    generate_total_degree_start_system,
    _coerce_rng,
    _random_unit_complex as _start_random_unit_complex,
    _rng_standard_normal,
)
from pycontinuum.tracking import track_paths, _validate_gamma
from pycontinuum.utils import (
    _coerce_point_for_variables,
    evaluate_backward_error_at_point,
    evaluate_jacobian_at_point,
    evaluate_scaled_system_at_point,
    evaluate_scaled_jacobian_at_point,
    newton_corrector,
    _evaluate_scaled_term,
    _mapping_coordinate_for_variable,
    _safe_scaled_coefficient,
    _scaled_evaluation_is_exact_zero,
    _scaled_euclidean_norm,
    _polynomial_coefficient_scale,
    _strict_json_value,
    solve_linear_system,
)


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
        self.values = _validate_solution_values(values)
        self.residual = _validate_solution_residual(residual)
        self.scaled_residual = None
        self.backward_error = None
        self.is_singular = _validate_boolean_option("is_singular", is_singular)
        self.path_index = _validate_solution_path_index(path_index)
        self.multiplicity = 1
        self.path_indices = (path_index,) if path_index is not None else ()
        self.root_indices = ()
        self.cluster_radius = 0.0
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

        if getattr(self, "multiplicity", 1) > 1:
            status += f", multiplicity={self.multiplicity}"

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
        tol = _validate_nonnegative_finite_float("tol", tol)
        return all(abs(val.imag) <= tol for val in self.values.values())

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
        tol = _validate_nonnegative_finite_float("tol", tol)
        # Ensure is_real is called correctly now
        return self.is_real(tol=tol) and all(val.real > -tol for val in self.values.values())

    def point(
        self,
        variables: Optional[List[Variable]] = None,
        *,
        real: bool = False,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """Return this solution as an ordered coordinate vector.

        Args:
            variables: Coordinate order. If omitted, solution variables are
                sorted by name.
            real: If True, return a real-valued array and reject coordinates
                with imaginary part larger than ``tol``.
            tol: Tolerance for real-coordinate validation.
        """
        real = _validate_boolean_option("real", real)
        if variables is None:
            if not isinstance(self.values, dict):
                raise TypeError("solution must expose a values dictionary")
            variables = tuple(
                sorted(
                    self.values.keys(),
                    key=lambda variable: getattr(variable, "name", repr(variable)),
                )
            )
            _validate_coordinate_variables(variables)
        else:
            variables = _normalize_coordinate_variables(variables)
        point = _solution_point_from_values(
            self.values,
            variables,
            label="Solution",
        )
        if real:
            tol = _validate_nonnegative_finite_float("tol", tol)
            if not np.all(np.abs(point.imag) <= tol):
                raise ValueError("Solution contains non-real coordinate(s)")
            return point.real.astype(float)
        return point

    def as_dict(
        self,
        include_metadata: bool = True,
        *,
        strict_json: bool = False,
    ) -> Dict[str, Any]:
        """Return a JSON-friendly representation of this solution."""
        include_metadata = _validate_boolean_option(
            "include_metadata",
            include_metadata,
        )
        strict_json = _validate_boolean_option("strict_json", strict_json)
        result: Dict[str, Any] = {
            "values": {
                variable.name: _complex_metadata(value)
                for variable, value in sorted(
                    self.values.items(),
                    key=lambda item: item[0].name,
                )
            },
            "residual": float(self.residual),
            "scaled_residual": (
                None if self.scaled_residual is None
                else float(self.scaled_residual)
            ),
            "backward_error": (
                None if self.backward_error is None
                else float(self.backward_error)
            ),
            "is_singular": bool(self.is_singular),
            "multiplicity": int(getattr(self, "multiplicity", 1)),
        }
        if include_metadata:
            result.update({
                "path_index": _metadata_value(self.path_index),
                "path_indices": _metadata_value(getattr(self, "path_indices", ())),
                "root_indices": _metadata_value(getattr(self, "root_indices", ())),
                "cluster_radius": float(getattr(self, "cluster_radius", 0.0)),
                "winding_number": _metadata_value(self.winding_number),
                "path_info": _metadata_value(getattr(self, "path_info", None)),
            })
            if hasattr(self, "refinement"):
                result["refinement"] = _metadata_value(self.refinement)
        if strict_json:
            return _strict_json_value(result)
        return result

    def distance(self, other: Any, variables: List[Variable]) -> float:
        """Compute distance to another coordinate record."""
        ordered_variables = _normalize_coordinate_variables(variables)
        left = _coerce_point_for_variables(
            self,
            list(ordered_variables),
            "Solution",
            allow_nonfinite=True,
        )
        right = _coerce_point_for_variables(
            other,
            list(ordered_variables),
            "other solution",
            allow_nonfinite=True,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            difference = left - right
        return _scaled_euclidean_norm(difference)

class SolutionSet:
    """Class representing a set of solutions to a polynomial system."""
    
    def __init__(self, solutions: List[Solution], system: PolynomialSystem):
        """Initialize a solution set.
        Args:
            solutions: Iterable of Solution objects
            system: The polynomial system that was solved
        """
        _validate_polynomial_system("system", system)
        if isinstance(solutions, (str, bytes)):
            raise TypeError(
                "solutions must be an iterable of Solution objects"
            )
        try:
            solution_list = list(solutions)
        except TypeError as exc:
            raise TypeError(
                "solutions must be an iterable of Solution objects"
            ) from exc
        for index, solution in enumerate(solution_list):
            if not isinstance(solution, Solution):
                raise TypeError(f"solutions[{index}] must be a Solution")

        self.solutions = solution_list
        self.system = system
        self._meta = {}  # For storing metadata about the solve process

    @classmethod
    def from_points(
        cls,
        points: Any,
        system: PolynomialSystem,
        variables: Optional[List[Variable]] = None,
        *,
        singularity_threshold: float = 1e12,
    ) -> 'SolutionSet':
        """Build a solution set from external coordinate records.

        Args:
            points: Iterable of coordinate vectors, coordinate mappings,
                ``Solution`` objects, or solution-like objects with a
                ``values`` mapping.
            system: Polynomial system used to compute residual metadata.
            variables: Optional coordinate order. Defaults to the system's
                deterministic variable order.
            singularity_threshold: Condition threshold used to mark singular
                imported candidates.

        Returns:
            A :class:`SolutionSet` containing standard :class:`Solution`
            objects with residual, scaled-residual, backward-error, and
            singularity metadata.
        """
        _validate_polynomial_system("system", system)
        singularity_threshold = _validate_positive_finite_float(
            "singularity_threshold",
            singularity_threshold,
        )
        ordered_variables = tuple(_ordered_variables(system, variables))
        _validate_variables_cover_system(system, list(ordered_variables))
        if isinstance(points, (str, bytes)):
            raise TypeError("points must be an iterable of coordinate records")
        try:
            point_list = list(points)
        except TypeError as exc:
            raise TypeError(
                "points must be an iterable of coordinate records"
            ) from exc

        solutions = []
        for index, point_record in enumerate(point_list):
            point = _coerce_point_for_variables(
                point_record,
                list(ordered_variables),
                f"points[{index}]",
                allow_nonfinite=True,
            )
            values = {
                variable: value
                for variable, value in zip(ordered_variables, point)
            }
            residual = _residual_norm(system, values)
            solution = Solution(
                values,
                residual=residual,
                is_singular=_is_singular(
                    system,
                    values,
                    ordered_variables,
                    singularity_threshold,
                    rank_tolerance=_rank_tolerance_for_tol(1e-12),
                ),
                path_index=getattr(point_record, "path_index", None),
            )
            if isinstance(point_record, Solution):
                _copy_solution_attributes(point_record, solution)
            solution.scaled_residual = _scaled_residual_norm(system, values)
            solution.backward_error = _backward_error_norm(system, values)
            solutions.append(solution)

        result = cls(solutions, system)
        result._meta["source"] = "from_points"
        result._meta["variables"] = [variable.name for variable in ordered_variables]
        result._meta["singularity_threshold"] = singularity_threshold
        return result
        
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

    def __iter__(self) -> Iterator[Solution]:
        """Iterate over solutions in stored order."""
        return iter(self.solutions)

    def to_array(
        self,
        variables: Optional[List[Variable]] = None,
        *,
        real: bool = False,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """Return solutions as a dense array with one row per solution.

        Args:
            variables: Coordinate order. If omitted, the solved system's
                deterministic variable order is used.
            real: If True, return a real-valued array and reject non-real
                solution coordinates.
            tol: Tolerance for real-coordinate validation.
        """
        real = _validate_boolean_option("real", real)
        ordered_variables = list(_solution_set_variables(self, variables))
        dtype = float if real else complex
        if not self.solutions:
            return np.empty((0, len(ordered_variables)), dtype=dtype)
        rows = [
            solution.point(ordered_variables, real=real, tol=tol)
            for solution in self.solutions
        ]
        return np.vstack(rows).astype(dtype, copy=False)

    def nearest(
        self,
        point: Any,
        variables: Optional[List[Variable]] = None,
        *,
        return_distance: bool = False,
    ) -> Union[Solution, Tuple[Solution, float]]:
        """Return the stored solution nearest to a coordinate record.

        Distances are computed with the same scaled Euclidean norm used by
        solution clustering, so very large coordinates remain comparable
        without overflowing.

        Args:
            point: Coordinate vector, mapping, :class:`Solution`, or
                solution-like object exposing a ``values`` mapping.
            variables: Coordinate order. If omitted, the solved system's
                deterministic variable order is used.
            return_distance: If True, return ``(solution, distance)``.
        """
        return_distance = _validate_boolean_option(
            "return_distance",
            return_distance,
        )
        if not self.solutions:
            raise ValueError("cannot find nearest solution in an empty SolutionSet")

        ordered_variables = list(_solution_set_variables(self, variables))
        target = _coerce_point_for_variables(
            point,
            ordered_variables,
            "point",
            allow_nonfinite=True,
        )

        nearest_solution = None
        nearest_distance = float("inf")
        for solution in self.solutions:
            solution_point = _coerce_point_for_variables(
                solution,
                ordered_variables,
                "solution",
                allow_nonfinite=True,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                difference = solution_point - target
            distance = _scaled_euclidean_norm(difference)
            if distance < nearest_distance:
                nearest_solution = solution
                nearest_distance = distance

        if nearest_solution is None:
            raise ValueError("no finite nearest solution distance could be computed")
        if return_distance:
            return nearest_solution, nearest_distance
        return nearest_solution

    def as_dicts(
        self,
        include_metadata: bool = True,
        *,
        strict_json: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return all solutions as JSON-friendly dictionaries."""
        include_metadata = _validate_boolean_option(
            "include_metadata",
            include_metadata,
        )
        strict_json = _validate_boolean_option("strict_json", strict_json)
        return [
            solution.as_dict(
                include_metadata=include_metadata,
                strict_json=strict_json,
            )
            for solution in self.solutions
        ]

    def as_dict(
        self,
        include_metadata: bool = True,
        *,
        strict_json: bool = False,
    ) -> Dict[str, Any]:
        """Return this solution set as a JSON-friendly dictionary."""
        include_metadata = _validate_boolean_option(
            "include_metadata",
            include_metadata,
        )
        strict_json = _validate_boolean_option("strict_json", strict_json)
        result: Dict[str, Any] = {
            "solutions": self.as_dicts(
                include_metadata=include_metadata,
                strict_json=strict_json,
            ),
            "solution_count": len(self.solutions),
        }
        if include_metadata:
            result.update({
                "metadata": _metadata_value(self._meta),
                "system": {
                    "equations": [
                        repr(equation)
                        for equation in self.system.equations
                    ],
                    "variables": [
                        variable.name
                        for variable in self.system.ordered_variables()
                    ],
                },
            })
        if strict_json:
            return _strict_json_value(result)
        return result
    
    def filter(self,
               real: Optional[bool] = None,
               positive: Optional[bool] = None,
               tol: float = 1e-10,
               max_residual: Optional[float] = None,
               max_scaled_residual: Optional[float] = None,
               max_backward_error: Optional[float] = None,
               custom_filter: Optional[Callable[[Solution], bool]] = None) -> 'SolutionSet':
        """Filter solutions based on criteria.

        Args:
            real: If True, only include real solutions. If False, only non-real. If None, no filter.
            positive: If True, only include positive real solutions. If False, only non-positive real. If None, no filter.
            tol: Tolerance used for real and positive checks (default: 1e-10).
            max_residual: Maximum residual threshold.
            max_scaled_residual: Maximum coefficient-scaled residual threshold.
            max_backward_error: Maximum backward-error threshold.
            custom_filter: Custom filter function taking a Solution and returning bool.

        Returns:
            A new SolutionSet with filtered solutions.
        """
        real = _validate_optional_boolean_filter("real", real)
        positive = _validate_optional_boolean_filter("positive", positive)
        tol = _validate_positive_finite_float("tol", tol)
        if max_residual is not None:
            max_residual = _validate_nonnegative_finite_float(
                "max_residual",
                max_residual,
            )
        if max_scaled_residual is not None:
            max_scaled_residual = _validate_nonnegative_finite_float(
                "max_scaled_residual",
                max_scaled_residual,
            )
        if max_backward_error is not None:
            max_backward_error = _validate_nonnegative_finite_float(
                "max_backward_error",
                max_backward_error,
            )
        if custom_filter is not None and not callable(custom_filter):
            raise TypeError("custom_filter must be callable")

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

        if max_scaled_residual is not None:
            filtered_sols = [
                sol for sol in filtered_sols
                if _solution_scaled_residual_or_inf(self.system, sol)
                <= max_scaled_residual
            ]

        if max_backward_error is not None:
            filtered_sols = [
                sol for sol in filtered_sols
                if _solution_backward_error_or_inf(self.system, sol)
                <= max_backward_error
            ]

        if custom_filter is not None:
            filtered_sols = [sol for sol in filtered_sols if custom_filter(sol)]

        result = SolutionSet(filtered_sols, self.system)
        result._meta = self._meta.copy()
        result._meta['is_filtered'] = True
        return result

    def diagnostics(self, **kwargs):
        """Audit residuals, Jacobian rank, conditioning, and duplicates.

        Keyword arguments are forwarded to
        :func:`pycontinuum.validation.diagnose_solutions`.
        """
        from pycontinuum.validation import diagnose_solutions

        return diagnose_solutions(self, **kwargs)

    def refine(self, **kwargs) -> 'SolutionSet':
        """Newton-polish every solution in this set.

        Keyword arguments are forwarded to :func:`refine_solutions`.
        The original solution set is not mutated.
        """
        return refine_solutions(self, **kwargs)

def solve(system: PolynomialSystem, 
          start_system=None,
          start_solutions=None,
          variables=None,
          tol: float = 1e-10,
          verbose: bool = False,
          store_paths: bool = False,
          use_endgame: bool = True,
          endgame_options: Optional[Dict[str, Any]] = None,
          tracking_options: Optional[Dict[str, Any]] = None,
          deduplication_tol_factor: float = 10.0,
          singular_deduplication_tol: float = 1e-3,
          allow_underdetermined: bool = False,
          scale_equations: bool = True,
          max_paths: Optional[int] = None,
          random_state: Any = None,
          _allow_zero_max_paths: bool = False) -> SolutionSet:
    """Solve a polynomial system using homotopy continuation.
    Args:
        system: The polynomial system to solve, a single polynomial equation,
            or a parseable system string accepted by ``PolynomialSystem.parse``.
        start_system: Optional custom start system (default: total-degree
            homotopy). Accepts the same one-equation and parseable string forms.
        start_solutions: Optional known solutions of the start system. Entries
            may be coordinate vectors, Solution objects, or mappings from
            variables/variable names to coordinates.
        variables: Optional list of variables to use (default: extracted from system)
        tol: Tolerance for path tracking and solution classification
        verbose: Whether to print progress information
        store_paths: Whether to store path tracking points
        use_endgame: Whether to use endgame methods for singular solutions
        endgame_options: Optional dictionary of options for the endgame procedure
        tracking_options: Optional path-tracker options. Supported keys are
            ``min_step_size``, ``max_step_size``, ``max_newton_iters``,
            ``max_steps``, ``max_predictor_norm``, ``gamma``,
            ``endgame_start``, ``singularity_threshold``, and
            ``final_singularity_threshold``, ``predictor``, and ``n_jobs``.
        deduplication_tol_factor: Factor multiplied by `tol` for regular solution deduplication.
        singular_deduplication_tol: Absolute tolerance for singular solution deduplication.
        allow_underdetermined: Permit explicit variable lists with free
            coordinates long enough to report a positive-dimensional solve
            error. ``solve`` still returns finite zero-dimensional solution
            sets only.
        scale_equations: Whether to normalize each equation by its largest
            coefficient magnitude before solving. Residuals are still reported
            against the original system.
        max_paths: Optional guard on the number of continuation paths to track.
            Generated total-degree homotopies are checked before start-solution
            generation; custom start data is checked after validation.
        random_state: Optional seed or NumPy random generator for reproducible start systems
        
    Returns:
        A SolutionSet containing all found solutions
    """
    start_time = time.time()
    if variables is not None:
        variables = _normalize_variable_list(variables)
    system = _coerce_parseable_polynomial_system(
        "system",
        system,
        variables=variables,
    )
    _validate_polynomial_system("system", system)
    verbose = _validate_boolean_option("verbose", verbose)
    store_paths = _validate_boolean_option("store_paths", store_paths)
    use_endgame = _validate_boolean_option("use_endgame", use_endgame)
    allow_underdetermined = _validate_boolean_option(
        "allow_underdetermined", allow_underdetermined
    )
    scale_equations = _validate_boolean_option("scale_equations", scale_equations)
    _allow_zero_max_paths = _validate_boolean_option(
        "_allow_zero_max_paths",
        _allow_zero_max_paths,
    )
    tol = _validate_positive_finite_float("tol", tol)
    deduplication_tol_factor = _validate_positive_finite_float(
        "deduplication_tol_factor", deduplication_tol_factor
    )
    singular_deduplication_tol = _validate_positive_finite_float(
        "singular_deduplication_tol", singular_deduplication_tol
    )
    max_paths = _validate_max_paths(
        max_paths,
        allow_zero=_allow_zero_max_paths,
    )
    
    tracker_kwargs = _validate_tracking_options(tracking_options)
    endgame_options = _validate_endgame_options(
        endgame_options,
        name="endgame_options",
    )
    rng = _coerce_rng(random_state)
    generated_gamma = False
    if "gamma" not in tracker_kwargs:
        tracker_kwargs["gamma"] = _random_unit_complex(rng)
        generated_gamma = True
    effective_endgame_options = (
        _endgame_options_with_random_state(
            endgame_options,
            rng,
            include_random_state=random_state is not None,
        )
        if use_endgame
        else endgame_options
    )

    # Get the variables in the system
    if variables is None:
        variables = system.ordered_variables()
    _validate_variables_cover_system(
        system,
        variables,
        allow_extra=allow_underdetermined,
    )
    custom_start_requested = start_system is not None or start_solutions is not None
    if (start_system is None) != (start_solutions is None):
        raise ValueError(
            "start_system and start_solutions must be provided together"
        )
    if start_system is not None:
        start_system = _coerce_parseable_polynomial_system(
            "start_system",
            start_system,
            variables=variables,
        )
        
    if verbose:
        print(f"Variables used for solving: {variables}")

    working_system, preprocessing = _preprocess_system(
        system,
        tol=tol,
        remove_duplicate_equations=not custom_start_requested,
    )
    working_system, equation_scaling = _scale_equation_system(
        working_system,
        enabled=scale_equations,
    )
    default_square_up = _square_up_metadata(working_system, variables, method="none")
    if preprocessing["inconsistent_constants"]:
        result = SolutionSet([], system)
        result._meta['total_paths'] = 0
        result._meta['successful_paths'] = 0
        result._meta['failed_paths'] = 0
        result._meta['solve_time'] = time.time() - start_time
        result._meta['raw_solutions_found'] = 0
        result._meta['tracking_options'] = tracker_kwargs.copy()
        result._meta['generated_gamma'] = generated_gamma
        result._meta['preprocessing'] = preprocessing
        result._meta['equation_scaling'] = equation_scaling
        result._meta['square_up'] = default_square_up
        result._meta['path_summary'] = _summarize_path_results([], [], 0)
        return result

    if preprocessing["removed_zero_equations"] and verbose:
        print(
            "Removed "
            f"{preprocessing['removed_zero_equations']} identically zero "
            "equation(s) before tracking"
        )
    if preprocessing["removed_duplicate_equations"] and verbose:
        print(
            "Removed "
            f"{preprocessing['removed_duplicate_equations']} scalar-multiple "
            "duplicate equation(s) before tracking"
        )

    if not system.equations:
        if variables:
            _raise_no_zero_dimensional_constraints_error(
                "empty system",
                variables,
            )
        solution = Solution(values={}, residual=0.0)
        solution.scaled_residual = 0.0
        solution.backward_error = 0.0
        result = SolutionSet([solution], system)
        result._meta['total_paths'] = 0
        result._meta['successful_paths'] = 0
        result._meta['failed_paths'] = 0
        result._meta['solve_time'] = time.time() - start_time
        result._meta['raw_solutions_found'] = 1
        result._meta['tracking_options'] = tracker_kwargs.copy()
        result._meta['generated_gamma'] = generated_gamma
        result._meta['preprocessing'] = preprocessing
        result._meta['equation_scaling'] = equation_scaling
        result._meta['square_up'] = default_square_up
        result._meta['path_summary'] = _summarize_path_results([], [], 0)
        return result

    if not working_system.equations:
        if variables:
            _raise_no_zero_dimensional_constraints_error(
                "After removing zero equations, the system",
                variables,
            )
        solution = Solution(values={}, residual=0.0)
        solution.scaled_residual = 0.0
        solution.backward_error = 0.0
        result = SolutionSet([solution], system)
        result._meta['total_paths'] = 0
        result._meta['successful_paths'] = 0
        result._meta['failed_paths'] = 0
        result._meta['solve_time'] = time.time() - start_time
        result._meta['raw_solutions_found'] = 1
        result._meta['tracking_options'] = tracker_kwargs.copy()
        result._meta['generated_gamma'] = generated_gamma
        result._meta['preprocessing'] = preprocessing
        result._meta['equation_scaling'] = equation_scaling
        result._meta['square_up'] = default_square_up
        result._meta['path_summary'] = _summarize_path_results([], [], 0)
        return result

    if not custom_start_requested and _is_linear_system(working_system):
        linear_result = _solve_linear_system_direct(
            original_system=system,
            working_system=working_system,
            variables=variables,
            tol=tol,
            start_time=start_time,
            tracker_kwargs=tracker_kwargs,
            generated_gamma=generated_gamma,
            preprocessing=preprocessing,
            equation_scaling=equation_scaling,
            square_up=default_square_up,
        )
        return linear_result

    if not custom_start_requested and _is_univariate_system(working_system, variables):
        univariate_result = _solve_univariate_system_direct(
            original_system=system,
            working_system=working_system,
            variables=variables,
            tol=tol,
            start_time=start_time,
            tracker_kwargs=tracker_kwargs,
            generated_gamma=generated_gamma,
            preprocessing=preprocessing,
            equation_scaling=equation_scaling,
            square_up=default_square_up,
            deduplication_tol_factor=deduplication_tol_factor,
            singular_deduplication_tol=singular_deduplication_tol,
        )
        if univariate_result is not None:
            return univariate_result

    if not custom_start_requested:
        reduction = _reduce_coordinate_assignments(
            working_system,
            variables,
            tol,
        )
        if reduction is not None:
            return _solve_reduced_coordinate_system(
                original_system=system,
                working_system=working_system,
                reduced_system=reduction["system"],
                assignments=reduction["assignments"],
                reduced_variables=reduction["variables"],
                all_variables=variables,
                tol=tol,
                verbose=verbose,
                store_paths=store_paths,
                use_endgame=use_endgame,
                endgame_options=effective_endgame_options,
                tracking_options=tracker_kwargs,
                deduplication_tol_factor=deduplication_tol_factor,
                singular_deduplication_tol=singular_deduplication_tol,
                allow_underdetermined=allow_underdetermined,
                rng=rng,
                generated_gamma=generated_gamma,
                preprocessing=preprocessing,
                equation_scaling=equation_scaling,
                square_up=default_square_up,
                start_time=start_time,
                scale_equations=scale_equations,
                max_paths=max_paths,
            )

        affine_reduction = _reduce_affine_linear_equation(
            working_system,
            variables,
            tol,
        )
        if affine_reduction is not None:
            return _solve_reduced_affine_system(
                original_system=system,
                working_system=working_system,
                reduced_system=affine_reduction["system"],
                eliminated_variable=affine_reduction["eliminated_variable"],
                expression=affine_reduction["expression"],
                reduced_variables=affine_reduction["variables"],
                all_variables=variables,
                metadata=affine_reduction["metadata"],
                tol=tol,
                verbose=verbose,
                store_paths=store_paths,
                use_endgame=use_endgame,
                endgame_options=effective_endgame_options,
                tracking_options=tracker_kwargs,
                deduplication_tol_factor=deduplication_tol_factor,
                singular_deduplication_tol=singular_deduplication_tol,
                allow_underdetermined=allow_underdetermined,
                rng=rng,
                generated_gamma=generated_gamma,
                preprocessing=preprocessing,
                equation_scaling=equation_scaling,
                square_up=default_square_up,
                start_time=start_time,
                scale_equations=scale_equations,
                max_paths=max_paths,
            )

        monomial_branch_result = _solve_monomial_zero_branches(
            original_system=system,
            working_system=working_system,
            variables=variables,
            tol=tol,
            verbose=verbose,
            store_paths=store_paths,
            use_endgame=use_endgame,
            endgame_options=effective_endgame_options,
            tracking_options=tracker_kwargs,
            deduplication_tol_factor=deduplication_tol_factor,
            singular_deduplication_tol=singular_deduplication_tol,
            allow_underdetermined=allow_underdetermined,
            rng=rng,
            generated_gamma=generated_gamma,
            preprocessing=preprocessing,
            equation_scaling=equation_scaling,
            square_up=default_square_up,
            start_time=start_time,
            max_paths=max_paths,
        )
        if monomial_branch_result is not None:
            return monomial_branch_result

        block_result = _solve_independent_blocks(
            original_system=system,
            working_system=working_system,
            variables=variables,
            tol=tol,
            verbose=verbose,
            store_paths=store_paths,
            use_endgame=use_endgame,
            endgame_options=effective_endgame_options,
            tracking_options=tracker_kwargs,
            deduplication_tol_factor=deduplication_tol_factor,
            singular_deduplication_tol=singular_deduplication_tol,
            allow_underdetermined=allow_underdetermined,
            rng=rng,
            generated_gamma=generated_gamma,
            preprocessing=preprocessing,
            equation_scaling=equation_scaling,
            square_up=default_square_up,
            start_time=start_time,
            max_paths=max_paths,
        )
        if block_result is not None:
            return block_result

        triangular_result = _solve_triangular_system_direct(
            original_system=system,
            working_system=working_system,
            variables=variables,
            tol=tol,
            start_time=start_time,
            tracker_kwargs=tracker_kwargs,
            generated_gamma=generated_gamma,
            preprocessing=preprocessing,
            equation_scaling=equation_scaling,
            square_up=default_square_up,
            deduplication_tol_factor=deduplication_tol_factor,
            singular_deduplication_tol=singular_deduplication_tol,
        )
        if triangular_result is not None:
            return triangular_result

        binomial_result = _solve_binomial_system_direct(
            original_system=system,
            working_system=working_system,
            variables=variables,
            tol=tol,
            start_time=start_time,
            tracker_kwargs=tracker_kwargs,
            generated_gamma=generated_gamma,
            preprocessing=preprocessing,
            equation_scaling=equation_scaling,
            square_up=default_square_up,
            deduplication_tol_factor=deduplication_tol_factor,
            singular_deduplication_tol=singular_deduplication_tol,
        )
        if binomial_result is not None:
            return binomial_result

        factorized_branch_result = _solve_factorized_branches(
            original_system=system,
            working_system=working_system,
            variables=variables,
            tol=tol,
            verbose=verbose,
            store_paths=store_paths,
            use_endgame=use_endgame,
            endgame_options=effective_endgame_options,
            tracking_options=tracker_kwargs,
            deduplication_tol_factor=deduplication_tol_factor,
            singular_deduplication_tol=singular_deduplication_tol,
            allow_underdetermined=allow_underdetermined,
            rng=rng,
            generated_gamma=generated_gamma,
            preprocessing=preprocessing,
            equation_scaling=equation_scaling,
            square_up=default_square_up,
            start_time=start_time,
            max_paths=max_paths,
        )
        if factorized_branch_result is not None:
            return factorized_branch_result

    if len(working_system.equations) < len(variables):
        raise ValueError(
            "Underdetermined polynomial systems generally have "
            "positive-dimensional solution sets; solve() returns finite "
            "zero-dimensional roots only. Add enough independent equations "
            "or use witness-set tools for positive-dimensional components."
        )

    auto_start_system = not custom_start_requested
    tracking_system = working_system
    square_up = default_square_up
    start_system_meta: Dict[str, Any]
    if auto_start_system:
        tracking_system, square_up = _square_up_system(working_system, variables, rng)

    # If no start system provided, generate a total-degree system
    if auto_start_system:
        estimated_paths = _total_degree_path_count(tracking_system.degrees())
        _check_path_limit(
            estimated_paths,
            max_paths,
            "total-degree start system",
        )
        if verbose:
            print("Generating total-degree start system...")
            
        # Pass allow_underdetermined parameter to start system generator
        start_system, start_solutions = generate_total_degree_start_system(
            tracking_system,
            variables,
            allow_underdetermined,
            random_state=rng,
            max_solutions=max_paths,
        )
        start_system_meta = {
            "source": "total_degree",
            "path_count": len(start_solutions),
            "degrees": tuple(tracking_system.degrees()),
        }
        
        if verbose:
            degrees = tracking_system.degrees()
            total_paths = _total_degree_path_count(degrees)
            print(f"Using total-degree homotopy with {total_paths} start paths ({' * '.join(map(str, degrees))})")
    else:
        start_system, start_solutions, start_system_meta = _validate_custom_start_data(
            start_system,
            start_solutions,
            tracking_system,
            variables,
            tol,
            scale_equations=scale_equations,
        )
        _check_path_limit(
            len(start_solutions),
            max_paths,
            "custom start system",
        )
    
    # Track the paths from start solutions to the target system
    if verbose:
        print(f"Tracking {len(start_solutions)} paths...")
    end_solutions, path_results = track_paths(
        start_system=start_system,
        target_system=tracking_system,
        start_solutions=start_solutions,
        variables=variables,
        tol=tol,
        verbose=verbose,
        store_paths=store_paths,
        use_endgame=use_endgame,
        endgame_options=effective_endgame_options,
        **tracker_kwargs
    )
    tracked_path_attempts = len(start_solutions)
    initial_successful_paths = _count_successful_path_results(path_results)
    tracking_retries = {
        "attempted": False,
        "accepted": False,
        "attempt_count": 1,
        "initial_successful_paths": initial_successful_paths,
        "total_attempted_paths": tracked_path_attempts,
    }
    if auto_start_system and generated_gamma and initial_successful_paths == 0:
        retry_kwargs = tracker_kwargs.copy()
        retry_kwargs["gamma"] = _random_unit_complex(rng)
        _check_path_limit(
            tracked_path_attempts + len(start_solutions),
            max_paths,
            "gamma retry",
        )
        if verbose:
            print("Retrying tracking with a new random gamma...")
        retry_end_solutions, retry_path_results = track_paths(
            start_system=start_system,
            target_system=tracking_system,
            start_solutions=start_solutions,
            variables=variables,
            tol=tol,
            verbose=verbose,
            store_paths=store_paths,
            use_endgame=use_endgame,
            endgame_options=effective_endgame_options,
            **retry_kwargs
        )
        tracked_path_attempts += len(start_solutions)
        retry_successful_paths = _count_successful_path_results(retry_path_results)
        tracking_retries = {
            "attempted": True,
            "accepted": retry_successful_paths > initial_successful_paths,
            "attempt_count": 2,
            "initial_gamma": tracker_kwargs["gamma"],
            "retry_gamma": retry_kwargs["gamma"],
            "initial_successful_paths": initial_successful_paths,
            "retry_successful_paths": retry_successful_paths,
            "total_attempted_paths": tracked_path_attempts,
        }
        if retry_successful_paths > initial_successful_paths:
            tracker_kwargs = retry_kwargs
            end_solutions = retry_end_solutions
            path_results = retry_path_results

    failed_retry_indices = [
        index
        for index, path_info in enumerate(path_results)
        if not path_info.get("success", False)
    ]
    if failed_retry_indices:
        _check_path_limit(
            tracked_path_attempts + len(failed_retry_indices),
            max_paths,
            "failed path retry",
        )
        failed_retry_kwargs = _failed_path_retry_options(tracker_kwargs)
        if verbose:
            print(
                "Retrying "
                f"{len(failed_retry_indices)} failed path(s) with smaller steps..."
            )
        retry_start_solutions = [
            start_solutions[index] for index in failed_retry_indices
        ]
        retry_end_solutions, retry_path_results = track_paths(
            start_system=start_system,
            target_system=tracking_system,
            start_solutions=retry_start_solutions,
            variables=variables,
            tol=tol,
            verbose=verbose,
            store_paths=store_paths,
            use_endgame=use_endgame,
            endgame_options=effective_endgame_options,
            **failed_retry_kwargs
        )
        tracked_path_attempts += len(failed_retry_indices)
        improved_paths = []
        for local_index, original_index in enumerate(failed_retry_indices):
            retry_info = retry_path_results[local_index]
            retry_info["retry_of_path_index"] = int(original_index)
            retry_info["retry_strategy"] = "smaller_steps"
            retry_info["path_index"] = int(original_index)
            if retry_info.get("success", False):
                end_solutions[original_index] = retry_end_solutions[local_index]
                path_results[original_index] = retry_info
                improved_paths.append(int(original_index))

        failed_retry_meta = {
            "attempted": True,
            "strategy": "smaller_steps",
            "path_indices": tuple(int(index) for index in failed_retry_indices),
            "attempted_path_count": len(failed_retry_indices),
            "successful_path_count": len(improved_paths),
            "improved_path_indices": tuple(improved_paths),
            "options": failed_retry_kwargs.copy(),
        }
        tracking_retries["attempted"] = True
        tracking_retries["accepted"] = (
            bool(tracking_retries.get("accepted", False)) or bool(improved_paths)
        )
        tracking_retries["attempt_count"] = (
            int(tracking_retries.get("attempt_count", 1)) + 1
        )
        tracking_retries["failed_path_retry"] = failed_retry_meta
        tracking_retries["total_attempted_paths"] = tracked_path_attempts
    else:
        tracking_retries["total_attempted_paths"] = tracked_path_attempts
    # Process the raw solutions
    raw_solutions = []
    
    if verbose:
        print("Processing and classifying solutions...")
    
    # Process each end solution and its path result info
    successful_paths = 0
    failed_paths = 0
    residual_rejections: List[Dict[str, Any]] = []

    # path_results should now be a list of dictionaries, one for each path
    # Each dict contains keys like 'success', 'singular', 'endgame_used', 'winding_number', 'predictions', etc.
    for i, path_result_info in enumerate(path_results):
        endpoint = end_solutions[i] # Get the endpoint corresponding to this path result

        # Check if the path was successful (either tracker reached t=0 or endgame succeeded)
        if not path_result_info.get('success', False):
             failed_paths += 1
             continue

        successful_paths += 1

        # Use the final point from the path or endgame prediction, then polish
        # it against the original system before residual filtering.
        final_point = np.array(path_result_info.get('final_point', endpoint), dtype=complex) # Assume track_paths adds 'final_point'
        final_point, residual, solution_polish = _polish_endpoint_against_system(
            system,
            final_point,
            variables,
            tol,
        )
        path_result_info["solution_polish"] = solution_polish

        # Create Solution object
        solution_dict = {var: val for var, val in zip(variables, final_point)}
        scaled_residual = _scaled_residual_norm(system, solution_dict)
        acceptance_scaled_residual = _scaled_residual_norm(
            working_system,
            solution_dict,
        )
        backward_error = _backward_error_norm(system, solution_dict)
        is_singular = bool(path_result_info.get('singular', False)) or _is_singular(
            system,
            solution_dict,
            tuple(variables),
            threshold=1e12,
            rank_tolerance=_rank_tolerance_for_tol(tol),
        )
        path_result_info["singular"] = is_singular
        winding_number = path_result_info.get('winding_number', None)

        residual_limit = 100 * tol
        if is_singular or path_result_info.get('endgame_used', False):
            residual_limit = max(residual_limit, singular_deduplication_tol)
        backward_error_limit = max(residual_limit, _backward_error_limit(tol))

        if not _solution_quality_within_limits(
            acceptance_scaled_residual,
            backward_error,
            residual_limit,
            backward_error_limit,
        ):
             if verbose:
                 print(f"Skipping solution from path {i} due to large residual ({residual:.2e})")
             residual_rejections.append({
                 "path_index": i,
                 "residual": float(residual),
                 "scaled_residual": float(scaled_residual),
                 "acceptance_scaled_residual": float(acceptance_scaled_residual),
                 "backward_error": float(backward_error),
                 "residual_limit": float(residual_limit),
                 "backward_error_limit": float(backward_error_limit),
             })
             failed_paths += 1 # Count as failed path if residual is too high
             continue

        solution = Solution(
            values=solution_dict,
            residual=residual,
            is_singular=is_singular,
            path_index=i
        )
        solution.scaled_residual = scaled_residual
        solution.backward_error = backward_error

        # Store winding number if available
        if winding_number is not None:
             solution.winding_number = winding_number

        solution.path_info = _compact_path_info(path_result_info)
        solution.path_info["accepted"] = True
        solution.path_info["solution_residual"] = float(residual)
        solution.path_info["scaled_solution_residual"] = float(scaled_residual)
        solution.path_info["backward_error"] = float(backward_error)
        solution.path_info["residual_limit"] = float(residual_limit)
        solution.path_info["backward_error_limit"] = float(backward_error_limit)

        # Store path points if available
        if store_paths and path_result_info.get('path_points'): # Access path_points from the specific path's result info
            solution.path_points = path_result_info['path_points']

        raw_solutions.append(solution)

    if verbose:
        print(f"Attempting to deduplicate {len(raw_solutions)} raw solutions...")
    unique_solutions = _deduplicate_solutions(
        raw_solutions,
        system,
        variables,
        regular_tolerance=tol * deduplication_tol_factor,
        singular_tolerance=singular_deduplication_tol,
        rank_tolerance=_rank_tolerance_for_tol(tol),
        polish_tolerance=tol,
    )

    # Create the result
    result = SolutionSet(unique_solutions, system)
    
    # Add metadata
    result._meta['total_paths'] = len(start_solutions)
    result._meta['successful_paths'] = successful_paths
    result._meta['failed_paths'] = failed_paths # This now includes paths skipped due to high residual
    result._meta['solve_time'] = time.time() - start_time
    result._meta['raw_solutions_found'] = len(raw_solutions) # Count of solutions *before* deduplication but *after* residual check
    result._meta['tracking_options'] = tracker_kwargs.copy()
    result._meta['generated_gamma'] = generated_gamma
    result._meta['preprocessing'] = preprocessing
    result._meta['equation_scaling'] = equation_scaling
    result._meta['square_up'] = square_up
    result._meta['start_system'] = start_system_meta
    result._meta['tracking_retries'] = tracking_retries
    result._meta['path_summary'] = _summarize_path_results(
        path_results,
        residual_rejections,
        len(raw_solutions),
    )
    result._meta['multiplicity_summary'] = _summarize_solution_multiplicities(
        unique_solutions
    )
    
    if verbose:
        print(f"Found {len(unique_solutions)} distinct solutions (from {len(raw_solutions)} raw solutions)")
        print(f"Solution process completed in {result._meta['solve_time']:.2f} seconds")
    
    return result


def _solve_linear_system_direct(
    *,
    original_system: PolynomialSystem,
    working_system: PolynomialSystem,
    variables: List[Variable],
    tol: float,
    start_time: float,
    tracker_kwargs: Dict[str, Any],
    generated_gamma: bool,
    preprocessing: Dict[str, Any],
    equation_scaling: Dict[str, Any],
    square_up: Dict[str, Any],
) -> SolutionSet:
    matrix, rhs, row_scales = _linear_system_matrix(working_system, variables)
    balanced_matrix, column_scales = _column_balance_linear_system(matrix)
    rhs_scale = _linear_rhs_scale(rhs)
    rank = _relative_matrix_rank(balanced_matrix)
    augmented_rank = _relative_matrix_rank(
        (
            np.column_stack([balanced_matrix, rhs / rhs_scale])
            if balanced_matrix.size
            else (rhs / rhs_scale).reshape(-1, 1)
        )
    )
    residual_limit = 100.0 * tol
    backward_error_limit = _backward_error_limit(tol)
    linear_meta: Dict[str, Any] = {
        "method": "direct_lstsq",
        "matrix_shape": tuple(matrix.shape),
        "rank": int(rank),
        "augmented_rank": int(augmented_rank),
        "residual_limit": float(residual_limit),
        "backward_error_limit": float(backward_error_limit),
        "row_scaling_method": "coefficient_max_norm",
        "row_coefficient_scales": tuple(
            _metadata_float(scale) for scale in row_scales
        ),
        "column_scaling_method": "column_max_norm",
        "column_scales": tuple(float(scale) for scale in column_scales),
        "rhs_scale": float(rhs_scale),
    }

    if augmented_rank > rank:
        linear_meta["status"] = "inconsistent"
        linear_meta["residual_norm"] = float("inf")
        return _linear_solution_set(
            [],
            original_system,
            start_time,
            tracker_kwargs,
            generated_gamma,
            preprocessing,
            equation_scaling,
            square_up,
            linear_meta,
        )

    if rank < len(variables):
        linear_meta["status"] = "positive_dimensional"
        raise ValueError(
            "Linear system is rank deficient and has infinitely many "
            "solutions; provide additional equations or use witness-set tools "
            "for positive-dimensional components"
        )

    scaled_point = solve_linear_system(balanced_matrix, rhs)
    point = scaled_point / column_scales
    values = {var: value for var, value in zip(variables, point)}
    residual = _residual_norm(original_system, values)
    scaled_residual = _scaled_residual_norm(original_system, values)
    acceptance_scaled_residual = _scaled_residual_norm(working_system, values)
    backward_error = _backward_error_norm(original_system, values)
    linear_meta["residual_norm"] = float(residual)
    linear_meta["scaled_residual_norm"] = float(scaled_residual)
    linear_meta["acceptance_scaled_residual_norm"] = float(
        acceptance_scaled_residual
    )
    linear_meta["backward_error_norm"] = float(backward_error)
    if not np.all(np.isfinite(point)) or not _solution_quality_within_limits(
        acceptance_scaled_residual,
        backward_error,
        residual_limit,
        backward_error_limit,
    ):
        linear_meta["status"] = "inconsistent"
        return _linear_solution_set(
            [],
            original_system,
            start_time,
            tracker_kwargs,
            generated_gamma,
            preprocessing,
            equation_scaling,
            square_up,
            linear_meta,
        )

    solution = Solution(
        values=values,
        residual=residual,
        is_singular=False,
        path_index=None,
    )
    solution.scaled_residual = scaled_residual
    solution.backward_error = backward_error
    solution.path_info = {
        "accepted": True,
        "method": "linear_direct",
        "solution_residual": float(residual),
        "scaled_solution_residual": float(scaled_residual),
        "backward_error": float(backward_error),
        "residual_limit": float(residual_limit),
        "backward_error_limit": float(backward_error_limit),
    }
    _attach_cluster_metadata(solution)
    linear_meta["status"] = "unique_solution"
    return _linear_solution_set(
        [solution],
        original_system,
        start_time,
        tracker_kwargs,
        generated_gamma,
        preprocessing,
        equation_scaling,
        square_up,
        linear_meta,
    )


def _linear_solution_set(
    solutions: List[Solution],
    system: PolynomialSystem,
    start_time: float,
    tracker_kwargs: Dict[str, Any],
    generated_gamma: bool,
    preprocessing: Dict[str, Any],
    equation_scaling: Dict[str, Any],
    square_up: Dict[str, Any],
    linear_meta: Dict[str, Any],
) -> SolutionSet:
    result = SolutionSet(solutions, system)
    result._meta['total_paths'] = 0
    result._meta['successful_paths'] = 0
    result._meta['failed_paths'] = 0
    result._meta['solve_time'] = time.time() - start_time
    result._meta['raw_solutions_found'] = len(solutions)
    result._meta['tracking_options'] = tracker_kwargs.copy()
    result._meta['generated_gamma'] = generated_gamma
    result._meta['preprocessing'] = preprocessing
    result._meta['equation_scaling'] = equation_scaling
    result._meta['square_up'] = square_up
    result._meta['start_system'] = {"source": "linear_direct", "path_count": 0}
    result._meta['path_summary'] = _summarize_path_results([], [], len(solutions))
    result._meta['multiplicity_summary'] = _summarize_solution_multiplicities(
        solutions
    )
    result._meta['linear_solve'] = linear_meta
    return result


def _is_linear_system(system: PolynomialSystem) -> bool:
    return all(equation.degree() <= 1 for equation in system.equations)


def _linear_system_matrix(
    system: PolynomialSystem,
    variables: List[Variable],
) -> Tuple[np.ndarray, np.ndarray, Tuple[Number, ...]]:
    variable_index = {variable: index for index, variable in enumerate(variables)}
    matrix = np.zeros((len(system.equations), len(variables)), dtype=complex)
    rhs = np.zeros(len(system.equations), dtype=complex)
    row_scales: List[Number] = []

    for row_index, equation in enumerate(system.equations):
        row_scale = _polynomial_coefficient_scale(equation)
        row_scales.append(row_scale)
        constant = 0.0 + 0.0j
        for term in equation.terms:
            coefficient = complex(
                _safe_scaled_coefficient(term.coefficient, row_scale)
            )
            if not term.variables:
                constant += coefficient
                continue
            if len(term.variables) != 1 or term.degree() != 1:
                raise ValueError("Internal error: expected a linear system")
            variable = next(iter(term.variables))
            matrix[row_index, variable_index[variable]] += coefficient
        rhs[row_index] = -constant

    return matrix, rhs, tuple(row_scales)


def _relative_matrix_rank(matrix: np.ndarray) -> int:
    if matrix.size == 0 or not np.all(np.isfinite(matrix)):
        return 0
    try:
        return int(np.linalg.matrix_rank(matrix))
    except (np.linalg.LinAlgError, ValueError, OverflowError, FloatingPointError):
        return 0


def _column_balance_linear_system(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ValueError("Internal error: expected a two-dimensional matrix")
    column_scales = np.ones(matrix.shape[1], dtype=float)
    if matrix.size == 0:
        return matrix.copy(), column_scales
    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        finite_magnitudes = np.abs(column[np.isfinite(column)])
        scale = float(np.max(finite_magnitudes)) if finite_magnitudes.size else 0.0
        if np.isfinite(scale) and scale > 0.0:
            column_scales[index] = scale
    return matrix / column_scales, column_scales


def _linear_rhs_scale(rhs: np.ndarray) -> float:
    finite_magnitudes = np.abs(rhs[np.isfinite(rhs)])
    scale = float(np.max(finite_magnitudes)) if finite_magnitudes.size else 0.0
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _solve_univariate_system_direct(
    *,
    original_system: PolynomialSystem,
    working_system: PolynomialSystem,
    variables: List[Variable],
    tol: float,
    start_time: float,
    tracker_kwargs: Dict[str, Any],
    generated_gamma: bool,
    preprocessing: Dict[str, Any],
    equation_scaling: Dict[str, Any],
    square_up: Dict[str, Any],
    deduplication_tol_factor: float,
    singular_deduplication_tol: float,
) -> Optional[SolutionSet]:
    variable = variables[0]
    solving_equation_index, solving_equation, coefficients, coefficients_lossy = (
        _select_univariate_solving_equation(working_system, variable)
    )
    common_factor_meta = {
        "attempted": False,
        "used": False,
        "method": None,
        "degree": None,
        "polynomial": None,
    }
    common_factor_data = None
    if not coefficients_lossy:
        common_factor_data = _univariate_common_factor_data(
            working_system,
            variable,
            coefficients,
        )
    if common_factor_data is not None:
        solving_equation_index = None
        solving_equation = common_factor_data["polynomial"]
        coefficients = common_factor_data["coefficients"]
        coefficients_lossy = False
        common_factor_meta = common_factor_data["metadata"]

    degree = solving_equation.degree() if coefficients_lossy else len(coefficients) - 1
    if degree <= 0:
        raise ValueError("Internal error: expected a positive-degree polynomial")

    root_solver_meta: Dict[str, Any] = {
        "method": "companion_roots",
        "coefficients_lossy": coefficients_lossy,
    }
    factorization_meta: Dict[str, Any]
    if coefficients_lossy:
        root_candidates, root_solver_meta = _lossy_univariate_root_candidates(
            solving_equation,
            variable,
            coefficients,
        )
        factorization_meta = {
            "attempted": False,
            "used": False,
            "method": None,
            "status": "skipped_lossy_coefficients",
            "original_degree": int(degree),
            "distinct_factor_degree": None,
            "total_factor_degree": None,
            "factor_count": None,
            "candidate_root_count": None,
            "factors": tuple(),
        }
    else:
        root_candidates, factorization_meta = _univariate_factor_root_candidates(
            solving_equation,
            variable,
            coefficients,
        )
    if root_candidates is None:
        roots = _safe_companion_roots(coefficients)
        if roots is None or len(roots) == 0:
            return None
        root_candidates = _root_candidates_from_roots(
            roots,
            source="companion_roots",
        )
        root_solver_meta = {
            "method": "companion_roots",
            "coefficients_lossy": coefficients_lossy,
            "candidate_root_count": len(root_candidates),
            "status": "used",
        }
    elif not root_candidates and root_solver_meta.get("root_solve_failure_count", 0):
        return None

    raw_solutions: List[Solution] = []
    residual_limit = 100.0 * tol
    base_backward_error_limit = _backward_error_limit(tol)
    candidate_filter_limit = max(1000.0 * tol, 1e-6)
    for candidate in root_candidates:
        root = candidate["root"]
        candidate_values = {variable: complex(root)}
        if len(working_system.equations) > 1:
            candidate_residual = _residual_norm(original_system, candidate_values)
            candidate_scaled_residual = _scaled_residual_norm(
                working_system,
                candidate_values,
            )
            candidate_backward_error = _backward_error_norm(
                original_system,
                candidate_values,
            )
            if not _solution_quality_within_limits(
                candidate_scaled_residual,
                candidate_backward_error,
                candidate_filter_limit,
                max(candidate_filter_limit, base_backward_error_limit),
            ):
                continue

        point = np.array([root], dtype=complex)
        point, residual, solution_polish = _polish_endpoint_against_system(
            original_system,
            point,
            variables,
            tol,
        )
        values = {variable: point[0]}
        scaled_residual = _scaled_residual_norm(original_system, values)
        acceptance_scaled_residual = _scaled_residual_norm(working_system, values)
        backward_error = _backward_error_norm(original_system, values)
        is_singular = _is_singular(
            original_system,
            values,
            tuple(variables),
            threshold=1e12,
            rank_tolerance=_rank_tolerance_for_tol(tol),
        )
        acceptance_limit = (
            max(residual_limit, singular_deduplication_tol)
            if is_singular else residual_limit
        )
        backward_error_limit = max(acceptance_limit, base_backward_error_limit)
        if not np.all(np.isfinite(point)) or not _solution_quality_within_limits(
            acceptance_scaled_residual,
            backward_error,
            acceptance_limit,
            backward_error_limit,
        ):
            continue

        solution = Solution(
            values=values,
            residual=residual,
            is_singular=is_singular,
            path_index=None,
        )
        solution.scaled_residual = scaled_residual
        solution.backward_error = backward_error
        solution.multiplicity = int(candidate["multiplicity"])
        solution.root_indices = tuple(candidate["root_indices"])
        solution.path_info = {
            "accepted": True,
            "method": "univariate_direct",
            "root_index": int(candidate["root_index"]),
            "root_indices": tuple(candidate["root_indices"]),
            "root_source": candidate["source"],
            "root_multiplicity": int(candidate["multiplicity"]),
            "solution_residual": float(residual),
            "scaled_solution_residual": float(scaled_residual),
            "acceptance_scaled_solution_residual": float(
                acceptance_scaled_residual
            ),
            "backward_error": float(backward_error),
            "residual_limit": float(acceptance_limit),
            "backward_error_limit": float(backward_error_limit),
            "solution_polish": solution_polish,
        }
        if candidate["factor_index"] is not None:
            solution.path_info["factor_index"] = int(candidate["factor_index"])
            solution.path_info["factor_root_index"] = int(
                candidate["factor_root_index"]
            )
        raw_solutions.append(solution)

    unique_solutions = _deduplicate_solutions(
        raw_solutions,
        original_system,
        variables,
        regular_tolerance=tol * deduplication_tol_factor,
        singular_tolerance=singular_deduplication_tol,
        rank_tolerance=_rank_tolerance_for_tol(tol),
        polish_tolerance=tol,
    )
    meta = {
        "method": (
            "factor_companion_roots"
            if factorization_meta["used"] else "companion_roots"
        ),
        "degree": degree,
        "equation_count": len(working_system.equations),
        "solving_equation_index": solving_equation_index,
        "equation_degrees": tuple(
            equation.degree() for equation in working_system.equations
        ),
        "raw_root_count": sum(
            len(candidate["root_indices"]) for candidate in root_candidates
        ),
        "raw_root_candidate_count": len(root_candidates),
        "accepted_root_count": sum(
            int(getattr(solution, "multiplicity", 1))
            for solution in raw_solutions
        ),
        "accepted_root_candidate_count": len(raw_solutions),
        "max_backward_error": max(
            (float(getattr(solution, "backward_error", float("inf")))
             for solution in raw_solutions),
            default=0.0,
        ),
        "backward_error_limit": float(base_backward_error_limit),
        "candidate_filter_limit": (
            candidate_filter_limit if len(working_system.equations) > 1 else None
        ),
        "leading_coefficient": _complex_metadata(coefficients[0]),
        "solving_equation": repr(solving_equation),
        "common_factor": common_factor_meta,
        "factorization": factorization_meta,
        "root_solver": root_solver_meta,
        "status": "solved",
    }
    return _univariate_solution_set(
        unique_solutions,
        original_system,
        start_time,
        tracker_kwargs,
        generated_gamma,
        preprocessing,
        equation_scaling,
        square_up,
        raw_count=sum(
            int(getattr(solution, "multiplicity", 1))
            for solution in raw_solutions
        ),
        univariate_meta=meta,
    )


def _univariate_solution_set(
    solutions: List[Solution],
    system: PolynomialSystem,
    start_time: float,
    tracker_kwargs: Dict[str, Any],
    generated_gamma: bool,
    preprocessing: Dict[str, Any],
    equation_scaling: Dict[str, Any],
    square_up: Dict[str, Any],
    *,
    raw_count: int,
    univariate_meta: Dict[str, Any],
) -> SolutionSet:
    result = SolutionSet(solutions, system)
    result._meta['total_paths'] = 0
    result._meta['successful_paths'] = 0
    result._meta['failed_paths'] = 0
    result._meta['solve_time'] = time.time() - start_time
    result._meta['raw_solutions_found'] = raw_count
    result._meta['tracking_options'] = tracker_kwargs.copy()
    result._meta['generated_gamma'] = generated_gamma
    result._meta['preprocessing'] = preprocessing
    result._meta['equation_scaling'] = equation_scaling
    result._meta['square_up'] = square_up
    result._meta['start_system'] = {"source": "univariate_direct", "path_count": 0}
    result._meta['path_summary'] = _summarize_path_results([], [], raw_count)
    result._meta['multiplicity_summary'] = _summarize_solution_multiplicities(
        solutions
    )
    result._meta['univariate_solve'] = univariate_meta
    return result


def _is_univariate_system(
    system: PolynomialSystem,
    variables: List[Variable],
) -> bool:
    if len(variables) != 1 or not system.equations:
        return False
    if len(system.equations) > 1 and any(
        equation.degree() == 1 for equation in system.equations
    ):
        return False
    variable = variables[0]
    for equation in system.equations:
        equation_variables = equation.variables()
        if any(equation_variable != variable for equation_variable in equation_variables):
            return False
    return any(equation.degree() > 0 for equation in system.equations)


def _select_univariate_solving_equation(
    system: PolynomialSystem,
    variable: Variable,
) -> Tuple[int, Polynomial, Tuple[complex, ...], bool]:
    candidates = []
    for equation_index, equation in enumerate(system.equations):
        degree = equation.degree()
        if degree <= 0:
            continue
        coefficients = _univariate_coefficients(equation, variable)
        coefficients_lossy = _univariate_coefficients_are_lossy(
            equation,
            variable,
            coefficients,
        )
        if len(coefficients) <= 1 and not coefficients_lossy:
            continue
        effective_degree = degree if coefficients_lossy else len(coefficients) - 1
        candidates.append((
            effective_degree,
            equation_index,
            equation,
            coefficients,
            coefficients_lossy,
        ))

    if not candidates:
        raise ValueError("Internal error: expected a positive-degree polynomial")

    _, equation_index, equation, coefficients, coefficients_lossy = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return equation_index, equation, coefficients, coefficients_lossy


def _univariate_coefficients_are_lossy(
    polynomial: Polynomial,
    variable: Variable,
    coefficients: Tuple[complex, ...],
) -> bool:
    if polynomial.degree() != len(coefficients) - 1:
        return True
    row_scale = _polynomial_coefficient_scale(polynomial)
    for term in polynomial.terms:
        if term.coefficient == 0:
            continue
        if term.variables and (
            len(term.variables) != 1 or variable not in term.variables
        ):
            raise ValueError("Internal error: expected a univariate polynomial")
        try:
            scaled_coefficient = _safe_scaled_coefficient(
                term.coefficient,
                row_scale,
            )
        except (OverflowError, FloatingPointError):
            return True
        if scaled_coefficient == 0:
            return True
    return False


def _safe_companion_roots(
    coefficients: Tuple[complex, ...],
) -> Optional[np.ndarray]:
    try:
        roots = np.roots(np.asarray(coefficients, dtype=complex))
    except Exception:
        return None
    try:
        return np.asarray(roots, dtype=complex)
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return None


def _root_candidates_from_roots(
    roots: np.ndarray,
    *,
    source: str,
    multiplicity: int = 1,
    root_index_offset: int = 0,
    factor_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    candidates = []
    for index, root in enumerate(roots):
        candidate_index = len(candidates)
        root_indices = tuple(
            range(
                root_index_offset + index * multiplicity,
                root_index_offset + (index + 1) * multiplicity,
            )
        )
        candidates.append({
            "root": root,
            "candidate_index": candidate_index,
            "root_index": candidate_index,
            "root_indices": root_indices,
            "multiplicity": int(multiplicity),
            "source": source,
            "factor_index": factor_index,
            "factor_root_index": index if factor_index is not None else None,
        })
    return candidates


def _lossy_univariate_root_candidates(
    polynomial: Polynomial,
    variable: Variable,
    coefficients: Tuple[complex, ...],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    roots: List[Tuple[complex, str]] = []
    base_root_count = 0
    base_root_status = "not_attempted"
    if len(coefficients) > 1:
        base_roots = _safe_companion_roots(coefficients)
        if base_roots is not None:
            base_root_count = len(base_roots)
            roots.extend((complex(root), "companion_roots") for root in base_roots)
            base_root_status = "used"
        else:
            base_root_status = "root_solve_failed"

    scaled_data = _scaled_univariate_companion_coefficients(polynomial, variable)
    scaled_root_count = 0
    scaled_root_status = "not_attempted"
    scaled_meta = {
        "status": "unavailable",
        "variable_scale": None,
        "coefficient_scale_log": None,
        "transformed_degree": None,
        "underflowed_transformed_terms": None,
    }
    if scaled_data is not None:
        scaled_coefficients, variable_scale, scaled_meta = scaled_data
        scaled_roots = _safe_companion_roots(scaled_coefficients)
        if scaled_roots is not None:
            scaled_root_count = len(scaled_roots)
            for root in scaled_roots:
                roots.append((complex(root) * variable_scale, "scaled_companion_roots"))
            scaled_root_status = "used"
        else:
            scaled_meta = {**scaled_meta, "status": "root_solve_failed"}
            scaled_root_status = "root_solve_failed"

    candidates = []
    for index, (root, source) in enumerate(roots):
        candidates.append({
            "root": root,
            "candidate_index": index,
            "root_index": index,
            "root_indices": (index,),
            "multiplicity": 1,
            "source": source,
            "factor_index": None,
            "factor_root_index": None,
        })

    metadata = {
        "method": "lossy_scaled_companion_roots",
        "coefficients_lossy": True,
        "base_root_solver_status": base_root_status,
        "scaled_root_solver_status": scaled_root_status,
        "root_solve_failure_count": sum(
            status == "root_solve_failed"
            for status in (base_root_status, scaled_root_status)
        ),
        "base_candidate_root_count": base_root_count,
        "scaled_candidate_root_count": scaled_root_count,
        "candidate_root_count": len(candidates),
        "scaled_companion": scaled_meta,
    }
    return candidates, metadata


def _scaled_univariate_companion_coefficients(
    polynomial: Polynomial,
    variable: Variable,
) -> Optional[Tuple[Tuple[complex, ...], float, Dict[str, Any]]]:
    term_data = []
    degree = 0
    for term in polynomial.terms:
        if not term.variables:
            exponent = 0
        else:
            if len(term.variables) != 1 or variable not in term.variables:
                raise ValueError("Internal error: expected a univariate polynomial")
            exponent = int(term.variables[variable])
        log_abs, phase = _coefficient_log_abs_and_phase(term.coefficient)
        if log_abs == float("-inf"):
            continue
        if not np.isfinite(log_abs):
            return None
        term_data.append((exponent, log_abs, phase))
        degree = max(degree, exponent)

    if degree <= 0 or not term_data:
        return None

    variable_log_scale = _best_univariate_variable_log_scale(term_data)
    transformed_logs = [
        log_abs + exponent * variable_log_scale
        for exponent, log_abs, _phase in term_data
    ]
    coefficient_scale_log = max(transformed_logs)
    if not np.isfinite(coefficient_scale_log):
        return None

    coefficients = [0.0 + 0.0j for _ in range(degree + 1)]
    log_min = math.log(np.nextafter(0.0, 1.0))
    underflowed_terms = 0
    for exponent, log_abs, phase in term_data:
        transformed_log = log_abs + exponent * variable_log_scale
        normalized_log = transformed_log - coefficient_scale_log
        if normalized_log < log_min:
            underflowed_terms += 1
            continue
        magnitude = math.exp(normalized_log)
        coefficients[degree - exponent] += complex(
            magnitude * math.cos(phase),
            magnitude * math.sin(phase),
        )

    while len(coefficients) > 1 and coefficients[0] == 0:
        coefficients.pop(0)
    if len(coefficients) <= 1 or not np.all(np.isfinite(coefficients)):
        return None

    try:
        variable_scale = math.exp(variable_log_scale)
    except OverflowError:
        return None
    if not np.isfinite(variable_scale) or variable_scale == 0.0:
        return None

    metadata = {
        "status": "used",
        "variable_scale": float(variable_scale),
        "variable_log_scale": float(variable_log_scale),
        "coefficient_scale_log": float(coefficient_scale_log),
        "transformed_degree": len(coefficients) - 1,
        "underflowed_transformed_terms": underflowed_terms,
    }
    return tuple(coefficients), variable_scale, metadata


def _best_univariate_variable_log_scale(
    term_data: List[Tuple[int, float, float]],
) -> float:
    log_scale_limit = math.log(np.finfo(float).max)
    candidates = [0.0]
    for left_index, (left_exp, left_log_abs, _left_phase) in enumerate(term_data):
        for right_exp, right_log_abs, _right_phase in term_data[left_index + 1:]:
            if left_exp == right_exp:
                continue
            candidate = (right_log_abs - left_log_abs) / (left_exp - right_exp)
            candidates.append(float(np.clip(
                candidate,
                -log_scale_limit,
                log_scale_limit,
            )))

    def spread(log_scale: float) -> float:
        values = [
            log_abs + exponent * log_scale
            for exponent, log_abs, _phase in term_data
        ]
        return max(values) - min(values)

    return min(candidates, key=lambda item: (spread(item), abs(item)))


def _coefficient_log_abs_and_phase(value: Any) -> Tuple[float, float]:
    if value == 0:
        return float("-inf"), 0.0
    if isinstance(value, Real):
        phase = math.pi if value < 0 else 0.0
        try:
            return math.log(abs(value)), phase
        except OverflowError:
            if isinstance(value, Integral):
                return math.log(abs(int(value))), phase
            return float("inf"), phase
    numeric_value = complex(value)
    if numeric_value == 0:
        return float("-inf"), 0.0
    if not np.isfinite(numeric_value.real) or not np.isfinite(numeric_value.imag):
        return float("inf"), 0.0
    return math.log(abs(numeric_value)), math.atan2(
        numeric_value.imag,
        numeric_value.real,
    )


def _univariate_common_factor_data(
    system: PolynomialSystem,
    variable: Variable,
    selected_coefficients: Tuple[complex, ...],
) -> Optional[Dict[str, Any]]:
    if len(system.equations) <= 1:
        return None

    try:
        import sympy as sp
    except Exception:
        return None

    symbols = (sp.Symbol(variable.name),)
    gcd_poly = None
    source_indices = []
    for equation_index, equation in enumerate(system.equations):
        if equation.degree() <= 0:
            continue
        _, expr = _polynomial_to_sympy_expr(equation, [variable], sp)
        if expr is None:
            return None
        try:
            current = sp.Poly(expr, *symbols)
        except Exception:
            return None
        gcd_poly = current if gcd_poly is None else sp.gcd(gcd_poly, current)
        source_indices.append(equation_index)

    if gcd_poly is None:
        return None

    try:
        gcd_expr = gcd_poly.as_expr()
        gcd_degree = int(sp.Poly(gcd_expr, *symbols).degree())
    except Exception:
        return None
    selected_degree = len(selected_coefficients) - 1
    if gcd_degree <= 0 or gcd_degree >= selected_degree:
        return None

    common_polynomial = _sympy_expr_to_polynomial(
        gcd_expr,
        [variable],
        symbols,
        sp,
    )
    if common_polynomial is None or common_polynomial.degree() != gcd_degree:
        return None
    try:
        common_coefficients = _univariate_coefficients(common_polynomial, variable)
    except ValueError:
        return None
    if len(common_coefficients) <= 1:
        return None

    return {
        "polynomial": common_polynomial,
        "coefficients": common_coefficients,
        "metadata": {
            "attempted": True,
            "used": True,
            "method": "sympy_gcd",
            "degree": gcd_degree,
            "source_equation_indices": tuple(source_indices),
            "polynomial": repr(common_polynomial),
            "selected_equation_degree": selected_degree,
        },
    }


def _univariate_factor_root_candidates(
    polynomial: Polynomial,
    variable: Variable,
    coefficients: Tuple[complex, ...],
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    original_degree = len(coefficients) - 1
    metadata: Dict[str, Any] = {
        "attempted": False,
        "used": False,
        "method": None,
        "status": "not_attempted",
        "original_degree": int(original_degree),
        "distinct_factor_degree": None,
        "total_factor_degree": None,
        "factor_count": None,
        "candidate_root_count": None,
        "factors": tuple(),
    }
    if original_degree <= 1:
        metadata["status"] = "linear_or_constant"
        return None, metadata

    try:
        import sympy as sp
    except Exception:
        metadata["status"] = "sympy_unavailable"
        return None, metadata

    metadata["attempted"] = True
    metadata["method"] = "sympy_factor_list"

    symbols, expr = _polynomial_to_sympy_expr(polynomial, [variable], sp)
    if expr is None:
        metadata["status"] = "unsupported_polynomial"
        return None, metadata

    try:
        _, raw_factors = sp.factor_list(expr, *symbols)
    except Exception:
        metadata["status"] = "factorization_failed"
        return None, metadata

    factors = []
    for factor_expr, multiplicity in raw_factors:
        factor_multiplicity = int(multiplicity)
        if factor_multiplicity <= 0:
            metadata["status"] = "invalid_factor_multiplicity"
            return None, metadata

        factor_polynomial = _sympy_expr_to_polynomial(
            factor_expr,
            [variable],
            symbols,
            sp,
        )
        if factor_polynomial is None:
            metadata["status"] = "unsupported_factor"
            return None, metadata
        try:
            factor_coefficients = _univariate_coefficients(
                factor_polynomial,
                variable,
            )
        except ValueError:
            metadata["status"] = "unsupported_factor"
            return None, metadata

        factor_degree = len(factor_coefficients) - 1
        if factor_degree <= 0:
            continue

        try:
            factor_expression = str(sp.factor(factor_expr))
        except Exception:
            factor_expression = str(factor_expr)
        factors.append({
            "expression": factor_expression,
            "degree": int(factor_degree),
            "multiplicity": factor_multiplicity,
            "coefficients": factor_coefficients,
        })

    if not factors:
        metadata["status"] = "no_nonconstant_factors"
        return None, metadata

    distinct_factor_degree = sum(factor["degree"] for factor in factors)
    total_factor_degree = sum(
        factor["degree"] * factor["multiplicity"] for factor in factors
    )
    metadata.update({
        "distinct_factor_degree": int(distinct_factor_degree),
        "total_factor_degree": int(total_factor_degree),
        "factor_count": len(factors),
        "factors": tuple(
            {
                "factor": factor["expression"],
                "degree": int(factor["degree"]),
                "multiplicity": int(factor["multiplicity"]),
            }
            for factor in factors
        ),
    })

    if total_factor_degree != original_degree:
        metadata["status"] = "degree_mismatch"
        return None, metadata
    if distinct_factor_degree >= original_degree:
        metadata["status"] = "square_free"
        return None, metadata

    root_candidates = []
    root_index_offset = 0
    for factor_index, factor in enumerate(factors):
        factor_roots = _safe_companion_roots(factor["coefficients"])
        if factor_roots is None or len(factor_roots) == 0:
            metadata["status"] = "factor_root_solve_failed"
            return None, metadata
        multiplicity = int(factor["multiplicity"])
        for factor_root_index, root in enumerate(factor_roots):
            candidate_index = len(root_candidates)
            root_indices = tuple(
                range(root_index_offset, root_index_offset + multiplicity)
            )
            root_index_offset += multiplicity
            root_candidates.append({
                "root": root,
                "candidate_index": candidate_index,
                "root_index": candidate_index,
                "root_indices": root_indices,
                "multiplicity": multiplicity,
                "source": "sympy_factor_roots",
                "factor_index": factor_index,
                "factor_root_index": factor_root_index,
            })

    metadata.update({
        "used": True,
        "status": "used",
        "candidate_root_count": len(root_candidates),
    })
    return root_candidates, metadata


def _solve_independent_blocks(
    *,
    original_system: PolynomialSystem,
    working_system: PolynomialSystem,
    variables: List[Variable],
    tol: float,
    verbose: bool,
    store_paths: bool,
    use_endgame: bool,
    endgame_options: Optional[Dict[str, Any]],
    tracking_options: Dict[str, Any],
    deduplication_tol_factor: float,
    singular_deduplication_tol: float,
    allow_underdetermined: bool,
    rng: Any,
    generated_gamma: bool,
    preprocessing: Dict[str, Any],
    equation_scaling: Dict[str, Any],
    square_up: Dict[str, Any],
    start_time: float,
    max_paths: Optional[int],
) -> Optional[SolutionSet]:
    blocks = _independent_variable_blocks(working_system, variables)
    if blocks is None:
        return None

    block_results = []
    used_paths = 0
    for block_index, block in enumerate(blocks):
        block_system = PolynomialSystem([
            working_system.equations[equation_index]
            for equation_index in block["equation_indices"]
        ])
        block_result = solve(
            block_system,
            variables=block["variables"],
            tol=tol,
            verbose=False,
            store_paths=store_paths,
            use_endgame=use_endgame,
            endgame_options=endgame_options,
            tracking_options=tracking_options.copy(),
            deduplication_tol_factor=deduplication_tol_factor,
            singular_deduplication_tol=singular_deduplication_tol,
            allow_underdetermined=allow_underdetermined,
            scale_equations=False,
            max_paths=_remaining_path_limit(max_paths, used_paths),
            _allow_zero_max_paths=True,
            random_state=rng,
        )
        block_results.append((block_index, block, block_result))
        used_paths += int(block_result._meta.get("total_paths", 0))
        _check_path_limit(
            used_paths,
            max_paths,
            "independent block decomposition",
        )

    total_paths = sum(
        int(result._meta.get("total_paths", 0))
        for _, _, result in block_results
    )
    successful_paths = sum(
        int(result._meta.get("successful_paths", 0))
        for _, _, result in block_results
    )
    failed_paths = sum(
        int(result._meta.get("failed_paths", 0))
        for _, _, result in block_results
    )
    path_offsets = []
    next_path_offset = 0
    for _, _, result in block_results:
        path_offsets.append(next_path_offset)
        next_path_offset += int(result._meta.get("total_paths", 0))

    raw_solutions: List[Solution] = []
    block_solution_lists = [
        list(enumerate(result))
        for _, _, result in block_results
    ]
    if all(block_solution_lists):
        residual_limit = 100.0 * tol
        base_backward_error_limit = _backward_error_limit(tol)
        for combination_index, combination in enumerate(product(*block_solution_lists)):
            values: Dict[Variable, complex] = {}
            multiplicity = 1
            path_indices = []
            block_solution_meta = []
            for block_position, (solution_index, block_solution) in enumerate(combination):
                block_index, block, _ = block_results[block_position]
                values.update(block_solution.values)
                multiplicity *= int(getattr(block_solution, "multiplicity", 1))
                path_indices.extend(
                    path_offsets[block_position] + path_index
                    for path_index in getattr(block_solution, "path_indices", ())
                    if path_index is not None
                )
                block_solution_meta.append({
                    "block_index": int(block_index),
                    "solution_index": int(solution_index),
                    "variables": tuple(variable.name for variable in block["variables"]),
                    "multiplicity": int(getattr(block_solution, "multiplicity", 1)),
                    "residual": float(getattr(block_solution, "residual", float("inf"))),
                })

            ordered_values = _solution_values_from_complete_coordinates(
                values,
                variables,
                label="combined solution",
            )
            residual = _residual_norm(original_system, ordered_values)
            scaled_residual = _scaled_residual_norm(original_system, ordered_values)
            acceptance_scaled_residual = _scaled_residual_norm(
                working_system,
                ordered_values,
            )
            backward_error = _backward_error_norm(original_system, ordered_values)
            is_singular = _is_singular(
                original_system,
                ordered_values,
                tuple(variables),
                threshold=1e12,
                rank_tolerance=_rank_tolerance_for_tol(tol),
            )
            acceptance_limit = (
                max(residual_limit, singular_deduplication_tol)
                if is_singular else residual_limit
            )
            backward_error_limit = max(acceptance_limit, base_backward_error_limit)
            if not _solution_quality_within_limits(
                acceptance_scaled_residual,
                backward_error,
                acceptance_limit,
                backward_error_limit,
            ):
                continue

            solution = Solution(
                values=ordered_values,
                residual=residual,
                is_singular=is_singular,
                path_index=path_indices[0] if path_indices else None,
            )
            solution.scaled_residual = scaled_residual
            solution.backward_error = backward_error
            solution.multiplicity = multiplicity
            solution.path_indices = tuple(sorted(set(path_indices)))
            solution.path_info = {
                "accepted": True,
                "method": "independent_blocks",
                "combination_index": int(combination_index),
                "solution_residual": float(residual),
                "scaled_solution_residual": float(scaled_residual),
                "acceptance_scaled_solution_residual": float(
                    acceptance_scaled_residual
                ),
                "backward_error": float(backward_error),
                "residual_limit": float(acceptance_limit),
                "backward_error_limit": float(backward_error_limit),
                "block_solutions": tuple(block_solution_meta),
            }
            _attach_cluster_metadata(solution)
            raw_solutions.append(solution)

    unique_solutions = _deduplicate_solutions(
        raw_solutions,
        original_system,
        variables,
        regular_tolerance=tol * deduplication_tol_factor,
        singular_tolerance=singular_deduplication_tol,
        rank_tolerance=_rank_tolerance_for_tol(tol),
        polish_tolerance=tol,
    )
    result = SolutionSet(unique_solutions, original_system)
    result._meta['total_paths'] = total_paths
    result._meta['successful_paths'] = successful_paths
    result._meta['failed_paths'] = failed_paths
    result._meta['solve_time'] = time.time() - start_time
    result._meta['raw_solutions_found'] = len(raw_solutions)
    result._meta['tracking_options'] = tracking_options.copy()
    result._meta['generated_gamma'] = generated_gamma
    result._meta['preprocessing'] = preprocessing
    result._meta['equation_scaling'] = equation_scaling
    result._meta['square_up'] = square_up
    result._meta['start_system'] = {
        "source": "independent_blocks",
        "path_count": total_paths,
        "block_count": len(blocks),
    }
    result._meta['path_summary'] = _summarize_path_results([], [], len(raw_solutions))
    result._meta['multiplicity_summary'] = _summarize_solution_multiplicities(
        unique_solutions
    )
    result._meta['independent_blocks'] = {
        "method": "variable_equation_incidence",
        "status": "solved" if raw_solutions else "no_solutions",
        "block_count": len(blocks),
        "raw_combination_count": _product_int(
            len(solution_list) for solution_list in block_solution_lists
        ),
        "accepted_combination_count": len(raw_solutions),
        "distinct_solution_count": len(unique_solutions),
        "blocks": tuple(
            _block_result_metadata(block_index, block, block_result)
            for block_index, block, block_result in block_results
        ),
    }
    return result


def _independent_variable_blocks(
    system: PolynomialSystem,
    variables: List[Variable],
) -> Optional[List[Dict[str, Any]]]:
    if len(variables) <= 1 or len(system.equations) <= 1:
        return None

    variable_index = {variable: index for index, variable in enumerate(variables)}
    parent = list(range(len(variables)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    equation_variables = []
    used_variables = set()
    for equation in system.equations:
        eq_vars = [
            variable for variable in variables
            if variable in equation.variables()
        ]
        if not eq_vars:
            return None
        used_variables.update(eq_vars)
        first_index = variable_index[eq_vars[0]]
        for variable in eq_vars[1:]:
            union(first_index, variable_index[variable])
        equation_variables.append(eq_vars)

    if used_variables != set(variables):
        return None

    root_to_variables: Dict[int, List[Variable]] = {}
    for variable in variables:
        root_to_variables.setdefault(find(variable_index[variable]), []).append(variable)
    if len(root_to_variables) <= 1:
        return None

    roots = sorted(
        root_to_variables,
        key=lambda root: min(variable_index[variable] for variable in root_to_variables[root]),
    )
    root_to_block = {
        root: {"variables": root_to_variables[root], "equation_indices": []}
        for root in roots
    }
    for equation_index, eq_vars in enumerate(equation_variables):
        root = find(variable_index[eq_vars[0]])
        root_to_block[root]["equation_indices"].append(equation_index)

    blocks = [root_to_block[root] for root in roots]
    if any(not block["equation_indices"] for block in blocks):
        return None
    return blocks


def _block_result_metadata(
    block_index: int,
    block: Dict[str, Any],
    block_result: SolutionSet,
) -> Dict[str, Any]:
    start_system_meta = block_result._meta.get("start_system", {})
    return {
        "block_index": int(block_index),
        "variables": tuple(variable.name for variable in block["variables"]),
        "equation_indices": tuple(int(index) for index in block["equation_indices"]),
        "equation_count": len(block["equation_indices"]),
        "solution_count": len(block_result),
        "raw_solutions_found": int(block_result._meta.get("raw_solutions_found", 0)),
        "total_paths": int(block_result._meta.get("total_paths", 0)),
        "start_source": start_system_meta.get("source"),
    }


def _product_int(values: Any) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _solve_triangular_system_direct(
    *,
    original_system: PolynomialSystem,
    working_system: PolynomialSystem,
    variables: List[Variable],
    tol: float,
    start_time: float,
    tracker_kwargs: Dict[str, Any],
    generated_gamma: bool,
    preprocessing: Dict[str, Any],
    equation_scaling: Dict[str, Any],
    square_up: Dict[str, Any],
    deduplication_tol_factor: float,
    singular_deduplication_tol: float,
) -> Optional[SolutionSet]:
    if len(working_system.equations) != len(variables) or not variables:
        return None

    branch_data = _enumerate_triangular_branches(
        working_system,
        variables,
        tol,
    )
    if branch_data is None:
        return None

    branches, triangular_meta = branch_data
    raw_solutions: List[Solution] = []
    residual_limit = 100.0 * tol
    base_backward_error_limit = _backward_error_limit(tol)
    for branch_index, branch in enumerate(branches):
        values = _solution_values_from_complete_coordinates(
            branch["assignments"],
            variables,
            label="triangular branch",
        )
        residual = _residual_norm(original_system, values)
        scaled_residual = _scaled_residual_norm(original_system, values)
        acceptance_scaled_residual = _scaled_residual_norm(working_system, values)
        backward_error = _backward_error_norm(original_system, values)
        is_singular = _is_singular(
            original_system,
            values,
            tuple(variables),
            threshold=1e12,
            rank_tolerance=_rank_tolerance_for_tol(tol),
        )
        acceptance_limit = (
            max(residual_limit, singular_deduplication_tol)
            if is_singular else residual_limit
        )
        backward_error_limit = max(acceptance_limit, base_backward_error_limit)
        if not _solution_quality_within_limits(
            acceptance_scaled_residual,
            backward_error,
            acceptance_limit,
            backward_error_limit,
        ):
            continue

        solution = Solution(
            values=values,
            residual=residual,
            is_singular=is_singular,
            path_index=None,
        )
        solution.scaled_residual = scaled_residual
        solution.backward_error = backward_error
        solution.multiplicity = int(branch.get("multiplicity", 1))
        solution.root_indices = tuple(branch["root_indices"])
        solution.path_info = {
            "accepted": True,
            "method": "triangular_direct",
            "branch_index": branch_index,
            "branch_multiplicity": int(branch.get("multiplicity", 1)),
            "solution_residual": float(residual),
            "scaled_solution_residual": float(scaled_residual),
            "acceptance_scaled_solution_residual": float(
                acceptance_scaled_residual
            ),
            "backward_error": float(backward_error),
            "residual_limit": float(acceptance_limit),
            "backward_error_limit": float(backward_error_limit),
            "triangular_steps": tuple(branch["steps"]),
        }
        raw_solutions.append(solution)

    unique_solutions = _deduplicate_solutions(
        raw_solutions,
        original_system,
        variables,
        regular_tolerance=tol * deduplication_tol_factor,
        singular_tolerance=singular_deduplication_tol,
        rank_tolerance=_rank_tolerance_for_tol(tol),
        polish_tolerance=tol,
    )
    triangular_meta = {
        **triangular_meta,
        "accepted_branch_count": sum(
            int(getattr(solution, "multiplicity", 1))
            for solution in raw_solutions
        ),
        "accepted_branch_candidate_count": len(raw_solutions),
        "distinct_solution_count": len(unique_solutions),
        "status": "solved",
    }
    return _triangular_solution_set(
        unique_solutions,
        original_system,
        start_time,
        tracker_kwargs,
        generated_gamma,
        preprocessing,
        equation_scaling,
        square_up,
        raw_count=sum(
            int(getattr(solution, "multiplicity", 1))
            for solution in raw_solutions
        ),
        triangular_meta=triangular_meta,
    )


def _triangular_solution_set(
    solutions: List[Solution],
    system: PolynomialSystem,
    start_time: float,
    tracker_kwargs: Dict[str, Any],
    generated_gamma: bool,
    preprocessing: Dict[str, Any],
    equation_scaling: Dict[str, Any],
    square_up: Dict[str, Any],
    *,
    raw_count: int,
    triangular_meta: Dict[str, Any],
) -> SolutionSet:
    result = SolutionSet(solutions, system)
    result._meta['total_paths'] = 0
    result._meta['successful_paths'] = 0
    result._meta['failed_paths'] = 0
    result._meta['solve_time'] = time.time() - start_time
    result._meta['raw_solutions_found'] = raw_count
    result._meta['tracking_options'] = tracker_kwargs.copy()
    result._meta['generated_gamma'] = generated_gamma
    result._meta['preprocessing'] = preprocessing
    result._meta['equation_scaling'] = equation_scaling
    result._meta['square_up'] = square_up
    result._meta['start_system'] = {"source": "triangular_direct", "path_count": 0}
    result._meta['path_summary'] = _summarize_path_results([], [], raw_count)
    result._meta['multiplicity_summary'] = _summarize_solution_multiplicities(
        solutions
    )
    result._meta['triangular_solve'] = triangular_meta
    return result


def _solve_binomial_system_direct(
    *,
    original_system: PolynomialSystem,
    working_system: PolynomialSystem,
    variables: List[Variable],
    tol: float,
    start_time: float,
    tracker_kwargs: Dict[str, Any],
    generated_gamma: bool,
    preprocessing: Dict[str, Any],
    equation_scaling: Dict[str, Any],
    square_up: Dict[str, Any],
    deduplication_tol_factor: float,
    singular_deduplication_tol: float,
) -> Optional[SolutionSet]:
    binomial_data = _binomial_torus_data(working_system, variables)
    if binomial_data is None:
        return None

    exponent_matrix, rhs_values, equation_metadata = binomial_data
    determinant = _integer_determinant(exponent_matrix)
    if determinant == 0:
        inconsistency = _rank_deficient_binomial_inconsistency(
            exponent_matrix,
            rhs_values,
            equation_metadata,
            tol,
        )
        if inconsistency is not None:
            binomial_meta = {
                "method": "log_lift_consistency",
                "status": "inconsistent",
                "determinant": determinant,
                "torus_solution_count": 0,
                "candidate_count": 0,
                "accepted_candidate_count": 0,
                "distinct_solution_count": 0,
                "equations": equation_metadata,
                "inconsistent_constraints": inconsistency,
            }
            return _binomial_solution_set(
                [],
                original_system,
                start_time,
                tracker_kwargs,
                generated_gamma,
                preprocessing,
                equation_scaling,
                square_up,
                raw_count=0,
                binomial_meta=binomial_meta,
            )
        raise ValueError(
            "Binomial system has rank-deficient exponent constraints and "
            "infinitely many torus solutions; provide additional independent "
            "equations or use witness-set tools for positive-dimensional "
            "components"
        )

    solution_count = abs(determinant)
    max_candidate_count = 200000
    candidates, enumeration_meta = _enumerate_binomial_torus_candidates(
        exponent_matrix,
        rhs_values,
        solution_count,
        tol,
        max_candidate_count=max_candidate_count,
    )
    if len(candidates) < solution_count:
        return None

    raw_solutions: List[Solution] = []
    residual_limit = 100.0 * tol
    base_backward_error_limit = _backward_error_limit(tol)
    for candidate_index, candidate in enumerate(candidates):
        point, residual, solution_polish = _polish_endpoint_against_system(
            original_system,
            candidate,
            variables,
            tol,
        )
        values = {
            variable: value
            for variable, value in zip(variables, point)
        }
        scaled_residual = _scaled_residual_norm(original_system, values)
        acceptance_scaled_residual = _scaled_residual_norm(working_system, values)
        backward_error = _backward_error_norm(original_system, values)
        is_singular = _is_singular(
            original_system,
            values,
            tuple(variables),
            threshold=1e12,
            rank_tolerance=_rank_tolerance_for_tol(tol),
        )
        acceptance_limit = (
            max(residual_limit, singular_deduplication_tol)
            if is_singular else residual_limit
        )
        backward_error_limit = max(acceptance_limit, base_backward_error_limit)
        if not np.all(np.isfinite(point)) or not _solution_quality_within_limits(
            acceptance_scaled_residual,
            backward_error,
            acceptance_limit,
            backward_error_limit,
        ):
            continue

        solution = Solution(
            values=values,
            residual=residual,
            is_singular=is_singular,
            path_index=None,
        )
        solution.scaled_residual = scaled_residual
        solution.backward_error = backward_error
        solution.root_indices = (candidate_index,)
        solution.path_info = {
            "accepted": True,
            "method": "binomial_direct",
            "candidate_index": candidate_index,
            "solution_residual": float(residual),
            "scaled_solution_residual": float(scaled_residual),
            "acceptance_scaled_solution_residual": float(
                acceptance_scaled_residual
            ),
            "backward_error": float(backward_error),
            "residual_limit": float(acceptance_limit),
            "backward_error_limit": float(backward_error_limit),
            "solution_polish": solution_polish,
        }
        raw_solutions.append(solution)

    if len(raw_solutions) < solution_count:
        return None

    unique_solutions = _deduplicate_solutions(
        raw_solutions,
        original_system,
        variables,
        regular_tolerance=tol * deduplication_tol_factor,
        singular_tolerance=singular_deduplication_tol,
        rank_tolerance=_rank_tolerance_for_tol(tol),
        polish_tolerance=tol,
    )
    if len(unique_solutions) < solution_count:
        return None

    binomial_meta = {
        "method": "log_lift_enumeration",
        "status": "solved",
        "determinant": determinant,
        "torus_solution_count": solution_count,
        "enumerated_lift_count": enumeration_meta["enumerated_lift_count"],
        "lift_enumeration_method": enumeration_meta["method"],
        "lift_search_space": enumeration_meta["search_space"],
        "lift_representative_count": enumeration_meta["representative_count"],
        "hermite_diagonal": enumeration_meta["hermite_diagonal"],
        "candidate_count": len(candidates),
        "accepted_candidate_count": len(raw_solutions),
        "distinct_solution_count": len(unique_solutions),
        "equations": equation_metadata,
    }
    return _binomial_solution_set(
        unique_solutions,
        original_system,
        start_time,
        tracker_kwargs,
        generated_gamma,
        preprocessing,
        equation_scaling,
        square_up,
        raw_count=len(raw_solutions),
        binomial_meta=binomial_meta,
    )


def _binomial_solution_set(
    solutions: List[Solution],
    system: PolynomialSystem,
    start_time: float,
    tracker_kwargs: Dict[str, Any],
    generated_gamma: bool,
    preprocessing: Dict[str, Any],
    equation_scaling: Dict[str, Any],
    square_up: Dict[str, Any],
    *,
    raw_count: int,
    binomial_meta: Dict[str, Any],
) -> SolutionSet:
    result = SolutionSet(solutions, system)
    result._meta['total_paths'] = 0
    result._meta['successful_paths'] = 0
    result._meta['failed_paths'] = 0
    result._meta['solve_time'] = time.time() - start_time
    result._meta['raw_solutions_found'] = raw_count
    result._meta['tracking_options'] = tracker_kwargs.copy()
    result._meta['generated_gamma'] = generated_gamma
    result._meta['preprocessing'] = preprocessing
    result._meta['equation_scaling'] = equation_scaling
    result._meta['square_up'] = square_up
    result._meta['start_system'] = {"source": "binomial_direct", "path_count": 0}
    result._meta['path_summary'] = _summarize_path_results([], [], raw_count)
    result._meta['multiplicity_summary'] = _summarize_solution_multiplicities(
        solutions
    )
    result._meta['binomial_solve'] = binomial_meta
    return result


def _binomial_torus_data(
    system: PolynomialSystem,
    variables: List[Variable],
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[Dict[str, Any], ...]]]:
    if len(system.equations) != len(variables) or not variables:
        return None

    exponent_rows = []
    rhs_values = []
    equation_metadata = []
    coefficient_tol = 1e-14

    for equation_index, equation in enumerate(system.equations):
        row_scale = _polynomial_coefficient_scale(equation)
        if _scaling_would_drop_nonzero_coefficient(equation, row_scale):
            return None
        terms = [
            (
                term,
                complex(_safe_scaled_coefficient(term.coefficient, row_scale)),
            )
            for term in equation.terms
        ]
        terms = [
            (term, coefficient) for term, coefficient in terms
            if abs(coefficient) > coefficient_tol
        ]
        if len(terms) != 2:
            return None

        (first, first_coefficient), (second, second_coefficient) = terms
        first_exponents = _monomial_exponent_vector(first, variables)
        second_exponents = _monomial_exponent_vector(second, variables)
        if np.any(np.minimum(first_exponents, second_exponents) > 0):
            return None

        exponent_difference = first_exponents - second_exponents
        if not np.any(exponent_difference):
            return None

        if abs(first_coefficient) <= coefficient_tol:
            return None
        rhs = -second_coefficient / first_coefficient
        if abs(rhs) <= coefficient_tol or not np.isfinite(rhs):
            return None

        exponent_rows.append(exponent_difference)
        rhs_values.append(rhs)
        equation_metadata.append({
            "equation_index": equation_index,
            "exponent_difference": tuple(int(value) for value in exponent_difference),
            "rhs": _complex_metadata(rhs),
            "first_exponents": tuple(int(value) for value in first_exponents),
            "second_exponents": tuple(int(value) for value in second_exponents),
        })

    return (
        np.asarray(exponent_rows, dtype=int),
        np.asarray(rhs_values, dtype=complex),
        tuple(equation_metadata),
    )


def _rank_deficient_binomial_inconsistency(
    exponent_matrix: np.ndarray,
    rhs_values: np.ndarray,
    equation_metadata: Tuple[Dict[str, Any], ...],
    tol: float,
) -> Optional[Tuple[Dict[str, Any], ...]]:
    grouped: Dict[Tuple[int, ...], List[Tuple[int, int, complex]]] = {}
    for row_index, row in enumerate(np.asarray(exponent_matrix, dtype=object)):
        primitive, multiplier = _primitive_exponent_direction(row)
        if primitive is None:
            continue
        grouped.setdefault(primitive, []).append(
            (row_index, multiplier, complex(rhs_values[row_index]))
        )

    consistency_tol = max(1000.0 * tol, 1e-8)
    inconsistencies: List[Dict[str, Any]] = []
    for primitive, entries in grouped.items():
        for left_pos in range(len(entries)):
            left_index, left_multiplier, left_rhs = entries[left_pos]
            for right_index, right_multiplier, right_rhs in entries[left_pos + 1:]:
                relation_gcd = gcd(abs(left_multiplier), abs(right_multiplier))
                if relation_gcd == 0:
                    continue
                left_power = right_multiplier // relation_gcd
                right_power = left_multiplier // relation_gcd
                left_value = _complex_integer_power(left_rhs, left_power)
                right_value = _complex_integer_power(right_rhs, right_power)
                if left_value is None or right_value is None:
                    continue
                scale = max(1.0, abs(left_value), abs(right_value))
                if abs(left_value - right_value) <= consistency_tol * scale:
                    continue
                inconsistencies.append({
                    "equation_indices": (
                        int(equation_metadata[left_index]["equation_index"]),
                        int(equation_metadata[right_index]["equation_index"]),
                    ),
                    "primitive_exponent": primitive,
                    "multipliers": (int(left_multiplier), int(right_multiplier)),
                    "left_relation_power": int(left_power),
                    "right_relation_power": int(right_power),
                    "left_value": _complex_metadata(left_value),
                    "right_value": _complex_metadata(right_value),
                    "tolerance": float(consistency_tol),
                })

    if inconsistencies:
        return tuple(inconsistencies)

    for relation in _integer_left_nullspace(exponent_matrix):
        relation_value = _binomial_relation_value(rhs_values, relation)
        if relation_value is None:
            continue
        if abs(relation_value - 1.0) <= consistency_tol * max(
            1.0,
            abs(relation_value),
        ):
            continue
        equation_indices = tuple(
            int(equation_metadata[index]["equation_index"])
            for index, coefficient in enumerate(relation)
            if coefficient != 0
        )
        inconsistencies.append({
            "equation_indices": equation_indices,
            "relation": tuple(int(coefficient) for coefficient in relation),
            "relation_value": _complex_metadata(relation_value),
            "expected_value": _complex_metadata(1.0 + 0.0j),
            "tolerance": float(consistency_tol),
        })

    return tuple(inconsistencies) if inconsistencies else None


def _integer_left_nullspace(exponent_matrix: np.ndarray) -> Tuple[Tuple[int, ...], ...]:
    transposed = np.asarray(exponent_matrix, dtype=object).T.tolist()
    if not transposed:
        return ()

    row_count = len(transposed)
    column_count = len(transposed[0]) if transposed else 0
    matrix = [
        [Fraction(int(value), 1) for value in row]
        for row in transposed
    ]
    pivot_columns: List[int] = []
    pivot_row = 0

    for column in range(column_count):
        pivot = None
        for candidate in range(pivot_row, row_count):
            if matrix[candidate][column] != 0:
                pivot = candidate
                break
        if pivot is None:
            continue

        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        pivot_value = matrix[pivot_row][column]
        matrix[pivot_row] = [
            value / pivot_value
            for value in matrix[pivot_row]
        ]

        for row_index in range(row_count):
            if row_index == pivot_row:
                continue
            factor = matrix[row_index][column]
            if factor == 0:
                continue
            matrix[row_index] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(
                    matrix[row_index],
                    matrix[pivot_row],
                )
            ]

        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break

    free_columns = [
        column
        for column in range(column_count)
        if column not in pivot_columns
    ]
    relations = []
    for free_column in free_columns:
        vector = [Fraction(0, 1) for _ in range(column_count)]
        vector[free_column] = Fraction(1, 1)
        for row_index, pivot_column in enumerate(pivot_columns):
            vector[pivot_column] = -matrix[row_index][free_column]
        relation = _primitive_integer_vector(vector)
        if relation is not None:
            relations.append(relation)

    return tuple(relations)


def _primitive_integer_vector(
    values: List[Fraction],
) -> Optional[Tuple[int, ...]]:
    denominator_lcm = 1
    for value in values:
        denominator_lcm = _integer_lcm(denominator_lcm, value.denominator)

    integers = [
        int(value * denominator_lcm)
        for value in values
    ]
    divisor = 0
    for value in integers:
        divisor = gcd(divisor, abs(value))
    if divisor == 0:
        return None

    integers = [value // divisor for value in integers]
    for value in integers:
        if value == 0:
            continue
        if value < 0:
            integers = [-entry for entry in integers]
        break
    return tuple(integers)


def _integer_lcm(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return abs(left * right) // gcd(abs(left), abs(right))


def _binomial_relation_value(
    rhs_values: np.ndarray,
    relation: Tuple[int, ...],
) -> Optional[complex]:
    result = 1.0 + 0.0j
    for rhs, exponent in zip(rhs_values, relation):
        if exponent == 0:
            continue
        powered = _complex_integer_power(complex(rhs), exponent)
        if powered is None:
            return None
        result *= powered
        if not np.isfinite(result):
            return None
    return result


def _primitive_exponent_direction(
    row: np.ndarray,
) -> Tuple[Optional[Tuple[int, ...]], int]:
    entries = tuple(int(value) for value in row)
    divisor = 0
    for value in entries:
        divisor = gcd(divisor, abs(value))
    if divisor == 0:
        return None, 0

    primitive = tuple(value // divisor for value in entries)
    for value in primitive:
        if value == 0:
            continue
        if value < 0:
            primitive = tuple(-entry for entry in primitive)
            divisor = -divisor
        break
    return primitive, divisor


def _complex_integer_power(value: complex, exponent: int) -> Optional[complex]:
    try:
        if exponent == 0:
            result = 1.0 + 0.0j
        elif exponent > 0:
            result = value ** exponent
        else:
            result = 1.0 / (value ** (-exponent))
    except (ZeroDivisionError, OverflowError, FloatingPointError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _monomial_exponent_vector(
    monomial: Monomial,
    variables: List[Variable],
) -> np.ndarray:
    return np.asarray(
        [int(monomial.variables.get(variable, 0)) for variable in variables],
        dtype=int,
    )


def _integer_determinant(matrix: np.ndarray) -> int:
    values = [
        [int(entry) for entry in row]
        for row in np.asarray(matrix, dtype=object).tolist()
    ]
    size = len(values)
    if size == 0:
        return 1
    if any(len(row) != size for row in values):
        raise ValueError("Internal error: determinant requires a square matrix")
    if size == 1:
        return values[0][0]

    sign = 1
    previous_pivot = 1
    for index in range(size - 1):
        pivot_row = None
        for candidate_row in range(index, size):
            if values[candidate_row][index] != 0:
                pivot_row = candidate_row
                break
        if pivot_row is None:
            return 0
        if pivot_row != index:
            values[index], values[pivot_row] = values[pivot_row], values[index]
            sign *= -1

        pivot = values[index][index]
        for row in range(index + 1, size):
            for column in range(index + 1, size):
                values[row][column] = (
                    values[row][column] * pivot
                    - values[row][index] * values[index][column]
                ) // previous_pivot
        previous_pivot = pivot

    return sign * values[size - 1][size - 1]


def _enumerate_binomial_torus_candidates(
    exponent_matrix: np.ndarray,
    rhs_values: np.ndarray,
    solution_count: int,
    tol: float,
    *,
    max_candidate_count: int,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    lift_representatives, enumeration_meta = _binomial_lift_representatives(
        exponent_matrix,
        solution_count,
        max_candidate_count=max_candidate_count,
    )
    if lift_representatives is None:
        return [], enumeration_meta
    solve_matrix = np.asarray(exponent_matrix, dtype=float)
    enumeration_meta = {
        **enumeration_meta,
        "log_linear_solver": "solve_linear_system",
        "log_linear_solve_failure_count": 0,
    }
    if not np.all(np.isfinite(solve_matrix)):
        enumeration_meta["log_linear_solve_failure_count"] = len(
            lift_representatives
        )
        enumeration_meta["log_linear_solver_status"] = "nonfinite_exponent_matrix"
        return [], enumeration_meta

    principal_logs = np.log(rhs_values)
    if not np.all(np.isfinite(principal_logs)):
        enumeration_meta["log_linear_solve_failure_count"] = len(
            lift_representatives
        )
        enumeration_meta["log_linear_solver_status"] = "nonfinite_rhs_log"
        return [], enumeration_meta

    candidates: List[np.ndarray] = []
    uniqueness_tol = max(1000.0 * tol, 1e-8)
    log_residual_limit = max(1000.0 * tol, 1e-8)
    failure_count = 0

    for lift_indices in lift_representatives:
        lift = np.asarray(lift_indices, dtype=float)
        lifted_logs = principal_logs + 2j * np.pi * lift
        log_point = solve_linear_system(solve_matrix, lifted_logs)
        residual = _binomial_log_linear_residual_norm(
            solve_matrix,
            log_point,
            lifted_logs,
        )
        rhs_norm = _scaled_euclidean_norm(lifted_logs)
        residual_limit = log_residual_limit * max(1.0, rhs_norm)
        if (
            not np.all(np.isfinite(log_point))
            or not np.isfinite(residual)
            or residual > residual_limit
        ):
            failure_count += 1
            continue
        point = np.exp(log_point)
        if not np.all(np.isfinite(point)):
            failure_count += 1
            continue
        if any(
            _scaled_euclidean_norm(point - candidate) <= uniqueness_tol
            for candidate in candidates
        ):
            continue
        candidates.append(np.asarray(point, dtype=complex))
        if len(candidates) == solution_count:
            break

    enumeration_meta["log_linear_solve_failure_count"] = failure_count
    enumeration_meta["log_linear_residual_limit"] = float(log_residual_limit)
    enumeration_meta["log_linear_solver_status"] = (
        "used" if candidates else "no_finite_candidates"
    )
    return candidates, enumeration_meta


def _binomial_log_linear_residual_norm(
    matrix: np.ndarray,
    log_point: np.ndarray,
    rhs: np.ndarray,
) -> float:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            residual = matrix @ log_point - rhs
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return float("inf")
    return _scaled_euclidean_norm(residual)


def _binomial_lift_representatives(
    exponent_matrix: np.ndarray,
    solution_count: int,
    *,
    max_candidate_count: int,
) -> Tuple[Optional[List[Tuple[int, ...]]], Dict[str, Any]]:
    dimension = int(exponent_matrix.shape[0])
    fallback_search_space = int(solution_count) ** dimension
    metadata: Dict[str, Any] = {
        "method": "bruteforce_lift_representatives",
        "status": "pending",
        "search_space": fallback_search_space,
        "enumerated_lift_count": fallback_search_space,
        "representative_count": None,
        "hermite_diagonal": None,
    }

    hermite_representatives, hermite_meta = _hermite_lift_representatives(
        exponent_matrix,
        solution_count,
    )
    if hermite_representatives is not None:
        metadata.update(hermite_meta)
        return hermite_representatives, metadata

    if fallback_search_space > max_candidate_count:
        metadata["status"] = "too_many_bruteforce_lifts"
        return None, metadata

    ranges = tuple(range(int(solution_count)) for _ in range(dimension))
    representatives = [tuple(int(value) for value in lift) for lift in product(*ranges)]
    metadata.update({
        "status": "used",
        "representative_count": len(representatives),
    })
    return representatives, metadata


def _hermite_lift_representatives(
    exponent_matrix: np.ndarray,
    solution_count: int,
) -> Tuple[Optional[List[Tuple[int, ...]]], Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "method": "hermite_lift_representatives",
        "status": "unavailable",
        "search_space": int(solution_count),
        "enumerated_lift_count": int(solution_count),
        "representative_count": None,
        "hermite_diagonal": None,
    }
    try:
        import sympy as sp
        from sympy.matrices.normalforms import hermite_normal_form
    except Exception:
        return None, metadata

    try:
        hermite = hermite_normal_form(sp.Matrix(exponent_matrix))
        hermite_array = np.asarray(hermite.tolist(), dtype=object)
    except Exception:
        metadata["status"] = "failed"
        return None, metadata

    if hermite_array.shape != exponent_matrix.shape:
        metadata["status"] = "shape_mismatch"
        return None, metadata

    try:
        diagonal = tuple(
            abs(int(hermite_array[index, index]))
            for index in range(hermite_array.shape[0])
        )
    except Exception:
        metadata["status"] = "invalid_diagonal"
        return None, metadata

    if not diagonal or any(value <= 0 for value in diagonal):
        metadata["status"] = "singular_diagonal"
        return None, metadata
    if _product_int(diagonal) != int(solution_count):
        metadata["status"] = "degree_mismatch"
        metadata["hermite_diagonal"] = diagonal
        return None, metadata

    representatives = [
        tuple(int(value) for value in lift)
        for lift in product(*(range(value) for value in diagonal))
    ]
    metadata.update({
        "status": "used",
        "representative_count": len(representatives),
        "hermite_diagonal": diagonal,
    })
    return representatives, metadata


def _enumerate_triangular_branches(
    system: PolynomialSystem,
    variables: List[Variable],
    tol: float,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    constant_tol = max(1000.0 * tol, 1e-10)
    initial_state = {
        "assignments": {},
        "unused_indices": tuple(range(len(system.equations))),
        "steps": tuple(),
        "root_indices": tuple(),
        "multiplicity": 1,
    }
    states = [initial_state]
    triangular_steps: List[Dict[str, Any]] = []
    used_equation_indices = set()

    while states:
        complete = [
            state for state in states
            if len(state["assignments"]) == len(variables)
        ]
        if len(complete) == len(states):
            complete = _assign_triangular_branch_root_indices(complete)
            return complete, {
                "method": "recursive_univariate",
                "branch_count": sum(
                    int(branch.get("multiplicity", 1))
                    for branch in complete
                ),
                "branch_candidate_count": len(complete),
                "steps": tuple(triangular_steps),
                "used_equation_indices": tuple(sorted(used_equation_indices)),
            }

        next_states = []
        made_progress = False
        for state in states:
            assignments = state["assignments"]
            unused_indices = list(state["unused_indices"])
            state_valid = True
            retained_indices = []
            for equation_index in unused_indices:
                reduced = _substitute_constants(
                    system.equations[equation_index],
                    assignments,
                    cancellation_tol=constant_tol,
                )
                if not reduced.variables():
                    if not _constant_abs_within_tolerance(
                        reduced.evaluate({}),
                        constant_tol,
                    ):
                        state_valid = False
                        break
                    continue
                retained_indices.append(equation_index)
            if not state_valid:
                continue

            if len(assignments) == len(variables):
                next_states.append({
                    **state,
                    "unused_indices": tuple(retained_indices),
                })
                continue

            candidate = _select_triangular_equation(
                system,
                retained_indices,
                assignments,
                variables,
                cancellation_tol=constant_tol,
            )
            if candidate is None:
                return None

            equation_index, variable, reduced, degree = candidate
            coefficients = _univariate_coefficients(reduced, variable)
            coefficients_lossy = _univariate_coefficients_are_lossy(
                reduced,
                variable,
                coefficients,
            )
            root_solver_meta: Dict[str, Any] = {
                "method": "companion_roots",
                "coefficients_lossy": coefficients_lossy,
            }
            if coefficients_lossy:
                root_candidates, root_solver_meta = _lossy_univariate_root_candidates(
                    reduced,
                    variable,
                    coefficients,
                )
                factorization_meta = {
                    "attempted": False,
                    "used": False,
                    "method": None,
                    "status": "skipped_lossy_coefficients",
                    "original_degree": int(degree),
                    "distinct_factor_degree": None,
                    "total_factor_degree": None,
                    "factor_count": None,
                    "candidate_root_count": None,
                    "factors": tuple(),
                }
            else:
                root_candidates, factorization_meta = _univariate_factor_root_candidates(
                    reduced,
                    variable,
                    coefficients,
                )
            if root_candidates is None:
                roots = _safe_companion_roots(coefficients)
                if roots is None or len(roots) == 0:
                    return None
                root_candidates = _root_candidates_from_roots(
                    roots,
                    source="companion_roots",
                )
                root_solver_meta = {
                    "method": "companion_roots",
                    "coefficients_lossy": coefficients_lossy,
                    "candidate_root_count": len(root_candidates),
                    "status": "used",
                }
            elif (
                not root_candidates
                and root_solver_meta.get("root_solve_failure_count", 0)
            ):
                return None
            step_metadata = {
                "equation_index": int(equation_index),
                "variable": variable.name,
                "degree": int(degree),
                "root_count": int(
                    sum(
                        int(candidate["multiplicity"])
                        for candidate in root_candidates
                    )
                ),
                "root_candidate_count": int(len(root_candidates)),
                "factorization": factorization_meta,
                "root_solver": root_solver_meta,
            }
            triangular_steps.append(step_metadata)
            used_equation_indices.add(equation_index)
            for root_candidate in root_candidates:
                next_assignments = dict(assignments)
                next_assignments[variable] = complex(root_candidate["root"])
                next_unused = tuple(
                    index for index in retained_indices if index != equation_index
                )
                next_states.append({
                    "assignments": next_assignments,
                    "unused_indices": next_unused,
                    "steps": tuple(state["steps"]) + (step_metadata,),
                    "root_indices": tuple(state["root_indices"]) + tuple(
                        root_candidate["root_indices"]
                    ),
                    "multiplicity": (
                        int(state.get("multiplicity", 1))
                        * int(root_candidate["multiplicity"])
                    ),
                })
            made_progress = True

        if not made_progress:
            return [], {
                "method": "recursive_univariate",
                "branch_count": 0,
                "branch_candidate_count": 0,
                "steps": tuple(triangular_steps),
                "used_equation_indices": tuple(sorted(used_equation_indices)),
            }
        states = next_states

    return [], {
        "method": "recursive_univariate",
        "branch_count": 0,
        "branch_candidate_count": 0,
        "steps": tuple(triangular_steps),
        "used_equation_indices": tuple(sorted(used_equation_indices)),
    }


def _assign_triangular_branch_root_indices(
    branches: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    indexed_branches = []
    root_index_offset = 0
    for branch in branches:
        multiplicity = int(branch.get("multiplicity", 1))
        root_indices = tuple(
            range(root_index_offset, root_index_offset + multiplicity)
        )
        root_index_offset += multiplicity
        indexed_branches.append({
            **branch,
            "root_indices": root_indices,
            "multiplicity": multiplicity,
        })
    return indexed_branches


def _select_triangular_equation(
    system: PolynomialSystem,
    equation_indices: List[int],
    assignments: Dict[Variable, complex],
    variables: List[Variable],
    *,
    cancellation_tol: float = 0.0,
) -> Optional[Tuple[int, Variable, Polynomial, int]]:
    assigned = set(assignments)
    allowed = set(variables)
    candidates = []
    for equation_index in equation_indices:
        reduced = _substitute_constants(
            system.equations[equation_index],
            assignments,
            cancellation_tol=cancellation_tol,
        )
        remaining_variables = tuple(
            variable for variable in reduced.variables()
            if variable in allowed and variable not in assigned
        )
        if len(remaining_variables) != 1:
            continue
        variable = remaining_variables[0]
        if any(variable not in allowed for variable in reduced.variables()):
            continue
        degree = reduced.degree()
        if degree <= 0:
            continue
        try:
            _univariate_coefficients(reduced, variable)
        except ValueError:
            continue
        candidates.append((degree, equation_index, variable, reduced))
    if not candidates:
        return None
    degree, equation_index, variable, reduced = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2].name),
    )
    return equation_index, variable, reduced, degree


def _univariate_coefficients(
    polynomial: Polynomial,
    variable: Variable,
) -> Tuple[complex, ...]:
    row_scale = _polynomial_coefficient_scale(polynomial)
    degree = polynomial.degree()
    coefficients = [0.0 + 0.0j for _ in range(degree + 1)]
    for term in polynomial.terms:
        if not term.variables:
            exponent = 0
        else:
            if len(term.variables) != 1 or variable not in term.variables:
                raise ValueError("Internal error: expected a univariate polynomial")
            exponent = term.variables[variable]
        coefficients[degree - exponent] += complex(
            _safe_scaled_coefficient(term.coefficient, row_scale)
        )

    while len(coefficients) > 1 and coefficients[0] == 0:
        coefficients.pop(0)
    return tuple(coefficients)


def _reduce_coordinate_assignments(
    system: PolynomialSystem,
    variables: List[Variable],
    tol: float,
) -> Optional[Dict[str, Any]]:
    assignments: Dict[Variable, complex] = {}
    assignment_sources = []
    remaining_equations = []
    assignment_tol = max(1000.0 * tol, 1e-10)

    for index, equation in enumerate(system.equations):
        assignment = _coordinate_assignment(equation, assignment_tol)
        if assignment is None:
            remaining_equations.append(equation)
            continue

        variable, value, coefficient = assignment
        if variable in assignments:
            if abs(assignments[variable] - value) > assignment_tol:
                return {
                    "system": PolynomialSystem([Polynomial([1])]),
                    "variables": [],
                    "assignments": assignments,
                    "assignment_sources": assignment_sources,
                    "inconsistent": True,
                }
        else:
            assignments[variable] = value
            assignment_sources.append({
                "equation_index": index,
                "variable": variable.name,
                "value": _complex_metadata(value),
                "coefficient": _complex_metadata(coefficient),
            })

    if not assignments:
        return None

    reduced_equations = [
        _substitute_constants(equation, assignments)
        for equation in remaining_equations
    ]
    if _system_uses_assigned_variables(reduced_equations, assignments):
        return None
    reduced_variables = [
        variable for variable in variables if variable not in assignments
    ]
    return {
        "system": PolynomialSystem(reduced_equations),
        "variables": reduced_variables,
        "assignments": assignments,
        "assignment_sources": tuple(assignment_sources),
        "inconsistent": False,
    }


def _system_uses_assigned_variables(
    equations: List[Polynomial],
    assignments: Dict[Variable, complex],
) -> bool:
    assigned = set(assignments)
    return any(
        variable in assigned
        for equation in equations
        for variable in equation.variables()
    )


def _coordinate_assignment(
    polynomial: Polynomial,
    tol: float,
) -> Optional[Tuple[Variable, complex, complex]]:
    row_scale = _polynomial_coefficient_scale(polynomial)
    constant = 0.0 + 0.0j
    linear_terms: Dict[Variable, complex] = {}
    for term in polynomial.terms:
        coefficient = complex(
            _safe_scaled_coefficient(term.coefficient, row_scale)
        )
        if not term.variables:
            constant += coefficient
            continue
        if term.degree() != 1 or len(term.variables) != 1:
            return None
        variable = next(iter(term.variables))
        linear_terms[variable] = (
            linear_terms.get(variable, 0.0 + 0.0j)
            + coefficient
        )

    nonzero_terms = {
        variable: coefficient
        for variable, coefficient in linear_terms.items()
        if abs(coefficient) > tol
    }
    if len(nonzero_terms) != 1:
        return None
    variable, coefficient = next(iter(nonzero_terms.items()))
    return variable, -constant / coefficient, coefficient


def _substitute_constants(
    polynomial: Polynomial,
    assignments: Dict[Variable, complex],
    *,
    cancellation_tol: float = 0.0,
) -> Polynomial:
    row_scale = (
        _polynomial_coefficient_scale(polynomial)
        if _requires_row_coefficient_scaling(polynomial)
        else 1.0
    )
    if _substitution_would_drop_unassigned_term(polynomial, assignments, row_scale):
        exact_reduced = _substitute_constants_with_exact_assignments(
            polynomial,
            assignments,
        )
        if exact_reduced is not None:
            return exact_reduced
        return polynomial
    grouped_terms: Dict[
        Tuple[Tuple[Variable, int], ...],
        Dict[str, Any],
    ] = {}
    for term in polynomial.terms:
        coefficient, variables = _substitute_term_constants_with_row_scale(
            term,
            assignments,
            row_scale,
        )
        key = tuple(sorted(variables.items(), key=lambda item: item[0].name))
        if key not in grouped_terms:
            grouped_terms[key] = {
                "variables": variables,
                "coefficient": 0.0 + 0.0j,
                "coefficient_scale": 0.0,
            }
        grouped_terms[key]["coefficient"] += coefficient
        grouped_terms[key]["coefficient_scale"] += abs(coefficient)

    terms = []
    for grouped in grouped_terms.values():
        coefficient = grouped["coefficient"]
        coefficient_scale = grouped["coefficient_scale"]
        if coefficient == 0:
            continue
        if (
            cancellation_tol > 0.0
            and coefficient_scale > 0.0
            and abs(coefficient) <= cancellation_tol * coefficient_scale
        ):
            continue
        terms.append(Monomial(grouped["variables"], coefficient=coefficient))
    return Polynomial(terms)


def _substitute_constants_with_exact_assignments(
    polynomial: Polynomial,
    assignments: Dict[Variable, complex],
) -> Optional[Polynomial]:
    terms = []
    for term in polynomial.terms:
        coefficient: Number = term.coefficient
        variables = {}
        for variable, exponent in term.variables.items():
            if variable not in assignments:
                variables[variable] = exponent
                continue
            power = _exact_assignment_power(assignments[variable], int(exponent))
            if power is None:
                return None
            try:
                coefficient = coefficient * power
            except OverflowError:
                return None
        terms.append(Monomial(variables, coefficient=coefficient))
    try:
        return Polynomial(terms)
    except (TypeError, ValueError, OverflowError):
        return None


def _exact_assignment_power(value: complex, exponent: int) -> Optional[int]:
    if exponent == 0:
        return 1
    coordinate = complex(value)
    scale = max(1.0, abs(coordinate.real))
    if abs(coordinate.imag) > 1e-12 * scale:
        return None
    rounded = round(coordinate.real)
    if abs(coordinate.real - rounded) > 1e-12 * max(1.0, abs(rounded)):
        return None
    return int(rounded) ** exponent


def _substitution_would_drop_unassigned_term(
    polynomial: Polynomial,
    assignments: Dict[Variable, complex],
    row_scale: Number,
) -> bool:
    if row_scale == 1.0:
        return False
    for term in polynomial.terms:
        if term.coefficient == 0:
            continue
        try:
            substituted_coefficient, _ = _substitute_term_constants_with_row_scale(
                term,
                assignments,
                row_scale,
            )
        except (OverflowError, FloatingPointError):
            return True
        if substituted_coefficient == 0:
            exact_nonzero = _term_survives_exact_assignment(term, assignments)
            if exact_nonzero is not False:
                return True
    return False


def _term_survives_exact_assignment(
    term: Monomial,
    assignments: Dict[Variable, complex],
) -> Optional[bool]:
    coefficient: Number = term.coefficient
    for variable, exponent in term.variables.items():
        if variable not in assignments:
            continue
        power = _exact_assignment_power(assignments[variable], int(exponent))
        if power is None:
            return None
        try:
            coefficient = coefficient * power
        except OverflowError:
            return None
    try:
        return coefficient != 0
    except Exception:
        return None


def _substitute_term_constants_with_row_scale(
    term: Monomial,
    assignments: Dict[Variable, complex],
    row_scale: Number,
) -> Tuple[complex, Dict[Variable, int]]:
    assigned_variables = {}
    remaining_variables = {}
    for variable, exponent in term.variables.items():
        if variable in assignments:
            assigned_variables[variable] = exponent
        else:
            remaining_variables[variable] = exponent

    if assigned_variables:
        assigned_term = Monomial(
            assigned_variables,
            coefficient=term.coefficient,
        )
        coefficient = _evaluate_scaled_term(assigned_term, assignments, row_scale)
    else:
        coefficient = complex(
            _safe_scaled_coefficient(term.coefficient, row_scale)
        )
    return coefficient, remaining_variables


def _solve_monomial_zero_branches(
    *,
    original_system: PolynomialSystem,
    working_system: PolynomialSystem,
    variables: List[Variable],
    tol: float,
    verbose: bool,
    store_paths: bool,
    use_endgame: bool,
    endgame_options: Optional[Dict[str, Any]],
    tracking_options: Dict[str, Any],
    deduplication_tol_factor: float,
    singular_deduplication_tol: float,
    allow_underdetermined: bool,
    rng: Any,
    generated_gamma: bool,
    preprocessing: Dict[str, Any],
    equation_scaling: Dict[str, Any],
    square_up: Dict[str, Any],
    start_time: float,
    max_paths: Optional[int],
) -> Optional[SolutionSet]:
    branch_data = _monomial_zero_branch_data(working_system, variables, tol)
    if branch_data is None:
        return None

    equation_index = branch_data["equation_index"]
    branch_specs = list(branch_data["branches"])
    branch_results = []
    raw_solutions: List[Solution] = []
    residual_limit = 100.0 * tol
    base_backward_error_limit = _backward_error_limit(tol)
    path_offset = 0

    for branch_index, branch_spec in enumerate(branch_specs):
        branch_type = branch_spec["type"]
        assignments = dict(branch_spec.get("assignments", {}))
        if branch_type == "cofactor":
            reduced_system = PolynomialSystem([
                branch_spec["polynomial"] if index == equation_index else equation
                for index, equation in enumerate(working_system.equations)
            ])
            reduced_variables = list(variables)
        else:
            reduced_system = PolynomialSystem([
                _substitute_constants(equation, assignments)
                for index, equation in enumerate(working_system.equations)
                if index != equation_index
            ])
            reduced_variables = [
                variable for variable in variables if variable not in assignments
            ]

        if _reduced_branch_has_free_variables(reduced_system, reduced_variables, tol):
            _raise_positive_dimensional_branch_error(
                "monomial zero",
                equation_index,
                branch_index,
                reduced_system,
                reduced_variables,
            )
        branch_result = solve(
            reduced_system,
            variables=reduced_variables,
            tol=tol,
            verbose=False,
            store_paths=store_paths,
            use_endgame=use_endgame,
            endgame_options=endgame_options,
            tracking_options=tracking_options.copy(),
            deduplication_tol_factor=deduplication_tol_factor,
            singular_deduplication_tol=singular_deduplication_tol,
            allow_underdetermined=allow_underdetermined,
            scale_equations=False,
            max_paths=_remaining_path_limit(max_paths, path_offset),
            _allow_zero_max_paths=True,
            random_state=rng,
        )

        branch_results.append({
            "branch_index": int(branch_index),
            "branch_type": branch_type,
            "assigned_variables": tuple(
                variable.name for variable in assignments
            ),
            "factor_multiplicity": int(branch_spec["multiplicity"]),
            "factor": branch_spec["description"],
            "reduced_variables": tuple(
                variable.name for variable in reduced_variables
            ),
            "reduced_equations": len(reduced_system.equations),
            "solution_count": len(branch_result),
            "raw_solutions_found": int(
                branch_result._meta.get("raw_solutions_found", 0)
            ),
            "total_paths": int(branch_result._meta.get("total_paths", 0)),
            "successful_paths": int(
                branch_result._meta.get("successful_paths", 0)
            ),
            "failed_paths": int(branch_result._meta.get("failed_paths", 0)),
            "start_source": branch_result._meta.get("start_system", {}).get(
                "source"
            ),
            "reduced_meta": branch_result._meta.copy(),
        })

        branch_multiplicity = int(branch_spec["multiplicity"])
        for solution_index, reduced_solution in enumerate(branch_result):
            values = dict(assignments)
            values.update(reduced_solution.values)
            ordered_values = _solution_values_from_complete_coordinates(
                values,
                variables,
                label="lifted branch solution",
            )
            residual = _residual_norm(original_system, ordered_values)
            scaled_residual = _scaled_residual_norm(original_system, ordered_values)
            acceptance_scaled_residual = _scaled_residual_norm(
                working_system,
                ordered_values,
            )
            backward_error = _backward_error_norm(original_system, ordered_values)
            is_singular = _is_singular(
                original_system,
                ordered_values,
                tuple(variables),
                threshold=1e12,
                rank_tolerance=_rank_tolerance_for_tol(tol),
            )
            acceptance_limit = (
                max(residual_limit, singular_deduplication_tol)
                if is_singular else residual_limit
            )
            backward_error_limit = max(acceptance_limit, base_backward_error_limit)
            if not _solution_quality_within_limits(
                acceptance_scaled_residual,
                backward_error,
                acceptance_limit,
                backward_error_limit,
            ):
                continue

            path_indices = tuple(
                path_offset + path_index
                for path_index in getattr(reduced_solution, "path_indices", ())
                if path_index is not None
            )
            lifted = Solution(
                values=ordered_values,
                residual=residual,
                is_singular=is_singular,
                path_index=path_indices[0] if path_indices else None,
            )
            _copy_solution_attributes(reduced_solution, lifted)
            lifted.values = ordered_values
            lifted.residual = residual
            lifted.scaled_residual = scaled_residual
            lifted.backward_error = backward_error
            lifted.path_indices = tuple(sorted(set(path_indices)))
            lifted.multiplicity = (
                int(getattr(reduced_solution, "multiplicity", 1))
                * branch_multiplicity
            )
            if (
                not hasattr(lifted, "path_info")
                or not isinstance(lifted.path_info, dict)
            ):
                lifted.path_info = {}
            lifted.path_info["monomial_zero_branch"] = {
                "equation_index": int(equation_index),
                "branch_index": int(branch_index),
                "branch_type": branch_type,
                "assigned_variables": tuple(
                    variable.name for variable in assignments
                ),
                "factor": branch_spec["description"],
                "factor_multiplicity": branch_multiplicity,
                "solution_index": int(solution_index),
                "lifted": True,
            }
            lifted.path_info["method"] = "monomial_zero_branches"
            lifted.path_info["accepted"] = True
            lifted.path_info["solution_residual"] = float(residual)
            lifted.path_info["scaled_solution_residual"] = float(scaled_residual)
            lifted.path_info["acceptance_scaled_solution_residual"] = float(
                acceptance_scaled_residual
            )
            lifted.path_info["backward_error"] = float(backward_error)
            lifted.path_info["residual_limit"] = float(acceptance_limit)
            lifted.path_info["backward_error_limit"] = float(backward_error_limit)
            _attach_cluster_metadata(lifted)
            raw_solutions.append(lifted)

        path_offset += int(branch_result._meta.get("total_paths", 0))
        _check_path_limit(
            path_offset,
            max_paths,
            "monomial zero branch decomposition",
        )

    unique_solutions = _deduplicate_solutions(
        raw_solutions,
        original_system,
        variables,
        regular_tolerance=tol * deduplication_tol_factor,
        singular_tolerance=singular_deduplication_tol,
        rank_tolerance=_rank_tolerance_for_tol(tol),
        polish_tolerance=tol,
    )
    total_paths = sum(branch["total_paths"] for branch in branch_results)
    successful_paths = sum(branch["successful_paths"] for branch in branch_results)
    failed_paths = sum(branch["failed_paths"] for branch in branch_results)

    result = SolutionSet(unique_solutions, original_system)
    result._meta['total_paths'] = total_paths
    result._meta['successful_paths'] = successful_paths
    result._meta['failed_paths'] = failed_paths
    result._meta['solve_time'] = time.time() - start_time
    result._meta['raw_solutions_found'] = len(raw_solutions)
    result._meta['tracking_options'] = tracking_options.copy()
    result._meta['generated_gamma'] = generated_gamma
    result._meta['preprocessing'] = preprocessing
    result._meta['equation_scaling'] = equation_scaling
    result._meta['square_up'] = square_up
    result._meta['start_system'] = {
        "source": "monomial_zero_branches",
        "path_count": total_paths,
        "branch_count": len(branch_specs),
    }
    result._meta['path_summary'] = _summarize_path_results([], [], len(raw_solutions))
    result._meta['multiplicity_summary'] = _summarize_solution_multiplicities(
        unique_solutions
    )
    result._meta['monomial_zero_branches'] = {
        "method": "coordinate_union",
        "status": "solved" if raw_solutions else "no_solutions",
        "equation_index": int(equation_index),
        "coefficient": branch_data["coefficient"],
        "monomial_degree": int(branch_data["degree"]),
        "branch_variables": tuple(
            variable.name for variable in branch_data["branch_variables"]
        ),
        "exponents": branch_data["exponents"],
        "cofactor": branch_data.get("cofactor"),
        "cofactor_degree": branch_data.get("cofactor_degree"),
        "common_factor_kind": branch_data["kind"],
        "branch_count": len(branch_specs),
        "accepted_branch_solution_count": len(raw_solutions),
        "distinct_solution_count": len(unique_solutions),
        "branches": tuple(branch_results),
    }
    return result


def _monomial_zero_branch_data(
    system: PolynomialSystem,
    variables: List[Variable],
    tol: float,
) -> Optional[Dict[str, Any]]:
    if len(variables) <= 1 or len(system.equations) <= 1:
        return None

    coefficient_tol = max(1000.0 * tol, 1e-12)
    variable_order = {variable: index for index, variable in enumerate(variables)}
    candidates = []
    for equation_index, equation in enumerate(system.equations):
        row_scale = _polynomial_coefficient_scale(equation)
        nonzero_terms = [
            term for term in equation.terms
            if abs(
                complex(_safe_scaled_coefficient(term.coefficient, row_scale))
            ) > coefficient_tol
        ]
        if not nonzero_terms:
            continue
        if len(nonzero_terms) == 1:
            term = nonzero_terms[0]
            if not term.variables:
                continue
            branch_variables = tuple(
                sorted(term.variables, key=lambda variable: variable_order[variable])
            )
            if len(branch_variables) <= 1:
                continue
            branches = tuple(
                {
                    "type": "coordinate",
                    "assignments": {variable: 0.0 + 0.0j},
                    "multiplicity": int(term.variables[variable]),
                    "description": variable.name,
                }
                for variable in branch_variables
            )
            candidates.append((
                len(branches),
                term.degree(),
                equation_index,
                {
                    "kind": "monomial",
                    "coefficient": _complex_metadata(term.coefficient),
                    "degree": term.degree(),
                    "branch_variables": branch_variables,
                    "exponents": {
                        variable.name: int(term.variables[variable])
                        for variable in branch_variables
                    },
                    "branches": branches,
                },
            ))
            continue

        common_exponents = _common_monomial_exponents(nonzero_terms)
        if not common_exponents:
            continue
        cofactor = _divide_polynomial_by_monomial(equation, common_exponents)
        if cofactor.degree() <= 0:
            continue
        branch_variables = tuple(
            sorted(common_exponents, key=lambda variable: variable_order[variable])
        )
        branches = [
            {
                "type": "coordinate",
                "assignments": {variable: 0.0 + 0.0j},
                "multiplicity": int(common_exponents[variable]),
                "description": variable.name,
            }
            for variable in branch_variables
        ]
        branches.append({
            "type": "cofactor",
            "assignments": {},
            "multiplicity": 1,
            "description": repr(cofactor),
            "polynomial": cofactor,
        })
        candidates.append((
            len(branches),
            equation.degree(),
            equation_index,
            {
                "kind": "common_monomial_factor",
                "coefficient": None,
                "degree": sum(common_exponents.values()),
                "branch_variables": branch_variables,
                "exponents": {
                    variable.name: int(common_exponents[variable])
                    for variable in branch_variables
                },
                "cofactor": repr(cofactor),
                "cofactor_degree": cofactor.degree(),
                "branches": tuple(branches),
            },
        ))

    if not candidates:
        return None

    _, _, equation_index, data = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    return {"equation_index": equation_index, **data}


def _common_monomial_exponents(terms: List[Monomial]) -> Dict[Variable, int]:
    common_variables = set(terms[0].variables)
    for term in terms[1:]:
        common_variables.intersection_update(term.variables)

    common_exponents = {}
    for variable in common_variables:
        exponent = min(int(term.variables.get(variable, 0)) for term in terms)
        if exponent > 0:
            common_exponents[variable] = exponent
    return common_exponents


def _divide_polynomial_by_monomial(
    polynomial: Polynomial,
    exponents: Dict[Variable, int],
) -> Polynomial:
    terms = []
    for term in polynomial.terms:
        variables = term.variables.copy()
        for variable, exponent in exponents.items():
            remaining = variables.get(variable, 0) - exponent
            if remaining > 0:
                variables[variable] = remaining
            else:
                variables.pop(variable, None)
        terms.append(Monomial(variables, coefficient=term.coefficient))
    return Polynomial(terms)


def _reduced_branch_has_free_variables(
    reduced_system: PolynomialSystem,
    reduced_variables: List[Variable],
    tol: float,
) -> bool:
    if not reduced_variables:
        return False
    used_variables = reduced_system.variables()
    if set(reduced_variables).issubset(used_variables):
        return False

    constant_tol = float(tol)
    has_inconsistent_constant = any(
        not equation.variables()
        and not _constant_abs_within_tolerance(equation.evaluate({}), constant_tol)
        for equation in reduced_system.equations
    )
    return not has_inconsistent_constant


def _raise_positive_dimensional_branch_error(
    decomposition: str,
    equation_index: int,
    branch_index: int,
    reduced_system: PolynomialSystem,
    reduced_variables: List[Variable],
) -> None:
    used_variables = reduced_system.variables()
    free_variables = tuple(
        variable.name
        for variable in reduced_variables
        if variable not in used_variables
    )
    raise ValueError(
        f"{decomposition} branch decomposition found a "
        "positive-dimensional component "
        f"(equation {equation_index}, branch {branch_index}, "
        f"free variable(s): {', '.join(free_variables)}). "
        "solve() returns finite zero-dimensional roots only; use "
        "witness-set tools for positive-dimensional components."
    )


def _raise_no_zero_dimensional_constraints_error(
    context: str,
    variables: List[Variable],
) -> None:
    variable_names = ", ".join(variable.name for variable in variables)
    raise ValueError(
        f"{context} has no finite zero-dimensional constraints "
        f"on variable(s): {variable_names}. The solution set is "
        "positive-dimensional; add independent equations or use "
        "witness-set tools for positive-dimensional components."
    )


def _solve_factorized_branches(
    *,
    original_system: PolynomialSystem,
    working_system: PolynomialSystem,
    variables: List[Variable],
    tol: float,
    verbose: bool,
    store_paths: bool,
    use_endgame: bool,
    endgame_options: Optional[Dict[str, Any]],
    tracking_options: Dict[str, Any],
    deduplication_tol_factor: float,
    singular_deduplication_tol: float,
    allow_underdetermined: bool,
    rng: Any,
    generated_gamma: bool,
    preprocessing: Dict[str, Any],
    equation_scaling: Dict[str, Any],
    square_up: Dict[str, Any],
    start_time: float,
    max_paths: Optional[int],
) -> Optional[SolutionSet]:
    branch_data = _factorized_branch_data(working_system, variables, tol)
    if branch_data is None:
        return None

    equation_index = branch_data["equation_index"]
    branch_factors = branch_data["factors"]
    branch_results = []
    raw_solutions: List[Solution] = []
    residual_limit = 100.0 * tol
    base_backward_error_limit = _backward_error_limit(tol)
    path_offset = 0

    for branch_index, factor_data in enumerate(branch_factors):
        factor_polynomial = factor_data["polynomial"]
        reduced_system = PolynomialSystem([
            factor_polynomial if index == equation_index else equation
            for index, equation in enumerate(working_system.equations)
        ])
        if _reduced_branch_has_free_variables(reduced_system, variables, tol):
            _raise_positive_dimensional_branch_error(
                "factorized",
                equation_index,
                branch_index,
                reduced_system,
                variables,
            )

        branch_result = solve(
            reduced_system,
            variables=variables,
            tol=tol,
            verbose=False,
            store_paths=store_paths,
            use_endgame=use_endgame,
            endgame_options=endgame_options,
            tracking_options=tracking_options.copy(),
            deduplication_tol_factor=deduplication_tol_factor,
            singular_deduplication_tol=singular_deduplication_tol,
            allow_underdetermined=allow_underdetermined,
            scale_equations=False,
            max_paths=_remaining_path_limit(max_paths, path_offset),
            _allow_zero_max_paths=True,
            random_state=rng,
        )

        branch_results.append({
            "branch_index": int(branch_index),
            "factor": factor_data["expression"],
            "factor_degree": int(factor_polynomial.degree()),
            "factor_multiplicity": int(factor_data["multiplicity"]),
            "solution_count": len(branch_result),
            "raw_solutions_found": int(
                branch_result._meta.get("raw_solutions_found", 0)
            ),
            "total_paths": int(branch_result._meta.get("total_paths", 0)),
            "successful_paths": int(
                branch_result._meta.get("successful_paths", 0)
            ),
            "failed_paths": int(branch_result._meta.get("failed_paths", 0)),
            "start_source": branch_result._meta.get("start_system", {}).get(
                "source"
            ),
            "reduced_meta": branch_result._meta.copy(),
        })

        factor_multiplicity = int(factor_data["multiplicity"])
        for solution_index, branch_solution in enumerate(branch_result):
            ordered_values = _solution_values_from_complete_coordinates(
                branch_solution.values,
                variables,
                label="branch solution",
            )
            residual = _residual_norm(original_system, ordered_values)
            scaled_residual = _scaled_residual_norm(original_system, ordered_values)
            acceptance_scaled_residual = _scaled_residual_norm(
                working_system,
                ordered_values,
            )
            backward_error = _backward_error_norm(original_system, ordered_values)
            is_singular = _is_singular(
                original_system,
                ordered_values,
                tuple(variables),
                threshold=1e12,
                rank_tolerance=_rank_tolerance_for_tol(tol),
            )
            acceptance_limit = (
                max(residual_limit, singular_deduplication_tol)
                if is_singular else residual_limit
            )
            backward_error_limit = max(acceptance_limit, base_backward_error_limit)
            if not _solution_quality_within_limits(
                acceptance_scaled_residual,
                backward_error,
                acceptance_limit,
                backward_error_limit,
            ):
                continue

            path_indices = tuple(
                path_offset + path_index
                for path_index in getattr(branch_solution, "path_indices", ())
                if path_index is not None
            )
            lifted = Solution(
                values=ordered_values,
                residual=residual,
                is_singular=is_singular,
                path_index=path_indices[0] if path_indices else None,
            )
            _copy_solution_attributes(branch_solution, lifted)
            lifted.values = ordered_values
            lifted.residual = residual
            lifted.scaled_residual = scaled_residual
            lifted.backward_error = backward_error
            lifted.path_indices = tuple(sorted(set(path_indices)))
            lifted.multiplicity = (
                int(getattr(branch_solution, "multiplicity", 1))
                * factor_multiplicity
            )
            if (
                not hasattr(lifted, "path_info")
                or not isinstance(lifted.path_info, dict)
            ):
                lifted.path_info = {}
            lifted.path_info["factorized_branch"] = {
                "equation_index": int(equation_index),
                "branch_index": int(branch_index),
                "factor": factor_data["expression"],
                "factor_multiplicity": factor_multiplicity,
                "solution_index": int(solution_index),
                "lifted": True,
            }
            lifted.path_info["method"] = "factorized_branches"
            lifted.path_info["accepted"] = True
            lifted.path_info["solution_residual"] = float(residual)
            lifted.path_info["scaled_solution_residual"] = float(scaled_residual)
            lifted.path_info["acceptance_scaled_solution_residual"] = float(
                acceptance_scaled_residual
            )
            lifted.path_info["backward_error"] = float(backward_error)
            lifted.path_info["residual_limit"] = float(acceptance_limit)
            lifted.path_info["backward_error_limit"] = float(backward_error_limit)
            _attach_cluster_metadata(lifted)
            raw_solutions.append(lifted)

        path_offset += int(branch_result._meta.get("total_paths", 0))
        _check_path_limit(
            path_offset,
            max_paths,
            "factorized branch decomposition",
        )

    unique_solutions = _deduplicate_solutions(
        raw_solutions,
        original_system,
        variables,
        regular_tolerance=tol * deduplication_tol_factor,
        singular_tolerance=singular_deduplication_tol,
        rank_tolerance=_rank_tolerance_for_tol(tol),
        polish_tolerance=tol,
    )
    total_paths = sum(branch["total_paths"] for branch in branch_results)
    successful_paths = sum(branch["successful_paths"] for branch in branch_results)
    failed_paths = sum(branch["failed_paths"] for branch in branch_results)

    result = SolutionSet(unique_solutions, original_system)
    result._meta['total_paths'] = total_paths
    result._meta['successful_paths'] = successful_paths
    result._meta['failed_paths'] = failed_paths
    result._meta['solve_time'] = time.time() - start_time
    result._meta['raw_solutions_found'] = len(raw_solutions)
    result._meta['tracking_options'] = tracking_options.copy()
    result._meta['generated_gamma'] = generated_gamma
    result._meta['preprocessing'] = preprocessing
    result._meta['equation_scaling'] = equation_scaling
    result._meta['square_up'] = square_up
    result._meta['start_system'] = {
        "source": "factorized_branches",
        "path_count": total_paths,
        "branch_count": len(branch_factors),
    }
    result._meta['path_summary'] = _summarize_path_results([], [], len(raw_solutions))
    result._meta['multiplicity_summary'] = _summarize_solution_multiplicities(
        unique_solutions
    )
    result._meta['factorized_branches'] = {
        "method": "sympy_factor_list",
        "status": "solved" if raw_solutions else "no_solutions",
        "equation_index": int(equation_index),
        "original_degree": int(branch_data["original_degree"]),
        "distinct_factor_degree": int(branch_data["distinct_factor_degree"]),
        "total_factor_degree": int(branch_data["total_factor_degree"]),
        "factor_count": len(branch_factors),
        "accepted_branch_solution_count": len(raw_solutions),
        "distinct_solution_count": len(unique_solutions),
        "factors": tuple(
            {
                "factor": factor["expression"],
                "degree": int(factor["polynomial"].degree()),
                "multiplicity": int(factor["multiplicity"]),
            }
            for factor in branch_factors
        ),
        "branches": tuple(branch_results),
    }
    return result


def _factorized_branch_data(
    system: PolynomialSystem,
    variables: List[Variable],
    tol: float,
) -> Optional[Dict[str, Any]]:
    if len(system.equations) <= 1:
        return None

    try:
        import sympy as sp
    except Exception:
        return None

    candidates = []
    for equation_index, equation in enumerate(system.equations):
        if equation.degree() <= 1:
            continue
        symbols, expr = _polynomial_to_sympy_expr(equation, variables, sp)
        if expr is None:
            continue
        try:
            _, raw_factors = sp.factor_list(expr, *symbols)
        except Exception:
            continue

        factor_data = []
        for factor_expr, multiplicity in raw_factors:
            polynomial = _sympy_expr_to_polynomial(factor_expr, variables, symbols, sp)
            if polynomial is None or polynomial.degree() <= 0:
                continue
            factor_data.append({
                "expression": str(sp.factor(factor_expr)),
                "multiplicity": int(multiplicity),
                "polynomial": polynomial,
            })

        if len(factor_data) <= 1:
            continue
        original_degree = equation.degree()
        distinct_factor_degree = sum(
            factor["polynomial"].degree() for factor in factor_data
        )
        total_factor_degree = sum(
            factor["polynomial"].degree() * factor["multiplicity"]
            for factor in factor_data
        )
        if total_factor_degree != original_degree:
            continue
        candidates.append((
            distinct_factor_degree,
            len(factor_data),
            original_degree,
            equation_index,
            factor_data,
            total_factor_degree,
        ))

    if not candidates:
        return None

    (
        distinct_factor_degree,
        _,
        original_degree,
        equation_index,
        factor_data,
        total_factor_degree,
    ) = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    return {
        "equation_index": equation_index,
        "original_degree": original_degree,
        "distinct_factor_degree": distinct_factor_degree,
        "total_factor_degree": total_factor_degree,
        "factors": tuple(factor_data),
    }


def _polynomial_to_sympy_expr(
    polynomial: Polynomial,
    variables: List[Variable],
    sp: Any,
) -> Tuple[Tuple[Any, ...], Optional[Any]]:
    symbols = tuple(sp.Symbol(variable.name) for variable in variables)
    symbol_by_variable = {
        variable: symbol for variable, symbol in zip(variables, symbols)
    }
    expr = sp.Integer(0)
    for term in polynomial.terms:
        coefficient = _coefficient_to_sympy(term.coefficient, sp)
        if coefficient is None:
            return symbols, None
        term_expr = coefficient
        for variable, exponent in term.variables.items():
            symbol = symbol_by_variable.get(variable)
            if symbol is None:
                return symbols, None
            term_expr *= symbol ** int(exponent)
        expr += term_expr
    return symbols, expr


def _sympy_expr_to_polynomial(
    expr: Any,
    variables: List[Variable],
    symbols: Tuple[Any, ...],
    sp: Any,
) -> Optional[Polynomial]:
    try:
        poly = sp.Poly(sp.expand(expr), *symbols)
    except Exception:
        return None

    terms = []
    for exponents, coefficient in poly.terms():
        coefficient_value = _safe_coefficient_from_sympy(coefficient)
        if coefficient_value is None:
            return None
        monomial_variables = {
            variable: int(exponent)
            for variable, exponent in zip(variables, exponents)
            if int(exponent) != 0
        }
        terms.append(Monomial(monomial_variables, coefficient=coefficient_value))
    return Polynomial(terms)


def _coefficient_to_sympy(value: Any, sp: Any) -> Optional[Any]:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, Integral):
        return sp.Integer(int(value))
    if isinstance(value, Rational):
        return sp.Rational(int(value.numerator), int(value.denominator))
    if isinstance(value, Real):
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            return None
        return sp.Float(numeric_value)
    try:
        numeric_value = complex(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(numeric_value.real) or not np.isfinite(numeric_value.imag):
        return None
    return sp.Float(numeric_value.real) + sp.Float(numeric_value.imag) * sp.I


def _safe_coefficient_from_sympy(value: Any) -> Optional[Number]:
    if getattr(value, "is_Integer", False):
        try:
            return int(value)
        except Exception:
            return None
    if getattr(value, "is_Rational", False):
        try:
            numerator, denominator = int(value.p), int(value.q)
        except Exception:
            return None
        if denominator == 1:
            return numerator
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(numeric_value) or (value != 0 and numeric_value == 0.0):
            return None
        return numeric_value
    try:
        numeric_value = complex(value)
    except (TypeError, ValueError):
        try:
            numeric_value = complex(value.evalf())
        except Exception:
            return None
    if not np.isfinite(numeric_value.real) or not np.isfinite(numeric_value.imag):
        return None
    if numeric_value.imag == 0:
        return numeric_value.real
    return numeric_value


def _solve_reduced_coordinate_system(
    *,
    original_system: PolynomialSystem,
    working_system: PolynomialSystem,
    reduced_system: PolynomialSystem,
    assignments: Dict[Variable, complex],
    reduced_variables: List[Variable],
    all_variables: List[Variable],
    tol: float,
    verbose: bool,
    store_paths: bool,
    use_endgame: bool,
    endgame_options: Optional[Dict[str, Any]],
    tracking_options: Dict[str, Any],
    deduplication_tol_factor: float,
    singular_deduplication_tol: float,
    allow_underdetermined: bool,
    rng: Any,
    generated_gamma: bool,
    preprocessing: Dict[str, Any],
    equation_scaling: Dict[str, Any],
    square_up: Dict[str, Any],
    start_time: float,
    scale_equations: bool,
    max_paths: Optional[int],
) -> SolutionSet:
    reduced_result = solve(
        reduced_system,
        variables=reduced_variables,
        tol=tol,
        verbose=verbose,
        store_paths=store_paths,
        use_endgame=use_endgame,
        endgame_options=endgame_options,
        tracking_options=tracking_options.copy(),
        deduplication_tol_factor=deduplication_tol_factor,
        singular_deduplication_tol=singular_deduplication_tol,
        allow_underdetermined=allow_underdetermined,
        scale_equations=scale_equations,
        max_paths=max_paths,
        _allow_zero_max_paths=True,
        random_state=rng,
    )

    lifted_solutions = []
    rejected_lift_count = 0
    residual_limit = 100.0 * tol
    base_backward_error_limit = _backward_error_limit(tol)
    for reduced_solution in reduced_result:
        values = dict(assignments)
        values.update(reduced_solution.values)
        ordered_values = _solution_values_from_complete_coordinates(
            values,
            all_variables,
            label="lifted solution",
        )
        residual = _residual_norm(original_system, ordered_values)
        scaled_residual = _scaled_residual_norm(original_system, ordered_values)
        acceptance_scaled_residual = _scaled_residual_norm(
            working_system,
            ordered_values,
        )
        backward_error = _backward_error_norm(original_system, ordered_values)
        is_singular = _is_singular(
            original_system,
            ordered_values,
            tuple(all_variables),
            threshold=1e12,
            rank_tolerance=_rank_tolerance_for_tol(tol),
        )
        acceptance_limit = (
            max(residual_limit, singular_deduplication_tol)
            if is_singular else residual_limit
        )
        backward_error_limit = max(acceptance_limit, base_backward_error_limit)
        if not _solution_quality_within_limits(
            acceptance_scaled_residual,
            backward_error,
            acceptance_limit,
            backward_error_limit,
        ):
            rejected_lift_count += int(getattr(reduced_solution, "multiplicity", 1))
            continue
        lifted = Solution(
            values=ordered_values,
            residual=residual,
            is_singular=is_singular,
            path_index=getattr(reduced_solution, "path_index", None),
        )
        _copy_solution_attributes(reduced_solution, lifted)
        lifted.values = ordered_values
        lifted.residual = residual
        lifted.scaled_residual = scaled_residual
        lifted.backward_error = backward_error
        if not hasattr(lifted, "path_info") or not isinstance(lifted.path_info, dict):
            lifted.path_info = {}
        lifted.path_info["coordinate_reduction"] = {
            "assigned_variables": tuple(variable.name for variable in assignments),
            "lifted": True,
        }
        lifted.path_info["solution_residual"] = float(residual)
        lifted.path_info["scaled_solution_residual"] = float(scaled_residual)
        lifted.path_info["acceptance_scaled_solution_residual"] = float(
            acceptance_scaled_residual
        )
        lifted.path_info["backward_error"] = float(backward_error)
        lifted.path_info["residual_limit"] = float(acceptance_limit)
        lifted.path_info["backward_error_limit"] = float(backward_error_limit)
        _attach_cluster_metadata(lifted)
        lifted_solutions.append(lifted)

    accepted_raw_count = sum(
        int(getattr(solution, "multiplicity", 1))
        for solution in lifted_solutions
    )
    result = SolutionSet(lifted_solutions, original_system)
    result._meta = reduced_result._meta.copy()
    result._meta['solve_time'] = time.time() - start_time
    result._meta['raw_solutions_found'] = accepted_raw_count
    result._meta['generated_gamma'] = generated_gamma
    result._meta['preprocessing'] = preprocessing
    result._meta['equation_scaling'] = equation_scaling
    result._meta['square_up'] = square_up
    result._meta['multiplicity_summary'] = _summarize_solution_multiplicities(
        lifted_solutions
    )
    result._meta['coordinate_reduction'] = {
        "assigned_variables": tuple(variable.name for variable in assignments),
        "assignments": {
            variable.name: _complex_metadata(value)
            for variable, value in assignments.items()
        },
        "reduced_variables": tuple(variable.name for variable in reduced_variables),
        "reduced_equations": len(reduced_system.equations),
        "reduced_meta": reduced_result._meta.copy(),
        "accepted_lift_count": int(accepted_raw_count),
        "rejected_lift_count": int(rejected_lift_count),
    }
    return result


def _reduce_affine_linear_equation(
    system: PolynomialSystem,
    variables: List[Variable],
    tol: float,
) -> Optional[Dict[str, Any]]:
    linear_tol = max(1000.0 * tol, 1e-10)
    for equation_index, equation in enumerate(system.equations):
        parts = _linear_equation_parts(equation, linear_tol)
        if parts is None:
            continue

        constant, coefficients = parts
        if len(coefficients) < 2:
            continue

        eliminated_variable, pivot = max(
            coefficients.items(),
            key=lambda item: abs(item[1]),
        )
        expression = Polynomial([Monomial({}, coefficient=-constant / pivot)])
        for variable, coefficient in coefficients.items():
            if variable == eliminated_variable:
                continue
            expression = expression + Monomial(
                {variable: 1},
                coefficient=-coefficient / pivot,
            )

        reduced_equations = [
            _substitute_polynomial(
                other_equation,
                eliminated_variable,
                expression,
            )
            for index, other_equation in enumerate(system.equations)
            if index != equation_index
        ]
        reduced_variables = [
            variable for variable in variables if variable != eliminated_variable
        ]
        return {
            "system": PolynomialSystem(reduced_equations),
            "variables": reduced_variables,
            "eliminated_variable": eliminated_variable,
            "expression": expression,
            "metadata": {
                "equation_index": equation_index,
                "eliminated_variable": eliminated_variable.name,
                "pivot": _complex_metadata(pivot),
                "expression": repr(expression),
                "reduced_variables": tuple(
                    variable.name for variable in reduced_variables
                ),
                "reduced_equations": len(reduced_equations),
            },
        }
    return None


def _linear_equation_parts(
    polynomial: Polynomial,
    tol: float,
) -> Optional[Tuple[complex, Dict[Variable, complex]]]:
    row_scale = _polynomial_coefficient_scale(polynomial)
    constant = 0.0 + 0.0j
    coefficients: Dict[Variable, complex] = {}
    for term in polynomial.terms:
        coefficient = complex(
            _safe_scaled_coefficient(term.coefficient, row_scale)
        )
        if not term.variables:
            constant += coefficient
            continue
        if term.degree() != 1 or len(term.variables) != 1:
            return None
        variable = next(iter(term.variables))
        coefficients[variable] = (
            coefficients.get(variable, 0.0 + 0.0j)
            + coefficient
        )

    coefficients = {
        variable: coefficient
        for variable, coefficient in coefficients.items()
        if abs(coefficient) > tol
    }
    if not coefficients:
        return None
    return constant, coefficients


def _substitute_polynomial(
    polynomial: Polynomial,
    variable: Variable,
    expression: Polynomial,
) -> Polynomial:
    row_scale = (
        _polynomial_coefficient_scale(polynomial)
        if _requires_row_coefficient_scaling(polynomial)
        else 1.0
    )
    result = Polynomial([0])
    for term in polynomial.terms:
        term_polynomial = Polynomial([
            _safe_scaled_coefficient(term.coefficient, row_scale)
        ])
        for term_variable, exponent in term.variables.items():
            if term_variable == variable:
                factor = expression ** exponent
            else:
                factor = term_variable ** exponent
            term_polynomial = term_polynomial * factor
        result = result + term_polynomial
    return result


def _requires_row_coefficient_scaling(polynomial: Polynomial) -> bool:
    for term in polynomial.terms:
        try:
            coefficient = complex(term.coefficient)
        except (TypeError, ValueError, OverflowError):
            return True
        if not np.isfinite(coefficient):
            return True
    return False


def _solve_reduced_affine_system(
    *,
    original_system: PolynomialSystem,
    working_system: PolynomialSystem,
    reduced_system: PolynomialSystem,
    eliminated_variable: Variable,
    expression: Polynomial,
    reduced_variables: List[Variable],
    all_variables: List[Variable],
    metadata: Dict[str, Any],
    tol: float,
    verbose: bool,
    store_paths: bool,
    use_endgame: bool,
    endgame_options: Optional[Dict[str, Any]],
    tracking_options: Dict[str, Any],
    deduplication_tol_factor: float,
    singular_deduplication_tol: float,
    allow_underdetermined: bool,
    rng: Any,
    generated_gamma: bool,
    preprocessing: Dict[str, Any],
    equation_scaling: Dict[str, Any],
    square_up: Dict[str, Any],
    start_time: float,
    scale_equations: bool,
    max_paths: Optional[int],
) -> SolutionSet:
    reduced_result = solve(
        reduced_system,
        variables=reduced_variables,
        tol=tol,
        verbose=verbose,
        store_paths=store_paths,
        use_endgame=use_endgame,
        endgame_options=endgame_options,
        tracking_options=tracking_options.copy(),
        deduplication_tol_factor=deduplication_tol_factor,
        singular_deduplication_tol=singular_deduplication_tol,
        allow_underdetermined=allow_underdetermined,
        scale_equations=scale_equations,
        max_paths=max_paths,
        _allow_zero_max_paths=True,
        random_state=rng,
    )

    lifted_solutions = []
    rejected_lift_count = 0
    residual_limit = 100.0 * tol
    base_backward_error_limit = _backward_error_limit(tol)
    for reduced_solution in reduced_result:
        reduced_values = dict(reduced_solution.values)
        eliminated_value = expression.evaluate(reduced_values)
        values = dict(reduced_values)
        values[eliminated_variable] = eliminated_value
        ordered_values = _solution_values_from_complete_coordinates(
            values,
            all_variables,
            label="lifted solution",
        )
        residual = _residual_norm(original_system, ordered_values)
        scaled_residual = _scaled_residual_norm(original_system, ordered_values)
        acceptance_scaled_residual = _scaled_residual_norm(
            working_system,
            ordered_values,
        )
        backward_error = _backward_error_norm(original_system, ordered_values)
        is_singular = _is_singular(
            original_system,
            ordered_values,
            tuple(all_variables),
            threshold=1e12,
            rank_tolerance=_rank_tolerance_for_tol(tol),
        )
        acceptance_limit = (
            max(residual_limit, singular_deduplication_tol)
            if is_singular else residual_limit
        )
        backward_error_limit = max(acceptance_limit, base_backward_error_limit)
        if not _solution_quality_within_limits(
            acceptance_scaled_residual,
            backward_error,
            acceptance_limit,
            backward_error_limit,
        ):
            rejected_lift_count += int(getattr(reduced_solution, "multiplicity", 1))
            continue
        lifted = Solution(
            values=ordered_values,
            residual=residual,
            is_singular=is_singular,
            path_index=getattr(reduced_solution, "path_index", None),
        )
        _copy_solution_attributes(reduced_solution, lifted)
        lifted.values = ordered_values
        lifted.residual = residual
        lifted.scaled_residual = scaled_residual
        lifted.backward_error = backward_error
        if not hasattr(lifted, "path_info") or not isinstance(lifted.path_info, dict):
            lifted.path_info = {}
        lifted.path_info["linear_reduction"] = {
            "eliminated_variable": eliminated_variable.name,
            "lifted": True,
        }
        lifted.path_info["solution_residual"] = float(residual)
        lifted.path_info["scaled_solution_residual"] = float(scaled_residual)
        lifted.path_info["acceptance_scaled_solution_residual"] = float(
            acceptance_scaled_residual
        )
        lifted.path_info["backward_error"] = float(backward_error)
        lifted.path_info["residual_limit"] = float(acceptance_limit)
        lifted.path_info["backward_error_limit"] = float(backward_error_limit)
        _attach_cluster_metadata(lifted)
        lifted_solutions.append(lifted)

    accepted_raw_count = sum(
        int(getattr(solution, "multiplicity", 1))
        for solution in lifted_solutions
    )
    result = SolutionSet(lifted_solutions, original_system)
    result._meta = reduced_result._meta.copy()
    result._meta['solve_time'] = time.time() - start_time
    result._meta['raw_solutions_found'] = accepted_raw_count
    result._meta['generated_gamma'] = generated_gamma
    result._meta['preprocessing'] = preprocessing
    result._meta['equation_scaling'] = equation_scaling
    result._meta['square_up'] = square_up
    result._meta['multiplicity_summary'] = _summarize_solution_multiplicities(
        lifted_solutions
    )
    result._meta['linear_reduction'] = {
        **metadata,
        "reduced_meta": reduced_result._meta.copy(),
        "accepted_lift_count": int(accepted_raw_count),
        "rejected_lift_count": int(rejected_lift_count),
    }
    return result


def _deduplicate_solutions(
    raw_solutions: List[Solution],
    system: PolynomialSystem,
    variables: List[Variable],
    *,
    regular_tolerance: float,
    singular_tolerance: float,
    rank_tolerance: float,
    polish_tolerance: float,
) -> List[Solution]:
    if not raw_solutions:
        return []
    variables = tuple(_normalize_coordinate_variables(variables))

    parent = list(range(len(raw_solutions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, solution in enumerate(raw_solutions):
        for other_index in range(index):
            other = raw_solutions[other_index]
            tolerance = _deduplication_tolerance_for_pair(
                solution,
                other,
                regular_tolerance=regular_tolerance,
                singular_tolerance=singular_tolerance,
            )
            if solution.distance(other, variables) < tolerance:
                union(index, other_index)

    clusters_by_root: Dict[int, List[Solution]] = {}
    for index, solution in enumerate(raw_solutions):
        clusters_by_root.setdefault(find(index), []).append(solution)
    clusters = list(clusters_by_root.values())

    return [
        _merge_solution_cluster(
            cluster,
            system,
            variables,
            rank_tolerance=rank_tolerance,
            polish_tolerance=polish_tolerance,
        )
        for cluster in clusters
    ]


def _deduplication_tolerance_for_pair(
    left: Solution,
    right: Solution,
    *,
    regular_tolerance: float,
    singular_tolerance: float,
) -> float:
    return (
        singular_tolerance
        if left.is_singular and right.is_singular
        else regular_tolerance
    )


def _merge_solution_cluster(
    cluster: List[Solution],
    system: PolynomialSystem,
    variables: List[Variable],
    *,
    rank_tolerance: float,
    polish_tolerance: float,
) -> Solution:
    variables = tuple(_normalize_coordinate_variables(variables))
    representative = min(
        cluster,
        key=lambda solution: _solution_quality_key(system, solution),
    )
    multiplicity = sum(getattr(solution, "multiplicity", 1) for solution in cluster)
    path_indices = tuple(
        sorted({
            path_index
            for solution in cluster
            for path_index in getattr(solution, "path_indices", ())
            if path_index is not None
        })
    )
    root_indices = tuple(
        sorted({
            root_index
            for solution in cluster
            for root_index in getattr(solution, "root_indices", ())
            if root_index is not None
        })
    )

    if len(cluster) > 1:
        candidate_points = np.vstack([
            _solution_point_from_values(
                solution.values,
                variables,
                label="solution",
            )
            for solution in cluster
        ])
    else:
        candidate_points = np.empty((len(cluster), len(variables)), dtype=complex)
    if len(cluster) > 1 and np.all(np.isfinite(candidate_points)):
        centroid = np.mean(candidate_points, axis=0)
        centroid, centroid_residual, centroid_polish = _polish_endpoint_against_system(
            system,
            centroid,
            variables,
            polish_tolerance,
        )
        centroid_values = {
            var: value for var, value in zip(variables, centroid)
        }
        centroid_quality = _solution_values_quality_key(
            system,
            centroid_values,
            centroid_residual,
        )
        if (
            np.isfinite(centroid_residual)
            and centroid_quality <= _solution_quality_key(system, representative)
        ):
            representative.values = {
                var: value for var, value in zip(variables, centroid)
            }
            representative.residual = centroid_residual
            representative.scaled_residual = centroid_quality[3]
            representative.backward_error = centroid_quality[4]
            if hasattr(representative, "path_info"):
                representative.path_info["cluster_centroid_polish"] = centroid_polish

    representative.multiplicity = int(multiplicity)
    representative.path_indices = path_indices
    representative.root_indices = root_indices
    representative.path_index = path_indices[0] if path_indices else representative.path_index
    representative.is_singular = bool(
        any(solution.is_singular for solution in cluster)
        or _is_singular(
            system,
            representative.values,
            tuple(variables),
            threshold=1e12,
            rank_tolerance=rank_tolerance,
        )
    )
    representative.cluster_radius = _cluster_radius(representative, cluster, variables)
    _attach_cluster_metadata(representative)
    return representative


def _cluster_radius(
    representative: Solution,
    cluster: List[Solution],
    variables: List[Variable],
) -> float:
    distances = [representative.distance(solution, variables) for solution in cluster]
    finite_distances = [distance for distance in distances if np.isfinite(distance)]
    return max(finite_distances, default=0.0)


def _attach_cluster_metadata(solution: Solution) -> None:
    if not hasattr(solution, "path_info") or not isinstance(solution.path_info, dict):
        solution.path_info = {}
    solution.path_info["cluster"] = {
        "multiplicity": int(getattr(solution, "multiplicity", 1)),
        "path_indices": tuple(getattr(solution, "path_indices", ())),
        "radius": float(getattr(solution, "cluster_radius", 0.0)),
    }
    root_indices = tuple(getattr(solution, "root_indices", ()))
    if root_indices:
        solution.path_info["cluster"]["root_indices"] = root_indices


def _summarize_solution_multiplicities(
    solutions: List[Solution],
) -> Dict[str, Any]:
    multiplicities = tuple(
        int(getattr(solution, "multiplicity", 1)) for solution in solutions
    )
    return {
        "distinct_solutions": len(solutions),
        "total_multiplicity": sum(multiplicities),
        "max_multiplicity": max(multiplicities, default=0),
        "multiple_root_count": sum(value > 1 for value in multiplicities),
    }


def refine_solution(
    system: PolynomialSystem,
    solution: Any,
    variables: Optional[List[Variable]] = None,
    *,
    tol: float = 1e-12,
    max_iters: int = 50,
    singularity_threshold: float = 1e12,
    keep_failed: bool = True,
) -> Solution:
    """Newton-polish one candidate solution against a polynomial system.

    Args:
        system: Polynomial system to refine against.
        solution: Candidate coordinate vector, coordinate mapping, or object
            with a ``values`` mapping.
        variables: Optional variable order. Defaults to system variable order.
        tol: Newton residual/step tolerance.
        max_iters: Maximum Newton iterations.
        singularity_threshold: Condition threshold used to mark singularity.
        keep_failed: If True, keep the original candidate when Newton fails or
            worsens the residual.

    Returns:
        A new :class:`Solution` with refinement metadata attached.
    """
    _validate_polynomial_system("system", system)
    tol = _validate_positive_finite_float("tol", tol)
    max_iters = _validate_positive_integer_option("max_iters", max_iters)
    singularity_threshold = _validate_positive_finite_float(
        "singularity_threshold", singularity_threshold
    )
    keep_failed = _validate_boolean_option("keep_failed", keep_failed)

    ordered_variables = tuple(_ordered_variables(system, variables))
    _validate_variables_cover_system(system, list(ordered_variables))
    initial_point = _coerce_point_for_variables(
        solution,
        list(ordered_variables),
        "solution",
        allow_nonfinite=True,
    )
    initial_values = {
        var: value for var, value in zip(ordered_variables, initial_point)
    }
    initial_residual = _residual_norm(system, initial_values)
    initial_scaled_residual = _scaled_residual_norm(system, initial_values)
    initial_backward_error = _backward_error_norm(system, initial_values)

    corrected, success, iterations = newton_corrector(
        system,
        initial_point,
        list(ordered_variables),
        max_iters=max_iters,
        tol=tol,
    )
    refined_values = {
        var: value for var, value in zip(ordered_variables, corrected)
    }
    refined_residual = _residual_norm(system, refined_values)
    refined_scaled_residual = _scaled_residual_norm(system, refined_values)
    refined_backward_error = _backward_error_norm(system, refined_values)

    use_refined = (
        np.isfinite(refined_residual)
        and np.isfinite(refined_scaled_residual)
        and np.isfinite(refined_backward_error)
        and (
            refined_scaled_residual <= tol
            or refined_backward_error <= _backward_error_limit(tol)
            or refined_scaled_residual < initial_scaled_residual
            or refined_backward_error < initial_backward_error
        )
    )
    if keep_failed and (not success or not use_refined):
        final_values = (
            dict(solution.values)
            if isinstance(solution, Solution)
            else dict(initial_values)
        )
        final_residual = initial_residual
        final_scaled_residual = initial_scaled_residual
        final_backward_error = initial_backward_error
        accepted = False
    else:
        final_values = refined_values
        final_residual = refined_residual
        final_scaled_residual = refined_scaled_residual
        final_backward_error = refined_backward_error
        accepted = use_refined

    refined_solution = Solution(
        values=final_values,
        residual=final_residual,
        is_singular=_is_singular(
            system,
            final_values,
            ordered_variables,
            singularity_threshold,
            rank_tolerance=_rank_tolerance_for_tol(tol),
        ),
        path_index=getattr(solution, "path_index", None),
    )
    _copy_solution_attributes(solution, refined_solution)
    refined_solution.scaled_residual = final_scaled_residual
    refined_solution.backward_error = final_backward_error
    refined_solution.refinement = {
        "success": bool(accepted),
        "newton_success": bool(success),
        "accepted": bool(accepted),
        "iterations": int(iterations),
        "initial_residual": float(initial_residual),
        "initial_scaled_residual": float(initial_scaled_residual),
        "initial_backward_error": float(initial_backward_error),
        "final_residual": float(final_residual),
        "final_scaled_residual": float(final_scaled_residual),
        "final_backward_error": float(final_backward_error),
        "tol": tol,
        "max_iters": max_iters,
    }
    return refined_solution


def refine_solutions(
    solutions: Any,
    system: Optional[PolynomialSystem] = None,
    variables: Optional[List[Variable]] = None,
    *,
    tol: float = 1e-12,
    max_iters: int = 50,
    singularity_threshold: float = 1e12,
    keep_failed: bool = True,
) -> SolutionSet:
    """Newton-polish a solution set or iterable of solutions.

    Args:
        solutions: A :class:`SolutionSet` or iterable of coordinate records.
        system: Polynomial system. Optional when ``solutions`` has ``system``.
        variables: Optional variable order.
        tol: Newton residual/step tolerance.
        max_iters: Maximum Newton iterations per solution.
        singularity_threshold: Condition threshold used to mark singularity.
        keep_failed: Keep original candidates if Newton fails or worsens them.

    Returns:
        A new :class:`SolutionSet` containing polished candidates.
    """
    inferred_system = system if system is not None else getattr(solutions, "system", None)
    if inferred_system is None:
        raise ValueError("system is required when refining a bare solution iterable")
    _validate_polynomial_system("system", inferred_system)
    tol = _validate_positive_finite_float("tol", tol)
    max_iters = _validate_positive_integer_option("max_iters", max_iters)
    singularity_threshold = _validate_positive_finite_float(
        "singularity_threshold", singularity_threshold
    )
    keep_failed = _validate_boolean_option("keep_failed", keep_failed)
    if variables is not None:
        variables = _normalize_variable_list(variables)
        _validate_variables_cover_system(inferred_system, variables)

    source_solutions = getattr(solutions, "solutions", solutions)
    if isinstance(source_solutions, (str, bytes)):
        raise TypeError("solutions must be an iterable of coordinate records")
    try:
        solution_list = list(source_solutions)
    except TypeError as exc:
        raise TypeError(
            "solutions must be an iterable of coordinate records"
        ) from exc

    refined = [
        refine_solution(
            inferred_system,
            solution,
            variables=variables,
            tol=tol,
            max_iters=max_iters,
            singularity_threshold=singularity_threshold,
            keep_failed=keep_failed,
        )
        for solution in solution_list
    ]

    result = SolutionSet(refined, inferred_system)
    result._meta = getattr(solutions, "_meta", {}).copy()
    result._meta["is_refined"] = True
    result._meta["refinement"] = {
        "tol": tol,
        "max_iters": max_iters,
        "singularity_threshold": singularity_threshold,
        "accepted_count": sum(
            bool(getattr(solution, "refinement", {}).get("accepted", False))
            for solution in refined
        ),
        "success_count": sum(
            bool(getattr(solution, "refinement", {}).get("success", False))
            for solution in refined
        ),
    }
    return result


def polyvar(*names: str) -> Union[Variable, Tuple[Variable, ...]]:
    """Create polynomial variables with the given names.

    Args:
        *names: Variable names

    Returns:
        A single Variable or a tuple of Variables
    """
    return _polyvar(*names)


def _validate_tracking_options(
    tracking_options: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if tracking_options is None:
        return {}
    if not isinstance(tracking_options, dict):
        raise TypeError("tracking_options must be a dictionary")

    allowed = {
        "min_step_size",
        "max_step_size",
        "max_newton_iters",
        "max_steps",
        "max_predictor_norm",
        "gamma",
        "endgame_start",
        "singularity_threshold",
        "final_singularity_threshold",
        "predictor",
        "n_jobs",
    }
    unknown = sorted(set(tracking_options) - allowed)
    if unknown:
        raise ValueError(
            "Unknown tracking option(s): " + ", ".join(unknown)
        )

    validated_options = tracking_options.copy()
    for option in ("max_newton_iters", "max_steps"):
        if option in validated_options:
            validated_options[option] = _validate_tracking_integer_option(
                option,
                validated_options[option],
            )
    if "n_jobs" in validated_options:
        n_jobs = validated_options["n_jobs"]
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral):
            raise TypeError("n_jobs must be an integer")
        if n_jobs <= 0:
            raise ValueError("n_jobs must be positive")
        validated_options["n_jobs"] = int(n_jobs)
    if "predictor" in validated_options:
        predictor = validated_options["predictor"]
        if not isinstance(predictor, str):
            raise TypeError("predictor must be a string")
        normalized = predictor.lower()
        if normalized not in {"euler", "heun", "rk4"}:
            raise ValueError("predictor must be 'euler', 'heun', or 'rk4'")
        validated_options["predictor"] = normalized
    if "gamma" in validated_options:
        validated_options["gamma"] = _validate_gamma(validated_options["gamma"])

    float_options = {
        "min_step_size",
        "max_step_size",
        "max_predictor_norm",
        "endgame_start",
        "singularity_threshold",
        "final_singularity_threshold",
    }
    for option in sorted(float_options.intersection(validated_options)):
        validated_options[option] = _validate_tracking_float_option(
            option,
            validated_options[option],
            allow_infinite=option == "max_predictor_norm",
        )

    min_step_size = validated_options.get("min_step_size", 1e-6)
    max_step_size = validated_options.get("max_step_size", 0.05)
    max_predictor_norm = validated_options.get("max_predictor_norm", float("inf"))
    endgame_start = validated_options.get("endgame_start", 0.1)
    singularity_threshold = validated_options.get("singularity_threshold", 1e3)
    final_singularity_threshold = validated_options.get(
        "final_singularity_threshold",
        1e8,
    )
    _validate_tracking_float_ranges(
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        max_predictor_norm=max_predictor_norm,
        endgame_start=endgame_start,
        singularity_threshold=singularity_threshold,
        final_singularity_threshold=final_singularity_threshold,
    )

    return validated_options


def _validate_tracking_integer_option(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _validate_tracking_float_option(
    name: str,
    value: Any,
    *,
    allow_infinite: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if np.isnan(numeric_value):
        if name == "max_predictor_norm":
            raise ValueError("max_predictor_norm cannot be NaN")
        raise ValueError(f"{name} must be finite")
    if not allow_infinite and not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def _validate_tracking_float_ranges(
    *,
    min_step_size: float,
    max_step_size: float,
    max_predictor_norm: float,
    endgame_start: float,
    singularity_threshold: float,
    final_singularity_threshold: float,
) -> None:
    if min_step_size <= 0:
        raise ValueError("min_step_size must be positive")
    if max_step_size <= 0:
        raise ValueError("max_step_size must be positive")
    if min_step_size > max_step_size:
        raise ValueError("min_step_size cannot exceed max_step_size")
    if max_predictor_norm <= 0:
        raise ValueError("max_predictor_norm must be positive")
    if not 0 <= endgame_start <= 1:
        raise ValueError("endgame_start must be between 0 and 1")
    if singularity_threshold <= 0:
        raise ValueError("singularity_threshold must be positive")
    if final_singularity_threshold <= 0:
        raise ValueError("final_singularity_threshold must be positive")


def _validate_boolean_option(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _validate_solution_values(values: Any) -> Dict[Variable, complex]:
    if not isinstance(values, dict):
        raise TypeError(
            "values must be a dictionary mapping Variable objects to numeric coordinates"
        )

    validated: Dict[Variable, complex] = {}
    for index, (variable, value) in enumerate(values.items()):
        if not isinstance(variable, Variable):
            raise TypeError(f"values key {index} must be a Variable")
        validated[variable] = _validate_solution_coordinate(
            f"values[{variable.name}]",
            value,
        )
    return validated


def _validate_solution_coordinate(name: str, value: Any) -> complex:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
        raise TypeError(f"{name} must be a numeric coordinate")
    try:
        return complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a numeric coordinate") from exc


def _validate_solution_residual(residual: Any) -> float:
    if isinstance(residual, (bool, np.bool_)) or not isinstance(residual, Number):
        raise TypeError("residual must be a real number")
    try:
        numeric_residual = float(residual)
    except (TypeError, ValueError) as exc:
        raise TypeError("residual must be a real number") from exc
    if numeric_residual < 0:
        raise ValueError("residual must be nonnegative")
    return numeric_residual


def _validate_solution_path_index(path_index: Optional[Any]) -> Optional[int]:
    if path_index is None:
        return None
    if isinstance(path_index, (bool, np.bool_)) or not isinstance(path_index, Integral):
        raise TypeError("path_index must be an integer or None")
    if path_index < 0:
        raise ValueError("path_index must be nonnegative")
    return int(path_index)


def _validate_optional_boolean_filter(name: str, value: Optional[bool]) -> Optional[bool]:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean or None")
    return value


def _validate_positive_finite_float(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return numeric_value


def _validate_nonnegative_finite_float(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return numeric_value


def _validate_positive_integer_option(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _validate_polynomial_system(name: str, system: Any) -> None:
    if not isinstance(system, PolynomialSystem):
        raise TypeError(f"{name} must be a PolynomialSystem")


def _coerce_parseable_polynomial_system(
    name: str,
    system: Any,
    *,
    variables: Optional[List[Variable]] = None,
) -> Any:
    if isinstance(system, str):
        return PolynomialSystem.parse(
            system,
            variables=_parse_variable_map(variables),
        )
    if (
        isinstance(system, (list, tuple))
        and system
        and all(isinstance(equation, str) for equation in system)
    ):
        return PolynomialSystem.parse(
            system,
            variables=_parse_variable_map(variables),
        )
    if isinstance(system, (Polynomial, Monomial, Variable)):
        return PolynomialSystem([system])
    if _is_numeric_system_equation(system):
        return PolynomialSystem([system])
    return system


def _is_numeric_system_equation(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, (bool, np.bool_))


def _parse_variable_map(
    variables: Optional[List[Variable]],
) -> Optional[Dict[str, Variable]]:
    if variables is None:
        return None
    variable_map: Dict[str, Variable] = {}
    duplicates = []
    for index, variable in enumerate(variables):
        if not isinstance(variable, Variable):
            raise TypeError(f"variables[{index}] must be a Variable")
        if variable.name in variable_map:
            duplicates.append(variable.name)
        variable_map[variable.name] = variable
    if duplicates:
        raise ValueError(
            "Variable list contains duplicate variable(s): "
            + ", ".join(sorted(set(duplicates)))
        )
    return variable_map


def _normalize_variable_list(variables: Any) -> List[Variable]:
    try:
        normalized = list(variables)
    except TypeError as exc:
        raise TypeError(
            "variables must be an iterable of Variable objects"
        ) from exc
    return normalized


def _validate_max_paths(
    max_paths: Optional[int],
    *,
    allow_zero: bool = False,
) -> Optional[int]:
    if max_paths is None:
        return None
    if isinstance(max_paths, bool) or not isinstance(max_paths, Integral):
        raise TypeError("max_paths must be an integer or None")
    if max_paths < 0 or (max_paths == 0 and not allow_zero):
        raise ValueError("max_paths must be positive")
    return int(max_paths)


def _total_degree_path_count(degrees: List[int]) -> int:
    total = 1
    for degree in degrees:
        total *= int(degree)
    return total


def _check_path_limit(
    path_count: int,
    max_paths: Optional[int],
    source: str,
) -> None:
    if max_paths is None:
        return
    if path_count > max_paths:
        raise ValueError(
            f"{source} would track {path_count} path(s), exceeding "
            f"max_paths={max_paths}"
        )


def _remaining_path_limit(
    max_paths: Optional[int],
    used_paths: int,
) -> Optional[int]:
    if max_paths is None:
        return None
    return max(int(max_paths) - int(used_paths), 0)


def _failed_path_retry_options(tracker_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    retry_kwargs = tracker_kwargs.copy()
    current_max_step = float(retry_kwargs.get("max_step_size", 0.05))
    retry_kwargs["max_step_size"] = max(current_max_step / 2.0, 1e-8)
    retry_kwargs["max_newton_iters"] = max(
        int(retry_kwargs.get("max_newton_iters", 10)),
        20,
    )
    retry_kwargs["max_steps"] = max(
        int(retry_kwargs.get("max_steps", 10000)),
        20000,
    )
    if "max_predictor_norm" in retry_kwargs:
        predictor_norm = float(retry_kwargs["max_predictor_norm"])
        if np.isfinite(predictor_norm):
            retry_kwargs["max_predictor_norm"] = max(predictor_norm / 2.0, 1e-12)
    else:
        retry_kwargs["max_predictor_norm"] = 0.1
    retry_kwargs.setdefault("predictor", "rk4")
    return retry_kwargs


def _endgame_options_with_random_state(
    endgame_options: Optional[Dict[str, Any]],
    rng: Any,
    include_random_state: bool,
) -> Optional[Dict[str, Any]]:
    if endgame_options is None and not include_random_state:
        return None

    options = {} if endgame_options is None else dict(endgame_options)
    if include_random_state:
        options.setdefault("random_state", rng)
    return options


def _random_unit_complex(rng: Any) -> complex:
    return _start_random_unit_complex(rng, context="gamma")


def _ordered_variables(
    system: PolynomialSystem, variables: Optional[List[Variable]]
) -> Tuple[Variable, ...]:
    if variables is None:
        return tuple(system.ordered_variables())
    return tuple(_normalize_variable_list(variables))


def _solution_set_variables(
    solution_set: SolutionSet,
    variables: Optional[List[Variable]],
) -> Tuple[Variable, ...]:
    if variables is not None:
        return _normalize_coordinate_variables(variables)

    ordered_variables = tuple(solution_set.system.ordered_variables())
    if ordered_variables or not solution_set.solutions:
        return ordered_variables

    inferred = set()
    for solution in solution_set.solutions:
        inferred.update(solution.values)
    variables = tuple(
        sorted(inferred, key=lambda variable: getattr(variable, "name", repr(variable)))
    )
    _validate_coordinate_variables(variables)
    return variables


def _normalize_coordinate_variables(variables: Any) -> Tuple[Variable, ...]:
    normalized = tuple(_normalize_variable_list(variables))
    _validate_coordinate_variables(normalized)
    return normalized


def _validate_coordinate_variables(variables: Tuple[Variable, ...]) -> None:
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


def _solution_point_from_values(
    values: Any,
    variables: Tuple[Variable, ...],
    *,
    label: str,
) -> np.ndarray:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must expose a values dictionary")
    missing = []
    coordinates = []
    for variable in variables:
        found, coordinate = _mapping_coordinate_for_variable(
            values,
            variable,
            label,
        )
        if found:
            coordinates.append(coordinate)
        else:
            missing.append(variable.name)
    if missing:
        raise ValueError(
            f"{label} is missing coordinate(s): "
            + ", ".join(sorted(missing))
        )
    try:
        return np.array(coordinates, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} coordinate(s) must be numeric") from exc


def _solution_values_from_complete_coordinates(
    values: Any,
    variables: Any,
    *,
    label: str,
) -> Dict[Variable, complex]:
    ordered_variables = _normalize_coordinate_variables(variables)
    point = _solution_point_from_values(
        values,
        ordered_variables,
        label=label,
    )
    return {
        variable: value
        for variable, value in zip(ordered_variables, point)
    }


def _residual_norm(
    system: PolynomialSystem, values: Dict[Variable, complex]
) -> float:
    try:
        with np.errstate(over="raise", invalid="raise"):
            residuals = np.array(system.evaluate(values), dtype=complex)
    except (OverflowError, FloatingPointError):
        return _raw_residual_overflow_norm(system, values)
    if residuals.size == 0:
        return 0.0
    if not np.all(np.isfinite(residuals)):
        return _raw_residual_overflow_norm(system, values)
    return _scaled_euclidean_norm(residuals)


def _raw_residual_overflow_norm(
    system: PolynomialSystem,
    values: Dict[Variable, complex],
) -> float:
    try:
        scaled_residual = _scaled_residual_norm(system, values)
    except (OverflowError, FloatingPointError, TypeError, ValueError):
        return float("inf")
    if scaled_residual != 0.0:
        return float("inf")
    if all(
        _scaled_evaluation_is_exact_zero(equation, values)
        for equation in system.equations
    ):
        return 0.0
    return float("inf")


def _scaled_residual_norm(
    system: PolynomialSystem, values: Dict[Variable, complex]
) -> float:
    variables = tuple(system.ordered_variables())
    point = _solution_point_from_values(values, variables, label="solution")
    residuals = evaluate_scaled_system_at_point(system, point, variables)
    if residuals.size == 0:
        return 0.0
    if not np.all(np.isfinite(residuals)):
        return float("inf")
    return _scaled_euclidean_norm(residuals)


def _backward_error_norm(
    system: PolynomialSystem, values: Dict[Variable, complex]
) -> float:
    variables = tuple(system.ordered_variables())
    point = _solution_point_from_values(values, variables, label="solution")
    errors = evaluate_backward_error_at_point(system, point, variables)
    if errors.size == 0:
        return 0.0
    if not np.all(np.isfinite(errors)):
        return float("inf")
    return _scaled_euclidean_norm(errors)


def _is_singular(
    system: PolynomialSystem,
    values: Dict[Variable, complex],
    variables: Tuple[Variable, ...],
    threshold: float,
    rank_tolerance: Optional[float] = None,
) -> bool:
    variables = _normalize_coordinate_variables(variables)
    if not variables:
        return False
    point = _solution_point_from_values(values, variables, label="solution")
    if not np.all(np.isfinite(point)):
        return True
    jacobian = evaluate_scaled_jacobian_at_point(system, point, list(variables))
    if jacobian.size == 0:
        return False
    if not np.all(np.isfinite(jacobian)):
        return True
    rank = _matrix_rank(jacobian, rank_tolerance)
    if rank < min(jacobian.shape):
        return True
    try:
        condition = float(np.linalg.cond(jacobian))
    except (np.linalg.LinAlgError, ValueError, OverflowError, FloatingPointError):
        return True
    if not np.isfinite(condition):
        return True
    return bool(condition > threshold)


def _matrix_rank(matrix: np.ndarray, tolerance: Optional[float] = None) -> int:
    if matrix.size == 0 or not np.all(np.isfinite(matrix)):
        return 0
    try:
        if tolerance is None:
            return int(np.linalg.matrix_rank(matrix))
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    except (np.linalg.LinAlgError, ValueError, OverflowError, FloatingPointError):
        return 0
    if not np.all(np.isfinite(singular_values)):
        return 0
    return int(np.sum(singular_values > tolerance))


def _rank_tolerance_for_tol(tol: float) -> float:
    return max(1000.0 * min(float(tol), 1e-10), np.sqrt(np.finfo(float).eps))


def _backward_error_limit(tol: float) -> float:
    return max(100.0 * float(tol), 1000.0 * float(np.finfo(float).eps))


def _solution_quality_within_limits(
    scaled_residual: float,
    backward_error: float,
    residual_limit: float,
    backward_error_limit: float,
) -> bool:
    return (
        (
            np.isfinite(scaled_residual)
            and scaled_residual <= residual_limit
        )
        or (
            np.isfinite(backward_error)
            and backward_error <= backward_error_limit
        )
    )


def _copy_solution_attributes(source: Solution, target: Solution) -> None:
    target.path_points = getattr(source, "path_points", None)
    target.winding_number = getattr(source, "winding_number", None)
    target.path_info = getattr(source, "path_info", None)
    target.multiplicity = getattr(source, "multiplicity", 1)
    target.path_indices = getattr(source, "path_indices", ())
    target.root_indices = getattr(source, "root_indices", ())
    target.cluster_radius = getattr(source, "cluster_radius", 0.0)
    target.scaled_residual = getattr(source, "scaled_residual", None)
    target.backward_error = getattr(source, "backward_error", None)


def _preprocess_system(
    system: PolynomialSystem,
    tol: float,
    *,
    remove_duplicate_equations: bool,
) -> Tuple[PolynomialSystem, Dict[str, Any]]:
    constant_tol = float(tol)
    duplicate_tol = 1e-12
    retained_equations = []
    inconsistent_constants = []
    removed_zero_equations = 0
    duplicate_equations = []
    seen_by_support: Dict[
        Tuple[Tuple[Tuple[str, int], ...], ...],
        List[Dict[str, Any]],
    ] = {}

    for index, equation in enumerate(system.equations):
        if equation.variables():
            if remove_duplicate_equations:
                coefficient_map = _polynomial_coefficient_map(
                    equation,
                    zero_tol=duplicate_tol,
                )
                if coefficient_map is not None:
                    support = tuple(sorted(coefficient_map))
                    duplicate = _find_scalar_multiple_duplicate(
                        coefficient_map,
                        seen_by_support.get(support, ()),
                        tolerance=duplicate_tol,
                    )
                    if duplicate is not None:
                        duplicate_equations.append({
                            "index": index,
                            "duplicate_of": duplicate["index"],
                            "scale": _complex_metadata(duplicate["scale"]),
                        })
                        continue
                    seen_by_support.setdefault(support, []).append({
                        "index": index,
                        "coefficients": coefficient_map,
                    })
            retained_equations.append(equation)
            continue

        constant_value = equation.evaluate({})
        if _constant_abs_within_tolerance(constant_value, constant_tol):
            removed_zero_equations += 1
        else:
            inconsistent_constants.append({
                "index": index,
                "value": _complex_metadata(constant_value),
            })

    return PolynomialSystem(retained_equations), {
        "removed_zero_equations": removed_zero_equations,
        "duplicate_removal_enabled": bool(remove_duplicate_equations),
        "removed_duplicate_equations": len(duplicate_equations),
        "duplicate_equations": tuple(duplicate_equations),
        "duplicate_tolerance": duplicate_tol if remove_duplicate_equations else None,
        "inconsistent_constants": tuple(inconsistent_constants),
        "constant_tolerance": constant_tol,
    }


def _constant_abs_within_tolerance(value: Number, tolerance: float) -> bool:
    try:
        return abs(value) <= tolerance
    except (TypeError, ValueError, OverflowError):
        return False


def _polynomial_coefficient_map(
    polynomial: Polynomial,
    *,
    zero_tol: float,
) -> Optional[Dict[Tuple[Tuple[str, int], ...], Number]]:
    coefficients: Dict[Tuple[Tuple[str, int], ...], Number] = {}
    for term in polynomial.terms:
        coefficient = _duplicate_detection_coefficient(term.coefficient)
        if coefficient is None:
            return None
        key = tuple(
            sorted(
                (
                    (variable.name, int(exponent))
                    for variable, exponent in term.variables.items()
                ),
                key=lambda item: item[0],
            )
        )
        try:
            coefficients[key] = coefficients.get(key, 0) + coefficient
        except OverflowError:
            return None

    coefficients = {
        key: coefficient
        for key, coefficient in coefficients.items()
        if _coefficient_abs_exceeds_tolerance(coefficient, zero_tol)
    }
    if not coefficients:
        return None
    return coefficients


def _duplicate_detection_coefficient(value: Any) -> Optional[Number]:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, Integral):
        return int(value)
    try:
        coefficient = complex(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(coefficient):
        return None
    return coefficient


def _coefficient_abs_exceeds_tolerance(value: Number, tolerance: float) -> bool:
    try:
        return abs(value) > tolerance
    except (TypeError, OverflowError):
        return False


def _coefficient_sort_magnitude(value: Number) -> Number:
    try:
        return abs(value)
    except (TypeError, OverflowError):
        return 0


def _find_scalar_multiple_duplicate(
    coefficients: Dict[Tuple[Tuple[str, int], ...], Number],
    candidates: Any,
    *,
    tolerance: float,
) -> Optional[Dict[str, Any]]:
    for candidate in candidates:
        candidate_coefficients = candidate["coefficients"]
        if set(coefficients) != set(candidate_coefficients):
            continue
        anchor_key = max(
            candidate_coefficients,
            key=lambda key: (
                _coefficient_sort_magnitude(candidate_coefficients[key]),
                repr(key),
            ),
        )
        anchor = candidate_coefficients[anchor_key]
        if not _coefficient_abs_exceeds_tolerance(anchor, tolerance):
            continue
        scale = _duplicate_detection_scale(coefficients[anchor_key], anchor)
        if scale is None:
            continue
        if _coefficients_match_scaled(
            coefficients,
            candidate_coefficients,
            scale,
            tolerance=tolerance,
        ):
            return {"index": candidate["index"], "scale": scale}
    return None


def _duplicate_detection_scale(left: Number, right: Number) -> Optional[Number]:
    try:
        if isinstance(left, Integral) and isinstance(right, Integral):
            scale = Fraction(int(left), int(right))
        else:
            scale = complex(left) / complex(right)
    except (ZeroDivisionError, TypeError, ValueError, OverflowError):
        return None
    if not _complex_metadata_representable(scale):
        return None
    return scale


def _complex_metadata_representable(value: Number) -> bool:
    try:
        value = complex(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(value))


def _coefficients_match_scaled(
    left: Dict[Tuple[Tuple[str, int], ...], Number],
    right: Dict[Tuple[Tuple[str, int], ...], Number],
    scale: Number,
    *,
    tolerance: float,
) -> bool:
    if not _complex_metadata_representable(scale):
        return False
    for key, left_value in left.items():
        try:
            right_value = scale * right[key]
            difference = left_value - right_value
        except (TypeError, ValueError, OverflowError):
            return False
        if difference == 0:
            continue
        try:
            allowed_error = tolerance * max(
                1.0,
                abs(left_value),
                abs(right_value),
            )
            error = abs(difference)
        except (TypeError, ValueError, OverflowError):
            return False
        if error > allowed_error:
            return False
    return True


def _scale_equation_system(
    system: PolynomialSystem,
    *,
    enabled: bool,
) -> Tuple[PolynomialSystem, Dict[str, Any]]:
    exact_coefficient_scales = tuple(
        _polynomial_coefficient_scale(equation)
        for equation in system.equations
    )
    coefficient_scales = tuple(
        _metadata_float(scale) for scale in exact_coefficient_scales
    )
    metadata: Dict[str, Any] = {
        "enabled": bool(enabled),
        "method": "coefficient_max_norm" if enabled else "none",
        "equation_count": len(system.equations),
        "coefficient_scales": coefficient_scales,
        "max_coefficient_scale": max(coefficient_scales, default=1.0),
        "scaled_equation_count": 0,
        "skipped_destructive_scaling_count": 0,
    }
    if not enabled:
        return system, metadata

    scaled_equations = []
    scaled_count = 0
    skipped_count = 0
    for equation, scale in zip(system.equations, exact_coefficient_scales):
        if scale != 1.0:
            if _scaling_would_drop_nonzero_coefficient(equation, scale):
                scaled_equations.append(equation)
                skipped_count += 1
                continue
            scaled_equations.append(_scale_polynomial_coefficients(equation, scale))
            scaled_count += 1
        else:
            scaled_equations.append(equation)

    metadata["scaled_equation_count"] = scaled_count
    metadata["skipped_destructive_scaling_count"] = skipped_count
    return PolynomialSystem(scaled_equations), metadata


def _scale_polynomial_coefficients(polynomial: Polynomial, scale: Number) -> Polynomial:
    scaled = Polynomial.__new__(Polynomial)
    scaled.terms = [
        Monomial(
            term.variables.copy(),
            coefficient=_safe_scaled_coefficient(term.coefficient, scale),
        )
        for term in polynomial.terms
    ]
    return scaled


def _scaling_would_drop_nonzero_coefficient(
    polynomial: Polynomial,
    scale: Number,
) -> bool:
    for term in polynomial.terms:
        coefficient = term.coefficient
        if coefficient == 0:
            continue
        try:
            scaled_coefficient = _safe_scaled_coefficient(coefficient, scale)
        except (OverflowError, FloatingPointError):
            return True
        if scaled_coefficient == 0:
            return True
        try:
            scaled_complex = complex(scaled_coefficient)
        except (OverflowError, TypeError, ValueError):
            return True
        if not np.isfinite(scaled_complex.real) or not np.isfinite(scaled_complex.imag):
            return True
    return False


def _metadata_float(value: Any) -> float:
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return float("inf")
    return result if np.isfinite(result) else float("inf")


def _compact_path_info(path_info: Dict[str, Any]) -> Dict[str, Any]:
    scalar_keys = (
        "success",
        "singular",
        "endgame_used",
        "polished",
        "steps",
        "newton_iters",
        "start_t",
        "end_t",
        "direction",
        "tol",
        "min_step_size",
        "max_step_size",
        "initial_step_size",
        "max_newton_iters",
        "max_steps",
        "max_predictor_norm",
        "gamma",
        "endgame_start",
        "singularity_threshold",
        "final_singularity_threshold",
        "store_paths",
        "use_endgame",
        "step_reductions",
        "max_observed_predictor_norm",
        "max_predictor_correction_norm",
        "predictor",
        "predictor_fallbacks",
        "final_t",
        "final_residual",
        "start_residual",
        "start_scaled_residual",
        "start_residual_limit",
        "failure_reason",
        "winding_number",
    )
    compact = {
        key: _metadata_value(path_info[key])
        for key in scalar_keys
        if key in path_info
    }
    polish = path_info.get("polish")
    if isinstance(polish, dict):
        compact["polish"] = {
            key: _metadata_value(value)
            for key, value in polish.items()
            if key != "predictions"
        }
    solution_polish = path_info.get("solution_polish")
    if isinstance(solution_polish, dict):
        compact["solution_polish"] = {
            key: _metadata_value(value)
            for key, value in solution_polish.items()
        }
    return compact


def _polish_endpoint_against_system(
    system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
    tol: float,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    initial_point = np.array(point, dtype=complex)
    initial_values = {var: val for var, val in zip(variables, initial_point)}
    initial_residual = _residual_norm(system, initial_values)
    initial_scaled_residual = _scaled_residual_norm(system, initial_values)
    initial_backward_error = _backward_error_norm(system, initial_values)
    metadata: Dict[str, Any] = {
        "attempted": False,
        "success": False,
        "accepted": False,
        "iterations": 0,
        "initial_residual": float(initial_residual),
        "initial_scaled_residual": float(initial_scaled_residual),
        "initial_backward_error": float(initial_backward_error),
        "final_residual": float(initial_residual),
        "final_scaled_residual": float(initial_scaled_residual),
        "final_backward_error": float(initial_backward_error),
        "tol": _endpoint_polish_tolerance(tol),
    }
    if not np.all(np.isfinite(initial_point)):
        return initial_point, initial_residual, metadata

    corrected, success, iterations = newton_corrector(
        system,
        initial_point,
        variables,
        max_iters=8,
        tol=metadata["tol"],
    )
    refined_values = {var: val for var, val in zip(variables, corrected)}
    refined_residual = _residual_norm(system, refined_values)
    refined_scaled_residual = _scaled_residual_norm(system, refined_values)
    refined_backward_error = _backward_error_norm(system, refined_values)
    accepted = (
        np.all(np.isfinite(corrected))
        and np.isfinite(refined_residual)
        and np.isfinite(refined_scaled_residual)
        and np.isfinite(refined_backward_error)
        and (
            not np.isfinite(initial_scaled_residual)
            or refined_scaled_residual < initial_scaled_residual
            or refined_backward_error < initial_backward_error
        )
    )
    final_point = corrected if accepted else initial_point
    final_residual = refined_residual if accepted else initial_residual
    final_scaled_residual = (
        refined_scaled_residual if accepted else initial_scaled_residual
    )
    final_backward_error = (
        refined_backward_error if accepted else initial_backward_error
    )

    metadata.update({
        "attempted": True,
        "success": bool(success),
        "accepted": bool(accepted),
        "iterations": int(iterations),
        "candidate_residual": float(refined_residual),
        "candidate_scaled_residual": float(refined_scaled_residual),
        "candidate_backward_error": float(refined_backward_error),
        "final_residual": float(final_residual),
        "final_scaled_residual": float(final_scaled_residual),
        "final_backward_error": float(final_backward_error),
    })
    return final_point, final_residual, metadata


def _endpoint_polish_tolerance(tol: float) -> float:
    return max(float(np.finfo(float).eps), min(tol * 1e-6, 1e-14))


def _summarize_path_results(
    path_results: List[Dict[str, Any]],
    residual_rejections: List[Dict[str, Any]],
    accepted_count: int,
) -> Dict[str, Any]:
    failure_reasons: Counter = Counter()
    tracked_successful = 0
    endgame_paths = 0
    polished_paths = 0
    polish_attempted = 0
    polish_failed = 0
    max_final_residual = 0.0
    max_steps = 0
    max_newton_iters = 0
    max_step_reductions = 0
    max_predictor_norm = 0.0
    max_predictor_correction_norm = 0.0
    predictor_fallbacks = 0
    predictors = Counter()

    for path_info in path_results:
        predictor = path_info.get("predictor")
        if predictor:
            predictors[str(predictor)] += 1

        if path_info.get("success", False):
            tracked_successful += 1
        else:
            failure_reasons[path_info.get("failure_reason") or "unspecified"] += 1

        if path_info.get("endgame_used", False):
            endgame_paths += 1
        if path_info.get("polished", False):
            polished_paths += 1

        polish = path_info.get("polish")
        if isinstance(polish, dict):
            polish_attempted += 1
            if not polish.get("success", False):
                polish_failed += 1

        final_residual = _finite_float_or_inf(path_info.get("final_residual", 0.0))
        max_final_residual = max(max_final_residual, final_residual)
        max_steps = max(max_steps, int(path_info.get("steps", 0) or 0))
        max_newton_iters = max(
            max_newton_iters, int(path_info.get("newton_iters", 0) or 0)
        )
        max_step_reductions = max(
            max_step_reductions, int(path_info.get("step_reductions", 0) or 0)
        )
        max_predictor_norm = max(
            max_predictor_norm,
            _finite_float_or_inf(path_info.get("max_observed_predictor_norm", 0.0)),
        )
        max_predictor_correction_norm = max(
            max_predictor_correction_norm,
            _finite_float_or_inf(path_info.get("max_predictor_correction_norm", 0.0)),
        )
        predictor_fallbacks += int(path_info.get("predictor_fallbacks", 0) or 0)

    rejected = tuple(
        {
            "path_index": int(item["path_index"]),
            "residual": float(item["residual"]),
            "scaled_residual": float(item.get("scaled_residual", item["residual"])),
            "residual_limit": float(item["residual_limit"]),
        }
        for item in residual_rejections
    )

    return {
        "total_paths": len(path_results),
        "tracked_successful_paths": tracked_successful,
        "tracker_failed_paths": len(path_results) - tracked_successful,
        "accepted_paths": accepted_count,
        "residual_rejected_paths": len(residual_rejections),
        "tracker_failure_reasons": dict(sorted(failure_reasons.items())),
        "residual_rejections": rejected,
        "endgame_paths": endgame_paths,
        "polished_paths": polished_paths,
        "polish_attempted_paths": polish_attempted,
        "polish_failed_paths": polish_failed,
        "max_final_residual": max_final_residual,
        "max_steps": max_steps,
        "max_newton_iters": max_newton_iters,
        "max_step_reductions": max_step_reductions,
        "max_observed_predictor_norm": max_predictor_norm,
        "max_predictor_correction_norm": max_predictor_correction_norm,
        "predictor_fallbacks": predictor_fallbacks,
        "predictors": dict(sorted(predictors.items())),
    }


def _metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _metadata_value(value.tolist())
    if isinstance(value, np.generic):
        return _metadata_value(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {
            _metadata_key(key): _metadata_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    return value


def _metadata_key(key: Any) -> Any:
    if isinstance(key, np.generic):
        key = key.item()
    if isinstance(key, (str, int, float, bool)) or key is None:
        return key
    return str(key)


def _finite_float_or_inf(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return result if np.isfinite(result) else float("inf")


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _solution_scaled_residual_or_inf(
    system: PolynomialSystem,
    solution: Solution,
) -> float:
    cached = _finite_float_or_none(getattr(solution, "scaled_residual", None))
    if cached is not None:
        return cached
    try:
        return _scaled_residual_norm(system, solution.values)
    except (TypeError, ValueError, OverflowError, FloatingPointError, KeyError):
        return float("inf")


def _solution_backward_error_or_inf(
    system: PolynomialSystem,
    solution: Solution,
) -> float:
    cached = _finite_float_or_none(getattr(solution, "backward_error", None))
    if cached is not None:
        return cached
    try:
        return _backward_error_norm(system, solution.values)
    except (TypeError, ValueError, OverflowError, FloatingPointError, KeyError):
        return float("inf")


def _solution_quality_key(
    system: PolynomialSystem,
    solution: Solution,
) -> Tuple[float, float, float, float, float]:
    return _solution_values_quality_key(
        system,
        solution.values,
        _finite_float_or_inf(getattr(solution, "residual", None)),
        scaled_residual=_solution_scaled_residual_or_inf(system, solution),
        backward_error=_solution_backward_error_or_inf(system, solution),
    )


def _solution_values_quality_key(
    system: PolynomialSystem,
    values: Dict[Variable, complex],
    residual: Any,
    *,
    scaled_residual: Optional[float] = None,
    backward_error: Optional[float] = None,
) -> Tuple[float, float, float, float, float]:
    raw_residual = _finite_float_or_inf(residual)
    if scaled_residual is None:
        try:
            scaled_residual = _scaled_residual_norm(system, values)
        except (TypeError, ValueError, OverflowError, FloatingPointError, KeyError):
            scaled_residual = float("inf")
    else:
        scaled_residual = _finite_float_or_inf(scaled_residual)
    if backward_error is None:
        try:
            backward_error = _backward_error_norm(system, values)
        except (TypeError, ValueError, OverflowError, FloatingPointError, KeyError):
            backward_error = float("inf")
    else:
        backward_error = _finite_float_or_inf(backward_error)
    best_normalized = min(scaled_residual, backward_error)
    worst_normalized = max(scaled_residual, backward_error)
    return (
        best_normalized,
        worst_normalized,
        raw_residual,
        scaled_residual,
        backward_error,
    )


def _count_successful_path_results(path_results: List[Dict[str, Any]]) -> int:
    return sum(1 for path_info in path_results if path_info.get("success", False))


def _coerce_custom_start_solution(
    solution: Any,
    variables: Tuple[Variable, ...],
    index: int,
) -> np.ndarray:
    label = f"Start solution {index}"
    if isinstance(solution, Solution):
        return _solution_point_from_values(
            getattr(solution, "values", None),
            variables,
            label=label,
        )
    if isinstance(solution, Mapping):
        return _solution_point_from_values(
            solution,
            variables,
            label=label,
        )

    try:
        point = np.asarray(solution, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a numeric point") from exc
    if point.ndim != 1 or point.size != len(variables):
        raise ValueError(
            "Each start solution must be a one-dimensional point with "
            f"{len(variables)} coordinate(s); solution {index} has "
            f"shape {point.shape}"
        )
    return point


def _validate_custom_start_data(
    start_system: Any,
    start_solutions: Any,
    target_system: PolynomialSystem,
    variables: List[Variable],
    tol: float,
    *,
    scale_equations: bool,
) -> Tuple[PolynomialSystem, List[np.ndarray], Dict[str, Any]]:
    _validate_polynomial_system("start_system", start_system)
    _validate_variables_cover_system(start_system, variables)

    if len(start_system.equations) != len(target_system.equations):
        raise ValueError(
            "Custom start system must have the same number of equations as "
            "the tracked target system"
        )

    tracking_start_system, start_equation_scaling = _scale_equation_system(
        start_system,
        enabled=scale_equations,
    )

    try:
        raw_solutions = list(start_solutions)
    except TypeError as exc:
        raise TypeError("start_solutions must be an iterable of points") from exc
    if not raw_solutions:
        raise ValueError(
            "start_solutions must contain at least one start point for a "
            "custom start system"
        )

    ordered_variables = _normalize_coordinate_variables(variables)
    start_tol = max(1000.0 * tol, 1e-8)
    validated_solutions: List[np.ndarray] = []
    residuals: List[float] = []
    scaled_residuals: List[float] = []

    for index, solution in enumerate(raw_solutions):
        point = _coerce_custom_start_solution(solution, ordered_variables, index)
        if not np.all(np.isfinite(point)):
            raise ValueError(f"Start solution {index} contains nonfinite values")

        values = {var: value for var, value in zip(ordered_variables, point)}
        residual = _residual_norm(tracking_start_system, values)
        scaled_residual = _scaled_residual_norm(tracking_start_system, values)
        if scaled_residual > start_tol:
            raise ValueError(
                f"Start solution {index} does not satisfy the custom start "
                "system within tolerance "
                f"({residual:.2e} raw, {scaled_residual:.2e} scaled > "
                f"{start_tol:.2e})"
            )
        validated_solutions.append(point)
        residuals.append(residual)
        scaled_residuals.append(scaled_residual)

    return tracking_start_system, validated_solutions, {
        "source": "custom",
        "path_count": len(validated_solutions),
        "start_solution_tolerance": start_tol,
        "max_start_residual": max(residuals, default=0.0),
        "max_start_scaled_residual": max(scaled_residuals, default=0.0),
        "equation_scaling": start_equation_scaling,
    }


def _square_up_system(
    system: PolynomialSystem,
    variables: List[Variable],
    rng: Any,
) -> Tuple[PolynomialSystem, Dict[str, Any]]:
    n_eqs = len(system.equations)
    n_vars = len(variables)
    if n_eqs <= n_vars:
        return system, _square_up_metadata(system, variables, method="none")
    if n_vars == 0:
        raise ValueError(
            "Cannot square up an overdetermined system with no variables"
        )

    coefficients = _random_complex_matrix(rng, n_vars, n_eqs)
    source_scales = tuple(
        _polynomial_coefficient_scale(equation)
        if _requires_row_coefficient_scaling(equation)
        else 1.0
        for equation in system.equations
    )
    source_equations = tuple(
        _scale_polynomial_coefficients(equation, scale)
        if scale != 1.0 else equation
        for equation, scale in zip(system.equations, source_scales)
    )
    equations = []
    for row in coefficients:
        combined = Polynomial([0])
        for coefficient, equation in zip(row, source_equations):
            combined = combined + complex(coefficient) * equation
        equations.append(combined)

    squared_system = PolynomialSystem(equations)
    return squared_system, _square_up_metadata(
        system,
        variables,
        method="random_linear_combinations",
        squared_equations=n_vars,
        coefficients=coefficients,
        source_equation_scales=source_scales,
    )


def _square_up_metadata(
    system: PolynomialSystem,
    variables: List[Variable],
    *,
    method: str,
    squared_equations: Optional[int] = None,
    coefficients: Optional[np.ndarray] = None,
    source_equation_scales: Optional[Tuple[Number, ...]] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "method": method,
        "original_equations": len(system.equations),
        "variables": tuple(variable.name for variable in variables),
    }
    if squared_equations is not None:
        metadata["squared_equations"] = squared_equations
    if coefficients is not None:
        metadata["coefficients"] = tuple(
            tuple(_complex_metadata(value) for value in row)
            for row in coefficients
        )
    if source_equation_scales is not None:
        metadata["source_equation_scales"] = tuple(
            _metadata_float(scale) for scale in source_equation_scales
        )
    return metadata


def _random_complex_matrix(rng: Any, rows: int, cols: int) -> np.ndarray:
    real = _rng_standard_normal(
        rng,
        (rows, cols),
        context="square-up matrix real part",
    )
    imag = _rng_standard_normal(
        rng,
        (rows, cols),
        context="square-up matrix imaginary part",
    )
    return real + 1j * imag


def _complex_metadata(value: complex) -> Dict[str, float]:
    try:
        complex_value = complex(value)
    except (TypeError, ValueError, OverflowError):
        if isinstance(value, Real):
            return {"real": _signed_metadata_float(value), "imag": 0.0}
        return {"real": float("nan"), "imag": float("nan")}
    return {
        "real": _signed_metadata_float(complex_value.real),
        "imag": _signed_metadata_float(complex_value.imag),
    }


def _signed_metadata_float(value: Any) -> float:
    try:
        result = float(value)
    except OverflowError:
        try:
            return float("-inf") if value < 0 else float("inf")
        except (TypeError, ValueError):
            return float("inf")
    except (TypeError, ValueError):
        return float("nan")
    return result


def _validate_variables_cover_system(
    system: PolynomialSystem,
    variables: List[Variable],
    *,
    allow_extra: bool = False,
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

    system_variables = system.variables()
    missing = sorted(
        (variable.name for variable in system_variables if variable not in seen)
    )
    if missing:
        raise ValueError(
            "Variable list is missing system variable(s): " + ", ".join(missing)
        )

    extra = sorted(
        variable.name for variable in seen if variable not in system_variables
    )
    if system_variables and extra and not allow_extra:
        raise ValueError(
            "Variable list contains variable(s) not used by system: "
            + ", ".join(extra)
        )
