"""Solution diagnostics for polynomial systems.

The solver returns numerical candidate roots. This module provides a
structured audit layer for residuals, Jacobian rank, conditioning, realness,
and duplicate clustering so downstream code can decide whether a solve is
good enough for its tolerance budget.
"""

from collections import Counter
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pycontinuum.polynomial import PolynomialSystem, Variable
from pycontinuum.utils import (
    _coerce_point_for_variables,
    evaluate_backward_error_at_point,
    evaluate_jacobian_at_point,
    evaluate_system_at_point,
    evaluate_scaled_system_at_point,
    evaluate_scaled_jacobian_at_point,
    solve_linear_system,
    _scaled_euclidean_norm,
    _strict_json_value,
)


@dataclass(frozen=True)
class SolutionDiagnostics:
    """Numerical diagnostics for one candidate solution."""

    index: int
    variables: Tuple[Variable, ...]
    point: Tuple[complex, ...]
    residuals: Tuple[complex, ...]
    scaled_residuals: Tuple[complex, ...]
    backward_errors: Tuple[float, ...]
    residual_norm: float
    scaled_residual_norm: float
    backward_error_norm: float
    max_residual: float
    max_scaled_residual: float
    max_backward_error: float
    jacobian_rank: int
    jacobian_shape: Tuple[int, int]
    condition_number: float
    newton_step_norm: float
    newton_error_bound: float
    is_finite: bool
    is_real: bool
    is_valid: bool
    is_scaled_valid: bool
    is_backward_stable: bool
    is_rank_deficient: bool
    is_well_conditioned: bool
    is_square_system: bool
    is_overdetermined: bool
    is_underdetermined: bool
    is_regular: bool
    is_isolated: bool
    is_certified_regular: bool
    is_newton_certified: bool
    regularity_status: str
    certification_status: str
    cluster_id: int
    nearest_neighbor_distance: float

    def as_dict(self, *, strict_json: bool = False) -> Dict[str, Any]:
        """Return a JSON-friendly representation of the diagnostics."""
        strict_json = _validate_boolean_option("strict_json", strict_json)
        result = {
            "index": self.index,
            "variables": [var.name for var in self.variables],
            "point": [_complex_as_dict(value) for value in self.point],
            "residuals": [_complex_as_dict(value) for value in self.residuals],
            "scaled_residuals": [
                _complex_as_dict(value) for value in self.scaled_residuals
            ],
            "backward_errors": list(self.backward_errors),
            "residual_norm": self.residual_norm,
            "scaled_residual_norm": self.scaled_residual_norm,
            "backward_error_norm": self.backward_error_norm,
            "max_residual": self.max_residual,
            "max_scaled_residual": self.max_scaled_residual,
            "max_backward_error": self.max_backward_error,
            "jacobian_rank": self.jacobian_rank,
            "jacobian_shape": self.jacobian_shape,
            "condition_number": self.condition_number,
            "newton_step_norm": self.newton_step_norm,
            "newton_error_bound": self.newton_error_bound,
            "is_finite": self.is_finite,
            "is_real": self.is_real,
            "is_valid": self.is_valid,
            "is_scaled_valid": self.is_scaled_valid,
            "is_backward_stable": self.is_backward_stable,
            "is_rank_deficient": self.is_rank_deficient,
            "is_well_conditioned": self.is_well_conditioned,
            "is_square_system": self.is_square_system,
            "is_overdetermined": self.is_overdetermined,
            "is_underdetermined": self.is_underdetermined,
            "is_regular": self.is_regular,
            "is_isolated": self.is_isolated,
            "is_certified_regular": self.is_certified_regular,
            "is_newton_certified": self.is_newton_certified,
            "regularity_status": self.regularity_status,
            "certification_status": self.certification_status,
            "cluster_id": self.cluster_id,
            "nearest_neighbor_distance": self.nearest_neighbor_distance,
        }
        if strict_json:
            return _strict_json_value(result)
        return result


@dataclass(frozen=True)
class SolutionAudit:
    """Aggregate diagnostics for a solution set."""

    diagnostics: Tuple[SolutionDiagnostics, ...]
    variables: Tuple[Variable, ...]
    tolerance: float
    duplicate_tolerance: float
    real_tolerance: float
    condition_threshold: float
    certificate_tolerance: float

    @property
    def count(self) -> int:
        """Number of candidate solutions audited."""
        return len(self.diagnostics)

    @property
    def valid_count(self) -> int:
        """Number of solutions with finite residual norm below tolerance."""
        return sum(item.is_valid for item in self.diagnostics)

    @property
    def invalid_count(self) -> int:
        """Number of solutions failing the residual/finite-value check."""
        return self.count - self.valid_count

    @property
    def scaled_valid_count(self) -> int:
        """Number of solutions with finite coefficient-scaled residuals."""
        return sum(item.is_scaled_valid for item in self.diagnostics)

    @property
    def scaled_invalid_count(self) -> int:
        """Number of candidates failing the scaled residual check."""
        return self.count - self.scaled_valid_count

    @property
    def backward_stable_count(self) -> int:
        """Number of candidates passing normwise backward-error checks."""
        return sum(item.is_backward_stable for item in self.diagnostics)

    @property
    def backward_unstable_count(self) -> int:
        """Number of candidates failing normwise backward-error checks."""
        return self.count - self.backward_stable_count

    @property
    def real_count(self) -> int:
        """Number of solutions whose coordinates are real within tolerance."""
        return sum(item.is_real for item in self.diagnostics)

    @property
    def rank_deficient_count(self) -> int:
        """Number of solutions with rank-deficient Jacobians."""
        return sum(item.is_rank_deficient for item in self.diagnostics)

    @property
    def well_conditioned_count(self) -> int:
        """Number of isolated solutions passing the conditioning threshold."""
        return sum(item.is_well_conditioned for item in self.diagnostics)

    @property
    def isolated_count(self) -> int:
        """Number of candidates with full-column-rank Jacobians."""
        return sum(item.is_isolated for item in self.diagnostics)

    @property
    def certified_regular_count(self) -> int:
        """Number of candidates passing the numerical regularity certificate."""
        return sum(item.is_certified_regular for item in self.diagnostics)

    @property
    def newton_certified_count(self) -> int:
        """Number of candidates with separated Newton error balls."""
        return sum(item.is_newton_certified for item in self.diagnostics)

    @property
    def all_valid(self) -> bool:
        """Whether every candidate passes the residual/finite-value check."""
        return all(item.is_valid for item in self.diagnostics)

    @property
    def all_scaled_valid(self) -> bool:
        """Whether every candidate passes coefficient-scaled residual checks."""
        return all(item.is_scaled_valid for item in self.diagnostics)

    @property
    def all_backward_stable(self) -> bool:
        """Whether every candidate passes normwise backward-error checks."""
        return all(item.is_backward_stable for item in self.diagnostics)

    @property
    def all_newton_certified(self) -> bool:
        """Whether every candidate passes the Newton certificate."""
        return all(item.is_newton_certified for item in self.diagnostics)

    @property
    def max_residual(self) -> float:
        """Largest residual norm among all candidates."""
        if not self.diagnostics:
            return 0.0
        return max(item.residual_norm for item in self.diagnostics)

    @property
    def max_scaled_residual(self) -> float:
        """Largest coefficient-scaled residual norm among all candidates."""
        if not self.diagnostics:
            return 0.0
        return max(item.scaled_residual_norm for item in self.diagnostics)

    @property
    def max_backward_error(self) -> float:
        """Largest normwise backward error among all candidates."""
        if not self.diagnostics:
            return 0.0
        return max(item.backward_error_norm for item in self.diagnostics)

    @property
    def worst_condition_number(self) -> float:
        """Largest finite or infinite Jacobian condition number."""
        if not self.diagnostics:
            return 0.0
        return max(item.condition_number for item in self.diagnostics)

    @property
    def max_newton_error_bound(self) -> float:
        """Largest Newton error estimate among all candidates."""
        if not self.diagnostics:
            return 0.0
        return max(item.newton_error_bound for item in self.diagnostics)

    @property
    def cluster_count(self) -> int:
        """Number of duplicate clusters after tolerance-based grouping."""
        return len({item.cluster_id for item in self.diagnostics})

    @property
    def duplicate_count(self) -> int:
        """Number of candidates beyond the first representative per cluster."""
        return self.count - self.cluster_count

    @property
    def failed_indices(self) -> Tuple[int, ...]:
        """Indices of candidates that failed the residual/finite-value check."""
        return tuple(item.index for item in self.diagnostics if not item.is_valid)

    def duplicate_clusters(self) -> Dict[int, Tuple[int, ...]]:
        """Return clusters containing more than one candidate index."""
        clusters: Dict[int, List[int]] = {}
        for item in self.diagnostics:
            clusters.setdefault(item.cluster_id, []).append(item.index)
        return {
            cluster_id: tuple(indices)
            for cluster_id, indices in clusters.items()
            if len(indices) > 1
        }

    def regularity_status_counts(self) -> Dict[str, int]:
        """Return candidate counts grouped by regularity status."""
        return dict(Counter(item.regularity_status for item in self.diagnostics))

    def certification_status_counts(self) -> Dict[str, int]:
        """Return candidate counts grouped by certification status."""
        return dict(Counter(item.certification_status for item in self.diagnostics))

    def summary(self, *, strict_json: bool = False) -> Dict[str, Any]:
        """Return compact aggregate diagnostics."""
        strict_json = _validate_boolean_option("strict_json", strict_json)
        result = {
            "count": self.count,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "scaled_valid_count": self.scaled_valid_count,
            "scaled_invalid_count": self.scaled_invalid_count,
            "backward_stable_count": self.backward_stable_count,
            "backward_unstable_count": self.backward_unstable_count,
            "real_count": self.real_count,
            "rank_deficient_count": self.rank_deficient_count,
            "well_conditioned_count": self.well_conditioned_count,
            "isolated_count": self.isolated_count,
            "certified_regular_count": self.certified_regular_count,
            "newton_certified_count": self.newton_certified_count,
            "all_valid": self.all_valid,
            "all_scaled_valid": self.all_scaled_valid,
            "all_backward_stable": self.all_backward_stable,
            "all_newton_certified": self.all_newton_certified,
            "max_residual": self.max_residual,
            "max_scaled_residual": self.max_scaled_residual,
            "max_backward_error": self.max_backward_error,
            "max_newton_error_bound": self.max_newton_error_bound,
            "worst_condition_number": self.worst_condition_number,
            "cluster_count": self.cluster_count,
            "duplicate_count": self.duplicate_count,
            "failed_indices": self.failed_indices,
            "regularity_status_counts": self.regularity_status_counts(),
            "certification_status_counts": self.certification_status_counts(),
            "tolerance": self.tolerance,
            "duplicate_tolerance": self.duplicate_tolerance,
            "real_tolerance": self.real_tolerance,
            "condition_threshold": self.condition_threshold,
            "certificate_tolerance": self.certificate_tolerance,
        }
        if strict_json:
            return _strict_json_value(result)
        return result

    def as_dict(self, *, strict_json: bool = False) -> Dict[str, Any]:
        """Return a JSON-friendly representation of the full audit."""
        strict_json = _validate_boolean_option("strict_json", strict_json)
        result = self.summary()
        result["variables"] = [var.name for var in self.variables]
        result["diagnostics"] = [item.as_dict() for item in self.diagnostics]
        result["duplicate_clusters"] = self.duplicate_clusters()
        if strict_json:
            return _strict_json_value(result)
        return result


def diagnose_solution(
    system: PolynomialSystem,
    solution: Any,
    variables: Optional[Sequence[Variable]] = None,
    *,
    tolerance: float = 1e-8,
    real_tolerance: Optional[float] = None,
    condition_threshold: float = 1e12,
    certificate_tolerance: Optional[float] = None,
    index: int = 0,
    cluster_id: int = 0,
    nearest_neighbor_distance: float = float("inf"),
) -> SolutionDiagnostics:
    """Compute residual and Jacobian diagnostics for one solution.

    Args:
        system: Polynomial system used for residual/Jacobian evaluation.
        solution: Object with a ``values`` mapping from variables to values.
        variables: Optional variable order. Defaults to the system order.
        tolerance: Residual norm threshold for ``is_valid``.
        real_tolerance: Imaginary-part threshold for ``is_real``.
        condition_threshold: Maximum condition number for isolated roots marked
            ``is_well_conditioned``.
        certificate_tolerance: Maximum Newton correction norm for the
            lightweight regularity certificate. Defaults to
            ``max(10 * tolerance, 1e-12)``.
        index: Index to report in the resulting diagnostics.
        cluster_id: Duplicate cluster identifier to attach to this solution.
        nearest_neighbor_distance: Distance to the nearest other solution.

    Returns:
        A :class:`SolutionDiagnostics` instance.
    """
    _validate_polynomial_system("system", system)
    (
        tolerance,
        real_tolerance,
        condition_threshold,
        certificate_tolerance,
    ) = _validate_thresholds(
        tolerance, real_tolerance, condition_threshold, certificate_tolerance
    )
    ordered_variables = _coerce_variables(system, variables)
    real_tol = tolerance if real_tolerance is None else real_tolerance
    cert_tol = (
        max(10.0 * tolerance, 1e-12)
        if certificate_tolerance is None
        else certificate_tolerance
    )

    point = _solution_point(solution, ordered_variables)
    values = {var: value for var, value in zip(ordered_variables, point)}
    residuals_array = evaluate_system_at_point(
        system, point, list(ordered_variables)
    )
    scaled_residuals_array = evaluate_scaled_system_at_point(
        system, point, list(ordered_variables)
    )
    backward_errors_array = evaluate_backward_error_at_point(
        system, point, list(ordered_variables)
    )
    jacobian = evaluate_jacobian_at_point(system, point, list(ordered_variables))
    scaled_jacobian = evaluate_scaled_jacobian_at_point(
        system, point, list(ordered_variables)
    )

    is_raw_residual_finite = bool(np.all(np.isfinite(residuals_array)))
    is_raw_jacobian_finite = bool(np.all(np.isfinite(jacobian)))
    is_finite = bool(
        np.all(np.isfinite(point))
        and is_raw_residual_finite
        and np.all(np.isfinite(scaled_residuals_array))
        and np.all(np.isfinite(backward_errors_array))
        and np.all(np.isfinite(scaled_jacobian))
    )
    is_scaled_finite = bool(
        np.all(np.isfinite(point))
        and np.all(np.isfinite(scaled_residuals_array))
        and np.all(np.isfinite(scaled_jacobian))
    )
    is_backward_error_finite = bool(
        np.all(np.isfinite(point))
        and np.all(np.isfinite(backward_errors_array))
    )
    is_scaled_jacobian_finite = bool(np.all(np.isfinite(scaled_jacobian)))
    residual_norm = _norm(residuals_array)
    scaled_residual_norm = _norm(scaled_residuals_array)
    backward_error_norm = _norm(backward_errors_array)
    max_residual = _max_abs(residuals_array)
    max_scaled_residual = _max_abs(scaled_residuals_array)
    max_backward_error = _max_abs(backward_errors_array)
    rank_tolerance = _rank_tolerance_for_residual_tolerance(tolerance)
    jacobian_rank = _matrix_rank(
        scaled_jacobian,
        is_scaled_jacobian_finite,
        rank_tolerance,
    )
    condition_number = _condition_number(
        scaled_jacobian,
        is_scaled_jacobian_finite,
        rank_tolerance,
    )
    if is_raw_jacobian_finite:
        newton_jacobian = jacobian
        newton_residuals = residuals_array
        newton_inputs_finite = is_finite
    else:
        newton_jacobian = scaled_jacobian
        newton_residuals = scaled_residuals_array
        newton_inputs_finite = is_scaled_finite
    newton_step_norm = _newton_step_norm(
        newton_jacobian,
        newton_residuals,
        newton_inputs_finite,
    )
    newton_error_bound = _newton_error_bound(newton_step_norm)
    n_equations, n_variables = jacobian.shape
    is_square_system = n_equations == n_variables
    is_overdetermined = n_equations > n_variables
    is_underdetermined = n_equations < n_variables
    is_rank_deficient = jacobian_rank < min(jacobian.shape, default=0)
    is_real = bool(np.all(np.abs(np.imag(point)) <= real_tol))
    is_scaled_valid = is_scaled_finite and scaled_residual_norm <= tolerance
    is_valid = (
        is_finite
        and residual_norm <= tolerance
        and is_scaled_valid
    )
    is_backward_stable = (
        is_backward_error_finite and backward_error_norm <= tolerance
    )
    is_regular = (
        is_scaled_jacobian_finite
        and np.all(np.isfinite(point))
        and jacobian_rank == n_variables
    )
    is_isolated = is_regular and not is_underdetermined
    is_well_conditioned = is_isolated and condition_number <= condition_threshold
    is_certified_regular = (
        is_valid
        and is_isolated
        and is_well_conditioned
        and newton_step_norm <= cert_tol
    )
    is_newton_certified = (
        is_certified_regular
        and newton_error_bound <= 2.0 * cert_tol
        and _newton_ball_is_separated(
            newton_error_bound,
            nearest_neighbor_distance,
        )
    )
    regularity_status = _regularity_status(
        is_finite=is_finite,
        is_valid=is_valid,
        is_isolated=is_isolated,
        is_well_conditioned=is_well_conditioned,
        newton_step_norm=newton_step_norm,
        certificate_tolerance=cert_tol,
        is_underdetermined=is_underdetermined,
    )
    certification_status = _certification_status(
        regularity_status=regularity_status,
        is_certified_regular=is_certified_regular,
        is_newton_certified=is_newton_certified,
        newton_error_bound=newton_error_bound,
        certificate_tolerance=cert_tol,
        nearest_neighbor_distance=nearest_neighbor_distance,
    )

    return SolutionDiagnostics(
        index=index,
        variables=ordered_variables,
        point=tuple(complex(value) for value in point),
        residuals=tuple(complex(value) for value in residuals_array),
        scaled_residuals=tuple(complex(value) for value in scaled_residuals_array),
        backward_errors=tuple(float(value) for value in backward_errors_array),
        residual_norm=residual_norm,
        scaled_residual_norm=scaled_residual_norm,
        backward_error_norm=backward_error_norm,
        max_residual=max_residual,
        max_scaled_residual=max_scaled_residual,
        max_backward_error=max_backward_error,
        jacobian_rank=jacobian_rank,
        jacobian_shape=tuple(jacobian.shape),
        condition_number=condition_number,
        newton_step_norm=newton_step_norm,
        newton_error_bound=newton_error_bound,
        is_finite=is_finite,
        is_real=is_real,
        is_valid=is_valid,
        is_scaled_valid=is_scaled_valid,
        is_backward_stable=is_backward_stable,
        is_rank_deficient=is_rank_deficient,
        is_well_conditioned=is_well_conditioned,
        is_square_system=is_square_system,
        is_overdetermined=is_overdetermined,
        is_underdetermined=is_underdetermined,
        is_regular=is_regular,
        is_isolated=is_isolated,
        is_certified_regular=is_certified_regular,
        is_newton_certified=is_newton_certified,
        regularity_status=regularity_status,
        certification_status=certification_status,
        cluster_id=cluster_id,
        nearest_neighbor_distance=nearest_neighbor_distance,
    )


def diagnose_solutions(
    solutions: Any,
    system: Optional[PolynomialSystem] = None,
    variables: Optional[Sequence[Variable]] = None,
    *,
    tolerance: float = 1e-8,
    duplicate_tolerance: Optional[float] = None,
    real_tolerance: Optional[float] = None,
    condition_threshold: float = 1e12,
    certificate_tolerance: Optional[float] = None,
) -> SolutionAudit:
    """Audit a solution set or iterable of solution objects.

    Args:
        solutions: A ``SolutionSet``-like object or iterable of solutions.
        system: Polynomial system. Optional when ``solutions`` has ``system``.
        variables: Optional variable order. Defaults to the system order.
        tolerance: Residual norm threshold for validity.
        duplicate_tolerance: Euclidean distance threshold for duplicate clusters.
        real_tolerance: Imaginary-part threshold for real classification.
        condition_threshold: Maximum condition number for well-conditioned
            isolated roots.
        certificate_tolerance: Maximum Newton correction norm for the
            lightweight regularity certificate.

    Returns:
        A :class:`SolutionAudit` instance with per-solution and aggregate metrics.
    """
    (
        tolerance,
        real_tolerance,
        condition_threshold,
        certificate_tolerance,
    ) = _validate_thresholds(
        tolerance, real_tolerance, condition_threshold, certificate_tolerance
    )
    if duplicate_tolerance is None:
        duplicate_tolerance = max(10.0 * tolerance, 1e-12)
    else:
        duplicate_tolerance = _validate_nonnegative_finite_float(
            "duplicate_tolerance", duplicate_tolerance
        )
    cert_tol = (
        max(10.0 * tolerance, 1e-12)
        if certificate_tolerance is None
        else certificate_tolerance
    )

    inferred_system = (
        system if system is not None else getattr(solutions, "system", None)
    )
    if inferred_system is None:
        raise ValueError("system is required when auditing a bare solution iterable")
    _validate_polynomial_system("system", inferred_system)

    ordered_variables = _coerce_variables(inferred_system, variables)
    solution_list = list(getattr(solutions, "solutions", solutions))
    points = [
        _solution_point(solution, ordered_variables) for solution in solution_list
    ]
    cluster_ids, nearest_distances = _cluster_points(points, duplicate_tolerance)
    real_tol = tolerance if real_tolerance is None else real_tolerance

    diagnostics = tuple(
        diagnose_solution(
            inferred_system,
            solution,
            ordered_variables,
            tolerance=tolerance,
            real_tolerance=real_tol,
            condition_threshold=condition_threshold,
            certificate_tolerance=cert_tol,
            index=index,
            cluster_id=cluster_ids[index],
            nearest_neighbor_distance=nearest_distances[index],
        )
        for index, solution in enumerate(solution_list)
    )

    return SolutionAudit(
        diagnostics=diagnostics,
        variables=ordered_variables,
        tolerance=tolerance,
        duplicate_tolerance=duplicate_tolerance,
        real_tolerance=real_tol,
        condition_threshold=condition_threshold,
        certificate_tolerance=cert_tol,
    )


def validate_solutions(
    solutions: Any,
    system: Optional[PolynomialSystem] = None,
    variables: Optional[Sequence[Variable]] = None,
    *,
    tolerance: float = 1e-8,
    duplicate_tolerance: Optional[float] = None,
    real_tolerance: Optional[float] = None,
    condition_threshold: float = 1e12,
    certificate_tolerance: Optional[float] = None,
) -> bool:
    """Return ``True`` when every candidate passes residual validation."""
    return diagnose_solutions(
        solutions,
        system,
        variables,
        tolerance=tolerance,
        duplicate_tolerance=duplicate_tolerance,
        real_tolerance=real_tolerance,
        condition_threshold=condition_threshold,
        certificate_tolerance=certificate_tolerance,
    ).all_valid


def _coerce_variables(
    system: PolynomialSystem, variables: Optional[Sequence[Variable]]
) -> Tuple[Variable, ...]:
    if variables is None:
        if hasattr(system, "ordered_variables"):
            return tuple(system.ordered_variables())
        return tuple(sorted(system.variables(), key=lambda var: var.name))
    try:
        ordered_variables = tuple(variables)
    except TypeError as exc:
        raise TypeError(
            "variables must be an iterable of Variable objects"
        ) from exc
    seen = set()
    duplicates = []
    for index, variable in enumerate(ordered_variables):
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
        variable.name for variable in system.variables() if variable not in seen
    )
    if missing:
        raise ValueError(
            "Variable list is missing system variable(s): " + ", ".join(missing)
        )
    return ordered_variables


def _solution_point(solution: Any, variables: Sequence[Variable]) -> np.ndarray:
    return _coerce_point_for_variables(
        solution,
        list(variables),
        "solution",
        allow_nonfinite=True,
    )


def _cluster_points(
    points: Sequence[np.ndarray], tolerance: float
) -> Tuple[List[int], List[float]]:
    count = len(points)
    parent = list(range(count))
    nearest = [float("inf")] * count

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

    for left in range(count):
        for right in range(left + 1, count):
            distance = _point_distance(points[left], points[right])
            nearest[left] = min(nearest[left], distance)
            nearest[right] = min(nearest[right], distance)
            if distance <= tolerance:
                union(left, right)

    root_to_cluster: Dict[int, int] = {}
    cluster_ids: List[int] = []
    for index in range(count):
        root = find(index)
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        cluster_ids.append(root_to_cluster[root])
    return cluster_ids, nearest


def _point_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return float("inf")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = left - right
    return _scaled_euclidean_norm(difference)


def _norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if not np.all(np.isfinite(values)):
        return float("inf")
    return _scaled_euclidean_norm(values)


def _max_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if not np.all(np.isfinite(values)):
        return float("inf")
    return float(np.max(np.abs(values)))


def _matrix_rank(
    matrix: np.ndarray,
    is_finite: bool,
    rank_tolerance: Optional[float] = None,
) -> int:
    if not is_finite or matrix.size == 0:
        return 0
    try:
        if rank_tolerance is not None:
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            if not np.all(np.isfinite(singular_values)):
                return 0
            return int(np.sum(singular_values > rank_tolerance))
        return int(np.linalg.matrix_rank(matrix))
    except (np.linalg.LinAlgError, ValueError, OverflowError, FloatingPointError):
        return 0


def _condition_number(
    matrix: np.ndarray,
    is_finite: bool,
    rank_tolerance: Optional[float] = None,
) -> float:
    if not is_finite or matrix.size == 0 or min(matrix.shape, default=0) == 0:
        return float("inf")
    try:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    except (np.linalg.LinAlgError, ValueError, OverflowError, FloatingPointError):
        return float("inf")
    if not np.all(np.isfinite(singular_values)):
        return float("inf")
    if singular_values.size == 0 or singular_values[-1] == 0:
        return float("inf")
    if rank_tolerance is not None and singular_values[-1] <= rank_tolerance:
        return float("inf")
    with np.errstate(over="ignore", invalid="ignore"):
        condition = float(singular_values[0] / singular_values[-1])
    return condition if np.isfinite(condition) else float("inf")


def _rank_tolerance_for_residual_tolerance(tolerance: float) -> float:
    return max(1000.0 * min(float(tolerance), 1e-10), np.sqrt(np.finfo(float).eps))


def _newton_step_norm(
    jacobian: np.ndarray, residuals: np.ndarray, is_finite: bool
) -> float:
    if not is_finite:
        return float("inf")
    if residuals.size == 0:
        return 0.0
    step = solve_linear_system(jacobian, -residuals)
    if not np.all(np.isfinite(step)):
        return float("inf")
    return _scaled_euclidean_norm(step)


def _newton_error_bound(newton_step_norm: float) -> float:
    if not np.isfinite(newton_step_norm):
        return float("inf")
    return float(2.0 * newton_step_norm)


def _newton_ball_is_separated(
    newton_error_bound: float,
    nearest_neighbor_distance: float,
) -> bool:
    if not np.isfinite(newton_error_bound):
        return False
    if np.isinf(nearest_neighbor_distance):
        return True
    if nearest_neighbor_distance < 0 or not np.isfinite(nearest_neighbor_distance):
        return False
    return 2.0 * newton_error_bound < nearest_neighbor_distance


def _regularity_status(
    *,
    is_finite: bool,
    is_valid: bool,
    is_isolated: bool,
    is_well_conditioned: bool,
    newton_step_norm: float,
    certificate_tolerance: float,
    is_underdetermined: bool,
) -> str:
    if not is_finite:
        return "nonfinite"
    if not is_valid:
        return "large_residual"
    if is_underdetermined:
        return "underdetermined"
    if not is_isolated:
        return "singular_or_nonisolated"
    if not is_well_conditioned:
        return "ill_conditioned"
    if newton_step_norm > certificate_tolerance:
        return "large_newton_step"
    return "certified_regular"


def _certification_status(
    *,
    regularity_status: str,
    is_certified_regular: bool,
    is_newton_certified: bool,
    newton_error_bound: float,
    certificate_tolerance: float,
    nearest_neighbor_distance: float,
) -> str:
    if is_newton_certified:
        return "certified_unique"
    if not is_certified_regular:
        return regularity_status
    if not np.isfinite(newton_error_bound):
        return "nonfinite_newton_error"
    if newton_error_bound > 2.0 * certificate_tolerance:
        return "large_newton_error"
    if (
        np.isfinite(nearest_neighbor_distance)
        and 2.0 * newton_error_bound >= nearest_neighbor_distance
    ):
        return "overlapping_newton_ball"
    return "uncertified"


def _validate_thresholds(
    tolerance: float,
    real_tolerance: Optional[float],
    condition_threshold: float,
    certificate_tolerance: Optional[float] = None,
) -> Tuple[float, Optional[float], float, Optional[float]]:
    tolerance = _validate_nonnegative_finite_float("tolerance", tolerance)
    if real_tolerance is not None:
        real_tolerance = _validate_nonnegative_finite_float(
            "real_tolerance", real_tolerance
        )
    condition_threshold = _validate_nonnegative_finite_float(
        "condition_threshold", condition_threshold
    )
    if certificate_tolerance is not None:
        certificate_tolerance = _validate_nonnegative_finite_float(
            "certificate_tolerance", certificate_tolerance
        )
    return tolerance, real_tolerance, condition_threshold, certificate_tolerance


def _validate_polynomial_system(name: str, system: Any) -> None:
    if not isinstance(system, PolynomialSystem):
        raise TypeError(f"{name} must be a PolynomialSystem")


def _validate_nonnegative_finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value < 0:
        raise ValueError(f"{name} must be non-negative and finite")
    return numeric_value


def _validate_boolean_option(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _complex_as_dict(value: complex) -> Dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}
