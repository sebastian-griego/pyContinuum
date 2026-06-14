"""Solution diagnostics for polynomial systems.

The solver returns numerical candidate roots. This module provides a
structured audit layer for residuals, Jacobian rank, conditioning, realness,
and duplicate clustering so downstream code can decide whether a solve is
good enough for its tolerance budget.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pycontinuum.polynomial import PolynomialSystem, Variable
from pycontinuum.utils import evaluate_jacobian_at_point


@dataclass(frozen=True)
class SolutionDiagnostics:
    """Numerical diagnostics for one candidate solution."""

    index: int
    variables: Tuple[Variable, ...]
    point: Tuple[complex, ...]
    residuals: Tuple[complex, ...]
    residual_norm: float
    max_residual: float
    jacobian_rank: int
    jacobian_shape: Tuple[int, int]
    condition_number: float
    is_finite: bool
    is_real: bool
    is_valid: bool
    is_rank_deficient: bool
    is_well_conditioned: bool
    cluster_id: int
    nearest_neighbor_distance: float

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation of the diagnostics."""
        return {
            "index": self.index,
            "variables": [var.name for var in self.variables],
            "point": [_complex_as_dict(value) for value in self.point],
            "residuals": [_complex_as_dict(value) for value in self.residuals],
            "residual_norm": self.residual_norm,
            "max_residual": self.max_residual,
            "jacobian_rank": self.jacobian_rank,
            "jacobian_shape": self.jacobian_shape,
            "condition_number": self.condition_number,
            "is_finite": self.is_finite,
            "is_real": self.is_real,
            "is_valid": self.is_valid,
            "is_rank_deficient": self.is_rank_deficient,
            "is_well_conditioned": self.is_well_conditioned,
            "cluster_id": self.cluster_id,
            "nearest_neighbor_distance": self.nearest_neighbor_distance,
        }


@dataclass(frozen=True)
class SolutionAudit:
    """Aggregate diagnostics for a solution set."""

    diagnostics: Tuple[SolutionDiagnostics, ...]
    variables: Tuple[Variable, ...]
    tolerance: float
    duplicate_tolerance: float
    real_tolerance: float
    condition_threshold: float

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
    def real_count(self) -> int:
        """Number of solutions whose coordinates are real within tolerance."""
        return sum(item.is_real for item in self.diagnostics)

    @property
    def rank_deficient_count(self) -> int:
        """Number of solutions with rank-deficient Jacobians."""
        return sum(item.is_rank_deficient for item in self.diagnostics)

    @property
    def well_conditioned_count(self) -> int:
        """Number of solutions passing the conditioning threshold."""
        return sum(item.is_well_conditioned for item in self.diagnostics)

    @property
    def all_valid(self) -> bool:
        """Whether every candidate passes the residual/finite-value check."""
        return all(item.is_valid for item in self.diagnostics)

    @property
    def max_residual(self) -> float:
        """Largest residual norm among all candidates."""
        if not self.diagnostics:
            return 0.0
        return max(item.residual_norm for item in self.diagnostics)

    @property
    def worst_condition_number(self) -> float:
        """Largest finite or infinite Jacobian condition number."""
        if not self.diagnostics:
            return 0.0
        return max(item.condition_number for item in self.diagnostics)

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

    def summary(self) -> Dict[str, Any]:
        """Return compact aggregate diagnostics."""
        return {
            "count": self.count,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "real_count": self.real_count,
            "rank_deficient_count": self.rank_deficient_count,
            "well_conditioned_count": self.well_conditioned_count,
            "all_valid": self.all_valid,
            "max_residual": self.max_residual,
            "worst_condition_number": self.worst_condition_number,
            "cluster_count": self.cluster_count,
            "duplicate_count": self.duplicate_count,
            "failed_indices": self.failed_indices,
            "tolerance": self.tolerance,
            "duplicate_tolerance": self.duplicate_tolerance,
            "real_tolerance": self.real_tolerance,
            "condition_threshold": self.condition_threshold,
        }

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation of the full audit."""
        result = self.summary()
        result["variables"] = [var.name for var in self.variables]
        result["diagnostics"] = [item.as_dict() for item in self.diagnostics]
        result["duplicate_clusters"] = self.duplicate_clusters()
        return result


def diagnose_solution(
    system: PolynomialSystem,
    solution: Any,
    variables: Optional[Sequence[Variable]] = None,
    *,
    tolerance: float = 1e-8,
    real_tolerance: Optional[float] = None,
    condition_threshold: float = 1e12,
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
        condition_threshold: Maximum condition number for ``is_well_conditioned``.
        index: Index to report in the resulting diagnostics.
        cluster_id: Duplicate cluster identifier to attach to this solution.
        nearest_neighbor_distance: Distance to the nearest other solution.

    Returns:
        A :class:`SolutionDiagnostics` instance.
    """
    _validate_thresholds(tolerance, real_tolerance, condition_threshold)
    ordered_variables = _coerce_variables(system, variables)
    real_tol = tolerance if real_tolerance is None else real_tolerance

    point = _solution_point(solution, ordered_variables)
    values = {var: value for var, value in zip(ordered_variables, point)}
    residuals_array = np.array(system.evaluate(values), dtype=complex)
    jacobian = evaluate_jacobian_at_point(system, point, list(ordered_variables))

    is_finite = bool(
        np.all(np.isfinite(point))
        and np.all(np.isfinite(residuals_array))
        and np.all(np.isfinite(jacobian))
    )
    residual_norm = _norm(residuals_array)
    max_residual = _max_abs(residuals_array)
    jacobian_rank = _matrix_rank(jacobian, is_finite)
    condition_number = _condition_number(jacobian, is_finite)
    is_rank_deficient = jacobian_rank < min(jacobian.shape, default=0)
    is_real = bool(np.all(np.abs(np.imag(point)) <= real_tol))
    is_valid = is_finite and residual_norm <= tolerance
    is_well_conditioned = (
        is_finite and not is_rank_deficient and condition_number <= condition_threshold
    )

    return SolutionDiagnostics(
        index=index,
        variables=ordered_variables,
        point=tuple(complex(value) for value in point),
        residuals=tuple(complex(value) for value in residuals_array),
        residual_norm=residual_norm,
        max_residual=max_residual,
        jacobian_rank=jacobian_rank,
        jacobian_shape=tuple(jacobian.shape),
        condition_number=condition_number,
        is_finite=is_finite,
        is_real=is_real,
        is_valid=is_valid,
        is_rank_deficient=is_rank_deficient,
        is_well_conditioned=is_well_conditioned,
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
) -> SolutionAudit:
    """Audit a solution set or iterable of solution objects.

    Args:
        solutions: A ``SolutionSet``-like object or iterable of solutions.
        system: Polynomial system. Optional when ``solutions`` has ``system``.
        variables: Optional variable order. Defaults to the system order.
        tolerance: Residual norm threshold for validity.
        duplicate_tolerance: Euclidean distance threshold for duplicate clusters.
        real_tolerance: Imaginary-part threshold for real classification.
        condition_threshold: Maximum condition number for well-conditioned roots.

    Returns:
        A :class:`SolutionAudit` instance with per-solution and aggregate metrics.
    """
    _validate_thresholds(tolerance, real_tolerance, condition_threshold)
    if duplicate_tolerance is None:
        duplicate_tolerance = max(10.0 * tolerance, 1e-12)
    if duplicate_tolerance < 0:
        raise ValueError("duplicate_tolerance must be non-negative")

    inferred_system = (
        system if system is not None else getattr(solutions, "system", None)
    )
    if inferred_system is None:
        raise ValueError("system is required when auditing a bare solution iterable")

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
    ).all_valid


def _coerce_variables(
    system: PolynomialSystem, variables: Optional[Sequence[Variable]]
) -> Tuple[Variable, ...]:
    if variables is None:
        if hasattr(system, "ordered_variables"):
            return tuple(system.ordered_variables())
        return tuple(sorted(system.variables(), key=lambda var: var.name))
    return tuple(variables)


def _solution_point(solution: Any, variables: Sequence[Variable]) -> np.ndarray:
    values = getattr(solution, "values", None)
    if not isinstance(values, dict):
        raise TypeError("solution must expose a values dictionary")
    return np.array([values.get(var, 0) for var in variables], dtype=complex)


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
    return float(np.linalg.norm(left - right))


def _norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if not np.all(np.isfinite(values)):
        return float("inf")
    return float(np.linalg.norm(values))


def _max_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if not np.all(np.isfinite(values)):
        return float("inf")
    return float(np.max(np.abs(values)))


def _matrix_rank(matrix: np.ndarray, is_finite: bool) -> int:
    if not is_finite or matrix.size == 0:
        return 0
    return int(np.linalg.matrix_rank(matrix))


def _condition_number(matrix: np.ndarray, is_finite: bool) -> float:
    if not is_finite or matrix.size == 0 or min(matrix.shape, default=0) == 0:
        return float("inf")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] == 0:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def _validate_thresholds(
    tolerance: float,
    real_tolerance: Optional[float],
    condition_threshold: float,
) -> None:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if real_tolerance is not None and real_tolerance < 0:
        raise ValueError("real_tolerance must be non-negative")
    if condition_threshold < 0:
        raise ValueError("condition_threshold must be non-negative")


def _complex_as_dict(value: complex) -> Dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}
