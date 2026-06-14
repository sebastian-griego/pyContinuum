import math

import numpy as np

from pycontinuum import (
    PolynomialSystem,
    Solution,
    SolutionSet,
    diagnose_solution,
    diagnose_solutions,
    polyvar,
    solve,
    validate_solutions,
)


def test_diagnose_solutions_reports_residual_rank_and_clusters():
    x = polyvar("x")
    system = PolynomialSystem([x**2 - 1])
    solutions = solve(system, random_state=123)

    audit = diagnose_solutions(solutions, tolerance=1e-8)

    assert audit.count == 2
    assert audit.all_valid
    assert audit.valid_count == 2
    assert audit.real_count == 2
    assert audit.rank_deficient_count == 0
    assert audit.cluster_count == 2
    assert audit.duplicate_count == 0
    assert audit.max_residual < 1e-8
    assert validate_solutions(solutions, tolerance=1e-8)

    first = audit.diagnostics[0]
    assert first.jacobian_shape == (1, 1)
    assert first.jacobian_rank == 1
    assert first.is_well_conditioned
    assert first.as_dict()["variables"] == ["x"]


def test_solution_set_diagnostics_shortcut_matches_public_function():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1, x - y])
    solutions = solve(system, random_state=123)

    via_method = solutions.diagnostics(tolerance=1e-8)
    via_function = diagnose_solutions(solutions, tolerance=1e-8)

    assert via_method.summary() == via_function.summary()
    assert via_method.count == 1
    assert via_method.diagnostics[0].point == via_function.diagnostics[0].point


def test_diagnose_solution_flags_bad_candidate():
    x = polyvar("x")
    system = PolynomialSystem([x**2 - 1])
    bad_solution = Solution({x: 0.25}, residual=0.0)

    diagnostics = diagnose_solution(system, bad_solution, tolerance=1e-8)

    assert not diagnostics.is_valid
    assert diagnostics.residual_norm > 0.9
    assert diagnostics.max_residual > 0.9
    assert diagnostics.is_finite


def test_duplicate_clusters_group_nearby_candidates():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])
    solutions = SolutionSet(
        [
            Solution({x: 1.0 + 1e-9}, residual=1e-9),
            Solution({x: 1.0 + 2e-9}, residual=2e-9),
            Solution({x: -1.0}, residual=2.0),
        ],
        system,
    )

    audit = diagnose_solutions(
        solutions,
        tolerance=1e-6,
        duplicate_tolerance=1e-6,
    )

    assert audit.cluster_count == 2
    assert audit.duplicate_count == 1
    assert audit.duplicate_clusters() == {0: (0, 1)}
    assert audit.invalid_count == 1
    assert audit.failed_indices == (2,)
    assert audit.diagnostics[0].cluster_id == audit.diagnostics[1].cluster_id
    assert audit.diagnostics[0].nearest_neighbor_distance < 1e-6


def test_singular_solution_is_valid_but_rank_deficient():
    x = polyvar("x")
    system = PolynomialSystem([x**2])
    solution = Solution({x: 0.0}, residual=0.0, is_singular=True)

    diagnostics = diagnose_solution(system, solution, tolerance=1e-12)

    assert diagnostics.is_valid
    assert diagnostics.is_rank_deficient
    assert not diagnostics.is_well_conditioned
    assert diagnostics.jacobian_rank == 0
    assert math.isinf(diagnostics.condition_number)


def test_nonfinite_candidate_fails_cleanly():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])
    solution = Solution({x: np.nan}, residual=float("nan"))

    diagnostics = diagnose_solution(system, solution)

    assert not diagnostics.is_finite
    assert not diagnostics.is_valid
    assert math.isinf(diagnostics.residual_norm)
