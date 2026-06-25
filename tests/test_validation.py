import json
import math

import numpy as np
import pytest

from pycontinuum import (
    PolynomialSystem,
    Solution,
    SolutionSet,
    diagnose_solution,
    diagnose_solutions,
    polyvar,
    refine_solution,
    refine_solutions,
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
    assert audit.isolated_count == 2
    assert audit.certified_regular_count == 2
    assert audit.newton_certified_count == 2
    assert audit.all_newton_certified
    assert audit.max_residual < 1e-8
    assert audit.max_newton_error_bound < 1e-8
    assert validate_solutions(solutions, tolerance=1e-8)

    first = audit.diagnostics[0]
    assert first.jacobian_shape == (1, 1)
    assert first.jacobian_rank == 1
    assert first.is_well_conditioned
    assert first.is_isolated
    assert first.is_certified_regular
    assert first.is_newton_certified
    assert first.regularity_status == "certified_regular"
    assert first.certification_status == "certified_unique"
    assert first.newton_step_norm < 1e-8
    assert first.newton_error_bound < 1e-8
    assert first.as_dict()["variables"] == ["x"]
    assert first.as_dict()["is_newton_certified"]


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
    assert not diagnostics.is_certified_regular
    assert diagnostics.regularity_status == "large_residual"


def test_solution_audit_reports_status_count_breakdowns():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])
    solutions = SolutionSet(
        [
            Solution({x: 1.0 + 0j}, residual=0.0),
            Solution({x: 2.0 + 0j}, residual=1.0),
        ],
        system,
    )

    audit = diagnose_solutions(solutions, tolerance=1e-8)
    regularity_counts = audit.regularity_status_counts()
    certification_counts = audit.certification_status_counts()
    summary = audit.summary()
    record = audit.as_dict()

    assert regularity_counts == {
        "certified_regular": 1,
        "large_residual": 1,
    }
    assert certification_counts == {
        "certified_unique": 1,
        "large_residual": 1,
    }
    assert summary["regularity_status_counts"] == regularity_counts
    assert summary["certification_status_counts"] == certification_counts
    assert record["regularity_status_counts"] == regularity_counts
    assert record["certification_status_counts"] == certification_counts


def test_diagnostics_report_coefficient_scaled_residuals():
    x = polyvar("x")
    system = PolynomialSystem([1e12 * (x - 1)])
    candidate = Solution({x: 1.0 + 1e-10}, residual=100.0)

    diagnostics = diagnose_solution(system, candidate, tolerance=1e-8)
    audit = diagnose_solutions(
        SolutionSet([candidate], system),
        tolerance=1e-8,
    )

    assert diagnostics.residual_norm > 1.0
    assert diagnostics.scaled_residual_norm < 1e-8
    assert not diagnostics.is_valid
    assert diagnostics.is_scaled_valid
    assert (
        diagnostics.as_dict()["scaled_residual_norm"]
        == diagnostics.scaled_residual_norm
    )
    assert not audit.all_valid
    assert audit.all_scaled_valid
    assert audit.max_scaled_residual < 1e-8


def test_diagnostics_reject_tiny_row_false_positive():
    x = polyvar("x")
    system = PolynomialSystem([1e-12 * (x - 2)])
    candidate = Solution({x: 1.0 + 0j}, residual=1e-12)
    solutions = SolutionSet([candidate], system)

    diagnostics = diagnose_solution(system, candidate, tolerance=1e-8)
    audit = diagnose_solutions(solutions, tolerance=1e-8)

    assert diagnostics.residual_norm < 1e-8
    assert diagnostics.scaled_residual_norm == pytest.approx(0.5)
    assert diagnostics.backward_error_norm == pytest.approx(1.0 / 3.0)
    assert not diagnostics.is_valid
    assert not diagnostics.is_scaled_valid
    assert not diagnostics.is_backward_stable
    assert diagnostics.regularity_status == "large_residual"
    assert not audit.all_valid
    assert audit.failed_indices == (0,)
    assert not validate_solutions(solutions, tolerance=1e-8)


def test_diagnostics_export_strict_json_for_infinite_neighbor_distance():
    x = polyvar("x")
    system = PolynomialSystem([(10**400) * (x - 1)])
    solution = Solution({x: 1.0 + 0j}, residual=float("inf"))
    solutions = SolutionSet([solution], system)

    diagnostics = diagnose_solution(system, solution, tolerance=1e-8)
    audit = diagnose_solutions(solutions, tolerance=1e-8)
    record = diagnostics.as_dict(strict_json=True)
    summary = audit.summary(strict_json=True)
    audit_record = audit.as_dict(strict_json=True)

    json.dumps(record, allow_nan=False)
    json.dumps(summary, allow_nan=False)
    json.dumps(audit_record, allow_nan=False)
    assert record["residual_norm"] == 0.0
    assert record["max_residual"] == 0.0
    assert record["nearest_neighbor_distance"] == "Infinity"
    assert summary["max_residual"] == 0.0
    assert audit_record["diagnostics"][0]["residual_norm"] == 0.0

    with pytest.raises(TypeError, match="strict_json must be a boolean"):
        diagnostics.as_dict(strict_json=0)
    with pytest.raises(TypeError, match="strict_json must be a boolean"):
        audit.summary(strict_json="yes")
    with pytest.raises(TypeError, match="strict_json must be a boolean"):
        audit.as_dict(strict_json=1)


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


def test_newton_certificate_rejects_overlapping_candidates():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])
    solutions = SolutionSet(
        [
            Solution({x: 1.0 + 1e-10}, residual=1e-10),
            Solution({x: 1.0 + 1.5e-10}, residual=1.5e-10),
        ],
        system,
    )

    audit = diagnose_solutions(
        solutions,
        tolerance=1e-6,
        duplicate_tolerance=1e-12,
    )

    assert audit.valid_count == 2
    assert audit.certified_regular_count == 2
    assert audit.newton_certified_count == 0
    assert not audit.all_newton_certified
    assert audit.max_newton_error_bound > 0.0
    assert audit.diagnostics[0].is_certified_regular
    assert not audit.diagnostics[0].is_newton_certified
    assert audit.diagnostics[0].certification_status == "overlapping_newton_ball"


def test_singular_solution_is_valid_but_rank_deficient():
    x = polyvar("x")
    system = PolynomialSystem([x**2])
    solution = Solution({x: 0.0}, residual=0.0, is_singular=True)

    diagnostics = diagnose_solution(system, solution, tolerance=1e-12)

    assert diagnostics.is_valid
    assert diagnostics.is_rank_deficient
    assert not diagnostics.is_regular
    assert not diagnostics.is_isolated
    assert not diagnostics.is_certified_regular
    assert not diagnostics.is_newton_certified
    assert diagnostics.regularity_status == "singular_or_nonisolated"
    assert diagnostics.certification_status == "singular_or_nonisolated"
    assert not diagnostics.is_well_conditioned
    assert diagnostics.jacobian_rank == 0
    assert math.isinf(diagnostics.condition_number)


def test_diagnostics_fail_closed_when_svd_fails(monkeypatch):
    x = polyvar("x")
    system = PolynomialSystem([x - 1])
    solution = Solution({x: 1.0 + 0j}, residual=0.0)

    def failing_svd(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np.linalg, "svd", failing_svd)

    diagnostics = diagnose_solution(system, solution, tolerance=1e-8)

    assert diagnostics.is_valid
    assert diagnostics.jacobian_rank == 0
    assert math.isinf(diagnostics.condition_number)
    assert diagnostics.is_rank_deficient
    assert diagnostics.regularity_status == "singular_or_nonisolated"


def test_underdetermined_diagnostics_are_not_isolated_roots():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x])
    solution = Solution({x: 0.0, y: 2.0}, residual=0.0)

    diagnostics = diagnose_solution(
        system,
        solution,
        variables=[x, y],
        tolerance=1e-12,
    )

    assert diagnostics.is_valid
    assert diagnostics.jacobian_shape == (1, 2)
    assert diagnostics.is_underdetermined
    assert not diagnostics.is_regular
    assert not diagnostics.is_isolated
    assert not diagnostics.is_well_conditioned
    assert not diagnostics.is_certified_regular
    assert not diagnostics.is_newton_certified
    assert diagnostics.regularity_status == "underdetermined"
    assert diagnostics.certification_status == "underdetermined"


def test_nonfinite_candidate_fails_cleanly():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])
    solution = Solution({x: np.nan}, residual=float("nan"))

    diagnostics = diagnose_solution(system, solution)

    assert not diagnostics.is_finite
    assert not diagnostics.is_valid
    assert not diagnostics.is_certified_regular
    assert not diagnostics.is_newton_certified
    assert diagnostics.regularity_status == "nonfinite"
    assert diagnostics.certification_status == "nonfinite"
    assert math.isinf(diagnostics.residual_norm)


def test_overflowing_residual_candidate_fails_cleanly():
    x = polyvar("x")
    system = PolynomialSystem([(10**400) * x])
    solution = Solution({x: 1.0 + 0j}, residual=float("inf"))
    solutions = SolutionSet([solution], system)

    diagnostics = diagnose_solution(system, solution)
    audit = diagnose_solutions(solutions)

    assert not diagnostics.is_finite
    assert not diagnostics.is_valid
    assert diagnostics.regularity_status == "nonfinite"
    assert diagnostics.certification_status == "nonfinite"
    assert math.isinf(diagnostics.residual_norm)
    assert math.isinf(diagnostics.max_residual)
    assert audit.invalid_count == 1
    assert audit.failed_indices == (0,)
    assert not validate_solutions(solutions)


def test_diagnostics_use_scaled_backward_error_for_huge_exact_root():
    x = polyvar("x")
    system = PolynomialSystem([(10**400) * (x - 1)])
    solution = Solution({x: 1.0 + 0j}, residual=float("inf"))
    solutions = SolutionSet([solution], system)

    diagnostics = diagnose_solution(system, solution, tolerance=1e-8)
    audit = diagnose_solutions(solutions, tolerance=1e-8)

    assert diagnostics.is_valid
    assert diagnostics.is_scaled_valid
    assert diagnostics.is_backward_stable
    assert diagnostics.is_certified_regular
    assert diagnostics.is_newton_certified
    assert diagnostics.regularity_status == "certified_regular"
    assert diagnostics.certification_status == "certified_unique"
    assert diagnostics.newton_step_norm == 0.0
    assert diagnostics.scaled_residual_norm == 0.0
    assert diagnostics.backward_error_norm == 0.0
    assert audit.all_valid
    assert audit.all_scaled_valid
    assert audit.all_backward_stable
    assert audit.all_newton_certified
    assert validate_solutions(solutions)


def test_diagnostics_keep_large_finite_residual_norms_finite():
    x = polyvar("x")
    huge = 1e200
    system = PolynomialSystem([huge * x, huge * x])
    solution = Solution({x: 1.0 + 0j}, residual=float("inf"))

    diagnostics = diagnose_solution(system, solution, tolerance=1e-8)

    assert math.isfinite(diagnostics.residual_norm)
    assert diagnostics.residual_norm == pytest.approx(math.sqrt(2.0) * huge)
    assert diagnostics.max_residual == pytest.approx(huge)
    assert not diagnostics.is_valid


def test_diagnostics_keep_large_finite_neighbor_distances_finite():
    x, y = polyvar("x", "y")
    huge = 1e200
    system = PolynomialSystem([x, y])
    solutions = SolutionSet(
        [
            Solution({x: 0.0 + 0j, y: 0.0 + 0j}, residual=0.0),
            Solution({x: huge + 0j, y: huge + 0j}, residual=float("inf")),
        ],
        system,
    )

    audit = diagnose_solutions(
        solutions,
        tolerance=1e-8,
        duplicate_tolerance=0.0,
    )

    assert audit.cluster_count == 2
    assert math.isfinite(audit.diagnostics[0].nearest_neighbor_distance)
    assert audit.diagnostics[0].nearest_neighbor_distance == pytest.approx(
        math.sqrt(2.0) * huge
    )
    assert audit.diagnostics[1].nearest_neighbor_distance == pytest.approx(
        audit.diagnostics[0].nearest_neighbor_distance
    )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"tolerance": float("nan")}, ValueError, "tolerance must be non-negative"),
        ({"tolerance": True}, TypeError, "tolerance must be a number"),
        ({"tolerance": "1e-8"}, TypeError, "tolerance must be a number"),
        (
            {"real_tolerance": float("inf")},
            ValueError,
            "real_tolerance must be non-negative",
        ),
        (
            {"real_tolerance": "1e-8"},
            TypeError,
            "real_tolerance must be a number",
        ),
        (
            {"condition_threshold": -1.0},
            ValueError,
            "condition_threshold must be non-negative",
        ),
        (
            {"condition_threshold": "1e12"},
            TypeError,
            "condition_threshold must be a number",
        ),
        (
            {"certificate_tolerance": "tight"},
            TypeError,
            "certificate_tolerance must be a number",
        ),
        (
            {"certificate_tolerance": "1e-8"},
            TypeError,
            "certificate_tolerance must be a number",
        ),
    ],
)
def test_diagnose_solution_rejects_invalid_thresholds(
    kwargs, error_type, message
):
    x = polyvar("x")
    system = PolynomialSystem([x - 1])
    solution = Solution({x: 1.0}, residual=0.0)

    with pytest.raises(error_type, match=message):
        diagnose_solution(system, solution, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            {"duplicate_tolerance": float("inf")},
            ValueError,
            "duplicate_tolerance must be non-negative",
        ),
        (
            {"duplicate_tolerance": False},
            TypeError,
            "duplicate_tolerance must be a number",
        ),
        (
            {"duplicate_tolerance": "1e-8"},
            TypeError,
            "duplicate_tolerance must be a number",
        ),
    ],
)
def test_diagnose_solutions_rejects_invalid_duplicate_tolerance(
    kwargs, error_type, message
):
    x = polyvar("x")
    system = PolynomialSystem([x - 1])
    solutions = SolutionSet([Solution({x: 1.0}, residual=0.0)], system)

    with pytest.raises(error_type, match=message):
        diagnose_solutions(solutions, **kwargs)


def test_diagnostics_reject_malformed_variable_lists():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1])
    solution = Solution({x: 0.5, y: 0.5}, residual=0.0)

    with pytest.raises(TypeError, match="variables must be an iterable"):
        diagnose_solution(system, solution, variables=x)
    with pytest.raises(TypeError, match=r"variables\[1\] must be a Variable"):
        diagnose_solution(system, solution, variables=[x, "extra"])
    with pytest.raises(ValueError, match="missing system variable"):
        diagnose_solution(system, solution, variables=[x])


def test_diagnostics_accept_coordinate_records():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 3, x * y - 2])

    class SolutionLike:
        def __init__(self, values):
            self.values = values

    vector = [1.0 + 0j, 2.0 + 0j]
    direct_mapping = {"y": 2.0 + 0j, x: 1.0 + 0j}
    solution_like = SolutionLike({"x": 1.0 + 0j, "y": 2.0 + 0j})

    diagnostics = diagnose_solution(system, vector, variables=[x, y])
    audit = diagnose_solutions(
        [vector, direct_mapping, solution_like],
        system=system,
        variables=[x, y],
        tolerance=1e-8,
    )

    assert diagnostics.is_valid
    assert diagnostics.point == (1.0 + 0j, 2.0 + 0j)
    assert audit.all_valid
    assert audit.count == 3
    assert validate_solutions(
        [vector, direct_mapping, solution_like],
        system=system,
        variables=[x, y],
        tolerance=1e-8,
    )


def test_diagnostics_reject_missing_solution_coordinates():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1])
    solution = Solution({x: 0.5}, residual=0.0)
    solutions = SolutionSet([solution], system)

    with pytest.raises(ValueError, match=r"missing coordinate.*y"):
        diagnose_solution(system, solution)
    with pytest.raises(ValueError, match=r"missing coordinate.*y"):
        diagnose_solution(system, {"x": 0.5})
    with pytest.raises(ValueError, match=r"missing coordinate.*y"):
        diagnose_solutions(solutions)


def test_refine_solution_improves_bad_candidate():
    x = polyvar("x")
    system = PolynomialSystem([x**2 - 1])
    candidate = Solution({x: 0.25}, residual=0.9375)

    refined = refine_solution(system, candidate, tol=1e-14)

    assert abs(abs(refined.values[x]) - 1.0) < 1e-10
    assert refined.residual < 1e-12
    assert refined.refinement["accepted"]
    assert refined.refinement["final_residual"] < refined.refinement["initial_residual"]
    assert candidate.values[x] == 0.25


def test_refine_solution_uses_scaled_newton_fallback_for_extreme_rows():
    x = polyvar("x")
    system = PolynomialSystem([(10**400) * (x - 1)])
    candidate = Solution({x: 0.9 + 0j}, residual=float("inf"))

    refined = refine_solution(system, candidate, tol=1e-12, max_iters=5)

    assert refined.refinement["accepted"]
    assert refined.refinement["newton_success"]
    assert math.isinf(refined.refinement["initial_residual"])
    assert refined.refinement["final_residual"] == 0.0
    assert refined.scaled_residual == 0.0
    assert refined.backward_error == 0.0
    assert abs(refined.values[x] - 1.0) < 1e-12
    assert candidate.values[x] == 0.9 + 0j


def test_refinement_accepts_coordinate_records():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 3, x * y - 2])

    class SolutionLike:
        def __init__(self, values):
            self.values = values

    vector = [0.8 + 0j, 2.1 + 0j]
    direct_mapping = {"y": 2.1 + 0j, x: 0.8 + 0j}
    solution_like = SolutionLike({"x": 0.8 + 0j, "y": 2.1 + 0j})

    refined = refine_solution(
        system,
        vector,
        variables=[x, y],
        tol=1e-12,
    )
    refined_set = refine_solutions(
        [vector, direct_mapping, solution_like],
        system=system,
        variables=[x, y],
        tol=1e-12,
    )

    assert isinstance(refined, Solution)
    assert abs(refined.values[x] - 1.0) < 1e-10
    assert abs(refined.values[y] - 2.0) < 1e-10
    assert refined.refinement["accepted"]
    assert all(isinstance(solution, Solution) for solution in refined_set)
    assert all(solution.refinement["accepted"] for solution in refined_set)


def test_refined_solution_exports_refinement_metadata():
    x = polyvar("x")
    system = PolynomialSystem([x**2 - 1])
    candidate = Solution({x: 0.25}, residual=0.9375)

    refined = refine_solution(system, candidate, tol=1e-14)
    refined_set = refine_solutions([candidate], system=system, tol=1e-14)
    direct_record = refined.as_dict()
    set_record = refined_set.as_dicts()[0]

    json.dumps(direct_record)
    json.dumps(set_record)
    assert direct_record["refinement"]["accepted"]
    assert direct_record["refinement"]["final_residual"] == 0.0
    assert set_record["refinement"]["accepted"]
    assert "refinement" not in candidate.as_dict()
    assert "refinement" not in refined.as_dict(include_metadata=False)


def test_solution_set_refine_preserves_metadata_and_marks_refined():
    x = polyvar("x")
    system = PolynomialSystem([x - 2])
    solutions = SolutionSet([Solution({x: 1.9}, residual=0.1)], system)
    solutions._meta["source"] = "manual"

    refined = solutions.refine(tol=1e-14)
    via_function = refine_solutions(solutions, tol=1e-14)

    assert refined is not solutions
    assert refined._meta["source"] == "manual"
    assert refined._meta["is_refined"]
    assert refined._meta["refinement"]["accepted_count"] == 1
    assert abs(refined[0].values[x] - 2.0) < 1e-12
    assert via_function._meta["refinement"]["success_count"] == 1


def test_refine_solutions_accepts_generators_and_rejects_bad_iterables():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    refined = refine_solutions(
        (Solution({x: 0.9}, residual=0.1) for _ in range(1)),
        system=system,
        tol=1e-14,
    )

    assert len(refined) == 1
    assert refined[0].refinement["accepted"]

    with pytest.raises(TypeError, match="solutions must be an iterable"):
        refine_solutions(None, system=system)
    with pytest.raises(TypeError, match="solutions must be an iterable"):
        refine_solutions("not-solutions", system=system)
    with pytest.raises(TypeError, match="numeric one-dimensional point"):
        refine_solutions([object()], system=system)


def test_refine_solution_keeps_candidate_when_newton_worsens():
    x = polyvar("x")
    system = PolynomialSystem([x**2 - 1])
    candidate = Solution({x: 0.0}, residual=1.0)

    refined = refine_solution(system, candidate, tol=1e-14, max_iters=1)

    assert refined.values[x] == candidate.values[x]
    assert refined.residual == 1.0
    assert not refined.refinement["accepted"]
    assert not refined.refinement["success"]


def test_refine_solution_rejects_tiny_raw_residual_false_positive():
    x = polyvar("x")
    system = PolynomialSystem([1e-12 * (x - 1)])
    candidate = Solution({x: 2.0 + 0j}, residual=1e-12)

    refined = refine_solution(system, candidate, tol=1e-10, max_iters=1)

    assert refined.values[x] == 2.0 + 0j
    assert not refined.refinement["accepted"]
    assert not refined.refinement["success"]
    assert refined.refinement["newton_success"]
    assert refined.residual == pytest.approx(1e-12)
    assert refined.scaled_residual == pytest.approx(1.0)
    assert refined.backward_error > 0.1


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"tol": 0.0}, ValueError, "tol must be positive and finite"),
        ({"tol": True}, TypeError, "tol must be a number"),
        ({"max_iters": 1.5}, TypeError, "max_iters must be an integer"),
        ({"max_iters": 0}, ValueError, "max_iters must be positive"),
        (
            {"singularity_threshold": float("nan")},
            ValueError,
            "singularity_threshold must be positive and finite",
        ),
        ({"keep_failed": "yes"}, TypeError, "keep_failed must be a boolean"),
    ],
)
def test_refine_solution_rejects_invalid_options(kwargs, error_type, message):
    x = polyvar("x")
    system = PolynomialSystem([x - 1])
    candidate = Solution({x: 0.9}, residual=0.1)

    with pytest.raises(error_type, match=message):
        refine_solution(system, candidate, **kwargs)


def test_refine_solution_rejects_invalid_system_and_variables():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1])
    candidate = Solution({x: 0.5, y: 0.5}, residual=0.0)
    partial_candidate = Solution({x: 0.5}, residual=0.0)

    with pytest.raises(TypeError, match="system must be a PolynomialSystem"):
        refine_solution([x - 1], candidate)
    with pytest.raises(TypeError, match=r"variables\[1\] must be a Variable"):
        refine_solution(system, candidate, variables=[x, "extra"])
    with pytest.raises(ValueError, match="missing system variable"):
        refine_solution(system, candidate, variables=[x])
    with pytest.raises(ValueError, match=r"missing coordinate.*y"):
        refine_solution(system, partial_candidate)
    with pytest.raises(ValueError, match=r"missing coordinate.*y"):
        refine_solution(system, {"x": 0.5})
