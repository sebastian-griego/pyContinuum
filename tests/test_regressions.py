import numpy as np
import pytest

from pycontinuum import Monomial, Polynomial, PolynomialSystem, polyvar, solve
from pycontinuum.start_systems import generate_total_degree_start_system
from pycontinuum.witness_set import generate_generic_slice


def test_polyvar_rejects_invalid_names():
    with pytest.raises(ValueError, match="At least one"):
        polyvar()
    with pytest.raises(ValueError, match="non-empty"):
        polyvar("")
    with pytest.raises(ValueError, match="whitespace"):
        polyvar("bad name")


def test_polynomial_exponents_must_be_non_negative_integers():
    x = polyvar("x")

    with pytest.raises(ValueError, match="non-negative integer"):
        x ** -1
    with pytest.raises(ValueError, match="non-negative integer"):
        x ** 1.5
    with pytest.raises(ValueError, match="non-negative"):
        Monomial({x: -1})


def test_parse_preserves_alphanumeric_and_underscore_variable_names():
    variables = {}
    polynomial = Polynomial.parse("x1 + 2*var_name - 3", variables=variables)

    assert set(variables) == {"x1", "var_name"}
    value = polynomial.evaluate({
        variables["x1"]: 3,
        variables["var_name"]: 4,
    })
    assert np.isclose(value, 8.0)


def test_total_degree_start_system_random_state_is_reproducible():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**2 - 1, y**2 - 1])

    start_1, solutions_1 = generate_total_degree_start_system(
        system, [x, y], random_state=123
    )
    start_2, solutions_2 = generate_total_degree_start_system(
        system, [x, y], random_state=123
    )
    start_3, solutions_3 = generate_total_degree_start_system(
        system, [x, y], random_state=456
    )

    assert repr(start_1) == repr(start_2)
    np.testing.assert_allclose(np.asarray(solutions_1), np.asarray(solutions_2))
    assert repr(start_1) != repr(start_3)
    assert not np.allclose(np.asarray(solutions_1), np.asarray(solutions_3))


def test_generic_slice_random_state_is_reproducible():
    x, y = polyvar("x", "y")

    slice_1 = generate_generic_slice(2, [x, y], random_state=99)
    slice_2 = generate_generic_slice(2, [x, y], random_state=99)
    slice_3 = generate_generic_slice(2, [x, y], random_state=100)

    assert repr(slice_1) == repr(slice_2)
    assert repr(slice_1) != repr(slice_3)


def test_regular_solution_is_not_marked_singular_when_endgame_enabled():
    x = polyvar("x")
    solutions = solve(PolynomialSystem([x - 1]), use_endgame=True, random_state=0)

    assert len(solutions) == 1
    assert not solutions[0].is_singular
    assert abs(solutions[0].values[x] - 1) < 1e-8
