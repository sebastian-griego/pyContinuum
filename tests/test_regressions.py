import json
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import pycontinuum.endgame as endgame_module
import pycontinuum.monodromy as monodromy_module
import pycontinuum.parameter_homotopy as parameter_homotopy_module
import pycontinuum.solver as solver_module
import pycontinuum.tracking as tracking_module
import pycontinuum.witness_set as witness_set_module
from pycontinuum import (
    Monomial,
    Polynomial,
    PolynomialSystem,
    Solution,
    SolutionSet,
    Variable,
    WitnessSet,
    compute_witness_superset,
    polyvar,
    solve,
)
from pycontinuum.start_systems import (
    generate_total_degree_solutions,
    generate_total_degree_start_system,
)
from pycontinuum.solver import (
    _backward_error_norm,
    _deduplicate_solutions,
    _integer_determinant,
    _is_singular,
    _residual_norm,
    _scaled_residual_norm,
    _solution_values_from_complete_coordinates,
)
from pycontinuum.tracking import (
    check_singularity,
    compute_tangent,
    homotopy_function,
    homotopy_jacobian,
    track_paths,
)
from pycontinuum.utils import (
    evaluate_backward_error_at_point,
    evaluate_equation_scaled_jacobian_at_point,
    evaluate_jacobian_at_point,
    evaluate_jacobian_polynomials,
    evaluate_scaled_jacobian_at_point,
    evaluate_scaled_system_at_point,
    evaluate_system_at_point,
    newton_corrector,
    newton_corrector_numeric,
    solve_linear_system,
    _scaled_euclidean_norm,
)
from pycontinuum.witness_set import generate_generic_slice


def test_polyvar_rejects_invalid_names():
    with pytest.raises(ValueError, match="At least one"):
        polyvar()
    with pytest.raises(TypeError, match="Variable name must be a string"):
        polyvar(1)
    with pytest.raises(TypeError, match="Variable name must be a string"):
        polyvar([])
    with pytest.raises(TypeError, match="Variable name must be a string"):
        polyvar("x", [])
    with pytest.raises(ValueError, match="valid identifier"):
        polyvar("")
    with pytest.raises(ValueError, match="valid identifier"):
        polyvar("bad name")
    with pytest.raises(ValueError, match="valid identifier"):
        polyvar("1x")
    with pytest.raises(ValueError, match="valid identifier"):
        polyvar("x+y")
    with pytest.raises(ValueError, match="valid identifier"):
        Variable("x-y")
    with pytest.raises(ValueError, match="Variable names must be unique.*x"):
        polyvar("x", "x")
    with pytest.raises(ValueError, match="Variable names must be unique.*x"):
        solver_module.polyvar("x", "y", "x")


def test_polynomial_exponents_must_be_non_negative_integers():
    x = polyvar("x")

    with pytest.raises(ValueError, match="non-negative integer"):
        x ** -1
    with pytest.raises(ValueError, match="non-negative integer"):
        x ** 1.5
    with pytest.raises(ValueError, match="non-negative integer"):
        (x + 1) ** True
    with pytest.raises(ValueError, match="non-negative"):
        Monomial({x: -1})


def test_polynomial_coefficients_must_be_numeric_nonboolean_and_finite():
    x = polyvar("x")

    with pytest.raises(TypeError, match="Monomial coefficient must be a numeric"):
        Monomial({x: 1}, coefficient=True)
    with pytest.raises(ValueError, match="Monomial coefficient must be finite"):
        Monomial({x: 1}, coefficient=float("nan"))
    with pytest.raises(TypeError, match="Unsupported term type"):
        Polynomial([False])
    with pytest.raises(ValueError, match="Monomial coefficient must be finite"):
        Polynomial([float("inf")])
    with pytest.raises(TypeError, match="Unsupported equation type"):
        PolynomialSystem([True])
    with pytest.raises(TypeError):
        x + True
    with pytest.raises(TypeError):
        x - False
    with pytest.raises(TypeError):
        True * x


def test_polynomial_constructors_reject_noniterable_inputs_clearly():
    with pytest.raises(TypeError, match="Monomial variables must be a mapping"):
        Monomial([])
    with pytest.raises(TypeError, match="Monomial variables must be a mapping"):
        Monomial(None)
    with pytest.raises(TypeError, match="Polynomial terms must be an iterable"):
        Polynomial(None)
    with pytest.raises(TypeError, match="Polynomial terms must be an iterable"):
        Polynomial("x")
    with pytest.raises(TypeError, match="PolynomialSystem equations must be an iterable"):
        PolynomialSystem(None)
    with pytest.raises(TypeError, match="PolynomialSystem equations must be an iterable"):
        PolynomialSystem("x")


def test_polynomial_preserves_tiny_nonzero_coefficients():
    x = polyvar("x")

    tiny = Polynomial([Monomial({x: 1}, coefficient=1e-20)])
    scaled = 1e-20 * (x - 1)

    assert tiny.degree() == 1
    assert tiny.variables() == {x}
    assert tiny.evaluate({x: 3}) == pytest.approx(3e-20)
    assert scaled.degree() == 1
    assert scaled.variables() == {x}
    assert scaled.evaluate({x: 1}) == 0
    assert scaled.evaluate({x: 2}) == pytest.approx(1e-20)


def test_variable_self_multiplication_builds_quadratic():
    x = polyvar("x")

    squared = x * x

    assert squared.variables[x] == 2
    assert squared.degree() == 2
    assert squared.evaluate({x: 3}) == 9

    solutions = solve(PolynomialSystem([x * x - 4]), random_state=0)
    roots = sorted(round(solution.values[x].real, 10) for solution in solutions)
    assert roots == [-2.0, 2.0]


def test_unary_negation_for_polynomial_objects():
    x, y = polyvar("x", "y")

    negated_variable = -x
    negated_monomial = -(x * y)
    negated_polynomial = -(x + y - 1)

    assert negated_variable.evaluate({x: 2}) == -2
    assert negated_monomial.evaluate({x: 2, y: 3}) == -6
    assert negated_polynomial.evaluate({x: 2, y: 3}) == -4


def test_monomial_variables_attribute_is_dict_like_and_callable():
    x, y = polyvar("x", "y")
    monomial = x * x * y
    constant = Monomial({}, coefficient=3)

    assert monomial.variables[x] == 2
    assert monomial.variables[y] == 1
    assert monomial.variables() == {x, y}
    assert constant.variables == {}
    assert constant.variables() == set()


def test_monomial_repr_uses_canonical_variable_order():
    x, y = polyvar("x", "y")

    assert repr(x * y) == "x*y"
    assert repr(y * x) == "x*y"
    assert repr((y * x) * x) == "x^2*y"
    assert repr(3 * y * x) == "3*x*y"


def test_derivative_and_jacobian_reject_invalid_variables():
    x, y = polyvar("x", "y")
    monomial = x * y
    polynomial = monomial + y
    system = PolynomialSystem([polynomial])

    with pytest.raises(TypeError, match="var must be a Variable"):
        monomial.partial_derivative("x")
    with pytest.raises(TypeError, match="var must be a Variable"):
        polynomial.partial_derivative("x")
    with pytest.raises(TypeError, match="vars_list must be an iterable"):
        polynomial.jacobian("x")
    with pytest.raises(TypeError, match=r"vars_list\[1\] must be a Variable"):
        system.jacobian([x, "y"])
    with pytest.raises(ValueError, match="duplicate variable"):
        system.jacobian([x, x])

    assert repr(polynomial.jacobian((x, y))) == "[[y, x + 1]]"


def test_polynomial_objects_support_scalar_division():
    x, y = polyvar("x", "y")

    divided_variable = x / 2
    divided_monomial = (x * y) / 2
    divided_polynomial = (x + y - 1) / 2

    assert divided_variable.evaluate({x: 4}) == 2
    assert divided_monomial.evaluate({x: 2, y: 3}) == 3
    assert divided_polynomial.evaluate({x: 2, y: 3}) == 2

    solutions = solve(PolynomialSystem([(x**2 - 4) / 2]), random_state=0)
    roots = sorted(round(solution.values[x].real, 10) for solution in solutions)
    assert roots == [-2.0, 2.0]


def test_polynomial_scalar_division_rejects_invalid_divisors():
    x = polyvar("x")

    with pytest.raises(ZeroDivisionError, match="division by zero"):
        x / 0
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        (x + 1) / 0
    with pytest.raises(TypeError):
        x / True
    with pytest.raises(TypeError):
        (x + 1) / "2"
    with pytest.raises(TypeError):
        1 / x


def test_parse_preserves_alphanumeric_and_underscore_variable_names():
    variables = {}
    polynomial = Polynomial.parse("x1 + 2*var_name - 3", variables=variables)

    assert set(variables) == {"x1", "var_name"}
    value = polynomial.evaluate({
        variables["x1"]: 3,
        variables["var_name"]: 4,
    })
    assert np.isclose(value, 8.0)


def test_polynomial_parse_validates_reusable_variable_map():
    x = polyvar("x")
    variables = {"x": x}

    polynomial = Polynomial.parse("x + y", variables=variables)

    assert variables["x"] is x
    assert isinstance(variables["y"], Variable)
    assert variables["y"].name == "y"
    assert repr(polynomial) == "x + y"

    with pytest.raises(TypeError, match="variables must be a mutable mapping"):
        Polynomial.parse("x", variables=[])
    with pytest.raises(TypeError, match="variables keys must be strings"):
        Polynomial.parse("x", variables={1: x})
    with pytest.raises(TypeError, match="variables values must be Variable objects"):
        Polynomial.parse("x", variables={"x": object()})
    with pytest.raises(ValueError, match="keys must match"):
        Polynomial.parse("x", variables={"x": polyvar("y")})


def test_polynomial_parse_exponents_are_exact_integers():
    polynomial = Polynomial.parse("x^9007199254740993")
    variable = next(iter(polynomial.variables()))

    assert polynomial.terms[0].variables[variable] == 9007199254740993
    assert Polynomial.parse("x^1e3").degree() == 1000
    variables = {}
    python_power = Polynomial.parse("x**3 + 2*x*y**2", variables=variables)
    assert python_power.degree() == 3
    assert python_power.evaluate({variables["x"]: 2, variables["y"]: 3}) == 44

    with pytest.raises(ValueError, match="non-negative integers"):
        Polynomial.parse("x^1.5")
    with pytest.raises(ValueError, match=r"Expected integer exponent after '\*\*'"):
        Polynomial.parse("x**")


def test_polynomial_parse_accepts_python_power_and_imaginary_literals():
    variables = {}

    polynomial = Polynomial.parse("x**3 + 2j*x*y**2 + 1 - 3j", variables=variables)

    assert set(variables) == {"x", "y"}
    assert polynomial.degree() == 3
    value = polynomial.evaluate({variables["x"]: 2, variables["y"]: 3})
    assert value == 9 + 33j


def test_polynomial_parse_keeps_plain_j_as_variable_name():
    variables = {}

    polynomial = Polynomial.parse("j + 2j*x", variables=variables)

    assert set(variables) == {"j", "x"}
    value = polynomial.evaluate({variables["j"]: 5, variables["x"]: 3})
    assert value == 5 + 6j


def test_polynomial_parse_rejects_oversized_imaginary_literals():
    with pytest.raises(ValueError, match="too large"):
        Polynomial.parse("1e309j*x")


def test_polynomial_parse_accepts_parenthesized_polynomial_products():
    variables = {}

    polynomial = Polynomial.parse(
        "(x + 1)*(y - 2) + 2(x - y)",
        variables=variables,
    )

    assert set(variables) == {"x", "y"}
    value = polynomial.evaluate({variables["x"]: 3, variables["y"]: 5})
    assert value == 8
    assert polynomial.degree() == 2


def test_polynomial_parse_accepts_parenthesized_polynomial_powers():
    variables = {}

    polynomial = Polynomial.parse("-(x + 1)**2 + (y - 2)^3", variables=variables)

    assert polynomial.degree() == 3
    value = polynomial.evaluate({variables["x"]: 2, variables["y"]: 4})
    assert value == -1


def test_polynomial_parse_rejects_unmatched_parentheses():
    with pytest.raises(ValueError, match="Expected"):
        Polynomial.parse("(x + 1")
    with pytest.raises(ValueError, match="Unexpected token"):
        Polynomial.parse("x + 1)")


def test_polynomial_system_parse_accepts_equations_and_solves():
    variables = {}

    system = PolynomialSystem.parse(
        "x^2 + y^2 = 2; x - y = 0",
        variables=variables,
    )
    x = variables["x"]
    y = variables["y"]

    assert len(system.equations) == 2
    assert system.ordered_variables() == [x, y]
    assert system.evaluate({x: 1, y: 1}) == [0, 0]

    solutions = solve(system, variables=[x, y], random_state=0)

    assert len(solutions) == 2
    roots = sorted(round(solution.values[x].real) for solution in solutions)
    assert roots == [-1, 1]
    for solution in solutions:
        assert abs(solution.values[x] - solution.values[y]) < 1e-10


def test_polynomial_system_parse_accepts_newlines_and_iterables():
    variables = {}

    newline_system = PolynomialSystem.parse(
        """
        x + y - 1
        (x - 1)*(y - 2) = 0
        """,
        variables=variables,
    )
    iterable_system = PolynomialSystem.parse(
        ["x + y - 1", "(x - 1)*(y - 2) = 0"],
        variables=variables,
    )

    x = variables["x"]
    y = variables["y"]
    values = {x: 1, y: 0}
    assert newline_system.evaluate(values) == [0, 0]
    assert iterable_system.evaluate(values) == [0, 0]


def test_polynomial_system_parse_validates_inputs():
    x = polyvar("x")

    with pytest.raises(TypeError, match="variables must be a mutable mapping"):
        PolynomialSystem.parse("x", variables=[])
    with pytest.raises(TypeError, match="equations must be a string"):
        PolynomialSystem.parse(1)
    with pytest.raises(ValueError, match="non-empty string"):
        PolynomialSystem.parse("")
    with pytest.raises(ValueError, match="must contain equations"):
        PolynomialSystem.parse([])
    with pytest.raises(ValueError, match="non-empty string"):
        PolynomialSystem.parse(["x", ""])
    with pytest.raises(ValueError, match="at most one"):
        PolynomialSystem.parse("x = y = 1")
    with pytest.raises(ValueError, match="both sides"):
        PolynomialSystem.parse("x = ")
    with pytest.raises(ValueError, match="Expected"):
        PolynomialSystem.parse("(x + 1 = 0")
    with pytest.raises(ValueError, match="keys must match"):
        PolynomialSystem.parse("x", variables={"x": polyvar("y")})

    system = PolynomialSystem.parse("x = 1", variables={"x": x})
    assert x in system.variables()


def test_solve_accepts_parseable_system_string():
    solutions = solve("x^2 = 1", random_state=0)

    assert len(solutions) == 2
    x = solutions.system.ordered_variables()[0]
    roots = sorted(round(solution.values[x].real) for solution in solutions)
    assert roots == [-1, 1]
    assert all(abs(solution.values[x].imag) < 1e-10 for solution in solutions)


def test_solve_accepts_system_string_with_explicit_variables():
    x, y = polyvar("x", "y")

    solutions = solve(
        "x^2 + y^2 = 2; x - y = 0",
        variables=[x, y],
        random_state=0,
    )

    assert len(solutions) == 2
    for solution in solutions:
        assert x in solution.values
        assert y in solution.values
        assert abs(solution.values[x] - solution.values[y]) < 1e-10


def test_solve_string_target_rejects_malformed_explicit_variables():
    x = polyvar("x")

    with pytest.raises(TypeError, match=r"variables\[1\] must be a Variable"):
        solve("x - 1", variables=[x, "extra"], random_state=0)
    with pytest.raises(ValueError, match="duplicate variable"):
        solve("x - 1", variables=[x, x], random_state=0)


def test_solve_accepts_iterable_of_equation_strings():
    x = polyvar("x")

    solutions = solve(["x^2 - 1"], variables=[x], random_state=0)

    assert len(solutions) == 2
    roots = sorted(round(solution.values[x].real) for solution in solutions)
    assert roots == [-1, 1]


def test_solve_accepts_single_polynomial_equation():
    x = polyvar("x")

    solutions = solve(x**2 - 1, random_state=0)

    assert len(solutions) == 2
    roots = sorted(round(solution.values[x].real) for solution in solutions)
    assert roots == [-1, 1]
    assert solutions.system.equations[0].degree() == 2


def test_solve_accepts_single_monomial_and_variable_equations():
    x = polyvar("x")

    monomial_solution = solve(x * x, random_state=0)
    variable_solution = solve(x, random_state=0)

    assert len(monomial_solution) == 1
    assert len(variable_solution) == 1
    assert abs(monomial_solution[0].values[x]) < 1e-8
    assert abs(variable_solution[0].values[x]) < 1e-12


def test_solve_accepts_numeric_constant_equations():
    inconsistent = solve(1, random_state=0)
    zero = solve(0, random_state=0)

    assert len(inconsistent) == 0
    assert len(zero) == 1
    assert zero[0].values == {}


def test_integral_coefficients_remain_exact():
    x = polyvar("x")
    huge = 10**400

    parsed = Polynomial.parse("9007199254740993*x")
    parsed_decimal = Polynomial.parse("9007199254740993.0*x")
    parsed_scientific = Polynomial.parse("9007199254740993e0*x")
    parsed_huge_scientific = Polynomial.parse("1.5e309*x")
    parsed_variable = next(iter(parsed.variables()))
    direct = Monomial({x: 1}, coefficient=huge)

    assert parsed.terms[0].coefficient == 9007199254740993
    assert isinstance(parsed.terms[0].coefficient, int)
    assert parsed_decimal.terms[0].coefficient == 9007199254740993
    assert isinstance(parsed_decimal.terms[0].coefficient, int)
    assert parsed_scientific.terms[0].coefficient == 9007199254740993
    assert isinstance(parsed_scientific.terms[0].coefficient, int)
    assert parsed_huge_scientific.terms[0].coefficient == 15 * 10**308
    assert isinstance(parsed_huge_scientific.terms[0].coefficient, int)
    assert parsed.evaluate({parsed_variable: 2}) == 18014398509481986
    assert direct.coefficient == huge
    assert isinstance(direct.coefficient, int)


def test_exact_constructor_coefficients_remain_exact():
    x = polyvar("x")

    decimal_monomial = Monomial({x: 1}, coefficient=Decimal("1.5e309"))
    fraction_monomial = Monomial({x: 1}, coefficient=Fraction(10**400, 1))
    decimal_constant = Polynomial([Decimal("9007199254740993.0")])

    assert decimal_monomial.coefficient == 15 * 10**308
    assert isinstance(decimal_monomial.coefficient, int)
    assert fraction_monomial.coefficient == 10**400
    assert isinstance(fraction_monomial.coefficient, int)
    assert decimal_constant.terms[0].coefficient == 9007199254740993
    assert isinstance(decimal_constant.terms[0].coefficient, int)


def test_polynomial_parse_rejects_inexact_numeric_underflow():
    with pytest.raises(ValueError, match="too small"):
        Polynomial.parse("1e-400*x")


def test_polynomial_constructors_reject_inexact_numeric_underflow():
    x = polyvar("x")

    with pytest.raises(ValueError, match="too small"):
        Monomial({x: 1}, coefficient=Decimal("1e-400"))
    with pytest.raises(ValueError, match="too small"):
        Monomial({x: 1}, coefficient=Fraction(1, 10**400))
    with pytest.raises(ValueError, match="too small"):
        Polynomial([Decimal("1e-400")])


def test_exact_scalar_arithmetic_rejects_inexact_underflow():
    x = polyvar("x")
    tiny = Fraction(1, 10**400)
    huge = Fraction(10**400, 1)

    with pytest.raises(ValueError, match="too small"):
        x / huge
    with pytest.raises(ValueError, match="too small"):
        (x * x) / huge
    with pytest.raises(ValueError, match="too small"):
        (x + 1) / huge
    with pytest.raises(ValueError, match="too small"):
        Monomial({x: 1}, coefficient=0.5) * tiny
    with pytest.raises(ValueError, match="too small"):
        tiny * Monomial({x: 1}, coefficient=0.5)
    with pytest.raises(ValueError, match="too small"):
        Monomial({x: 1}, coefficient=1j) * tiny
    with pytest.raises(ValueError, match="too small"):
        Monomial({x: 1}, coefficient=1j) / huge


def test_exact_scalar_division_preserves_representable_quotients():
    x = polyvar("x")
    huge_int = 10**400

    reciprocal_fraction = x / Fraction(1, huge_int)
    reciprocal_decimal = x / Decimal("1e-400")
    reduced = (huge_int * x) / Fraction(huge_int, 1)
    normalized = Monomial({x: 1}, coefficient=0.5) / Fraction(1, 2)

    assert reciprocal_fraction.coefficient == huge_int
    assert isinstance(reciprocal_fraction.coefficient, int)
    assert reciprocal_decimal.coefficient == huge_int
    assert isinstance(reciprocal_decimal.coefficient, int)
    assert reduced.coefficient == 1
    assert isinstance(reduced.coefficient, int)
    assert normalized.coefficient == 1
    assert isinstance(normalized.coefficient, int)


def test_exact_like_term_combination_preserves_integer_coefficients():
    x = polyvar("x")
    huge_int = 10**400

    polynomial = Polynomial([
        Monomial({x: 1}, coefficient=huge_int),
        Monomial({x: 1}, coefficient=1),
    ])

    assert len(polynomial.terms) == 1
    assert polynomial.terms[0].coefficient == huge_int + 1
    assert isinstance(polynomial.terms[0].coefficient, int)


def test_exact_like_term_combination_rejects_unrepresentable_inexact_sum():
    x = polyvar("x")
    huge_int = 10**400

    with pytest.raises(ValueError, match="too large"):
        Polynomial([
            Monomial({x: 1}, coefficient=huge_int),
            Monomial({x: 1}, coefficient=0.5),
        ])
    with pytest.raises(ValueError, match="too large"):
        Polynomial([
            Monomial({x: 1}, coefficient=huge_int),
            Monomial({x: 1}, coefficient=1j),
        ])


def test_derivative_coefficient_scaling_uses_checked_arithmetic():
    x = polyvar("x")
    huge_int = 10**400

    derivative = Monomial({x: huge_int}, coefficient=0.5).partial_derivative(x)
    assert derivative.coefficient == huge_int // 2
    assert isinstance(derivative.coefficient, int)

    with pytest.raises(ValueError, match="too large"):
        Monomial({x: huge_int}, coefficient=1j).partial_derivative(x)


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


def test_total_degree_start_system_rejects_nonboolean_underdetermined_flag():
    x = polyvar("x")

    with pytest.raises(TypeError, match="allow_underdetermined must be a boolean"):
        generate_total_degree_start_system(
            PolynomialSystem([x - 1]),
            [x],
            allow_underdetermined="yes",
            random_state=0,
        )


def test_total_degree_start_system_rejects_malformed_rng_output():
    class NonfiniteRng:
        def uniform(self, *args, **kwargs):
            return float("nan")

    class VectorRng:
        def uniform(self, *args, **kwargs):
            return np.array([0.0, 1.0])

    x = polyvar("x")
    system = PolynomialSystem([x**2 - 1])

    with pytest.raises(ValueError, match="finite scalar.*total-degree coefficient 0"):
        generate_total_degree_start_system(system, [x], random_state=NonfiniteRng())
    with pytest.raises(ValueError, match="finite scalar.*total-degree coefficient 0"):
        generate_total_degree_start_system(system, [x], random_state=VectorRng())


def test_solve_rejects_malformed_rng_output_for_generated_gamma():
    class VectorRng:
        def uniform(self, *args, **kwargs):
            return np.array([0.0, 1.0])

    x = polyvar("x")

    with pytest.raises(ValueError, match="finite scalar.*gamma"):
        solve(PolynomialSystem([x - 1]), variables=[x], random_state=VectorRng())


def test_total_degree_solution_generation_validates_inputs():
    with pytest.raises(TypeError, match=r"degrees\[0\] must be an integer"):
        generate_total_degree_solutions([True], [1.0 + 0j])
    with pytest.raises(ValueError, match=r"degrees\[0\] must be positive"):
        generate_total_degree_solutions([0], [1.0 + 0j])
    with pytest.raises(TypeError, match=r"c_values\[0\] must be a numeric"):
        generate_total_degree_solutions([1], ["1"])
    with pytest.raises(ValueError, match=r"c_values\[0\] must be finite"):
        generate_total_degree_solutions([1], [float("inf")])
    with pytest.raises(ValueError, match=r"c_values\[0\] must be nonzero"):
        generate_total_degree_solutions([2], [0.0 + 0j])
    with pytest.raises(ValueError, match="same length"):
        generate_total_degree_solutions([1, 2], [1.0 + 0j])
    with pytest.raises(TypeError, match="max_solutions must be an integer"):
        generate_total_degree_solutions([1], [1.0 + 0j], max_solutions=True)
    with pytest.raises(ValueError, match="max_solutions must be nonnegative"):
        generate_total_degree_solutions([1], [1.0 + 0j], max_solutions=-1)
    with pytest.raises(
        ValueError,
        match=r"would generate 6 solution\(s\).*max_solutions=5",
    ):
        generate_total_degree_solutions(
            [2, 3],
            [1.0 + 0j, 1.0 + 0j],
            max_solutions=5,
        )

    assert generate_total_degree_solutions([], []) == [[]]


def test_total_degree_solution_generation_handles_many_degree_one_variables():
    degrees = [1] * 1200
    c_values = [1.0 + 0j] * len(degrees)

    solutions = generate_total_degree_solutions(
        degrees,
        c_values,
        max_solutions=1,
    )

    assert len(solutions) == 1
    assert len(solutions[0]) == len(degrees)
    np.testing.assert_allclose(np.asarray(solutions[0]), np.ones(len(degrees)))


def test_total_degree_start_system_honors_solution_cap():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**2 - 1, y**2 - 1])

    with pytest.raises(
        ValueError,
        match=r"total-degree start system would generate 4 solution\(s\).*"
        "max_solutions=3",
    ):
        generate_total_degree_start_system(
            system,
            [x, y],
            random_state=0,
            max_solutions=3,
        )


def test_generic_slice_random_state_is_reproducible():
    x, y = polyvar("x", "y")

    slice_1 = generate_generic_slice(2, [x, y], random_state=99)
    slice_2 = generate_generic_slice(2, [x, y], random_state=99)
    slice_3 = generate_generic_slice(2, [x, y], random_state=100)

    assert repr(slice_1) == repr(slice_2)
    assert repr(slice_1) != repr(slice_3)


def test_generic_slice_rejects_invalid_dimension_and_variables():
    x = polyvar("x")

    with pytest.raises(ValueError, match="dimension must be non-negative"):
        generate_generic_slice(-1, [x])
    with pytest.raises(TypeError, match="dimension must be an integer"):
        generate_generic_slice(True, [x])
    with pytest.raises(ValueError, match="dimension cannot exceed"):
        generate_generic_slice(2, [x])
    with pytest.raises(TypeError, match=r"variables\[1\] must be a Variable"):
        generate_generic_slice(1, [x, "bad"])


def test_generic_slice_rejects_malformed_standard_normal_output():
    class WrongShapeRng:
        def uniform(self, *args, **kwargs):
            return 0.0

        def standard_normal(self, size=None):
            return np.array([0.0])

    class NonfiniteRng:
        def uniform(self, *args, **kwargs):
            return 0.0

        def standard_normal(self, size=None):
            if size is None:
                return float("nan")
            return np.zeros(size)

    x, y = polyvar("x", "y")

    with pytest.raises(
        ValueError,
        match=r"standard_normal.*shape \(2,\).*generic slice 0 coefficients real",
    ):
        generate_generic_slice(1, [x, y], random_state=WrongShapeRng())
    with pytest.raises(
        ValueError,
        match=r"standard_normal.*shape \(\).*generic slice 0 constant real",
    ):
        generate_generic_slice(1, [x, y], random_state=NonfiniteRng())


def test_regular_solution_is_not_marked_singular_when_endgame_enabled():
    x = polyvar("x")
    solutions = solve(PolynomialSystem([x - 1]), use_endgame=True, random_state=0)

    assert len(solutions) == 1
    assert not solutions[0].is_singular
    assert abs(solutions[0].values[x] - 1) < 1e-8


def test_homotopy_tangent_uses_actual_derivative_by_default():
    x = polyvar("x")
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 5])

    tangent = compute_tangent(
        start_system,
        target_system,
        np.array([3.0 + 0j]),
        0.5,
        [x],
        gamma=1.0 + 0j,
    )
    unit_tangent = compute_tangent(
        start_system,
        target_system,
        np.array([3.0 + 0j]),
        0.5,
        [x],
        gamma=1.0 + 0j,
        normalize=True,
    )

    np.testing.assert_allclose(tangent, np.array([-4.0 + 0j]))
    np.testing.assert_allclose(unit_tangent, np.array([-1.0 + 0j]))


def test_homotopy_helpers_use_scaled_fallback_for_huge_coefficients():
    x = polyvar("x")
    start_system = PolynomialSystem([x - 2])
    target_system = PolynomialSystem([(10**400) * (x - 1)])
    point = np.array([1.5 + 0j])

    value = homotopy_function(
        start_system,
        target_system,
        point,
        0.5,
        [x],
        gamma=1.0 + 0j,
    )
    jacobian = homotopy_jacobian(
        start_system,
        target_system,
        point,
        0.5,
        [x],
        gamma=1.0 + 0j,
    )
    tangent = compute_tangent(
        start_system,
        target_system,
        point,
        0.5,
        [x],
        gamma=1.0 + 0j,
    )

    assert np.all(np.isfinite(value))
    assert np.all(np.isfinite(jacobian))
    assert np.all(np.isfinite(tangent))
    np.testing.assert_allclose(value, np.array([0.125 + 0j]))
    np.testing.assert_allclose(jacobian, np.array([[0.75 + 0j]]))
    np.testing.assert_allclose(tangent, np.array([1.0 + 0j]))


def test_homotopy_helpers_accept_coordinate_records():
    x, y = polyvar("x", "y")
    start_system = PolynomialSystem([x - 1, y - 1])
    target_system = PolynomialSystem([x - 3, y - 5])

    class SolutionLike:
        def __init__(self, values):
            self.values = values

    mapping_point = {"y": 3.0 + 0j, x: 2.0 + 0j}
    solution_point = Solution({x: 2.0 + 0j, y: 3.0 + 0j}, residual=0.0)
    solution_like = SolutionLike({"x": 2.0 + 0j, "y": 3.0 + 0j})

    np.testing.assert_allclose(
        homotopy_function(
            start_system,
            target_system,
            mapping_point,
            0.5,
            [x, y],
            gamma=1.0 + 0j,
        ),
        np.array([1.0 / 3.0 + 0j, 0.8 + 0j]),
    )
    np.testing.assert_allclose(
        homotopy_jacobian(
            start_system,
            target_system,
            solution_point,
            0.5,
            [x, y],
            gamma=1.0 + 0j,
        ),
        np.array([[2.0 / 3.0, 0.0], [0.0, 0.6]], dtype=complex),
    )
    np.testing.assert_allclose(
        compute_tangent(
            start_system,
            target_system,
            solution_like,
            0.5,
            [x, y],
            gamma=1.0 + 0j,
        ),
        np.array([-2.0 + 0j, -4.0 + 0j]),
    )

    with pytest.raises(ValueError, match=r"point is missing coordinate\(s\): y"):
        homotopy_function(
            start_system,
            target_system,
            {"x": 2.0 + 0j},
            0.5,
            [x, y],
            gamma=1.0 + 0j,
        )


def test_homotopy_helpers_reject_invalid_direct_inputs():
    x, y = polyvar("x", "y")
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    with pytest.raises(ValueError, match="same number of equations"):
        homotopy_function(
            start_system,
            PolynomialSystem([x - 2, x + 1]),
            np.array([1.0 + 0j]),
            0.5,
            [x],
        )
    with pytest.raises(TypeError, match="t must be a number"):
        homotopy_jacobian(
            start_system,
            target_system,
            np.array([1.0 + 0j]),
            "0.5",
            [x],
        )
    with pytest.raises(TypeError, match="gamma must be a complex number"):
        compute_tangent(
            start_system,
            target_system,
            np.array([1.0 + 0j]),
            0.5,
            [x],
            gamma=np.bool_(True),
        )
    with pytest.raises(ValueError, match="missing homotopy variable.*y"):
        compute_tangent(
            start_system,
            PolynomialSystem([x + y - 2]),
            np.array([1.0 + 0j]),
            0.5,
            [x],
        )
    with pytest.raises(ValueError, match="not used by homotopy.*y"):
        homotopy_function(
            start_system,
            target_system,
            np.array([1.0 + 0j, 0.0 + 0j]),
            0.5,
            [x, y],
        )


def test_check_singularity_rejects_invalid_direct_options():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    with pytest.raises(TypeError, match="threshold must be a number"):
        check_singularity(system, np.array([1.0 + 0j]), [x], "1e3")
    with pytest.raises(ValueError, match="threshold must be positive"):
        check_singularity(system, np.array([1.0 + 0j]), [x], 0.0)
    with pytest.raises(TypeError, match="verbose must be a boolean"):
        check_singularity(
            system,
            np.array([1.0 + 0j]),
            [x],
            1e3,
            verbose="yes",
        )
    with pytest.raises(TypeError, match="debug must be a boolean"):
        check_singularity(
            system,
            np.array([1.0 + 0j]),
            [x],
            1e3,
            debug="yes",
        )


def test_track_paths_parallel_matches_sequential_order():
    x = polyvar("x")
    start_system = PolynomialSystem([x**2 - 1])
    target_system = PolynomialSystem([x**2 - 4])
    start_solutions = [[1.0 + 0j], [-1.0 + 0j]]

    sequential_end, sequential_info = track_paths(
        start_system,
        target_system,
        start_solutions,
        [x],
        gamma=1.0 + 0j,
        use_endgame=False,
        n_jobs=1,
    )
    parallel_end, parallel_info = track_paths(
        start_system,
        target_system,
        start_solutions,
        [x],
        gamma=1.0 + 0j,
        use_endgame=False,
        n_jobs=2,
    )

    np.testing.assert_allclose(np.asarray(parallel_end), np.asarray(sequential_end))
    assert [info["path_index"] for info in parallel_info] == [0, 1]
    assert [info["success"] for info in parallel_info] == [
        info["success"] for info in sequential_info
    ]
    assert all(info["start_residual"] == 0.0 for info in parallel_info)
    assert all(info["start_residual_limit"] > 0.0 for info in parallel_info)


def test_track_paths_accepts_start_solution_generators():
    x = polyvar("x")
    start_system = PolynomialSystem([x**2 - 1])
    target_system = PolynomialSystem([x**2 - 4])
    start_solutions = ([value + 0j] for value in (1.0, -1.0))

    endpoints, path_info = track_paths(
        start_system,
        target_system,
        start_solutions,
        [x],
        gamma=1.0 + 0j,
        use_endgame=False,
    )

    np.testing.assert_allclose(np.asarray(endpoints), np.array([[2.0], [-2.0]]))
    assert [info["path_index"] for info in path_info] == [0, 1]
    assert all(info["success"] for info in path_info)


def test_track_paths_accepts_mapping_start_solutions():
    x, y = polyvar("x", "y")
    start_system = PolynomialSystem([x - 1, y - 1])
    target_system = PolynomialSystem([x - 2, y - 3])

    endpoints, path_info = track_paths(
        start_system,
        target_system,
        [{"y": 1.0 + 0j, x: 1.0 + 0j}],
        [x, y],
        gamma=1.0 + 0j,
        use_endgame=False,
    )

    np.testing.assert_allclose(
        np.asarray(endpoints),
        np.array([[2.0 + 0j, 3.0 + 0j]]),
        atol=1e-8,
    )
    assert path_info[0]["success"]
    assert path_info[0]["start_residual"] == 0.0


def test_track_paths_records_effective_tracking_options():
    x = polyvar("x")
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    endpoints, path_info = track_paths(
        start_system,
        target_system,
        [[1.0 + 0j]],
        [x],
        tol=1e-9,
        min_step_size=1e-5,
        max_step_size=0.04,
        max_newton_iters=7,
        max_steps=200,
        max_predictor_norm=0.2,
        gamma=1.0 + 0j,
        endgame_start=0.2,
        singularity_threshold=1e4,
        final_singularity_threshold=1e6,
        store_paths=True,
        use_endgame=False,
        predictor="heun",
    )

    info = path_info[0]
    assert info["success"]
    assert info["start_t"] == pytest.approx(1.0)
    assert info["end_t"] == pytest.approx(0.0)
    assert info["direction"] == -1
    assert info["tol"] == pytest.approx(1e-9)
    assert info["min_step_size"] == pytest.approx(1e-5)
    assert info["max_step_size"] == pytest.approx(0.04)
    assert info["initial_step_size"] == pytest.approx(0.04)
    assert info["max_newton_iters"] == 7
    assert info["max_steps"] == 200
    assert info["max_predictor_norm"] == pytest.approx(0.2)
    assert info["gamma"] == 1.0 + 0j
    assert info["endgame_start"] == pytest.approx(0.2)
    assert info["singularity_threshold"] == pytest.approx(1e4)
    assert info["final_singularity_threshold"] == pytest.approx(1e6)
    assert info["store_paths"] is True
    assert info["use_endgame"] is False
    assert info["predictor"] == "heun"
    assert info["path_points"]
    np.testing.assert_allclose(np.asarray(endpoints), np.array([[2.0 + 0j]]))


def test_track_paths_heun_predictor_matches_euler_endpoint_order():
    x = polyvar("x")
    start_system = PolynomialSystem([x**2 - 1])
    target_system = PolynomialSystem([x**2 - 4])
    start_solutions = [[1.0 + 0j], [-1.0 + 0j]]

    euler_end, _ = track_paths(
        start_system,
        target_system,
        start_solutions,
        [x],
        gamma=1.0 + 0j,
        use_endgame=False,
        predictor="euler",
    )
    heun_end, heun_info = track_paths(
        start_system,
        target_system,
        start_solutions,
        [x],
        gamma=1.0 + 0j,
        use_endgame=False,
        predictor="heun",
    )

    np.testing.assert_allclose(np.asarray(heun_end), np.asarray(euler_end))
    assert [info["path_index"] for info in heun_info] == [0, 1]
    assert all(info["predictor"] == "heun" for info in heun_info)
    assert all(info["predictor_fallbacks"] == 0 for info in heun_info)
    assert any(info["max_predictor_correction_norm"] > 0 for info in heun_info)


def test_track_paths_rk4_predictor_matches_euler_endpoint_order():
    x = polyvar("x")
    start_system = PolynomialSystem([x**2 - 1])
    target_system = PolynomialSystem([x**2 - 4])
    start_solutions = [[1.0 + 0j], [-1.0 + 0j]]

    euler_end, _ = track_paths(
        start_system,
        target_system,
        start_solutions,
        [x],
        gamma=1.0 + 0j,
        use_endgame=False,
        predictor="euler",
    )
    rk4_end, rk4_info = track_paths(
        start_system,
        target_system,
        start_solutions,
        [x],
        gamma=1.0 + 0j,
        use_endgame=False,
        predictor="rk4",
    )

    np.testing.assert_allclose(np.asarray(rk4_end), np.asarray(euler_end))
    assert [info["path_index"] for info in rk4_info] == [0, 1]
    assert all(info["predictor"] == "rk4" for info in rk4_info)
    assert all(info["predictor_fallbacks"] == 0 for info in rk4_info)
    assert any(info["max_predictor_correction_norm"] > 0 for info in rk4_info)


def test_solve_accepts_and_records_tracking_options():
    x = polyvar("x")
    system = PolynomialSystem([x - 5])

    solutions = solve(
        system,
        use_endgame=False,
        random_state=0,
        tracking_options={
            "gamma": 1.0 + 0j,
            "max_step_size": 0.02,
            "max_predictor_norm": 0.05,
            "n_jobs": 2,
        },
    )

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 5) < 1e-8
    assert solutions._meta["tracking_options"]["max_step_size"] == 0.02
    assert solutions._meta["tracking_options"]["max_predictor_norm"] == 0.05
    assert solutions._meta["tracking_options"]["gamma"] == 1.0 + 0j
    assert solutions._meta["tracking_options"]["n_jobs"] == 2
    assert not solutions._meta["generated_gamma"]


def test_solve_records_heun_predictor_path_metadata():
    x = polyvar("x")
    target_system = PolynomialSystem([x**2 - 4])
    start_system = PolynomialSystem([x**2 - 1])

    solutions = solve(
        target_system,
        start_system=start_system,
        start_solutions=[[1.0 + 0j], [-1.0 + 0j]],
        tracking_options={"gamma": 1.0 + 0j, "predictor": "heun"},
        use_endgame=False,
        random_state=0,
    )

    assert len(solutions) == 2
    assert solutions._meta["tracking_options"]["predictor"] == "heun"
    assert solutions._meta["path_summary"]["predictors"] == {"heun": 2}
    assert solutions._meta["path_summary"]["predictor_fallbacks"] == 0
    assert solutions._meta["path_summary"]["max_predictor_correction_norm"] > 0
    assert all(solution.path_info["predictor"] == "heun" for solution in solutions)
    assert all(solution.path_info["start_residual"] == 0.0 for solution in solutions)


def test_solve_preserves_effective_tracking_options_in_solution_path_info():
    x = polyvar("x")
    target_system = PolynomialSystem([x**2 - 4])
    start_system = PolynomialSystem([x**2 - 1])

    solutions = solve(
        target_system,
        start_system=start_system,
        start_solutions=[[1.0 + 0j], [-1.0 + 0j]],
        variables=[x],
        tol=1e-9,
        store_paths=True,
        use_endgame=False,
        tracking_options={
            "gamma": 1.0 + 0j,
            "predictor": "rk4",
            "min_step_size": 1e-5,
            "max_step_size": 0.04,
            "max_newton_iters": 7,
            "max_steps": 200,
            "max_predictor_norm": 0.2,
            "endgame_start": 0.2,
            "singularity_threshold": 1e4,
            "final_singularity_threshold": 1e6,
        },
        random_state=0,
    )

    assert len(solutions) == 2
    for solution in solutions:
        info = solution.path_info
        assert info["start_t"] == pytest.approx(1.0)
        assert info["end_t"] == pytest.approx(0.0)
        assert info["direction"] == -1
        assert info["tol"] == pytest.approx(1e-9)
        assert info["min_step_size"] == pytest.approx(1e-5)
        assert info["max_step_size"] == pytest.approx(0.04)
        assert info["initial_step_size"] == pytest.approx(0.04)
        assert info["max_newton_iters"] == 7
        assert info["max_steps"] == 200
        assert info["max_predictor_norm"] == pytest.approx(0.2)
        assert info["gamma"] == {"real": 1.0, "imag": 0.0}
        assert info["endgame_start"] == pytest.approx(0.2)
        assert info["singularity_threshold"] == pytest.approx(1e4)
        assert info["final_singularity_threshold"] == pytest.approx(1e6)
        assert info["store_paths"] is True
        assert info["use_endgame"] is False
        assert info["predictor"] == "rk4"


def test_solve_records_rk4_predictor_path_metadata():
    x = polyvar("x")
    target_system = PolynomialSystem([x**2 - 4])
    start_system = PolynomialSystem([x**2 - 1])

    solutions = solve(
        target_system,
        start_system=start_system,
        start_solutions=[[1.0 + 0j], [-1.0 + 0j]],
        tracking_options={"gamma": 1.0 + 0j, "predictor": "rk4"},
        use_endgame=False,
        random_state=0,
    )

    assert len(solutions) == 2
    assert solutions._meta["tracking_options"]["predictor"] == "rk4"
    assert solutions._meta["path_summary"]["predictors"] == {"rk4": 2}
    assert solutions._meta["path_summary"]["predictor_fallbacks"] == 0
    assert solutions._meta["path_summary"]["max_predictor_correction_norm"] > 0
    assert all(solution.path_info["predictor"] == "rk4" for solution in solutions)


def test_solve_rejects_unknown_tracking_options():
    x = polyvar("x")

    with pytest.raises(ValueError, match="Unknown tracking option"):
        solve(
            PolynomialSystem([x - 1]),
            tracking_options={"stepz": 0.1},
            random_state=0,
        )


def test_solve_rejects_unknown_predictor():
    x = polyvar("x")

    with pytest.raises(ValueError, match="predictor must be"):
        solve(
            PolynomialSystem([x - 1]),
            tracking_options={"predictor": "bogus"},
            random_state=0,
        )


def test_solve_rejects_invalid_parallel_worker_count():
    x = polyvar("x")

    with pytest.raises(ValueError, match="n_jobs must be positive"):
        solve(
            PolynomialSystem([x - 1]),
            tracking_options={"n_jobs": 0},
            random_state=0,
        )


@pytest.mark.parametrize(
    ("tracking_options", "error_type", "message"),
    [
        ({"max_steps": True}, TypeError, "max_steps must be an integer"),
        ({"max_steps": "10"}, TypeError, "max_steps must be an integer"),
        ({"max_steps": 0}, ValueError, "max_steps must be positive"),
        (
            {"max_newton_iters": True},
            TypeError,
            "max_newton_iters must be an integer",
        ),
        (
            {"max_newton_iters": "10"},
            TypeError,
            "max_newton_iters must be an integer",
        ),
        (
            {"max_newton_iters": 0},
            ValueError,
            "max_newton_iters must be positive",
        ),
    ],
)
def test_solve_rejects_invalid_integer_tracking_options(
    tracking_options,
    error_type,
    message,
):
    x = polyvar("x")

    with pytest.raises(error_type, match=message):
        solve(
            PolynomialSystem([x - 1]),
            tracking_options=tracking_options,
            random_state=0,
        )


@pytest.mark.parametrize(
    ("gamma", "error_type", "message"),
    [
        (0.0, ValueError, "gamma must be finite and nonzero"),
        (complex(float("nan"), 0.0), ValueError, "gamma must be finite and nonzero"),
        (True, TypeError, "gamma must be a complex number"),
        (np.bool_(True), TypeError, "gamma must be a complex number"),
        ("not-a-gamma", TypeError, "gamma must be a complex number"),
    ],
)
def test_solve_rejects_invalid_gamma_tracking_option(
    gamma, error_type, message
):
    x = polyvar("x")

    with pytest.raises(error_type, match=message):
        solve(
            PolynomialSystem([x - 1]),
            tracking_options={"gamma": gamma},
            random_state=0,
        )


@pytest.mark.parametrize(
    ("tracking_options", "error_type", "message"),
    [
        ({"min_step_size": True}, TypeError, "min_step_size must be a number"),
        ({"max_step_size": "0.05"}, TypeError, "max_step_size must be a number"),
        (
            {"max_predictor_norm": True},
            TypeError,
            "max_predictor_norm must be a number",
        ),
        (
            {"max_predictor_norm": float("nan")},
            ValueError,
            "max_predictor_norm cannot be NaN",
        ),
        ({"endgame_start": "0.1"}, TypeError, "endgame_start must be a number"),
        ({"singularity_threshold": 0.0}, ValueError, "singularity_threshold must be positive"),
        (
            {"final_singularity_threshold": "1e8"},
            TypeError,
            "final_singularity_threshold must be a number",
        ),
        (
            {"min_step_size": 0.1, "max_step_size": 0.05},
            ValueError,
            "min_step_size cannot exceed max_step_size",
        ),
    ],
)
def test_solve_rejects_invalid_float_tracking_options(
    tracking_options,
    error_type,
    message,
):
    x = polyvar("x")

    with pytest.raises(error_type, match=message):
        solve(
            PolynomialSystem([x - 1]),
            tracking_options=tracking_options,
            random_state=0,
        )


def test_track_paths_rejects_invalid_gamma():
    x = polyvar("x")

    with pytest.raises(ValueError, match="gamma must be finite and nonzero"):
        track_paths(
            start_system=PolynomialSystem([x - 1]),
            target_system=PolynomialSystem([x - 2]),
            start_solutions=[[1.0 + 0j]],
            variables=[x],
            gamma=0.0,
        )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"min_step_size": True}, TypeError, "min_step_size must be a number"),
        ({"min_step_size": "small"}, TypeError, "min_step_size must be a number"),
        ({"min_step_size": "0.05"}, TypeError, "min_step_size must be a number"),
        ({"max_step_size": "0.05"}, TypeError, "max_step_size must be a number"),
        ({"max_step_size": float("inf")}, ValueError, "max_step_size must be finite"),
        (
            {"max_predictor_norm": True},
            TypeError,
            "max_predictor_norm must be a number",
        ),
        (
            {"max_predictor_norm": "wide"},
            TypeError,
            "max_predictor_norm must be a number",
        ),
        (
            {"max_predictor_norm": "0.1"},
            TypeError,
            "max_predictor_norm must be a number",
        ),
        (
            {"max_predictor_norm": float("nan")},
            ValueError,
            "max_predictor_norm cannot be NaN",
        ),
        ({"endgame_start": "late"}, TypeError, "endgame_start must be a number"),
        ({"endgame_start": "0.1"}, TypeError, "endgame_start must be a number"),
        (
            {"singularity_threshold": False},
            TypeError,
            "singularity_threshold must be a number",
        ),
        (
            {"singularity_threshold": "1e3"},
            TypeError,
            "singularity_threshold must be a number",
        ),
        (
            {"final_singularity_threshold": "large"},
            TypeError,
            "final_singularity_threshold must be a number",
        ),
    ],
)
def test_track_paths_rejects_invalid_float_parameters(
    kwargs,
    error_type,
    message,
):
    x = polyvar("x")

    with pytest.raises(error_type, match=message):
        track_paths(
            start_system=PolynomialSystem([x - 1]),
            target_system=PolynomialSystem([x - 2]),
            start_solutions=[[1.0 + 0j]],
            variables=[x],
            gamma=1.0 + 0j,
            **kwargs,
        )


def test_track_paths_rejects_malformed_variable_list():
    x = polyvar("x")

    with pytest.raises(TypeError, match=r"variables\[0\] must be a Variable"):
        track_paths(
            start_system=PolynomialSystem([x - 1]),
            target_system=PolynomialSystem([x - 2]),
            start_solutions=[[1.0 + 0j]],
            variables=["x"],
            gamma=1.0 + 0j,
        )


def test_track_paths_rejects_missing_homotopy_variable():
    x, y = polyvar("x", "y")

    with pytest.raises(ValueError, match="missing homotopy variable.*y"):
        track_paths(
            start_system=PolynomialSystem([x - 1]),
            target_system=PolynomialSystem([x + y - 2]),
            start_solutions=[[1.0 + 0j]],
            variables=[x],
            gamma=1.0 + 0j,
        )


def test_track_paths_rejects_extra_variables():
    x, y = polyvar("x", "y")

    with pytest.raises(ValueError, match="not used by homotopy.*y"):
        track_paths(
            start_system=PolynomialSystem([x - 1]),
            target_system=PolynomialSystem([x - 2]),
            start_solutions=[[1.0 + 0j, 0.0 + 0j]],
            variables=[x, y],
            gamma=1.0 + 0j,
        )


def test_track_paths_rejects_start_solution_off_start_system():
    x = polyvar("x")

    with pytest.raises(
        ValueError,
        match=r"start_solutions\[0\].*does not satisfy start_system",
    ):
        track_paths(
            start_system=PolynomialSystem([x - 1]),
            target_system=PolynomialSystem([x - 2]),
            start_solutions=[[0.0 + 0j]],
            variables=[x],
            gamma=1.0 + 0j,
            use_endgame=False,
        )


def test_track_paths_rejects_mapping_start_solution_missing_coordinate():
    x, y = polyvar("x", "y")

    with pytest.raises(
        ValueError,
        match=r"start_solutions\[0\] is missing coordinate\(s\): y",
    ):
        track_paths(
            start_system=PolynomialSystem([x - 1, y - 1]),
            target_system=PolynomialSystem([x - 2, y - 3]),
            start_solutions=[{x: 1.0 + 0j}],
            variables=[x, y],
            gamma=1.0 + 0j,
            use_endgame=False,
        )


def test_track_paths_rejects_tiny_row_start_solution_false_positive():
    x = polyvar("x")

    with pytest.raises(
        ValueError,
        match=r"start_solutions\[0\].*scaled",
    ):
        track_paths(
            start_system=PolynomialSystem([1e-12 * (x - 1)]),
            target_system=PolynomialSystem([x - 2]),
            start_solutions=[[2.0 + 0j]],
            variables=[x],
            gamma=1.0 + 0j,
            use_endgame=False,
        )


def test_track_single_path_rejects_invalid_gamma():
    x = polyvar("x")

    with pytest.raises(ValueError, match="gamma must be finite and nonzero"):
        tracking_module.track_single_path(
            start_system=PolynomialSystem([x - 1]),
            target_system=PolynomialSystem([x - 2]),
            start_solution=np.array([1.0 + 0j]),
            variables=[x],
            gamma=complex(1.0, float("inf")),
        )


def test_track_single_path_rejects_start_solution_off_start_system():
    x = polyvar("x")

    with pytest.raises(
        ValueError,
        match="start_solution does not satisfy start_system",
    ):
        tracking_module.track_single_path(
            start_system=PolynomialSystem([x - 1]),
            target_system=PolynomialSystem([x - 2]),
            start_solution=np.array([0.0 + 0j]),
            variables=[x],
            gamma=1.0 + 0j,
            use_endgame=False,
        )


def test_track_single_path_accepts_solution_start_solution():
    x = polyvar("x")

    point, info = tracking_module.track_single_path(
        start_system=PolynomialSystem([x - 1]),
        target_system=PolynomialSystem([x - 2]),
        start_solution=Solution({x: 1.0 + 0j}, residual=0.0),
        variables=[x],
        gamma=1.0 + 0j,
        use_endgame=False,
    )

    np.testing.assert_allclose(point, np.array([2.0 + 0j]), atol=1e-8)
    assert info["success"]
    assert info["start_residual"] == 0.0


def test_track_single_path_rejects_tiny_row_start_solution_false_positive():
    x = polyvar("x")

    with pytest.raises(ValueError, match="start_solution.*scaled"):
        tracking_module.track_single_path(
            start_system=PolynomialSystem([1e-12 * (x - 1)]),
            target_system=PolynomialSystem([x - 2]),
            start_solution=np.array([2.0 + 0j]),
            variables=[x],
            gamma=1.0 + 0j,
            use_endgame=False,
        )


def test_track_single_path_reports_zero_start_residual_for_huge_exact_cancellation():
    x = polyvar("x")
    huge = 10**400

    point, info = tracking_module.track_single_path(
        start_system=PolynomialSystem([huge * (x - 1)]),
        target_system=PolynomialSystem([x - 2]),
        start_solution=np.array([1.0 + 0j]),
        variables=[x],
        gamma=1.0 + 0j,
        use_endgame=False,
        max_steps=20,
        max_step_size=0.2,
    )

    assert info["success"]
    assert info["failure_reason"] is None
    assert info["start_residual"] == 0.0
    assert info["start_scaled_residual"] == 0.0
    np.testing.assert_allclose(point, np.array([2.0 + 0j]), atol=1e-8)


def test_track_single_path_uses_scaled_target_residual_for_overflowing_target():
    x = polyvar("x")
    huge = 10**400

    point, info = tracking_module.track_single_path(
        start_system=PolynomialSystem([x - 1]),
        target_system=PolynomialSystem([huge * (x - 2)]),
        start_solution=np.array([1.0 + 0j]),
        variables=[x],
        gamma=1.0 + 0j,
        use_endgame=False,
        max_steps=20,
        max_step_size=0.2,
    )

    assert info["success"]
    assert info["failure_reason"] is None
    assert info["final_residual"] == pytest.approx(0.0)
    np.testing.assert_allclose(point, np.array([2.0 + 0j]), atol=1e-8)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"tol": 0.0}, ValueError, "tol must be positive and finite"),
        ({"tol": True}, TypeError, "tol must be a number"),
        ({"tol": np.bool_(True)}, TypeError, "tol must be a number"),
        ({"tol": "1e-8"}, TypeError, "tol must be a number"),
        ({"verbose": "yes"}, TypeError, "verbose must be a boolean"),
        ({"store_paths": "yes"}, TypeError, "store_paths must be a boolean"),
        ({"use_endgame": "yes"}, TypeError, "use_endgame must be a boolean"),
        (
            {"endgame_options": "fast"},
            TypeError,
            "endgame_options must be a dictionary",
        ),
        (
            {"endgame_options": {"newton_iterations": 5}},
            ValueError,
            "Unknown endgame option",
        ),
    ],
)
def test_track_paths_rejects_invalid_public_options(kwargs, error_type, message):
    x = polyvar("x")

    with pytest.raises(error_type, match=message):
        track_paths(
            start_system=PolynomialSystem([x - 1]),
            target_system=PolynomialSystem([x - 2]),
            start_solutions=[[1.0 + 0j]],
            variables=[x],
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"tol": float("nan")}, ValueError, "tol must be positive and finite"),
        ({"tol": np.bool_(True)}, TypeError, "tol must be a number"),
        ({"tol": "1e-8"}, TypeError, "tol must be a number"),
        ({"debug": "yes"}, TypeError, "debug must be a boolean"),
        ({"verbose": "yes"}, TypeError, "verbose must be a boolean"),
        ({"store_paths": "yes"}, TypeError, "store_paths must be a boolean"),
        ({"use_endgame": "yes"}, TypeError, "use_endgame must be a boolean"),
        (
            {"endgame_options": {"newton_iterations": 5}},
            ValueError,
            "Unknown endgame option",
        ),
    ],
)
def test_track_single_path_rejects_invalid_public_options(
    kwargs, error_type, message
):
    x = polyvar("x")

    with pytest.raises(error_type, match=message):
        tracking_module.track_single_path(
            start_system=PolynomialSystem([x - 1]),
            target_system=PolynomialSystem([x - 2]),
            start_solution=np.array([1.0 + 0j]),
            variables=[x],
            **kwargs,
        )


def test_solve_rejects_total_degree_start_system_over_path_limit():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**2 + y**2 - 1, x * y - 0.25])

    with pytest.raises(ValueError, match="exceeding max_paths=3"):
        solve(system, variables=[x, y], max_paths=3, random_state=0)


def test_solve_rejects_custom_start_system_over_path_limit():
    x = polyvar("x")
    target_system = PolynomialSystem([x**2 - 1])
    start_system = PolynomialSystem([x**2 - 4])

    with pytest.raises(ValueError, match="custom start system would track 2 path"):
        solve(
            target_system,
            start_system=start_system,
            start_solutions=[[2.0 + 0j], [-2.0 + 0j]],
            tracking_options={"gamma": 1.0 + 0j},
            max_paths=1,
            random_state=0,
        )


def test_solve_applies_max_paths_across_independent_blocks():
    x, y, u, v = polyvar("x", "y", "u", "v")
    system = PolynomialSystem([
        x**2 + y**2 - 1,
        x * y - 0.25,
        u**2 + v**2 - 1,
        u * v - 0.25,
    ])

    with pytest.raises(ValueError, match="exceeding max_paths=2"):
        solve(system, variables=[x, y, u, v], max_paths=6, random_state=0)


def test_solve_applies_max_paths_across_monomial_branches():
    x, y, a, b = polyvar("x", "y", "a", "b")
    system = PolynomialSystem([
        x * y,
        x**2 + y**2 - 1,
        a**2 + b**2 - 1,
        a * b - 0.25,
    ])

    with pytest.raises(ValueError, match="exceeding max_paths=2"):
        solve(system, variables=[x, y, a, b], max_paths=6, random_state=0)


def test_internal_remaining_path_limit_can_be_zero():
    assert solver_module._remaining_path_limit(6, 5) == 1
    assert solver_module._remaining_path_limit(6, 6) == 0
    assert solver_module._remaining_path_limit(6, 7) == 0


def test_internal_zero_remaining_path_budget_allows_direct_solves():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    solutions = solver_module.solve(
        system,
        max_paths=0,
        random_state=0,
        _allow_zero_max_paths=True,
    )

    assert len(solutions) == 1
    assert solutions._meta["total_paths"] == 0


def test_internal_zero_remaining_path_budget_rejects_tracking_before_generation():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**2 + y**2 - 1, x * y - 0.25])

    with pytest.raises(
        ValueError,
        match="total-degree start system would track 4 path.*max_paths=0",
    ):
        solver_module.solve(
            system,
            variables=[x, y],
            max_paths=0,
            random_state=0,
            _allow_zero_max_paths=True,
        )


def test_solve_rejects_invalid_max_paths():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    with pytest.raises(ValueError, match="max_paths must be positive"):
        solve(system, max_paths=0)
    with pytest.raises(TypeError, match="max_paths must be an integer"):
        solve(system, max_paths=True)


@pytest.mark.parametrize(
    ("option", "value", "error_type", "message"),
    [
        ("tol", 0.0, ValueError, "tol must be positive and finite"),
        ("tol", float("nan"), ValueError, "tol must be positive and finite"),
        ("tol", True, TypeError, "tol must be a number"),
        ("tol", "1e-8", TypeError, "tol must be a number"),
        (
            "deduplication_tol_factor",
            0.0,
            ValueError,
            "deduplication_tol_factor must be positive and finite",
        ),
        (
            "deduplication_tol_factor",
            "10",
            TypeError,
            "deduplication_tol_factor must be a number",
        ),
        (
            "singular_deduplication_tol",
            float("inf"),
            ValueError,
            "singular_deduplication_tol must be positive and finite",
        ),
        (
            "singular_deduplication_tol",
            "1e-3",
            TypeError,
            "singular_deduplication_tol must be a number",
        ),
    ],
)
def test_solve_rejects_invalid_tolerance_options(
    option, value, error_type, message
):
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    with pytest.raises(error_type, match=message):
        solve(system, random_state=0, **{option: value})


@pytest.mark.parametrize(
    "option",
    [
        "verbose",
        "store_paths",
        "use_endgame",
        "allow_underdetermined",
        "scale_equations",
    ],
)
def test_solve_rejects_nonboolean_solver_flags(option):
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    with pytest.raises(TypeError, match=f"{option} must be a boolean"):
        solve(system, random_state=0, **{option: "yes"})


def test_solve_rejects_invalid_endgame_options():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    with pytest.raises(TypeError, match="endgame_options must be a dictionary"):
        solve(system, endgame_options="fast", random_state=0)
    with pytest.raises(ValueError, match="Unknown endgame option"):
        solve(system, endgame_options={"newton_iterations": 5}, random_state=0)
    with pytest.raises(TypeError, match="abstol must be a number"):
        solve(system, endgame_options={"abstol": "1e-8"}, random_state=0)
    with pytest.raises(TypeError, match="samples_per_loop must be an integer"):
        solve(system, endgame_options={"samples_per_loop": True}, random_state=0)
    with pytest.raises(ValueError, match="samples_per_loop must be positive"):
        solve(system, endgame_options={"samples_per_loop": 0}, random_state=0)
    with pytest.raises(ValueError, match="L must be between 0 and 1"):
        solve(system, endgame_options={"L": 1.0}, random_state=0)


def test_solution_set_exports_ordered_arrays_and_dicts():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x - 1, y - 2])

    solutions = solve(system, variables=[x, y], random_state=0)

    np.testing.assert_allclose(
        solutions.to_array([y, x]),
        np.array([[2.0 + 0j, 1.0 + 0j]]),
    )
    np.testing.assert_allclose(
        solutions.to_array([x, y], real=True),
        np.array([[1.0, 2.0]]),
    )
    record = solutions.as_dicts()[0]
    assert record["values"]["x"] == {"real": 1.0, "imag": 0.0}
    assert record["values"]["y"] == {"real": 2.0, "imag": 0.0}
    assert record["multiplicity"] == 1
    assert "path_info" in record


def test_solution_set_exports_json_serializable_full_record():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x - 1, y - 2])

    solutions = solve(
        system,
        variables=[x, y],
        random_state=0,
        tracking_options={"gamma": 1.0 + 0.0j},
    )
    record = solutions.as_dict()

    json.dumps(record)
    assert record["solution_count"] == 1
    assert record["system"] == {
        "equations": ["x - 1", "y - 2"],
        "variables": ["x", "y"],
    }
    assert record["metadata"]["tracking_options"]["gamma"] == {
        "real": 1.0,
        "imag": 0.0,
    }
    assert "path_info" in record["solutions"][0]

    compact = solutions.as_dict(include_metadata=False)
    assert set(compact) == {"solutions", "solution_count"}
    assert "path_info" not in compact["solutions"][0]

    with pytest.raises(TypeError, match="include_metadata must be a boolean"):
        solutions.as_dict(include_metadata="yes")


def test_solution_dict_exports_validate_metadata_flag():
    x = polyvar("x")
    solution = Solution({x: 1.0 + 0j}, residual=0.0, path_index=3)
    solutions = SolutionSet([solution], PolynomialSystem([x - 1]))

    assert "path_info" in solution.as_dict(include_metadata=True)
    assert "path_info" not in solution.as_dict(include_metadata=False)
    assert "path_info" in solutions.as_dicts(include_metadata=True)[0]
    assert "path_info" not in solutions.as_dicts(include_metadata=False)[0]

    with pytest.raises(TypeError, match="include_metadata must be a boolean"):
        solution.as_dict(include_metadata="no")
    with pytest.raises(TypeError, match="include_metadata must be a boolean"):
        solutions.as_dicts(include_metadata=0)


def test_solution_dict_exports_metadata_as_json_serializable_values():
    x = polyvar("x")
    solution = Solution({x: np.float64(1.0)}, residual=np.float64(0.0))
    solution.path_index = np.int64(2)
    solution.path_indices = (np.int64(2),)
    solution.root_indices = (np.int64(0), np.int64(1))
    solution.winding_number = np.int64(3)
    solution.path_info = {
        "nested": {
            "array": np.array([1.0 + 2.0j, 3.0 + 0.0j]),
            "scalar": np.float64(1.25),
        },
        "complex": 1.0 + 2.0j,
    }

    record = solution.as_dict()

    json.dumps(record)
    assert record["path_index"] == 2
    assert record["path_indices"] == [2]
    assert record["root_indices"] == [0, 1]
    assert record["winding_number"] == 3
    assert record["path_info"]["nested"]["array"][0] == {
        "real": 1.0,
        "imag": 2.0,
    }
    assert record["path_info"]["complex"] == {"real": 1.0, "imag": 2.0}


def test_solution_dict_exports_strict_json_for_nonfinite_values():
    x = polyvar("x")
    solution = Solution(
        {x: complex(float("nan"), float("inf"))},
        residual=float("inf"),
    )
    solution.scaled_residual = np.float64(float("nan"))
    solution.backward_error = np.float64(float("inf"))
    solution.path_info = {
        "array": np.array(
            [float("inf"), complex(float("-inf"), float("nan"))]
        )
    }
    solutions = SolutionSet([solution], PolynomialSystem([x]))
    solutions._meta["largest_scale"] = float("inf")

    record = solution.as_dict(strict_json=True)
    set_record = solutions.as_dict(strict_json=True)

    json.dumps(record, allow_nan=False)
    json.dumps(set_record, allow_nan=False)
    assert record["values"]["x"] == {"real": "NaN", "imag": "Infinity"}
    assert record["residual"] == "Infinity"
    assert record["scaled_residual"] == "NaN"
    assert record["backward_error"] == "Infinity"
    assert record["path_info"]["array"][0] == {
        "real": "Infinity",
        "imag": 0.0,
    }
    assert record["path_info"]["array"][1] == {
        "real": "-Infinity",
        "imag": "NaN",
    }
    assert set_record["metadata"]["largest_scale"] == "Infinity"
    assert "path_info" not in solution.as_dict(
        include_metadata=False,
        strict_json=True,
    )

    with pytest.raises(TypeError, match="strict_json must be a boolean"):
        solution.as_dict(strict_json=1)
    with pytest.raises(TypeError, match="strict_json must be a boolean"):
        solutions.as_dicts(strict_json="yes")
    with pytest.raises(TypeError, match="strict_json must be a boolean"):
        solutions.as_dict(strict_json=0)


def test_solution_set_real_array_rejects_complex_coordinates():
    x = polyvar("x")
    solutions = solve(PolynomialSystem([x**2 + 1]), random_state=0)

    with pytest.raises(ValueError, match="non-real coordinate"):
        solutions.to_array([x], real=True)


def test_solution_vector_exports_reject_nonboolean_real_flag():
    x = polyvar("x")
    solution = Solution({x: 1.0 + 0.0j}, residual=0.0)
    solutions = SolutionSet([solution], PolynomialSystem([x - 1]))

    with pytest.raises(TypeError, match="real must be a boolean"):
        solution.point([x], real="yes")
    with pytest.raises(TypeError, match="real must be a boolean"):
        solutions.to_array([x], real=1)


@pytest.mark.parametrize(
    ("method", "kwargs", "error_type", "message"),
    [
        ("is_real", {"tol": -1.0}, ValueError, "tol must be nonnegative and finite"),
        ("is_real", {"tol": "tight"}, TypeError, "tol must be a number"),
        ("is_positive", {"tol": "1e-8"}, TypeError, "tol must be a number"),
        ("is_positive", {"tol": float("inf")}, ValueError, "tol must be nonnegative and finite"),
        ("point", {"real": True, "tol": float("nan")}, ValueError, "tol must be nonnegative and finite"),
        ("point", {"real": True, "tol": True}, TypeError, "tol must be a number"),
    ],
)
def test_solution_coordinate_checks_reject_invalid_tolerances(
    method, kwargs, error_type, message
):
    x = polyvar("x")
    solution = Solution({x: 1.0 + 0.0j}, residual=0.0)

    with pytest.raises(error_type, match=message):
        getattr(solution, method)(**kwargs)


def test_solution_coordinate_checks_accept_zero_tolerance():
    x = polyvar("x")
    solution = Solution({x: 1.0 + 0.0j}, residual=0.0)

    assert solution.is_real(tol=0.0)
    assert solution.is_positive(tol=0.0)
    np.testing.assert_allclose(solution.point([x], real=True, tol=0.0), [1.0])


def test_solution_constructor_normalizes_numeric_inputs():
    x = polyvar("x")
    solution = Solution(
        {x: np.float64(1.5)},
        residual=np.float64(0.25),
        is_singular=False,
        path_index=np.int64(2),
    )
    invalid = Solution({x: np.nan}, residual=float("nan"))

    assert solution.values[x] == 1.5 + 0j
    assert solution.residual == 0.25
    assert solution.path_index == 2
    assert np.isnan(invalid.values[x].real)
    assert np.isnan(invalid.residual)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"values": None}, TypeError, "values must be a dictionary"),
        ({"values": {"x": 1.0}}, TypeError, "values key 0 must be a Variable"),
        ({"values": "not-a-dict"}, TypeError, "values must be a dictionary"),
    ],
)
def test_solution_constructor_rejects_invalid_values(kwargs, error_type, message):
    x = polyvar("x")
    arguments = {"values": {x: 1.0}, "residual": 0.0}
    arguments.update(kwargs)

    with pytest.raises(error_type, match=message):
        Solution(**arguments)


def test_solution_constructor_rejects_nonnumeric_coordinate():
    x = polyvar("x")

    with pytest.raises(
        TypeError,
        match=r"values\[x\] must be a numeric coordinate",
    ):
        Solution({x: "not-numeric"}, residual=0.0)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"residual": "0"}, TypeError, "residual must be a real number"),
        ({"residual": 1.0 + 0j}, TypeError, "residual must be a real number"),
        ({"residual": -1.0}, ValueError, "residual must be nonnegative"),
        ({"is_singular": "no"}, TypeError, "is_singular must be a boolean"),
        ({"path_index": True}, TypeError, "path_index must be an integer or None"),
        ({"path_index": 1.5}, TypeError, "path_index must be an integer or None"),
        ({"path_index": -1}, ValueError, "path_index must be nonnegative"),
    ],
)
def test_solution_constructor_rejects_invalid_metadata(kwargs, error_type, message):
    x = polyvar("x")
    arguments = {"residual": 0.0}
    arguments.update(kwargs)

    with pytest.raises(error_type, match=message):
        Solution({x: 1.0}, **arguments)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"real": "yes"}, TypeError, "real must be a boolean or None"),
        ({"positive": "no"}, TypeError, "positive must be a boolean or None"),
        ({"tol": "tight"}, TypeError, "tol must be a number"),
        ({"tol": "1e-8"}, TypeError, "tol must be a number"),
        ({"tol": 0.0}, ValueError, "tol must be positive and finite"),
        ({"max_residual": "1e-8"}, TypeError, "max_residual must be a number"),
        (
            {"max_scaled_residual": "1e-8"},
            TypeError,
            "max_scaled_residual must be a number",
        ),
        (
            {"max_backward_error": "1e-8"},
            TypeError,
            "max_backward_error must be a number",
        ),
        (
            {"max_residual": float("inf")},
            ValueError,
            "max_residual must be nonnegative and finite",
        ),
        (
            {"max_scaled_residual": float("inf")},
            ValueError,
            "max_scaled_residual must be nonnegative and finite",
        ),
        (
            {"max_backward_error": float("inf")},
            ValueError,
            "max_backward_error must be nonnegative and finite",
        ),
        (
            {"max_residual": -1.0},
            ValueError,
            "max_residual must be nonnegative and finite",
        ),
        (
            {"max_scaled_residual": -1.0},
            ValueError,
            "max_scaled_residual must be nonnegative and finite",
        ),
        (
            {"max_backward_error": -1.0},
            ValueError,
            "max_backward_error must be nonnegative and finite",
        ),
        ({"custom_filter": "not-callable"}, TypeError, "custom_filter must be callable"),
    ],
)
def test_solution_set_filter_rejects_invalid_options(kwargs, error_type, message):
    x = polyvar("x")
    solutions = solve(PolynomialSystem([x**2 - 1]), random_state=0)

    with pytest.raises(error_type, match=message):
        solutions.filter(**kwargs)


def test_solution_set_filter_validates_and_applies_predicates():
    x = polyvar("x")
    solutions = solve(PolynomialSystem([x**2 - 1]), random_state=0)

    positive = solutions.filter(
        real=True,
        positive=True,
        max_residual=1e-8,
        custom_filter=lambda solution: solution.values[x].real > 0,
    )

    assert len(positive) == 1
    assert positive[0].values[x].real > 0
    assert positive._meta["is_filtered"]


def test_solution_set_filter_applies_scaled_and_backward_error_cutoffs():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    accepted = Solution({x: 1.0 + 0j}, residual=1e-2)
    accepted.scaled_residual = 1e-12
    accepted.backward_error = 1e-12

    large_scaled = Solution({x: 1.0 + 0j}, residual=0.0)
    large_scaled.scaled_residual = 1e-4
    large_scaled.backward_error = 1e-12

    large_backward = Solution({x: 1.0 + 0j}, residual=0.0)
    large_backward.scaled_residual = 1e-12
    large_backward.backward_error = 1e-4

    missing_metrics_valid = Solution({x: 1.0 + 0j}, residual=0.0)
    missing_metrics_invalid = Solution({x: 2.0 + 0j}, residual=0.0)

    solutions = SolutionSet(
        [
            accepted,
            large_scaled,
            large_backward,
            missing_metrics_valid,
            missing_metrics_invalid,
        ],
        system,
    )

    filtered = solutions.filter(
        max_scaled_residual=1e-8,
        max_backward_error=1e-8,
    )

    assert list(filtered) == [accepted, missing_metrics_valid]
    assert filtered._meta["is_filtered"]


def test_solution_set_nearest_accepts_coordinate_records():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y])
    first = Solution({x: 0.0 + 0j, y: 0.0 + 0j}, residual=0.0)
    second = Solution({x: 10.0 + 0j, y: 0.0 + 0j}, residual=0.0)
    solutions = SolutionSet([first, second], system)

    class SolutionLike:
        def __init__(self, values):
            self.values = values

    nearest, distance = solutions.nearest(
        {"y": 0.0 + 0j, "x": 9.0 + 0j},
        variables=[x, y],
        return_distance=True,
    )

    assert nearest is second
    assert distance == pytest.approx(1.0)
    assert solutions.nearest(SolutionLike({x: 1.0 + 0j, y: 0.0 + 0j})) is first


def test_solution_set_nearest_uses_scaled_distance_for_huge_coordinates():
    x = polyvar("x")
    system = PolynomialSystem([x])
    first = Solution({x: 1.0e300 + 0j}, residual=0.0)
    second = Solution({x: 2.0e300 + 0j}, residual=0.0)
    solutions = SolutionSet([first, second], system)

    nearest, distance = solutions.nearest(
        [1.1e300 + 0j],
        variables=[x],
        return_distance=True,
    )

    assert nearest is first
    assert np.isfinite(distance)


def test_solution_set_nearest_validates_inputs():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y])
    solution = Solution({x: 0.0 + 0j, y: 0.0 + 0j}, residual=0.0)
    solutions = SolutionSet([solution], system)

    with pytest.raises(ValueError, match="empty SolutionSet"):
        SolutionSet([], system).nearest([0.0 + 0j, 0.0 + 0j], variables=[x, y])
    with pytest.raises(TypeError, match="return_distance must be a boolean"):
        solutions.nearest([0.0 + 0j, 0.0 + 0j], return_distance=1)
    with pytest.raises(ValueError, match="missing coordinate.*y"):
        solutions.nearest({"x": 0.0 + 0j}, variables=[x, y])
    with pytest.raises(ValueError, match="duplicate variable"):
        solutions.nearest([0.0 + 0j, 0.0 + 0j], variables=[x, x])


def test_solution_set_constructor_validates_inputs():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])
    generated = SolutionSet(
        (Solution({x: 1.0 + 0j}, residual=0.0) for _ in range(1)),
        system,
    )

    assert len(generated) == 1
    assert isinstance(generated[0], Solution)

    with pytest.raises(TypeError, match="solutions must be an iterable"):
        SolutionSet(None, system)
    with pytest.raises(TypeError, match="solutions must be an iterable"):
        SolutionSet("not-a-solution-list", system)
    with pytest.raises(TypeError, match=r"solutions\[0\] must be a Solution"):
        SolutionSet([object()], system)
    with pytest.raises(TypeError, match="system must be a PolynomialSystem"):
        SolutionSet([], [x - 1])


def test_solution_set_from_points_imports_coordinate_records():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 3, x * y - 2])

    class SolutionLike:
        def __init__(self, values):
            self.values = values

    imported = SolutionSet.from_points(
        [
            [1.0 + 0j, 2.0 + 0j],
            {"y": 2.0 + 0j, x: 1.0 + 0j},
            SolutionLike({"x": 1.0 + 0j, "y": 2.0 + 0j}),
        ],
        system,
        variables=[x, y],
    )

    assert len(imported) == 3
    assert imported._meta["source"] == "from_points"
    assert imported._meta["variables"] == ["x", "y"]
    assert all(isinstance(solution, Solution) for solution in imported)
    np.testing.assert_allclose(
        imported.to_array([x, y]),
        np.array(
            [
                [1.0 + 0j, 2.0 + 0j],
                [1.0 + 0j, 2.0 + 0j],
                [1.0 + 0j, 2.0 + 0j],
            ],
            dtype=complex,
        ),
    )
    assert all(solution.residual == 0.0 for solution in imported)
    assert all(solution.scaled_residual == 0.0 for solution in imported)
    assert all(solution.backward_error == 0.0 for solution in imported)
    assert imported.diagnostics(tolerance=1e-8).all_valid


def test_solution_set_from_points_marks_singular_imported_candidates():
    x = polyvar("x")
    imported = SolutionSet.from_points(
        [[0.0 + 0j]],
        PolynomialSystem([x**2]),
        variables=[x],
    )

    assert len(imported) == 1
    assert imported[0].is_singular
    assert imported.diagnostics(tolerance=1e-12).diagnostics[0].is_rank_deficient


def test_solution_set_from_points_validates_inputs():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1])

    with pytest.raises(TypeError, match="points must be an iterable"):
        SolutionSet.from_points(None, system, variables=[x, y])
    with pytest.raises(TypeError, match="points must be an iterable"):
        SolutionSet.from_points("not-points", system, variables=[x, y])
    with pytest.raises(ValueError, match=r"points\[0\].*2 coordinate"):
        SolutionSet.from_points([[1.0 + 0j]], system, variables=[x, y])
    with pytest.raises(ValueError, match=r"points\[0\].*missing coordinate.*y"):
        SolutionSet.from_points([{"x": 1.0 + 0j}], system, variables=[x, y])
    with pytest.raises(ValueError, match=r"points\[0\].*conflicting coordinates.*x"):
        SolutionSet.from_points(
            [{x: 1.0 + 0j, "x": 2.0 + 0j, y: 0.0 + 0j}],
            system,
            variables=[x, y],
        )
    with pytest.raises(TypeError, match="singularity_threshold must be a number"):
        SolutionSet.from_points(
            [[1.0 + 0j, 0.0 + 0j]],
            system,
            variables=[x, y],
            singularity_threshold="loose",
        )


def test_solution_vector_exports_reject_missing_coordinates():
    x, y = polyvar("x", "y")
    partial = Solution({x: 1.0 + 0j}, residual=0.0)
    solutions = SolutionSet([partial], PolynomialSystem([x + y - 2]))

    with pytest.raises(ValueError, match=r"missing coordinate.*y"):
        partial.point([x, y])
    with pytest.raises(ValueError, match=r"missing coordinate.*y"):
        solutions.to_array([x, y])


def test_solution_distance_uses_validated_coordinate_vectors():
    x, y = polyvar("x", "y")
    left = Solution({x: 1.0 + 0j, y: 2.0 + 0j}, residual=0.0)
    right = Solution({x: 4.0 + 0j, y: 6.0 + 0j}, residual=0.0)

    class SolutionLike:
        def __init__(self, values):
            self.values = values

    assert left.distance(right, [x, y]) == pytest.approx(5.0)
    assert left.distance([4.0 + 0j, 6.0 + 0j], [x, y]) == pytest.approx(5.0)
    assert left.distance({"y": 6.0 + 0j, x: 4.0 + 0j}, [x, y]) == pytest.approx(5.0)
    assert left.distance(
        SolutionLike({"x": 4.0 + 0j, "y": 6.0 + 0j}),
        [x, y],
    ) == pytest.approx(5.0)

    with pytest.raises(ValueError, match="duplicate variable"):
        left.distance(right, [x, x])
    with pytest.raises(TypeError, match=r"variables\[1\] must be a Variable"):
        left.distance(right, [x, "y"])


def test_solution_distance_keeps_large_finite_distances_finite():
    x, y = polyvar("x", "y")
    huge = 1e200
    left = Solution({x: 0.0 + 0j, y: 0.0 + 0j}, residual=0.0)
    right = Solution({x: huge + 0j, y: huge + 0j}, residual=0.0)

    distance = left.distance(right, [x, y])

    assert np.isfinite(distance)
    assert distance == pytest.approx(np.sqrt(2.0) * huge)


def test_solution_distance_rejects_missing_coordinates():
    x, y = polyvar("x", "y")
    partial = Solution({x: 1.0 + 0j}, residual=0.0)
    complete = Solution({x: 1.0 + 0j, y: 2.0 + 0j}, residual=0.0)

    with pytest.raises(ValueError, match=r"Solution is missing coordinate.*y"):
        partial.distance(complete, [x, y])
    with pytest.raises(ValueError, match=r"other solution is missing coordinate.*y"):
        complete.distance(partial, [x, y])


def test_residual_and_rank_helpers_reject_missing_coordinates():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1])
    values = {x: 1.0 + 0j}

    with pytest.raises(ValueError, match=r"solution is missing coordinate.*y"):
        _scaled_residual_norm(system, values)
    with pytest.raises(ValueError, match=r"solution is missing coordinate.*y"):
        _is_singular(system, values, (x, y), threshold=1e12)


def test_solver_norm_helpers_keep_large_finite_values_finite():
    x = polyvar("x")
    huge = 1e200

    residual_system = PolynomialSystem([huge * x, huge * x])
    unit_values = {x: 1.0 + 0j}
    residual = _residual_norm(residual_system, unit_values)

    scaled_system = PolynomialSystem([x, x])
    huge_values = {x: huge + 0j}
    scaled_residual = _scaled_residual_norm(scaled_system, huge_values)
    backward_error = _backward_error_norm(scaled_system, huge_values)

    assert np.isfinite(residual)
    assert residual == pytest.approx(np.sqrt(2.0) * huge)
    assert np.isfinite(scaled_residual)
    assert scaled_residual == pytest.approx(np.sqrt(2.0) * huge)
    assert np.isfinite(backward_error)
    assert backward_error == pytest.approx(np.sqrt(2.0))


def test_witness_and_endgame_residual_helpers_keep_large_finite_norms_finite():
    x, y = polyvar("x", "y")
    huge = 1e200
    system = PolynomialSystem([x, y])
    point = np.array([huge + 0j, huge + 0j])

    witness_residual = witness_set_module._witness_system_residual(
        system,
        point,
        [x, y],
    )
    endgame_residual, used_scaled = endgame_module._system_residual_norm_with_source(
        system,
        point,
        [x, y],
    )

    assert np.isfinite(witness_residual)
    assert witness_residual == pytest.approx(np.sqrt(2.0) * huge)
    assert np.isfinite(endgame_residual)
    assert endgame_residual == pytest.approx(np.sqrt(2.0) * huge)
    assert not used_scaled


def test_newton_corrector_allows_large_finite_steps():
    x, y = polyvar("x", "y")
    tiny = 1e-200
    huge = 1e200
    system = PolynomialSystem([tiny * x - 1, tiny * y - 1])

    point, success, iters = newton_corrector(
        system,
        np.array([0.0 + 0j, 0.0 + 0j]),
        [x, y],
        max_iters=3,
        tol=1e-12,
    )

    assert success
    assert iters == 1
    np.testing.assert_allclose(point, np.array([huge + 0j, huge + 0j]))


def test_numeric_newton_corrector_allows_large_finite_steps():
    tiny = 1e-200
    huge = 1e200

    def f(point):
        return tiny * point - np.ones(2, dtype=complex)

    def jac(point):
        return np.eye(2, dtype=complex) * tiny

    point, success, iters = newton_corrector_numeric(
        f,
        jac,
        np.array([0.0 + 0j, 0.0 + 0j]),
        max_iters=3,
        tol=1e-12,
    )

    assert success
    assert iters == 1
    np.testing.assert_allclose(point, np.array([huge + 0j, huge + 0j]))


def test_complete_coordinate_helper_preserves_order_and_rejects_missing():
    x, y = polyvar("x", "y")

    ordered = _solution_values_from_complete_coordinates(
        {y: 2.0, x: 1.0},
        [x, y],
        label="lifted solution",
    )

    assert list(ordered) == [x, y]
    assert ordered == {x: 1.0 + 0j, y: 2.0 + 0j}
    with pytest.raises(ValueError, match=r"lifted solution is missing coordinate.*y"):
        _solution_values_from_complete_coordinates(
            {x: 1.0},
            [x, y],
            label="lifted solution",
        )
    with pytest.raises(ValueError, match="duplicate variable"):
        _solution_values_from_complete_coordinates(
            {x: 1.0},
            [x, x],
            label="lifted solution",
        )


def test_solution_clustering_rejects_invalid_coordinate_inputs():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1])
    partial = Solution({x: 1.0 + 0j}, residual=0.0)
    complete = Solution({x: 1.0 + 0j, y: 0.0 + 0j}, residual=0.0)

    with pytest.raises(ValueError, match=r"solution is missing coordinate.*y"):
        _deduplicate_solutions(
            [partial],
            system,
            [x, y],
            regular_tolerance=1e-6,
            singular_tolerance=1e-3,
            rank_tolerance=1e-12,
            polish_tolerance=1e-12,
        )
    with pytest.raises(ValueError, match="duplicate variable"):
        _deduplicate_solutions(
            [complete],
            system,
            [x, x],
            regular_tolerance=1e-6,
            singular_tolerance=1e-3,
            rank_tolerance=1e-12,
            polish_tolerance=1e-12,
        )


def test_solution_clustering_prefers_scaled_quality_over_raw_residual(monkeypatch):
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    raw_residual_best = Solution({x: 1.0 + 1e-10}, residual=0.0, path_index=0)
    raw_residual_best.scaled_residual = 1.0
    raw_residual_best.backward_error = 1.0

    normalized_quality_best = Solution(
        {x: 1.0 + 2e-10},
        residual=1e-3,
        path_index=1,
    )
    normalized_quality_best.scaled_residual = 1e-12
    normalized_quality_best.backward_error = 1e-12

    def skip_centroid_polish(system, point, variables, tol):
        return point, float("inf"), {"attempted": False}

    monkeypatch.setattr(
        solver_module,
        "_polish_endpoint_against_system",
        skip_centroid_polish,
    )

    unique = _deduplicate_solutions(
        [raw_residual_best, normalized_quality_best],
        system,
        [x],
        regular_tolerance=1e-6,
        singular_tolerance=1e-3,
        rank_tolerance=1e-12,
        polish_tolerance=1e-12,
    )

    assert unique == [normalized_quality_best]
    assert unique[0].path_indices == (0, 1)
    assert unique[0].path_info["cluster"]["multiplicity"] == 2


def test_solution_clustering_is_order_independent_for_duplicate_chains(monkeypatch):
    x = polyvar("x")
    system = PolynomialSystem([x - 1])
    raw_solutions = [
        Solution({x: 1.0 + 0.0e-6}, residual=0.0, path_index=0),
        Solution({x: 1.0 + 0.9e-6}, residual=0.0, path_index=1),
        Solution({x: 1.0 + 1.8e-6}, residual=0.0, path_index=2),
    ]
    for solution in raw_solutions:
        solution.scaled_residual = 0.0
        solution.backward_error = 0.0

    def skip_centroid_polish(system, point, variables, tol):
        return point, float("inf"), {"attempted": False}

    monkeypatch.setattr(
        solver_module,
        "_polish_endpoint_against_system",
        skip_centroid_polish,
    )

    unique = _deduplicate_solutions(
        raw_solutions,
        system,
        [x],
        regular_tolerance=1e-6,
        singular_tolerance=1e-3,
        rank_tolerance=1e-12,
        polish_tolerance=1e-12,
    )

    assert len(unique) == 1
    assert unique[0].multiplicity == 3
    assert unique[0].path_indices == (0, 1, 2)
    assert unique[0].path_info["cluster"]["multiplicity"] == 3


def test_solve_requires_custom_start_system_and_solutions_together():
    x = polyvar("x")
    target_system = PolynomialSystem([x - 2])
    start_system = PolynomialSystem([x - 1])

    with pytest.raises(ValueError, match="provided together"):
        solve(target_system, start_system=start_system, random_state=0)

    with pytest.raises(ValueError, match="provided together"):
        solve(target_system, start_solutions=[[1.0 + 0j]], random_state=0)


def test_solve_uses_validated_custom_start_system():
    x = polyvar("x")
    target_system = PolynomialSystem([x - 2])
    start_system = PolynomialSystem([x - 1])

    solutions = solve(
        target_system,
        start_system=start_system,
        start_solutions=[[1.0 + 0j]],
        use_endgame=False,
        tracking_options={"gamma": 1.0 + 0j},
    )

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 2.0) < 1e-8
    assert solutions._meta["start_system"]["source"] == "custom"
    assert solutions._meta["start_system"]["path_count"] == 1
    assert solutions._meta["start_system"]["max_start_residual"] == 0.0
    assert solutions._meta["path_summary"]["accepted_paths"] == 1
    assert solutions._meta["path_summary"]["tracked_successful_paths"] == 1
    assert solutions._meta["path_summary"]["residual_rejected_paths"] == 0
    assert solutions[0].path_info["accepted"]
    assert solutions[0].path_info["solution_residual"] < 1e-8


def test_solve_accepts_parseable_custom_start_system():
    x = polyvar("x")

    solutions = solve(
        "x - 2",
        variables=[x],
        start_system="x - 1",
        start_solutions=[[1.0 + 0j]],
        use_endgame=False,
        tracking_options={"gamma": 1.0 + 0j},
    )

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 2.0) < 1e-8
    assert solutions._meta["start_system"]["source"] == "custom"
    assert solutions._meta["start_system"]["path_count"] == 1


def test_solve_accepts_iterable_custom_start_system_strings():
    x = polyvar("x")

    solutions = solve(
        ["x - 2"],
        variables=[x],
        start_system=["x - 1"],
        start_solutions=[[1.0 + 0j]],
        use_endgame=False,
        tracking_options={"gamma": 1.0 + 0j},
    )

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 2.0) < 1e-8
    assert solutions._meta["start_system"]["source"] == "custom"


def test_solve_accepts_single_polynomial_custom_start_system():
    x = polyvar("x")

    solutions = solve(
        x - 2,
        start_system=x - 1,
        start_solutions=[[1.0 + 0j]],
        use_endgame=False,
        tracking_options={"gamma": 1.0 + 0j},
    )

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 2.0) < 1e-8
    assert solutions._meta["start_system"]["source"] == "custom"


def test_solve_accepts_solution_objects_as_custom_start_solutions():
    x = polyvar("x")

    solutions = solve(
        x - 2,
        start_system=x - 1,
        start_solutions=[Solution({x: 1.0 + 0j}, residual=0.0)],
        use_endgame=False,
        tracking_options={"gamma": 1.0 + 0j},
    )

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 2.0) < 1e-8
    assert solutions._meta["start_system"]["path_count"] == 1


def test_solve_accepts_mapping_custom_start_solutions():
    x, y = polyvar("x", "y")

    solutions = solve(
        PolynomialSystem([x - 2, y - 3]),
        variables=[x, y],
        start_system=PolynomialSystem([x - 1, y - 1]),
        start_solutions=[{"y": 1.0 + 0j, x: 1.0 + 0j}],
        use_endgame=False,
        tracking_options={"gamma": 1.0 + 0j},
    )

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 2.0) < 1e-8
    assert abs(solutions[0].values[y] - 3.0) < 1e-8


def test_solve_rejects_mapping_custom_start_solution_missing_coordinate():
    x, y = polyvar("x", "y")

    with pytest.raises(
        ValueError,
        match=r"Start solution 0 is missing coordinate\(s\): y",
    ):
        solve(
            PolynomialSystem([x - 2, y - 3]),
            variables=[x, y],
            start_system=PolynomialSystem([x - 1, y - 1]),
            start_solutions=[{x: 1.0 + 0j}],
            random_state=0,
        )


def test_solve_rejects_empty_custom_start_solutions():
    x = polyvar("x")
    target_system = PolynomialSystem([x - 2])
    start_system = PolynomialSystem([x - 1])

    with pytest.raises(ValueError, match="start_solutions.*at least one"):
        solve(
            target_system,
            start_system=start_system,
            start_solutions=[],
            random_state=0,
        )


def test_custom_start_system_is_scaled_for_huge_exact_coefficients():
    x = polyvar("x")
    huge = 10**400
    target_system = PolynomialSystem([huge * (x - 2)])
    start_system = PolynomialSystem([huge * (x - 1)])

    solutions = solve(
        target_system,
        start_system=start_system,
        start_solutions=[[1.0 + 0j]],
        variables=[x],
        use_endgame=False,
        tracking_options={"gamma": 1.0 + 0j},
        random_state=0,
    )

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 2.0) < 1e-6
    assert solutions[0].residual == 0.0
    assert solutions[0].scaled_residual == 0.0
    assert solutions[0].backward_error == 0.0
    start_meta = solutions._meta["start_system"]
    assert start_meta["source"] == "custom"
    assert start_meta["max_start_residual"] == 0.0
    assert start_meta["max_start_scaled_residual"] == 0.0
    assert start_meta["equation_scaling"]["enabled"]
    assert start_meta["equation_scaling"]["coefficient_scales"] == (float("inf"),)
    assert solutions[0].path_info["start_scaled_residual"] == 0.0


def test_solve_rejects_custom_start_solution_with_wrong_dimension():
    x, y = polyvar("x", "y")
    target_system = PolynomialSystem([x - 2, y - 3])
    start_system = PolynomialSystem([x - 1, y - 1])

    with pytest.raises(ValueError, match="one-dimensional point"):
        solve(
            target_system,
            start_system=start_system,
            start_solutions=[[1.0 + 0j]],
            random_state=0,
        )


def test_solve_rejects_custom_start_solution_not_on_start_system():
    x = polyvar("x")
    target_system = PolynomialSystem([x - 2])
    start_system = PolynomialSystem([x - 1])

    with pytest.raises(ValueError, match="does not satisfy"):
        solve(
            target_system,
            start_system=start_system,
            start_solutions=[[3.0 + 0j]],
            random_state=0,
        )


def test_solve_rejects_tiny_row_custom_start_solution_false_positive():
    x = polyvar("x")
    target_system = PolynomialSystem([x - 2])
    start_system = PolynomialSystem([1e-12 * (x - 1)])

    with pytest.raises(ValueError, match="custom start.*scaled"):
        solve(
            target_system,
            start_system=start_system,
            start_solutions=[[2.0 + 0j]],
            variables=[x],
            random_state=0,
        )


def test_solve_rejects_successful_path_with_only_tiny_raw_residual(monkeypatch):
    x = polyvar("x")
    target_system = PolynomialSystem([1e-12 * (x - 1)])
    start_system = PolynomialSystem([x - 2])

    def fake_track_paths(*args, **kwargs):
        return [np.array([2.0 + 0j])], [
            {
                "success": True,
                "singular": False,
                "final_t": 0.0,
                "final_residual": 1e-12,
                "failure_reason": None,
            }
        ]

    monkeypatch.setattr(solver_module, "track_paths", fake_track_paths)
    monkeypatch.setattr(
        solver_module,
        "_polish_endpoint_against_system",
        lambda system, point, variables, tol: (
            np.asarray(point, dtype=complex),
            1e-12,
            {
                "attempted": True,
                "accepted": False,
                "final_residual": 1e-12,
                "final_scaled_residual": 1.0,
            },
        ),
    )

    solutions = solve(
        target_system,
        start_system=start_system,
        start_solutions=[[2.0 + 0j]],
        variables=[x],
        random_state=0,
        use_endgame=False,
        tracking_options={"gamma": 1.0 + 0j},
    )

    assert len(solutions) == 0
    summary = solutions._meta["path_summary"]
    assert summary["tracked_successful_paths"] == 1
    assert summary["residual_rejected_paths"] == 1
    rejection = summary["residual_rejections"][0]
    assert rejection["residual"] == pytest.approx(1e-12)
    assert rejection["scaled_residual"] == pytest.approx(1.0)
    assert rejection["scaled_residual"] > rejection["residual_limit"]


def test_solve_random_state_controls_generated_gamma():
    x = polyvar("x")
    system = PolynomialSystem([x**2 - 1])

    first = solve(system, random_state=123)
    second = solve(system, random_state=123)
    third = solve(system, random_state=456)

    gamma_first = first._meta["tracking_options"]["gamma"]
    gamma_second = second._meta["tracking_options"]["gamma"]
    gamma_third = third._meta["tracking_options"]["gamma"]

    assert first._meta["generated_gamma"]
    assert second._meta["generated_gamma"]
    assert gamma_first == gamma_second
    assert gamma_first != gamma_third
    assert abs(abs(gamma_first) - 1.0) < 1e-12


def test_solve_retries_generated_gamma_when_all_paths_fail(monkeypatch):
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**2 + y - 1, y**2 + x - 1])
    root = np.array([0.0 + 0j, 1.0 + 0j])
    calls = []

    def fake_track_paths(
        *,
        start_system,
        target_system,
        start_solutions,
        variables,
        tol,
        verbose,
        store_paths,
        use_endgame,
        endgame_options,
        **kwargs,
    ):
        calls.append(kwargs["gamma"])
        path_count = len(start_solutions)
        if len(calls) == 1:
            return (
                [np.zeros(len(variables), dtype=complex) for _ in range(path_count)],
                [
                    {"success": False, "failure_reason": "forced_failure"}
                    for _ in range(path_count)
                ],
            )

        return (
            [root.copy() for _ in range(path_count)],
            [
                {
                    "success": True,
                    "singular": False,
                    "final_point": root.copy(),
                    "steps": 1,
                    "newton_iters": 1,
                }
                for _ in range(path_count)
            ],
        )

    monkeypatch.setattr(solver_module, "track_paths", fake_track_paths)

    solutions = solve(system, variables=[x, y], random_state=0, use_endgame=False)

    assert len(calls) == 2
    assert len(solutions) == 1
    assert abs(solutions[0].values[x]) < 1e-12
    assert abs(solutions[0].values[y] - 1) < 1e-12
    assert solutions._meta["tracking_options"]["gamma"] == calls[1]
    retry_meta = solutions._meta["tracking_retries"]
    assert retry_meta["attempted"]
    assert retry_meta["accepted"]
    assert retry_meta["attempt_count"] == 2
    assert retry_meta["initial_successful_paths"] == 0
    assert retry_meta["retry_successful_paths"] == 4
    assert retry_meta["initial_gamma"] == calls[0]
    assert retry_meta["retry_gamma"] == calls[1]


def test_solve_retries_failed_paths_with_smaller_steps(monkeypatch):
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**2 + y - 1, y**2 + x - 1])
    first_root = np.array([0.0 + 0j, 1.0 + 0j])
    retry_root = np.array([1.0 + 0j, 0.0 + 0j])
    calls = []

    def fake_track_paths(
        *,
        start_system,
        target_system,
        start_solutions,
        variables,
        tol,
        verbose,
        store_paths,
        use_endgame,
        endgame_options,
        **kwargs,
    ):
        calls.append({"count": len(start_solutions), "options": kwargs.copy()})
        if len(calls) == 1:
            failed = [
                {"success": False, "failure_reason": "forced_failure"}
                for _ in range(len(start_solutions) - 1)
            ]
            return (
                [first_root.copy()]
                + [np.zeros(len(variables), dtype=complex) for _ in failed],
                [
                    {
                        "success": True,
                        "singular": False,
                        "final_point": first_root.copy(),
                        "steps": 1,
                        "newton_iters": 1,
                    },
                    *failed,
                ],
            )

        return (
            [retry_root.copy() for _ in start_solutions],
            [
                {
                    "success": True,
                    "singular": False,
                    "final_point": retry_root.copy(),
                    "steps": 2,
                    "newton_iters": 2,
                }
                for _ in start_solutions
            ],
        )

    monkeypatch.setattr(solver_module, "track_paths", fake_track_paths)

    solutions = solve(system, variables=[x, y], random_state=0, use_endgame=False)
    points = sorted(
        (round(solution.values[x].real), round(solution.values[y].real))
        for solution in solutions
    )

    assert points == [(0, 1), (1, 0)]
    assert len(calls) == 2
    assert calls[0]["count"] == 4
    assert calls[1]["count"] == 3
    assert calls[1]["options"]["gamma"] == calls[0]["options"]["gamma"]
    assert calls[1]["options"]["max_step_size"] < 0.05
    assert calls[1]["options"]["max_newton_iters"] == 20
    assert calls[1]["options"]["max_steps"] == 20000
    assert calls[1]["options"]["predictor"] == "rk4"
    retry_meta = solutions._meta["tracking_retries"]
    assert retry_meta["attempted"]
    assert retry_meta["accepted"]
    assert retry_meta["attempt_count"] == 2
    assert retry_meta["failed_path_retry"]["path_indices"] == (1, 2, 3)
    assert retry_meta["failed_path_retry"]["successful_path_count"] == 3
    assert retry_meta["total_attempted_paths"] == 7


def test_failed_path_retry_respects_max_paths(monkeypatch):
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**2 + y - 1, y**2 + x - 1])
    root = np.array([0.0 + 0j, 1.0 + 0j])

    def fake_track_paths(
        *,
        start_system,
        target_system,
        start_solutions,
        variables,
        tol,
        verbose,
        store_paths,
        use_endgame,
        endgame_options,
        **kwargs,
    ):
        return (
            [root.copy() for _ in start_solutions],
            [
                {
                    "success": True,
                    "singular": False,
                    "final_point": root.copy(),
                },
                *(
                    {"success": False, "failure_reason": "forced_failure"}
                    for _ in range(len(start_solutions) - 1)
                ),
            ],
        )

    monkeypatch.setattr(solver_module, "track_paths", fake_track_paths)

    with pytest.raises(ValueError, match="failed path retry"):
        solve(
            system,
            variables=[x, y],
            max_paths=4,
            random_state=0,
            use_endgame=False,
        )


def test_solve_random_state_seeds_endgame_polish(monkeypatch):
    x = polyvar("x")
    calls = []

    def fake_endgame(
        start_system,
        target_system,
        point,
        t,
        variables,
        options=None,
    ):
        calls.append(dict(options or {}))
        point = np.asarray(point, dtype=complex)
        return point, {
            "success": True,
            "steps": 0,
            "winding_number": 1,
            "final_residual": 0.0,
            "final_point": point.copy(),
        }

    monkeypatch.setattr(tracking_module, "run_cauchy_endgame", fake_endgame)

    solve(
        PolynomialSystem([x - 2]),
        start_system=PolynomialSystem([x - 1]),
        start_solutions=[[1.0 + 0j]],
        tracking_options={"gamma": 1.0 + 0j},
        use_endgame=True,
        random_state=123,
    )

    expected_seed = int(np.random.default_rng(123).integers(0, 2**32 - 1))
    assert calls[0]["random_state"] == expected_seed


def test_endgame_random_state_spawning_rejects_malformed_rng_output():
    class VectorIntegerRng:
        def uniform(self, *args, **kwargs):
            return 0.0

        def integers(self, *args, **kwargs):
            return np.array([1, 2])

    class OutOfRangeIntegerRng:
        def uniform(self, *args, **kwargs):
            return 0.0

        def randint(self, low, high):
            return high

    class NonfiniteUniformRng:
        def uniform(self, *args, **kwargs):
            return float("nan")

    with pytest.raises(ValueError, match="scalar integer endgame random seed"):
        tracking_module._spawn_endgame_random_states(
            {"random_state": VectorIntegerRng()},
            1,
        )
    with pytest.raises(ValueError, match=r"seed in \[0,"):
        tracking_module._spawn_endgame_random_states(
            {"random_state": OutOfRangeIntegerRng()},
            1,
        )
    with pytest.raises(ValueError, match="finite scalar.*endgame random seed"):
        tracking_module._spawn_endgame_random_states(
            {"random_state": NonfiniteUniformRng()},
            1,
        )


def test_randomized_gamma_keeps_close_scalar_roots_separated():
    x = polyvar("x")
    system = PolynomialSystem([x**2 - 2.1 * x + 1.1])

    solutions = solve(system, random_state=1)
    roots = sorted(sol.values[x].real for sol in solutions)

    assert len(solutions) == 2
    np.testing.assert_allclose(roots, [1.0, 1.1], atol=1e-6)


def test_empty_zero_variable_system_returns_empty_solution():
    solutions = solve(PolynomialSystem([]), random_state=0)

    assert len(solutions) == 1
    assert solutions[0].values == {}
    assert solutions[0].residual == 0.0
    assert solutions._meta["total_paths"] == 0


def test_zero_constant_equations_are_removed_before_tracking():
    x = polyvar("x")
    system = PolynomialSystem([x - 1, 0])

    solutions = solve(system, variables=[x], random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 1) < 1e-12
    assert solutions[0].residual == 0.0
    assert solutions._meta["preprocessing"]["removed_zero_equations"] == 1
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["linear_solve"]["status"] == "unique_solution"


def test_inconsistent_constant_equation_returns_empty_solution_set():
    x = polyvar("x")
    system = PolynomialSystem([x - 1, 1])

    solutions = solve(system, variables=[x], random_state=0)

    assert len(solutions) == 0
    assert solutions._meta["total_paths"] == 0
    constants = solutions._meta["preprocessing"]["inconsistent_constants"]
    assert constants[0]["index"] == 1
    assert constants[0]["value"] == {"real": 1.0, "imag": 0.0}


def test_constant_equation_preprocessing_honors_solver_tolerance():
    x = polyvar("x")
    system = PolynomialSystem([x - 1, 1e-15])

    tight = solve(system, variables=[x], tol=1e-16, random_state=0)
    default = solve(system, variables=[x], random_state=0)

    assert len(tight) == 0
    assert tight._meta["preprocessing"]["removed_zero_equations"] == 0
    assert tight._meta["preprocessing"]["constant_tolerance"] == 1e-16
    assert tight._meta["preprocessing"]["inconsistent_constants"] == (
        {
            "index": 1,
            "value": {"real": 1e-15, "imag": 0.0},
        },
    )

    assert len(default) == 1
    assert abs(default[0].values[x] - 1.0) < 1e-12
    assert default._meta["preprocessing"]["removed_zero_equations"] == 1
    assert default._meta["preprocessing"]["constant_tolerance"] == 1e-10


def test_huge_exact_integer_constant_equation_returns_empty_solution_set():
    system = PolynomialSystem([10**400])

    solutions = solve(system, random_state=0)

    assert len(solutions) == 0
    assert solutions._meta["total_paths"] == 0
    constants = solutions._meta["preprocessing"]["inconsistent_constants"]
    assert constants == (
        {
            "index": 0,
            "value": {"real": float("inf"), "imag": 0.0},
        },
    )


def test_scalar_multiple_equations_are_removed_before_direct_solving():
    x, y = polyvar("x", "y")
    repeated = y**2 - 1
    system = PolynomialSystem([x**2 + y - 1, repeated, 3 * repeated])

    solutions = solve(system, variables=[x, y], random_state=0)

    assert len(solutions) == 3
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["start_system"]["source"] == "triangular_direct"
    preprocessing = solutions._meta["preprocessing"]
    assert preprocessing["removed_duplicate_equations"] == 1
    assert preprocessing["duplicate_equations"] == (
        {
            "index": 2,
            "duplicate_of": 1,
            "scale": {"real": 3.0, "imag": 0.0},
        },
    )
    points = sorted(
        (round(solution.values[x].real, 8), round(solution.values[y].real, 8))
        for solution in solutions
    )
    assert points == [
        (round(-np.sqrt(2), 8), -1.0),
        (0.0, 1.0),
        (round(np.sqrt(2), 8), -1.0),
    ]


def test_huge_exact_integer_scalar_multiple_equations_are_removed():
    x = polyvar("x")
    huge = 10**400
    repeated = huge * (x - 1)
    system = PolynomialSystem([repeated, 3 * repeated])

    solutions = solve(system, variables=[x], random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 1.0) < 1e-12
    assert solutions[0].scaled_residual == 0.0
    preprocessing = solutions._meta["preprocessing"]
    assert preprocessing["removed_duplicate_equations"] == 1
    assert preprocessing["duplicate_equations"] == (
        {
            "index": 1,
            "duplicate_of": 0,
            "scale": {"real": 3.0, "imag": 0.0},
        },
    )


def test_preprocessing_keeps_same_support_nonduplicate_equations():
    x = polyvar("x")
    system = PolynomialSystem([x - 1, 2 * x - 2.0001])

    solutions = solve(system, variables=[x], random_state=0)

    assert len(solutions) == 0
    assert solutions._meta["preprocessing"]["removed_duplicate_equations"] == 0
    assert solutions._meta["preprocessing"]["duplicate_equations"] == ()
    assert solutions._meta["linear_solve"]["status"] == "inconsistent"


def test_custom_start_system_preserves_duplicate_target_equation_shape():
    x = polyvar("x")
    target_system = PolynomialSystem([x - 1, 2 * x - 2])
    start_system = PolynomialSystem([x - 2, x - 2])

    solutions = solve(
        target_system,
        start_system=start_system,
        start_solutions=[[2.0 + 0j]],
        tracking_options={"gamma": 1.0 + 0j},
        use_endgame=False,
        random_state=0,
    )

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 1) < 1e-8
    assert solutions._meta["total_paths"] == 1
    assert solutions._meta["start_system"]["source"] == "custom"
    preprocessing = solutions._meta["preprocessing"]
    assert not preprocessing["duplicate_removal_enabled"]
    assert preprocessing["removed_duplicate_equations"] == 0
    assert preprocessing["duplicate_tolerance"] is None


def test_solve_overdetermined_consistent_linear_system_directly():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x - 1, y - 2, x + y - 3])

    solutions = solve(system, variables=[x, y], random_state=123)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 1) < 1e-8
    assert abs(solutions[0].values[y] - 2) < 1e-8
    assert solutions[0].residual < 1e-8
    assert solutions._meta["total_paths"] == 0
    assert solutions[0].path_info["method"] == "linear_direct"
    assert solutions._meta["linear_solve"]["status"] == "unique_solution"
    assert solutions._meta["linear_solve"]["rank"] == 2
    assert solutions._meta["linear_solve"]["matrix_shape"] == (3, 2)


def test_linear_direct_uses_shared_solver_after_lstsq_failure(monkeypatch):
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 3, x - y - 1, 2 * x - 4])
    original_lstsq = np.linalg.lstsq
    calls = {"lstsq": 0}

    def fail_first_lstsq(matrix, rhs, rcond=None):
        calls["lstsq"] += 1
        if calls["lstsq"] == 1:
            raise np.linalg.LinAlgError("raw lstsq failed")
        return original_lstsq(matrix, rhs, rcond=rcond)

    monkeypatch.setattr(np.linalg, "lstsq", fail_first_lstsq)

    solutions = solve(system, variables=[x, y], random_state=123)

    assert calls["lstsq"] >= 2
    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 2) < 1e-8
    assert abs(solutions[0].values[y] - 1) < 1e-8
    assert solutions[0].path_info["method"] == "linear_direct"
    assert solutions._meta["linear_solve"]["status"] == "unique_solution"


def test_solver_records_scaled_residuals_on_solutions():
    x = polyvar("x")
    system = PolynomialSystem([1e12 * (x - 1)])

    solutions = solve(system, random_state=0)

    assert len(solutions) == 1
    assert solutions[0].scaled_residual is not None
    assert solutions[0].scaled_residual <= solutions[0].residual
    assert (
        solutions[0].path_info["scaled_solution_residual"]
        == solutions[0].scaled_residual
    )
    assert (
        solutions._meta["linear_solve"]["scaled_residual_norm"]
        == solutions[0].scaled_residual
    )


def test_solver_scales_large_coefficient_equations_by_default():
    x = polyvar("x")
    system = PolynomialSystem([1e12 * (x - 1)])

    solutions = solve(system, random_state=0)

    scaling = solutions._meta["equation_scaling"]
    assert scaling["enabled"]
    assert scaling["method"] == "coefficient_max_norm"
    assert scaling["coefficient_scales"] == (1e12,)
    assert scaling["scaled_equation_count"] == 1
    assert abs(solutions[0].values[x] - 1) < 1e-12


def test_solver_scales_huge_exact_integer_coefficients():
    x = polyvar("x")
    huge = 10**400
    system = PolynomialSystem([huge * (x - 1)])

    solutions = solve(system, random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 1.0) < 1e-12
    assert solutions[0].residual == 0.0
    assert solutions[0].scaled_residual == 0.0
    scaling = solutions._meta["equation_scaling"]
    assert scaling["enabled"]
    assert scaling["coefficient_scales"] == (float("inf"),)
    assert scaling["max_coefficient_scale"] == float("inf")
    assert scaling["scaled_equation_count"] == 1
    linear_solve = solutions._meta["linear_solve"]
    assert linear_solve["status"] == "unique_solution"
    assert linear_solve["residual_norm"] == 0.0
    assert linear_solve["scaled_residual_norm"] == 0.0


def test_solver_equation_scaling_can_be_disabled():
    x = polyvar("x")
    system = PolynomialSystem([1e12 * (x - 1)])

    solutions = solve(system, random_state=0, scale_equations=False)

    scaling = solutions._meta["equation_scaling"]
    assert not scaling["enabled"]
    assert scaling["method"] == "none"
    assert scaling["coefficient_scales"] == (1e12,)
    assert scaling["scaled_equation_count"] == 0
    assert abs(solutions[0].values[x] - 1) < 1e-12


def test_solver_rejects_nonboolean_equation_scaling_option():
    x = polyvar("x")

    with pytest.raises(TypeError, match="scale_equations must be a boolean"):
        solve(PolynomialSystem([x - 1]), scale_equations="yes", random_state=0)


def test_equation_scaling_preserves_tiny_relative_terms():
    x = polyvar("x")
    system = PolynomialSystem([1e16 * x + 1])

    solutions = solve(system, random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] + 1e-16) < 1e-24
    assert solutions._meta["equation_scaling"]["coefficient_scales"] == (1e16,)


def test_equation_scaling_skips_extreme_coefficients_that_would_drop_terms():
    x, y = polyvar("x", "y")
    huge = 10**400
    system = PolynomialSystem([huge * x + y**4 - huge])

    scaled_system, scaling = solver_module._scale_equation_system(
        system,
        enabled=True,
    )
    residuals = evaluate_scaled_system_at_point(
        scaled_system,
        [1.0 + 0j, 1e100 + 0j],
        [x, y],
    )

    assert scaled_system.equations[0] is system.equations[0]
    assert scaling["coefficient_scales"] == (float("inf"),)
    assert scaling["scaled_equation_count"] == 0
    assert scaling["skipped_destructive_scaling_count"] == 1
    assert np.isclose(residuals[0], 1.0 + 0j, rtol=1e-12, atol=1e-12)


def test_solver_preserves_tiny_absolute_linear_equations():
    x = polyvar("x")
    system = PolynomialSystem([1e-20 * (x - 1)])

    solutions = solve(system, random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 1.0) < 1e-12
    assert solutions._meta["linear_solve"]["status"] == "unique_solution"
    assert solutions._meta["equation_scaling"]["coefficient_scales"] == (1e-20,)


def test_univariate_direct_accepts_backward_stable_extreme_roots():
    x = polyvar("x")
    system = PolynomialSystem([1e-16 * x**2 + x - 1])

    solutions = solve(system, random_state=0)
    roots = sorted(solutions, key=lambda solution: solution.values[x].real)

    assert len(roots) == 2
    assert abs(roots[0].values[x].real + 1e16) / 1e16 < 1e-12
    assert abs(roots[1].values[x].real - 1.0) < 1e-12
    assert roots[0].residual > 0.1
    assert roots[0].backward_error < 1e-12
    assert (
        roots[0].path_info["backward_error"]
        == roots[0].backward_error
        == _backward_error_norm(system, roots[0].values)
    )
    assert roots[0].as_dict()["backward_error"] == roots[0].backward_error

    meta = solutions._meta["univariate_solve"]
    assert meta["accepted_root_count"] == 2
    assert meta["accepted_root_candidate_count"] == 2
    assert meta["max_backward_error"] < 1e-12

    audit = solutions.diagnostics(tolerance=1e-8)
    assert not audit.all_valid
    assert audit.all_backward_stable
    assert audit.backward_stable_count == 2
    assert audit.max_backward_error < 1e-8


def test_univariate_direct_falls_back_when_companion_roots_fail(monkeypatch):
    x = polyvar("x")
    system = PolynomialSystem([x**2 + x + 1])

    def fail_roots(coefficients):
        raise np.linalg.LinAlgError("companion eigenproblem failed")

    monkeypatch.setattr(np, "roots", fail_roots)

    solutions = solve(system, variables=[x], random_state=0)

    assert len(solutions) == 2
    assert solutions._meta["start_system"]["source"] == "total_degree"
    assert "univariate_solve" not in solutions._meta
    for solution in solutions:
        assert solution.scaled_residual < 1e-8
        assert solution.backward_error < 1e-8


def test_univariate_direct_recovers_large_roots_from_underflowed_leading_term():
    x = polyvar("x")
    huge = 10**400
    system = PolynomialSystem([x**4 + huge * x - huge])

    solutions = solve(
        system,
        variables=[x],
        scale_equations=False,
        random_state=0,
    )
    roots = sorted((solution.values[x] for solution in solutions), key=abs)
    large_root_scale = 10 ** (400 / 3)

    assert len(roots) == 4
    assert abs(roots[0] - 1.0) < 1e-12
    assert all(
        abs(abs(root) - large_root_scale) / large_root_scale < 1e-12
        for root in roots[1:]
    )
    assert max(solution.backward_error for solution in solutions) < 1e-10

    meta = solutions._meta["univariate_solve"]
    assert meta["degree"] == 4
    assert meta["root_solver"]["coefficients_lossy"]
    assert meta["root_solver"]["method"] == "lossy_scaled_companion_roots"
    assert meta["root_solver"]["scaled_candidate_root_count"] == 4
    assert meta["factorization"]["status"] == "skipped_lossy_coefficients"


def test_custom_start_tracking_uses_scaled_target_equations():
    x = polyvar("x")
    target_system = PolynomialSystem([1e9 * (x - 2)])
    start_system = PolynomialSystem([x - 1])

    solutions = solve(
        target_system,
        start_system=start_system,
        start_solutions=[[1.0 + 0j]],
        tracking_options={"gamma": 1.0 + 0j},
        use_endgame=False,
    )

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 2.0) < 1e-8
    assert solutions._meta["equation_scaling"]["coefficient_scales"] == (2e9,)
    assert solutions._meta["start_system"]["source"] == "custom"


def test_solve_overdetermined_inconsistent_linear_system_directly():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x - 1, y - 2, x + y - 4])

    solutions = solve(system, variables=[x, y], random_state=123)

    assert len(solutions) == 0
    assert solutions._meta["raw_solutions_found"] == 0
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["linear_solve"]["status"] == "inconsistent"
    assert solutions._meta["linear_solve"]["augmented_rank"] == 3


def test_linear_direct_row_scales_huge_exact_coefficients_when_global_scaling_disabled():
    x, y = polyvar("x", "y")
    huge = 10**400
    system = PolynomialSystem([huge * x - huge, y - 2])

    solutions = solve(
        system,
        variables=[x, y],
        scale_equations=False,
        random_state=0,
    )

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 1) < 1e-12
    assert abs(solutions[0].values[y] - 2) < 1e-12
    assert solutions._meta["equation_scaling"]["enabled"] is False
    assert solutions._meta["start_system"]["source"] == "linear_direct"
    linear_meta = solutions._meta["linear_solve"]
    assert linear_meta["status"] == "unique_solution"
    assert linear_meta["row_scaling_method"] == "coefficient_max_norm"
    assert linear_meta["row_coefficient_scales"] == (float("inf"), 2.0)


def test_linear_direct_column_scales_tiny_row_scaled_pivots():
    x, y = polyvar("x", "y")
    huge = 10**400
    system = PolynomialSystem([huge * x + y - huge, y - 1e100])

    solutions = solve(
        system,
        variables=[x, y],
        scale_equations=False,
        random_state=0,
    )

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 1.0) < 1e-12
    assert abs(solutions[0].values[y] - 1e100) / 1e100 < 1e-12
    assert np.isinf(solutions[0].residual)
    assert solutions[0].scaled_residual == 0.0
    assert solutions[0].backward_error == 0.0
    assert solutions[0].path_info["backward_error"] == 0.0

    linear_meta = solutions._meta["linear_solve"]
    assert linear_meta["status"] == "unique_solution"
    assert linear_meta["rank"] == 2
    assert linear_meta["augmented_rank"] == 2
    assert linear_meta["column_scaling_method"] == "column_max_norm"
    assert linear_meta["column_scales"] == (1.0, 1e-100)
    assert linear_meta["residual_norm"] == float("inf")
    assert linear_meta["scaled_residual_norm"] == 0.0
    assert linear_meta["backward_error_norm"] == 0.0


def test_solve_rank_deficient_consistent_linear_system_raises():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1, 2 * x + 2 * y - 2])

    with pytest.raises(ValueError, match="infinitely many solutions"):
        solve(system, variables=[x, y], random_state=0)


def test_solve_polishes_regular_endpoint_against_original_system():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x, y])

    solutions = solve(
        system,
        start_system=PolynomialSystem([x - 1, y - 1]),
        start_solutions=[[1.0 + 0j, 1.0 + 0j]],
        tracking_options={"gamma": 1.0 + 0j},
        use_endgame=False,
    )

    assert len(solutions) == 1
    assert abs(solutions[0].values[x]) < 1e-15
    assert abs(solutions[0].values[y]) < 1e-15
    assert solutions[0].path_info["solution_polish"]["attempted"]
    assert solutions[0].path_info["solution_residual"] < 1e-15


def test_solve_reports_multiple_root_multiplicity():
    x = polyvar("x")
    system = PolynomialSystem([x**2])

    solutions = solve(system, random_state=0)

    assert len(solutions) == 1
    solution = solutions[0]
    assert abs(solution.values[x]) < 1e-12
    assert solution.residual == 0.0
    assert solution.is_singular
    assert solution.multiplicity == 2
    assert solution.path_indices == ()
    assert solution.root_indices == (0, 1)
    assert solution.path_info["cluster"]["multiplicity"] == 2
    assert solution.path_info["cluster"]["root_indices"] == (0, 1)
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["start_system"]["source"] == "univariate_direct"
    assert solutions._meta["univariate_solve"]["degree"] == 2
    assert solutions._meta["multiplicity_summary"] == {
        "distinct_solutions": 1,
        "total_multiplicity": 2,
        "max_multiplicity": 2,
        "multiple_root_count": 1,
    }

    audit = solutions.diagnostics(tolerance=1e-8)
    assert audit.rank_deficient_count == 1
    assert audit.certified_regular_count == 0


def test_solve_roots_repeated_univariate_factor_once_when_sympy_available():
    pytest.importorskip("sympy")
    x = polyvar("x")
    system = PolynomialSystem([(x - 1) ** 8])

    solutions = solve(system, random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 1) < 1e-12
    assert solutions[0].multiplicity == 8
    assert solutions[0].root_indices == tuple(range(8))
    assert solutions._meta["raw_solutions_found"] == 8

    meta = solutions._meta["univariate_solve"]
    assert meta["method"] == "factor_companion_roots"
    assert meta["raw_root_count"] == 8
    assert meta["raw_root_candidate_count"] == 1
    assert meta["accepted_root_count"] == 8
    assert meta["accepted_root_candidate_count"] == 1
    assert meta["factorization"]["used"]
    assert meta["factorization"]["method"] == "sympy_factor_list"
    assert meta["factorization"]["original_degree"] == 8
    assert meta["factorization"]["distinct_factor_degree"] == 1
    assert meta["factorization"]["candidate_root_count"] == 1
    assert meta["factorization"]["factors"][0]["multiplicity"] == 8
    assert solutions._meta["multiplicity_summary"]["total_multiplicity"] == 8


def test_sympy_conversion_preserves_extreme_coefficients_and_hidden_terms():
    sp = pytest.importorskip("sympy")
    x, y, z = polyvar("x", "y", "z")
    huge = 10**400

    symbols, expr = solver_module._polynomial_to_sympy_expr(
        x**4 + huge * x - huge,
        [x],
        sp,
    )

    assert expr is not None
    expanded = sp.expand(expr)
    poly = sp.Poly(expanded, *symbols)
    assert poly.degree(symbols[0]) == 4
    assert poly.coeff_monomial((4,)) == 1
    assert poly.coeff_monomial((1,)) == huge
    assert expanded.subs({symbols[0]: 1}) == 1

    symbols, expr = solver_module._polynomial_to_sympy_expr(
        (huge * x + y - huge) * (z - 1),
        [x, y, z],
        sp,
    )

    assert expr is not None
    expanded = sp.expand(expr)
    poly = sp.Poly(expanded, *symbols)
    assert poly.total_degree() == 2
    assert poly.coeff_monomial((1, 0, 1)) == huge
    assert poly.coeff_monomial((1, 0, 0)) == -huge
    assert poly.coeff_monomial((0, 1, 1)) == 1
    assert poly.coeff_monomial((0, 1, 0)) == -1
    assert expanded.subs({
        symbols[0]: 1,
        symbols[1]: 2,
        symbols[2]: 2,
    }) == 2

    roundtrip = solver_module._sympy_expr_to_polynomial(
        expr,
        [x, y, z],
        symbols,
        sp,
    )

    assert roundtrip is not None
    _, roundtrip_expr = solver_module._polynomial_to_sympy_expr(
        roundtrip,
        [x, y, z],
        sp,
    )
    assert sp.expand(roundtrip_expr - expanded) == 0


def test_solve_preserves_multiplicity_for_repeated_nonlinear_univariate_factor():
    pytest.importorskip("sympy")
    x = polyvar("x")
    system = PolynomialSystem([(x**2 + 1) ** 3])

    solutions = solve(system, random_state=0)
    roots = sorted(solutions, key=lambda solution: solution.values[x].imag)

    assert len(roots) == 2
    assert [solution.multiplicity for solution in roots] == [3, 3]
    assert abs(roots[0].values[x] + 1j) < 1e-12
    assert abs(roots[1].values[x] - 1j) < 1e-12

    meta = solutions._meta["univariate_solve"]
    assert meta["method"] == "factor_companion_roots"
    assert meta["raw_root_count"] == 6
    assert meta["raw_root_candidate_count"] == 2
    assert meta["factorization"]["distinct_factor_degree"] == 2
    assert meta["factorization"]["factors"][0]["degree"] == 2
    assert meta["factorization"]["factors"][0]["multiplicity"] == 3
    assert solutions._meta["multiplicity_summary"] == {
        "distinct_solutions": 2,
        "total_multiplicity": 6,
        "max_multiplicity": 3,
        "multiple_root_count": 2,
    }


def test_solve_keeps_close_regular_roots_separate():
    x = polyvar("x")
    system = PolynomialSystem([(x - 1) * (x - (1 + 1e-6))])

    solutions = solve(system, random_state=0)
    roots = sorted(solution.values[x].real for solution in solutions)

    assert len(solutions) == 2
    assert [solution.multiplicity for solution in solutions] == [1, 1]
    assert not any(solution.is_singular for solution in solutions)
    np.testing.assert_allclose(roots, [1.0, 1.000001], atol=1e-8)
    assert solutions._meta["univariate_solve"]["method"] == "companion_roots"


def test_solve_multi_equation_univariate_common_roots_directly():
    x = polyvar("x")
    system = PolynomialSystem([x**2 - 1, (x + 1) * (x + 2)])

    solutions = solve(system, random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] + 1) < 1e-10
    assert solutions[0].residual < 1e-10
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["start_system"]["source"] == "univariate_direct"
    assert solutions._meta["univariate_solve"]["equation_count"] == 2
    common_factor = solutions._meta["univariate_solve"]["common_factor"]
    if common_factor["used"]:
        assert common_factor["method"] == "sympy_gcd"
        assert solutions._meta["univariate_solve"]["degree"] == 1
        assert solutions._meta["univariate_solve"]["raw_root_count"] == 1
    else:
        assert solutions._meta["univariate_solve"]["raw_root_count"] == 2
    assert solutions._meta["univariate_solve"]["accepted_root_count"] == 1


def test_multi_equation_univariate_roots_common_factor_when_sympy_available():
    pytest.importorskip("sympy")
    x = polyvar("x")
    system = PolynomialSystem([
        (x - 1) * (x - 2) * (x - 3) * (x - 4),
        (x - 2) * (x - 4) * (x - 5) * (x - 6),
    ])

    solutions = solve(system, random_state=0)
    roots = sorted(round(solution.values[x].real) for solution in solutions)

    assert roots == [2, 4]
    assert solutions._meta["total_paths"] == 0
    meta = solutions._meta["univariate_solve"]
    assert meta["common_factor"]["used"]
    assert meta["common_factor"]["method"] == "sympy_gcd"
    assert meta["common_factor"]["selected_equation_degree"] == 4
    assert meta["degree"] == 2
    assert meta["raw_root_count"] == 2
    assert meta["accepted_root_count"] == 2


def test_multi_equation_univariate_uses_lowest_degree_equation():
    x = polyvar("x")
    system = PolynomialSystem([(x - 1) * (x - 2) * (x - 3), (x - 2) * (x - 3)])

    solutions = solve(system, random_state=0)
    roots = sorted(round(solution.values[x].real) for solution in solutions)

    assert roots == [2, 3]
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["univariate_solve"]["solving_equation_index"] == 1
    assert solutions._meta["univariate_solve"]["degree"] == 2
    assert solutions._meta["univariate_solve"]["raw_root_count"] == 2


def test_inconsistent_multi_equation_univariate_system_returns_no_roots():
    x = polyvar("x")
    system = PolynomialSystem([x**2 - 1, x**2 + 1])

    solutions = solve(system, random_state=0)

    assert len(solutions) == 0
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["start_system"]["source"] == "univariate_direct"
    assert solutions._meta["univariate_solve"]["accepted_root_count"] == 0


def test_univariate_direct_row_scales_huge_exact_coefficients_when_global_scaling_disabled():
    x = polyvar("x")
    huge = 10**400
    system = PolynomialSystem([huge * x**2 - huge])

    solutions = solve(
        system,
        variables=[x],
        scale_equations=False,
        random_state=0,
    )
    roots = sorted(round(solution.values[x].real) for solution in solutions)

    assert roots == [-1, 1]
    assert solutions._meta["equation_scaling"]["enabled"] is False
    assert solutions._meta["start_system"]["source"] == "univariate_direct"
    assert solutions._meta["univariate_solve"]["degree"] == 2
    assert solutions._meta["univariate_solve"]["accepted_root_count"] == 2


def test_custom_univariate_start_system_still_tracks_requested_homotopy():
    x = polyvar("x")
    target_system = PolynomialSystem([x**2 - 1])
    start_system = PolynomialSystem([x**2 - 4])

    solutions = solve(
        target_system,
        start_system=start_system,
        start_solutions=[[2.0 + 0j], [-2.0 + 0j]],
        tracking_options={"gamma": 1.0 + 0j},
        use_endgame=False,
        random_state=0,
    )

    roots = sorted(solution.values[x].real for solution in solutions)
    assert len(solutions) == 2
    np.testing.assert_allclose(roots, [-1.0, 1.0], atol=1e-8)
    assert solutions._meta["total_paths"] == 2
    assert solutions._meta["start_system"]["source"] == "custom"
    assert "univariate_solve" not in solutions._meta


def test_solve_triangular_nonlinear_system_directly():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**2 - 1, y - x**2])

    solutions = solve(system, variables=[x, y], random_state=0)
    points = sorted(
        (round(solution.values[x].real), round(solution.values[y].real))
        for solution in solutions
    )

    assert points == [(-1, 1), (1, 1)]
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["start_system"]["source"] == "triangular_direct"
    assert solutions._meta["triangular_solve"]["method"] == "recursive_univariate"
    assert solutions._meta["triangular_solve"]["branch_count"] == 2
    assert solutions._meta["triangular_solve"]["accepted_branch_count"] == 2
    assert all(
        solution.path_info["method"] == "triangular_direct"
        for solution in solutions
    )


def test_triangular_direct_validates_branches_against_original_system():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**2 - 1, (x + 1) * y + (x - 1)])

    solutions = solve(system, variables=[x, y], random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 1) < 1e-12
    assert abs(solutions[0].values[y]) < 1e-12
    assert solutions._meta["triangular_solve"]["branch_count"] == 1
    assert solutions._meta["triangular_solve"]["accepted_branch_count"] == 1


def test_triangular_direct_rejects_branch_with_only_tiny_raw_residual(monkeypatch):
    x = polyvar("x")
    system = PolynomialSystem([1e-12 * (x - 1)])

    def fake_enumerate_triangular_branches(*args, **kwargs):
        return [
            {
                "assignments": {x: 2.0 + 0.0j},
                "steps": (),
                "root_indices": (0,),
                "multiplicity": 1,
            }
        ], {
            "method": "fake_recursive_univariate",
            "branch_count": 1,
            "branch_candidate_count": 1,
            "steps": (),
            "used_equation_indices": (0,),
        }

    monkeypatch.setattr(
        solver_module,
        "_enumerate_triangular_branches",
        fake_enumerate_triangular_branches,
    )

    solutions = solver_module._solve_triangular_system_direct(
        original_system=system,
        working_system=system,
        variables=[x],
        tol=1e-10,
        start_time=0.0,
        tracker_kwargs={},
        generated_gamma=False,
        preprocessing={},
        equation_scaling={},
        square_up={},
        deduplication_tol_factor=10.0,
        singular_deduplication_tol=1e-8,
    )

    assert len(solutions) == 0
    meta = solutions._meta["triangular_solve"]
    assert meta["branch_count"] == 1
    assert meta["accepted_branch_count"] == 0
    assert meta["accepted_branch_candidate_count"] == 0


def test_triangular_direct_roots_repeated_steps_once_when_sympy_available():
    pytest.importorskip("sympy")
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**6, y**4 - x])

    solutions = solve(system, variables=[x, y], random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x]) < 1e-12
    assert abs(solutions[0].values[y]) < 1e-12
    assert solutions[0].multiplicity == 24
    assert solutions[0].root_indices == tuple(range(24))
    assert solutions._meta["raw_solutions_found"] == 24

    meta = solutions._meta["triangular_solve"]
    assert meta["method"] == "recursive_univariate"
    assert meta["branch_count"] == 24
    assert meta["branch_candidate_count"] == 1
    assert meta["accepted_branch_count"] == 24
    assert meta["accepted_branch_candidate_count"] == 1
    assert meta["distinct_solution_count"] == 1
    assert [step["root_count"] for step in meta["steps"]] == [6, 4]
    assert [step["root_candidate_count"] for step in meta["steps"]] == [1, 1]
    assert all(step["factorization"]["used"] for step in meta["steps"])
    assert solutions[0].path_info["branch_multiplicity"] == 24
    assert solutions._meta["multiplicity_summary"] == {
        "distinct_solutions": 1,
        "total_multiplicity": 24,
        "max_multiplicity": 24,
        "multiple_root_count": 1,
    }


def test_triangular_constant_checks_honor_tolerance_after_substitution():
    x = polyvar("x")

    loose_branches, loose_meta = solver_module._enumerate_triangular_branches(
        PolynomialSystem([x, 1e-8]),
        [x],
        tol=1e-10,
    )
    tight_branches, tight_meta = solver_module._enumerate_triangular_branches(
        PolynomialSystem([x, 1e-8]),
        [x],
        tol=1e-12,
    )

    assert len(loose_branches) == 1
    assert loose_meta["branch_count"] == 1
    assert loose_branches[0]["assignments"][x] == 0
    assert tight_branches == []
    assert tight_meta["branch_count"] == 0


def test_triangular_constant_checks_handle_huge_exact_constants():
    x = polyvar("x")

    branches, meta = solver_module._enumerate_triangular_branches(
        PolynomialSystem([x, 10**400]),
        [x],
        tol=1e-10,
    )

    assert branches == []
    assert meta["branch_count"] == 0


def test_custom_start_system_bypasses_triangular_direct_path():
    x, y = polyvar("x", "y")
    target_system = PolynomialSystem([x**2 - 1, y - x**2])
    start_system = PolynomialSystem([x**2 - 4, y - 4])

    solutions = solve(
        target_system,
        start_system=start_system,
        start_solutions=[
            [2.0 + 0j, 4.0 + 0j],
            [-2.0 + 0j, 4.0 + 0j],
        ],
        tracking_options={"gamma": 1.0 + 0j},
        use_endgame=False,
        random_state=0,
    )

    roots = sorted(solution.values[x].real for solution in solutions)
    assert len(solutions) == 2
    np.testing.assert_allclose(roots, [-1.0, 1.0], atol=1e-8)
    assert solutions._meta["start_system"]["source"] == "custom"
    assert "triangular_solve" not in solutions._meta


def test_solve_full_rank_binomial_system_directly():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x * y - 1, x**2 - y])

    solutions = solve(system, variables=[x, y], random_state=0)

    assert len(solutions) == 3
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["start_system"]["source"] == "binomial_direct"
    assert solutions._meta["binomial_solve"]["method"] == "log_lift_enumeration"
    assert solutions._meta["binomial_solve"]["torus_solution_count"] == 3
    assert all(
        solution.path_info["method"] == "binomial_direct"
        for solution in solutions
    )
    for solution in solutions:
        x_value = solution.values[x]
        y_value = solution.values[y]
        assert abs(x_value) > 1e-8
        assert abs(y_value) > 1e-8
        assert abs(x_value**3 - 1) < 1e-8
        assert abs(y_value - x_value**2) < 1e-8
        assert solution.residual < 1e-8


def test_binomial_direct_does_not_depend_on_explicit_inverse(monkeypatch):
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x * y - 1, x**2 - y])

    def fail_inverse(matrix):
        raise np.linalg.LinAlgError("inverse failed")

    monkeypatch.setattr(np.linalg, "inv", fail_inverse)

    solutions = solve(system, variables=[x, y], random_state=0)

    assert len(solutions) == 3
    assert solutions._meta["start_system"]["source"] == "binomial_direct"
    assert solutions._meta["binomial_solve"]["status"] == "solved"
    assert solutions._meta["binomial_solve"]["torus_solution_count"] == 3


def test_binomial_torus_enumeration_uses_overflow_safe_uniqueness_norm():
    candidates, meta = solver_module._enumerate_binomial_torus_candidates(
        np.array([[2]], dtype=int),
        np.array([1e308 + 0j], dtype=complex),
        solution_count=2,
        tol=1e152,
        max_candidate_count=10,
    )

    assert meta["status"] == "used"
    assert len(candidates) == 1
    assert np.isfinite(candidates[0][0])


def test_direct_and_lifted_solution_paths_record_backward_error():
    x, y = polyvar("x", "y")
    cases = [
        ("binomial_direct", PolynomialSystem([x * y - 1, x**2 - y])),
        ("independent_blocks", PolynomialSystem([x**2 - 1, y**2 - 1])),
        ("coordinate_reduction", PolynomialSystem([y - 1, x**2 + y])),
        ("linear_reduction", PolynomialSystem([x + y - 1, y**2 - 1])),
    ]

    for label, system in cases:
        solutions = solve(system, variables=[x, y], random_state=0)
        assert len(solutions) > 0, label
        for solution in solutions:
            assert solution.backward_error is not None, label
            assert solution.backward_error == _backward_error_norm(
                system,
                solution.values,
            )
            assert solution.path_info["backward_error"] == solution.backward_error

    refined = solver_module.refine_solution(
        PolynomialSystem([x - 1]),
        Solution({x: 0.9 + 0j}, residual=0.1),
        [x],
    )
    assert refined.backward_error == 0.0
    assert refined.refinement["initial_backward_error"] > 0.0
    assert refined.refinement["final_backward_error"] == 0.0


def test_binomial_direct_row_scales_huge_exact_coefficients_when_global_scaling_disabled():
    x, y = polyvar("x", "y")
    huge = 10**400
    system = PolynomialSystem([huge * x * y - huge, huge * x**2 - huge * y])

    solutions = solve(
        system,
        variables=[x, y],
        scale_equations=False,
        max_paths=10,
        random_state=0,
    )

    assert len(solutions) == 3
    assert solutions._meta["equation_scaling"]["enabled"] is False
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["start_system"]["source"] == "binomial_direct"
    assert solutions._meta["binomial_solve"]["torus_solution_count"] == 3
    for solution in solutions:
        x_value = solution.values[x]
        y_value = solution.values[y]
        assert abs(x_value**3 - 1) < 1e-8
        assert abs(y_value - x_value**2) < 1e-8


def test_binomial_direct_uses_hermite_lift_representatives_for_large_lattice():
    pytest.importorskip("sympy")
    x, y, z = polyvar("x", "y", "z")
    system = PolynomialSystem([x * y - 1, y**6 * z - 1, x * z**10 - 1])

    solutions = solve(system, variables=[x, y, z], max_paths=100, random_state=0)

    assert len(solutions) == 61
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["start_system"]["source"] == "binomial_direct"
    meta = solutions._meta["binomial_solve"]
    assert meta["torus_solution_count"] == 61
    assert meta["lift_enumeration_method"] == "hermite_lift_representatives"
    assert meta["enumerated_lift_count"] == 61
    assert meta["lift_representative_count"] == 61
    assert meta["lift_search_space"] == 61
    diagonal_product = 1
    for value in meta["hermite_diagonal"]:
        diagonal_product *= value
    assert diagonal_product == 61
    assert max(solution.residual for solution in solutions) < 1e-8


def test_rank_deficient_binomial_inconsistency_returns_empty_directly():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x * y - 1, x * y - 2])

    solutions = solve(system, variables=[x, y], max_paths=1, random_state=0)

    assert len(solutions) == 0
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["start_system"]["source"] == "binomial_direct"
    meta = solutions._meta["binomial_solve"]
    assert meta["status"] == "inconsistent"
    assert meta["determinant"] == 0
    assert meta["torus_solution_count"] == 0
    assert meta["inconsistent_constraints"][0]["equation_indices"] == (0, 1)


def test_rank_deficient_binomial_inconsistency_row_scales_huge_exact_coefficients_when_global_scaling_disabled():
    x, y = polyvar("x", "y")
    huge = 10**400
    system = PolynomialSystem([huge * x * y - huge, huge * x * y - 2 * huge])

    solutions = solve(
        system,
        variables=[x, y],
        scale_equations=False,
        max_paths=1,
        random_state=0,
    )

    assert len(solutions) == 0
    assert solutions._meta["equation_scaling"]["enabled"] is False
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["binomial_solve"]["status"] == "inconsistent"


def test_rank_deficient_binomial_power_inconsistency_returns_empty_directly():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**2 * y**2 - 1, x * y - 2])

    solutions = solve(system, variables=[x, y], max_paths=1, random_state=0)

    assert len(solutions) == 0
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["binomial_solve"]["status"] == "inconsistent"


def test_rank_deficient_binomial_dependency_inconsistency_returns_empty_directly():
    x, y, z = polyvar("x", "y", "z")
    system = PolynomialSystem([x * y - 1, y * z - 1, x * y**2 * z - 3])

    solutions = solve(system, variables=[x, y, z], max_paths=1, random_state=0)

    assert len(solutions) == 0
    assert solutions._meta["total_paths"] == 0
    meta = solutions._meta["binomial_solve"]
    assert meta["status"] == "inconsistent"
    assert meta["determinant"] == 0
    assert meta["inconsistent_constraints"][0]["equation_indices"] == (0, 1, 2)
    assert meta["inconsistent_constraints"][0]["relation"] == (1, 1, -1)


def test_rank_deficient_binomial_consistent_dependency_is_not_marked_empty():
    x, y, z = polyvar("x", "y", "z")
    system = PolynomialSystem([x * y - 1, y * z - 1, x * y**2 * z - 1])

    with pytest.raises(ValueError, match="rank-deficient.*infinitely many"):
        solve(system, variables=[x, y, z], max_paths=1, random_state=0)


def test_custom_start_system_bypasses_binomial_direct_path():
    x, y = polyvar("x", "y")
    target_system = PolynomialSystem([x * y - 1, x**2 - y])
    omega = np.exp(2j * np.pi / 3)
    start_solutions = [
        [1.0 + 0j, 1.0 + 0j],
        [omega, omega**2],
        [omega**2, omega],
    ]

    solutions = solve(
        target_system,
        start_system=target_system,
        start_solutions=start_solutions,
        variables=[x, y],
        tracking_options={"gamma": 1.0 + 0j},
        use_endgame=False,
        random_state=0,
    )

    assert len(solutions) == 3
    assert solutions._meta["total_paths"] == 3
    assert solutions._meta["start_system"]["source"] == "custom"
    assert "binomial_solve" not in solutions._meta


def test_monomial_zero_equation_splits_coordinate_branches():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x * y, x**2 + y**2 - 1])

    solutions = solve(system, variables=[x, y], random_state=0)
    points = sorted(
        (round(solution.values[x].real), round(solution.values[y].real))
        for solution in solutions
    )

    assert points == [(-1, 0), (0, -1), (0, 1), (1, 0)]
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["start_system"]["source"] == "monomial_zero_branches"
    branch_meta = solutions._meta["monomial_zero_branches"]
    assert branch_meta["equation_index"] == 0
    assert branch_meta["branch_variables"] == ("x", "y")
    assert branch_meta["accepted_branch_solution_count"] == 4
    assert branch_meta["distinct_solution_count"] == 4
    assert [branch["start_source"] for branch in branch_meta["branches"]] == [
        "univariate_direct",
        "univariate_direct",
    ]
    assert all(
        solution.path_info["method"] == "monomial_zero_branches"
        for solution in solutions
    )


def test_monomial_zero_branch_overlaps_are_deduplicated():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x * y, x**2 + y**2])

    solutions = solve(system, variables=[x, y], random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x]) < 1e-12
    assert abs(solutions[0].values[y]) < 1e-12
    assert solutions[0].is_singular
    assert solutions[0].multiplicity == 4
    branch_meta = solutions._meta["monomial_zero_branches"]
    assert branch_meta["accepted_branch_solution_count"] == 2
    assert branch_meta["distinct_solution_count"] == 1
    assert solutions._meta["multiplicity_summary"]["total_multiplicity"] == 4


def test_monomial_zero_split_rejects_positive_dimensional_branch():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**2, x * y])

    with pytest.raises(ValueError, match="positive-dimensional.*y.*witness-set"):
        solve(system, variables=[x, y], random_state=0)


def test_monomial_zero_split_keeps_finite_branch_when_other_branch_inconsistent():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x * y, (x - 1) ** 2])

    solutions = solve(system, variables=[x, y], random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 1.0) < 1e-12
    assert abs(solutions[0].values[y]) < 1e-12
    assert solutions[0].multiplicity == 2
    assert solutions._meta["start_system"]["source"] == "monomial_zero_branches"
    assert solutions._meta["monomial_zero_branches"]["distinct_solution_count"] == 1


def test_monomial_branch_free_variable_check_honors_solver_tolerance():
    x = polyvar("x")
    reduced_system = PolynomialSystem([1e-15])

    assert not solver_module._reduced_branch_has_free_variables(
        reduced_system,
        [x],
        tol=1e-16,
    )
    assert solver_module._reduced_branch_has_free_variables(
        reduced_system,
        [x],
        tol=1e-10,
    )


def test_monomial_branch_free_variable_check_handles_huge_exact_constants():
    x = polyvar("x")
    reduced_system = PolynomialSystem([10**400])

    assert not solver_module._reduced_branch_has_free_variables(
        reduced_system,
        [x],
        tol=1e-10,
    )


def test_common_monomial_factor_splits_coordinate_and_cofactor_branches():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x * (y - 1), x**2 + y**2 - 1])

    solutions = solve(system, variables=[x, y], random_state=0)
    points = sorted(
        (round(solution.values[x].real), round(solution.values[y].real))
        for solution in solutions
    )

    assert points == [(0, -1), (0, 1)]
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["start_system"]["source"] == "monomial_zero_branches"
    assert "factorized_branches" not in solutions._meta
    branch_meta = solutions._meta["monomial_zero_branches"]
    assert branch_meta["common_factor_kind"] == "common_monomial_factor"
    assert branch_meta["branch_variables"] == ("x",)
    assert branch_meta["cofactor"] == "y - 1"
    assert branch_meta["cofactor_degree"] == 1
    assert branch_meta["accepted_branch_solution_count"] == 3
    assert branch_meta["distinct_solution_count"] == 2
    assert [branch["branch_type"] for branch in branch_meta["branches"]] == [
        "coordinate",
        "cofactor",
    ]
    assert max(solution.multiplicity for solution in solutions) == 3


def test_factorized_equation_splits_nontrivial_branches_when_sympy_available():
    pytest.importorskip("sympy")
    x, y = polyvar("x", "y")
    system = PolynomialSystem([(x - 1) * (y - 2), x**2 + y**2 - 5])

    solutions = solve(system, variables=[x, y], random_state=0)
    points = sorted(
        (round(solution.values[x].real), round(solution.values[y].real))
        for solution in solutions
    )

    assert points == [(-1, 2), (1, -2), (1, 2)]
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["start_system"]["source"] == "factorized_branches"
    branch_meta = solutions._meta["factorized_branches"]
    assert branch_meta["method"] == "sympy_factor_list"
    assert branch_meta["factor_count"] == 2
    assert branch_meta["accepted_branch_solution_count"] == 4
    assert branch_meta["distinct_solution_count"] == 3
    assert all(
        solution.path_info["method"] == "factorized_branches"
        for solution in solutions
    )
    assert all(solution.backward_error is not None for solution in solutions)
    assert all(
        solution.path_info["backward_error"] == solution.backward_error
        for solution in solutions
    )


def test_factorized_equation_preserves_repeated_branch_multiplicity():
    pytest.importorskip("sympy")
    x, y = polyvar("x", "y")
    system = PolynomialSystem([(x - 1) ** 2 * (y - 2), x**2 + y**2 - 5])

    solutions = solve(system, variables=[x, y], random_state=0)
    summary = sorted(
        (
            round(solution.values[x].real),
            round(solution.values[y].real),
            solution.multiplicity,
        )
        for solution in solutions
    )

    assert summary == [(-1, 2, 1), (1, -2, 2), (1, 2, 3)]
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["start_system"]["source"] == "factorized_branches"
    branch_meta = solutions._meta["factorized_branches"]
    assert branch_meta["factor_count"] == 2
    assert branch_meta["original_degree"] == 3
    assert branch_meta["distinct_factor_degree"] == 2
    assert branch_meta["total_factor_degree"] == 3
    assert branch_meta["accepted_branch_solution_count"] == 4
    assert branch_meta["distinct_solution_count"] == 3
    assert sorted(factor["multiplicity"] for factor in branch_meta["factors"]) == [
        1,
        2,
    ]
    assert solutions._meta["multiplicity_summary"] == {
        "distinct_solutions": 3,
        "total_multiplicity": 6,
        "max_multiplicity": 3,
        "multiple_root_count": 2,
    }


def test_factorized_branch_overlaps_are_deduplicated_when_sympy_available():
    pytest.importorskip("sympy")
    x, y = polyvar("x", "y")
    system = PolynomialSystem([(x - 1) * (y - 1), (x - y)**2 + (x - 1)**2])

    solutions = solve(system, variables=[x, y], random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 1) < 1e-10
    assert abs(solutions[0].values[y] - 1) < 1e-10
    assert solutions[0].is_singular
    assert solutions[0].multiplicity == 4
    branch_meta = solutions._meta["factorized_branches"]
    assert branch_meta["accepted_branch_solution_count"] == 2
    assert branch_meta["distinct_solution_count"] == 1
    assert solutions._meta["multiplicity_summary"]["total_multiplicity"] == 4


def test_factorized_branch_rejects_positive_dimensional_component():
    pytest.importorskip("sympy")
    x, y = polyvar("x", "y")
    system = PolynomialSystem([(x - 1) * (y - 2), (x - 1) ** 2])

    with pytest.raises(ValueError, match="positive-dimensional.*y.*witness-set"):
        solve(system, variables=[x, y], random_state=0)


def test_solve_decomposes_independent_nonlinear_blocks():
    x, y, z = polyvar("x", "y", "z")
    system = PolynomialSystem([x * y - 1, x**2 + y**2 - 2, z**2 - 1])

    solutions = solve(system, variables=[x, y, z], random_state=0)
    points = sorted(
        (
            round(solution.values[x].real),
            round(solution.values[y].real),
            round(solution.values[z].real),
        )
        for solution in solutions
    )

    assert points == [(-1, -1, -1), (-1, -1, 1), (1, 1, -1), (1, 1, 1)]
    assert solutions._meta["start_system"]["source"] == "independent_blocks"
    assert solutions._meta["total_paths"] == 4
    assert solutions._meta["independent_blocks"]["block_count"] == 2
    assert solutions._meta["independent_blocks"]["raw_combination_count"] == 4
    assert [block["variables"] for block in solutions._meta["independent_blocks"]["blocks"]] == [
        ("x", "y"),
        ("z",),
    ]
    assert all(
        solution.path_info["method"] == "independent_blocks"
        for solution in solutions
    )


def test_independent_blocks_preserve_product_multiplicity():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**2, y**2])

    solutions = solve(system, variables=[x, y], random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x]) < 1e-12
    assert abs(solutions[0].values[y]) < 1e-12
    assert solutions[0].multiplicity == 4
    assert solutions._meta["start_system"]["source"] == "independent_blocks"
    assert solutions._meta["multiplicity_summary"]["total_multiplicity"] == 4


def test_integer_determinant_handles_row_swaps_and_singular_matrices():
    assert _integer_determinant(np.array([[0, 2], [3, 4]])) == -6
    assert _integer_determinant(np.array([[1, 2], [2, 4]])) == 0
    assert _integer_determinant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])) == -3


def test_solve_reduces_coordinate_assignment_before_univariate_solve():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([y - 1, x**2 + y])

    solutions = solve(system, random_state=0)
    roots = sorted(solutions, key=lambda solution: solution.values[x].imag)

    assert len(solutions) == 2
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["coordinate_reduction"]["assigned_variables"] == ("y",)
    assert solutions._meta["univariate_solve"]["method"] == "companion_roots"
    assert abs(roots[0].values[x] + 1j) < 1e-10
    assert abs(roots[1].values[x] - 1j) < 1e-10
    assert all(abs(solution.values[y] - 1) < 1e-12 for solution in solutions)
    assert all(
        solution.path_info["coordinate_reduction"]["lifted"]
        for solution in solutions
    )


def test_coordinate_reduction_row_scales_huge_exact_coefficients_when_global_scaling_disabled():
    x, y = polyvar("x", "y")
    huge = 10**400
    system = PolynomialSystem([huge * x - huge, y**2 + x])

    solutions = solve(
        system,
        variables=[x, y],
        scale_equations=False,
        random_state=0,
    )
    roots = sorted(solutions, key=lambda solution: solution.values[y].imag)

    assert len(solutions) == 2
    assert solutions._meta["equation_scaling"]["enabled"] is False
    assert solutions._meta["coordinate_reduction"]["assigned_variables"] == ("x",)
    assert all(abs(solution.values[x] - 1) < 1e-12 for solution in solutions)
    assert abs(roots[0].values[y] + 1j) < 1e-10
    assert abs(roots[1].values[y] - 1j) < 1e-10


def test_constant_substitution_preserves_scaled_large_assigned_powers():
    x, y = polyvar("x", "y")
    huge = 10**400
    polynomial = huge * x + y**4 - huge

    reduced = solver_module._substitute_constants(
        polynomial,
        {y: 1e100 + 0j},
    )

    assert reduced.degree() == 1
    assert np.isclose(reduced.evaluate({x: 0.0 + 0j}), 0.0 + 0j, atol=1e-12)
    assert np.isclose(reduced.evaluate({x: 1.0 + 0j}), 1.0 + 0j, atol=1e-12)


def test_coordinate_reduction_preserves_hidden_terms_after_exact_assignments():
    x, y, z = polyvar("x", "y", "z")
    huge = 10**400
    polynomial = (huge * x + y - huge) * (z - 1)

    reduced = solver_module._substitute_constants(
        polynomial,
        {x: 1.0 + 0.0j, y: 2.0 + 0.0j},
    )

    assert reduced.degree() == 1
    assert reduced.evaluate({z: 0.0 + 0.0j}) == -2
    assert reduced.evaluate({z: 2.0 + 0.0j}) == 2

    solutions = solve(
        PolynomialSystem([polynomial, x - 1, y - 2]),
        variables=[x, y, z],
        scale_equations=False,
        random_state=0,
    )

    assert len(solutions) == 1
    assert solutions._meta["coordinate_reduction"]["assigned_variables"] == (
        "x",
        "y",
    )
    assert solutions[0].values[x] == 1
    assert solutions[0].values[y] == 2
    assert solutions[0].values[z] == 1
    assert solutions[0].residual == 0.0
    assert solutions[0].scaled_residual == 0.0
    assert solutions[0].backward_error == 0.0


def test_triangular_solve_preserves_unassigned_terms_that_underflow_when_scaled():
    x, y = polyvar("x", "y")
    huge = 10**400
    system = PolynomialSystem([huge * x + y**4 - huge, y - 1e100])

    selected = solver_module._select_triangular_equation(
        system,
        [0, 1],
        {},
        [x, y],
        cancellation_tol=1e-10,
    )
    solutions = solve(system, variables=[x, y], random_state=0)

    assert selected is not None
    assert selected[0] == 1
    assert selected[1] == y
    assert len(solutions) == 1
    assert abs(solutions[0].values[x]) < 1e-10
    assert abs(solutions[0].values[y] - 1e100) / 1e100 < 1e-12
    assert solutions[0].scaled_residual < 1e-10
    assert solutions[0].backward_error < 1e-10
    assert solutions[0].path_info["backward_error"] == solutions[0].backward_error


def test_triangular_direct_recovers_large_roots_after_exact_branch_substitution():
    x, y = polyvar("x", "y")
    huge = 10**400
    system = PolynomialSystem([x**4 + huge * x - huge + y**2 - 1, y**2 - 1])

    solutions = solve(
        system,
        variables=[x, y],
        scale_equations=False,
        random_state=0,
    )
    roots = sorted(solutions, key=lambda solution: (solution.values[y].real, abs(solution.values[x])))
    large_root_scale = 10 ** (400 / 3)
    small_roots = [solution for solution in roots if abs(solution.values[x]) < 2.0]
    large_roots = [solution for solution in roots if abs(solution.values[x]) >= 2.0]
    triangular_steps = solutions._meta["triangular_solve"]["steps"]

    assert solver_module._binomial_torus_data(system, [x, y]) is None
    assert len(solutions) == 8
    assert len(small_roots) == 2
    assert len(large_roots) == 6
    assert all(abs(solution.values[x] - 1.0) < 1e-12 for solution in small_roots)
    assert all(
        abs(abs(solution.values[x]) - large_root_scale) / large_root_scale < 1e-12
        for solution in large_roots
    )
    assert max(solution.backward_error for solution in solutions) < 1e-10
    assert solutions._meta["start_system"]["source"] == "triangular_direct"
    assert solutions._meta["triangular_solve"]["accepted_branch_count"] == 8
    assert any(
        step["root_solver"]["method"] == "lossy_scaled_companion_roots"
        for step in triangular_steps
    )


def test_coordinate_reduction_preserves_multiple_root_multiplicity():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([y - 1, x**2])

    solutions = solve(system, random_state=0)

    assert len(solutions) == 1
    assert solutions[0].is_singular
    assert solutions[0].multiplicity == 2
    assert solutions[0].root_indices == (0, 1)
    assert solutions[0].values[y] == 1
    assert solutions._meta["raw_solutions_found"] == 2
    assert solutions._meta["multiplicity_summary"]["total_multiplicity"] == 2


def test_coordinate_reduction_detects_inconsistent_substitution():
    x = polyvar("x")
    system = PolynomialSystem([x - 1, x**2 - 4])

    solutions = solve(system, random_state=0)

    assert len(solutions) == 0
    assert solutions._meta["coordinate_reduction"]["assigned_variables"] == ("x",)
    reduced_preprocessing = solutions._meta["coordinate_reduction"]["reduced_meta"][
        "preprocessing"
    ]
    assert reduced_preprocessing["inconsistent_constants"][0]["value"] == {
        "real": -0.75,
        "imag": 0.0,
    }
    assert solutions._meta["equation_scaling"]["coefficient_scales"] == (
        1.0,
        4.0,
    )


def test_coordinate_reduction_rejects_lift_with_only_tiny_raw_residual(monkeypatch):
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x, 1e-12 * (y - 1) ** 2])

    def fake_recursive_solve(reduced_system, *args, **kwargs):
        solution = Solution(values={y: 2.0 + 0.0j}, residual=1e-12)
        solution.scaled_residual = 1.0
        solution.backward_error = 1.0
        result = SolutionSet([solution], reduced_system)
        result._meta.update({
            "raw_solutions_found": 1,
            "total_paths": 0,
            "successful_paths": 0,
            "failed_paths": 0,
        })
        return result

    monkeypatch.setattr(solver_module, "solve", fake_recursive_solve)

    solutions = solve(system, variables=[x, y], random_state=0)

    assert len(solutions) == 0
    assert solutions._meta["raw_solutions_found"] == 0
    meta = solutions._meta["coordinate_reduction"]
    assert meta["accepted_lift_count"] == 0
    assert meta["rejected_lift_count"] == 1
    assert meta["reduced_meta"]["raw_solutions_found"] == 1


def test_solve_reduces_affine_linear_equation_before_univariate_solve():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1, x**2 + y**2 - 1])

    solutions = solve(system, random_state=0)
    points = sorted(
        [(round(solution.values[x].real), round(solution.values[y].real))
         for solution in solutions]
    )

    assert len(solutions) == 2
    assert points == [(0, 1), (1, 0)]
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["linear_reduction"]["eliminated_variable"] in {"x", "y"}
    assert solutions._meta["univariate_solve"]["method"] == "companion_roots"
    assert all(
        solution.path_info["linear_reduction"]["lifted"]
        for solution in solutions
    )


def test_affine_linear_reduction_row_scales_huge_exact_coefficients_when_global_scaling_disabled():
    x, y = polyvar("x", "y")
    huge = 10**400
    system = PolynomialSystem([huge * x + huge * y - huge, y**2 - 1])

    solutions = solve(
        system,
        variables=[x, y],
        scale_equations=False,
        random_state=0,
    )
    points = sorted(
        (round(solution.values[x].real), round(solution.values[y].real))
        for solution in solutions
    )

    assert points == [(0, 1), (2, -1)]
    assert solutions._meta["equation_scaling"]["enabled"] is False
    assert solutions._meta["linear_reduction"]["eliminated_variable"] == "x"
    assert solutions._meta["linear_reduction"]["expression"] == "-y + 1.0"
    assert solutions._meta["univariate_solve"]["method"] == "companion_roots"


def test_affine_substitution_row_scales_huge_exact_nonlinear_rows_when_global_scaling_disabled():
    x, y = polyvar("x", "y")
    huge = 10**400
    system = PolynomialSystem([x + y - 1, huge * x**2 - huge])

    solutions = solve(
        system,
        variables=[x, y],
        scale_equations=False,
        random_state=0,
    )
    points = sorted(
        (round(solution.values[x].real), round(solution.values[y].real))
        for solution in solutions
    )

    assert points == [(-1, 2), (1, 0)]
    assert solutions._meta["equation_scaling"]["enabled"] is False
    assert solutions._meta["linear_reduction"]["eliminated_variable"] == "x"
    assert solutions._meta["univariate_solve"]["method"] == "companion_roots"


def test_affine_linear_reduction_handles_complex_lifts():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1, x * y - 2])

    solutions = solve(system, random_state=0)

    assert len(solutions) == 2
    assert solutions._meta["total_paths"] == 0
    assert solutions._meta["linear_reduction"]["reduced_variables"] == ("y",)
    for solution in solutions:
        assert abs(solution.values[x] + solution.values[y] - 1) < 1e-12
        assert abs(solution.values[x] * solution.values[y] - 2) < 1e-12


def test_affine_linear_reduction_rejects_positive_dimensional_remainder():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1, (x + y - 1) ** 2])

    with pytest.raises(ValueError, match="positive-dimensional.*witness-set"):
        solve(system, random_state=0)


def test_affine_reduction_rejects_lift_with_only_tiny_raw_residual(monkeypatch):
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y, 1e-12 * (y - 1) ** 2])

    def fake_recursive_solve(reduced_system, *args, **kwargs):
        solution = Solution(values={y: 2.0 + 0.0j}, residual=1e-12)
        solution.scaled_residual = 1.0
        solution.backward_error = 1.0
        result = SolutionSet([solution], reduced_system)
        result._meta.update({
            "raw_solutions_found": 1,
            "total_paths": 0,
            "successful_paths": 0,
            "failed_paths": 0,
        })
        return result

    monkeypatch.setattr(solver_module, "solve", fake_recursive_solve)

    solutions = solve(system, variables=[x, y], random_state=0)

    assert len(solutions) == 0
    assert solutions._meta["raw_solutions_found"] == 0
    meta = solutions._meta["linear_reduction"]
    assert meta["accepted_lift_count"] == 0
    assert meta["rejected_lift_count"] == 1
    assert meta["reduced_meta"]["raw_solutions_found"] == 1


@pytest.mark.parametrize("system", [PolynomialSystem([]), PolynomialSystem([0])])
def test_zero_equation_system_with_variables_reports_positive_dimensional(system):
    x = polyvar("x")

    with pytest.raises(ValueError, match="positive-dimensional.*witness-set"):
        solve(system, variables=[x], random_state=0)


def test_solve_overdetermined_square_up_is_reproducible():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([
        x**2 + y - 1,
        y**2 + x - 1,
        x**2 + y**2 - 1,
    ])

    first = solve(system, variables=[x, y], random_state=321)
    second = solve(system, variables=[x, y], random_state=321)
    third = solve(system, variables=[x, y], random_state=322)

    assert (
        first._meta["square_up"]["coefficients"]
        == second._meta["square_up"]["coefficients"]
    )
    assert (
        first._meta["square_up"]["coefficients"]
        != third._meta["square_up"]["coefficients"]
    )


def test_square_up_rejects_malformed_standard_normal_output():
    class WrongShapeRng:
        def uniform(self, *args, **kwargs):
            return 0.0

        def standard_normal(self, size=None):
            return np.zeros((1, 1))

    x, y = polyvar("x", "y")
    system = PolynomialSystem([
        x**2 + y - 1,
        y**2 + x - 1,
        x**2 + y**2 - 1,
    ])

    with pytest.raises(
        ValueError,
        match=r"standard_normal.*shape \(2, 3\).*square-up matrix real part",
    ):
        solve(
            system,
            variables=[x, y],
            random_state=WrongShapeRng(),
            max_paths=20,
        )


def test_square_up_row_scales_huge_exact_coefficients_when_global_scaling_disabled():
    x, y = polyvar("x", "y")
    huge = 10**400
    system = PolynomialSystem([
        huge * (x**2 + y**2 - 1),
        x**2 + y - 1,
        x + y**2 - 1,
    ])

    solutions = solve(
        system,
        variables=[x, y],
        scale_equations=False,
        max_paths=20,
        random_state=0,
    )
    points = sorted(
        (round(solution.values[x].real), round(solution.values[y].real))
        for solution in solutions
    )

    assert points == [(0, 1), (1, 0)]
    assert solutions._meta["equation_scaling"]["enabled"] is False
    assert solutions._meta["start_system"]["source"] == "total_degree"
    assert solutions._meta["square_up"]["method"] == "random_linear_combinations"
    assert solutions._meta["square_up"]["source_equation_scales"] == (
        float("inf"),
        1.0,
        1.0,
    )


def test_endpoint_polish_accepts_endpoint_already_within_tolerance():
    x = polyvar("x")
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    end_solutions, path_results = track_paths(
        start_system=start_system,
        target_system=target_system,
        start_solutions=[[1.0 + 0j]],
        variables=[x],
        tol=1e-10,
        gamma=1.0 + 0j,
        use_endgame=True,
        endgame_options={"newton_max_iters": 0},
    )

    path_info = path_results[0]
    assert path_info["success"]
    assert path_info["polish"]["attempted"]
    assert path_info["polish"]["success"]
    assert path_info["polish"]["accepted"]
    assert path_info["polished"]
    assert path_info["steps"] > 0
    assert abs(end_solutions[0][0] - 2) < 1e-8


def test_track_paths_rejects_mismatched_homotopy_equation_counts():
    x = polyvar("x")

    with pytest.raises(ValueError, match="same number of equations"):
        track_paths(
            start_system=PolynomialSystem([x - 1, x + 1]),
            target_system=PolynomialSystem([x - 2]),
            start_solutions=[[1.0 + 0j]],
            variables=[x],
        )


def test_track_paths_rejects_start_solution_dimension_mismatch():
    x, y = polyvar("x", "y")

    with pytest.raises(ValueError, match="start_solutions\\[0\\]"):
        track_paths(
            start_system=PolynomialSystem([x - 1, y - 1]),
            target_system=PolynomialSystem([x - 2, y - 2]),
            start_solutions=[[1.0 + 0j]],
            variables=[x, y],
        )


def test_track_single_path_newton_failure_reports_trial_state(monkeypatch):
    x = polyvar("x")
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    def failing_corrector(f, jac, point, max_iters=10, tol=1e-10):
        return np.asarray(point, dtype=complex) + np.array([0.25 + 0j]), False, 1

    monkeypatch.setattr(
        tracking_module,
        "newton_corrector_numeric",
        failing_corrector,
    )

    point, info = tracking_module.track_single_path(
        start_system=start_system,
        target_system=target_system,
        start_solution=np.array([1.0 + 0j]),
        variables=[x],
        gamma=1.0 + 0j,
        min_step_size=0.25,
        max_step_size=0.25,
        use_endgame=False,
    )

    expected_residual = tracking_module.homotopy_residual_at(
        start_system,
        target_system,
        point,
        info["final_t"],
        [x],
        1.0 + 0j,
    )
    assert not info["success"]
    assert info["failure_reason"] == "newton_failed"
    assert info["final_t"] == 0.75
    np.testing.assert_allclose(info["final_point"], point)
    assert info["final_residual"] == expected_residual


def test_track_single_path_store_paths_is_quiet_without_verbose(monkeypatch, capsys):
    x = polyvar("x")
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    def failing_corrector(f, jac, point, max_iters=10, tol=1e-10):
        return np.asarray(point, dtype=complex) + np.array([0.25 + 0j]), False, 1

    monkeypatch.setattr(
        tracking_module,
        "newton_corrector_numeric",
        failing_corrector,
    )

    _point, info = tracking_module.track_single_path(
        start_system=start_system,
        target_system=target_system,
        start_solution=np.array([1.0 + 0j]),
        variables=[x],
        gamma=1.0 + 0j,
        min_step_size=0.25,
        max_step_size=0.25,
        store_paths=True,
        use_endgame=False,
        verbose=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert info["failure_reason"] == "newton_failed"
    assert len(info["path_points"]) == 1


def test_track_single_path_nonfinite_corrector_reports_infinite_residual(monkeypatch):
    x = polyvar("x")
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    def nonfinite_corrector(f, jac, point, max_iters=10, tol=1e-10):
        return np.array([np.nan + 0j]), False, 1

    monkeypatch.setattr(
        tracking_module,
        "newton_corrector_numeric",
        nonfinite_corrector,
    )

    point, info = tracking_module.track_single_path(
        start_system=start_system,
        target_system=target_system,
        start_solution=np.array([1.0 + 0j]),
        variables=[x],
        gamma=1.0 + 0j,
        min_step_size=0.25,
        max_step_size=0.25,
        use_endgame=False,
    )

    assert not info["success"]
    assert info["failure_reason"] == "nonfinite_corrector"
    np.testing.assert_allclose(info["final_point"], point)
    assert np.isinf(info["final_residual"])


def test_track_single_path_nonfinite_predictor_reports_trial_state(monkeypatch):
    x = polyvar("x")
    huge = np.finfo(float).max
    start_system = PolynomialSystem([x - huge])
    target_system = PolynomialSystem([x])

    def overflowing_tangent(*args, **kwargs):
        return np.array([-huge + 0j])

    monkeypatch.setattr(
        tracking_module,
        "compute_tangent",
        overflowing_tangent,
    )

    point, info = tracking_module.track_single_path(
        start_system=start_system,
        target_system=target_system,
        start_solution=np.array([huge + 0j]),
        variables=[x],
        gamma=1.0 + 0j,
        min_step_size=1.0,
        max_step_size=1.0,
        use_endgame=False,
    )

    assert not info["success"]
    assert info["failure_reason"] == "nonfinite_predictor"
    assert info["final_t"] == 1.0
    assert info["trial_t"] == 0.0
    assert np.isinf(info["trial_residual"])
    assert not np.all(np.isfinite(info["trial_point"]))
    np.testing.assert_allclose(info["final_point"], point)
    assert info["final_residual"] < 1e-12


def test_track_single_path_rejects_successful_endgame_with_large_residual(monkeypatch):
    import pycontinuum.endgame as endgame_module

    x = polyvar("x")
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    monkeypatch.setattr(
        tracking_module,
        "check_singularity",
        lambda *args, **kwargs: True,
    )

    def fake_run_cauchy_endgame(*args, **kwargs):
        return (
            np.array([100.0 + 0j]),
            {
                "success": True,
                "winding_number": 1,
                "predictions": [],
            },
        )

    monkeypatch.setattr(
        endgame_module,
        "run_cauchy_endgame",
        fake_run_cauchy_endgame,
    )

    point, info = tracking_module.track_single_path(
        start_system=start_system,
        target_system=target_system,
        start_solution=np.array([1.0 + 0j]),
        variables=[x],
        tol=1e-10,
        endgame_start=1.0,
        use_endgame=True,
    )

    assert not info["success"]
    assert info["failure_reason"] == "large_final_residual"
    np.testing.assert_allclose(point, np.array([100.0 + 0j]))
    assert info["final_residual"] > 1.0


def test_singular_path_uses_target_polish_before_cauchy_endgame(monkeypatch):
    x, y = polyvar("x", "y")
    target_system = PolynomialSystem([x * y - 1, x**2 + y**2 - 2])
    start_system, start_solutions = generate_total_degree_start_system(
        target_system,
        [x, y],
        random_state=np.random.default_rng(0),
    )

    def fail_cauchy(*args, **kwargs):
        raise AssertionError("Cauchy endgame should not be needed")

    monkeypatch.setattr(endgame_module, "run_cauchy_endgame", fail_cauchy)

    point, info = tracking_module.track_single_path(
        start_system=start_system,
        target_system=target_system,
        start_solution=start_solutions[0],
        variables=[x, y],
        tol=1e-10,
        gamma=-0.6520162635843662 - 0.7582049802141122j,
        endgame_start=0.1,
        use_endgame=True,
    )

    assert info["success"]
    assert info["singular"]
    assert info["endgame_target_polish"]["accepted"]
    assert info["final_t"] == 0.0
    assert info["final_residual"] < 100 * 1e-10
    np.testing.assert_allclose(point[0], point[1], atol=1e-5)


def test_track_single_path_start_residual_limit_honors_tolerance():
    x = polyvar("x")
    start_system = PolynomialSystem([x])
    target_system = PolynomialSystem([x - 1])

    with pytest.raises(ValueError, match="does not satisfy start_system"):
        tracking_module.track_single_path(
            start_system,
            target_system,
            start_solution=np.array([2e-9 + 0j]),
            variables=[x],
            tol=1e-12,
            use_endgame=False,
        )

    point, info = tracking_module.track_single_path(
        start_system,
        target_system,
        start_solution=np.array([2e-9 + 0j]),
        variables=[x],
        tol=1e-8,
        use_endgame=False,
    )

    assert info["success"]
    assert info["start_residual"] == pytest.approx(2e-9)
    assert info["start_residual_limit"] == pytest.approx(1e-5)
    np.testing.assert_allclose(point, np.array([1.0 + 0j]), atol=1e-8)


def test_tracking_residual_helpers_return_infinity_for_malformed_points():
    x = polyvar("x")
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([x - 2])

    assert np.isinf(
        tracking_module.target_residual_at(
            target_system,
            np.array([2.0 + 0j, 999.0 + 0j]),
            [x],
        )
    )
    assert np.isinf(
        tracking_module.target_residual_at(
            target_system,
            np.array([], dtype=complex),
            [x],
        )
    )
    assert np.isinf(
        tracking_module.homotopy_residual_at(
            start_system,
            target_system,
            np.array([1.0 + 0j, 999.0 + 0j]),
            1.0,
            [x],
            1.0 + 0j,
        )
    )
    assert np.isinf(
        tracking_module.homotopy_residual_at(
            start_system,
            target_system,
            np.array([np.nan + 0j]),
            1.0,
            [x],
            1.0 + 0j,
        )
    )
    assert (
        tracking_module.target_residual_at(
            target_system,
            np.array([2.0 + 0j]),
            [x],
        )
        == 0.0
    )


def test_tracking_residual_helpers_use_scaled_fallback_for_overflowing_values():
    x = polyvar("x")
    huge_system = PolynomialSystem([(10**400) * x])

    target_residual = tracking_module.target_residual_at(
        huge_system,
        np.array([1.0 + 0j]),
        [x],
    )
    assert np.isfinite(target_residual)
    assert target_residual == pytest.approx(1.0)
    homotopy_residual = tracking_module.homotopy_residual_at(
        huge_system,
        huge_system,
        np.array([1.0 + 0j]),
        0.5,
        [x],
        1.0 + 0j,
    )
    assert np.isfinite(homotopy_residual)
    assert homotopy_residual == pytest.approx(1.0)


def test_tracking_residual_helpers_keep_large_finite_norms_finite():
    x, y = polyvar("x", "y")
    huge = 1e200
    huge_system = PolynomialSystem([x, y])
    point = np.array([huge + 0j, huge + 0j])

    target_residual = tracking_module.target_residual_at(
        huge_system,
        point,
        [x, y],
    )
    homotopy_residual = tracking_module.homotopy_residual_at(
        huge_system,
        huge_system,
        point,
        0.5,
        [x, y],
        1.0 + 0j,
    )

    assert np.isfinite(target_residual)
    assert target_residual == pytest.approx(np.sqrt(2.0) * huge)
    assert np.isfinite(homotopy_residual)
    assert homotopy_residual == pytest.approx(np.sqrt(2.0) * huge)


def test_parameter_homotopy_norm_keeps_large_finite_values_finite():
    huge = 1e200
    norm = parameter_homotopy_module._norm_or_inf(
        np.array([huge + 0j, huge + 0j])
    )

    assert np.isfinite(norm)
    assert norm == pytest.approx(np.sqrt(2.0) * huge)


def test_tracking_residual_helpers_use_scaled_residual_for_finite_rows():
    x = polyvar("x")
    large_system = PolynomialSystem([1e12 * (x - 2)])
    tiny_system = PolynomialSystem([1e-12 * (x - 2)])

    assert tracking_module.target_residual_at(
        large_system,
        np.array([2.0 + 1e-10 + 0j]),
        [x],
    ) < 1e-8
    assert tracking_module.target_residual_at(
        tiny_system,
        np.array([1.0 + 0j]),
        [x],
    ) == pytest.approx(0.5)


def test_check_singularity_uses_scaled_jacobian_for_large_coefficients():
    x = polyvar("x")
    huge_system = PolynomialSystem([(10**400) * x])

    assert not check_singularity(
        huge_system,
        np.array([1.0 + 0j]),
        [x],
        threshold=1e3,
    )


def test_check_singularity_is_invariant_to_equation_row_scaling():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([(10**100) * x, y])

    assert not check_singularity(
        system,
        np.array([1.0 + 0j, 1.0 + 0j]),
        [x, y],
        threshold=1e3,
    )


def test_check_singularity_treats_nonisolated_jacobians_as_singular():
    x, y = polyvar("x", "y")

    assert check_singularity(
        PolynomialSystem([x]),
        np.array([0.0 + 0j, 1.0 + 0j]),
        [x, y],
        threshold=1e12,
    )
    assert check_singularity(
        PolynomialSystem([x, 2 * x]),
        np.array([0.0 + 0j, 1.0 + 0j]),
        [x, y],
        threshold=1e12,
    )


def test_check_singularity_treats_unscalable_nonfinite_jacobian_as_singular():
    x = polyvar("x")
    system = PolynomialSystem([x**2])

    assert check_singularity(
        system,
        np.array([np.nan + 0j]),
        [x],
        threshold=1e3,
    )


def test_check_singularity_treats_nonfinite_condition_estimate_as_singular(monkeypatch):
    x = polyvar("x")
    system = PolynomialSystem([x])

    monkeypatch.setattr(np.linalg, "cond", lambda matrix: float("nan"))

    assert check_singularity(
        system,
        np.array([1.0 + 0j]),
        [x],
        threshold=1e3,
    )


def test_check_singularity_treats_condition_exceptions_as_singular(monkeypatch):
    x = polyvar("x")
    system = PolynomialSystem([x])

    def failing_condition(matrix):
        raise ValueError("condition estimate failed")

    monkeypatch.setattr(np.linalg, "cond", failing_condition)

    assert check_singularity(
        system,
        np.array([1.0 + 0j]),
        [x],
        threshold=1e3,
    )


def test_solver_rank_helpers_fail_closed_when_linalg_fails(monkeypatch):
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    def failing_svd(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    def failing_matrix_rank(*args, **kwargs):
        raise np.linalg.LinAlgError("rank failed")

    monkeypatch.setattr(np.linalg, "svd", failing_svd)
    assert solver_module._matrix_rank(np.eye(1), tolerance=1e-12) == 0
    assert solver_module._is_singular(
        system,
        {x: 1.0 + 0j},
        (x,),
        threshold=1e12,
        rank_tolerance=1e-12,
    )

    monkeypatch.setattr(np.linalg, "matrix_rank", failing_matrix_rank)
    assert solver_module._matrix_rank(np.eye(1)) == 0
    assert solver_module._relative_matrix_rank(np.eye(1)) == 0


def test_rank_diagnostics_fail_closed_on_non_linalg_errors(monkeypatch):
    x = polyvar("x")
    system = PolynomialSystem([x - 1])
    solution_set = SolutionSet([Solution({x: 1.0 + 0j}, residual=0.0)], system)

    def failing_svd(*args, **kwargs):
        raise FloatingPointError("SVD overflow")

    def failing_matrix_rank(*args, **kwargs):
        raise OverflowError("rank overflow")

    monkeypatch.setattr(np.linalg, "svd", failing_svd)
    audit = solution_set.diagnostics(tolerance=1e-8)
    diagnostics = audit.diagnostics[0]

    assert diagnostics.jacobian_rank == 0
    assert np.isinf(diagnostics.condition_number)
    assert diagnostics.is_rank_deficient
    assert monodromy_module._monodromy_jacobian_rank(
        system,
        np.array([1.0 + 0j]),
        [x],
    ) == 0
    assert solver_module._matrix_rank(np.eye(1), tolerance=1e-12) == 0

    monkeypatch.setattr(np.linalg, "matrix_rank", failing_matrix_rank)
    assert solver_module._matrix_rank(np.eye(1)) == 0
    assert solver_module._relative_matrix_rank(np.eye(1)) == 0


def test_solver_singularity_treats_condition_exceptions_as_singular(monkeypatch):
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    def failing_condition(matrix):
        raise OverflowError("condition overflow")

    monkeypatch.setattr(np.linalg, "cond", failing_condition)

    assert solver_module._is_singular(
        system,
        {x: 1.0 + 0j},
        (x,),
        threshold=1e12,
    )


def test_tracking_does_not_use_endgame_for_scaled_regular_target():
    x = polyvar("x")
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([(10**400) * (x - 2)])

    point, info = tracking_module.track_single_path(
        start_system,
        target_system,
        start_solution=np.array([1.0 + 0j]),
        variables=[x],
        gamma=1.0 + 0j,
        min_step_size=0.01,
        max_step_size=0.05,
        endgame_start=0.5,
        singularity_threshold=1e3,
        final_singularity_threshold=1e3,
        use_endgame=True,
    )

    assert info["success"]
    assert not info["singular"]
    assert not info.get("endgame_used", False)
    assert info["failure_reason"] is None
    np.testing.assert_allclose(point, np.array([2.0 + 0j]))
    assert info["final_residual"] == 0.0


@pytest.mark.parametrize("scale", [1e12, 1e-12])
def test_track_single_path_is_invariant_to_finite_target_row_scaling(scale):
    x = polyvar("x")
    start_system = PolynomialSystem([x - 1])
    target_system = PolynomialSystem([scale * (x - 2)])

    point, info = tracking_module.track_single_path(
        start_system,
        target_system,
        start_solution=np.array([1.0 + 0j]),
        variables=[x],
        gamma=1.0 + 0j,
        min_step_size=0.01,
        max_step_size=0.05,
        use_endgame=False,
        tol=1e-8,
    )

    assert info["success"]
    assert info["failure_reason"] is None
    np.testing.assert_allclose(point, np.array([2.0 + 0j]), atol=1e-8)
    assert info["final_residual"] < 1e-8


def test_compute_witness_superset_random_state_controls_slice_and_solver():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x - y])

    slice_1, witnesses_1 = compute_witness_superset(
        system, [x, y], dimension=1, random_state=99
    )
    slice_2, witnesses_2 = compute_witness_superset(
        system, [x, y], dimension=1, random_state=99
    )
    slice_3, witnesses_3 = compute_witness_superset(
        system, [x, y], dimension=1, random_state=100
    )

    assert repr(slice_1) == repr(slice_2)
    assert repr(slice_1) != repr(slice_3)
    assert (
        witnesses_1._meta["tracking_options"]["gamma"]
        == witnesses_2._meta["tracking_options"]["gamma"]
    )
    assert (
        witnesses_1._meta["tracking_options"]["gamma"]
        != witnesses_3._meta["tracking_options"]["gamma"]
    )


def test_compute_witness_superset_handles_redundant_overdetermined_system():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x - y, 2 * x - 2 * y])

    slicing, witnesses = compute_witness_superset(
        system, [x, y], dimension=1, random_state=123
    )

    assert len(slicing.equations) == 1
    assert len(witnesses) == 1
    assert witnesses[0].residual < 1e-8
    assert witnesses._meta["linear_solve"]["status"] == "unique_solution"
    assert witnesses._meta["total_paths"] == 0
    assert witnesses._meta["witness_set"]["augmented_equations"] == 3


def test_compute_witness_superset_rejects_still_underdetermined_slice():
    x, y, z = polyvar("x", "y", "z")
    system = PolynomialSystem([x])

    with pytest.raises(ValueError, match="underdetermined augmented system"):
        compute_witness_superset(system, [x, y, z], dimension=1, random_state=0)


def test_compute_witness_superset_validates_dimension_and_variables():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y])

    with pytest.raises(TypeError, match="dimension must be an integer"):
        compute_witness_superset(system, [x, y], dimension=True, random_state=0)
    with pytest.raises(ValueError, match="dimension cannot exceed"):
        compute_witness_superset(
            PolynomialSystem([x]),
            [x],
            dimension=2,
            random_state=0,
        )
    with pytest.raises(ValueError, match="missing system variable"):
        compute_witness_superset(system, [x], dimension=1, random_state=0)
    with pytest.raises(ValueError, match="duplicates"):
        compute_witness_superset(system, [x, x], dimension=1, random_state=0)


@pytest.mark.parametrize("solver_options", ["bad", ["verbose"]])
def test_compute_witness_superset_rejects_malformed_solver_options(
    solver_options,
):
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y])

    with pytest.raises(TypeError, match="solver_options must be a dictionary"):
        compute_witness_superset(
            system,
            [x, y],
            dimension=1,
            solver_options=solver_options,
            random_state=0,
        )


@pytest.mark.parametrize("option", ["system", "variables"])
def test_compute_witness_superset_rejects_conflicting_solver_options(option):
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y])

    with pytest.raises(ValueError, match=f"cannot override.*{option}"):
        compute_witness_superset(
            system,
            [x, y],
            dimension=1,
            solver_options={option: [x, y]},
            random_state=0,
        )


def test_compute_witness_superset_validates_verbose_solver_option():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y])

    with pytest.raises(TypeError, match=r"solver_options\['verbose'\]"):
        compute_witness_superset(
            system,
            [x, y],
            dimension=1,
            solver_options={"verbose": "yes"},
            random_state=0,
        )


def test_witness_set_constructor_validates_inputs():
    x = polyvar("x")

    with pytest.raises(TypeError, match="original_system must be"):
        WitnessSet([], PolynomialSystem([]), [], dimension=0)
    with pytest.raises(TypeError, match="dimension must be an integer"):
        WitnessSet(
            PolynomialSystem([]),
            PolynomialSystem([]),
            [],
            dimension=True,
        )
    with pytest.raises(ValueError, match="dimension must be non-negative"):
        WitnessSet(
            PolynomialSystem([]),
            PolynomialSystem([]),
            [],
            dimension=-1,
        )
    with pytest.raises(ValueError, match="dimension cannot exceed"):
        WitnessSet(
            PolynomialSystem([]),
            PolynomialSystem([x - 1]),
            [],
            dimension=2,
        )
    with pytest.raises(ValueError, match="slicing_system must contain 0 equation"):
        WitnessSet(
            PolynomialSystem([]),
            PolynomialSystem([x - 1]),
            [],
            dimension=0,
        )
    with pytest.raises(TypeError, match=r"witness_points\[0\]"):
        WitnessSet(
            PolynomialSystem([]),
            PolynomialSystem([x - 1]),
            [object()],
            dimension=1,
        )


def test_witness_set_constructor_accepts_coordinate_records():
    x, y = polyvar("x", "y")

    class SolutionLike:
        def __init__(self, values):
            self.values = values

    witness_set = WitnessSet(
        PolynomialSystem([x - y]),
        PolynomialSystem([x - 1]),
        [
            [1.0 + 0j, 1.0 + 0j],
            {"x": 1.0 + 0j, "y": 1.0 + 0j},
            SolutionLike({x: 1.0 + 0j, "y": 1.0 + 0j}),
        ],
        dimension=1,
    )

    assert witness_set.degree == 3
    assert all(isinstance(point, Solution) for point in witness_set.witness_points)
    assert all(point.values[x] == 1.0 + 0j for point in witness_set.witness_points)
    assert all(point.values[y] == 1.0 + 0j for point in witness_set.witness_points)

    with pytest.raises(ValueError, match="conflicting coordinates.*x"):
        WitnessSet(
            PolynomialSystem([x - y]),
            PolynomialSystem([x - 1]),
            [{x: 1.0 + 0j, "x": 2.0 + 0j, y: 1.0 + 0j}],
            dimension=1,
        )


def test_witness_set_constructor_validates_witness_residuals():
    x, y = polyvar("x", "y")

    with pytest.raises(ValueError, match="original_system"):
        WitnessSet(
            PolynomialSystem([x - y]),
            PolynomialSystem([x - 1]),
            [Solution({x: 1.0 + 0j, y: 2.0 + 0j}, residual=0.0)],
            dimension=1,
        )
    with pytest.raises(ValueError, match="slicing_system"):
        WitnessSet(
            PolynomialSystem([x - y]),
            PolynomialSystem([x - 1]),
            [Solution({x: 2.0 + 0j, y: 2.0 + 0j}, residual=0.0)],
            dimension=1,
        )

    approximate = WitnessSet(
        PolynomialSystem([x - 1]),
        PolynomialSystem([]),
        [Solution({x: 1.0 + 1e-7}, residual=1e-7)],
        dimension=0,
        validation_tolerance=1e-6,
    )
    assert approximate.validation_tolerance == 1e-6

    with pytest.raises(TypeError, match="validation_tolerance must be a number"):
        WitnessSet(
            PolynomialSystem([]),
            PolynomialSystem([]),
            [],
            dimension=0,
            validation_tolerance="1e-8",
        )


@pytest.mark.parametrize("extra_value", [99.0 + 0j, float("inf")])
def test_witness_set_constructor_rejects_extra_witness_coordinates(extra_value):
    x, y = polyvar("x", "y")

    with pytest.raises(ValueError, match="outside the witness ambient.*y"):
        WitnessSet(
            PolynomialSystem([x - 1]),
            PolynomialSystem([]),
            [Solution({x: 1.0 + 0j, y: extra_value}, residual=0.0)],
            dimension=0,
        )

    with pytest.raises(ValueError, match="outside the witness ambient.*z"):
        WitnessSet(
            PolynomialSystem([x - 1]),
            PolynomialSystem([]),
            [{"x": 1.0 + 0j, "z": extra_value}],
            dimension=0,
        )


def test_witness_set_infers_variables_from_slicing_system_for_sampling():
    x = polyvar("x")
    witness_set = WitnessSet(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        [Solution({x: 1.0 + 0j}, residual=0.0)],
        dimension=1,
    )

    sample = witness_set.sample_point(
        target_slice=PolynomialSystem([x - 2]),
        random_state=0,
        options={"tol": 1e-10},
    )

    assert [var.name for var in witness_set.variables] == ["x"]
    assert sample is not None
    assert np.allclose(sample, np.array([2.0 + 0j]), atol=1e-8)


def test_witness_set_sample_points_tracks_all_witness_paths():
    x, y = polyvar("x", "y")
    witness_set = WitnessSet(
        PolynomialSystem([x**2 + y**2 - 1]),
        PolynomialSystem([x]),
        [
            Solution({x: 0.0 + 0.0j, y: 1.0 + 0.0j}, residual=0.0),
            Solution({x: 0.0 + 0.0j, y: -1.0 + 0.0j}, residual=0.0),
        ],
        dimension=1,
    )

    points = witness_set.sample_points(
        target_slice=PolynomialSystem([y]),
        variables=[x, y],
        random_state=0,
        options={"tol": 1e-10},
    )
    rounded = sorted(
        (round(point[0].real), round(point[1].real))
        for point in points
    )

    assert rounded == [(-1, 0), (1, 0)]
    for point in points:
        values = {x: point[0], y: point[1]}
        assert abs(witness_set.original_system.evaluate(values)[0]) < 1e-8
        assert abs(point[1]) < 1e-8

    points_with_info, info_records = witness_set.sample_points(
        target_slice=PolynomialSystem([y]),
        variables=[x, y],
        random_state=0,
        options={"tol": 1e-10},
        return_info=True,
    )

    assert len(points_with_info) == 2
    assert len(info_records) == 2
    assert [record["witness_index"] for record in info_records] == [0, 1]
    assert all(record["success"] for record in info_records)
    assert all(record["validated"] for record in info_records)
    assert all(record["tracking_success"] for record in info_records)
    assert all(record["failure_reason"] is None for record in info_records)
    assert all(record["sample_tolerance"] == pytest.approx(1e-8) for record in info_records)
    assert all(isinstance(record["tracking_info"], dict) for record in info_records)


def test_witness_set_sampling_rejects_invalid_successful_tracker_output(
    monkeypatch,
):
    import pycontinuum.parameter_homotopy as parameter_homotopy

    x = polyvar("x")
    witness_set = WitnessSet(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        [Solution({x: 1.0 + 0j}, residual=0.0)],
        dimension=1,
    )

    def fake_track_parameter_path(
        parameter_homotopy,
        start_point,
        start_t=0.0,
        end_t=1.0,
        options=None,
    ):
        return start_point, {"success": True}

    monkeypatch.setattr(
        parameter_homotopy,
        "track_parameter_path",
        fake_track_parameter_path,
    )

    target_slice = PolynomialSystem([x - 2])

    assert witness_set.sample_point(
        target_slice=target_slice,
        random_state=0,
        options={"tol": 1e-10},
    ) is None
    sample, info = witness_set.sample_point(
        target_slice=target_slice,
        random_state=0,
        options={"tol": 1e-10},
        return_info=True,
    )
    assert witness_set.sample_points(
        target_slice=target_slice,
        random_state=0,
        options={"tol": 1e-10},
    ) == []
    points, records = witness_set.sample_points(
        target_slice=target_slice,
        random_state=0,
        options={"tol": 1e-10},
        return_info=True,
    )

    assert sample is None
    assert info["success"] is False
    assert info["tracking_success"] is True
    assert info["failure_reason"] == "invalid_sample"
    assert points == []
    assert records[0]["success"] is False
    assert records[0]["failure_reason"] == "invalid_sample"


def test_witness_set_sample_points_validates_options_and_coordinates():
    x, y = polyvar("x", "y")
    witness_set = WitnessSet(
        PolynomialSystem([]),
        PolynomialSystem([x - y]),
        [Solution({x: 1.0 + 0.0j, y: 1.0 + 0.0j}, residual=0.0)],
        dimension=1,
    )

    with pytest.raises(TypeError, match="options must be a dictionary"):
        witness_set.sample_points(
            target_slice=PolynomialSystem([x - 1]),
            variables=[x, y],
            options="bad",
        )
    with pytest.raises(TypeError, match="return_info must be a boolean"):
        witness_set.sample_point(
            target_slice=PolynomialSystem([x - 1]),
            variables=[x, y],
            return_info="yes",
        )
    with pytest.raises(TypeError, match="return_info must be a boolean"):
        witness_set.sample_points(
            target_slice=PolynomialSystem([x - 1]),
            variables=[x, y],
            return_info=1,
        )
    with pytest.raises(ValueError, match=r"witness_points\[0\].*y"):
        WitnessSet(
            PolynomialSystem([]),
            PolynomialSystem([x - y]),
            [Solution({x: 1.0 + 0.0j}, residual=0.0)],
            dimension=1,
        )
    with pytest.raises(ValueError, match="target_slice must contain 1 equation"):
        witness_set.sample_points(
            target_slice=PolynomialSystem([]),
            variables=[x, y],
            options={"tol": 1e-10},
        )


def test_witness_set_sample_point_is_quiet_on_tracking_failure(
    monkeypatch,
    capsys,
):
    import pycontinuum.parameter_homotopy as parameter_homotopy

    x = polyvar("x")
    witness_set = WitnessSet(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        [Solution({x: 1.0 + 0j}, residual=0.0)],
        dimension=1,
    )

    def fake_track_parameter_path(
        parameter_homotopy,
        start_point,
        start_t=0.0,
        end_t=1.0,
        options=None,
    ):
        return start_point, {"success": False}

    monkeypatch.setattr(
        parameter_homotopy,
        "track_parameter_path",
        fake_track_parameter_path,
    )

    sample = witness_set.sample_point(
        target_slice=PolynomialSystem([x - 2]),
        variables=[x],
        random_state=0,
    )

    captured = capsys.readouterr()
    assert sample is None
    assert captured.out == ""
    assert captured.err == ""


def test_witness_set_empty_sampling_info():
    x = polyvar("x")
    witness_set = WitnessSet(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        [],
        dimension=1,
    )

    sample, info = witness_set.sample_point(return_info=True, random_state=0)
    points, records = witness_set.sample_points(return_info=True, random_state=0)

    assert sample is None
    assert info["success"] is False
    assert info["failure_reason"] == "empty_witness_set"
    assert info["witness_index"] is None
    assert points == []
    assert records == []


def test_witness_set_sample_point_rejects_malformed_rng_index():
    class VectorIndexRng:
        def uniform(self, *args, **kwargs):
            return 0.0

        def integers(self, *args, **kwargs):
            return np.array([0, 1])

    class OutOfRangeUniformRng:
        def uniform(self, *args, **kwargs):
            return 1.0

    x = polyvar("x")
    witness_set = WitnessSet(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        [Solution({x: 1.0 + 0j}, residual=0.0)],
        dimension=1,
    )

    with pytest.raises(ValueError, match="scalar witness point index"):
        witness_set.sample_point(
            target_slice=PolynomialSystem([x - 2]),
            variables=[x],
            random_state=VectorIndexRng(),
        )
    with pytest.raises(ValueError, match=r"witness point index in \[0, 1\)"):
        witness_set.sample_point(
            target_slice=PolynomialSystem([x - 2]),
            variables=[x],
            random_state=OutOfRangeUniformRng(),
        )


def test_witness_set_membership_defaults_to_inferred_variables():
    x = polyvar("x")
    witness_set = WitnessSet(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        [Solution({x: 1.0 + 0j}, residual=0.0)],
        dimension=1,
    )

    assert witness_set.is_point_on_component(
        np.array([2.0 + 0j]), tolerance=1e-8, random_state=0
    )


def test_witness_set_membership_accepts_solution_and_mapping_points():
    x, y = polyvar("x", "y")
    line_witness = WitnessSet(
        PolynomialSystem([x - y]),
        PolynomialSystem([x - 1]),
        [Solution({x: 1.0 + 0j, y: 1.0 + 0j}, residual=0.0)],
        dimension=1,
    )

    assert line_witness.is_point_on_component(
        Solution({x: 2.0 + 0j, y: 2.0 + 0j}, residual=0.0),
        variables=[x, y],
        tolerance=1e-6,
        random_state=0,
    )
    assert line_witness.is_point_on_component(
        {"y": 3.0 + 0j, x: 3.0 + 0j},
        variables=[x, y],
        tolerance=1e-6,
        random_state=0,
    )
    assert not line_witness.is_point_on_component(
        {x: 2.0 + 0j, "y": 3.0 + 0j},
        variables=[x, y],
        tolerance=1e-6,
        random_state=0,
    )


def test_witness_set_membership_rejects_mapping_point_missing_coordinate():
    x, y = polyvar("x", "y")
    witness_set = WitnessSet(
        PolynomialSystem([x - y]),
        PolynomialSystem([x - 1]),
        [Solution({x: 1.0 + 0j, y: 1.0 + 0j}, residual=0.0)],
        dimension=1,
    )

    with pytest.raises(ValueError, match=r"point is missing coordinate\(s\): y"):
        witness_set.is_point_on_component(
            {x: 2.0 + 0j},
            variables=[x, y],
            random_state=0,
        )


def test_witness_set_membership_rejects_malformed_slice_rng_output():
    class WrongShapeRng:
        def uniform(self, *args, **kwargs):
            return 0.0

        def standard_normal(self, size=None):
            return np.array([0.0])

    x, y = polyvar("x", "y")
    witness_set = WitnessSet(
        PolynomialSystem([]),
        PolynomialSystem([x - y]),
        [Solution({x: 1.0 + 0j, y: 1.0 + 0j}, residual=0.0)],
        dimension=1,
    )

    with pytest.raises(
        ValueError,
        match=r"standard_normal.*shape \(2,\).*point slice 0 coefficients real",
    ):
        witness_set.is_point_on_component(
            np.array([1.0 + 0j, 1.0 + 0j]),
            variables=[x, y],
            random_state=WrongShapeRng(),
        )


def test_witness_set_membership_uses_scaled_residual_for_extreme_system():
    x, y = polyvar("x", "y")
    huge = 10**400
    witness_set = WitnessSet(
        PolynomialSystem([huge * (x - y)]),
        PolynomialSystem([x - 1]),
        [Solution({x: 1.0 + 0j, y: 1.0 + 0j}, residual=0.0)],
        dimension=1,
    )

    assert witness_set.is_point_on_component(
        np.array([2.0 + 0j, 2.0 + 0j]),
        variables=[x, y],
        tolerance=1e-6,
        random_state=0,
        options={"tol": 1e-8, "max_steps": 2000},
    )
    assert not witness_set.is_point_on_component(
        np.array([2.0 + 0j, 3.0 + 0j]),
        variables=[x, y],
        tolerance=1e-6,
        random_state=0,
        options={"tol": 1e-8, "max_steps": 2000},
    )


def test_witness_set_membership_tracks_all_degree_two_witness_paths():
    x, y = polyvar("x", "y")
    witness_set = WitnessSet(
        PolynomialSystem([x**2 + y**2 - 1]),
        PolynomialSystem([x]),
        [
            Solution({x: 0.0 + 0.0j, y: 1.0 + 0.0j}, residual=0.0),
            Solution({x: 0.0 + 0.0j, y: -1.0 + 0.0j}, residual=0.0),
        ],
        dimension=1,
    )

    point = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    for seed in range(12):
        assert witness_set.is_point_on_component(
            point,
            variables=[x, y],
            tolerance=1e-6,
            random_state=seed,
        )
    assert not witness_set.is_point_on_component(
        np.array([2.0 + 0.0j, 0.0 + 0.0j]),
        variables=[x, y],
        tolerance=1e-6,
        random_state=0,
    )


def test_witness_set_membership_rejects_missing_witness_coordinate():
    x, y = polyvar("x", "y")
    with pytest.raises(ValueError, match=r"witness_points\[0\].*y"):
        WitnessSet(
            PolynomialSystem([]),
            PolynomialSystem([x - y]),
            [Solution({x: 1.0 + 0.0j}, residual=0.0)],
            dimension=1,
        )


def test_witness_set_membership_rejects_invalid_tolerance():
    x = polyvar("x")
    witness_set = WitnessSet(
        PolynomialSystem([]),
        PolynomialSystem([x - 1]),
        [Solution({x: 1.0 + 0j}, residual=0.0)],
        dimension=1,
    )

    with pytest.raises(TypeError, match="tolerance must be a number"):
        witness_set.is_point_on_component(
            np.array([1.0 + 0j]),
            tolerance="1e-8",
            random_state=0,
        )
    with pytest.raises(ValueError, match="tolerance must be positive and finite"):
        witness_set.is_point_on_component(
            np.array([1.0 + 0j]),
            tolerance=0.0,
            random_state=0,
        )


def test_total_degree_start_system_rejects_constant_equations():
    x = polyvar("x")
    system = PolynomialSystem([1])

    with pytest.raises(ValueError, match="positive-degree"):
        generate_total_degree_start_system(system, [x], random_state=0)


def test_solve_rejects_variable_list_missing_system_variable():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1])

    with pytest.raises(ValueError, match="missing system variable"):
        solve(system, variables=[x], random_state=0)


def test_solve_rejects_variable_list_with_unused_variable():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x - 1])

    with pytest.raises(ValueError, match="not used by system.*y"):
        solve(system, variables=[x, y], random_state=0)


def test_solve_reports_free_variable_as_positive_dimensional_when_allowed():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x**2 - 1])

    with pytest.raises(ValueError, match="positive-dimensional.*witness-set"):
        solve(
            system,
            variables=[x, y],
            allow_underdetermined=True,
            random_state=0,
        )


def test_solve_rejects_underdetermined_nonlinear_tracking_when_allowed():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x * y - 1])

    with pytest.raises(ValueError, match="positive-dimensional.*witness-set"):
        solve(
            system,
            variables=[x, y],
            allow_underdetermined=True,
            use_endgame=False,
            random_state=0,
        )


def test_custom_start_cannot_bypass_underdetermined_solve_rejection():
    x, y = polyvar("x", "y")
    target_system = PolynomialSystem([x * y - 1])
    start_system = PolynomialSystem([x - 1])

    with pytest.raises(ValueError, match="positive-dimensional.*witness-set"):
        solve(
            target_system,
            start_system=start_system,
            start_solutions=[[1.0 + 0j, 1.0 + 0j]],
            variables=[x, y],
            allow_underdetermined=True,
            use_endgame=False,
            random_state=0,
        )


def test_solve_rejects_non_polynomial_system_target():
    x = polyvar("x")

    with pytest.raises(TypeError, match="system must be a PolynomialSystem"):
        solve([x - 1], random_state=0)


def test_solve_rejects_noniterable_variable_argument():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    with pytest.raises(TypeError, match="variables must be an iterable"):
        solve(system, variables=x, random_state=0)


def test_solve_rejects_nonvariable_entries():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    with pytest.raises(TypeError, match=r"variables\[1\] must be a Variable"):
        solve(system, variables=[x, "extra"], random_state=0)


def test_solve_rejects_duplicate_variables():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    with pytest.raises(ValueError, match="duplicate variable"):
        solve(system, variables=[x, x], random_state=0)


def test_total_degree_start_system_rejects_non_polynomial_target():
    x = polyvar("x")

    with pytest.raises(TypeError, match="target_system must be a PolynomialSystem"):
        generate_total_degree_start_system([x - 1], [x], random_state=0)


def test_total_degree_start_system_rejects_nonvariable_entries():
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    with pytest.raises(TypeError, match=r"variables\[1\] must be a Variable"):
        generate_total_degree_start_system(system, [x, "extra"], random_state=0)


def test_total_degree_start_system_rejects_unused_variables_by_default():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x - 1])

    with pytest.raises(ValueError, match="not used by system.*y"):
        generate_total_degree_start_system(system, [x, y], random_state=0)


def test_total_degree_start_system_rejects_incomplete_variable_list():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1])

    with pytest.raises(ValueError, match="missing system variable"):
        generate_total_degree_start_system(system, [x], random_state=0)


def test_total_degree_start_system_rejects_overdetermined_systems():
    x = polyvar("x")
    system = PolynomialSystem([x - 1, x + 1])

    with pytest.raises(ValueError, match="overdetermined systems"):
        generate_total_degree_start_system(
            system,
            [x],
            allow_underdetermined=True,
            random_state=0,
        )


def test_evaluation_helpers_reject_malformed_points_and_variables():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1])

    with pytest.raises(TypeError, match="system must be a PolynomialSystem"):
        evaluate_system_at_point([x + y - 1], [1.0 + 0j, 0.0 + 0j], [x, y])
    with pytest.raises(ValueError, match="point must have 2 coordinate"):
        evaluate_system_at_point(system, [1.0 + 0j], [x, y])
    residuals = evaluate_system_at_point(system, [np.nan + 0j, 0.0 + 0j], [x, y])
    assert not np.all(np.isfinite(residuals))
    with pytest.raises(TypeError, match=r"variables\[1\] must be a Variable"):
        evaluate_system_at_point(system, [1.0 + 0j, 0.0 + 0j], [x, "y"])
    with pytest.raises(ValueError, match="duplicate variable"):
        evaluate_system_at_point(PolynomialSystem([x - 1]), [1.0 + 0j, 2.0 + 0j], [x, x])


def test_evaluation_helpers_accept_solution_and_mapping_points():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 3, x * y - 2])
    solution = Solution({x: 1.0 + 0j, y: 2.0 + 0j}, residual=0.0)
    mapping = {"y": 2.0 + 0j, x: 1.0 + 0j, "x": 1.0 + 0j}

    np.testing.assert_allclose(
        evaluate_system_at_point(system, solution, [x, y]),
        np.array([0.0 + 0j, 0.0 + 0j]),
    )
    np.testing.assert_allclose(
        evaluate_scaled_system_at_point(system, mapping, [x, y]),
        np.array([0.0 + 0j, 0.0 + 0j]),
    )
    np.testing.assert_allclose(
        evaluate_backward_error_at_point(system, mapping, [x, y]),
        np.array([0.0, 0.0]),
    )
    expected_jacobian = np.array([[1.0, 1.0], [2.0, 1.0]], dtype=complex)
    np.testing.assert_allclose(
        evaluate_jacobian_at_point(system, solution, [x, y]),
        expected_jacobian,
    )
    np.testing.assert_allclose(
        evaluate_scaled_jacobian_at_point(system, mapping, [x, y]),
        expected_jacobian,
    )
    np.testing.assert_allclose(
        evaluate_equation_scaled_jacobian_at_point(system, mapping, [x, y]),
        np.array([[1.0 / 3.0, 1.0 / 3.0], [1.0, 0.5]], dtype=complex),
    )
    with pytest.raises(ValueError, match="point coordinate\\(s\\) must be numeric"):
        evaluate_system_at_point(
            PolynomialSystem([0]),
            {x: 10**400, "x": 10**400, y: 0.0 + 0j},
            [x, y],
        )


def test_evaluation_helpers_reject_mapping_points_missing_coordinates():
    x, y = polyvar("x", "y")
    system = PolynomialSystem([x + y - 1])

    with pytest.raises(ValueError, match=r"point is missing coordinate\(s\): y"):
        evaluate_system_at_point(system, {x: 1.0 + 0j}, [x, y])
    with pytest.raises(ValueError, match="conflicting coordinates.*x"):
        evaluate_system_at_point(
            system,
            {x: 1.0 + 0j, "x": 2.0 + 0j, y: 0.0 + 0j},
            [x, y],
        )


def test_evaluation_helpers_return_nonfinite_entries_for_overflow():
    x = polyvar("x")
    system = PolynomialSystem([(10**400) * x])

    residuals = evaluate_system_at_point(system, [1.0 + 0j], [x])
    jacobian = evaluate_jacobian_at_point(system, [1.0 + 0j], [x])
    jacobian_from_polys = evaluate_jacobian_polynomials(
        system.jacobian([x]),
        {x: 1.0 + 0j},
    )

    assert residuals.shape == (1,)
    assert jacobian.shape == (1, 1)
    assert jacobian_from_polys.shape == (1, 1)
    assert not np.all(np.isfinite(residuals))
    assert not np.all(np.isfinite(jacobian))
    assert not np.all(np.isfinite(jacobian_from_polys))


def test_evaluation_helpers_return_zero_for_huge_exact_cancellation():
    x = polyvar("x")
    huge = 10**400
    system = PolynomialSystem([huge * x - huge])

    exact_residuals = evaluate_system_at_point(system, [1.0 + 0j], [x])
    nonroot_residuals = evaluate_system_at_point(system, [2.0 + 0j], [x])

    assert exact_residuals.shape == (1,)
    assert exact_residuals[0] == 0
    assert not np.all(np.isfinite(nonroot_residuals))


def test_evaluation_helpers_do_not_report_zero_for_hidden_nonzero_terms():
    x, y = polyvar("x", "y")
    huge = 10**400
    constant_system = PolynomialSystem([huge * x + 1 - huge])
    linear_system = PolynomialSystem([huge * x + y - huge])

    constant_residuals = evaluate_system_at_point(
        constant_system,
        [1.0 + 0j],
        [x],
    )
    linear_residuals = evaluate_system_at_point(
        linear_system,
        [1.0 + 0j, 2.0 + 0j],
        [x, y],
    )
    exact_zero_residuals = evaluate_system_at_point(
        linear_system,
        [1.0 + 0j, 0.0 + 0j],
        [x, y],
    )

    assert not np.all(np.isfinite(constant_residuals))
    assert not np.all(np.isfinite(linear_residuals))
    assert exact_zero_residuals[0] == 0


def test_scaled_evaluation_preserves_underflowed_coefficient_times_large_coordinate():
    x, y = polyvar("x", "y")
    huge = 10**400
    system = PolynomialSystem([huge * x + y**4 - huge])

    point = [1.0 + 0j, 1e100 + 0j]
    raw_residuals = evaluate_system_at_point(system, point, [x, y])
    scaled_residuals = evaluate_scaled_system_at_point(system, point, [x, y])
    backward_errors = evaluate_backward_error_at_point(system, point, [x, y])

    assert not np.all(np.isfinite(raw_residuals))
    assert scaled_residuals.shape == (1,)
    assert backward_errors.shape == (1,)
    assert np.isclose(scaled_residuals[0], 1.0 + 0j, rtol=1e-12, atol=1e-12)
    assert np.isclose(backward_errors[0], 1.0 / 3.0, rtol=1e-12, atol=1e-12)


def test_solver_and_diagnostics_report_zero_raw_residual_for_huge_exact_cancellation():
    x = polyvar("x")
    huge = 10**400
    system = PolynomialSystem([huge * x - huge])

    solutions = solve(
        system,
        variables=[x],
        scale_equations=False,
        random_state=0,
    )
    audit = solutions.diagnostics()

    assert len(solutions) == 1
    assert solutions[0].values[x] == 1
    assert solutions[0].residual == 0.0
    assert solutions[0].scaled_residual == 0.0
    assert solutions._meta["linear_solve"]["residual_norm"] == 0.0
    assert audit.max_residual == 0.0
    assert audit.max_scaled_residual == 0.0
    assert audit.all_valid


def test_newton_corrector_does_not_accept_stalled_nonroot():
    x = polyvar("x")
    system = PolynomialSystem([x**2 - 1])

    point, success, iters = newton_corrector(
        system,
        np.array([0.0 + 0j]),
        [x],
        max_iters=3,
        tol=1e-12,
    )

    assert not success
    assert iters == 1
    assert abs(point[0]) < 1e-15


def test_newton_corrector_damps_residual_increasing_step():
    x = polyvar("x")
    system = PolynomialSystem([x**3 - 1])
    start = np.array([0.1 + 0j])
    initial_residual = abs(system.evaluate({x: start[0]})[0])

    point, success, iters = newton_corrector(
        system,
        start,
        [x],
        max_iters=1,
        tol=1e-12,
    )
    final_residual = abs(system.evaluate({x: point[0]})[0])

    assert not success
    assert iters == 1
    assert final_residual < initial_residual
    assert abs(point[0]) < 2.0


def test_newton_corrector_uses_scaled_fallback_for_overflowing_rows():
    x = polyvar("x")
    system = PolynomialSystem([(10**400) * (x - 1)])

    point, success, iters = newton_corrector(
        system,
        np.array([0.9 + 0j]),
        [x],
        max_iters=5,
        tol=1e-12,
    )

    assert success
    assert iters == 1
    assert abs(point[0] - 1.0) < 1e-12


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"max_iters": True}, TypeError, "max_iters must be an integer"),
        ({"max_iters": -1}, ValueError, "max_iters must be nonnegative"),
        ({"tol": "1e-8"}, TypeError, "tol must be a number"),
        ({"tol": 0.0}, ValueError, "tol must be positive and finite"),
    ],
)
def test_newton_corrector_rejects_invalid_options(kwargs, error_type, message):
    x = polyvar("x")
    system = PolynomialSystem([x - 1])

    with pytest.raises(error_type, match=message):
        newton_corrector(
            system,
            np.array([0.0 + 0j]),
            [x],
            **kwargs,
        )


def test_numeric_newton_corrector_does_not_accept_stalled_nonroot():
    def f(point):
        return np.array([point[0] ** 2 - 1], dtype=complex)

    def jac(point):
        return np.array([[2 * point[0]]], dtype=complex)

    point, success, iters = newton_corrector_numeric(
        f,
        jac,
        np.array([0.0 + 0j]),
        max_iters=3,
        tol=1e-12,
    )

    assert not success
    assert iters == 1
    assert abs(point[0]) < 1e-15


def test_numeric_newton_corrector_accepts_list_outputs():
    point, success, iters = newton_corrector_numeric(
        lambda point: [point[0] - 1],
        lambda point: [[1]],
        np.array([0.0 + 0j]),
        max_iters=3,
        tol=1e-12,
    )

    assert success
    assert iters == 1
    np.testing.assert_allclose(point, np.array([1.0 + 0j]))


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"max_iters": np.bool_(True)}, TypeError, "max_iters must be an integer"),
        ({"max_iters": -1}, ValueError, "max_iters must be nonnegative"),
        ({"tol": "1e-8"}, TypeError, "tol must be a number"),
        ({"tol": float("nan")}, ValueError, "tol must be positive and finite"),
    ],
)
def test_numeric_newton_corrector_rejects_invalid_options(
    kwargs, error_type, message
):
    with pytest.raises(error_type, match=message):
        newton_corrector_numeric(
            lambda point: np.array([point[0] - 1], dtype=complex),
            lambda point: np.array([[1]], dtype=complex),
            np.array([0.0 + 0j]),
            **kwargs,
        )


def test_numeric_newton_corrector_rejects_noncallables():
    with pytest.raises(TypeError, match="f must be callable"):
        newton_corrector_numeric(
            "not-callable",
            lambda point: np.array([[1]], dtype=complex),
            np.array([0.0 + 0j]),
        )
    with pytest.raises(TypeError, match="jac must be callable"):
        newton_corrector_numeric(
            lambda point: np.array([point[0] - 1], dtype=complex),
            "not-callable",
            np.array([0.0 + 0j]),
        )


def test_numeric_newton_corrector_rejects_nonnumeric_callable_outputs():
    with pytest.raises(TypeError, match=r"f\(point\) must be numeric"):
        newton_corrector_numeric(
            lambda point: ["not-numeric"],
            lambda point: np.array([[1]], dtype=complex),
            np.array([0.0 + 0j]),
        )
    with pytest.raises(TypeError, match=r"jac\(point\) must be numeric"):
        newton_corrector_numeric(
            lambda point: np.array([point[0] - 1], dtype=complex),
            lambda point: [["not-numeric"]],
            np.array([0.0 + 0j]),
        )


def test_numeric_newton_corrector_damps_residual_increasing_step():
    def f(point):
        return np.array([point[0] ** 3 - 1], dtype=complex)

    def jac(point):
        return np.array([[3 * point[0] ** 2]], dtype=complex)

    start = np.array([0.1 + 0j])
    initial_residual = abs(f(start)[0])
    point, success, iters = newton_corrector_numeric(
        f,
        jac,
        start,
        max_iters=1,
        tol=1e-12,
    )

    assert not success
    assert iters == 1
    assert abs(f(point)[0]) < initial_residual
    assert abs(point[0]) < 2.0


def test_numeric_newton_corrector_backtracks_over_invalid_trial_points():
    def f(point):
        if point[0].real > 0.75:
            raise ValueError("outside numeric model domain")
        return np.array([point[0] - 0.5], dtype=complex)

    def jac(point):
        return np.array([[0.25]], dtype=complex)

    point, success, iters = newton_corrector_numeric(
        f,
        jac,
        np.array([0.0 + 0j]),
        max_iters=4,
        tol=1e-12,
    )

    assert success
    assert iters == 1
    np.testing.assert_allclose(point, np.array([0.5 + 0j]))


def test_numeric_newton_corrector_fails_cleanly_when_all_trials_invalid():
    def f(point):
        if point[0].real > 0.0:
            raise ValueError("outside numeric model domain")
        return np.array([point[0] - 1], dtype=complex)

    def jac(point):
        return np.array([[1]], dtype=complex)

    point, success, iters = newton_corrector_numeric(
        f,
        jac,
        np.array([0.0 + 0j]),
        max_iters=3,
        tol=1e-12,
    )

    assert not success
    assert iters == 1
    np.testing.assert_allclose(point, np.array([0.0 + 0j]))


def test_scaled_euclidean_norm_falls_back_when_numpy_norm_fails(monkeypatch):
    def failing_norm(values):
        raise FloatingPointError("norm backend failed")

    monkeypatch.setattr(np.linalg, "norm", failing_norm)

    norm = _scaled_euclidean_norm(
        np.array([3e200 + 0j, 4e200 + 0j], dtype=complex)
    )

    assert norm == pytest.approx(5e200)


def test_solve_uses_scaled_norm_fallback_when_numpy_norm_fails(monkeypatch):
    x = polyvar("x")

    def failing_norm(values):
        raise FloatingPointError("norm backend failed")

    monkeypatch.setattr(np.linalg, "norm", failing_norm)

    solutions = solve(PolynomialSystem([x - 1]), variables=[x], random_state=0)

    assert len(solutions) == 1
    assert abs(solutions[0].values[x] - 1.0) < 1e-12
    assert solutions[0].scaled_residual == 0.0
    assert solutions._meta["linear_solve"]["status"] == "unique_solution"


def test_solve_linear_system_uses_balanced_fallback_after_raw_lapack_failure(monkeypatch):
    original_solve = np.linalg.solve
    original_lstsq = np.linalg.lstsq

    def fail_on_unbalanced_solve(matrix, rhs):
        if np.max(np.abs(matrix)) > 1e100:
            raise np.linalg.LinAlgError("raw solve failed")
        return original_solve(matrix, rhs)

    def fail_on_unbalanced_lstsq(matrix, rhs, rcond=None):
        if np.max(np.abs(matrix)) > 1e100:
            raise np.linalg.LinAlgError("raw lstsq failed")
        return original_lstsq(matrix, rhs, rcond=rcond)

    monkeypatch.setattr(np.linalg, "solve", fail_on_unbalanced_solve)
    monkeypatch.setattr(np.linalg, "lstsq", fail_on_unbalanced_lstsq)

    solution = solve_linear_system(
        np.array([[1e200 + 0j, 0.0 + 0j], [0.0 + 0j, 1e-200 + 0j]]),
        np.array([1e200 + 0j, 1e-200 + 0j]),
    )

    np.testing.assert_allclose(solution, np.array([1.0 + 0j, 1.0 + 0j]))


def test_solve_linear_system_selects_lowest_residual_finite_candidate(monkeypatch):
    monkeypatch.setattr(
        np.linalg,
        "solve",
        lambda matrix, rhs: np.zeros(matrix.shape[1], dtype=complex),
    )

    solution = solve_linear_system(
        np.eye(2, dtype=complex),
        np.array([1.0 + 0j, -2.0 + 0j]),
    )

    np.testing.assert_allclose(solution, np.array([1.0 + 0j, -2.0 + 0j]))


def test_solve_linear_system_returns_nonfinite_vector_for_invalid_inputs():
    nonfinite = solve_linear_system(
        np.array([[np.nan + 0j]], dtype=complex),
        np.array([1.0 + 0j], dtype=complex),
    )
    malformed = solve_linear_system(
        np.array([1.0 + 0j], dtype=complex),
        np.array([1.0 + 0j], dtype=complex),
    )

    assert nonfinite.shape == (1,)
    assert malformed.shape == (1,)
    assert not np.all(np.isfinite(nonfinite))
    assert not np.all(np.isfinite(malformed))


def test_solve_linear_system_rejects_nonnumeric_inputs():
    with pytest.raises(TypeError, match="jac must be numeric"):
        solve_linear_system([["not-numeric"]], np.array([1.0 + 0j]))
    with pytest.raises(TypeError, match="rhs must be numeric"):
        solve_linear_system(np.array([[1.0 + 0j]]), ["not-numeric"])


def test_numeric_newton_corrector_fails_on_nonfinite_linear_subproblem():
    start = np.array([0.5 + 0j])

    def f(point):
        return np.array([point[0] - 1], dtype=complex)

    def jac(point):
        return np.array([[np.nan + 0j]], dtype=complex)

    point, success, iters = newton_corrector_numeric(
        f,
        jac,
        start,
        max_iters=3,
        tol=1e-12,
    )

    assert not success
    assert iters == 1
    np.testing.assert_allclose(point, start)
