# tests/test_polynomial.py
import pytest
import numpy as np

from pycontinuum import polyvar, PolynomialSystem, Polynomial, Variable

def test_create_simple_polynomial_system():
    """Tests the creation of a basic PolynomialSystem using native API."""
    # Use polyvar to create variables
    x, y = polyvar('x', 'y')
    
    # Create polynomials using operator overloading
    f1 = x**2 - y
    f2 = x - 1
    
    # Create the system
    system = PolynomialSystem([f1, f2])
    
    assert isinstance(system, PolynomialSystem)
    assert len(system.equations) == 2
    
    # Check that the system has the right variables
    system_vars = system.variables()
    assert len(system_vars) == 2
    assert all(isinstance(var, Variable) for var in system_vars)


def test_polynomial_system_accepts_numeric_constant_equations():
    system = PolynomialSystem([0, 1, 1 + 2j])

    assert len(system.equations) == 3
    assert system.variables() == set()
    np.testing.assert_allclose(
        np.array(system.evaluate({}), dtype=complex),
        np.array([0, 1, 1 + 2j], dtype=complex),
    )


def test_evaluate_polynomial_system():
    """Tests the evaluation of a PolynomialSystem at a given point."""
    x, y = polyvar('x', 'y')
    
    # Create system: x^2 - y = 0, x - 1 = 0
    system = PolynomialSystem([x**2 - y, x - 1])
    
    # Evaluate at point (2, 3)
    point_dict = {x: 2+0j, y: 3+0j}
    values = system.evaluate(point_dict)
    
    assert isinstance(values, list)
    assert len(values) == 2
    # x^2 - y at (2,3) = 4 - 3 = 1
    assert np.isclose(values[0], 1.0 + 0j)
    # x - 1 at (2,3) = 2 - 1 = 1  
    assert np.isclose(values[1], 1.0 + 0j)


def test_evaluate_rejects_missing_variable_values():
    x, y = polyvar('x', 'y')
    polynomial = x + y - 1
    system = PolynomialSystem([polynomial])
    monomial = x * y

    with pytest.raises(ValueError, match="Missing value.*y"):
        monomial.evaluate({x: 2.0 + 0j})
    with pytest.raises(ValueError, match="Missing value.*y"):
        polynomial.evaluate({x: 1.0 + 0j})
    with pytest.raises(ValueError, match="Missing value.*y"):
        system.evaluate({x: 1.0 + 0j})


def test_evaluate_rejects_non_mapping_values_but_allows_constants():
    x = polyvar('x')
    polynomial = x - 1
    constant_system = PolynomialSystem([polynomial - x + 1])

    with pytest.raises(TypeError, match="values must be a mapping"):
        polynomial.evaluate([1.0 + 0j])
    assert constant_system.evaluate({}) == [0]


def test_evaluate_rejects_nonnumeric_variable_values():
    x, y = polyvar('x', 'y')
    polynomial = x + y - 1
    system = PolynomialSystem([polynomial])
    monomial = x * y

    with pytest.raises(TypeError, match=r"values\[x\] must be a numeric coordinate"):
        monomial.evaluate({x: "1", y: 2.0 + 0j})
    with pytest.raises(TypeError, match=r"values\[x\] must be a numeric coordinate"):
        polynomial.evaluate({x: object(), y: 2.0 + 0j})
    with pytest.raises(TypeError, match=r"values\[x\] must be a numeric coordinate"):
        system.evaluate({x: True, y: 2.0 + 0j})


def test_evaluate_allows_nonfinite_numeric_values_for_diagnostics():
    x = polyvar('x')
    polynomial = x + 1
    system = PolynomialSystem([polynomial])

    assert np.isnan(polynomial.evaluate({x: float('nan')}))
    assert np.isinf(system.evaluate({x: float('inf')})[0])


def test_parse_polynomial_simple():
    # Parse a polynomial string and evaluate
    p = Polynomial.parse("x^2 + 3*x*y - 1")
    x, y = polyvar('x', 'y')
    # Map parsed variables to our created variables' names
    # Build a value dict for evaluation
    values = {x: 2.0 + 0j, y: 1.0 + 0j}
    # Re-evaluate by converting names in p's vars to these objects
    # Build a name->Variable dict from our created vars
    name_to_var = {x.name: x, y.name: y}
    # Evaluate p by translating variable keys
    # Create a dict using p.variables()
    test_values = {}
    for term in p.terms:
        for var in term.variables:
            test_values[var] = values[name_to_var[var.name]]
    val = p.evaluate(test_values)
    # 2^2 + 3*2*1 - 1 = 4 + 6 - 1 = 9
    assert np.isclose(val, 9.0 + 0j)
