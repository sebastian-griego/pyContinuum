"""
Tests for the polynomial module of PyContinuum.
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import pycontinuum
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pycontinuum.polynomial import Variable, Polynomial, PolynomialSystem


def test_variable_creation():
    """Test the creation of variables."""
    x = Variable('x')
    assert x.name == 'x'
    assert str(x) == 'x'


def test_polynomial_creation():
    """Test the creation of polynomials."""
    x = Variable('x')
    y = Variable('y')
    
    # Test simple polynomials
    p1 = Polynomial([x])
    assert str(p1) == 'x'
    
    p2 = Polynomial([x, y])
    assert str(p2) == 'x + y'
    
    p3 = Polynomial([x, -1])
    assert str(p3) == 'x + -1'


def test_polynomial_addition():
    """Test polynomial addition."""
    x = Variable('x')
    y = Variable('y')
    
    p1 = Polynomial([x])
    p2 = Polynomial([y])
    
    # Variable + Variable
    p3 = p1 + p2
    assert str(p3) == 'x + y'
    
    # Polynomial + constant
    p4 = p1 + 5
    assert str(p4) == 'x + 5'
    
    # Polynomial + Polynomial
    p5 = p1 + p1
    assert str(p5) == 'x + x'  # This should ideally combine to 2*x


def test_polynomial_multiplication():
    """Test polynomial multiplication."""
    x = Variable('x')
    y = Variable('y')
    
    p1 = Polynomial([x])
    p2 = Polynomial([y])
    
    # Variable * Variable
    p3 = p1 * p2
    assert 'x*y' in str(p3)
    
    # Polynomial * constant
    p4 = p1 * 3
    assert '3*x' in str(p4) or '3x' in str(p4)
    
    # Polynomial * Polynomial
    p5 = p1 * p1
    assert 'x^2' in str(p5) or 'x*x' in str(p5)


def test_polynomial_evaluation():
    """Test polynomial evaluation."""
    x = Variable('x')
    y = Variable('y')
    
    # p = x^2 + 2*y
    p = Polynomial([Polynomial([x]) * Polynomial([x]), Polynomial([y]) * 2])
    
    # Evaluate at x=3, y=4
    values = {x: 3, y: 4}
    result = p.evaluate(values)
    
    # Should be 3^2 + 2*4 = 9 + 8 = 17
    assert abs(result - 17) < 1e-10


def test_polynomial_degree():
    """Test polynomial degree calculation."""
    x = Variable('x')
    y = Variable('y')
    
    # p = x^2 + y^3
    p = x**2 + y**3
    
    assert p.degree() == 3  # Highest degree term is y^3


def test_polynomial_system():
    """Test polynomial system functionality."""
    x = Variable('x')
    y = Variable('y')
    
    # System: x^2 + y^2 = 1, x + y = 0
    eq1 = x**2 + y**2 - 1
    eq2 = x + y
    
    system = PolynomialSystem([eq1, eq2])
    
    # Check variables
    vars_set = system.variables()
    assert x in vars_set
    assert y in vars_set
    assert len(vars_set) == 2
    
    # Check evaluation
    values = {x: 1/np.sqrt(2), y: -1/np.sqrt(2)}
    result = system.evaluate(values)
    
    # First equation should be close to 0
    assert abs(result[0]) < 1e-10
    # Second equation should be close to 0
    assert abs(result[1]) < 1e-10


def test_polynomial_derivatives():
    """Test polynomial derivatives."""
    x = Variable('x')
    y = Variable('y')
    
    # p = x^3 + 2*x*y + y^2
    p = x**3 + 2*x*y + y**2
    
    # dp/dx = 3*x^2 + 2*y
    dp_dx = p.partial_derivative(x)
    
    # Evaluate at x=2, y=1
    values = {x: 2, y: 1}
    result = dp_dx.evaluate(values)
    
    # Should be 3*(2^2) + 2*1 = 12 + 2 = 14
    assert abs(result - 14) < 1e-10
    
    # dp/dy = 2*x + 2*y
    dp_dy = p.partial_derivative(y)
    result = dp_dy.evaluate(values)
    
    # Should be 2*2 + 2*1 = 4 + 2 = 6
    assert abs(result - 6) < 1e-10


def test_jacobian():
    """Test Jacobian matrix computation."""
    x = Variable('x')
    y = Variable('y')
    
    # System: x^2 + y^2 = 1, x + y = 0
    eq1 = x**2 + y**2 - 1
    eq2 = x + y
    
    system = PolynomialSystem([eq1, eq2])
    
    # Compute Jacobian
    jac = system.jacobian([x, y])
    
    # Check dimensions
    assert len(jac) == 2
    assert len(jac[0]) == 2
    
    # Jacobian should be:
    # [2x, 2y]
    # [1,  1 ]
    
    # Evaluate at x=1, y=2
    values = {x: 1, y: 2}
    
    assert abs(jac[0][0].evaluate(values) - 2) < 1e-10  # 2x = 2*1 = 2
    assert abs(jac[0][1].evaluate(values) - 4) < 1e-10  # 2y = 2*2 = 4
    assert abs(jac[1][0].evaluate(values) - 1) < 1e-10  # 1
    assert abs(jac[1][1].evaluate(values) - 1) < 1e-10  # 1