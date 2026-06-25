"""
Polynomial representation module for PyContinuum.

This module provides classes and functions for representing and manipulating
multivariate polynomials and polynomial systems.
"""

from collections.abc import Iterable, Mapping, MutableMapping
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from numbers import Integral, Number, Rational
from typing import Dict, List, Tuple, Union, Set, Any, Optional

import numpy as np


_VARIABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_numeric_coefficient(value: Any) -> bool:
    return (
        isinstance(value, Number)
        and not isinstance(value, (bool, np.bool_))
    )


def _coerce_coefficient(name: str, value: Any) -> Number:
    if not _is_numeric_coefficient(value):
        raise TypeError(f"{name} must be a numeric scalar")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Decimal):
        return _coerce_decimal_coefficient(name, value)
    if isinstance(value, Rational):
        return _coerce_rational_coefficient(name, value)
    try:
        coefficient = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a numeric scalar") from exc
    if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag):
        raise ValueError(f"{name} must be finite")
    if coefficient.imag == 0:
        real = float(coefficient.real)
        if real.is_integer() and isinstance(value, Integral):
            return int(value)
        return real
    return coefficient


def _coerce_decimal_coefficient(name: str, value: Decimal) -> Number:
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    if value == value.to_integral_value():
        return int(value)
    try:
        numeric_value = float(value)
    except OverflowError as exc:
        raise ValueError(
            f"{name} is too large for floating-point representation"
        ) from exc
    if not np.isfinite(numeric_value):
        raise ValueError(
            f"{name} is too large for floating-point representation"
        )
    if value != 0 and numeric_value == 0.0:
        raise ValueError(
            f"{name} is too small for floating-point representation"
        )
    return numeric_value


def _coerce_rational_coefficient(name: str, value: Rational) -> Number:
    if value.denominator == 1:
        return int(value)
    try:
        numeric_value = float(value)
    except OverflowError as exc:
        raise ValueError(
            f"{name} is too large for floating-point representation"
        ) from exc
    if not np.isfinite(numeric_value):
        raise ValueError(
            f"{name} is too large for floating-point representation"
        )
    if value != 0 and numeric_value == 0.0:
        raise ValueError(
            f"{name} is too small for floating-point representation"
        )
    return numeric_value


def _coerce_nonzero_divisor(value: Any) -> Number:
    if not _is_numeric_coefficient(value):
        raise TypeError("divisor must be a numeric scalar")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("divisor must be finite")
        divisor = value
    elif isinstance(value, Rational):
        divisor = value
    else:
        divisor = _coerce_coefficient("divisor", value)
    if divisor == 0:
        raise ZeroDivisionError("division by zero")
    return divisor


def _as_exact_real_fraction(name: str, value: Any) -> Optional[Fraction]:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, Integral):
        return Fraction(int(value), 1)
    if isinstance(value, Rational):
        return Fraction(value.numerator, value.denominator)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name} must be finite")
        return Fraction(value)
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError(f"{name} must be finite")
        return Fraction.from_float(numeric_value)
    return None


def _as_complex_scalar(name: str, value: Any) -> Optional[complex]:
    if not isinstance(value, (complex, np.complexfloating)):
        return None
    try:
        numeric_value = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a numeric scalar") from exc
    if not np.isfinite(numeric_value.real) or not np.isfinite(numeric_value.imag):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def _coerce_complex_components(name: str, real: Number, imag: Number) -> Number:
    try:
        value = complex(real, imag)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} is too large for floating-point representation"
        ) from exc
    return _coerce_coefficient(name, value)


def _coerce_inexact_operation_result(
    name: str,
    value: Any,
    *,
    expect_nonzero: bool,
) -> Number:
    if expect_nonzero:
        try:
            numeric_value = complex(value)
        except (TypeError, ValueError, OverflowError):
            numeric_value = None
        if numeric_value == 0:
            raise ValueError(
                f"{name} is too small for floating-point representation"
            )
    try:
        return _coerce_coefficient(name, value)
    except OverflowError as exc:
        raise ValueError(
            f"{name} is too large for floating-point representation"
        ) from exc


def _add_coefficients(name: str, left: Number, right: Number) -> Number:
    left_fraction = _as_exact_real_fraction(name, left)
    right_fraction = _as_exact_real_fraction(name, right)
    if left_fraction is not None and right_fraction is not None:
        return _coerce_coefficient(name, left_fraction + right_fraction)

    left_complex = _as_complex_scalar(name, left)
    right_complex = _as_complex_scalar(name, right)
    if left_complex is not None and right_fraction is not None:
        real = _add_coefficients(name, left_complex.real, right)
        return _coerce_complex_components(name, real, left_complex.imag)
    if right_complex is not None and left_fraction is not None:
        real = _add_coefficients(name, left, right_complex.real)
        return _coerce_complex_components(name, real, right_complex.imag)

    try:
        value = left + right
    except OverflowError as exc:
        raise ValueError(
            f"{name} is too large for floating-point representation"
        ) from exc
    return _coerce_inexact_operation_result(
        name,
        value,
        expect_nonzero=False,
    )


def _multiply_coefficients(name: str, left: Number, right: Number) -> Number:
    left_fraction = _as_exact_real_fraction(name, left)
    right_fraction = _as_exact_real_fraction(name, right)
    if left_fraction is not None and right_fraction is not None:
        return _coerce_coefficient(name, left_fraction * right_fraction)

    left_complex = _as_complex_scalar(name, left)
    right_complex = _as_complex_scalar(name, right)
    if left_complex is not None and right_fraction is not None:
        real = (
            _multiply_coefficients(name, left_complex.real, right)
            if left_complex.real != 0
            else 0.0
        )
        imag = (
            _multiply_coefficients(name, left_complex.imag, right)
            if left_complex.imag != 0
            else 0.0
        )
        return _coerce_complex_components(name, real, imag)
    if right_complex is not None and left_fraction is not None:
        real = (
            _multiply_coefficients(name, left, right_complex.real)
            if right_complex.real != 0
            else 0.0
        )
        imag = (
            _multiply_coefficients(name, left, right_complex.imag)
            if right_complex.imag != 0
            else 0.0
        )
        return _coerce_complex_components(name, real, imag)

    try:
        value = left * right
    except OverflowError as exc:
        raise ValueError(
            f"{name} is too large for floating-point representation"
        ) from exc
    return _coerce_inexact_operation_result(
        name,
        value,
        expect_nonzero=left != 0 and right != 0,
    )


def _divide_coefficients(name: str, numerator: Number, denominator: Number) -> Number:
    divisor = _coerce_nonzero_divisor(denominator)
    numerator_fraction = _as_exact_real_fraction(name, numerator)
    divisor_fraction = _as_exact_real_fraction("divisor", divisor)
    if numerator_fraction is not None and divisor_fraction is not None:
        return _coerce_coefficient(name, numerator_fraction / divisor_fraction)

    numerator_complex = _as_complex_scalar(name, numerator)
    if numerator_complex is not None and divisor_fraction is not None:
        real = (
            _divide_coefficients(name, numerator_complex.real, divisor)
            if numerator_complex.real != 0
            else 0.0
        )
        imag = (
            _divide_coefficients(name, numerator_complex.imag, divisor)
            if numerator_complex.imag != 0
            else 0.0
        )
        return _coerce_complex_components(name, real, imag)

    try:
        value = numerator / divisor
    except OverflowError as exc:
        if numerator != 0:
            raise ValueError(
                f"{name} is too small for floating-point representation"
            ) from exc
        raise
    return _coerce_inexact_operation_result(
        name,
        value,
        expect_nonzero=numerator != 0,
    )


def _validate_evaluation_values(values: Any, variables: Set["Variable"]) -> None:
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping from Variable objects to values")
    missing = sorted(variable.name for variable in variables if variable not in values)
    if missing:
        raise ValueError(
            "Missing value for variable(s): " + ", ".join(missing)
        )
    for variable in sorted(variables, key=lambda var: var.name):
        value = values[variable]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
            raise TypeError(f"values[{variable.name}] must be a numeric coordinate")


def _validate_variable(name: str, value: Any) -> "Variable":
    if not isinstance(value, Variable):
        raise TypeError(f"{name} must be a Variable")
    return value


def _normalize_variable_sequence(name: str, variables: Any) -> Tuple["Variable", ...]:
    if isinstance(variables, (str, bytes)) or not isinstance(variables, Iterable):
        raise TypeError(f"{name} must be an iterable of Variable objects")

    normalized = tuple(variables)
    seen = set()
    duplicates = []
    for index, variable in enumerate(normalized):
        if not isinstance(variable, Variable):
            raise TypeError(f"{name}[{index}] must be a Variable")
        if variable in seen:
            duplicates.append(variable.name)
        seen.add(variable)

    if duplicates:
        raise ValueError(
            f"{name} contains duplicate variable(s): "
            + ", ".join(sorted(set(duplicates)))
        )

    return normalized


def _validate_parse_variables(variables: Any) -> MutableMapping[str, "Variable"]:
    if not isinstance(variables, MutableMapping):
        raise TypeError(
            "variables must be a mutable mapping from names to Variable objects"
        )

    for key, variable in variables.items():
        if not isinstance(key, str):
            raise TypeError("variables keys must be strings")
        if not isinstance(variable, Variable):
            raise TypeError("variables values must be Variable objects")
        if key != variable.name:
            raise ValueError("variables mapping keys must match Variable.name")

    return variables


def _split_system_expression(expr: str) -> List[str]:
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("Polynomial system expression must be a non-empty string")

    equations = []
    start = 0
    depth = 0
    for index, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unexpected ')' in polynomial system expression")
        elif depth == 0 and char in ";\n":
            equation = expr[start:index].strip()
            if equation:
                equations.append(equation)
            start = index + 1

    if depth != 0:
        raise ValueError("Expected ')' in polynomial system expression")

    equation = expr[start:].strip()
    if equation:
        equations.append(equation)
    if not equations:
        raise ValueError("Polynomial system expression must contain equations")
    return equations


def _split_equation_expression(expr: str) -> Optional[Tuple[str, str]]:
    depth = 0
    equals_index = None
    for index, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unexpected ')' in polynomial equation")
        elif char == "=" and depth == 0:
            if equals_index is not None:
                raise ValueError("Polynomial equation may contain at most one '='")
            equals_index = index
    if depth != 0:
        raise ValueError("Expected ')' in polynomial equation")
    if equals_index is None:
        return None
    left = expr[:equals_index].strip()
    right = expr[equals_index + 1:].strip()
    if not left or not right:
        raise ValueError("Polynomial equation must have expressions on both sides")
    return left, right


def _parse_system_equation(
    expr: str,
    variables: MutableMapping[str, "Variable"],
) -> "Polynomial":
    equation = _split_equation_expression(expr)
    if equation is None:
        return Polynomial.parse(expr, variables=variables)
    left, right = equation
    return (
        Polynomial.parse(left, variables=variables)
        - Polynomial.parse(right, variables=variables)
    )


class Variable:
    """Representation of a polynomial variable."""
    
    def __init__(self, name: str):
        """Initialize a variable with a name.
        
        Args:
            name: String name of the variable
        """
        if not isinstance(name, str):
            raise TypeError("Variable name must be a string")
        if not _VARIABLE_NAME_RE.fullmatch(name):
            raise ValueError(
                "Variable name must be a valid identifier matching "
                "[A-Za-z_][A-Za-z0-9_]*"
            )
        self.name = name
        
    def __repr__(self) -> str:
        return self.name
    
    # Add these methods to properly handle equality and hashing
    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)

    def __neg__(self) -> "Monomial":
        """Return the additive inverse of this variable."""
        return Monomial({self: 1}, coefficient=-1)
    
    def __pow__(self, exponent: int) -> "Polynomial":
        """Raise the variable to a power, creating a polynomial."""
        if isinstance(exponent, (bool, np.bool_)) or not isinstance(exponent, Integral):
            raise ValueError("Exponent must be a non-negative integer")
        if exponent < 0:
            raise ValueError("Exponent must be a non-negative integer")
        return Polynomial([Monomial({self: exponent})])
    
    def __mul__(self, other: Any) -> Union["Polynomial", "Monomial"]:
        """Multiply the variable by another object."""
        if _is_numeric_coefficient(other):
            return Monomial({self: 1}, coefficient=other)
        elif isinstance(other, Variable):
            variables = {self: 1}
            variables[other] = variables.get(other, 0) + 1
            return Monomial(variables)
        elif isinstance(other, Monomial):
            new_vars = other.variables.copy()
            if self in new_vars:
                new_vars[self] += 1
            else:
                new_vars[self] = 1
            return Monomial(new_vars, coefficient=other.coefficient)
        elif isinstance(other, Polynomial):
            return other * self
        else:
            return NotImplemented

    def __truediv__(self, other: Any) -> "Monomial":
        """Divide this variable by a numeric scalar."""
        if _is_numeric_coefficient(other):
            return Monomial(
                {self: 1},
                coefficient=_divide_coefficients(
                    "Monomial coefficient",
                    1,
                    other,
                ),
            )
        return NotImplemented
    
    def __rmul__(self, other: Any) -> "Monomial":
        """Handle multiplication when the variable is on the right."""
        if _is_numeric_coefficient(other):
            return Monomial({self: 1}, coefficient=other)
        return NotImplemented
    
    def __add__(self, other: Any) -> "Polynomial":
        """Add the variable to another object."""
        if _is_numeric_coefficient(other):
            return Polynomial([Monomial({self: 1}), Monomial({}, coefficient=other)])
        elif isinstance(other, Variable):
            return Polynomial([Monomial({self: 1}), Monomial({other: 1})])
        elif isinstance(other, (Monomial, Polynomial)):
            return Polynomial([Monomial({self: 1})]) + other
        else:
            return NotImplemented
    
    def __radd__(self, other: Any) -> "Polynomial":
        """Handle addition when the variable is on the right."""
        if _is_numeric_coefficient(other):
            return Polynomial([Monomial({self: 1}), Monomial({}, coefficient=other)])
        return NotImplemented
    
    def __sub__(self, other: Any) -> "Polynomial":
        """Subtract another object from the variable."""
        if _is_numeric_coefficient(other):
            return Polynomial([Monomial({self: 1}), Monomial({}, coefficient=-other)])
        elif isinstance(other, Variable):
            return Polynomial([Monomial({self: 1}), Monomial({other: 1}, coefficient=-1)])
        elif isinstance(other, (Monomial, Polynomial)):
            return Polynomial([Monomial({self: 1})]) - other
        else:
            return NotImplemented
    
    def __rsub__(self, other: Any) -> "Polynomial":
        """Handle subtraction when the variable is on the right."""
        if _is_numeric_coefficient(other):
            return Polynomial([Monomial({self: 1}, coefficient=-1), Monomial({}, coefficient=other)])
        return NotImplemented


class _VariableExponentMap(dict):
    """Dictionary of variable exponents that is also callable as a variable set."""

    def __call__(self) -> Set[Variable]:
        return set(self.keys())


class Monomial:
    """Representation of a monomial (term in a polynomial)."""
    
    def __init__(self, variables: Dict[Variable, int], coefficient: complex = 1):
        """Initialize a monomial with variables and their exponents.
        
        Args:
            variables: Dict mapping Variable objects to their exponents
            coefficient: Coefficient of the monomial (default: 1)
        """
        if not isinstance(variables, Mapping):
            raise TypeError(
                "Monomial variables must be a mapping from Variable objects to exponents"
            )
        for var, exp in variables.items():
            if not isinstance(var, Variable):
                raise TypeError("Monomial variables must be Variable instances")
            if isinstance(exp, (bool, np.bool_)) or not isinstance(exp, Integral):
                raise TypeError("Monomial exponents must be integers")
            if exp < 0:
                raise ValueError("Monomial exponents must be non-negative")

        # Filter out zero exponents
        self.variables = _VariableExponentMap(
            {var: exp for var, exp in variables.items() if exp != 0}
        )
        self.coefficient = _coerce_coefficient("Monomial coefficient", coefficient)
        
    def __repr__(self) -> str:
        if not self.variables and self.coefficient == 0:
            return "0"
        
        if not self.variables:
            return str(self.coefficient)
        
        coef_str = ""
        if self.coefficient != 1:
            if self.coefficient == -1:
                coef_str = "-"
            else:
                coef_str = f"{self.coefficient}*"
        
        var_strs = []
        for var, exp in sorted(self.variables.items(), key=lambda item: item[0].name):
            if exp == 1:
                var_strs.append(f"{var.name}")
            else:
                var_strs.append(f"{var.name}^{exp}")
        
        return f"{coef_str}{'*'.join(var_strs)}"
    
    def degree(self) -> int:
        """Get the total degree of the monomial."""
        return sum(self.variables.values())
    
    def evaluate(self, values: Dict[Variable, complex]) -> complex:
        """Evaluate the monomial at specific variable values.
        
        Args:
            values: Dict mapping variables to their values
            
        Returns:
            The evaluated value of the monomial
        """
        _validate_evaluation_values(values, set(self.variables))
        return self._evaluate_unchecked(values)

    def _evaluate_unchecked(self, values: Dict[Variable, complex]) -> complex:
        result = self.coefficient
        for var, exp in self.variables.items():
            result *= values[var] ** exp
        return result
    
    def __mul__(self, other: Any) -> Union["Polynomial", "Monomial"]:
        """Multiply the monomial by another object."""
        if _is_numeric_coefficient(other):
            return Monomial(
                self.variables.copy(),
                coefficient=_multiply_coefficients(
                    "Monomial coefficient",
                    self.coefficient,
                    other,
                ),
            )
        elif isinstance(other, Variable):
            new_vars = self.variables.copy()
            if other in new_vars:
                new_vars[other] += 1
            else:
                new_vars[other] = 1
            return Monomial(new_vars, coefficient=self.coefficient)
        elif isinstance(other, Monomial):
            new_vars = self.variables.copy()
            for var, exp in other.variables.items():
                if var in new_vars:
                    new_vars[var] += exp
                else:
                    new_vars[var] = exp
            return Monomial(
                new_vars,
                coefficient=_multiply_coefficients(
                    "Monomial coefficient",
                    self.coefficient,
                    other.coefficient,
                ),
            )
        elif isinstance(other, Polynomial):
            return other * self
        else:
            return NotImplemented

    def __truediv__(self, other: Any) -> "Monomial":
        """Divide this monomial by a numeric scalar."""
        if _is_numeric_coefficient(other):
            return Monomial(
                self.variables.copy(),
                coefficient=_divide_coefficients(
                    "Monomial coefficient",
                    self.coefficient,
                    other,
                ),
            )
        return NotImplemented
    
    def __rmul__(self, other: Any) -> "Monomial":
        """Handle multiplication when the monomial is on the right."""
        if _is_numeric_coefficient(other):
            return Monomial(
                self.variables.copy(),
                coefficient=_multiply_coefficients(
                    "Monomial coefficient",
                    self.coefficient,
                    other,
                ),
            )
        return NotImplemented
    
    def __add__(self, other: Any) -> "Polynomial":
        """Add the monomial to another object."""
        if _is_numeric_coefficient(other):
            return Polynomial([self, Monomial({}, coefficient=other)])
        elif isinstance(other, Variable):
            return Polynomial([self, Monomial({other: 1})])
        elif isinstance(other, Monomial):
            return Polynomial([self, other])
        elif isinstance(other, Polynomial):
            return other + self
        else:
            return NotImplemented
    
    def __radd__(self, other: Any) -> "Polynomial":
        """Handle addition when the monomial is on the right."""
        if _is_numeric_coefficient(other):
            return Polynomial([self, Monomial({}, coefficient=other)])
        return NotImplemented
    
    def __sub__(self, other: Any) -> "Polynomial":
        """Subtract another object from the monomial."""
        if _is_numeric_coefficient(other):
            return Polynomial([self, Monomial({}, coefficient=-other)])
        elif isinstance(other, Variable):
            return Polynomial([self, Monomial({other: 1}, coefficient=-1)])
        elif isinstance(other, Monomial):
            return Polynomial([self, Monomial(other.variables, coefficient=-other.coefficient)])
        elif isinstance(other, Polynomial):
            return Polynomial([self]) - other
        else:
            return NotImplemented
    
    def __rsub__(self, other: Any) -> "Polynomial":
        """Handle subtraction when the monomial is on the right."""
        if _is_numeric_coefficient(other):
            return Polynomial([Monomial(self.variables, coefficient=-self.coefficient), 
                             Monomial({}, coefficient=other)])
        return NotImplemented
    
    def partial_derivative(self, var: Variable) -> "Monomial":
        """Compute partial derivative with respect to a variable.
        
        Args:
            var: Variable to differentiate with respect to
            
        Returns:
            Derivative as a new Monomial
        """
        var = _validate_variable("var", var)
        if var not in self.variables:
            return Monomial({}, coefficient=0)
        
        exp = self.variables[var]
        new_vars = self.variables.copy()
        new_coef = _multiply_coefficients(
            "Monomial coefficient",
            self.coefficient,
            exp,
        )
        
        if exp == 1:
            del new_vars[var]
        else:
            new_vars[var] = exp - 1
            
        return Monomial(new_vars, coefficient=new_coef)
    
    def as_polynomial(self) -> "Polynomial":
        """Convert this monomial to a polynomial."""
        return Polynomial([self])
    
    def variables(self) -> Set[Variable]:
        """Get the set of variables in this monomial."""
        return set(self.variables.keys())

    def __neg__(self) -> "Monomial":
        """Return the additive inverse of this monomial."""
        return Monomial(self.variables.copy(), coefficient=-self.coefficient)


class Polynomial:
    """Representation of a multivariate polynomial."""
    
    def __init__(self, terms: List[Union[Monomial, Variable, int, float, complex]]):
        """Initialize a polynomial from a list of terms."""
        if isinstance(terms, (str, bytes)) or not isinstance(terms, Iterable):
            raise TypeError(
                "Polynomial terms must be an iterable of Monomial, Variable, "
                "or numeric terms"
            )
        processed_terms = []
        for term in terms:
            if isinstance(term, Monomial):
                processed_terms.append(term)
            elif isinstance(term, Variable):
                processed_terms.append(Monomial({term: 1}))
            elif _is_numeric_coefficient(term):
                processed_terms.append(Monomial({}, coefficient=term))
            else:
                raise TypeError(f"Unsupported term type: {type(term)}")

        # Combine like terms immediately upon initialization
        self.terms = self._combine_like_terms(processed_terms)
    
    def _combine_like_terms(self, terms: List[Monomial]) -> List[Monomial]:
        """Combine terms with the same variable exponents."""
        # Use a dictionary to group terms by their variable exponents (as a tuple of tuples)
        term_dict: Dict[Tuple[Tuple[Variable, int], ...], complex] = {}

        for term in terms:
            # Create a hashable representation of the variables and exponents
            # Sort the variable items for consistent keys
            var_key = tuple(sorted(term.variables.items(), key=lambda item: item[0].name))

            if var_key in term_dict:
                term_dict[var_key] = _add_coefficients(
                    "Monomial coefficient",
                    term_dict[var_key],
                    term.coefficient,
                )
            else:
                term_dict[var_key] = term.coefficient

        # Create new Monomial objects from the combined terms.  Do not use an
        # absolute tolerance here: tiny coefficients can define a legitimate
        # scaled equation and should be handled by solver-level scaling.
        combined_terms = []
        for var_key, coef in term_dict.items():
            if coef != 0:
                # Convert the tuple key back to a dictionary
                variables_dict = dict(var_key)
                combined_terms.append(Monomial(variables_dict, coefficient=coef))

        # Optionally sort terms for consistent representation (e.g., by total degree, then lexicographically)
        # This is not strictly necessary for correctness but makes debugging easier.
        # A simple sort by string representation of variables might suffice for now.
        combined_terms.sort(key=lambda m: repr(m.variables)) # Simple sort key

        return combined_terms
    
    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        
        term_strs: List[str] = []
        for i, term in enumerate(self.terms):
            term_repr = str(term)
            if i == 0:
                term_strs.append(term_repr)
            else:
                if term_repr.startswith("-"):
                    term_strs.append(f"- {term_repr[1:]}")
                else:
                    term_strs.append(f"+ {term_repr}")
        
        return " ".join(term_strs)

    def __neg__(self) -> "Polynomial":
        """Return the additive inverse of this polynomial."""
        return Polynomial([-term for term in self.terms])

    @staticmethod
    def parse(expr: str, variables: Optional[Dict[str, "Variable"]] = None) -> "Polynomial":
        """Parse a simple polynomial string into a Polynomial.
        Supported syntax examples: "x^2 + 3*x*y - 1",
        "x**2 + 3*x*y - 1", "2x^2y - 5", "x + 2j*y",
        "(x + 1)*(y - 2)".
        This is a lightweight parser intended for convenience and tests.

        Args:
            expr: Polynomial expression string
            variables: Optional pre-existing name->Variable map to reuse

        Returns:
            Polynomial instance
        """
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError("Polynomial expression must be a non-empty string")
        if variables is None:
            variables = {}
        else:
            variables = _validate_parse_variables(variables)

        token_pattern = re.compile(
            r"(?P<imag>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[jJ])"
            r"|(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
            r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
            r"|(?P<op>\*\*|[+\-*^()])"
        )
        compact = "".join(expr.split())
        tokens: List[Tuple[str, str]] = []
        pos = 0
        while pos < len(compact):
            match = token_pattern.match(compact, pos)
            if not match:
                raise ValueError(f"Unexpected token at position {pos}: {compact[pos]!r}")
            kind = match.lastgroup
            value = match.group()
            tokens.append((kind, value))
            pos = match.end()

        def parse_exponent(value: str) -> int:
            try:
                numeric = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError(f"Invalid exponent: {value}") from exc
            if (
                not numeric.is_finite()
                or numeric != numeric.to_integral_value()
                or numeric < 0
            ):
                raise ValueError("Exponents must be non-negative integers")
            return int(numeric)

        def parse_number(value: str) -> Number:
            try:
                numeric = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError(f"Invalid numeric coefficient: {value}") from exc
            if not numeric.is_finite():
                raise ValueError("Numeric coefficients must be finite")
            if numeric == numeric.to_integral_value():
                return int(numeric)

            numeric_float = float(numeric)
            if not np.isfinite(numeric_float):
                raise ValueError(
                    "Non-integer numeric coefficient is too large for "
                    "floating-point representation"
                )
            if numeric != 0 and numeric_float == 0.0:
                raise ValueError(
                    "Nonzero numeric coefficient is too small for "
                    "floating-point representation"
                )
            return numeric_float

        def parse_imag_number(value: str) -> Number:
            numeric = parse_number(value[:-1])
            if numeric == 0:
                return 0.0
            return _coerce_complex_components(
                "Imaginary numeric coefficient",
                0,
                numeric,
            )

        class Parser:
            def __init__(self, parsed_tokens: List[Tuple[str, str]]):
                self.tokens = parsed_tokens
                self.index = 0

            def current(self) -> Optional[Tuple[str, str]]:
                if self.index >= len(self.tokens):
                    return None
                return self.tokens[self.index]

            def advance(self) -> Tuple[str, str]:
                token = self.tokens[self.index]
                self.index += 1
                return token

            def parse(self) -> "Polynomial":
                polynomial = self.parse_expression()
                if self.current() is not None:
                    _kind, value = self.current()
                    raise ValueError(f"Unexpected token: {value}")
                return polynomial

            def parse_expression(self) -> "Polynomial":
                result = self.parse_term()
                while self.current() is not None:
                    kind, value = self.current()
                    if kind != "op" or value not in "+-":
                        break
                    self.advance()
                    rhs = self.parse_term()
                    result = result + rhs if value == "+" else result - rhs
                return result

            def parse_term(self) -> "Polynomial":
                result = self.parse_factor()
                while self.current() is not None:
                    kind, value = self.current()
                    if kind == "op" and value == "*":
                        self.advance()
                        result = result * self.parse_factor()
                    elif self._starts_implicit_factor(kind, value):
                        result = result * self.parse_factor()
                    else:
                        break
                return result

            def parse_factor(self) -> "Polynomial":
                sign = 1
                if self.current() is not None:
                    kind, value = self.current()
                    if kind == "op" and value in "+-":
                        sign = -1 if value == "-" else 1
                        self.advance()
                factor = self.parse_power_base()
                if sign < 0:
                    return -factor
                return factor

            def parse_power_base(self) -> "Polynomial":
                base = self.parse_primary()
                if self.current() is not None:
                    kind, value = self.current()
                    if kind == "op" and value in {"^", "**"}:
                        exponent_operator = value
                        self.advance()
                        if (
                            self.current() is None
                            or self.current()[0] != "number"
                        ):
                            raise ValueError(
                                "Expected integer exponent after "
                                f"'{exponent_operator}'"
                            )
                        _number_kind, exponent_value = self.advance()
                        base = base ** parse_exponent(exponent_value)
                return base

            def parse_primary(self) -> "Polynomial":
                token = self.current()
                if token is None:
                    raise ValueError("Incomplete polynomial term")
                kind, value = token
                if kind == "number":
                    self.advance()
                    return Polynomial([parse_number(value)])
                if kind == "imag":
                    self.advance()
                    return Polynomial([parse_imag_number(value)])
                if kind == "name":
                    self.advance()
                    if value not in variables:
                        variables[value] = Variable(value)
                    return Polynomial([Monomial({variables[value]: 1})])
                if kind == "op" and value == "(":
                    self.advance()
                    expression = self.parse_expression()
                    if self.current() != ("op", ")"):
                        raise ValueError("Expected ')' in polynomial expression")
                    self.advance()
                    return expression
                if kind == "op" and value in {"^", "**"}:
                    raise ValueError(
                        f"Unexpected '{value}' in polynomial expression"
                    )
                raise ValueError(f"Unexpected token: {value}")

            @staticmethod
            def _starts_implicit_factor(kind: str, value: str) -> bool:
                return kind in {"number", "imag", "name"} or (
                    kind == "op" and value == "("
                )

        return Parser(tokens).parse()
    
    def degree(self) -> int:
        """Get the maximum degree of any term in the polynomial."""
        if not self.terms:
            return 0
        return max(term.degree() for term in self.terms)
    
    def variables(self) -> Set[Variable]:
        """Get the set of all variables in the polynomial."""
        vars_set = set()
        for term in self.terms:
            vars_set.update(term.variables.keys())
        return vars_set
    
    def evaluate(self, values: Dict[Variable, complex]) -> complex:
        """Evaluate the polynomial at specific variable values.
        
        Args:
            values: Dict mapping variables to their values
            
        Returns:
            The evaluated value of the polynomial
        """
        _validate_evaluation_values(values, self.variables())
        return self._evaluate_unchecked(values)

    def _evaluate_unchecked(self, values: Dict[Variable, complex]) -> complex:
        return sum(term._evaluate_unchecked(values) for term in self.terms)
    
    def __add__(self, other: Any) -> "Polynomial":
        """Add another object to the polynomial."""
        if _is_numeric_coefficient(other):
            return Polynomial(self.terms + [Monomial({}, coefficient=other)])
        elif isinstance(other, Variable):
            return Polynomial(self.terms + [Monomial({other: 1})])
        elif isinstance(other, Monomial):
            return Polynomial(self.terms + [other])
        elif isinstance(other, Polynomial):
            return Polynomial(self.terms + other.terms)
        else:
            return NotImplemented
    
    def __radd__(self, other: Any) -> "Polynomial":
        """Handle addition when the polynomial is on the right."""
        return self + other
    
    def __sub__(self, other: Any) -> "Polynomial":
        """Subtract another object from the polynomial."""
        if _is_numeric_coefficient(other):
            return Polynomial(self.terms + [Monomial({}, coefficient=-other)])
        elif isinstance(other, Variable):
            return Polynomial(self.terms + [Monomial({other: 1}, coefficient=-1)])
        elif isinstance(other, Monomial):
            return Polynomial(self.terms + [Monomial(other.variables, coefficient=-other.coefficient)])
        elif isinstance(other, Polynomial):
            return Polynomial(self.terms + [Monomial(term.variables, coefficient=-term.coefficient) 
                                          for term in other.terms])
        else:
            return NotImplemented
    
    def __rsub__(self, other: Any) -> "Polynomial":
        """Handle subtraction when the polynomial is on the right."""
        if _is_numeric_coefficient(other):
            return Polynomial([Monomial({}, coefficient=other)] + 
                            [Monomial(term.variables, coefficient=-term.coefficient) for term in self.terms])
        return NotImplemented
    
    def __mul__(self, other: Any) -> "Polynomial":
        """Multiply the polynomial by another object."""
        if _is_numeric_coefficient(other):
            return Polynomial([term * other for term in self.terms])
        elif isinstance(other, Variable):
            return Polynomial([term * other for term in self.terms])
        elif isinstance(other, Monomial):
            return Polynomial([term * other for term in self.terms])
        elif isinstance(other, Polynomial):
            result_terms = []
            for term1 in self.terms:
                for term2 in other.terms:
                    result_terms.append(term1 * term2)
            return Polynomial(result_terms)
        else:
            return NotImplemented

    def __truediv__(self, other: Any) -> "Polynomial":
        """Divide this polynomial by a numeric scalar."""
        if _is_numeric_coefficient(other):
            divisor = _coerce_nonzero_divisor(other)
            return Polynomial([term / divisor for term in self.terms])
        return NotImplemented
    
    def __rmul__(self, other: Any) -> "Polynomial":
        """Handle multiplication when the polynomial is on the right."""
        return self * other
    
    def __pow__(self, exponent: int) -> "Polynomial":
        """Raise the polynomial to a power.
        
        Args:
            exponent: Non-negative integer exponent
            
        Returns:
            The polynomial raised to the given power
        """
        if (
            isinstance(exponent, (bool, np.bool_))
            or not isinstance(exponent, Integral)
            or exponent < 0
        ):
            raise ValueError("Exponent must be a non-negative integer")
        exponent = int(exponent)
        
        if exponent == 0:
            return Polynomial([1])
        
        if exponent == 1:
            return Polynomial(self.terms)
        
        # Use binary exponentiation for efficiency
        result = Polynomial([1])
        base = Polynomial(self.terms)
        while exponent > 0:
            if exponent & 1:  # exponent is odd
                result = result * base
            base = base * base
            exponent >>= 1
            
        return result
    
    def partial_derivative(self, var: Variable) -> "Polynomial":
        """Compute partial derivative with respect to a variable.
        
        Args:
            var: Variable to differentiate with respect to
            
        Returns:
            Derivative as a new Polynomial
        """
        var = _validate_variable("var", var)
        return Polynomial([term.partial_derivative(var) for term in self.terms])
    
    def jacobian(self, vars_list: List[Variable]) -> List[List["Polynomial"]]:
        """Compute the Jacobian matrix of partial derivatives.
        
        Args:
            vars_list: List of variables for the Jacobian
            
        Returns:
            Jacobian matrix as a list of lists of polynomials
        """
        variables = _normalize_variable_sequence("vars_list", vars_list)
        return [[self.partial_derivative(var) for var in variables]]


class PolynomialSystem:
    """Representation of a system of polynomial equations."""
    
    def __init__(
        self,
        equations: List[Union[Polynomial, Monomial, Variable, int, float, complex]],
    ):
        """Initialize a polynomial system from a list of equations."""
        if isinstance(equations, (str, bytes)) or not isinstance(equations, Iterable):
            raise TypeError(
                "PolynomialSystem equations must be an iterable of Polynomial, "
                "Monomial, Variable, or numeric equations"
            )
        self.equations = []
        for eq in equations:
            if isinstance(eq, Polynomial):
                self.equations.append(eq)
            elif isinstance(eq, (Monomial, Variable)):
                # Convert Monomial or Variable to Polynomial
                self.equations.append(Polynomial([eq]))
            elif _is_numeric_coefficient(eq):
                self.equations.append(Polynomial([eq]))
            else:
                raise TypeError(f"Unsupported equation type: {type(eq)}")

    @staticmethod
    def parse(
        equations: Any,
        variables: Optional[Dict[str, Variable]] = None,
    ) -> "PolynomialSystem":
        """Parse polynomial equations into a PolynomialSystem.

        A string input may contain equations separated by semicolons or
        newlines. Each equation can be either an expression interpreted as
        equal to zero, or a single ``left = right`` equation that is converted
        to ``left - right``.
        """
        if variables is None:
            variables = {}
        else:
            variables = _validate_parse_variables(variables)

        if isinstance(equations, str):
            equation_exprs = _split_system_expression(equations)
        elif isinstance(equations, Iterable):
            equation_exprs = []
            for index, equation in enumerate(equations):
                if not isinstance(equation, str) or not equation.strip():
                    raise ValueError(
                        f"equations[{index}] must be a non-empty string"
                    )
                equation_exprs.append(equation)
            if not equation_exprs:
                raise ValueError("Polynomial system must contain equations")
        else:
            raise TypeError(
                "equations must be a string or iterable of equation strings"
            )

        return PolynomialSystem(
            _parse_system_equation(expr, variables)
            for expr in equation_exprs
        )
    
    def __repr__(self) -> str:
        return "\n".join([f"{i}: {eq}" for i, eq in enumerate(self.equations)])
    
    def variables(self) -> Set[Variable]:
        """Get the set of all variables in the system."""
        vars_set = set()
        for eq in self.equations:
            vars_set.update(eq.variables())
        return vars_set

    def ordered_variables(self) -> List[Variable]:
        """Get system variables in deterministic name order."""
        return sorted(self.variables(), key=lambda var: var.name)
    
    def evaluate(self, values: Dict[Variable, complex]) -> List[complex]:
        """Evaluate the system at specific variable values.
        
        Args:
            values: Dict mapping variables to their values
        Returns:
            List of evaluated values for each equation
        """
        _validate_evaluation_values(values, self.variables())
        return [eq._evaluate_unchecked(values) for eq in self.equations]
    
    def jacobian(self, vars_list: List[Variable]) -> List[List[Polynomial]]:
        """Compute the Jacobian matrix of partial derivatives.
        
        Args:
            vars_list: List of variables for the Jacobian
            
        Returns:
            Jacobian matrix as a list of lists of polynomials
        """
        variables = _normalize_variable_sequence("vars_list", vars_list)
        return [eq.jacobian(variables)[0] for eq in self.equations]
    
    def degrees(self) -> List[int]:
        """Get the degrees of each polynomial in the system.
        
        Returns:
            List of degrees for each polynomial
        """
        return [eq.degree() for eq in self.equations]
    

def polyvar(*names: str) -> Union[Variable, Tuple[Variable, ...]]:
    """Create polynomial variables with the given names.
    
    Args:
        *names: Variable names
        
    Returns:
        A single Variable or a tuple of Variables
    """
    if not names:
        raise ValueError("At least one variable name is required")
    variables = tuple(Variable(name) for name in names)
    seen = set()
    duplicates = []
    for variable in variables:
        name = variable.name
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            "Variable names must be unique: " + ", ".join(sorted(set(duplicates)))
        )
    return variables[0] if len(variables) == 1 else variables

