"""
Utility functions for PyContinuum.

This module provides common utility functions used across the library.
"""

from collections.abc import Mapping
from fractions import Fraction
from numbers import Integral, Number, Real, Rational
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import math

import numpy as np

from pycontinuum.polynomial import Variable, PolynomialSystem


def _strict_json_value(value: Any) -> Any:
    """Return a JSON value that avoids NaN and Infinity numeric literals."""
    if isinstance(value, np.ndarray):
        return _strict_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, complex):
        return {
            "real": _strict_json_real(value.real),
            "imag": _strict_json_real(value.imag),
        }
    if isinstance(value, dict):
        return {
            _strict_json_key(key): _strict_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return _strict_json_real(value)
    return value


def _strict_json_key(key: Any) -> Any:
    strict_key = _strict_json_value(key)
    if isinstance(strict_key, (str, int, float, bool)) or strict_key is None:
        return strict_key
    return str(strict_key)


def _strict_json_real(value: Any) -> Any:
    try:
        numeric_value = float(value)
    except OverflowError:
        try:
            return "-Infinity" if value < 0 else "Infinity"
        except (TypeError, ValueError):
            return "NaN"
    except (TypeError, ValueError):
        return "NaN"
    if np.isnan(numeric_value):
        return "NaN"
    if np.isposinf(numeric_value):
        return "Infinity"
    if np.isneginf(numeric_value):
        return "-Infinity"
    return numeric_value


def _validate_polynomial_system(name: str, system: Any) -> PolynomialSystem:
    if not isinstance(system, PolynomialSystem):
        raise TypeError(f"{name} must be a PolynomialSystem")
    return system


def _normalize_variables(variables: Any) -> List[Variable]:
    try:
        normalized = list(variables)
    except TypeError as exc:
        raise TypeError("variables must be an iterable of Variable objects") from exc

    seen = set()
    duplicates = []
    for index, variable in enumerate(normalized):
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
    return normalized


def _coerce_numeric_vector(
    point: Any,
    label: str,
    expected_dimension: Optional[int] = None,
    *,
    allow_nonfinite: bool = False,
) -> np.ndarray:
    try:
        array = np.asarray(point, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a numeric one-dimensional point") from exc
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional; got shape {array.shape}")
    if expected_dimension is not None and array.size != expected_dimension:
        raise ValueError(
            f"{label} must have {expected_dimension} coordinate(s); "
            f"got shape {array.shape}"
        )
    if not allow_nonfinite and not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains nonfinite values")
    return array


def _coerce_point_for_variables(
    point: Any,
    variables: List[Variable],
    label: str,
    *,
    allow_nonfinite: bool = False,
) -> np.ndarray:
    values = None
    if isinstance(point, Mapping):
        values = point
    elif isinstance(getattr(point, "values", None), Mapping):
        values = point.values

    if values is None:
        return _coerce_numeric_vector(
            point,
            label,
            len(variables),
            allow_nonfinite=allow_nonfinite,
        )

    coordinates = []
    missing = []
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
        array = np.asarray(coordinates, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} coordinate(s) must be numeric") from exc
    if not allow_nonfinite and not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains nonfinite values")
    return array


def _mapping_coordinate_for_variable(
    values: Mapping,
    variable: Variable,
    label: str,
) -> Tuple[bool, Any]:
    """Return a coordinate from a mapping keyed by Variable or variable name."""
    has_variable_key = variable in values
    has_name_key = variable.name in values
    if has_variable_key and has_name_key:
        variable_value = values[variable]
        name_value = values[variable.name]
        if _coordinate_values_match(variable_value, name_value):
            return True, variable_value
        raise ValueError(
            f"{label} has conflicting coordinates for variable {variable.name}"
        )
    if has_variable_key:
        return True, values[variable]
    if has_name_key:
        return True, values[variable.name]
    return False, None


def _coordinate_values_match(left: Any, right: Any) -> bool:
    try:
        direct_match = left == right
    except (TypeError, ValueError, OverflowError):
        direct_match = False
    try:
        direct_match_array = np.asarray(direct_match)
        if direct_match_array.shape == () and bool(direct_match_array.item()):
            return True
    except (TypeError, ValueError, OverflowError):
        pass

    try:
        return complex(left) == complex(right)
    except (TypeError, ValueError, OverflowError):
        return False


def _coerce_complex_array(value: Any, name: str) -> np.ndarray:
    try:
        return np.array(value, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc


def _validate_nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def _validate_positive_finite_float(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return numeric_value


def _validate_callable(name: str, value: Any) -> Callable[[np.ndarray], np.ndarray]:
    if not callable(value):
        raise TypeError(f"{name} must be callable")
    return value


def evaluate_system_at_point(system: PolynomialSystem,
                            point: Any,
                            variables: List[Variable]) -> np.ndarray:
    """Evaluate a polynomial system at a point."""
    system = _validate_polynomial_system("system", system)
    variables = _normalize_variables(variables)
    point = _coerce_point_for_variables(
        point,
        variables,
        "point",
        allow_nonfinite=True,
    )
    # Create a dictionary mapping variables to their values
    var_dict = {var: val for var, val in zip(variables, point)}
    
    values = [
        _evaluate_polynomial_as_complex(equation, var_dict)
        for equation in system.equations
    ]
    return np.array(values, dtype=complex)


def evaluate_scaled_system_at_point(system: PolynomialSystem,
                                   point: Any,
                                   variables: List[Variable]) -> np.ndarray:
    """Evaluate equations after normalizing each by coefficient scale."""
    system = _validate_polynomial_system("system", system)
    variables = _normalize_variables(variables)
    point = _coerce_point_for_variables(
        point,
        variables,
        "point",
        allow_nonfinite=True,
    )
    var_dict = {var: val for var, val in zip(variables, point)}
    values = []
    for equation in system.equations:
        scale = _polynomial_coefficient_scale(equation)
        values.append(_evaluate_scaled_polynomial(equation, var_dict, scale))
    return np.array(values, dtype=complex)


def evaluate_backward_error_at_point(system: PolynomialSystem,
                                     point: Any,
                                     variables: List[Variable]) -> np.ndarray:
    """Evaluate normwise per-equation backward errors at a point.

    Each entry is ``|f_i(x)| / sum_j |c_ij x^a_j|``.  This catches roots whose
    absolute residual is inflated by cancellation between very large terms
    while still rejecting points that are not roots of a nearby coefficient
    perturbation.
    """
    system = _validate_polynomial_system("system", system)
    variables = _normalize_variables(variables)
    point = _coerce_point_for_variables(
        point,
        variables,
        "point",
        allow_nonfinite=True,
    )
    var_dict = {var: val for var, val in zip(variables, point)}
    errors = []
    for equation in system.equations:
        residual = _evaluate_polynomial_as_complex(equation, var_dict)
        scale = _polynomial_evaluation_scale(equation, var_dict)
        if (
            np.isfinite(residual.real)
            and np.isfinite(residual.imag)
            and np.isfinite(scale)
            and scale > 0.0
        ):
            errors.append(abs(residual) / scale)
            continue

        coefficient_scale = _polynomial_coefficient_scale(equation)
        try:
            scaled_residual = _evaluate_scaled_polynomial(
                equation,
                var_dict,
                coefficient_scale,
            )
        except (OverflowError, FloatingPointError):
            errors.append(float("inf"))
            continue
        scaled_scale = _scaled_polynomial_evaluation_scale(
            equation,
            var_dict,
            coefficient_scale,
        )
        if (
            np.isfinite(scaled_residual.real)
            and np.isfinite(scaled_residual.imag)
            and np.isfinite(scaled_scale)
            and scaled_scale > 0.0
        ):
            errors.append(abs(scaled_residual) / scaled_scale)
        else:
            errors.append(float("inf"))
    return np.array(errors, dtype=float)


def evaluate_jacobian_at_point(system: PolynomialSystem,
                              point: Any,
                              variables: List[Variable]) -> np.ndarray:
    """Evaluate the Jacobian of a polynomial system at a point."""
    system = _validate_polynomial_system("system", system)
    variables = _normalize_variables(variables)
    point = _coerce_point_for_variables(
        point,
        variables,
        "point",
        allow_nonfinite=True,
    )
    # Create a dictionary mapping variables to their values
    var_dict = {var: val for var, val in zip(variables, point)}
    
    # Get the Jacobian polynomials
    jac_polys = system.jacobian(variables)
    
    # Evaluate each polynomial in the Jacobian
    jac_values = []
    for row in jac_polys:
        jac_row = []
        for poly in row:
            jac_row.append(_evaluate_polynomial_as_complex(poly, var_dict))
        jac_values.append(jac_row)
    
    # Handle zero-equation systems: return a (0, n_vars) empty matrix
    if len(jac_values) == 0:
        return np.zeros((0, len(variables)), dtype=complex)
    
    # Convert to numpy array
    return np.array(jac_values, dtype=complex)


def evaluate_scaled_jacobian_at_point(system: PolynomialSystem,
                                     point: Any,
                                     variables: List[Variable]) -> np.ndarray:
    """Evaluate a coefficient-scaled Jacobian matrix at a point.

    Each derivative polynomial is divided by its largest coefficient magnitude
    before evaluation. This keeps rank tests from declaring an approximate
    multiple root regular merely because a tiny nonzero derivative appears in a
    raw floating-point endpoint.
    """
    system = _validate_polynomial_system("system", system)
    variables = _normalize_variables(variables)
    point = _coerce_point_for_variables(
        point,
        variables,
        "point",
        allow_nonfinite=True,
    )
    var_dict = {var: val for var, val in zip(variables, point)}
    jac_polys = system.jacobian(variables)
    jac_values = []
    for row in jac_polys:
        jac_row = []
        for poly in row:
            scale = _polynomial_coefficient_scale(poly)
            jac_row.append(_evaluate_scaled_polynomial(poly, var_dict, scale))
        jac_values.append(jac_row)

    if len(jac_values) == 0:
        return np.zeros((0, len(variables)), dtype=complex)
    return np.array(jac_values, dtype=complex)


def evaluate_equation_scaled_jacobian_at_point(
    system: PolynomialSystem,
    point: Any,
    variables: List[Variable],
) -> np.ndarray:
    """Evaluate the Jacobian of equations scaled by original row scale.

    This differs from :func:`evaluate_scaled_jacobian_at_point`, which scales
    each derivative polynomial by its own coefficient scale for rank tests.
    Predictor-corrector methods need the Jacobian of the same scaled equations
    used for residual evaluation, so each row here is divided by the scale of
    the corresponding original equation.
    """
    system = _validate_polynomial_system("system", system)
    variables = _normalize_variables(variables)
    point = _coerce_point_for_variables(
        point,
        variables,
        "point",
        allow_nonfinite=True,
    )
    var_dict = {var: val for var, val in zip(variables, point)}
    jac_polys = system.jacobian(variables)
    jac_values = []
    for equation, row in zip(system.equations, jac_polys):
        scale = _polynomial_coefficient_scale(equation)
        jac_values.append([
            _evaluate_scaled_polynomial(poly, var_dict, scale)
            for poly in row
        ])

    if len(jac_values) == 0:
        return np.zeros((0, len(variables)), dtype=complex)
    return np.array(jac_values, dtype=complex)


def _polynomial_coefficient_scale(polynomial: Any) -> Number:
    terms = getattr(polynomial, "terms", ())
    scale: Number = 0
    for term in terms:
        coefficient = getattr(term, "coefficient", 0.0)
        magnitude = _coefficient_magnitude(coefficient)
        if magnitude is not None and magnitude > scale:
            scale = magnitude
    return scale if scale > 0.0 else 1.0


def _polynomial_evaluation_scale(polynomial: Any, values: Dict[Variable, complex]) -> float:
    terms = getattr(polynomial, "terms", ())
    scale: Number = 0
    for term in terms:
        term_scale = _coefficient_magnitude(getattr(term, "coefficient", 0.0))
        if term_scale is None:
            return float("inf")
        for variable, exponent in getattr(term, "variables", {}).items():
            if variable not in values:
                raise ValueError(f"Missing value for variable(s): {variable.name}")
            coordinate_magnitude = abs(values[variable])
            if not np.isfinite(coordinate_magnitude):
                return float("inf")
            try:
                with np.errstate(over="raise", invalid="raise"):
                    coordinate_factor = coordinate_magnitude ** int(exponent)
                    if coordinate_factor != 1.0:
                        term_scale *= coordinate_factor
            except (OverflowError, FloatingPointError):
                return float("inf")
            if not _is_finite_magnitude(term_scale):
                return float("inf")
        try:
            scale += term_scale
        except OverflowError:
            return float("inf")
        if not _is_finite_magnitude(scale):
            return float("inf")
    return _float_or_inf(scale if scale > 0.0 else 1.0)


def _scaled_polynomial_evaluation_scale(
    polynomial: Any,
    values: Dict[Variable, complex],
    coefficient_scale: Number,
) -> float:
    scale: Number = 0
    for term in getattr(polynomial, "terms", ()):
        term_scale = _scaled_term_evaluation_magnitude(
            term,
            values,
            coefficient_scale,
        )
        if not np.isfinite(term_scale):
            return float("inf")
        try:
            scale += term_scale
        except OverflowError:
            return float("inf")
        if not _is_finite_magnitude(scale):
            return float("inf")
    return _float_or_inf(scale if scale > 0.0 else 1.0)


def _evaluate_scaled_polynomial(
    polynomial: Any,
    values: Dict[Variable, complex],
    scale: Number,
) -> complex:
    total = 0.0 + 0.0j
    for term in getattr(polynomial, "terms", ()):
        total += _evaluate_scaled_term(term, values, scale)
    return total


def _evaluate_scaled_term(
    term: Any,
    values: Dict[Variable, complex],
    scale: Number,
) -> complex:
    coefficient = getattr(term, "coefficient", 0.0)
    variables = getattr(term, "variables", {})
    try:
        term_value = _safe_scaled_coefficient(
            coefficient,
            scale,
        )
        scaled_coefficient = term_value
        with np.errstate(over="raise", invalid="raise"):
            for variable, exponent in variables.items():
                if variable not in values:
                    raise ValueError(f"Missing value for variable(s): {variable.name}")
                term_value *= values[variable] ** int(exponent)
        term_value = complex(term_value)
    except (OverflowError, FloatingPointError):
        return _evaluate_scaled_term_log_domain(term, values, scale)

    if not np.isfinite(term_value.real) or not np.isfinite(term_value.imag):
        return _evaluate_scaled_term_log_domain(term, values, scale)
    if scaled_coefficient == 0 and coefficient != 0 and variables:
        return _evaluate_scaled_term_log_domain(term, values, scale)
    return term_value


def _scaled_term_evaluation_magnitude(
    term: Any,
    values: Dict[Variable, complex],
    scale: Number,
) -> float:
    coefficient = getattr(term, "coefficient", 0.0)
    variables = getattr(term, "variables", {})
    try:
        scaled_coefficient = _safe_scaled_coefficient(coefficient, scale)
        term_scale = _coefficient_magnitude(scaled_coefficient)
        if term_scale is None:
            return float("inf")
        with np.errstate(over="raise", invalid="raise"):
            for variable, exponent in variables.items():
                if variable not in values:
                    raise ValueError(f"Missing value for variable(s): {variable.name}")
                coordinate_magnitude = abs(values[variable])
                if not np.isfinite(coordinate_magnitude):
                    return float("inf")
                coordinate_factor = coordinate_magnitude ** int(exponent)
                if coordinate_factor != 1.0:
                    term_scale *= coordinate_factor
        if not _is_finite_magnitude(term_scale):
            return _scaled_term_evaluation_magnitude_log_domain(term, values, scale)
    except (OverflowError, FloatingPointError):
        return _scaled_term_evaluation_magnitude_log_domain(term, values, scale)

    if term_scale == 0 and coefficient != 0 and variables:
        return _scaled_term_evaluation_magnitude_log_domain(term, values, scale)
    return _float_or_inf(term_scale)


def _scaled_term_evaluation_magnitude_log_domain(
    term: Any,
    values: Dict[Variable, complex],
    scale: Number,
) -> float:
    log_abs, _phase = _scaled_term_log_abs_and_phase(term, values, scale)
    if math.isnan(log_abs):
        return float("inf")
    if log_abs == float("-inf"):
        return 0.0
    if log_abs > math.log(np.finfo(float).max):
        return float("inf")
    if log_abs < math.log(np.nextafter(0.0, 1.0)):
        return 0.0
    return math.exp(log_abs)


def _evaluate_scaled_term_log_domain(
    term: Any,
    values: Dict[Variable, complex],
    scale: Number,
) -> complex:
    log_abs, phase = _scaled_term_log_abs_and_phase(term, values, scale)
    if log_abs == float("-inf"):
        return 0.0 + 0.0j
    if math.isnan(log_abs):
        return complex(float("nan"), 0.0)
    if log_abs > math.log(np.finfo(float).max):
        return complex(float("inf"), 0.0)
    if log_abs < math.log(np.nextafter(0.0, 1.0)):
        return 0.0 + 0.0j
    magnitude = math.exp(log_abs)
    return complex(magnitude * math.cos(phase), magnitude * math.sin(phase))


def _scaled_term_log_abs_and_phase(
    term: Any,
    values: Dict[Variable, complex],
    scale: Number,
) -> Tuple[float, float]:
    coefficient = getattr(term, "coefficient", 0.0)
    log_abs, phase = _log_abs_and_phase(coefficient)
    if log_abs == float("-inf"):
        return log_abs, phase
    scale_log_abs = _positive_log_abs(scale)
    if not np.isfinite(scale_log_abs):
        return float("-inf"), 0.0
    log_abs -= scale_log_abs

    for variable, exponent in getattr(term, "variables", {}).items():
        if variable not in values:
            raise ValueError(f"Missing value for variable(s): {variable.name}")
        exponent = int(exponent)
        if exponent == 0:
            continue
        coordinate = complex(values[variable])
        if coordinate == 0:
            return float("-inf"), 0.0
        if not np.isfinite(coordinate.real) or not np.isfinite(coordinate.imag):
            return float("inf"), 0.0
        magnitude = abs(coordinate)
        if magnitude == 0:
            return float("-inf"), 0.0
        log_abs += exponent * math.log(magnitude)
        phase += exponent * math.atan2(coordinate.imag, coordinate.real)
    return log_abs, phase


def _log_abs_and_phase(value: Any) -> Tuple[float, float]:
    if isinstance(value, Real):
        if value == 0:
            return float("-inf"), 0.0
        phase = math.pi if value < 0 else 0.0
        return _positive_log_abs(abs(value)), phase
    numeric_value = complex(value)
    if numeric_value == 0:
        return float("-inf"), 0.0
    if not np.isfinite(numeric_value.real) or not np.isfinite(numeric_value.imag):
        return float("inf"), 0.0
    return math.log(abs(numeric_value)), math.atan2(
        numeric_value.imag,
        numeric_value.real,
    )


def _positive_log_abs(value: Any) -> float:
    if value == 0:
        return float("-inf")
    if isinstance(value, Integral):
        return math.log(abs(int(value)))
    try:
        magnitude = abs(value)
    except (TypeError, ValueError, OverflowError):
        return float("inf")
    if isinstance(magnitude, Integral):
        return math.log(abs(int(magnitude)))
    try:
        return math.log(float(magnitude))
    except (TypeError, ValueError, OverflowError):
        return float("inf")


def _safe_scaled_coefficient(coefficient: Any, scale: Number) -> Number:
    try:
        return coefficient / scale
    except OverflowError:
        coefficient_magnitude = _coefficient_magnitude(coefficient)
        scale_magnitude = _coefficient_magnitude(scale)
        if (
            coefficient_magnitude is not None
            and scale_magnitude is not None
            and coefficient_magnitude < scale_magnitude
        ):
            return 0.0 + 0.0j if isinstance(coefficient, complex) else 0.0
        raise


def _evaluate_polynomial_as_complex(
    polynomial: Any,
    values: Dict[Variable, complex],
) -> complex:
    try:
        with np.errstate(over="raise", invalid="raise"):
            value = polynomial.evaluate(values)
    except (OverflowError, FloatingPointError):
        exact_zero = _scaled_evaluation_is_exact_zero(polynomial, values)
        if exact_zero:
            return 0.0 + 0.0j
        return complex(float("inf"), 0.0)
    value = _complex_or_infinite(value)
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        exact_zero = _scaled_evaluation_is_exact_zero(polynomial, values)
        if exact_zero:
            return 0.0 + 0.0j
    return value


def _scaled_evaluation_is_exact_zero(
    polynomial: Any,
    values: Dict[Variable, complex],
) -> bool:
    try:
        exact_real_zero = _exact_real_polynomial_is_zero(polynomial, values)
        if exact_real_zero is not None:
            return exact_real_zero
        scale = _polynomial_coefficient_scale(polynomial)
        if _scaled_evaluation_loses_nonzero_term(polynomial, values, scale):
            return False
        scaled_value = _evaluate_scaled_polynomial(polynomial, values, scale)
    except (OverflowError, FloatingPointError, TypeError, ValueError):
        return False
    return scaled_value == 0


def _exact_real_polynomial_is_zero(
    polynomial: Any,
    values: Dict[Variable, complex],
) -> Optional[bool]:
    total = Fraction(0, 1)
    for term in getattr(polynomial, "terms", ()):
        coefficient = _exact_real_fraction(getattr(term, "coefficient", 0.0))
        if coefficient is None:
            return None
        term_value = coefficient
        for variable, exponent in getattr(term, "variables", {}).items():
            if variable not in values:
                raise ValueError(f"Missing value for variable(s): {variable.name}")
            coordinate = _exact_real_fraction(values[variable])
            if coordinate is None:
                return None
            term_value *= coordinate ** int(exponent)
        total += term_value
    return total == 0


def _exact_real_fraction(value: Any) -> Optional[Fraction]:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, Integral):
        return Fraction(int(value), 1)
    if isinstance(value, Rational):
        return Fraction(value.numerator, value.denominator)
    if isinstance(value, Real):
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            return None
        return Fraction.from_float(numeric_value)
    try:
        numeric_value = complex(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(numeric_value.real) or not np.isfinite(numeric_value.imag):
        return None
    if numeric_value.imag != 0:
        return None
    return Fraction.from_float(float(numeric_value.real))


def _scaled_evaluation_loses_nonzero_term(
    polynomial: Any,
    values: Dict[Variable, complex],
    scale: Number,
) -> bool:
    for term in getattr(polynomial, "terms", ()):
        coefficient = getattr(term, "coefficient", 0.0)
        if coefficient == 0:
            continue
        try:
            scaled_coefficient = _safe_scaled_coefficient(coefficient, scale)
        except (OverflowError, FloatingPointError):
            return True
        if scaled_coefficient != 0:
            continue
        if _term_is_zero_from_coordinates(term, values):
            continue
        return True
    return False


def _term_is_zero_from_coordinates(
    term: Any,
    values: Dict[Variable, complex],
) -> bool:
    for variable, exponent in getattr(term, "variables", {}).items():
        if variable not in values:
            raise ValueError(f"Missing value for variable(s): {variable.name}")
        if int(exponent) > 0 and values[variable] == 0:
            return True
    return False


def _complex_or_infinite(value: Any) -> complex:
    try:
        return complex(value)
    except OverflowError:
        if isinstance(value, Real):
            return complex(float("-inf") if value < 0 else float("inf"), 0.0)
        return complex(float("inf"), 0.0)


def _coefficient_magnitude(value: Any) -> Optional[Number]:
    try:
        magnitude = abs(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(magnitude, Integral):
        return int(magnitude)
    if _is_finite_magnitude(magnitude):
        return float(magnitude)
    return None


def _is_finite_magnitude(value: Any) -> bool:
    if isinstance(value, Integral):
        return True
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError):
        return np.isfinite(_float_or_inf(value))


def _float_or_inf(value: Any) -> float:
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return float("inf")
    return result if np.isfinite(result) else float("inf")

def newton_corrector(system: PolynomialSystem,
                    point: np.ndarray,
                    variables: List[Variable],
                    max_iters: int = 10,
                    tol: float = 1e-10) -> Tuple[np.ndarray, bool, int]:
    """Apply Newton's method to correct a point to a solution.
    
    Args:
        system: The polynomial system
        point: Initial point for correction
        variables: The variables in the system
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence
        
    Returns:
        Tuple of (corrected point, success flag, number of iterations)
    """
    _validate_polynomial_system("system", system)
    variables = _normalize_variables(variables)
    max_iters = _validate_nonnegative_integer("max_iters", max_iters)
    tol = _validate_positive_finite_float("tol", tol)
    current = _coerce_numeric_vector(point, "point", len(variables))
    
    for i in range(max_iters):
        f_val, jac, residual, use_scaled = _newton_system_data(
            system,
            current,
            variables,
        )
        
        # Check if we're already at a solution
        if residual < tol:
            return current, True, i
        
        # Solve the linear system J * delta = -f
        delta = solve_linear_system(jac, -f_val)
        delta_norm = _scaled_euclidean_norm(delta)
        if not np.isfinite(delta_norm):
            return current, False, i + 1

        # A tiny Newton step alone can occur at a singular non-root, so require
        # the current residual to be small too.
        if delta_norm < tol:
            return current, False, i + 1

        def residual_at(candidate: np.ndarray) -> float:
            if use_scaled:
                return _norm_or_inf(
                    evaluate_scaled_system_at_point(system, candidate, variables)
                )
            raw_residual = _norm_or_inf(
                evaluate_system_at_point(system, candidate, variables)
            )
            if np.isfinite(raw_residual):
                return raw_residual
            return _norm_or_inf(
                evaluate_scaled_system_at_point(system, candidate, variables)
            )
        
        next_point, next_residual, accepted, step_scale = _damped_update(
            current,
            delta,
            residual,
            residual_at,
        )
        if not accepted:
            return current, False, i + 1

        current = next_point
        if next_residual < tol:
            return current, True, i + 1
        
        if step_scale * delta_norm < tol:
            return current, False, i + 1
    
    # If we got here, we didn't converge
    return current, False, max_iters


def _newton_system_data(
    system: PolynomialSystem,
    point: np.ndarray,
    variables: List[Variable],
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    f_val = evaluate_system_at_point(system, point, variables)
    residual = _norm_or_inf(f_val)
    jac = evaluate_jacobian_at_point(system, point, variables)
    if np.isfinite(residual) and np.all(np.isfinite(jac)):
        return f_val, jac, residual, False

    scaled_f_val = evaluate_scaled_system_at_point(system, point, variables)
    scaled_residual = _norm_or_inf(scaled_f_val)
    scaled_jac = evaluate_equation_scaled_jacobian_at_point(
        system,
        point,
        variables,
    )
    return scaled_f_val, scaled_jac, scaled_residual, True


def solve_linear_system(jac: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a linear system robustly with least-squares fallback.

    Args:
        jac: Jacobian matrix (m x n), typically square
        rhs: Right-hand side vector (m,)

    Returns:
        Solution vector x minimizing ||Jx - rhs||.
    """
    jacobian = _coerce_complex_array(jac, "jac").copy()
    right_hand_side = _coerce_complex_array(rhs, "rhs").copy()
    failure = _linear_solve_failure_vector(jacobian, right_hand_side)
    if (
        jacobian.ndim != 2
        or right_hand_side.ndim != 1
        or jacobian.shape[0] != right_hand_side.shape[0]
        or not np.all(np.isfinite(jacobian))
        or not np.all(np.isfinite(right_hand_side))
    ):
        return failure

    candidates: List[np.ndarray] = []
    if jacobian.shape[0] == jacobian.shape[1]:
        try:
            result = np.linalg.solve(jacobian, right_hand_side)
            if np.all(np.isfinite(result)):
                candidates.append(np.asarray(result, dtype=complex))
        except (np.linalg.LinAlgError, OverflowError, FloatingPointError, ValueError):
            pass

    # Fallback to least squares
    try:
        result = np.linalg.lstsq(jacobian, right_hand_side, rcond=None)[0]
        if np.all(np.isfinite(result)):
            candidates.append(np.asarray(result, dtype=complex))
    except (np.linalg.LinAlgError, OverflowError, FloatingPointError, ValueError):
        pass

    balanced = _solve_balanced_linear_system(jacobian, right_hand_side)
    if np.all(np.isfinite(balanced)):
        candidates.append(np.asarray(balanced, dtype=complex))
    return _select_best_linear_solution(
        candidates,
        jacobian,
        right_hand_side,
        failure,
    )


def _solve_balanced_linear_system(
    jacobian: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Solve after simple row and column equilibration."""
    failure = _linear_solve_failure_vector(jacobian, right_hand_side)
    if (
        jacobian.ndim != 2
        or right_hand_side.ndim != 1
        or jacobian.shape[0] != right_hand_side.shape[0]
        or not np.all(np.isfinite(jacobian))
        or not np.all(np.isfinite(right_hand_side))
    ):
        return failure

    row_scales = np.maximum(
        np.max(np.abs(jacobian), axis=1),
        np.abs(right_hand_side),
    )
    row_scales = np.where(row_scales > 0, row_scales, 1.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        row_scaled_matrix = jacobian / row_scales[:, None]
        row_scaled_rhs = right_hand_side / row_scales
    if (
        not np.all(np.isfinite(row_scaled_matrix))
        or not np.all(np.isfinite(row_scaled_rhs))
    ):
        return failure

    column_scales = np.max(np.abs(row_scaled_matrix), axis=0)
    column_scales = np.where(column_scales > 0, column_scales, 1.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        balanced_matrix = row_scaled_matrix / column_scales
    if not np.all(np.isfinite(balanced_matrix)):
        return failure

    try:
        if balanced_matrix.shape[0] == balanced_matrix.shape[1]:
            balanced_solution = np.linalg.solve(
                balanced_matrix,
                row_scaled_rhs,
            )
            if np.all(np.isfinite(balanced_solution)):
                return balanced_solution / column_scales
    except (np.linalg.LinAlgError, OverflowError, FloatingPointError, ValueError):
        pass

    try:
        balanced_solution = np.linalg.lstsq(
            balanced_matrix,
            row_scaled_rhs,
            rcond=None,
        )[0]
        if np.all(np.isfinite(balanced_solution)):
            return balanced_solution / column_scales
    except (np.linalg.LinAlgError, OverflowError, FloatingPointError, ValueError):
        pass
    return failure


def _select_best_linear_solution(
    candidates: List[np.ndarray],
    jacobian: np.ndarray,
    right_hand_side: np.ndarray,
    failure: np.ndarray,
) -> np.ndarray:
    best_candidate = None
    best_quality = (float("inf"), float("inf"))
    expected_size = jacobian.shape[1] if jacobian.ndim == 2 else failure.size
    for candidate in candidates:
        if candidate.ndim != 1 or candidate.size != expected_size:
            continue
        residual_norm = _linear_solution_residual_norm(
            jacobian,
            right_hand_side,
            candidate,
        )
        if not np.isfinite(residual_norm):
            continue
        quality = (residual_norm, _scaled_euclidean_norm(candidate))
        if quality < best_quality:
            best_quality = quality
            best_candidate = candidate
    return failure if best_candidate is None else best_candidate


def _linear_solution_residual_norm(
    jacobian: np.ndarray,
    right_hand_side: np.ndarray,
    candidate: np.ndarray,
) -> float:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            residual = jacobian @ candidate - right_hand_side
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return float("inf")
    return _scaled_euclidean_norm(residual)


def _linear_solve_failure_vector(
    jacobian: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Return a shape-compatible nonfinite vector for failed linear solves."""
    if jacobian.ndim == 2:
        columns = jacobian.shape[1]
    elif right_hand_side.ndim == 1:
        columns = right_hand_side.size
    else:
        columns = 0
    return np.full(columns, np.nan + 0j, dtype=complex)


def newton_corrector_numeric(
    f: Callable[[np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    max_iters: int = 10,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, bool, int]:
    """Newton's method for numeric function and Jacobian callables.

    Args:
        f: Function mapping x -> f(x)
        jac: Function mapping x -> J(x)
        point: Initial guess
        max_iters: Maximum iterations
        tol: Convergence tolerance on step and residual

    Returns:
        (x, success, iters)
    """
    f = _validate_callable("f", f)
    jac = _validate_callable("jac", jac)
    max_iters = _validate_nonnegative_integer("max_iters", max_iters)
    tol = _validate_positive_finite_float("tol", tol)
    current = _coerce_numeric_vector(point, "point")
    for i in range(max_iters):
        f_val = _coerce_complex_array(f(current), "f(point)")
        residual = _norm_or_inf(f_val)
        if residual < tol:
            return current, True, i
        J = _coerce_complex_array(jac(current), "jac(point)")
        delta = solve_linear_system(J, -f_val)
        delta_norm = _scaled_euclidean_norm(delta)
        if not np.isfinite(delta_norm):
            return current, False, i + 1
        if delta_norm < tol:
            return current, False, i + 1

        next_point, next_residual, accepted, step_scale = _damped_update(
            current,
            delta,
            residual,
            lambda candidate: _norm_or_inf(
                _coerce_complex_array(f(candidate), "f(point)")
            ),
        )
        if not accepted:
            return current, False, i + 1

        current = next_point
        if next_residual < tol:
            return current, True, i + 1
        if step_scale * delta_norm < tol:
            return current, False, i + 1
    return current, False, max_iters


def _damped_update(
    current: np.ndarray,
    delta: np.ndarray,
    current_residual: float,
    residual_at: Callable[[np.ndarray], float],
    *,
    max_backtracks: int = 12,
) -> Tuple[np.ndarray, float, bool, float]:
    """Return a residual-decreasing Newton update, backing off if needed."""
    step_scale = 1.0
    for _ in range(max_backtracks + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = current + step_scale * delta
        if not np.all(np.isfinite(candidate)):
            candidate_residual = float("inf")
        else:
            try:
                candidate_residual = residual_at(candidate)
            except (TypeError, ValueError, OverflowError, FloatingPointError):
                candidate_residual = float("inf")
        if candidate_residual < current_residual:
            return candidate, candidate_residual, True, step_scale
        step_scale *= 0.5
    return current, current_residual, False, 0.0


def _norm_or_inf(values: np.ndarray) -> float:
    return _scaled_euclidean_norm(values)


def _scaled_euclidean_norm(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=complex)
    if values.size == 0:
        return 0.0
    if not np.all(np.isfinite(values)):
        return float("inf")
    magnitudes = np.abs(values)
    scale = float(np.max(magnitudes))
    if scale == 0.0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_values = values / scale
    scaled_norm = _safe_vector_norm(scaled_values)
    if not np.isfinite(scaled_norm):
        scaled_norm = _fallback_vector_norm(scaled_values)
    with np.errstate(over="ignore", invalid="ignore"):
        result = scale * scaled_norm
    return result if np.isfinite(result) else float("inf")


def _safe_vector_norm(values: np.ndarray) -> float:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(np.linalg.norm(values))
    except (
        np.linalg.LinAlgError,
        ValueError,
        OverflowError,
        FloatingPointError,
        TypeError,
    ):
        return float("inf")
    return result if np.isfinite(result) else float("inf")


def _fallback_vector_norm(values: np.ndarray) -> float:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            magnitudes = np.abs(values)
            total = float(np.sum(magnitudes * magnitudes))
    except (ValueError, OverflowError, FloatingPointError, TypeError):
        return float("inf")
    if total < 0.0 or not np.isfinite(total):
        return float("inf")
    return math.sqrt(total)


def evaluate_jacobian_polynomials(jac_polys: List[List[Any]], var_dict: Dict[Variable, complex]) -> np.ndarray:
    """Evaluate a Jacobian represented as polynomials at a point.

    Args:
        jac_polys: List of rows, each a list of Polynomial
        var_dict: Mapping variable -> value

    Returns:
        Numeric Jacobian matrix.
    """
    rows: List[List[complex]] = []
    for row in jac_polys:
        rows.append([
            _evaluate_polynomial_as_complex(poly, var_dict)
            for poly in row
        ])
    return np.array(rows, dtype=complex)
