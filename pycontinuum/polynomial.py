"""
Polynomial representation module for PyContinuum.

This module provides classes and functions for representing and manipulating
multivariate polynomials and polynomial systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Set, Any, Optional


class Variable:
    """Representation of a polynomial variable."""
    
    def __init__(self, name: str):
        """Initialize a variable with a name.
        
        Args:
            name: String name of the variable
        """
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
    
    def __pow__(self, exponent: int) -> "Polynomial":
        """Raise the variable to a power, creating a polynomial."""
        return Polynomial([Monomial({self: exponent})])
    
    def __mul__(self, other: Any) -> Union["Polynomial", "Monomial"]:
        """Multiply the variable by another object."""
        if isinstance(other, (int, float, complex)):
            return Monomial({self: 1}, coefficient=other)
        elif isinstance(other, Variable):
            return Monomial({self: 1, other: 1})
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
    
    def __rmul__(self, other: Any) -> "Monomial":
        """Handle multiplication when the variable is on the right."""
        if isinstance(other, (int, float, complex)):
            return Monomial({self: 1}, coefficient=other)
        return NotImplemented
    
    def __add__(self, other: Any) -> "Polynomial":
        """Add the variable to another object."""
        if isinstance(other, (int, float, complex)):
            return Polynomial([Monomial({self: 1}), Monomial({}, coefficient=other)])
        elif isinstance(other, Variable):
            return Polynomial([Monomial({self: 1}), Monomial({other: 1})])
        elif isinstance(other, (Monomial, Polynomial)):
            return Polynomial([Monomial({self: 1})]) + other
        else:
            return NotImplemented
    
    def __radd__(self, other: Any) -> "Polynomial":
        """Handle addition when the variable is on the right."""
        if isinstance(other, (int, float, complex)):
            return Polynomial([Monomial({self: 1}), Monomial({}, coefficient=other)])
        return NotImplemented
    
    def __sub__(self, other: Any) -> "Polynomial":
        """Subtract another object from the variable."""
        if isinstance(other, (int, float, complex)):
            return Polynomial([Monomial({self: 1}), Monomial({}, coefficient=-other)])
        elif isinstance(other, Variable):
            return Polynomial([Monomial({self: 1}), Monomial({other: 1}, coefficient=-1)])
        elif isinstance(other, (Monomial, Polynomial)):
            return Polynomial([Monomial({self: 1})]) - other
        else:
            return NotImplemented
    
    def __rsub__(self, other: Any) -> "Polynomial":
        """Handle subtraction when the variable is on the right."""
        if isinstance(other, (int, float, complex)):
            return Polynomial([Monomial({self: 1}, coefficient=-1), Monomial({}, coefficient=other)])
        return NotImplemented


class Monomial:
    """Representation of a monomial (term in a polynomial)."""
    
    def __init__(self, variables: Dict[Variable, int], coefficient: complex = 1):
        """Initialize a monomial with variables and their exponents.
        
        Args:
            variables: Dict mapping Variable objects to their exponents
            coefficient: Coefficient of the monomial (default: 1)
        """
        # Filter out zero exponents
        self.variables = {var: exp for var, exp in variables.items() if exp != 0}
        self.coefficient = coefficient
        
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
        for var, exp in self.variables.items():
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
        result = self.coefficient
        for var, exp in self.variables.items():
            result *= values.get(var, 0) ** exp
        return result
    
    def __mul__(self, other: Any) -> Union["Polynomial", "Monomial"]:
        """Multiply the monomial by another object."""
        if isinstance(other, (int, float, complex)):
            return Monomial(self.variables.copy(), coefficient=self.coefficient * other)
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
            return Monomial(new_vars, coefficient=self.coefficient * other.coefficient)
        elif isinstance(other, Polynomial):
            return other * self
        else:
            return NotImplemented
    
    def __rmul__(self, other: Any) -> "Monomial":
        """Handle multiplication when the monomial is on the right."""
        if isinstance(other, (int, float, complex)):
            return Monomial(self.variables.copy(), coefficient=self.coefficient * other)
        return NotImplemented
    
    def __add__(self, other: Any) -> "Polynomial":
        """Add the monomial to another object."""
        if isinstance(other, (int, float, complex)):
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
        if isinstance(other, (int, float, complex)):
            return Polynomial([self, Monomial({}, coefficient=other)])
        return NotImplemented
    
    def __sub__(self, other: Any) -> "Polynomial":
        """Subtract another object from the monomial."""
        if isinstance(other, (int, float, complex)):
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
        if isinstance(other, (int, float, complex)):
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
        if var not in self.variables:
            return Monomial({}, coefficient=0)
        
        exp = self.variables[var]
        new_vars = self.variables.copy()
        new_coef = self.coefficient * exp
        
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


class Polynomial:
    """Representation of a multivariate polynomial."""
    
    def __init__(self, terms: List[Union[Monomial, Variable, int, float, complex]]):
        """Initialize a polynomial from a list of terms."""
        processed_terms = []
        for term in terms:
            if isinstance(term, Monomial):
                processed_terms.append(term)
            elif isinstance(term, Variable):
                processed_terms.append(Monomial({term: 1}))
            elif isinstance(term, (int, float, complex)):
                processed_terms.append(Monomial({}, coefficient=complex(term)))
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
                term_dict[var_key] += term.coefficient
            else:
                term_dict[var_key] = term.coefficient

        # Create new Monomial objects from the combined terms, filtering out zero coefficients
        combined_terms = []
        for var_key, coef in term_dict.items():
            if abs(coef) > 1e-15:  # Use a small tolerance for floating point zeros
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
            if i == 0:
                term_strs.append(str(term))
            else:
                if term.coefficient.real > 0:
                    # + always gets a space after it
                    term_strs.append(f"+ {term}")
                else:
                    # negative: strip the leading "-" from str(term) and
                    # then prepend "- " so we get " - foo" instead of "-foo"
                    s = str(term)
                    assert s.startswith("-"), "unexpected repr for negative term"
                    term_strs.append(f"- {s[1:]}")
        
        return " ".join(term_strs)
    
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
        return sum(term.evaluate(values) for term in self.terms)
    
    def __add__(self, other: Any) -> "Polynomial":
        """Add another object to the polynomial."""
        if isinstance(other, (int, float, complex)):
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
        if isinstance(other, (int, float, complex)):
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
        if isinstance(other, (int, float, complex)):
            return Polynomial([Monomial({}, coefficient=other)] + 
                            [Monomial(term.variables, coefficient=-term.coefficient) for term in self.terms])
        return NotImplemented
    
    def __mul__(self, other: Any) -> "Polynomial":
        """Multiply the polynomial by another object."""
        if isinstance(other, (int, float, complex)):
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
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Exponent must be a non-negative integer")
        
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
        return Polynomial([term.partial_derivative(var) for term in self.terms])
    
    def jacobian(self, vars_list: List[Variable]) -> List[List["Polynomial"]]:
        """Compute the Jacobian matrix of partial derivatives.
        
        Args:
            vars_list: List of variables for the Jacobian
            
        Returns:
            Jacobian matrix as a list of lists of polynomials
        """
        return [[self.partial_derivative(var) for var in vars_list]]


class PolynomialSystem:
    """Representation of a system of polynomial equations."""
    
    def __init__(self, equations: List[Union[Polynomial, Monomial, Variable]]):
        """Initialize a polynomial system from a list of equations."""
        self.equations = []
        for eq in equations:
            if isinstance(eq, Polynomial):
                self.equations.append(eq)
            elif isinstance(eq, (Monomial, Variable)):
                # Convert Monomial or Variable to Polynomial
                self.equations.append(Polynomial([eq]))
            else:
                raise TypeError(f"Unsupported equation type: {type(eq)}")
    
    def __repr__(self) -> str:
        return "\n".join([f"{i}: {eq}" for i, eq in enumerate(self.equations)])
    
    def variables(self) -> Set[Variable]:
        """Get the set of all variables in the system."""
        vars_set = set()
        for eq in self.equations:
            vars_set.update(eq.variables())
        return vars_set
    
    def evaluate(self, values: Dict[Variable, complex]) -> List[complex]:
        """Evaluate the system at specific variable values.
        
        Args:
            values: Dict mapping variables to their values
        Returns:
            List of evaluated values for each equation
        """
        return [eq.evaluate(values) for eq in self.equations]
    
    def jacobian(self, vars_list: List[Variable]) -> List[List[Polynomial]]:
        """Compute the Jacobian matrix of partial derivatives.
        
        Args:
            vars_list: List of variables for the Jacobian
            
        Returns:
            Jacobian matrix as a list of lists of polynomials
        """
        return [eq.jacobian(vars_list)[0] for eq in self.equations]
    
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
    variables = tuple(Variable(name) for name in names)
    return variables[0] if len(variables) == 1 else variables


def make_system(*equations) -> "PolynomialSystem":
    """Create a polynomial system from various types of equations.
    
    Args:
        *equations: Polynomial equations to include in the system
        
    Returns:
        A PolynomialSystem object containing the processed equations
        
    Raises:
        TypeError: If an equation cannot be converted to a Polynomial
    """
    processed_equations = []
    for eq in equations:
        if isinstance(eq, Polynomial):
            processed_equations.append(eq)
        elif isinstance(eq, (Monomial, Variable)):
            processed_equations.append(Polynomial([eq]))
        else:
            try:
                # Try to convert to Polynomial if it's not already one
                processed_equations.append(Polynomial([eq]))
            except TypeError:
                raise TypeError(f"Cannot convert {type(eq)} to polynomial equation")
    
    return PolynomialSystem(processed_equations)
