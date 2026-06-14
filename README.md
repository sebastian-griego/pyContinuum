# PyContinuum

A pure Python library for polynomial homotopy continuation methods.


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

PyContinuum is a modern, pip-installable, user-friendly Python library for solving polynomial systems using numerical homotopy continuation methods. It provides a clean, Pythonic interface for computing all solutions to systems of polynomial equations without requiring external compiled binaries.

**Key Features** (planned):

- **Clean Python API**: Define polynomials and solve systems with an intuitive interface
- **Core Homotopy Methods**: Robust implementation of predictor-corrector path tracking
- **Start System Generation**: Automated total-degree homotopy and custom start systems
- **Solution Processing**: Classification, refinement, and filtering of solutions
- **Visualization**: Tools to visualize path tracking and solution sets
- **No External Dependencies**: Pure Python implementation with optional accelerators
- **Educational**: Clear documentation and examples for teaching and learning

## Quick Example

```python
from pycontinuum import polyvar, PolynomialSystem, solve

# Define variables
x, y = polyvar('x', 'y')

# Define polynomial system
f1 = x**2 + y**2 - 1      # circle
f2 = x**2 - y             # parabola
system = PolynomialSystem([f1, f2])

# Solve the system
solutions = solve(system)

# Display solutions
print(solutions)
```

## Installation

```bash
pip install pycontinuum
pip install pycontinuum[viz]
pip install pycontinuum[monodromy]
```

## Reproducible Solves

Random generic start systems are part of homotopy continuation. Pass
`random_state` when you need reproducible start systems and solution ordering:

```python
solutions = solve(system, random_state=123)
```

## Solution Diagnostics

Use the diagnostics layer to turn numerical roots into an auditable report with
residual norms, Jacobian rank, condition numbers, real-solution counts, and
duplicate clusters:

```python
from pycontinuum import diagnose_solutions

solutions = solve(system, random_state=123)
audit = diagnose_solutions(solutions, tolerance=1e-8)

print(audit.summary())
assert audit.all_valid
```

You can also call `solutions.diagnostics(tolerance=1e-8)` directly on a
`SolutionSet`.

## Publishing (maintainers)

```bash
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
# TestPyPI
setx TWINE_USERNAME __token__
setx TWINE_PASSWORD pypi-XXXXX
python -m twine upload --repository testpypi dist/*
# PyPI
python -m twine upload dist/*
```


## Status

This project is under active development. Current focus is on implementing the core functionality for the v1.0 release.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

PyContinuum draws inspiration from existing software in numerical algebraic geometry, including:
- PHCpack and phcpy
- Bertini and PyBertini
- HomotopyContinuation.jl

