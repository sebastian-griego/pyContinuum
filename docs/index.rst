Welcome to PyContinuum's documentation!
====================================

PyContinuum is a pure Python library for solving polynomial systems using numerical homotopy continuation methods.

.. image:: https://img.shields.io/pypi/v/pycontinuum.svg
   :target: https://pypi.org/project/pycontinuum/
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/pycontinuum/badge/?version=latest
   :target: https://pycontinuum.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Overview
--------

PyContinuum is a modern, pip-installable, user-friendly Python library for solving systems of polynomial equations. It implements the homotopy continuation method, which is a powerful numerical technique that can reliably find all complex solutions to a polynomial system.

Key Features
~~~~~~~~~~~

- **Clean Python API**: Define polynomials and solve systems with an intuitive interface
- **Core Homotopy Methods**: Robust implementation of predictor-corrector path tracking
- **Start System Generation**: Automated total-degree homotopy and custom start systems
- **Solution Processing**: Classification, refinement, and filtering of solutions
- **Visualization**: Tools to visualize path tracking and solution sets
- **No External Dependencies**: Pure Python implementation with optional accelerators
- **Educational**: Clear documentation and examples for teaching and learning

Quick Example
------------

.. code-block:: python

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

Installation
-----------

.. code-block:: bash

   pip install pycontinuum

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/installation
   guide/getting_started
   guide/theory
   guide/examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/polynomial
   api/solver
   api/tracking
   api/start_systems
   api/visualization

.. toctree::
   :maxdepth: 2
   :caption: Development

   dev/contributing
   dev/roadmap
   dev/changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`