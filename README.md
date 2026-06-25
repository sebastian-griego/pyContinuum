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

The monodromy extra exposes positive-dimensional component tools without
importing them into the default namespace:

```python
from pycontinuum.monodromy import MonodromyBreakup, trace_monodromy_loops
```

## Reproducible Solves

Random generic start systems and randomized gamma choices are part of homotopy
continuation. Pass `random_state` when you need reproducible paths, start
systems, and solution ordering:

```python
solutions = solve(system, random_state=123)
```

When passing an explicit `variables=[...]` order, entries must be
`Variable` objects and the list must contain each variable used by the system
exactly once. Missing, duplicate, or malformed variables are rejected so the
solver cannot silently solve a different polynomial system.
Variable names must be ASCII identifiers matching
`[A-Za-z_][A-Za-z0-9_]*`, which keeps printed polynomials, parser input, and
optional symbolic factorization consistent.
`Polynomial.parse()` accepts both `^` and Python-style `**` for exponents, with
the same exact non-negative integer exponent checks as native polynomial
construction. It also supports parenthesized polynomial expressions such as
`(x + 1)*(y - 2)` and implicit multiplication around parentheses. Numeric
imaginary literals such as `2j*x` are accepted as complex coefficients; a plain
`j` remains a normal variable name.
Use `PolynomialSystem.parse("x^2 + y^2 = 1; x^2 = y")` to build a whole system
from semicolon- or newline-separated equations; equations with `=` are stored
as left-minus-right polynomials. The high-level `solve()` function accepts the
same system strings directly, so `solve("x^2 = 1")` is valid for quick
experiments. A single `Polynomial`, `Monomial`, `Variable`, or numeric constant
can also be passed directly as a one-equation system, for example
`solve(x**2 - 1)`.
Direct polynomial and system evaluation helpers accept coordinate vectors,
`Solution` objects, or mappings keyed by `Variable` objects or variable-name
strings. They require values for every variable in the explicit variable order;
incomplete coordinate records are rejected instead of treating missing
variables as zero.
`PolynomialSystem` accepts numeric constants directly, so constant constraints
can be written as `PolynomialSystem([x - 1, 0])` or `PolynomialSystem([1])`
without manually wrapping them as `Polynomial([1])`.

## Custom Start Systems

Advanced users can pass a custom `start_system` with matching
`start_solutions`. They must be provided together. The start system must use the
chosen variable list, have the same number of equations as the tracked target
system, and each finite start point must have one coordinate per variable and
satisfy the start equations within the solver tolerance budget. Valid custom
start data is recorded in `solutions._meta["start_system"]`. Like target
systems, custom start systems may be supplied as `PolynomialSystem` objects,
system strings, or iterables of equation strings. Custom start solutions may be
coordinate vectors, existing `Solution` objects, or mappings keyed by `Variable`
objects or variable-name strings.

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
Diagnostics validate numeric thresholds and variable lists before computing
residual, rank, conditioning, duplicate-cluster, or certification fields, so an
audit cannot silently use a malformed tolerance budget.
They accept coordinate vectors, `Solution` objects, solution-like objects with
a `values` mapping, or coordinate mappings keyed by `Variable` objects or
variable-name strings.
Use `SolutionSet.from_points(points, system, variables=[...])` to import roots
from external data, arrays, or name-keyed records; it computes residual,
scaled-residual, backward-error, and singularity metadata before returning a
standard `SolutionSet`.

Use `solutions.to_array([x, y], real=True)` to move validated real roots into
NumPy workflows, `solutions.as_dicts()` / `solution.as_dict()` for individual
records, or `solutions.as_dict()` for a JSON-friendly full result with solve
metadata.
`solution.distance(other, [x, y])` accepts the same coordinate-vector and
mapping record forms for pairwise comparisons.
Use `solutions.nearest(point, [x, y], return_distance=True)` to match an
external coordinate record to the closest stored solution with the same
scaled-distance norm used internally for clustering.
Pass `strict_json=True` to solution, solution-set, diagnostics, or audit
exports when the result must satisfy `json.dumps(..., allow_nan=False)`;
nonfinite numeric evidence is preserved as `"NaN"`, `"Infinity"`, or
`"-Infinity"` strings in that mode.
Array export, diagnostics, and refinement reject solution records with missing
coordinates instead of silently treating absent variables as zero.
Coordinate mappings that provide both a `Variable` key and the matching
variable-name string key must agree; conflicting duplicate coordinates are
rejected before evaluation or tracking.
`solutions.filter()` validates boolean flags, tolerance values, raw/scaled
residual cutoffs, backward-error cutoffs, and custom predicates before applying
them. Scaled residual and backward-error filters compute missing metrics from
the solution set's system. Solution-level realness checks and real-array
conversion also reject malformed or nonfinite tolerances.

Diagnostics also report whether each candidate is an isolated full-rank root.
The `newton_step_norm`, `is_certified_regular`, and `regularity_status` fields
separate small-residual singular or positive-dimensional candidates from
regular isolated solutions. Aggregate counts are available as
`audit.isolated_count` and `audit.certified_regular_count`; use
`audit.regularity_status_counts()` to see why candidates failed the regularity
audit.

For regular candidates, diagnostics also include a local Newton error estimate.
Use `newton_error_bound`, `is_newton_certified`, and
`certification_status` to distinguish roots whose Newton error balls are
separated from nearby candidates; aggregate counts are available as
`audit.newton_certified_count` and `audit.all_newton_certified`. Use
`audit.certification_status_counts()` to summarize uncertified-candidate
failure modes.

Raw and coefficient-scaled residuals are reported separately. Use
`scaled_residual_norm`, `is_scaled_valid`, and `audit.max_scaled_residual` to
audit systems whose equations have very large or very small coefficients without
losing the raw residual norm.
Diagnostics and univariate direct solves also report a normwise
`backward_error`, measuring residual size relative to the sum of absolute term
contributions at the candidate point. This helps audit extreme-scale roots
where floating-point cancellation can make a mathematically valid root have a
large absolute residual.

Solver results also include compact path accounting in
`solutions._meta["path_summary"]`, separating tracker failures from endpoints
that tracked successfully but were rejected by the original-system residual
filter. Accepted `Solution` objects carry a compact `solution.path_info`
snapshot with path steps, residuals, polish/endgame status, and the final
non-destructive Newton polish attempted against the original system.

When multiple continuation paths converge to the same endpoint, the solver now
keeps that evidence instead of discarding it during deduplication. Each
`Solution` reports `multiplicity`, `path_indices`, and a cluster radius in
`solution.path_info["cluster"]`; aggregate counts are stored in
`solutions._meta["multiplicity_summary"]`.

## Solution Refinement

Use Newton refinement to polish loose candidates from a solve, a witness-set
workflow, or external data before auditing:

```python
refined = solutions.refine(tol=1e-12)
assert refined.diagnostics(tolerance=1e-10).all_valid
```

The same functionality is available as `refine_solution(system, solution)` and
`refine_solutions(solutions)`. Refinement returns new objects rather than
mutating the original candidates, and it keeps the original point if Newton
would fail or worsen the residual. Newton correction uses residual-decreasing
damped steps and accepts convergence only when the residual is small, not merely
because an update step is tiny.
Standalone refinement functions accept coordinate vectors, `Solution` objects,
solution-like objects with a `values` mapping, or coordinate mappings keyed by
`Variable` objects or variable-name strings.
Refinement options are checked up front: tolerances and singularity thresholds
must be positive finite numbers, iteration counts must be positive integers,
and `keep_failed` must be a boolean.

## Path Tracking Controls

The default solver uses adaptive predictor-corrector path tracking. For harder
systems, pass `tracking_options` to tune the numerical continuation without
leaving the high-level API:

```python
solutions = solve(
    system,
    max_paths=5000,
    tracking_options={
        "max_step_size": 0.02,
        "max_predictor_norm": 0.05,
        "predictor": "rk4",
        "max_newton_iters": 20,
        "n_jobs": 4,
    },
    random_state=123,
)
```

Use `max_paths` as a guardrail for large continuation jobs. Generated
total-degree homotopies are checked against this limit before start solutions
are allocated, and custom start systems are checked before tracking begins.
Recursive decomposition paths, including independent blocks and branch splits,
propagate the remaining path budget to sub-solves so the aggregate solve cannot
silently exceed the requested limit.
When calling `generate_total_degree_start_system()` or
`generate_total_degree_solutions()` directly, pass `max_solutions` to enforce
the same kind of allocation cap before the Cartesian product of start roots is
materialized.
Solver-level numeric tolerances must be positive finite numbers, and boolean
flags such as `use_endgame`, `store_paths`, `allow_underdetermined`, and
`scale_equations` are validated before any numerical work starts.
The high-level `solve()` API returns finite zero-dimensional solution sets; it
rejects underdetermined polynomial systems as positive-dimensional even when
`allow_underdetermined=True`, instead of returning arbitrary samples from a
curve or surface. Use witness-set tools for positive-dimensional components.

Unknown tracking options are rejected so spelling mistakes do not silently
change a numerical run. The solver chooses a random unit-complex `gamma` by
default; provide `"gamma": 1.0 + 0.0j` or another explicit value when you need
to reproduce a particular homotopy independent of `random_state`. Explicit
`gamma` values must be finite and nonzero. Independent paths can be tracked
concurrently with `"n_jobs"`; endpoint ordering and path metadata remain
indexed by the original start-solution order.
If an automatically generated homotopy produces zero successful paths, the
solver retries once with a fresh generated `gamma` and records the decision in
`solutions._meta["tracking_retries"]`. Explicit `gamma` values are never
overridden. Failed individual paths are retried once with smaller steps, a
larger Newton iteration budget, and an RK4 predictor while preserving the same
gamma; these attempts are also recorded in `tracking_retries` and count against
`max_paths`.

The default predictor is first-order Euler. Use `"predictor": "heun"` for a
second-order tangent predictor or `"predictor": "rk4"` for a fourth-order
Runge-Kutta tangent predictor. Path metadata records predictor fallbacks and
the largest higher-order correction norm. Direct `track_paths()` and
`track_single_path()` calls apply the same validation to tolerances, boolean
flags, explicit `gamma`, variable lists, start-point dimensions, and endgame
options before tracking begins. They also reject start points that do not
satisfy the start system within the tracker tolerance budget and report the
accepted `start_residual` in path metadata. `track_paths()` accepts any
iterable of start points and consumes it once before validation. Direct start
points may be coordinate vectors, mappings keyed by `Variable` objects or
variable-name strings, or solution-like objects with a `values` mapping.
Direct path records include the validated effective tracking options, including
step bounds, predictor, `gamma`, endgame thresholds, and path limits.
The direct homotopy helpers `homotopy_function()`, `homotopy_jacobian()`, and
`compute_tangent()` accept the same coordinate-record forms for their point
argument.
Path singularity checks treat nonfinite, rank-deficient, or underdetermined
Jacobians as singular even when a raw condition-number estimate would be
finite.

By default, `solve()` normalizes each working equation by its largest
coefficient magnitude before direct solving, reduction, square-up, or path
tracking. This improves numerical conditioning for systems with uneven
coefficient scales while all reported residuals remain evaluated against the
original system. The divisors are recorded in
`solutions._meta["equation_scaling"]`; pass `scale_equations=False` to disable
this normalization for experiments that require the raw homotopy.

The lower-level `track_parameter_path(..., options={...})` API accepts the same
step-size, predictor-cap, and higher-order predictor controls for parameter
homotopies used in witness-set workflows. It validates finite start points and
checks that they satisfy the parameter homotopy at `start_t` before tracking.
Parameter homotopy start points and public evaluator points may be coordinate
vectors, mappings keyed by `Variable` objects or variable-name strings, or
solution-like objects with a `values` mapping.
Use `parameter_homotopy.system_at(t)` to materialize the coefficient-scaled
polynomial system actually tracked at an intermediate parameter value.
It validates option values, rejects unknown option names and non-boolean flags,
uses damped Newton correction at each parameter step, and reports
nonconvergence details such as `max_steps_exceeded`, `newton_failed`, or
`nonfinite_tangent` in `info["failure_reason"]`. Failed parameter paths include
the last accepted `final_point`, `final_residual`, and, when a Newton trial is
rejected, the `trial_point`, `trial_t`, and `trial_residual`.
All parameter path results include the validated effective tracking options,
including `tol`, step bounds, predictor, path bounds, and `max_steps`, so a
failure report is reproducible without reconstructing defaults by hand.
If `start_t == end_t`, `track_parameter_path()` validates the start point,
returns it without predictor or Newton steps, and records `direction == 0` with
`initial_step_size == 0.0`.

Endpoint Newton polishing is recorded under each path's `polish` metadata. A
failed polish attempt does not invalidate an otherwise successful tracked path;
the tracker keeps the better endpoint. Direct Cauchy endgame calls report
`status`, `failure_code`, `final_residual`, and `final_point` so singular-end
handling can be audited without parsing console output. Pass
`options={"random_state": ...}` to `run_cauchy_endgame()` when you need
reproducible heuristic projections. Direct endgame starting points may be
coordinate vectors, mappings keyed by `Variable` objects or variable-name
strings, or solution-like objects with a `values` mapping. Unknown endgame
option names are rejected by both `solve(..., endgame_options={...})` and
direct endgame calls, and
endgame `gamma` values must also be finite and nonzero. Cauchy endgame loops
are bounded by `max_iterations` and report `max_iterations_exceeded` instead
of running without an explicit iteration guard. Failed path-tracking
attempts also record
the trial `final_t`, `final_point`, and `final_residual` where available;
nonfinite Newton-corrector outputs are reported as `nonfinite_corrector` with an
infinite residual rather than `nan` metadata.

## Witness Sets

`compute_witness_superset()` builds a generic slice, solves the augmented
system through the standard solver, and records witness metadata in
`witnesses._meta["witness_set"]`. Its `random_state` controls both the slice and
the underlying solve, so witness computations are reproducible. Redundant
overdetermined augmented systems use the solver's square-up path, while
requests that still leave the augmented system underdetermined are rejected
instead of returning artificial dummy-slice points. Solver options passed to
this helper are validated and cannot override the controlled target system or
ambient variable list.
`WitnessSet` stores a deterministic ambient variable list inferred from the
original system and its slice, so sampling and membership tests keep working
when free variables only appear in the slicing equations. Use
`witness_set.sample_points()` to move the whole witness set to a new slice, or
`sample_point()` when a single random path is enough. Membership testing tracks
every stored witness point to the slice through the query point, avoiding
degree-greater-than-one false negatives caused by sampling a single witness.
Pass `return_info=True` to witness sampling methods to get per-witness tracking
and validation records alongside the returned point(s).
Constructor witness points may be coordinate vectors, `Solution` objects,
solution-like objects with a `values` mapping, or coordinate mappings keyed by
`Variable` objects or variable-name strings; they are normalized to `Solution`
objects after validation.
Membership query points may be coordinate vectors, mappings keyed by `Variable`
objects or variable-name strings, or solution-like objects with a `values`
mapping.
Sampled points are checked against the original system and the target slice
before they are returned.
Witness dimensions, slicing equation counts, ambient variable lists, target
slice sizes, and exact witness coordinate records are validated before tracking.
Constructed witness points must satisfy both the original system and the
slicing system within `validation_tolerance`.

Monodromy loop tracing also accepts `random_state` directly, and
`numerical_irreducible_decomposition()` accepts
`monodromy_options={"random_state": ...}` so loop target slices can be
reproduced during component breakup. It accepts witness supersets as
`SolutionSet` objects or as iterables of coordinate records. Loop tracing and
decomposition routines are quiet by default; pass `verbose=True` in the
monodromy options for progress output. Monodromy start witness points may be
coordinate vectors, `Solution` objects, mappings keyed by `Variable` objects or
variable-name strings, or solution-like objects with a `values` mapping. The
monodromy loop count, matching tolerance, option dictionaries, variable list,
exact witness point coordinates, and witness residuals against the original
system and start slice are validated before tracking starts; unknown monodromy
tracker option names are rejected up front.

## System Preprocessing

Before path tracking, `solve()` removes identically zero constant equations,
detects inconsistent constant equations, and drops nonconstant equations that
are exact scalar multiples of an earlier equation. Inconsistent constants return
an empty `SolutionSet` immediately. Removed duplicates, scale factors, and other
preprocessing details are recorded in `solutions._meta["preprocessing"]`.
Duplicate removal is skipped for custom start systems so the supplied start and
target systems keep matching equation counts.
If removing zero equations leaves variables with no constraints, the solver
reports a positive-dimensional system instead of returning a dummy point.

Full-rank linear systems are solved directly with a rank-aware least-squares
path instead of numerical continuation. Consistent square or overdetermined
linear systems return exact `SolutionSet` metadata under
`solutions._meta["linear_solve"]`; inconsistent linear systems return no
solutions, and rank-deficient consistent linear systems raise because they have
positive-dimensional solution sets.

Univariate systems are solved directly through companion roots and endpoint
polishing. For multiple equations in one variable, the solver roots the
lowest-degree positive equation and filters candidates through the full system.
When optional SymPy support is installed, it first tries a univariate polynomial
GCD and roots the lower-degree common factor when that reduces candidate roots.
It also factors repeated univariate components and roots each distinct factor
once, carrying algebraic multiplicity on the resulting `Solution` objects instead
of relying on fragile companion eigenvalue clusters for high-multiplicity roots.
The result still uses standard `Solution` objects with multiplicity clustering,
and records `solutions._meta["univariate_solve"]`. Custom univariate start
systems continue to track the requested homotopy.

For mixed systems, simple coordinate assignments such as `y - 1` and affine
linear equations such as `x + y - 1` are eliminated before solving the reduced
system. Lifted solutions record `solutions._meta["coordinate_reduction"]` or
`solutions._meta["linear_reduction"]`, and the reduced problem can still use the
direct linear, direct univariate, square-up, or path-tracking solver path.

Monomial zero equations such as `x*y = 0` are split into exact coordinate
branches before generic path tracking. Equations with a common monomial factor,
such as `x*(y - 1) = 0`, are split into the coordinate factor branches plus a
cofactor branch. Each branch solves the reduced system, lifts candidates back to
the original variables, and deduplicates overlapping branch endpoints. Results
record `solutions._meta["monomial_zero_branches"]`.
If a consistent branch leaves a free variable, the solver reports a
positive-dimensional component instead of falling back to generic tracking and
returning an arbitrary singular sample point.

When optional SymPy support is installed, reducible polynomial equations such
as `(x - 1)*(y - 2) = 0` can also be factored into branch systems before
generic path tracking. Repeated factors are solved once per distinct branch and
their multiplicities are carried into the lifted solutions. Each factor branch
is solved with the normal solver, validated against the original equations, and
merged with overlap deduplication; metadata is recorded in
`solutions._meta["factorized_branches"]`. Consistent factor branches with free
variables are rejected as positive-dimensional components. Without SymPy, the
solver falls back to the standard homotopy and direct-solve paths.

Square triangular systems are solved by recursive univariate elimination when
each step exposes one new variable after prior substitutions. For example,
`x**2 - 1, y - x**2` is solved directly without a total-degree start system.
When optional SymPy support is available, repeated factors at each triangular
step are rooted once and their multiplicities are multiplied through the branch.
Results record `solutions._meta["triangular_solve"]` and still validate every
branch against the original equations.

Square full-rank binomial systems with no common monomial factors are solved
directly in the algebraic torus. For example, `x*y - 1, x**2 - y` is solved by
enumerating the three logarithmic lifts implied by the exponent matrix instead
of tracking four total-degree paths. With optional SymPy support, Hermite normal
form enumerates one representative per integer-lattice coset, avoiding
`|det(A)|**n` brute-force lift searches. Results record
`solutions._meta["binomial_solve"]`; unsupported sparse systems fall back to
the standard homotopy path.

When equations split into independent variable blocks, `solve()` solves each
block separately and forms the Cartesian product of validated block solutions.
This keeps unrelated subsystems from sharing one large continuation run and
records the split under `solutions._meta["independent_blocks"]`.

When the default total-degree homotopy is used for a system with more equations
than variables, `solve()` squares the system up with reproducible random linear
combinations before tracking. Candidate endpoints are still filtered against
the original equations, so inconsistent overdetermined systems return no
solutions. The random coefficients and dimensions are recorded in
`solutions._meta["square_up"]`.

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

