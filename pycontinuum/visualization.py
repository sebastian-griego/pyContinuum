"""
Visualization module for PyContinuum.

This module provides functions to visualize solution paths and results
from homotopy continuation.
"""

from collections.abc import Mapping
from numbers import Integral, Real
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots

from pycontinuum.polynomial import Variable
from pycontinuum.utils import _mapping_coordinate_for_variable


def _validate_plot_variable(name: str, variable: Any) -> Variable:
    if not isinstance(variable, Variable):
        raise TypeError(f"{name} must be a Variable")
    return variable


def _solution_coordinate(
    solution: Any,
    variable: Variable,
    label: str = "solution",
) -> complex:
    if isinstance(solution, Mapping):
        values = solution
    else:
        values = getattr(solution, "values", None)
        if not isinstance(values, Mapping):
            raise TypeError(
                f"{label} must be a coordinate mapping or expose a values mapping"
            )

    found, value = _mapping_coordinate_for_variable(values, variable, label)
    if not found:
        raise ValueError(
            f"{label} is missing coordinate for variable {variable.name}"
        )

    try:
        return complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{label} coordinate for variable {variable.name} must be numeric"
        ) from exc


def _normalize_parameter_continuation_results(
    results: Any,
) -> List[Tuple[float, List[Any]]]:
    if isinstance(results, (str, bytes)):
        raise TypeError(
            "results must be an iterable of (parameter, solutions) pairs"
        )

    try:
        result_list = list(results)
    except TypeError as exc:
        raise TypeError(
            "results must be an iterable of (parameter, solutions) pairs"
        ) from exc

    if not result_list:
        raise ValueError("results must contain at least one parameter sample")

    normalized = []
    for sample_index, sample in enumerate(result_list):
        if isinstance(sample, (str, bytes)):
            raise ValueError(
                f"results[{sample_index}] must be a (parameter, solutions) pair"
            )

        try:
            parameter, solutions = sample
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"results[{sample_index}] must be a (parameter, solutions) pair"
            ) from exc

        if isinstance(parameter, (bool, np.bool_)):
            raise TypeError(f"results[{sample_index}][0] must be a number")
        try:
            parameter_value = float(parameter)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"results[{sample_index}][0] must be a number") from exc
        if not np.isfinite(parameter_value):
            raise ValueError(f"results[{sample_index}][0] must be finite")

        if isinstance(solutions, (str, bytes, Mapping)):
            raise TypeError(
                f"results[{sample_index}][1] must be an iterable of solution records"
            )
        try:
            solution_list = list(solutions)
        except TypeError as exc:
            raise TypeError(
                f"results[{sample_index}][1] must be an iterable of solution records"
            ) from exc

        normalized.append((parameter_value, solution_list))

    return normalized


def _validate_path_variable_index(var_idx: Any) -> int:
    if isinstance(var_idx, bool) or not isinstance(var_idx, Integral):
        raise TypeError("var_idx must be an integer")
    var_idx = int(var_idx)
    if var_idx < 0:
        raise ValueError("var_idx must be non-negative")
    return var_idx


def _normalize_path_points(
    path_points: Any,
    var_idx: int,
    name: str,
    *,
    require_nonempty: bool,
) -> Tuple[List[float], List[complex]]:
    if isinstance(path_points, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of (t, point) pairs")
    try:
        samples = list(path_points)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an iterable of (t, point) pairs"
        ) from exc

    if require_nonempty and not samples:
        raise ValueError(f"{name} must contain at least one path point")

    t_values = []
    var_values = []
    for sample_index, sample in enumerate(samples):
        label = f"{name}[{sample_index}]"
        if isinstance(sample, (str, bytes)):
            raise ValueError(f"{label} must be a (t, point) pair")

        try:
            t_value, point = sample
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a (t, point) pair") from exc

        if isinstance(t_value, bool) or not isinstance(t_value, Real):
            raise TypeError(f"{label}[0] must be a number")
        t_value = float(t_value)
        if not np.isfinite(t_value):
            raise ValueError(f"{label}[0] must be finite")

        try:
            point_array = np.asarray(point, dtype=complex)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label}[1] must be an array-like point") from exc
        if point_array.ndim != 1:
            raise ValueError(f"{label}[1] must be a one-dimensional point")
        if point_array.shape[0] <= var_idx:
            raise ValueError(
                f"{label}[1] must contain coordinate index {var_idx}"
            )

        var_value = point_array[var_idx]
        if not np.isfinite(var_value):
            raise ValueError(f"{label}[1][{var_idx}] must be finite")

        t_values.append(t_value)
        var_values.append(complex(var_value))

    return t_values, var_values


def _normalize_all_path_points(
    all_path_points: Any,
    var_idx: int,
) -> List[Tuple[List[float], List[complex]]]:
    if isinstance(all_path_points, (str, bytes)):
        raise TypeError(
            "all_path_points must be an iterable of path point iterables"
        )
    try:
        path_collections = list(all_path_points)
    except TypeError as exc:
        raise TypeError(
            "all_path_points must be an iterable of path point iterables"
        ) from exc

    return [
        _normalize_path_points(
            path_points,
            var_idx,
            f"all_path_points[{path_index}]",
            require_nonempty=False,
        )
        for path_index, path_points in enumerate(path_collections)
    ]


def _validate_plot_alpha(alpha: Any) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a number")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")
    return alpha


def plot_path(path_points: List[Tuple[float, np.ndarray]], 
              var_idx: int = 0,
              title: Optional[str] = None,
              figsize: Tuple[int, int] = (10, 8),
              show_endpoints: bool = True) -> plt.Figure:
    """Plot a solution path in the complex plane.
    
    Args:
        path_points: List of (t, solution) pairs along the path
        var_idx: Index of the variable to plot (default: 0)
        title: Plot title (default: auto-generated)
        figsize: Figure size
        show_endpoints: Whether to mark the start and end points
        
    Returns:
        The created matplotlib figure
    """
    var_idx = _validate_path_variable_index(var_idx)
    t_values, var_values = _normalize_path_points(
        path_points,
        var_idx,
        "path_points",
        require_nonempty=True,
    )
    
    real_parts = [z.real for z in var_values]
    imag_parts = [z.imag for z in var_values]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the path
    scatter = ax.scatter(real_parts, imag_parts, c=t_values, cmap='viridis', 
                        s=30, alpha=0.7)
    
    # Add a colorbar for t values
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('t value (1 = start, 0 = target)')
    
    # Connect the points with a line
    ax.plot(real_parts, imag_parts, 'k-', alpha=0.3)
    
    # Highlight the start and end points if requested
    if show_endpoints:
        ax.plot(real_parts[0], imag_parts[0], 'go', markersize=10, label='Start (t=1)')
        ax.plot(real_parts[-1], imag_parts[-1], 'ro', markersize=10, label='End (t=0)')
        ax.legend()
    
    # Set axis labels and title
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    
    if title is None:
        title = f'Solution Path in Complex Plane (Variable {var_idx})'
    ax.set_title(title)
    
    # Equal aspect ratio for the complex plane
    ax.axis('equal')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_all_paths(all_path_points: List[List[Tuple[float, np.ndarray]]],
                  var_idx: int = 0,
                  title: Optional[str] = None,
                  figsize: Tuple[int, int] = (12, 10),
                  show_endpoints: bool = True,
                  alpha: float = 0.5) -> plt.Figure:
    """Plot multiple solution paths in the complex plane.
    
    Args:
        all_path_points: List of path_points for multiple paths
        var_idx: Index of the variable to plot (default: 0)
        title: Plot title (default: auto-generated)
        figsize: Figure size
        show_endpoints: Whether to mark the end points
        alpha: Transparency for the paths
        
    Returns:
        The created matplotlib figure
    """
    var_idx = _validate_path_variable_index(var_idx)
    alpha = _validate_plot_alpha(alpha)
    normalized_paths = _normalize_all_path_points(all_path_points, var_idx)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use a different color for each path
    n_paths = len(normalized_paths)
    cmap = plt.cm.rainbow
    colors = [cmap(i / max(n_paths, 1)) for i in range(n_paths)]
    
    # Plot each path
    plotted_endpoints = False
    for i, (t_values, var_values) in enumerate(normalized_paths):
        if not t_values:
            continue
        
        real_parts = [z.real for z in var_values]
        imag_parts = [z.imag for z in var_values]
        
        # Plot the path
        ax.plot(real_parts, imag_parts, '-', color=colors[i], alpha=alpha, linewidth=1.5)
        
        # Highlight the end points if requested
        if show_endpoints:
            if not plotted_endpoints:
                ax.plot(real_parts[-1], imag_parts[-1], 'ro', markersize=5, alpha=0.7, 
                       label='Solutions (t=0)')
                plotted_endpoints = True
            else:
                ax.plot(real_parts[-1], imag_parts[-1], 'ro', markersize=5, alpha=0.7)
    
    # Set axis labels and title
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    
    if title is None:
        title = f'All Solution Paths in Complex Plane (Variable {var_idx})'
    ax.set_title(title)
    
    # Equal aspect ratio for the complex plane
    ax.axis('equal')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    if show_endpoints and plotted_endpoints:
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_solutions_2d(solution_set, 
                     var_x: Variable,
                     var_y: Variable,
                     title: Optional[str] = None,
                     figsize: Tuple[int, int] = (10, 8),
                     real_only: bool = False,
                     marker_size: int = 100) -> plt.Figure:
    """Plot solutions in 2D for two selected variables.
    
    Args:
        solution_set: Set of solutions to plot
        var_x: Variable for the x-axis
        var_y: Variable for the y-axis
        title: Plot title (default: auto-generated)
        figsize: Figure size
        real_only: Whether to only plot real solutions
        marker_size: Size of the solution markers
        
    Returns:
        The created matplotlib figure
    """
    var_x = _validate_plot_variable("var_x", var_x)
    var_y = _validate_plot_variable("var_y", var_y)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter solutions if requested
    solutions = solution_set.solutions
    if real_only:
        solutions = [sol for sol in solutions if sol.is_real()]
    
    # Extract x and y coordinates
    x_coords = []
    y_coords = []
    complex_coords = []
    colors = []  # For distinguishing regular and singular solutions
    
    for sol in solutions:
        # Get values and take real part if real_only
        x_val = _solution_coordinate(sol, var_x)
        y_val = _solution_coordinate(sol, var_y)
        complex_coords.append((x_val, y_val))
        
        if real_only:
            x_val = x_val.real
            y_val = y_val.real
            
        x_coords.append(x_val.real if not real_only else x_val)
        y_coords.append(y_val.real if not real_only else y_val)
        
        # Color based on singularity
        colors.append('red' if sol.is_singular else 'blue')
    
    # Plot the solutions
    if real_only:
        scatter = ax.scatter(x_coords, y_coords, c=colors, s=marker_size, alpha=0.7,
                           edgecolors='k')
    else:
        # For complex solutions, use real parts on axes and size for imaginary part
        sizes = [
            marker_size * (1 + 0.5 * abs(x.imag) + 0.5 * abs(y.imag))
            for x, y in complex_coords
        ]
        scatter = ax.scatter(x_coords, y_coords, c=colors, s=sizes, alpha=0.7,
                           edgecolors='k')
    
    # Set axis labels and title
    ax.set_xlabel(f'{var_x.name} ({"Real" if real_only else "Real Part"})')
    ax.set_ylabel(f'{var_y.name} ({"Real" if real_only else "Real Part"})')
    
    if title is None:
        title = f'Solutions in {var_x.name}-{var_y.name} Plane'
        if real_only:
            title += ' (Real Solutions Only)'
    ax.set_title(title)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Regular'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Singular')
    ]
    ax.legend(handles=legend_elements)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_solutions_3d(solution_set, 
                     var_x: Variable,
                     var_y: Variable,
                     var_z: Variable,
                     title: Optional[str] = None,
                     figsize: Tuple[int, int] = (12, 10),
                     real_only: bool = True,
                     marker_size: int = 100) -> plt.Figure:
    """Plot solutions in 3D for three selected variables.
    
    Args:
        solution_set: Set of solutions to plot
        var_x: Variable for the x-axis
        var_y: Variable for the y-axis
        var_z: Variable for the z-axis
        title: Plot title (default: auto-generated)
        figsize: Figure size
        real_only: Whether to only plot real solutions
        marker_size: Size of the solution markers
        
    Returns:
        The created matplotlib figure
    """
    var_x = _validate_plot_variable("var_x", var_x)
    var_y = _validate_plot_variable("var_y", var_y)
    var_z = _validate_plot_variable("var_z", var_z)

    # Create the 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Filter solutions if requested
    solutions = solution_set.solutions
    if real_only:
        solutions = [sol for sol in solutions if sol.is_real()]
    
    # Extract coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    complex_coords = []
    colors = []  # For distinguishing regular and singular solutions
    
    for sol in solutions:
        # Get values and take real part if real_only
        x_val = _solution_coordinate(sol, var_x)
        y_val = _solution_coordinate(sol, var_y)
        z_val = _solution_coordinate(sol, var_z)
        complex_coords.append((x_val, y_val, z_val))
        
        if real_only:
            x_val = x_val.real
            y_val = y_val.real
            z_val = z_val.real
            
        x_coords.append(x_val.real if not real_only else x_val)
        y_coords.append(y_val.real if not real_only else y_val)
        z_coords.append(z_val.real if not real_only else z_val)
        
        # Color based on singularity
        colors.append('red' if sol.is_singular else 'blue')
    
    # Plot the solutions
    if real_only:
        sizes = marker_size
    else:
        sizes = [
            marker_size
            * (
                1
                + (abs(x.imag) + abs(y.imag) + abs(z.imag)) / 3.0
            )
            for x, y, z in complex_coords
        ]
    scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, s=sizes, alpha=0.7,
                         edgecolors='k')
    
    # Set axis labels and title
    ax.set_xlabel(f'{var_x.name} ({"Real" if real_only else "Real Part"})')
    ax.set_ylabel(f'{var_y.name} ({"Real" if real_only else "Real Part"})')
    ax.set_zlabel(f'{var_z.name} ({"Real" if real_only else "Real Part"})')
    
    if title is None:
        title = f'Solutions in {var_x.name}-{var_y.name}-{var_z.name} Space'
        if real_only:
            title += ' (Real Solutions Only)'
    ax.set_title(title)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Regular'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Singular')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    return fig


def plot_parameter_continuation(results: List[Tuple[float, List[Dict[Variable, complex]]]],
                               param_var: Variable,
                               plot_var: Variable,
                               title: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 8),
                               plot_imag: bool = False) -> plt.Figure:
    """Plot solution branches for parameter continuation.
    
    Args:
        results: List of (parameter_value, solutions) pairs
        param_var: Parameter variable (x-axis)
        plot_var: Variable to plot on y-axis
        title: Plot title (default: auto-generated)
        figsize: Figure size
        plot_imag: Whether to plot imaginary parts (default: False)
        
    Returns:
        The created matplotlib figure
    """
    param_var = _validate_plot_variable("param_var", param_var)
    plot_var = _validate_plot_variable("plot_var", plot_var)
    results = _normalize_parameter_continuation_results(results)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Track solution branches across parameter values
    branches = []
    
    # Start with the solutions at the first parameter value
    initial_solutions = results[0][1]
    for solution_index, init_sol in enumerate(initial_solutions):
        solution_value = _solution_coordinate(
            init_sol,
            plot_var,
            label=f"results[0][1][{solution_index}]",
        )
        branch = [(results[0][0], solution_value)]
        branches.append(branch)
    
    # For each subsequent parameter value, match solutions to existing branches
    for i in range(1, len(results)):
        param, solutions = results[i]

        current_values = [
            _solution_coordinate(
                solution,
                plot_var,
                label=f"results[{i}][1][{solution_index}]",
            )
            for solution_index, solution in enumerate(solutions)
        ]

        match_candidates = []
        for branch_index, branch in enumerate(branches):
            last_sol = branch[-1][1]
            for solution_index, sol_val in enumerate(current_values):
                match_candidates.append((
                    abs(sol_val - last_sol),
                    branch_index,
                    solution_index,
                ))

        matched_branches = set()
        matched_solutions = set()
        for _, branch_index, solution_index in sorted(match_candidates):
            if branch_index in matched_branches or solution_index in matched_solutions:
                continue
            branches[branch_index].append((param, current_values[solution_index]))
            matched_branches.add(branch_index)
            matched_solutions.add(solution_index)
    
    # Plot each branch
    for branch in branches:
        param_vals = [p for p, _ in branch]
        
        if plot_imag:
            real_vals = [sol.real for _, sol in branch]
            imag_vals = [sol.imag for _, sol in branch]
            
            ax.plot(param_vals, real_vals, '-o', markersize=4, label='Real Part')
            ax.plot(param_vals, imag_vals, '--o', markersize=4, label='Imaginary Part')
        else:
            real_vals = [sol.real for _, sol in branch]
            ax.plot(param_vals, real_vals, '-o', markersize=4)
    
    # Set axis labels and title
    ax.set_xlabel(f'Parameter {param_var.name}')
    
    if plot_imag:
        ax.set_ylabel(f'{plot_var.name} (Real and Imaginary Parts)')
    else:
        ax.set_ylabel(f'{plot_var.name} (Real Part)')
    
    if title is None:
        title = f'Parameter Continuation: {plot_var.name} vs {param_var.name}'
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    if plot_imag:
        ax.legend()
    
    plt.tight_layout()
    return fig
