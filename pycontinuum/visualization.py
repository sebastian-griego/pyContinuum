"""
Visualization module for PyContinuum.

This module provides functions to visualize solution paths and results
from homotopy continuation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots

from pycontinuum.polynomial import Variable


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
    # Extract the t values and corresponding variable values
    t_values = [p[0] for p in path_points]
    var_values = [p[1][var_idx] for p in path_points]
    
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
    if show_endpoints and len(path_points) > 0:
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
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use a different color for each path
    n_paths = len(all_path_points)
    cmap = plt.cm.rainbow
    colors = [cmap(i / n_paths) for i in range(n_paths)]
    
    # Plot each path
    for i, path_points in enumerate(all_path_points):
        if not path_points:
            continue
            
        t_values = [p[0] for p in path_points]
        var_values = [p[1][var_idx] for p in path_points]
        
        real_parts = [z.real for z in var_values]
        imag_parts = [z.imag for z in var_values]
        
        # Plot the path
        ax.plot(real_parts, imag_parts, '-', color=colors[i], alpha=alpha, linewidth=1.5)
        
        # Highlight the end points if requested
        if show_endpoints and len(path_points) > 0:
            if i == 0:  # Add labels only once
                ax.plot(real_parts[-1], imag_parts[-1], 'ro', markersize=5, alpha=0.7, 
                       label='Solutions (t=0)')
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
    if show_endpoints:
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
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter solutions if requested
    solutions = solution_set.solutions
    if real_only:
        solutions = [sol for sol in solutions if sol.is_real()]
    
    # Extract x and y coordinates
    x_coords = []
    y_coords = []
    colors = []  # For distinguishing regular and singular solutions
    
    for sol in solutions:
        # Get values and take real part if real_only
        x_val = sol.values[var_x]
        y_val = sol.values[var_y]
        
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
        sizes = [marker_size * (1 + 0.5*abs(x.imag) + 0.5*abs(y.imag)) 
                for x, y in zip(x_coords, y_coords)]
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
    colors = []  # For distinguishing regular and singular solutions
    
    for sol in solutions:
        # Get values and take real part if real_only
        x_val = sol.values[var_x]
        y_val = sol.values[var_y]
        z_val = sol.values[var_z]
        
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
    scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, s=marker_size, alpha=0.7,
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
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract parameter values
    param_values = [param for param, _ in results]
    
    # Track solution branches across parameter values
    branches = []
    
    # Start with the solutions at the first parameter value
    initial_solutions = results[0][1]
    for init_sol in initial_solutions:
        branch = [(results[0][0], init_sol[plot_var])]
        branches.append(branch)
    
    # For each subsequent parameter value, match solutions to existing branches
    for i in range(1, len(results)):
        param, solutions = results[i]
        
        # For each existing branch, find the closest solution
        for branch in branches:
            last_sol = branch[-1][1]
            
            # Find closest solution at this parameter value
            closest_sol = None
            min_dist = float('inf')
            
            for sol_dict in solutions:
                sol_val = sol_dict[plot_var]
                dist = abs(sol_val - last_sol)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_sol = sol_val
            
            # Add to branch if we found a match
            if closest_sol is not None:
                branch.append((param, closest_sol))
    
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