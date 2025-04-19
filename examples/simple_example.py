"""
Simple example demonstrating the basic usage of PyContinuum.

This example solves a system of two equations:
- x^2 + y^2 = 1 (a circle)
- x^2 = y (a parabola)

The intersection points are the solutions to this system.
"""

import sys
import os
import time
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import pycontinuum
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pycontinuum import polyvar, PolynomialSystem, solve

def main():
    """Run the simple example."""
    print("PyContinuum Simple Example")
    print("=========================")
    # Define variables
    print("Defining variables and equations...")
    x, y = polyvar('x', 'y')
    
    # Define polynomial system
    f1 = x**2 + y**2 - 1      # circle: x^2 + y^2 = 1
    f2 = x**2 - y             # parabola: x^2 = y
    system = PolynomialSystem([f1, f2])
    
    # Debug the variables
    vars_set = system.variables()
    print(f"Variables in the system: {vars_set}")
    print(f"Number of variables: {len(vars_set)}")
    variables_list = list(vars_set)
    
    print(f"System to solve:\n{system}\n")
    
    # Solve the system
    print("Solving the system...")
    start_time = time.time()
    solutions = solve(system, variables=variables_list, verbose=True)
    solve_time = time.time() - start_time
    
    print(f"\nSolve completed in {solve_time:.3f} seconds")
    print(f"Found {len(solutions)} solutions:")
    
    # Display each solution
    for i, sol in enumerate(solutions):
        print(f"\nSolution {i+1}:")
        print(sol)
    
    # Visualize the system and solutions
    print("\nCreating visualization...")
    
    # Plot the algebraic curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate points for the circle
    t = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(t)
    circle_y = np.sin(t)
    
    # Generate points for the parabola
    px = np.linspace(-1.5, 1.5, 100)
    py = px**2
    
    # Plot curves
    ax.plot(circle_x, circle_y, 'b-', linewidth=2, label='$x^2 + y^2 = 1$')
    ax.plot(px, py, 'r-', linewidth=2, label='$x^2 = y$')
    
    # Plot solutions
    real_solutions = solutions.filter(real=True)
    
    solution_x = [sol.values[x].real for sol in real_solutions]
    solution_y = [sol.values[y].real for sol in real_solutions]
    
    ax.scatter(solution_x, solution_y, color='green', s=100, zorder=5, label='Solutions')
    
    # Set up the plot
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Circle and Parabola Intersection')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('circle_parabola_intersection.png')
    print("Visualization saved as 'circle_parabola_intersection.png'")
    
    # Show the plot if running interactively
    plt.show()
    
    return solutions

if __name__ == "__main__":
    # Execute the example
    try:
        import numpy as np
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the required dependencies: numpy, matplotlib")
    except Exception as e:
        print(f"Error: {e}")