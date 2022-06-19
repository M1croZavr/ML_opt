from matplotlib import pyplot as plt
import numpy as np


def make_level_lines_plot(X, Y, Z, points_history, var1, var2):
    """Draw level lines plot according to passed values. Draws countour and extrema as dots."""
    plt.figure(figsize=(16, 8))
    initial_point = points_history[0]
    final_point = points_history[-1]
    plt.scatter(initial_point[0], initial_point[1], c='black', label='Starting point', s=100)
    plt.scatter(final_point[0], final_point[1], c='red', label='Final point', s=100)
    plt.plot(np.array([point[0] for point in points_history]),
             np.array([point[1] for point in points_history]), 'b--', label='Convergence trajectory')
    plt.contourf(X, Y, Z, 10, cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.xlabel(str(var1))
    plt.ylabel(str(var2))
    plt.legend()
    plt.title('Level lines and algorithm points')
    plt.show()


def make_annealing_plot_2d(x, y, points):
    plt.figure(figsize=(16, 8))
    plt.plot(x, y, 'b-', label='Function')
    plt.vlines(points[:, 0], 0, points[:, 1], colors='r', linestyles='dashed')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Function and iteration points')
    plt.show()


def plot_energy_history(energy):
    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(0, len(energy)), energy)
    plt.xlabel('Iteration')
    plt.ylabel('Energy (Function value)')
    plt.title('Dependence of the value of a given function on iteration')
    plt.show()
