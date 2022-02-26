from matplotlib import pyplot as plt, cm
import numpy as np


def make_3d_plot(X, Y, Z, m_points, var1, var2, fun):
    """Draw 3D plot according to passed values. Draws plane and extrema as dots."""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15, 7))
    for i, m in enumerate(m_points):
        coord1, coord2 = m.values()
        ax.scatter(coord1,
                   coord2,
                   float(fun.subs([(var1, coord1), (var2, coord2)])),
                   c='black',
                   s=40,
                   label=f'Extrema{i}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot_surface(X, Y, Z, cmap=cm.winter, alpha=0.5)
    if m_points:
        plt.legend()
    ax.set_title('Plane and extrema points')
    plt.show()


def make_3d_plot_lagrange(X, Y, Z1, Z2, m_points, var1, var2, fun):
    """Draw 3D plot according to passed values. Draws plane and extrema as dots."""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15, 7))
    for i, m in enumerate(m_points):
        coord1, coord2 = list(m.values())[1:]
        ax.scatter(coord1,
                   coord2,
                   float(fun.subs([(var1, coord1), (var2, coord2)])),
                   c='red',
                   s=30,
                   label=f'Extrema{i}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot_surface(X, Y, Z1, cmap=cm.winter, alpha=0.5)
    ax.plot_surface(X, Y, Z2, cmap=cm.hot, alpha=0.6)
    if m_points:
        plt.legend()
    ax.set_title('Plane and extrema points')
    plt.show()


def make_level_lines_plot(X, Y, Z, m_points, var1, var2, Z_c=None):
    """Draw level lines plot according to passed values. Draws countour and extrema as dots."""
    plt.figure(figsize=(10, 6))
    for i, m in enumerate(m_points):
        coord1, coord2 = m[var1], m[var2]
        plt.scatter(coord1,
                    coord2,
                    c='black',
                    label=f'Extrema{i}')
    plt.contourf(X, Y, Z, 10, cmap='viridis', alpha=0.6)
    plt.colorbar()
    if not(Z_c is None):
        plt.contour(X, Y, Z_c, 0)
    plt.xlabel('X')
    plt.ylabel('Y')
    if m_points:
        plt.legend()
    plt.title('Level lines and extrema points')
    plt.show()
