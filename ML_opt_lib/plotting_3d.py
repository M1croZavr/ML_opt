from matplotlib import pyplot as plt, cm


def make_3d_plot(X, Y, Z, m_points, var1, var2, fun):
    """Draw 3D plot according to passed values. Draws plane and extemums as dots."""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15, 7))
    for i, m in enumerate(m_points):
        coord1, coord2 = m.values()
        ax.scatter(coord1,
                   coord2,
                   float(fun.subs([(var1, coord1), (var2, coord2)])),
                   c='black',
                   s=40,
                   label=f'Extremum{i}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot_surface(X, Y, Z, cmap=cm.winter, alpha=0.5)
    plt.legend()
    ax.set_title('Plane and extremum points')
    plt.show()


def make_3d_plot_lagrange(X, Y, Z1, Z2, m_points, var1, var2, fun, fun_c):
    """Draw 3D plot according to passed values. Draws plane and extemums as dots."""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15, 7))
    for i, m in enumerate(m_points):
        coord1, coord2 = list(m.values())[1:]
        ax.scatter(coord1,
                   coord2,
                   float(fun.subs([(var1, coord1), (var2, coord2)])),
                   c='red',
                   s=30,
                   label=f'Extremum{i}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot_surface(X, Y, Z1, cmap=cm.hot, alpha=0.3)
    ax.plot_surface(X, Y, Z2, cmap=cm.winter, alpha=0.5)
    plt.legend()
    ax.set_title('Plane and extremum points')
    plt.show()
