from matplotlib import pyplot as plt, cm


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
    ax.set_xlabel(str(var1))
    ax.set_ylabel(str(var2))
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
    plt.xlabel(str(var1))
    plt.ylabel(str(var2))
    if m_points:
        plt.legend()
    plt.title('Level lines and extrema points')
    plt.show()


def regression_3d(data, targets, x1_mesh, x2_mesh, y_mesh):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(18, 9))
    ax.scatter(data[:, 0], data[:, 1], targets, s=6, c='blue', label='Data samples')
    # x_mesh, y_mesh = np.meshgrid(np.linspace(min(data[:, 0]), max(data[:, 0]), 25),
    #                              np.linspace(min(data[:, 1]), max(data[:, 1]), 25))
    # z_mesh = np.array([[w[0] + w[1] * x + w[2] * y for x, y in zip(x_i, y_i)] for x_i, y_i in zip(x_mesh, y_mesh)])
    ax.set_title('Data samples and regression plane')
    ax.plot_surface(x1_mesh, x2_mesh, y_mesh, cmap=cm.winter, alpha=0.5)
    ax.legend()
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.show()


def regression_2d(data, targets, x1, y):
    plt.figure(figsize=(18, 9))
    plt.scatter(data[:, 0], targets, s=6, c='blue', label='Data samples')
    # lin = np.linspace(min(data[:, 0]), max(data[:, 0]), 50)
    plt.plot(x1, y, c='teal', label='Regression line')
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.title('Data samples and regression line')
    plt.legend()
    plt.show()
