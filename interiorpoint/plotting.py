from matplotlib import pyplot as plt


def draw_level_lines(X, Y, Z, points, variables):
    plt.figure(figsize=(12, 8))
    plt.plot([point[variables[0]] for point in points],
             [point[variables[1]] for point in points], '-o', color='black')
    plt.contourf(X, Y, Z, 15, cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.xlabel(variables[0])
    plt.ylabel(variables[1])
    plt.show()


def draw_lines(X, Y, points, variables):
    plt.figure(figsize=(12, 8))
    plt.plot([point[0] for point in points],
             [point[1] for point in points], '-o', color='black')
    plt.plot(X, Y, alpha=0.5)
    plt.xlabel(variables[0])
    plt.ylabel('y')
    plt.show()
