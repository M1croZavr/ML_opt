from matplotlib import pyplot as plt


def plot_scatter_and_line(X, Y, x, y, names):
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, 'b-', label='Function')
    coordinates = list(zip(x, y))
    for i, name in enumerate(names):
        plt.scatter(coordinates[i][0], coordinates[i][1], c='r')
        plt.annotate(name, (coordinates[i][0], coordinates[i][1]), fontsize=20)
    plt.title('Function and algorithms dots')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()


def plot_parabola_and_line(X, Y, u, a, b, c):
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, 'b-', label='Function')
    plt.plot(X, [a * x_i ** 2 + b * x_i + c for x_i in X], 'r--', label='Approximating parabola')
    plt.scatter(u, a * u ** 2 + b * u + c, c='black')
    plt.annotate('u', (u, a * u ** 2 + b * u + c), fontsize=20)
    plt.title('Function and approximating parabola')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()
