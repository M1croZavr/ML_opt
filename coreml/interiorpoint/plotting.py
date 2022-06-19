from matplotlib import pyplot as plt
from ..twovarextremas.utils_twovarextremas import preproc_fun
import sympy
import functools


def draw_level_lines(X, Y, Z, points, variables):
    plt.figure(figsize=(12, 8))
    plt.plot([point[variables[0]] for point in points],
             [point[variables[1]] for point in points], '-o', color='black')
    plt.contourf(X, Y, Z, 15, cmap='winter', alpha=0.5)
    plt.colorbar()
    plt.title('Level lines, iteration points.')
    plt.xlabel(variables[0])
    plt.ylabel(variables[1])


def draw_lines(X, Y, points, variables):
    plt.figure(figsize=(12, 8))
    plt.plot([point[0] for point in points],
             [point[1] for point in points], '-o', color='black', label='Min points')
    plt.plot(X, Y, alpha=0.5)
    plt.xlabel(variables[0])
    plt.ylabel('y')


def draw_feasible_polygon(X, Y, g, variables):
    c = [sympy.lambdify(variables, preproc_fun(g_i)) for g_i in g]
    plt.imshow(
        functools.reduce(lambda one, two: one & (two(X, Y) >= 0), c, True),
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        origin='lower',
        cmap='summer')
    plt.title('Level lines, iteration points and feasible set.')
