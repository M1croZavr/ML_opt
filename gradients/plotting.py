from twovarextremas import plotting_3d, utils_twovarextremas
from onedimensionaloptimization import plotting
import numpy as np
import functools
import sympy


def plotter(left, right, n_points=100):
    """Decorator makes result plot for optimization method if parameter plot is True

    Positional arguments:
    left -- left bound of result plot
    right -- right bound of result plot
    """
    def plotting_decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            res = f(*args, **kwargs)
            if 'plot' in kwargs.keys():
                plot_flag = kwargs['plot']
            elif len(args) >= 5:
                plot_flag = args[4]
            else:
                plot_flag = False
            fun = utils_twovarextremas.preproc_fun(args[0]) if args \
                                                            else utils_twovarextremas.preproc_fun(kwargs['fun_anl'])
            variables = tuple(fun.atoms(sympy.Symbol))
            if plot_flag and len(variables) == 2:
                X, Y = np.meshgrid(np.linspace(left, right, n_points), np.linspace(left, right, n_points))
                Z = np.array(
                    [[float(fun.subs([(variables[0], x), (variables[1], y)])) for x, y in zip(x_i, y_i)] for x_i, y_i in
                     zip(X, Y)])
                plotting_3d.make_3d_plot(X, Y, Z, [res[1]], variables[0], variables[1], fun)
                plotting_3d.make_level_lines_plot(X, Y, Z, [res[1]], variables[0], variables[1])
            elif plot_flag and len(variables) == 1:
                X = np.linspace(left, right, n_points)
                Y = np.array([float(fun.subs(variables[0], x)) for x in X])
                plotting.plot_scatter_and_line(X, Y, list(res[1].values()), [res[0]], [str(variables[0])])
            else:
                print('Cannot draw plot for this function.')
            return res
        return wrapper
    return plotting_decorator
