import sympy
import numpy as np
from ML_opt_lib.plotting_3d import make_3d_plot, make_3d_plot_lagrange
from utils_check import preproc_fun, check_constraints
from ML_opt_lib.extrema_methods import find_extremum, find_extremum_lagrange


def find_local_2var(vars_, fun_anl, constr=None, plot=False):
    """Finds local extrema of a function of two variables.

    Positional arguments:
    vars_ -- list of variable names
    fun_anl -- function analytic form

    Keyword arguments:
    constr -- list (list of tuples) of function constraints(default=None)
    plot -- Draw plot with surface and points(default=False)
    """
    var1, var2 = sympy.symbols(' '.join(vars_))
    fun = preproc_fun(fun_anl)

    if constr is None:
        m_points = sympy.solve([fun.diff(var1), fun.diff(var2)], dict=True)
    else:
        m_points = check_constraints(sympy.solve([fun.diff(var1), fun.diff(var2)], dict=True), constr, vars_)

    for m in m_points:
        coord1, coord2, label = find_extremum(fun, var1, var2, m)
        print(f'Point: ({float(coord1)}, {float(coord2)}) | Extremum: {label}')

    if plot:
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = np.array([[float(fun.subs([('x', x), ('y', y)])) for x, y in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)])
        make_3d_plot(X, Y, Z, m_points, var1, var2, fun)
    return None


def find_lagrange_2var(vars_, fun_anl, fun_constr, constr=None, plot=False):
    """Finds extrema of a function of two variables by Lagrange method.

    Positional arguments:
    vars_ -- list of variable names
    fun_anl -- function analytic form
    fun_constr -- limiting function analytic form

    Keyword arguments:
    constr -- list (list of tuples) of function constraints(default=None)
    plot -- Draw plot with surface and points(default=False)
    """
    l, var1, var2 = sympy.symbols('lambda ' + ' '.join(vars_))
    fun = preproc_fun(fun_anl)
    fun_c = preproc_fun(fun_constr)
    lagrange = fun + l * fun_c
    if constr is None:
        m_points = sympy.solve([lagrange.diff(var1), lagrange.diff(var2), fun_c], dict=True)
    else:
        m_points = check_constraints(sympy.solve([lagrange.diff(var1), lagrange.diff(var2), fun_c], dict=True),
                                     # Искусственное ограничение на лямбду
                                     [(float('-inf'), float('+inf'))] + constr,
                                     vars_)

    for m in m_points:
        coord1, coord2, label = find_extremum_lagrange(fun_c, lagrange, m, var1, var2)
        print(f'Point: ({float(coord1)}, {float(coord2)}) | Extremum: {label}')

    if plot:
        X = np.arange(-10, 10, 0.25)
        Y = np.arange(-10, 10, 0.25)
        X, Y = np.meshgrid(X, Y)
        Z1 = np.array([[float(fun.subs([('x', x), ('y', y)])) for x, y in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)])
        Z2 = np.array([[float(fun_c.subs([('x', x), ('y', y)])) for x, y in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)])
        make_3d_plot_lagrange(X, Y, Z1, Z2, m_points, var1, var2, fun, fun_c)
    return None
