import sympy
import numpy as np
from plotting_3d import make_3d_plot, make_3d_plot_lagrange, make_level_lines_plot
from utils_check import preproc_fun, check_constraints
from extrema_methods import find_extremum, find_extremum_lagrange


def find_local_2var(vars_, fun_anl, constr=None, plot=False, plot_ll=False):
    """Finds local extrema of a function of two variables.

    Positional arguments:
    vars_ -- list of variable names
    fun_anl -- function analytic form

    Keyword arguments:
    constr -- list (list of tuples) of function constraints(default=None)
    plot -- Draw plot with surface and points(default=False)
    plot_ll -- Draw level lines(default=False)
    """
    var1, var2 = sympy.symbols(' '.join(vars_))
    fun = preproc_fun(fun_anl)

    if constr is None:
        m_points = sympy.solve([fun.diff(var1), fun.diff(var2)], dict=True)
    else:
        m_points = check_constraints(sympy.solve([fun.diff(var1), fun.diff(var2)], dict=True), constr, vars_)

    if m_points:
        for m in m_points:
            coord1, coord2, label = find_extremum(fun, var1, var2, m)
            print(f'Point: ({float(coord1)}, {float(coord2)}) | Z: {float(fun.subs(m))} | Extrema: {label}')
    else:
        print('No extrema has been found! ')
        need_draw = input('Draw graphs? (yes/no): ').lower()
        if not(need_draw and need_draw != 'no'):
            return None

    X, Y, Z = None, None, None
    if plot:
        X = np.arange(-10, 10, 0.5)
        Y = np.arange(-10, 10, 0.5)
        X, Y = np.meshgrid(X, Y)
        try:
            Z = np.array([[float(fun.subs([('x', x), ('y', y)])) for x, y in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)])
        except TypeError:
            print('Looks like the function is not differentiable!')
            return None
        make_3d_plot(X, Y, Z, m_points, var1, var2, fun)

    if plot_ll:
        if not((X is None) and (Y is None) and (Z is None)):
            make_level_lines_plot(X, Y, Z, m_points, var1, var2)
        else:
            # if constr:
            #     X = np.linspace(*constr[0], 50)
            #     Y = np.linspace(*constr[1], 50)
            # else:
            X = np.arange(-10, 10, 0.5)
            Y = np.arange(-10, 10, 0.5)
            X, Y = np.meshgrid(X, Y)
            try:
                Z = np.array([[float(fun.subs([('x', x), ('y', y)])) for x, y in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)])
            except TypeError:
                raise Error('Looks like the function is not differentiable!')
            make_level_lines_plot(X, Y, Z, m_points, var1, var2)
    return None


def find_lagrange_2var(vars_, fun_anl, fun_constr, constr=None, plot=False, plot_ll=False):
    """Finds extrema of a function of two variables by Lagrange method.

    Positional arguments:
    vars_ -- list of variable names
    fun_anl -- function analytic form
    fun_constr -- limiting function analytic form

    Keyword arguments:
    constr -- list (list of tuples) of function constraints(default=None)
    plot -- Draw plot with surface and points(default=False)
    plot_ll -- Draw level lines(default=False)
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

    if m_points:
        for m in m_points:
            coord1, coord2, label = find_extremum_lagrange(fun_c, lagrange, m, var1, var2)
            print(f'Point: ({float(coord1)}, {float(coord2)}) | Z: {float(fun.subs(m))} | Extrema: {label}')
    else:
        print('No extrema has been found! ')
        need_draw = input('Draw graphs? (yes/no): ').lower()
        if not(need_draw and need_draw != 'no'):
            return None

    Z = None
    if plot:
        X = np.arange(-10, 10, 0.5)
        Y = np.arange(-10, 10, 0.5)
        X, Y = np.meshgrid(X, Y)
        try:
            Z1 = np.array([[float(fun.subs([('x', x), ('y', y)])) for x, y in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)])
            Z2 = np.array([[float(fun_c.subs([('x', x), ('y', y)])) for x, y in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)])
        except TypeError:
            print('Looks like the function is not differentiable!')
            return None
        make_3d_plot_lagrange(X, Y, Z1, Z2, m_points, var1, var2, fun)

    if plot_ll:
        # if constr:
        #     X = np.linspace(*constr[0], 50)
        #     Y = np.linspace(*constr[1], 50)
        # else:
        X = np.arange(-10, 10, 0.5)
        Y = np.arange(-10, 10, 0.5)
        X, Y = np.meshgrid(X, Y)
        if not(Z is None):
            make_level_lines_plot(X, Y, Z, m_points, var1, var2, fun_c)
        else:
            try:
                Z = np.array([[float(fun.subs([('x', x), ('y', y)])) for x, y in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)])
            except TypeError:
                raise Error('Looks like the function is not differentiable!')
            make_level_lines_plot(X, Y, Z, m_points, var1, var2, fun_c)
    return None
