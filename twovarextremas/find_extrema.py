import sympy
import numpy as np
from plotting_3d import make_3d_plot, make_3d_plot_lagrange, make_level_lines_plot
from utils_check import preproc_fun, check_constraints, create_xyz
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
        print('No extrema has been found!')
        need_draw = input('Draw graphs? (yes/no): ').lower()
        if not(need_draw and need_draw != 'no'):
            return None

    if plot or plot_ll:
        X, Y, Z = create_xyz(-10, 10, 0.5, constr, fun, var1, var2)
        if plot:
            make_3d_plot(X, Y, Z, m_points, var1, var2, fun)
        if plot_ll:
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

    if plot or plot_ll:
        X, Y, Z = create_xyz(-10, 10, 0.5, constr, fun, var1, var2)
        _, _, Z_c = create_xyz(-10, 10, 0.5, constr, fun_c, var1, var2)
        if plot:
            make_3d_plot_lagrange(X, Y, Z, Z_c, m_points, var1, var2, fun)
        if plot_ll:
            make_level_lines_plot(X, Y, Z, m_points, var1, var2, Z_c)
    return None
