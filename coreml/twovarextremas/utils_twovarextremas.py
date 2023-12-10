import re

import numpy as np
from sympy.parsing.sympy_parser import standard_transformations, \
    implicit_multiplication_application, parse_expr


class NonDifferentialError(Exception):
    pass


def preproc_fun(fun_anl):
    """Return preprocessed function as sympy object function."""
    fun = fun_anl.replace('^', '**').replace('–', '-')
    fun = fun.replace('pi', str(np.pi)).replace('sympy.pi', str(np.pi))
    assert not('=' in fun), '"=" symbol is not allowed!'
    logs = re.findall(r'log[\d\s]+\([^).]+\)', fun)
    for log in logs:
        fun = fun.replace(log, f'log({log[log.find("(") + 1:log.find(")")] + ", " + log[3:log.find("(")].strip()})')
    transformations = (standard_transformations + (implicit_multiplication_application,))
    fun = parse_expr(fun, transformations=transformations)
    return fun


def check_constraints(init_points, constr, vars_):
    """Return filtered by constraints conditions stationary points.

    Positional arguments:
    init_points -- saddle points of equation. Dict like {name: coordinate}
    constr -- list (list of tuples) of constraints on each variable
    vars_ -- list of variable names
    """
    m_points = []
    for m in init_points:
        for (coord_name, coord), coord_constraint in zip(m.items(), constr):
            if not(coord_constraint[0] <= coord <= coord_constraint[1]):
                break
            elif not(str(coord_name) in vars_):
                continue
        else:
            m_points.append(m)
    return m_points


def create_xyz(left, right, step, constr, fun, var1='x', var2='y'):
    """Return X, Y, Z arrays created with meshgrid and with applied function.

    Positional arguments:
    left -- value 'from' in generating X and Y
    right -- value 'to' in generating X and Y
    step -- step in generating X and Y
    constr -- constraints on x and y variable
    fun -- function in analytic form
    var1 -- variable 1(default='x')
    var2 -- variable 2(default='y')
    """
    X = np.arange(left, right, step)
    Y = np.arange(left, right, step)
    X, Y = np.meshgrid(X, Y)
    try:
        Z = np.array([[float(fun.subs([(var1, x), (var2, y)])) for x, y in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)])
    except TypeError:
        try:
            X = np.arange(*constr[0], step)
            Y = np.arange(*constr[1], step)
            X, Y = np.meshgrid(X, Y)
            Z = np.array([[float(fun.subs([(var1, x), (var2, y)])) for x, y in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)])
        except TypeError:
            raise NonDifferentialError('Looks like the function is not differentiable in its domain or in R!')
    return X, Y, Z
