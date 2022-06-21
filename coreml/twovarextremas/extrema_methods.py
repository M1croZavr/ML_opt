import numpy as np


def find_extrema(fun, var1, var2, m):
    """Return coordinates and kind of extrema or additional research required otherwise."""
    A = fun.diff(var1).diff(var1).subs([(var1, m[var1]), (var2, m[var2])])
    B = fun.diff(var1).diff(var2).subs([(var1, m[var1]), (var2, m[var2])])
    C = fun.diff(var2).diff(var2).subs([(var1, m[var1]), (var2, m[var2])])
    condition = A * C - B ** 2
    if condition < 0:
        label = 'saddle point'
    elif condition == 0:
        label = 'additional research required'
    else:
        if A > 0:
            label = 'Minimum'
        elif A < 0:
            label = 'Maximum'
    return m[var1], m[var2], label


def find_extrema_lagrange(fun_c, lagrange, m, var1, var2):
    """Return coordinates and kind of extrema found by matrix forms."""
    fun_c_var1 = float(fun_c.diff(var1).subs(m))
    fun_c_var2 = float(fun_c.diff(var2).subs(m))
    l_11 = float(lagrange.diff(var1).diff(var1).subs(m))
    l_12 = float(lagrange.diff(var1).diff(var2).subs(m))
    l_22 = float(lagrange.diff(var2).diff(var2).subs(m))
    A = np.array([[0, fun_c_var1, fun_c_var2],
                  [fun_c_var1, l_11, l_12],
                  [fun_c_var2, l_12, l_22]])
    det = np.linalg.det(A)
    if det > 0:
        label = 'Maximum'
    elif det < 0:
        label = 'Minimum'
    else:
        label = 'Saddle point'
    return m[var1], m[var2], label
