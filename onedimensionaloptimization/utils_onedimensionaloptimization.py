import sympy


def update_vars(x1, x2, x3, x_, f1, f2, f3, f_x_):
    """Updates variables in parabolic method."""
    if x1 < x_ < x2 < x3:
        if f_x_ > f2:
            x1 = x_
            f1 = f_x_
        else:
            x3 = x2
            f3 = f2
            x2 = x_
            f2 = f_x_
    elif x1 < x2 < x_ < x3:
        if f_x_ > f2:
            x3 = x_
            f3 = f_x_
        else:
            x1 = x2
            f1 = f2
            x2 = x_
            f2 = f_x_
    return x1, x2, x3, f1, f2, f3


def sign(x):
    """Sign function, returns sign of x"""
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0


def parabolic_approximation(x1, x2, x3, f1, f2, f3):
    """Calculate parameters of parabolic equation.
    :return u (top of a parabola), a, b, c (parameters)"""
    if x1 > x3:
        x3, x1 = x1, x3
        f3, f1 = f1, f3
    a, b, c = sympy.symbols('a b c')
    cfs = sympy.solve([a * x1 ** 2 + b * x1 + c - f1, a * x2 ** 2 + b * x2 + c - f2, a * x3 ** 2 + b * x3 + c - f3],
                      [a, b, c],
                      dict=True)[0]
    u = -cfs[b] / (2 * cfs[a])
    return u, cfs[a], cfs[b], cfs[c]


def spi_behaving(v, w, x, fv, fw, fx, x_l, x_r, g):
    if (x != w) and (x != v) and (w != v):
        u, a, b, c = parabolic_approximation(v, x, w, fv, fx, fw)  # Параболическая аппроксимация, находим u
        if (x_l <= u <= x_r) and (abs(u - x) <= abs(g) / 2):
            return u, a, b, c
    return False



