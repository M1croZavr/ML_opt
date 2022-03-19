import sympy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from twovarextremas import utils_twovarextremas
from .utils_onedimensionaloptimization import update_vars, sign, parabolic_approximation, spi_behaving
from .plotting import plot_scatter_and_line, plot_parabola_and_line
from sympy.utilities.lambdify import lambdify
from scipy.optimize import line_search


def golden_section_search(fun_anl, x_l, x_r,
                          eps=1e-5, max_iter=500, print_info=False, record_info=False, plot=False):
    """Optimizes function of 1 variable by golden section method.

    Positional arguments:
    fun_anl -- function analytic form
    x_l -- left bound of the interval
    x_r -- right bound of the interval

    Keyword arguments:
    eps -- float or int. Precision of result(default=1e-5)
    max_iter -- Maximum iteration of algorithm(default=500)
    print_info -- Print information each iteration(default=False)
    record_info -- Make pd.DataFrame with recorder information(default=False)
    plot -- Draw plot(default=False)
    """
    fun = utils_twovarextremas.preproc_fun(fun_anl)
    var, = fun.atoms(sympy.Symbol)
    phi = (1 + np.sqrt(5)) / 2
    a, b = float(x_l), float(x_r)
    c, d = b + (a - b) / phi, a + (b - a) / phi
    fc, fd = fun.subs(var, c), fun.subs(var, d)
    if plot:
        X = np.linspace(a, b, 30)
        Y = [fun.subs(var, x_i) for x_i in X]
        plot_scatter_and_line(X, Y, [a, b, c, d],
                              [fun.subs(var, a), fun.subs(var, b), fc, fd],
                              ['a', 'b', 'c', 'd'])
    if record_info:
        df = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'fc', 'fd'])
    else:
        df = None
    for i in range(max_iter):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b + (a - b) / phi
            fc = fun.subs(var, c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) / phi
            fd = fun.subs(var, d)
        if plot:
            plot_scatter_and_line(X, Y, [a, b, c, d],
                                  [fun.subs(var, a), fun.subs(var, b), fc, fd],
                                  ['a', 'b', 'c', 'd'])
        if print_info:
            print(f'k: {i + 1} | a: {a} | b: {b} | c: {c} | d: {d} | fc: {fc} | fd: {fd}')
        if record_info:
            df = df.append({'a': a, 'b': b, 'c': c, 'd': d, 'fc': fc, 'fd': fd}, ignore_index=True)
        if abs(b - a) < eps:
            print('Найдено значение с заданной точностью')
            break
    else:
        print('Достигнуто максимальное количество итераций')
    res_f, res_x = fun.subs(var, (a + b) / 2), (a + b) / 2
    print(f'f(x) = {res_f}, x = {res_x}')
    if record_info:
        df.index.name = 'k'
        return res_f, res_x, df
    else:
        return res_f, res_x


def parabola_method(fun_anl, x_l, x_r,
                    eps=1e-5, max_iter=500, print_info=False, record_info=False, plot=False):
    """Optimizes function of 1 variable by parabola method.

    Positional arguments:
    fun_anl -- function analytic form
    x_l -- left bound of the interval
    x_r -- right bound of the interval

    Keyword arguments:
    eps -- float or int. Precision of result(default=1e-5)
    max_iter -- Maximum iteration of algorithm(default=500)
    print_info -- Print information each iteration(default=False)
    record_info -- Make pd.DataFrame with recorder information(default=False)
    plot -- Draw plot(default=False)
    """
    fun = utils_twovarextremas.preproc_fun(fun_anl)
    var, = fun.atoms(sympy.Symbol)
    x1, x3 = float(x_l), float(x_r)
    u, f_u = [], None
    f1, f3 = fun.subs(var, x_l), fun.subs(var, x_r)
    for possible_x2 in np.linspace(x1, x3, 15):
        f2 = fun.subs(var, possible_x2)
        if (x1 < possible_x2 < x3) and (f1 > f2 and f3 > f2):
            x2 = possible_x2
            break
    else:
        print('Выполнено с ошибкой (нет точки минимума)')
        return None
    if plot:
        X = np.linspace(x1, x3, 30)
        Y = [fun.subs(var, x_i) for x_i in X]
    if record_info:
        df = pd.DataFrame(columns=['x1', 'x2', 'x3', 'f_u', 'u'])
    else:
        df = None
    for i in range(max_iter):
        u_temp, a, b, c = parabolic_approximation(x1, x2, x3, f1, f2, f3)
        u.append(u_temp)
        f_u = fun.subs(var, u[-1])
        x1, x2, x3, f1, f2, f3 = update_vars(x1, x2, x3, u[-1], f1, f2, f3, f_u)
        if len(u) > 1 and abs(u[-2] - u[-1]) < eps:
            print('Найдено значение с заданной точностью')
            break
        if print_info:
            print(f'k: {i + 1} | x1: {x1} | x2: {x2} | x3: {x3} | f_u: {f_u} | u: {u[-1]}')
        if record_info:
            df = df.append({'x1': x1, 'x2': x2, 'x3': x3, 'f_u': f_u, 'u': u[-1]}, ignore_index=True)
        if plot:
            plot_parabola_and_line(X, Y, u[-1], a, b, c)
    else:
        print('Достигнуто максимальное количество итераций')
    res_f, res_x = f_u, u[-1]
    print(f'f(x) = {res_f}, x = {res_x}')
    if record_info:
        df.index.name = 'k'
        return res_f, res_x, df
    else:
        return res_f, res_x


def brent_method(fun_anl, x_l, x_r,
                 eps=1e-5, max_iter=500, print_info=False, record_info=False, plot=False):
    """Optimizes function of 1 variable by Brent's method.

    Positional arguments:
    fun_anl -- function analytic form
    x_l -- left bound of the interval
    x_r -- right bound of the interval

    Keyword arguments:
    eps -- float or int. Precision of result(default=1e-5)
    max_iter -- Maximum iteration of algorithm(default=500)
    print_info -- Print information each iteration(default=False)
    record_info -- Make pd.DataFrame with recorder information(default=False)
    plot -- Draw plot(default=False)
    """
    fun = utils_twovarextremas.preproc_fun(fun_anl)
    var, = fun.atoms(sympy.Symbol)
    # left and right flexible variables
    a, b = float(x_l), float(x_r)
    # Golden section variables
    phi1 = (3 - np.sqrt(5)) / 2
    phi2 = (np.sqrt(5) - 1) / 2
    # initialize v w x
    v = w = x = a + phi1 * (b - a)
    fv = fw = fx = fun.subs(var, x)
    d = e = b - a
    if plot:
        X = np.linspace(a, b, 30)
        Y = [fun.subs(var, x_i) for x_i in X]
    if record_info:
        df = pd.DataFrame(columns=['a', 'b', 'x', 'w', 'v', 'fx', 'method'])
    else:
        df = None
    for i in range(max_iter):
        g = e
        e = d
        spi_condition = spi_behaving(v, w, x, fv, fw, fx, a, b, g, eps)
        if spi_condition:
            u, a_, b_, c_ = spi_condition
            method = 'Parabolic'
        else:
            method = 'Golden search'
            m = a + (b - a) / 2
            if x >= m:
                u = phi2 * x + phi1 * a  # Золотое сечение [x, b]
            else:
                u = phi2 * x + phi1 * b  # Золотое сечение [a, x]
        d = abs(u - x)
        fu = fun.subs(var, u)
        if fu <= fx:
            if u >= x:
                a = x
                x = u
                fx = fu
                v = w
                fv = fw
                w = a
                fw = fun.subs(var, w)
            else:
                b = x
                x = u
                fx = fu
                v = w
                fv = fw
                w = b
                fw = fun.subs(var, w)
        else:
            if u >= x:
                b = u
                v = w
                fv = fw
                w = b
                fw = fun.subs(var, w)

            else:
                a = u
                v = w
                fv = fw
                w = a
                fw = fun.subs(var, w)
        if plot:
            if spi_condition:
                plot_parabola_and_line(X, Y, u, a_, b_, c_)
            else:
                plot_scatter_and_line(X, Y, [a, b, w, v, x],
                                      [fun.subs(var, a), fun.subs(var, b), fw, fv, fx],
                                      ['a', 'b', 'w', 'v', 'x'])
        if print_info:
            print(f'k: {i + 1} | a: {a} | b: {b} | x: {x} | w: {w} | v: {v} | fx: {fx} | method: {method}')
        if record_info:
            df = df.append({'a': a, 'b': b, 'x': x, 'w': w, 'v': v, 'fx': fx, 'method': method}, ignore_index=True)
        if (b - a) <= eps:
            print('Найдено значение с заданной точностью')
            break
    else:
        print('Достигнуто максимальное количество итераций')
    res_f, res_x = fun.subs(var, (a + b) / 2), (a + b) / 2
    print(f'f(x) = {res_f}, x = {res_x}')
    if record_info:
        df.index.name = 'k'
        return res_f, res_x, df
    else:
        return res_f, res_x


def bfgs(fun_anl, x0,
         c1=0.0001, c2=0.1, x_max=100, threshold=1e-5, max_iter=500, print_info=False, record_info=False):
    """Optimizes function of 1 variable by Broyden — Fletcher — Goldfarb — Shanno algorithm.

    Positional arguments:
    fun_anl -- function analytic form
    x0 -- starting point

    Keyword arguments:
    c1 -- first constant of the Wolfe condition(default=0.0001)
    c2 -- second constant of the Wolfe condition(default=0.1)
    x_max -- max value of function argument(default=100)
    threshold -- exit threshold by search interval length(default=1e-8)
    max_iter -- Maximum iteration of algorithm(default=500)
    print_info -- Print information each iteration(default=False)
    record_info -- Make pd.DataFrame with recorder information(default=False)
    plot -- Draw plot(default=False)
    """
    fun = utils_twovarextremas.preproc_fun(fun_anl)
    var, = fun.atoms(sympy.Symbol)
    lambda_fun = lambdify(var, fun)
    lambda_fun_gradient = lambdify(var, fun.diff(var))
    x = float(x0)
    grf = lambda_fun_gradient(x)
    C = 1
    if record_info:
        df = pd.DataFrame(columns=['x', 'alpha', 'p', 's', 'y', 'c', 'delta_fun'])
    else:
        df = None
    for i in range(max_iter):
        try:
            # 0.01 - kind of learning rate
            p = 0.01 * (-C * grf)
            # Find alpha that satisfies strong Wolfe conditions
            alpha = line_search(lambda_fun, lambda_fun_gradient, x, p, c1=c1, c2=c2, gfk=grf)[0]
            x_next = x + alpha * p
            s = x_next - x
            y = lambda_fun_gradient(x_next) - grf
            ro = 1.0 / (y * s)
            C = ((1 - ro * s * y) * C * (1 - ro * y * s)) + ro * s * s
            x = x_next
            grf = lambda_fun_gradient(x)
        except Exception:
            print('Выполнено с ошибкой')
            return None
        if print_info:
            print(f'k: {i + 1} | x: {x} | alpha: {alpha} | p: {p} | s: {s} | y: {y} | c: {C} | delta_fun: {grf}')
        if record_info:
            df = df.append({'x': x, 'alpha': alpha, 'p': p, 's': s, 'y': y, 'c': C, 'delta_fun': grf},
                           ignore_index=True)
        if x > x_max:
            print('Достигнуто ограничение на максимально возможное значение аргумента')
            break
        if abs(grf) <= threshold:
            print('Точка, удовлетворяющая условию Вольфе, найдена с заданной точностью')
            break
    else:
        print('Достигнуто максимальное количество итераций')
    res_f, res_x = lambda_fun(x), x
    print(f'f(x) = {res_f}, x = {res_x}')
    if record_info:
        df.index.name = 'k'
        return res_f, res_x, df
    else:
        return res_f, res_x
