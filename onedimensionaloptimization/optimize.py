import sympy
import pandas as pd
import numpy as np
from twovarextremas import utils_twovarextremas
from .utils_onedimensionaloptimization import update_vars, sign, parabolic_approximation, spi_behaving
from matplotlib import pyplot as plt


def golden_section_search(fun_anl, x_l, x_r, eps=1e-5, max_iter=500, print_info=False, record_info=False):
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
    """
    fun = utils_twovarextremas.preproc_fun(fun_anl)
    var, = fun.atoms(sympy.Symbol)
    phi = (1 + np.sqrt(5)) / 2
    x_l = float(x_l)
    x_r = float(x_r)
    c = x_r + (x_l - x_r) / phi
    d = x_l + (x_r - x_l) / phi
    fc = fun.subs(var, c)
    fd = fun.subs(var, d)
    if record_info:
        df = pd.DataFrame(columns=['x_l', 'x_r', 'c', 'd', 'fc', 'fd'])
    else:
        df = None
    for i in range(max_iter):
        if fc < fd:
            x_r = d
            d = c
            fd = fc
            c = x_r + (x_l - x_r) / phi
            fc = fun.subs(var, c)
        else:
            x_l = c
            c = d
            fc = fd
            d = x_l + (x_r - x_l) / phi
            fd = fun.subs(var, d)
        if print_info:
            print(f'k: {i + 1} | x_l: {x_l} | x_r: {x_r} | c: {c} | d: {d} | fc: {fc} | fd: {fd}')
        if record_info:
            df = df.append({'x_l': x_l, 'x_r': x_r, 'c': c, 'd': d, 'fc': fc, 'fd': fd}, ignore_index=True)
        if abs(x_r - x_l) < eps:
            print('Найдено значение с заданной точностью')
            break
    else:
        print('Достигнуто максимальное количество итераций')
    res_f, res_x = fun.subs(var, (x_l + x_r) / 2), (x_l + x_r) / 2
    if record_info:
        df.index.name = 'k'
        return res_f, res_x, df
    else:
        return res_f, res_x


def parabola_method(fun_anl, x_l, x_r, eps=1e-5, max_iter=500, print_info=False, record_info=False, plot=False):
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
    x1 = float(x_l)
    x3 = float(x_r)
    u = []
    f_u = None
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
        plt.figure(figsize=(10, 6))
        x = np.linspace(x_l, x_r, 15)
        y = [fun.subs(var, x_i) for x_i in x]
        plt.plot(x, y, 'b-')

    if record_info:
        df = pd.DataFrame(columns=['x1', 'x2', 'x3', 'f_u', 'u'])
    else:
        df = None

    for i in range(max_iter):
        u_temp, a, b, c = parabolic_approximation(x1, x2, x3, f1, f2, f3)
        u.append(u_temp)
        if i == 0:
            f_u = fun.subs(var, u[-1])
            x1, x2, x3, f1, f2, f3 = update_vars(x1, x2, x3, u[-1], f1, f2, f3, f_u)
        else:
            f_u = fun.subs(var, u[-1])
            if abs(u[-2] - u[-1]) < eps:
                print('Найдено значение с заданной точностью')
                break
            else:
                x1, x2, x3, f1, f2, f3 = update_vars(x1, x2, x3, u[-1], f1, f2, f3, f_u)
        if print_info:
            print(f'k: {i + 1} | x1: {x1} | x2: {x2} | x3: {x3} | f_u: {f_u} | u: {u[-1]}')
        if record_info:
            df = df.append({'x1': x1, 'x2': x2, 'x3': x3, 'f_u': f_u, 'u': u[-1]}, ignore_index=True)
        if plot:
            x = np.linspace(x_l, x_r, 15)
            y = [a * x_i ** 2 + b * x_i + c for x_i in x]
            plt.plot(x, y, 'r--')
    else:
        print('Достигнуто максимальное количество итераций')
    res_f, res_x = f_u, u[-1]
    if record_info:
        df.index.name = 'k'
        return res_f, res_x, df
    else:
        return res_f, res_x


# def brent_method(fun_anl, x_l, x_r, eps=1e-5, max_iter=500, print_info=False, record_info=False, plot=False):
#     """Optimizes function of 1 variable by Brent's method.
#
#     Positional arguments:
#     fun_anl -- function analytic form
#     x_l -- left bound of the interval
#     x_r -- right bound of the interval
#
#     Keyword arguments:
#     eps -- float or int. Precision of result(default=1e-5)
#     max_iter -- Maximum iteration of algorithm(default=500)
#     print_info -- Print information each iteration(default=False)
#     record_info -- Make pd.DataFrame with recorder information(default=False)
#     plot -- Draw plot(default=False)
#     """
#     fun = utils_twovarextremas.preproc_fun(fun_anl)
#     var, = fun.atoms(sympy.Symbol)
#
#     x_l = float(x_l)
#     x_r = float(x_r)
#     K = (3 - 5 ** 0.5) / 2
#     v = w = x = x_l + K * (x_r - x_l)
#     fv = fw = fx = fun.subs(var, x)
#     d = e = x_r - x_l
#     u = None
#     while (x_r - x_l) > eps:
#         g = e
#         e = d
#
#         if spi_behaving(v, w, x, fv, fw, fx, x_l, x_r, g):
#             u, a, b, c = parabolic_approximation(v, x, w, fv, fx, fw)
#             d = u - x
#         else:
#             if x < (x_r - x_l) / 2:
#                 u = x + K * (x_r - x)  # Золотое сечение [x, x_u]
#                 d = x_r - x
#             else:
#                 u = x - K * (x - x_l)  # Золотое сечение [x_l, x]
#                 d = x - x_l
#
#         fu = fun.subs(var, u)
#         if fu <= fx:
#             if u >= x:
#                 x_l = x
#             else:
#                 x_r = x
#             v, w, x, fv, fw, fx = w, x, u, fw, fx, fu
#         else:
#             if u >= x:
#                 x_r = u
#             else:
#                 x_l = u
#             if fu <= fw or w == x:
#                 v, w, fv, fw = w, u, fw, fu
#             elif fu <= fv or v == x or v == w:
#                 v, fv = u, fu
#         print(u, x)




def brent_method(fun_anl, x_l, x_r, eps=1e-5, max_iter=500, print_info=False, record_info=False, plot=False):
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
    x_l_init = float(x_l)
    x_r_init = float(x_r)
    x_l = float(x_l)
    x_r = float(x_r)
    K = (3 - 5 ** 0.5) / 2
    x = w = v = (x_l + x_r) / 2
    fx = fw = fv = fun.subs(var, x)
    d = e = x_r - x_l
    u, a, b, c = None, None, None, None

    if plot:
        plt.figure(figsize=(10, 6))
        x_plot = np.linspace(x_l_init, x_r_init, 50)
        y_plot = [fun.subs(var, x_i) for x_i in x_plot]
        plt.plot(x_plot, y_plot, 'b-')

    if record_info:
        df = pd.DataFrame(columns=['x_l', 'x_r', 'u', 'fu'])
    else:
        df = None

    for i in range(max_iter):
        g = e
        e = d
        if x_r - x_l < eps:
            print((x_l + x_r) / 2)
        if spi_behaving(v, w, x, fv, fw, fx, x_l, x_r, g):
                u, a, b, c = parabolic_approximation(v, x, w, fv, fx, fw)
                d = abs(u - x)


        else:
            if x < (x_r - x_l) / 2:
                u = x + K * (x_r - x)  # Золотое сечение [x, x_u]
                d = x_r - x
            else:
                u = x - K * (x - x_l)  # Золотое сечение [x_l, x]
                d = x - x_l

        if abs(u - x) < eps:
            u = x + sign(u - x) * eps  # Минимальная близость между x и u
        fu = fun.subs(var, u)

        if fu <= fx:
            if u >= x:
                x_l = x
            else:
                x_r = x
            v, w, x, fv, fw, fx = w, x, u, fw, fx, fu
        else:
            if u >= x:
                x_r = u
            else:
                x_l = u
            if fu <= fw or w == x:
                v, w, fv, fw = w, u, fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu

        if print_info:
            print(f'k: {i + 1} | x_l: {x_l} | x_r: {x_r} | u: {u} | fu: {fu}')
        if record_info:
            df = df.append({'x_l': x_l, 'x_r': x_r, 'u': u, 'fu': fu}, ignore_index=True)
        if plot and a and b and c:
            x_plot = np.linspace(x_l_init, x_r_init, 50)
            y_plot = [a * x_i ** 2 + b * x_i + c for x_i in x_plot]
            plt.plot(x_plot, y_plot, 'r--')

    else:
        print('Достигнуто максимальное количество итераций')
    res_f = fu
    res_x = u
    if record_info:
        df.index.name = 'k'
        return res_f, res_x, df
    else:
        return res_f, res_x

