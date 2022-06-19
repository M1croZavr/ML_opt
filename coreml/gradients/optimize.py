import sympy
import pandas as pd
import numpy as np
from ..twovarextremas import utils_twovarextremas
from ..twovarextremas import plotting_3d
from .plotting import plotter
from ..onedimensionaloptimization import optimize as one_optimization, plotting
from sympy import lambdify, derive_by_array
from scipy.optimize import fmin_cg


__all__ = ['constant_descent', 'splitting_step_descent', 'steepest_descent', 'conjugate_gradients']


@plotter(-10, 10, 40)
def constant_descent(fun_anl, lr,
                     eps=1e-5, max_iter=500,
                     print_info=False, plot=False, plot_iteratively=False, record_info=False, x_init=None):
    """Optimizes function of n variables by gradient descent method with constant step.

    Positional arguments:
    fun_anl -- function analytic form
    lr -- constant learning rate for descent

    Keyword arguments:
    eps -- float or int. Precision of result(default=1e-5)
    max_iter -- Maximum iteration of algorithm(default=500)
    print_info -- Print information each iteration(default=False)
    plot -- Draw plot(default=False), plot is available if less than 3 variables in function
    plot_iteratively -- Draw plot iteratively(default=False), plot is available if less than 3 variables in function
    record_info -- Make pd.DataFrame with recorder information(default=False)
    x_init -- Initial x vector(default=None), leave None to initialize by ones
    """
    fun = utils_twovarextremas.preproc_fun(fun_anl)
    variables = tuple(fun.atoms(sympy.Symbol))
    fun_grad = lambdify(variables, derive_by_array(fun, variables))
    if x_init is None:
        x_i = np.ones((len(variables), ))
    else:
        x_i = np.array(x_init)
        assert len(x_i) == len(variables), 'Initial x vector does not match number of variables in function'
    x_i_1 = None
    if record_info:
        df = pd.DataFrame(columns=variables)
    else:
        df = None
    if plot_iteratively:
        if len(variables) == 1:
            X = np.linspace(-10, 10, 40)
            Y = np.array([float(fun.subs(variables[0], x)) for x in X])
        elif len(variables) == 2:
            X, Y = np.meshgrid(np.linspace(-10, 10, 40), np.linspace(-10, 10, 40))
            Z = np.array(
                [[float(fun.subs([(variables[0], xx), (variables[1], yy)])) for xx, yy in zip(x, y)] for x, y in
                 zip(X, Y)])
    for i in range(max_iter):
        x_i_1 = x_i - lr * np.array(fun_grad(*x_i))
        if print_info:
            print(('k: {} | ' + ' : {} | '.join(map(str, variables)) + ' : {}').format(i + 1, *x_i_1))
        if record_info:
            df = df.append(dict(zip(variables, x_i_1)), ignore_index=True)
        if plot_iteratively and i % 5 == 0:
            if len(variables) == 1:
                x_d = dict(zip(variables, x_i_1))
                plotting.plot_scatter_and_line(X, Y, list(x_d.values()),
                                               [fun.subs(x_d)], [str(variables[0])])
            if len(variables) == 2:
                x_d = dict(zip(variables, x_i_1))
                plotting_3d.make_level_lines_plot(X, Y, Z, [x_d], variables[0], variables[1])
        if np.sum(np.sqrt((x_i_1 - x_i) ** 2)) <= eps:
            print('Найдено значение с заданной точностью')
            break
        else:
            x_i = x_i_1
    else:
        print('Достигнуто максимальное количество итераций')
    res_x = dict(zip(variables, x_i_1))
    res_f = fun.subs(res_x)
    print(f'f(X) = {res_f}, X = {res_x}')
    if record_info:
        df.index.name = 'k'
        return res_f, res_x, df
    else:
        return res_f, res_x


@plotter(-10, 10, 40)
def splitting_step_descent(fun_anl, lr, e, d,
                           eps=1e-5, plot=False, plot_iteratively=False,
                           max_iter=500, print_info=False, record_info=False, x_init=None):
    """Optimizes function of n variables by gradient descent method with splitting step.

    Positional arguments:
    fun_anl -- function analytic form
    lr -- initializing learning rate for descent
    e -- constant which helps to choose lr each step
    d -- constant which splits lr each step

    Keyword arguments:
    eps -- float or int. Precision of result(default=1e-5)
    plot -- Draw plot(default=False)
    plot_iteratively -- Draw plot iteratively(default=False), plot is available if less than 3 variables in function
    max_iter -- Maximum iteration of algorithm(default=500)
    print_info -- Print information each iteration(default=False)
    record_info -- Make pd.DataFrame with recorder information(default=False)
    x_init -- Initial x vector(default=None), leave None to initialize by ones
    """
    fun = utils_twovarextremas.preproc_fun(fun_anl)
    variables = tuple(fun.atoms(sympy.Symbol))
    fun_grad = lambdify(variables, derive_by_array(fun, variables))
    if x_init is None:
        x_i = np.ones((len(variables), ))
    else:
        x_i = np.array(x_init)
        assert len(x_i) == len(variables), 'Initial x vector does not match number of variables in function'
    x_i_1 = None
    if record_info:
        df = pd.DataFrame(columns=variables)
    else:
        df = None
    if plot_iteratively:
        if len(variables) == 1:
            X = np.linspace(-10, 10, 40)
            Y = np.array([float(fun.subs(variables[0], x)) for x in X])
        elif len(variables) == 2:
            X, Y = np.meshgrid(np.linspace(-10, 10, 40), np.linspace(-10, 10, 40))
            Z = np.array(
                [[float(fun.subs([(variables[0], xx), (variables[1], yy)])) for xx, yy in zip(x, y)] for x, y in
                 zip(X, Y)])
    for i in range(max_iter):
        x_i_1 = x_i - lr * np.array(fun_grad(*x_i))
        while fun.subs(dict(zip(variables, x_i - lr * np.array(fun_grad(*x_i))))) > (fun.subs(dict(zip(variables, x_i)))
                                                                                     - e * lr * np.sum(np.array(fun_grad(*x_i))) ** 2):
            lr *= d
        if print_info:
            print(('k: {} | ' + ' : {} | '.join(map(str, variables)) + ' : {}').format(i + 1, *x_i_1))
        if record_info:
            df = df.append(dict(zip(variables, x_i_1)), ignore_index=True)
        if plot_iteratively and i % 5 == 0:
            if len(variables) == 1:
                x_d = dict(zip(variables, x_i_1))
                plotting.plot_scatter_and_line(X, Y, list(x_d.values()),
                                               [fun.subs(x_d)], [str(variables[0])])
            if len(variables) == 2:
                x_d = dict(zip(variables, x_i_1))
                plotting_3d.make_level_lines_plot(X, Y, Z, [x_d], variables[0], variables[1])
        if np.sum(np.sqrt((x_i_1 - x_i) ** 2)) <= eps:
            print('Найдено значение с заданной точностью')
            break
        else:
            x_i = x_i_1
    else:
        print('Достигнуто максимальное количество итераций')
    res_x = dict(zip(variables, x_i_1))
    res_f = fun.subs(res_x)
    print(f'f(X) = {res_f}, X = {res_x}')
    if record_info:
        df.index.name = 'k'
        return res_f, res_x, df
    else:
        return res_f, res_x


@plotter(-10, 10, 40)
def steepest_descent(fun_anl,
                     eps=1e-5, max_iter=500, print_info=False, record_info=False, plot=False,
                     plot_iteratively=False, x_init=None, lr_finder='golden_search'):
    """Optimizes function of n variables by gradient descent method with solving one-dimensional optimization problem
     each iteration to figure out suitable learning rate.

    Positional arguments:
    fun_anl -- function analytic form

    Keyword arguments:
    eps -- float or int. Precision of result(default=1e-5)
    max_iter -- Maximum iteration of algorithm(default=500)
    print_info -- Print information each iteration(default=False)
    record_info -- Make pd.DataFrame with recorder information(default=False)
    plot -- Draw plot(default=False)
    plot_iteratively -- Draw plot iteratively(default=False), plot is available if less than 3 variables in function
    x_init -- Initial x vector(default=None), leave None to initialize by ones
    lr_fined -- One dimensional optimization solver(parabolic, brent, golden_search)(default=golden_search)
    """
    fun = utils_twovarextremas.preproc_fun(fun_anl)
    variables = tuple(fun.atoms(sympy.Symbol))
    fun_grad = lambdify(variables, derive_by_array(fun, variables))
    if x_init is None:
        x_i = np.ones((len(variables), ))
    else:
        x_i = np.array(x_init)
        assert len(x_i) == len(variables), 'Initial x vector does not match number of variables in function'
    x_i_1 = None
    lm = sympy.Symbol('l')
    if record_info:
        df = pd.DataFrame(columns=list(variables) + [lm])
    else:
        df = None
    if plot_iteratively:
        if len(variables) == 1:
            X = np.linspace(-10, 10, 40)
            Y = np.array([float(fun.subs(variables[0], x)) for x in X])
        elif len(variables) == 2:
            X, Y = np.meshgrid(np.linspace(-10, 10, 40), np.linspace(-10, 10, 40))
            Z = np.array(
                [[float(fun.subs([(variables[0], xx), (variables[1], yy)])) for xx, yy in zip(x, y)] for x, y in
                 zip(X, Y)])
    for i in range(max_iter):
        if np.all(np.array(fun_grad(*x_i)) == 0):
            print('Градиент вышел на плато')
            break
        if lr_finder == 'golden_search':
            _, lr = one_optimization.golden_section_search(str(fun.subs(dict(zip(variables, x_i - lm * np.array(fun_grad(*x_i)))))),
                                                           0, 1)
        elif lr_finder == 'parabolic':
            _, lr = one_optimization.parabola_method(str(fun.subs(dict(zip(variables, x_i - lm * np.array(fun_grad(*x_i)))))),
                                                     0, 1)
        elif lr_finder == 'brent':
            _, lr = one_optimization.brent_method(str(fun.subs(dict(zip(variables, x_i - lm * np.array(fun_grad(*x_i)))))),
                                                  0, 1)
        lr = float(lr)
        x_i_1 = x_i - lr * np.array(fun_grad(*x_i))
        if print_info:
            print(('k: {} | ' + ' : {} | '.join(map(str, variables)) + ' : {}' + ' lr: {}').format(i + 1, *x_i_1, lr))
        if record_info:
            df = df.append(dict(zip(list(variables) + [lm], list(x_i_1) + [lr])), ignore_index=True)
        if plot_iteratively and i % 5 == 0:
            if len(variables) == 1:
                x_d = dict(zip(variables, x_i_1))
                plotting.plot_scatter_and_line(X, Y, list(x_d.values()),
                                               [fun.subs(x_d)], [str(variables[0])])
            if len(variables) == 2:
                x_d = dict(zip(variables, x_i_1))
                plotting_3d.make_level_lines_plot(X, Y, Z, [x_d], variables[0], variables[1])
        if np.sum(np.sqrt((x_i_1 - x_i) ** 2)) <= eps:
            print('Найдено значение с заданной точностью')
            break
        else:
            x_i = x_i_1
    else:
        print('Достигнуто максимальное количество итераций')
    res_x = dict(zip(variables, x_i_1))
    res_f = fun.subs(res_x)
    print(f'f(X) = {res_f}, X = {res_x}')
    if record_info:
        df.index.name = 'k'
        return res_f, res_x, df
    else:
        return res_f, res_x


@plotter(-10, 10, 40)
def conjugate_gradients(fun_anl,
                        eps=1e-5, max_iter=500,
                        print_info=False, record_info=False, plot=False, plot_iteratively=False, x_init=None):
    """Optimizes function of n variables by conjugate gradients method.

    Positional arguments:
    fun_anl -- function analytic form

    Keyword arguments:
    eps -- float or int. Precision of result(default=1e-5)
    max_iter -- Maximum iteration of algorithm(default=500)
    print_info -- Print information each iteration(default=False)
    record_info -- Make pd.DataFrame with recorder information(default=False)
    plot -- Draw plot(default=False)
    plot_iteratively -- Draw plot iteratively(default=False), plot is available if less than 3 variables in function
    x_init -- Initial x vector(default=None), leave None to initialize by ones
    """
    fun = utils_twovarextremas.preproc_fun(fun_anl)
    variables = tuple(fun.atoms(sympy.Symbol))
    fun_grad = lambdify(variables, derive_by_array(fun, variables))
    if x_init is None:
        x_i = np.ones((len(variables), ))
    else:
        x_i = np.array(x_init)
        assert len(x_i) == len(variables), 'Initial x vector does not match number of variables in function'

    if plot_iteratively:
        if len(variables) == 1:
            X = np.linspace(-10, 10, 40)
            Y = np.array([float(fun.subs(variables[0], x)) for x in X])

            def callback_plot(xk):
                x_d = dict(zip(variables, xk))
                plotting.plot_scatter_and_line(X, Y, list(x_d.values()),
                                               [fun.subs(x_d)], [str(variables[0])])

        elif len(variables) == 2:
            X, Y = np.meshgrid(np.linspace(-10, 10, 40), np.linspace(-10, 10, 40))
            Z = np.array(
                [[float(fun.subs([(variables[0], xx), (variables[1], yy)])) for xx, yy in zip(x, y)] for x, y in
                 zip(X, Y)])

            def callback_plot(xk):
                x_d = dict(zip(variables, xk))
                plotting_3d.make_level_lines_plot(X, Y, Z, [x_d], variables[0], variables[1])

    def f(x, *args):
        fun, variables, _ = args
        return float(fun.subs(dict(zip(variables, x))))

    def grad_f(x, *args):
        _, _, fun_grad = args
        return np.array(fun_grad(*x), dtype=np.float32)
    if record_info:
        df = pd.DataFrame(columns=list(variables))
    else:
        df = None
    res_x, res_f, _, _, warn, x_n = fmin_cg(f, x_i, grad_f, args=(fun, variables, fun_grad),
                                            callback=callback_plot if plot_iteratively is True else None,
                                            gtol=eps, maxiter=max_iter, full_output=True, retall=True, disp=False)
    if print_info:
        for i, values in enumerate(x_n):
            print(('k: {} | ' + ' : {} | '.join(map(str, variables)) + ' : {}').format(i + 1, *values))
    if warn == 0:
        print('Найдено значение с заданной точностью')
    elif warn == 1:
        print('Достигнуто максимальное количество итераций')
    else:
        print('Выполнено с ошибкой')
    res_x = dict(zip(variables, res_x))
    print(f'f(X) = {res_f}, X = {res_x}')
    if record_info:
        for values in x_n:
            df = df.append(dict(zip(variables, values)), ignore_index=True)
        df.index.name = 'k'
        return res_f, res_x, df
    else:
        return res_f, res_x
