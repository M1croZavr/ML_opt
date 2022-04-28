from twovarextremas.utils_twovarextremas import preproc_fun
import numpy as np
import sympy
from scipy.optimize import minimize
from .plotting import draw_level_lines, draw_lines, draw_feasible_polygon


def newton_dual(fun, g, x_init, plot=True):
    """Optimization of a function with equality-type constraints.

    Positional arguments:
    fun -- str, function in analytic form
    g -- sequence(list, tuple ...) of str, sequence of equality type constraints functions in analytic form
    x_init -- numpy.ndarray, initial approximation

    Keyword arguments:
    plot -- bool, Draw plot(default=False)
    """
    fun = preproc_fun(fun)
    variables = list(fun.atoms(sympy.Symbol))
    constraints = []
    for g_i in g:
        left, right = tuple(map(lambda x: x.strip(), g_i.split('=')))

        def constraint(x, left=left, right=right):
            return sympy.lambdify(variables, preproc_fun(left) - preproc_fun(right))(*x)

        constraints.append({'type': 'eq', 'fun': constraint})
    f = sympy.lambdify(variables, fun)

    def callback(xk):
        x_history.append(xk)

    x_history = [x_init]
    minimize_result = minimize(lambda x: f(*x),
                               x_init,
                               constraints=constraints,
                               method='SLSQP',
                               callback=callback)

    if plot:
        if len(variables) == 2:
            X = np.linspace(-10, 10)
            Y = np.linspace(-10, 10)
            X, Y = np.meshgrid(X, Y)
            Z = np.array([[float(fun.subs([(variables[0], x), (variables[1], y)]))
                           for x, y in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)])
            points = []
            for x in x_history:
                points.append(dict(zip(variables, x)))
            draw_level_lines(X, Y, Z, points, variables)
        elif len(variables) == 1:
            X = np.linspace(-10, 10)
            Y = np.array([float(fun.subs(variables[0], x_i)) for x_i in X])
            points = []
            for x in x_history:
                points.append((x, float(fun.subs(variables[0], x[0]))))
            draw_lines(X, Y, points, variables)
    res_f = minimize_result["fun"]
    res_x = dict(zip(variables, minimize_result["x"]))
    message = minimize_result["message"]
    print(f'f(X): {res_f} | X: {res_x} | Message: {message}')
    return res_f, res_x
    # b = []
    # A = []
    # for g_i in g:
    #     left, right = g_i.split('=')
    #     left, right = preproc_fun(left), float(right)
    #     A.append([left.coeff(variable) for variable in variables])
    #     b.append(right)
    # A = np.array(A, dtype=np.float32)
    # b = np.array(b, dtype=np.float32).reshape(-1, 1)
    # mu = np.array(sympy.symbols(' '.join([f'mu{i + 1}' for i in range(len(g))]))).reshape(-1, 1)
    # # Objective function for mu -> max
    # f_mu = -1 * (-mu.T.dot(b).item() - f.subs(dict(zip(variables, map(lambda x: x.item(), -A.T.dot(mu))))))
    # f_mu = sympy.lambdify(mu.flatten(), f_mu)
    # mu_opt = minimize(lambda x: f_mu(*x), np.zeros((len(g), )))['x']
    # # Objective function for x -> max
    # f_x = -1 * (variables.dot(-A.T.dot(mu_opt.reshape(-1, 1))).item() - f)
    # f_x = sympy.lambdify(variables, f_x)
    # minimize_result = minimize(lambda x: f_x(*x), x_init)
    # print(f'f(x): {minimize_result["fun"]} | x: {minimize_result["x"]} | Message: {minimize_result["message"]}')


def log_barriers(fun, g, x_init, t=0.1, c=2., tol_barrier=1e-4, tol_newton=1e-4, maxiter=50, plot=True):
    """Optimization of a Function with Inequality Constraints by the Logarithmic Barrier Method
    Be careful to have doubly differentiable fun to properly run function.
    Parameter x_init must satisfy constraints to better converge on the right solution.

    Positional arguments:
    fun -- str, function in analytic form
    g -- sequence(list, tuple ...) of str, sequence of inequality type constraints functions in analytic form
    x_init -- numpy.ndarray, initial approximation

    Keyword arguments:
    t -- float, initialization for logarithmic barrier coefficient(default=0.1)
    c -- float, multiplier of t each iteration of barrier method(default=2.)
    tol_barrier -- float, precision for logarithmic barrier part of solution(default=1e-4)
    tol_newton -- float, precision for newton gradient method part of solution(default=1e-4)
    max_iter -- int, number of iterations in newton gradient method(default=200)
    plot -- bool, Draw plot(default=False)
    """
    x = x_init
    g = g.copy()
    for i in range(len(g)):
        if '>=' in g[i]:
            # if '>=' in g[i]:
            left, right = g[i].split('>=')
            # elif '=' in g[i]:
            #     left, right = g[i].split('=')
            g[i] = f'({left.strip()} - {right.strip()})'
        elif '<=' in g[i]:
            left, right = g[i].split('<=')
            g[i] = f'(-({left.strip()}) + {right.strip()})'
    fun_prep = preproc_fun(fun)
    z = preproc_fun(f'{fun} - (1 / t) * ln({"*".join(g)})')
    variables = list(fun_prep.atoms(sympy.Symbol))
    grad = sympy.derive_by_array(z, variables)
    hess = sympy.hessian(z, variables)
    j = 0
    x_history = [x]
    while len(g) / t > tol_barrier and j < maxiter:
        i = 0
        hg = np.ones((len(variables), )).reshape(-1, 1)
        while np.linalg.norm(hg) > tol_newton and i < maxiter:
            gx = np.array(grad.subs(zip(variables, x)).subs({'t': t}), dtype=np.float64)
            hx = np.array(hess.subs(zip(variables, x)).subs({'t': t}), dtype=np.float64)
            hg = -1 * (np.linalg.inv(hx) @ gx).flatten()
            x = x + hg
            x_history.append(x)
            i += 1
        t = c * t
        # z = preproc_fun(f'{fun} - (1 / {t}) * ln({"*".join(g)})')
        # grad = sympy.derive_by_array(z, variables)
        # hess = sympy.hessian(z, variables)
        j += 1
    if j != maxiter:
        message = 'Optimization terminated successfully'
    else:
        message = 'Reached the limit of iterations'
    fun = preproc_fun(fun)
    if plot:
        if len(variables) == 2:
            X, Y = np.meshgrid(np.linspace(-10, 10, 25), np.linspace(-10, 10, 25))
            Z = np.array([[float(fun.subs([(variables[0], x), (variables[1], y)]))
                           for x, y in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)])
            points = []
            for x in x_history:
                points.append(dict(zip(variables, x)))
            draw_level_lines(X, Y, Z, points, variables)
            X, Y = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
            draw_feasible_polygon(X, Y, g, variables)
        elif len(variables) == 1:
            X = np.linspace(-10, 10)
            Y = np.array([float(fun.subs(variables[0], x_i)) for x_i in X])
            points = []
            for x in x_history:
                points.append((x, float(fun.subs(variables[0], x[0]))))
            draw_lines(X, Y, points, variables)
    res_f = fun.subs(zip(variables, x))
    res_x = dict(zip(variables, x))
    print(f'f(X): {res_f} | X: {res_x} | Message: {message}')
    return res_f, res_x


def direct_dual_interior_point(fun, g, x_init, plot=True):
    """Optimization of a function with equality and inequality-type constraints.

    Positional arguments:
    fun -- str, function in analytic form
    g -- sequence(list, tuple ...) of str, sequence of equality type constraints functions in analytic form
    x_init -- numpy.ndarray, initial approximation

    Keyword arguments:
    plot -- Draw plot(default=False)
    """
    fun = preproc_fun(fun)
    g = g.copy()
    variables = list(fun.atoms(sympy.Symbol))
    constraints = []
    for i, g_i in enumerate(g):
        if '>=' in g_i:
            left, right = tuple(map(lambda x: x.strip(), g_i.split('>=')))
            g_str = f'({left}) - ({right})'

            def constraint(x, left=left, right=right):
                return sympy.lambdify(variables, preproc_fun(left) - preproc_fun(right))(*x)

            constraints.append({'type': 'ineq', 'fun': constraint})

        elif '<=' in g_i:
            left, right = tuple(map(lambda x: x.strip(), g_i.split('<=')))
            g_str = f'-({left}) + ({right})'

            def constraint(x, left=left, right=right):
                return sympy.lambdify(variables, -1 * preproc_fun(left) + preproc_fun(right))(*x)

            constraints.append({'type': 'ineq', 'fun': constraint})
        else:
            left, right = tuple(map(lambda x: x.strip(), g_i.split('=')))

            def constraint(x, left=left, right=right):
                return sympy.lambdify(variables, preproc_fun(left) - preproc_fun(right))(*x)

            constraints.append({'type': 'eq', 'fun': constraint})
        g[i] = g_str
    f = sympy.lambdify(variables, fun)

    def callback(xk, state):
        x_history.append(xk)

    x_history = [x_init]
    minimize_result = minimize(lambda x: f(*x),
                               x_init,
                               constraints=constraints,
                               method='trust-constr',
                               callback=callback)

    if plot:
        if len(variables) == 2:
            X, Y = np.meshgrid(np.linspace(-10, 10, 25), np.linspace(-10, 10, 25))
            Z = np.array([[float(fun.subs([(variables[0], x), (variables[1], y)]))
                           for x, y in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)])
            points = []
            for x in x_history:
                points.append(dict(zip(variables, x)))
            draw_level_lines(X, Y, Z, points, variables)
            X, Y = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
            draw_feasible_polygon(X, Y, g, variables)
        elif len(variables) == 1:
            X = np.linspace(-10, 10)
            Y = np.array([float(fun.subs(variables[0], x_i)) for x_i in X])
            points = []
            for x in x_history:
                points.append((x, float(fun.subs(variables[0], x[0]))))
            draw_lines(X, Y, points, variables)
    res_f = minimize_result["fun"]
    res_x = dict(zip(variables, minimize_result["x"]))
    message = minimize_result["message"]
    print(f'f(X): {res_f} | X: {res_x} | Message: {message}')
    return res_f, res_x

