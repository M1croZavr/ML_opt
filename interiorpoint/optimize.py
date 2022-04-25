from twovarextremas.utils_twovarextremas import preproc_fun
import numpy as np
import sympy
from scipy.optimize import minimize
from .plotting import draw_level_lines, draw_lines


def newton_dual(fun, g, x_init, plot=True):
    """Optimizes function of 1 variable by golden section method.

    Positional arguments:
    fun -- str, function in analytic form
    g -- sequence(list, tuple ...) of str, sequence of equality type constraints functions in analytic form
    x_init -- numpy.ndarray, initial approximation

    Keyword arguments:
    plot -- Draw plot(default=False)
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

    print(f'f(X): {minimize_result["fun"]} | X: {dict(zip(variables, minimize_result["x"]))} | Message: {minimize_result["message"]}')
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


def log_barriers(fun, g, x_init, t, tol_barrier=1e-5, tol_newton=1e-5, maxiter=200, plot=True):
    """Optimizes function of 1 variable by golden section method.

    Positional arguments:
    fun -- str, function in analytic form
    g -- sequence(list, tuple ...) of str, sequence of inequality/equality type constraints functions in analytic form
    x_init -- numpy.ndarray, initial approximation

    Keyword arguments:
    plot -- Draw plot(default=False)
    """
    x = x_init
    for i in range(len(g)):
        if (('>=' in g[i]) or ('=' in g[i])) and (not('<=' in g[i])):
            if '>=' in g[i]:
                left, right = g[i].split('>=')
            elif '=' in g[i]:
                left, right = g[i].split('=')
            g[i] = f'({left.strip()} - {right.strip()})'
        elif '<=' in g[i]:
            left, right = g[i].split('<=')
            g[i] = f'(-({left.strip()}) + {right.strip()})'
    # Пересчитываем гессиану с новым t
    z = preproc_fun(f'{fun} + 1 / {t} * ln({"*".join(g)})')
    variables = list(z.atoms(sympy.Symbol))
    grad = sympy.derive_by_array(z, variables)
    hess = sympy.hessian(z, variables)
    while len(g) / t > tol_barrier:
        i = 0
        hg = np.ones((len(variables), )).reshape(-1, 1)
        while np.linalg.norm(hg) > tol_newton and i < maxiter:
            gx = np.array(grad.subs(dict(zip(variables, x))), dtype=np.float64)
            hx = np.array(hess.subs(dict(zip(variables, x))), dtype=np.float64)
            hg = -1 * (np.linalg.inv(hx) @ gx).flatten()
            x = x + hg
            i += 1
        # !!!!
        t = (1 + 1/(13 * np.sqrt(0.01))) * t
        z = preproc_fun(f'{fun} + 1 / {t} * ln({"*".join(g)})')
        grad = sympy.derive_by_array(z, variables)
        hess = sympy.hessian(z, variables)
    return x, variables


