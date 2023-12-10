import numpy as np
import pandas as pd
import sympy
from sympy import lambdify, derive_by_array

from . import plotting
from ..onedimensionaloptimization import plotting as one_plotting
from ..twovarextremas import plotting_3d
from ..twovarextremas import utils_twovarextremas

__all__ = ['stoch_descent', 'simulated_annealing', 'genetic_algorithm']


def stoch_descent(fun_anl, lr,
                  eps=1e-5, max_iter=500,
                  print_info=False, plot=False, record_info=False, x_init=None, momentum=0.85):
    """Optimizes function of n variables by stochastic gradient descent method with momentum.

    Positional arguments:
    fun_anl -- function analytic form
    lr -- learning rate for descent

    Keyword arguments:
    eps -- float or int., Precision of result(default=1e-5)
    max_iter -- int, Maximum iteration of algorithm(default=500)
    print_info -- bool, Print information each iteration(default=False)
    plot -- bool, Draw plot(default=False), plot is available if less than 3 variables in function
    record_info -- bool, Make pd.DataFrame with recorder information(default=False)
    x_init -- np.ndarray or another sequence data type, Initial x vector(default=None), leave None to initialize by ones
    momentum -- momentum parameter for optimizing by gradient descent(default=0.85)
    """
    fun = utils_twovarextremas.preproc_fun(fun_anl)
    variables = tuple(fun.atoms(sympy.Symbol))
    fun_grad = lambdify(variables, derive_by_array(fun, variables))
    v = 1
    x_history = []
    if x_init is None:
        x_i = np.ones((len(variables), ))
    else:
        x_i = np.array(x_init)
        assert len(x_i) == len(variables), 'Initial x vector does not match number of variables in function'
    x_history.append(x_i)
    x_i_1 = None
    if record_info:
        df = pd.DataFrame(columns=variables)
    else:
        df = None
    if plot:
        if len(variables) == 1:
            X = np.linspace(-15, 15, 50)
            Y = np.array([float(fun.subs(variables[0], x)) for x in X])
            one_plotting.plot_scatter_and_line(X,
                                               Y,
                                               x_i,
                                               [fun.subs(dict(zip(variables, x_i)))],
                                               [str(variables[0])])
        elif len(variables) == 2:
            X, Y = np.meshgrid(np.linspace(-15, 15, 50), np.linspace(-15, 15, 50))
            Z = np.array(
                [[float(fun.subs([(variables[0], xx), (variables[1], yy)])) for xx, yy in zip(x, y)] for x, y in
                 zip(X, Y)])
    for i in range(max_iter):
        v = momentum * v - lr * np.array(fun_grad(*x_i))
        x_i_1 = x_i + v
        if print_info:
            print(('k: {} | ' + ' : {} | '.join(map(str, variables)) + ' : {}').format(i + 1, *x_i_1))
        if record_info:
            df = df.append(dict(zip(variables, x_i_1)), ignore_index=True)
        if plot:
            x_history.append(x_i_1)
        if np.sum(np.sqrt((x_i_1 - x_i) ** 2)) <= eps:
            print('Найдено значение с заданной точностью')
            break
        else:
            x_i = x_i_1
    else:
        print('Достигнуто максимальное количество итераций')

    if plot:
        if len(variables) == 2:
            plotting_3d.make_3d_plot(X,
                                     Y,
                                     Z,
                                     [{variables[0]: x_history[0][0],
                                       variables[1]: x_history[0][1]},
                                      {variables[0]: x_history[-1][0],
                                       variables[1]: x_history[-1][1]}],
                                     variables[0],
                                     variables[1],
                                     fun)
            plotting.make_level_lines_plot(X, Y, Z,
                                           x_history,
                                           *variables)
        elif len(variables) == 1:
            one_plotting.plot_scatter_and_line(X,
                                               Y,
                                               x_i,
                                               [fun.subs(dict(zip(variables, x_i_1)))],
                                               [str(variables[0])])
    res_x = dict(zip(variables, x_i_1))
    res_f = fun.subs(res_x)
    print(f'f(X) = {res_f}, X = {res_x}')
    if record_info:
        df.index.name = 'k'
        return res_f, res_x, df
    else:
        return res_f, res_x


def simulated_annealing(fun_anl, bounds,
                        t_max=10, t_min=0.001, t_ch=0.5, max_iter=500, plot=False):
    """Optimizes function of n variable -> min by simulated annealing method.
    linear temperature changing.

    Positional arguments:
    fun_anl -- function analytic form
    bounds -- dict, dictionary of bounds for each variable

    Keyword arguments:
    t_max -- initial temperature
    t_min -- minimal temperature
    t_ch -- constant for linear temperature changing each iteration
    max_iter -- Maximum iteration of algorithm(default=500)
    plot -- Draw plot(default=False)
    """
    fun = utils_twovarextremas.preproc_fun(fun_anl)
    variables = tuple(fun.atoms(sympy.Symbol))
    fun_lambda = lambdify(variables, fun)
    state = np.array([np.random.choice(np.arange(*bounds[str(variable)], 0.1))
                      for variable in variables])
    current_energy = fun_lambda(*state)
    t = t_max
    points = []
    energy_history = [current_energy]

    def generate_state_candidate(x, fraction):
        """Move x to the right or to the left"""
        new_state = []
        for idx, variable in enumerate(variables):
            bound = bounds[str(variable)]
            amplitude = (max(bound) - min(bound)) * fraction / 10
            delta = (-1 * amplitude / 2.) + amplitude * np.random.random_sample()
            new_state.append(max(min(x[idx] + delta, bound[1]), bound[0]))
        return np.array(new_state)

    for i in range(1, max_iter + 1):
        state_candidate = generate_state_candidate(state, i / max_iter)
        energy_candidate = fun_lambda(*state_candidate)
        if energy_candidate < current_energy:
            current_energy = energy_candidate
            state = state_candidate
        else:
            p = np.exp(energy_candidate - current_energy / t)
            if np.random.rand() <= p:
                current_energy = energy_candidate
                state = state_candidate
        t = t_max * t_ch / i
        if t <= t_min:
            print('Температура достигла минимума')
            break
        if plot:
            energy_history.append(current_energy)
            if i % 10 == 0:
                if len(variables) == 1:
                    points.append([state[0], current_energy])
                elif len(variables) == 2:
                    points.append([state[0], state[1], current_energy])
    else:
        print('Достигнуто максимальное количество итераций')
    if plot:
        if len(variables) == 1:
            x_draw = np.arange(*bounds[str(variables[0])], 0.05)
            plotting.make_annealing_plot_2d(x_draw, [fun_lambda(x_i) for x_i in x_draw], np.array(points))
        plotting.plot_energy_history(energy_history)

    res_f, res_x = fun_lambda(*state), dict(zip(variables, state))
    print(f'f(X) = {res_f}, X = {res_x}')
    return res_f, res_x


def genetic_algorithm(fun_anl, bounds,
                      n_bits=16, n_pop=100, p_c=0.85, max_iter=500):
    """Optimizes continuous function of n variable -> min by genetic algorithm.

    Positional arguments:
    fun_anl -- function analytic form
    bounds -- dict, dictionary of bounds for each variable

    Keyword arguments:
    n_bits -- number of candidate's bits(chromosomes) per one variable(default=16)
    n_pop -- actual population size(default=100)
    p_c -- probability of making crossover between two parents(default=0.85)
    max_iter -- Number of generations(default=500)
    """
    fun = utils_twovarextremas.preproc_fun(fun_anl)
    variables = tuple(fun.atoms(sympy.Symbol))
    fun_lambda = lambdify(variables, fun)
    # Probability of children bit mutation
    p_m = 1.0 / (float(n_bits) * len(bounds))

    # Decoding bitstring to float number for each variable
    def decode(bounds, n_bits, bitstring, variables):
        # Decoded consists of float values for each variable figured out from bitstring
        decoded = []
        largest = 2 ** n_bits
        for i, variable in enumerate(variables):
            variable = str(variable)
            # Extract the substring for this variable
            start, end = i * n_bits, (i * n_bits) + n_bits
            substring = bitstring[start:end]
            # Convert bitstring to a string of bit chars
            chars = ''.join([str(s) for s in substring])
            # Convert string to integer from binary system
            integer = int(chars, 2)
            # Scale value and fit into the bounds
            value = bounds[variable][0] + (integer / largest) * (bounds[variable][1] - bounds[variable][0])
            decoded.append(value)
        return decoded

    # Tournament selection
    # Selects the candidate which has the best value (-> min) among k random relatives
    def selection(pop, scores, k=5):
        # first random selection
        selection_ix = np.random.randint(len(pop))
        relatives = np.random.randint(0, len(pop), k - 1)
        for ix in relatives:
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]

    # Crossover two parents to create two children with probability
    def crossover(p1, p2, p):
        # Initially children are equal to their parents
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() <= p:
            # Perform crossover
            crossover_point = np.random.randint(1, len(p1) - 1)
            c1 = p1[:crossover_point] + p2[crossover_point:]
            c2 = p2[:crossover_point] + p1[crossover_point:]
        return [c1, c2]

    # Mutation of children with probability
    def mutation(bitstring, p):
        for i in range(len(bitstring)):
            if np.random.rand() < p:
                bitstring[i] = 1 - bitstring[i]

    def overall(objective, bounds, chromosomes, generations, population_size, p_c, p_m, vars):
        population = [np.random.randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(population_size)]
        best, best_eval = 0, objective(*decode(bounds, chromosomes, population[0], vars))
        for gen in range(generations):
            decoded = [decode(bounds, chromosomes, p, vars) for p in population]
            scores = [objective(*d) for d in decoded]
            for i in range(population_size):
                if scores[i] < best_eval:
                    best, best_eval = decoded[i], scores[i]
                    print(f'Generation: {gen} | X = {best} | F(X) = {best_eval}')
            selected = [selection(population, scores) for _ in range(population_size)]
            # Next generation is coming
            children = []
            for pair in range(0, population_size, 2):
                p1, p2 = selected[pair], selected[pair + 1]
                for child in crossover(p1, p2, p_c):
                    mutation(child, p_m)
                    children.append(child)
            population = children
        return [best, best_eval]
    res_x, res_f = overall(fun_lambda, bounds, n_bits, max_iter, n_pop, p_c, p_m, variables)
    print(f'f(X) = {res_f}, X = {dict(zip(variables, res_x))}')
    return res_f, res_x
