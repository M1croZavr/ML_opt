import sympy
import pandas as pd
import numpy as np
from twovarextremas import utils_twovarextremas, plotting_3d
from onedimensionaloptimization import optimize as one_optimization, plotting as one_plotting
from . import plotting
from sympy import lambdify, derive_by_array


__all__ = ['stoch_descent']


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
                        t_max=10, t_min=0.001, max_iter=500, plot=False):
    """Optimizes function of n variable -> min by simulated annealing method.

    Positional arguments:
    fun_anl -- function analytic form
    bounds -- dict, dictionary of bounds for each variable

    Keyword arguments:
    t_max -- initial temperature
    t_min -- minimal temperature
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
        t = t_max * 0.1 / i
        if t <= t_min:
            print('Температура достигла минимума')
            break
        if i % 5 == 1:
            if plot:
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
    res_f, res_x = fun_lambda(*state), dict(zip(variables, state))
    print(f'f(X) = {res_f}, X = {res_x}')
    return res_f, res_x


def genetic_algorithm(fun_anl, bounds,
                      n_bits=16, n_pop=100, r_cross=0.9,
                      max_iter=500, plot=False):
    """Optimizes function of n variable -> min by simulated annealing method.

    Positional arguments:
    fun_anl -- function analytic form
    bounds -- dict, dictionary of bounds for each variable

    Keyword arguments:
    n_bits -- bits per one variable(default=16)
    n_pop -- population size(default=100)
    r_cross -- crossover rate
    max_iter -- Maximum iteration of algorithm(default=500)
    plot -- Draw plot(default=False)
    """
    fun = utils_twovarextremas.preproc_fun(fun_anl)
    variables = tuple(fun.atoms(sympy.Symbol))
    fun_lambda = lambdify(variables, fun)
    r_mut = 1.0 / (float(n_bits) * len(bounds))

    # decode bitstring to numbers
    def decode(bounds, n_bits, bitstring):
        decoded = list()
        largest = 2 ** n_bits
        for i, variable in enumerate(variables):
            variable = str(variable)
            # extract the substring
            start, end = i * n_bits, (i * n_bits) + n_bits
            substring = bitstring[start:end]
            # convert bitstring to a string of chars
            chars = ''.join([str(s) for s in substring])
            # convert string to integer
            integer = int(chars, 2)
            # scale integer to desired range
            value = bounds[variable][0] + (integer / largest) * (bounds[variable][1] - bounds[variable][0])
            # store
            decoded.append(value)
        return decoded

    # tournament selection
    def selection(pop, scores, k=3):
        # first random selection
        selection_ix = np.random.randint(len(pop))
        for ix in np.random.randint(0, len(pop), k - 1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]

    # crossover two parents to create two children
    def crossover(p1, p2, r_cross):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if np.random.rand() < r_cross:
            # select crossover point that is not on the end of the string
            pt = np.random.randint(1, len(p1) - 2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    # mutation operator
    def mutation(bitstring, r_mut):
        for i in range(len(bitstring)):
            # check for a mutation
            if np.random.rand() < r_mut:
                # flip the bit
                bitstring[i] = 1 - bitstring[i]

    # genetic algorithm
    def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
        # initial population of random bitstring
        pop = [np.random.randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
        # keep track of best solution
        best, best_eval = 0, objective(*decode(bounds, n_bits, pop[0]))
        # enumerate generations
        for gen in range(n_iter):
            # decode population
            decoded = [decode(bounds, n_bits, p) for p in pop]
            # evaluate all candidates in the population
            scores = [objective(*d) for d in decoded]
            # check for new best solution
            for i in range(n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
            # select parents
            selected = [selection(pop, scores) for _ in range(n_pop)]
            # create the next generation
            children = list()
            for i in range(0, n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]
                # crossover and mutation
                for c in crossover(p1, p2, r_cross):
                    # mutation
                    mutation(c, r_mut)
                    # store for next generation
                    children.append(c)
            # replace population
            pop = children
        return [best, best_eval]
    res_x, res_f = genetic_algorithm(fun_lambda, bounds, n_bits, max_iter, n_pop, r_cross, r_mut)
    print(f'f(X) = {res_f}, X = {res_x}')
    return res_f, res_x
