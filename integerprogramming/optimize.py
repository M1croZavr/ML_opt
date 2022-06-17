import numpy as np
from . import utils


def make_simplex(A, B, C, basis, variables, extrema='max'):
    """Linear programming simplex method.
    Find extrema of linear programming problem. Z -> extrema by tabular simplex method.
    Necessary conditions:
    1. The problem has a canonical form.
        Z = <C, x> -> extrema
        Ax = B
        x >= 0, b >= 0
    2. The problem has an explicitly distinguished basis.

    Positional arguments:
        A -- np.ndarray, Coefficients of constraint variables
        B -- np.ndarray, Free members of restrictions
        C -- np.ndarray, Coefficients of objective function (Without multiplying by -1)
        basis -- np.ndarray, Basis variables names in right order
        variables -- np.ndarray, All variables names in right order

    Keyword arguments:
        extrema -- str, which extrema to find \'max\' or \'min\'

    Return:
        Full coefficient matrix M, basis, variables
    """
    B = np.hstack((B, [0]))
    C = -1 * C
    M = np.vstack((A, C))
    M = np.hstack((M, B.reshape(-1, 1)))
    M = M.astype(np.float64)
    flag = True

    while flag:
        if extrema.lower() == 'max':
            if np.any(M[-1, :] < 0):
                leading_column_idx = np.argmin(M[-1, :-1])
                leading_column = M[:, leading_column_idx]
            else:
                print('All coefficients in string Z are >= 0')
                break
        elif extrema.lower() == 'min':
            if np.any(M[-1, :] > 0):
                leading_column_idx = np.argmax(M[-1, :-1])
                leading_column = M[:, leading_column_idx]
            else:
                print('All coefficients in string Z are <= 0')
                break
        else:
            raise ValueError('Extrema must be one of {\'min\', \'max\'}')

        b_div_lead_col = M[:, -1][: len(basis)] / leading_column[: len(basis)]
        # В случае деления на 0
        b_div_lead_col[np.isinf(b_div_lead_col)] = -1
        if np.any(b_div_lead_col > 0):
            valid_idxs = np.where(b_div_lead_col > 0)[0]
            leading_row_idx = valid_idxs[np.argmin(b_div_lead_col[valid_idxs])]
            leading_row = M[leading_row_idx, :]
        else:
            print('No leading row found')
            break

        leading_element = M[leading_row_idx, leading_column_idx]

        basis[leading_row_idx] = variables[leading_column_idx]
        M[leading_row_idx, :] = M[leading_row_idx, :] / leading_element

        for row_idx in range(M.shape[0]):
            if row_idx != leading_row_idx:
                M[row_idx, :] = M[row_idx, :] - leading_column[row_idx] * M[leading_row_idx, :]

    return M, basis, variables


def gomori_method(A, B, C, basis, variables, target_vars, extrema='max'):
    """Integer linear programming gomori method by simplex method.
    Find extrema of integer linear programming problem. Z -> extrema by tabular simplex method.
    Necessary conditions:
    1. The problem has a canonical form.
        Z = <C, x> -> extrema
        Ax = B
        x >= 0, b >= 0
    2. The problem has an explicitly distinguished basis.

    Positional arguments:
        A -- np.ndarray, Coefficients of constraint variables
        B -- np.ndarray, Free members of restrictions
        C -- np.ndarray, Coefficients of objective function
        basis -- np.ndarray, Basis variables names in right order
        variables -- np.ndarray, All variables names in right order
        target_vars -- set, list ... seq, Names of needed variables, for example {\'x1\', \'x2\'}

    Keyword arguments:
        extrema -- str, which extrema to find \'max\' or \'min\'

    Return:
        Full coefficient matrix M, basis, variables
    """
    s_M, s_basis, s_variables = make_simplex(A, B, C, basis, variables, extrema)
    t_count = 0

    while t_count < 10:
        max_fraction = 0.
        target_idx = None
        for var_idx in range(len(s_basis)):
            if s_basis[var_idx] in target_vars:
                wh, fr = utils.find_whole_fraction(s_M[:, -1][var_idx])
                if fr > max_fraction:
                    max_fraction = fr
                    target_idx = var_idx

        if max_fraction == 0:
            break
        else:
            t_count += 1

        new_constraint = list(map(lambda x: -1 * utils.find_whole_fraction(x)[1], s_M[target_idx, :]))
        new_constraint.insert(-1, 1)
        new_constraint = np.array(new_constraint, dtype=np.float64)
        s_variables = np.concatenate((s_variables, [f't{t_count}']))
        s_basis = np.concatenate((s_basis, [f't{t_count}']))

        s_M = np.insert(s_M, -1, 0, axis=1)
        s_M = np.insert(s_M, -1, new_constraint, axis=0)
        s_M[-1, :] = -1 * s_M[-1, :]

        min_value = float('inf')
        min_idx = None
        for j in range(s_M.shape[0] - 2):
            if s_M[-2, j] != 0:
                current_value = s_M[-1, j] / s_M[-2, j]
                if current_value < min_value:
                    min_value = current_value
                    min_idx = j

        leading_row_idx = s_M.shape[0] - 2
        leading_column_idx = min_idx
        leading_element = s_M[leading_row_idx, leading_column_idx]
        leading_column = s_M[:, leading_column_idx]

        s_basis[len(s_basis) - 1] = s_variables[leading_column_idx]
        s_M[leading_row_idx, :] = s_M[leading_row_idx, :] / leading_element

        for row_idx in range(s_M.shape[0]):
            if row_idx != leading_row_idx:
                s_M[row_idx, :] = s_M[row_idx, :] - leading_column[row_idx] * s_M[leading_row_idx, :]
    print(f'F(X) = {-1 * s_M[-1, -1]}, X: {"  ".join([f"{var}={value}" for var, value in zip(s_basis, s_M[:-1, -1]) if var in target_vars])}')
    return s_M, s_basis, s_variables
