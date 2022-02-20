from sympy.parsing.sympy_parser import standard_transformations, \
                                       implicit_multiplication_application, parse_expr


def preproc_fun(fun_anl):
    """Return preprocessed function as sympy object function."""
    fun = fun_anl.replace('^', '**').replace('–', '-')
    transformations = (standard_transformations + (implicit_multiplication_application,))
    fun = parse_expr(fun, transformations=transformations)
    return fun


def check_constraints(init_points, constr, vars_):
    """Return filtered by constraints conditions stationary points.

    Positional arguments:
    init_points -- saddle points of equation. Dict like {name: coordinate}
    constr -- list (list of tuples) of constraints on each variable
    vars_ -- list of variable names
    """
    m_points = []
    for m in init_points:
        for (coord_name, coord), coord_constraint in zip(m.items(), constr):
            if not(str(coord_name) in vars_) or not(coord_constraint[0] <= coord <= coord_constraint[1]):
                break
        else:
            m_points.append(m)
    return m_points
