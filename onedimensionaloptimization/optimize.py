from twovarextremas import utils_check


def golden_section_search(fun_anl, x_l, x_u, eps=1e-5, max_iter=500, print_info=False, record_info=False):


    fun = utils_check.preproc_fun(fun_anl)
    return fun