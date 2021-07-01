"""

miqp-clf2lin
------------

Functions to extract raw attributes from instances in the datasets.
For each instance, *static* attributes are extracted (refer to doc/raw_features.txt for the list of 58 attributes).
Raw features are grouped in different functions, which are then all called by compute_static_features.

NOTE: dynamic features are computed using information from the benchmark, the same used for labeling.

"""

import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse import linalg
import time


""" GENERAL INFORMATION """


def fix_parameters(c, time_limit=3600):
    """
    :param c: cplex.Cplex instance of an MIQP
    :param time_limit: float, timelimit
    Fix parameters for the instance to be solved.
    """
    c.parameters.timelimit.set(time_limit)  # time limit is 1h (3600 sec)
    c.parameters.threads.set(1)  # specify the correct number of threads

    c.set_log_stream(None)
    # c.set_error_stream(None)
    c.set_warning_stream(None)
    c.set_results_stream(None)
    return


def var_type_lists(c, size):
    """
    :param c: cplex.Cplex instance of an MIQP
    :param size: int, size of the problem, i.e. number of variables
    :return: lists, containing (cplex) indices of variables, depending on their type (binary, continuous, integer)
    """
    bin_list = [i for i in range(size) if c.variables.get_types(i) == 'B']
    cont_list = [i for i in range(size) if c.variables.get_types(i) == 'C']
    int_list = [i for i in range(size) if c.variables.get_types(i) == 'I']

    # dummy check for new instances that might contain different types
    if len(bin_list) + len(cont_list) != size:
        print("Variables of other types detected for {}. B: {}, C: {}, I: {}, tot: {}".format(c.get_problem_name(),
                                                                                              len(bin_list),
                                                                                              len(cont_list),
                                                                                              len(int_list), size))
    return bin_list, cont_list, int_list


def general_info(c, filename, abs_tol, eig_max):
    """
    :param c: cplex.Cplex instance
    :param filename: str, name of the processed instance
    :param abs_tol: float, specified tolerance on absolute value to determine zero eigenvalues (9*1e-6)
    :param eig_max: int, maximal number of eigenvalues computed (500)
    :return: information required for multiple features, computed once and for all
    q_dok, q_ind_tup, var_deg, w, w_mod, cons_dok, cons_ind_tup
    """

    c.read(filename)
    fix_parameters(c)
    size = c.variables.get_num()

    # quadratic objective in dok format
    qmat_sp = c.objective.get_quadratic()  # SparsePair format list
    var_deg = {}  # variables degrees

    q_dok = dok_matrix((size + 1, size + 1))  # size+1 to get full spectrum in sparse mode when size <= eig_max

    for i in range(size):
        ind, val = qmat_sp[i].unpack()
        var_deg[i] = len(ind) - 1 if i in ind else len(ind)  # remove diagonal from the count
        for j in range(len(ind)):
            q_dok[i, ind[j]] = val[j]
    q_ind_tup = q_dok.keys()  # tuples of indices of nonzero entries (contains both symmetric entries)

    # spectrum corrected
    if size <= eig_max:
        w, v = linalg.eigsh(q_dok, k=size, which='LM')
    else:
        print("\tPartial spectrum computed - eig_max {}, size {}".format(eig_max, size))
        w, v = linalg.eigsh(q_dok, k=eig_max, which='LM')  # only eig_max < size eig computed (Largest Magnitude)
    w_mod = np.where(np.abs(w) < abs_tol, 0., w)

    # constraints in dok format
    cons_sp = c.linear_constraints.get_rows()
    cons_dok = dok_matrix((c.linear_constraints.get_num(), size))
    for i in range(c.linear_constraints.get_num()):
        ind, val = cons_sp[i].unpack()
        for j in range(len(ind)):
            cons_dok[i, ind[j]] = val[j]
    cons_ind_tup = cons_dok.keys()  # tuples of indices of nonzero entries

    return q_dok, q_ind_tup, var_deg, w, w_mod, cons_dok, cons_ind_tup


""" GENERAL """


# var_constr_general
def var_constr_general(c):
    # 1   optimization sense (1=min or -1=max)
    # 2   # binary variables
    # 3   # integer variables
    # 4   total number of variables (size of the pb)
    # 5   total number of constraints
    return [c.objective.get_sense(), float(c.variables.get_num_binary()), float(c.variables.get_num_integer()),
            float(c.variables.get_num()), float(c.linear_constraints.get_num())]


""" QUADRATIC OBJECTIVE """


# q_nnz_square_diag
def q_nnz_square_diag(bin_list, cont_list, int_list, q_ind_tup):
    # 6   # nnz square binaries (diagonal only)
    # 7   # nnz square continuous (diagonal only)
    # 8   # nnz square integer (diagonal only)
    return [sum(1 for (i, j) in q_ind_tup if (i in bin_list) and (j in bin_list) and (i == j)),
            sum(1 for (i, j) in q_ind_tup if (i in cont_list) and (j in cont_list) and (i == j)),
            sum(1 for (i, j) in q_ind_tup if (i in int_list) and (j in int_list) and (i == j))]


# q_nnz_prod
def q_nnz_prod(bin_list, cont_list, int_list, q_ind_tup):
    # 9   # of bin*bin out-diagonal products (/2) !! UpperDiagonal only
    # 10  # of cont*cont out-diagonal products (/2) !! UpperDiagonal only
    # 11  # of int*int out-diagonal products (/2) !! UpperDiagonal only
    # 12  # of bin*cont out-diagonal products (NOT*2) !! UpperDiagonal only
    # 13  # of bin*int out-diagonal products (NOT*2) !! UpperDiagonal only
    # 14  # of int*cont out-diagonal products (NOT*2) !! UpperDiagonal only
    return [sum(1 for (i, j) in q_ind_tup if (i in bin_list) and (j in bin_list) and (i != j)) / 2.,
            sum(1 for (i, j) in q_ind_tup if (i in cont_list) and (j in cont_list) and (i != j)) / 2.,
            sum(1 for (i, j) in q_ind_tup if (i in int_list) and (j in int_list) and (i != j)) / 2.,
            sum(1 for (i, j) in q_ind_tup if (i in bin_list) and (j in cont_list)),
            sum(1 for (i, j) in q_ind_tup if (i in bin_list) and (j in int_list)),
            sum(1 for (i, j) in q_ind_tup if (i in int_list) and (j in cont_list))]


# q_bin_deg
def q_bin_deg(bin_list, var_deg):
    # 15  max degree of binary variables
    # 16  min degree of binary variables
    # 17  avg degree of binary variables
    bin_deg = [var_deg[k] for k in bin_list]
    if not bin_deg:
        return [0, 0, 0]
    return [max(bin_deg or [0]), min(bin_deg or [0]), sum(bin_deg) / float(len(bin_deg))]


# q_cont_deg
def q_cont_deg(cont_list, var_deg):
    # 18  max degree of continuous variables
    # 19  min degree of continuous variables
    # 20  avg degree of continuous variables
    cont_deg = [var_deg[k] for k in cont_list]
    if not cont_deg:
        return [0, 0, 0]
    return [max(cont_deg or [0]), min(cont_deg or [0]), sum(cont_deg) / float(len(cont_deg))]


# q_int_deg
def q_int_deg(int_list, var_deg):
    # 21  max degree of integer variables
    # 22  min degree of integer variables
    # 23  avg degree of integer variables
    int_deg = [var_deg[k] for k in int_list]
    if not int_deg:
        return [0, 0, 0]
    return [max(int_deg or [0]), min(int_deg or [0]), sum(int_deg) / float(len(int_deg))]


# q_conn_graph
def q_conn_graph(var_deg, size):
    # 24  density of connectivity graph
    return sum(var_deg.values()) / float(size * (size - 1)) if float(size * (size - 1)) != 0 else 0


# q_coeffs
def q_coeffs(q_dok, q_ind_tup):
    # 25  smallest nnz |q_ii| (diagonal)
    # 26  biggest nnz |q_ii| (diagonal)
    # 27  smallest nnz |q_ij| (all: diagonal and out-diagonal)
    # 28  biggest nnz |q_ij| (all: diagonal and out-diagonal)
    return [min(list(q_dok[(i, j)] for (i, j) in q_ind_tup if i == j) or [0]),
            max(list(q_dok[(i, j)] for (i, j) in q_ind_tup if i == j) or [0]),
            min(list(q_dok[(i, j)] for (i, j) in q_ind_tup) or [0]),
            max(list(q_dok[(i, j)] for (i, j) in q_ind_tup) or [0])]


# q_diag_dom
def q_diag_dom(q_dok, q_ind_tup, size):
    # 29  averaged `diagonal dominance' on rows
    sum_delta = 0
    for i in range(size):
        sum_delta += (2 * abs(q_dok[i, i]) - np.sum([abs(q_dok[i, j]) for j in range(size) if (i, j) in q_ind_tup]))
    return sum_delta / float(size)


""" LINEAR OBJECTIVE """


# linear_nnz
def linear_nnz(c, bin_list, cont_list, int_list):
    # 30  # of nnz binary in linear objective
    # 31  # of nnz continuous in linear objective
    # 32  # of nnz integer in linear objective
    if not bin_list and cont_list and int_list:
        return [0, np.count_nonzero(c.objective.get_linear(cont_list)),
                np.count_nonzero(c.objective.get_linear(int_list))]
    elif not cont_list and bin_list and int_list:
        return [np.count_nonzero(c.objective.get_linear(bin_list)), 0,
                np.count_nonzero(c.objective.get_linear(int_list))]
    elif not int_list and bin_list and cont_list:
        return [np.count_nonzero(c.objective.get_linear(bin_list)), np.count_nonzero(c.objective.get_linear(cont_list)),
                0]
    elif not bin_list and not cont_list and int_list:
        return [0, 0, np.count_nonzero(c.objective.get_linear(int_list))]
    elif not cont_list and not int_list and bin_list:
        return [np.count_nonzero(c.objective.get_linear(bin_list)), 0, 0]
    # case of continuous only should not be considered, but for completeness
    elif not bin_list and not int_list and cont_list:
        return [0, np.count_nonzero(c.objective.get_linear(cont_list)), 0]
    # all variables present
    elif bin_list and cont_list and int_list:
        return [np.count_nonzero(c.objective.get_linear(bin_list)),
                np.count_nonzero(c.objective.get_linear(cont_list)), np.count_nonzero(c.objective.get_linear(int_list))]


# linear_coeffs
def linear_coeffs(c):
    # 33  min nnz c_i
    # 34  max nnz c_i
    return [min(c.objective.get_linear() or [0]), max(c.objective.get_linear() or [0])]


""" CONSTRAINTS """


# constr_nnz
def constr_nnz(bin_list, cont_list, int_list, cons_ind_tup):
    # 35  # of nnz binary in constraints
    # 36  # of nnz continuous in constraints
    # 37  # of nnz integer in constraints
    return [sum(1 for (_, j) in cons_ind_tup if j in bin_list),
            sum(1 for (_, j) in cons_ind_tup if j in cont_list), sum(1 for (_, j) in cons_ind_tup if j in int_list)]


# constr_var_type
def constr_var_type(c, bin_list, cont_list, int_list, cons_ind_tup):
    # 38  # of constraints involving binary variables
    # 39  # of constraints involving continuous variables
    # 40  # of constraints involving integer variables
    count_bin = 0
    count_cont = 0
    count_int = 0
    for i in range(c.linear_constraints.get_num()):
        if any(j in bin_list for (k, j) in cons_ind_tup if k == i):
            count_bin += 1
        if any(j in cont_list for (k, j) in cons_ind_tup if k == i):
            count_cont += 1
        if any(j in int_list for (k, j) in cons_ind_tup if k == i):
            count_int += 1
    return [count_bin, count_cont, count_int]


# constr_coeffs_rhs
def constr_coeffs_rhs(c, cons_dok, cons_ind_tup):
    # 41  min nnz |a_ij|
    # 42  max nnz |a_ij|
    # 43  min rhs nnz
    # 44  max rhs nnz
    return [min(list(cons_dok[(i, j)] for (i, j) in cons_ind_tup) or [0]),
            max(list(cons_dok[(i, j)] for (i, j) in cons_ind_tup) or [0]),
            min(c.linear_constraints.get_rhs() or [0]), max(c.linear_constraints.get_rhs() or [0])]


""" SPECTRUM """


# eigen_sign
def eigen_sign(w):
    # 45  # of positive eigenvalues
    # 46  # of negative eigenvalues
    # 47  # of zero eigenvalues
    # 48  value lambda_max
    # 49  value lambda_min
    return [sum(1 for eig in w if eig > 0), sum(1 for eig in w if eig < 0), sum(1 for eig in w if eig == 0),
            max(w), min(w)]


# spectrum_prop
def spectrum_prop(w):
    # 50  trace of Q
    # 51  spectral norm of Q
    return [sum(w), max(abs(w))]


# eigen_sign_mod
def eigen_sign_mod(w_mod):
    # 52  # of positive eigenvalues (after abs_tol correction)
    # 53  # of negative eigenvalues (after abs_tol correction)
    # 54  # of zero eigenvalues (after abs_tol correction)
    # 55  value lambda_max (after abs_tol correction)
    # 56  value lambda_min (after abs_tol correction)
    return [sum(1 for eig in w_mod if eig > 0), sum(1 for eig in w_mod if eig < 0), sum(1 for eig in w_mod if eig == 0),
            max(w_mod), min(w_mod)]


# spectrum_prop_mod
def spectrum_prop_mod(w_mod):
    # 57  trace of Q (after abs_tol correction)
    # 58  spectral norm of Q (after abs_tol correction)
    return [sum(w_mod), max(abs(w_mod))]


""" STATIC (raw) FEATURES EXTRACTION """


def compute_static_raw_features(c, filename, abs_tol, eig_max):
    """
    :param c: cplex.Cplex instance of an MIQP
    :param filename: str, pathway to the processed instance
    :param abs_tol: float, specified tolerance on absolute value to determine zero eigenvalues (9*1e-6)
    :param eig_max: int, maximal number of eigenvalues computed (500)
    Read MIQP instance and extract general info and basic static features, which are returned as an array.
    """
    c.read(filename)
    fix_parameters(c)
    size = c.variables.get_num()
    bin_list, cont_list, int_list = var_type_lists(c, size)

    q_dok, q_ind_tup, var_deg, w, w_mod, cons_dok, cons_ind_tup = general_info(c, filename, abs_tol, eig_max)

    ft_list = []
    # print("\tStatic features")
    st0 = time.clock()
    ft_list.extend(var_constr_general(c))
    ft_list.extend(q_nnz_square_diag(bin_list, cont_list, int_list, q_ind_tup))
    ft_list.extend(q_nnz_prod(bin_list, cont_list, int_list, q_ind_tup))
    ft_list.extend(q_bin_deg(bin_list, var_deg))
    ft_list.extend(q_cont_deg(cont_list, var_deg))
    ft_list.extend(q_int_deg(int_list, var_deg))
    ft_list.append(q_conn_graph(var_deg, size))
    ft_list.extend(q_coeffs(q_dok, q_ind_tup))
    ft_list.append(q_diag_dom(q_dok, q_ind_tup, size))
    ft_list.extend(linear_nnz(c, bin_list, cont_list, int_list))
    ft_list.extend(linear_coeffs(c))
    ft_list.extend(constr_nnz(bin_list, cont_list, int_list, cons_ind_tup))
    ft_list.extend(constr_var_type(c, bin_list, cont_list, int_list, cons_ind_tup))
    ft_list.extend(constr_coeffs_rhs(c, cons_dok, cons_ind_tup))
    ft_list.extend(eigen_sign(w))
    ft_list.extend(spectrum_prop(w))
    ft_list.extend(eigen_sign_mod(w_mod))
    ft_list.extend(spectrum_prop_mod(w_mod))
    st_time = time.clock() - st0

    # print("\nFeatures are: ")
    # print(np.asarray(ft_list), "\n")
    return np.asarray(ft_list), st_time
