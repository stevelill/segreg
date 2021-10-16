"""
Alternative implementations of ordinary regression routines.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import math

# NOTE: numba needs numpy.linalg, rather than scipy.linalg
import numpy.linalg
import scipy.linalg
import numpy as np
import statsmodels.api as sm

from segreg.model.alt import likelihood_util

try:
    from numba import jit
except ImportError as e:
    from segreg.mockjit import jit


# TODO: make namedtuple for ols_terms; use it everywhere
@jit(nopython=True)
def ols_terms(x_arr, y_arr):
    # NOTE: cannot use vdot with numba
    # vdot faster than sum(arr*arr) without numba, but similar with numba
    return np.array([len(x_arr),
                     np.sum(x_arr),
                     np.sum(y_arr),
                     np.sum(x_arr * x_arr),
                     np.sum(y_arr * y_arr),
                     np.sum(x_arr * y_arr)])

##########################################################################
# Hessian
##########################################################################


def rss(params, ols_data):
    """
    Parameters
    ----------
    params: list
        [intercept, slope]
    ols_data: list
    """
    v, m = params

    return likelihood_util.rss_line_segment([0.0, v, m], ols_data)

#     num_data = ols_data[0]
#     sum_x = ols_data[1]
#     sum_y = ols_data[2]
#     sum_xx = ols_data[3]
#     sum_yy = ols_data[4]
#     sum_xy = ols_data[5]
#
#     term1 = num_data * v * v + sum_xx * m * m + 2.0 * sum_x * v * m
#     term2 = -2.0 * sum_y * v - 2.0 * sum_xy * m + sum_yy
#
#     return term1 + term2


def ols_loglik(params, indep, dep):
    """
    Parameters
    ----------
    params: list
        [intercept, slope, residual_variance]
    ols_data: list
        ols_terms
    """
    [intercept, slope, resid_variance] = params

    ols_data = ols_terms(indep, dep)

    num_data = ols_data[0]

    rss_term = rss([intercept, slope], ols_data)

    result = likelihood_util.loglikelihood(rss=rss_term,
                                           resid_variance=resid_variance,
                                           num_data=num_data)
    return result


##########################################################################
# OLS LINEAR RESTRICTIONS
##########################################################################


def ols_fixed_slope(indep, dep, slope):
    intercept = (np.sum(dep) - slope * np.sum(indep)) / len(indep)
    return intercept

##########################################################################
# OLS
##########################################################################


@jit(nopython=True)
def mat_by_hand_ols(indep, dep):
    indep_trans = indep.T
    term1 = np.dot(indep_trans, indep)
    term2 = np.dot(indep_trans, dep)

    result = numpy.linalg.solve(term1, term2)
    result = result.T[0]
    return result


##########################################################################
# BEGIN: numba jit methods
##########################################################################


@jit(nopython=True, cache=False)
def ols_verbose(indep, dep, slope=np.nan):
    """
    Ordinary Least Squares estimation for one-dimensional data.

    This method automatically includes an intercept.

    Parameters
    ----------
    indep: array-like ndim 1
        the independent data, ie: the regressor
    dep: array-like ndim 1
        the dependent data, ie: the regressand
    slope: float
        if set to a float, will perform OLS estimation for the fixed slope

    Returns
    -------    
    intercept: float 
    slope: float
    rss: float 
    ols_data: numpy array ndim 1
    """
    # NOTE: unexplained numba failure when using slope=None, which is why we
    # have used np.nan
    ols_data = ols_terms(indep, dep)

    num_data = ols_data[0]
    sum_x = ols_data[1]
    sum_y = ols_data[2]
    sum_xx = ols_data[3]
    sum_yy = ols_data[4]
    sum_xy = ols_data[5]

    mean_x = sum_x / num_data
    mean_y = sum_y / num_data

    if math.isnan(slope):
        slope = (num_data * sum_xy - sum_y * sum_x) / \
            (num_data * sum_xx - sum_x * sum_x)

    intercept = mean_y - slope * mean_x

    two_intercept = 2.0 * intercept

    rss = (sum_yy - two_intercept * sum_y - 2.0 * slope * sum_xy +
           intercept * intercept * num_data + two_intercept * slope * sum_x +
           slope * slope * sum_xx)

    return intercept, slope, rss, ols_data


@jit(nopython=True, cache=False)
def _to_ols_matrix(arr):
    """
    Converts input to a 2d array where the columns are the observations of the
    data.

    Warning: assumes num obs is greater than num variables.
    """
    arr_2d = np.atleast_2d(arr)
    arr_2d_shape = arr_2d.shape
    num_col = min(arr_2d_shape[0], arr_2d_shape[1])
    arr_2d = arr_2d.reshape(-1, num_col)
    return arr_2d


@jit(nopython=True, cache=False)
def matrix_ols(indep, dep, add_const=True):
    """
    TODO: this is new: test it out a bunch

    Parameters
    ----------
    indep: array-like shape (num_obs, indep_dim)
        ie: data in columns
    indep: array-like shape (num_obs, indep_dim)
        ie: data in columns        
    """
    indep_to_use = _to_ols_matrix(indep)
    return _matrix_ols_impl(indep_to_use, dep, add_const)


@jit(nopython=True, cache=False)
def _matrix_ols_impl(indep, dep, add_const=True):
    """
    Parameters
    ----------
    indep: arraylike shape (num_obs, indep_dim)
        ie: data in columns
    """
    indep_to_use = indep
    if(add_const == True):
        indep_to_use = np.hstack((np.ones((indep_to_use.shape[0], 1)),
                                  indep_to_use))

    indep_trans = indep_to_use.T

    term1 = indep_trans.dot(indep_to_use)
    term2 = indep_trans.dot(dep)

    beta = np.linalg.solve(term1, term2)

    const = beta[0]
    rest_beta = beta[1:]

    return const, rest_beta

##########################################################################
# END: numba jit methods
##########################################################################


def fast_ols(indep, dep):
    """
    Assumes intercept.
    1D.
    """
    sum_y = np.sum(dep)
    sum_x = np.sum(indep)
    sum_xy = np.vdot(indep, dep)
    sum_xx = np.vdot(indep, indep)

    num = len(indep)
    mean_x = sum_x / num
    mean_y = sum_y / num

    slope = (num * sum_xy - sum_y * sum_x) / (num * sum_xx - sum_x * sum_x)
    intercept = mean_y - slope * mean_x

    return intercept, slope


def fast_ols_with_rss(indep, dep):
    """
    Assumes intercept.
    1D
    """
    sum_y = np.sum(dep)
    sum_x = np.sum(indep)
    sum_xy = np.vdot(indep, dep)
    sum_xx = np.vdot(indep, indep)

    num = len(indep)
    mean_x = sum_x / num
    mean_y = sum_y / num

    slope = (num * sum_xy - sum_y * sum_x) / (num * sum_xx - sum_x * sum_x)
    intercept = mean_y - slope * mean_x

    sum_yy = np.vdot(dep, dep)

    two_intercept = 2.0 * intercept

    rss = (sum_yy - two_intercept * sum_y - 2.0 * slope * sum_xy +
           intercept * intercept * num + two_intercept * slope * sum_x + slope * slope * sum_xx)

    return intercept, slope, rss


def statsmodels_ols(indep, dep, add_const=True, verbose=False):
    indep_to_use = indep
    if(add_const == True):
        indep_to_use = sm.add_constant(indep_to_use, has_constant='add')

    model = sm.OLS(dep, indep_to_use)
    results = model.fit()

    if verbose:
        print(results.summary())

    return results.params[0], results.params[1]

# cannot jit this due to third-party calls


def bare_bones_ols(indep, dep, add_const=True):
    indep_to_use = indep
    if(add_const == True):
        indep_to_use = sm.add_constant(indep_to_use, has_constant='add')

    coeff = scipy.linalg.lstsq(indep_to_use, dep)[0]

    # TODO: this only the case if have intercept
    est_const = coeff[0]
    est_mat = coeff[1:]

    if(len(est_mat.shape) == 1 and len(est_mat) == 1):
        est_mat = est_mat[0]

    return est_const, est_mat

# TODO: everywhere; either code for or document when scipy array needed


def ols_with_resid(indep, dep, add_const=True):
    indep_to_use = indep
    if(add_const == True):
        indep_to_use = sm.add_constant(indep_to_use, has_constant='add')
    coeff = scipy.linalg.lstsq(indep_to_use, dep)[0]

    # TODO: this only the case if have intercept
    est_const = coeff[0]
    est_mat = coeff[1:]

    if(len(indep.shape) > 1):
        resid = dep - (est_const + np.dot(indep, est_mat.reshape(-1, 1)))
    else:
        resid = dep - (est_const + indep * est_mat)

    if(len(est_mat.shape) == 1 and len(est_mat) == 1):
        est_mat = est_mat[0]

    return est_const, est_mat, resid


def ols_with_rss(indep, dep, add_const=True):

    ##print("DEPRECATED: HAS POTENTIAL BUG")

    indep_to_use = indep
    if(add_const == True):
        indep_to_use = sm.add_constant(indep_to_use, has_constant='add')
    coeff = scipy.linalg.lstsq(indep_to_use, dep)[0]

    # TODO: this only the case if have intercept
    est_const = coeff[0]
    est_mat = coeff[1:]

    if(len(indep.shape) > 1):
        term = est_const + np.dot(indep, est_mat.reshape(-1, 1))
        resid = dep - term.T[0]
    else:
        resid = dep - (est_const + indep * est_mat)

    if(len(est_mat.shape) == 1 and len(est_mat) == 1):
        est_mat = est_mat[0]

    rss = np.vdot(resid, resid)

    return est_const, est_mat, rss


@jit(nopython=True, cache=False)
def invert_two_by_two(a, b, c, d, e, f):
    r"""
    Solves a linear system of two equations with two unknowns.

    The rationale for this implementation is that it is much faster than 
    calling scipy.linalg.solve.

    Notes
    -----
    Solves the system for :math:`x`:

    .. math::
        A x = b

    where

    .. math::
        A = \begin{pmatrix}
            a & b \\
            c & d
            \end{pmatrix}
        \qquad
        x = \begin{pmatrix}
            x_1 \\
            x_2
            \end{pmatrix}
        \qquad
        b = \begin{pmatrix}
            e \\
            f
            \end{pmatrix}

    Returns
    -------
    tuple: x_1, x_2
    """
    denom = a * d - b * c
    first = (d * e - b * f) / denom
    second = (a * f - c * e) / denom

    return first, second
