"""
Ordinary Least Squares Regression core routines.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import numpy as np

from libc.math cimport isnan
from libc.math cimport NAN
cimport numpy as np
cimport cython

from segreg.model.regression cimport OLSData
from segreg.model.regression cimport OlsEstTerms


################################################################################
# PYTHON
################################################################################


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ols_with_rss(double[:] indep, double[:] dep, slope=None):

    cdef OLSData ols_data_to_use
    ols_data_to_use = ols_data(indep, dep)

    cdef OlsEstTerms ols_est_terms

    if slope is not None:
        ols_est_terms = ols_from_formula_with_rss_cimpl(ols_data_to_use, slope)
    else:
        ols_est_terms = ols_from_formula_with_rss_cimpl(ols_data_to_use)

    return ols_est_terms.intercept, ols_est_terms.slope, ols_est_terms.rss


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ols_verbose(double[:] indep, double[:] dep, slope=None):

    cdef OLSData ols_data_to_use
    ols_data_to_use = ols_data(indep, dep)

    cdef OlsEstTerms ols_est_terms

    if slope is not None:
        ols_est_terms = ols_from_formula_with_rss_cimpl(ols_data_to_use, slope)
    else:
        ols_est_terms = ols_from_formula_with_rss_cimpl(ols_data_to_use)

    # note: if we return the struct, it converts to python dict
    ols_data_arr = [ols_data_to_use.num,
                    ols_data_to_use.sum_x,
                    ols_data_to_use.sum_y,
                    ols_data_to_use.sum_xx,
                    ols_data_to_use.sum_yy,
                    ols_data_to_use.sum_xy]

    return (ols_est_terms.intercept,
            ols_est_terms.slope,
            ols_est_terms.rss,
            np.array(ols_data_arr))


################################################################################
# CYTHON
################################################################################

cdef OLSData add(OLSData ols_data_lhs, OLSData ols_data_rhs):
    cdef OLSData result
    result.num = ols_data_lhs.num + ols_data_rhs.num
    result.sum_x = ols_data_lhs.sum_x + ols_data_rhs.sum_x
    result.sum_y = ols_data_lhs.sum_y + ols_data_rhs.sum_y
    result.sum_xx = ols_data_lhs.sum_xx + ols_data_rhs.sum_xx
    result.sum_yy = ols_data_lhs.sum_yy + ols_data_rhs.sum_yy
    result.sum_xy = ols_data_lhs.sum_xy + ols_data_rhs.sum_xy
    return result

cdef OLSData subtract(OLSData ols_data_lhs, OLSData ols_data_rhs):
    cdef OLSData result
    result.num = ols_data_lhs.num - ols_data_rhs.num
    result.sum_x = ols_data_lhs.sum_x - ols_data_rhs.sum_x
    result.sum_y = ols_data_lhs.sum_y - ols_data_rhs.sum_y
    result.sum_xx = ols_data_lhs.sum_xx - ols_data_rhs.sum_xx
    result.sum_yy = ols_data_lhs.sum_yy - ols_data_rhs.sum_yy
    result.sum_xy = ols_data_lhs.sum_xy - ols_data_rhs.sum_xy
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double sum(double[:] arr):
    cdef double result = 0.0
    cdef size_t length = arr.shape[0]
    for i in xrange(length):
        result += arr[i]
    return result

# TODO: when do we want these?


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double vdot(double[:] lhs_arr, double[:] rhs_arr):
    cdef double result = 0.0
    cdef size_t length = lhs_arr.shape[0]
    for i in xrange(length):
        result += lhs_arr[i] * rhs_arr[i]
    return result

# TODO: pointers, references (is this copying under the hood for return?)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef OLSData ols_data(double[:] x_arr, double[:] y_arr):
    cdef OLSData result
    result.num = x_arr.shape[0]
    result.sum_x = sum(x_arr)
    result.sum_y = sum(y_arr)
    result.sum_xx = vdot(x_arr, x_arr)
    result.sum_yy = vdot(y_arr, y_arr)
    result.sum_xy = vdot(x_arr, y_arr)
    return result

# NOTE: c++ NAN is a float, but double seems ok here and ties out with other
# pure python impl -- somehow we want double here and not float (TODO check why)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef OlsEstTerms ols_from_formula_with_rss_cimpl(OLSData ols_data, double slope=NAN):
    """
    Assumes intercept.
    1D
    """
    cdef size_t num_data = ols_data.num
    cdef double sum_x = ols_data.sum_x
    cdef double sum_y = ols_data.sum_y
    cdef double sum_xx = ols_data.sum_xx
    cdef double sum_yy = ols_data.sum_yy
    cdef double sum_xy = ols_data.sum_xy

    cdef double mean_x = sum_x / num_data
    cdef double mean_y = sum_y / num_data

    if isnan(slope):
        slope = (num_data * sum_xy - sum_y * sum_x) / (num_data * sum_xx - sum_x * sum_x)

    cdef double intercept = mean_y - slope * mean_x

    cdef double two_intercept = 2.0 * intercept

    cdef double rss = (sum_yy - two_intercept * sum_y - 2.0 * slope * sum_xy +
                       intercept * intercept * num_data + two_intercept * slope * sum_x + slope * slope * sum_xx)

    cdef OlsEstTerms ols_est_terms
    ols_est_terms.intercept = intercept
    ols_est_terms.slope = slope
    ols_est_terms.rss = rss

    return ols_est_terms
