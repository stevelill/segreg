"""
Alternative implementations of segmented regression routines.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import numpy as np
from segreg.model.alt import regression_alt, likelihood_util


try:
    from numba import jit
except ImportError as e:
    from segreg.mockjit import jit

##########################################################################
# Hessian
##########################################################################


def ols_terms(indep, dep, u):
    index_to_use = np.searchsorted(indep, u, side='right')
    indep1 = indep[0:index_to_use]
    dep1 = dep[0:index_to_use]
    indep2 = indep[index_to_use:]
    dep2 = dep[index_to_use:]

    ols_terms1 = regression_alt.ols_terms(indep1, dep1)
    ols_terms2 = regression_alt.ols_terms(indep2, dep2)
    return ols_terms1, ols_terms2


def rss(params, ols_data1, ols_data2):
    u, v, m1, m2 = params

#    rss1 = rss_for_region2([u, v, m1], ols_data1)
#    rss2 = rss_for_region2([u, v, m2], ols_data2)

    rss1 = likelihood_util.rss_line_segment([u, v, m1], ols_data1)
    rss2 = likelihood_util.rss_line_segment([u, v, m2], ols_data2)

    return rss1 + rss2


def one_bkpt_loglik(params, indep, dep):
    """
    Parameters
    ----------
    params: list
        [u, v, m1, m2, residual_variance]
    indep: array-like
        independent data
    dep: array-like
        dependent data
    """
    u, v, m1, m2, resid_variance = params

    ols_data1, ols_data2 = ols_terms(indep, dep, u)

    rss_term = rss([u, v, m1, m2], ols_data1, ols_data2)

    num_data = ols_data1[0] + ols_data2[0]

    result = likelihood_util.loglikelihood(rss=rss_term,
                                           resid_variance=resid_variance,
                                           num_data=num_data)

    return result


def rss_for_region2(params, ols_data):
    # deprecated
    """
    DEPRECATED: use likelihood_util.rss_line_segment
    """

    print("DEPRECATED: use likelihood_util.rss_line_segment")

    u, v, m = params

    num_data = ols_data[0]
    sum_x = ols_data[1]
    sum_y = ols_data[2]
    sum_xx = ols_data[3]
    sum_yy = ols_data[4]
    sum_xy = ols_data[5]

    two_m = 2.0 * m
    mm = m * m
    mmu = mm * u
    return ((v * v - two_m * u * v + mmu * u) * num_data
            + 2.0 * (m * v - mmu) * sum_x
            + 2.0 * (m * u - v) * sum_y
            + mm * sum_xx
            + sum_yy
            - two_m * sum_xy)


##########################################################################
# End Hessian
##########################################################################


def segmented_func(u, v, m1, m2):
    """
    NOTE: dupe from one_bkpt_segreg
    """
    def func(x):

        # this guards against incorrect results when array of int passed in
        x_arr = np.asarray(x, dtype=float)

        return np.piecewise(x_arr,
                            [x_arr <= u,
                             x_arr > u],
                            [lambda x: v + m1 * (x - u),
                             lambda x: v + m2 * (x - u)])
    return func


@jit(nopython=True)
def _fixed_bkpt_ls_impl(ols_terms_1, ols_terms_2, u):
    """
    Computes the segmented fit for a fixed value of the break point.

    This implementation uses the formulas of Section 6.2 of 
    "Segmented Regression" by Steven Lillywhite.

    Note
    ----
    Does not depend on segreg cython.

    See Also
    --------
    fixed_bkpt_ls_regression
    one_bkpt_segreg.fixed_bkpt_least_squares

    Parameters
    ----------
    ols_terms_1

    ols_terms_2

    u: float
        the x-coordinate of the breakpoint

    Returns
    -------
    v: float
        the y-coordinate of the breakpoint ( ie: bkpt is (u,v) )
    m1: float
        the left-hand-side slope
    m2: float
        the right-hand side slope

    """
    num_data_1, sum_x_1, sum_y_1, sum_xx_1, sum_yy_1, sum_xy_1 = ols_terms_1
    num_data_2, sum_x_2, sum_y_2, sum_xx_2, sum_yy_2, sum_xy_2 = ols_terms_2

    sum_x_minus_u_sq_1 = sum_xx_1 - 2.0 * u * sum_x_1 + u * u * num_data_1
    sum_x_minus_u_sq_2 = sum_xx_2 - 2.0 * u * sum_x_2 + u * u * num_data_2

    # BEGIN: v
    v_num_term_1 = (sum_xy_1 - u * sum_y_1) * \
        (sum_x_1 - u * num_data_1) / sum_x_minus_u_sq_1
    v_num_term_2 = (sum_xy_2 - u * sum_y_2) * \
        (sum_x_2 - u * num_data_2) / sum_x_minus_u_sq_2

    v_numerator = sum_y_1 + sum_y_2 - v_num_term_1 - v_num_term_2

    piece1 = (sum_x_1 - u * num_data_1)
    v_denom_term_1 = piece1 * piece1 / sum_x_minus_u_sq_1
    piece2 = (sum_x_2 - u * num_data_2)
    v_denom_term_2 = piece2 * piece2 / sum_x_minus_u_sq_2

    v_denominator = num_data_1 + num_data_2 - v_denom_term_1 - v_denom_term_2

    v = v_numerator / v_denominator
    # END: v

    # BEGIN: slopes
    uv = u * v
    m1_numerator = sum_xy_1 - v * sum_x_1 - u * sum_y_1 + uv * num_data_1
    m2_numerator = sum_xy_2 - v * sum_x_2 - u * sum_y_2 + uv * num_data_2

    m1 = m1_numerator / sum_x_minus_u_sq_1
    m2 = m2_numerator / sum_x_minus_u_sq_2
    # END: slopes

    rss = rss_for_region(ols_terms_1, u, v, m1) + \
        rss_for_region(ols_terms_2, u, v, m2)
    return v, m1, m2, rss


@jit(nopython=True)
def rss_for_region(ols_data, u, v, m):

    num_data = ols_data[0]
    sum_x = ols_data[1]
    sum_y = ols_data[2]
    sum_xx = ols_data[3]
    sum_yy = ols_data[4]
    sum_xy = ols_data[5]

    two_m = 2.0 * m
    mm = m * m
    mmu = mm * u
    return ((v * v - two_m * u * v + mmu * u) * num_data
            + 2.0 * (m * v - mmu) * sum_x
            + 2.0 * (m * u - v) * sum_y
            + mm * sum_xx
            + sum_yy
            - two_m * sum_xy)


@jit(nopython=True)
def fixed_bkpt_ls(indep, dep, u):
    index_to_use = np.searchsorted(indep, u, side='right')
    indep1 = indep[0:index_to_use]
    dep1 = dep[0:index_to_use]
    indep2 = indep[index_to_use:]
    dep2 = dep[index_to_use:]

    ols_terms1 = regression_alt.ols_terms(indep1, dep1)
    ols_terms2 = regression_alt.ols_terms(indep2, dep2)
    return _fixed_bkpt_ls_impl(ols_terms1, ols_terms2, u)


@jit(nopython=True)
def fixed_bkpt_ls_regression(indep, dep, u):
    """
    Computes the segmented fit for a fixed value of the break point.

    This implementation uses the regression formula Section 4.1.1 of
    "Segmented Regression" by Steven Lillywhite.  In particular, it does not
    depend on specialized formulas of Section 6.2.

    Note
    ----
    Does not depend on segreg cython.

    See Also
    --------
    _fixed_bkpt_ls_impl
    one_bkpt_segreg.fixed_bkpt_least_squares

    Parameters
    ----------
    indep: array-like
        independent variable data
    dep: array-like
        dependent variable data
    u: float
        the x-coordinate of the breakpoint

    Returns
    -------
    v: float
        the y-coordinate of the breakpoint ( ie: bkpt is (u,v) )
    m1: float
        the left-hand-side slope
    m2: float
        the right-hand side slope
    """
    index_to_use = np.searchsorted(indep, u, side='right')

    shift_indep = indep - u

    lhs_ols_data = np.copy(shift_indep)
    lhs_ols_data[index_to_use:] = 0.0
    rhs_ols_data = shift_indep
    rhs_ols_data[0:index_to_use] = 0.0

    # NOTE: numba cannot handle this
    #data = np.array([lhs_ols_data, rhs_ols_data])

#    data = np.vstack((lhs_ols_data, rhs_ols_data))
#    data = data.T

    data = np.column_stack((lhs_ols_data, rhs_ols_data))

    # this calls third-party libs and is slower and also incompatible with jit
    #est_const, est_mat = regression_alt.bare_bones_ols(data, dep)

    est_const, est_mat = regression_alt.matrix_ols(data, dep)

    v = est_const
    m1 = est_mat[0]
    m2 = est_mat[1]

    return v, m1, m2


@jit(nopython=True, cache=False)
def estimate_one_bkpt_segreg(indep,
                             dep,
                             num_end_to_skip=3,
                             verbose=False,
                             extra_verbose=False,
                             optimize=True,
                             check_near_middle=False):
    """
    NOTES
    -----
    This implementation does not compute anything when there are 2n+1 distinct
    data points, and num_end_to_skip is n-1.  In such a case, there is only
    one possible configuration, where the bkpt is at the nth data point.
    """
    min_value = np.inf
    min_params = None

    # list call gets set out of order, so we call sort here on indices
    unique_indep = list(set(indep))
    unique_indep_lhs_indices = np.searchsorted(indep, unique_indep)
    unique_indep_lhs_indices.sort()
    unique_indep = indep[unique_indep_lhs_indices]

    num_data = len(unique_indep)

    # STEP 2. check for local min in intervals between data points
    index1_begin = num_end_to_skip + 2
    index1_end = num_data - num_end_to_skip - 1

    indices_to_check = unique_indep_lhs_indices[index1_begin:index1_end]

    last_index = indices_to_check[-1]

    for ind in indices_to_check:

        indep1 = indep[0:ind]
        dep1 = dep[0:ind]
        indep2 = indep[ind:]
        dep2 = dep[ind:]

        if verbose:
            print()
            print("-" * 50)
            print("lhs indep")
            print(indep1)
            print()
            print("rhs indep")
            print(indep2)
            print()

        # TODO: put in check that mins actually hit
        if len(indep1) < num_end_to_skip:
            raise Exception("region one has too few data")
        if len(indep2) < num_end_to_skip:
            raise Exception("region two has too few data")

        u1_data = indep1[-1]
        u1_data_next = indep2[0]

        if verbose:
            print()
            print("u1_data:      ", u1_data)
            print("u1_data_next: ", u1_data_next)

        ####################################################################
        # interior of interval
        # check regressions for right place
        ####################################################################
        (ols_intercept1,
         ols_slope1,
         rss1,
         ols_terms1) = regression_alt.ols_verbose(indep1, dep1)

        # TRICK: get out early if possible
        if rss1 > min_value and optimize:
            continue

        (ols_intercept2,
         ols_slope2,
         rss2,
         ols_terms2) = regression_alt.ols_verbose(indep2, dep2)

        rss = rss1 + rss2

        # TRICK: get out early if possible
        if rss > min_value and optimize:
            continue

        slope_diff = ols_slope2 - ols_slope1

        if abs(slope_diff) > 1.0e-14:

            u1_intersect = (ols_intercept1 - ols_intercept2) / slope_diff
            u1_right_place = (u1_data < u1_intersect) and (u1_intersect < u1_data_next)

            if verbose:
                print()
                print("OLS INTERSECTION: ", u1_intersect)

            if u1_right_place:
                if rss < min_value:
                    min_value = rss
                    v1 = ols_intercept1 + ols_slope1 * u1_intersect
                    min_params = [u1_intersect, v1, ols_slope1, ols_slope2]

                    if verbose:
                        print()
                        print("NEW LOW BOTH IN RIGHT PLACE")
                        print("params: ", min_params)
                        print("RSS: ", rss)
                        print()
                        print("bndies: ", u1_data, u1_data_next)

        ####################################################################
        # check corner boundaries
        ####################################################################

        v1, m1, m2, rss = _fixed_bkpt_ls_impl(ols_terms1,
                                              ols_terms2,
                                              u1_data)

        if verbose:
            print()
            print("checking corner u: ", u1_data, " ; RSS: ", rss)

        if rss < min_value:
            min_value = rss
            min_params = [u1_data, v1, m1, m2]

            if verbose:
                print()
                print("NEW LOW WITH CORNER VALUE")
                print("params: ", min_params)
                print("RSS: ", rss)
                print()

        # we check lhs endpoint for each interval; we thus miss rhs of last
        # interval; so we need to check it separately
        if ind == last_index:
            v1, m1, m2, rss = _fixed_bkpt_ls_impl(ols_terms1,
                                                  ols_terms2,
                                                  u1_data_next)

            if verbose:
                print()
                print("checking LAST corner u: ", u1_data_next, " ; RSS: ", rss)

            if rss < min_value:
                min_value = rss
                min_params = [u1_data_next, v1, m1, m2]

                if verbose:
                    print()
                    print("NEW LOW WITH CORNER VALUE")
                    print("params: ", min_params)
                    print("RSS: ", rss)
                    print()

    return min_params, min_value


def estimate_one_bkpt_segreg_basic(indep,
                                   dep,
                                   num_end_to_skip=2,
                                   verbose=False):
    """
    Implements one bkpt segmented regression without using specialized formulas
    for regression calculations.

    This is intended to be an implementation that uses off-the-shelf python 
    libraries without any special treatment of the algorithms.  This may be 
    thought of as a somewhat "brute force" implementation.
    """

    def ols_func(indep, dep):
        (intercept,
         slope,
         rss,
         dummy) = regression_alt.ols_verbose(indep, dep)
        return intercept, slope, rss

    return _estimate_one_bkpt_segreg_basic_impl(indep=indep,
                                                dep=dep,
                                                num_end_to_skip=num_end_to_skip,
                                                ols_func=ols_func,
                                                verbose=verbose)


def _estimate_one_bkpt_segreg_basic_impl(indep,
                                         dep,
                                         num_end_to_skip=2,
                                         ols_func=regression_alt.ols_verbose,
                                         fixed_bkpt_ls_func=fixed_bkpt_ls_regression,
                                         verbose=False):
    """
    PARAMETERS
    ----------
    ols_func: takes params (indep, dep), performs OLS regression, and returns
        slope, intercept, rss
    fixed_bkpt_ls_func: takes params (indep, dep, u), performs segmented
        regression with fixed bkpt u, and returns (v, m1, m2), the y-coord
        of the bkpt, the lhs slope, and the rhs slope
    """

    min_value = np.inf

    # list call gets set out of order, so we call sort here on indices
    unique_indep = list(set(indep))
    unique_indep_lhs_indices = np.searchsorted(indep, unique_indep)
    unique_indep_lhs_indices.sort()
    unique_indep = indep[unique_indep_lhs_indices]

    num_data = len(unique_indep)
    index1_begin = num_end_to_skip + 1
    index1_end = num_data - num_end_to_skip - 2

    data_points_to_check = unique_indep[index1_begin:index1_end + 1]

    if verbose:
        print()
        print("checking at data points:")
        print(data_points_to_check)
        print()

    # STEP 1. evaluate at the data points
    for u in data_points_to_check:

        v, m1, m2 = fixed_bkpt_ls_func(indep, dep, u)

        curr_params = [u, v, m1, m2]

        func = segmented_func(u, v, m1, m2)
        residuals = dep - func(indep)
        func_val = np.vdot(residuals, residuals)

        if verbose:
            print("data point check ; u: ", u, " ; RSS: ", func_val)
            print("diff: ", func_val - min_value)
            print()

        if(func_val < min_value):
            min_params = curr_params
            min_value = func_val

    # STEP 2. check for local min in intervals between data points
    # for index in unique_indep_lhs_indices[range_begin + 1:-range_begin]:

    if verbose:
        print()
        print("unique indices")
        print(unique_indep_lhs_indices)
        print("unique indep")
        print(unique_indep)
        print()

    for index in range(index1_begin, index1_end):

        index_to_use = unique_indep_lhs_indices[index + 1]

        lhs_indep = indep[0:index_to_use]
        lhs_dep = dep[0:index_to_use]

        rhs_indep = indep[index_to_use:]
        rhs_dep = dep[index_to_use:]

        u_data = lhs_indep[-1]
        u_data_next = rhs_indep[0]

        if verbose:
            print()
            print("lhs indep")
            print(lhs_indep)
            print()
            print("rhs indep")
            print(rhs_indep)
            print()
            print("interval: ", u_data, " , ", u_data_next)
            print()

        (lhs_ols_intercept,
         lhs_ols_slope,
         lhs_rss) = ols_func(lhs_indep, lhs_dep)

        (rhs_ols_intercept,
         rhs_ols_slope,
         rhs_rss) = ols_func(rhs_indep, rhs_dep)

        slope_diff = rhs_ols_slope - lhs_ols_slope

        # note: is slopes same, AND intercepts same, then every point in
        # interval has same rss (since it is one line), and hence the
        # same rss at the endpoints; so we can skip interval in such a case
        # as the endpt check will cover it
        if abs(slope_diff) > 1.0e-14:

            u = (lhs_ols_intercept - rhs_ols_intercept) / slope_diff

            if verbose:
                print("OLS INTERSECTION: ", u)

            if(u_data < u and u < u_data_next):

                func_val = lhs_rss + rhs_rss

                if func_val < min_value:
                    min_value = func_val

                    v_numer = (lhs_ols_intercept * rhs_ols_slope - rhs_ols_intercept * lhs_ols_slope)
                    v_denom = (rhs_ols_slope - lhs_ols_slope)
                    v = v_numer / v_denom
                    min_params = [u, v, lhs_ols_slope, rhs_ols_slope]

    return min_params, min_value
