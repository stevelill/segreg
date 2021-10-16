"""
Alternative implementations of segmented regression routines.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np

from segreg.model import one_bkpt_segreg
from segreg.model import regression
from segreg.model.alt import regression_alt, one_bkpt_segreg_alt


try:
    from numba import jit
except ImportError as e:
    from segreg.mockjit import jit


def segmented_func_impl(x, params):
    """
    PARAMETERS
    ----------
    x: array-like (non-scalar)
    """
    # TODO: REMEMBER THIS FUNCTION GIVES ODD RESULTS WITH INTEGER INPUT
    x_arr = np.array(x, dtype=float)

    u1, v1, u2, v2, m1, m2 = params

    mid_slope = (v2 - v1) / (u2 - u1)

    # we sort the data
    argsort_inds = x_arr.argsort()

    sorted_arr = x_arr[argsort_inds]

    first = sorted_arr[sorted_arr <= u1]
    second = sorted_arr[np.logical_and(u1 < sorted_arr, sorted_arr <= u2)]
    third = sorted_arr[u2 < sorted_arr]

    first_vals = v1 + m1 * (first - u1)
    second_vals = v1 + mid_slope * (second - u1)
    third_vals = v2 + m2 * (third - u2)

#    print "IN FUNC"
#    print "------------------------"
#    print x_arr <= u1
#    print "VAL1: ", v1 + m1 * (x_arr - u1)
#    print "------------------------"
#    print np.logical_and(u1 < x_arr, x_arr <= u2)
#    print "VAL2: ", v1 + mid_slope * (x_arr - u1)
#    print "------------------------"
#    print u2 < x_arr
#    print "VAL: ", v2 + m2*(x_arr-u2)
#    print
#
#    print "first"
#    print first
#    print "second"
#    print second
#    print "third"
#    print third
#    print "first vals"
#    print first_vals
#    print "second vals"
#    print second_vals
#    print "third vals"
#    print third_vals

    sorted_result = np.append(first_vals, second_vals)
    sorted_result = np.append(sorted_result, third_vals)

    result = sorted_result[argsort_inds]

    return result

# TODO: duped: get rid of this impl


def segmented_func(x, params):
    if np.isscalar(x):
        result = segmented_func_impl([x], params)
        return result[0]
    else:
        return segmented_func_impl(x, params)

# NOTE: bug in scipy?  does not work when three or more conditions and input
# is a scalar


def segmented_funcORIG(x, params):
    # TODO: REMEMBER THIS FUNCTION GIVES ODD RESULTS WITH INTEGER INPUT
    x_arr = np.array(x, dtype=float)

    u1, v1, u2, v2, m1, m2 = params

    mid_slope = (v2 - v1) / (u2 - u1)

    first = x_arr[x_arr <= u1]

    print("IN FUNC")
    print(u2 < x_arr)
    print("VAL: ", v2 + m2 * (x - u2))
    print()

    return np.piecewise(x_arr,
                        [x_arr <= u1,
                         np.logical_and(u1 < x_arr, x_arr <= u2),
                         u2 < x_arr],
                        [lambda z: v1 + m1 * (z - u1),
                         lambda z: v1 + mid_slope * (z - u1),
                         lambda z: v2 + m2 * (z - u2)])


@jit(nopython=True)
def fixed_bkpt_ls_regression(indep, dep, u1, u2):
    """
    Pure python implementation of the main cython impl:
    two_bkpt_segreg.fixed_bkpt_least_squares

    Segmented function params: (u1,v1,u2,v2,m1,m2), where (u1,v1) and (u2,v2)
    are breakpoints, and m1,m2 are the slope of the line segments in regions
    1 and 3 (the slope in region 2 being determined)

    This implementation uses the regression formula Section 5.1.1 of
    "Segmented Regression" by Steven Lillywhite.  It gives the same answer as
    the method fixed_bkpt_ls
    """

    index1 = np.searchsorted(indep, u1, side='right')
    index2 = np.searchsorted(indep, u2, side='right')

    data_shiftu1 = indep - u1
    data_shiftu2 = indep - u2

    diff = u2 - u1

    data0 = 1.0 - np.copy(data_shiftu1) / diff
    data0[0:index1] = 1.0
    data0[index2:] = 0.0

    data1 = np.copy(data_shiftu1) / diff
    data1[0:index1] = 0.0
    data1[index2:] = 1.0

    data2 = np.copy(data_shiftu1)
    data2[index1:] = 0.0

    data3 = np.copy(data_shiftu2)
    data3[0:index2] = 0.0

    data = np.vstack((data0, data1, data2, data3))
    data = data.T

    # matrix mult by hand faster than canned OLS routines
    dep = dep.reshape(-1, 1)
    v1, v2, m1, m2 = regression_alt.mat_by_hand_ols(data, dep)

    return v1, v2, m1, m2


def fixed_bkpt_ls_from_data(indep, dep, u1, u2):
    index1 = np.searchsorted(indep, u1, side='right')
    index2 = np.searchsorted(indep, u2, side='right')
    indep1 = indep[0:index1]
    dep1 = dep[0:index1]
    indep2 = indep[index1:index2]
    dep2 = dep[index1:index2]
    indep3 = indep[index2:]
    dep3 = dep[index2:]

    ols_terms_1 = regression_alt.ols_terms(indep1, dep1)
    ols_terms_2 = regression_alt.ols_terms(indep2, dep2)
    ols_terms_3 = regression_alt.ols_terms(indep3, dep3)

    return fixed_bkpt_ls(ols_terms_1, ols_terms_2, ols_terms_3, u1, u2)


@jit(nopython=True, cache=False)
def fixed_bkpt_ls(ols_terms_1, ols_terms_2, ols_terms_3, u1, u2):
    """
    Pure python implementation of the main cython impl:
    two_bkpt_segreg.fixed_bkpt_least_squares

    Segmented function params: (u1,v1,u2,v2,m1,m2), where (u1,v1) and (u2,v2)
    are breakpoints, and m1,m2 are the slope of the line segments in regions
    1 and 3 (the slope in region 2 being determined)

    NOTES
    -----
    The notation below follows the document
        "Segmented Regression" by Steven Lillywhite
    """
    num_data_1, sum_x_1, sum_y_1, sum_xx_1, sum_yy_1, sum_xy_1 = ols_terms_1
    num_data_2, sum_x_2, sum_y_2, sum_xx_2, sum_yy_2, sum_xy_2 = ols_terms_2
    num_data_3, sum_x_3, sum_y_3, sum_xx_3, sum_yy_3, sum_xy_3 = ols_terms_3

    u1_sq = u1 * u1
    u2_sq = u2 * u2
    two_u1 = 2.0 * u1
    two_u2 = 2.0 * u2
    diff = u2 - u1
    diff_sq = diff * diff

    A1 = sum_y_1
    A2 = sum_y_2
    A3 = sum_y_3

    B11 = sum_xy_1 - u1 * A1
    B22 = sum_xy_2 - u2 * A2
    B21 = sum_xy_2 - u1 * A2
    B32 = sum_xy_3 - u2 * A3

    C11 = sum_x_1 - u1 * num_data_1
    C21 = sum_x_2 - u1 * num_data_2
    C32 = sum_x_3 - u2 * num_data_3

    D11 = sum_xx_1 - two_u1 * sum_x_1 + u1_sq * num_data_1
    D22 = sum_xx_2 - two_u2 * sum_x_2 + u2_sq * num_data_2
    D21 = sum_xx_2 - two_u1 * sum_x_2 + u1_sq * num_data_2
    D32 = sum_xx_3 - two_u2 * sum_x_3 + u2_sq * num_data_3

    E = sum_yy_1 + sum_yy_2 + sum_yy_3

    F2 = sum_xx_2 - (u1 + u2) * sum_x_2 + u1 * u2 * num_data_2

    ##
    term = D21 / diff_sq
    a = -num_data_1 + C11 * C11 / D11 - D22 / diff_sq
    b = F2 / diff_sq
    c = b
    d = -num_data_3 + C32 * C32 / D32 - term
    e = -A1 + B11 * C11 / D11 + B22 / diff
    f = -A3 + B32 * C32 / D32 - B21 / diff

    # v estimates
    v1, v2 = regression_alt.invert_two_by_two(a, b, c, d, e, f)

    ## BEGIN: slopes
    m1 = (B11 - v1 * C11) / D11
    m2 = (B32 - v2 * C32) / D32
    ## END: slopes

    m = (v2 - v1) / (u2 - u1)
    two_v1 = 2.0 * v1
    two_v2 = 2.0 * v2
    rss = (E - two_v1 * (A1 + A2) - two_v2 * A3
           - 2.0 * m1 * B11 - 2.0 * m * B21 - 2.0 * m2 * B32
           + v1 * v1 * (num_data_1 + num_data_2) + v2 * v2 * num_data_3
           + two_v1 * (m1 * C11 + m * C21) + two_v2 * m2 * C32
           + m1 * m1 * D11 + m * m * D21 + m2 * m2 * D32)

    return v1, v2, m1, m2, rss


@jit(nopython=True, cache=False)
def estimate_two_bkpt_segreg(indep,
                             dep,
                             num_end_to_skip=3,
                             num_between_to_skip=4,
                             verbose=False,
                             optimize=True):
    # TODO: can we raise Exception with Numba?
    # if num_between_to_skip < 2:
    #    pass
        #raise Exception("num_between_to_skip must be greater than zero")

    min_value = np.inf
    min_params = None

    # list call gets set out of order, so we call sort here on indices
    unique_indep = list(set(indep))
    unique_indep_lhs_indices = np.searchsorted(indep, unique_indep)
    unique_indep_lhs_indices.sort()
    unique_indep = indep[unique_indep_lhs_indices]

    # STEP 2. check for local min in intervals between data points
    # NOTE: we use array mask with minus, like: x[0:-3]
    # these are indices of LHS of allowable intervals to check
    num_uniq = len(unique_indep)
    index1_begin = num_end_to_skip + 1
    index2_end = num_uniq - num_end_to_skip - 3
    index1_end = index2_end - num_between_to_skip

    if verbose:
        print()
        print("indep")
        print(indep)
        print()
        print("unique_indep_lhs_indices")
        print(unique_indep_lhs_indices)
        print()
        print()
        print("num_end_to_skip: ", num_end_to_skip)
        print("num_between_to_skip: ", num_between_to_skip)
        print("unique range: ", np.arange(num_uniq))
        print("index1_begin: ", index1_begin)
        print("index1_end: ", index1_end)
        print("index2_end: ", index2_end)
        print()

    index1_range = np.arange(index1_begin, index1_end + 1)

    for index1 in index1_range:
        index2_begin = index1 + num_between_to_skip

        index2_range = np.arange(index2_begin, index2_end + 1)

        for index2 in index2_range:

            ind1 = unique_indep_lhs_indices[index1 + 1]
            ind2 = unique_indep_lhs_indices[index2 + 1]

            indep1 = indep[0:ind1]
            dep1 = dep[0:ind1]
            indep2 = indep[ind1:ind2]
            dep2 = dep[ind1:ind2]
            indep3 = indep[ind2:]
            dep3 = dep[ind2:]

            # TODO: put in check that mins actually hit
            if len(indep1) < num_end_to_skip:
                raise Exception("region one has too few data")
            if len(indep2) < num_between_to_skip:
                raise Exception("region two has too few data")
            if len(indep3) < num_end_to_skip:
                raise Exception("region three has too few data")

            u1_data = indep1[-1]
            u1_data_next = indep2[0]

            u2_data = indep2[-1]
            u2_data_next = indep3[0]

            if verbose:
                print()
                print("--------------------------------------------------------")
                print("INDICES: ", index1, ",", index2)
                print()
                print("u1 interval: ", u1_data, ", ", u1_data_next)
                print("u2 interval: ", u2_data, ", ", u2_data_next)
                print()
                print("indep1: len: ", len(indep1))
                print(indep1)
                print("indep2: len: ", len(indep2))
                print(indep2)
                print("indep3: len: ", len(indep3))
                print(indep3)
                print()

            ###################################################################
            # interior of square
            # check regressions for right place
            #
            # get out early trick: based on fact that unrestricted regressions
            # rss1 + rss2 + rss3 is lower bound for the segmented solution rss
            ###################################################################
            (ols_intercept1,
             ols_slope1,
             rss1,
             ols_terms1) = regression_alt.ols_verbose(indep1, dep1)

            # TRICK: get out early if possible
            if rss1 > min_value and optimize:
                if verbose:
                    print("OUT EARLY on rss1")
                continue

            (ols_intercept2,
             ols_slope2,
             rss2,
             ols_terms2) = regression_alt.ols_verbose(indep2, dep2)

            # TRICK: get out early if possible
            if rss1 + rss2 > min_value and optimize:
                if verbose:
                    print("OUT EARLY on rss1 + rss2")
                continue

            (ols_intercept3,
             ols_slope3,
             rss3,
             ols_terms3) = regression_alt.ols_verbose(indep3, dep3)

            rss = rss1 + rss2 + rss3

            if rss > min_value and optimize:
                if verbose:
                    print("OUT EARLY on rss1 + rss2 + rss3")
                continue

            lhs_slope_diff = ols_slope2 - ols_slope1
            rhs_slope_diff = ols_slope3 - ols_slope2

            non_zero_slopes = abs(lhs_slope_diff) > 1.0e-14 and abs(rhs_slope_diff) > 1.0e-14

            # either or both slopes are zero, then these cases will get covered
            # by the square boundary calculations (edges, corners)

            if non_zero_slopes:
                u1_intersect = (ols_intercept1 - ols_intercept2) / lhs_slope_diff

                u2_intersect = (ols_intercept2 - ols_intercept3) / rhs_slope_diff

                u1_right_place = ((u1_data < u1_intersect) and
                                  (u1_intersect < u1_data_next))
                u2_right_place = ((u2_data < u2_intersect) and
                                  (u2_intersect < u2_data_next))

                if u1_right_place and u2_right_place:
                    if rss < min_value:
                        min_value = rss
                        v1 = ols_intercept1 + ols_slope1 * u1_intersect
                        v2 = ols_intercept2 + ols_slope2 * u2_intersect
                        min_params = np.array([u1_intersect,
                                               v1,
                                               u2_intersect,
                                               v2,
                                               ols_slope1,
                                               ols_slope3])

                        if verbose:
                            print()
                            print("NEW LOW BOTH IN RIGHT PLACE")
                            print("params: ", min_params)
                            print("RSS: ", rss)
                            print()
                            print("bndies: ", u1_data, u1_data_next)
                            print("bndies: ", u2_data, u2_data_next)
                    continue

            ###################################################################
            # sides of square
            ###################################################################

            ##########
            # fix u1
            ##########

            (check_min_params,
             check_min_value) = _fix_u1_bndy(u1_fixed=u1_data,
                                             ols_terms1=ols_terms1,
                                             ols_terms2=ols_terms2,
                                             ols_intercept3=ols_intercept3,
                                             ols_slope3=ols_slope3,
                                             rss3=rss3,
                                             u2_data=u2_data,
                                             u2_data_next=u2_data_next,
                                             min_value=min_value,
                                             verbose=verbose)

            if check_min_params is not None:
                min_params = check_min_params
                min_value = check_min_value

            (check_min_params,
             check_min_value) = _fix_u1_bndy(u1_fixed=u1_data_next,
                                             ols_terms1=ols_terms1,
                                             ols_terms2=ols_terms2,
                                             ols_intercept3=ols_intercept3,
                                             ols_slope3=ols_slope3,
                                             rss3=rss3,
                                             u2_data=u2_data,
                                             u2_data_next=u2_data_next,
                                             min_value=min_value,
                                             verbose=verbose)

            if check_min_params is not None:
                min_params = check_min_params
                min_value = check_min_value

            ##########
            # fix u2
            ##########

            (check_min_params,
             check_min_value) = _fix_u2_bndy(u2_fixed=u2_data,
                                             ols_terms2=ols_terms2,
                                             ols_terms3=ols_terms3,
                                             ols_intercept1=ols_intercept1,
                                             ols_slope1=ols_slope1,
                                             rss1=rss1,
                                             u1_data=u1_data,
                                             u1_data_next=u1_data_next,
                                             min_value=min_value,
                                             verbose=verbose)

            if check_min_params is not None:
                min_params = check_min_params
                min_value = check_min_value

            (check_min_params,
             check_min_value) = _fix_u2_bndy(u2_fixed=u2_data_next,
                                             ols_terms2=ols_terms2,
                                             ols_terms3=ols_terms3,
                                             ols_intercept1=ols_intercept1,
                                             ols_slope1=ols_slope1,
                                             rss1=rss1,
                                             u1_data=u1_data,
                                             u1_data_next=u1_data_next,
                                             min_value=min_value,
                                             verbose=verbose)

            if check_min_params is not None:
                min_params = check_min_params
                min_value = check_min_value

            ###################################################################
            # check corner boundaries
            ###################################################################

            (check_min_params,
             check_min_value) = _corner(u1=u1_data,
                                        u2=u2_data,
                                        ols_terms1=ols_terms1,
                                        ols_terms2=ols_terms2,
                                        ols_terms3=ols_terms3,
                                        min_value=min_value,
                                        verbose=verbose)

            if check_min_params is not None:
                min_params = check_min_params
                min_value = check_min_value

            (check_min_params,
             check_min_value) = _corner(u1=u1_data_next,
                                        u2=u2_data,
                                        ols_terms1=ols_terms1,
                                        ols_terms2=ols_terms2,
                                        ols_terms3=ols_terms3,
                                        min_value=min_value,
                                        verbose=verbose)

            if check_min_params is not None:
                min_params = check_min_params
                min_value = check_min_value

            (check_min_params,
             check_min_value) = _corner(u1=u1_data,
                                        u2=u2_data_next,
                                        ols_terms1=ols_terms1,
                                        ols_terms2=ols_terms2,
                                        ols_terms3=ols_terms3,
                                        min_value=min_value,
                                        verbose=verbose)

            if check_min_params is not None:
                min_params = check_min_params
                min_value = check_min_value

            (check_min_params,
             check_min_value) = _corner(u1=u1_data_next,
                                        u2=u2_data_next,
                                        ols_terms1=ols_terms1,
                                        ols_terms2=ols_terms2,
                                        ols_terms3=ols_terms3,
                                        min_value=min_value,
                                        verbose=verbose)

            if check_min_params is not None:
                min_params = check_min_params
                min_value = check_min_value

    return min_params, min_value


@jit(nopython=True)
def _corner(u1,
            u2,
            ols_terms1,
            ols_terms2,
            ols_terms3,
            min_value,
            verbose):
    v1, v2, m1, m2, rss = fixed_bkpt_ls(ols_terms1,
                                        ols_terms2,
                                        ols_terms3,
                                        u1,
                                        u2)

    min_params = None

    if rss < min_value:
        min_value = rss
        min_params = np.array([u1, v1, u2, v2, m1, m2])

        if verbose:
            print()
            print("NEW LOW WITH CORNER VALUE")
            print("params: ", min_params)
            print("RSS: ", rss)
            print("CORNER: u1: ", u1, " ; u2: ", u2)
            print()

    return min_params, min_value


@jit(nopython=True)
def _fix_u1_bndy(u1_fixed,
                 ols_terms1,
                 ols_terms2,
                 ols_intercept3,
                 ols_slope3,
                 rss3,
                 u2_data,
                 u2_data_next,
                 min_value,
                 verbose):
    v1, m1, m2, rss_12 = one_bkpt_segreg_alt._fixed_bkpt_ls_impl(ols_terms1,
                                                                 ols_terms2,
                                                                 u1_fixed)

    rss = rss_12 + rss3

    min_params = None

    # TODO: what if equals here?  (ie: two solutions?)
    if rss < min_value:

        slope_diff = ols_slope3 - m2

        if abs(slope_diff) > 1.0e-14:

            u2_intersect = (v1 - ols_intercept3 - m2 * u1_fixed) / slope_diff
            u2_right_place = ((u2_data < u2_intersect) and
                              (u2_intersect < u2_data_next))

            if u2_right_place:
                min_value = rss
                v2 = ols_intercept3 + ols_slope3 * u2_intersect
                min_params = np.array([u1_fixed,
                                       v1,
                                       u2_intersect,
                                       v2,
                                       m1,
                                       ols_slope3])

                if verbose:
                    print()
                    print("NEW LOW FIXED u1=", u1_fixed)
                    print("u2 interval")
                    print(u2_data, u2_data_next)
                    print("u2 intersect: ", u2_intersect)
                    print()
                    print("RSS: ", rss)

    return min_params, min_value


@jit(nopython=True)
def _fix_u2_bndy(u2_fixed,
                 ols_terms2,
                 ols_terms3,
                 ols_intercept1,
                 ols_slope1,
                 rss1,
                 u1_data,
                 u1_data_next,
                 min_value,
                 verbose):
    v2, m2, m3, rss_23 = one_bkpt_segreg_alt._fixed_bkpt_ls_impl(ols_terms2,
                                                                 ols_terms3,
                                                                 u2_fixed)

    rss = rss1 + rss_23

    min_params = None

    if rss < min_value:
        slope_diff = m2 - ols_slope1

        if abs(slope_diff) > 1.0e-14:
            u1_intersect = (ols_intercept1 - v2 + m2 * u2_fixed) / slope_diff

            u1_right_place = ((u1_data < u1_intersect) and (u1_intersect < u1_data_next))

            if u1_right_place:
                min_value = rss
                v1 = ols_intercept1 + ols_slope1 * u1_intersect
                min_params = np.array([u1_intersect,
                                       v1,
                                       u2_fixed,
                                       v2,
                                       ols_slope1,
                                       m3])

                if verbose:
                    print()
                    print("NEW LOW FIXED u2=", u2_fixed)
                    print("u1 interval")
                    print(u1_data, u1_data)
                    print("u1 intersect: ", u1_intersect)
                    print()
                    print("RSS: ", rss)

    return min_params, min_value
