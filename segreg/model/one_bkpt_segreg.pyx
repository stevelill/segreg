"""
One-Bkpt Segmented Regression core routines.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


# cython: boundscheck=False, wrap_around=False, cdivision=True

from cpython cimport array
cimport cython
import array

import numpy as np
cimport numpy as np

from libcpp.set cimport set
from libc.math cimport isnan
from libc.math cimport NAN

from segreg.model.regression cimport OLSData
from segreg.model.regression cimport OlsEstTerms
from segreg.model.regression cimport add
from segreg.model.regression cimport subtract
from segreg.model.regression cimport ols_data
from segreg.model.regression cimport ols_from_formula_with_rss_cimpl

from segreg.model.one_bkpt_segreg cimport FixedBkptTerms


################################################################################


def segmented_func(u, v, m1, m2):
    """
    TODO: force float here?
    """
    def func(x):

        # TODO: keep?
        x_arr = np.asarray(x, dtype=float)

        result = np.piecewise(x_arr,
                              [x_arr <= u,
                               x_arr > u],
                              [lambda x: v + m1 * (x - u),
                                  lambda x: v + m2 * (x - u)])
        # np.piecewise returns zero-dim array when scalar
        if np.isscalar(x):
            result = float(result)
        return result
    return func


def fixed_bkpt_ls_for_data(indep, dep, bkpt):
    """    
    NOTES
    -----
    assumes input is sorted

    TODO: check where this is used
    """
    index = np.searchsorted(indep, bkpt)

    lhs_indep = indep[0:index]
    lhs_dep = dep[0:index]

    rhs_indep = indep[index:]
    rhs_dep = dep[index:]

    cdef OLSData ols_1
    cdef OLSData ols_2
    ols_1 = ols_data(lhs_indep, lhs_dep)
    ols_2 = ols_data(rhs_indep, rhs_dep)

    cdef FixedBkptTerms fixed_bkpt_terms = fixed_breakpt_ls(ols_1, ols_2, bkpt)
    return (fixed_bkpt_terms.v,
            fixed_bkpt_terms.m1,
            fixed_bkpt_terms.m2,
            fixed_bkpt_terms.rss)


def one_bkpt_seg_profile_rss_func(indep, dep):

    # TODO: sort input here

    def func(u):
        v, m1, m2, rss = fixed_bkpt_ls_for_data(indep, dep, u)
        return rss
    return func

################################################################################


cdef struct OneBkptParams:
    double x0
    double y0
    double lhs_slope
    double rhs_slope

# TODO: add fixed m1
cdef FixedBkptTerms fixed_breakpt_ls(OLSData ols_terms_1,
                                     OLSData ols_terms_2,
                                     double u,
                                     double m2=NAN):
    """
    Computes segmented regression for one break-point conditional on u.
    Segmented function params: (u,v,m1,m2), where u is breakpoint, and v is the
    y-coordinate of the u (u,v); m1,m2 are lhs and rhs slopes, respectively

    PARAMETERS
    ----------
    ols_terms_1 : functions of data on left-hand side of u which are needed in
        regression calculation
    ols_terms_2 : functions of data on right-hand side of u which are needed in
        regression calculation
    """
    cdef size_t num_data_1 = ols_terms_1.num
    cdef double sum_x_1 = ols_terms_1.sum_x
    cdef double sum_y_1 = ols_terms_1.sum_y
    cdef double sum_xx_1 = ols_terms_1.sum_xx
    cdef double sum_yy_1 = ols_terms_1.sum_yy
    cdef double sum_xy_1 = ols_terms_1.sum_xy

    cdef size_t num_data_2 = ols_terms_2.num
    cdef double sum_x_2 = ols_terms_2.sum_x
    cdef double sum_y_2 = ols_terms_2.sum_y
    cdef double sum_xx_2 = ols_terms_2.sum_xx
    cdef double sum_yy_2 = ols_terms_2.sum_yy
    cdef double sum_xy_2 = ols_terms_2.sum_xy

    cdef double sum_x_minus_u_sq_1 = sum_xx_1 - 2.0 * u * sum_x_1 + u * u * num_data_1
    cdef double sum_x_minus_u_sq_2 = sum_xx_2 - 2.0 * u * sum_x_2 + u * u * num_data_2

    ## BEGIN: v
    # numerator
    cdef double v_num_term_1 = (sum_xy_1 - u * sum_y_1) * (sum_x_1 - u * num_data_1) / sum_x_minus_u_sq_1

    cdef double v_num_term_2

    if isnan(m2):
        v_num_term_2 = (sum_xy_2 - u * sum_y_2) * (sum_x_2 - u * num_data_2) / sum_x_minus_u_sq_2
    else:
        v_num_term_2 = m2 * (sum_x_2 - u * num_data_2)

    cdef double v_numerator = sum_y_1 + sum_y_2 - v_num_term_1 - v_num_term_2

    # denominator
    cdef double piece1 = (sum_x_1 - u * num_data_1)
    cdef double v_denom_term_1 = piece1 * piece1 / sum_x_minus_u_sq_1
    cdef double piece2 = (sum_x_2 - u * num_data_2)

    cdef double v_denom_term_2
    if isnan(m2):
        v_denom_term_2 = piece2 * piece2 / sum_x_minus_u_sq_2
    else:
        v_denom_term_2 = 0.0

    cdef double v_denominator = num_data_1 + num_data_2 - v_denom_term_1 - v_denom_term_2

    cdef double v = v_numerator / v_denominator
    ## END: v

    ## BEGIN: slopes
    cdef double uv = u * v

    cdef double m1_numerator = sum_xy_1 - v * sum_x_1 - u * sum_y_1 + uv * num_data_1
    cdef double m1 = m1_numerator / sum_x_minus_u_sq_1

    cdef double m2_numerator
    if isnan(m2):
        m2_numerator = sum_xy_2 - v * sum_x_2 - u * sum_y_2 + uv * num_data_2
        m2 = m2_numerator / sum_x_minus_u_sq_2
    ## END: slopes

    cdef double rss = rss_for_region(ols_terms_1, u, v, m1) + rss_for_region(ols_terms_2, u, v, m2)

    cdef FixedBkptTerms fixed_bkpt_terms
    fixed_bkpt_terms.v = v
    fixed_bkpt_terms.m1 = m1
    fixed_bkpt_terms.m2 = m2
    fixed_bkpt_terms.rss = rss

    return fixed_bkpt_terms

cdef double rss_for_region(OLSData ols_data, double u, double v, double m):
    cdef size_t num_data = ols_data.num
    cdef double sum_x = ols_data.sum_x
    cdef double sum_y = ols_data.sum_y
    cdef double sum_xx = ols_data.sum_xx
    cdef double sum_yy = ols_data.sum_yy
    cdef double sum_xy = ols_data.sum_xy

    cdef double two_m = 2.0 * m
    cdef double mm = m * m
    cdef double mmu = mm * u
    return ((v * v - two_m * u * v + mmu * u) * num_data
            + 2.0 * (m * v - mmu) * sum_x
            + 2.0 * (m * u - v) * sum_y
            + mm * sum_xx
            + sum_yy
            - two_m * sum_xy)

# python wrapper


def fixed_bkpt_least_squares(double[:] ols_data1,
                             double[:] ols_data2,
                             double u):
    cdef OLSData ols_data_to_use1
    ols_data_to_use1.num = int(ols_data1[0])
    ols_data_to_use1.sum_x = ols_data1[1]
    ols_data_to_use1.sum_y = ols_data1[2]
    ols_data_to_use1.sum_xx = ols_data1[3]
    ols_data_to_use1.sum_yy = ols_data1[4]
    ols_data_to_use1.sum_xy = ols_data1[5]

    cdef OLSData ols_data_to_use2
    ols_data_to_use2.num = int(ols_data2[0])
    ols_data_to_use2.sum_x = ols_data2[1]
    ols_data_to_use2.sum_y = ols_data2[2]
    ols_data_to_use2.sum_xx = ols_data2[3]
    ols_data_to_use2.sum_yy = ols_data2[4]
    ols_data_to_use2.sum_xy = ols_data2[5]

    cdef FixedBkptTerms fixed_bkpt_terms = fixed_breakpt_ls(ols_data_to_use1,
                                                            ols_data_to_use2,
                                                            u)
    return (fixed_bkpt_terms.v,
            fixed_bkpt_terms.m1,
            fixed_bkpt_terms.m2,
            fixed_bkpt_terms.rss)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# TODO: expand later to any fixed params
# TODO: add data checks; eg: num_end_to_skip too big for data
# NOTE: assumes data is already sorted by indep
# num_end_to_skip is with reference to bkpt; by default, bkpt cannot be
# at last datapoint; so num_end_to_skip = 0 automatically skips first and
# last datapoint
# moreover, for num_end_to_skip = 0, we don't let the solution lie in last
# interval between last two datapoints
def estimate_one_bkpt_segreg(double[:] indep, double[:] dep, **kwargs):
    # TODO: finish with fixed rhs slope
    m2 = kwargs.pop('m2', None)
    cdef size_t num_end_to_skip = kwargs.pop('num_end_to_skip', 2)

    # TODO: this does not seem to help; why?
    cdef bint check_near_middle = kwargs.pop('check_near_middle', True)

    if m2 is None:
        m2 = NAN

    # we can have multiple y values for same x value in the data, so we
    # need to determine the unique set of x values
    # list call gets set out of order, so we call sort here on indices
    # TODO: write these in C?
    unique_indep = np.unique(indep)
    unique_indep_lhs_indices = np.searchsorted(indep, unique_indep)
    unique_indep_lhs_indices.sort()

    # c++ std::set
    cdef set[int] indices_can_skip

    ########################################################################
    cdef size_t num_unique_indep = unique_indep_lhs_indices.shape[0]
    cdef size_t half_num = int(num_unique_indep / 2)

    cdef size_t half_index
    cdef double x0
    cdef double y0
    cdef double lhs_slope
    cdef double rhs_slope
    cdef size_t index_to_use
    cdef FixedBkptTerms fixed_bkpt_terms

    cdef OneBkptParams one_bkpt_params

    cdef OLSData ols_1
    cdef OLSData ols_2

    cdef OLSData ols_all

    cdef bint in_right_place

    # for testing
    #####check_near_middle = False

    cdef min_value = float("inf")

    if check_near_middle:
        # let's check randomly near the middle -- hopefully near the solution
        # so that later checks may allow more skipping

        # TODO: how many to check here?
        half_index = unique_indep_lhs_indices[half_num]

        indices_can_skip.insert(half_index)

        x0 = indep[half_index]

        index_to_use = np.searchsorted(indep, x0)
        # don't need to add 1 here as terms all zero when x = x0
        ols_1 = ols_data(indep[0:index_to_use], dep[0:index_to_use])
        ols_2 = ols_data(indep[index_to_use:], dep[index_to_use:])

        #y0, lhs_slope, rhs_slope, rss = fixed_breakpt_ls(ols_1, ols_2, x0)
        fixed_bkpt_terms = fixed_breakpt_ls(ols_1, ols_2, x0, m2)
        y0 = fixed_bkpt_terms.v
        lhs_slope = fixed_bkpt_terms.m1
        rhs_slope = fixed_bkpt_terms.m2
        rss = fixed_bkpt_terms.rss

        one_bkpt_params.x0 = x0
        one_bkpt_params.y0 = y0
        one_bkpt_params.lhs_slope = lhs_slope
        one_bkpt_params.rhs_slope = rhs_slope

        min_value = rss

    ########################################################################

    # special interval: x in (indep[0], indep[1])
    # this must be the same as the sum_sq val at indep[1], so no need to compute
    # similarly for x in (indep[-2], indep[-1])

    # NOTE: the actually number from end will be num_end_to_skip + 2
    # this number is modulo multiple dep values at same indep
    # ie: it is number of distinct indep from end
    ##
    # eg:  num_end_to_skip = 0, then in first loop iteration, lhs has 2
    # distinct indep,
    # at last loop iter, rhs has 2 distinct indep
    ##
    # eg:  num_end_to_skip = 2, then in first loop iteration, lhs has 4
    # distinct indep,
    # at last loop iter, rhs has 4 distinct indep

    cdef size_t first_index = 1 + num_end_to_skip

    # double check all the when repeated x vals
    cdef size_t prev_index = unique_indep_lhs_indices[first_index]

    first_y = dep[0:prev_index]
    first_x = indep[0:prev_index]

    last_y = dep[prev_index:]
    last_x = indep[prev_index:]

    ols_all = ols_data(indep, dep)

    ols_1 = ols_data(first_x, first_y)
    ols_2 = ols_data(last_x, last_y)

    # STEP 1. check for local min in intervals between data points
    # for index in xrange(2, self._num_obs - 1):
    cdef int count = -1
    cdef size_t index
    cdef OLSData ols_next
    cdef OlsEstTerms ols_est_terms

    cdef double rhs_ols_intercept
    cdef double rhs_ols_slope
    cdef double rhs_rss

    cdef double lhs_ols_intercept
    cdef double lhs_ols_slope
    cdef double lhs_rss

    cdef double tot_sum_sq_resid

    cdef size_t range_begin = first_index + 1

    cdef size_t range_end = unique_indep_lhs_indices.shape[0] - num_end_to_skip - 1

    last_index = unique_indep_lhs_indices[range_end - 1]
    last_plus_one_index = unique_indep_lhs_indices[range_end]

    for index in unique_indep_lhs_indices[range_begin:range_end]:

        next_y = dep[prev_index:index]
        next_x = indep[prev_index:index]

        prev_index = index

        ols_next = ols_data(next_x, next_y)

        ols_1 = add(ols_1, ols_next)

        ########################################################################
        # CHANGE
        ########################################################################
        # aug-2020: this was orig; it appears to suffer at 1e-11 small
        # rounding errors from successive math operations
        #ols_2 = subtract(ols_2, ols_next)

        # aug-2020: NEW this is equivalent; hopefully less rounding errors
        ols_2 = subtract(ols_all, ols_1)
        ########################################################################
        # END: CHANGE
        ########################################################################

        index_minus_one = index - 1

        # we try to reduce the number of ols calls; the more data, the
        # higher we expect the sum resid squares to be, which is why
        # we bifurcate here on the data
        count += 1
        if count < half_num:
            ols_est_terms = ols_from_formula_with_rss_cimpl(ols_2, m2)
            rhs_ols_intercept = ols_est_terms.intercept
            rhs_ols_slope = ols_est_terms.slope
            rhs_rss = ols_est_terms.rss

            # imposing continuity constraint of linear spline only increases
            # sum of squared residuals (since ols residuals are a minimum)
            # so if we are already bigger than some previous potential min
            ## value, stop
            if rhs_rss > min_value:
                indices_can_skip.insert(index_minus_one)
                indices_can_skip.insert(index)
                continue

            ols_est_terms = ols_from_formula_with_rss_cimpl(ols_1)
            lhs_ols_intercept = ols_est_terms.intercept
            lhs_ols_slope = ols_est_terms.slope
            lhs_rss = ols_est_terms.rss

            tot_sum_sq_resid = lhs_rss + rhs_rss
            if tot_sum_sq_resid > min_value:
                indices_can_skip.insert(index_minus_one)
                indices_can_skip.insert(index)
                continue

        else:
            ols_est_terms = ols_from_formula_with_rss_cimpl(ols_1)
            lhs_ols_intercept = ols_est_terms.intercept
            lhs_ols_slope = ols_est_terms.slope
            lhs_rss = ols_est_terms.rss

            # imposing continuity constraint of linear spline only increases
            # sum of squared residuals (since ols residuals are a minimum)
            # so if we are already bigger than some previous potential min
            ## value, stop
            if lhs_rss > min_value:
                indices_can_skip.insert(index_minus_one)
                indices_can_skip.insert(index)
                continue

            ols_est_terms = ols_from_formula_with_rss_cimpl(ols_2, m2)
            rhs_ols_intercept = ols_est_terms.intercept
            rhs_ols_slope = ols_est_terms.slope
            rhs_rss = ols_est_terms.rss

            tot_sum_sq_resid = lhs_rss + rhs_rss
            if tot_sum_sq_resid > min_value:
                indices_can_skip.insert(index_minus_one)
                indices_can_skip.insert(index)
                continue

        in_right_place = False

        if abs(rhs_ols_slope - lhs_ols_slope) > 1.0e-14:

            x0 = (lhs_ols_intercept - rhs_ols_intercept) / (rhs_ols_slope - lhs_ols_slope)

            in_right_place = (indep[index_minus_one] < x0 and x0 < indep[index])

        if in_right_place:

            # in this case, we have a local min on the interval, so we may
            # exclude checking the endpts
            indices_can_skip.insert(index_minus_one)
            indices_can_skip.insert(index)

            if tot_sum_sq_resid < min_value:
                min_value = tot_sum_sq_resid

                ## TODO: y0 = lhs_ols_intercept + lhs_ols_slope * x0
                y0 = (lhs_ols_intercept * rhs_ols_slope - rhs_ols_intercept * lhs_ols_slope) / (rhs_ols_slope - lhs_ols_slope)

                one_bkpt_params.x0 = x0
                one_bkpt_params.y0 = y0
                one_bkpt_params.lhs_slope = lhs_ols_slope
                one_bkpt_params.rhs_slope = rhs_ols_slope

        else:
            # else we need to check the endpts
            # if index_minus_one not in indices_can_skip:
            if indices_can_skip.find(index_minus_one) == indices_can_skip.end():
                x0 = indep[index_minus_one]

                fixed_bkpt_terms = fixed_breakpt_ls(ols_1, ols_2, x0, m2)
                y0 = fixed_bkpt_terms.v
                lhs_slope = fixed_bkpt_terms.m1
                rhs_slope = fixed_bkpt_terms.m2
                rss = fixed_bkpt_terms.rss

                if(rss < min_value):
                    one_bkpt_params.x0 = x0
                    one_bkpt_params.y0 = y0
                    one_bkpt_params.lhs_slope = lhs_slope
                    one_bkpt_params.rhs_slope = rhs_slope

                    min_value = rss

            if index == last_index and indices_can_skip.find(index) == indices_can_skip.end():
                # don't need to add over next_x ols; same answer at a datapt
                x0 = indep[index]
                fixed_bkpt_terms = fixed_breakpt_ls(ols_1, ols_2, x0, m2)
                y0 = fixed_bkpt_terms.v
                lhs_slope = fixed_bkpt_terms.m1
                rhs_slope = fixed_bkpt_terms.m2
                rss = fixed_bkpt_terms.rss

                if(rss < min_value):
                    one_bkpt_params.x0 = x0
                    one_bkpt_params.y0 = y0
                    one_bkpt_params.lhs_slope = lhs_slope
                    one_bkpt_params.rhs_slope = rhs_slope

                    min_value = rss

    # old code did this
    # TODO: keep?
    if isnan(m2):
        min_params = [one_bkpt_params.x0, one_bkpt_params.y0, one_bkpt_params.lhs_slope, one_bkpt_params.rhs_slope]
    else:
        min_params = [one_bkpt_params.x0, one_bkpt_params.y0, one_bkpt_params.lhs_slope]

    return min_params, min_value
