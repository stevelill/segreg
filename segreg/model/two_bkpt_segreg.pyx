"""
Two-Bkpt Segmented Regression core routines.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np

cimport cython
cimport numpy as np

from segreg.model.regression cimport OLSData
from segreg.model.regression cimport OlsEstTerms
from segreg.model.regression cimport ols_data
from segreg.model.regression cimport ols_from_formula_with_rss_cimpl

from segreg.model.one_bkpt_segreg cimport FixedBkptTerms
from segreg.model.one_bkpt_segreg cimport fixed_breakpt_ls


##########################################################################
# PYTHON
##########################################################################


def fixed_bkpt_ls_for_data(double[:] indep, double[:] dep, u1, u2):
    index1 = np.searchsorted(indep, u1, side='right')
    index2 = np.searchsorted(indep, u2, side='right')
    indep1 = indep[0:index1]
    dep1 = dep[0:index1]
    indep2 = indep[index1:index2]
    dep2 = dep[index1:index2]
    indep3 = indep[index2:]
    dep3 = dep[index2:]

    ols_data_1 = ols_data(indep1, dep1)
    ols_data_2 = ols_data(indep2, dep2)
    ols_data_3 = ols_data(indep3, dep3)

    cdef TwoFixedBkptTerms fixed_bkpt_terms = two_fixed_bkpt_ls(ols_data_1,
                                                                ols_data_2,
                                                                ols_data_3,
                                                                u1,
                                                                u2)
    return (fixed_bkpt_terms.v1,
            fixed_bkpt_terms.v2,
            fixed_bkpt_terms.m1,
            fixed_bkpt_terms.m2,
            fixed_bkpt_terms.rss)


def fixed_bkpt_least_squares(double[:] ols_data1,
                             double[:] ols_data2,
                             double[:] ols_data3,
                             double u1,
                             double u2):

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

    cdef OLSData ols_data_to_use3
    ols_data_to_use3.num = int(ols_data3[0])
    ols_data_to_use3.sum_x = ols_data3[1]
    ols_data_to_use3.sum_y = ols_data3[2]
    ols_data_to_use3.sum_xx = ols_data3[3]
    ols_data_to_use3.sum_yy = ols_data3[4]
    ols_data_to_use3.sum_xy = ols_data3[5]

    cdef TwoFixedBkptTerms fixed_bkpt_terms = two_fixed_bkpt_ls(ols_data_to_use1,
                                                                ols_data_to_use2,
                                                                ols_data_to_use3,
                                                                u1,
                                                                u2)
    return (fixed_bkpt_terms.v1,
            fixed_bkpt_terms.v2,
            fixed_bkpt_terms.m1,
            fixed_bkpt_terms.m2,
            fixed_bkpt_terms.rss)

##########################################################################
# CYTHON
##########################################################################


cdef struct TwoFixedBkptTerms:
    double v1
    double v2
    double m1
    double m2
    double rss

cdef struct TwoDimVect:
    double x1
    double x2

cdef struct TwoSegregParams:
    double u1
    double v1
    double u2
    double v2
    double m1
    double m2

cdef struct EstimationResult:
    TwoSegregParams minParams
    double minValue
    bint canSkipCorners

cdef TwoDimVect invert_two_by_two(double a,
                                  double b,
                                  double c,
                                  double d,
                                  double e,
                                  double f):
    """
    Solves the two-dimensional linear system for x:
        A x = b
    where
        A = | a b |  x = |x_1|  b = |e|
            | c d |      |x_2|      |f|

    NOTE
    ----
    This is much faster than calling scipy.linalg.solve

    RETURNS
    -------
    tuple: x_1, x_2
    """
    cdef double denom = a * d - b * c
    cdef double first = (d * e - b * f) / denom
    cdef double second = (a * f - c * e) / denom

    cdef TwoDimVect result
    result.x1 = first
    result.x2 = second

    return result

# TODO: add check if u1 = u2
cdef TwoFixedBkptTerms two_fixed_bkpt_ls(OLSData ols_terms_1,
                                         OLSData ols_terms_2,
                                         OLSData ols_terms_3,
                                         double u1,
                                         double u2):
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

    cdef size_t num_data_3 = ols_terms_3.num
    cdef double sum_x_3 = ols_terms_3.sum_x
    cdef double sum_y_3 = ols_terms_3.sum_y
    cdef double sum_xx_3 = ols_terms_3.sum_xx
    cdef double sum_yy_3 = ols_terms_3.sum_yy
    cdef double sum_xy_3 = ols_terms_3.sum_xy

    cdef double u1_sq = u1 * u1
    cdef double u2_sq = u2 * u2
    cdef double two_u1 = 2.0 * u1
    cdef double two_u2 = 2.0 * u2
    cdef double diff = u2 - u1
    cdef double diff_sq = diff * diff

    cdef double A1 = sum_y_1
    cdef double A2 = sum_y_2
    cdef double A3 = sum_y_3

    cdef double B11 = sum_xy_1 - u1 * A1
    cdef double B22 = sum_xy_2 - u2 * A2
    cdef double B21 = sum_xy_2 - u1 * A2
    cdef double B32 = sum_xy_3 - u2 * A3

    cdef double C11 = sum_x_1 - u1 * num_data_1
    cdef double C21 = sum_x_2 - u1 * num_data_2
    cdef double C32 = sum_x_3 - u2 * num_data_3

    cdef double D11 = sum_xx_1 - two_u1 * sum_x_1 + u1_sq * num_data_1
    cdef double D22 = sum_xx_2 - two_u2 * sum_x_2 + u2_sq * num_data_2
    cdef double D21 = sum_xx_2 - two_u1 * sum_x_2 + u1_sq * num_data_2
    cdef double D32 = sum_xx_3 - two_u2 * sum_x_3 + u2_sq * num_data_3

    cdef double E = sum_yy_1 + sum_yy_2 + sum_yy_3

    cdef double F2 = sum_xx_2 - (u1 + u2) * sum_x_2 + u1 * u2 * num_data_2

    ##
    cdef double term = D21 / diff_sq
    cdef double a = -1.0 * num_data_1 + C11 * C11 / D11 - D22 / diff_sq
    cdef double b = F2 / diff_sq
    cdef double c = b
    cdef double d = -1.0 * num_data_3 + C32 * C32 / D32 - term
    cdef double e = -A1 + B11 * C11 / D11 + B22 / diff
    cdef double f = -A3 + B32 * C32 / D32 - B21 / diff

    # v estimates
    cdef TwoDimVect inversion
    inversion = invert_two_by_two(a, b, c, d, e, f)
    cdef double v1 = inversion.x1
    cdef double v2 = inversion.x2

    # BEGIN: slopes
    cdef double m1 = (B11 - v1 * C11) / D11
    cdef double m2 = (B32 - v2 * C32) / D32
    # END: slopes

    cdef double m = (v2 - v1) / (u2 - u1)
    cdef double two_v1 = 2.0 * v1
    cdef double two_v2 = 2.0 * v2

    # TODO: double-check this formula vs rss_for_region
    cdef double rss = (E - two_v1 * (A1 + A2) - two_v2 * A3
                       - 2.0 * m1 * B11 - 2.0 * m * B21 - 2.0 * m2 * B32
                       + v1 * v1 * (num_data_1 + num_data_2) +
                       v2 * v2 * num_data_3
                       + two_v1 * (m1 * C11 + m * C21) + two_v2 * m2 * C32
                       + m1 * m1 * D11 + m * m * D21 + m2 * m2 * D32)

    cdef TwoFixedBkptTerms fixed_bkpt_terms
    fixed_bkpt_terms.v1 = v1
    fixed_bkpt_terms.v2 = v2
    fixed_bkpt_terms.m1 = m1
    fixed_bkpt_terms.m2 = m2
    fixed_bkpt_terms.rss = rss

    return fixed_bkpt_terms


cdef EstimationResult corner(OLSData ols_data1,
                             OLSData ols_data2,
                             OLSData ols_data3,
                             double u1,
                             double u2,
                             TwoSegregParams minParams,
                             double min_value,
                             bint verbose):
    twoFixedBkptTerms = two_fixed_bkpt_ls(ols_data1,
                                          ols_data2,
                                          ols_data3,
                                          u1,
                                          u2)

    v1 = twoFixedBkptTerms.v1
    v2 = twoFixedBkptTerms.v2
    m1 = twoFixedBkptTerms.m1
    m2 = twoFixedBkptTerms.m2
    rss = twoFixedBkptTerms.rss

    if rss < min_value:
        min_value = rss

        minParams.u1 = u1
        minParams.v1 = v1
        minParams.u2 = u2
        minParams.v2 = v2
        minParams.m1 = m1
        minParams.m2 = m2

        if verbose:
            print()
            print("NEW LOW WITH CORNER VALUE")
            print("params: ", minParams)
            print("RSS: ", rss)
            print("CORNER: u1: ", u1, " ; u2: ", u2)
            print()

    cdef EstimationResult result
    result.minParams = minParams
    result.minValue = min_value

    return result

cdef EstimationResult fix_u1_bndy(double u1_fixed,
                                  OLSData ols_data1,
                                  OLSData ols_data2,
                                  double ols_intercept3,
                                  double ols_slope3,
                                  double rss3,
                                  double u2_data,
                                  double u2_data_next,
                                  TwoSegregParams minParams,
                                  double min_value,
                                  bint verbose):
    fixedBkptTerms12 = fixed_breakpt_ls(ols_data1,
                                        ols_data2,
                                        u1_fixed)
    v1 = fixedBkptTerms12.v
    m1 = fixedBkptTerms12.m1
    m2 = fixedBkptTerms12.m2
    rss_12 = fixedBkptTerms12.rss

    rss = rss_12 + rss3

    cdef double u2_intersect
    cdef bint u2_right_place = False

    cdef bint can_skip_corners = True

    cdef double slope_diff = ols_slope3 - m2

    # TODO: what if equals here?  (ie: two solutions?)
    if rss < min_value:

        can_skip_corners = False

        # note: if slopes essentially equal, then solution here is 1bkpt --
        # the params and rss will be covered when we check corners

        if abs(slope_diff) > 1.0e-14:

            u2_intersect = (v1 - ols_intercept3 - m2 * u1_fixed) / slope_diff
            u2_right_place = (u2_data < u2_intersect) and (u2_intersect < u2_data_next)

            can_skip_corners = u2_right_place

            if u2_right_place:
                min_value = rss
                v2 = ols_intercept3 + ols_slope3 * u2_intersect

                minParams.u1 = u1_fixed
                minParams.v1 = v1
                minParams.u2 = u2_intersect
                minParams.v2 = v2
                minParams.m1 = m1
                minParams.m2 = ols_slope3

                if verbose:
                    print()
                    print("NEW LOW FIXED u1=", u1_fixed)
                    print("u2 interval")
                    print(u2_data, u2_data_next)
                    print("u2 intersect: ", u2_intersect)
                    print()
                    print("RSS: ", rss)

    cdef EstimationResult result
    result.minParams = minParams
    result.minValue = min_value
    result.canSkipCorners = can_skip_corners

    return result


cdef EstimationResult fix_u2_bndy(double u2_fixed,
                                  OLSData ols_data2,
                                  OLSData ols_data3,
                                  double ols_intercept1,
                                  double ols_slope1,
                                  double rss1,
                                  double u1_data,
                                  double u1_data_next,
                                  TwoSegregParams minParams,
                                  double min_value,
                                  bint verbose):
    fixedBkptTerms23 = fixed_breakpt_ls(ols_data2,
                                        ols_data3,
                                        u2_fixed)
    v2 = fixedBkptTerms23.v
    m2 = fixedBkptTerms23.m1
    m3 = fixedBkptTerms23.m2
    rss_23 = fixedBkptTerms23.rss

    rss = rss1 + rss_23

    cdef double u1_intersect
    cdef bint u1_right_place = False

    cdef bint can_skip_corners = True

    cdef double slope_diff = m2 - ols_slope1

    if rss < min_value:

        can_skip_corners = False

        # note: if slopes essentially equal, then solution here is 1bkpt --
        # the params and rss will be covered when we check corners

        if abs(slope_diff) > 1.0e-14:

            u1_intersect = (ols_intercept1 - v2 + m2 * u2_fixed) / slope_diff
            u1_right_place = (u1_data < u1_intersect) and (u1_intersect < u1_data_next)

            can_skip_corners = u1_right_place

            if u1_right_place:
                min_value = rss
                v1 = ols_intercept1 + ols_slope1 * u1_intersect

                minParams.u1 = u1_intersect
                minParams.v1 = v1
                minParams.u2 = u2_fixed
                minParams.v2 = v2
                minParams.m1 = ols_slope1
                minParams.m2 = m3

                if verbose:
                    print()
                    print("NEW LOW FIXED u2=", u2_fixed)
                    print("u1 interval")
                    print(u1_data, u1_data)
                    print("u1 intersect: ", u1_intersect)
                    print()
                    print("RSS: ", rss)

    cdef EstimationResult result
    result.minParams = minParams
    result.minValue = min_value
    result.canSkipCorners = can_skip_corners

    return result


# TODO: finish converting to cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# TODO: expand later to any fixed params
# TODO: add data checks; eg: num_end_to_skip too big for data
# NOTE: assumes data is already sorted by indep
# TODO: don't allow num_between less than 2
def estimate_two_bkpt_segreg(double[:] indep, double[:] dep, **kwargs):
    cdef size_t num_end_to_skip = kwargs.pop('num_end_to_skip', 3)
    cdef size_t num_between_to_skip = kwargs.pop('num_between_to_skip', 4)

    cdef bint verbose = kwargs.pop('verbose', False)

    cdef size_t index1_begin
    cdef size_t index1_end
    cdef size_t index2_begin
    cdef size_t index2_end
    cdef size_t index1
    cdef size_t index2
    cdef size_t ind1
    cdef size_t ind2

    cdef OLSData ols_data1
    cdef OlsEstTerms ols_result1
    cdef OLSData ols_data2
    cdef OlsEstTerms ols_result2
    cdef OLSData ols_data3
    cdef OlsEstTerms ols_result3

    cdef FixedBkptTerms fixedBkptTerms12
    cdef FixedBkptTerms fixedBkptTerms23
    cdef TwoFixedBkptTerms twoFixedBkptTerms

    cdef TwoSegregParams minParams


#     cdef double ols_intercept1
#     cdef double ols_intercept2
#     cdef double ols_intercept3
#     cdef double ols_slope1
#     cdef double ols_slope2
#     cdef double ols_slope3

    cdef double u1_data
    cdef double u1_data_next
    cdef double u2_data
    cdef double u2_data_next

    # helpers
    cdef double u1_data_to_use
    cdef double u2_data_to_use

    cdef double u1_intersect
    cdef double u2_intersect

    cdef bint u1_right_place
    cdef bint u2_right_place

    cdef double lhs_slope_diff
    cdef double rhs_slope_diff

    cdef bint non_zero_slopes

    cdef bint can_skip_lower_left_corner

    # we can have multiple y values for same x value in the data, so we
    # need to determine the unique set of x values
    # list call gets set out of order, so we call sort here on indices
    # TODO: write these in C?
# cdef double[:] unique_indep_lhs_indices

    unique_indep = np.unique(indep)
    unique_indep_lhs_indices = np.searchsorted(indep, unique_indep)
    unique_indep_lhs_indices.sort()

    cdef double min_value = float("inf")

    # STEP 2. check for local min in intervals between data points
    cdef size_t num_inds = unique_indep_lhs_indices.shape[0]

    # the left-most and right-most ends are consistent with 1BKPT
    # these are indices of LHS of allowable intervals to check
    num_uniq = len(unique_indep)
    index1_begin = num_end_to_skip + 1
    index2_end = num_uniq - num_end_to_skip - 3
    index1_end = index2_end - num_between_to_skip

    cdef double[:] indep1
    cdef double[:] indep2
    cdef double[:] indep3

    cdef size_t num_index2_range
    cdef size_t first_index1
    cdef size_t first_index2
    cdef size_t last_index2

    if verbose:
        print()
        print("indep")
        print(list(indep))
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

    first_index1 = index1_range[0]

    for index1 in index1_range:

        index2_begin = index1 + num_between_to_skip
        index2_range = np.arange(index2_begin, index2_end + 1)

        num_index2_range = len(index2_range)
        first_index2 = index2_range[0]
        last_index2 = index2_range[num_index2_range - 1]

        for index2 in index2_range:

            # reset for each square
            can_skip_lower_left_corner = False

            if verbose:
                print()
                print("--------------------------------------------------------")
                print(index1, ",", index2)

            ind1 = unique_indep_lhs_indices[index1 + 1]
            ind2 = unique_indep_lhs_indices[index2 + 1]

            indep1 = indep[0:ind1]
            dep1 = dep[0:ind1]
            indep2 = indep[ind1:ind2]
            dep2 = dep[ind1:ind2]
            indep3 = indep[ind2:]
            dep3 = dep[ind2:]

            # TODO: put in check that mins actually hit
            if indep1.shape[0] < num_end_to_skip:
                raise Exception("region one has too few data")
            if indep2.shape[0] < num_between_to_skip:
                raise Exception("region two has too few data")
            if indep3.shape[0] < num_end_to_skip:
                raise Exception("region three has too few data")

            u1_data = indep[ind1 - 1]
            u1_data_next = indep2[0]

            u2_data = indep[ind2 - 1]
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
                print(list(indep1))
                print("indep2: len: ", len(indep2))
                print(list(indep2))
                print("indep3: len: ", len(indep3))
                print(list(indep3))
                print()

            ###################################################################
            # interior of square
            # check regressions for right place
            ###################################################################

            ols_data1 = ols_data(indep1, dep1)
            ols_result1 = ols_from_formula_with_rss_cimpl(ols_data1)
            ols_intercept1 = ols_result1.intercept
            ols_slope1 = ols_result1.slope
            rss1 = ols_result1.rss

            # TRICK: get out early if possible
            if rss1 > min_value:
                continue

            ols_data2 = ols_data(indep2, dep2)
            ols_result2 = ols_from_formula_with_rss_cimpl(ols_data2)
            ols_intercept2 = ols_result2.intercept
            ols_slope2 = ols_result2.slope
            rss2 = ols_result2.rss

            # TRICK: get out early if possible
            if rss1 + rss2 > min_value:
                continue

            ols_data3 = ols_data(indep3, dep3)
            ols_result3 = ols_from_formula_with_rss_cimpl(ols_data3)
            ols_intercept3 = ols_result3.intercept
            ols_slope3 = ols_result3.slope
            rss3 = ols_result3.rss

            rss = rss1 + rss2 + rss3

            if rss > min_value:
                continue

            lhs_slope_diff = ols_slope2 - ols_slope1
            rhs_slope_diff = ols_slope3 - ols_slope2

            non_zero_slopes = abs(lhs_slope_diff) > 1.0e-14 and abs(rhs_slope_diff) > 1.0e-14

            if non_zero_slopes:
                u1_intersect = (ols_intercept1 - ols_intercept2) / lhs_slope_diff
                u2_intersect = (ols_intercept2 - ols_intercept3) / rhs_slope_diff

                u1_right_place = (u1_data < u1_intersect) and (u1_intersect < u1_data_next)
                u2_right_place = (u2_data < u2_intersect) and (u2_intersect < u2_data_next)

                if u1_right_place and u2_right_place:
                    if rss < min_value:
                        min_value = rss
                        v1 = ols_intercept1 + ols_slope1 * u1_intersect
                        v2 = ols_intercept2 + ols_slope2 * u2_intersect

                        minParams.u1 = u1_intersect
                        minParams.v1 = v1
                        minParams.u2 = u2_intersect
                        minParams.v2 = v2
                        minParams.m1 = ols_slope1
                        minParams.m2 = ols_slope3

                        if verbose:
                            print()
                            print("NEW LOW BOTH IN RIGHT PLACE")
                            print("params: ", minParams)
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

            result = fix_u1_bndy(u1_fixed=u1_data,
                                 ols_data1=ols_data1,
                                 ols_data2=ols_data2,
                                 ols_intercept3=ols_intercept3,
                                 ols_slope3=ols_slope3,
                                 rss3=rss3,
                                 u2_data=u2_data,
                                 u2_data_next=u2_data_next,
                                 minParams=minParams,
                                 min_value=min_value,
                                 verbose=verbose)

            minParams = result.minParams
            min_value = result.minValue

            if result.canSkipCorners:
                can_skip_lower_left_corner = True

            # extra side-of-square bndy we need to check
            if index2 == first_index2:
                # here we need to additionally do with bkpt u1_data_next

                if verbose:
                    print("EXTRA SIDE BNDY FIX u1_data_next: ", u1_data_next,
                          " ; u1_data: ", u1_data,
                          " ; u2_data: ", u2_data)

                result = fix_u1_bndy(u1_fixed=u1_data_next,
                                     ols_data1=ols_data1,
                                     ols_data2=ols_data2,
                                     ols_intercept3=ols_intercept3,
                                     ols_slope3=ols_slope3,
                                     rss3=rss3,
                                     u2_data=u2_data,
                                     u2_data_next=u2_data_next,
                                     minParams=minParams,
                                     min_value=min_value,
                                     verbose=verbose)

                minParams = result.minParams
                min_value = result.minValue

                if not result.canSkipCorners:
                    # need to check lower-right corner
                    result = corner(ols_data1=ols_data1,
                                    ols_data2=ols_data2,
                                    ols_data3=ols_data3,
                                    u1=u1_data_next,
                                    u2=u2_data,
                                    minParams=minParams,
                                    min_value=min_value,
                                    verbose=verbose)

                    minParams = result.minParams
                    min_value = result.minValue

            ##########
            # fix u2
            ##########
            # TODO: better names between one, two bkpt versions

            result = fix_u2_bndy(u2_fixed=u2_data,
                                 ols_data2=ols_data2,
                                 ols_data3=ols_data3,
                                 ols_intercept1=ols_intercept1,
                                 ols_slope1=ols_slope1,
                                 rss1=rss1,
                                 u1_data=u1_data,
                                 u1_data_next=u1_data_next,
                                 minParams=minParams,
                                 min_value=min_value,
                                 verbose=verbose)

            minParams = result.minParams
            min_value = result.minValue

            if result.canSkipCorners:
                can_skip_lower_left_corner = True

            # extra side-of-square bndy we need to check
            if index2 == last_index2:
                # here we need to additionally do with bkpt u1_data_next
                if verbose:
                    print("EXTRA SIDE BNDY FIX u2_data_next: ", u2_data_next,
                          " ; u1_data: ", u1_data,
                          " ; u2_data: ", u2_data)
                    print()

                result = fix_u2_bndy(u2_fixed=u2_data_next,
                                     ols_data2=ols_data2,
                                     ols_data3=ols_data3,
                                     ols_intercept1=ols_intercept1,
                                     ols_slope1=ols_slope1,
                                     rss1=rss1,
                                     u1_data=u1_data,
                                     u1_data_next=u1_data_next,
                                     minParams=minParams,
                                     min_value=min_value,
                                     verbose=verbose)

                minParams = result.minParams
                min_value = result.minValue

                if not result.canSkipCorners:
                    # need to check upper-right corner
                    result = corner(ols_data1=ols_data1,
                                    ols_data2=ols_data2,
                                    ols_data3=ols_data3,
                                    u1=u1_data_next,
                                    u2=u2_data_next,
                                    minParams=minParams,
                                    min_value=min_value,
                                    verbose=verbose)

                    minParams = result.minParams
                    min_value = result.minValue

                    # only miss one corner at most upper-left, so need to
                    # check it separately
                    if index1 == first_index1:
                        # need to check upper-left corner of upper-left square
                        result = corner(ols_data1=ols_data1,
                                        ols_data2=ols_data2,
                                        ols_data3=ols_data3,
                                        u1=u1_data,
                                        u2=u2_data_next,
                                        minParams=minParams,
                                        min_value=min_value,
                                        verbose=verbose)

                        minParams = result.minParams
                        min_value = result.minValue

            ###################################################################
            # check corner boundaries
            ###################################################################

            if not can_skip_lower_left_corner:
                result = corner(ols_data1=ols_data1,
                                ols_data2=ols_data2,
                                ols_data3=ols_data3,
                                u1=u1_data,
                                u2=u2_data,
                                minParams=minParams,
                                min_value=min_value,
                                verbose=verbose)

                minParams = result.minParams
                min_value = result.minValue

    # last step: convert to python return
    min_params = [minParams.u1,
                  minParams.v1,
                  minParams.u2,
                  minParams.v2,
                  minParams.m1,
                  minParams.m2]

    return min_params, min_value
