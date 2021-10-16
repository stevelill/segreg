"""
Utility functions associated with the likelihood function.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np


def loglikelihood(rss, resid_variance, num_data):
    """
    Parameters
    ----------
    rss: float
    resid_variance: float
    num_data: int
    """
    term1 = num_data * np.log(2.0 * np.pi * resid_variance)
    term2 = rss / resid_variance
    return -0.5 * (term1 + term2)


def rss_line_segment(params, ols_data):
    """
    For the line model defined by:
        y = v + m(x-u)

    For data {x_i, y_i}, computes the RSS (residual sum of squares), defined
    as

        Sum[ y_i -v - m(x_i - u) ]

    For OLS, this is the complete RSS.  For bkpt models, we can add these over
    each linear segment.
    """

    u, v, m = params

    num_data = ols_data[0]
    sum_x = ols_data[1]
    sum_y = ols_data[2]
    sum_xx = ols_data[3]
    sum_yy = ols_data[4]
    sum_xy = ols_data[5]

    term = (v - m * u)

    return (sum_yy - 2.0 * term * sum_y - 2.0 * m * sum_xy
            + m * m * sum_xx + 2.0 * m * term * sum_x + term * term * num_data)

# TODO: maybe check the precision of diffs between these two ways ???
#     two_m = 2.0 * m
#     mm = m * m
#     mmu = mm * u
#     return ((v * v - two_m * u * v + mmu * u) * num_data
#             + 2.0 * (m * v - mmu) * sum_x
#             + 2.0 * (m * u - v) * sum_y
#             + mm * sum_xx
#             + sum_yy
#             - two_m * sum_xy)


def contains(arr, val):
    arr_to_use = np.array(arr)
    smallest = min(abs(arr_to_use - val))
    return (smallest < 1.0e-14)


def remove_nearest(arr, value):
    arr_to_use = np.array(arr)
    idx = (np.abs(arr_to_use - value)).argmin()
    result = np.delete(arr_to_use, idx)
    return result


def perturb_nearest(arr, value, epsilon):
    """
    Finds entry in arr nearest to value, and changes this entry by adding
    epsilon.

    NOTES
    -----
    Non-ambiguous results only when arr has unique entries.
    """
    arr_to_use = np.array(arr)
    idx = (np.abs(arr_to_use - value)).argmin()
    arr_val = arr_to_use[idx]
    arr_to_use[idx] = arr_val + epsilon
    return arr_to_use
