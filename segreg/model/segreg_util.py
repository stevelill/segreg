"""
Useful routines for segmented regression problems.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np


def one_bkpt_segmented_func(u, v, m1, m2):
    """
    Returns the one-bkpt function corresponding to the given parameters.

    ``(u,v)`` is the breakpoint (in x-y plane)

    ``m1`` is the slope of the left-hand segment

    ``m2`` is the slope of the right-hand segment


    Parameters
    ----------
    u: float
    v: float
    m1: float
    m2: float

    Returns
    -------
    func: function object
    """
    def func(x):

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


def two_bkpt_segmented_func(u1, v1, u2, v2, m1, m2):
    """
    Returns the two-bkpt function corresponding to the given parameters.

    ``(u1,v1), (u2, v2)`` are the breakpoints (in x-y plane), ordered such
    that ``u1 < u2``

    ``m1`` is the slope of the left-most segment

    ``m2`` is the slope of the right-most segment

    Parameters
    ----------
    u1: float
    v1: float
    u2: float
    v2: float
    m1: float
    m2: float

    Returns
    -------
    func: function object
    """
    params = [u1, v1, u2, v2, m1, m2]

    def func(x):
        return _two_bkpt_segmented_func(x, params)

    return func


def _two_bkpt_segmented_func(x, params):
    if np.isscalar(x):
        result = _two_bkpt_segmented_func_impl([x], params)
        return result[0]
    else:
        return _two_bkpt_segmented_func_impl(x, params)


def _two_bkpt_segmented_func_impl(x, params):
    """
    Parameters
    ----------
    x: array-like (non-scalar)
    """
    # TODO: remember this function gives odd results with integer input
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

    sorted_result = np.append(first_vals, second_vals)
    sorted_result = np.append(sorted_result, third_vals)

    result = sorted_result[argsort_inds]

    return result
