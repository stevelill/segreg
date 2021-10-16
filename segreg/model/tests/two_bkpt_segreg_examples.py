"""
Provides a suite of special data configurations for testing segmented
regression.

Notation: geographic: N,S,E,W
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from collections import namedtuple

from matplotlib import pyplot as plt

import numpy as np
from segreg.analysis import stats_plotting
from segreg.model import two_bkpt_segreg


TwoBkptExample = namedtuple("TwoBkptExample", ["indep",
                                               "dep",
                                               "num_end_to_skip",
                                               "num_between_to_skip",
                                               "params",
                                               "rss"])


def _plot(indep, dep, num_end_to_skip, num_between_to_skip):

    (min_params,
     min_value) = two_bkpt_segreg.estimate_two_bkpt_segreg(indep,
                                                           dep,
                                                           num_end_to_skip=num_end_to_skip,
                                                           num_between_to_skip=num_between_to_skip)

    func = two_bkpt_segreg.segmented_func(min_params)
    stats_plotting.plot_model(func=func,
                              indep=indep,
                              dep=dep,
                              extra_pts=[min_params[0],
                                         min_params[2]],
                              full_size_scatter=True)

    plt.ylim([min(dep) - 2, max(dep) + 2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def _create_muliple_y(indep_core, dep_core):

    indep = [[x, x, x] for x in indep_core]
    indep = np.array(indep, dtype=float)
    indep = indep.flatten()

    dep = [[y - 1, y, y + 1] for y in dep_core]
    dep = np.array(dep, dtype=float)
    dep = dep.flatten()

    return indep, dep


def corner_NW_square_NW(multiple_y=False, plot=False):
    """
    Solution at corner u1,u2_next, most upper-left square.

    rss = residual sum squares
    """
    indep = np.arange(9)
    num_distinct_indep = len(indep)
    dep = [1, 0, 0, 0, 0, 0, 0, 0, 1]

    params = [1.0, 0.0, 7.0, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0
    num_between_to_skip = 3

    if plot:
        _plot(indep, dep, num_end_to_skip, num_between_to_skip)

    return TwoBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          num_between_to_skip=num_between_to_skip,
                          params=params,
                          rss=rss)


def corner_SE_square_SW(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)
    dep = [2, 1, 0, 0, 0, 1, 2, 3, 4]

    params = [2.0, 0.0, 4.0, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0
    num_between_to_skip = 3

    if plot:
        _plot(indep, dep, num_end_to_skip, num_between_to_skip)

    return TwoBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          num_between_to_skip=num_between_to_skip,
                          params=params,
                          rss=rss)


def corner_NE_square_NE(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)
    dep = [4, 3, 2, 1, 0, 0, 0, 0, 1]

    params = [4.0, 0.0, 7.0, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0
    num_between_to_skip = 3

    if plot:
        _plot(indep, dep, num_end_to_skip, num_between_to_skip)

    return TwoBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          num_between_to_skip=num_between_to_skip,
                          params=params,
                          rss=rss)


def corner_NE_square_NW(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)
    dep = [2, 1, 0, 0, 0, 0, 0, 0, 1]

    params = [2.0, 0.0, 7.0, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0
    num_between_to_skip = 3

    if plot:
        _plot(indep, dep, num_end_to_skip, num_between_to_skip)

    return TwoBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          num_between_to_skip=num_between_to_skip,
                          params=params,
                          rss=rss)


def side_W_square_NW(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)

    x0 = 6.5

    def line(x):
        return x - x0

    dep = [1, 0, 0, 0, 0, 0, 0, line(7), line(8)]

    params = [1.0, 0.0, x0, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0
    num_between_to_skip = 3

    if plot:
        _plot(indep, dep, num_end_to_skip, num_between_to_skip)

    return TwoBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          num_between_to_skip=num_between_to_skip,
                          params=params,
                          rss=rss)


def side_E_square_NW(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)

    x0 = 6.5

    def line(x):
        return x - x0

    dep = [2, 1, 0, 0, 0, 0, 0, line(7), line(8)]

    params = [2.0, 0.0, x0, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0
    num_between_to_skip = 3

    if plot:
        _plot(indep, dep, num_end_to_skip, num_between_to_skip)

    return TwoBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          num_between_to_skip=num_between_to_skip,
                          params=params,
                          rss=rss)


def side_E_square_SW(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)

    x0 = 4.5

    def line(x):
        return x - x0

    dep = [2, 1, 0, 0, 0, line(5), line(6), line(7), line(8)]

    params = [2.0, 0.0, x0, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0
    num_between_to_skip = 3

    if plot:
        _plot(indep, dep, num_end_to_skip, num_between_to_skip)

    return TwoBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          num_between_to_skip=num_between_to_skip,
                          params=params,
                          rss=rss)


def side_E_square_NE(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)

    x0 = 6.5

    def line(x):
        return x - x0

    dep = [4, 3, 2, 1, 0, 0, 0, line(7), line(8)]

    params = [4.0, 0.0, x0, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0
    num_between_to_skip = 3

    if plot:
        _plot(indep, dep, num_end_to_skip, num_between_to_skip)

    return TwoBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          num_between_to_skip=num_between_to_skip,
                          params=params,
                          rss=rss)


def side_S_square_SW(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)

    x0 = 1.5

    def line(x):
        return x0 - x

    dep = [line(0), line(1), 0, 0, 0, 1, 2, 3, 4]

    params = [x0, 0.0, 4.0, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0
    num_between_to_skip = 3

    if plot:
        _plot(indep, dep, num_end_to_skip, num_between_to_skip)

    return TwoBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          num_between_to_skip=num_between_to_skip,
                          params=params,
                          rss=rss)


def side_N_square_SW(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)

    x0 = 1.5

    def line(x):
        return x0 - x

    dep = [line(0), line(1), 0, 0, 0, 0, 1, 2, 3]

    params = [x0, 0.0, 5.0, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0
    num_between_to_skip = 3

    if plot:
        _plot(indep, dep, num_end_to_skip, num_between_to_skip)

    return TwoBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          num_between_to_skip=num_between_to_skip,
                          params=params,
                          rss=rss)


def side_N_square_NW(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)

    x0 = 1.5

    def line(x):
        return x0 - x

    dep = [line(0), line(1), 0, 0, 0, 0, 0, 0, 1]

    params = [x0, 0.0, 7.0, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0
    num_between_to_skip = 3

    if plot:
        _plot(indep, dep, num_end_to_skip, num_between_to_skip)

    return TwoBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          num_between_to_skip=num_between_to_skip,
                          params=params,
                          rss=rss)


def interior_square_NW(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)

    u1 = 1.5

    def line_left(x):
        return u1 - x

    u2 = 6.5

    def line_right(x):
        return x - u2

    dep = [line_left(0), line_left(1), 0, 0, 0, 0, 0, line_right(7), line_right(8)]

    params = [u1, 0.0, u2, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0
    num_between_to_skip = 3

    if plot:
        _plot(indep, dep, num_end_to_skip, num_between_to_skip)

    return TwoBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          num_between_to_skip=num_between_to_skip,
                          params=params,
                          rss=rss)


def interior_square_NE(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)

    u1 = 3.5

    def line_left(x):
        return u1 - x

    u2 = 6.5

    def line_right(x):
        return x - u2

    dep = [line_left(0), line_left(1), line_left(2), line_left(3), 0, 0, 0, line_right(7), line_right(8)]

    params = [u1, 0.0, u2, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0
    num_between_to_skip = 3

    if plot:
        _plot(indep, dep, num_end_to_skip, num_between_to_skip)

    return TwoBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          num_between_to_skip=num_between_to_skip,
                          params=params,
                          rss=rss)


if __name__ == "__main__":
    interior_square_NE(plot=True, multiple_y=False)
