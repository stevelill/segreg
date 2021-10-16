"""
Testing one-bkpt segreg.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt

from segreg.analysis import stats_plotting
from segreg.model import one_bkpt_segreg


OneBkptExample = namedtuple("OneBkptExample", ["indep",
                                               "dep",
                                               "num_end_to_skip",
                                               "params",
                                               "rss"])


def _plot(indep, dep, num_end_to_skip):

    (min_params,
     min_value) = one_bkpt_segreg.estimate_one_bkpt_segreg(indep,
                                                           dep,
                                                           num_end_to_skip=num_end_to_skip)

    func = one_bkpt_segreg.segmented_func(*min_params)
    stats_plotting.plot_model(func=func,
                              indep=indep,
                              dep=dep,
                              extra_pts=[min_params[0]],
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


def corner_E_interval_E(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)
    dep = [0, 0, 0, 0, 0, 0, 0, 0, 1]

    params = [7.0, 0.0, 0.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0

    if plot:
        _plot(indep, dep, num_end_to_skip)

    return OneBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          params=params,
                          rss=rss)


def corner_W_interval_W(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)
    dep = [1, 0, 0, 0, 0, 0, 0, 0, 0]

    params = [1.0, 0.0, -1.0, 0.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0

    if plot:
        _plot(indep, dep, num_end_to_skip)

    return OneBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          params=params,
                          rss=rss)


def corner_E_interval_W(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)
    dep = [2, 1, 0, 0, 0, 0, 0, 0, 0]

    params = [2.0, 0.0, -1.0, 0.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0

    if plot:
        _plot(indep, dep, num_end_to_skip)

    return OneBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          params=params,
                          rss=rss)


def corner_W_interval_E(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)
    dep = [0, 0, 0, 0, 0, 0, 0, 1, 2]

    params = [6.0, 0.0, 0.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0

    if plot:
        _plot(indep, dep, num_end_to_skip)

    return OneBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          params=params,
                          rss=rss)


def corner_interval_middle(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)
    dep = [4, 3, 2, 1, 0, 1, 2, 3, 4]

    params = [4.0, 0.0, -1.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0

    if plot:
        _plot(indep, dep, num_end_to_skip)

    return OneBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          params=params,
                          rss=rss)


def interior_interval_E(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)

    x0 = 6.5

    def line(x):
        return x - x0

    dep = [0, 0, 0, 0, 0, 0, 0, line(7), line(8)]

    params = [x0, 0.0, 0.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0

    if plot:
        _plot(indep, dep, num_end_to_skip)

    return OneBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          params=params,
                          rss=rss)

def interior_interval_E_minusone(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)

    x0 = 5.5

    def line(x):
        return x - x0

    dep = [0, 0, 0, 0, 0, 0, line(6), line(7), line(8)]

    params = [x0, 0.0, 0.0, 1.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0

    if plot:
        _plot(indep, dep, num_end_to_skip)

    return OneBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          params=params,
                          rss=rss)

def interior_interval_W(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)

    x0 = 1.5

    def line(x):
        return x0 - x

    dep = [line(0), line(1), 0, 0, 0, 0, 0, 0, 0]

    params = [x0, 0.0, -1.0, 0.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0

    if plot:
        _plot(indep, dep, num_end_to_skip)

    return OneBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          params=params,
                          rss=rss)

def interior_interval_W_plusone(multiple_y=False, plot=False):
    indep = np.arange(9)
    num_distinct_indep = len(indep)

    x0 = 2.5

    def line(x):
        return x0 - x

    dep = [line(0), line(1), line(2), 0, 0, 0, 0, 0, 0]

    params = [x0, 0.0, -1.0, 0.0]
    rss = 0.0

    if multiple_y:
        indep, dep = _create_muliple_y(indep, dep)
        rss = 2.0 * num_distinct_indep

    # ensure float arrays
    indep = np.array(indep, dtype=float)
    dep = np.array(dep, dtype=float)

    num_end_to_skip = 0

    if plot:
        _plot(indep, dep, num_end_to_skip)

    return OneBkptExample(indep=indep,
                          dep=dep,
                          num_end_to_skip=num_end_to_skip,
                          params=params,
                          rss=rss)


if __name__ == "__main__":
    interior_interval_W_plusone(plot=True, multiple_y=False)
