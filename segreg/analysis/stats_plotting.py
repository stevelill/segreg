"""
Routines to plot segmented regression results.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

from segreg.model import one_bkpt_rss_func
from segreg.model import two_bkpt_rss_func

_DEFAULT_SCATTER_SIZE = 3


def plot_models(func_arr,
                indep,
                dep,
                domain_begin_arr=None,
                domain_end_arr=None,
                extra_pts_arr=None,
                title=None,
                xlabel=None,
                ylabel=None,
                mark_extra_pts=True,
                scatter_size=3,
                scatter_color="gray",
                marker="o",
                legend=None,
                ax=None):
    """
    Plots univariate functions together with scatter of the data.

    This plotting utility is geared towards segmented regression problems where  
    the function inputs would be the piecewise-linear segmented functions.

    Notes
    -----
    For plotting segmented regression models, it is recommended to pass in the 
    bkpts via the parameter ``extra_pts_arr`` to ensure the plot looks correct.  
    This is because the bkpts might not coincide with data points, and the plot
    interpolation between plotted points could appear wrong near the bkpts.

    Parameters
    ----------
    func_arr: array-like
        An array of function objects.  Each function should be defined on the
        domain of the independent data.
    indep: array-like
        The independent data.  Also called predictor, explanatory variable,
        regressor, or exogenous variable.
    dep: array-like
        The dependent data.  Also called response, regressand, or endogenous
        variable.
    domain_begin_arr: array-like
        An array of float.  Each element specifies the left endpoint of the
        domain to plot for corresponding function in ``func_arr``. Must have 
        same length as the parameter ``func_arr``.
    domain_end_arr: array-like
        An array of float.  Each element specifies the right endpoint of the
        domain to plot for corresponding function in ``func_arr``. Must have 
        same length as the parameter ``func_arr``.
    extra_pts_arr: array-like
        An array of arrays.  Each element is an array containing extra points
        to plot.  The points are in the x-axis domain (ie: the domain of the
        independent data).  Must have same length as the parameter ``func_arr``.
        Each element must be either array-like or None.  
            Eg: [ [1,2], [3] ].

            Eg: [ None, [4,5] ].
    title: str
    xlabel: str
    ylabel: str
    mark_extra_pts: bool
        If True, will add marker to any plotted extra points, as set by the
        parameter ``extra_pts_arr``.
    full_size_scatter: bool
    scatter_size: int
    scatter_color: str
    marker: str
    legend: array-like
        An array of ``str``.  Must have same length as the parameter ``func_arr``.
    ax: matplotlib axes object, default None
    """
    num_series = len(func_arr)
    # todo: put in array length checks

    if domain_begin_arr is None:
        left_endpt = min(indep)
        domain_begin_arr = [left_endpt for dummy in range(num_series)]
    if domain_end_arr is None:
        right_endpt = max(indep)
        domain_end_arr = [right_endpt for dummy in range(num_series)]

    domain_orig_arr = []
    for domain_begin, domain_end in zip(domain_begin_arr, domain_end_arr):

        domain_orig = np.linspace(start=domain_begin,
                                  stop=domain_end,
                                  num=100)

        domain_orig_arr.append(domain_orig)

    if ax is None:
        f, ax = plt.subplots()

    if scatter_size is None:
        scatter_size = _DEFAULT_SCATTER_SIZE

    ax.scatter(indep,
               dep,
               color=scatter_color,
               s=scatter_size,
               marker=marker)

    if extra_pts_arr is None:
        extra_pts_arr = [None for x in func_arr]

    plotted_lines = []
    for func, domain_orig, extra_pts in zip(func_arr,
                                            domain_orig_arr,
                                            extra_pts_arr):

        domain = np.copy(domain_orig)

        if extra_pts is not None:

            domain = np.concatenate((domain, extra_pts))
            domain.sort()

            if mark_extra_pts:
                for extra_pt in extra_pts:
                    ax.plot(extra_pt, func(extra_pt), 'o', color="red")

        line, = ax.plot(domain, func(domain))
        plotted_lines.append(line)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if legend is not None:
        ax.legend(plotted_lines, legend)

    return ax


def plot_one_bkpt_segreg_rss(indep, dep, lines=True):

    argsort_for_indep = indep.argsort()
    indep_to_use = indep[argsort_for_indep]
    dep_to_use = dep[argsort_for_indep]

    # TODO: make epsilon better
    epsilon = 0.01
    domain = np.linspace(start=indep_to_use[0] + epsilon,
                         stop=indep_to_use[-1] - epsilon,
                         num=100)

    domain = np.concatenate((domain, indep[1:-2]))
    domain = np.array(list(set(domain)))
    domain.sort()

    func = one_bkpt_rss_func(indep=indep_to_use, dep=dep_to_use)

    myrange = [func(x) for x in domain]

    if lines:
        for val in indep_to_use:
            plt.axvline(val, color="red", linewidth=0.5)

    plt.plot(domain, myrange)
    # keep it tight
#    plt.axis('tight')
#    plt.ticklabel_format(useOffset=False, style='plain')


def _compute_surf(X, Y, func):
    rownum = X.shape[0]
    colnum = X.shape[1]
    Z = np.empty((rownum, colnum))
    for i in range(rownum):
        for j in range(colnum):

            x_val = X[j, i]
            y_val = Y[j, i]

            if x_val < y_val:
                val = func(x_val, y_val)
            else:
                val = np.nan

            Z[j, i] = val
    return Z


def plot_two_bkpt_segreg_rss(indep, dep):

    argsort_for_indep = indep.argsort()
    indep_to_use = indep[argsort_for_indep]
    dep_to_use = dep[argsort_for_indep]

    # TODO: make epsilon better
    epsilon = 0.1
    domain = np.linspace(start=indep_to_use[0] + epsilon,
                         stop=indep_to_use[-1] - epsilon,
                         num=100)

    domain = np.concatenate((domain, indep[1:-2]))
    domain.sort()

    func = two_bkpt_rss_func(indep_to_use, dep_to_use)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    X, Y = np.meshgrid(domain, domain)
    Z = _compute_surf(X, Y, func)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('u1')
    ax.set_ylabel('u2')
    ax.set_zlabel('RSS')
