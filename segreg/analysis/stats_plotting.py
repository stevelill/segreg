"""
Routines to plot segmented regression results.

DEPRECATED -- WILL BE REMOVED OR MODIFIED SOON
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import os

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np


def _handle_plot(save_dir, save_name, show=True):
    if save_dir is not None:
        #plt.savefig(os.path.join(save_dir, save_name + ".pdf"))
        plt.savefig(os.path.join(save_dir, save_name))
    else:
        if show:
            pass
            # plt.show()

# called by mathsci


def plot_model(func_arr,
               indep,
               dep,
               domain_begin_arr=None,
               domain_end_arr=None,
               extra_pts_arr=None,
               title=None,
               xlabel=None,
               ylabel=None,
               mark_extra_pts=True,
               padding_scale=5.0,
               full_size_scatter=False,
               scatter_size=3,
               scatter_color="gray",
               marker="o",
               ax=None):
    """
    Assumes data is sorted.
    """
    num_series = len(func_arr)
    # todo: put in array length checks

    if domain_begin_arr is None:
        domain_begin_arr = [indep[0] for dummy in range(num_series)]
    if domain_end_arr is None:
        domain_end_arr = [indep[-1] for dummy in range(num_series)]

    domain_orig_arr = []
    for domain_begin, domain_end in zip(domain_begin_arr, domain_end_arr):
        # todo clean up dx
        dx = (domain_end - domain_begin) / 100.0
        #domain_orig = np.arange(domain_begin, domain_end + dx, dx)

        domain_orig = np.linspace(start=domain_begin,
                                  stop=domain_end,
                                  num=100)

        domain_orig_arr.append(domain_orig)

    if ax is None:
        # todo: finish
        f, ax = plt.subplots()

    if full_size_scatter:
        ax.scatter(indep, dep, color=scatter_color)
    else:
        #plt.scatter(indep, dep, color="gray", s=2)
        ax.scatter(indep,
                   dep,
                   color=scatter_color,
                   s=scatter_size,
                   marker=marker)

    if extra_pts_arr is None:
        extra_pts_arr = [None for x in func_arr]
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

        # print(domain)
        ax.plot(domain, func(domain))


#    plt.grid()

    # if padding is None:
    padding = padding_scale * dx
    #print("padding: ", padding)

    #padding_left = padding * abs(domain_begin)
    #padding_right = padding * abs(domain_end)
    #padding_to_use = max(padding_left, padding_right)


# MOST RECENT WAS THIS for xlim
#    plt.xlim([min(domain_begin_arr) - padding,
#              max(domain_end_arr) + padding])


#     ax = plt.gca()
#     lim = ax.get_xlim()
#     curr_xticks = list(ax.get_xticks())
#     ax.set_xticks(curr_xticks[0:-1] + [domain_end])
#     ax.set_xlim(lim)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

#    import matplotlib.dates as mdates
#    ax = plt.gca()
#    myticks = ax.get_xticks()
#    mylabels = ax.get_xticklabels()

#    newticks = [datetime.date(year=int(x), month=1, day=1) for x in myticks]

#    ax.set_xticklabels(newticks)
#    ax.xaxis.set_major_locator(mdates.YearLocator(40, 1))

#    fig = plt.gcf()

#    fig.autofmt_xdate()

#    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

#    print("my x axis ticks")
#    print(myticks)

#    print("my x axis labels")
#    print(mylabels)

#    gu.date_ticks(ax=ax, num_years=5, month=Month.SEP, date_format='%Y')

#    ax.xaxis.set_major_locator(MaxNLocator(20))

    return ax
#    plt.close()


def plot_fitted_model(domain_begin,
                      domain_end,
                      func,
                      indep,
                      dep,
                      extra_pts=None,
                      title=None,
                      xlabel=None,
                      ylabel=None,
                      save_dir=None,
                      save_name=None,
                      mark_extra_pts=False,
                      padding=0.005,
                      show=True,
                      full_size_scatter=False):
    """
    Assumes data is sorted.
    """
    dx = (domain_end - domain_begin) / 100.0
    domain = np.arange(domain_begin, domain_end + dx, dx)
    if extra_pts is not None:
        domain = np.concatenate((domain, extra_pts))
    domain.sort()

    myrange = func(domain)

#    plt.figure()

    if full_size_scatter:
        plt.scatter(indep, dep, color="gray")
    else:
        #plt.scatter(indep, dep, color="gray", s=2)
        plt.scatter(indep, dep, color="gray", s=3)

    plt.plot(domain, myrange)
    plt.grid()

    padding_left = padding * abs(domain_begin)
    padding_right = padding * abs(domain_end)
    padding_to_use = max(padding_left, padding_right)

    plt.xlim([domain_begin - padding_to_use,
              domain_end + padding_to_use])

#     ax = plt.gca()
#     lim = ax.get_xlim()
#     curr_xticks = list(ax.get_xticks())
#     ax.set_xticks(curr_xticks[0:-1] + [domain_end])
#     ax.set_xlim(lim)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if mark_extra_pts:
        for extra_pt in extra_pts:
            plt.plot(extra_pt, func(extra_pt), 'o', color="red")

    _handle_plot(save_dir, save_name, show)

#    plt.close()


def plot_fit(estimator):
    pass


# DEPRECATE???

# called by notebooks


def plot_segmented_sumsq(indep, segmented_sumsq_func, **kwargs):
    title = kwargs.pop('title', None)
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    savedir = kwargs.pop('savedir', None)
    name = kwargs.pop('name', "")
    lines = kwargs.pop('lines', False)

    argsort_for_indep = indep.argsort()
    indep_to_use = indep[argsort_for_indep]

    dx = 0.01
    domain = np.arange(indep_to_use[0] + dx, indep_to_use[-1], dx)

    #domain = np.arange(indep_to_use[1], indep_to_use[-1], 0.01)

    myrange = []
    for val in domain:
        myrange.append(segmented_sumsq_func(val))

    if lines:
        for val in indep_to_use:
            plt.axvline(val, color="red", linewidth=0.5)

    plt.plot(domain, myrange)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    else:
        plt.title("Residual Sum Squares for Segmented Regression; " + name)
    # keep it tight
    plt.axis('tight')
    plt.ticklabel_format(useOffset=False, style='plain')

    if savedir is not None:
        filepath = os.path.join(savedir, name + "_segreg_sumsq")
        plt.savefig(filepath)

    plt.grid()
    # plt.show()
