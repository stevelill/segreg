"""
Routines to plot segmented regression results.

DEPRECATED -- WILL BE REMOVED OR MODIFIED SOON
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import datetime
import os

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import statsmodels.api as sm


_FIGSIZE = (12, 6)

# TODO ???


def _handle_plot(save_dir, save_name, show=True):
    if save_dir is not None:
        #plt.savefig(os.path.join(save_dir, save_name + ".pdf"))
        plt.savefig(os.path.join(save_dir, save_name))
    else:
        if show:
            pass
            # plt.show()


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


def plot_segmented_fit(indep,
                       dep,
                       func,
                       bkpt,
                       **kwargs):
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    savedir = kwargs.pop('savedir', None)
    name = kwargs.pop('name', None)
    lines = kwargs.pop('lines', False)
    figsize = kwargs.pop('figsize', None)
    segreg_title = kwargs.pop('segreg_title', None)

    func2 = kwargs.pop('func2', None)
    func2_title = kwargs.pop('func2_title', None)
    func2_is_segreg = kwargs.pop('func2_is_segreg', False)
    func2_bkpt = kwargs.pop('func2_bkpt', None)

    segreg_title_to_use = "Segmented Regression Fit"
    if segreg_title is not None:
        segreg_title_to_use = segreg_title
    # TODO: check who calls this; eventually change this default
    func2_title_to_use = "Residual Sum Squares for Segmented Regression"
    if func2_title is not None:
        func2_title_to_use = func2_title
    # TODO: reverse order of func1, func2

    argsort_for_indep = indep.argsort()
    indep_to_use = indep[argsort_for_indep]
    dep_to_use = dep[argsort_for_indep]

    if func2 is not None:
        if figsize is None:
            figsize = (12, 8)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)

        if func2_is_segreg:
            _plot_segreg(ax1,
                         indep_to_use,
                         dep_to_use,
                         func2,
                         func2_bkpt,
                         func2_title_to_use,
                         xlabel,
                         ylabel,
                         name)
        else:
            _plot_func(ax1,
                       indep_to_use,
                       dep_to_use,
                       func2,
                       bkpt,
                       lines,
                       func2_title_to_use,
                       xlabel,
                       ylabel="rss",
                       name=name)

        _plot_segreg(ax2,
                     indep_to_use,
                     dep_to_use,
                     func,
                     bkpt,
                     segreg_title_to_use,
                     xlabel,
                     ylabel,
                     name)

    else:
        if figsize is None:
            # pass
            figsize = _FIGSIZE

        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        _plot_segreg(ax1,
                     indep_to_use,
                     dep_to_use,
                     func,
                     bkpt,
                     segreg_title_to_use,
                     xlabel,
                     ylabel,
                     name)

    # keep it tight
    # plt.axis('tight')
    #plt.ticklabel_format(useOffset=False, style='plain')

    if savedir is not None:
        if name is not None:
            name_for_file = name.replace(" ", "")

        filepath = os.path.join(savedir, name_for_file + "_segreg_rss")
        plt.savefig(filepath)
    else:
        # TODO: put back erase this pass here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pass
        # plt.show()


def _plot_segreg(ax,
                 indep,
                 dep,
                 func,
                 bkpt,
                 title=None,
                 xlabel=None,
                 ylabel=None,
                 name=None):
    ax.scatter(indep, dep, color="gray", s=2)
    fitted = []
    for val in indep:
        fitted.append(func(val))

    # let's also add breakpoint to plot
    domain = indep
    domain = np.append(domain, bkpt)
    fitted.append(func(bkpt))

    # see also: argsort (it is less magical)
    # resort them pairwise after adding breakpoint
    domain, fitted = list(zip(*sorted(zip(domain, fitted))))

    ax.plot(domain, fitted)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.axvline(bkpt, color="red")

    if title is not None:
        title_to_use = title
        if name is not None:
            title_to_use += " ; " + name

        ax.set_title(title_to_use)

    ax.grid()


def _plot_func(ax,
               indep,
               dep,
               func,
               bkpt,
               lines,
               title=None,
               xlabel=None,
               ylabel=None,
               name=None):
    # TODO: don't we want the scatter?
    ax.scatter(indep, dep, color="gray", s=2)

    domain = np.arange(indep[0], indep[-1], 0.01)

    myrange = []
    for val in domain:
        myrange.append(func(val))

    if lines:
        for val in indep:
            ax.axvline(val, color="red")

    ax.plot(domain, myrange)
    ax.axvline(bkpt, color="red")

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if name is not None:
        title += " ; " + name

    ax.set_title(title)
    ax.grid()

# DEPRECATE???


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

# TODO: expand to multiple breakpoints; ie: x0 becomes array
# DEPRECATED


def plot_segmented_fitORIG(indep, dep, func, x0, **kwargs):
    title = kwargs.pop('title', None)
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    savedir = kwargs.pop('savedir', None)
    shortname = kwargs.pop('shortname', "")

    plt.scatter(indep, dep)
    fitted = []
    for val in indep:
        fitted.append(func(val))

    # let's also add breakpoint to plot
    domain = indep
    domain = np.append(domain, x0)
    fitted.append(func(x0))

    # see also: argsort (it is less magical)
    # let's sort them pairwise
    domain, fitted = list(zip(*sorted(zip(domain, fitted))))

    plt.plot(domain, fitted)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if savedir is not None:
        filepath = os.path.join(savedir, shortname + "_segreg")
        plt.savefig(filepath)

    plt.show()

# TODO: need a smart histogram bin calculator


def plot_boot_sample(series,
                     statistic_name,
                     display_name,
                     show_graph=False,
                     save_dir=None,
                     **kwargs):
    """
    Parameters
    ----------
    series : numpy ndarray, shape: (num,)
        represents bootstrap sample estimates for a single statistic

    kwargs
    ------
    trim: list [lower_percentile, upper_percentile]
        in plot, will trim away below lower_percentile and above
        upper_percentile

    """
    trim = kwargs.pop('trim', None)
    show_estimate = kwargs.pop('show_estimate', None)
    show_mean = kwargs.pop('show_mean', True)

    if trim is not None:
        trim_value_bounds = np.percentile(series, trim)
        series_to_use = series[np.logical_and(series > trim_value_bounds[0],
                                              series < trim_value_bounds[1])]
    else:
        series_to_use = series

    num_iter = len(series_to_use)

    min_rhs = min(series_to_use)
    max_rhs = max(series_to_use)
    binwidth = (max_rhs - min_rhs) / 20.0
    num_bins = 100
    if num_iter < 10000:
        num_bins = 100
    else:
        num_bins = int(num_iter / 100.0)

    # n, bins, patches = plt.hist(rhs_slope_estimates, bins=scipy.arange(min_rhs, max_rhs + binwidth, binwidth), normed=False, histtype='bar')
#    n, bins, patches = plt.hist(series_to_use, num_bins, normed=False, histtype='bar')

    # normed appears to be gone in recent versions

    n, bins, patches = plt.hist(series_to_use,
                                num_bins,
                                # normed=False,
                                density=True,
                                histtype='bar')

    # plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

    plt.xlabel(statistic_name)
    plt.title("Bootstrap Sample Distribution: " + statistic_name
              + "\n" + display_name)

    if show_mean:
        mean = np.mean(series)
        plt.axvline(mean, color="green", label="mean")
    if show_estimate is not None:
        plt.axvline(show_estimate, color="red", label="estimate")

    plt.legend(loc="best")
    plt.grid()

    if save_dir is not None:
        boot_hist_name = "boot_histogram_" + statistic_name + "_" + display_name

        boot_hist_name = boot_hist_name.replace(" ", "_")

        plt.savefig(os.path.join(save_dir, boot_hist_name))
        # plt.close()
    if show_graph:
        plt.plot()

    probplot = sm.ProbPlot(series_to_use, dist="norm", fit=True)
    probplot.qqplot(line='45')

    plt.title("Bootstrap Sample QQ Plot Versus Normal: " + statistic_name
              + "\n" + display_name)
    if save_dir is not None:
        qqplot_name = "boot_qqplot_" + statistic_name + "_" + display_name

        qqplot_name = qqplot_name.replace(" ", "_")
        plt.savefig(os.path.join(save_dir, qqplot_name))
        # plt.close()
    if show_graph:
        plt.show()
