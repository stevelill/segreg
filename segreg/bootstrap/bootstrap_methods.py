"""
Routines used in statistical bootstrap methods.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np
from scipy.stats import norm


def _bca_acceleration_impl(jackknife_estimates):
    jackknife_estimates = np.array(jackknife_estimates)
    mean = np.mean(jackknife_estimates, axis=0)

    demeaned = mean - jackknife_estimates

    # TODO: make sure formula not this way
    # begin erase trying
    #####demeaned = jackknife_estimates - mean
    # end erase trying

    numerator = np.sum(demeaned ** 3, axis=0)

    term2 = np.sum(demeaned ** 2, axis=0)
    denom = 6.0 * np.power(term2, 1.5)

    acceleration = numerator / denom

    return acceleration

# assumes dep has shape (num,)


def bca_acceleration(estimator, indep, dep):
    """
    see: stata manual on the bootstrap

    Returns
    -------
    bca acceleration: scipy array shape (num_params, )
    """
    num_data = len(indep)

    jackknife_estimates = []
    for i in range(num_data):
        sub_indep = np.delete(indep, i, axis=0)
        sub_dep = np.delete(dep, i)

        sub_params = estimator.fit(sub_indep, sub_dep)

        # in case of fixed params in model
        sub_params = sub_params[estimator.estimated_params_indices()]

        jackknife_estimates.append(sub_params)

    acceleration = _bca_acceleration_impl(jackknife_estimates)

    # we guarantee to return array_lik
    if not hasattr(acceleration, "__iter__"):
        acceleration = np.array([acceleration])
    return acceleration

# TODO: put resampling inside this method -- in principle, bca algo depends
# on resampling method: eg: parametric versus non-parametric


def model_bca(boot_sims,
              orig_sample_estimate,
              estimator,
              indep,
              dep,
              significance=0.05,
              **kwargs):
    """
    Computes bca confidence intervals for regression-like data and a
    regression-like estimator.

    TODO: see my write-up for formula and refs.
    (We follow formula in stata bootstrap manual.)

    Note
    ----
    This computes bca confidence intervals for each parameter independently.
    In particular, it does not compute any joint parameter region.

    Warning
    -------
    If there are any fixed parameters in the model, they are assumed to be
    excluded from the arguments passed to this method.

    Parameters
    ----------
    boot_sims : scipy ndarray shape: (num_simulations, num_params)
        ie: each column represents bootstrap simulations for a single statistic
        should not include any columns corresponding to fixed params in the
        model

    orig_sample_estimate : scipy ndarray shape: (num_params,)
        the estimate for the parameters from the original data sample
        should not include fixed params
    estimator: type segreg.statistics.estimator.Estimator

    indep: scipy array shape (num_data,)
        original data
    dep: scipy array shape (num_data,)
        original data

    Returns
    -------
    bca confidence intervals: scipy array shape (num_boot_params, 2)
        columns are endpts of confidence intervals, left, right, respectively
    """
    no_acceleration = kwargs.pop('no_acceleration', False)
    verbose = kwargs.pop('verbose', False)

    num_boot_params = boot_sims.shape[1]

    if len(orig_sample_estimate) != num_boot_params:
        raise Exception("num boot columns: " + str(num_boot_params) +
                        " not equal to num orig_sample_estimate: " +
                        str(len(orig_sample_estimate)))

    if no_acceleration:
        if verbose:
            print()
            print("bca: acceleration set to zero")
            print()
        bca_acceleration_arr = np.zeros(num_boot_params)
    else:
        bca_acceleration_arr = bca_acceleration(estimator, indep, dep)

    boot_sims_trans = boot_sims.T

    bca_arr = []
    for param_sims, param_est, acceleration in zip(boot_sims_trans,
                                                   orig_sample_estimate,
                                                   bca_acceleration_arr):

        if np.isnan(acceleration):
            acceleration = 0.0
            print("cannot compute acceleration; setting to: ", acceleration)

        bca_arr.append(bca(param_sims, param_est, acceleration, significance))

    return np.array(bca_arr)


def bca(boot_sims,
        orig_sample_estimate,
        acceleration,
        significance,
        ties=False,
        respect_right=False):
    """
    Computes bca confidence intervals for a single parameter.

    TODO: see my write-up for formula and refs.
    (We follow formula in stata bootstrap manual.)

    Parameters
    ----------
    boot_sims: scipy array shape (num_sims,)
    orig_sample_estimate: float
        the estimate for the parameter from the original data sample
    acceleration: float
        acceleration term for bca (can be zero)
    significance : float
        the coverage of the confidence interval would be: 1 - significance
        eg: a 95% confidence interval corresponds to a significance value
        of 0.05

    Returns
    -------
    bca confidence interval: scipy array shape (2,)
        endpts of confidence interval, left, right, respectively
    """
    sorted_boot_sims = np.sort(boot_sims)

    num_sims = len(sorted_boot_sims)

    numerator = len(sorted_boot_sims[sorted_boot_sims <= orig_sample_estimate])

    if ties:
        numerator -= 0.5 * len(sorted_boot_sims[sorted_boot_sims == orig_sample_estimate])

    denom = num_sims

    median_bias = norm.ppf(numerator / denom)

    std_normal_quantile = norm.ppf(1.0 - significance / 2.0)

    term = median_bias - std_normal_quantile
    lower_arg = median_bias + term / (1.0 - acceleration * term)

    term = median_bias + std_normal_quantile
    upper_arg = median_bias + term / (1.0 - acceleration * term)

    lower_quantile = norm.cdf(lower_arg)
    upper_quantile = norm.cdf(upper_arg)

    # convert to percentages
    quantiles = 100.0 * np.array([lower_quantile, upper_quantile])

    quantiles_to_use = quantiles

    if respect_right:
        # if right bca quantile less than that for standard percentile
        # interval, do not use it; use the standard percentile interval
        # instead
        even_right_quantile = 100.0 * (1.0 - significance / 2.0)
        if quantiles[1] < even_right_quantile:
            even_left_quantile = 100.0 - even_right_quantile
            quantiles_to_use = [even_left_quantile, even_right_quantile]

    bounds = np.percentile(sorted_boot_sims, quantiles_to_use)

    # TODO: put a try/except???
#     try:
#         bounds = np.percentile(sorted_boot_sims, quantiles)
#     except Exception:
#         print("blah")
#         raise Exception()

    return bounds


def boot_basic_conf_interval(boot_sims,
                             orig_sample_estimate,
                             significance=0.05):
    """
    Name follows Davison and Hinkley.

    Warning
    -------
    If there are any fixed parameters in the model, they are assumed to be
    excluded from the arguments passed to this method.

    Parameters
    ----------
    boot_sims : scipy ndarray shape: (num_simulations, num_params)
        ie: each column represents bootstrap simulations for a single statistic
        should not include any columns corresponding to fixed params in the
        model

    orig_sample_estimate : scipy ndarray shape: (num_params,)
        should not include fixed params
    """
    num_boot_params = boot_sims.shape[1]

    if len(orig_sample_estimate) != num_boot_params:
        raise Exception("num boot columns: " + str(num_boot_params) +
                        " not equal to num orig_sample_estimate: " +
                        str(len(orig_sample_estimate)))

    # convert to percentage
    significance *= 100.0

    lower_quantile = significance / 2.0
    upper_quantile = 100.0 - lower_quantile

    boot_quantiles = np.percentile(boot_sims,
                                   [upper_quantile, lower_quantile],
                                   axis=0)

    boot_bounds = 2.0 * orig_sample_estimate - boot_quantiles
    boot_bounds = boot_bounds.T

    return boot_bounds


def boot_percentile_conf_interval(boot_sims,
                                  orig_sample_estimate,
                                  significance=0.05):
    """

    Warning
    -------
    If there are any fixed parameters in the model, they are assumed to be
    excluded from the arguments passed to this method.

    Parameters
    ----------
    boot_sims : scipy ndarray shape: (num_simulations, num_params)
        ie: each column represents bootstrap simulations for a single statistic
        should not include any columns corresponding to fixed params in the
        model

    orig_sample_estimate : scipy ndarray shape: (num_params,)
        should not include fixed params
    """
    num_boot_params = boot_sims.shape[1]

    if len(orig_sample_estimate) != num_boot_params:
        raise Exception("num boot columns: " + str(num_boot_params) +
                        " not equal to num orig_sample_estimate: " +
                        str(len(orig_sample_estimate)))

    # convert to percentage
    significance *= 100.0

    lower_quantile = significance / 2.0
    upper_quantile = 100.0 - lower_quantile

    boot_quantiles = np.percentile(boot_sims,
                                   [lower_quantile, upper_quantile],
                                   axis=0)

    boot_bounds = boot_quantiles.T

    return boot_bounds
