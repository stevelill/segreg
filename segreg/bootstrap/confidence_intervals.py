"""
Convenience methods to produce formatted outputs for bootstrap confidence 
intervals.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


from pandas import DataFrame
import numpy as np

from segreg.bootstrap import bootstrap_methods
from segreg.bootstrap import resampling


def _create_df(conf_interval, est_params, cols, param_names):
    ci_data = {cols[1]: est_params,
               cols[0]: conf_interval.T[0],
               cols[2]: conf_interval.T[1]}
    ci_df = DataFrame(ci_data, index=param_names, columns=cols)
    ci_df.index.name = "Parameter"

    return ci_df


def _confidence_intervals(params_arr,
                          est_params,
                          significance,
                          estimator,
                          indep,
                          dep):

    param_names = estimator.param_names()

    # in case there are restricted params
    param_names = np.array(param_names)
    param_names = param_names[estimator.estimated_params_indices()]
    cols = ["Left", "Estimate", "Right"]

    # BCa
    bca_conf_interval = bootstrap_methods.model_bca(params_arr,
                                                    est_params,
                                                    estimator,
                                                    indep,
                                                    dep,
                                                    significance=significance)

    bca_ci_df = _create_df(conf_interval=bca_conf_interval,
                           est_params=est_params,
                           cols=cols,
                           param_names=param_names)

    percentile_conf_interval = bootstrap_methods.boot_percentile_conf_interval(params_arr,
                                                                               est_params,
                                                                               significance=significance)

    percentile_ci_df = _create_df(conf_interval=percentile_conf_interval,
                                  est_params=est_params,
                                  cols=cols,
                                  param_names=param_names)

    basic_conf_interval = bootstrap_methods.boot_basic_conf_interval(params_arr,
                                                                     est_params,
                                                                     significance=significance)

    basic_ci_df = _create_df(conf_interval=basic_conf_interval,
                             est_params=est_params,
                             cols=cols,
                             param_names=param_names)

    return bca_ci_df, percentile_ci_df, basic_ci_df


def boot_conf_intervals(indep,
                        dep,
                        estimator,
                        display_name=None,
                        resample_cases=False,
                        significance=0.05,
                        num_sims=10000,
                        verbose=True,
                        seed=None,
                        precision=4):
    """
    Bootstrap confidence intervals for regression models.

    A convenience method to produce formatted outputs for presenting results.
    Returns confidence intervals of the regression model parameters
    for three types of confidence interval: 
        BCa (bootstrapped bias-corrected and accelerated
        percentile
        basic

    These are described in Davison and Hinkley, "Bootstrap Methods and their 
    Application", as well as numerous other sources such as: 
    DiCiccio, T. J., & Efron, B. (1996). "Bootstrap confidence intervals." 
    Statistical Science, 11 (3), 189-228

    The "basic" bootstrap confidence intervals follows the nomenclature of 
    Davison and Hinkley.

    Parameters
    ----------
    indep: array-like
        The independent data.  Also called predictor, explanatory variable,
        regressor, or exogenous variable.
    dep: array-like
        The dependent data.  Also called response, regressand, or endogenous
        variable.
    display_name: str
    resample_cases : boolean, default False
        If True, the bootstrap will resample pairs with replacement 
        from (indep, dep).  See Section 6.2.4 in Davison and Hinkley, 
        "Bootstrap Methods and their Application".
    significance : float
        The coverage of the confidence interval would be: 1 - significance.
        For example, a 95% confidence interval corresponds to a ``significance`` 
        value of 0.05.
    num_sims: int
        Number of bootstrap simulations.
    verbose: bool
    seed: int
        Seed for random generator driving bootstrap simulations.
    precision: int
        Decimal precision for outputs. 

    Returns
    -------
    bca_ci_df: pandas DataFrame
    percentile_ci_df: pandas DataFrame
    basic_ci_df: pandas DataFrame
    """
    if display_name is None:
        display_name = ""

    est_params = estimator.fit(indep, dep)
    est_params = np.array(est_params)

    params_arr = resampling.boot_param_dist(indep=indep,
                                            dep=dep,
                                            estimator=estimator,
                                            num_iter=num_sims,
                                            resample_cases=resample_cases,
                                            seed=seed,
                                            include_fixed_params=False,
                                            verbose=verbose)

    if estimator.has_restricted_params():
        est_params = est_params[estimator.estimated_params_indices()]

    (bca_ci_df,
     percentile_ci_df,
     basic_ci_df) = _confidence_intervals(params_arr=params_arr,
                                          est_params=est_params,
                                          significance=significance,
                                          estimator=estimator,
                                          indep=indep,
                                          dep=dep)

    if verbose:
        def my_formatter(x):
            format_str = '.' + str(precision) + 'f'
            return format(x, format_str)

        formatters = [my_formatter for dummy in range(len(bca_ci_df.columns))]

        print()
        print("confidence level: ", 100.0 * (1.0 - significance), "%")
        print()
        print("bootstrap bca confidence intervals")
        print()
        print(bca_ci_df.to_string(formatters=formatters))
#        if latex:
#            print(bca_ci_df.to_latex(escape=False, formatters=formatters))
#        else:
        print("bootstrap percentile confidence intervals")
        print()
        print(percentile_ci_df.to_string(formatters=formatters))
        print()
        print("bootstrap basic confidence intervals")
        print()
        print(basic_ci_df.to_string(formatters=formatters))
        print()

    return bca_ci_df, percentile_ci_df, basic_ci_df
