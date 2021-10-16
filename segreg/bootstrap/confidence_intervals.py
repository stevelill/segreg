"""

"""

# Author: Steven Lillywhite
# License: BSD 3 clause

# from segreg.segmented_regression

from pandas import DataFrame
import numpy as np

from segreg.bootstrap import bootstrap_methods
from segreg.bootstrap import resampling


# TODO: clean this one up!!!
FIGSIZE = (12, 6)


def _confidence_intervals(params_arr,
                          est_params,
                          significance,
                          estimator,
                          indep,
                          dep,
                          verbose=False,
                          **kwargs):
    add_basic_ci = kwargs.pop('add_basic_ci', False)
    add_percentile_ci = kwargs.pop('add_percentile_ci', False)
    latex = kwargs.pop('latex', False)
    precision = kwargs.pop('precision', 4)

    param_names = estimator.param_names()

    # in case there are restricted params
    param_names = np.array(param_names)
    param_names = param_names[estimator.estimated_params_indices()]

#    cols = ["estimate", "lower", "upper"]

    cols = ["Left", "Estimate", "Right"]

    def my_formatter(x):
        format_str = '.' + str(precision) + 'f'
        return format(x, format_str)


#    my_formatter = text_util.get_decimal_formatter(precision=precision)
    formatters = [my_formatter for dummy in range(len(cols))]

    bca_conf_interval = bootstrap_methods.model_bca(params_arr,
                                                    est_params,
                                                    estimator,
                                                    indep,
                                                    dep,
                                                    significance=significance)

    ci_data = {cols[1]: est_params,
               cols[0]: bca_conf_interval.T[0],
               cols[2]: bca_conf_interval.T[1]}
    bca_ci_df = DataFrame(ci_data, index=param_names, columns=cols)
    bca_ci_df.index.name = "Parameter"

    if verbose:
        print()
        print("confidence level: ", 100.0 * (1.0 - significance), "%")
        print()
        print("bootstrap bca confidence intervals")
        print()
        if latex:
            print(bca_ci_df.to_latex(escape=False, formatters=formatters))
        else:
            print(bca_ci_df.to_string(formatters=formatters))
        print()

    if add_basic_ci:
        basic_conf_interval = bootstrap_methods.boot_basic_conf_interval(params_arr,
                                                                         est_params,
                                                                         significance=significance)

        ci_data = {cols[1]: est_params,
                   cols[0]: basic_conf_interval.T[0],
                   cols[2]: basic_conf_interval.T[1]}

        basic_ci_df = DataFrame(ci_data, index=param_names, columns=cols)

        if verbose:
            print()
            print("bootstrap basic confidence intervals")
            print()
            if latex:
                print(basic_ci_df.to_latex(escape=False, formatters=formatters))
            else:
                print(basic_ci_df.to_string(formatters=formatters))
            print()

    if add_percentile_ci:
        percentile_conf_interval = bootstrap_methods.boot_percentile_conf_interval(params_arr,
                                                                                   est_params,
                                                                                   significance=significance)

        ci_data = {cols[1]: est_params,
                   cols[0]: percentile_conf_interval.T[0],
                   cols[2]: percentile_conf_interval.T[1]}

        ci_df = DataFrame(ci_data, index=param_names, columns=cols)

        if verbose:
            print()
            print("bootstrap percentile confidence intervals")
            print()
            if latex:
                print(ci_df.to_latex(escape=False, formatters=formatters))
            else:
                print(ci_df.to_string(formatters=formatters))
            print()

    # NEW 2021
    return bca_ci_df


def boot_conf_intervals(indep,
                        dep,
                        estimator,
                        display_name=None,
                        resample_cases=False,
                        significance=0.05,
                        num_iter=10000,
                        verbose=True,
                        add_basic_ci=False,
                        add_percentile_ci=False,
                        preserve_cond_var=False,
                        seed=None,
                        precision=4):

    if display_name is None:
        display_name = ""

    est_params = estimator.fit(indep, dep)
    est_params = np.array(est_params)

    # TODO: take this back outside this method
#     if verbose:
#         print()
#         print(display_name + " estimated parameters: ")
#         latex = False
#         param_names = estimator.param_names(latex=latex)
#         for i, name in enumerate(param_names):
#             print(name, ": ", text_util.format_num(est_params[i]))

    params_arr = resampling.boot_param_dist(indep,
                                            dep,
                                            estimator,
                                            num_iter,
                                            resample_cases=resample_cases,
                                            seed=seed,
                                            preserve_cond_var=preserve_cond_var,
                                            include_fixed_params=False,
                                            verbose=verbose)

    if estimator.has_restricted_params():
        est_params = est_params[estimator.estimated_params_indices()]

    bca_ci_df = _confidence_intervals(params_arr,
                                      est_params,
                                      significance,
                                      estimator,
                                      indep,
                                      dep,
                                      add_basic_ci=add_basic_ci,
                                      add_percentile_ci=add_percentile_ci,
                                      precision=precision,
                                      verbose=verbose)

    return bca_ci_df
