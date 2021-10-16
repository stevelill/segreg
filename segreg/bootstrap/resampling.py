"""
Bootstrap resampling methods.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import numpy as np
import statsmodels.api as sm


def _normalize_residuals(indep, resid):
    ########################################################################
    # get estimate of resid variance
    # new -- put aic
    # TODO: one should check both ls-cv and aic visually
    model = sm.nonparametric.KernelReg(resid * resid,
                                       indep,
                                       "c",
                                       reg_type="ll",
                                       bw="aic")
    mean, mfx = model.fit()
    est_cond_stddev = np.sqrt(mean)
    resid_to_use = resid / est_cond_stddev

    return resid_to_use, est_cond_stddev


def boot_resample(indep,
                  dep,
                  fitted_values=None,
                  resid=None,
                  resample_cases=False,
                  preserve_cond_var=False,
                  **kwargs):
    cond_var_residuals = kwargs.pop('cond_var_residuals', None)
    est_cond_stddev = kwargs.pop('est_cond_stddev', None)
    if resample_cases:
        indep_resample, dep_resample = random_selection_with_replacement_two_series(indep, dep)
    else:
        # or from empirical residuals
        if preserve_cond_var:
            epsilon = random_selection_with_replacement(cond_var_residuals)
            epsilon = np.multiply(epsilon, est_cond_stddev)
        else:
            epsilon = random_selection_with_replacement(resid)

        dep_resample = fitted_values + epsilon
        indep_resample = indep

    return indep_resample, dep_resample


def boot_param_dist(indep,
                    dep,
                    estimator,
                    num_iter,
                    resample_cases=False,
                    seed=None,
                    **kwargs):
    """
    Computes sampling distribution of model parameters.

    TODO: finish

    Parameters
    ----------
    estimator : subclass of segreg.statistics.estimator.Estimator
    resample_cases : boolean, default False
        see 6.2.4 Davison and Hinkley


    Returns
    -------
    param sims: scipy array shape (num_iter, num_params)
        returns sampling distribution for the parameters; a matrix where each
        row represents parameter estimates for a sample from the data-generating
        distribution
    """
    verbose = kwargs.pop('verbose', True)
    preserve_cond_var = kwargs.pop('preserve_cond_var', False)
    diagnoser = kwargs.pop('diagnoser', None)
    include_fixed_params = kwargs.pop('include_fixed_params', True)

    estimator.fit(indep, dep)
    func = estimator.get_func()

    fitted_values = func(indep)

    if seed is not None:
        np.random.seed(seed)

    num_data = len(indep)

    if verbose:
        boot_type = "resample residuals"
        if resample_cases:
            boot_type = "resample cases"
        print()
        print("Bootstrap parameter statistics")
        print("num bootstrap resamples: ", str(num_iter))
        print(boot_type)
        print()

    resid = estimator.residuals()

    if preserve_cond_var:
        resid_to_use, est_cond_stddev = _normalize_residuals(indep, resid)
        print()
        print("TRY TO PRESERVE ORIG CONDITIONAL VARIANCE")
        print("orig resid stddev: ", np.std(resid))
        print("new resid stddev: ", np.std(resid_to_use))
        print()
        ########################################################################

    params_arr = []

    # small speed-up to pre-compute these
    #resample_indices = np.random.choice(num_data, (num_iter, num_data) )

    for i in range(num_iter):
        if (i % 200000 == 0):
            if verbose:
                print("iter: ", i)

        indep_curr, dep_curr = boot_resample(indep,
                                             dep,
                                             fitted_values,
                                             resid,
                                             resample_cases,
                                             preserve_cond_var)

        # check resampled data size for sanity
        curr_num_indep = len(indep_curr)
        curr_num_dep = len(dep_curr)
        if (curr_num_indep != num_data) or (curr_num_dep != num_data):
            raise Exception("num boot resamples not equal to orig num data!!!")

        curr_params = estimator.fit(indep_curr, dep_curr)

        params_arr.append(curr_params)

        # potential diagnosis
        if diagnoser is not None:
            diagnoser(estimator, curr_params, indep_curr, dep_curr)

    params_arr = np.array(params_arr)

    if not include_fixed_params:
        estimated_params_indices = estimator.estimated_params_indices()
        params_arr = params_arr[:, estimated_params_indices]

    return params_arr


def random_selection_with_replacement(series):
    """
    Random draw with replacement from series.  Returns same number of elements
    as the input.

    PARAMETERS
    ----------
    series : array-like

    RETURNS
    -------
    numpy array
    """
    series_to_use = np.array(series)
    num = len(series_to_use)

    choice_indices = np.random.choice(num, num)

    return series_to_use[choice_indices]


def random_selection_with_replacement_two_series(series1, series2):
    """
    Random draw with replacement.  Each draw is from the same index for the
    two series.  The two series are assumed to have the same length, and
    the number of draws is equal to the number of elements of the series.

    PARAMETERS
    ----------
    series1 : array-like
    series2 : array-like
        must have same number of elements as series1

    RETURNS
    -------
    tuple
        tuple of numpy array
    """
    num = len(series1)
    if num != len(series2):
        raise Exception("length of series1: " + str(num) +
                        " not equal to " +
                        "length of series2: " + str(len(series2)))

    series_to_use1 = np.array(series1)
    series_to_use2 = np.array(series2)

    choice_indices = np.random.choice(num, num)

    return series_to_use1[choice_indices], series_to_use2[choice_indices]
