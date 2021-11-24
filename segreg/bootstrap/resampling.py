"""
Bootstrap resampling methods.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import numpy as np


def boot_resample(indep,
                  dep,
                  fitted_values=None,
                  resid=None,
                  resample_cases=False):
    """
    A single boot resampling for a regression model.

    If ``resample_cases`` is False (the default), then both ``fitted_values``
    and ``resid`` must be set.  In this case, it is assumed that there is some 
    (fitted) model of the form:

        .. math::
            y = f(x) + \epsilon

    and the ``fitted_values`` are :math:`\{f(x_i)\}`, where 
    ``indep`` = :math:`\{x_1, x_2, \dots, x_n\}`.  If 
    ``resid`` = :math:`\{\epsilon_1, \epsilon_2, \dots, \epsilon_n\}`, then
    this function takes a random sample with replacement from ``resid``, 
    :math:`\{\epsilon_1^*, \epsilon_2^*, \dots, \epsilon_n^*\}`, and returns

        .. math::
            \{f(x_i) + \epsilon_i^*\}
    as the resampled dependent values.
    

    Parameters
    ----------
    indep: array-like
        The independent data.  Also called predictor, explanatory variable,
        regressor, or exogenous variable.
    dep: array-like
        The dependent data.  Also called response, regressand, or endogenous
        variable.
    fitted_values: array-like of shape (len(indep),)
        The returned dependent data will be these values plus bootstrap 
        residuals (random draws with replacement from ``resid``).
    resid: array-like of shape (len(``indep``),)
        Residuals from which to draw a bootstrap resample (random sample with
        replacement).
    resample_cases : boolean, default False
        If True, the bootstrap will resample pairs with replacement 
        from (``indep``, ``dep``).  See Section 6.2.4 in Davison and Hinkley, 
        "Bootstrap Methods and their Application".

    Returns
    -------
    indep_resample: numpy array
        When ``resample_cases`` is False (the default), the returned array is
        the input ``indep``.  That is, a copy is not made.
    dep_resample: numpy array
    """

    if resample_cases:
        (indep_resample,
         dep_resample) = random_selection_with_replacement_two_series(indep,
                                                                      dep)
    else:
        if resid is None:
            raise ValueError("resid must be set")
        if fitted_values is None:
            raise ValueError("fitted values must be set")

        # or from empirical residuals
        epsilon = random_selection_with_replacement(resid)
        dep_resample = fitted_values + epsilon
        indep_resample = indep

    return indep_resample, dep_resample


def boot_param_dist(indep,
                    dep,
                    estimator,
                    num_sims,
                    resample_cases=False,
                    seed=None,
                    verbose=False,
                    diagnoser=None,
                    include_fixed_params=True):
    """
    Computes bootstrap sampling distribution of model parameters.

    Parameters
    ----------
    indep: array-like
        The independent data.  Also called predictor, explanatory variable,
        regressor, or exogenous variable.
    dep: array-like
        The dependent data.  Also called response, regressand, or endogenous
        variable.
    estimator : subclass of segreg.model.estimator.Estimator
    num_sims: int
        Number of bootstrap simulations.
    resample_cases : boolean, default False
        If True, the bootstrap will resample pairs with replacement 
        from (indep, dep).  See Section 6.2.4 in Davison and Hinkley, 
        "Bootstrap Methods and their Application".
    seed: int
        Seed for random generator driving bootstrap simulations.
    verbose: bool
    diagnoser: function object taking params: estimator, params, indep, dep
        Used for diagnosing each bootstrap resample.  This is currently a
        developmental feature.
    include_fixed_params: bool

    Returns
    -------
    param sims: numpy array shape (num_sims, num_params)
        Returns bootstrap sampling distribution for the parameters: panel data 
        where each row represents parameter estimates for a bootstrap sample.
        The columns are in the same order as given by the ``estimator`` input.
    """

    estimator.fit(indep, dep)
    func = estimator.model_function

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
        print("num bootstrap resamples: ", str(num_sims))
        print(boot_type)
        print()

    resid = estimator.residuals

    params_arr = []

    # small speed-up to pre-compute these
    #resample_indices = np.random.choice(num_data, (num_sims, num_data) )

    for i in range(num_sims):
        if (i % 200000 == 0):
            if verbose:
                print("iter: ", i)

        indep_curr, dep_curr = boot_resample(indep=indep,
                                             dep=dep,
                                             fitted_values=fitted_values,
                                             resid=resid,
                                             resample_cases=resample_cases)

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
        estimated_params_indices = estimator.estimated_params_indices
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
        Must have same number of elements as series1.

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
