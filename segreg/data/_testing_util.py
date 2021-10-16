"""
Routines to generate fake data to use for testing purposes.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import numpy as np


def generate_fake_data(num_data, x_min, x_max, seed=None):
    """
    Generates an 1D array of fake data.

    Parameters
    ----------
    num_data: int
    x_min: float
        lower bound for the independent data
    x_max: float
        upper bound for the independent data
    seed: int
        sets the seed for the random number generator

    Returns
    -------
    indep: numpy array ndim 1
        data is returned sorted
    """
    if seed is not None:
        np.random.seed(seed)

    x_range = x_max - x_min
    indep = np.random.random(num_data) * x_range + x_min
    indep = np.sort(indep)
    return indep


def _generate_fake_data_no_noise(num_data, x_min, x_max, func, seed=None):
    """
    Generates fake one-dimensional OLS data.

    The independent data is generated randomly, and the dependent data is a
    deterministic function of the independent data.

    Parameters
    ----------
    num_data: int
    x_min: float
        lower bound for the independent data
    x_max: float
        upper bound for the independent data
    func: a function taking an array and returning an array
        generates the dep data
    seed: int
        sets the seed for the random number generator

    Returns
    -------
    indep: numpy array ndim 1
        data is returned sorted
    dep:  numpy array ndim 1
        same length as indep
    """
    indep = generate_fake_data(num_data, x_min, x_max, seed=seed)

    dep = func(indep)

    return indep, dep


def generate_fake_data_normal_errors(num_data,
                                     x_min,
                                     x_max,
                                     func,
                                     std_dev,
                                     seed=None):
    """
    Generates fake one-dimensional OLS data.

    The independent data is generated randomly, and the dependent data is a
    function of the independent data plus random noise.

    The noise is Gaussian with mean zero and standard deviation as specified by
    the input.

    Parameters
    ----------
    num_data: int
    x_min: float
        lower bound for the independent data
    x_max: float
        upper bound for the independent data
    func: a function taking an array and returning an array
        generates the dep data
    std_dev: float
        the standard deviation of the noise
    seed: int
        sets the seed for the random number generator

    Returns
    -------
    indep: numpy array ndim 1
        data is returned sorted
    dep:  numpy array ndim 1
        same length as indep
    """

    if seed is not None:
        np.random.seed(seed)
    indep, dep = _generate_fake_data_no_noise(num_data,
                                              x_min,
                                              x_max,
                                              func,
                                              seed=seed)
    resid = std_dev * np.random.randn(num_data)
    dep += resid

    return indep, dep
