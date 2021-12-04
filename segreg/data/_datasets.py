"""
The ``segreg.datasets`` module provides datasets for testing segmented 
regression.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np


def hinkley():
    """
    A dataset from an academic paper by David Hinkley.

    Inference in Two-Phase Regression
    David Hinkley
    Journal of the American Statistical Association
    Vol. 66, No. 336 (Dec., 1971), pp. 736-743

    Returns
    -------
    indep : numpy array
    dep : numpy array        
    """
    indep = np.array([2.0,
                      2.52288,
                      3.0,
                      3.52288,
                      4.0,
                      4.52288,
                      5.0,
                      5.52288,
                      6.0,
                      6.52288,
                      7.0,
                      7.52288,
                      8.0,
                      8.52288,
                      9.0])
    dep = np.array([0.370483,
                    .537970,
                    .607684,
                    .723323,
                    .761856,
                    .892063,
                    .956707,
                    .940349,
                    .898609,
                    .953850,
                    .990834,
                    .890291,
                    .990779,
                    1.050865,
                    .982785])
    return indep, dep


def test1():
    """
    Random dataset for testing.

    ``dep`` created with one-bkpt model defined by parameters:

    (u=20, v=10, m1=0.0, m2=-0.1) 

    and normal errors.

    Returns
    -------
    indep : numpy array
    dep : numpy array        
    """
    indep = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
                      13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
                      26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.,
                      39.], dtype=float)

    dep = np.array([10.78862702, 10.80865967, 10.36303396, 10.06787099, 10.63787434,
                    10.17632513, 10.03836726, 10.21000339, 10.1880592, 10.83228144,
                    10.4639768, 10.02054785, 10.06725762, 10.36737546, 10.07423906,
                    10.27969998, 10.11165485, 10.18817926, 10.92853834, 10.64732206,
                    10.43788206, 10.24099838, 10.39779815, 10.67018116, 9.61842261,
                    9.86287077, 9.85279224, 9.89176769, 9.33674018, 9.23876682,
                    9.84379126, 9.26903955, 9.4313239, 9.23073589, 8.62025779,
                    8.75997475, 8.69917697, 9.10815078, 8.92514738, 8.11896596],
                   dtype=float)
    return indep, dep


def test2():
    """
    Random dataset for testing.

    ``dep`` created with one-bkpt model defined by parameters:

    (u=50, v=20, m1=0.0, m2=-0.1) 

    and normal errors.

    Returns
    -------
    indep : numpy array
    dep : numpy array        
    """
    indep = np.array([10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
                      21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31.,
                      32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42.,
                      43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53.,
                      54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64.,
                      65., 66., 67., 68., 69., 70., 71., 72., 73., 74., 75.,
                      76., 77., 78., 79., 80., 81., 82., 83., 84., 85., 86.,
                      87., 88., 89., 90., 91., 92., 93., 94., 95., 96., 97.,
                      98., 99., 100., 101., 102., 103., 104., 105., 106., 107.,
                      108., 109.])
    dep = np.array([20.1, 20.2, 19.6, 20.5, 20.4, 20.3, 20.4, 19.9, 19.7, 20.1,
                    20.3, 20.3, 20.3, 19.3, 20.4, 20.7, 20.2, 19.5, 20., 20.7,
                    20.7, 20.3, 20.3, 20.1, 20., 20., 20., 19.9, 19.6, 19.7,
                    20.6, 20.1, 20.2, 20., 20.6, 20.9, 18.8, 20.2, 19., 20.,
                    19.7, 20.2, 20.2, 19.4, 18.7, 19.6, 18.8, 18.6, 19., 18.1,
                    19., 20.2, 18.9, 19.2, 19., 19.3, 18., 17.7, 17.9, 18.3,
                    18.3, 17.1, 18.3, 18.1, 17.3, 17.5, 17., 17.2, 16.4, 17.3,
                    17.1, 16.7, 17.1, 17.6, 15.7, 17.1, 16.7, 16.8, 16.2, 15.1,
                    15.5, 15.7, 15.1, 15.6, 15.2, 16.2, 15.3, 15.1, 14.9, 15.,
                    14.5, 15.4, 14.6, 15., 14.6, 15., 14.2, 13.3, 13.8, 14.])
    return indep, dep
