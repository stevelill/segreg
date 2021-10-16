"""
The ``segreg.model.alt`` module implements segmented regression models.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from segreg.model.alt.brute_force_segreg import estimate_one_bkpt_segreg as brute_fit_one_bkpt
from segreg.model.alt.brute_force_segreg import estimate_two_bkpt_segreg as brute_fit_two_bkpt


#from segreg.model.alt.likelihood_util import loglikelihood as foober

# TODO: fill out alt interface

__all__ = ["brute_fit_one_bkpt",
           "brute_fit_two_bkpt",
           ]
