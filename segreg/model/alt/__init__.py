"""
The ``segreg.model.alt`` module implements segmented regression models.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

from segreg.model.alt.brute_force_segreg import estimate_one_bkpt_segreg as brute_fit_one_bkpt
from segreg.model.alt.brute_force_segreg import estimate_two_bkpt_segreg as brute_fit_two_bkpt

from segreg.model.alt.one_bkpt_segreg_alt import estimate_one_bkpt_segreg as fit_one_bkpt
from segreg.model.alt.two_bkpt_segreg_alt import estimate_two_bkpt_segreg as fit_two_bkpt


__all__ = ["brute_fit_one_bkpt",
           "brute_fit_two_bkpt",
           "fit_one_bkpt",
           "fit_two_bkpt"
           ]
