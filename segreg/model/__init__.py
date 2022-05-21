"""
The ``segreg.model`` module implements segmented regression models.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


from segreg.model.ols_estimator import OLSRegressionEstimator
from segreg.model.segreg_estimator import OneBkptSegRegEstimator
from segreg.model.segreg_estimator import TwoBkptSegRegEstimator

from segreg.model.one_bkpt_segreg import one_bkpt_rss_func
from segreg.model.segreg_util import two_bkpt_rss_func

__all__ = ["OLSRegressionEstimator",
           "OneBkptSegRegEstimator",
           "TwoBkptSegRegEstimator",
           "one_bkpt_rss_func",
           "two_bkpt_rss_func"]
