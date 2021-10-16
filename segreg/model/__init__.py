"""
The ``segreg.model`` module implements segmented regression models.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


from segreg.model.ols_estimator import OLSRegressionEstimator
from segreg.model.segreg_estimator import OneBkptSegRegEstimator
from segreg.model.segreg_estimator import TwoBkptSegRegEstimator

__all__ = ["OLSRegressionEstimator",
           "OneBkptSegRegEstimator",
           "TwoBkptSegRegEstimator"]
