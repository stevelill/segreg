"""
Ordinary Least Squares Estimator.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np

from segreg.model import regression
from segreg.model.estimator import Estimator


# TODO: frozen paradigm?
class OLSRegressionEstimator(Estimator):
    """
    classdocs
    """

    def __init__(self):

        # TODO: make this better
        # currently not using in OLS
        # we include estimate of residual variance as a parameter
        self._num_params = 3
        self._fixed_params_indices = []
        self._estimated_params_indices = np.setdiff1d(np.arange(self._num_params),
                                                      self._fixed_params_indices)

    ############################################################################
    # OVERRIDE Estimator
    ############################################################################

    @property
    def num_params(self):
        return self._num_params

    def has_restricted_params(self):
        return False

    def _set_data(self, indep, dep):
        self._clear()
        self._indep = indep
        self._dep = dep

    def _estimate(self):
        est_const, est_mat, rss = regression.ols_with_rss(self._indep,
                                                          self._dep)

        num_obs = len(self._indep)

        self._rss = rss
        # TODO: mle way or non-bias way here?
        #variance = self._rss / num_obs

        variance = self._rss / (num_obs - 2)
        resid_stddev = np.sqrt(variance)

        self._params = [est_const, est_mat, resid_stddev]
        self._params = np.array(self._params)
        self._is_estimated = True

        return self._params

    def estimation_func_val_at_estimate(self):
        """
        For regression model such as this, this gives RSS.
        """
        if not self._is_estimated:
            raise Exception("Need to call fit first")
        return self._rss

    # TODO: for multivariate
    def get_func_for_params(self, params):
        intercept = params[0]
        slope = params[1]

        # TODO: handle for multivariate
        def ols_func(x):
            return intercept + slope * x

        return ols_func

    def param_names(self, **kwargs):
        # return ["intercept", "slope"]

        latex = kwargs.pop('latex', False)

        if latex:
            result = ["$intercept$", "$slope$", "$\sigma$"]
        else:
            result = ["intercept", "slope", "sigma"]

        return result
