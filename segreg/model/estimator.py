"""
General code for statistical estimators.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import abc

import numpy as np

from segreg.model import regression


# TODO: add access to estimated params

class Estimator(object, metaclass=abc.ABCMeta):

    """
    Base class for estimators
    """

    def __init__(self, **kwargs):
        """
        """
        self._indep = None
        self._dep = None
        self._params = None
        self._residuals = None
        self._rss = None
        self._loglikelihood = None
        self._is_estimated = False

    def _clear(self):
        self._indep = None
        self._dep = None
        self._params = None
        self._residuals = None
        self._rss = None
        self._loglikelihood = None
        self._is_estimated = False

    @abc.abstractmethod
    def num_params(self):
        pass

    @abc.abstractmethod
    def has_restricted_params(self):
        pass

    @abc.abstractmethod
    def get_func_for_params(self, *params):
        pass

    @abc.abstractmethod
    def estimation_func_val_at_estimate(self):
        pass

    @abc.abstractmethod
    def param_names(self):
        pass

    def estimated_params_indices(self):
        """
        In case the model has fixed params, this give indices in the params
        array of the non-fixed params.  That is, the params that are actually
        estimated.
        """
        return self._estimated_params_indices

    # TODO: have a "fast version" just for boot that only returns params
    def fit(self, indep, dep):
        """
        TODO
        """
        self._set_data(indep, dep)
        return self._estimate()

    @property
    def params(self):
        return self._params

    def get_func(self):
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        return self.get_func_for_params(self._params)

    def residuals(self):
        """
        Depends on data being set and estimation performed.
        """
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        func = self.get_func()
        self._residuals = self._dep - func(self._indep)

        return self._residuals

    def loglikelihood(self):
        """
        Computes loglikelihood at the MLE.
        """
        # TODO: check calling this before estimation is raising this exception
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        # TODO: could make num_obs this a class variable
        num_obs = len(self._indep)
        self._loglikelihood = regression.loglikelihood(num_obs, self._rss)

        return self._loglikelihood

    def rss(self):
        return self._rss

    def r_squared(self):
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        dep_mean = np.mean(self._dep)

        # orig
        #regression_vals = self._func(self._indep) - dep_mean
        func = self.get_func()

        regression_vals = func(self._indep) - dep_mean

        regression_sum_sq = np.vdot(regression_vals, regression_vals)

        tot_vals = self._dep - dep_mean
        tot_sum_sq = np.vdot(tot_vals, tot_vals)

        return regression_sum_sq / tot_sum_sq
