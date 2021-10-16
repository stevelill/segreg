"""
Segmented Regression Estimators.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np

from segreg.model import one_bkpt_segreg
from segreg.model import two_bkpt_segreg
from segreg.model.estimator import Estimator


class OneBkptSegRegEstimator(Estimator):

    """
    classdocs
    """

    def __init__(self, **kwargs):
        self._restrict_rhs_slope = kwargs.pop('restrict_rhs_slope', None)
        self._num_end_to_skip = kwargs.pop('num_end_to_skip', None)
        self._no_bias_variance = kwargs.pop('no_bias_variance', False)

        # TODO: make this better
        # keep noise variance as param
        self._num_params = 5
        self._fixed_params_indices = []
        if self._restrict_rhs_slope is not None:
            self._fixed_params_indices = [3]
        self._estimated_params_indices = np.setdiff1d(np.arange(self._num_params),
                                                      self._fixed_params_indices)

    ##########################################################################
    # OVERRIDE Estimator
    ##########################################################################

    @property
    def num_params(self):
        return self._num_params

    def has_restricted_params(self):
        # TODO change later this impl
        return self._restrict_rhs_slope is not None

    def _set_data(self, indep, dep):
        self._clear()

        # TODO: put this in cython for faster ?
        # we sort the data
        argsort_for_indep = indep.argsort()
        self._indep = indep[argsort_for_indep]
        self._dep = dep[argsort_for_indep]

    def _estimate(self):

        est_params, est_value = self._argmin_sum_squares()

        u = est_params[0]
        v = est_params[1]
        m1 = est_params[2]

        if self._restrict_rhs_slope is None:
            m2 = est_params[3]
        else:
            m2 = self._restrict_rhs_slope

        self._params = [u, v, m1, m2]

        self._rss = est_value

        # rss = np.vdot(self._residuals, self._residuals)
        # TODO: mle way or non-bias way here?
        num_obs = len(self._indep)
        variance = self._rss / num_obs

        if self._no_bias_variance:
            # subtract one representing the resid stddev itself
            num_primary_params = len(self.estimated_params_indices()) - 1
            variance = self._rss / (num_obs - num_primary_params)

        resid_stddev = np.sqrt(variance)

        # TODO: maybe not do this?  keep resid_stddev separate?
        self._params.append(resid_stddev)
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

    def get_func_for_params(self, params):
        u = params[0]
        v = params[1]
        m1 = params[2]

        if self._restrict_rhs_slope is None:
            m2 = params[3]
        else:
            m2 = self._restrict_rhs_slope

        return one_bkpt_segreg.segmented_func(u, v, m1, m2)

    # TODO: static?
    def param_names(self, **kwargs):
        latex = kwargs.pop('latex', False)

        if latex:
            result = ["$u$", "$v$", "$m_1$", "$m_2$", "$\sigma$"]
        else:
            result = ["u", "v", "m1", "m2", "sigma"]

        return result

    ##########################################################################
    # IMPL
    ##########################################################################

    def _argmin_sum_squares(self):

        # TODO: can we pass this bool as arg to the function and get rid of
        # this if/else ???
        if self._num_end_to_skip is not None:
            result = one_bkpt_segreg.estimate_one_bkpt_segreg(self._indep,
                                                              self._dep,
                                                              num_end_to_skip=self._num_end_to_skip,
                                                              m2=self._restrict_rhs_slope)
        else:
            result = one_bkpt_segreg.estimate_one_bkpt_segreg(self._indep,
                                                              self._dep,
                                                              m2=self._restrict_rhs_slope)

        return result


class TwoBkptSegRegEstimator(Estimator):

    """
    classdocs
    """

    def __init__(self, **kwargs):
        self._num_end_to_skip = kwargs.pop('num_end_to_skip', None)
        self._num_between_to_skip = kwargs.pop('num_between_to_skip', 5)
        self._no_bias_variance = kwargs.pop('no_bias_variance', False)

        # TODO: make this better
        self._num_params = 7
        self._fixed_params_indices = []
        self._estimated_params_indices = np.setdiff1d(np.arange(self._num_params),
                                                      self._fixed_params_indices)

    ##########################################################################
    # OVERRIDE Estimator
    ##########################################################################

    @property
    def num_params(self):
        return self._num_params

    def has_restricted_params(self):
        # TODO change later this impl
        return len(self._fixed_params_indices) > 0

    def _set_data(self, indep, dep):
        self._clear()

        # TODO: put this in cython for faster
        # we sort the data
        argsort_for_indep = indep.argsort()
        self._indep = indep[argsort_for_indep]
        self._dep = dep[argsort_for_indep]

    def _estimate(self):
        est_params, est_value = self._argmin_sum_squares()

        u1 = est_params[0]
        v1 = est_params[1]
        u2 = est_params[2]
        v2 = est_params[3]
        m1 = est_params[4]
        m2 = est_params[5]

        self._params = [u1, v1, u2, v2, m1, m2]

        self._rss = est_value

        # TODO: mle way or non-bias way here?
        num_obs = len(self._indep)
        variance = self._rss / num_obs

        if self._no_bias_variance:
            # subtract one representing the resid stddev itself
            num_primary_params = len(self.estimated_params_indices()) - 1
            variance = self._rss / (num_obs - num_primary_params)

        resid_stddev = np.sqrt(variance)

        self._params.append(resid_stddev)
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

    def get_func_for_params(self, params):
        u1 = params[0]
        v1 = params[1]
        u2 = params[2]
        v2 = params[3]
        m1 = params[4]
        m2 = params[5]
        # TODO: fix this
        # don't use last param which was appended resid stddev
        # return two_bkpt_segreg.segmented_func(params[0:-1])
        return two_bkpt_segreg.segmented_func([u1, v1, u2, v2, m1, m2])

    # TODO: static?
    def param_names(self, **kwargs):
        latex = kwargs.pop('latex', False)

        if latex:
            result = ["$u1$", "$v1$", "$u2$",
                      "$v2$", "$m_1$", "$m_2$", "$\sigma$"]
        else:
            result = ["u1", "v1", "u2", "v2", "m1", "m2", "sigma"]

        return result

    ##########################################################################
    # IMPL
    ##########################################################################

    def _argmin_sum_squares(self):

        if self._num_end_to_skip is not None:
            est_params, est_value = two_bkpt_segreg.estimate_two_bkpt_segreg(self._indep,
                                                                             self._dep,
                                                                             num_end_to_skip=self._num_end_to_skip,
                                                                             num_between_to_skip=self._num_between_to_skip)
        else:
            est_params, est_value = two_bkpt_segreg.estimate_two_bkpt_segreg(self._indep,
                                                                             self._dep)

        return est_params, est_value
