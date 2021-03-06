"""
Segmented Regression Estimators.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np

from segreg.model import one_bkpt_segreg
from segreg.model import two_bkpt_segreg
from segreg.model import segreg_util
from segreg.model.estimator import Estimator


class OneBkptSegRegEstimator(Estimator):

    """
    Estimator for one-bkpt segmented regression.

    This estimator is limited to univariate, continuous, linear, one-bkpt 
    segmented regression problems.  The model fitting estimates the parameters: 

        ``[u, v, m1, m2, sigma]``

    where

        ``(u,v)`` is the breakpoint (in x-y plane)

        ``m1`` is the slope of the left-hand segment

        ``m2`` is the slope of the right-hand segment

        ``sigma`` is the standard deviation of the residuals

    Examples
    --------
    >>> from segreg.model import OneBkptSegRegEstimator
    >>> indep = [1,2,3,4,5,6,7,8,9]
    >>> dep = [1,2,3,4,5,4,3,2,1]
    >>> estimator = OneBkptSegRegEstimator()
    >>> estimator.fit(indep, dep)
    array([ 5.,  5.,  1., -1.,  0.])


    Parameters
    ----------
    num_end_to_skip: int
        Number of data points to skip at each end of the data when solving for
        the bkpts.  As such, this determines a guaranteed minimum number of data 
        points in the left and right segments in the returned fit.
        If None, defaults to the underlying implementation.
        TODO: explain
    restrict_rhs_slope: float or None
        If not ``None``, will fix the rhs slope, ``m2``, to the given value.  As 
        such, the rhs slope ``m2`` will not be estimated when ``fit`` is called.
    no_bias_variance: bool
        If True, will modify the MLE estimate of the variance so that it is
        unbiased. 

    See Also
    --------
    OLSRegressionEstimator
    TwoBkptSegRegEstimator
    """

    def __init__(self,
                 num_end_to_skip=None,
                 restrict_rhs_slope=None,
                 no_bias_variance=False):
        super().__init__()

        # TODO: check and restrict
        self._num_end_to_skip = num_end_to_skip
        self._restrict_rhs_slope = restrict_rhs_slope
        self._no_bias_variance = no_bias_variance

        # TODO: make this better
        # keep noise variance as param
        self._num_params = 5
        self._fixed_params_indices = []
        if self._restrict_rhs_slope is not None:
            self._fixed_params_indices = [3]
        self._estimated_params_indices = np.setdiff1d(np.arange(self._num_params),
                                                      self._fixed_params_indices)

    def fit(self, indep, dep):
        """
        Fit the model to the given data.

        Parameters
        ----------
        indep: array-like of shape (num_data,)
            The independent data.  Also called predictor, explanatory variable,
            regressor, or exogenous variable.
        dep: array-like of shape (num_data,)
            The dependent data.  Also called response, regressand, or endogenous
            variable.

        Returns
        -------
        params: array of shape (num_params,)
            The estimated parameters.  The returned parameters are, in order,
            [u, v, m1, m2, sigma].
        """
        # some cython methods only accept double array arguments
        indep_to_use = np.asarray(indep, dtype=float)
        dep_to_use = np.asarray(dep, dtype=float)

        self._set_data(indep_to_use, dep_to_use)
        return self._estimate()

    @property
    def params(self):
        """
        Returns the fitted parameters.

        The returned parameters are, in order,

            [u, v, m1, m2, sigma]
        """
        return super().params
    params.__doc__ += Estimator.params.__doc__

    @property
    def num_params(self):
        return self._num_params

    @property
    def has_restricted_params(self):
        # TODO change later this impl
        return self._restrict_rhs_slope is not None

    @property
    def estimated_params_indices(self):
        return self._estimated_params_indices

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
            num_primary_params = len(self.estimated_params_indices) - 1
            variance = self._rss / (num_obs - num_primary_params)

        resid_stddev = np.sqrt(variance)

        # TODO: maybe not do this?  keep resid_stddev separate?
        self._params.append(resid_stddev)
        self._params = np.array(self._params)

        self._is_estimated = True

        return self._params

    def get_func_for_params(self, params):
        """
        Returns the regression model function defined by the given parameters.

        Parameters
        ----------
        params: array-like
            First four elements should be: [u, v, m1, m2, ...].  Any further 
            elements in ``params`` are ignored.  If ``restrict_rhs_slope`` has
            been set, then ``m2`` will be ignored here.

        """
        u = params[0]
        v = params[1]
        m1 = params[2]

        if self._restrict_rhs_slope is None:
            m2 = params[3]
        else:
            m2 = self._restrict_rhs_slope

        return segreg_util.one_bkpt_segmented_func(u, v, m1, m2)
    get_func_for_params.__doc__ += Estimator.get_func_for_params.__doc__

    @property
    def param_names(self):

        result = ["u", "v", "m1", "m2", "sigma"]

        return result

    def _argmin_sum_squares(self):
        """
        Main routine that calls cython-derived code.
        """

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
    Estimator for two-bkpt segmented regression.

    This estimator is limited to univariate, continuous, linear, two-bkpt 
    segmented regression problems.  The model fitting estimates the parameters: 

    ``[u1, v1, u2, v2, m1, m2, sigma]``

    where

        ``(u1,v1), (u2, v2)`` are the breakpoints (in x-y plane), ordered such
        that ``u1 < u2``

        ``m1`` is the slope of the left-most segment

        ``m2`` is the slope of the right-most segment

        ``sigma`` is the standard deviation of the residuals

    Notes
    -----
    The slope of the middle segment of the two-bkpt model does not appear as a 
    parameter since it is implied by the parameters ``(u1, v1)`` and 
    ``(u2, v2)``.

    Parameters
    ----------
    num_end_to_skip: int
        Number of data points to skip at each end of the data when solving for
        the bkpts.  As such, this determines a guaranteed minimum number of data 
        points in the left-most and right-most segments in the returned fit.
        If None, defaults to the underlying implementation.
        TODO: explain
    num_between_to_skip: int
        Number of data points to skip between the two bkpts (ie: the middle
        segment) when solving for the bkpts.  Specifically, for each choice of
        left bkpt ``u1``, will skip this many data points between ``u1`` and
        ``u2``.  As such, this determines a guaranteed minimum number of data 
        points between the bkpts in the returned fit.

    Examples
    --------
    >>> from segreg.model import TwoBkptSegRegEstimator
    >>> indep = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    >>> dep = [1,2,3,4,5,4,3,2,1,0,1,2,3,4]
    >>> estimator = TwoBkptSegRegEstimator()
    >>> estimator.fit(indep, dep)
    array([ 5.,  5., 10., -0.,  1.,  1.,  0.])

    See Also
    --------
    OLSRegressionEstimator
    OneBkptSegRegEstimator
    """

    def __init__(self,
                 num_end_to_skip=None,
                 num_between_to_skip=None,
                 no_bias_variance=False):

        super().__init__()

        self._num_end_to_skip = num_end_to_skip
        self._num_between_to_skip = num_between_to_skip
        self._no_bias_variance = no_bias_variance

        # TODO: make this better
        self._num_params = 7
        self._fixed_params_indices = []
        self._estimated_params_indices = np.setdiff1d(np.arange(self._num_params),
                                                      self._fixed_params_indices)

    @property
    def num_params(self):
        return self._num_params

    @property
    def has_restricted_params(self):
        # TODO change later this impl
        return len(self._fixed_params_indices) > 0

    @property
    def estimated_params_indices(self):
        return self._estimated_params_indices

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

        # for straight-line data, the fitted rss can sometimes be negative,
        # due to noise in the computations
        if abs(est_value) < 1.0e-13:
            est_value = 0.0

        self._rss = est_value

        # TODO: mle way or non-bias way here?
        num_obs = len(self._indep)
        variance = self._rss / num_obs

        if self._no_bias_variance:
            # subtract one representing the resid stddev itself
            num_primary_params = len(self.estimated_params_indices) - 1
            variance = self._rss / (num_obs - num_primary_params)

        resid_stddev = np.sqrt(variance)

        self._params.append(resid_stddev)
        self._params = np.array(self._params)

        self._is_estimated = True

        return self._params

    def fit(self, indep, dep):
        """
        Fit the model to the given data.

        Parameters
        ----------
        indep: array-like of shape (num_data,)
            The independent data.  Also called predictor, explanatory variable,
            regressor, or exogenous variable.
        dep: array-like of shape (num_data,)
            The dependent data.  Also called response, regressand, or endogenous
            variable.

        Returns
        -------
        params: array of shape (num_params,)
            The estimated parameters.  The returned parameters are, in order,

            [u1, v1, u2, v2, m1, m2, sigma].
        """
        # some cython methods only accept double array arguments
        indep_to_use = np.asarray(indep, dtype=float)
        dep_to_use = np.asarray(dep, dtype=float)

        self._set_data(indep_to_use, dep_to_use)
        return self._estimate()

    @property
    def params(self):
        """
        Returns the fitted parameters.

        The returned parameters are, in order,

            [u1, v1, u2, v2, m1, m2, sigma]
        """
        return super().params
    params.__doc__ += Estimator.params.__doc__

    def get_func_for_params(self, params):
        """
        Returns the regression model function defined by the given parameters.

        Parameters
        ----------
        params: array-like
            First six elements should be: [u1, v1, u2, v2, m1, m2, ...].  Any 
            further elements in ``params`` are ignored.

        """
        u1 = params[0]
        v1 = params[1]
        u2 = params[2]
        v2 = params[3]
        m1 = params[4]
        m2 = params[5]

        return segreg_util.two_bkpt_segmented_func(u1, v1, u2, v2, m1, m2)
    get_func_for_params.__doc__ += Estimator.get_func_for_params.__doc__

    @property
    def param_names(self):

        result = ["u1", "v1", "u2", "v2", "m1", "m2", "sigma"]

        return result

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
