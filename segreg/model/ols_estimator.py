"""
Ordinary Least Squares Estimator.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import numpy as np

from segreg.model import regression
from segreg.model.estimator import Estimator


class OLSRegressionEstimator(Estimator):
    """
    Estimator for ordinary least-squares regression.

    This estimator is limited to univariate, linear, regression problems.  The 
    model fitting estimates the parameters: 

        ``[intercept, slope, sigma]``

    where the fitted line is defined by

        y = ``intercept`` + ``slope`` * x

    and ``sigma`` is the standard deviation of the residuals.

    Notes
    -----
    There are many standard python libraries for this type of OLS. This class is 
    provided as a convenience to implement the same interface as the estimators 
    for segmented regression.  Moreover, the underlying implementation has been 
    customized for the univariate regression problems for which this class is 
    limited, for the purpose of greater calculation speed.

    Examples
    --------
    >>> from segreg.model import OLSRegressionEstimator
    >>> indep = [1,2,3,4,5]
    >>> dep = [2,3,4,5,6]
    >>> estimator = OLSRegressionEstimator()
    >>> estimator.fit(indep, dep)
    array([1., 1., 0.])

    See Also
    --------
    OneBkptSegRegEstimator
    TwoBkptSegRegEstimator
    """

    def __init__(self):

        super().__init__()

        # we include estimate of residual variance as a parameter
        self._num_params = 3
        self._fixed_params_indices = []
        self._estimated_params_indices = np.setdiff1d(np.arange(self._num_params),
                                                      self._fixed_params_indices)

    def fit(self, indep, dep):
        """
        Fit the model to the given data.

        The fit automatically includes an intercept.  There is no need to add
        a column of ones to the ``indep`` input.

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
            [intercept, slope, sigma].
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

            [intercept, slope, sigma]
        """
        return super().params
    params.__doc__ += Estimator.params.__doc__

    @property
    def num_params(self):
        return self._num_params

    @property
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

    @property
    def estimated_params_indices(self):
        return self._estimated_params_indices

    def get_func_for_params(self, params):
        """
        Returns the regression model function defined by the given parameters.

        Parameters
        ----------
        params: array-like
            First element should be the ``intercept``, second element the 
            ``slope``.  Any further elements in ``params`` are ignored.

        """
        intercept = params[0]
        slope = params[1]

        def ols_func(x):
            return intercept + slope * x

        return ols_func
    get_func_for_params.__doc__ += Estimator.get_func_for_params.__doc__

    @property
    def param_names(self):

        result = ["intercept", "slope", "sigma"]

        return result
