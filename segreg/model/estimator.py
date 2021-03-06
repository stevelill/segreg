"""
General code for statistical regression estimators.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import abc

import numpy as np


def _loglikelihood(num_data, rss):
    """
    Loglikelihood evaluated at the MLE.

    Assumes the residual errors are normally distributed i.i.d. with zero mean 
    and constant variance.  This is valid for any regression model of the form: 
        y = f(x) + e
    where
        e ~ N(0, v)

    Parameters
    ----------
    num_data: int
    rss: float
        The residual sum of squares.

    Returns
    -------
    loglikelihood: float
    """
    return -0.5 * num_data * (np.log(2.0 * np.pi)
                              - np.log(num_data)
                              + 1.0
                              + np.log(rss))


class Estimator(object, metaclass=abc.ABCMeta):

    """
    Base class for estimators.

    This class is geared to estimation problems of the form:

        y_i = f(x_i; theta) + epsilon_i

    where f(x) is generally a non-linear function of x, defined by a vector
    theta of parameters, and {epsilon_i} are independently and identically
    distributed errors with mean zero and constant variance. For given data
    {x_i, y_i} (i=1,...,n), the parameters theta are determined by means of
    least-squares estimation.

    TODO: define special exception for not fitted
    """

    def __init__(self):
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

    @property
    @abc.abstractmethod
    def num_params(self):
        """
        Number of model parameters.

        Includes the residual variance.

        Returns
        -------
        num_params: int
        """
        pass

    @property
    @abc.abstractmethod
    def has_restricted_params(self):
        """
        Whether there are any model parameters set to a fixed value.

        Returns
        -------
        has_restricted_params: bool
        """
        pass

    @abc.abstractmethod
    def get_func_for_params(self, *params):
        r"""
        More generally, if the model is of the form:

        .. math::
            y = f(x; \theta) + \varepsilon

        where :math:`f(x)` is a function of x depending on the parameter vector 
        :math:`\theta`, then this returns the function

        .. math::
            f(x; \theta)

        where :math:`\theta` is passed to this function as ``params``.

        See Also
        --------
        model_function

        Returns
        -------
        func_for_params: function object
        """
        pass

    @property
    @abc.abstractmethod
    def param_names(self):
        """
        Names of the parameters.

        Returns
        -------
        param_names: array-like of shape (``num_params``,)
        """
        pass

    @property
    @abc.abstractmethod
    def estimated_params_indices(self):
        """
        Indices in the parameter array of the fitted parameters.

        This is only useful when there are fixed parameters in the model.

        Returns
        -------
        estimated_params_indices: array-like of shape (num fitted,)
        """
        pass

    @abc.abstractmethod
    def fit(self, indep, dep):
        """
        Fit the model to the given data.

        Parameters
        ----------
        indep: array-like
            The independent data.  Also called predictor, explanatory variable,
            regressor, or exogenous variable.
        dep: array-like
            The dependent data.  Also called response, regressand, or endogenous
            variable.
        """
        pass

    @property
    def params(self):
        r"""
        More generally, if the (fitted) model is of the form:

        .. math::
            y_i = f(x_i; \theta) + \varepsilon_i

        where :math:`f(x)` is a function of x depending on the parameter vector 
        :math:`\theta`, and :math:`\{\varepsilon_i\}` is i.i.d. random error of 
        mean zero and variance :math:`\sigma^2`, then the returned parameter
        vector is of the form:

        .. math::
            [\theta_1, \dots, \theta_k, \sigma] 

        where :math:`\theta = [\theta_1, \dots, \theta_k]`.

        Notes
        -----
        The error variance :math:`\sigma^2` is often considered a nuisance
        parameter, and is not required for many applications.


        Raises
        ------
        Exception
            If ``fit`` has not been called.

        Returns
        -------
        params: array-like
        """
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        return self._params

    @property
    def model_function(self):
        r"""
        Returns the regression model function defined by the estimated 
        parameters.  

        That is, if the model is of the form:

        .. math::
            y = f(x; \theta) + \varepsilon

        where :math:`f(x)` is a function of x depending on the parameter vector 
        :math:`\theta`, then this returns the function

        .. math::
            f(x; \widehat{\theta})

        where :math:`\widehat{\theta}` is the estimated parameter vector from 
        fitting the model to data.

        See Also
        --------
        get_func_for_params

        Raises
        ------
        Exception
            If ``fit`` has not been called.

        Returns
        -------
        model_function: function object
        """
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        return self.get_func_for_params(self._params)

    @property
    def residuals(self):
        """
        Returns the residuals from the fit.

        Raises
        ------
        Exception
            If ``fit`` has not been called.

        Returns
        -------
        residuals: array-like
        """
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        func = self.model_function
        self._residuals = self._dep - func(self._indep)

        return self._residuals

    @property
    def loglikelihood(self):
        """
        Computes loglikelihood at the MLE (maximum likelihood estimate).

        This assumes the residual errors are normally distributed i.i.d. with
        zero mean and constant variance.

        Raises
        ------
        Exception
            If ``fit`` has not been called.

        Returns
        -------
        loglikelihood: float
        """
        # TODO: check calling this before estimation is raising this exception
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        # TODO: could make num_obs this a class variable
        num_obs = len(self._indep)
        self._loglikelihood = _loglikelihood(num_obs, self._rss)

        return self._loglikelihood

    @property
    def rss(self):
        """
        Residual sum of squares of the fit.

        Raises
        ------
        Exception
            If ``fit`` has not been called.

        Returns
        -------
        rss: float
        """
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        return self._rss

    @property
    def r_squared(self):
        """
        R-squared of the fit.

        Raises
        ------
        Exception
            If ``fit`` has not been called.

        Returns
        -------
        r_squared: float
        """
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        dep_mean = np.mean(self._dep)

        func = self.model_function

        regression_vals = func(self._indep) - dep_mean

        regression_sum_sq = np.vdot(regression_vals, regression_vals)

        tot_vals = self._dep - dep_mean
        tot_sum_sq = np.vdot(tot_vals, tot_vals)

        return regression_sum_sq / tot_sum_sq
