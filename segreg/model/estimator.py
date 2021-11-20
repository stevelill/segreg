"""
General code for statistical estimators.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import abc

import numpy as np

# TODO: make more properties


def loglikelihood(num_data, rss):
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

    @abc.abstractmethod
    def num_params(self):
        """
        Number of model parameters.

        Includes the residual variance.

        Returns
        -------
        int
        """
        pass

    @abc.abstractmethod
    def has_restricted_params(self):
        """
        Returns
        -------
        bool
        """
        pass

    @abc.abstractmethod
    def get_func_for_params(self, *params):
        r"""
        Returns the function defined by the given parameters.  

        That is, if the model is of the form:

        .. math::
            y = f(x; \theta) + \varepsilon

        where :math:`f(x)` is a function of x depending on the parameter vector 
        :math:`\theta`, then this returns the function

        .. math::
            f(x; \theta)

        where :math:`\theta` is passed to this function as `params`.

        See Also
        --------
        `get_func`

        Returns
        -------
        function
        """
        pass

    @abc.abstractmethod
    def param_names(self):
        pass

    def estimated_params_indices(self):
        """
        In case the model has fixed parameters, this give indices in the 
        parameter array of the non-fixed parameters.  That is, the parameters 
        that are actually estimated.
        """
        return self._estimated_params_indices

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
        self._set_data(indep, dep)
        return self._estimate()

    @property
    def params(self):
        r"""
        Returns the fitted parameters.

        If the model is of the form:

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
        parameter.


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

    def get_func(self):
        r"""
        Returns the function defined by the estimated parameters.  

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
        `get_func_for_params`

        Raises
        ------
        Exception
            If ``fit`` has not been called.

        Returns
        -------
        function object
        """
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        return self.get_func_for_params(self._params)

    def residuals(self):
        """
        Returns the residuals from a fit of the model to data.

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

        func = self.get_func()
        self._residuals = self._dep - func(self._indep)

        return self._residuals

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
        float
        """
        # TODO: check calling this before estimation is raising this exception
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        # TODO: could make num_obs this a class variable
        num_obs = len(self._indep)
        self._loglikelihood = loglikelihood(num_obs, self._rss)

        return self._loglikelihood

    @property
    def rss(self):
        """
        Residual sum of squares from the fit.

        Raises
        ------
        Exception
            If ``fit`` has not been called.

        Returns
        -------
        float
        """
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        return self._rss

    # TODO: make property
    def r_squared(self):
        """
        Computes R-squared of the fit.

        Raises
        ------
        Exception
            If ``fit`` has not been called.

        Returns
        -------
        float
        """
        if not self._is_estimated:
            raise Exception("Need to call fit first")

        dep_mean = np.mean(self._dep)

        func = self.get_func()

        regression_vals = func(self._indep) - dep_mean

        regression_sum_sq = np.vdot(regression_vals, regression_vals)

        tot_vals = self._dep - dep_mean
        tot_sum_sq = np.vdot(tot_vals, tot_vals)

        return regression_sum_sq / tot_sum_sq
