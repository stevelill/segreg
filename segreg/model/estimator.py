"""
General code for statistical estimators.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import abc

import numpy as np

from segreg.model import regression


class Estimator(object, metaclass=abc.ABCMeta):

    """
    Base class for estimators.
    
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
        """
        Returns the function defined by the given parameters.  
        
        That is, if the model is of the form:
        
        .. math::
            y = f(x; a_1, a_2, \dots, a_k) + \epsilon

        where :math:`f(x)` is a function of x depending on parameters 
        :math:`a_1, \dots, a_k`, then this returns the function

        .. math::
            f(x; a_1, \dots, a_k)
            
        where :math:`a_1, \dots, a_k` are passed to this function as `params`.
        
        See Also
        --------
        `get_func`
        
        Returns
        -------
        function
        """
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
        """
        Returns the fitted params.

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
        """
        Returns the function defined by the estimated parameters.  
        
        That is, if the model is of the form:
        
        .. math::
            y = f(x; a_1, a_2, \dots, a_k) + \epsilon

        where :math:`f(x)` is a function of x depending on parameters 
        :math:`a_1, \dots, a_k`, then this returns the function

        .. math::
            f(x; \widehat{a}_1, \dots, \widehat{a}_k)
            
        where :math:`\widehat{a}_1, \dots, \widehat{a}_k` are the estimated
        parameters from fitting the model to data.

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
        self._loglikelihood = regression.loglikelihood(num_obs, self._rss)

        return self._loglikelihood

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

        # orig
        #regression_vals = self._func(self._indep) - dep_mean
        func = self.get_func()

        regression_vals = func(self._indep) - dep_mean

        regression_sum_sq = np.vdot(regression_vals, regression_vals)

        tot_vals = self._dep - dep_mean
        tot_sum_sq = np.vdot(tot_vals, tot_vals)

        return regression_sum_sq / tot_sum_sq
