"""
Unittest for OlsRegressionEstimator.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest

import numpy as np

from segreg.model import OLSRegressionEstimator, regression
from segreg.data import _testing_util


class TestOlsEstimator(unittest.TestCase):

    def setUp(self):
        self._num_data = 100
        self._x_min = 0
        self._x_max = 100

        self._intercept = 20
        self._slope = -0.05

        def func(x):
            return self._intercept + self._slope * x

        self._func = func
        self._stddev = 2.0

        seed = 32402304
        self._indep, self._dep = _testing_util.generate_fake_data_normal_errors(self._num_data,
                                                                                self._x_min,
                                                                                self._x_max,
                                                                                self._func,
                                                                                self._stddev,
                                                                                seed=seed)

        self._ols = OLSRegressionEstimator()

    def test_estimate(self):
        est_params = self._ols.fit(self._indep, self._dep)

        # before no-bias variance estimate
#        expected_est_params = np.array([19.616934487467692,
#                                        -0.042793152668902268,
#                                        1.939118953065025])

        expected_est_params = np.array([19.616934487467695,
                                        - 0.04279315266890229,
                                        1.9588059446280579])

        close = np.allclose(expected_est_params, est_params)
        self.assertTrue(close)

    def test_estimation_func_val_at_estimate(self):
        self._ols.fit(self._indep, self._dep)
        computed_func_val = self._ols.estimation_func_val_at_estimate()
        expected_func_val = 376.01823141359989
        self.assertAlmostEqual(expected_func_val, computed_func_val, places=10)

        # compute by hand -- this matches with less precision
        residuals = self._ols.residuals()
        rss = np.vdot(residuals, residuals)

        self.assertAlmostEqual(rss, computed_func_val, places=9)

    def test_get_func(self):
        est_params = self._ols.fit(self._indep, self._dep)

        def expected_func(x):
            return est_params[0] + est_params[1] * x

        computed_func = self._ols.get_func()

        expected_func_vals = expected_func(self._indep)
        computed_func_vals = computed_func(self._indep)

        close = np.allclose(expected_func_vals, computed_func_vals)
        self.assertTrue(close)

    def test_func_for_params(self):
        params = [12.0, 0.24]
        computed_func = self._ols.get_func_for_params(params)

        def expected_func(x):
            return params[0] + params[1] * x

        expected_func_vals = expected_func(self._indep)

        computed_func_vals = computed_func(self._indep)

        close = np.allclose(expected_func_vals, computed_func_vals)
        self.assertTrue(close)

    def test_residuals(self):
        self._ols.fit(self._indep, self._dep)
        computed_residuals = self._ols.residuals()

        func = self._ols.get_func()
        expected_residuals = self._dep - func(self._indep)

        close = np.allclose(expected_residuals, computed_residuals)
        self.assertTrue(close)

    def test_loglikelihood(self):
        self._ols.fit(self._indep, self._dep)
        computed_loglikelihood = self._ols.loglikelihood()

        residuals = self._ols.residuals()
        rss = np.vdot(residuals, residuals)

        expected_loglikelihood = regression.loglikelihood(self._num_data, rss)

        self.assertAlmostEqual(expected_loglikelihood,
                               computed_loglikelihood,
                               places=10)

        # TODO: add r sq to interface of Estimator?
#    def test_r_squared(self):
#        computed_r_squared = self._segreg.r_squared()
#
#        dep_mean = np.mean(self._dep)
#        func = self._segreg.get_func()
#
#        regression_vals = func(self._indep) - dep_mean
#        regression_sum_sq = np.vdot(regression_vals, regression_vals)
#
#        tot_vals = self._dep - dep_mean
#        tot_sum_sq = np.vdot(tot_vals, tot_vals)
#
#        expected_r_squared =  regression_sum_sq / tot_sum_sq
#
#        self.assertAlmostEqual(expected_r_squared,
#                               computed_r_squared,
#                               places=12)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
