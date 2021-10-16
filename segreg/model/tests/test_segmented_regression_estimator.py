"""
Testing OneBkptSegRegEstimator.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest

import numpy as np

from segreg.model import regression
from segreg.model import one_bkpt_segreg
from segreg.model import OneBkptSegRegEstimator
from segreg.data import _testing_util


class TestOneBkptSegRegEstimator(unittest.TestCase):

    def setUp(self):
        self._num_data = 100
        self._x_min = 0
        self._x_max = 100

        self._u = 20
        self._v = 10
        self._m1 = -0.05
        self._m2 = 0.4
        self._func = one_bkpt_segreg.segmented_func(self._u,
                                                    self._v,
                                                    self._m1,
                                                    self._m2)
        self._stddev = 2.0

        seed = 32402304
        self._indep, self._dep = _testing_util.generate_fake_data_normal_errors(self._num_data,
                                                                                self._x_min,
                                                                                self._x_max,
                                                                                self._func,
                                                                                self._stddev,
                                                                                seed=seed)

        self._segreg = OneBkptSegRegEstimator()
        self._segreg.fit(self._indep, self._dep)

    def test_estimate(self):
        computed_est_params = self._segreg.fit(self._indep, self._dep)
        computed_est_params = np.array(computed_est_params)

        expected_est_params = np.array([26.63732814126465,
                                        11.626933181693003,
                                        0.06626623424297813,
                                        0.42653383431718306,
                                        1.8826792196921438])
        close = np.allclose(expected_est_params, computed_est_params)
        self.assertTrue(close)
        ##########
        stddev = 3.0
        seed = 546543
        (indep,
         dep) = _testing_util.generate_fake_data_normal_errors(self._num_data,
                                                               self._x_min,
                                                               self._x_max,
                                                               self._func,
                                                               stddev,
                                                               seed=seed)

        computed_est_params = self._segreg.fit(indep, dep)
        computed_est_params = np.array(computed_est_params)

        expected_est_params = np.array([25.322690022950244,
                                        12.050685330570786,
                                        0.050857833019324,
                                        0.4098067211721176,
                                        2.6631593422221491])

        close = np.allclose(expected_est_params, computed_est_params)
        self.assertTrue(close)

    def test_estimation_func_val_at_estimate(self):
        computed_func_val = self._segreg.estimation_func_val_at_estimate()

        # subtract rhs rss bit by bit
        #expected_func_val = 354.4481044261281

        # rhs rss subtracted from total
        expected_func_val = 354.44810442605524

        self.assertAlmostEqual(expected_func_val,
                               computed_func_val,
                               delta=1.0e-14)

        # compute by hand -- this matches with less precision
        residuals = self._segreg.residuals()
        rss = np.vdot(residuals, residuals)

        self.assertAlmostEqual(rss, computed_func_val, places=9)

    def test_func(self):
        computed_func = self._segreg.get_func()

        est_params = self._segreg.fit(self._indep, self._dep)

        def expected_func(x):
            return np.piecewise(x,
                                [x <= est_params[0],
                                 x > est_params[0]],
                                [lambda x: est_params[1] + est_params[2] * (x - est_params[0]),
                                 lambda x: est_params[1] + est_params[3] * (x - est_params[0])])

        expected_func_vals = expected_func(self._indep)

        computed_func_vals = computed_func(self._indep)

        close = np.allclose(expected_func_vals, computed_func_vals)
        self.assertTrue(close)

    def test_func_for_params(self):
        params = [self._u, self._v, self._m1, self._m2]
        computed_func = self._segreg.get_func_for_params(params)

        def expected_func(x):
            return np.piecewise(x,
                                [x <= params[0],
                                 x > params[0]],
                                [lambda x: params[1] + params[2] * (x - params[0]),
                                 lambda x: params[1] + params[3] * (x - params[0])])

        expected_func_vals = expected_func(self._indep)

        computed_func_vals = computed_func(self._indep)

        close = np.allclose(expected_func_vals, computed_func_vals)
        self.assertTrue(close)

    def test_residuals(self):
        computed_residuals = self._segreg.residuals()

        func = self._segreg.get_func()
        expected_residuals = self._dep - func(self._indep)

        close = np.allclose(expected_residuals, computed_residuals)
        self.assertTrue(close)

    def test_loglikelihood(self):
        computed_loglikelihood = self._segreg.loglikelihood()

        rss = self._segreg.estimation_func_val_at_estimate()
        expected_loglikelihood = regression.loglikelihood(self._num_data, rss)

        self.assertAlmostEqual(expected_loglikelihood,
                               computed_loglikelihood,
                               places=15)

    def test_r_squared(self):
        computed_r_squared = self._segreg.r_squared()

        dep_mean = np.mean(self._dep)
        func = self._segreg.get_func()

        regression_vals = func(self._indep) - dep_mean
        regression_sum_sq = np.vdot(regression_vals, regression_vals)

        tot_vals = self._dep - dep_mean
        tot_sum_sq = np.vdot(tot_vals, tot_vals)

        expected_r_squared = regression_sum_sq / tot_sum_sq

        self.assertAlmostEqual(expected_r_squared,
                               computed_r_squared,
                               places=12)


if __name__ == "__main__":
    unittest.main()
