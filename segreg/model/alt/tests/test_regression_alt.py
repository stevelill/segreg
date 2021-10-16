"""
Testing alternative regression.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import unittest

import scipy.linalg

import numpy as np

from segreg import data
from segreg.data import _testing_util
from segreg.model import one_bkpt_segreg
from segreg.model.alt import regression_alt, likelihood_util


class TestRegressionAlt(unittest.TestCase):

    def setUp(self):
        num_data = 100
        x_min = 0
        x_max = 100

        u = 20
        v = 10
        m1 = -0.05
        m2 = 0.4
        func = one_bkpt_segreg.segmented_func(u, v, m1, m2)
        stddev = 2.0

        seed = 123456789
        self._indep, self._dep = _testing_util.generate_fake_data_normal_errors(num_data,
                                                                                x_min,
                                                                                x_max,
                                                                                func,
                                                                                stddev,
                                                                                seed=seed)

        seed = 3403947
        fake_data = _testing_util.generate_fake_data(num_data, x_min, x_max)
        self._indep2 = np.stack((self._indep, fake_data), axis=1)

    def test_invert_two_by_two(self):
        a = 1.0
        b = 2.0
        c = 3.0
        d = 4.0

        e = 2.5
        f = 3.5

        mat = np.array([[a, b], [c, d]])
        vec = np.array([[e], [f]])

        expected = scipy.linalg.solve(mat, vec)

        computed = regression_alt.invert_two_by_two(a, b, c, d, e, f)

        self.assertEqual(expected[0], computed[0])
        self.assertEqual(expected[1], computed[1])

    def test_regression_tieout_1d(self):
        """
        Tie out matrix_ols and ols using scipy.linalg.lstsq.
        """
        tol = 1.0e-12

        # this beta is array
        intercept, beta = regression_alt.matrix_ols(
            self._indep, self._dep)
        # here, we know beta has only one element
        beta_result = beta[0]

        py_intercept, py_beta = regression_alt.bare_bones_ols(
            self._indep, self._dep)

        computed = np.array([intercept, beta_result])
        expected = np.array([py_intercept, py_beta])
        close = np.allclose(computed, expected, rtol=0.0, atol=tol)

        self.assertTrue(close)

    def test_regression_tieout_2d(self):
        """
        Tie out matrix_ols and ols using scipy.linalg.lstsq.
        """
        tol = 1.0e-12

        intercept, beta = regression_alt.matrix_ols(
            self._indep2, self._dep)

        py_intercept, py_beta = regression_alt.bare_bones_ols(
            self._indep2, self._dep)

        self.assertAlmostEqual(intercept, py_intercept, delta=tol)

        computed = np.array([beta])
        expected = np.array([py_beta])
        close = np.allclose(computed, expected, rtol=0.0, atol=tol)

        self.assertTrue(close)

    def test_ls_rss(self):
        """
        Tie out two impls with rss.
        """
        tol = 1.0e-10

        # this beta is array
        (intercept,
         slope,
         rss,
         ols_data) = regression_alt.ols_verbose(
            self._indep, self._dep)

        (intercept2,
         slope2,
         rss2) = regression_alt.ols_with_rss(
            self._indep, self._dep)

        computed = np.array([intercept, slope, rss])
        expected = np.array([intercept2, slope2, rss2])
        close = np.allclose(computed, expected, rtol=0.0, atol=tol)

        self.assertTrue(close)

    def test_ols(self):
        """
        foo
        """
        tol = 1.0e-14

        indep = np.array([0, 1, 2], dtype=float)
        dep = np.array([0, 1, 0], dtype=float)

        # this beta is array
        (intercept,
         slope,
         rss,
         ols_data) = regression_alt.ols_verbose(indep, dep)

        computed = np.array([intercept, slope, rss])
        expected = np.array([1.0 / 3.0, 0.0, 2.0 / 3.0])
        close = np.allclose(computed, expected, rtol=0.0, atol=tol)
        self.assertTrue(close)

    def test_ols_loglik(self):
        indep, dep = data.test1()

        intercept, slope, rss, ols_data = regression_alt.ols_verbose(indep, dep)
        num_data = ols_data[0]
        resid_variance = rss / num_data

        expected = likelihood_util.loglikelihood(rss=rss,
                                                 resid_variance=resid_variance,
                                                 num_data=num_data)

        computed = regression_alt.ols_loglik([intercept, slope, resid_variance],
                                             indep, dep)

        self.assertAlmostEqual(expected, computed, delta=1.0e-11)


if __name__ == "__main__":
    unittest.main()
