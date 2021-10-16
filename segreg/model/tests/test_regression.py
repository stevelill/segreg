"""
Testing regression.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest

import numpy as np

from segreg.model import one_bkpt_segreg
from segreg.model import regression
from segreg.model.alt import regression_alt
from segreg.data import _testing_util


class TestRegression(unittest.TestCase):

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
        ##
        self._x_arr = np.array([1, 2, 3, 4])
        self._y_arr = np.array([2, 3, 4, 5])
        num = 4
        sum_x = 10
        sum_y = 14
        sum_xx = 30
        sum_yy = 54
        sum_xy = 40
        self._ols_terms = np.array([num,
                                    sum_x,
                                    sum_y,
                                    sum_xx,
                                    sum_yy,
                                    sum_xy],
                                   dtype=float)

    def test_regression_tieout_fixed_slope(self):
        """
        Tie out cython with all-python version.
        """
        fixed_slopes = [-1.2, 0.0, 3.0]
        tol = 1.0e-12

        for fixed_slope in fixed_slopes:
            intercept, slope, rss = regression.ols_with_rss(
                self._indep, self._dep, fixed_slope)

            ru_intercept = regression_alt.ols_fixed_slope(
                self._indep, self._dep, slope)
            self.assertAlmostEqual(intercept, ru_intercept, delta=tol)

    def test_regression_tieout(self):
        """
        Tie out cython with all-python version.
        """
        tol = 1.0e-12

        intercept, slope, rss = regression.ols_with_rss(self._indep, self._dep)

        py_intercept, py_slope, py_rss = regression_alt.fast_ols_with_rss(
            self._indep, self._dep)

        py_intercept2, py_slope2 = regression_alt.fast_ols(
            self._indep, self._dep)

        computed = np.array([intercept, slope, rss])
        expected = np.array([py_intercept, py_slope, py_rss])
        close = np.allclose(computed, expected)

        self.assertTrue(close)

        close2 = np.allclose([intercept, slope], [py_intercept2, py_slope2])
        self.assertTrue(close2)

    # TODO: deprecate: test tieout of fast ols with statsmodels elsewhere
    def test_regression_tieout2(self):
        intercept, slope, rss = regression.ols_with_rss(self._indep, self._dep)

        (sm_intercept,
         sm_slope) = regression_alt.statsmodels_ols(self._indep,
                                                    self._dep)

        tol = 1.0e-10

        self.assertAlmostEqual(intercept, sm_intercept, delta=tol)
        self.assertAlmostEqual(slope, sm_slope, delta=tol)

        (ru_intercept,
         ru_slope,
         ru_rss) = regression_alt.fast_ols_with_rss(self._indep, self._dep)

        self.assertAlmostEqual(intercept, ru_intercept, delta=tol)
        self.assertAlmostEqual(slope, ru_slope, delta=tol)
        self.assertAlmostEqual(rss, ru_rss, delta=tol)

    ##########################################################################
    # ols_terms
    ##########################################################################
    def test_ols_terms(self):
        expected = self._ols_terms

        computed = regression_alt.ols_terms(self._x_arr, self._y_arr)

        close = np.allclose(expected, computed)
        self.assertTrue(close)

        ##########
        computed_ols_terms = regression_alt.ols_terms(
            self._indep, self._dep)

        expected_old_terms = np.array([1.000000000000000e+02,
                                       4.934767010474161e+03,
                                       2.238873535354354e+03,
                                       3.308688715792259e+05,
                                       6.162908768484979e+04,
                                       1.412910032530363e+05])

        close = np.allclose(computed_ols_terms, expected_old_terms)
        self.assertTrue(close)

    def test_ols_verbose(self):
        (intercept1,
         slope1,
         rss1,
         ols_terms1) = regression.ols_verbose(self._indep, self._dep)

        (intercept2,
         slope2,
         rss2,
         ols_terms2) = regression_alt.ols_verbose(self._indep, self._dep)

        tol = 1.0e-11
        self.assertAlmostEqual(intercept1, intercept2, delta=tol)
        self.assertAlmostEqual(slope1, slope2, delta=tol)
        self.assertAlmostEqual(rss1, rss2, delta=tol)
        close = np.allclose(ols_terms1, ols_terms2, rtol=0.0, atol=tol)
        self.assertTrue(close)


if __name__ == "__main__":
    unittest.main()
