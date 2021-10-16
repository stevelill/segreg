"""
Testing alternative one-bkpt segreg.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest

import numpy.testing

from segreg import data
from segreg.model import one_bkpt_segreg
from segreg.model.alt import one_bkpt_segreg_alt
from segreg.model.alt import regression_alt
from segreg.data import _testing_util


class TestOneBkptSegregAlt(unittest.TestCase):

    def setUp(self):
        num_data = 100
        x_min = 0
        x_max = 100

        u = 50
        v = 10
        m1 = -0.05
        m2 = 0.4
        func = one_bkpt_segreg.segmented_func(u, v, m1, m2)
        stddev = 2.0

        seed = 123456789
        #seed = None
        self._indep, self._dep = _testing_util.generate_fake_data_normal_errors(num_data,
                                                                                x_min,
                                                                                x_max,
                                                                                func,
                                                                                stddev,
                                                                                seed=seed)

    def test_one_fixed_bkpt_tieout(self):
        """
        Tie out new cython one bkpt regression with previous all-python
        regression_util.
        """
        index = 7
        ols_data1 = regression_alt.ols_terms(
            self._indep[0:index], self._dep[0:index])
        ols_data2 = regression_alt.ols_terms(
            self._indep[index:], self._dep[index:])

        # TODO: double check that cases where bkpt equals data point are ok
        # consider bkpt to be on left, right, or middle
        u1 = self._indep[index - 1]
        u2 = self._indep[index]
        u3 = 0.5 * (u1 + u2)

        tol = 1.0e-11

        for u in [u1, u2, u3]:
            v, m1, m2, rss = one_bkpt_segreg.fixed_bkpt_least_squares(ols_data1,
                                                                      ols_data2, u)

            a_v, a_m1, a_m2, a_rss = one_bkpt_segreg_alt._fixed_bkpt_ls_impl(ols_data1,
                                                                             ols_data2, u)

            self.assertAlmostEqual(v, a_v, delta=tol)
            self.assertAlmostEqual(m1, a_m1, delta=tol)
            self.assertAlmostEqual(m2, a_m2, delta=tol)
            self.assertAlmostEqual(rss, a_rss, delta=tol)

            b_v, b_m1, b_m2 = one_bkpt_segreg_alt.fixed_bkpt_ls_regression(self._indep,
                                                                           self._dep, u)

            self.assertAlmostEqual(v, b_v, delta=tol)
            self.assertAlmostEqual(m1, b_m1, delta=tol)
            self.assertAlmostEqual(m2, b_m2, delta=tol)

            c_v, c_m1, c_m2, c_rss = one_bkpt_segreg_alt.fixed_bkpt_ls(self._indep,
                                                                       self._dep, u)

            self.assertAlmostEqual(v, c_v, delta=tol)
            self.assertAlmostEqual(m1, c_m1, delta=tol)
            self.assertAlmostEqual(m2, c_m2, delta=tol)
            self.assertAlmostEqual(rss, c_rss, delta=tol)

    def test_one_bkpt_estimation_tie_out(self):
        tol = 1.0e-8

        (est_params,
         est_value) = one_bkpt_segreg_alt.estimate_one_bkpt_segreg_basic(self._indep,
                                                                         self._dep)

        (est_params2,
         est_value2) = one_bkpt_segreg_alt.estimate_one_bkpt_segreg(self._indep,
                                                                    self._dep)

        numpy.testing.assert_allclose(est_params, est_params2)
        self.assertAlmostEqual(est_value, est_value2, delta=tol)

        (est_params3,
         est_value3) = one_bkpt_segreg.estimate_one_bkpt_segreg(self._indep,
                                                                self._dep)

        numpy.testing.assert_allclose(est_params2, est_params3)
        self.assertAlmostEqual(est_value2, est_value3, delta=tol)

    def test_rss_for_region(self):
        indep, dep = data.test1()

        ols_data = regression_alt.ols_terms(indep, dep)

        num_data, sum_x, sum_y, sum_xx, sum_yy, sum_xy = ols_data

        u = 20.0
        v = 8.0
        m = 0.02

        term = v - m * u

        expected = (sum_yy
                    - 2.0 * term * sum_y
                    - 2.0 * m * sum_xy
                    + m * m * sum_xx
                    + 2.0 * m * term * sum_x +
                    term * term * num_data)

        computed = one_bkpt_segreg_alt.rss_for_region(ols_data, u, v, m)

        self.assertAlmostEqual(expected, computed, delta=1.0e-12)


if __name__ == "__main__":
    unittest.main()
