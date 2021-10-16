"""
Unittest for one_bkpt_segreg.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest

import numpy as np

from segreg.model import one_bkpt_segreg
from segreg.model.alt import one_bkpt_segreg_alt
from segreg.model.alt import regression_alt
from segreg.data import _testing_util

_DELTA = 1.0e-11


class TestOneBkptSegReg(unittest.TestCase):

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

    ##########################################################################
    # fixed_bkpt_least_squares
    ##########################################################################

    def test_one_bkpt_regression_tieout(self):
        """
        Tie out new cython one bkpt regression with previous all-python
        regression_util.
        """
        index = 7
        ols_data1 = regression_alt.ols_terms(
            self._indep[0:index], self._dep[0:index])
        ols_data2 = regression_alt.ols_terms(
            self._indep[index:], self._dep[index:])

        u1 = self._indep[index - 1]
        u2 = self._indep[index]
        u3 = 0.5 * (u1 + u2)

        tol = 1.0e-12

        for u in [u1, u2, u3]:
            v, m1, m2, rss = one_bkpt_segreg.fixed_bkpt_least_squares(
                ols_data1, ols_data2, u)
            ru_v, ru_m1, ru_m2, ru_rss = one_bkpt_segreg_alt._fixed_bkpt_ls_impl(
                ols_data1, ols_data2, u)

            self.assertAlmostEqual(v, ru_v, delta=tol)
            self.assertAlmostEqual(m1, ru_m1, delta=tol)
            self.assertAlmostEqual(m2, ru_m2, delta=tol)
            self.assertAlmostEqual(rss, ru_rss, delta=tol)

    def test_fixed_bkpt_least_squares(self):
        """
        Tests cython impl versus previously-computed known values.
        """
        ind = 50
        lhs_indep = self._indep[0:ind]
        lhs_dep = self._dep[0:ind]
        rhs_indep = self._indep[ind:]
        rhs_dep = self._dep[ind:]

        lhs_ols_terms = regression_alt.ols_terms(lhs_indep, lhs_dep)
        rhs_ols_terms = regression_alt.ols_terms(rhs_indep, rhs_dep)

        u = self._indep[ind]
        v, m1, m2, rss = one_bkpt_segreg.fixed_bkpt_least_squares(lhs_ols_terms,
                                                                  rhs_ols_terms,
                                                                  u)

        computed = np.array([v, m1, m2, rss])

        expected = np.array([2.004382671236669e+01,
                             2.624209010855996e-01,
                             4.347576641826885e-01,
                             4.956903975490332e+02])

        close = np.allclose(expected, computed)
        self.assertTrue(close)

    ##########################################################################
    # estimate_one_bkpt_segreg
    ##########################################################################

    def test_estimate_one_bkpt_segreg(self):
        (computed_est_params,
         computed_est_value) = one_bkpt_segreg.estimate_one_bkpt_segreg(self._indep,
                                                                        self._dep)

        expected_est_params = np.array([19.823644118891426,
                                        9.81779810550313,
                                        -0.113661299203566,
                                        0.395549681365013])
        #expected_est_value = 313.8556519862591

        expected_est_value = 313.85565198624533

        close = np.allclose(expected_est_params, computed_est_params)
        self.assertTrue(close)

        self.assertAlmostEqual(expected_est_value,
                               computed_est_value,
                               delta=_DELTA)

        ##########
        (computed_est_params,
         computed_est_value) = one_bkpt_segreg.estimate_one_bkpt_segreg(self._indep,
                                                                        self._dep,
                                                                        num_end_to_skip=0)

        expected_est_params = np.array([19.823644118891426,
                                        9.817798105503106,
                                        -0.113661299203568,
                                        0.395549681365014])
        #expected_est_value = 313.855651986245
        expected_est_value = 313.85565198624533

        close = np.allclose(expected_est_params, computed_est_params)
        self.assertTrue(close)

        self.assertAlmostEqual(expected_est_value,
                               computed_est_value,
                               delta=_DELTA)
        ##########
        (computed_est_params,
         computed_est_value) = one_bkpt_segreg.estimate_one_bkpt_segreg(self._indep,
                                                                        self._dep,
                                                                        num_end_to_skip=1)

        expected_est_params = np.array([19.823644118891426,
                                        9.81779810550313,
                                        -0.113661299203566,
                                        0.395549681365013])

        #expected_est_value = 313.85565198627364

        expected_est_value = 313.85565198624533

        close = np.allclose(expected_est_params, computed_est_params)
        self.assertTrue(close)

        self.assertAlmostEqual(expected_est_value,
                               computed_est_value,
                               delta=_DELTA)

        ##########
        (computed_est_params,
         computed_est_value) = one_bkpt_segreg.estimate_one_bkpt_segreg(self._indep,
                                                                        self._dep,
                                                                        num_end_to_skip=10)

        expected_est_params = np.array([19.823644118891426,
                                        9.81779810550311,
                                        -0.113661299203568,
                                        0.395549681365014])

        #expected_est_value = 313.85565198621714

        expected_est_value = 313.85565198624533

        close = np.allclose(expected_est_params, computed_est_params)
        self.assertTrue(close)

        self.assertAlmostEqual(expected_est_value,
                               computed_est_value,
                               delta=_DELTA)
        ##########
        (computed_est_params,
         computed_est_value) = one_bkpt_segreg.estimate_one_bkpt_segreg(self._indep,
                                                                        self._dep,
                                                                        m2=0)

        expected_est_params = np.array([97.314862590113847,
                                        39.353578042674471,
                                        0.353390573215835])
        #expected_est_value = 642.2100221862838
        expected_est_value = 642.2100221862638

        close = np.allclose(expected_est_params, computed_est_params)
        self.assertTrue(close)

        self.assertAlmostEqual(expected_est_value,
                               computed_est_value,
                               delta=_DELTA)

        ##########
        (computed_est_params,
         computed_est_value) = one_bkpt_segreg.estimate_one_bkpt_segreg(self._indep,
                                                                        self._dep,
                                                                        num_end_to_skip=10,
                                                                        m2=0)

        expected_est_params = np.array([88.387918368954359,
                                        36.823770136742397,
                                        0.363272815551632])
        #expected_est_value = 774.8647504839428

        expected_est_value = 774.8647504839901

        close = np.allclose(expected_est_params, computed_est_params)
        self.assertTrue(close)

        self.assertAlmostEqual(expected_est_value,
                               computed_est_value,
                               delta=_DELTA)
        ##########
        (computed_est_params,
         computed_est_value) = one_bkpt_segreg.estimate_one_bkpt_segreg(self._indep,
                                                                        self._dep,
                                                                        m2=-0.1)

        expected_est_params = np.array([97.314862590113847,
                                        39.367716099083275,
                                        0.353604389035283])
        #expected_est_value = 643.7676277954997

        expected_est_value = 643.7676277955397

        close = np.allclose(expected_est_params, computed_est_params)
        self.assertTrue(close)

        self.assertAlmostEqual(expected_est_value,
                               computed_est_value,
                               delta=_DELTA)


if __name__ == "__main__":
    unittest.main()
