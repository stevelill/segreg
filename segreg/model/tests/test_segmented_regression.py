"""
Testing one-bkpt segreg versus known values.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import unittest

import numpy as np

from segreg import data
from segreg.model import one_bkpt_segreg
from segreg.model import OneBkptSegRegEstimator


class TestSegmentedRegression(unittest.TestCase):

    def setUp(self):
        """
        Example from: David Hinkley, 'Inference in Two-Phase Regression', 1971
        """
        self._indep, self._dep = data.hinkley()

    def test_get_concentrated_params(self):
        u = 3.2

        # eventually can switch to regression.pyx
        (v,
         m1,
         m2,
         rss) = one_bkpt_segreg.fixed_bkpt_ls_for_data(self._indep,
                                                       self._dep,
                                                       u)

        expected_y0 = 0.76502984252021233
        expected_lhs_slope = 0.33978593251421768
        expected_rhs_slope = 0.048418840979537602

        self.assertAlmostEqual(expected_y0, v, places=12)
        self.assertAlmostEqual(expected_lhs_slope, m1, places=12)
        self.assertAlmostEqual(expected_rhs_slope, m2, places=12)

    def test_get_concentrated_params_match_x_val(self):
        u = 4.0
        (v,
         m1,
         m2,
         rss) = one_bkpt_segreg.fixed_bkpt_ls_for_data(self._indep,
                                                       self._dep,
                                                       u)

        expected_y0 = 0.85049000018459431
        expected_lhs_slope = 0.23282504604709114
        expected_rhs_slope = 0.034205673610849002

        self.assertAlmostEqual(expected_y0, v, places=12)
        self.assertAlmostEqual(expected_lhs_slope, m1, places=12)
        self.assertAlmostEqual(expected_rhs_slope, m2, places=12)

    def test_get_residuals(self):
        seg_reg = OneBkptSegRegEstimator()
        seg_reg.fit(self._indep, self._dep)
        residuals = seg_reg.residuals()

        expected_residuals = np.array([-0.034275543372, 0.032013734373,
                                       0.009386361128, 0.023827638872,
                                       -0.029980734373, -0.000971456628,
                                       0.03229308002, 0.006242996413,
                                       -0.04434088159, 0.001208034804,
                                       0.029348156801, -0.080886926806,
                                       0.010757195191, 0.061151111585,
                                       -0.015772766418])

        close = np.allclose(expected_residuals, residuals)
        self.assertTrue(close)

    def test_get_func(self):
        restrict_rhs_slope = 0.0
        expected_vals = np.array([0.404758543372215, 0.505956265627365, 0.598297638872425,
                                  0.699495361127575, 0.791836734372635, 0.893034456627785,
                                  0.924413919980346, 0.934106003586703, 0.942949881589812,
                                  0.95264196519617, 0.961485843199278, 0.971177926805636,
                                  0.980021804808744, 0.989713888415102, 0.99855776641821])

        expected_res_vals = np.array([0.404758543372215, 0.505956265627365, 0.598297638872425,
                                      0.699495361127575, 0.791836734372635, 0.893034456627784,
                                      0.961674333333334, 0.961674333333334, 0.961674333333334,
                                      0.961674333333334, 0.961674333333334, 0.961674333333334,
                                      0.961674333333334, 0.961674333333334, 0.961674333333334])

        self._test_get_func(restrict_rhs_slope, expected_vals, expected_res_vals)

        #####

        # np.set_printoptions(precision=15)

        restrict_rhs_slope = -0.1
        expected_vals = np.array([0.404758543372215, 0.505956265627365, 0.598297638872425,
                                  0.699495361127575, 0.791836734372635, 0.893034456627785,
                                  0.924413919980346, 0.934106003586703, 0.942949881589812,
                                  0.95264196519617, 0.961485843199278, 0.971177926805636,
                                  0.980021804808744, 0.989713888415102, 0.99855776641821])

        expected_res_vals = np.array([0.498595520531996, 0.558225344576934, 0.61263664668893,
                                      0.672266470733868, 0.726677772845864, 0.786307596890802,
                                      0.840718899002799, 0.900348723047736, 0.954760025159733,
                                      1.01438984920467, 1.068801151316667, 1.052536,
                                      1.004824, 0.952536, 0.904824])

        self._test_get_func(restrict_rhs_slope, expected_vals, expected_res_vals)

    def _test_get_func(self, restrict_rhs_slope, expected_vals, expected_res_vals):

        restricted_estimator = OneBkptSegRegEstimator(restrict_rhs_slope=restrict_rhs_slope)
        estimator = OneBkptSegRegEstimator()

        estimator.fit(self._indep, self._dep)
        restricted_estimator.fit(self._indep, self._dep)

        func = estimator.get_func()
        res_func = restricted_estimator.get_func()

        vals = func(self._indep)

        # begin eras
#        print
#        print "expected: "
#        print expected_vals
#        print
#        print
#        print "computed: "
#        print vals
#        print
#        exit()
        # end erae

        close = np.allclose(vals, expected_vals)
        self.assertTrue(close)

        res_vals = res_func(self._indep)
        res_close = np.allclose(res_vals, expected_res_vals)
        self.assertTrue(res_close)

    #@unittest.skip("skipping")
    def test_single_break_point_regression(self):
        seg_reg = OneBkptSegRegEstimator()
        est_params = seg_reg.fit(self._indep, self._dep)
        est_value = seg_reg.estimation_func_val_at_estimate()

        expected_params = [4.6516524673749009,
                           0.91795696348885369,
                           0.19353909550021009,
                           0.018535961609466453,
                           0.034940613059983604]
        expected_value = 0.018312696615112443

        close = np.allclose(expected_params, est_params)

        self.assertTrue(close)
        self.assertAlmostEqual(expected_value, est_value, places=12)

    #@unittest.skip("skipping")
    def test_single_break_point_regression_restricted_slope(self):
        rhs_slope = 0.0

        seg_reg = OneBkptSegRegEstimator(restrict_rhs_slope=rhs_slope)
        est_params = seg_reg.fit(self._indep, self._dep)
        est_value = seg_reg.estimation_func_val_at_estimate()

        expected_params = [4.8775363888197658,
                           0.9616743333333333,
                           0.19353909550021009,
                           rhs_slope,
                           0.039553191529800594]

        expected_value = 0.0234668244029

        close = np.allclose(expected_params, est_params)

        self.assertTrue(close)
        self.assertAlmostEqual(expected_value, est_value, places=12)


if __name__ == "__main__":
    unittest.main()
