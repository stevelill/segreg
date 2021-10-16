"""
Compare various implementations of segmented regression estimation for 
consistency.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest

import numpy as np

from segreg.model import OneBkptSegRegEstimator
from segreg.model.alt import one_bkpt_segreg_alt
from segreg.data import _testing_util

_DELTA = 1.0e-9


class TestSegRegTieout(unittest.TestCase):
    """
    Tie out one bkpt segmented regression estimator with a pure python method
    that uses the most basic regression formulas.
    """

    def setUp(self):
        pass

    def _create_data(self, num_data, stddev, seed=None):
        x_min = 20
        x_max = 100

        self._x0 = 50
        self._y0 = 20
        self._lhs_slope = -0.5
        self._rhs_slope = 0.2
        params = [self._x0, self._y0, self._lhs_slope, self._rhs_slope]

        seg_reg_estimator = OneBkptSegRegEstimator()
        self._func = seg_reg_estimator.get_func_for_params(params)

        if seed is not None:
            np.random.seed(seed)

        self._indep, self._dep = _testing_util.generate_fake_data_normal_errors(num_data,
                                                                                x_min,
                                                                                x_max,
                                                                                self._func,
                                                                                stddev)

    #@unittest.skip("skipping")
    def testTieoutToBruteOLS(self):

        num_data_arr = [10, 20, 30, 50, 100, 1000]
        stddev = 2.0

        seed_arr = [123, 456, 454545, 123456, 9685, 3495340]

        for num_data, seed in zip(num_data_arr, seed_arr):
            self._create_data(num_data, stddev, seed)


#             print()
#             print("indep: ")
#             print(self._indep)
#             print()
#             print("dep: ")
#             print(self._dep)
#             print()

            num_end_to_skip = 0

            good_segmented = OneBkptSegRegEstimator(
                num_end_to_skip=num_end_to_skip)
            min_params = good_segmented.fit(self._indep, self._dep)
            min_value = good_segmented.estimation_func_val_at_estimate()

#             (basic_min_params,
#              basic_min_value) = one_bkpt_segreg_alt.estimate_one_bkpt_segreg_basic(self._indep,
#                                                                                    self._dep,
#                                                                                    num_end_to_skip=num_end_to_skip,
#                                                                                    verbose=False)

            (basic_min_params,
             basic_min_value) = one_bkpt_segreg_alt.estimate_one_bkpt_segreg(self._indep,
                                                                             self._dep,
                                                                             num_end_to_skip=num_end_to_skip,
                                                                             verbose=False)

            # estimator has estimated residual stddev as last element,
            # but basic does not
            close = np.allclose(min_params[:-1], basic_min_params)

#             print()
#             print("num_data: ", num_data)
#             #print("cy:    ", min_params)
#             print("cy    rss: ", min_value)
#             # print()
#             #print("basic: ", basic_min_params)
#             print("basic rss: ", basic_min_value)
#             print()
#             print()

#            exit()

            self.assertTrue(close)
            self.assertAlmostEqual(min_value, basic_min_value, delta=_DELTA)

    #@unittest.skip("skipping")
    def testTieoutToBruteOLS2(self):

        num_tests = 100
        num_data = 50
        seed_arr = np.arange(1000, 1000 + num_tests)
        stddev_arr = np.arange(1, num_tests + 1) * 0.1

        for stddev, seed in zip(stddev_arr, seed_arr):
            self._create_data(num_data, stddev, seed)

            num_end_to_skip = 0

            good_segmented = OneBkptSegRegEstimator(
                num_end_to_skip=num_end_to_skip)
            min_params = good_segmented.fit(self._indep, self._dep)
            min_value = good_segmented.estimation_func_val_at_estimate()

            (basic_min_params,
             basic_min_value) = one_bkpt_segreg_alt.estimate_one_bkpt_segreg_basic(self._indep,
                                                                                   self._dep,
                                                                                   num_end_to_skip=num_end_to_skip)

            # estimator has estimated residual stddev as last element,
            # but basic does not
            close = np.allclose(min_params[:-1], basic_min_params)

            self.assertTrue(close)
            self.assertAlmostEqual(min_value, basic_min_value, delta=_DELTA)

    #@unittest.skip("skipping")
    def testTieoutToBruteOLS3(self):

        num_data = 50
        seed = 1095
        stddev = 9.6

        self._create_data(num_data, stddev, seed)

        num_end_to_skip = 0

        good_segmented = OneBkptSegRegEstimator(
            num_end_to_skip=num_end_to_skip)
        min_params = good_segmented.fit(self._indep, self._dep)
        min_value = good_segmented.estimation_func_val_at_estimate()

        (basic_min_params,
         basic_min_value) = one_bkpt_segreg_alt.estimate_one_bkpt_segreg_basic(self._indep,
                                                                               self._dep,
                                                                               num_end_to_skip=num_end_to_skip)

        # estimator has estimated residual stddev as last element,
        # but basic does not
        close = np.allclose(min_params[:-1], basic_min_params)

        self.assertTrue(close)
        self.assertAlmostEqual(min_value, basic_min_value, delta=_DELTA)

    #@unittest.skip("skipping")
    def testTieoutToBruteOLS4(self):

        num_end_to_skip = 2
        good_segmented = OneBkptSegRegEstimator(
            num_end_to_skip=num_end_to_skip)
        func = good_segmented.get_func_for_params([8.0, 1.0, -0.5, 0.5])

        indep = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
                          6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10], dtype=float)
        dep = func(indep)

        stddev = 0.1
        resid = stddev * np.random.randn(len(indep))
        dep += resid

        min_params = good_segmented.fit(indep, dep)
        min_value = good_segmented.estimation_func_val_at_estimate()

        (basic_min_params,
         basic_min_value) = one_bkpt_segreg_alt.estimate_one_bkpt_segreg_basic(indep,
                                                                               dep,
                                                                               num_end_to_skip=num_end_to_skip,
                                                                               verbose=False)

        # estimator has estimated residual stddev as last element,
        # but basic does not
        close = np.allclose(min_params[:-1], basic_min_params)

        self.assertTrue(close)
        self.assertAlmostEqual(min_value, basic_min_value, delta=_DELTA)

    #@unittest.skip("skipping")
    def testTieoutToBruteOLS5(self):

        num_end_to_skip = 2
        good_segmented = OneBkptSegRegEstimator(
            num_end_to_skip=num_end_to_skip)
        func = good_segmented.get_func_for_params([8.0, 1.0, -0.5, 0.5])

        indep = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        dep = func(indep)

        stddev = 0.1
        resid = stddev * np.random.randn(len(indep))
        dep += resid

        min_params = good_segmented.fit(indep, dep)
        min_value = good_segmented.estimation_func_val_at_estimate()

        (basic_min_params,
         basic_min_value) = one_bkpt_segreg_alt.estimate_one_bkpt_segreg_basic(indep,
                                                                               dep,
                                                                               num_end_to_skip=num_end_to_skip,
                                                                               verbose=False)

        # estimator has estimated residual stddev as last element,
        # but basic does not
        close = np.allclose(min_params[:-1], basic_min_params)

        self.assertTrue(close)
        self.assertAlmostEqual(min_value, basic_min_value, delta=_DELTA)


if __name__ == "__main__":
    unittest.main()
