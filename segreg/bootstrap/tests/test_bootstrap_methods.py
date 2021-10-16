"""

"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import unittest

import scipy.stats
import numpy as np

from segreg.bootstrap import bootstrap_methods
from segreg.bootstrap import resampling
from segreg.model import OneBkptSegRegEstimator


class TestBootstrapMethods(unittest.TestCase):

    def setUp(self):
        """
        Example from: David Hinkley, 'Inference in Two-Phase Regression', 1971
        """
        self._indep = np.array([2.0,
                                2.52288,
                                3.0,
                                3.52288,
                                4.0,
                                4.52288,
                                5.0,
                                5.52288,
                                6.0,
                                6.52288,
                                7.0,
                                7.52288,
                                8.0,
                                8.52288,
                                9.0])
        self._dep = np.array([0.370483,
                              .537970,
                              .607684,
                              .723323,
                              .761856,
                              .892063,
                              .956707,
                              .940349,
                              .898609,
                              .953850,
                              .990834,
                              .890291,
                              .990779,
                              1.050865,
                              .982785])

    def test_model_bca(self):
        estimator = OneBkptSegRegEstimator()

        est_params = estimator.fit(self._indep, self._dep)

        seed = 342343
        num_iter = 1000

        resample_cases = False
        params_arr = resampling.boot_param_dist(self._indep,
                                                self._dep,
                                                estimator,
                                                num_iter,
                                                resample_cases=resample_cases,
                                                seed=seed,
                                                verbose=False)

        significance = 0.1
        computed_bca = bootstrap_methods.model_bca(params_arr,
                                                   est_params,
                                                   estimator,
                                                   self._indep,
                                                   self._dep,
                                                   significance=significance)
        np.set_printoptions(precision=15)

        expected_bca = np.array([[4.306710977377564e+00, 5.207583578462149e+00],
                                 [8.707166259334380e-01, 9.612950561131541e-01],
                                 [1.595660266309449e-01, 2.231276420801550e-01],
                                 [3.388604173345641e-03, 3.318876736175657e-02],
                                 [3.087118567686039e-02, 4.714196419361153e-02]])

        close = np.allclose(expected_bca, computed_bca)
        self.assertTrue(close)

    def test_bca(self):
        estimator = OneBkptSegRegEstimator()

        est_params = estimator.fit(self._indep, self._dep)

        seed = 342343
        num_iter = 1000

        resample_cases = False
        params_arr = resampling.boot_param_dist(self._indep,
                                                self._dep,
                                                estimator,
                                                num_iter,
                                                resample_cases=resample_cases,
                                                seed=seed,
                                                verbose=False)

        significance = 0.1

        param_index = 0
        one_param_sims = params_arr[:, param_index]
        est_param = est_params[param_index]

        acceleration = bootstrap_methods.bca_acceleration(estimator, self._indep, self._dep)
        acceleration = acceleration[param_index]

        computed_bca = bootstrap_methods.bca(one_param_sims,
                                             est_param,
                                             acceleration,
                                             significance)

        expected_bca = np.array([4.306710977377564, 5.207583578462149])

        close = np.allclose(expected_bca, computed_bca)
        self.assertTrue(close)

        ########################################################################
        # do zero acceleratin
        acceleration = 0

        computed_bca = bootstrap_methods.bca(one_param_sims,
                                             est_param,
                                             acceleration,
                                             significance)

        expected_bca = np.array([4.249920064427225, 5.117859757816824])

        close = np.allclose(expected_bca, computed_bca)
        self.assertTrue(close)

    def test_boot_basic_conf_interval(self):
        estimator = OneBkptSegRegEstimator()

        est_params = estimator.fit(self._indep, self._dep)

        seed = 342343
        num_iter = 1000

        resample_cases = False
        params_arr = resampling.boot_param_dist(self._indep,
                                                self._dep,
                                                estimator,
                                                num_iter,
                                                resample_cases=resample_cases,
                                                seed=seed,
                                                verbose=False)

        significance = 0.1

        computed = bootstrap_methods.boot_basic_conf_interval(params_arr,
                                                              est_params,
                                                              significance)

        expected = np.array([[4.213861423850026e+00, 5.069736823136535e+00],
                             [8.758594006703491e-01, 9.664762750569093e-01],
                             [1.550231327596568e-01, 2.221395892229018e-01],
                             [4.459478107224107e-03, 3.463033613847930e-02],
                             [3.003010034070960e-02, 5.066285313144510e-02]])

        close = np.allclose(expected, computed)
        self.assertTrue(close)

    def test_boot_percentile_conf_interval(self):
        estimator = OneBkptSegRegEstimator()

        est_params = estimator.fit(self._indep, self._dep)

        seed = 342343
        num_iter = 1000

        resample_cases = False
        params_arr = resampling.boot_param_dist(self._indep,
                                                self._dep,
                                                estimator,
                                                num_iter,
                                                resample_cases=resample_cases,
                                                seed=seed,
                                                verbose=False)

        significance = 0.1

        computed = bootstrap_methods.boot_percentile_conf_interval(params_arr,
                                                                   est_params,
                                                                   significance)

        expected = np.array([[4.23356811161327595272e+00, 5.08944351089978486158e+00],
                             [8.69437651920798137972e-01, 9.60054526307358324644e-01],
                             [1.64938601777516985480e-01, 2.32055058240762002164e-01],
                             [2.44158708045375243828e-03, 3.26124451117089375618e-02],
                             [1.92183729885245339897e-02, 3.98511257792600326333e-02]])

        close = np.allclose(expected, computed)
        self.assertTrue(close)


if __name__ == "__main__":
    unittest.main()
