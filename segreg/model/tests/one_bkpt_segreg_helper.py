"""
Testing one-bkpt segreg by comparing with a brute-force method.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import unittest

from matplotlib import pyplot as plt
import numpy.testing

import numpy as np
from segreg.analysis import stats_plotting
from segreg.model import one_bkpt_segreg
from segreg.model.alt import brute_force_segreg
from segreg.model.alt import one_bkpt_segreg_alt


class OneBkptSegregHelper(unittest.TestCase):

    def setUp(self):
        self._seed = None

    def tearDown(self):
        if self._seed is not None:
            print()
            print("seed: ", self._seed)
            print()

    def compare(self,
                lhs_module,
                rhs_module,
                indep,
                dep,
                num_end_to_skip,
                tol,
                expected_params=None,
                expected_value=None,
                verbose=False,
                plot=False):

        (lhs_min_params,
         lhs_min_value) = lhs_module.estimate_one_bkpt_segreg(indep,
                                                              dep,
                                                              num_end_to_skip=num_end_to_skip,
                                                              check_near_middle=False)

        (rhs_min_params,
         rhs_min_value) = rhs_module.estimate_one_bkpt_segreg(indep,
                                                              dep,
                                                              num_end_to_skip=num_end_to_skip,
                                                              check_near_middle=False)

        if verbose:
            print()
            print()
            print(lhs_module.__name__)
            print(np.array(lhs_min_params))
            print(lhs_min_value)
            print()
            print(rhs_module.__name__)
            print(np.array(rhs_min_params))
            print(rhs_min_value)
            print()

        if plot:
            func = one_bkpt_segreg.segmented_func(*lhs_min_params)
            stats_plotting.plot_model(func=func,
                                      indep=indep,
                                      dep=dep,
                                      extra_pts=[lhs_min_params[0]],
                                      full_size_scatter=True)
            plt.show()

        numpy.testing.assert_allclose(lhs_min_params,
                                      rhs_min_params,
                                      rtol=0.0,
                                      atol=tol)

        self.assertAlmostEqual(lhs_min_value, rhs_min_value, delta=1000.0 * tol)

        if expected_params is not None:
            self.check_known_value(expected_params=expected_params,
                                   computed_params=lhs_min_params,
                                   expected_value=expected_value,
                                   computed_value=lhs_min_value,
                                   tol=tol)

            self.check_known_value(expected_params=expected_params,
                                   computed_params=rhs_min_params,
                                   expected_value=expected_value,
                                   computed_value=rhs_min_value,
                                   tol=tol)

    def compare_to_brute_force(self,
                               lhs_module,
                               indep,
                               dep,
                               num_end_to_skip,
                               dx=0.01,
                               verbose=False,
                               seed=None):
        """
        NOTE: turn off optimization "check_near_middle" to better compare the
        methods.  Sometimes this can find different solution when there are 
        ties amongst breakpoints.
        """

        (lhs_min_params,
         lhs_min_value) = lhs_module.estimate_one_bkpt_segreg(indep,
                                                              dep,
                                                              num_end_to_skip=num_end_to_skip,
                                                              check_near_middle=False)

        (rhs_min_params,
         rhs_min_value) = brute_force_segreg.estimate_one_bkpt_segreg(indep,
                                                                      dep,
                                                                      num_end_to_skip=num_end_to_skip,
                                                                      dx=dx)

        if verbose:
            print()
            print()
            print(lhs_module.__name__)
            print(lhs_min_params)
            print(lhs_min_value)
            print()
            print("BRUTE FORCE")
            print(brute_force_segreg.__name__)
            print(rhs_min_params)
            print(rhs_min_value)
            print()

        tol = dx

        bkpt_super_close = abs(lhs_min_params[0] - rhs_min_params[0]) < 1.0e-10

#         print()
#         print("bkpt diff")
#         print(abs(lhs_min_params[0] - rhs_min_params[0]))
#         print("brute force: ", rhs_min_params[0])
#         print()

        # if bkpt diff not super tiny, make sure that brute force RSS is
        # greater than the method we are checking against
        # note: the epsilon tol here depends on overall magnitude of RSS
        # we skip this check if bkpts essentially the same, due to natural
        # spurious precisions diffs that can occur in different impls, even
        # when bkpts exactly the same
        # what tol; depends on RSS magnitude
        if not bkpt_super_close:
            if abs(rhs_min_value - lhs_min_value) > 1.0e-11:
                self.assertTrue(lhs_min_value <= rhs_min_value)

        self.assertAlmostEqual(lhs_min_value, rhs_min_value, delta=tol)

        try:
            # check bkpts close
            self.assertAlmostEqual(lhs_min_params[0],
                                   rhs_min_params[0],
                                   delta=tol)
        except:
            # if bkpts not close, brute may have gotten stuck somewhere else;
            # let's make sure it is not better than lhs solution -- up to
            # spurious precision
            [v1, m1, m2, rss] = one_bkpt_segreg_alt.fixed_bkpt_ls(indep,
                                                                  dep,
                                                                  u=rhs_min_params[0])
            if abs(rhs_min_value - lhs_min_value) > 1.0e-10:
                self.assertTrue(lhs_min_value < rss)
            print()
            print("-" * 50)
            print("EXCEPTION: OK")
            print()
            print()
            print(lhs_module.__name__)
            print(lhs_min_params)
            print(lhs_min_value)
            print()
            print("BRUTE FORCE")
            print(brute_force_segreg.__name__)
            print(rhs_min_params)
            print(rhs_min_value)
            print()

            if seed is not None:
                print("seed: ", seed)
            print()

    def brute_force_test_suite(self,
                               lhs_module,
                               num_data=10,
                               num_tests=100,
                               seeds=None,
                               investigate=False,
                               num_end_to_skip=0):
        """
        hypothesis?
        """
        indep = np.arange(num_data, dtype=float)

        if seeds is None:
            np.random.seed(3483407)
            seeds = np.random.randint(10, 1000000, num_tests)

        for i, seed in enumerate(seeds):

            if i % 500 == 0:
                print("iter: ", i)

            self._seed = seed

            np.random.seed(seed)

            dep = np.random.randint(low=-10, high=10, size=num_data)
            dep = np.array(dep, dtype=float)

            if investigate:
                import pprint
                print()
                pprint.pprint(indep)
                print()
                pprint.pprint(dep)
                print()
                plt.scatter(indep, dep)
                plt.show()

            self.compare_to_brute_force(lhs_module,
                                        indep,
                                        dep,
                                        num_end_to_skip,
                                        verbose=investigate,
                                        seed=seed)

            dep = np.random.rand(num_data)

            if investigate:
                print()
                pprint.pprint(indep)
                print()
                pprint.pprint(dep)
                print()
                plt.scatter(indep, dep)
                plt.show()

            self.compare_to_brute_force(lhs_module,
                                        indep,
                                        dep,
                                        num_end_to_skip,
                                        verbose=investigate,
                                        seed=seed)

            dep = np.random.randn(num_data)

            if investigate:
                print()
                pprint.pprint(indep)
                print()
                pprint.pprint(dep)
                print()
                plt.scatter(indep, dep)
                plt.show()

            self.compare_to_brute_force(lhs_module,
                                        indep,
                                        dep,
                                        num_end_to_skip,
                                        verbose=investigate,
                                        seed=seed)

    def compare_test_suite(self,
                           lhs_module,
                           rhs_module,
                           num_data=10,
                           num_tests=100,
                           seeds=None,
                           investigate=False,
                           num_end_to_skip=0,
                           tol=1.0e-12):
        """
        hypothesis?
        """

        indep = np.arange(num_data, dtype=float)

        if seeds is None:
            np.random.seed(3483407)
            seeds = np.random.randint(10, 1000000, num_tests)

        for i, seed in enumerate(seeds):

            if i % 10000 == 0:
                print("iter: ", i)

            self._seed = seed

            np.random.seed(seed)

            dep = np.random.randint(low=-10, high=10, size=num_data)
            dep = np.array(dep, dtype=float)

            if investigate:
                import pprint
                print()
                pprint.pprint(indep)
                print()
                pprint.pprint(dep)
                print()
                plt.scatter(indep, dep)
                plt.show()

            self.compare(lhs_module,
                         rhs_module,
                         indep,
                         dep,
                         num_end_to_skip,
                         tol,
                         verbose=investigate)

            dep = np.random.rand(num_data)

            if investigate:
                print()
                pprint.pprint(indep)
                print()
                pprint.pprint(dep)
                print()
                plt.scatter(indep, dep)
                plt.show()

            self.compare(lhs_module,
                         rhs_module,
                         indep,
                         dep,
                         num_end_to_skip,
                         tol,
                         verbose=investigate)

            dep = np.random.randn(num_data)

            if investigate:
                print()
                pprint.pprint(indep)
                print()
                pprint.pprint(dep)
                print()
                plt.scatter(indep, dep)
                plt.show()

            self.compare(lhs_module,
                         rhs_module,
                         indep,
                         dep,
                         num_end_to_skip,
                         tol,
                         verbose=investigate)

    def check_known_value(self,
                          expected_params,
                          computed_params,
                          expected_value=None,
                          computed_value=None,
                          tol=1.0e-12):

        numpy.testing.assert_allclose(computed_params,
                                      expected_params,
                                      rtol=0.0,
                                      atol=tol)
        if expected_value is not None:
            self.assertAlmostEqual(computed_value,
                                   expected_value,
                                   delta=tol)
