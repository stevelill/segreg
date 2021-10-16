"""
Testing two-bkpt segreg by comparing with a brute-force method.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest


from matplotlib import pyplot as plt
import numpy as np
import numpy.testing

from segreg.analysis import stats_plotting
from segreg.model import two_bkpt_segreg
from segreg.model.alt import brute_force_segreg
from segreg.model.alt import two_bkpt_segreg_alt


class TwoBkptSegregHelper(unittest.TestCase):

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
                num_between_to_skip,
                tol,
                verbose=False,
                expected_params=None,
                expected_value=None,
                plot=False,
                seed=None):
        (lhs_min_params,
         lhs_min_value) = lhs_module.estimate_two_bkpt_segreg(indep,
                                                              dep,
                                                              num_end_to_skip=num_end_to_skip,
                                                              num_between_to_skip=num_between_to_skip)

        (rhs_min_params,
         rhs_min_value) = rhs_module.estimate_two_bkpt_segreg(indep,
                                                              dep,
                                                              num_end_to_skip=num_end_to_skip,
                                                              num_between_to_skip=num_between_to_skip)

        if verbose:
            print()
            print()
            print(lhs_module.__name__)
            print(lhs_min_params)
            print(lhs_min_value)
            print()
            print(rhs_module.__name__)
            print(rhs_min_params)
            print(rhs_min_value)
            print()

        if plot:
            func = two_bkpt_segreg.segmented_func(lhs_min_params)
            stats_plotting.plot_model(func=func,
                                      indep=indep,
                                      dep=dep,
                                      extra_pts=[lhs_min_params[0],
                                                 lhs_min_params[2]],
                                      full_size_scatter=True)
            plt.show()

        self.assertAlmostEqual(lhs_min_value,
                               rhs_min_value,
                               delta=1000.0 * tol)

        try:
            # sometimes different bkpts found due to precision diffs only
            numpy.testing.assert_allclose(lhs_min_params,
                                          rhs_min_params,
                                          rtol=0.0,
                                          atol=tol)
        except:
            print()
            print("-" * 50)
            print("EXCEPTION: OK")
            print(lhs_module.__name__)
            print(lhs_min_params)
            print(lhs_min_value)
            print(rhs_module.__name__)
            print(rhs_min_params)
            print(rhs_min_value)
            print()

            if seed is not None:
                print("seed: ", seed)

            print("-" * 50)
            print()

        if expected_params is not None:
            self.check_known_value(expected_params=expected_params,
                                   computed_params=lhs_min_params,
                                   expected_value=expected_value,
                                   computed_value=lhs_min_value)

            self.check_known_value(expected_params=expected_params,
                                   computed_params=rhs_min_params,
                                   expected_value=expected_value,
                                   computed_value=rhs_min_value)

    def compare_to_brute_force(self,
                               lhs_module,
                               indep,
                               dep,
                               num_end_to_skip,
                               num_between_to_skip,
                               dx=0.01,
                               verbose=False,
                               seed=None):
        (lhs_min_params,
         lhs_min_value) = lhs_module.estimate_two_bkpt_segreg(indep,
                                                              dep,
                                                              num_end_to_skip=num_end_to_skip,
                                                              num_between_to_skip=num_between_to_skip)

        (rhs_min_params,
         rhs_min_value) = brute_force_segreg.estimate_two_bkpt_segreg(indep,
                                                                      dep,
                                                                      num_end_to_skip=num_end_to_skip,
                                                                      num_between_to_skip=num_between_to_skip,
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
#         numpy.testing.assert_allclose(lhs_min_params,
#                                       rhs_min_params,
#                                       rtol=0.0,
#                                       atol=tol)

        # if bkpt diff not super tiny, make sure that brute force RSS is
        # greater than the method we are checking against
        if abs(rhs_min_value - lhs_min_value) > 1.0e-10:
            self.assertTrue(lhs_min_value <= rhs_min_value)

        self.assertAlmostEqual(lhs_min_value, rhs_min_value, delta=tol)

        try:
            # check bkpts close
            self.assertAlmostEqual(lhs_min_params[0],
                                   rhs_min_params[0],
                                   delta=tol)
            self.assertAlmostEqual(lhs_min_params[2],
                                   rhs_min_params[2],
                                   delta=tol)
        except:
            # if bkpts not close, brute may have gotten stuck somewhere else;
            # let's make sure it is not better than lhs solution
            [v1, v2, m1, m2, rss] = two_bkpt_segreg_alt.fixed_bkpt_ls_from_data(indep,
                                                                                dep,
                                                                                u1=rhs_min_params[0],
                                                                                u2=rhs_min_params[2])
            if abs(rhs_min_value - lhs_min_value) > 1.0e-10:
                self.assertTrue(lhs_min_value < rss)

            print()
            print("-" * 50)
            print("EXCEPTION: OK")
            print(lhs_module.__name__)
            print(lhs_min_params)
            print(lhs_min_value)
            print(brute_force_segreg.__name__)
            print(rhs_min_params)
            print(rhs_min_value)
            print()

            if seed is not None:
                print("seed: ", seed)

            print("-" * 50)
            print()

    def brute_force_test_suite(self,
                               lhs_module,
                               num_data=10,
                               num_tests=100,
                               num_end_to_skip=0,
                               num_between_to_skip=2,
                               seeds=None,
                               investigate=False):
        """
        hypothesis?
        """

        indep = np.arange(num_data, dtype=float)

        if seeds is None:
            np.random.seed(3483407)
            seeds = np.random.randint(10, 1000000, num_tests)

        for i, seed in enumerate(seeds):

            if i % 100 == 0:
                print("iter: ", i)

            self._seed = seed

            np.random.seed(seed)

            dep = np.random.randint(low=-10, high=10, size=num_data)
            dep = np.array(dep, dtype=float)

            printdata = investigate
            if printdata:
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
                                        num_between_to_skip,
                                        verbose=investigate,
                                        seed=seed)

            dep = np.random.rand(num_data)

            printdata = investigate
            if printdata:
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
                                        num_between_to_skip,
                                        verbose=investigate,
                                        seed=seed)

            dep = np.random.randn(num_data)

            printdata = investigate
            if printdata:
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
                                        num_between_to_skip,
                                        verbose=investigate,
                                        seed=seed)

    def compare_test_suite(self,
                           lhs_module,
                           rhs_module,
                           num_data=10,
                           num_tests=100,
                           num_end_to_skip=0,
                           num_between_to_skip=2,
                           tol=1.0e-12,
                           seeds=None,
                           investigate=False):
        """
        hypothesis?
        """

        indep = np.arange(num_data, dtype=float)

        if seeds is None:
            np.random.seed(3483407)
            seeds = np.random.randint(10, 1000000, num_tests)

        for i, seed in enumerate(seeds):

            if i % 1000 == 0:
                print("iter: ", i)

            self._seed = seed

            np.random.seed(seed)

            dep = np.random.randint(low=-10, high=10, size=num_data)
            dep = np.array(dep, dtype=float)

            printdata = investigate
            if printdata:
                import pprint
                print()
                pprint.pprint(indep)
                print()
                pprint.pprint(dep)
                print()
                plt.scatter(indep, dep)
                plt.show()

            self.compare(lhs_module=lhs_module,
                         rhs_module=rhs_module,
                         indep=indep,
                         dep=dep,
                         num_end_to_skip=num_end_to_skip,
                         num_between_to_skip=num_between_to_skip,
                         tol=tol,
                         verbose=investigate,
                         seed=seed)

            dep = np.random.rand(num_data)

            printdata = investigate
            if printdata:
                print()
                pprint.pprint(indep)
                print()
                pprint.pprint(dep)
                print()
                plt.scatter(indep, dep)
                plt.show()

            self.compare(lhs_module=lhs_module,
                         rhs_module=rhs_module,
                         indep=indep,
                         dep=dep,
                         num_end_to_skip=num_end_to_skip,
                         num_between_to_skip=num_between_to_skip,
                         tol=tol,
                         verbose=investigate,
                         seed=seed)

            dep = np.random.randn(num_data)

            printdata = investigate
            if printdata:
                print()
                pprint.pprint(indep)
                print()
                pprint.pprint(dep)
                print()
                plt.scatter(indep, dep)
                plt.show()

            self.compare(lhs_module=lhs_module,
                         rhs_module=rhs_module,
                         indep=indep,
                         dep=dep,
                         num_end_to_skip=num_end_to_skip,
                         num_between_to_skip=num_between_to_skip,
                         tol=tol,
                         verbose=investigate,
                         seed=seed)

    def check_known_value(self,
                          expected_params,
                          computed_params,
                          expected_value=None,
                          computed_value=None):
        tol = 1.0e-12

        numpy.testing.assert_allclose(computed_params,
                                      expected_params,
                                      rtol=0.0,
                                      atol=tol)
        if expected_value is not None:
            self.assertAlmostEqual(computed_value,
                                   expected_value,
                                   delta=tol)
