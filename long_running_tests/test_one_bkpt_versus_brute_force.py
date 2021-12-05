"""
Test one-bkpt segmented regression estimation against a brute-force method of
calculating the fit in a large test suite.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest

from segreg.model import one_bkpt_segreg
from segreg.model.alt import one_bkpt_segreg_alt
from segreg.model.tests.one_bkpt_segreg_helper import OneBkptSegregHelper


class TestOneBkptVersusBruteForce(OneBkptSegregHelper):

    #@unittest.skip("skipping")
    def test1(self):
        num_data = 10
        num_tests = 10000

        lhs_module = one_bkpt_segreg_alt

        seeds = None

        #seeds = [585834]

        self.brute_force_test_suite(lhs_module,
                                    num_data,
                                    num_tests,
                                    seeds=seeds,
                                    investigate=False)

        self.brute_force_test_suite(lhs_module,
                                    num_data,
                                    num_tests,
                                    seeds=seeds,
                                    investigate=False,
                                    num_end_to_skip=2)

    #@unittest.skip("skipping")
    def test2(self):
        num_data = 100
        num_tests = 10000

        lhs_module = one_bkpt_segreg_alt

        seeds = None

        self.brute_force_test_suite(lhs_module,
                                    num_data,
                                    num_tests,
                                    seeds=seeds,
                                    investigate=False)

        self.brute_force_test_suite(lhs_module,
                                    num_data,
                                    num_tests,
                                    seeds=seeds,
                                    investigate=False,
                                    num_end_to_skip=20)

    #@unittest.skip("skipping")
    def test3(self):
        num_data = 10
        num_tests = 10000

        lhs_module = one_bkpt_segreg

        #seeds = [924305]
        seeds = None

        self.brute_force_test_suite(lhs_module,
                                    num_data,
                                    num_tests,
                                    seeds=seeds)

        self.brute_force_test_suite(lhs_module,
                                    num_data,
                                    num_tests,
                                    num_end_to_skip=2,
                                    seeds=seeds,
                                    investigate=False)

    #@unittest.skip("skipping")
    def test4(self):
        num_data = 100
        num_tests = 10000

        lhs_module = one_bkpt_segreg

        self.brute_force_test_suite(lhs_module,
                                    num_data,
                                    num_tests)

        self.brute_force_test_suite(lhs_module,
                                    num_data,
                                    num_tests,
                                    num_end_to_skip=20)


if __name__ == "__main__":
    unittest.main()
