"""
Test two-bkpt segmented regression estimation against a brute-force method of
calculating the fit in a large test suite.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest

from segreg.model import two_bkpt_segreg
from segreg.model.alt import two_bkpt_segreg_alt
from segreg.model.tests.two_bkpt_segreg_helper import TwoBkptSegregHelper


class TestTwoBkptVersusBruteForce(TwoBkptSegregHelper):

    def test1(self):
        num_data = 10
        num_tests = 1000

        num_end_to_skip = 0
        num_between_to_skip = 2

        lhs_module = two_bkpt_segreg_alt

        self.brute_force_test_suite(lhs_module,
                                    num_data,
                                    num_tests,
                                    num_end_to_skip=num_end_to_skip,
                                    num_between_to_skip=num_between_to_skip,
                                    investigate=False)

    def test2(self):
        num_data = 10
        num_tests = 1000

        num_end_to_skip = 0
        num_between_to_skip = 2

        lhs_module = two_bkpt_segreg

        self.brute_force_test_suite(lhs_module,
                                    num_data,
                                    num_tests,
                                    num_end_to_skip=num_end_to_skip,
                                    num_between_to_skip=num_between_to_skip,
                                    investigate=False)

    def test3(self):
        num_data = 20
        num_tests = 100

        num_end_to_skip = 2
        num_between_to_skip = 5

        lhs_module = two_bkpt_segreg_alt

        self.brute_force_test_suite(lhs_module,
                                    num_data,
                                    num_tests,
                                    num_end_to_skip=num_end_to_skip,
                                    num_between_to_skip=num_between_to_skip,
                                    investigate=False)

    def test4(self):
        num_data = 20
        num_tests = 100

        num_end_to_skip = 2
        num_between_to_skip = 5

        lhs_module = two_bkpt_segreg

        self.brute_force_test_suite(lhs_module,
                                    num_data,
                                    num_tests,
                                    num_end_to_skip=num_end_to_skip,
                                    num_between_to_skip=num_between_to_skip,
                                    investigate=False)


if __name__ == "__main__":
    unittest.main()
