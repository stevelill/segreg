"""
Test various implementations of two-bkpt segmented regression estimation against
each other in a large test suite.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import time
import unittest

from segreg.model import two_bkpt_segreg
from segreg.model.alt import two_bkpt_segreg_alt
from segreg.model.alt import two_bkpt_segreg_alt_SLOW
from segreg.model.tests.two_bkpt_segreg_helper import TwoBkptSegregHelper


class TestTwoBkptVersions(TwoBkptSegregHelper):

    def run_test(self,
                 lhs_module,
                 rhs_module,
                 num_data=100,
                 num_tests=2000,
                 num_end_to_skip=0,
                 num_between_to_skip=2):

        print()
        print("COMPARING:")
        print(lhs_module.__name__)
        print(rhs_module.__name__)
        print()

        start = time.time()

        # begin erase
        #seeds = [486838]
        seeds = None
        # end erase

        tol = 1.0e-11

        self.compare_test_suite(lhs_module=lhs_module,
                                rhs_module=rhs_module,
                                num_data=num_data,
                                num_tests=num_tests,
                                num_end_to_skip=num_end_to_skip,
                                num_between_to_skip=num_between_to_skip,
                                tol=tol,
                                seeds=seeds,
                                investigate=False)
        end = time.time()
        print()
        print("time: ", end - start)
        print()

    def test1(self):

        lhs_module = two_bkpt_segreg_alt
        rhs_module = two_bkpt_segreg

        num_data = 100
        num_tests = 2000
        num_end_to_skip = 0
        num_between_to_skip = 2

        self.run_test(lhs_module=lhs_module,
                      rhs_module=rhs_module,
                      num_data=num_data,
                      num_tests=num_tests,
                      num_end_to_skip=num_end_to_skip,
                      num_between_to_skip=num_between_to_skip)

    def test2(self):

        lhs_module = two_bkpt_segreg_alt_SLOW
        rhs_module = two_bkpt_segreg_alt

        num_data = 100
        num_tests = 2000
        num_end_to_skip = 0
        num_between_to_skip = 2

        self.run_test(lhs_module=lhs_module,
                      rhs_module=rhs_module,
                      num_data=num_data,
                      num_tests=num_tests,
                      num_end_to_skip=num_end_to_skip,
                      num_between_to_skip=num_between_to_skip)

    def test3(self):

        lhs_module = two_bkpt_segreg_alt
        rhs_module = two_bkpt_segreg

        num_data = 50
        num_tests = 2000
        num_end_to_skip = 10
        num_between_to_skip = 10

        self.run_test(lhs_module=lhs_module,
                      rhs_module=rhs_module,
                      num_data=num_data,
                      num_tests=num_tests,
                      num_end_to_skip=num_end_to_skip,
                      num_between_to_skip=num_between_to_skip)

    def test4(self):

        lhs_module = two_bkpt_segreg_alt_SLOW
        rhs_module = two_bkpt_segreg_alt

        num_data = 50
        num_tests = 2000
        num_end_to_skip = 10
        num_between_to_skip = 10

        self.run_test(lhs_module=lhs_module,
                      rhs_module=rhs_module,
                      num_data=num_data,
                      num_tests=num_tests,
                      num_end_to_skip=num_end_to_skip,
                      num_between_to_skip=num_between_to_skip)

    def test5(self):

        lhs_module = two_bkpt_segreg_alt
        rhs_module = two_bkpt_segreg

        num_data = 10
        num_tests = 100000
        num_end_to_skip = 1
        num_between_to_skip = 2

        self.run_test(lhs_module=lhs_module,
                      rhs_module=rhs_module,
                      num_data=num_data,
                      num_tests=num_tests,
                      num_end_to_skip=num_end_to_skip,
                      num_between_to_skip=num_between_to_skip)

    def test6(self):

        lhs_module = two_bkpt_segreg_alt_SLOW
        rhs_module = two_bkpt_segreg_alt

        num_data = 10
        num_tests = 100000
        num_end_to_skip = 1
        num_between_to_skip = 2

        self.run_test(lhs_module=lhs_module,
                      rhs_module=rhs_module,
                      num_data=num_data,
                      num_tests=num_tests,
                      num_end_to_skip=num_end_to_skip,
                      num_between_to_skip=num_between_to_skip)


if __name__ == "__main__":
    unittest.main()
