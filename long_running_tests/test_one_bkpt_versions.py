"""
Test various implementations of one-bkpt segmented regression estimation against
each other in a large test suite.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import time
import unittest

from segreg.model import one_bkpt_segreg
from segreg.model.alt import one_bkpt_segreg_alt
from segreg.model.tests.one_bkpt_segreg_helper import OneBkptSegregHelper


class TestOneBkptVersions(OneBkptSegregHelper):

    def run_test(self,
                 lhs_module,
                 rhs_module,
                 num_data=100,
                 num_tests=2000,
                 num_end_to_skip=0,
                 tol=1.0e-11):

        print()
        print("COMPARING:")
        print(lhs_module.__name__)
        print(rhs_module.__name__)
        print()

        start = time.time()

        seeds = None
        investigate = False

        # for investigation
        #seeds = [350054]
        #seeds = [924305]
        #seeds = [314519]
        #investigate = True

        self.compare_test_suite(lhs_module,
                                rhs_module,
                                num_data,
                                num_tests,
                                seeds=seeds,
                                investigate=investigate,
                                num_end_to_skip=num_end_to_skip,
                                tol=tol)
        end = time.time()
        print()
        print("time: ", end - start)
        print()

    #@unittest.skip("skipping")
    def test1(self):

        lhs_module = one_bkpt_segreg_alt
        rhs_module = one_bkpt_segreg

        num_data = 10
        num_tests = 100000
        num_end_to_skip = 0

        self.run_test(lhs_module=lhs_module,
                      rhs_module=rhs_module,
                      num_data=num_data,
                      num_tests=num_tests,
                      num_end_to_skip=num_end_to_skip)

    #@unittest.skip("skipping")
    def test2(self):

        lhs_module = one_bkpt_segreg_alt
        rhs_module = one_bkpt_segreg

        num_data = 10
        num_tests = 100000
        num_end_to_skip = 2

        self.run_test(lhs_module=lhs_module,
                      rhs_module=rhs_module,
                      num_data=num_data,
                      num_tests=num_tests,
                      num_end_to_skip=num_end_to_skip)

    #@unittest.skip("skipping")
    def test3(self):

        lhs_module = one_bkpt_segreg_alt
        rhs_module = one_bkpt_segreg

        num_data = 100
        num_tests = 10000
        num_end_to_skip = 0

        self.run_test(lhs_module=lhs_module,
                      rhs_module=rhs_module,
                      num_data=num_data,
                      num_tests=num_tests,
                      num_end_to_skip=num_end_to_skip,
                      tol=1.0e-10)

    #@unittest.skip("skipping")
    def test4(self):

        lhs_module = one_bkpt_segreg_alt
        rhs_module = one_bkpt_segreg

        num_data = 100
        num_tests = 10000
        num_end_to_skip = 20

        self.run_test(lhs_module=lhs_module,
                      rhs_module=rhs_module,
                      num_data=num_data,
                      num_tests=num_tests,
                      num_end_to_skip=num_end_to_skip,
                      tol=1.0e-10)

    #@unittest.skip("skipping")
    def test5(self):

        lhs_module = one_bkpt_segreg_alt
        rhs_module = one_bkpt_segreg

        num_data = 10
        num_tests = 100000
        num_end_to_skip = 3

        self.run_test(lhs_module=lhs_module,
                      rhs_module=rhs_module,
                      num_data=num_data,
                      num_tests=num_tests,
                      num_end_to_skip=num_end_to_skip)

    #@unittest.skip("skipping")
    def test6(self):

        lhs_module = one_bkpt_segreg_alt
        rhs_module = one_bkpt_segreg

        num_data = 20
        num_tests = 100000
        num_end_to_skip = 5

        self.run_test(lhs_module=lhs_module,
                      rhs_module=rhs_module,
                      num_data=num_data,
                      num_tests=num_tests,
                      num_end_to_skip=num_end_to_skip)


if __name__ == "__main__":
    unittest.main()
