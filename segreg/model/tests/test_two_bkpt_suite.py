"""
Testing two-bkpt segreg against alternative versions for a special suite of
data.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest


from segreg.model import two_bkpt_segreg
from segreg.model.alt import two_bkpt_segreg_alt, two_bkpt_segreg_alt_SLOW
from segreg.model.tests import two_bkpt_segreg_examples
from segreg.model.tests.two_bkpt_segreg_helper import TwoBkptSegregHelper

from segreg.model.alt import brute_force_segreg

_TOL = 1.0e-12


class TestTwoBkptSuite(TwoBkptSegregHelper):

    def test_cython_versus_alt(self):
        lhs_module = two_bkpt_segreg_alt
        rhs_module = two_bkpt_segreg

        self.suite(lhs_module, rhs_module)

    def test_alt_versus_alt_slow(self):
        lhs_module = two_bkpt_segreg_alt_SLOW
        rhs_module = two_bkpt_segreg_alt

        self.suite(lhs_module, rhs_module)

    def test_cython_versus_brute(self):
        """
        Note: brute passes here with high tolerance because the examples are
        constructed with bkpts on brute force's search grid.
        """
        lhs_module = brute_force_segreg
        rhs_module = two_bkpt_segreg

        self.suite(lhs_module, rhs_module)

    def suite(self, lhs_module, rhs_module):

        func_names = ["corner_NW_square_NW",
                      "corner_SE_square_SW",
                      "corner_NE_square_NE",
                      "corner_NE_square_NW",
                      "side_W_square_NW",
                      "side_E_square_NW",
                      "side_E_square_SW",
                      "side_E_square_NE",
                      "side_S_square_SW",
                      "side_N_square_SW",
                      "side_N_square_NW",
                      "interior_square_NW",
                      "interior_square_NE"]

        self._suite_impl(lhs_module=lhs_module,
                         rhs_module=rhs_module,
                         func_names=func_names)

    def _suite_impl(self, lhs_module, rhs_module, func_names):
        for func_name in func_names:

            self.comparison(lhs_module=lhs_module,
                            rhs_module=rhs_module,
                            func_name=func_name,
                            plot=False)

    def comparison(self, lhs_module, rhs_module, func_name, plot=False):

        example = getattr(two_bkpt_segreg_examples, func_name)()
        self.compare(lhs_module=lhs_module,
                     rhs_module=rhs_module,
                     indep=example.indep,
                     dep=example.dep,
                     num_end_to_skip=example.num_end_to_skip,
                     num_between_to_skip=example.num_between_to_skip,
                     tol=_TOL,
                     expected_params=example.params,
                     expected_value=example.rss,
                     plot=plot,
                     verbose=False)

        example = getattr(two_bkpt_segreg_examples, func_name)(multiple_y=True)

        self.compare(lhs_module=lhs_module,
                     rhs_module=rhs_module,
                     indep=example.indep,
                     dep=example.dep,
                     num_end_to_skip=example.num_end_to_skip,
                     num_between_to_skip=example.num_between_to_skip,
                     tol=_TOL,
                     expected_params=example.params,
                     expected_value=example.rss,
                     plot=plot)


if __name__ == "__main__":
    unittest.main()
