"""
Testing one-bkpt segreg by comparing with a brute-force method.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import unittest

from segreg.model import one_bkpt_segreg
from segreg.model.alt import brute_force_segreg, one_bkpt_segreg_alt
from segreg.model.tests import one_bkpt_segreg_examples
from segreg.model.tests.one_bkpt_segreg_helper import OneBkptSegregHelper


class TestOneBkptSuite(OneBkptSegregHelper):

    #@unittest.skip("skipping")
    def test_cython_versus_alt(self):
        lhs_module = one_bkpt_segreg_alt
        rhs_module = one_bkpt_segreg

        self.suite(lhs_module, rhs_module, tol=1.0e-12)

    def test_cython_versus_brute(self):
        """
        Note: brute passes here with high tolerance because the examples are
        constructed with bkpts on brute force's search grid.

        np.arange can have very small diffs to expected values; hence we loosen
        tol a little
        """
        lhs_module = one_bkpt_segreg
        rhs_module = brute_force_segreg

        self.suite(lhs_module, rhs_module, tol=1.0e-11)

    def suite(self, lhs_module, rhs_module, tol):

        func_names = ["corner_E_interval_E",
                      "corner_W_interval_E",
                      "corner_W_interval_W",
                      "corner_E_interval_W",
                      "corner_interval_middle",
                      "interior_interval_E",
                      "interior_interval_E_minusone",
                      "interior_interval_W",
                      "interior_interval_W_plusone"]

        self._suite_impl(lhs_module=lhs_module,
                         rhs_module=rhs_module,
                         func_names=func_names,
                         tol=tol)

    def _suite_impl(self, lhs_module, rhs_module, func_names, tol):
        for func_name in func_names:

            self.comparison(lhs_module=lhs_module,
                            rhs_module=rhs_module,
                            func_name=func_name,
                            tol=tol,
                            plot=False)

    def comparison(self, lhs_module, rhs_module, func_name, tol, plot=False):

        example = getattr(one_bkpt_segreg_examples, func_name)()
        self.compare(lhs_module=lhs_module,
                     rhs_module=rhs_module,
                     indep=example.indep,
                     dep=example.dep,
                     num_end_to_skip=example.num_end_to_skip,
                     tol=tol,
                     expected_params=example.params,
                     expected_value=example.rss,
                     plot=plot,
                     verbose=False)

        self.compare(lhs_module=lhs_module,
                     rhs_module=rhs_module,
                     indep=example.indep,
                     dep=example.dep,
                     num_end_to_skip=2,
                     tol=tol,
                     expected_params=None,
                     expected_value=None,
                     plot=False,
                     verbose=False)

        example = getattr(one_bkpt_segreg_examples, func_name)(multiple_y=True)

        self.compare(lhs_module=lhs_module,
                     rhs_module=rhs_module,
                     indep=example.indep,
                     dep=example.dep,
                     num_end_to_skip=example.num_end_to_skip,
                     tol=tol,
                     expected_params=example.params,
                     expected_value=example.rss,
                     plot=plot)


if __name__ == "__main__":
    unittest.main()
