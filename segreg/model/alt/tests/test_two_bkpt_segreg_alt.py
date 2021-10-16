"""
Testing alternative two-bkpt segreg.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import unittest

import numpy as np
import numpy.testing

from segreg.model import one_bkpt_segreg
from segreg.model import two_bkpt_segreg
from segreg.model.alt import regression_alt
from segreg.model.alt import two_bkpt_segreg_alt
from segreg.data import _testing_util


class TestTwoBkptSegregAlt(unittest.TestCase):

    def setUp(self):
        num_data = 100
        x_min = 0
        x_max = 100

        u = 50
        v = 10
        m1 = -0.05
        m2 = 0.4
        func = one_bkpt_segreg.segmented_func(u, v, m1, m2)
        stddev = 2.0

        seed = 123456789
        #seed = None
        self._indep, self._dep = _testing_util.generate_fake_data_normal_errors(num_data,
                                                                                x_min,
                                                                                x_max,
                                                                                func,
                                                                                stddev,
                                                                                seed=seed)

        num_data = 100
        x_min = 0
        x_max = 100

        u1 = 30
        v1 = 10
        u2 = 70
        v2 = 40
        m1 = -0.2
        m2 = 0.1
        params = [u1, v1, u2, v2, m1, m2]

        def func(x):
            return two_bkpt_segreg_alt.segmented_func(x, params)

        stddev = 0.1
        stddev = 1.0

        seed = 123456789
        #seed = None
        self._indep2, self._dep2 = _testing_util.generate_fake_data_normal_errors(num_data,
                                                                                  x_min,
                                                                                  x_max,
                                                                                  func,
                                                                                  stddev,
                                                                                  seed=seed)

    def test_fixed_breakpt_ls(self):
        """
        RSS of two bkpt regression when the bkpts u1 and u2 are fixed.
        This compares the brute-force algebraic formula versus an OLS version
        and shows that they tie out.
        """

        ind1 = 30
        ind2 = 70
        u1 = self._indep[ind1]
        u2 = self._indep[ind2]

        indep1 = self._indep[0:ind1]
        dep1 = self._dep[0:ind1]
        indep2 = self._indep[ind1:ind2]
        dep2 = self._dep[ind1:ind2]
        indep3 = self._indep[ind2:]
        dep3 = self._dep[ind2:]

        ols_terms_1 = regression_alt.ols_terms(indep1, dep1)
        ols_terms_2 = regression_alt.ols_terms(indep2, dep2)
        ols_terms_3 = regression_alt.ols_terms(indep3, dep3)

        assert(ols_terms_1[0] + ols_terms_2[0] + ols_terms_3[0] == len(self._indep))
#        print
#        print ols_terms_1
#        print ols_terms_2
#        print ols_terms_3
#        print

        v1, v2, m1, m2, rss = two_bkpt_segreg_alt.fixed_bkpt_ls(ols_terms_1,
                                                                ols_terms_2,
                                                                ols_terms_3,
                                                                u1,
                                                                u2)
        m = (v2 - v1) / (u2 - u1)

#        print
#        print "GIVEN"
#        print "u1: ", u1
#        print "u2: ", u2
#        print "DETERMINED"
#        print "v1: ", v1
#        print "v2: ", v2
#        print "m1: ", m1
#        print "m: ", m
#        print "m2: ", m2
#        print "rss: ", rss
#        print

        ##########
        v1_reg, v2_reg, m1_reg, m2_reg = two_bkpt_segreg_alt.fixed_bkpt_ls_regression(self._indep,
                                                                                      self._dep,
                                                                                      u1,
                                                                                      u2)

#        print
#        print "FROM REGRESSION WAY"
#        print "v1: ", v1_reg
#        print "v2: ", v2_reg
#        print "m1: ", m1_reg
#        print "m2: ", m2_reg
#        print

        close = np.allclose([v1, v2, m1, m2], [v1_reg, v2_reg, m1_reg, m2_reg])
        self.assertTrue(close)

#        plt.scatter(self._indep, self._dep)
#        plt.plot(u1, v1, 'x', color="red")
#        plt.plot(u2, v2, 'x', color="red")
#        plt.show()

        # second data set
        indep1 = self._indep2[0:ind1]
        dep1 = self._dep2[0:ind1]
        indep2 = self._indep2[ind1:ind2]
        dep2 = self._dep2[ind1:ind2]
        indep3 = self._indep2[ind2:]
        dep3 = self._dep2[ind2:]

        ols_terms_1 = regression_alt.ols_terms(indep1, dep1)
        ols_terms_2 = regression_alt.ols_terms(indep2, dep2)
        ols_terms_3 = regression_alt.ols_terms(indep3, dep3)

        assert(ols_terms_1[0] + ols_terms_2[0] + ols_terms_3[0] == len(self._indep2))

        v1, v2, m1, m2, rss = two_bkpt_segreg_alt.fixed_bkpt_ls(ols_terms_1,
                                                                ols_terms_2,
                                                                ols_terms_3,
                                                                u1,
                                                                u2)

        v1_reg, v2_reg, m1_reg, m2_reg = two_bkpt_segreg_alt.fixed_bkpt_ls_regression(self._indep2,
                                                                                      self._dep2,
                                                                                      u1,
                                                                                      u2)

        close = np.allclose([v1, v2, m1, m2], [v1_reg, v2_reg, m1_reg, m2_reg])
        self.assertTrue(close)


#        plt.scatter(self._indep2, self._dep2)
#        plt.plot(u1, v1, 'x', color="red")
#        plt.plot(u2, v2, 'x', color="red")
#        plt.show()

    def test_estimate_two_bkpt_segreg(self):
        min_params, min_value = two_bkpt_segreg_alt.estimate_two_bkpt_segreg(self._indep,
                                                                             self._dep)

#         expected_min_params = [47.822870172839473,
#                                9.6830334655471741,
#                                67.054643834892644,
#                                15.577251393176446,
#                                - 0.07085705058507843,
#                                0.4461709018475565]
#         expected_min_value = 304.0652184920431

        expected_min_params = np.array([47.85387666197152,
                                        9.68083643717828,
                                        67.38018840213282,
                                        15.704649432906,
                                        -0.07085705058508,
                                        0.44700063087945])

        expected_min_value = 304.0601316900478

        # todo: numpy utils
        for expected_val, val in zip(expected_min_params, min_params):
            self.assertAlmostEqual(expected_val, val, delta=1.0e-12)

        self.assertAlmostEqual(expected_min_value, min_value, delta=1.0e-12)

    def test_segmented_func(self):
        u1 = 20
        v1 = 10

        u2 = 70
        v2 = 40

        m1 = -0.2
        m2 = 0.1

        params = [u1, v1, u2, v2, m1, m2]

        domain = np.arange(0, 100, 0.1)

#        fig = plt.figure(figsize=(12,6))
#        plt.ylim([0,50])
#        plt.xlim([0,100])
#        plt.plot(domain, two_bkpt_segreg_alt.segmented_func(domain, params))
#        plt.show()

    def test_fixed_bkpt_ls_from_data(self):
        u1 = 40.0
        u2 = 80.0
        alt_arr = two_bkpt_segreg_alt.fixed_bkpt_ls_from_data(self._indep,
                                                              self._dep,
                                                              u1=u1,
                                                              u2=u2)
        alt_arr = np.array(alt_arr)

        cy_arr = two_bkpt_segreg.fixed_bkpt_ls_for_data(self._indep,
                                                        self._dep,
                                                        u1=u1,
                                                        u2=u2)
        cy_arr = np.array(cy_arr)

        numpy.testing.assert_allclose(alt_arr, cy_arr, rtol=0.0, atol=1.0e-12)


if __name__ == "__main__":

    unittest.main()
