"""
Testing one-bkpt segreg by comparing with alternative versions.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import unittest

import numpy.testing
import numpy as np

from segreg.model import one_bkpt_segreg
from segreg.model.alt import one_bkpt_segreg_alt


_TOL = 1.0e-10


class TestOneBkptTieouts(unittest.TestCase):

    def setUp(self):

        self._seed = None

    def tearDown(self):
        # unittest.TestCase.tearDown(self)
        if self._seed is not None:
            print()
            print("seed: ", self._seed)
            print()

    def _compare(self,
                 indep,
                 dep,
                 num_end_to_skip,
                 expected_params=None,
                 expected_value=None,
                 second_tol=1.0e-13,
                 verbose=False,
                 include_basic=True):
        tol = _TOL

        if include_basic:
            (balt_min_params,
             balt_min_value) = one_bkpt_segreg_alt.estimate_one_bkpt_segreg_basic(indep,
                                                                                  dep,
                                                                                  num_end_to_skip=num_end_to_skip,
                                                                                  verbose=False)

        (alt_min_params,
         alt_min_value) = one_bkpt_segreg_alt.estimate_one_bkpt_segreg(indep,
                                                                       dep,
                                                                       num_end_to_skip=num_end_to_skip,
                                                                       verbose=False,
                                                                       optimize=True)

        (min_params,
         min_value) = one_bkpt_segreg.estimate_one_bkpt_segreg(indep,
                                                               dep,
                                                               num_end_to_skip=num_end_to_skip)

        if verbose:
            print()
            if include_basic:
                print("BASIC")
                print(balt_min_params)
                print(balt_min_value)
            print()
            print("ALT")
            print(alt_min_params)
            print(alt_min_value)
            print()
            print("CY")
            print(min_params)
            print(min_value)
            print()

        if include_basic:
            numpy.testing.assert_allclose(balt_min_params,
                                          alt_min_params,
                                          rtol=0.0,
                                          atol=tol)
            self.assertAlmostEqual(balt_min_value, alt_min_value, delta=tol)

        numpy.testing.assert_allclose(alt_min_params,
                                      min_params,
                                      rtol=0.0,
                                      atol=tol)
        self.assertAlmostEqual(alt_min_value, min_value, delta=tol)

        if include_basic:
            numpy.testing.assert_allclose(balt_min_params,
                                          min_params,
                                          rtol=0.0,
                                          atol=tol)
            self.assertAlmostEqual(balt_min_value, min_value, delta=tol)

        if expected_params is not None:

            if include_basic:
                numpy.testing.assert_allclose(balt_min_params,
                                              expected_params,
                                              rtol=0.0,
                                              atol=second_tol)
            numpy.testing.assert_allclose(alt_min_params,
                                          expected_params,
                                          rtol=0.0,
                                          atol=second_tol)
            numpy.testing.assert_allclose(min_params,
                                          expected_params,
                                          rtol=0.0,
                                          atol=second_tol)

        if expected_value is not None:
            if include_basic:
                self.assertAlmostEqual(balt_min_value,
                                       expected_value,
                                       delta=second_tol)
            self.assertAlmostEqual(alt_min_value,
                                   expected_value,
                                   delta=second_tol)
            self.assertAlmostEqual(min_value,
                                   expected_value,
                                   delta=second_tol)


#         return (balt_min_params,
#                 alt_min_params,
#                 min_params,
#                 balt_min_value,
#                 alt_min_value,
#                 min_value)

    #@unittest.skip("skipping")
    def test1(self):
        indep = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.,
                          12., 13., 14., 15., 16., 17., 18., 19., 20., 21.,
                          22., 23., 24., 25., 26., 27., 28., 29., 30., 31.,
                          32., 33., 34., 35., 36., 37., 38., 39., 40., 41.,
                          42., 43., 44., 45., 46., 47., 48., 49., 50., 51.,
                          52., 53., 54., 55., 56., 57., 58., 59., 60., 61.,
                          62., 63., 64., 65., 66., 67., 68., 69., 70., 71.,
                          72., 73., 74., 75., 76., 77., 78., 79., 80., 81.,
                          82., 83., 84., 85., 86., 87., 88., 89., 90., 91.,
                          92., 93., 94., 95., 96., 97., 98., 99., 100.])

        dep = np.array([16.06501634, 16.06141074, 15.49089292, 16.30904663,
                        16.33143052, 16.2516089, 16.16180664, 16.32820128,
                        16.31755969, 16.35650915, 16.3292358, 16.33553528,
                        16.29592206, 16.4008524, 16.26001568, 16.36654044,
                        16.29750291, 16.46294102, 16.31973992, 16.38632926,
                        16.10478248, 16.63026159, 15.71441776, 15.94107017,
                        16.260645, 16.02549504, 16.37268008, 15.54402784,
                        16.27627501, 15.88011635, 16.23204586, 16.09751806,
                        16.32581788, 16.22833577, 15.99564897, 16.06862442,
                        16.0494935, 16.23432959, 16.01379412, 16.26723599,
                        16.42932023, 16.91108154, 16.46860592, 16.28050295,
                        16.91067701, 16.67644894, 15.8460774, 15.52023902,
                        15.49538282, 15.56716808, 15.92551513, 16.69256654,
                        15.84690985, 15.71940234, 15.20830776, 15.5496192,
                        15.67038942, 15.8609684, 15.84706269, 16.01575608,
                        15.54918627, 15.97054007, 16.08650573, 16.42132966,
                        15.61184011, 15.45724874, 15.50198465, 15.84561376,
                        16.23863, 15.9638286, 15.34425553, 15.51389262,
                        15.79518257, 15.83325209, 15.48641236, 15.07206709,
                        15.58696864, 15.73813031, 15.55142235, 16.20395037,
                        15.3095687, 15.26205539, 16.01480885, 15.56003138,
                        15.69351603, 15.77611415, 15.94623323, 16.07418793,
                        15.3646572, 15.49572429, 14.74988748, 16.00946868,
                        16.29645457, 15.95550385, 15.85595943, 16.29112181,
                        15.61662205, 15.7692196, 15.78984613, 16.08150237,
                        16.31294536])

        # this one has min value at last data point to check
        self._compare(indep, dep, num_end_to_skip=10)

    #@unittest.skip("skipping")
    def test2(self):
        tol = 1.0e-14

        num_end_to_skip = 0

        indep = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
        dep = np.array([3, 2, 1, 0, 1, 2, 3], dtype=float)

        self._compare(indep,
                      dep,
                      num_end_to_skip,
                      expected_params=[3.0, 0.0, -1.0, 1.0],
                      expected_value=0.0,
                      second_tol=tol)

    #@unittest.skip("skipping")
    def test3(self):

        num_end_to_skip = 0

        indep = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6], dtype=float)
        dep = np.array([3, 4, 2, 2, 3, 1, 1, 0, 2, 0, -1, 1, 1, 2, 0, 2, 1, 3, 3, 4, 2], dtype=float)

#        plt.scatter(indep, dep)
#        plt.show()

        self._compare(indep,
                      dep,
                      num_end_to_skip,
                      expected_params=[3.0, 0.0, -1.0, 1.0],
                      expected_value=14.0,
                      verbose=False)

    #@unittest.skip("skipping")
    def test4(self):

        num_end_to_skip = 0

        indep = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6], dtype=float)
        dep = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5], dtype=float)

        #plt.scatter(indep, dep)
        # plt.show()

        self._compare(indep,
                      dep,
                      num_end_to_skip,
                      expected_params=[3.0, 1.0, 0.0, 1.0],
                      expected_value=14.0,
                      verbose=False)

    #@unittest.skip("skipping")
    def test5(self):
        """
        Solution at last data point to check.
        """

        num_end_to_skip = 0

        indep = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6], dtype=float)
        dep = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 3], dtype=float)

#        plt.scatter(indep, dep)
#        plt.show()

        self._compare(indep,
                      dep,
                      num_end_to_skip,
                      expected_params=[5.0, 1.0, 0.0, 1.0],
                      expected_value=14.0,
                      verbose=False)

    #@unittest.skip("skipping")
    def test6(self):
        """
        Solution at last data point to check.
        """

        num_end_to_skip = 1

        indep = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6], dtype=float)
        dep = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 3, 2, 3, 4], dtype=float)

        #plt.scatter(indep, dep)
        # plt.show()

        self._compare(indep,
                      dep,
                      num_end_to_skip,
                      expected_params=[4.0, 1.0, 0.0, 1.0],
                      expected_value=14.0,
                      verbose=False)

    #@unittest.skip("skipping")
    def test7(self):
        """
        Solution at first data point to check.
        Mirror-image of test6.
        """

        num_end_to_skip = 1

        indep = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6], dtype=float)
        dep = np.array([2, 3, 4, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=float)

        #plt.scatter(indep, dep)
        # plt.show()

        self._compare(indep,
                      dep,
                      num_end_to_skip,
                      expected_params=[2.0, 1.0, -1.0, 0.0],
                      expected_value=14.0,
                      verbose=False)

    #@unittest.skip("skipping")
    def test8(self):
        """
        Like test7, but change num_end_to_skip.
        """

        num_end_to_skip = 0

        indep = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6], dtype=float)
        dep = np.array([2, 3, 4, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=float)

        #plt.scatter(indep, dep)
        # plt.show()

        self._compare(indep,
                      dep,
                      num_end_to_skip,
                      expected_params=[2.0, 1.0, -1.0, 0.0],
                      expected_value=14.0,
                      verbose=False)

    #@unittest.skip("skipping")
    def test9(self):
        """
        Solution at interior of last interval.
        """

        num_end_to_skip = 0

        indep = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
        dep = np.array([0, 0, 0, 0, 0, -1, -5], dtype=float)

        #plt.scatter(indep, dep)
        # plt.show()

        self._compare(indep,
                      dep,
                      num_end_to_skip,
                      expected_params=[4.75, 0.0, 0.0, -4.0],
                      expected_value=0.0,
                      verbose=False)

    #@unittest.skip("skipping")
    def test10(self):
        """
        Solution at interior of first interval.
        Mirror-image of test8.
        """

        num_end_to_skip = 0

        indep = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
        dep = np.array([-5, -1, 0, 0, 0, 0, 0], dtype=float)

        #plt.scatter(indep, dep)
        # plt.show()

        self._compare(indep,
                      dep,
                      num_end_to_skip,
                      expected_params=[1.25, 0.0, 4.0, 0.0],
                      expected_value=0.0,
                      verbose=False)

    #@unittest.skip("skipping")
    def test11(self):

        num_end_to_skip = 1

        indep = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
        dep = np.array([0, 1, 0, 0, 0, 0, 0], dtype=float)

        #plt.scatter(indep, dep)
        # plt.show()

        self._compare(indep,
                      dep,
                      num_end_to_skip,
                      expected_params=None,
                      expected_value=None,
                      verbose=False)

    #@unittest.skip("skipping")
    def test12(self):
        """
        This has two solutions at u=1, u=3.  The algorithm only finds u=1.
        Idea: start in middle and work out towards ends???
        """

        num_end_to_skip = 0

        indep = np.array([0, 1, 2, 3, 4], dtype=float)
        dep = np.array([0, 1, 0, 1, 0], dtype=float)

        #plt.scatter(indep, dep)
        # plt.show()

        self._compare(indep,
                      dep,
                      num_end_to_skip,
                      expected_params=None,
                      expected_value=None,
                      verbose=False)


if __name__ == "__main__":
    unittest.main()
