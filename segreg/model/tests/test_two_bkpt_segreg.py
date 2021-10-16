"""
Testing two-bkpt segreg against alternative versions.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause


import unittest

import numpy as np
import numpy.testing

from segreg.model import one_bkpt_segreg, two_bkpt_segreg
from segreg.model.alt import regression_alt
from segreg.model.alt import two_bkpt_segreg_alt
from segreg.model.tests import two_bkpt_segreg_examples
from segreg.data import _testing_util


class TestTwoBkptSegreg(unittest.TestCase):

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

    #@unittest.skip("skipping")
    def test_fixed_bkpt_least_squares(self):
        """
        Tie out cython with pure python version.
        """
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
        indep, dep = _testing_util.generate_fake_data_normal_errors(num_data,
                                                                    x_min,
                                                                    x_max,
                                                                    func,
                                                                    stddev,
                                                                    seed=seed)

        ind1 = 30
        ind2 = 70
        u1 = indep[ind1]
        u2 = indep[ind2]

        indep1 = indep[0:ind1]
        dep1 = dep[0:ind1]
        indep2 = indep[ind1:ind2]
        dep2 = dep[ind1:ind2]
        indep3 = indep[ind2:]
        dep3 = dep[ind2:]

        ols_terms_1 = regression_alt.ols_terms(indep1, dep1)
        ols_terms_2 = regression_alt.ols_terms(indep2, dep2)
        ols_terms_3 = regression_alt.ols_terms(indep3, dep3)

        assert(ols_terms_1[0] + ols_terms_2[0] + ols_terms_3[0] == len(indep))

        v1alt, v2alt, m1alt, m2alt, rssalt = two_bkpt_segreg_alt.fixed_bkpt_ls(ols_terms_1,
                                                                               ols_terms_2,
                                                                               ols_terms_3,
                                                                               u1,
                                                                               u2)

        v1, v2, m1, m2, rss = two_bkpt_segreg.fixed_bkpt_least_squares(ols_terms_1,
                                                                       ols_terms_2,
                                                                       ols_terms_3,
                                                                       u1,
                                                                       u2)

        self.assertEqual(v1, v1alt)
        self.assertEqual(v2, v2alt)
        self.assertEqual(m1, m1alt)
        self.assertEqual(m2, m2alt)
        self.assertEqual(rss, rssalt)

        v1b, v2b, m1b, m2b, rssb = two_bkpt_segreg.fixed_bkpt_ls_for_data(indep,
                                                                          dep,
                                                                          u1,
                                                                          u2)
        # TODO: this tol ok?
        delta = 1.0e-11
        self.assertAlmostEqual(v1, v1b, delta=delta)
        self.assertAlmostEqual(v2, v2b, delta=delta)
        self.assertAlmostEqual(m1, m1b, delta=delta)
        self.assertAlmostEqual(m2, m2b, delta=delta)
        self.assertAlmostEqual(rss, rssb, delta=delta)

    #@unittest.skip("skipping")
    def test_fixed_bkpt_ls(self):
        """
        Tie out cython with pure python version.
        """

        example = two_bkpt_segreg_examples.corner_NW_square_NW(multiple_y=True)
        indep = example.indep
        dep = example.dep

        u1 = 1
        u2 = 7

        v1alt, v2alt, m1alt, m2alt, rssalt = two_bkpt_segreg_alt.fixed_bkpt_ls_from_data(indep, dep, u1, u2)

        v1, v2, m1, m2, rss = two_bkpt_segreg.fixed_bkpt_ls_for_data(indep,
                                                                     dep,
                                                                     u1,
                                                                     u2)

        alt_params = [u1, v1alt, u2, v2alt, m1alt, m2alt]
        cy_params = [u1, v1, u2, v2, m1, m2]

        expected_params = example.params
        expected_rss = example.rss

        numpy.testing.assert_allclose(alt_params, expected_params, rtol=0.0, atol=1.0e-12)
        numpy.testing.assert_allclose(cy_params, expected_params, rtol=0.0, atol=1.0e-12)

        self.assertEqual(rssalt, expected_rss)
        self.assertEqual(rss, expected_rss)

        u1 = 2
        u2 = 6

        v1alt, v2alt, m1alt, m2alt, rssalt = two_bkpt_segreg_alt.fixed_bkpt_ls_from_data(indep, dep, u1, u2)

        v1, v2, m1, m2, rss = two_bkpt_segreg.fixed_bkpt_ls_for_data(indep,
                                                                     dep,
                                                                     u1,
                                                                     u2)

        alt_params = [u1, v1alt, u2, v2alt, m1alt, m2alt]
        cy_params = [u1, v1, u2, v2, m1, m2]

        numpy.testing.assert_allclose(alt_params,
                                      cy_params,
                                      rtol=0.0,
                                      atol=1.0e-12)
        self.assertAlmostEqual(rssalt, rss, delta=1.0e-12)

    def _check_fixed_bkpt_ls_internals(self, indep, dep, u1, u2):
        side_pairs = [['left', 'left'],
                      ['right', 'left'],
                      ['left', 'right'],
                      ['right', 'right']]

        results = []

        for pair in side_pairs:

            index1 = np.searchsorted(indep, u1, side=pair[0])
            index2 = np.searchsorted(indep, u2, side=pair[1])

            indep1 = indep[0:index1]
            dep1 = dep[0:index1]
            indep2 = indep[index1:index2]
            dep2 = dep[index1:index2]
            indep3 = indep[index2:]
            dep3 = dep[index2:]

            ols_terms_1 = regression_alt.ols_terms(indep1, dep1)
            ols_terms_2 = regression_alt.ols_terms(indep2, dep2)
            ols_terms_3 = regression_alt.ols_terms(indep3, dep3)

            v1, v2, m1, m2, rss = two_bkpt_segreg_alt.fixed_bkpt_ls(ols_terms_1,
                                                                    ols_terms_2,
                                                                    ols_terms_3,
                                                                    u1,
                                                                    u2)
            results.append([v1, v2, m1, m2, rss])

        first_result = results[0]
        for result in results[1:]:
            numpy.testing.assert_allclose(result,
                                          first_result,
                                          rtol=0.0,
                                          atol=1.0e-12)

    def test_fixed_bkpt_ls_internals(self):
        """
        We know this mathematically.  This is an extra check on implementation,
        albeit quite in the weeds.
        """

        example = two_bkpt_segreg_examples.corner_NW_square_NW(multiple_y=True)
        indep = example.indep
        dep = example.dep

        u1 = 1
        u2 = 7

        self._check_fixed_bkpt_ls_internals(indep, dep, u1, u2)

        u1 = 2
        u2 = 6

        self._check_fixed_bkpt_ls_internals(indep, dep, u1, u2)

        u1 = 2.5
        u2 = 6.5

        self._check_fixed_bkpt_ls_internals(indep, dep, u1, u2)

        u1 = 4.0
        u2 = 5.0

        self._check_fixed_bkpt_ls_internals(indep, dep, u1, u2)

        example = two_bkpt_segreg_examples.interior_square_NE(multiple_y=True)
        indep = example.indep
        dep = example.dep

        u1 = 1
        u2 = 7

        self._check_fixed_bkpt_ls_internals(indep, dep, u1, u2)

        u1 = 2
        u2 = 6

        self._check_fixed_bkpt_ls_internals(indep, dep, u1, u2)

        u1 = 2.5
        u2 = 6.5

        self._check_fixed_bkpt_ls_internals(indep, dep, u1, u2)

        u1 = 4.0
        u2 = 5.0

        self._check_fixed_bkpt_ls_internals(indep, dep, u1, u2)

    #@unittest.skip("skipping")
    def test_tie_out_pure_python(self):
        (min_params,
         min_value) = two_bkpt_segreg.estimate_two_bkpt_segreg(self._indep2,
                                                               self._dep2)

        (min_params_alt,
         min_value_alt) = two_bkpt_segreg_alt.estimate_two_bkpt_segreg(self._indep2,
                                                                       self._dep2)

        for val, val_alt in zip(min_params, min_params_alt):
            self.assertAlmostEqual(val, val_alt, delta=1.0e-15)

        self.assertAlmostEqual(min_value, min_value_alt, delta=1.0e-15)

#         min_params_alt_2, min_value_alt_2 = two_bkpt_segreg_alt.estimate_two_bkpt_segreg_purepy(
#             self._indep2, self._dep2)
#
#         for val, val_alt in zip(min_params, min_params_alt_2):
#             self.assertAlmostEqual(val, val_alt, delta=1.0e-15)
#
#         self.assertAlmostEqual(min_value, min_value_alt_2, delta=1.0e-15)

    #@unittest.skip("skipping")
    def _test_data(self,
                   indep,
                   dep,
                   num_end_to_skip=2,
                   num_between_to_skip=3,
                   verbose=False):
        (min_params,
         min_value) = two_bkpt_segreg.estimate_two_bkpt_segreg(indep,
                                                               dep,
                                                               num_end_to_skip=num_end_to_skip,
                                                               num_between_to_skip=num_between_to_skip)

        (min_params_alt,
         min_value_alt) = two_bkpt_segreg_alt.estimate_two_bkpt_segreg(indep,
                                                                       dep,
                                                                       num_end_to_skip=num_end_to_skip,
                                                                       num_between_to_skip=num_between_to_skip,
                                                                       verbose=verbose)

        for val, val_alt in zip(min_params, min_params_alt):
            self.assertAlmostEqual(val, val_alt, delta=1.0e-15)

        self.assertAlmostEqual(min_value, min_value_alt, delta=1.0e-15)

    #@unittest.skip("skipping")
    def test_tie_out_pure_python_suite(self):
        # min at: interior square
        # EXTRA SIDE BNDY FIX u1_data_next,
        # EXTRA SIDE BNDY FIX u2_data_next

        indep = np.array([12.56825089, 14.73056038, 14.87957976, 23.80398437, 31.74834797,
                          39.92624099, 44.87686087, 45.52516479, 46.33365116, 48.75110813,
                          51.3044325, 58.45439029, 67.42858676, 74.43507968, 79.57414322])
        dep = np.array([13.77579137, 13.02596512, 11.87033656, 12.27814005, 10.96322397,
                        18.13720115, 20.00043867, 21.74219926, 22.92424179, 25.85955849,
                        26.8671903, 30.59422735, 37.10138493, 40.70041576, 40.54570971])
        self._test_data(indep, dep, num_end_to_skip=2, num_between_to_skip=3)

        # min at: EXTRA SIDE BNDY FIX u1_data_next
        indep = np.array([0.51739052, 5.20473534, 5.94975717, 5.99641505, 8.36676359,
                          9.11816633, 12.22169496, 12.59375897, 13.90929782, 15.21107239,
                          15.29658846, 17.50264876, 18.87098409, 25.97668985, 29.31074364])
        dep = np.array([11.66438517, 8.02568725, 12.90131284, 8.70738677, 10.35105175,
                        11.02956934, 8.76443499, 7.29721396, 6.51778999, 7.09774966,
                        9.10643211, 6.47674903, 5.65178239, 5.5705972, 6.0352771])

        self._test_data(indep, dep, num_end_to_skip=2, num_between_to_skip=3)

        # min at corner
        indep = np.array([0.7930158, 3.2447625, 6.87880775, 7.97081275, 8.32884528,
                          9.66847643, 15.95139702, 18.17720255, 20.91994331, 24.97242289,
                          25.16074953, 28.92913367, 29.83305062, 29.95396095, 30.23427343,
                          31.60396851, 33.59667219, 33.98585426, 34.40921118, 35.48855417,
                          37.60909674, 42.21543843, 44.57496728, 46.22633874, 46.68802967,
                          48.09493334, 49.53394491, 50.37688477, 51.63372092, 56.90257275,
                          58.24414288, 65.80053113, 68.96265157, 72.32545634, 77.57383429,
                          77.88597358, 78.36135956, 81.38491175, 81.55329211, 82.78186442,
                          85.05426059, 85.92944832, 89.44020277, 93.55622434, 94.42423776,
                          94.93668466, 95.89008503, 96.51179355, 98.57934447, 99.41618822])
        dep = np.array([7.04995658, 17.30496965, 15.65506018, 12.19832504, 17.62550455,
                        17.01470301, 1.29226539, 9.77368334, 4.07926561, 12.89252611,
                        8.58948489, 12.63013518, 14.37948781, 6.67770865, 11.76886755,
                        6.06128959, 7.68036803, 4.18247965, 2.49320766, 1.49491777,
                        0.18997578, 11.43896708, 13.47351645, 6.22897925, 14.78403295,
                        12.29469744, 0.99453236, 10.31892922, 0.8995087, 9.53824917,
                        -4.94009965, -5.04900931, 2.32295004, 7.3099462, -1.23041764,
                        1.87537023, 3.40448574, 6.65287646, 6.58817392, -2.05845787,
                        -4.44969555, -4.87227791, -5.68559734, 3.43123333, -0.68007438,
                        -2.71273329, -6.02971448, 0.69115426, -4.26797726, -1.36975162])

        self._test_data(indep,
                        dep,
                        num_end_to_skip=10,
                        num_between_to_skip=10)

        # min at: EXTRA SIDE BNDY FIX u2_data_next
        indep = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.,
                          12., 13., 14., 15., 16., 17., 18., 19., 20.])
        dep = np.array([16.7657374, 16.66304924, 16.42356086, 16.55386542,
                        16.01315871, 16.63416859, 16.00543082, 16.0115388,
                        16.30811176, 16.07282908, 15.65418164, 15.62406448,
                        16.10103264, 16.1665398, 15.73820585, 15.68296928,
                        16.11564633, 16.08361701, 15.9797384, 15.39192604,
                        15.22326413])

        self._test_data(indep,
                        dep,
                        num_end_to_skip=3,
                        num_between_to_skip=3,
                        verbose=False)


if __name__ == "__main__":
    unittest.main()
