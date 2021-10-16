"""

"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import pprint
import unittest

import numpy as np

from segreg import analysis
from segreg.model import OneBkptSegRegEstimator
from segreg.bootstrap import resampling


class TestResampling(unittest.TestCase):

    # TODO: duped from datasets
    def setUp(self):
        """
        Example from: David Hinkley, 'Inference in Two-Phase Regression', 1971
        """
        self._indep = np.array([2.0,
                                2.52288,
                                3.0,
                                3.52288,
                                4.0,
                                4.52288,
                                5.0,
                                5.52288,
                                6.0,
                                6.52288,
                                7.0,
                                7.52288,
                                8.0,
                                8.52288,
                                9.0])
        self._dep = np.array([0.370483,
                              .537970,
                              .607684,
                              .723323,
                              .761856,
                              .892063,
                              .956707,
                              .940349,
                              .898609,
                              .953850,
                              .990834,
                              .890291,
                              .990779,
                              1.050865,
                              .982785])

    def test_semi_parametric_bootstrap_resample(self):
        estimator = OneBkptSegRegEstimator(num_end_to_skip=0)

        seed = 342343
        num_iter = 10

        resample_cases = True
        params_arr = resampling.boot_param_dist(self._indep,
                                                self._dep,
                                                estimator,
                                                num_iter,
                                                resample_cases=resample_cases,
                                                seed=seed,
                                                verbose=False)

        expected = np.array([[4.663045880781e+00, 9.092036748601e-01, 1.994605454545e-01, 1.906437121668e-02, 4.859911840769e-02],
                             [4.197719760840e+00, 9.085099339168e-01, 2.385258748803e-01, 2.130573498099e-02, 3.399214694145e-02],
                             [4.772911728460e+00, 9.506374837316e-01, 1.925103098210e-01, 2.672756381143e-02, 1.602571410528e-02],
                             [4.684152834668e+00, 9.311480417797e-01, 2.029772609044e-01, 1.106781463511e-02, 4.018413194015e-02],
                             [4.657515333315e+00, 9.016006140562e-01, 1.788527879114e-01, 2.347108893531e-02, 2.943247616816e-02],
                             [4.699663518513e+00, 9.105060660894e-01, 1.824126675224e-01, 2.147946656316e-02, 3.881545483262e-02],
                             [5.000000000000e+00, 9.147090733425e-01, 1.522757171638e-01, 2.261830577785e-02, 4.174316098543e-02],
                             [4.603863660097e+00, 8.960950968498e-01, 1.785171562506e-01, 1.832187774830e-02, 3.945264620571e-02],
                             [5.000000000000e+00, 9.424308766929e-01, 1.655693435840e-01, 1.469128875129e-02, 1.145257386852e-02],
                             [5.000000000000e+00, 9.334019927411e-01, 1.613030508095e-01, 3.806604145486e-03, 2.821932932183e-02]])

        close = np.allclose(expected, params_arr)
        self.assertTrue(close)

    #@unittest.skip("skipping")
    def test_semi_parametric_bootstrap(self):
        estimator = OneBkptSegRegEstimator()

        seed = 342343
        num_iter = 10

        resample_cases = False
        params_arr = resampling.boot_param_dist(self._indep,
                                                self._dep,
                                                estimator,
                                                num_iter,
                                                resample_cases=resample_cases,
                                                seed=seed,
                                                verbose=False)

        expected = np.array([[4.098564506022e+00, 8.565612919352e-01, 2.453913681664e-01, 3.236303923697e-02, 4.400187239635e-02],
                             [4.536010040046e+00, 9.110783139309e-01, 2.049730093852e-01, 2.288898315948e-02, 3.513068941465e-02],
                             [4.522880000000e+00, 9.201430002557e-01, 1.889503581583e-01, 2.274854567541e-02, 2.354278444337e-02],
                             [4.107613693420e+00, 8.759506862199e-01, 2.355757488474e-01, 2.502096633088e-02, 3.338913637944e-02],
                             [4.522880000000e+00, 8.904256854168e-01, 1.799798725057e-01, 2.269492276102e-02, 2.515728863654e-02],
                             [5.000000000000e+00, 9.570526600678e-01, 1.749727935147e-01, -7.429417737916e-04, 2.856097966763e-02],
                             [5.000000000000e+00, 9.515742411434e-01, 1.844283004179e-01, 7.140609145948e-03, 3.829145673279e-02],
                             [4.787158417064e+00, 8.930998536077e-01, 1.683336956008e-01, 2.444209758306e-02, 3.856400912024e-02],
                             [4.713071121737e+00, 9.342891745366e-01, 1.972022023895e-01, 1.844140087943e-02, 1.642522710201e-02],
                             [4.271845606595e+00, 8.844255725031e-01, 2.293782972599e-01, 2.440943146911e-02, 3.450330836177e-02]])

        close = np.allclose(expected, params_arr)
        self.assertTrue(close)


if __name__ == "__main__":
    unittest.main()
