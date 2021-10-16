"""
Sanity check segmented regression routines by comparing results to those in an
academic paper of Hinkley.
"""

# Author: Steven Lillywhite
# License: BSD 3 clause

import unittest

import numpy as np

from segreg.model import OneBkptSegRegEstimator
from segreg import data


class HinkleyTieout(unittest.TestCase):
    """
    This ties out with published data and estimation results in:

    Inference in Two-Phase Regression
    David Hinkley
    Journal of the American Statistical Association
    Vol. 66, No. 336 (Dec., 1971), pp. 736-743
    """

    def setUp(self):
        self._indep, self._dep = data.hinkley()

    def test_segmented_regression(self):
        estimator = OneBkptSegRegEstimator(no_bias_variance=True)

        est_params = estimator.fit(self._indep, self._dep)

        # Hinkley does not seem to provide estimate for m2
        # these are [u,v,m1,sigma**2]
        expected = [4.652, 0.9180, 0.1935, 0.00166]

        computed = est_params[[0, 1, 2, 4]]
        computed[-1] = computed[-1] ** 2

        # Hinkley prints results to various precision
        computed[0] = round(computed[0], 3)
        computed[1] = round(computed[1], 4)
        computed[2] = round(computed[2], 4)
        computed[3] = round(computed[3], 5)

        close = np.allclose(expected, computed)
        self.assertTrue(close)

    def test_restricted_segmented_regression(self):
        estimator = OneBkptSegRegEstimator(no_bias_variance=True,
                                           restrict_rhs_slope=0)

        est_params = estimator.fit(self._indep, self._dep)

        # Hinkley does not seem to provide estimate for m2
        # these are [u,v,m1,sigma**2]
        expected = [4.878, 0.9617, 0.1935, 0.00196]

        computed = est_params[[0, 1, 2, 4]]
        computed[-1] = computed[-1] ** 2

        # Hinkley prints results to various precision
        computed[0] = round(computed[0], 3)
        computed[1] = round(computed[1], 4)
        computed[2] = round(computed[2], 4)
        computed[3] = round(computed[3], 5)

        close = np.allclose(expected, computed)
        self.assertTrue(close)


if __name__ == "__main__":
    unittest.main()
