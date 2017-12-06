import unittest

import numpy as np
from numba import vectorize, float64


@vectorize([float64(float64)])
def _radiance_middle(x):
    # Radiance.MIDDLE: triangle(x1=30, x2=50, x3=100)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 0.0
    if x < 0.5:
        return x / 0.5
    return 1.0 - (x - 0.5) / 0.5


class GeneratorsTest(unittest.TestCase):
    def test_true(self):
        y = _radiance_middle(np.array([-0.25, 0., 0.25, 0.5, 0.75, 1., 1.25]))
        np.testing.assert_array_almost_equal(y, [0., 0., 0.5, 1., 0.5, 0., 0.])
