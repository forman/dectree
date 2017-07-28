import unittest

import dectree.propfuncs as pf
import numpy as np


class GeneratorsTest(unittest.TestCase):
    def test_true(self):
        self.run_func_tests(pf.true, [(-0.25, 1.0),
                                      (+0.00, 1.0),
                                      (+0.25, 1.0),
                                      (+0.50, 1.0),
                                      (+0.75, 1.0),
                                      (+1.00, 1.0),
                                      (+1.25, 1.0)])

    def test_false(self):
        self.run_func_tests(pf.false, [(-0.25, 0.0),
                                       (+0.00, 0.0),
                                       (+0.25, 0.0),
                                       (+0.50, 0.0),
                                       (+0.75, 0.0),
                                       (+1.00, 0.0),
                                       (+1.25, 0.0)])

    def test_ramp_down(self):
        self.run_func_tests(pf.ramp_down, [(-0.25, 1.0),
                                           (+0.00, 1.0),
                                           (+0.25, 0.5),
                                           (+0.50, 0.0),
                                           (+0.75, 0.0),
                                           (+1.00, 0.0),
                                           (+1.25, 0.0)])

    def test_ramp_up(self):
        self.run_func_tests(pf.ramp_up, [(-0.25, 0.0),
                                         (+0.00, 0.0),
                                         (+0.25, 0.0),
                                         (+0.50, 0.0),
                                         (+0.75, 0.5),
                                         (+1.00, 1.0),
                                         (+1.25, 1.0)])

    def test_triangle(self):
        self.run_func_tests(pf.triangle, [(-0.25, 0.0),
                                          (+0.00, 0.0),
                                          (+0.25, 0.5),
                                          (+0.50, 1.0),
                                          (+0.75, 0.5),
                                          (+1.00, 0.0),
                                          (+1.25, 0.0)])

    def run_func_tests(self, f, points):
        g1, g1_code = gen_func(f)
        for i in range(len(points)):
            x, y = points[i]
            y_actual = g1(x)
            self.assertAlmostEqual(y_actual, y,
                                   msg='at index={}: point=(x={}, y={}), got y={} instead\n{}'.format(i, x, y, y_actual,
                                                                                                      g1_code))

        g2, g2_code = gen_func(f, vectorize=True)
        x, y = zip(*points)
        x = np.array(x, dtype=np.float64)
        y_actual = g2(x)
        np.testing.assert_array_almost_equal(y_actual, y, err_msg=g2_code)


def gen_func(f, vectorize=False):
    func_params, func_body_pattern = f()
    func_body = func_body_pattern.format(**func_params)
    code_lines = []
    if vectorize:
        code_lines.append('from numba import vectorize, float64')
        code_lines.append('@vectorize([float64(float64)])')
    code_lines.append("def g(x):")
    code_lines.extend(map(lambda l: '    ' + l, func_body.split('\n')))
    code = '\n'.join(code_lines)
    locals = {}
    exec(code, None, locals)
    g = locals['g']
    return g, code
