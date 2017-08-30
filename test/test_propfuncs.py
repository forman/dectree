import unittest

import numpy as np

import dectree.propfuncs as pf


class GeneratorsTest(unittest.TestCase):
    def test_true(self):
        self.run_func_tests(pf.true, {}, [(-0.25, 1.0),
                                          (+0.00, 1.0),
                                          (+0.25, 1.0),
                                          (+0.50, 1.0),
                                          (+0.75, 1.0),
                                          (+1.00, 1.0),
                                          (+1.25, 1.0)])

    def test_false(self):
        self.run_func_tests(pf.false, {}, [(-0.25, 0.0),
                                           (+0.00, 0.0),
                                           (+0.25, 0.0),
                                           (+0.50, 0.0),
                                           (+0.75, 0.0),
                                           (+1.00, 0.0),
                                           (+1.25, 0.0)])

    def test_eq(self):
        self.run_func_tests(pf.eq, dict(x0=0.5), [(-0.25, 0.0),
                                                  (+0.00, 0.0),
                                                  (+0.25, 0.0),
                                                  (+0.50, 1.0),
                                                  (+0.75, 0.0),
                                                  (+1.00, 0.0),
                                                  (+1.25, 0.0)])

    def test_eq_dx(self):
        self.run_func_tests(pf.eq, dict(x0=0.5, dx=0.4), [(-0.25, 0.0),
                                                          (+0.00, 0.0),
                                                          (+0.30, 0.5),
                                                          (+0.50, 1.0),
                                                          (+0.70, 0.5),
                                                          (+1.00, 0.0),
                                                          (+1.25, 0.0)])

    def test_ne(self):
        self.run_func_tests(pf.ne, dict(x0=0.5), [(-0.25, 1.0),
                                                  (+0.00, 1.0),
                                                  (+0.25, 1.0),
                                                  (+0.50, 0.0),
                                                  (+0.75, 1.0),
                                                  (+1.00, 1.0),
                                                  (+1.25, 1.0)])

    def test_ne_dx(self):
        self.run_func_tests(pf.ne, dict(x0=0.5, dx=0.4), [(-0.25, 1.0),
                                                          (+0.00, 1.0),
                                                          (+0.30, 0.5),
                                                          (+0.50, 0.0),
                                                          (+0.70, 0.5),
                                                          (+1.00, 1.0),
                                                          (+1.25, 1.0)])

    def test_gt(self):
        self.run_func_tests(pf.gt, dict(x0=0.5), [(-0.25, 0.0),
                                                  (+0.00, 0.0),
                                                  (+0.25, 0.0),
                                                  (+0.50, 0.0),
                                                  (+0.75, 1.0),
                                                  (+1.00, 1.0),
                                                  (+1.25, 1.0)])

    def test_ge(self):
        self.run_func_tests(pf.ge, dict(x0=0.5), [(-0.25, 0.0),
                                                  (+0.00, 0.0),
                                                  (+0.25, 0.0),
                                                  (+0.50, 1.0),
                                                  (+0.75, 1.0),
                                                  (+1.00, 1.0),
                                                  (+1.25, 1.0)])

    def test_gt_ge_dx(self):
        f_params = dict(x0=0.5, dx=0.4)
        points = [(-0.25, 0.0),
                  (+0.00, 0.0),
                  (+0.30, 0.25),
                  (+0.50, 0.5),
                  (+0.70, 0.75),
                  (+1.00, 1.0),
                  (+1.25, 1.0)]
        self.run_func_tests(pf.gt, f_params, points)
        self.run_func_tests(pf.ge, f_params, points)

    def test_lt(self):
        self.run_func_tests(pf.lt, dict(x0=0.5), [(-0.25, 1.0),
                                                  (+0.00, 1.0),
                                                  (+0.25, 1.0),
                                                  (+0.50, 0.0),
                                                  (+0.75, 0.0),
                                                  (+1.00, 0.0),
                                                  (+1.25, 0.0)])

    def test_le(self):
        self.run_func_tests(pf.le, dict(x0=0.5), [(-0.25, 1.0),
                                                  (+0.00, 1.0),
                                                  (+0.25, 1.0),
                                                  (+0.50, 1.0),
                                                  (+0.75, 0.0),
                                                  (+1.00, 0.0),
                                                  (+1.25, 0.0)])

    def test_lt_le_dx(self):
        f_params = dict(x0=0.5, dx=0.4)
        points = [(-0.25, 1.0),
                  (+0.00, 1.0),
                  (+0.30, 0.75),
                  (+0.50, 0.5),
                  (+0.70, 0.25),
                  (+1.00, 0.0),
                  (+1.25, 0.0)]
        self.run_func_tests(pf.lt, f_params, points)
        self.run_func_tests(pf.le, f_params, points)

    def test_ramp(self):
        self.run_func_tests(pf.ramp, {}, [(-0.25, 0.0),
                                          (+0.00, 0.0),
                                          (+0.25, 0.25),
                                          (+0.50, 0.5),
                                          (+0.75, 0.75),
                                          (+1.00, 1.0),
                                          (+1.25, 1.0)])

    def test_inv_ramp(self):
        self.run_func_tests(pf.inv_ramp, {}, [(-0.25, 1.0),
                                              (+0.00, 1.0),
                                              (+0.25, 0.75),
                                              (+0.50, 0.5),
                                              (+0.75, 0.25),
                                              (+1.00, 0.0),
                                              (+1.25, 0.0)])

    def test_triangular(self):
        self.run_func_tests(pf.triangular, {}, [(-0.25, 0.0),
                                                (+0.00, 0.0),
                                                (+0.25, 0.5),
                                                (+0.50, 1.0),
                                                (+0.75, 0.5),
                                                (+1.00, 0.0),
                                                (+1.25, 0.0)])

    def test_inv_triangular(self):
        self.run_func_tests(pf.inv_triangular, {}, [(-0.25, 1.0),
                                                    (+0.00, 1.0),
                                                    (+0.25, 0.5),
                                                    (+0.50, 0.0),
                                                    (+0.75, 0.5),
                                                    (+1.00, 1.0),
                                                    (+1.25, 1.0)])

    def test_trapezoid(self):
        self.run_func_tests(pf.trapezoid, {}, [(-0.25, 0.0),
                                               (+0.00, 0.0),
                                               (+0.25, 0.75),
                                               (+0.35, 1.0),
                                               (+0.50, 1.0),
                                               (+0.65, 1.0),
                                               (+0.75, 0.75),
                                               (+1.00, 0.0),
                                               (+1.25, 0.0)])

    def test_inv_trapezoid(self):
        self.run_func_tests(pf.inv_trapezoid, {}, [(-0.25, 1.0),
                                                   (+0.00, 1.0),
                                                   (+0.25, 0.25),
                                                   (+0.35, 0.0),
                                                   (+0.50, 0.0),
                                                   (+0.65, 0.0),
                                                   (+0.75, 0.25),
                                                   (+1.00, 1.0),
                                                   (+1.25, 1.0)])

    def run_func_tests(self, f, f_params, points):
        msg = 'at index={}: point=(x={}, y={}), got y={} instead\n{}'
        g1, g1_code = gen_func(f, f_params)
        for i in range(len(points)):
            x, y = points[i]
            y_actual = g1(x)
            self.assertAlmostEqual(y_actual, y,
                                   msg=msg.format(i, x, y, y_actual, g1_code))

        g2, g2_code = gen_func(f, f_params, vectorize=True)
        x, y = zip(*points)
        x = np.array(x, dtype=np.float64)
        y_actual = g2(x)
        np.testing.assert_array_almost_equal(y_actual, y, err_msg=g2_code)


def gen_func(f, f_params, vectorize=False):
    func_params, func_body_pattern = f(**f_params)
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
