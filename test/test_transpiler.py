import unittest

import os.path

from dectree.transpiler import transpile, Transpiler


# http://numba.pydata.org/numba-doc/dev/user/jitclass.html

class TranspilerTest(unittest.TestCase):
    def test_transpile_rule_condition(self):
        type_defs = dict(XType=dict(HI='true()', LO='false()'),
                         YType=dict(FAST='true()', SLOW='false()'))
        var_defs = dict(x='XType', y='YType')
        options = dict()
        self.assertEqual(Transpiler.transpile_rule_condition('y == FAST',
                                                             type_defs, var_defs, options),
                         '_YType_FAST(input.y)')
        self.assertEqual(Transpiler.transpile_rule_condition('x != HI',
                                                             type_defs, var_defs, options),
                         '1.0 - (_XType_HI(input.x))')
        self.assertEqual(Transpiler.transpile_rule_condition('y is FAST',
                                                             type_defs, var_defs, options),
                         '_YType_FAST(input.y)')
        self.assertEqual(Transpiler.transpile_rule_condition('x is not HI',
                                                             type_defs, var_defs, options),
                         '1.0 - (_XType_HI(input.x))')
        self.assertEqual(Transpiler.transpile_rule_condition('x != HI and y == SLOW',
                                                             type_defs, var_defs, options),
                         'min(1.0 - (_XType_HI(input.x)), _YType_SLOW(input.y))')
        self.assertEqual(Transpiler.transpile_rule_condition('x != HI or y == SLOW',
                                                             type_defs, var_defs, options),
                         'max(1.0 - (_XType_HI(input.x)), _YType_SLOW(input.y))')
        self.assertEqual(Transpiler.transpile_rule_condition('x != HI or y == SLOW or y == FAST',
                                                             type_defs, var_defs, options),
                         'max(max(1.0 - (_XType_HI(input.x)), _YType_SLOW(input.y)), _YType_FAST(input.y))')
        self.assertEqual(Transpiler.transpile_rule_condition('x == HI or not y != SLOW',
                                                             type_defs, var_defs, options),
                         'max(_XType_HI(input.x), 1.0 - (1.0 - (_YType_SLOW(input.y))))')
        self.assertEqual(Transpiler.transpile_rule_condition('x == HI and not (y == FAST or x == LO)',
                                                             type_defs, var_defs, options),
                         'min(_XType_HI(input.x), 1.0 - (max(_YType_FAST(input.y), _XType_LO(input.x))))')

        options = dict(vectorize=True)
        self.assertEqual(Transpiler.transpile_rule_condition('x == HI and not (y == FAST or x == LO)',
                                                             type_defs, var_defs, options),
                         'np.minimum(_XType_HI(input.x), 1.0 - (np.maximum(_YType_FAST(input.y), _XType_LO(input.x))))')

    def test_write_module(self):
        src_file = os.path.join(os.path.dirname(__file__), 'dc1.yml')
        out_file = os.path.join(os.path.dirname(__file__), 'dc1.py')
        if os.path.exists(out_file):
            os.remove(out_file)
        transpile(src_file)
        self.assertTrue(os.path.exists(out_file))
        m = __import__('test.dc1')
        self.assertTrue(hasattr(m, 'dc1'))
        self.assertTrue(hasattr(m.dc1, 'Input'))
        self.assertTrue(hasattr(m.dc1, 'Output'))
        self.assertTrue(hasattr(m.dc1, 'apply_rules'))
        input = m.dc1.Input()
        output = m.dc1.Output()

        input.glint = 0.2
        input.radiance = 60.
        m.dc1.apply_rules(input, output)
        self.assertAlmostEqual(output.cloudy, 0.8)
        self.assertAlmostEqual(output.certain, 1.0)


def eval_func(f, x):
    body = f()
    code_lines = ["def y(x):"] + list(map(lambda l: '    ' + l, body.split('\n')))
    code = '\n'.join(code_lines)
    locals = {}
    exec(code, None, locals)
    y = locals['y']
    return y(x)
