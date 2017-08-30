import unittest
import os.path
import numpy as np
from dectree.codegen import VECTORIZE_PROP, ExprGen
from dectree.transpiler import transpile
from io import StringIO


# http://numba.pydata.org/numba-doc/dev/user/jitclass.html


def get_src(no1='false()', a='a', p1='P1', b='b', no2='NO'):
    code = \
        """
        types:

            P1:
                LOW: inv_ramp()
                HIGH: ramp()

            P2:
                "YES": true()
                "NO": {no1}

        inputs:
            - {a}: {p1}

        outputs:
            - b: P2

        rules:
            -
                - if a == LOW:
                    - {b} = {no2}
                - else:
                    - b = YES
        """

    return code.format(a=a, b=b, p1=p1, no1=no1, no2=no2)


class TranspileTest(unittest.TestCase):
    def test_transpile_success(self):
        src_file = StringIO(get_src())
        out_file = StringIO()
        transpile(src_file, out_file=out_file)
        self.assertIsNotNone(out_file.getvalue())

    def test_transpile_failures(self):
        src_file = StringIO("")
        with self.assertRaises(ValueError) as cm:
            transpile(src_file)
        self.assertEqual(str(cm.exception), 'Empty decision tree definition')

        src_file = StringIO("")
        out_file = StringIO()
        with self.assertRaises(ValueError) as cm:
            transpile(src_file, out_file=out_file)
        self.assertEqual(str(cm.exception), 'Empty decision tree definition')

        src_file = StringIO("types:\n    ")
        out_file = StringIO()
        with self.assertRaises(ValueError) as cm:
            transpile(src_file, out_file=out_file)
        self.assertEqual(str(cm.exception), "Invalid decision tree definition: missing section "
                                            "('types', 'inputs', 'outputs', 'rules') or all of them")

        src_file = StringIO("types: null\ninputs: null\noutputs: null\nrules: null\n")
        out_file = StringIO()
        with self.assertRaises(ValueError) as cm:
            transpile(src_file, out_file=out_file)
        self.assertEqual(str(cm.exception), "Invalid decision tree definition: section 'types' is empty")

        src_file = StringIO(get_src(a='u'))
        out_file = StringIO()
        with self.assertRaises(ValueError) as cm:
            transpile(src_file, out_file=out_file)
        self.assertEqual(str(cm.exception), 'Variable "a" is undefined')

        src_file = StringIO(get_src(b='u'))
        out_file = StringIO()
        with self.assertRaises(ValueError) as cm:
            transpile(src_file, out_file=out_file)
        self.assertEqual(str(cm.exception), 'Variable "u" is undefined')

        src_file = StringIO(get_src(no1='false'))
        out_file = StringIO()
        with self.assertRaises(ValueError) as cm:
            transpile(src_file, out_file=out_file)
        self.assertEqual(str(cm.exception), 'Illegal value for property "NO" of type "P2": False')

        src_file = StringIO(get_src(p1='Radiance'))
        out_file = StringIO()
        with self.assertRaises(ValueError) as cm:
            transpile(src_file, out_file=out_file)
        self.assertEqual(str(cm.exception), 'Type "Radiance" of variable "a" is undefined')

        src_file = StringIO(get_src(no2='Radiance'))
        out_file = StringIO()
        with self.assertRaises(ValueError) as cm:
            transpile(src_file, out_file=out_file)
        self.assertEqual(str(cm.exception), '"Radiance" is not a property of type "P2" of variable "b"')

    def test_transpile_with_defaults(self):
        src_file = os.path.join(os.path.dirname(__file__), 'dectree_test.yml')
        out_file = os.path.join(os.path.dirname(__file__), 'dectree_test.py')
        if os.path.exists(out_file):
            os.remove(out_file)
        transpile(src_file)
        self.assertTrue(os.path.exists(out_file))
        m = __import__('test.dectree_test')
        self.assertTrue(hasattr(m, 'dectree_test'))
        self.assertTrue(hasattr(m.dectree_test, 'Input'))
        self.assertTrue(hasattr(m.dectree_test, 'Output'))
        self.assertTrue(hasattr(m.dectree_test, 'apply_rules'))
        inputs = m.dectree_test.Input()
        outputs = m.dectree_test.Output()

        inputs.glint = 0.2
        inputs.radiance = 60.
        m.dectree_test.apply_rules(inputs, outputs)
        self.assertAlmostEqual(outputs.cloudy, 0.6)
        self.assertAlmostEqual(outputs.certain, 1.0)

    def test_transpile_parameterized(self):
        src_file = os.path.join(os.path.dirname(__file__), 'dectree_test.yml')
        out_file = os.path.join(os.path.dirname(__file__), 'dectree_test_p.py')
        if os.path.exists(out_file):
            os.remove(out_file)
        transpile(src_file, out_file=out_file, parameterize=True)
        self.assertTrue(os.path.exists(out_file))
        m = __import__('test.dectree_test_p')
        self.assertTrue(hasattr(m, 'dectree_test_p'))
        self.assertTrue(hasattr(m.dectree_test_p, 'Input'))
        self.assertTrue(hasattr(m.dectree_test_p, 'Output'))
        self.assertTrue(hasattr(m.dectree_test_p, 'Params'))
        self.assertTrue(hasattr(m.dectree_test_p, 'apply_rules'))
        inputs = m.dectree_test_p.Input()
        outputs = m.dectree_test_p.Output()
        params = m.dectree_test_p.Params()

        inputs.glint = 0.2
        inputs.radiance = 60.
        m.dectree_test_p.apply_rules(inputs, outputs, params)
        self.assertAlmostEqual(outputs.cloudy, 0.6)
        self.assertAlmostEqual(outputs.certain, 1.0)

    def test_transpile_vectorized(self):
        src_file = os.path.join(os.path.dirname(__file__), 'dectree_test.yml')
        out_file = os.path.join(os.path.dirname(__file__), 'dectree_test_v.py')
        if os.path.exists(out_file):
            os.remove(out_file)
        transpile(src_file, out_file=out_file, vectorize=VECTORIZE_PROP)
        self.assertTrue(os.path.exists(out_file))
        m = __import__('test.dectree_test_v')
        self.assertTrue(hasattr(m, 'dectree_test_v'))
        self.assertTrue(hasattr(m.dectree_test_v, 'Input'))
        self.assertTrue(hasattr(m.dectree_test_v, 'Output'))
        self.assertTrue(hasattr(m.dectree_test_v, 'apply_rules'))
        input = m.dectree_test_v.Input()
        output = m.dectree_test_v.Output()

        input.glint = np.array([0.2, 0.3])
        input.radiance = np.array([60.0, 10.0])
        m.dectree_test_v.apply_rules(input, output)
        np.testing.assert_almost_equal(output.cloudy, np.array([0.6, 0.0]))
        np.testing.assert_almost_equal(output.certain, np.array([1.0, 1.0]))


def eval_func(f, x):
    body = f()
    code_lines = ["def y(x):"] + list(map(lambda l: '    ' + l, body.split('\n')))
    code = '\n'.join(code_lines)
    local_vars = {}
    exec(code, None, local_vars)
    y = local_vars['y']
    return y(x)


class ConditionTranspilerTest(unittest.TestCase):
    def test_transpile_success(self):
        type_defs = dict(
            XType=dict(HI=('ramp()', dict(x1=0.5, x2=1.0), ''),
                       LO=('inv_ramp()', dict(x1=0.0, x2=0.5), '')),
            YType=dict(FAST=('true()', {}, ''),
                       SLOW=('false()', {}, ''))
        )
        var_defs = dict(x='XType', y='YType')
        options = dict()
        transpiler = ExprGen(type_defs, var_defs, options)
        self.assertEqual(transpiler.gen_expr('y == FAST'),
                         '_YType_FAST(input.y)')
        self.assertEqual(transpiler.gen_expr('x != HI'),
                         '1.0 - (_XType_HI(input.x))')
        self.assertEqual(transpiler.gen_expr('y is FAST'),
                         '_YType_FAST(input.y)')
        self.assertEqual(transpiler.gen_expr('x is not HI'),
                         '1.0 - (_XType_HI(input.x))')
        self.assertEqual(transpiler.gen_expr('x != HI and y == SLOW'),
                         'min(1.0 - (_XType_HI(input.x)), _YType_SLOW(input.y))')
        self.assertEqual(transpiler.gen_expr('x != HI or y == SLOW'),
                         'max(1.0 - (_XType_HI(input.x)), _YType_SLOW(input.y))')
        self.assertEqual(transpiler.gen_expr('x != HI or y == SLOW or y == FAST'),
                         'max(max(1.0 - (_XType_HI(input.x)), _YType_SLOW(input.y)), _YType_FAST(input.y))')
        self.assertEqual(transpiler.gen_expr('x == HI or not y != SLOW'),
                         'max(_XType_HI(input.x), 1.0 - (1.0 - (_YType_SLOW(input.y))))')
        self.assertEqual(transpiler.gen_expr('x == HI and not (y == FAST or x == LO)'),
                         'min(_XType_HI(input.x), 1.0 - (max(_YType_FAST(input.y), _XType_LO(input.x))))')

        options = dict(vectorize=VECTORIZE_PROP)
        transpiler = ExprGen(type_defs, var_defs, options)
        self.assertEqual(transpiler.gen_expr('x == HI and not (y == FAST or x == LO)'),
                         'np.minimum(_XType_HI(input.x), 1.0 - (np.maximum(_YType_FAST(input.y), '
                         '_XType_LO(input.x))))')

        options = dict(parameterize=True)
        transpiler = ExprGen(type_defs, var_defs, options)
        self.assertEqual(transpiler.gen_expr('x == HI and not (y == FAST or x == LO)'),
                         'min(_XType_HI(input.x, x1=params.XType_HI_x1, x2=params.XType_HI_x2), '
                         '1.0 - (max(_YType_FAST(input.y), '
                         '_XType_LO(input.x, x1=params.XType_LO_x1, x2=params.XType_LO_x2))))')

        options = dict(vectorize=VECTORIZE_PROP, parameterize=True)
        transpiler = ExprGen(type_defs, var_defs, options)
        self.assertEqual(transpiler.gen_expr('x == HI and not (y == FAST or x == LO)'),
                         'np.minimum(_XType_HI(input.x, x1=params.XType_HI_x1, x2=params.XType_HI_x2), '
                         '1.0 - (np.maximum(_YType_FAST(input.y), '
                         '_XType_LO(input.x, x1=params.XType_LO_x1, x2=params.XType_LO_x2))))')

    def test_transpile_failure(self):
        type_defs = dict(
            XType=dict(HI=('ramp()', dict(x1=0.5, x2=1.0), ''),
                       LO=('inv_ramp()', dict(x1=0.0, x2=0.5), '')),
            YType=dict(FAST=('true()', {}, ''),
                       SLOW=('false()', {}, ''))
        )
        var_defs = dict(x='XType', y='YType')
        options = dict()
        expr_gen = ExprGen(type_defs, var_defs, options)

        with self.assertRaises(ValueError) as cm:
            expr_gen.gen_expr('for i in range(3): pass')
        self.assertEqual(str(cm.exception), 'Invalid condition expression: [for i in range(3): pass]')

        with self.assertRaises(ValueError) as cm:
            expr_gen.gen_expr('7 < x')
        self.assertEqual(str(cm.exception), 'Left side of comparison must be the name of an input')

        with self.assertRaises(ValueError) as cm:
            expr_gen.gen_expr('x <= HI')
        self.assertEqual(str(cm.exception),
                         '"==", "!=", "is", and "is not" are the only supported comparison operators')

        with self.assertRaises(ValueError) as cm:
            expr_gen.gen_expr('-(x == HI)')
        self.assertEqual(str(cm.exception), '"not" is the only supported unary operator')

        with self.assertRaises(ValueError) as cm:
            expr_gen.gen_expr('x + HI')
        self.assertTrue(str(cm.exception) != '"and" and "or" are the only supported binary operators')

        with self.assertRaises(ValueError) as cm:
            expr_gen.gen_expr('max(x, HI)')
        self.assertEqual(str(cm.exception), 'Unsupported expression')

        with self.assertRaises(SyntaxError) as cm:
            expr_gen.gen_expr('x && HI')
        self.assertTrue(str(cm.exception) != '')
