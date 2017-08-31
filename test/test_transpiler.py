import unittest
import os.path
import numpy as np
from dectree.codegen import VECTORIZE_PROP, ExprGen
from dectree.transpiler import transpile, compile
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
        self.assertTrue(hasattr(m.dectree_test, 'Inputs'))
        self.assertTrue(hasattr(m.dectree_test, 'Outputs'))
        self.assertTrue(hasattr(m.dectree_test, 'apply_rules'))
        inputs = m.dectree_test.Inputs()
        outputs = m.dectree_test.Outputs()

        inputs.glint = 0.2
        inputs.radiance = 60.
        m.dectree_test.apply_rules(inputs, outputs)
        self.assertAlmostEqual(outputs.cloudy, 0.6)
        self.assertAlmostEqual(outputs.certain, 1.0)

    def test_compile_with_defaults(self):
        src_file = os.path.join(os.path.dirname(__file__), 'dectree_test.yml')
        apply_rules, Inputs, Outputs = compile(src_file)
        self.assertIsNotNone(apply_rules)
        self.assertIsNotNone(Inputs)
        self.assertIsNotNone(Outputs)
        inputs = Inputs()
        outputs = Outputs()

        inputs.glint = 0.2
        inputs.radiance = 60.
        apply_rules(inputs, outputs)
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
        self.assertTrue(hasattr(m.dectree_test_p, 'Inputs'))
        self.assertTrue(hasattr(m.dectree_test_p, 'Outputs'))
        self.assertTrue(hasattr(m.dectree_test_p, 'Params'))
        self.assertTrue(hasattr(m.dectree_test_p, 'apply_rules'))
        inputs = m.dectree_test_p.Inputs()
        outputs = m.dectree_test_p.Outputs()
        params = m.dectree_test_p.Params()

        inputs.glint = 0.2
        inputs.radiance = 60.
        m.dectree_test_p.apply_rules(inputs, outputs, params)
        self.assertAlmostEqual(outputs.cloudy, 0.6)
        self.assertAlmostEqual(outputs.certain, 1.0)

    def test_compile_parameterized(self):
        src_file = os.path.join(os.path.dirname(__file__), 'dectree_test.yml')
        apply_rules, Inputs, Outputs, Params = compile(src_file, parameterize=True)
        self.assertIsNotNone(apply_rules)
        self.assertIsNotNone(Inputs)
        self.assertIsNotNone(Outputs)
        self.assertIsNotNone(Params)
        inputs = Inputs()
        outputs = Outputs()
        params = Params()

        inputs.glint = 0.2
        inputs.radiance = 60.
        apply_rules(inputs, outputs, params)
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
        self.assertTrue(hasattr(m.dectree_test_v, 'Inputs'))
        self.assertTrue(hasattr(m.dectree_test_v, 'Outputs'))
        self.assertTrue(hasattr(m.dectree_test_v, 'apply_rules'))
        inputs = m.dectree_test_v.Inputs()
        outputs = m.dectree_test_v.Outputs()

        inputs.glint = np.array([0.2, 0.3])
        inputs.radiance = np.array([60.0, 10.0])
        m.dectree_test_v.apply_rules(inputs, outputs)
        np.testing.assert_almost_equal(outputs.cloudy, np.array([0.6, 0.0]))
        np.testing.assert_almost_equal(outputs.certain, np.array([1.0, 1.0]))


def eval_func(f, x):
    body = f()
    code_lines = ["def y(x):"] + list(map(lambda l: '    ' + l, body.split('\n')))
    code = '\n'.join(code_lines)
    local_vars = {}
    exec(code, None, local_vars)
    y = local_vars['y']
    return y(x)

