import unittest

from dectree.codegen import VECTORIZE_PROP, ExprGen, _get_effective_op_pattern


class ExprGenTest(unittest.TestCase):
    def test_success(self):
        type_defs = dict(
            XType=dict(HI=('ramp()', dict(x1=0.5, x2=1.0), ''),
                       LO=('inv_ramp()', dict(x1=0.0, x2=0.5), '')),
            YType=dict(FAST=('true()', {}, ''),
                       SLOW=('false()', {}, ''))
        )
        var_defs = dict(x='XType', y='YType')
        transpiler = ExprGen(type_defs, var_defs)
        self.assertEqual(transpiler.gen_expr('y == FAST'),
                         '_YType_FAST(inputs.y)')
        self.assertEqual(transpiler.gen_expr('x != HI'),
                         '1.0 - (_XType_HI(inputs.x))')
        self.assertEqual(transpiler.gen_expr('y is FAST'),
                         '_YType_FAST(inputs.y)')
        self.assertEqual(transpiler.gen_expr('x is not HI'),
                         '1.0 - (_XType_HI(inputs.x))')
        self.assertEqual(transpiler.gen_expr('x != HI and y == SLOW'),
                         'min(1.0 - (_XType_HI(inputs.x)), _YType_SLOW(inputs.y))')
        self.assertEqual(transpiler.gen_expr('x != HI or y == SLOW'),
                         'max(1.0 - (_XType_HI(inputs.x)), _YType_SLOW(inputs.y))')
        self.assertEqual(transpiler.gen_expr('x != HI or y == SLOW or y == FAST'),
                         'max(max(1.0 - (_XType_HI(inputs.x)), _YType_SLOW(inputs.y)), _YType_FAST(inputs.y))')
        self.assertEqual(transpiler.gen_expr('x == HI or not y != SLOW'),
                         'max(_XType_HI(inputs.x), 1.0 - (1.0 - (_YType_SLOW(inputs.y))))')
        self.assertEqual(transpiler.gen_expr('x == HI and not (y == FAST or x == LO)'),
                         'min(_XType_HI(inputs.x), 1.0 - (max(_YType_FAST(inputs.y), _XType_LO(inputs.x))))')

        op_patters = dict(not_pattern=_get_effective_op_pattern('1.0 - ({x})', vectorize=VECTORIZE_PROP),
                          and_pattern=_get_effective_op_pattern('min({x}, {y})', vectorize=VECTORIZE_PROP),
                          or_pattern=_get_effective_op_pattern('max({x}, {y})', vectorize=VECTORIZE_PROP))

        transpiler = ExprGen(type_defs, var_defs, vectorize=VECTORIZE_PROP, **op_patters)
        self.assertEqual(transpiler.gen_expr('x == HI and not (y == FAST or x == LO)'),
                         'np.minimum(_XType_HI(inputs.x), 1.0 - (np.maximum(_YType_FAST(inputs.y), '
                         '_XType_LO(inputs.x))))')

        transpiler = ExprGen(type_defs, var_defs, parameterize=True)
        self.assertEqual(transpiler.gen_expr('x == HI and not (y == FAST or x == LO)'),
                         'min(_XType_HI(inputs.x, x1=params.XType_HI_x1, x2=params.XType_HI_x2), '
                         '1.0 - (max(_YType_FAST(inputs.y), '
                         '_XType_LO(inputs.x, x1=params.XType_LO_x1, x2=params.XType_LO_x2))))')

        transpiler = ExprGen(type_defs, var_defs, vectorize=VECTORIZE_PROP, parameterize=True, **op_patters)
        self.assertEqual(transpiler.gen_expr('x == HI and not (y == FAST or x == LO)'),
                         'np.minimum(_XType_HI(inputs.x, x1=params.XType_HI_x1, x2=params.XType_HI_x2), '
                         '1.0 - (np.maximum(_YType_FAST(inputs.y), '
                         '_XType_LO(inputs.x, x1=params.XType_LO_x1, x2=params.XType_LO_x2))))')

    def test_failure(self):
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
