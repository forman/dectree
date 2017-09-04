import ast
import unittest

from dectree.decompiler import ExprDecompiler


class AstTest(unittest.TestCase):
    def test_unary(self):
        self.assertEqual(transform_expr('-(-+(-x))'), '--+-x')
        self.assertEqual(transform_expr('-((a-b)-c)'), '-(a - b - c)')
        self.assertEqual(transform_expr('not x'), 'not x')
        self.assertEqual(transform_expr('not -x'), 'not -x')
        self.assertEqual(transform_expr('not not x'), 'not not x')

    def test_binary(self):
        self.assertEqual(transform_expr('a-b-c-d'), 'a - b - c - d')
        self.assertEqual(transform_expr('(a-b)-c-d'), 'a - b - c - d')
        self.assertEqual(transform_expr('(a-b-c)-d'), 'a - b - c - d')
        self.assertEqual(transform_expr('a-(b-c)-d'), 'a - (b - c) - d')
        self.assertEqual(transform_expr('a-(b-c-d)'), 'a - (b - c - d)')
        self.assertEqual(transform_expr('a-b-(c-d)'), 'a - b - (c - d)')
        self.assertEqual(transform_expr('a-(b-(c-d))'), 'a - (b - (c - d))')
        self.assertEqual(transform_expr('(a-(b-c))-d'), 'a - (b - c) - d')

        self.assertEqual(transform_expr('a----b'), 'a - ---b')
        self.assertEqual(transform_expr('---a-b'), '---a - b')

        self.assertEqual(transform_expr('a*b+c/d'), 'a * b + c / d')
        self.assertEqual(transform_expr('a+b*c-d'), 'a + b * c - d')
        self.assertEqual(transform_expr('(a+b)*(c-d)'), '(a + b) * (c - d)')

    def test_bool(self):
        self.assertEqual(transform_expr('a and b and c'), 'a and b and c')
        self.assertEqual(transform_expr('(a and b) and c'), 'a and b and c')
        self.assertEqual(transform_expr('a and (b and c)'), 'a and (b and c)')

        self.assertEqual(transform_expr('a and b or c and d'), 'a and b or c and d')
        self.assertEqual(transform_expr('a or b and c or d'), 'a or b and c or d')
        self.assertEqual(transform_expr('(a or b) and (c or d)'), '(a or b) and (c or d)')
        self.assertEqual(transform_expr('(a or b) and not (c or not d)'), '(a or b) and not (c or not d)')

    def test_compare(self):
        self.assertEqual(transform_expr('a < 2'), 'a < 2')
        self.assertEqual(transform_expr('a >= 2 == b'), 'a >= 2 == b')
        self.assertEqual(transform_expr('0 < x <= 1'), '0 < x <= 1')
        self.assertEqual(transform_expr('a is not Null'), 'a is not Null')
        self.assertEqual(transform_expr('a in data'), 'a in data')

    def test_mixed(self):
        self.assertEqual(transform_expr('a+sin(x + 2.8)'), 'a + sin(x + 2.8)')
        self.assertEqual(transform_expr('a+max(1, sin(x+2.8), x**0.5)'), 'a + max(1, sin(x + 2.8), x ** 0.5)')


def transform_expr(expr: str) -> str:
    decompiler = ExprDecompiler()
    return decompiler._transform(ast.parse(expr))


