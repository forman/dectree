import ast
import unittest


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
        self.assertEqual(transform_expr('a >= 2 == b'), 'a < 2 == b')

    def test_mixed(self):
        self.assertEqual(transform_expr('a+sin(x + 2.8)'), 'a + sin(x + 2.8)')
        self.assertEqual(transform_expr('a+max(1, sin(x+2.8), x**0.5)'), 'a + max(1, sin(x + 2.8), x ** 0.5)')


def transform_expr(expr: str) -> str:
    transformer = ExprDecompiler()
    return transformer.transform(ast.parse(expr))


class ExprDecompiler:
    """
    See https://greentreesnakes.readthedocs.io/en/latest/nodes.html#expressions
    """
    _KEYWORDS = {'and', 'or', 'not', 'True', 'False', 'None'}

    _OP_INFOS = {
        ast.Or: ('or', 100, 'L'),
        ast.And: ('and', 200, 'L'),
        ast.Not: ('not', 300, None),
        ast.UAdd: ('+', 500, None),
        ast.USub: ('-', 500, None),
        ast.Add: ('+', 500, 'E'),
        ast.Sub: ('-', 500, 'L'),
        ast.Mult: ('*', 600, 'E'),
        ast.Div: ('/', 600, 'L'),
        ast.FloorDiv: ('//', 600, 'L'),
        ast.Mod: ('%', 600, 'L'),
        ast.Pow: ('**', 700, 'L'),
    }
    @classmethod
    def get_op_info(cls, op: ast.AST):
        return cls._OP_INFOS.get(type(op), (None, None, None))

    def transform(self, node: ast.expr) -> str:
        if isinstance(node, ast.Module):
            return self.transform(node.body[0])
        if isinstance(node, ast.Expr):
            return self.transform(node.value)
        if isinstance(node, ast.Name):
            return self.transform_name(node)
        if isinstance(node, ast.Attribute):
            pat = self.transform_attribute(node.value, node.attr, node.ctx)
            x = self.transform(node.value)
            return pat.format(x=x)
        if isinstance(node, ast.Call):
            pat = self.transform_call(node.func, node.args)
            xes = {'x%s' % i: self.transform(node.args[i]) for i in range(len(node.args))}
            return pat.format(**xes)
        if isinstance(node, ast.UnaryOp):
            pat = self.transform_unary_op(node.op, node.operand)
            arg = self.transform(node.operand)
            return pat.format(x=arg)
        if isinstance(node, ast.BinOp):
            pat = self.transform_bin_op(node.op, node.left, node.right)
            x = self.transform(node.left)
            y = self.transform(node.right)
            return pat.format(x=x, y=y)
        if isinstance(node, ast.BoolOp):
            pat = self.transform_bool_op(node.op, node.values)
            xes = {'x%s' % i: self.transform(node.values[i]) for i in range(len(node.values))}
            return pat.format(**xes)
        if isinstance(node, ast.Compare):
            pat = self.transform_compare(node.left, node.ops, node.comparators)
            xes = {'x%s' % i: self.transform(node.values[i]) for i in range(len(node.values))}
            return pat.format(**xes)
        if isinstance(node, ast.Num):
            return str(node.n)
        if isinstance(node, ast.NameConstant):
            return str(node.value)
        raise ValueError('unrecognized expression node: %s' % node.__class__.__name__)

    def transform_name(self, name: ast.Name):
        return "%s" % name.id

    def transform_call(self, func: ast.Name, args):
        args = ', '.join(['{x%d}' % i for i in range(len(args))])
        return "%s(%s)" % (func.id, args)

    def transform_attribute(self, value: ast.AST, attr: str, ctx):
        return "{x}.%s" % attr

    def transform_unary_op(self, op, operand):
        name, precedence, _ = self.get_op_info(op)

        x = '{x}'

        right_op = getattr(operand, 'op', None)
        if right_op:
            _, other_precedence, other_assoc = self.get_op_info(right_op)
            if other_precedence < precedence or other_precedence == precedence \
                    and other_assoc is not None:
                x = '({x})'

        if name in self._KEYWORDS:
            return "%s %s" % (name, x)
        else:
            return "%s%s" % (name, x)

    def transform_bin_op(self, op, left, right):
        name, precedence, assoc = ExprDecompiler.get_op_info(op)

        x = '{x}'
        y = '{y}'

        left_op = getattr(left, 'op', None)
        if left_op:
            _, other_precedence, other_assoc = self.get_op_info(left_op)
            if other_precedence < precedence or other_precedence == precedence \
                    and assoc == 'R' and other_assoc is not None:
                x = '({x})'

        right_op = getattr(right, 'op', None)
        if right_op:
            _, other_precedence, other_assoc = self.get_op_info(right_op)
            if other_precedence < precedence or other_precedence == precedence \
                    and assoc == 'L' and other_assoc is not None:
                y = '({y})'

        return "%s %s %s" % (x, name, y)

    def transform_bool_op(self, op, values):
        name, precedence, assoc = ExprDecompiler.get_op_info(op)

        xes = []
        for i in range(len(values)):
            value = values[i]
            x = '{x%d}' % i
            other_op = getattr(value, 'op', None)
            if other_op:
                _, other_precedence, other_assoc = self.get_op_info(other_op)
                if i == 0 and other_precedence < precedence \
                        or i > 0 and other_precedence <= precedence:
                    x = '(%s)' % x
            xes.append(x)

        return (' %s ' % name).join(xes)

    # Compare(left, ops, comparators
    def transform_compare(self, left, ops, comparators):


        parts = []
        for i in range(len(ops)):
            op = ops[i]
            comparator = comparators[i]
            name, precedence, assoc = ExprDecompiler.get_op_info(op)
            if i == 0:
                x = '{x0}'
                left_op = getattr(left, 'op', None)
                if left_op:
                    name, other_precedence, assoc = ExprDecompiler.get_op_info(left_op)
                    if other_precedence < precedence:
                        x = '(%s)' % x
                parts.append(x)

            parts.append(name)

            x = '{x%d}' % (i + 1)
            right_op = getattr(comparator, 'op', None)
            if right_op:
                _, other_precedence, other_assoc = self.get_op_info(right_op)
                if i == 0 and other_precedence < precedence \
                        or i > 0 and other_precedence <= precedence:
                    x = '(%s)' % x

            parts.append(x)

        return ' '.join(parts)