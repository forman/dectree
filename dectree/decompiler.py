import ast


# noinspection PyMethodMayBeStatic
class ExprDecompiler:
    """
    Decompiles an AST expression into a text string.
    Decompilation is performed by calling the ``decompile(expr)`` method with an AST expression ``expr``.
    Decompilation can be customized by overriding various methods of the form ``transform_<part>(...)``.

    See https://greentreesnakes.readthedocs.io/en/latest/nodes.html#expressions
    """

    _KEYWORDS = {'in', 'not in', 'is', 'is not', 'and', 'or', 'not', 'True', 'False', 'None'}

    _OP_INFOS = {

        ast.Eq: ('==', 100, 'R'),
        ast.NotEq: ('!=', 100, 'R'),
        ast.Lt: ('<', 100, 'R'),
        ast.LtE: ('<=', 100, 'R'),
        ast.Gt: ('>', 100, 'R'),
        ast.GtE: ('>=', 100, 'R'),
        ast.Is: ('is', 100, 'R'),
        ast.IsNot: ('is not', 100, 'R'),
        ast.In: ('in', 100, 'R'),
        ast.NotIn: ('not in', 100, 'R'),

        ast.Or: ('or', 300, 'L'),
        ast.And: ('and', 400, 'L'),
        ast.Not: ('not', 500, None),

        ast.UAdd: ('+', 600, None),
        ast.USub: ('-', 600, None),

        ast.Add: ('+', 600, 'E'),
        ast.Sub: ('-', 600, 'L'),
        ast.Mult: ('*', 700, 'E'),
        ast.Div: ('/', 700, 'L'),
        ast.FloorDiv: ('//', 700, 'L'),
        ast.Mod: ('%', 800, 'L'),
        ast.Pow: ('**', 900, 'L'),
    }

    @classmethod
    def get_op_info(cls, op: ast.AST):
        return cls._OP_INFOS.get(type(op), (None, None, None))

    def decompile(self, node: ast.AST) -> str:
        return self._transform(node)

    def _transform(self, node: ast.AST) -> str:
        if isinstance(node, ast.Module):
            return self._transform(node.body[0])
        if isinstance(node, ast.Expr):
            return self._transform(node.value)
        if isinstance(node, ast.Name):
            return self.transform_name(node)
        if isinstance(node, ast.Attribute):
            pat = self.transform_attribute(node.value, node.attr, node.ctx)
            x = self._transform(node.value)
            return pat.format(x=x)
        if isinstance(node, ast.Call):
            pat = self.transform_call(node.func, node.args)
            xes = {'x%s' % i: self._transform(node.args[i]) for i in range(len(node.args))}
            return pat.format(**xes)
        if isinstance(node, ast.UnaryOp):
            pat = self.transform_unary_op(node.op, node.operand)
            arg = self._transform(node.operand)
            return pat.format(x=arg)
        if isinstance(node, ast.BinOp):
            pat = self.transform_bin_op(node.op, node.left, node.right)
            x = self._transform(node.left)
            y = self._transform(node.right)
            return pat.format(x=x, y=y)
        if isinstance(node, ast.BoolOp):
            pat = self.transform_bool_op(node.op, node.values)
            xes = {'x%s' % i: self._transform(node.values[i]) for i in range(len(node.values))}
            return pat.format(**xes)
        if isinstance(node, ast.Compare):
            pat = self.transform_compare(node.left, node.ops, node.comparators)
            xes = {'x0': self._transform(node.left)}
            xes.update({'x%s' % (i + 1): self._transform(node.comparators[i]) for i in range(len(node.comparators))})
            return pat.format(**xes)
        if isinstance(node, ast.Num):
            return str(node.n)
        if isinstance(node, ast.NameConstant):
            return str(node.value)
        raise ValueError('unrecognized expression node: %s' % node.__class__.__name__)

    def transform_name(self, name: ast.Name):
        return name.id

    def transform_call(self, func: ast.Name, args):
        args = ', '.join(['{x%d}' % i for i in range(len(args))])
        return "%s(%s)" % (self.transform_function_name(func), args)

    def transform_function_name(self, func: ast.Name):
        return func.id

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
