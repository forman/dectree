import ast
from collections import OrderedDict
from io import StringIO
from typing import List, Dict, Any, Tuple, Optional

import dectree.propfuncs as propfuncs
from .types import VarName, PropName, TypeName, PropDef, TypeDefs, VarDefs, PropFuncParamName

PARAMS_CLASS_NAME = 'Params'

OUTPUT_CLASS_NAME = 'Output'

INPUT_CLASS_NAME = 'Input'

CONFIG_NAME_OR_PATTERN = 'or_pattern'
CONFIG_NAME_AND_PATTERN = 'and_pattern'
CONFIG_NAME_NOT_PATTERN = 'not_pattern'
CONFIG_NAME_FUNCTION_NAME = 'func_name'
CONFIG_NAME_TYPES = 'types'
CONFIG_NAME_NO_JIT = 'no_jit'
CONFIG_NAME_VECTORIZE = 'vectorize'
CONFIG_NAME_PARAMETERIZE = 'parameterize'

VECTORIZE_NONE = 'off'
VECTORIZE_PROP = 'prop'
VECTORIZE_FUNC = 'func'

VECTORIZE_CHOICES = [VECTORIZE_NONE, VECTORIZE_PROP, VECTORIZE_FUNC]

CONFIG_DEFAULTS = {
    CONFIG_NAME_OR_PATTERN:
        ['max({x}, {y})',
         'pattern to translate "x or y" expressions; default is "{default}"', None],
    CONFIG_NAME_AND_PATTERN:
        ['min({x}, {y})',
         'pattern to translate "x and y" expressions; default is "{default}"', None],
    CONFIG_NAME_NOT_PATTERN:
        ['1.0 - ({x})',
         'pattern to translate "not x" expressions; default is "{default}"', None],
    CONFIG_NAME_FUNCTION_NAME:
        ['apply_rules',
         'name of the generated function which implements the decision tree; default is "{default}"', None],
    CONFIG_NAME_TYPES:
        [False,
         'whether to use Python 3.3+ type annotations in generated code; off by default', None],
    CONFIG_NAME_NO_JIT:
        [False,
         'whether to disable just-in-time-compilation (JIT) using Numba in generated code; JIT is on by default',
         None],
    CONFIG_NAME_PARAMETERIZE:
        [False,
         'whether to generate parameterized fuzzy sets, so thresholds can be changed later; off by default',
         None],
    CONFIG_NAME_VECTORIZE:
        [VECTORIZE_NONE,
         'whether to generated vectorized functions for Numpy arrays; '
         '"' + VECTORIZE_PROP + '" vectorizes membership functions (requires Numba), '
                                '"' + VECTORIZE_FUNC + '" vectorizes the decision tree function; '
                                                       'default is "{default}"',
         VECTORIZE_CHOICES],
}


def gen_code(type_defs,
             input_defs,
             output_defs,
             rules,
             **options):
    text_io = StringIO()
    code_gen = CodeGen(type_defs, input_defs, output_defs, rules, text_io, options)
    code_gen.gen_code()
    return text_io.getvalue()


class CodeGen:
    def __init__(self,
                 type_defs,
                 input_defs,
                 output_defs,
                 rules,
                 out_file,
                 options):

        assert type_defs
        assert input_defs
        assert output_defs
        assert rules
        assert out_file

        options = dict(options or {})
        for k, v in CONFIG_DEFAULTS.items():
            if k not in options:
                options[k] = v[0]

        self.type_defs = type_defs
        self.input_defs = input_defs
        self.output_defs = output_defs
        self.rules = rules
        self.options = options
        self.out_file = out_file
        self.output_assignments = None

        self.expr_gen = ExprGen(type_defs, input_defs, options)

    def gen_code(self):
        self.output_assignments = {}
        self._write_imports()
        self._write_type_prop_functions()
        self._write_io_class(self.input_defs, 'Input')
        self._write_io_class(self.output_defs, 'Output')
        self._write_params()
        self._write_apply_rules_function()

    def _write_imports(self):
        no_jit = _get_config_value(self.options, CONFIG_NAME_NO_JIT)
        vectorize = _get_config_value(self.options, CONFIG_NAME_VECTORIZE)

        numba_import = 'from numba import jit, jitclass, float64'
        numpy_import = 'import numpy as np'

        if no_jit:
            if vectorize == VECTORIZE_FUNC:
                self._write_lines('', numpy_import)
        else:
            if vectorize == VECTORIZE_PROP:
                self._write_lines('', numba_import + ', vectorize', numpy_import)
            elif vectorize == VECTORIZE_FUNC:
                self._write_lines('', numba_import, numpy_import)
            else:
                self._write_lines('', numba_import)

    def _write_type_prop_functions(self):
        parameterize = _get_config_value(self.options, CONFIG_NAME_PARAMETERIZE)
        numba_decorator = self._get_numba_decorator(prop_func=True)
        for type_name, type_def in self.type_defs.items():
            for prop_name, prop_def in type_def.items():
                prop_value, func_params, func_body_pattern = prop_def
                if parameterize and func_params:
                    func_header = 'def _{}_{}(x{}):'.format(type_name, prop_name, ', ' + ', '.join(func_params.keys()))
                    func_body = func_body_pattern.format(**{key: key for key in func_params.keys()})
                else:
                    func_header = 'def _{}_{}(x):'.format(type_name, prop_name)
                    func_body = func_body_pattern.format(**func_params)

                func_body_lines = map(lambda line: '    ' + str(line), func_body.split('\n'))
                self._write_lines('', '',
                                  numba_decorator,
                                  func_header,
                                  '    # {}.{}: {}'.format(type_name, prop_name, prop_value),
                                  *func_body_lines)

    def _write_apply_rules_function(self):
        vectorize = _get_config_value(self.options, CONFIG_NAME_VECTORIZE)
        parameterize = _get_config_value(self.options, CONFIG_NAME_PARAMETERIZE)
        function_name = _get_config_value(self.options, CONFIG_NAME_FUNCTION_NAME)
        if parameterize:
            function_params = [('input', INPUT_CLASS_NAME),
                               ('output', OUTPUT_CLASS_NAME),
                               ('params', PARAMS_CLASS_NAME)]
        else:
            function_params = [('input', INPUT_CLASS_NAME),
                               ('output', OUTPUT_CLASS_NAME)]

        type_annotations = _get_config_value(self.options, CONFIG_NAME_TYPES)
        if type_annotations:
            function_args = ', '.join(['{}: {}'.format(param_name, param_type)
                                       for param_name, param_type in function_params])
        else:
            function_args = ', '.join(['{}'.format(param_name)
                                       for param_name, _ in function_params])

        numba_decorator = self._get_numba_decorator()
        self._write_lines('', '',
                          numba_decorator,
                          'def {}({}):'.format(function_name, function_args))

        if vectorize == VECTORIZE_FUNC:
            output_var = list(self.output_defs.keys())[0]
            self._write_lines('    for i in range(len(output.{output_var})):'.format(output_var=output_var))
            self._write_lines('        t0 = 1.0')
        else:
            self._write_lines('    t0 = 1.0')

        for rule in self.rules:
            self._write_rule(rule, 1, 1)

    def _get_numba_decorator(self, prop_func=False):
        vectorize = _get_config_value(self.options, CONFIG_NAME_VECTORIZE)
        if vectorize == VECTORIZE_PROP and prop_func:
            numba_decorator = '@vectorize([float64(float64)])'
        else:
            numba_decorator = '@jit(nopython=True)'
        no_jit = _get_config_value(self.options, CONFIG_NAME_NO_JIT)
        if no_jit:
            numba_decorator = '# ' + numba_decorator
        return numba_decorator

    def _write_io_class(self, var_defs, type_name):
        self._write_class(type_name, var_defs.keys())

    def _write_params(self):
        parameterize = _get_config_value(self.options, CONFIG_NAME_PARAMETERIZE)
        if not parameterize:
            return
        param_names = []
        param_values = {}
        for type_name, type_def in self.type_defs.items():
            for prop_name, prop_def in type_def.items():
                prop_value, func_params, func_body = prop_def
                for param_name, param_value in func_params.items():
                    qualified_param_name = _get_qualified_param_name(type_name, prop_name, param_name)
                    param_names.append(qualified_param_name)
                    param_values[qualified_param_name] = param_value
        self._write_class('Params', param_names, param_values)

    def _write_class(self, class_name, var_names, param_values: Optional[Dict[str, Any]] = None):
        no_jit = _get_config_value(self.options, CONFIG_NAME_NO_JIT)
        vectorize = _get_config_value(self.options, CONFIG_NAME_VECTORIZE)
        types = _get_config_value(self.options, CONFIG_NAME_TYPES)
        is_io = param_values is None

        spec_name = '_{}Spec'.format(class_name)
        spec_lines = ['{} = ['.format(spec_name)]
        for var_name in var_names:
            if param_values:
                spec_lines.append('    ("{}", float64),'.format(var_name))
            elif not no_jit and vectorize != VECTORIZE_NONE:
                spec_lines.append('    ("{}", float64[:]),'.format(var_name))
            else:
                spec_lines.append('    ("{}", float64),'.format(var_name))
        spec_lines.append(']')

        if no_jit:
            spec_lines = map(lambda line: '# ' + line, spec_lines)

        self._write_lines('', '', *spec_lines)

        numba_line = '@jitclass({})'.format(spec_name)
        if no_jit:
            numba_line = '# ' + numba_line

        if is_io and vectorize == VECTORIZE_FUNC:
            if types:
                init_head = '    def __init__(self, size: int):'
            else:
                init_head = '    def __init__(self, size):'
        else:
            init_head = '    def __init__(self):'

        self._write_lines('', '',
                          numba_line,
                          'class {}:'.format(class_name),
                          init_head)
        for var_name in var_names:
            if param_values:
                self._write_lines('        self.{} = {}'.format(var_name, param_values[var_name]))
            elif is_io and vectorize == VECTORIZE_FUNC:
                self._write_lines('        self.{} = np.zeros(size, dtype=np.float64)'.format(var_name))
            elif vectorize != VECTORIZE_NONE:
                self._write_lines('        self.{} = np.zeros(1, dtype=np.float64)'.format(var_name))
            else:
                self._write_lines('        self.{} = 0.0'.format(var_name))

    def _write_rule(self, rule: List, source_level: int, target_level: int):
        sub_target_level = target_level
        for stmt in rule:
            keyword = stmt[0]
            if keyword == 'if':
                sub_target_level = target_level
                self._write_stmt(keyword, stmt[1], stmt[2], source_level, sub_target_level)
            elif keyword == 'elif':
                sub_target_level += 1
                self._write_stmt(keyword, stmt[1], stmt[2], source_level, sub_target_level)
            elif keyword == 'else':
                self._write_stmt(keyword, stmt[1], stmt[2], source_level, sub_target_level)
            elif keyword == '=':
                self._write_assignment(stmt[1], stmt[2], source_level, sub_target_level)
            else:
                raise NotImplemented

    def _write_stmt(self, keyword: str, condition_expr: str, body: List, source_level: int, target_level: int):
        vectorize = _get_config_value(self.options, CONFIG_NAME_VECTORIZE)
        and_pattern = _get_config_op_pattern(self.options, CONFIG_NAME_AND_PATTERN)
        not_pattern = '1.0 - {x}'  # _get_config_op_pattern(self.options, CONFIG_NAME_NOT_PATTERN)

        source_indent = (4 * source_level) * ' '
        if vectorize == VECTORIZE_FUNC:
            target_indent = 8 * ' '
        else:
            target_indent = 4 * ' '

        if keyword == 'if' or keyword == 'elif':
            condition = self.expr_gen.gen_expr(condition_expr)
            if keyword == 'if':
                t0 = 't' + str(target_level - 1)
                t1 = 't' + str(target_level - 0)
                self._write_lines('{tind}#{sind}{key} {expr}:'.format(tind=target_indent, sind=source_indent,
                                                                      key=keyword, expr=condition_expr))
                target_value = and_pattern.format(x=t0, y=condition)
            else:
                tp = 't' + str(target_level - 2)
                t0 = 't' + str(target_level - 1)
                t1 = 't' + str(target_level - 0)
                self._write_lines('{tind}#{sind}{key} {expr}:'.format(tind=target_indent, sind=source_indent,
                                                                      key=keyword, expr=condition_expr))
                target_value = and_pattern.format(x=tp, y=not_pattern.format(x=t0))
                self._write_lines('{tind}{tvar} = {tval}'.format(tind=target_indent, tvar=t0, tval=target_value))
                target_value = and_pattern.format(x=t0, y=condition)
        else:
            t0 = 't' + str(target_level - 1)
            t1 = 't' + str(target_level - 0)
            self._write_lines('{tind}#{sind}else:'.format(tind=target_indent, sind=source_indent))
            target_value = and_pattern.format(x=t0, y=not_pattern.format(x=t1))
        self._write_lines('{tind}{tvar} = {tval}'.format(tind=target_indent, tvar=t1, tval=target_value))
        self._write_rule(body, source_level + 1, target_level + 1)

    def _write_assignment(self, var_name: str, var_value: str, source_level: int, target_level: int):
        vectorize = _get_config_value(self.options, CONFIG_NAME_VECTORIZE)
        or_pattern = _get_config_op_pattern(self.options, CONFIG_NAME_OR_PATTERN)
        not_pattern = _get_config_op_pattern(self.options, CONFIG_NAME_NOT_PATTERN)

        source_indent = (source_level * 4) * ' '
        if vectorize == VECTORIZE_FUNC:
            target_indent = 8 * ' '
        else:
            target_indent = 4 * ' '

        t0 = 't' + str(target_level - 1)

        _, prop_def = self._get_output_def(var_name, var_value)
        prop_value, _, _ = prop_def
        if prop_value == 'true()':
            assignment_value = t0
        elif prop_value == 'false()':
            assignment_value = not_pattern.format(x=t0)
        else:
            raise ValueError('Currently you can only assign properties,'
                             ' whose values are "true()" or "false()')

        output_assignments = self.output_assignments.get(var_name)
        if output_assignments is None:
            output_assignments = [assignment_value]
            self.output_assignments[var_name] = output_assignments
        else:
            output_assignments.append(assignment_value)

        out_pattern = '{tval}'
        if len(output_assignments) > 1:
            if vectorize == VECTORIZE_FUNC:
                out_pattern = or_pattern.format(x='output.{name}[i]', y=out_pattern)
            else:
                out_pattern = or_pattern.format(x='output.{name}', y=out_pattern)
        if vectorize == VECTORIZE_FUNC:
            line_pattern = '{tind}output.{name}[i] = ' + out_pattern
        else:
            line_pattern = '{tind}output.{name} = ' + out_pattern

        self._write_lines('{tind}#{sind}{name} = {sval}'.format(tind=target_indent, sind=source_indent,
                                                                name=var_name, sval=var_value))
        self._write_lines(line_pattern.format(tind=target_indent, name=var_name,
                                              tval=assignment_value))

    def _get_output_def(self, var_name: VarName, prop_name: PropName) -> Tuple[TypeName, PropDef]:
        return _get_type_name_and_prop_def(var_name, prop_name, self.type_defs, self.output_defs)

    def _write_lines(self, *lines):
        for line in lines:
            self.out_file.write('%s\n' % line)


class ExprGen:
    def __init__(self,
                 type_defs: TypeDefs,
                 var_defs: VarDefs,
                 options: Dict[str, Any]):
        self.type_defs = type_defs
        self.var_defs = var_defs
        self.options = options

    def gen_expr(self, rule_condition: str) -> str:
        mod = ast.parse(rule_condition)

        body = mod.body
        if len(body) != 1 or not isinstance(body[0], ast.Expr):
            raise ValueError('Invalid condition expression: [{}]'.format(rule_condition))

        expr = body[0].value
        return self._transpile_expression(expr)

    def _transpile_expression(self, expr) -> str:
        if isinstance(expr, ast.Compare):
            vectorize = _get_config_value(self.options, CONFIG_NAME_VECTORIZE)

            left = expr.left
            if not isinstance(left, ast.Name):
                raise ValueError('Left side of comparison must be the name of an input')
            var_name = expr.left.id
            prop_name = expr.comparators[0].id
            compare_op = expr.ops[0]
            if isinstance(compare_op, ast.Eq) or isinstance(compare_op, ast.Is):
                if vectorize == VECTORIZE_FUNC:
                    op_pattern = '_{t}_{r}(input.{l}{p}[i])'
                else:
                    op_pattern = '_{t}_{r}(input.{l}{p})'
            elif isinstance(compare_op, ast.NotEq) or isinstance(compare_op, ast.IsNot):
                not_pattern = _get_config_op_pattern(self.options, CONFIG_NAME_NOT_PATTERN)
                if vectorize == VECTORIZE_FUNC:
                    op_pattern = not_pattern.format(x='_{t}_{r}(input.{l}{p}[i])')
                else:
                    op_pattern = not_pattern.format(x='_{t}_{r}(input.{l}{p})')
            else:
                raise ValueError('"==", "!=", "is", and "is not" are the only supported comparison operators')
            type_name, prop_def = _get_type_name_and_prop_def(var_name, prop_name, self.type_defs, self.var_defs)
            _, func_params, _ = prop_def
            parameterize = _get_config_value(self.options, CONFIG_NAME_PARAMETERIZE)
            if parameterize and func_params:
                params = ', ' + ', '.join(['{p}=params.{qp}'.format(p=param_name,
                                                                    qp=_get_qualified_param_name(type_name,
                                                                                                 prop_name,
                                                                                                 param_name))
                                           for param_name in func_params.keys()])
            else:
                params = ''
            return op_pattern.format(t=type_name, r=prop_name, l=var_name, p=params)

        if isinstance(expr, ast.UnaryOp):
            op = expr.op
            if isinstance(op, ast.Not):
                op_pattern = _get_config_op_pattern(self.options, CONFIG_NAME_NOT_PATTERN)
            else:
                raise ValueError('"not" is the only supported unary operator')
            v = expr.operand
            t = self._transpile_expression(v)
            return op_pattern.format(x=t)

        if isinstance(expr, ast.BoolOp):
            op = expr.op
            if isinstance(op, ast.And):
                op_pattern = _get_config_op_pattern(self.options, CONFIG_NAME_AND_PATTERN)
            elif isinstance(op, ast.Or):
                op_pattern = _get_config_op_pattern(self.options, CONFIG_NAME_OR_PATTERN)
            else:
                raise ValueError('"and" and "or" are the only supported binary operators')

            t1 = None
            for v in expr.values:
                if t1 is None:
                    t1 = self._transpile_expression(v)
                else:
                    t2 = self._transpile_expression(v)
                    t1 = op_pattern.format(x=t1, y=t2)

            return t1

        raise ValueError('Unsupported expression')


def _get_config_value(config, name):
    assert name in CONFIG_DEFAULTS
    if name in config:
        return config[name]
    return CONFIG_DEFAULTS[name][0]


def _get_config_op_pattern(options, op_pattern_name):
    op_pattern = _get_config_value(options, op_pattern_name)
    no_jit = _get_config_value(options, 'no_jit')
    vectorize = _get_config_value(options, 'vectorize')
    if not no_jit and vectorize == VECTORIZE_PROP:
        return op_pattern.replace('min(', 'np.minimum(').replace('max(', 'np.maximum(')
    else:
        return op_pattern


def _types_to_type_defs(types: Dict[str, Dict[str, str]]) -> TypeDefs:
    type_defs = OrderedDict()
    for type_name, type_properties in types.items():
        type_def = {}
        type_defs[type_name] = type_def
        for prop_name, prop_value in type_properties.items():
            try:
                prop_result = eval(prop_value, vars(propfuncs), {})
            except Exception:
                raise ValueError('Illegal value for property "{}" of type "{}": {}'.format(prop_name,
                                                                                           type_name,
                                                                                           prop_value))
            func_params, func_body = prop_result
            type_def[prop_name] = prop_value, func_params, func_body
    return type_defs


def _get_type_name_and_prop_def(var_name: VarName,
                                prop_name: PropName,
                                type_defs: TypeDefs,
                                var_defs: VarDefs) -> Tuple[TypeName, PropDef]:
    type_name = var_defs.get(var_name)
    if type_name is None:
        raise ValueError('Variable "{}" is undefined'.format(var_name))
    type_def = type_defs.get(type_name)
    if type_def is None:
        raise ValueError('Type "{}" of variable "{}" is undefined'.format(type_name, var_name))
    if prop_name not in type_def:
        raise ValueError('"{}" is not a property of type "{}" of variable "{}"'.format(prop_name, type_name, var_name))
    prop_def = type_def[prop_name]
    return type_name, prop_def


def _get_qualified_param_name(type_name: TypeName,
                              prop_name: PropName,
                              param_name: PropFuncParamName) -> str:
    return '{t}_{p}_{k}'.format(t=type_name, p=prop_name, k=param_name)
