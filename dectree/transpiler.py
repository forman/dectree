from typing import List, Dict, Any, Tuple, Optional
import ast
import os.path

# noinspection PyPackageRequirements
import yaml  # from pyyaml

import dectree.propfuncs as propfuncs

CONFIG_NAME_OR_PATTERN = 'or_pattern'
CONFIG_NAME_AND_PATTERN = 'and_pattern'
CONFIG_NAME_NOT_PATTERN = 'not_pattern'
CONFIG_NAME_FUNCTION_NAME = 'func_name'
CONFIG_NAME_TYPES = 'types'
CONFIG_NAME_NO_JIT = 'no_jit'
CONFIG_NAME_VECTORIZE = 'vectorize'

CONFIG_DEFAULTS = {
    CONFIG_NAME_OR_PATTERN:
        ['max({x}, {y})', 'a pattern to translate or-expressions of form "x or y", default is "{default}"'],
    CONFIG_NAME_AND_PATTERN:
        ['min({x}, {y})', 'a pattern to translate and-expressions of form "x and y", default is "{default}"'],
    CONFIG_NAME_NOT_PATTERN:
        ['1.0 - ({x})', 'a pattern to translate NOT expressions of form "not x", default is "{default}"'],
    CONFIG_NAME_FUNCTION_NAME:
        ['apply_rules', 'name of the generated function which implements the decision tree, default is "{default}"'],
    CONFIG_NAME_TYPES:
        [False, 'whether to use Python 3.3+ type annotations in generated code, default is {default}'],
    CONFIG_NAME_NO_JIT:
        [False,
         'whether to disable just-in-time-compilation (JIT) using Numba in generated code, default is {default}'],
    CONFIG_NAME_VECTORIZE:
        [False, 'whether to generate a vectorized decision tree function using Numba, default is {default}'],
}


def transpile(src_file, out_file=None, **options: Dict[str, Any]) -> Optional[str]:
    """
    Generate a decision tree function by transpiling *src_file* to *out_file* using the given *options*.

    :param src_file: A file descriptor or a path-like object to the decision tree definition source file (YAML format)
    :param out_file: A file descriptor or a path-like object to the module output file (Python)
    :param options: Transpiler options
    :return: A path to the written module output file (Python) or None if *out_file* is a file descriptor
    """
    transpiler = Transpiler(src_file, out_file, **options)
    transpiler.transpile()
    return transpiler.out_path


def _get_config_value(config, name):
    if name not in CONFIG_DEFAULTS:
        raise ValueError('Unknown configuration option "{}"'.format(name))
    if name in config:
        return config[name]
    return CONFIG_DEFAULTS[name][0]


def _get_config_op_pattern(options, op_pattern_name):
    op_pattern = _get_config_value(options, op_pattern_name)
    no_jit = _get_config_value(options, 'no_jit')
    vectorize = _get_config_value(options, 'vectorize')
    if not no_jit and vectorize:
        return op_pattern.replace('min(', 'np.minimum(').replace('max(', 'np.maximum(')
    else:
        return op_pattern


class Transpiler:
    def __init__(self, src_file, out_file=None, **options):

        try:
            fd = open(src_file)
            self.src_path = src_file
        except TypeError:
            fd = src_file
            self.src_path = None

        if not out_file:
            dir_name = os.path.dirname(self.src_path)
            base_name = os.path.splitext(os.path.basename(self.src_path))[0]
            self.out_file = os.path.join(dir_name, base_name + ".py")
            self.out_path = out_file
        else:
            self.out_file = out_file
            self.out_path = None

        try:
            code = yaml.load(fd)
        finally:
            if self.src_path:
                fd.close()

        self.options = code.get('options') or {}
        self.options.update(options)
        self.types = code['types']
        self.inputs = _io_def_to_type_dict(code['inputs'])
        self.outputs = _io_def_to_type_dict(code['outputs'])
        self.rules = code['rules']

        self.output_assignments = None

    def transpile(self):

        self.output_assignments = {}

        try:
            out_path = self.out_file
            # noinspection PyTypeChecker
            self.out_file = open(out_path, mode='w')
            self.out_path = out_path
        except TypeError:
            self.out_path = None

        try:
            self._write_imports()
            self._write_type_prop_functions()
            self._write_io_class(self.inputs, 'Input')
            self._write_io_class(self.outputs, 'Output')
            self._write_apply_rules_function()
        finally:
            if self.out_path:
                self.out_file.close()

    def _write_imports(self):
        no_jit = _get_config_value(self.options, CONFIG_NAME_NO_JIT)
        vectorize = _get_config_value(self.options, CONFIG_NAME_VECTORIZE)
        if no_jit:
            pass
        elif vectorize:
            self._write_lines('', 'from numba import jit, jitclass, float32, float64, vectorize', 'import numpy as np')
        else:
            self._write_lines('', 'from numba import jit, jitclass, float64')

    @classmethod
    def transpile_rule_condition(cls,
                                 rule_condition: str,
                                 type_defs: Dict[str, Dict[str, str]],
                                 var_defs: Dict[str, str],
                                 options: Dict[str, Any]) -> str:
        mod = ast.parse(rule_condition)

        body = mod.body
        if len(body) != 1 or not isinstance(body[0], ast.Expr):
            raise ValueError('Invalid condition expression: {}'.format(rule_condition))

        expr = body[0].value
        return cls._transpile_expression(expr, type_defs, var_defs, options)

    @classmethod
    def _transpile_expression(cls,
                              expr,
                              type_defs: Dict[str, Dict[str, str]],
                              var_defs: Dict[str, str],
                              options: Dict[str, Any]) -> str:
        if isinstance(expr, ast.Compare):
            left = expr.left
            if not isinstance(left, ast.Name):
                raise ValueError('Left side of comparison must be the name of an input')
            left_id = expr.left.id
            right_id = expr.comparators[0].id
            compare_op = expr.ops[0]
            if isinstance(compare_op, ast.Eq) or isinstance(compare_op, ast.Is):
                op_pattern = '_{t}_{r}(input.{l})'
            elif isinstance(compare_op, ast.NotEq) or isinstance(compare_op, ast.IsNot):
                not_pattern = _get_config_op_pattern(options, CONFIG_NAME_NOT_PATTERN)
                op_pattern = not_pattern.format(x='_{t}_{r}(input.{l})')
            else:
                raise ValueError('"==", "!=", "is", and "is not" are the only supported comparison operators')
            type_name, _ = cls.__get_type_value(left_id, right_id, type_defs, var_defs)
            return op_pattern.format(t=type_name, r=right_id, l=left_id)

        if isinstance(expr, ast.UnaryOp):
            op = expr.op
            if isinstance(op, ast.Not):
                op_pattern = _get_config_op_pattern(options, CONFIG_NAME_NOT_PATTERN)
            else:
                raise ValueError('"not" is the only supported unary operator')
            v = expr.operand
            t = cls._transpile_expression(v, type_defs, var_defs, options)
            return op_pattern.format(x=t)

        if isinstance(expr, ast.BoolOp):
            op = expr.op
            if isinstance(op, ast.And):
                op_pattern = _get_config_op_pattern(options, CONFIG_NAME_AND_PATTERN)
            elif isinstance(op, ast.Or):
                op_pattern = _get_config_op_pattern(options, CONFIG_NAME_OR_PATTERN)
            else:
                raise ValueError('"and" and "or" are the only supported binary operators')

            t1 = None
            for v in expr.values:
                if t1 is None:
                    t1 = cls._transpile_expression(v, type_defs, var_defs, options)
                else:
                    t2 = cls._transpile_expression(v, type_defs, var_defs, options)
                    t1 = op_pattern.format(x=t1, y=t2)

            return t1

        raise ValueError('Unsupported expression')

    def _write_type_prop_functions(self):
        numba_decorator = self._get_numba_decorator(prop_func=True)
        for type_name, type_properties in self.types.items():
            for prop_name, prop_value in type_properties.items():
                func_body = eval(prop_value, vars(propfuncs), {})
                func_body_lines = map(lambda line: '    ' + str(line), func_body.split('\n'))
                self._write_lines('', '',
                                  numba_decorator,
                                  'def _{}_{}(x):'.format(type_name, prop_name),
                                  '    # {}.{}: {}'.format(type_name, prop_name, prop_value),
                                  *func_body_lines)

    def _write_apply_rules_function(self):

        function_name = _get_config_value(self.options, CONFIG_NAME_FUNCTION_NAME)
        function_params = [('input', 'Input'), ('output', 'Output')]

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
                          'def {}({}):'.format(function_name, function_args),
                          '    t0 = 1.0')

        for rule in self.rules:
            self._write_rule(rule, 1)

    def _get_numba_decorator(self, prop_func=False):
        vectorize = _get_config_value(self.options, CONFIG_NAME_VECTORIZE)
        if vectorize and prop_func:
            numba_decorator = '@vectorize([float32(float32), float64(float64)])'
        else:
            numba_decorator = '@jit(nopython=True)'
        no_jit = _get_config_value(self.options, CONFIG_NAME_NO_JIT)
        if no_jit:
            numba_decorator = '# ' + numba_decorator
        return numba_decorator

    def _write_io_class(self, var_defs, type_name):
        no_jit = _get_config_value(self.options, CONFIG_NAME_NO_JIT)
        vectorize = _get_config_value(self.options, CONFIG_NAME_VECTORIZE)

        spec_lines = ['_{}_spec = ['.format(type_name.lower())]
        for var_name in var_defs.keys():
            if not no_jit and vectorize:
                spec_lines.append('    ("{}", float64[:]),'.format(var_name))
            else:
                spec_lines.append('    ("{}", float64),'.format(var_name))
        spec_lines.append(']')

        if no_jit:
            spec_lines = map(lambda line: '# ' + line, spec_lines)

        self._write_lines('', '', *spec_lines)

        numba_line = '@jitclass(_{}_spec)'.format(type_name.lower())
        if no_jit:
            numba_line = '# ' + numba_line

        self._write_lines('', '',
                          numba_line,
                          'class {}:'.format(type_name),
                          '    def __init__(self):')
        for var_name in var_defs.keys():
            if not no_jit and vectorize:
                self._write_lines('        self.{} = np.zeros(1, dtype=np.float64)'.format(var_name))
            else:
                self._write_lines('        self.{} = 0.0'.format(var_name))

    def _write_rule(self, rule, level):
        rule = dict(rule)
        try:
            else_body = rule.pop('else')
        except KeyError:
            # ok, no "else" part
            else_body = None

        msg = 'Each rule must have a "if <condition>" part and can have an "else" part'
        try:
            if_cond, then_body = rule.popitem()
            try:
                rule.popitem()
                raise ValueError(msg)
            except KeyError:
                pass  # ok!
        except KeyError:
            raise ValueError(msg)

        if not if_cond.startswith('if '):
            raise ValueError('{}, rule: {}'.format(msg, rule))

        and_pattern = _get_config_op_pattern(self.options, CONFIG_NAME_AND_PATTERN)
        not_pattern = _get_config_op_pattern(self.options, CONFIG_NAME_NOT_PATTERN)

        condition = self.transpile_rule_condition(if_cond[3:], self.types, self.inputs, self.options)
        t0 = 't' + str(level - 1)
        t1 = 't' + str(level)
        indent = (4 * level) * ' '
        self._write_lines('    #{}{}:'.format(indent, if_cond))
        self._write_lines('    {} = {}'.format(t1, and_pattern.format(x=t0, y=condition)))
        self._write_rule_body(then_body, level + 1)
        if else_body:
            self._write_lines('    #{}else:'.format(indent))
            self._write_lines('    {} = {}'.format(t1, not_pattern.format(x=t1)))
            self._write_rule_body(else_body, level + 1)

    def _write_rule_body(self, rule_body, level):
        if isinstance(rule_body, dict):
            self._write_rule(rule_body, level)
        else:
            or_pattern = _get_config_op_pattern(self.options, CONFIG_NAME_OR_PATTERN)
            not_pattern = _get_config_op_pattern(self.options, CONFIG_NAME_NOT_PATTERN)

            t0 = 't' + str(level - 1)
            for body_item in rule_body:
                for var_name, var_value in body_item.items():
                    _, type_value = self._get_output_type_value(var_name, var_value)
                    if type_value == 'true()':
                        assignment_value = t0
                    elif type_value == 'false()':
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

                    out_pattern = '{val}'
                    if len(output_assignments) > 1:
                        out_pattern = or_pattern.format(x='output.{name}', y=out_pattern)

                    line_pattern = '    output.{name} = ' + out_pattern
                    self._write_lines('    #{}{}: {}'.format((level * 4) * ' ', var_name, var_value))
                    self._write_lines(line_pattern.format(name=var_name, val=assignment_value))

    def _get_output_type_value(self, var_name, var_value):
        return self.__get_type_value(var_name, var_value, self.types, self.outputs)

    @classmethod
    def __get_type_value(cls,
                         var_name: str,
                         var_value: str,
                         type_defs: Dict[str, Dict[str, str]],
                         var_defs: Dict[str, str]) -> Tuple[str, str]:
        type_name = var_defs.get(var_name)
        if type_name is None:
            raise ValueError('Variable {} is undefined'.format(var_name))
        type_def = type_defs.get(type_name)
        if type_def is None:
            raise ValueError('Type {} of variable {} is undefined'.format(type_name, var_name))
        if var_value not in type_def:
            raise ValueError('{} is not a property of type {} of variable {}'.format(var_value, type_name, var_name))
        type_value = type_def[var_value]
        return type_name, type_value

    def _write_lines(self, *lines):
        for line in lines:
            self.out_file.write('%s\n' % line)


def _io_def_to_type_dict(io_defs: List[Dict[str, str]]) -> Dict[str, str]:
    type_dict = {}
    for io_def in io_defs:
        var_name, var_type = dict(io_def).popitem()
        type_dict[var_name] = var_type
    return type_dict
