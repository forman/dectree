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
CONFIG_NAME_PARAMETERIZE = 'parameterize'

CONFIG_DEFAULTS = {
    CONFIG_NAME_OR_PATTERN:
        ['max({x}, {y})', 'pattern to translate "x or y" expressions; default is "{default}"'],
    CONFIG_NAME_AND_PATTERN:
        ['min({x}, {y})', 'pattern to translate "x and y" expressions; default is "{default}"'],
    CONFIG_NAME_NOT_PATTERN:
        ['1.0 - ({x})', 'pattern to translate "not x" expressions; default is "{default}"'],
    CONFIG_NAME_FUNCTION_NAME:
        ['apply_rules', 'name of the generated function which implements the decision tree; default is "{default}"'],
    CONFIG_NAME_TYPES:
        [False, 'whether to use Python 3.3+ type annotations in generated code; off by default'],
    CONFIG_NAME_NO_JIT:
        [False,
         'whether to disable just-in-time-compilation (JIT) using Numba in generated code; JIT is on by default'],
    CONFIG_NAME_VECTORIZE:
        [False, 'whether to generate a vectorized decision tree function using Numba; off by default'],
    CONFIG_NAME_PARAMETERIZE:
        [False,
         'whether to generate parameterized fuzzy sets, so thresholds can later be changed; off by default'],
}

_PropName = str
_PropValue = str
_PropFuncParamName = str
_PropFuncParamValue = Any
_PropFuncParams = Dict[_PropFuncParamName, _PropFuncParamValue]
_PropFuncBody = str
_PropDef = Tuple[_PropValue, _PropFuncParams, _PropFuncBody]

_TypeName = str
_TypeDef = Dict[_PropName, _PropDef]
_TypeDefs = Dict[_TypeName, _TypeDef]

_VarName = str
_VarDefs = Dict[_VarName, _TypeName]


def transpile(src_file, out_file=None, **options: Dict[str, Any]) -> Optional[str]:
    """
    Generate a decision tree function by transpiling *src_file* to *out_file* using the given *options*.

    :param src_file: A file descriptor or a path-like object to the decision tree definition source file (YAML format)
    :param out_file: A file descriptor or a path-like object to the module output file (Python)
    :param options: Transpiler options
    :return: A path to the written module output file (Python) or None if *out_file* is a file descriptor
    """
    transpiler = _Transpiler(src_file, out_file, **options)
    transpiler.transpile()
    return transpiler.out_path


def _get_config_value(config, name):
    assert name in CONFIG_DEFAULTS
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


class _Transpiler:
    def __init__(self, src_file, out_file=None, **options):

        try:
            fd = open(src_file)
            self.src_path = src_file
        except TypeError:
            fd = src_file
            self.src_path = None

        if not out_file:
            assert self.src_path
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

        if not code:
            raise ValueError('Empty decision tree definition')

        sections = ('types', 'inputs', 'outputs', 'rules')
        if not all([section in code for section in sections]):
            raise ValueError('Invalid decision tree definition: missing section {} or all of them'.format(sections))

        for section in sections:
            if not code[section]:
                raise ValueError("Invalid decision tree definition: section '{}' is empty".format(section))

        self.options = code.get('options') or {}
        self.options.update(options)
        self.type_defs = _types_to_type_defs(code['types'])
        self.input_defs = _io_to_var_defs(code['inputs'])
        self.output_defs = _io_to_var_defs(code['outputs'])
        self.rules = code['rules']

        self.output_assignments = None

        self.condition_transpiler = _ConditionTranspiler(self.type_defs, self.input_defs, self.options)

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
            self._write_io_class(self.input_defs, 'Input')
            self._write_io_class(self.output_defs, 'Output')
            self._write_params()
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
        parameterize = _get_config_value(self.options, CONFIG_NAME_PARAMETERIZE)
        function_name = _get_config_value(self.options, CONFIG_NAME_FUNCTION_NAME)
        if parameterize:
            function_params = [('input', 'Input'), ('output', 'Output'), ('params', 'Params')]
        else:
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

        spec_name = '_{}Spec'.format(class_name)
        spec_lines = ['{} = ['.format(spec_name)]
        for var_name in var_names:
            if param_values:
                spec_lines.append('    ("{}", float64),'.format(var_name))
            elif not no_jit and vectorize:
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

        self._write_lines('', '',
                          numba_line,
                          'class {}:'.format(class_name),
                          '    def __init__(self):')
        for var_name in var_names:
            if param_values:
                self._write_lines('        self.{} = {}'.format(var_name, param_values[var_name]))
            elif not no_jit and vectorize:
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

        condition = self.condition_transpiler.transpile(if_cond[3:])
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

                    out_pattern = '{val}'
                    if len(output_assignments) > 1:
                        out_pattern = or_pattern.format(x='output.{name}', y=out_pattern)

                    line_pattern = '    output.{name} = ' + out_pattern
                    self._write_lines('    #{}{}: {}'.format((level * 4) * ' ', var_name, var_value))
                    self._write_lines(line_pattern.format(name=var_name, val=assignment_value))

    def _get_output_def(self, var_name: _VarName, prop_name: _PropName) -> Tuple[_TypeName, _PropDef]:
        return _get_type_name_and_prop_def(var_name, prop_name, self.type_defs, self.output_defs)

    def _write_lines(self, *lines):
        for line in lines:
            self.out_file.write('%s\n' % line)


class _ConditionTranspiler:
    def __init__(self,
                 type_defs: _TypeDefs,
                 var_defs: _VarDefs,
                 options: Dict[str, Any]):
        self.type_defs = type_defs
        self.var_defs = var_defs
        self.options = options

    def transpile(self, rule_condition: str) -> str:
        mod = ast.parse(rule_condition)

        body = mod.body
        if len(body) != 1 or not isinstance(body[0], ast.Expr):
            raise ValueError('Invalid condition expression: [{}]'.format(rule_condition))

        expr = body[0].value
        return self._transpile_expression(expr)

    def _transpile_expression(self, expr) -> str:
        if isinstance(expr, ast.Compare):
            left = expr.left
            if not isinstance(left, ast.Name):
                raise ValueError('Left side of comparison must be the name of an input')
            var_name = expr.left.id
            prop_name = expr.comparators[0].id
            compare_op = expr.ops[0]
            if isinstance(compare_op, ast.Eq) or isinstance(compare_op, ast.Is):
                op_pattern = '_{t}_{r}(input.{l}{p})'
            elif isinstance(compare_op, ast.NotEq) or isinstance(compare_op, ast.IsNot):
                not_pattern = _get_config_op_pattern(self.options, CONFIG_NAME_NOT_PATTERN)
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


def _types_to_type_defs(types: Dict[str, Dict[str, str]]) -> _TypeDefs:
    type_defs = {}
    for type_name, type_properties in types.items():
        type_def = {}
        type_defs[type_name] = type_def
        for prop_name, prop_value in type_properties.items():
            try:
                prop_result = eval(prop_value, vars(propfuncs), {})
            except Exception:
                raise ValueError('Illegal value for property "{}" of type "{}": [{}]'.format(prop_name,
                                                                                             type_name,
                                                                                             prop_value))
            func_params, func_body = prop_result
            type_def[prop_name] = prop_value, func_params, func_body
    return type_defs


def _io_to_var_defs(io: List[Dict[str, str]]) -> _VarDefs:
    io_defs = {}
    for item in io:
        var_name, var_type = dict(item).popitem()
        io_defs[var_name] = var_type
    return io_defs


def _get_type_name_and_prop_def(var_name: _VarName,
                                prop_name: _PropName,
                                type_defs: _TypeDefs,
                                var_defs: _VarDefs) -> Tuple[_TypeName, _PropDef]:
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


def _get_qualified_param_name(type_name: _TypeName,
                              prop_name: _PropName,
                              param_name: _PropFuncParamName) -> str:
    return '{t}_{p}_{k}'.format(t=type_name, p=prop_name, k=param_name)
