import ast
from collections import OrderedDict
from io import StringIO
from typing import Dict, Any, Tuple, Optional, Union

from .config import CONFIG_NAME_AND_PATTERN
from .config import CONFIG_NAME_FUNCTION_NAME
from .config import CONFIG_NAME_INPUTS_NAME
from .config import CONFIG_NAME_NOT_PATTERN
from .config import CONFIG_NAME_NO_JIT
from .config import CONFIG_NAME_OR_PATTERN
from .config import CONFIG_NAME_OUTPUTS_NAME
from .config import CONFIG_NAME_PARAMETERIZE
from .config import CONFIG_NAME_PARAMS_NAME
from .config import CONFIG_NAME_TYPES
from .config import CONFIG_NAME_VECTORIZE
from .config import VECTORIZE_FUNC
from .config import VECTORIZE_NONE
from .config import VECTORIZE_PROP
from .config import get_config_value
from .decompiler import ExprDecompiler
from .types import DerivedDef
from .types import DerivedDefs
from .types import PropDef
from .types import PropFuncParamName
from .types import PropName
from .types import RuleBody
from .types import Rules
from .types import TypeDefs
from .types import TypeName
from .types import VarDefs
from .types import VarName

BUILT_IN_SCALAR_TYPES = {'float', 'int', 'boolean'}


def gen_code(type_defs: TypeDefs,
             input_defs: VarDefs,
             output_defs: VarDefs,
             derived_defs: DerivedDefs,
             rules: Rules,
             **options):
    text_io = StringIO()
    code_gen = CodeGen(type_defs,
                       input_defs,
                       output_defs,
                       derived_defs,
                       rules,
                       text_io,
                       options)
    code_gen.gen_code()
    return text_io.getvalue()


class CodeGen:
    def __init__(self,
                 type_defs: TypeDefs,
                 input_defs: VarDefs,
                 output_defs: VarDefs,
                 derived_defs: DerivedDefs,
                 rules: Rules,
                 out_file: Union[str, Any],
                 options: Dict[str, Any]):
        assert type_defs
        assert input_defs
        assert output_defs
        assert rules
        assert out_file

        self.type_defs = OrderedDict(type_defs)
        self.input_defs = OrderedDict(input_defs)
        self.output_defs = OrderedDict(output_defs)
        self.derived_defs = list(derived_defs or [])
        self.rules = list(rules)
        self.out_file = out_file
        self.output_assignments = None

        for var_name, type_name, _ in derived_defs:
            if var_name not in self.output_defs:
                self.output_defs[var_name] = type_name

        self._check_var_types(self.input_defs)
        self._check_var_types(self.output_defs)

        options = dict(options or {})
        self.no_jit = get_config_value(options,
                                       CONFIG_NAME_NO_JIT)
        self.vectorize = get_config_value(options,
                                          CONFIG_NAME_VECTORIZE)
        self.parameterize = get_config_value(options,
                                             CONFIG_NAME_PARAMETERIZE)
        self.function_name = get_config_value(options,
                                              CONFIG_NAME_FUNCTION_NAME)
        self.inputs_name = get_config_value(options,
                                            CONFIG_NAME_INPUTS_NAME)
        self.outputs_name = get_config_value(options,
                                             CONFIG_NAME_OUTPUTS_NAME)
        self.params_name = get_config_value(options,
                                            CONFIG_NAME_PARAMS_NAME)
        self.use_py_types = get_config_value(options,
                                             CONFIG_NAME_TYPES)
        self.not_pattern = _get_config_op_pattern(options,
                                                  CONFIG_NAME_NOT_PATTERN)
        self.and_pattern = _get_config_op_pattern(options,
                                                  CONFIG_NAME_AND_PATTERN)
        self.or_pattern = _get_config_op_pattern(options,
                                                 CONFIG_NAME_OR_PATTERN)

        self.expr_gen = FuzzyExprGen(type_defs,
                                     self.input_defs,
                                     self.output_defs,
                                     self.derived_defs,
                                     parameterize=self.parameterize,
                                     vectorize=self.vectorize,
                                     no_jit=self.no_jit,
                                     not_pattern=self.not_pattern,
                                     and_pattern=self.and_pattern,
                                     or_pattern=self.or_pattern)

    def gen_code(self):
        self.output_assignments = {}
        self._write_imports()
        self._write_type_prop_functions()
        self._write_inputs_class()
        self._write_outputs_class()
        self._write_params()
        self._write_apply_rules_function()

    def _write_imports(self):
        math_import = 'import math'
        numba_import = 'from numba import jit, float64'
        numba_experimental_import = 'from numba.experimental import jitclass'
        numpy_import = 'import numpy as np'

        if self.derived_defs:
            self._write_lines('', math_import)

        if self.no_jit:
            if self.vectorize == VECTORIZE_FUNC:
                self._write_lines('', numpy_import)
        else:
            if self.vectorize == VECTORIZE_PROP:
                self._write_lines('',
                                  numba_import + ', vectorize',
                                  numba_experimental_import,
                                  numpy_import)
            elif self.vectorize == VECTORIZE_FUNC:
                self._write_lines('',
                                  numba_import,
                                  numba_experimental_import,
                                  numpy_import)
            else:
                self._write_lines('',
                                  numba_import,
                                  numba_experimental_import)

    def _write_type_prop_functions(self):
        numba_decorator = self._get_numba_decorator(prop_func=True)
        for type_name, type_def in self.type_defs.items():
            for prop_name, prop_def in type_def.items():
                prop_value, func_params, func_body_pattern = prop_def
                if self.parameterize and func_params:
                    args = ', '.join(func_params.keys())
                    func_header = f'def _{type_name}_{prop_name}(x, {args}):'
                    func_body = func_body_pattern.format(
                        **{key: key for key in func_params.keys()})
                else:
                    func_header = f'def _{type_name}_{prop_name}(x):'
                    func_body = func_body_pattern.format(**func_params)

                func_body_lines = [f'    {line}'
                                   for line in func_body.split('\n')]
                self._write_lines(
                    '', '',
                    numba_decorator,
                    func_header,
                    f'    # {type_name}.{prop_name}: {prop_value}',
                    *func_body_lines
                )

    def _write_names_accessor(self, target: str, var_names):
        tab = '     '
        self._write_lines(
            '', '',
            f'def get_{target}_names():',
            '    return ('
        )
        for var_name in var_names:
            self._write_lines(f'{tab}{tab}{var_name!r},')
        self._write_lines(
            '    )'
        )

    def _write_apply_rules_function(self):
        if self.parameterize:
            function_params = [('inputs', self.inputs_name),
                               ('outputs', self.outputs_name),
                               ('params', self.params_name)]
        else:
            function_params = [('inputs', self.inputs_name),
                               ('outputs', self.outputs_name)]

        if self.use_py_types:
            function_args = ', '.join([
                f'{param_name}: {param_type}'
                for param_name, param_type in function_params
            ])
        else:
            function_args = ', '.join(['{}'.format(param_name)
                                       for param_name, _ in function_params])

        numba_decorator = self._get_numba_decorator()
        self._write_lines('', '',
                          numba_decorator,
                          f'def {self.function_name}({function_args}):')

        if self.vectorize == VECTORIZE_FUNC:
            output_var = list(self.output_defs.keys())[0]
            self._write_lines(
                f'    for i in range(len(outputs.{output_var})):'
            )
            self._write_lines(
                f'        t0 = 1.0'
            )
        else:
            self._write_lines('    t0 = 1.0')

        for derived_def in self.derived_defs:
            self._write_derived_var(derived_def)

        for rule in self.rules:
            self._write_rule_body(rule, 1, 1)

    def _get_numba_decorator(self, prop_func=False):
        if self.vectorize == VECTORIZE_PROP and prop_func:
            numba_decorator = '@vectorize([float64(float64)])'
        else:
            numba_decorator = '@jit(nopython=True)'
        if self.no_jit:
            numba_decorator = '# ' + numba_decorator
        return numba_decorator

    def _write_inputs_class(self):
        self._write_io_class(self.inputs_name, self.input_defs)
        self._write_names_accessor('input', self.input_defs.keys())

    def _write_outputs_class(self):
        self._write_io_class(self.outputs_name, self.output_defs)
        self._write_names_accessor('output', self.output_defs.keys())

    def _write_io_class(self, class_name, var_defs):
        self._write_class(class_name, var_defs.keys())

    def _write_params(self):
        if not self.parameterize:
            return
        param_names = []
        param_values = {}
        for type_name, type_def in self.type_defs.items():
            for prop_name, prop_def in type_def.items():
                prop_value, func_params, func_body = prop_def
                for param_name, param_value in func_params.items():
                    qualified_param_name = _get_qualified_param_name(
                        type_name,
                        prop_name,
                        param_name
                    )
                    param_names.append(qualified_param_name)
                    param_values[qualified_param_name] = param_value
        self._write_class(self.params_name, param_names, param_values)

    def _write_class(self,
                     class_name,
                     var_names,
                     param_values: Optional[Dict[str, Any]] = None):

        tab = '    '
        is_io = param_values is None

        spec_name = '_{}Spec'.format(class_name)
        spec_lines = ['{} = ['.format(spec_name)]
        for var_name in var_names:
            if param_values:
                spec_lines.append(f'{tab}("{var_name}", float64),')
            elif not self.no_jit and self.vectorize != VECTORIZE_NONE:
                spec_lines.append(f'{tab}("{var_name}", float64[:]),')
            else:
                spec_lines.append(f'{tab}("{var_name}", float64),')
        spec_lines.append(']')

        if self.no_jit:
            spec_lines = map(lambda line: '# ' + line, spec_lines)

        self._write_lines('', '', *spec_lines)

        numba_line = f'@jitclass({spec_name})'
        if self.no_jit:
            numba_line = '# ' + numba_line

        if is_io and self.vectorize == VECTORIZE_FUNC:
            if self.use_py_types:
                init_head = f'{tab}def __init__(self, size: int):'
            else:
                init_head = f'{tab}def __init__(self, size):'
        else:
            init_head = f'{tab}def __init__(self):'

        self._write_lines('', '',
                          numba_line,
                          f'class {class_name}:',
                          init_head)
        for var_name in var_names:
            if param_values:
                self._write_lines(
                    f'{tab}{tab}self.{var_name}'
                    f' = {param_values[var_name]}'
                )
            elif is_io and self.vectorize == VECTORIZE_FUNC:
                self._write_lines(
                    f'{tab}{tab}self.{var_name}'
                    f' = np.zeros(size, dtype=np.float64)'
                )
            elif self.vectorize != VECTORIZE_NONE:
                self._write_lines(
                    f'{tab}{tab}self.{var_name}'
                    f' = np.zeros(1, dtype=np.float64)'
                )
            else:
                self._write_lines(
                    f'{tab}{tab}self.{var_name} = 0.0')

    def _write_rule_body(self,
                         rule_body: RuleBody,
                         source_level: int,
                         target_level: int):
        sub_target_level = target_level
        for rule_stmt in rule_body:
            keyword = rule_stmt[0]
            if keyword == 'if':
                sub_target_level = target_level
                self._write_rule_if_stmt_part(keyword,
                                              rule_stmt[1],
                                              rule_stmt[2],
                                              source_level,
                                              sub_target_level)
            elif keyword == 'elif':
                sub_target_level += 1
                self._write_rule_if_stmt_part(keyword,
                                              rule_stmt[1],
                                              rule_stmt[2],
                                              source_level,
                                              sub_target_level)
            elif keyword == 'else':
                self._write_rule_if_stmt_part(keyword,
                                              None,
                                              rule_stmt[1],
                                              source_level,
                                              sub_target_level)
            elif keyword == '=':
                self._write_rule_assignment(rule_stmt[1],
                                            rule_stmt[2],
                                            source_level,
                                            sub_target_level)
            else:
                raise NotImplemented

    def _write_rule_if_stmt_part(self,
                                 keyword: str,
                                 condition_expr: Optional[str],
                                 rule_body: RuleBody,
                                 source_level: int,
                                 target_level: int):
        not_pattern = '1.0 - {x}'  # note, not using self.not_pattern here!

        source_indent = (4 * source_level) * ' '
        if self.vectorize == VECTORIZE_FUNC:
            target_indent = 8 * ' '
        else:
            target_indent = 4 * ' '

        t0 = 't' + str(target_level - 1)
        t1 = 't' + str(target_level - 0)

        if keyword == 'if' or keyword == 'elif':
            condition = self.expr_gen.gen_expr(condition_expr)
            if keyword == 'if':
                self._write_lines('{tind}#{sind}{key} {expr}:'.format(tind=target_indent, sind=source_indent,
                                                                      key=keyword, expr=condition_expr))
                target_value = self.and_pattern.format(x=t0, y=condition)
            else:
                tp = 't' + str(target_level - 2)
                self._write_lines('{tind}#{sind}{key} {expr}:'.format(tind=target_indent, sind=source_indent,
                                                                      key=keyword, expr=condition_expr))
                target_value = self.and_pattern.format(x=tp, y=not_pattern.format(x=t0))
                self._write_lines('{tind}{tvar} = {tval}'.format(tind=target_indent, tvar=t0, tval=target_value))
                target_value = self.and_pattern.format(x=t0, y=condition)
        else:
            self._write_lines('{tind}#{sind}else:'.format(tind=target_indent, sind=source_indent))
            target_value = self.and_pattern.format(x=t0, y=not_pattern.format(x=t1))
        self._write_lines('{tind}{tvar} = {tval}'.format(tind=target_indent, tvar=t1, tval=target_value))
        self._write_rule_body(rule_body, source_level + 1, target_level + 1)

    def _write_rule_assignment(self, var_name: str, var_value: str, source_level: int, target_level: int):

        source_indent = (source_level * 4) * ' '
        if self.vectorize == VECTORIZE_FUNC:
            target_indent = 8 * ' '
        else:
            target_indent = 4 * ' '

        t0 = 't' + str(target_level - 1)

        _, prop_def = self._get_output_def(var_name, var_value)
        prop_value, _, _ = prop_def
        if prop_value == 'true()':
            assignment_value = t0
        elif prop_value == 'false()':
            assignment_value = self.not_pattern.format(x=t0)
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
            if self.vectorize == VECTORIZE_FUNC:
                out_pattern = self.or_pattern.format(x='outputs.{name}[i]', y=out_pattern)
            else:
                out_pattern = self.or_pattern.format(x='outputs.{name}', y=out_pattern)

        source_line_pattern = '{tind}#{sind}{name} = {sval}'
        if self.vectorize == VECTORIZE_FUNC:
            target_line_pattern = '{tind}outputs.{name}[i] = ' + out_pattern
        else:
            target_line_pattern = '{tind}outputs.{name} = ' + out_pattern

        self._write_lines(source_line_pattern.format(tind=target_indent, sind=source_indent,
                                                     name=var_name, sval=var_value),
                          target_line_pattern.format(tind=target_indent, name=var_name,
                                                     tval=assignment_value))

    def _write_derived_var(self, var_assignment: DerivedDef):
        var_name, var_type, source_expr = var_assignment

        decompiler = _ExprDecompiler(self.input_defs, self.output_defs, self.vectorize)
        target_expr = decompiler.decompile(ast.parse(source_expr))

        source_indent = 4 * ' '
        if self.vectorize == VECTORIZE_FUNC:
            target_indent = 8 * ' '
        else:
            target_indent = 4 * ' '

        source_line_pattern = '{tind}#{sind}{name} = {expr}: {type}'
        if self.vectorize == VECTORIZE_FUNC:
            target_line_pattern = '{tind}outputs.{name}[i] = {expr}'
        else:
            target_line_pattern = '{tind}outputs.{name} = {expr}'

        self._write_lines(source_line_pattern.format(tind=target_indent, sind=source_indent,
                                                     name=var_name, expr=source_expr, type=var_type),
                          target_line_pattern.format(tind=target_indent, name=var_name,
                                                     expr=target_expr))

    def _get_output_def(self, var_name: VarName, prop_name: PropName) -> Tuple[TypeName, PropDef]:
        return _get_type_name_and_prop_def(var_name, prop_name, self.type_defs, self.output_defs)

    def _write_lines(self, *lines):
        for line in lines:
            self.out_file.write('%s\n' % line)

    def _check_var_types(self, var_defs):
        for var_name, var_type in var_defs.items():
            if var_type not in self.type_defs and var_type not in BUILT_IN_SCALAR_TYPES:
                raise ValueError('Type "{}" of variable "{}" is undefined'.format(var_type, var_name))


class FuzzyExprGen:
    def __init__(self,
                 type_defs: TypeDefs,
                 input_defs: VarDefs,
                 output_defs: VarDefs,
                 derived_defs: Optional[DerivedDefs] = None,
                 parameterize: bool = False,
                 vectorize: str = VECTORIZE_NONE,
                 no_jit: bool = False,
                 not_pattern: str = '1.0 - ({x})',
                 and_pattern: str = 'min({x}, {y})',
                 or_pattern: str = 'max({x}, {y})'):

        assert type_defs is not None
        assert input_defs is not None
        assert output_defs is not None
        assert vectorize
        assert not_pattern
        assert and_pattern
        assert or_pattern

        self.type_defs = type_defs
        self.input_defs = input_defs
        self.output_defs = output_defs
        self.derived_defs = derived_defs or []
        self.var_defs = dict(input_defs)
        self.var_defs.update(output_defs)
        self.var_defs.update({item[0]: item[1]
                              for item in self.derived_defs})

        self.parameterize = parameterize
        self.vectorize = vectorize
        self.no_jit = no_jit
        self.not_pattern = not_pattern
        self.and_pattern = and_pattern
        self.or_pattern = or_pattern

    def gen_expr(self, rule_condition: str) -> str:
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
                raise ValueError('Left side of comparison must be the name of an input or an output')

            var_name = expr.left.id
            if var_name in self.input_defs:
                container_ref = 'inputs.'
            elif var_name in self.output_defs:
                container_ref = 'outputs.'
            else:
                container_ref = ''

            prop_name = expr.comparators[0].id
            compare_op = expr.ops[0]
            if isinstance(compare_op, ast.Eq) or isinstance(compare_op, ast.Is):
                if self.vectorize == VECTORIZE_FUNC:
                    op_pattern = '_{t}_{r}({c}{l}{p}[i])'
                else:
                    op_pattern = '_{t}_{r}({c}{l}{p})'
            elif isinstance(compare_op, ast.NotEq) or isinstance(compare_op, ast.IsNot):
                if self.vectorize == VECTORIZE_FUNC:
                    op_pattern = self.not_pattern.format(x='_{t}_{r}({c}{l}{p}[i])')
                else:
                    op_pattern = self.not_pattern.format(x='_{t}_{r}({c}{l}{p})')
            else:
                raise ValueError('"==", "!=", "is", and "is not" are the only supported comparison operators')
            type_name, prop_def = _get_type_name_and_prop_def(var_name, prop_name, self.type_defs, self.var_defs)
            _, func_params, _ = prop_def
            if self.parameterize and func_params:
                params = ', ' + ', '.join(['{p}=params.{qp}'.format(p=param_name,
                                                                    qp=_get_qualified_param_name(type_name,
                                                                                                 prop_name,
                                                                                                 param_name))
                                           for param_name in func_params.keys()])
            else:
                params = ''
            return op_pattern.format(t=type_name, r=prop_name, l=var_name, p=params, c=container_ref)

        if isinstance(expr, ast.UnaryOp):
            op = expr.op
            if isinstance(op, ast.Not):
                op_pattern = self.not_pattern
            else:
                raise ValueError('"not" is the only supported unary operator')
            v = expr.operand
            t = self._transpile_expression(v)
            return op_pattern.format(x=t)

        if isinstance(expr, ast.BoolOp):
            op = expr.op
            if isinstance(op, ast.And):
                op_pattern = self.and_pattern
            elif isinstance(op, ast.Or):
                op_pattern = self.or_pattern
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


def _get_config_op_pattern(options, op_pattern_name):
    op_pattern = get_config_value(options, op_pattern_name)
    no_jit = get_config_value(options, CONFIG_NAME_NO_JIT)
    vectorize = get_config_value(options, CONFIG_NAME_VECTORIZE)
    return _get_effective_op_pattern(op_pattern, no_jit=no_jit, vectorize=vectorize)


def _get_effective_op_pattern(op_pattern, no_jit=False, vectorize=VECTORIZE_NONE):
    if not no_jit and vectorize == VECTORIZE_PROP:
        # TODO: improve following naive replacements, e.g. use regex-based approach
        return op_pattern.replace('min(', 'np.minimum(').replace('max(', 'np.maximum(')
    else:
        return op_pattern


class _ExprDecompiler(ExprDecompiler):
    def __init__(self, input_defs: VarDefs, output_defs: VarDefs, vectorize: str = None):
        self.input_defs = input_defs
        self.output_defs = output_defs
        self.vectorize = vectorize

    def transform_name(self, name: ast.Name):

        var_name = name.id

        container_ref = ''
        if var_name in self.input_defs:
            container_ref = 'inputs.'
        elif var_name in self.input_defs:
            container_ref = 'outputs.'

        subscript = ''
        if self.vectorize == VECTORIZE_FUNC:
            subscript = '[i]'

        return '{c}{n}{s}'.format(c=container_ref, n=var_name, s=subscript)

    def transform_function_name(self, func: ast.Name):
        func_name = func.id
        if self.vectorize == VECTORIZE_PROP:
            return 'np.{}'.format(func_name)
        else:
            return 'math.{}'.format(func_name)
