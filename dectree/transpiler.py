import os
import os.path
import sys
import tempfile
from collections import OrderedDict
from io import StringIO
from typing import List, Dict, Any, Tuple, Union

# noinspection PyPackageRequirements
import yaml  # from pyyaml

import dectree.propfuncs as propfuncs
from .codegen import gen_code
from .config import CONFIG_NAME_FUNCTION_NAME, CONFIG_NAME_INPUTS_NAME, CONFIG_NAME_OUTPUTS_NAME, \
    CONFIG_NAME_PARAMS_NAME, CONFIG_NAME_PARAMETERIZE, get_config_value
from .types import TypeDefs, DerivedDef, DerivedDefs


def compile(src_file, **options: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Generate a decision tree function by compiling *src_file* using the given options.
    Return a tuple (function, Inputs, Outputs). If option ``parameterize`` is set,
    return (function, Inputs, Outputs, Params).

    Usage:::

        apply_rules, Inputs, Outputs, Params = compile(src_file, parameterize=True)
        inputs = Inputs()
        outputs = Outputs()
        params = Params()
        # set inputs members here
        # set params members here
        apply_rules(inputs, outputs, params)
        # get outputs members here

    :param src_file: A file descriptor or a path-like object to the decision tree definition source file (YAML format)
    :param options: Compiler/Transpiler options
    :return: A tuple containing the compiler function and the classes used to generate the functions's arguments.
    """
    text_io = StringIO()
    transpile(src_file, text_io, **options)

    py_code = text_io.getvalue()

    _, out_path = tempfile.mkstemp(suffix='.py', prefix='dectree_', text=True)
    with open(out_path, 'w') as out_fp:
        out_fp.write(py_code)

    dectree_module = _import_module_from_file(out_path)

    names = [CONFIG_NAME_FUNCTION_NAME, CONFIG_NAME_INPUTS_NAME, CONFIG_NAME_OUTPUTS_NAME]
    if get_config_value(options, CONFIG_NAME_PARAMETERIZE):
        names += [CONFIG_NAME_PARAMS_NAME]
    names = [get_config_value(options, name) for name in names]

    return tuple(getattr(dectree_module, name) for name in names)


def transpile(src_file, out_file=None, **options: Dict[str, Any]) -> str:
    """
    Generate a decision tree function by transpiling *src_file* to *out_file* using the given *options*.
    Return the generated output file, if any, otherwise return None.

    :param src_file: A file descriptor or a path-like object to the decision tree definition source file (YAML format)
    :param out_file: A file descriptor or a path-like object to the module output file (Python)
    :param options: Compiler/Transpiler options
    :return: A path to the written module output file (Python) or None if *out_file* is a file descriptor
    """

    try:
        fd = open(src_file)
        src_path = src_file
    except TypeError:
        fd = src_file
        src_path = None

    try:
        src_code = yaml.load(fd)
    finally:
        if src_path:
            fd.close()

    if not src_code:
        raise ValueError('Empty decision tree definition')

    _validate_src_code(src_code)

    type_defs = _normalize_types(_to_omap(src_code['types'], recursive=True))
    input_defs = _to_omap(src_code['inputs'])
    output_defs = _to_omap(src_code['outputs'])
    derived_defs = _parse_raw_var_assignments(_to_omap(src_code.get('derived')) or {})
    rules = _normalize_rules(src_code['rules'])

    src_options = dict(src_code.get('options') or {})
    src_options.update(options or {})

    py_code = gen_code(type_defs,
                       input_defs,
                       output_defs,
                       derived_defs,
                       rules,
                       **src_options)

    if out_file:
        try:
            fd = open(out_file, 'w')
            out_path = out_file
        except TypeError:
            fd = out_file
            out_path = None
    else:
        assert src_path
        dir_name = os.path.dirname(src_path)
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        out_path = os.path.join(dir_name, base_name + ".py")
        fd = open(out_path, mode='w')

    fd.write(py_code)
    if out_path is not None:
        fd.close()

    return out_path


def _validate_src_code(src_code):
    required_sections = ('types', 'inputs', 'outputs', 'rules')
    possible_sections = required_sections + ('derived', 'options')
    for section in required_sections:
        if section not in src_code:
            raise ValueError('Invalid decision tree definition: missing section "{}"'.format(section))
        if not src_code[section]:
            raise ValueError('Invalid decision tree definition: section "{}" is empty'.format(section))
    for section in src_code:
        if section not in possible_sections:
            raise ValueError('Invalid decision tree definition: unknown section "{}"'.format(section))


def _normalize_types(types: Dict[str, Dict[str, str]]) -> TypeDefs:
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


def _parse_raw_var_assignments(raw_var_assignments: Dict[str, str]) -> DerivedDefs:
    var_assignments = []
    for raw_var_assignment, var_type in raw_var_assignments.items():
        if not var_type:
            raise ValueError('illegal variable assignment: "{}: {}"'.format(raw_var_assignment, var_type))
        assignment_tokens = raw_var_assignment.split(None, 2)
        if len(assignment_tokens) != 3 \
                or not assignment_tokens[0] \
                or assignment_tokens[1] != '=' \
                or not assignment_tokens[2]:
            raise ValueError('illegal variable assignment: "{}"'.format(raw_var_assignment))
        var_name, _, expr = assignment_tokens
        var_assignments.append((var_name, var_type, expr))
    return var_assignments


def _normalize_rules(raw_rules):
    return [_normalize_rule(raw_rule) for raw_rule in raw_rules]


def _normalize_rule(raw_rule: Union[str, List]):
    if isinstance(raw_rule, str):
        raw_rule = _load_raw_rule(raw_rule)
    return _parse_raw_rule(raw_rule)


def _parse_raw_rule(raw_rule: List[Union[Dict, List]]) -> List[Union[Tuple, List]]:
    # print(raw_rule)
    n = len(raw_rule)
    parsed_rule = []
    for i in range(n):
        item = raw_rule[i]

        stmt_part, stmt_body, assignment = None, None, None
        if isinstance(item, dict):
            stmt_part, stmt_body = dict(item).popitem()
        else:
            assignment = item

        if stmt_part:
            stmt_tokens = stmt_part.split(None, 1)
            if len(stmt_tokens) == 0:
                raise ValueError('illegal rule part: {}'.format(stmt_part))

            keyword = stmt_tokens[0]

            if keyword == 'if':
                if i != 0:
                    raise ValueError('"if" must be first in rule: {}'.format(stmt_part))
                if len(stmt_tokens) != 2 or not stmt_tokens[1]:
                    raise ValueError('illegal rule part: {}'.format(stmt_part))
                condition = stmt_tokens[1]
            elif keyword == 'else':
                if len(stmt_tokens) == 1:
                    if i < n - 2:
                        raise ValueError('"else" must be last in rule: {}'.format(stmt_part))
                    condition = None
                else:
                    elif_stmt_tokens = stmt_tokens[1].split(None, 1)
                    if elif_stmt_tokens[0] == 'if':
                        keyword, condition = 'elif', elif_stmt_tokens[1]
                    else:
                        raise ValueError('illegal rule part: {}'.format(stmt_part))
            elif keyword == 'elif':
                if len(stmt_tokens) != 2 or not stmt_tokens[1]:
                    raise ValueError('illegal rule part: {}'.format(stmt_part))
                condition = stmt_tokens[1]
            else:
                raise ValueError('illegal rule part: {}'.format(stmt_part))

            if condition:
                parsed_rule.append((keyword, condition, _parse_raw_rule(stmt_body)))
            else:
                parsed_rule.append((keyword, _parse_raw_rule(stmt_body)))

        elif assignment:
            # noinspection PyUnresolvedReferences
            assignment_parts = assignment.split(None, 2)
            if len(assignment_parts) != 3 \
                    or not assignment_parts[0].isidentifier() \
                    or assignment_parts[1] != '=' \
                    or not assignment_parts[2]:
                raise ValueError('illegal rule part: {}'.format(stmt_part))

            parsed_rule.append(('=', assignment_parts[0], assignment_parts[2]))

        else:
            raise ValueError('illegal rule part: {}'.format(stmt_part))

    return parsed_rule


def _load_raw_rule(rule_code: str):
    raw_lines = rule_code.split('\n')
    yml_lines = []
    for raw_line in raw_lines:
        i = _count_leading_spaces(raw_line)
        indent = raw_line[0:i]
        content = raw_line[i:]
        if content:
            if content[0] != '#':
                yml_lines.append(indent + '- ' + content)
            else:
                yml_lines.append(indent + content)
    return yaml.load('\n'.join(yml_lines))


def _count_leading_spaces(s: str):
    i = 0
    for i in range(len(s)):
        if not s[i].isspace():
            return i
    return i


def _to_omap(list_or_dict, recursive=False):
    if not list_or_dict:
        return list_or_dict

    if _is_list_of_one_key_dicts(list_or_dict):
        dict_copy = OrderedDict()
        for item in list_or_dict:
            key, item = dict(item).popitem()
            dict_copy[key] = _to_omap(item) if recursive else item
        return dict_copy

    if recursive:
        if isinstance(list_or_dict, list):
            list_copy = []
            for item in list_or_dict:
                list_copy.append(_to_omap(item, recursive=True))
            return list_copy
        if isinstance(list_or_dict, dict):
            dict_copy = OrderedDict()
            for key, item in list_or_dict.items():
                dict_copy[key] = _to_omap(item, recursive=True)
            return dict_copy

    return list_or_dict


def _is_list_of_one_key_dicts(l):
    try:
        for item in l:
            # noinspection PyUnusedLocal
            (k, v), = item.items()
    except (AttributeError, TypeError):
        return False
    return True


def _import_module_from_file(full_path_to_module: str):
    module_dir, module_file = os.path.split(full_path_to_module)
    module_name, module_ext = os.path.splitext(module_file)
    try:
        sys.path.append(module_dir)
        module_obj = __import__(module_name)
        module_obj.__file__ = full_path_to_module
        globals()[module_name] = module_obj
        return module_obj
    finally:
        sys.path.remove(module_dir)
