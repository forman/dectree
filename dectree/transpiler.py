import os
import os.path
from collections import OrderedDict
from typing import List, Dict, Any, Tuple, Union

# noinspection PyPackageRequirements
import yaml  # from pyyaml

import dectree.propfuncs as propfuncs
from .codegen import gen_code
from .types import TypeDefs


def transpile(src_file, out_file=None, **options: Dict[str, Any]) -> str:
    """
    Generate a decision tree function by transpiling *src_file* to *out_file* using the given *options*.
    Return the path to the generated file, if any, otherwise return ``None``.

    :param src_file: A file descriptor or a path-like object to the decision tree definition source file (YAML format)
    :param out_file: A file descriptor or a path-like object to the module output file (Python)
    :param options: options, refer to `dectree --help`
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

    sections = ('types', 'inputs', 'outputs', 'rules')
    if not all([section in src_code for section in sections]):
        raise ValueError('Invalid decision tree definition: missing section {} or all of them'.format(sections))

    for section in sections:
        if not src_code[section]:
            raise ValueError("Invalid decision tree definition: section '{}' is empty".format(section))

    types = _normalize_types(_to_omap(src_code['types'], recursive=True))
    input_defs = _to_omap(src_code['inputs'])
    output_defs = _to_omap(src_code['outputs'])
    rules = _normalize_rules(src_code['rules'])

    src_options = dict(src_code.get('options') or {})
    src_options.update(options or {})

    py_code = gen_code(types, input_defs, output_defs, rules, **src_options)

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
