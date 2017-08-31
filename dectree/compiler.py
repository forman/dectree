import os
import os.path
import sys
import tempfile
from io import StringIO
from typing import Dict, Any, Tuple

from .config import CONFIG_NAME_FUNCTION_NAME, CONFIG_NAME_INPUTS_NAME, CONFIG_NAME_OUTPUTS_NAME, \
    CONFIG_NAME_PARAMS_NAME, CONFIG_NAME_PARAMETERIZE, get_config_value
from .transpiler import transpile


def compile(src_file, **options: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Generate a decision tree function by compiling *src_file* using the given *options*.
    Return a tuple:::

        (apply_rules, Inputs, Outputs)

    or if option ``parameterize=True`` return  tuple

        (apply_rules, Inputs, Outputs, Params)

    Usage pattern:::

        apply_rules, Inputs, Outputs, Params = compile(src_file, parameterize=True)
        inputs = Inputs()
        # set members of inputs object here...
        params = Params()
        # set members of params object here...
        outputs = Outputs()
        apply_rules(inputs, outputs, params)
        # get members of outputs members here...

    :param src_file: A file descriptor or a path-like object to the decision tree definition source file (YAML format)
    :param options: options, refer to `dectree --help`
    :return: A tuple ``(apply_rules, Inputs, Outputs)`` or ``(apply_rules, Inputs, Outputs, Params)``
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
