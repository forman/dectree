CONFIG_NAME_OR_PATTERN = 'or_pattern'
CONFIG_NAME_AND_PATTERN = 'and_pattern'
CONFIG_NAME_NOT_PATTERN = 'not_pattern'
CONFIG_NAME_INPUTS_NAME = 'inputs_name'
CONFIG_NAME_OUTPUTS_NAME = 'outputs_name'
CONFIG_NAME_PARAMS_NAME = 'params_name'
CONFIG_NAME_FUNCTION_NAME = 'func_name'
CONFIG_NAME_FLOAT_TYPE = 'float_type'
CONFIG_NAME_TYPES = 'types'
CONFIG_NAME_NO_JIT = 'no_jit'
CONFIG_NAME_VECTORIZE = 'vectorize'
CONFIG_NAME_PARAMETERIZE = 'parameterize'

VECTORIZE_NONE = 'off'
VECTORIZE_PROP = 'prop'
VECTORIZE_FUNC = 'func'

VECTORIZE_CHOICES = [VECTORIZE_NONE, VECTORIZE_PROP, VECTORIZE_FUNC]

FLOAT32_TYPE = 'float32'
FLOAT64_TYPE = 'float64'

FLOAT_TYPE_CHOICES = [FLOAT32_TYPE, FLOAT64_TYPE]

CONFIG_DEFAULTS = {
    CONFIG_NAME_OR_PATTERN:
        ['max({x}, {y})',
         'pattern to translate "x or y" expressions;'
         ' default is "{default}"',
         None],
    CONFIG_NAME_AND_PATTERN:
        ['min({x}, {y})',
         'pattern to translate "x and y" expressions;'
         ' default is "{default}"',
         None],
    CONFIG_NAME_NOT_PATTERN:
        ['1.0 - ({x})',
         'pattern to translate "not x" expressions;'
         ' default is "{default}"',
         None],
    CONFIG_NAME_INPUTS_NAME:
        ['Inputs',
         'name of the generated class that holds the inputs;'
         ' default is "{default}"',
         None],
    CONFIG_NAME_OUTPUTS_NAME:
        ['Outputs',
         'name of the generated class that holds the outputs;'
         ' default is "{default}"',
         None],
    CONFIG_NAME_PARAMS_NAME:
        ['Params',
         'name of the generated class that holds the parameters;'
         ' used with --parameterize; default is "{default}"',
         None],
    CONFIG_NAME_FUNCTION_NAME:
        ['apply_rules',
         'name of the generated function which implements the decision tree;'
         ' default is "{default}"',
         None],
    CONFIG_NAME_FLOAT_TYPE:
        [FLOAT64_TYPE,
         'name of the floating point type to be used;'
         ' default is "{default}"',
         FLOAT_TYPE_CHOICES],
    CONFIG_NAME_TYPES:
        [False,
         'whether to use Python 3.3+ type annotations in generated code;'
         ' off by default',
         None],
    CONFIG_NAME_NO_JIT:
        [False,
         'whether to disable just-in-time-compilation (JIT)'
         ' using Numba in generated code;'
         ' JIT is on by default',
         None],
    CONFIG_NAME_PARAMETERIZE:
        [False,
         'whether to generate parameterized fuzzy sets,'
         ' so thresholds can be changed later;'
         ' off by default',
         None],
    CONFIG_NAME_VECTORIZE:
        [VECTORIZE_NONE,
         'whether to generated vectorized functions for Numpy arrays; "'
         + VECTORIZE_PROP
         + '" vectorizes membership functions (requires Numba), "'
         + VECTORIZE_FUNC
         + '" vectorizes the decision tree function; '
           'default is "{default}"',
         VECTORIZE_CHOICES],
}


def get_config_value(config, name):
    assert name in CONFIG_DEFAULTS
    if name in config:
        return config[name]
    return CONFIG_DEFAULTS[name][0]
