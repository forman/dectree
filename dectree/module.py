from collections import OrderedDict

from dectree.types import TypeDefs, InputDefs, OutputDefs, RuleOrRules


def gen_module(types: TypeDefs = None,
               input: InputDefs = None,
               output: OutputDefs = None,
               rules: RuleOrRules = None,
               **options):
    ordered_dict = OrderedDict()
    ordered_dict['types'] = types
    ordered_dict['input'] = input
    ordered_dict['output'] = output
    ordered_dict['rules'] = rules
    return ordered_dict