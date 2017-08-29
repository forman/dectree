import unittest
from io import StringIO

import yaml

from transpiler import _load_raw_rule, _parse_raw_rule

text_1 = """
--- |
  if a is HI:
  
    if b is LO:
      out = True
    else if a is LOW:
      out = 0.5

  else if c is MID:
    out = 0.6

  else:
    out = False  
"""


class YamlTest(unittest.TestCase):
    def test_scalars(self):
        rule_code = yaml.load(StringIO(text_1))
        raw_rule = _load_raw_rule(rule_code)
        self.assertEqual(raw_rule,
                         [{'if a is HI':
                               [{'if b is LO':
                                     ['out = True']},
                                {'else if a is LOW':
                                     ['out = 0.5']}]},
                          {'else if c is MID':
                               ['out = 0.6']},
                          {'else':
                               ['out = False']}])

        parsed_rule = _parse_raw_rule(raw_rule)
        self.assertEqual(parsed_rule,
                         [('if', 'a is HI',
                           [('if', 'b is LO',
                             [('=', 'out', 'True')]),
                            ('elif', 'a is LOW',
                             [('=', 'out', '0.5')])]),
                          ('elif', 'c is MID',
                           [('=', 'out', '0.6')]),
                          ('else', None,
                           [('=', 'out', 'False')])])
