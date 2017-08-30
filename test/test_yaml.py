import unittest
from collections import OrderedDict
from io import StringIO

import yaml

from dectree.transpiler import _load_raw_rule, _parse_raw_rule, _to_omap

text_1 = """
--- |
  if a is HI:
  
    if b is LO:
      out = True
    else if a is LOW:
      out = 0.5

  else if c is MID:
    # Note: constant out value!
    out = 0.6

  else:
    out = False  
"""

text_2 = """
--- 
  - if a is HI:

    - if b is LO:
      - out = True
    - else if a is LOW:
      - out = 0.5

  - else if c is MID:
    # Note: constant out value!
    - out = 0.6

  - else:
    - out = False  
"""


class YamlTest(unittest.TestCase):
    def test_rule_conversion(self):
        rule_code_1 = yaml.load(StringIO(text_1))

        raw_rule = _load_raw_rule(rule_code_1)
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

        self.assertEqual(_to_omap(raw_rule, recursive=True),
                         OrderedDict([('if a is HI',
                                       OrderedDict([('if b is LO',
                                                     ['out = True']),
                                                    ('else if a is LOW',
                                                     ['out = 0.5'])])),
                                      ('else if c is MID',
                                       ['out = 0.6']),
                                      ('else',
                                       ['out = False'])]))

        rule_code_2 = yaml.load(StringIO(text_2))
        self.assertEqual(raw_rule, rule_code_2)

        parsed_rule = _parse_raw_rule(raw_rule)
        self.assertEqual(parsed_rule,
                         [('if', 'a is HI',
                           [('if', 'b is LO',
                             [('=', 'out', 'True')]),
                            ('elif', 'a is LOW',
                             [('=', 'out', '0.5')])]),
                          ('elif', 'c is MID',
                           [('=', 'out', '0.6')]),
                          ('else',
                           [('=', 'out', 'False')])])

    def test_omap(self):
        list_or_dict = []
        self.assertIs(_to_omap(list_or_dict), list_or_dict)

        list_or_dict = [1, 2, 3]
        self.assertIs(_to_omap(list_or_dict), list_or_dict)

        list_or_dict = [dict(z=3), dict(y=2), dict(x=1)]
        self.assertEqual(_to_omap(list_or_dict), OrderedDict([('z', 3), ('y', 2), ('x', 1)]))

        list_or_dict = [dict(z=3),
                        dict(y=2),
                        dict(x=[dict(a=8), dict(b=9)])]
        self.assertEqual(_to_omap(list_or_dict, recursive=False),
                         OrderedDict([('z', 3),
                                      ('y', 2),
                                      ('x', [dict(a=8), dict(b=9)])]))
        self.assertEqual(_to_omap(list_or_dict, recursive=True),
                         OrderedDict([('z', 3),
                                      ('y', 2),
                                      ('x', OrderedDict([('a', 8), ('b', 9)]))]))

        list_or_dict = OrderedDict([('z', 3),
                                    ('y', 2),
                                    ('x', [dict(a=8), dict(b=9)])])
        self.assertIs(_to_omap(list_or_dict, recursive=False),
                      list_or_dict)
        self.assertEqual(_to_omap(list_or_dict, recursive=True),
                         OrderedDict([('z', 3),
                                      ('y', 2),
                                      ('x', OrderedDict([('a', 8), ('b', 9)]))]))
