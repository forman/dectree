import unittest
from io import StringIO

import yaml

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
        result = yaml.load(StringIO(text_1))

        raw_lines = result.split('\n')
        yml_lines = []
        for raw_line in raw_lines:
            i = count_leading_spaces(raw_line)
            indent = raw_line[0:i]
            content = raw_line[i:]
            if content:
                yml_lines.append(indent + '- ' + content)

        result = yaml.load('\n'.join(yml_lines))
        self.assertEqual(result,
                         [{'if a is HI':
                           [{'if b is LO':
                                 ['out = True']},
                            {'else if a is LOW':
                                 ['out = 0.5']}]},
                          {'else if c is MID':
                               ['out = 0.6']},
                          {'else':
                               ['out = False']}])


def count_leading_spaces(s: str):
    i = 0
    for i in range(len(s)):
        if not s[i].isspace():
            return i
    return i
