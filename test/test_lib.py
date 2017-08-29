import unittest

from dectree import gen_module, ramp_up, ramp_down, true, false


class LibTest(unittest.TestCase):
    def test_gen_module(self):
        m = gen_module(
            types=dict(
                Inp=dict(LOW=ramp_down(x1=0.5, x2=0.6),
                         HIGH=ramp_up(x1=0.8, x2=1.0)),
                Out=dict(TRUE=true(),
                         FALSE=false())),
            inputs=dict(x='Inp'),
            outputs=dict(y='Out'),
            rules='if x is LOW:\n'
                  '    y = TRUE')

        self.assertIsNotNone(m)
