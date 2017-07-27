
from numba import jit, jitclass, float64


@jit(nopython=True)
def _Radiance_LOW(x):
    # Radiance.LOW: ramp_down(x1=0, x2=50)
    if x < 0.0:
        return 1.0
    if x > 50.0:
        return 0.0
    return 1.0 - (x - 0.0) / (50.0 - 0.0)


@jit(nopython=True)
def _Radiance_MIDDLE(x):
    # Radiance.MIDDLE: triangle(x1=30, x2=50, x3=100)
    if x < 30.0:
        return 0.0
    if x > 100.0:
        return 0.0
    if x < 50.0:
        return (x - 30.0) / (50.0 - 30.0)
    return 1.0 - (x - 50.0) / (100.0 - 50.0)


@jit(nopython=True)
def _Radiance_HIGH(x):
    # Radiance.HIGH: ramp_up(x1=50, x2=120)
    if x < 50.0:
        return 0.0
    if x > 120.0:
        return 1.0
    return (x - 50.0) / (120.0 - 50.0)


@jit(nopython=True)
def _Glint_LOW(x):
    # Glint.LOW: ramp_down()
    if x < 0.0:
        return 1.0
    if x > 0.5:
        return 0.0
    return 1.0 - (x - 0.0) / (0.5 - 0.0)


@jit(nopython=True)
def _Glint_HIGH(x):
    # Glint.HIGH: ramp_up()
    if x < 0.5:
        return 0.0
    if x > 1.0:
        return 1.0
    return (x - 0.5) / (1.0 - 0.5)


@jit(nopython=True)
def _Cloudy_True(x):
    # Cloudy.True: true()
    return 1.0


@jit(nopython=True)
def _Cloudy_False(x):
    # Cloudy.False: false()
    return 0.0


@jit(nopython=True)
def _Certain_HIGH(x):
    # Certain.HIGH: true()
    return 1.0


@jit(nopython=True)
def _Certain_LOW(x):
    # Certain.LOW: false()
    return 0.0


_input_spec = [
    ("glint", float64),
    ("radiance", float64),
]


@jitclass(_input_spec)
class Input:
    def __init__(self):
        self.glint = 0.0
        self.radiance = 0.0


_output_spec = [
    ("cloudy", float64),
    ("certain", float64),
]


@jitclass(_output_spec)
class Output:
    def __init__(self):
        self.cloudy = 0.0
        self.certain = 0.0


@jit(nopython=True)
def apply_rules(input, output):
    t0 = 1.0
    #    if radiance == HIGH or radiance == MIDDLE:
    t1 = min(t0, max(_Radiance_HIGH(input.radiance), _Radiance_MIDDLE(input.radiance)))
    #        if glint == LOW:
    t2 = min(t1, _Glint_LOW(input.glint))
    #            cloudy: True
    output.cloudy = t2
    #            certain: HIGH
    output.certain = t2
    #        else:
    t2 = 1.0 - (t2)
    #            if glint == HIGH:
    t3 = min(t2, _Glint_HIGH(input.glint))
    #                certain: LOW
    output.certain = max(output.certain, 1.0 - (t3))
    #    else:
    t1 = 1.0 - (t1)
    #        cloudy: False
    output.cloudy = max(output.cloudy, 1.0 - (t1))
    #        certain: HIGH
    output.certain = max(output.certain, t1)
