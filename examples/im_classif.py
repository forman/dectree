
from numba import jit, jitclass, float32, float64, vectorize
import numpy as np


@vectorize([float32(float32), float64(float64)])
def _Radiance_LOW(x, x1, x2):
    # Radiance.LOW: ramp_down(x1=0, x2=127)
    if x < x1:
        return 1.0
    if x > x2:
        return 0.0
    return 1.0 - (x - x1) / (x2 - x1)


@vectorize([float32(float32), float64(float64)])
def _Radiance_MIDDLE(x, x1, x2, x3):
    # Radiance.MIDDLE: triangle(x1=0, x2=127, x3=255)
    if x < x1:
        return 0.0
    if x > x3:
        return 0.0
    if x < x2:
        return (x - x1) / (x2 - x1)
    return 1.0 - (x - x2) / (x3 - x2)


@vectorize([float32(float32), float64(float64)])
def _Radiance_HIGH(x, x1, x2):
    # Radiance.HIGH: ramp_up(x1=127, x2=255)
    if x < x1:
        return 0.0
    if x > x2:
        return 1.0
    return (x - x1) / (x2 - x1)


@vectorize([float32(float32), float64(float64)])
def _Certainty_False(x):
    # Certainty.False: false()
    return 0.0


@vectorize([float32(float32), float64(float64)])
def _Certainty_True(x):
    # Certainty.True: true()
    return 1.0


_InputSpec = [
    ("red", float64[:]),
    ("green", float64[:]),
    ("blue", float64[:]),
]


@jitclass(_InputSpec)
class Input:
    def __init__(self):
        self.red = np.zeros(1, dtype=np.float64)
        self.green = np.zeros(1, dtype=np.float64)
        self.blue = np.zeros(1, dtype=np.float64)


_OutputSpec = [
    ("greyish", float64[:]),
    ("yellowish", float64[:]),
    ("darkredish", float64[:]),
    ("darkish", float64[:]),
    ("not_darkish", float64[:]),
]


@jitclass(_OutputSpec)
class Output:
    def __init__(self):
        self.greyish = np.zeros(1, dtype=np.float64)
        self.yellowish = np.zeros(1, dtype=np.float64)
        self.darkredish = np.zeros(1, dtype=np.float64)
        self.darkish = np.zeros(1, dtype=np.float64)
        self.not_darkish = np.zeros(1, dtype=np.float64)


_ParamsSpec = [
    ("Radiance_LOW_x1", float64),
    ("Radiance_LOW_x2", float64),
    ("Radiance_MIDDLE_x1", float64),
    ("Radiance_MIDDLE_x2", float64),
    ("Radiance_MIDDLE_x3", float64),
    ("Radiance_HIGH_x1", float64),
    ("Radiance_HIGH_x2", float64),
]


@jitclass(_ParamsSpec)
class Params:
    def __init__(self):
        self.Radiance_LOW_x1 = 0.0
        self.Radiance_LOW_x2 = 127.0
        self.Radiance_MIDDLE_x1 = 0.0
        self.Radiance_MIDDLE_x2 = 127.0
        self.Radiance_MIDDLE_x3 = 255.0
        self.Radiance_HIGH_x1 = 127.0
        self.Radiance_HIGH_x2 = 255.0


@jit(nopython=True)
def apply_rules(input: Input, output: Output, params: Params):
    t0 = 1.0
    #    if red is MIDDLE and green is MIDDLE and blue is MIDDLE:
    t1 = np.minimum(t0, np.minimum(np.minimum(_Radiance_MIDDLE(input.red, x1=params.Radiance_MIDDLE_x1, x2=params.Radiance_MIDDLE_x2, x3=params.Radiance_MIDDLE_x3), _Radiance_MIDDLE(input.green, x1=params.Radiance_MIDDLE_x1, x2=params.Radiance_MIDDLE_x2, x3=params.Radiance_MIDDLE_x3)), _Radiance_MIDDLE(input.blue, x1=params.Radiance_MIDDLE_x1, x2=params.Radiance_MIDDLE_x2, x3=params.Radiance_MIDDLE_x3)))
    #        greyish: True
    output.greyish = t1
    #    if red is HIGH and green is HIGH or red is MIDDLE and green is MIDDLE:
    t1 = np.minimum(t0, np.maximum(np.minimum(_Radiance_HIGH(input.red, x1=params.Radiance_HIGH_x1, x2=params.Radiance_HIGH_x2), _Radiance_HIGH(input.green, x1=params.Radiance_HIGH_x1, x2=params.Radiance_HIGH_x2)), np.minimum(_Radiance_MIDDLE(input.red, x1=params.Radiance_MIDDLE_x1, x2=params.Radiance_MIDDLE_x2, x3=params.Radiance_MIDDLE_x3), _Radiance_MIDDLE(input.green, x1=params.Radiance_MIDDLE_x1, x2=params.Radiance_MIDDLE_x2, x3=params.Radiance_MIDDLE_x3))))
    #        if blue is LOW:
    t2 = np.minimum(t1, _Radiance_LOW(input.blue, x1=params.Radiance_LOW_x1, x2=params.Radiance_LOW_x2))
    #            yellowish: True
    output.yellowish = t2
    #    if red is MIDDLE or red is LOW and green is MIDDLE or green is LOW:
    t1 = np.minimum(t0, np.maximum(np.maximum(_Radiance_MIDDLE(input.red, x1=params.Radiance_MIDDLE_x1, x2=params.Radiance_MIDDLE_x2, x3=params.Radiance_MIDDLE_x3), np.minimum(_Radiance_LOW(input.red, x1=params.Radiance_LOW_x1, x2=params.Radiance_LOW_x2), _Radiance_MIDDLE(input.green, x1=params.Radiance_MIDDLE_x1, x2=params.Radiance_MIDDLE_x2, x3=params.Radiance_MIDDLE_x3))), _Radiance_LOW(input.green, x1=params.Radiance_LOW_x1, x2=params.Radiance_LOW_x2)))
    #        if blue is not LOW and blue is not MIDDLE and blue is not HIGH:
    t2 = np.minimum(t1, np.minimum(np.minimum(1.0 - (_Radiance_LOW(input.blue, x1=params.Radiance_LOW_x1, x2=params.Radiance_LOW_x2)), 1.0 - (_Radiance_MIDDLE(input.blue, x1=params.Radiance_MIDDLE_x1, x2=params.Radiance_MIDDLE_x2, x3=params.Radiance_MIDDLE_x3))), 1.0 - (_Radiance_HIGH(input.blue, x1=params.Radiance_HIGH_x1, x2=params.Radiance_HIGH_x2))))
    #            darkredish: True
    output.darkredish = t2
    #    if red is LOW and green is LOW and blue is LOW:
    t1 = np.minimum(t0, np.minimum(np.minimum(_Radiance_LOW(input.red, x1=params.Radiance_LOW_x1, x2=params.Radiance_LOW_x2), _Radiance_LOW(input.green, x1=params.Radiance_LOW_x1, x2=params.Radiance_LOW_x2)), _Radiance_LOW(input.blue, x1=params.Radiance_LOW_x1, x2=params.Radiance_LOW_x2)))
    #        darkish: True
    output.darkish = t1
    #    else:
    t1 = 1.0 - (t1)
    #        not_darkish: True
    output.not_darkish = t1
