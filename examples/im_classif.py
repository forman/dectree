
from numba import jit, jitclass, float32, float64, vectorize
import numpy as np


@vectorize([float32(float32), float64(float64)])
def _Radiance_LOW(x):
    # Radiance.LOW: ramp_down(x1=0, x2=127)
    if x < 0.0:
        return 1.0
    if x < 127.0:
        return 1.0 - (x - 0.0) / (127.0 - 0.0)
    return 0.0


@vectorize([float32(float32), float64(float64)])
def _Radiance_MIDDLE(x):
    # Radiance.MIDDLE: triangular(x1=0, x2=127, x3=255)
    if x < 0.0:
        return 0.0
    if x < 127.0:
        return (x - 0.0) / (127.0 - 0.0)
    if x < 255.0:
        return 1.0 - (x - 127.0) / (255.0 - 127.0)
    return 0.0


@vectorize([float32(float32), float64(float64)])
def _Radiance_HIGH(x):
    # Radiance.HIGH: ramp_up(x1=127, x2=255)
    if x < 127.0:
        return 0.0
    if x < 255.0:
        return (x - 127.0) / (255.0 - 127.0)
    return 1.0


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


@jit(nopython=True)
def apply_rules(input, output):
    t0 = 1.0
    #    if red is MIDDLE and green is MIDDLE and blue is MIDDLE:
    t1 = np.minimum(t0, np.minimum(np.minimum(_Radiance_MIDDLE(input.red), _Radiance_MIDDLE(input.green)), _Radiance_MIDDLE(input.blue)))
    #        greyish: True
    output.greyish = t1
    #    if red is HIGH and green is HIGH or red is MIDDLE and green is MIDDLE:
    t1 = np.minimum(t0, np.maximum(np.minimum(_Radiance_HIGH(input.red), _Radiance_HIGH(input.green)), np.minimum(_Radiance_MIDDLE(input.red), _Radiance_MIDDLE(input.green))))
    #        if blue is LOW:
    t2 = np.minimum(t1, _Radiance_LOW(input.blue))
    #            yellowish: True
    output.yellowish = t2
    #    if red is MIDDLE or red is LOW and green is MIDDLE or green is LOW:
    t1 = np.minimum(t0, np.maximum(np.maximum(_Radiance_MIDDLE(input.red), np.minimum(_Radiance_LOW(input.red), _Radiance_MIDDLE(input.green))), _Radiance_LOW(input.green)))
    #        if blue is not LOW and blue is not MIDDLE and blue is not HIGH:
    t2 = np.minimum(t1, np.minimum(np.minimum(1.0 - (_Radiance_LOW(input.blue)), 1.0 - (_Radiance_MIDDLE(input.blue))), 1.0 - (_Radiance_HIGH(input.blue))))
    #            darkredish: True
    output.darkredish = t2
    #    if red is LOW and green is LOW and blue is LOW:
    t1 = np.minimum(t0, np.minimum(np.minimum(_Radiance_LOW(input.red), _Radiance_LOW(input.green)), _Radiance_LOW(input.blue)))
    #        darkish: True
    output.darkish = t1
    #    else:
    t1 = 1.0 - (t1)
    #        not_darkish: True
    output.not_darkish = t1
