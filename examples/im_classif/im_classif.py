
import math

from numba import jit, jitclass, float64, vectorize
import numpy as np


@vectorize([float64(float64)])
def _Radiance_LOW(x):
    # Radiance.LOW: inv_ramp(x1=0, x2=127)
    if x <= 0.0:
        return 1.0
    if x <= 127.0:
        return 1.0 - (x - 0.0) / (127.0 - 0.0)
    return 0.0


@vectorize([float64(float64)])
def _Radiance_MIDDLE(x):
    # Radiance.MIDDLE: triangular(x1=0, x2=127, x3=255)
    if x <= 0.0:
        return 0.0
    if x <= 127.0:
        return (x - 0.0) / (127.0 - 0.0)
    if x <= 255.0:
        return 1.0 - (x - 127.0) / (255.0 - 127.0)
    return 0.0


@vectorize([float64(float64)])
def _Radiance_HIGH(x):
    # Radiance.HIGH: ramp(x1=127, x2=255)
    if x <= 127.0:
        return 0.0
    if x <= 255.0:
        return (x - 127.0) / (255.0 - 127.0)
    return 1.0


@vectorize([float64(float64)])
def _Radiance_VERY_HIGH(x):
    # Radiance.VERY_HIGH: ramp(x1=180, x2=220)
    if x <= 180.0:
        return 0.0
    if x <= 220.0:
        return (x - 180.0) / (220.0 - 180.0)
    return 1.0


@vectorize([float64(float64)])
def _Certainty_FALSE(x):
    # Certainty.FALSE: false()
    return 0.0


@vectorize([float64(float64)])
def _Certainty_TRUE(x):
    # Certainty.TRUE: true()
    return 1.0


_InputsSpec = [
    ("red", float64[:]),
    ("green", float64[:]),
    ("blue", float64[:]),
]


@jitclass(_InputsSpec)
class Inputs:
    def __init__(self):
        self.red = np.zeros(1, dtype=np.float64)
        self.green = np.zeros(1, dtype=np.float64)
        self.blue = np.zeros(1, dtype=np.float64)


_OutputsSpec = [
    ("grey", float64[:]),
    ("yellow", float64[:]),
    ("dark_red", float64[:]),
    ("dark", float64[:]),
    ("not_dark", float64[:]),
    ("cloudy", float64[:]),
    ("mean", float64[:]),
]


@jitclass(_OutputsSpec)
class Outputs:
    def __init__(self):
        self.grey = np.zeros(1, dtype=np.float64)
        self.yellow = np.zeros(1, dtype=np.float64)
        self.dark_red = np.zeros(1, dtype=np.float64)
        self.dark = np.zeros(1, dtype=np.float64)
        self.not_dark = np.zeros(1, dtype=np.float64)
        self.cloudy = np.zeros(1, dtype=np.float64)
        self.mean = np.zeros(1, dtype=np.float64)


@jit(nopython=True)
def apply_rules(inputs, outputs):
    t0 = 1.0
    #    mean = (red + green + blue) / 3: Radiance
    outputs.mean = (inputs.red + inputs.green + inputs.blue) / 3
    #    if mean is VERY_HIGH:
    t1 = np.minimum(t0, _Radiance_VERY_HIGH(outputs.mean))
    #        cloudy = TRUE
    outputs.cloudy = t1
    #    if red is VERY_HIGH and green is VERY_HIGH and blue is VERY_HIGH:
    t1 = np.minimum(t0, np.minimum(np.minimum(_Radiance_VERY_HIGH(inputs.red), _Radiance_VERY_HIGH(inputs.green)), _Radiance_VERY_HIGH(inputs.blue)))
    #        cloudy = TRUE
    outputs.cloudy = np.maximum(outputs.cloudy, t1)
    #    if red is MIDDLE and green is MIDDLE and blue is MIDDLE:
    t1 = np.minimum(t0, np.minimum(np.minimum(_Radiance_MIDDLE(inputs.red), _Radiance_MIDDLE(inputs.green)), _Radiance_MIDDLE(inputs.blue)))
    #        grey = TRUE
    outputs.grey = t1
    #    if red is HIGH and green is HIGH or red is MIDDLE and green is MIDDLE:
    t1 = np.minimum(t0, np.maximum(np.minimum(_Radiance_HIGH(inputs.red), _Radiance_HIGH(inputs.green)), np.minimum(_Radiance_MIDDLE(inputs.red), _Radiance_MIDDLE(inputs.green))))
    #        if blue is LOW:
    t2 = np.minimum(t1, _Radiance_LOW(inputs.blue))
    #            yellow = TRUE
    outputs.yellow = t2
    #    if red is MIDDLE or red is LOW and green is MIDDLE or green is LOW:
    t1 = np.minimum(t0, np.maximum(np.maximum(_Radiance_MIDDLE(inputs.red), np.minimum(_Radiance_LOW(inputs.red), _Radiance_MIDDLE(inputs.green))), _Radiance_LOW(inputs.green)))
    #        if blue is not LOW and blue is not MIDDLE and blue is not HIGH:
    t2 = np.minimum(t1, np.minimum(np.minimum(1.0 - (_Radiance_LOW(inputs.blue)), 1.0 - (_Radiance_MIDDLE(inputs.blue))), 1.0 - (_Radiance_HIGH(inputs.blue))))
    #            dark_red = TRUE
    outputs.dark_red = t2
    #    if red is LOW and green is LOW and blue is LOW:
    t1 = np.minimum(t0, np.minimum(np.minimum(_Radiance_LOW(inputs.red), _Radiance_LOW(inputs.green)), _Radiance_LOW(inputs.blue)))
    #        dark = TRUE
    outputs.dark = t1
    #    else:
    t1 = np.minimum(t0, 1.0 - t1)
    #        not_dark = TRUE
    outputs.not_dark = t1
