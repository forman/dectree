
from numba import jit, jitclass, float32, float64, vectorize
import numpy as np


@vectorize([float32(float32), float64(float64)])
def _B1_LT_085(x):
    # B1.LT_085: lt(0.85)
    return 1.0 if x < 0.85 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B1_GT_1(x):
    # B1.GT_1: gt(1.0)
    return 1.0 if x > 1.0 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B2_GT_0(x):
    # B2.GT_0: gt(0.0)
    return 1.0 if x > 0.0 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B3_LT_025(x):
    # B3.LT_025: lt(0.25)
    return 1.0 if x < 0.25 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B3_LT_03(x):
    # B3.LT_03: lt(0.3)
    return 1.0 if x < 0.3 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B3_LT_035(x):
    # B3.LT_035: lt(0.35)
    return 1.0 if x < 0.35 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B3_LT_042(x):
    # B3.LT_042: lt(0.42)
    return 1.0 if x < 0.42 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B4_NODATA(x):
    # B4.NODATA: eq(0.0)
    return 1.0 if x == 0.0 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B5_LT_01(x):
    # B5.LT_01: lt(0.1)
    return 1.0 if x < 0.1 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B7_LT_05(x):
    # B7.LT_05: lt(0.5)
    return 1.0 if x < 0.5 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B8_GT_0(x):
    # B8.GT_0: gt(0.0)
    return 1.0 if x > 0.0 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B8_LT_009(x):
    # B8.LT_009: lt(0.09)
    return 1.0 if x < 0.09 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B8_GT_033(x):
    # B8.GT_033: gt(0.33)
    return 1.0 if x > 0.33 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B8_GT_035(x):
    # B8.GT_035: gt(0.35)
    return 1.0 if x > 0.35 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B8_GT_04(x):
    # B8.GT_04: gt(0.4)
    return 1.0 if x > 0.4 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B8_GT_045(x):
    # B8.GT_045: gt(0.45)
    return 1.0 if x > 0.45 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B8_LT_085(x):
    # B8.LT_085: lt(0.85)
    return 1.0 if x < 0.85 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B16_GT_0(x):
    # B16.GT_0: gt(0.0)
    return 1.0 if x > 0.0 else 0.0


@vectorize([float32(float32), float64(float64)])
def _B19_GT_015(x):
    # B19.GT_015: gt(0.15)
    return 1.0 if x > 0.15 else 0.0


@vectorize([float32(float32), float64(float64)])
def _BSum_GT_011(x):
    # BSum.GT_011: gt(0.11)
    return 1.0 if x > 0.11 else 0.0


@vectorize([float32(float32), float64(float64)])
def _BSum_GT_013(x):
    # BSum.GT_013: gt(0.13)
    return 1.0 if x > 0.13 else 0.0


@vectorize([float32(float32), float64(float64)])
def _BSum_GT_016(x):
    # BSum.GT_016: gt(0.16)
    return 1.0 if x > 0.16 else 0.0


@vectorize([float32(float32), float64(float64)])
def _Class_False(x):
    # Class.False: false()
    return 0.0


@vectorize([float32(float32), float64(float64)])
def _Class_True(x):
    # Class.True: true()
    return 1.0


_InputSpec = [
    ("b1", float64[:]),
    ("b2", float64[:]),
    ("b3", float64[:]),
    ("b4", float64[:]),
    ("b5", float64[:]),
    ("b6", float64[:]),
    ("b7", float64[:]),
    ("b8", float64[:]),
    ("b12", float64[:]),
    ("b13", float64[:]),
    ("b14", float64[:]),
    ("b15", float64[:]),
    ("b16", float64[:]),
    ("b19", float64[:]),
    ("b100", float64[:]),
    ("bsum", float64[:]),
]


@jitclass(_InputSpec)
class Input:
    def __init__(self):
        self.b1 = np.zeros(1, dtype=np.float64)
        self.b2 = np.zeros(1, dtype=np.float64)
        self.b3 = np.zeros(1, dtype=np.float64)
        self.b4 = np.zeros(1, dtype=np.float64)
        self.b5 = np.zeros(1, dtype=np.float64)
        self.b6 = np.zeros(1, dtype=np.float64)
        self.b7 = np.zeros(1, dtype=np.float64)
        self.b8 = np.zeros(1, dtype=np.float64)
        self.b12 = np.zeros(1, dtype=np.float64)
        self.b13 = np.zeros(1, dtype=np.float64)
        self.b14 = np.zeros(1, dtype=np.float64)
        self.b15 = np.zeros(1, dtype=np.float64)
        self.b16 = np.zeros(1, dtype=np.float64)
        self.b19 = np.zeros(1, dtype=np.float64)
        self.b100 = np.zeros(1, dtype=np.float64)
        self.bsum = np.zeros(1, dtype=np.float64)


_OutputSpec = [
    ("nodata", float64[:]),
    ("Wasser", float64[:]),
    ("Schill", float64[:]),
    ("Muschel", float64[:]),
    ("dense2", float64[:]),
    ("dense1", float64[:]),
    ("Strand", float64[:]),
    ("Sand", float64[:]),
    ("Misch", float64[:]),
    ("Misch2", float64[:]),
    ("Schlick", float64[:]),
    ("schlick_t", float64[:]),
    ("Wasser2", float64[:]),
]


@jitclass(_OutputSpec)
class Output:
    def __init__(self):
        self.nodata = np.zeros(1, dtype=np.float64)
        self.Wasser = np.zeros(1, dtype=np.float64)
        self.Schill = np.zeros(1, dtype=np.float64)
        self.Muschel = np.zeros(1, dtype=np.float64)
        self.dense2 = np.zeros(1, dtype=np.float64)
        self.dense1 = np.zeros(1, dtype=np.float64)
        self.Strand = np.zeros(1, dtype=np.float64)
        self.Sand = np.zeros(1, dtype=np.float64)
        self.Misch = np.zeros(1, dtype=np.float64)
        self.Misch2 = np.zeros(1, dtype=np.float64)
        self.Schlick = np.zeros(1, dtype=np.float64)
        self.schlick_t = np.zeros(1, dtype=np.float64)
        self.Wasser2 = np.zeros(1, dtype=np.float64)


@jit(nopython=True)
def apply_rules(input, output):
    t0 = 1.0
    #    if b4 is NODATA:
    t1 = np.minimum(t0, _B4_NODATA(input.b4))
    #        nodata: True
    output.nodata = t1
    #    else:
    t1 = 1.0 - (t1)
    #        if (b8 is GT_033 and b1 is LT_085) or b8 is LT_009:
    t2 = np.minimum(t1, np.maximum(np.minimum(_B8_GT_033(input.b8), _B1_LT_085(input.b1)), _B8_LT_009(input.b8)))
    #            if b5 is LT_01:
    t3 = np.minimum(t2, _B5_LT_01(input.b5))
    #                Wasser: True
    output.Wasser = t3
    #            else:
    t3 = 1.0 - (t3)
    #                if (b19 is GT_015 and (b8 is GT_04 and b8 is LT_085) and b7 is LT_05) or (b8 is GT_04 and bsum is GT_011) or (b8 is GT_035 and bsum is GT_016):
    t4 = np.minimum(t3, np.maximum(np.maximum(np.minimum(np.minimum(_B19_GT_015(input.b19), np.minimum(_B8_GT_04(input.b8), _B8_LT_085(input.b8))), _B7_LT_05(input.b7)), np.minimum(_B8_GT_04(input.b8), _BSum_GT_011(input.bsum))), np.minimum(_B8_GT_035(input.b8), _BSum_GT_016(input.bsum))))
    #                    if bsum is GT_013:
    t5 = np.minimum(t4, _BSum_GT_013(input.bsum))
    #                        Schill: True
    output.Schill = t5
    #                    else:
    t5 = 1.0 - (t5)
    #                        Muschel: True
    output.Muschel = t5
    #                else:
    t4 = 1.0 - (t4)
    #                    if b8 is GT_045:
    t5 = np.minimum(t4, _B8_GT_045(input.b8))
    #                        dense2: True
    output.dense2 = t5
    #                    else:
    t5 = 1.0 - (t5)
    #                        dense1: True
    output.dense1 = t5
    #        else:
    t2 = 1.0 - (t2)
    #            if b1 is GT_1:
    t3 = np.minimum(t2, _B1_GT_1(input.b1))
    #                Strand: True
    output.Strand = t3
    #            else:
    t3 = 1.0 - (t3)
    #                if b3 is LT_025:
    t4 = np.minimum(t3, _B3_LT_025(input.b3))
    #                    Sand: True
    output.Sand = t4
    #                else:
    t4 = 1.0 - (t4)
    #                    if b3 is LT_03 and b8 is GT_0:
    t5 = np.minimum(t4, np.minimum(_B3_LT_03(input.b3), _B8_GT_0(input.b8)))
    #                        Misch: True
    output.Misch = t5
    #                    else:
    t5 = 1.0 - (t5)
    #                        if b3 is LT_035 and b8 is GT_0:
    t6 = np.minimum(t5, np.minimum(_B3_LT_035(input.b3), _B8_GT_0(input.b8)))
    #                            Misch2: True
    output.Misch2 = t6
    #                        else:
    t6 = 1.0 - (t6)
    #                            if b3 is LT_042 and b2 is GT_0 and b8 is GT_0:
    t7 = np.minimum(t6, np.minimum(np.minimum(_B3_LT_042(input.b3), _B2_GT_0(input.b2)), _B8_GT_0(input.b8)))
    #                                Schlick: True
    output.Schlick = t7
    #                            else:
    t7 = 1.0 - (t7)
    #                                if b16 is GT_0 and b8 is GT_0:
    t8 = np.minimum(t7, np.minimum(_B16_GT_0(input.b16), _B8_GT_0(input.b8)))
    #                                    schlick_t: True
    output.schlick_t = t8
    #                                else:
    t8 = 1.0 - (t8)
    #                                    Wasser2: True
    output.Wasser2 = t8
