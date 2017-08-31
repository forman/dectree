
from numba import jit, jitclass, float64
import numpy as np


@jit(nopython=True)
def _B1_LT_085(x):
    # B1.LT_085: lt(0.85)
    if 0.0 == 0.0:
        return 1.0 if x < 0.85 else 0.0
    x1 = 0.85 - 0.0
    x2 = 0.85 + 0.0
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B1_GT_1(x):
    # B1.GT_1: gt(1.0)
    if 0.0 == 0.0:
        return 1.0 if x > 1.0 else 0.0
    x1 = 1.0 - 0.0
    x2 = 1.0 + 0.0
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B2_GT_0(x):
    # B2.GT_0: gt(0.0)
    if 0.0 == 0.0:
        return 1.0 if x > 0.0 else 0.0
    x1 = 0.0 - 0.0
    x2 = 0.0 + 0.0
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B3_LT_005(x):
    # B3.LT_005: lt(0.05)
    if 0.0 == 0.0:
        return 1.0 if x < 0.05 else 0.0
    x1 = 0.05 - 0.0
    x2 = 0.05 + 0.0
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B3_LT_01(x):
    # B3.LT_01: lt(0.1)
    if 0.0 == 0.0:
        return 1.0 if x < 0.1 else 0.0
    x1 = 0.1 - 0.0
    x2 = 0.1 + 0.0
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B3_LT_015(x):
    # B3.LT_015: lt(0.15)
    if 0.0 == 0.0:
        return 1.0 if x < 0.15 else 0.0
    x1 = 0.15 - 0.0
    x2 = 0.15 + 0.0
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B3_LT_02(x):
    # B3.LT_02: lt(0.2)
    if 0.0 == 0.0:
        return 1.0 if x < 0.2 else 0.0
    x1 = 0.2 - 0.0
    x2 = 0.2 + 0.0
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B4_NODATA(x):
    # B4.NODATA: eq(0.0)
    if 0.0 == 0.0:
        return 1.0 if x == 0.0 else 0.0
    x1 = 0.0 - 0.0
    x2 = 0.0
    x3 = 0.0 + 0.0
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    if x <= x3:
        return 1.0 - (x - x2) / (x3 - x2)
    return 0.0


@jit(nopython=True)
def _B5_LT_01(x):
    # B5.LT_01: lt(0.1)
    if 0.0 == 0.0:
        return 1.0 if x < 0.1 else 0.0
    x1 = 0.1 - 0.0
    x2 = 0.1 + 0.0
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B7_LT_05(x):
    # B7.LT_05: lt(0.5)
    if 0.0 == 0.0:
        return 1.0 if x < 0.5 else 0.0
    x1 = 0.5 - 0.0
    x2 = 0.5 + 0.0
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B8_GT_0(x):
    # B8.GT_0: gt(0.0)
    if 0.0 == 0.0:
        return 1.0 if x > 0.0 else 0.0
    x1 = 0.0 - 0.0
    x2 = 0.0 + 0.0
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B8_LT_009(x):
    # B8.LT_009: lt(0.09)
    if 0.0 == 0.0:
        return 1.0 if x < 0.09 else 0.0
    x1 = 0.09 - 0.0
    x2 = 0.09 + 0.0
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B8_GT_033(x):
    # B8.GT_033: gt(0.33)
    if 0.0 == 0.0:
        return 1.0 if x > 0.33 else 0.0
    x1 = 0.33 - 0.0
    x2 = 0.33 + 0.0
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B8_GT_035(x):
    # B8.GT_035: gt(0.35)
    if 0.0 == 0.0:
        return 1.0 if x > 0.35 else 0.0
    x1 = 0.35 - 0.0
    x2 = 0.35 + 0.0
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B8_GT_04(x):
    # B8.GT_04: gt(0.4)
    if 0.0 == 0.0:
        return 1.0 if x > 0.4 else 0.0
    x1 = 0.4 - 0.0
    x2 = 0.4 + 0.0
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B8_GT_045(x):
    # B8.GT_045: gt(0.45)
    if 0.0 == 0.0:
        return 1.0 if x > 0.45 else 0.0
    x1 = 0.45 - 0.0
    x2 = 0.45 + 0.0
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B8_LT_085(x):
    # B8.LT_085: lt(0.85)
    if 0.0 == 0.0:
        return 1.0 if x < 0.85 else 0.0
    x1 = 0.85 - 0.0
    x2 = 0.85 + 0.0
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B16_GT_0(x):
    # B16.GT_0: gt(0.0)
    if 0.0 == 0.0:
        return 1.0 if x > 0.0 else 0.0
    x1 = 0.0 - 0.0
    x2 = 0.0 + 0.0
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B19_GT_015(x):
    # B19.GT_015: gt(0.15)
    if 0.0 == 0.0:
        return 1.0 if x > 0.15 else 0.0
    x1 = 0.15 - 0.0
    x2 = 0.15 + 0.0
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _BSum_GT_011(x):
    # BSum.GT_011: gt(0.11)
    if 0.0 == 0.0:
        return 1.0 if x > 0.11 else 0.0
    x1 = 0.11 - 0.0
    x2 = 0.11 + 0.0
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _BSum_GT_013(x):
    # BSum.GT_013: gt(0.13)
    if 0.0 == 0.0:
        return 1.0 if x > 0.13 else 0.0
    x1 = 0.13 - 0.0
    x2 = 0.13 + 0.0
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _BSum_GT_016(x):
    # BSum.GT_016: gt(0.16)
    if 0.0 == 0.0:
        return 1.0 if x > 0.16 else 0.0
    x1 = 0.16 - 0.0
    x2 = 0.16 + 0.0
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _Class_FALSE(x):
    # Class.FALSE: false()
    return 0.0


@jit(nopython=True)
def _Class_TRUE(x):
    # Class.TRUE: true()
    return 1.0


_InputsSpec = [
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


@jitclass(_InputsSpec)
class Inputs:
    def __init__(self, size: int):
        self.b1 = np.zeros(size, dtype=np.float64)
        self.b2 = np.zeros(size, dtype=np.float64)
        self.b3 = np.zeros(size, dtype=np.float64)
        self.b4 = np.zeros(size, dtype=np.float64)
        self.b5 = np.zeros(size, dtype=np.float64)
        self.b6 = np.zeros(size, dtype=np.float64)
        self.b7 = np.zeros(size, dtype=np.float64)
        self.b8 = np.zeros(size, dtype=np.float64)
        self.b12 = np.zeros(size, dtype=np.float64)
        self.b13 = np.zeros(size, dtype=np.float64)
        self.b14 = np.zeros(size, dtype=np.float64)
        self.b15 = np.zeros(size, dtype=np.float64)
        self.b16 = np.zeros(size, dtype=np.float64)
        self.b19 = np.zeros(size, dtype=np.float64)
        self.b100 = np.zeros(size, dtype=np.float64)
        self.bsum = np.zeros(size, dtype=np.float64)


_OutputsSpec = [
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


@jitclass(_OutputsSpec)
class Outputs:
    def __init__(self, size: int):
        self.nodata = np.zeros(size, dtype=np.float64)
        self.Wasser = np.zeros(size, dtype=np.float64)
        self.Schill = np.zeros(size, dtype=np.float64)
        self.Muschel = np.zeros(size, dtype=np.float64)
        self.dense2 = np.zeros(size, dtype=np.float64)
        self.dense1 = np.zeros(size, dtype=np.float64)
        self.Strand = np.zeros(size, dtype=np.float64)
        self.Sand = np.zeros(size, dtype=np.float64)
        self.Misch = np.zeros(size, dtype=np.float64)
        self.Misch2 = np.zeros(size, dtype=np.float64)
        self.Schlick = np.zeros(size, dtype=np.float64)
        self.schlick_t = np.zeros(size, dtype=np.float64)
        self.Wasser2 = np.zeros(size, dtype=np.float64)


@jit(nopython=True)
def apply_rules(input: Inputs, output: Outputs):
    for i in range(len(output.nodata)):
        t0 = 1.0
        #    if b4 is NODATA:
        t1 = min(t0, _B4_NODATA(input.b4[i]))
        #        nodata = TRUE
        output.nodata[i] = t1
        #    elif (b8 is GT_033 and b1 is LT_085) or b8 is LT_009:
        t1 = min(t0, 1.0 - t1)
        t2 = min(t1, max(min(_B8_GT_033(input.b8[i]), _B1_LT_085(input.b1[i])), _B8_LT_009(input.b8[i])))
        #        if b5 is LT_01:
        t3 = min(t2, _B5_LT_01(input.b5[i]))
        #            Wasser = TRUE
        output.Wasser[i] = t3
        #        elif (b19 is GT_015 and (b8 is GT_04 and b8 is LT_085) and b7 is LT_05) or (b8 is GT_04 and bsum is GT_011) or (b8 is GT_035 and bsum is GT_016):
        t3 = min(t2, 1.0 - t3)
        t4 = min(t3, max(max(min(min(_B19_GT_015(input.b19[i]), min(_B8_GT_04(input.b8[i]), _B8_LT_085(input.b8[i]))), _B7_LT_05(input.b7[i])), min(_B8_GT_04(input.b8[i]), _BSum_GT_011(input.bsum[i]))), min(_B8_GT_035(input.b8[i]), _BSum_GT_016(input.bsum[i]))))
        #            if bsum is GT_013:
        t5 = min(t4, _BSum_GT_013(input.bsum[i]))
        #                Schill = TRUE
        output.Schill[i] = t5
        #            else:
        t5 = min(t4, 1.0 - t5)
        #                Muschel = TRUE
        output.Muschel[i] = t5
        #        elif b8 is GT_045:
        t4 = min(t3, 1.0 - t4)
        t5 = min(t4, _B8_GT_045(input.b8[i]))
        #            dense2 = TRUE
        output.dense2[i] = t5
        #        else:
        t5 = min(t4, 1.0 - t5)
        #            dense1 = TRUE
        output.dense1[i] = t5
        #    elif b1 is GT_1:
        t2 = min(t1, 1.0 - t2)
        t3 = min(t2, _B1_GT_1(input.b1[i]))
        #        Strand = TRUE
        output.Strand[i] = t3
        #    elif b3 is LT_005:
        t3 = min(t2, 1.0 - t3)
        t4 = min(t3, _B3_LT_005(input.b3[i]))
        #        Sand = TRUE
        output.Sand[i] = t4
        #    elif b3 is LT_01 and b8 is GT_0:
        t4 = min(t3, 1.0 - t4)
        t5 = min(t4, min(_B3_LT_01(input.b3[i]), _B8_GT_0(input.b8[i])))
        #        Misch = TRUE
        output.Misch[i] = t5
        #    elif b3 is LT_015 and b8 is GT_0:
        t5 = min(t4, 1.0 - t5)
        t6 = min(t5, min(_B3_LT_015(input.b3[i]), _B8_GT_0(input.b8[i])))
        #        Misch2 = TRUE
        output.Misch2[i] = t6
        #    elif b3 is LT_02 and b2 is GT_0 and b8 is GT_0:
        t6 = min(t5, 1.0 - t6)
        t7 = min(t6, min(min(_B3_LT_02(input.b3[i]), _B2_GT_0(input.b2[i])), _B8_GT_0(input.b8[i])))
        #        Schlick = TRUE
        output.Schlick[i] = t7
        #    elif b16 is GT_0 and b8 is GT_0:
        t7 = min(t6, 1.0 - t7)
        t8 = min(t7, min(_B16_GT_0(input.b16[i]), _B8_GT_0(input.b8[i])))
        #        schlick_t = TRUE
        output.schlick_t[i] = t8
        #    else:
        t8 = min(t7, 1.0 - t8)
        #        Wasser2 = TRUE
        output.Wasser2[i] = t8
