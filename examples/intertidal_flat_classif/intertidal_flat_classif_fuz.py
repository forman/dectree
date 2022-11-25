
import math

from numba import jit, jitclass, float64
import numpy as np


@jit(nopython=True)
def _B1_B1_veg(x):
    # B1.B1_veg: lt(0.85, dx=0.05)
    if 0.05 == 0.0:
        return 1.0 if x < 0.85 else 0.0
    x1 = 0.85 - 0.05
    x2 = 0.85 + 0.05
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B1_B1_strand(x):
    # B1.B1_strand: gt(1.00, dx=0.10)
    if 0.1 == 0.0:
        return 1.0 if x > 1.0 else 0.0
    x1 = 1.0 - 0.1
    x2 = 1.0 + 0.1
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B2_B2_schlick(x):
    # B2.B2_schlick: gt(0.00, dx=0.01)
    if 0.01 == 0.0:
        return 1.0 if x > 0.0 else 0.0
    x1 = 0.0 - 0.01
    x2 = 0.0 + 0.01
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B3_B3_sand(x):
    # B3.B3_sand: lt(0.05, dx=0.01)
    if 0.01 == 0.0:
        return 1.0 if x < 0.05 else 0.0
    x1 = 0.05 - 0.01
    x2 = 0.05 + 0.01
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B3_B3_sand2(x):
    # B3.B3_sand2: lt(0.10, dx=0.01)
    if 0.01 == 0.0:
        return 1.0 if x < 0.1 else 0.0
    x1 = 0.1 - 0.01
    x2 = 0.1 + 0.01
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B3_B3_misch(x):
    # B3.B3_misch: lt(0.15, dx=0.01)
    if 0.01 == 0.0:
        return 1.0 if x < 0.15 else 0.0
    x1 = 0.15 - 0.01
    x2 = 0.15 + 0.01
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B3_B3_schlick(x):
    # B3.B3_schlick: lt(0.20, dx=0.01)
    if 0.01 == 0.0:
        return 1.0 if x < 0.2 else 0.0
    x1 = 0.2 - 0.01
    x2 = 0.2 + 0.01
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B4_B4_nodata(x):
    # B4.B4_nodata: eq(0.0)
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
def _B5_B5_wasser(x):
    # B5.B5_wasser: lt(0.10, dx=0.05)
    if 0.05 == 0.0:
        return 1.0 if x < 0.1 else 0.0
    x1 = 0.1 - 0.05
    x2 = 0.1 + 0.05
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B7_B7_muschel(x):
    # B7.B7_muschel: lt(0.50, dx=0.05)
    if 0.05 == 0.0:
        return 1.0 if x < 0.5 else 0.0
    x1 = 0.5 - 0.05
    x2 = 0.5 + 0.05
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B8_B8_sediment_wasser(x):
    # B8.B8_sediment_wasser: gt(0.00, dx=0.01)
    if 0.01 == 0.0:
        return 1.0 if x > 0.0 else 0.0
    x1 = 0.0 - 0.01
    x2 = 0.0 + 0.01
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B8_B8_veg_wasser(x):
    # B8.B8_veg_wasser: lt(0.09, dx=0.01)
    if 0.01 == 0.0:
        return 1.0 if x < 0.09 else 0.0
    x1 = 0.09 - 0.01
    x2 = 0.09 + 0.01
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B8_B8_veg(x):
    # B8.B8_veg: gt(0.33, dx=0.02)
    if 0.02 == 0.0:
        return 1.0 if x > 0.33 else 0.0
    x1 = 0.33 - 0.02
    x2 = 0.33 + 0.02
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B8_B8_muschel_schill(x):
    # B8.B8_muschel_schill: gt(0.35, dx=0.02)
    if 0.02 == 0.0:
        return 1.0 if x > 0.35 else 0.0
    x1 = 0.35 - 0.02
    x2 = 0.35 + 0.02
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B8_B8_muschel_min(x):
    # B8.B8_muschel_min: gt(0.40, dx=0.02)
    if 0.02 == 0.0:
        return 1.0 if x > 0.4 else 0.0
    x1 = 0.4 - 0.02
    x2 = 0.4 + 0.02
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B8_B8_veg_dicht(x):
    # B8.B8_veg_dicht: gt(0.45, dx=0.02)
    if 0.02 == 0.0:
        return 1.0 if x > 0.45 else 0.0
    x1 = 0.45 - 0.02
    x2 = 0.45 + 0.02
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B8_B8_muschel_max(x):
    # B8.B8_muschel_max: lt(0.85, dx=0.02)
    if 0.02 == 0.0:
        return 1.0 if x < 0.85 else 0.0
    x1 = 0.85 - 0.02
    x2 = 0.85 + 0.02
    if x <= x1:
        return 1.0
    if x <= x2:
        return 1.0 - (x - x1) / (x2 - x1)
    return 0.0


@jit(nopython=True)
def _B16_B16_sediment_wasser(x):
    # B16.B16_sediment_wasser: gt(0.00, dx=0.01)
    if 0.01 == 0.0:
        return 1.0 if x > 0.0 else 0.0
    x1 = 0.0 - 0.01
    x2 = 0.0 + 0.01
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _B19_B19_muschel(x):
    # B19.B19_muschel: gt(0.15, dx=0.01)
    if 0.01 == 0.0:
        return 1.0 if x > 0.15 else 0.0
    x1 = 0.15 - 0.01
    x2 = 0.15 + 0.01
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _BSum_BSum_schill_1(x):
    # BSum.BSum_schill_1: gt(0.11, dx=0.02)
    if 0.02 == 0.0:
        return 1.0 if x > 0.11 else 0.0
    x1 = 0.11 - 0.02
    x2 = 0.11 + 0.02
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _BSum_BSum_schill_1a(x):
    # BSum.BSum_schill_1a: gt(0.13, dx=0.02)
    if 0.02 == 0.0:
        return 1.0 if x > 0.13 else 0.0
    x1 = 0.13 - 0.02
    x2 = 0.13 + 0.02
    if x <= x1:
        return 0.0
    if x <= x2:
        return (x - x1) / (x2 - x1)
    return 1.0


@jit(nopython=True)
def _BSum_BSum_schill_2(x):
    # BSum.BSum_schill_2: gt(0.16, dx=0.01)
    if 0.01 == 0.0:
        return 1.0 if x > 0.16 else 0.0
    x1 = 0.16 - 0.01
    x2 = 0.16 + 0.01
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
    ("b7", float64[:]),
    ("b8", float64[:]),
    ("b12", float64[:]),
    ("b13", float64[:]),
    ("b14", float64[:]),
    ("b16", float64[:]),
    ("b19", float64[:]),
]


@jitclass(_InputsSpec)
class Inputs:
    def __init__(self, size: int):
        self.b1 = np.zeros(size, dtype=np.float64)
        self.b2 = np.zeros(size, dtype=np.float64)
        self.b3 = np.zeros(size, dtype=np.float64)
        self.b4 = np.zeros(size, dtype=np.float64)
        self.b5 = np.zeros(size, dtype=np.float64)
        self.b7 = np.zeros(size, dtype=np.float64)
        self.b8 = np.zeros(size, dtype=np.float64)
        self.b12 = np.zeros(size, dtype=np.float64)
        self.b13 = np.zeros(size, dtype=np.float64)
        self.b14 = np.zeros(size, dtype=np.float64)
        self.b16 = np.zeros(size, dtype=np.float64)
        self.b19 = np.zeros(size, dtype=np.float64)


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
    ("bsum", float64[:]),
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
        self.bsum = np.zeros(size, dtype=np.float64)


@jit(nopython=True)
def apply_rules(inputs: Inputs, outputs: Outputs):
    for i in range(len(outputs.nodata)):
        t0 = 1.0
        #    bsum = b12 + b13 + b14: BSum
        outputs.bsum[i] = inputs.b12[i] + inputs.b13[i] + inputs.b14[i]
        #    if b4 is B4_nodata:
        t1 = min(t0, _B4_B4_nodata(inputs.b4[i]))
        #        nodata = TRUE
        outputs.nodata[i] = t1
        #    elif (b8 is B8_veg and b1 is B1_veg) or b8 is B8_veg_wasser:
        t1 = min(t0, 1.0 - t1)
        t2 = min(t1, max(min(_B8_B8_veg(inputs.b8[i]), _B1_B1_veg(inputs.b1[i])), _B8_B8_veg_wasser(inputs.b8[i])))
        #        if b5 is B5_wasser:
        t3 = min(t2, _B5_B5_wasser(inputs.b5[i]))
        #            Wasser = TRUE
        outputs.Wasser[i] = t3
        #        elif (b19 is B19_muschel and (b8 is B8_muschel_min and b8 is B8_muschel_max) and b7 is B7_muschel) or (b8 is B8_muschel_min and bsum is BSum_schill_1) or (b8 is B8_muschel_schill and bsum is BSum_schill_2):
        t3 = min(t2, 1.0 - t3)
        t4 = min(t3, max(max(min(min(_B19_B19_muschel(inputs.b19[i]), min(_B8_B8_muschel_min(inputs.b8[i]), _B8_B8_muschel_max(inputs.b8[i]))), _B7_B7_muschel(inputs.b7[i])), min(_B8_B8_muschel_min(inputs.b8[i]), _BSum_BSum_schill_1(outputs.bsum[i]))), min(_B8_B8_muschel_schill(inputs.b8[i]), _BSum_BSum_schill_2(outputs.bsum[i]))))
        #            if bsum is BSum_schill_1a:
        t5 = min(t4, _BSum_BSum_schill_1a(outputs.bsum[i]))
        #                Schill = TRUE
        outputs.Schill[i] = t5
        #            else:
        t5 = min(t4, 1.0 - t5)
        #                Muschel = TRUE
        outputs.Muschel[i] = t5
        #        elif b8 is B8_veg_dicht:
        t4 = min(t3, 1.0 - t4)
        t5 = min(t4, _B8_B8_veg_dicht(inputs.b8[i]))
        #            dense2 = TRUE
        outputs.dense2[i] = t5
        #        else:
        t5 = min(t4, 1.0 - t5)
        #            dense1 = TRUE
        outputs.dense1[i] = t5
        #    elif b1 is B1_strand:
        t2 = min(t1, 1.0 - t2)
        t3 = min(t2, _B1_B1_strand(inputs.b1[i]))
        #        Strand = TRUE
        outputs.Strand[i] = t3
        #    elif b3 is B3_sand:
        t3 = min(t2, 1.0 - t3)
        t4 = min(t3, _B3_B3_sand(inputs.b3[i]))
        #        Sand = TRUE
        outputs.Sand[i] = t4
        #    elif b3 is B3_sand2 and b8 is B8_sediment_wasser:
        t4 = min(t3, 1.0 - t4)
        t5 = min(t4, min(_B3_B3_sand2(inputs.b3[i]), _B8_B8_sediment_wasser(inputs.b8[i])))
        #        Misch = TRUE
        outputs.Misch[i] = t5
        #    elif b3 is B3_misch and b8 is B8_sediment_wasser:
        t5 = min(t4, 1.0 - t5)
        t6 = min(t5, min(_B3_B3_misch(inputs.b3[i]), _B8_B8_sediment_wasser(inputs.b8[i])))
        #        Misch2 = TRUE
        outputs.Misch2[i] = t6
        #    elif b3 is B3_schlick and b2 is B2_schlick and b8 is B8_sediment_wasser:
        t6 = min(t5, 1.0 - t6)
        t7 = min(t6, min(min(_B3_B3_schlick(inputs.b3[i]), _B2_B2_schlick(inputs.b2[i])), _B8_B8_sediment_wasser(inputs.b8[i])))
        #        Schlick = TRUE
        outputs.Schlick[i] = t7
        #    elif b16 is B16_sediment_wasser and b8 is B8_sediment_wasser:
        t7 = min(t6, 1.0 - t7)
        t8 = min(t7, min(_B16_B16_sediment_wasser(inputs.b16[i]), _B8_B8_sediment_wasser(inputs.b8[i])))
        #        schlick_t = TRUE
        outputs.schlick_t[i] = t8
        #    else:
        t8 = min(t7, 1.0 - t8)
        #        Wasser2 = TRUE
        outputs.Wasser2[i] = t8
