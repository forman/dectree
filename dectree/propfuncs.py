"""
Define possible functions which may be used as values for property definitions within a fuzzy set.
"""

from typing import Tuple, Dict, Any

_PropFuncParamName = str
_PropFuncParamValue = Any
_PropFuncParams = Dict[_PropFuncParamName, _PropFuncParamValue]
_PropFuncBody = str
_PropFuncResult = Tuple[_PropFuncParams, _PropFuncBody]


def true() -> _PropFuncResult:
    return {}, "return 1.0"


def false() -> _PropFuncResult:
    return {}, "return 0.0"


def ramp_down(x1: float = 0.0, x2: float = 0.5) -> _PropFuncResult:
    return dict(x1=float(x1), x2=float(x2)), (
        "if x < {x1}:\n"
        "    return 1.0\n"
        "if x < {x2}:\n"
        "    return 1.0 - (x - {x1}) / ({x2} - {x1})\n"
        "return 0.0"
    )


def ramp_up(x1: float = 0.5, x2: float = 1.0) -> _PropFuncResult:
    return dict(x1=float(x1), x2=float(x2)), (
        "if x < {x1}:\n"
        "    return 0.0\n"
        "if x < {x2}:\n"
        "    return (x - {x1}) / ({x2} - {x1})\n"
        "return 1.0"
    )


def triangular(x1: float = 0.0, x2: float = 0.5, x3: float = 1.0) -> _PropFuncResult:
    return dict(x1=float(x1), x2=float(x2), x3=float(x3)), (
        "if x < {x1}:\n"
        "    return 0.0\n"
        "if x < {x2}:\n"
        "    return (x - {x1}) / ({x2} - {x1})\n"
        "if x < {x3}:\n"
        "    return 1.0 - (x - {x2}) / ({x3} - {x2})\n"
        "return 0.0"
    )


def trapezoid(x1: float = 0.0, x2: float = 1.0 / 3.0, x3: float = 2.0 / 3.0, x4: float = 1.0) -> _PropFuncResult:
    return dict(x1=float(x1), x2=float(x2), x3=float(x3), x4=float(x4)), (
        "if x < {x1}:\n"
        "    return 0.0\n"
        "if x < {x2}:\n"
        "    return (x - {x1}) / ({x2} - {x1})\n"
        "if x < {x3}:\n"
        "    return 1.0\n"
        "if x < {x4}:\n"
        "    return 1.0 - (x - {x3}) / ({x4} - {x3})\n"
        "return 0.0"
    )
