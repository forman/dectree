"""
Define possible functions which may be used as values for property definitions within a fuzzy set.
"""

from typing import Tuple, Dict, Any

PropFuncParamName = str
PropFuncParamValue = Any
PropFuncParams = Dict[PropFuncParamName, PropFuncParamValue]
PropFuncBody = str
PropFuncResult = Tuple[PropFuncParams, PropFuncBody]


def true() -> PropFuncResult:
    return {}, "return 1.0"


def false() -> PropFuncResult:
    return {}, "return 0.0"


def const(t: float) -> PropFuncResult:
    return dict(t=float(t)), (
        "return {t}"
    )


def eq(x1: float) -> PropFuncResult:
    return dict(x1=float(x1)), (
        "return 1.0 if x == {x1} else 0.0"
    )


def ne(x1: float) -> PropFuncResult:
    return dict(x1=float(x1)), (
        "return 1.0 if x != {x1} else 0.0"
    )


def gt(x1: float) -> PropFuncResult:
    return dict(x1=float(x1)), (
        "return 1.0 if x > {x1} else 0.0"
    )


def ge(x1: float) -> PropFuncResult:
    return dict(x1=float(x1)), (
        "return 1.0 if x >= {x1} else 0.0"
    )


def lt(x1: float) -> PropFuncResult:
    return dict(x1=float(x1)), (
        "return 1.0 if x < {x1} else 0.0"
    )


def le(x1: float) -> PropFuncResult:
    return dict(x1=float(x1)), (
        "return 1.0 if x <= {x1} else 0.0"
    )


def ramp_down(x1: float = 0.0, x2: float = 0.5) -> PropFuncResult:
    return dict(x1=float(x1), x2=float(x2)), (
        "if x <= {x1}:\n"
        "    return 1.0\n"
        "if x <= {x2}:\n"
        "    return 1.0 - (x - {x1}) / ({x2} - {x1})\n"
        "return 0.0"
    )


def ramp_up(x1: float = 0.5, x2: float = 1.0) -> PropFuncResult:
    return dict(x1=float(x1), x2=float(x2)), (
        "if x <= {x1}:\n"
        "    return 0.0\n"
        "if x <= {x2}:\n"
        "    return (x - {x1}) / ({x2} - {x1})\n"
        "return 1.0"
    )


def triangular(x1: float = 0.0, x2: float = 0.5, x3: float = 1.0) -> PropFuncResult:
    return dict(x1=float(x1), x2=float(x2), x3=float(x3)), (
        "if x <= {x1}:\n"
        "    return 0.0\n"
        "if x <= {x2}:\n"
        "    return (x - {x1}) / ({x2} - {x1})\n"
        "if x <= {x3}:\n"
        "    return 1.0 - (x - {x2}) / ({x3} - {x2})\n"
        "return 0.0"
    )


def trapezoid(x1: float = 0.0, x2: float = 1.0 / 3.0, x3: float = 2.0 / 3.0, x4: float = 1.0) -> PropFuncResult:
    return dict(x1=float(x1), x2=float(x2), x3=float(x3), x4=float(x4)), (
        "if x <= {x1}:\n"
        "    return 0.0\n"
        "if x <= {x2}:\n"
        "    return (x - {x1}) / ({x2} - {x1})\n"
        "if x <= {x3}:\n"
        "    return 1.0\n"
        "if x <= {x4}:\n"
        "    return 1.0 - (x - {x3}) / ({x4} - {x3})\n"
        "return 0.0"
    )
