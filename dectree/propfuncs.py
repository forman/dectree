"""
Define possible functions which may be used as values for
property definitions within a fuzzy set.
"""

from .types import PropFuncResult


def true() -> PropFuncResult:
    return {}, "return 1.0"


def false() -> PropFuncResult:
    return {}, "return 0.0"


def const(t: float) -> PropFuncResult:
    return dict(t=float(t)), (
        "return {t}"
    )


def eq(x0: float, dx: float = 0.0) -> PropFuncResult:
    return dict(x0=float(x0), dx=float(dx)), (
        "return 1.0 if x == {x0} else 0.0"
        if dx == 0.0 else
        "x1 = {x0} - {dx}\n"
        "x2 = {x0}\n"
        "x3 = {x0} + {dx}\n"
        "if x <= x1:\n"
        "    return 0.0\n"
        "if x <= x2:\n"
        "    return (x - x1) / (x2 - x1)\n"
        "if x <= x3:\n"
        "    return 1.0 - (x - x2) / (x3 - x2)\n"
        "return 0.0"
    )


def ne(x0: float, dx: float = 0.0) -> PropFuncResult:
    return dict(x0=float(x0), dx=float(dx)), (
        "return 1.0 if x != {x0} else 0.0"
        if dx == 0.0 else
        "x1 = {x0} - {dx}\n"
        "x2 = {x0}\n"
        "x3 = {x0} + {dx}\n"
        "if x <= x1:\n"
        "    return 1.0\n"
        "if x <= x2:\n"
        "    return 1.0 - (x - x1) / (x2 - x1)\n"
        "if x <= x3:\n"
        "    return (x - x2) / (x3 - x2)\n"
        "return 1.0"
    )


def gt(x0: float, dx: float = 0.0) -> PropFuncResult:
    return _greater_op('>', x0, dx)


def ge(x0: float, dx: float = 0.0) -> PropFuncResult:
    return _greater_op('>=', x0, dx)


def lt(x0: float, dx: float = 0.0) -> PropFuncResult:
    return _less_op('<', x0, dx)


def le(x0: float, dx: float = 0.0) -> PropFuncResult:
    return _less_op('<=', x0, dx)


def ramp(x1: float = 0.0,
         x2: float = 1.0) -> PropFuncResult:
    return dict(x1=float(x1), x2=float(x2)), (
        "if x <= {x1}:\n"
        "    return 0.0\n"
        "if x <= {x2}:\n"
        "    return (x - {x1}) / ({x2} - {x1})\n"
        "return 1.0"
    )


def inv_ramp(x1: float = 0.0,
             x2: float = 1.0) -> PropFuncResult:
    return dict(x1=float(x1), x2=float(x2)), (
        "if x <= {x1}:\n"
        "    return 1.0\n"
        "if x <= {x2}:\n"
        "    return 1.0 - (x - {x1}) / ({x2} - {x1})\n"
        "return 0.0"
    )


def triangular(x1: float = 0.0,
               x2: float = 0.5,
               x3: float = 1.0) -> PropFuncResult:
    return dict(x1=float(x1), x2=float(x2), x3=float(x3)), (
        "if x <= {x1}:\n"
        "    return 0.0\n"
        "if x <= {x2}:\n"
        "    return (x - {x1}) / ({x2} - {x1})\n"
        "if x <= {x3}:\n"
        "    return 1.0 - (x - {x2}) / ({x3} - {x2})\n"
        "return 0.0"
    )


def inv_triangular(x1: float = 0.0,
                   x2: float = 0.5,
                   x3: float = 1.0) -> PropFuncResult:
    return dict(x1=float(x1), x2=float(x2), x3=float(x3)), (
        "if x <= {x1}:\n"
        "    return 1.0\n"
        "if x <= {x2}:\n"
        "    return 1.0 - (x - {x1}) / ({x2} - {x1})\n"
        "if x <= {x3}:\n"
        "    return (x - {x2}) / ({x3} - {x2})\n"
        "return 1.0"
    )


def trapezoid(x1: float = 0.0,
              x2: float = 1.0 / 3.0,
              x3: float = 2.0 / 3.0,
              x4: float = 1.0) -> PropFuncResult:
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


def inv_trapezoid(x1: float = 0.0,
                  x2: float = 1.0 / 3.0,
                  x3: float = 2.0 / 3.0,
                  x4: float = 1.0) -> PropFuncResult:
    return dict(x1=float(x1), x2=float(x2), x3=float(x3), x4=float(x4)), (
        "if x <= {x1}:\n"
        "    return 1.0\n"
        "if x <= {x2}:\n"
        "    return 1.0 - (x - {x1}) / ({x2} - {x1})\n"
        "if x <= {x3}:\n"
        "    return 0.0\n"
        "if x <= {x4}:\n"
        "    return (x - {x3}) / ({x4} - {x3})\n"
        "return 1.0"
    )


def _greater_op(op: str, x0: float, dx: float) -> PropFuncResult:
    return dict(x0=float(x0), dx=float(dx)), (
        ("return 1.0 if x %s {x0} else 0.0" % op)
        if dx == 0.0 else
        ("x1 = {x0} - {dx}\n"
         "x2 = {x0} + {dx}\n"
         "if x <= x1:\n"
         "    return 0.0\n"
         "if x <= x2:\n"
         "    return (x - x1) / (x2 - x1)\n"
         "return 1.0")
    )


def _less_op(op: str, x0: float, dx: float) -> PropFuncResult:
    return dict(x0=float(x0), dx=float(dx)), (
        ("return 1.0 if x %s {x0} else 0.0" % op)
        if dx == 0.0 else
        ("x1 = {x0} - {dx}\n"
         "x2 = {x0} + {dx}\n"
         "if x <= x1:\n"
         "    return 1.0\n"
         "if x <= x2:\n"
         "    return 1.0 - (x - x1) / (x2 - x1)\n"
         "return 0.0")
    )
