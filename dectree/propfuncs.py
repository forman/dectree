# y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def true():
    return "return 1.0"


def false():
    return "return 0.0"


def ramp_down(x1: float = 0.0, x2: float = 0.5) -> str:
    code = ("if x < {x1}:\n"
            "    return 1.0\n"
            "if x > {x2}:\n"
            "    return 0.0\n"
            "return 1.0 - (x - {x1}) / ({x2} - {x1})")
    return code.format(x1=float(x1), x2=float(x2))


def ramp_up(x1: float = 0.5, x2: float = 1.0) -> str:
    code = ("if x < {x1}:\n"
            "    return 0.0\n"
            "if x > {x2}:\n"
            "    return 1.0\n"
            "return (x - {x1}) / ({x2} - {x1})")
    return code.format(x1=float(x1), x2=float(x2))


def triangle(x1: float = 0.0, x2: float = 0.5, x3: float = 1.0) -> str:
    code = ("if x < {x1}:\n"
            "    return 0.0\n"
            "if x > {x3}:\n"
            "    return 0.0\n"
            "if x < {x2}:\n"
            "    return (x - {x1}) / ({x2} - {x1})\n"
            "return 1.0 - (x - {x2}) / ({x3} - {x2})")
    return code.format(x1=float(x1), x2=float(x2), x3=float(x3))
