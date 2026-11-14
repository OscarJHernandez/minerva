"""Substitution problems: premises supplied by the user make a result conditional."""

from __future__ import annotations

import sympy as sp

from ..latexparse import parse_latex
from ..service import Engine
from .core import Problem, expr, verdict


def _substitution(engine: Engine) -> tuple[str, str]:
    """x^2 + y with the premise y = 3x: the result at x = 2 is 4 + 6, by plain arithmetic."""
    step = engine.substitute(expr("x^2+y"), ["y = 3x"])
    if step.status.value != "ok" or not step.result_latex:
        return verdict("correct", False, None, f"status {step.status.value}")
    x = sp.Symbol("x")
    value = parse_latex(step.result_latex).subs(x, 2)
    return verdict(
        "correct",
        True,
        value == 10 and step.evidence == "conditional",
        "value at x = 2 and conditional evidence",
    )


def _substitution_order(engine: Engine) -> tuple[str, str]:
    """x = y and y = 2 applied to x + y: the order changes the result (4 versus y + 2), so Minerva must ask."""
    x, y = sp.symbols("x y")
    in_order = (x + y).subs(x, y).subs(y, 2)
    at_once = (x + y).subs({x: y, y: 2}, simultaneous=True)
    if in_order == at_once:  # the oracle: the two orders disagree
        raise RuntimeError("oracle: the order does not matter, the problem is ill-posed")
    step = engine.substitute(expr("x+y"), ["x = y", "y = 2"])
    return verdict("refused", step.status.value == "ok", None, f"status {step.status.value}")


PROBLEMS: list[Problem] = [
    Problem(
        "substitution-with-premise",
        "substitution",
        "arithmetic",
        "correct",
        _substitution,
        oracle="arithmetic",
        evidence="conditional",
    ),
    Problem(
        "substitution-order-refused",
        "substitution",
        "arithmetic (the two orders differ)",
        "refused",
        _substitution_order,
        oracle="arithmetic",
        evidence="refusal",
    ),
]
