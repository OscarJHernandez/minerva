"""Taylor polynomials against `mpmath.taylor`."""

from __future__ import annotations

from collections.abc import Callable

import mpmath as mp
import sympy as sp

from ..service import Engine
from ..textparse import parse_text
from .core import Judge, Problem, text, verdict

X = sp.Symbol("x")


def _taylor(source: str, function: Callable[[mp.mpf], mp.mpf], order: int = 6) -> Judge:
    def run(engine: Engine) -> tuple[str, str]:
        result = engine.series(text(source), "x", "0", order)
        if result.status.value != "ok" or not result.result_text:
            return verdict("correct", False, None, f"status {result.status.value}")
        poly = sp.Poly(parse_text(result.result_text), X)
        reference = mp.taylor(function, 0, order - 1)
        agrees = poly.degree() <= order - 1 and all(
            abs(mp.mpmathify(str(sp.N(poly.coeff_monomial(X**k), 30))) - reference[k])
            < mp.mpf("1e-20")
            for k in range(order)
        )
        return verdict("correct", True, agrees, "coefficients vs mpmath.taylor")

    return run


def _refused(source: str, why: str) -> Judge:
    def run(engine: Engine) -> tuple[str, str]:
        result = engine.series(text(source), "x", "0", 4)
        return verdict(
            "refused", result.status.value == "ok", None, f"{why}: {result.status.value}"
        )

    return run


PROBLEMS: list[Problem] = [
    Problem(
        "series-sine",
        "series",
        "DLMF §4.19 (Maclaurin series of the sine)",
        "correct",
        _taylor("sin(x)", mp.sin),
        oracle="mpmath_taylor",
    ),
    Problem(
        "series-exponential-times-cosine",
        "series",
        "Cauchy product of the Maclaurin series of exp and cos",
        "correct",
        _taylor("exp(x)*cos(x)", lambda x: mp.exp(x) * mp.cos(x)),
        oracle="mpmath_taylor",
    ),
    Problem(
        "series-arctangent",
        "series",
        "DLMF §4.24 (Maclaurin series of the arctangent)",
        "correct",
        _taylor("atan(x)", mp.atan),
        oracle="mpmath_taylor",
    ),
    Problem(
        "series-pole-refused",
        "series",
        "1/x has a pole at 0: no Taylor series (a Laurent series would be a different object)",
        "refused",
        _refused("1/x", "pole"),
        oracle="closed_form",
        evidence="refusal",
    ),
    Problem(
        "series-branch-point-refused",
        "series",
        "sqrt(x) is not analytic at 0: no Taylor series",
        "refused",
        _refused("sqrt(x)", "branch point"),
        oracle="closed_form",
        evidence="refusal",
    ),
]
