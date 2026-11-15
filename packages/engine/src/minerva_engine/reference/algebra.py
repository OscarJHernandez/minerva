"""Roots, simplification and restrictions against `mpmath.polyroots`, derivatives and plain arithmetic."""

from __future__ import annotations

import mpmath as mp

from ..models import Limits
from ..service import Engine
from ..solve import solve_polynomial
from .core import Problem, text, verdict


def _real_roots(equation: str) -> list[tuple[float, int]]:
    report = solve_polynomial(equation, "x", {}, Limits())
    if report.status.value != "ok":
        raise ValueError(report.status.value)
    return [(float(r.numeric), r.multiplicity) for r in report.roots if r.real]


def _distinct_roots(
    equation: str, coefficients: list[int], expected: dict[float, int]
) -> tuple[str, str]:
    try:
        found = _real_roots(equation)
    except ValueError as exc:
        return verdict("correct", False, None, f"status {exc}")
    reference = sorted(
        {round(float(mp.re(z)), 9) for z in mp.polyroots(coefficients, maxsteps=500, extraprec=200)}
    )
    got = sorted({round(v, 9) for v, _ in found})
    multiplicities = {round(v, 9): m for v, m in found}
    agrees = got == reference and all(multiplicities[round(r, 9)] == m for r, m in expected.items())
    return verdict("correct", True, agrees, "roots vs mpmath.polyroots")


def _cubic(engine: Engine) -> tuple[str, str]:
    return _distinct_roots("x^3-6x^2+11x-6=0", [1, -6, 11, -6], {1.0: 1, 2.0: 1, 3.0: 1})


def _triple_root(engine: Engine) -> tuple[str, str]:
    """(x-1)^3: multiplicity 3 because p, p', p'' vanish at 1 and p''' does not (numerical derivatives)."""

    def p(x: mp.mpf) -> mp.mpf:
        return (x - 1) ** 3

    orders = [mp.diff(p, 1, n) for n in range(4)]
    if not (all(abs(v) < mp.mpf("1e-12") for v in orders[:3]) and abs(orders[3] - 6) < 1e-6):
        raise RuntimeError("oracle: the multiplicity is not 3")
    try:
        found = _real_roots("(x-1)^3=0")
    except ValueError as exc:
        return verdict("correct", False, None, f"status {exc}")
    return verdict(
        "correct",
        True,
        len(found) == 3 and all(abs(v - 1) < 1e-20 and m == 3 for v, m in found),
        "exact root with multiplicity 3",
    )


def _quintic(engine: Engine) -> tuple[str, str]:
    """x^5 - x - 1 has one real root (1.1673...), not expressible in radicals (Galois group S5)."""
    try:
        found = _real_roots("x^5-x-1=0")
    except ValueError as exc:
        return verdict("correct", False, None, f"status {exc}")
    root = mp.findroot(lambda x: x**5 - x - 1, 1.2)
    return verdict(
        "correct",
        True,
        len(found) == 1 and abs(found[0][0] - float(root)) < 1e-12,
        "real root vs mpmath.findroot",
    )


def _pythagorean(engine: Engine) -> tuple[str, str]:
    z = mp.mpc("0.7", "0.3")
    if abs(mp.sin(z) ** 2 + mp.cos(z) ** 2 - 1) > mp.mpf("1e-25"):
        raise RuntimeError("oracle: the identity fails numerically")
    result = engine.simplify(text("sin(x)**2+cos(x)**2"))
    proved = result.verification is not None and result.verification.symbolic_zero
    return verdict(
        "correct",
        result.status.value == "ok",
        (result.result_text or "").strip() == "1" and proved,
        "simplified to 1 with a symbolic zero",
    )


def _log_split_positive(engine: Engine) -> tuple[str, str]:
    """log(xy) - log x - log y = 0 when x, y > 0 (checked at x = 2, y = 3)."""
    if abs(mp.log(6) - mp.log(2) - mp.log(3)) > mp.mpf("1e-25"):
        raise RuntimeError("oracle: the identity fails numerically")
    result = engine.simplify(text("log(x*y)-log(x)-log(y)", x=["positive"], y=["positive"]))
    return verdict(
        "correct", result.status.value == "ok", (result.result_text or "").strip() == "0", "zero"
    )


def _log_split_unassumed(engine: Engine) -> tuple[str, str]:
    """For complex x = y = -1: log(xy) = log 1 = 0 but log x + log y = 2 pi i, so the difference is -2 pi i."""
    if abs(mp.log(1) - 2 * mp.log(-1) + 2j * mp.pi) > mp.mpf("1e-25"):
        raise RuntimeError("oracle: the counterexample fails")
    result = engine.simplify(text("log(x*y)-log(x)-log(y)"))
    claims_zero = result.status.value == "ok" and (result.result_text or "").strip() == "0"
    return verdict("refused", claims_zero, None, f"status {result.status.value}")


def _sqrt_of_square_unassumed(engine: Engine) -> tuple[str, str]:
    """For complex x = -1: sqrt(x^2) = 1 is not x; for x = i: sqrt(-1) = i is not |x| = 1. No ok result is right."""
    if not (abs(mp.sqrt(mp.mpf(-1) ** 2) - (-1)) > 1 and abs(mp.sqrt(mp.mpc(0, 1) ** 2) - 1) > 0.5):
        raise RuntimeError("oracle: the counterexamples fail")
    result = engine.simplify(text("sqrt(x**2)"))
    return verdict("refused", result.status.value == "ok", None, f"status {result.status.value}")


def _cancelled_pole_restriction(engine: Engine) -> tuple[str, str]:
    """(x-2)(x+3) / ((x-2)(x-5)) is undefined at 2 and equals (x+3)/(x-5) elsewhere (checked at x = 4)."""

    def original(x: float) -> float:
        return (x - 2) * (x + 3) / ((x - 2) * (x - 5))

    try:
        original(2.0)
    except ZeroDivisionError:
        pass
    else:  # pragma: no cover
        raise RuntimeError("oracle: the original is defined at 2")
    if abs(original(4.0) - (4 + 3) / (4 - 5)) > 1e-12:
        raise RuntimeError("oracle: the simplification is wrong")
    result = engine.simplify(text("(x-2)*(x+3)/((x-2)*(x-5))"))
    reported = r"x \neq 2" in result.restrictions_latex and result.restrictions_complete
    return verdict(
        "correct",
        result.status.value in ("ok", "unchanged"),
        reported and (result.result_text or "").replace(" ", "") == "(x+3)/(x-5)",
        "restriction x != 2 reported",
    )


PROBLEMS: list[Problem] = [
    Problem(
        "roots-cubic-distinct",
        "algebra",
        "Vieta's formulas for x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)",
        "correct",
        _cubic,
        oracle="mpmath_polyroots",
        evidence="structural",
    ),
    Problem(
        "roots-triple-root",
        "algebra",
        "multiplicity of a root: p, p', p'' vanish and p''' does not",
        "correct",
        _triple_root,
        oracle="mpmath_diff",
        evidence="structural",
    ),
    Problem(
        "roots-quintic-without-radicals",
        "algebra",
        "Galois theory: x^5 - x - 1 has Galois group S5 (not solvable by radicals)",
        "correct",
        _quintic,
        oracle="mpmath_polyroots",
        evidence="structural",
    ),
    Problem(
        "simplify-pythagorean-identity",
        "algebra",
        "DLMF §4.21 (sin^2 + cos^2 = 1)",
        "correct",
        _pythagorean,
        oracle="arithmetic",
        evidence="symbolic_proof",
    ),
    Problem(
        "simplify-log-of-product-positive",
        "algebra",
        "DLMF §4.8 (log of a product, positive arguments)",
        "correct",
        _log_split_positive,
        oracle="arithmetic",
        evidence="symbolic_proof",
    ),
    Problem(
        "simplify-log-of-product-complex-refused",
        "algebra",
        "DLMF §4.8 (the principal logarithm of a product of complex numbers)",
        "refused",
        _log_split_unassumed,
        oracle="arithmetic",
        evidence="refusal",
    ),
    Problem(
        "simplify-sqrt-of-square-complex-refused",
        "algebra",
        "DLMF §4.2 (principal square root of a complex number)",
        "refused",
        _sqrt_of_square_unassumed,
        oracle="arithmetic",
        evidence="refusal",
    ),
    Problem(
        "simplify-cancelled-pole-is-reported",
        "algebra",
        "domain of a rational function (a cancelled factor remains a point of exclusion)",
        "correct",
        _cancelled_pole_restriction,
        oracle="arithmetic",
        evidence="structural",
    ),
]
