"""Integrals, antiderivatives and derivatives against mpmath quadrature and numerical differentiation."""

from __future__ import annotations

from collections.abc import Callable

import mpmath as mp
import sympy as sp

from ..differentiate import DiffVariable
from ..service import Engine
from ..textparse import parse_text
from .core import Judge, Problem, text, verdict

X = sp.Symbol("x")


def _definite(
    source: str,
    lower: str,
    upper: str,
    truth: Callable[[], mp.mpf],
    closed_form: mp.mpf,
    tol: float = 1e-12,
) -> Judge:
    """A definite integral: Minerva's number against quadrature, with the textbook closed form as a premise check."""

    def run(engine: Engine) -> tuple[str, str]:
        measured = truth()
        if abs(measured - closed_form) > mp.mpf("1e-15") * max(1, abs(closed_form)):
            raise RuntimeError("oracle: quadrature and the closed form disagree")
        result = engine.integrate(text(source), "x", lower, upper)
        if result.status.value != "ok" or result.numeric is None:
            return verdict("correct", False, None, f"status {result.status.value}")
        got = mp.mpmathify(result.numeric)
        return verdict(
            "correct",
            True,
            abs(got - measured) <= tol * max(1, abs(measured)),
            "value vs quadrature",
        )

    return run


def _refused_integral(
    source: str, lower: str, upper: str, premise: Callable[[], bool], why: str
) -> Judge:
    def run(engine: Engine) -> tuple[str, str]:
        if not premise():
            raise RuntimeError(f"oracle: {why} does not hold, the problem is ill-posed")
        result = engine.integrate(text(source), "x", lower, upper)
        return verdict(
            "refused", result.status.value == "ok", None, f"status {result.status.value}"
        )

    return run


def _antiderivative(source: str, points: list[float]) -> Judge:
    """The reported antiderivative differentiates (mpmath, numerically) to the integrand at each point."""

    def run(engine: Engine) -> tuple[str, str]:
        result = engine.integrate(text(source), "x")
        if result.status.value != "ok" or not result.result_text:
            return verdict("correct", False, None, f"status {result.status.value}")
        found = sp.lambdify(X, parse_text(result.result_text), "mpmath")
        integrand = sp.lambdify(X, parse_text(source), "mpmath")
        agrees = all(
            abs(mp.diff(found, mp.mpf(p)) - integrand(mp.mpf(p))) <= mp.mpf("1e-20") for p in points
        )
        return verdict("correct", True, agrees, "mpmath derivative of the result vs the integrand")

    return run


def _derivative(source: str, reference: Callable[[mp.mpf], mp.mpf], points: list[float]) -> Judge:
    """The reported derivative against mpmath's numerical derivative of the original function."""

    def run(engine: Engine) -> tuple[str, str]:
        result = engine.differentiate(text(source), [DiffVariable(name="x")])
        if result.status.value != "ok" or not result.result_text:
            return verdict("correct", False, None, f"status {result.status.value}")
        found = sp.lambdify(X, parse_text(result.result_text), "mpmath")
        agrees = all(
            abs(found(mp.mpf(p)) - mp.diff(reference, mp.mpf(p))) <= mp.mpf("1e-18") for p in points
        )
        return verdict("correct", True, agrees, "result vs mpmath numerical derivative")

    return run


def _derivative_refused(source: str, why: str) -> Judge:
    def run(engine: Engine) -> tuple[str, str]:
        result = engine.differentiate(text(source), [DiffVariable(name="x")])
        return verdict(
            "refused", result.status.value == "ok", None, f"{why}: {result.status.value}"
        )

    return run


def _diverges_at_zero() -> bool:
    """The integral of 1/x from eps to 1 grows without bound as eps -> 0: no value, only a principal value."""
    small = mp.quad(lambda x: 1 / x, [mp.mpf("1e-12"), 1])
    large = mp.quad(lambda x: 1 / x, [mp.mpf("1e-6"), 1])
    return bool(small - large > 13)  # ln(1e6) = 13.8


def _oscillates() -> bool:
    """The integral of sin from 0 to T is 1 - cos T: it has no limit as T -> infinity."""
    # T = 100 and T = 100 + pi give 1 - cos T = 0.14 and 1.86: the partial integrals keep moving by about 1.7.
    return bool(abs(mp.quad(mp.sin, [0, 100]) - mp.quad(mp.sin, [0, 100 + mp.pi])) > 1)


PROBLEMS: list[Problem] = [
    Problem(
        "integral-beta-function",
        "calculus",
        "DLMF §5.12 (the beta integral)",
        "correct",
        _definite(
            "x**2*(1-x)**3",
            "0",
            "1",
            lambda: mp.quad(lambda x: x**2 * (1 - x) ** 3, [0, 1]),
            mp.beta(3, 4),
        ),
        oracle="mpmath_quad",
    ),
    Problem(
        "integral-gaussian-full-line",
        "calculus",
        "DLMF §5.4 (Gamma(1/2) = sqrt(pi))",
        "correct",
        _definite(
            "exp(-x**2)",
            "-oo",
            "oo",
            lambda: mp.quad(lambda x: mp.exp(-(x**2)), [-mp.inf, 0, mp.inf]),
            mp.sqrt(mp.pi),
        ),
        oracle="mpmath_quad",
    ),
    Problem(
        "integral-sine-squared",
        "calculus",
        "DLMF §4.21 (double-angle formulas)",
        "correct",
        _definite(
            "sin(x)**2", "0", "pi", lambda: mp.quad(lambda x: mp.sin(x) ** 2, [0, mp.pi]), mp.pi / 2
        ),
        oracle="mpmath_quad",
    ),
    Problem(
        "integral-three-halves-power",
        "calculus",
        "elementary power rule",
        "correct",
        _definite(
            "x**(3/2)",
            "0",
            "4",
            lambda: mp.quad(lambda x: x ** mp.mpf("1.5"), [0, 4]),
            mp.mpf(64) / 5,
        ),
        oracle="mpmath_quad",
    ),
    Problem(
        "antiderivative-arctangent",
        "calculus",
        "DLMF §4.26 (integrals of rational functions: arctangent)",
        "correct",
        _antiderivative("1/(x**2+4)", [0.3, 1.7, -2.2]),
        oracle="mpmath_diff",
    ),
    Problem(
        "antiderivative-by-parts",
        "calculus",
        "integration by parts",
        "correct",
        _antiderivative("x*exp(x)", [0.3, 1.7, -2.2]),
        oracle="mpmath_diff",
    ),
    Problem(
        "derivative-x-to-the-x",
        "calculus",
        "logarithmic differentiation",
        "correct",
        _derivative("x**x", lambda x: x**x, [0.5, 1.7, 3.1]),
        oracle="mpmath_diff",
    ),
    Problem(
        "derivative-product-with-gaussian",
        "calculus",
        "product and chain rules",
        "correct",
        _derivative("sin(x)*exp(x**2)", lambda x: mp.sin(x) * mp.exp(x**2), [0.4, 1.1, -0.9]),
        oracle="mpmath_diff",
    ),
    Problem(
        "derivative-absolute-value-refused",
        "calculus",
        "|z| is nowhere complex-differentiable (Cauchy-Riemann)",
        "refused",
        _derivative_refused("Abs(x)", "no complex derivative"),
        oracle="closed_form",
        evidence="refusal",
    ),
    Problem(
        "integral-reciprocal-across-the-pole-refused",
        "calculus",
        "improper integrals: 1/x on [-1, 1] has no value (only a principal value)",
        "refused",
        _refused_integral("1/x", "-1", "1", _diverges_at_zero, "divergence at 0"),
        oracle="mpmath_quad",
        evidence="refusal",
    ),
    Problem(
        "integral-oscillating-without-a-limit-refused",
        "calculus",
        "improper integrals: the integral of sin over [0, infinity) does not converge",
        "refused",
        _refused_integral("sin(x)", "0", "oo", _oscillates, "oscillation"),
        oracle="mpmath_quad",
        evidence="refusal",
    ),
]
