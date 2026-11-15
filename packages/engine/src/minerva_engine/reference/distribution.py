"""Dirac delta problems: a regularised delta sequence integrated by quadrature is the oracle."""

from __future__ import annotations

from typing import Any

import mpmath as mp

from ..distributions.models import DistributionInput
from ..latexparse import parse_latex
from ..service import Engine
from .core import Judge, Problem, expr, verdict


def _delta(latex: str, lower: str, upper: str, endpoint: str | None = None) -> Any:
    def run(engine: Engine) -> Any:
        return engine.distribution(
            DistributionInput(
                expression=expr(latex, x=["real"]),
                action="integrate",
                variable="x",
                lower=lower,
                upper=upper,
                endpoint=endpoint,
            )
        )

    return run


def _gaussian_sequence(f: Any, g: Any, windows: list[list[float]], eps: str = "1e-3") -> mp.mpf:
    """integral of f(x) delta_eps(g(x)) with delta_eps(u) = exp(-(u/eps)^2) / (eps sqrt(pi))."""
    e = mp.mpf(eps)

    def seq(x: mp.mpf) -> mp.mpf:
        return f(x) * mp.exp(-((g(x) / e) ** 2)) / (e * mp.sqrt(mp.pi))

    return mp.fsum(mp.quad(seq, w) for w in windows)


def _against_sequence(
    latex: str, lower: str, upper: str, truth: mp.mpf, expected_closed_form: mp.mpf
) -> Judge:
    def run(engine: Engine) -> tuple[str, str]:
        result = _delta(latex, lower, upper)(engine)
        if result.status.value != "ok" or not result.result_latex:
            return verdict("correct", False, None, f"status {result.status.value}")
        if abs(truth - expected_closed_form) > mp.mpf("1e-4"):
            raise RuntimeError("oracle: the sequence disagrees with the textbook value")
        got = complex(parse_latex(result.result_latex).evalf(20))
        return verdict(
            "correct",
            True,
            abs(got - complex(truth)) < 1e-4,
            "result vs regularised delta sequence",
        )

    return run


def _delta_composition(engine: Engine) -> tuple[str, str]:
    """integral of (x^2+1) delta(x^2-4): sum over the roots of f(r)/|g'(r)| = 5/4 + 5/4, by a Gaussian sequence."""
    truth = _gaussian_sequence(
        lambda x: x**2 + 1, lambda x: x**2 - 4, [[-2.2, -2, -1.8], [1.8, 2, 2.2]]
    )
    return _against_sequence(
        r"(x^2+1)\operatorname{DiracDelta}(x^2-4)", r"-\infty", r"\infty", truth, mp.mpf(5) / 2
    )(engine)


def _delta_scaled(engine: Engine) -> tuple[str, str]:
    """integral of x^2 delta(2x - 6) = 3^2 / |2| = 9/2."""
    truth = _gaussian_sequence(lambda x: x**2, lambda x: 2 * x - 6, [[2.9, 3, 3.1]])
    return _against_sequence(
        r"x^2\operatorname{DiracDelta}(2x-6)", r"-\infty", r"\infty", truth, mp.mpf(9) / 2
    )(engine)


def _delta_sifting(engine: Engine) -> tuple[str, str]:
    """integral of cos(x) delta(x - 1) = cos(1) (the sifting property)."""
    truth = _gaussian_sequence(mp.cos, lambda x: x - 1, [[0.9, 1, 1.1]])
    return _against_sequence(
        r"\cos(x)\operatorname{DiracDelta}(x-1)", r"-\infty", r"\infty", truth, mp.cos(1)
    )(engine)


def _refused(latex: str, lower: str, upper: str) -> Judge:
    def run(engine: Engine) -> tuple[str, str]:
        result = _delta(latex, lower, upper)(engine)
        return verdict(
            "refused",
            result.status.value == "ok",
            None,
            f"status {result.status.value}, reason {result.reason}",
        )

    return run


def _delta_outside_the_range(engine: Engine) -> tuple[str, str]:
    """delta(x - 5) integrated over [0, 1] is 0 (the point is outside): the sequence agrees."""
    truth = _gaussian_sequence(lambda x: x, lambda x: x - 5, [[0, 1]])
    if abs(truth) > mp.mpf("1e-30"):
        raise RuntimeError("oracle: the delta sequence is not negligible outside its point")
    result = _delta(r"x\operatorname{DiracDelta}(x-5)", "0", "1")(engine)
    if result.status.value != "ok" or not result.result_latex:
        return verdict("correct", False, None, f"status {result.status.value}")
    return verdict(
        "correct",
        True,
        abs(complex(parse_latex(result.result_latex).evalf(20))) < 1e-12,
        "zero, by the sequence",
    )


def _heaviside_integral(engine: Engine) -> tuple[str, str]:
    """integral over [-1, 5] of Heaviside(x) e^{-x} = 1 - e^{-5}: plain quadrature over [0, 5]."""
    truth = mp.quad(lambda x: mp.exp(-x), [0, 5])
    result = _delta(r"\operatorname{Heaviside}(x) \exp(-x)", "-1", "5")(engine)
    if result.status.value != "ok" or not result.result_latex:
        return verdict("correct", False, None, f"status {result.status.value}")
    got = complex(parse_latex(result.result_latex).evalf(20))
    return verdict("correct", True, abs(got - complex(truth)) < 1e-12, "result vs quadrature")


PROBLEMS: list[Problem] = [
    Problem(
        "delta-composition-two-roots",
        "distribution",
        "Gel'fand and Shilov, Generalized Functions vol. 1 (delta of a function)",
        "correct",
        _delta_composition,
        oracle="mpmath_quad",
    ),
    Problem(
        "delta-repeated-root-refused",
        "distribution",
        "Gel'fand and Shilov, Generalized Functions vol. 1 (delta of a function with a repeated root is not defined)",
        "refused",
        _refused(r"\operatorname{DiracDelta}(x^2)", "-1", "1"),
        oracle="closed_form",
        evidence="refusal",
    ),
    Problem(
        "delta-on-the-limit-refused",
        "distribution",
        "convention of the integral at an endpoint (ambiguous without a stated one)",
        "refused",
        _refused(r"\operatorname{DiracDelta}(x)", "0", "1"),
        oracle="closed_form",
        evidence="refusal",
    ),
    Problem(
        "delta-scaled-argument",
        "distribution",
        "Gel'fand and Shilov, Generalized Functions vol. 1 (delta(ax) = delta(x) / |a|)",
        "correct",
        _delta_scaled,
        oracle="mpmath_quad",
    ),
    Problem(
        "delta-sifting-property",
        "distribution",
        "Gel'fand and Shilov, Generalized Functions vol. 1 (sifting property)",
        "correct",
        _delta_sifting,
        oracle="mpmath_quad",
    ),
    Problem(
        "delta-outside-the-range",
        "distribution",
        "Gel'fand and Shilov, Generalized Functions vol. 1 (support of the delta)",
        "correct",
        _delta_outside_the_range,
        oracle="mpmath_quad",
    ),
    Problem(
        "delta-squared-refused",
        "distribution",
        "Gel'fand and Shilov, Generalized Functions vol. 1 (the product of distributions is not defined in general)",
        "refused",
        _refused(r"\operatorname{DiracDelta}(x)^2", "-1", "1"),
        oracle="closed_form",
        evidence="refusal",
    ),
    Problem(
        "heaviside-weighted-integral",
        "distribution",
        "DLMF §1.16 (the Heaviside step function integrates to a truncated range)",
        "correct",
        _heaviside_integral,
        oracle="mpmath_quad",
        pending="CAP-001",
    ),
]
