"""Variational calculus: stationary paths and first variations against quadrature and finite differences."""

from __future__ import annotations

import mpmath as mp

from ..service import Engine
from .core import Problem, verdict


def _variation_action(engine: Engine) -> tuple[str, str]:
    """The stationary path of a harmonic action: S = int (x'^2 - x^2)/2 for x = 2 sin t / sin 1, by quadrature."""
    from ..variational import VariationInput

    report = engine.variation(
        VariationInput(
            action="solve",
            integrand=r"\frac12\dot{x}^2-\frac12 x^2",
            functions=["x"],
            lower="0",
            upper="1",
            conditions=[
                {"function": "x", "order": 0, "end": "lower", "kind": "fixed", "value": "0"},
                {"function": "x", "order": 0, "end": "upper", "kind": "fixed", "value": "2"},
            ],
        )
    )
    if report.status.value != "ok" or report.solution is None or not report.solution.paths:
        return verdict("correct", False, None, f"status {report.status.value}")

    def energy(s: mp.mpf) -> mp.mpf:
        return ((2 * mp.cos(s) / mp.sin(1)) ** 2 - (2 * mp.sin(s) / mp.sin(1)) ** 2) / 2

    exact = mp.quad(energy, [0, 1])
    found = mp.mpf(report.solution.paths[0].action)
    return verdict(
        "correct", True, abs(found - exact) < 1e-9, "action vs quadrature of the exact path"
    )


def _variation_first(engine: Engine) -> tuple[str, str]:
    """dS along eta for the arc-length functional on x = t^2, eta = t^3: central difference by quadrature."""
    from ..variational import VariationInput

    report = engine.variation(
        VariationInput(
            action="verify",
            integrand=r"\sqrt{1+\dot{x}^2}",
            functions=["x"],
            lower="0",
            upper="1",
            paths=[{"analytic": {"x": "t^2"}}],
            perturbations=[{"expressions": {"x": "t^3"}}],
            automatic_perturbations=False,
        )
    )
    if report.status.value != "ok" or report.verification is None:
        return verdict("correct", False, None, f"status {report.status.value}")

    def action(e: mp.mpf) -> mp.mpf:
        return mp.quad(lambda t: mp.sqrt(1 + (2 * t + 3 * e * t**2) ** 2), [0, 1])

    h = mp.mpf("1e-5")
    reference = (action(h) - action(-h)) / (2 * h)
    claimed = mp.mpf(report.verification.perturbations[0].analytic)
    return verdict(
        "correct", True, abs(claimed - reference) < 1e-8, "first variation vs central difference"
    )


def _variation_nonsmooth_refused(engine: Engine) -> tuple[str, str]:
    """|x'| has no first variation at x' = 0 and is outside the smooth theory: it must be refused."""
    from ..variational import VariationInput

    report = engine.variation(
        VariationInput(
            action="first", integrand=r"|\dot{x}|", functions=["x"], lower="0", upper="1"
        )
    )
    return verdict("refused", report.status.value == "ok", None, f"status {report.status.value}")


PROBLEMS: list[Problem] = [
    Problem(
        "variation-stationary-action-harmonic",
        "variational",
        "closed-form stationary path of the harmonic action (quadrature of x = 2 sin t / sin 1)",
        "correct",
        _variation_action,
        oracle="mpmath_quad",
    ),
    Problem(
        "variation-first-variation-arc-length",
        "variational",
        "central difference of the arc-length action (mpmath quadrature), independent of the symbolic variation",
        "correct",
        _variation_first,
        oracle="finite_difference",
    ),
    Problem(
        "variation-nonsmooth-integrand-refused",
        "variational",
        "the first variation needs a smooth integrand (|x'| is not differentiable at x' = 0)",
        "refused",
        _variation_nonsmooth_refused,
        oracle="closed_form",
        evidence="refusal",
    ),
]
