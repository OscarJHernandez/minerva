"""Agent tool for approximating an expression: series, rational and polynomial approximants, sums.

The model chooses a method (or `auto`) and passes only what the user said: the point, the interval,
the order or degree, the tolerance, the period, the small parameter. It never computes: every result
comes from the engine, which shows an approximation only if a check that does not use the route that
built it agreed. The card has a form for the numbers the user did not give.
"""

from __future__ import annotations

import uuid
from typing import Literal

from minerva_engine import ApproximationInput, ApproximationReport
from pydantic import Field

from .events import Card
from .tools import MAX_INPUT, SimplifyArgs, ToolContext, ToolOutput

MethodName = Literal[
    "auto",
    "taylor",
    "perturbation",
    "laurent",
    "asymptotic",
    "multivariable",
    "leading",
    "limit",
    "pade",
    "chebyshev",
    "legendre",
    "fourier",
    "piecewise",
    "continued_fraction",
    "euler_maclaurin",
    "integral_expansion",
    "root_expansion",
    "reversion",
    "laplace",
    "accelerate",
    "compare",
]

_USER = "Only if the user stated it; otherwise leave empty (the tool asks) - never invent it."


class ApproximationArgs(SimplifyArgs):
    latex: str = Field(
        max_length=MAX_INPUT,
        description="The expression to approximate in LaTeX (for a sum: the summand; for a number: the "
        "number, e.g. \\pi; for root_expansion: the equation's left side F with F = 0).",
    )
    method: MethodName = Field(
        default="auto",
        description="`auto` chooses from what the user said (a point, infinity, a small parameter, an interval, a "
        "period, a number). Name a method only when the user asked for it: taylor, laurent (poles), asymptotic "
        "(point oo or -oo), perturbation (small parameter), multivariable, leading, limit, pade, chebyshev, "
        "legendre, fourier, piecewise, continued_fraction, euler_maclaurin, integral_expansion, root_expansion, "
        "reversion, laplace, accelerate, or compare (several methods on an interval).",
    )
    variable: str = Field(
        default="x",
        min_length=1,
        max_length=32,
        description="The ONE variable of a single-variable method, a symbol of the expression (e.g. x). "
        "Several variables at once: method `multivariable` with `variables` and `points`, not this field.",
    )
    point: str | None = Field(
        default=None,
        max_length=64,
        description="Expansion point in plain text: 0, 1/2, pi, or oo / -oo for infinity. " + _USER,
    )
    direction: Literal["+", "-"] | None = Field(
        default=None, description="Which side for a one-sided limit or leading term. " + _USER
    )
    order: int | None = Field(
        default=None, ge=0, le=64, description="Order or number of terms. " + _USER
    )
    small: str | None = Field(
        default=None,
        max_length=64,
        description="The small parameter in plain text: eps, or a combination such as v/c (also the new variable "
        "name for reversion, and the large parameter M for laplace). " + _USER,
    )
    variables: list[str] = Field(
        default_factory=list, max_length=4, description="Variables for multivariable. " + _USER
    )
    points: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="One expansion point for each of `variables`. " + _USER,
    )
    lower: str | None = Field(
        default=None,
        max_length=64,
        description="Lower end of the interval or sum, plain text. " + _USER,
    )
    upper: str | None = Field(
        default=None,
        max_length=64,
        description="Upper end of the interval or sum, plain text. " + _USER,
    )
    degree: int | None = Field(
        default=None,
        ge=0,
        le=64,
        description="Polynomial degree, numerator degree, or the largest degree allowed. " + _USER,
    )
    denominator_degree: int | None = Field(
        default=None,
        ge=0,
        le=32,
        description="Denominator degree n of a [m/n] Pade approximant. " + _USER,
    )
    period: str | None = Field(
        default=None,
        max_length=64,
        description="The period of a Fourier series, plain text. " + _USER,
    )
    modes: int | None = Field(
        default=None, ge=0, le=128, description="Number of Fourier modes. " + _USER
    )
    segments: int | None = Field(
        default=None, ge=1, le=512, description="Number of piecewise segments. " + _USER
    )
    kind: str | None = Field(
        default=None,
        max_length=24,
        description="Piecewise: linear, pchip or cubic; Fourier: sigma for Lanczos factors. "
        + _USER,
    )
    tolerance: str | None = Field(
        default=None, max_length=32, description="Accuracy target as a number, e.g. 1e-6. " + _USER
    )
    tolerance_kind: Literal["absolute", "relative"] | None = Field(
        default=None,
        description="Whether the tolerance is an absolute or a relative error. " + _USER,
    )
    max_terms: int | None = Field(
        default=None, ge=1, le=128, description="Largest number of terms allowed. " + _USER
    )
    methods: list[str] = Field(
        default_factory=list, max_length=8, description="Methods to compare. " + _USER
    )


def request_of(args: ApproximationArgs) -> ApproximationInput:
    data = args.model_dump()
    data["input"] = data.pop("latex")
    data["format"] = "latex"
    return ApproximationInput(**data)


def _head(report: ApproximationReport) -> str:
    parts = [
        f"status={report.status.value}",
        f"method={report.method}",
        f"evidence={report.evidence}",
    ]
    if report.reason:
        parts.append(f"reason={report.reason}")
    return "; ".join(parts) + ". "


def known_formulas(report: ApproximationReport) -> list[str]:
    out: list[str] = []
    for text in (report.result_latex, report.statement_latex, report.remainder_latex):
        if text:
            out.append(text)
    out += [o.term_latex for o in report.orders]
    out += [o.coefficient_latex for o in report.orders]
    out += [e.latex for e in report.extras]
    out += [c.result_latex for c in report.candidates if c.result_latex and c.status.value == "ok"]
    return out


def run_approximation(args: ApproximationArgs, ctx: ToolContext) -> ToolOutput:
    request = request_of(args)
    report = ctx.engine.approximation(request)
    lines = [_head(report) + report.message]
    if report.input_latex:
        lines.append(f"input_as_read=${report.input_latex}$")
    if report.domain_latex:
        lines.append(f"domain=${report.domain_latex}$")
    if report.result_latex:
        lines.append(f"approximation=${report.result_latex}$")
    if report.statement_latex:
        lines.append(f"statement=${report.statement_latex}$")
    if report.remainder_latex:
        lines.append(f"remainder=${report.remainder_latex}$")
    for item in report.extras:
        lines.append(f"{item.label}: ${item.latex}$")
    for c in report.checks:
        lines.append(
            f"check [{'passed' if c.passed else 'FAILED' if c.passed is False else 'undecided'}]: {c.name}"
        )
    if report.error:
        lines.append(
            f"measured error on {report.error.samples} sampled points: max {report.error.max_abs}, "
            f"rms {report.error.rms}, max relative {report.error.max_rel} - a measurement, NOT a bound"
        )
    if report.tolerance_met is not None:
        lines.append(
            "tolerance "
            + ("met" if report.tolerance_met else "NOT met")
            + " on the sampled points (numerical support, not a guarantee)"
        )
    for cand in report.candidates:
        err = cand.error.max_abs if cand.error else "n/a"
        lines.append(
            f"candidate: {cand.label} [{cand.status.value}] max error {err}{' <- chosen' if cand.chosen else ''}"
        )
    if report.warnings:
        lines.append("warnings: " + ", ".join(report.warnings))
    for question in report.questions:
        lines.append(f"ASK THE USER: {question}")
    for note in report.notes[:6]:
        lines.append(f"note: {note}")
    if report.status.value == "ok":
        lines.append(
            "Say 'checked numerically' (or 'series coefficients checked'), never 'proved'. Repeat the warnings and the "
            "measured error as a measurement. An asymptotic expansion is not a convergent series."
        )
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="approximation",
        type="approximation",
        title="Approximation",
        data={"report": report.model_dump(mode="json"), "request": args.model_dump(mode="json")},
    )
    return ToolOutput(card, "\n".join(lines), known_formulas(report), report.status.value)
