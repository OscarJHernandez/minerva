"""Agent tool for creative telescoping of a parameterised definite integral.

The model names the integrand, the variable, the parameter and the bounds the USER gave. The engine
searches for an operator L and a certificate G with L f = dG/dx, proves that identity, analyses the
boundary and convergence under the declared assumptions, derives the differential equation of the
integral and evaluates it only if every condition and independent check agrees. The model never
supplies an operator, a certificate, an equation or an integration constant: the arguments below
have no field for any of them, and the search bounds are the engine's.
"""

from __future__ import annotations

import uuid

from minerva_engine import TelescopingInput, TelescopingResult
from pydantic import Field

from .events import Card
from .tools import MAX_INPUT, SimplifyArgs, ToolContext, ToolOutput


class TelescopingArgs(SimplifyArgs):
    latex: str = Field(
        max_length=MAX_INPUT,
        description="The integrand f(x, a) in LaTeX, e.g. \\exp(-a x^2) (no \\int or dx). Only "
        "products of rational functions, powers and exponentials of rational functions qualify.",
    )
    variable: str = Field(
        min_length=1, max_length=32, description="The variable of integration, e.g. x"
    )
    parameter: str = Field(
        min_length=1,
        max_length=32,
        description="The parameter the integral depends on, e.g. a. Declare it real and, where the "
        "problem says so, positive: convergence is decided from the declared assumptions only.",
    )
    lower: str | None = Field(
        default=None,
        max_length=64,
        description="Lower bound: a real number or -oo. Give both bounds or neither (neither: only "
        "the telescoping certificate is searched).",
    )
    upper: str | None = Field(
        default=None, max_length=64, description="Upper bound: a real number or oo."
    )


def _head(result: TelescopingResult) -> str:
    parts = [f"status={result.status.value}", f"outcome={result.outcome}"]
    if result.reason:
        parts.append(f"reason={result.reason}")
    return "; ".join(parts) + ". "


def run_creative_telescoping(args: TelescopingArgs, ctx: ToolContext) -> ToolOutput:
    request = TelescopingInput(
        input=args.latex,
        format="latex",
        assumptions=args.assumptions,
        variable=args.variable,
        parameter=args.parameter,
        lower=args.lower,
        upper=args.upper,
    )
    result = ctx.engine.creative_telescoping(request)
    lines = [_head(result) + result.message]
    if result.input_latex:
        lines.append(f"input_as_read=${result.input_latex}$")
    if result.operator:
        lines.append(f"operator (order {result.operator.order}): ${result.operator.latex}$")
    if result.certificate:
        check = result.certificate_check
        proof = (
            "PROVED exactly (L f - dG/dx simplifies to zero)"
            if check and check.symbolic_zero and check.passed
            else "not verified"
        )
        lines.append(f"certificate G: ${result.certificate.latex}$ ({proof})")
    for c in result.conditions:
        lines.append(
            f"condition [{'established' if c.established else 'NOT established'}]: ${c.latex}$ "
            f"({c.how})"
        )
    if result.boundary:
        lines.append(
            f"boundary contribution G(upper) - G(lower) = ${result.boundary.contribution_latex}$"
        )
    if result.equation:
        kind = "homogeneous" if result.equation.homogeneous else "inhomogeneous"
        lines.append(f"equation ({kind}): ${result.equation.latex}$")
    if result.solution:
        lines.append(f"general solution: ${result.solution.general_latex}$")
        lines.append(
            "initial values (independently verified): "
            + "; ".join(f"${v}$" for v in result.solution.initial_values_latex)
        )
    if result.result_latex:
        lines.append(
            f"value: $I = {result.result_latex}$ - derived from the equation and CHECKED against "
            "numerical quadrature; say 'checked numerically', never 'proved'."
        )
    for item in result.unresolved:
        lines.append(f"unresolved: {item}")
    if result.outcome in ("certificate", "equation"):
        lines.append(
            "PARTIAL RESULT: say plainly that the integral itself has NOT been evaluated; state "
            "only what is shown above and what remains unresolved."
        )
    for note in result.notes:
        lines.append(f"note: {note}")
    lines.append(
        "A telescoping relation found within the searched families is not a statement about other "
        "families; a missing relation does not prove that none exists."
    )
    known: list[str] = []
    if result.operator:
        known += [result.operator.latex, *result.operator.coefficients_latex]
    if result.certificate:
        known.append(result.certificate.latex)
    if result.equation:
        known.append(result.equation.latex)
    if result.boundary:
        known.append(result.boundary.contribution_latex)
    if result.solution:
        known.append(result.solution.general_latex)
    if result.result_latex:
        known.append(result.result_latex)
    known += [c.latex for c in result.conditions]
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="creative_telescoping",
        type="telescoping",
        title="Creative telescoping",
        data={**result.model_dump(mode="json"), "request": args.model_dump(mode="json")},
    )
    return ToolOutput(card, "\n".join(lines), known, result.status.value)
