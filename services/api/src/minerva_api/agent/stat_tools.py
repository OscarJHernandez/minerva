"""Agent tools for finite sums, expectations and statistical models.

The model chooses an operation and passes what the USER said; the engine derives and verifies. Three rules
are repeated in every description because the model is the weak point: independence is stated by the
user or the tool refuses; sizes, data and priors are the user's, never invented; a result that holds
given a premise is said to hold *given* it.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from minerva_engine import ExpectationInput, StatModelInput, SumsInput
from pydantic import BaseModel, Field

from .events import Card
from .tools import SimplifyArgs, ToolContext, ToolOutput, _expr

CONDITIONAL = "holds only given a premise that Minerva does not check"


class SumsArgs(SimplifyArgs):
    latex: str = Field(
        max_length=2000,
        description=(
            "An expression with finite sums or products in LaTeX, e.g. \\sum_{i=1}^{n}(x_i-\\mu)^2 "
            "or \\log\\prod_{i=1}^{n}x_i. The body of \\sum extends to the next + or -. A subscript "
            "that is the summation index makes x_i an element of the data x. Declare the upper bound "
            'an integer and positive in assumptions, e.g. {"n": ["integer", "positive"]}; infinite '
            "sums and x_{ij} are refused."
        ),
    )
    action: Literal["expand", "evaluate_constant", "factor", "log_of_product", "differentiate"] = (
        Field(
            default="expand",
            description="expand: linearity (split a sum, pull out what does not depend on the index). "
            "evaluate_constant: closed form of a sum whose summand has no data (n c, n(n+1)/2). factor: a "
            "double sum of a product becomes a product of sums. log_of_product: log of a product becomes "
            "a sum of logs, only if every factor is provably positive (declare it). differentiate needs "
            "`variable`.",
        )
    )
    variable: str | None = Field(
        default=None,
        max_length=32,
        description="Variable of differentiation (action differentiate)",
    )


class ExpectationArgs(SimplifyArgs):
    latex: str = Field(
        max_length=2000,
        description="An expression in E, Var and Cov, e.g. \\operatorname{Var}(X+Y) or "
        "\\mathbb{E}[aX+bY+c] or \\mathbb{E}[\\sum_{i=1}^{n}X_i].",
    )
    random: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Names of the random variables in the expression. A symbol not listed here is a "
        "constant (and an expectation with none of them is refused).",
    )
    independent: list[list[str]] = Field(
        default_factory=list,
        max_length=8,
        description='Groups of random variables the USER said are independent, e.g. [["X", "Y"]]; '
        '[["X"]] for an indexed X means its components are mutually independent. NEVER add a group '
        "on your own: without one, covariance terms are kept. A result that used a group holds "
        "only GIVEN that independence.",
    )
    action: Literal["expand", "to_moments"] = Field(
        default="expand",
        description="expand: linearity of E, bilinearity of Cov. to_moments: variance and covariance "
        "through raw moments E[X^2] - E[X]^2.",
    )


class StatModelArgs(BaseModel):
    """Arguments of `statistical_model`. Everything not given by the user stays empty."""

    family: str = Field(
        max_length=32,
        description="normal, bernoulli, binomial, poisson, beta, gamma (shape and rate), cauchy or "
        "uniform_0_theta. Anything else is refused.",
    )
    parametrization: str | None = Field(
        default=None,
        max_length=16,
        description='Only for normal: "sd" (parameters mu, sigma; the default) or "variance" (mu, v).',
    )
    parameters: dict[str, str] = Field(
        default_factory=dict,
        max_length=8,
        description="Parameters whose value is KNOWN, as expressions (names as in the family: normal "
        "mu/sigma or mu/v, bernoulli p, binomial n/p, poisson lam, beta/gamma alpha/beta). "
        "A parameter to be estimated is listed in `estimate` instead, not here.",
    )
    observation: str | None = Field(
        default=None, max_length=32, description="Name of the data, e.g. x (x_1 ... x_n)"
    )
    sample_size: str | None = Field(
        default=None, max_length=32, description="Name of the number of observations, e.g. n"
    )
    iid: bool | None = Field(
        default=None,
        description="Set true ONLY if the USER said the observations are independent and identically "
        "distributed. Never assume it: with several observations and no statement, the tool refuses "
        "and you must ask.",
    )
    estimate: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Parameters to estimate (mle), to differentiate by (score), whose information is "
        "wanted (fisher_information) or to update (posterior: exactly one).",
    )
    action: Literal[
        "moments", "log_likelihood", "score", "mle", "fisher_information", "posterior"
    ] = Field(
        description="moments: mean, variance and E[X^2] derived from the density. log_likelihood and "
        "score. mle: the maximum-likelihood estimator, with its admissibility conditions and whether "
        "the optimum is global (only when concavity is proved) or local. fisher_information. "
        "posterior: needs `prior`; found by matching the kernel, never from a table.",
    )
    prior: dict[str, Any] | None = Field(
        default=None,
        description='Only the USER\'s prior: {"family": "beta", "parameters": {"alpha": "1", "beta": "1"}}. '
        "Never choose a prior. The result holds for the prior given.",
    )
    data: list[str] | None = Field(
        default=None,
        max_length=64,
        description="Concrete observations, only if the USER gave them (numbers as text).",
    )
    assumptions: dict[str, list[str]] = Field(
        default_factory=dict,
        description='Properties of symbols, e.g. {"n": ["integer", "positive"], "sigma": ["positive"]}',
    )


# ---------------------------------------------------------------------------------- helpers


def _lines_for_checks(result: Any) -> list[str]:
    return [
        f"- check ({c.kind}): {c.method}: {c.points_agreed}/{c.points_checked}"
        + ("" if c.passed else " FAILED")
        for c in result.checks
    ]


def _conditions(result: Any) -> list[str]:
    return [
        f"- condition ${c.latex}$: {'established' if c.established else 'NOT established (' + c.how + ')'}"
        for c in result.conditions
    ]


def _known_and_quoted(
    formulas: list[str], conditional: bool, premises: list[str]
) -> tuple[list[str], list[tuple[str, str]]]:
    """A result that holds given a premise is quoted, never marked verified."""
    formulas = [f for f in formulas if f]
    if not conditional:
        return formulas, []
    why = f"{CONDITIONAL}: " + "; ".join(premises)
    return [], [(f, why) for f in formulas]


# ---------------------------------------------------------------------------------- sums


def run_sums(args: SumsArgs, ctx: ToolContext) -> ToolOutput:
    request = SumsInput(expression=_expr(args), action=args.action, variable=args.variable)
    result = ctx.engine.sums(request)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="sums",
        type="sums",
        title="Finite sums",
        data={
            "result": result.model_dump(mode="json"),
            "request": {
                **request.model_dump(mode="json", exclude={"expression"}),
                "latex": args.latex,
                "assumptions": args.assumptions,
            },
        },
    )
    head = f"status={result.status.value}; action={result.action}"
    if result.input_latex:
        head += f"; input_as_read=${result.input_latex}$"
    lines: list[str] = []
    if result.status.value == "ok":
        head += f"; result=${result.result_latex}$"
        if result.verification:
            head += (
                f"; {result.verification.instances_agreed}/{result.verification.instances_checked} "
                f"explicit instances agree ({result.message}). This is NOT a proof for every n: say "
                "'checked for these sizes'"
            )
        lines.extend(_conditions(result))
    else:
        head += f"; result=none ({result.reason or result.message})"
        lines.append(result.message)
        lines.extend(_conditions(result))
    known = [
        result.input_latex or "",
        result.result_latex or "",
        *[s.after_latex for s in result.steps],
    ]
    return ToolOutput(card, "\n".join([head, *lines]), [k for k in known if k], result.status.value)


# ---------------------------------------------------------------------------------- expectation


def run_expectation(args: ExpectationArgs, ctx: ToolContext) -> ToolOutput:
    request = ExpectationInput(
        expression=_expr(args), random=args.random, independent=args.independent, action=args.action
    )
    result = ctx.engine.expectation(request)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="expectation",
        type="expectation",
        title="Expectation, variance, covariance",
        data={
            "result": result.model_dump(mode="json"),
            "request": {
                **request.model_dump(mode="json", exclude={"expression"}),
                "latex": args.latex,
                "assumptions": args.assumptions,
            },
        },
    )
    head = f"status={result.status.value}; action={result.action}"
    if result.input_latex:
        head += f"; input_as_read=${result.input_latex}$"
    lines: list[str] = []
    conditional = result.evidence == "conditional"
    if result.status.value == "ok":
        head += f"; result=${result.result_latex}$; evidence={result.evidence}"
        if conditional:
            head += (
                "; the result HOLDS GIVEN: "
                + "; ".join(result.given)
                + ". Minerva did not check that "
                "this is true: say 'given', never 'proved'"
            )
        else:
            head += "; no independence was used"
        if result.verification:
            head += f"; {result.message}"
    elif result.status.value == "unchanged":
        head += "; result=none: nothing more follows without further assumptions (independence is never assumed)"
    else:
        head += f"; result=none ({result.reason or result.message})"
        lines.append(result.message)
    formulas = [
        result.input_latex or "",
        result.result_latex or "",
        *[s.after_latex for s in result.steps],
    ]
    known, quoted = _known_and_quoted(formulas[1:], conditional, result.given)
    if result.input_latex:
        known.insert(0, result.input_latex)
    return ToolOutput(
        card, "\n".join([head, *lines]), known, result.status.value, quoted_latex=quoted
    )


# ---------------------------------------------------------------------------------- statistical models


def run_statistical_model(args: StatModelArgs, ctx: ToolContext) -> ToolOutput:
    request = StatModelInput(**args.model_dump())
    result = ctx.engine.statistical_model(request)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="statistical_model",
        type="statistics",
        title="Statistical model",
        data={
            "result": result.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
        },
    )
    head = f"status={result.status.value}; action={result.action}; family={args.family}"
    lines: list[str] = []
    formulas: list[str] = []
    conditional = result.evidence == "conditional"
    if result.status.value == "ok":
        head += f"; evidence={result.evidence}"
        for name, text in result.moments.items():
            lines.append(f"- {name} = ${result.moments_latex.get(name, text)}$")
            formulas.append(result.moments_latex.get(name, ""))
        for name, latex in result.expressions_latex.items():
            lines.append(f"- {name}: ${latex}$")
            formulas.append(latex)
        for e in result.estimates:
            optimum = f"; optimum={e.optimum} ({e.optimum_evidence})" if e.optimum else ""
            lines.append(f"- estimate {e.parameter} = ${e.latex}${optimum}")
            lines.extend(f"  - valid only if ${c.latex}$ ({c.how})" for c in e.admissible_if)
            formulas.append(e.latex)
        if result.matrix_latex:
            lines.append(
                "- matrix: "
                + "; ".join(
                    "[" + ", ".join(f"${x}$" for x in row) + "]" for row in result.matrix_latex
                )
            )
            formulas.extend(x for row in result.matrix_latex for x in row)
        if result.posterior:
            lines.append(
                f"- posterior is the {result.posterior.family} family with "
                + ", ".join(f"{k} = ${v}$" for k, v in result.posterior.parameters_latex.items())
            )
            formulas.extend(result.posterior.parameters_latex.values())
        lines.extend(_lines_for_checks(result))
        lines.extend(_conditions(result))
        if result.premises:
            head += (
                "; the result HOLDS GIVEN: "
                + "; ".join(result.premises)
                + ". Minerva did not check "
                "these: say 'given', never 'proved'"
            )
        lines.append(result.message)
    else:
        head += f"; result=none (reason={result.reason or result.status.value}); {result.message}"
        if result.reason == "independence_not_declared":
            head += ". Ask the user whether the observations are independent and identically distributed"
        if result.reason == "candidate_inadmissible":
            head += ". Do not invent an estimate; explain that no maximum-likelihood estimate exists for these data"
    formulas.extend(s.after_latex for s in result.steps)
    known, quoted = _known_and_quoted(
        formulas, conditional or bool(result.premises), result.premises
    )
    return ToolOutput(
        card, "\n".join([head, *lines]), known, result.status.value, quoted_latex=quoted
    )
