"""Models of the ``expectation`` operation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..distributions.models import Condition
from ..indexed.models import SumStep
from ..models import ExpressionInput, Status

ExpectationAction = Literal["expand", "to_moments", "check"]
ExpectationReason = Literal[
    "parse_error",
    "no_random_variables",
    "verification_failed",
    "missing_candidate",
    "unsupported_form",
    "resource_limit",
]


class ExpectationInput(BaseModel):
    expression: ExpressionInput
    random: list[str] = Field(min_length=1, max_length=8)
    # Groups of mutually independent random variables, by name: [["X", "Y"]] says X and Y are
    # independent; [["X"]] for an indexed X says its components are mutually independent.
    # Independence is never assumed: without a group, variables may be dependent.
    independent: list[list[str]] = Field(default_factory=list, max_length=8)
    action: ExpectationAction = "expand"
    candidate: dict[str, str] | None = None


class PmfVerification(BaseModel):
    """Evidence from explicit finite joint distributions (exact rational arithmetic).

    Both sides are evaluated by weighted sums over the outcomes of a joint distribution that is
    dependent unless independence was declared; the identity has to hold for every distribution tried.
    """

    method: str = (
        "explicit finite joint distributions with exact rational weights, dependent unless "
        "independence was declared, at several sizes"
    )
    sizes: list[dict[str, int]] = Field(default_factory=list)
    models_checked: int = 0
    models_agreed: int = 0
    exact_zero_models: int = 0
    passed: bool = False
    note: str = ""

    def describe(self) -> str:
        sizes = (
            "; ".join(", ".join(f"{k} = {v}" for k, v in sorted(s.items())) for s in self.sizes)
            if self.sizes and self.sizes[0]
            else ""
        )
        where = f" for {sizes}" if sizes else ""
        return (
            f"agrees on {self.models_agreed} of {self.models_checked} explicit joint "
            f"distributions{where}; evidence, not a proof for every distribution"
        )


class ExpectationResult(BaseModel):
    status: Status
    action: ExpectationAction
    reason: ExpectationReason | None = None
    input_latex: str | None = None
    result_latex: str | None = None
    result_text: str | None = None
    # "conditional": the result holds given the independence that was declared
    evidence: Literal["identity", "conditional"] | None = None
    given: list[str] = Field(default_factory=list)
    steps: list[SumStep] = Field(default_factory=list)
    verification: PmfVerification | None = None
    message: str = ""
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------------- statistical models

StatAction = Literal["moments", "log_likelihood", "score", "mle", "fisher_information", "posterior"]
StatReason = Literal[
    "unsupported_family",
    "parameter_invalid",
    "parameter_missing",
    "assumptions_insufficient",
    "independence_not_declared",
    "regularity_not_satisfied",
    "convergence_not_established",
    "no_closed_form",
    "system_not_triangular",
    "candidate_inadmissible",
    "kernel_not_recognised",
    "verification_failed",
    "parse_error",
    "unsupported_form",
    "resource_limit",
]
Optimum = Literal["global", "local", "stationary_only"]


class PriorSpec(BaseModel):
    family: str = Field(max_length=32)
    parametrization: str | None = Field(default=None, max_length=16)
    parameters: dict[str, str] = Field(default_factory=dict, max_length=8)


class StatModelInput(BaseModel):
    """A probability model, and what to derive from it."""

    family: str = Field(max_length=32)
    parametrization: str | None = Field(default=None, max_length=16)
    # known parameters, as expressions; the others are the ones listed in `estimate`
    parameters: dict[str, str] = Field(default_factory=dict, max_length=8)
    observation: str | None = Field(default=None, max_length=32)  # name of the data, e.g. x
    sample_size: str | None = Field(default=None, max_length=32)  # e.g. n
    # Independence is never assumed: the user states it, or a likelihood of several observations
    # is refused (`independence_not_declared`).
    iid: bool | None = None
    estimate: list[str] = Field(default_factory=list, max_length=4)
    action: StatAction = "moments"
    prior: PriorSpec | None = None
    data: list[str] | None = Field(default=None, max_length=64)  # concrete observations
    # check an answer instead of deriving one, keyed like the benchmark outputs
    candidate: dict[str, str] | None = Field(default=None, max_length=16)
    assumptions: dict[str, list[str]] = Field(default_factory=dict, max_length=32)


class StatCheck(BaseModel):
    """One independent check that a result had to pass."""

    kind: Literal[
        "normalisation", "moment_quadrature", "score", "second_order", "information", "posterior"
    ]
    method: str
    points_checked: int = 0
    points_agreed: int = 0
    passed: bool = False
    note: str = ""


class StatStep(BaseModel):
    title: str
    before_latex: str = ""
    after_latex: str
    after_text: str = ""
    evidence: Literal["definition", "identity", "conditional"] = "identity"


class Estimate(BaseModel):
    parameter: str
    latex: str
    text: str
    optimum: Optimum | None = None
    optimum_evidence: Literal["proved", "measured"] | None = None
    admissible_if: list[Condition] = Field(default_factory=list)


class PosteriorOut(BaseModel):
    family: str
    parameters: dict[str, str] = Field(default_factory=dict)
    parameters_latex: dict[str, str] = Field(default_factory=dict)
    dropped_factors_latex: list[str] = Field(default_factory=list)


class StatModelResult(BaseModel):
    status: Status
    action: StatAction
    reason: StatReason | None = None
    steps: list[StatStep] = Field(default_factory=list)
    moments: dict[str, str] = Field(default_factory=dict)  # mean, variance, second (plain text)
    moments_latex: dict[str, str] = Field(default_factory=dict)
    # log-likelihood and score: named expressions in plain text and LaTeX ("loglik", "score:mu")
    expressions: dict[str, str] = Field(default_factory=dict)
    expressions_latex: dict[str, str] = Field(default_factory=dict)
    estimates: list[Estimate] = Field(default_factory=list)
    matrix_text: list[list[str]] | None = None
    matrix_latex: list[list[str]] | None = None
    posterior: PosteriorOut | None = None
    checks: list[StatCheck] = Field(default_factory=list)
    premises: list[str] = Field(default_factory=list)  # e.g. "the observations are independent"
    conditions: list[Condition] = Field(default_factory=list)
    evidence: Literal["identity", "conditional"] | None = None
    message: str = ""
    duration_ms: float = 0.0
