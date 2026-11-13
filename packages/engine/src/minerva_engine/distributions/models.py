"""Request and result models of the distribution operation.

``diagnostics`` is the engine-side trace: the sandboxed child cannot log, so it returns the events
as data and the API turns them into allow-listed log lines. Events and reasons are enumerations
and never carry expression text.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..models import ExpressionInput, Status

Action = Literal["simplify", "integrate", "differentiate"]
Endpoint = Literal["half", "include", "exclude"]

Reason = Literal[
    "unsupported_distribution",
    "singular_root",
    "assumptions_insufficient",
    "verification_failed",
    "integration_domain_ambiguous",
    "resource_limit",
    "parse_error",
]

Stage = Literal[
    "parsed",
    "identities_identified",
    "transformation_selected",
    "assumptions_evaluated",
    "verification_attempted",
    "accepted",
    "rejected",
    "completed",
]

# Handbook entries (data/entries/distributions.yaml) the operation cites by id.
ENTRY_SCALING = "distributions.delta_scaling"
ENTRY_COMPOSITE = "distributions.delta_composite_simple_roots"
ENTRY_TIMES_FUNCTION = "distributions.delta_times_function"
ENTRY_SIFTING = "distributions.delta_sifting"
ENTRY_HEAVISIDE_DERIVATIVE = "distributions.heaviside_derivative"
CITED_ENTRIES = (
    ENTRY_SCALING,
    ENTRY_COMPOSITE,
    ENTRY_TIMES_FUNCTION,
    ENTRY_SIFTING,
    ENTRY_HEAVISIDE_DERIVATIVE,
)


class DistributionInput(BaseModel):
    expression: ExpressionInput
    action: Action
    variable: str = Field(min_length=1, max_length=32)
    # `integrate` only: real numbers or infinity (write oo or \infty).
    lower: str | None = Field(default=None, max_length=64)
    upper: str | None = Field(default=None, max_length=64)
    # How a delta sitting exactly on a limit counts. Never chosen by Minerva: without it such an
    # integral is reported as ambiguous.
    endpoint: Endpoint | None = None


class Condition(BaseModel):
    """One validity condition of an identity and how it was established."""

    latex: str
    established: bool
    how: str


class DistributionStep(BaseModel):
    entry_id: str  # handbook entry of the identity applied
    title: str
    before_latex: str
    after_latex: str
    conditions: list[Condition] = Field(default_factory=list)


class Blocked(BaseModel):
    entry_id: str
    reason: Reason
    detail: str


class DistributionEvent(BaseModel):
    stage: Stage
    status: Literal["ok", "failed"] = "ok"
    duration_ms: float = 0.0
    reason: Reason | None = None


class Diagnostics(BaseModel):
    reason: Reason | None = None
    events: list[DistributionEvent] = Field(default_factory=list)


class DistributionVerification(BaseModel):
    """Test-function evidence. Pointwise sampling is never used for a distribution."""

    method: str = ""
    test_functions: list[str] = Field(default_factory=list)
    pairings_checked: int = 0
    pairings_agreed: int = 0
    parameter_samples: int = 0
    widths: list[float] = Field(default_factory=list)  # regularisation widths used
    convergence_observed: bool = False
    # The exact conditions (assumptions, simple roots, positions) are a separate leg.
    conditions_established: bool = False
    passed: bool = False
    note: str = ""


class DistributionResult(BaseModel):
    status: Status
    action: Action | None = None
    reason: Reason | None = None
    input_latex: str | None = None
    variable: str | None = None
    lower_latex: str | None = None
    upper_latex: str | None = None
    endpoint: Endpoint | None = None
    result_latex: str | None = None
    result_text: str | None = None
    # "distributional": equal as distributions (acts the same on every test function);
    # "value": an ordinary number or expression (the value of an integral).
    equality: Literal["distributional", "value"] | None = None
    steps: list[DistributionStep] = Field(default_factory=list)
    blocked: list[Blocked] = Field(default_factory=list)
    verification: DistributionVerification | None = None
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: float = 0.0
