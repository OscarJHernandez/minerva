"""Models of the ``statmech`` operation (statistical mechanics of explicit systems).

One request carries the whole problem (system, ensemble, parameters, assumptions, conventions); nothing
is stored on the server, so every call can be replayed. Results say what kind of statement they are
(``result_kind``) and which independent checks agreed. Diagnostics hold stage names, counts and enum
reasons only, never expressions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..distributions.models import Condition
from ..models import Status

Ensemble = Literal["canonical"]
Action = Literal["analyze", "partition", "derive", "expectation"]
TemperatureVariable = Literal["beta", "T"]
BaseQuantity = Literal["Z", "lnZ", "F", "U", "S", "Var_E"]
Statistic = Literal["mean", "moment", "central_moment", "cumulant", "variance", "covariance"]
ResultKind = Literal["exact", "representation", "numeric", "asymptotic"]

Reason = Literal[
    "parse_error",
    "invalid_degeneracy",
    "assumptions_insufficient",
    "ensemble_mismatch",
    "coupling_not_linear",
    "unknown_observable",
    "verification_failed",
    "unsupported_system",
    "resource_limit",
]
Stage = Literal["parse", "build", "symbolic", "verify", "expectation", "analyze"]

BASE_QUANTITIES: tuple[str, ...] = ("Z", "lnZ", "F", "U", "S", "Var_E")

MAX_LEVELS = 64
MAX_ORDER = 4


class LevelSpec(BaseModel):
    """One energy level: its energy, its degeneracy and the value of each named observable on it."""

    energy: str = Field(max_length=300)
    degeneracy: str = Field(default="1", max_length=100)
    observables: dict[str, str] = Field(default_factory=dict, max_length=8)


class LevelsSystem(BaseModel):
    kind: Literal["levels"] = "levels"
    levels: list[LevelSpec] = Field(min_length=1, max_length=MAX_LEVELS)


System = LevelsSystem


class Partial(BaseModel):
    """A partial derivative of a base quantity with the other state variables held fixed.

    The held-fixed set is part of the request and is checked: it must be exactly the other state
    variables (the temperature variable and the declared parameters). ``scale`` multiplies the
    derivative (for example ``-1`` for a pressure ``-dF/dV`` at fixed temperature).
    """

    of: BaseQuantity
    variable: str = Field(max_length=32)
    held_fixed: list[str] = Field(default_factory=list, max_length=8)
    order: int = Field(default=1, ge=1, le=2)
    scale: str = Field(default="1", max_length=64)
    label: str = Field(default="", max_length=64)


class Coupling(BaseModel):
    """Declares that the energies depend linearly on ``parameter`` as ``E_i(h) = E_i(0) - h A_i``."""

    parameter: str = Field(max_length=32)
    observable: str = Field(max_length=32)


class StatMechInput(BaseModel):
    ensemble: Ensemble = "canonical"
    system: System
    format: Literal["text", "latex"] = "text"
    temperature: TemperatureVariable = "beta"
    # Names of the temperature-related symbols; they are positive by the definition of a temperature.
    beta: str = Field(default="beta", max_length=16)
    T: str = Field(default="T", max_length=16)
    k_B: str = Field(default="k_B", max_length=16)
    parameters: list[str] = Field(default_factory=list, max_length=6)
    assumptions: dict[str, list[str]] = Field(default_factory=dict)
    action: Action = "partition"
    # derive
    quantities: list[BaseQuantity] = Field(default_factory=list, max_length=6)
    partials: list[Partial] = Field(default_factory=list, max_length=6)
    # expectation
    observable: str = Field(default="E", max_length=32)
    statistic: Statistic = "mean"
    order: int = Field(default=1, ge=1, le=MAX_ORDER)
    other_observable: str | None = Field(default=None, max_length=32)
    coupling: Coupling | None = None


class StatStep(BaseModel):
    name: str  # fixed vocabulary: levels, partition_function, probabilities, ...
    title: str
    latex: str
    text: str = ""
    result_kind: ResultKind = "exact"
    conditions: list[Condition] = Field(default_factory=list)


class StatCheck(BaseModel):
    """One independent check of a result. ``passed`` is None when it could not be decided."""

    name: str  # enumeration, finite_difference, normalisation, nonnegative_variance, symbolic_zero, ...
    subject: str  # the quantity checked
    passed: bool | None
    points: int = 0
    agreed: int = 0
    detail: str = ""


class StatQuantity(BaseModel):
    name: str
    title: str
    latex: str
    text: str
    result_kind: ResultKind = "exact"
    held_fixed: list[str] = Field(default_factory=list)
    variable: str | None = None
    route: str = ""  # how it was produced, in words
    checks: list[StatCheck] = Field(default_factory=list)


class MethodSuggestion(BaseModel):
    id: str
    title: str
    applies: bool
    implemented: bool
    why: str


class StatMechEvent(BaseModel):
    stage: Stage
    status: Literal["ok", "failed"] = "ok"
    duration_ms: float = 0.0
    reason: str | None = None


class StatMechDiagnostics(BaseModel):
    levels: int = 0
    parameters: int = 0
    points_checked: int = 0
    points_agreed: int = 0
    reason: Reason | None = None
    events: list[StatMechEvent] = Field(default_factory=list)


class StatMechVerification(BaseModel):
    method: str = (
        "mpmath enumeration of the microstates with log-sum-exp, and numerical differentiation of that "
        "enumeration, at sampled assumption-respecting points"
    )
    points_checked: int = 0
    points_agreed: int = 0
    symbolic_zero: bool | None = None  # extra evidence, never decides
    passed: bool = False
    note: str = ""


class StatMechResult(BaseModel):
    status: Status
    action: Action
    ensemble: Ensemble = "canonical"
    reason: Reason | None = None
    state_variables: list[str] = Field(default_factory=list)
    steps: list[StatStep] = Field(default_factory=list)
    quantities: list[StatQuantity] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    methods: list[MethodSuggestion] = Field(default_factory=list)
    not_supported: list[str] = Field(default_factory=list)
    verification: StatMechVerification = Field(default_factory=StatMechVerification)
    diagnostics: StatMechDiagnostics = Field(default_factory=StatMechDiagnostics)
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: float = 0.0
