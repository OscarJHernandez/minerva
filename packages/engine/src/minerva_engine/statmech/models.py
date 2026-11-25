"""Models of the ``statmech`` operation (statistical mechanics of explicit systems).

One request carries the whole problem (system, ensemble, parameters, assumptions, conventions); nothing
is stored on the server, so every call can be replayed. Results say what kind of statement they are
(``result_kind``) and which independent checks agreed. Diagnostics hold stage names, counts and enum
reasons only, never expressions.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ..distributions.models import Condition
from ..models import Status

Ensemble = Literal["canonical", "grand_canonical", "microcanonical"]
Action = Literal["analyze", "partition", "derive", "expectation", "evaluate", "approximate"]
Method = Literal["high_temperature", "low_temperature", "stirling", "thermodynamic_limit"]
TemperatureVariable = Literal["beta", "T"]
BaseQuantity = Literal["Z", "lnZ", "F", "U", "S", "Var_E", "Xi", "lnXi", "Omega", "N", "Var_N"]
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
    "divergent",
    "convergence_not_provable",
    "not_separable",
    "convention_unconfirmed",
    "not_hermitian",
    "operators_do_not_commute",
    "energy_not_attainable",
    "singular_expansion_point",
    "ordering_not_decided",
    "nothing_to_approximate",
]
Stage = Literal["parse", "build", "symbolic", "verify", "expectation", "analyze"]

BASE_QUANTITIES: tuple[str, ...] = ("Z", "lnZ", "F", "U", "S", "Var_E")
GRAND_QUANTITIES: tuple[str, ...] = ("Xi", "lnXi", "Omega", "N", "U", "S", "Var_N")

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


class FamilySystem(BaseModel):
    """Levels E(n) with degeneracy g(n) for an integer index, finite or infinite."""

    kind: Literal["level_family"] = "level_family"
    index: str = Field(default="n", max_length=8)
    energy: str = Field(max_length=300)
    degeneracy: str = Field(default="1", max_length=100)
    lower: str = Field(default="0", max_length=32)
    upper: str = Field(default="oo", max_length=32)


class DosSystem(BaseModel):
    """A continuous spectrum with density of states rho(E) on [lower, upper]."""

    kind: Literal["density_of_states"] = "density_of_states"
    variable: str = Field(default="En", max_length=8)
    density: str = Field(max_length=300)
    lower: str = Field(max_length=32)
    upper: str = Field(default="oo", max_length=32)


class PhaseVariable(BaseModel):
    name: str = Field(max_length=16)
    domain_lower: str = Field(default="-oo", max_length=64)
    domain_upper: str = Field(default="oo", max_length=64)


class ClassicalSystem(BaseModel):
    """A classical Hamiltonian H(p, q) of one copy of the system, integrated over phase space.

    ``phase_space_cell`` (the divisor of the phase-space measure, e.g. ``h^3`` for three degrees of
    freedom, or ``1``) is a convention only the user chooses; without it the result is
    ``convention_unconfirmed``.
    """

    kind: Literal["classical"] = "classical"
    hamiltonian: str = Field(max_length=400)
    variables: list[PhaseVariable] = Field(min_length=1, max_length=8)
    phase_space_cell: str | None = Field(default=None, max_length=64)


class QuantumSystem(BaseModel):
    """A finite-dimensional quantum system: an explicit Hermitian matrix H and optional operators.

    Operators are matrices of the same size; they are never assumed to commute with H or each other.
    """

    kind: Literal["quantum"] = "quantum"
    hamiltonian: list[list[str]] = Field(min_length=1, max_length=4)
    observables: dict[str, list[list[str]]] = Field(default_factory=dict, max_length=6)


class OccupationLevel(BaseModel):
    energy: str = Field(max_length=300)
    degeneracy: str = Field(default="1", max_length=100)


class OccupationSystem(BaseModel):
    """Non-interacting particles in single-particle levels, in the grand canonical ensemble.

    ``bose`` and ``fermi`` count each of the g single-particle states of a level separately;
    ``boltzmann`` is the classical-counting limit (Poisson occupation, no exclusion or bunching).
    """

    kind: Literal["occupation"] = "occupation"
    statistics: Literal["bose", "fermi", "boltzmann"]
    levels: list[OccupationLevel] = Field(min_length=1, max_length=MAX_LEVELS)
    mu: str = Field(default="mu", max_length=16)


class ChainSystem(BaseModel):
    """A periodic one-dimensional chain of N sites with nearest-neighbour coupling (transfer matrix).

    The energy of a configuration is the sum over bonds of ``bond(s, t)`` (s and t are the values of the two
    neighbouring sites, the bond must be symmetric) plus the sum over sites of ``site(s)``.
    """

    kind: Literal["chain"] = "chain"
    states: list[str] = Field(
        min_length=2, max_length=3, description='the values a site can take, e.g. ["1", "-1"]'
    )
    bond: str = Field(
        max_length=200,
        description="energy of two neighbouring sites in text syntax, in the names s and t (their "
        "values), symmetric in s and t; * for products, no subscripts, e.g. -J*s*t",
    )
    site: str = Field(
        default="0",
        max_length=200,
        description="energy of one site in text syntax, in the name s, e.g. -h*s",
    )
    length: str = Field(max_length=16)


System = Annotated[
    LevelsSystem
    | FamilySystem
    | DosSystem
    | ClassicalSystem
    | QuantumSystem
    | OccupationSystem
    | ChainSystem,
    Field(discriminator="kind"),
]


class Copies(BaseModel):
    """``count`` independent, non-interacting copies of the system.

    ``identical`` is the user's choice: true divides by count! (the Boltzmann counting of the
    classical limit, not exact quantum statistics); false treats the copies as distinguishable.
    """

    count: str = Field(max_length=32)
    identical: bool | None = None


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


MAX_SWEEP_POINTS = 60


class Sweep(BaseModel):
    """Evaluate over a range of one symbol (all numbers are the user's)."""

    variable: str = Field(max_length=32)
    start: str = Field(max_length=32)
    stop: str = Field(max_length=32)
    points: int = Field(default=11, ge=2, le=MAX_SWEEP_POINTS)
    scale: Literal["linear", "log"] = "linear"


class StatMechInput(BaseModel):
    ensemble: Ensemble = "canonical"
    system: System
    copies: Copies | None = None
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
    # evaluate: the numbers are chosen by the user, never by the model
    values: dict[str, str] = Field(default_factory=dict, max_length=16)
    sweep: Sweep | None = None
    # microcanonical: the fixed total energy of N distinguishable or counted copies of the levels
    total_energy: str | None = Field(default=None, max_length=32)
    # approximate: which approximation to apply (the order of an expansion is `order`)
    method: Method | None = None


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
    # The order n of a cumulant kappa_n of the observable named after `cumulant_`; None for every other quantity.
    # A structured field: nothing reads the order out of the name or of a check's subject text.
    order: int | None = None
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


class ErrorRow(BaseModel):
    """One measured comparison of an approximation with the independent exact value."""

    label: str
    exact: str
    approximate: str
    error: str
    note: str = ""


class Approximation(BaseModel):
    """What an approximation assumes, what it expands in, how accurate it is expected to be and where
    it holds, with measured errors. The table is a measurement, never a bound."""

    method: Method
    assumptions: list[str]
    parameter: str
    order: str
    validity: str
    error_table: list[ErrorRow] = Field(default_factory=list)


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


class StatTable(BaseModel):
    """Numerical values over a sweep: one column per quantity, numbers as decimal strings."""

    columns: list[str]
    rows: list[list[str]]
    variable: str = ""


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
    questions: list[str] = Field(default_factory=list)
    table: StatTable | None = None
    approximation: Approximation | None = None
    verification: StatMechVerification = Field(default_factory=StatMechVerification)
    diagnostics: StatMechDiagnostics = Field(default_factory=StatMechDiagnostics)
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: float = 0.0
