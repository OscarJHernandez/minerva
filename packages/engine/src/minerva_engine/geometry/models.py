"""Typed request and result models for coordinate-based differential geometry."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ..models import SYMBOL_NAME_MAX, Status

# A request may ask for at most this many coordinates; the benchmark raises the cap to measure growth.
MAX_DIMENSION = 4
HARD_MAX_DIMENSION = 8  # structural ceiling of the model itself
MAX_TENSOR_RANK = 3
MAX_CELL_CHARS = 300
MAX_FUNCTIONS = 6

Cell = Annotated[str, Field(min_length=1, max_length=MAX_CELL_CHARS)]
Variance = Literal["up", "down"]

QuantityName = Literal[
    "inverse_metric",
    "determinant",
    "christoffel",
    "riemann",
    "ricci",
    "ricci_scalar",
    "einstein",
    "kretschmann",
    "ricci_squared",
    "geodesics",
    "covariant_derivative",
    "raise_index",
    "lower_index",
]
QUANTITIES: tuple[str, ...] = QuantityName.__args__  # type: ignore[attr-defined]
TENSOR_QUANTITIES = frozenset({"covariant_derivative", "raise_index", "lower_index"})


class FunctionDecl(BaseModel):
    """A real-valued function of some of the coordinates, e.g. `f(r)` or `a(t)`."""

    name: str = Field(min_length=1, max_length=SYMBOL_NAME_MAX)
    args: list[str] = Field(min_length=1, max_length=4)


class TensorField(BaseModel):
    """A tensor field given by its components in the coordinates, row-major.

    `variance` lists the type of each index ("up" for V^m, "down" for w_m); `components` has
    n**rank entries, the first index varying slowest.
    """

    variance: list[Variance] = Field(min_length=1, max_length=MAX_TENSOR_RANK)
    components: list[Cell] = Field(min_length=1, max_length=HARD_MAX_DIMENSION**MAX_TENSOR_RANK)


class GeometryInput(BaseModel):
    coordinates: list[str] = Field(min_length=1, max_length=HARD_MAX_DIMENSION)
    metric: list[list[Cell]] = Field(min_length=1, max_length=HARD_MAX_DIMENSION)
    format: Literal["latex", "text"] = "latex"
    assumptions: dict[str, list[str]] = Field(default_factory=dict)
    functions: list[FunctionDecl] = Field(default_factory=list, max_length=6)
    quantities: list[QuantityName] = Field(min_length=1, max_length=len(QUANTITIES))
    tensor: TensorField | None = None
    # which index raise_index / lower_index acts on (0-based)
    tensor_index: int = Field(default=0, ge=0, le=MAX_TENSOR_RANK - 1)


class Component(BaseModel):
    indices: list[int]
    label_latex: str
    latex: str
    text: str


class Check(BaseModel):
    """One verification leg. `passed` is None when the leg did not run (and `note` says why)."""

    name: str
    kind: Literal["numeric", "symbolic"]
    passed: bool | None = None
    points_checked: int = 0
    points_agreed: int = 0
    note: str = ""


class GeometryVerification(BaseModel):
    method: str = ""
    # The independent numeric legs decide. Any failed numeric leg means the quantity is not shown.
    numeric_passed: bool = False
    # True only if every symbolic check that applies ran and found zero; None if none ran.
    symbolic_zero: bool | None = None
    checks: list[Check] = Field(default_factory=list)
    note: str = ""


class QuantityResult(BaseModel):
    kind: QuantityName
    status: Status
    title: str
    variance: list[Variance] = Field(default_factory=list)
    components: list[Component] = Field(default_factory=list)  # nonzero components only
    all_zero: bool = False
    scalar_latex: str | None = None
    scalar_text: str | None = None
    symmetry_note: str = ""
    verification: GeometryVerification | None = None
    notes: list[str] = Field(default_factory=list)
    message: str = ""


class StageDiagnostic(BaseModel):
    """Timing and size of one stage. Never contains an expression."""

    stage: str
    duration_ms: int = 0
    ops: int | None = None  # SymPy operation count of what the stage produced
    components: int | None = None  # number of nonzero components the stage produced
    status: Literal["ok", "failed", "skipped"] = "ok"
    failure: str | None = (
        None  # resource_limit | not_verified | unsupported | parse_error | internal
    )


class GeometryResult(BaseModel):
    status: Status
    dimension: int = 0
    coordinates_latex: list[str] = Field(default_factory=list)
    metric_latex: str | None = None
    signature: str | None = None
    signature_note: str = ""
    conventions: list[str] = Field(default_factory=list)
    assumptions_used: list[str] = Field(default_factory=list)
    restrictions_latex: list[str] = Field(default_factory=list)
    restrictions_complete: bool = True
    quantities: list[QuantityResult] = Field(default_factory=list)
    diagnostics: list[StageDiagnostic] = Field(default_factory=list)
    peak_memory_mb: float | None = None
    message: str = ""
    duration_ms: int = 0
