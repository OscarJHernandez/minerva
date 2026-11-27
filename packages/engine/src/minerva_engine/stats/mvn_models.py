"""Request and result of the bivariate Normal operation (`stats/mvn.py`, issue #229)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..distributions.models import Condition
from ..models import Status

MvnAction = Literal["density", "moments", "marginal", "conditional", "affine"]
MvnReason = Literal[
    "dimension_unsupported",
    "not_symmetric",
    "not_positive_definite",
    "singular_covariance",
    "map_singular",
    "variables_missing",
    "index_invalid",
    "parse_error",
    "verification_failed",
    "resource_limit",
]


class MvnInput(BaseModel):
    """A Normal distribution of two components, given by its mean and covariance, and one operation.

    Nothing has a default: the mean, the covariance, the names of the components, the component kept,
    the value conditioned on and the affine map all come from the user.
    """

    mean: list[str] = Field(min_length=1, max_length=8)
    covariance: list[list[str]] = Field(min_length=1, max_length=8)
    # names of the two components (x, y): needed for a density, a marginal and a conditional
    variables: list[str] | None = Field(default=None, max_length=8)
    action: MvnAction
    keep: str | None = Field(default=None, max_length=32)  # marginal: the component kept
    # conditional: {name of the component fixed: its value}
    given: dict[str, str] | None = Field(default=None, max_length=4)
    matrix: list[list[str]] | None = Field(default=None, max_length=4)  # affine: A (1 or 2 rows)
    offset: list[str] | None = Field(default=None, max_length=4)  # affine: b
    assumptions: dict[str, list[str]] = Field(default_factory=dict, max_length=32)


class MvnCheck(BaseModel):
    """One independent numerical check a result had to pass."""

    kind: Literal["normalisation", "moments", "marginal", "conditional", "affine"]
    method: str
    points_checked: int = 0
    points_agreed: int = 0
    passed: bool = False
    note: str = ""


class MvnResult(BaseModel):
    status: Status
    action: MvnAction
    reason: MvnReason | None = None
    message: str = ""
    # the distribution the result describes (the input for density and moments, the new one otherwise)
    variables: list[str] = Field(default_factory=list)
    mean_text: list[str] = Field(default_factory=list)
    mean_latex: list[str] = Field(default_factory=list)
    covariance_text: list[list[str]] = Field(default_factory=list)
    covariance_latex: list[list[str]] = Field(default_factory=list)
    density_text: str = ""
    density_latex: str = ""
    correlation_text: str = ""
    correlation_latex: str = ""
    # the model as read, so a follow-up can re-send exactly it
    input_mean_text: list[str] = Field(default_factory=list)
    input_covariance_text: list[list[str]] = Field(default_factory=list)
    checks: list[MvnCheck] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    premises: list[str] = Field(default_factory=list)
    evidence: Literal["identity", "conditional"] | None = None
    duration_ms: float = 0.0
