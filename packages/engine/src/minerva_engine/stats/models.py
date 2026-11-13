"""Models of the ``expectation`` operation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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
