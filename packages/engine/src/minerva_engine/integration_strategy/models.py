"""Result types of the integration strategy engine.

All text here is engine-written from fixed vocabulary plus formulas the engine printed; nothing is
free text from a model. `Technique` and `Reason` are enumerations so that logs and tests can rely
on them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Technique = Literal[
    "direct",
    "substitution",
    "parts",
    "trigonometric",
    "partial_fractions",
    "parity",
    "reflection",
    "frullani",
    "schwinger",
    "parameter_differentiation",
    "numerical",
]

# What a caller may ask for: `auto` explains a result with the technique that finds it; a name
# tries only that technique.
StrategyName = Literal[
    "auto",
    "substitution",
    "parts",
    "trigonometric",
    "partial_fractions",
    "parity",
    "reflection",
    "frullani",
    "schwinger",
    "parameter_differentiation",
    "numerical",
]

TECHNIQUES: tuple[Technique, ...] = (
    "substitution",
    "parts",
    "trigonometric",
    "partial_fractions",
    "parity",
    "reflection",
    "frullani",
    "schwinger",
    "parameter_differentiation",
    "numerical",
)

# Why a technique was suggested: a fixed code, never prose from the input.
Why = Literal[
    "composite_with_inner_derivative",
    "polynomial_times_transcendental",
    "trigonometric_powers_or_products",
    "rational_with_reducible_denominator",
    "odd_on_symmetric_interval",
    "even_on_symmetric_interval",
    "reflection_simplifies",
    "difference_at_two_scalings_over_x",
    "negative_power_of_a_positive_base",
    "free_parameter",
    "no_closed_form_for_a_number",
]

AttemptOutcome = Literal[
    "verified",
    "not_applicable",
    "no_progress",
    "failed_verification",
    "blocked",
    "deadline",
]

Verification = Literal["proved", "numerically_checked", "numeric_estimate", "none"]


class Suggestion(BaseModel):
    technique: Technique
    why: Why


class Step(BaseModel):
    """One shown step of the selected technique: what was done and the formula it produced."""

    what: str  # fixed vocabulary, e.g. "substitution", "transformed integral", "inner integral"
    latex: str


class Attempt(BaseModel):
    technique: Technique
    outcome: AttemptOutcome
    detail: str = ""  # a fixed code such as "convergence_not_established"; never an expression


class StrategyReport(BaseModel):
    """Which technique produced the result (or was suggested), and the evidence for it."""

    technique: Technique | None = None
    why: Why | None = None
    steps: list[Step] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)
    attempts: list[Attempt] = Field(default_factory=list)
    verification: Verification = "none"
    candidates_rejected: int = 0
    conditions_latex: list[str] = Field(default_factory=list)  # what had to hold, as established
    note: str = ""
