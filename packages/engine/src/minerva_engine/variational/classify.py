"""What may be said about a stationary path from second-variation evidence.

The result type has no member that says a path *is* a minimum. A negative direction (certified by
quadrature, by a second difference of the action and by the measured change of the action) proves
the path is not a weak local minimum; a violated Legendre condition does the same; the absence of
a negative direction among the perturbations that were tried proves nothing, and the sufficient
conditions (strengthened Legendre, no conjugate points) are listed as not established.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from .models import Fact

Stationarity = Literal["consistent", "violated", "not_tested"]


@dataclass(frozen=True)
class SecondEvidence:
    real_valued: bool
    stationary: Stationarity
    negative_certified: bool  # an admissible direction with δ²S < 0, confirmed three ways
    positive_certified: bool
    legendre_negative: bool  # W has a negative eigenvalue at a sampled point of the path
    legendre_positive: bool


NOT_ESTABLISHED = (
    "positivity of the second variation on all admissible perturbations",
    "the strengthened Legendre condition on the whole interval",
    "absence of conjugate points (Jacobi condition)",
)


def classify(evidence: SecondEvidence) -> tuple[list[Fact], list[str]]:
    """(facts, what is not established). Nothing is concluded for a path that is not stationary."""
    if not evidence.real_valued or evidence.stationary != "consistent":
        return [], list(NOT_ESTABLISHED)
    facts: list[Fact] = []
    if evidence.negative_certified:
        facts.append("not_a_weak_minimum")
    if evidence.positive_certified:
        facts.append("not_a_weak_maximum")
    if evidence.legendre_negative:
        facts.append("legendre_violated_for_minimum")
    if evidence.legendre_positive:
        facts.append("legendre_violated_for_maximum")
    if not evidence.negative_certified and not evidence.legendre_negative:
        facts.append("no_negative_direction_found")
    if not evidence.positive_certified and not evidence.legendre_positive:
        facts.append("no_positive_direction_found")
    return facts, list(NOT_ESTABLISHED)


ALL_FACTS: tuple[str, ...] = get_args(Fact)
