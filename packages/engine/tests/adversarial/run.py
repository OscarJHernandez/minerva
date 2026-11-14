"""Run corpus rows through the verifier and classify the outcome. Shared by the tests and the report."""

from __future__ import annotations

from typing import Literal

from minerva_engine.models import Verification
from minerva_engine.verify import verify_equal

from .corpus import CASES, Case

Outcome = Literal["correct", "false_accept", "false_reject"]


def verify_case(case: Case) -> Verification:
    """Sampling and probes only (`symbolic=False`): the symbolic zero test is separate evidence."""
    return verify_equal(case.a, case.b, symbolic=False)


def outcome_of(case: Case, verification: Verification) -> Outcome:
    accepted = verification.passed
    if case.expect == "accept":
        return "correct" if accepted else "false_reject"
    return "false_accept" if accepted else "correct"


def run_all(cases: list[Case] | None = None) -> list[tuple[Case, Outcome, Verification]]:
    return [
        (c, outcome_of(c, v := verify_case(c)), v) for c in (cases if cases is not None else CASES)
    ]
