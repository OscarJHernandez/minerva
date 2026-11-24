"""Run corpus rows through the verifier and classify the outcome. Shared by the tests and the report."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Literal

import minerva_engine.verify as verify_module
from minerva_engine.models import Verification
from minerva_engine.testing import shared_timeout_seconds
from minerva_engine.verify import verify_equal

from .corpus import CASES, Case

Outcome = Literal["correct", "false_accept", "false_reject"]


@contextlib.contextmanager
def patient_probes() -> Iterator[None]:
    """The probes' wall-clock safety stop (`PROBE_SECONDS`, 5 s) raised to the shared test budget.

    The probes' deterministic cap is their number of points; the seconds only protect a server from an expression
    that is slow to evaluate. A search the clock ends is incomplete and does not pass (V-017), so on an overloaded
    runner (a shard whose setup alone took 336 s) a true identity could be rejected for the load, not for the
    mathematics. The corpus measures the verifier's verdicts, so its probes are given the time to finish; a correct
    answer that arrives late is not a failure (docs/verification-defects.md V-017).
    """
    saved = verify_module.PROBE_SECONDS
    verify_module.PROBE_SECONDS = max(saved, shared_timeout_seconds())
    try:
        yield
    finally:
        verify_module.PROBE_SECONDS = saved


def verify_case(case: Case, *, probe: bool = True) -> Verification:
    """Sampling and probes only (`symbolic=False`): the symbolic zero test is separate evidence.

    `probe=False` is the verifier as it was before targeted probes existed (the "before" of a comparison).
    """
    with patient_probes():
        return verify_equal(case.a, case.b, symbolic=False, probe=probe)


def outcome_of(case: Case, verification: Verification) -> Outcome:
    accepted = verification.passed
    if case.expect == "accept":
        return "correct" if accepted else "false_reject"
    return "false_accept" if accepted else "correct"


def run_all(
    cases: list[Case] | None = None, *, probe: bool = True
) -> list[tuple[Case, Outcome, Verification]]:
    return [
        (c, outcome_of(c, v := verify_case(c, probe=probe)), v)
        for c in (cases if cases is not None else CASES)
    ]
