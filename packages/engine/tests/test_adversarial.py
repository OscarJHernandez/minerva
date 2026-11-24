"""False accepts and false rejects of `verify_equal` on the adversarial corpus (adversarial/corpus.py)."""

import collections

import pytest
import sympy as sp
from adversarial.corpus import CASES, Case
from adversarial.run import outcome_of, run_all, verify_case
from minerva_engine.ncverify import verify_equal_nc
from minerva_engine.verify import verify_equal

A = sp.Symbol("A", commutative=False)
B = sp.Symbol("B", commutative=False)


def _param(case: Case) -> pytest.param:  # type: ignore[valid-type]
    marks = []
    if case.defect:
        marks.append(
            pytest.mark.xfail(
                strict=True, reason=f"open defect {case.defect}: see docs/verification-defects.md"
            )
        )
    return pytest.param(case, id=case.id, marks=marks)


@pytest.mark.parametrize("case", [_param(c) for c in CASES])
def test_the_verifier_answers_each_adversarial_case_correctly(case: Case) -> None:
    verification = verify_case(case)
    assert outcome_of(case, verification) == "correct", (
        f"{case.id} ({case.note}): expected {case.expect}, got passed={verification.passed} "
        f"[{verification.numeric_points_agreed}/{verification.numeric_points_checked}, "
        f"probe={verification.probe_kind}] {verification.note}"
    )


def test_corpus_ids_are_unique_and_every_domain_has_both_kinds() -> None:
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))
    by_domain = collections.defaultdict(set)
    for c in CASES:
        by_domain[c.domain].add(c.truth)
    for domain in ("algebra", "singularity", "branch", "piecewise", "conditional", "precision"):
        assert by_domain[domain] == {"identity", "false"}, domain


def test_probes_only_tighten_never_loosen() -> None:
    # A result that passes with targeted probes also passes without them; a probe is never agreement.
    for case, _, with_probes in run_all():
        without = verify_equal(case.a, case.b, symbolic=False, probe=False)
        if with_probes.passed:
            assert without.passed, case.id
        assert with_probes.numeric_points_agreed == without.numeric_points_agreed, case.id


def test_noncommuting_polynomial_identities_in_both_directions() -> None:
    assert verify_equal_nc((A + B) ** 2, A**2 + A * B + B * A + B**2).passed
    assert not verify_equal_nc((A + B) ** 2, A**2 + 2 * A * B + B**2).passed  # commuting shortcut
    assert not verify_equal_nc(A * B, B * A).passed
    assert verify_equal_nc(A * (B + A), A * B + A**2).passed
    assert not verify_equal_nc((A + B) * (A - B), A**2 - B**2).passed  # cross terms do not cancel


def _cut_case_verification(seconds: float):  # type: ignore[no-untyped-def]
    import minerva_engine.verify as verify_module

    case = next(c for c in CASES if c.id == "branch.log_reciprocal_cut")
    saved = verify_module.PROBE_SECONDS
    verify_module.PROBE_SECONDS = seconds
    try:
        return verify_equal(case.a, case.b, symbolic=False)
    finally:
        verify_module.PROBE_SECONDS = saved


def test_a_search_the_clock_cut_short_is_incomplete_and_does_not_pass() -> None:
    """V-017. On a loaded machine the probes' wall-clock stop ended the search before the negative real axis and
    `log(1/z) = -log(z)` passed. A search the clock ends is now incomplete: it does not pass, it says why, and it
    is not reported as a disagreement (load may turn an answer into a refusal, never a refusal into an answer)."""
    cut = _cut_case_verification(0.0)  # the clock ends the search before its first point
    assert cut.passed is False  # the false identity is no longer accepted
    assert cut.probes_cut_short is True and cut.probe_outcome == "cut_short"
    assert cut.probe_kind is None  # no counterexample was found: incomplete, not different
    assert cut.numeric_points_agreed == cut.numeric_points_checked
    assert "cut short by the time limit" in cut.note and "incomplete" in cut.note
    assert "not a disagreement" in cut.note


def test_a_search_allowed_to_finish_finds_the_counterexample_and_is_not_cut_short() -> None:
    whole = _cut_case_verification(60.0)
    assert whole.probes_cut_short is False and whole.probe_outcome == "found"
    assert whole.passed is False and whole.probe_kind is not None  # the false identity is found
