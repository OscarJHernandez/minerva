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
