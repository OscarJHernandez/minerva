"""The identity benchmark: hand-written cases must hold, and every rule must apply to itself."""

import pytest
from minerva_engine import Applicability, Engine, ExpressionInput, Status
from minerva_knowledge import KnowledgeBase
from minerva_knowledge.benchmark import Case, load_cases, run_benchmark, run_case
from minerva_knowledge.identities import find_identities

CASES = load_cases()


def test_case_ids_are_unique_and_inputs_nonempty() -> None:
    assert len({c.id for c in CASES}) == len(CASES)
    assert all(c.input.strip() for c in CASES)


def test_asserted_cases_reference_real_entries(kb: KnowledgeBase) -> None:
    for case in CASES:
        if case.pending:
            continue
        for entry_id in [*case.expect, *case.forbid]:
            assert kb.get(entry_id) is not None, f"{case.id}: unknown entry {entry_id}"


def test_benchmark_has_no_failures(kb: KnowledgeBase, engine: Engine) -> None:
    report = run_benchmark(kb, engine, CASES)
    messages = [
        f"{r.case.id}: missing={r.missing} forbidden_shown={r.forbidden_shown} status={r.status}"
        for r in report.failures
    ]
    assert not messages, "\n".join(messages)


def test_pending_cases_are_not_stale(kb: KnowledgeBase, engine: Engine) -> None:
    """A pending case that already passes must be promoted, or the benchmark stops guarding it."""
    stale = [c.id for c in CASES if c.pending and run_case(kb, engine, c).passed]
    assert not stale, f"remove `pending` from: {stale}"


def test_pending_cases_do_not_hide_a_broken_case(kb: KnowledgeBase, engine: Engine) -> None:
    """Pending means the handbook lacks the entry, not that the input fails to parse."""
    for case in CASES:
        if case.pending:
            assert run_case(kb, engine, case).status == Status.OK.value, case.id


def test_unknown_expectation_is_reported_missing(kb: KnowledgeBase, engine: Engine) -> None:
    case = Case(id="x", input=r"\sin^2 x + \cos^2 x", expect=["no.such_entry"])
    result = run_case(kb, engine, case)
    assert result.missing == ["no.such_entry"]
    assert not result.passed


def test_forbidden_entry_that_applies_is_reported(kb: KnowledgeBase, engine: Engine) -> None:
    case = Case(id="x", input=r"\sin^2 x + \cos^2 x", forbid=["trig.pythagorean"])
    result = run_case(kb, engine, case)
    assert result.forbidden_shown == ["trig.pythagorean"]
    assert not result.passed


def _own_lhs_inputs(kb: KnowledgeBase) -> list[tuple[str, ExpressionInput]]:
    items = []
    for entry in kb.entries:
        if entry.rule is None:
            continue
        assumptions: dict[str, list[str]] = {}
        for cond in entry.rule.conditions:
            assumptions.setdefault(cond.variable, []).append(cond.assumption)
        items.append(
            (
                entry.id,
                ExpressionInput(input=entry.rule.lhs, format="text", assumptions=assumptions),
            )
        )
    return items


# Rules that cannot fire on their own lhs, by design. Each reason is checked below; a rule that starts
# working must be removed from this table so it is guarded again.
NOT_SELF_APPLICABLE = {
    "log.of_exp": "with `a` real, SymPy evaluates log(exp(a)) to a while parsing: nothing left to rewrite",
    "radical.sqrt_square_abs": "with `a` real, SymPy evaluates sqrt(a**2) to Abs(a) while parsing",
    "radical.sqrt_square_nonneg": "with `a` nonnegative, SymPy evaluates sqrt(a**2) to a while parsing",
    "hyp.cosh_cos": "the rewrite cos(I*a) invents the imaginary unit, which the matcher discards",
}


def _self_application_failures(kb: KnowledgeBase, engine: Engine) -> set[str]:
    failed = set()
    for entry_id, expression in _own_lhs_inputs(kb):
        response = find_identities(kb, engine, expression)
        applicable = {
            m.entry.id
            for m in response.matches
            if m.application.applicability is Applicability.APPLICABLE
        }
        if entry_id not in applicable:
            failed.add(entry_id)
    return failed


def test_every_rule_applies_to_its_own_left_hand_side(kb: KnowledgeBase, engine: Engine) -> None:
    """A rule that cannot fire on its own lhs is dead weight or has an auto-evaluating lhs."""
    failed = _self_application_failures(kb, engine)
    unexplained = sorted(failed - NOT_SELF_APPLICABLE.keys())
    assert not unexplained, f"rules that do not apply to their own lhs: {unexplained}"
    stale = sorted(NOT_SELF_APPLICABLE.keys() - failed)
    assert not stale, f"these now apply to their own lhs; remove them from the exemptions: {stale}"


def test_default_candidate_pool_equals_uncapped_pool(kb: KnowledgeBase, engine: Engine) -> None:
    """A small cap would silently drop applicable rules as the handbook grows."""
    for case in CASES:
        status, _, features, _ = engine.features(
            ExpressionInput(input=case.input, assumptions=case.assumptions)
        )
        if status is not Status.OK:
            continue
        default = {e.id for e in kb.candidate_rules(features)}
        uncapped = {e.id for e in kb.candidate_rules(features, limit=200)}
        assert default == uncapped, case.id


@pytest.mark.parametrize("case", [c for c in CASES if not c.pending], ids=lambda c: c.id)
def test_each_asserted_case(kb: KnowledgeBase, engine: Engine, case: Case) -> None:
    result = run_case(kb, engine, case)
    assert result.passed, f"missing={result.missing} forbidden_shown={result.forbidden_shown}"
