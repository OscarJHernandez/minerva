"""The transformation benchmark: well-formed cases, a reproducible baseline, honest pending cases.

Only correctness is asserted here. Latency and effort are measured with
`python -m minerva_knowledge step-benchmark --repeat 5`, never in CI.
"""

import pytest
from minerva_engine import Engine
from minerva_knowledge import KnowledgeBase
from minerva_knowledge.step_benchmark import (
    ADAPTERS,
    StepCase,
    StepOutcome,
    load_baseline,
    load_step_cases,
    run_step_benchmark,
    run_step_case,
    score,
)

CASES = load_step_cases()


def test_case_ids_are_unique_and_inputs_nonempty() -> None:
    assert len({c.id for c in CASES}) == len(CASES)
    assert all(c.input.strip() for c in CASES)


def test_cases_cover_every_operation_and_every_kind_of_outcome() -> None:
    assert {c.operation for c in CASES} == {"substitute", "collect", "isolate", "rewrite"}
    assert {c.status for c in CASES} >= {
        "ok",
        "unchanged",
        "unsupported",
        "parse_error",
        "conditions_not_satisfied",
    }
    assert any(c.pending for c in CASES), "unsolved cases are how progress is measured honestly"
    assert any("noncommutative" in c.tags for c in CASES)
    assert {c.domain for c in CASES} >= {"algebra", "physics", "applied", "operators"}


def test_rule_cases_reference_real_entries(kb: KnowledgeBase) -> None:
    for case in CASES:
        if case.rule_id is not None:
            entry = kb.get(case.rule_id)
            assert entry is not None and entry.rule is not None, f"{case.id}: {case.rule_id}"


@pytest.mark.parametrize(
    "bad",
    [
        {"operation": "isolate", "input": "a = b"},  # no variable
        {"operation": "isolate", "input": "a = b", "variable": "a"},  # ok without solutions
        {"operation": "substitute", "input": "x", "results": ["x"]},  # no relations
        {"operation": "collect", "input": "x", "results": ["x"]},  # no targets
        {"operation": "rewrite", "input": "x", "action": "expand", "results": ["x"]},  # no part
        {"operation": "collect", "input": "x", "by": ["x"]},  # ok without results
        {"operation": "collect", "input": "x", "by": ["x"], "results": ["x"], "solutions": []},
        {"operation": "rewrite", "input": "x", "part": "x", "action": "rule", "results": ["x"]},
    ],
)
def test_malformed_cases_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        StepCase(id="bad", **bad)  # type: ignore[arg-type]


# --- scoring is strict ----------------------------------------------------------------------------
def case(**fields: object) -> StepCase:
    base: dict[str, object] = {"id": "t", "operation": "collect", "input": "x", "by": ["x"]}
    return StepCase(**{**base, **fields})  # type: ignore[arg-type]


def test_structural_match_rejects_an_equal_but_differently_written_result() -> None:
    wanted = case(results=["(a + c)*x**2"], match="structural")
    assert score(wanted, StepOutcome("ok", ["(a + c)*x**2"])) == (True, True)
    assert score(wanted, StepOutcome("ok", ["a*x**2 + c*x**2"])) == (False, False)


def test_equivalent_match_accepts_another_form_of_the_same_function_but_not_another_function() -> (
    None
):
    wanted = case(operation="substitute", relations=["a = b"], results=["x**2 - 1"])
    assert score(wanted, StepOutcome("ok", ["(x - 1)*(x + 1)"])).count(True) == 2
    assert score(wanted, StepOutcome("ok", ["x**2 + 1"])) == (False, False)


def test_a_wrong_status_never_passes_even_with_the_right_text() -> None:
    wanted = case(results=["x"], match="structural")
    assert score(wanted, StepOutcome("not_verified", ["x"])) == (False, False)
    refusal = case(status="unsupported")
    assert score(refusal, StepOutcome("unsupported")) == (True, True)
    assert score(refusal, StepOutcome("ok", ["x"])) == (False, False)


def test_isolate_needs_the_complete_solution_set() -> None:
    wanted = case(operation="isolate", input="x^2 = 4", variable="x", solutions=["2", "-2"])
    assert score(wanted, StepOutcome("ok", ["2", "-2"])) == (True, True)
    assert score(wanted, StepOutcome("ok", ["2"])) == (False, False), "a dropped solution"
    assert score(wanted, StepOutcome("ok", ["2", "-2", "0"])) == (False, False), "an extra one"
    assert score(wanted, StepOutcome("ok", ["2", "2"])) == (False, False), "a repeated one"
    none = case(operation="isolate", input="x = 1", variable="x", solutions=[])
    assert score(none, StepOutcome("ok", [])) == (True, True)
    assert score(none, StepOutcome("ok", ["1"])) == (False, False)


def test_noncommutative_order_counts_when_scoring() -> None:
    nc = {"A": ["noncommutative"], "B": ["noncommutative"]}
    wanted = case(assumptions=nc, results=["A*B"], match="equivalent")
    assert score(wanted, StepOutcome("ok", ["A*B"])) == (True, True)
    assert score(wanted, StepOutcome("ok", ["B*A"])) == (False, False)


# --- the baseline -----------------------------------------------------------------------------------
def test_the_recorded_baseline_covers_every_case() -> None:
    assert set(load_baseline()) == {c.id for c in CASES}


def test_the_baseline_is_reproducible(kb: KnowledgeBase, engine: Engine) -> None:
    """The existing functionality is deterministic: re-running it reproduces the recorded pass/fail.
    A change here means an older operation changed behaviour, not that the new ones improved."""
    report = run_step_benchmark(kb, engine, "existing", CASES)
    recorded = load_baseline()
    changed = {i: p for i, p in report.solved_ids().items() if recorded[i] != p}
    assert not changed, f"baseline differs for {sorted(changed)}; regenerate with --write-baseline"


# --- the new operations -----------------------------------------------------------------------------
@pytest.mark.skipif(
    "steps" not in ADAPTERS, reason="the transformation operations are not wired in yet"
)
@pytest.mark.parametrize("case_", [c for c in CASES if not c.pending], ids=lambda c: c.id)
def test_every_scored_case_passes(case_: StepCase, kb: KnowledgeBase, engine: Engine) -> None:
    result = run_step_case(kb, engine, case_, ADAPTERS["steps"])
    assert result.passed, f"status {result.status}, results {result.results}: {result.message}"


@pytest.mark.skipif(
    "steps" not in ADAPTERS, reason="the transformation operations are not wired in yet"
)
def test_nothing_the_baseline_solved_is_lost(kb: KnowledgeBase, engine: Engine) -> None:
    report = run_step_benchmark(kb, engine, "steps", CASES)
    solved_now = report.solved_ids()
    lost = [i for i, was in load_baseline().items() if was and not solved_now[i]]
    assert not lost, f"solved by the existing functionality but not by the new one: {lost}"


@pytest.mark.skipif(
    "steps" not in ADAPTERS, reason="the transformation operations are not wired in yet"
)
def test_pending_cases_are_not_stale(kb: KnowledgeBase, engine: Engine) -> None:
    """A pending case that passes must be promoted, or the benchmark stops guarding it."""
    stale = [
        c.id for c in CASES if c.pending and run_step_case(kb, engine, c, ADAPTERS["steps"]).passed
    ]
    assert not stale, f"promote these (remove pending): {stale}"
