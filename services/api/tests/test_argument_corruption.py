"""The corruption test: every known model mistake, on every reference call of the benchmark, is harmless.

A corrupted call must give the clean call's result or fail with a typed reason. A verified result that differs
from the clean one means a normalisation changed the mathematics; a failure with no typed reason means the
model gets only prose. Both are defects and both must stay at zero. The operators are the inverses of the
normalisations, so a normalisation cannot be added without the mistake it repairs being tried.
"""

import json
from pathlib import Path

import pytest
from minerva_api.agent.evaluation import corruption
from minerva_api.agent.evaluation.corruption import (
    DEFECTS,
    DIAGNOSED,
    DIFFERENT,
    OPERATORS,
    SAME,
    UNTYPED,
    Reference,
    classify,
    run_call,
    run_corruptions,
)
from minerva_api.agent.tools import TOOLS, ToolContext
from minerva_engine import Engine, Limits
from minerva_engine.notation.transport import DAMAGE
from minerva_knowledge import KnowledgeBase

DOCS = Path(__file__).resolve().parents[3] / "docs" / "benchmarks"


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=120, max_concurrency=2)), kb=KnowledgeBase.load()
    )


@pytest.fixture(scope="module")
def report(ctx: ToolContext) -> dict:  # type: ignore[type-arg]
    return run_corruptions(ctx)


def test_every_damage_has_an_operator_and_the_reverse() -> None:
    assert {n for n in OPERATORS if n.startswith("transport:")} == {
        f"transport:{n}" for n in DAMAGE
    }


def test_every_registered_tool_has_a_reference_call_or_a_stated_reason() -> None:
    covered = {r.tool for r in corruption.references()}
    missing = set(TOOLS) - covered - set(corruption.EXEMPT)
    assert not missing, (
        f"add a reference call (benchmark case) or an EXEMPT reason for: {sorted(missing)}"
    )
    assert set(corruption.EXEMPT) <= set(TOOLS)
    assert all(reason for reason in corruption.EXEMPT.values())


def test_no_corruption_changes_a_result_or_fails_without_a_reason(report: dict) -> None:  # type: ignore[type-arg]
    assert report["defects"] == []
    assert all(report["totals"][d] == 0 for d in DEFECTS)


def test_the_cleaning_layer_recovers_almost_everything(report: dict) -> None:  # type: ignore[type-arg]
    assert report["applied"] >= 400
    assert report["recovery_rate"] >= 0.97


def test_the_stored_report_is_current(report: dict) -> None:  # type: ignore[type-arg]
    stored = json.loads((DOCS / "corruption.json").read_text())
    assert corruption.deterministic_view(stored) == corruption.deterministic_view(report)


# -- the oracle itself: a harness that cannot fail proves nothing ----------------------------------

_SIMPLIFY = Reference("sanity", "simplify", {"latex": r"\frac{x^2-1}{x-1}"})


def test_a_corruption_that_changes_the_mathematics_is_reported_first(ctx: ToolContext) -> None:
    clean = run_call("simplify", _SIMPLIFY.arguments, ctx)
    wrong = run_call("simplify", {"latex": r"\frac{x^2-1}{x+1}"}, ctx)
    assert classify(clean, wrong) == DIFFERENT


def test_a_refusal_without_a_typed_reason_is_a_defect(ctx: ToolContext) -> None:
    clean = run_call("simplify", _SIMPLIFY.arguments, ctx)
    unreadable = run_call("simplify", {"latex": r"\frac{x^2-1}{"}, ctx)
    assert unreadable.status == "parse_error" and unreadable.reason_code is None
    assert classify(clean, unreadable) == UNTYPED


def test_a_typed_refusal_and_an_identical_result_are_accepted(ctx: ToolContext) -> None:
    clean = run_call("simplify", _SIMPLIFY.arguments, ctx)
    assert classify(clean, run_call("simplify", _SIMPLIFY.arguments, ctx)) == SAME
    letters = run_call(
        "analyze_lagrangian",
        {"lagrangian": r"\frac12 m xdot^2", "coordinates": ["x"], "format": "latex"},
        ctx,
    )
    assert letters.reason_code == "unrecognized_derivative"
    assert classify(clean, letters) == DIAGNOSED


def test_an_operator_that_does_not_apply_is_skipped_not_counted(ctx: ToolContext) -> None:
    only = {"derivative:joined": OPERATORS["derivative:joined"]}
    one = run_corruptions(ctx, operators=only, refs=[_SIMPLIFY])
    assert one["applied"] == 0 and one["not_applicable"] == {"derivative:joined": 1}
