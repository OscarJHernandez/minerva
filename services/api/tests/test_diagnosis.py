"""The diagnosis names a subsystem only when a fixed rule decides it, and says `undetermined` otherwise."""

from typing import Any

import pytest
from minerva_api.agent.diagnosis import STAGE_OF_CATEGORY, STAGES, diagnose
from minerva_api.agent.events import TraceStep, TurnDebug
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_api.agent.tracing import ErrorCategory
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

from .test_agent_reliability import final, play


def step(category: str | None, name: str = "simplify") -> TraceStep:
    return TraceStep(
        id="c", name=name, arguments={}, for_model="", status="ok", error_category=category
    )  # type: ignore[call-arg]


def debug(stop: str = "completed", dropped: int = 0) -> TurnDebug:
    return TurnDebug(stop_reason=stop, events_dropped=dropped)


def test_a_clean_turn_has_no_stage() -> None:
    d = diagnose([step(None)], debug())
    assert d["stage"] == "none" and d["step"] is None and d["replayable"] is True


@pytest.mark.parametrize(
    ("category", "stage"),
    [
        ("invalid_arguments", "model"),
        ("parse_error", "parser"),
        ("unsupported", "engine"),
        ("not_verified", "verifier"),
        ("timeout", "infrastructure"),
        ("limit_reached", "orchestrator"),
    ],
)
def test_one_failing_subsystem_is_named(category: str, stage: str) -> None:
    d = diagnose([step(None), step(category)], debug())
    assert d["stage"] == stage and d["step"] == 1 and d["first_failure_code"] == category


def test_two_subsystems_are_not_ranked() -> None:
    d = diagnose([step("parse_error"), step("not_verified")], debug())
    assert d["stage"] == "undetermined" and d["candidates"] == ["parser", "verifier"]
    assert d["first_failure_code"] == "parse_error"  # the first failure is still reported


def test_an_incomplete_record_is_never_named() -> None:
    assert diagnose([step("parse_error")], debug(dropped=3))["stage"] == "undetermined"
    assert (
        diagnose([step("parse_error")], debug(dropped=3))["replayable_reason"] == "events_dropped"
    )


def test_a_stop_without_a_failed_step_points_at_its_owner() -> None:
    assert diagnose([], debug("model_error"))["stage"] == "infrastructure"
    assert diagnose([], debug("internal_error"))["stage"] == "orchestrator"
    assert diagnose([], debug("step_limit"))["stage"] == "undetermined"


def test_a_timed_out_turn_is_not_called_replayable() -> None:
    d = diagnose([step("timeout")], debug("time_limit"))
    assert d["replayable"] is False and d["replayable_reason"] == "timing_dependent"


def test_every_error_category_is_assigned_a_subsystem_or_listed_as_outside() -> None:
    outside = {"repeated_call"}  # assigned to the model; kept here to force a decision on new ones
    for category in ErrorCategory:
        assert category.value in STAGE_OF_CATEGORY or category.value in outside, category
    assert set(STAGE_OF_CATEGORY.values()) <= STAGES


def test_a_real_turn_stores_its_diagnosis() -> None:
    ctx = ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )
    bad = {"latex": r"\foo{a}"}
    message: dict[str, Any] = final(
        play(ReplayLLM([call("simplify", bad), say("Cannot read it.")]), ctx)
    )
    diagnosis = message["debug"]["diagnosis"]
    assert diagnosis["stage"] == "parser" and diagnosis["step"] == 0
    assert diagnosis["first_failure_code"] == "parse_error" and diagnosis["replayable"] is True
