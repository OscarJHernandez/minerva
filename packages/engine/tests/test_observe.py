"""Observation of sandbox calls: trace ids, outcomes and timings, and never an effect on the result."""

import threading

import pytest
from minerva_engine import (
    Engine,
    ExpressionInput,
    Limits,
    SandboxCall,
    current_trace_id,
    trace_scope,
)
from minerva_engine.sandbox import Sandbox


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(Limits(timeout_seconds=30, max_concurrency=2))


def test_every_sandbox_call_is_reported_with_the_bound_trace_id(engine: Engine) -> None:
    seen: list[SandboxCall] = []
    engine.observer = seen.append
    try:
        with trace_scope("abc123"):
            result = engine.simplify(ExpressionInput(input="(x^2-1)/(x-1)", format="text"))
        engine.simplify(ExpressionInput(input="x", format="text"))  # outside any scope
    finally:
        engine.observer = None
    assert result.status == "ok"
    assert [(c.operation, c.outcome, c.trace_id) for c in seen] == [
        ("simplify", "ok", "abc123"),
        ("simplify", "ok", None),
    ]
    assert all(c.ms >= 0 for c in seen)


def test_the_trace_id_follows_work_started_in_a_thread(engine: Engine) -> None:
    import contextvars

    seen: list[SandboxCall] = []
    engine.observer = seen.append
    try:
        with trace_scope("thread-1"):
            ctx = contextvars.copy_context()  # what a thread pool does for the caller
        worker = threading.Thread(
            target=lambda: ctx.run(engine.simplify, ExpressionInput(input="x + x", format="text"))
        )
        worker.start()
        worker.join()
    finally:
        engine.observer = None
    assert [c.trace_id for c in seen] == ["thread-1"]


def test_scope_is_restored(engine: Engine) -> None:
    assert current_trace_id() is None
    with trace_scope("outer"):
        with trace_scope("inner"):
            assert current_trace_id() == "inner"
        assert current_trace_id() == "outer"
    assert current_trace_id() is None


def test_a_failing_observer_never_changes_the_result(engine: Engine) -> None:
    def broken(call: SandboxCall) -> None:
        raise RuntimeError("observer failed")

    engine.observer = broken
    try:
        result = engine.simplify(ExpressionInput(input="(x^2-1)/(x-1)", format="text"))
    finally:
        engine.observer = None
    assert result.status == "ok" and result.result_text == "x + 1"


def test_a_resource_limit_is_reported_with_its_outcome() -> None:
    busy = Engine(Limits(timeout_seconds=0.001, max_concurrency=1))
    seen: list[SandboxCall] = []
    busy.observer = seen.append
    result = busy.simplify(ExpressionInput(input="(x^2-1)/(x-1)", format="text"))
    assert result.status == "resource_limit"
    assert seen and seen[0].outcome in {"timeout", "busy", "killed"}


def test_the_record_holds_no_expression() -> None:
    fields = set(SandboxCall.__dataclass_fields__)
    assert fields == {"operation", "outcome", "ms", "trace_id"}
    assert isinstance(Sandbox, type)
