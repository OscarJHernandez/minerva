"""Tracing: event schema, trace-id propagation, privacy of the server log, and the conversation's rules.

The claims under test: every agent request has one trace id that joins the stored trace, the stream, the
header and the log; events hold identifiers, counts, durations and fixed vocabulary only (never
content); tracing failures never reach the mathematics; and the detailed trace follows the same access,
deletion and storage rules as the conversation it belongs to.
"""

import asyncio
import json
import re
import types
import typing
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from minerva_api import logging as api_logging
from minerva_api.agent.llm import LLMError
from minerva_api.agent.loop import AgentLimits, TurnInfo, run_agent
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_api.agent.tracing import (
    EVENT_MODELS,
    IDENTIFIER_FIELDS,
    MAX_EVENTS,
    AgentStarted,
    Tracer,
    activate,
    deactivate,
    event_field_names,
)
from minerva_api.db import ChatMessage
from minerva_engine import Engine, Limits, SandboxCall
from minerva_knowledge import KnowledgeBase
from sqlalchemy import func, select

from .conftest import PASSWORD, build_app, register
from .test_chat import sse

SIMPLIFY = {"latex": r"\frac{x^2-1}{x-1}"}
EVENTS = {
    "agent.started",
    "llm.request",
    "llm.response",
    "tool.started",
    "tool.completed",
    "verification.completed",
    "agent.retry",
    "agent.stopped",
    "agent.completed",
}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def run(llm: Any, ctx: ToolContext, **kw: Any) -> dict[str, Any]:
    async def go() -> list[dict[str, Any]]:
        return [
            e
            async for e in run_agent(
                llm,
                [{"role": "user", "content": "q"}],
                ctx,
                AgentLimits(retry_delay_seconds=0),
                **kw,
            )
        ]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


# -------------------------------------------------------------------------------- the schema
def test_a_turn_records_the_documented_events_in_order(ctx: ToolContext) -> None:
    msg = run(ReplayLLM([call("simplify", SIMPLIFY), say("It is $x + 1$.")]), ctx)
    events = msg["debug"]["events"]
    names = [e["event"] for e in events]
    assert names[0] == "agent.started" and names[-1] == "agent.completed"
    assert set(names) <= EVENTS
    assert names.count("llm.request") == names.count("llm.response") == 2
    assert (
        names.index("tool.started")
        < names.index("tool.completed")
        < names.index("verification.completed")
    )
    assert names[-2] == "agent.stopped"
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert [e["t_ms"] for e in events] == sorted(e["t_ms"] for e in events)  # monotonic
    assert {e["trace_id"] for e in events} == {msg["debug"]["trace_id"]}
    assert len(msg["debug"]["trace_id"]) == 32


def test_event_fields_describe_what_happened(ctx: ToolContext) -> None:
    msg = run(ReplayLLM([call("simplify", SIMPLIFY), say("It is $x + 1$.")]), ctx)
    by_name = {e["event"]: e for e in msg["debug"]["events"]}
    started = by_name["agent.started"]
    assert (
        started["model"] == "replay"
        and started["max_steps"] == 6
        and started["max_tool_calls"] == 12
    )
    assert started["history_messages"] == 1 and len(started["prompt_sha"]) == 12
    tool_started = by_name["tool.started"]
    assert tool_started["tool"] == "simplify" and tool_started["arguments_valid"] is True
    completed = by_name["tool.completed"]
    assert completed["status"] == "ok" and completed["tool"] == "simplify"
    assert completed["duration_ms"] >= 0 and completed["engine_ms"] is not None
    assert completed["sandbox_calls"] >= 1 and completed["error_category"] is None
    verification = [e for e in msg["debug"]["events"] if e["event"] == "verification.completed"]
    tool_v = verification[0]
    assert (
        tool_v["scope"] == "tool"
        and tool_v["check"] == "symbolic_zero"
        and tool_v["accepted"] is True
    )
    assert tool_v["points_checked"] and tool_v["restrictions"] == 1
    stopped = by_name["agent.stopped"]
    assert stopped["reason"] == "completed" and stopped["steps"] == 2 and stopped["tool_calls"] == 1
    done = by_name["agent.completed"]
    assert done["status"] == "ok" and done["total_ms"] >= done["llm_ms"] >= 0
    assert done["tool_calls"] == 1 and done["tool_errors"] == 0 and done["formulas"] == 1
    assert done["evidence_pct"] == 100.0 and done["stop_reason"] == "completed"
    assert tool_v["verify_ms"] is not None


def test_a_refused_call_is_recorded_with_its_category(ctx: ToolContext) -> None:
    msg = run(ReplayLLM([call("simplify", {}), say("Sorry.")]), ctx)
    names = {e["event"]: e for e in msg["debug"]["events"]}
    assert names["tool.started"]["arguments_valid"] is False
    assert names["tool.completed"]["status"] == "error"
    assert names["tool.completed"]["error_category"] == "invalid_arguments"
    assert (
        names["agent.completed"]["invalid_calls"] == 1
        and names["agent.completed"]["tool_errors"] == 1
    )


def test_an_unknown_tool_name_is_never_logged_as_given(ctx: ToolContext) -> None:
    secret = "my_secret_tool_zzMARKERzz"
    msg = run(ReplayLLM([call(secret, {"latex": "x"}), say("Sorry.")]), ctx)
    events = msg["debug"]["events"]
    assert secret not in json.dumps(events)
    assert {e["tool"] for e in events if e["event"] == "tool.started"} == {"unknown"}
    assert next(e["proposed_tool"] for e in events if e["event"] == "llm.response") == "unknown"


def test_the_stored_trace_carries_the_same_ids(ctx: ToolContext) -> None:
    msg = run(ReplayLLM([call("simplify", SIMPLIFY), say("ok")]), ctx)
    step = msg["trace"][0]
    completed = next(e for e in msg["debug"]["events"] if e["event"] == "tool.completed")
    assert completed["call_id"] == step["id"] == msg["blocks"][0]["card"]["id"]


# --------------------------------------------------------------------- events hold no content
def _is_text(annotation: Any) -> bool:
    if annotation is str:
        return True
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return any(_is_text(a) for a in typing.get_args(annotation))
    return False


def test_no_event_has_a_free_text_field() -> None:
    """Plain text is allowed only for identifiers; everything else is a number, flag or fixed vocabulary."""
    for model in EVENT_MODELS:
        for name, field in model.model_fields.items():
            if _is_text(field.annotation):
                assert name in IDENTIFIER_FIELDS, f"{model.__name__}.{name} is free text"
    names = {m.model_fields["event"].default for m in EVENT_MODELS}
    assert names == EVENTS


def test_every_event_field_is_on_the_log_allow_list_and_nothing_riskier_is() -> None:
    assert event_field_names() <= api_logging.ALLOWED_FIELDS
    content_words = {
        "latex",
        "expression",
        "arguments",
        "result",
        "text",
        "message",
        "content",
        "prompt",
    }
    assert not (api_logging.ALLOWED_FIELDS & content_words)
    assert not any(
        w in f for f in api_logging.ALLOWED_FIELDS for w in ("latex", "expression", "body")
    )


def test_the_log_filter_drops_unlisted_fields() -> None:
    kept = api_logging._allow_list(
        None,
        "info",
        {"event": "x", "trace_id": "t", "expression": "x**2", "message": "hi", "latex": "y"},
    )
    assert dict(kept) == {"event": "x", "trace_id": "t"}


def test_the_api_has_no_telemetry_or_third_party_logging_dependency() -> None:
    """Events go to stdout through structlog and nowhere else: no SDK that could ship them off."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    banned = {
        "sentry-sdk",
        "opentelemetry",
        "datadog",
        "ddtrace",
        "newrelic",
        "posthog",
        "logfire",
        "rollbar",
        "bugsnag",
        "elasticsearch",
        "kafka-python",
        "confluent-kafka",
        "segment-analytics-python",
        "mixpanel",
        "honeycomb",
        "loguru",
        "fluent-logger",
        "logstash",
    }
    declared = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    names = {re.split(r"[\s<>=!~;\[]", d, maxsplit=1)[0].lower() for d in declared}
    assert not any(n.startswith(tuple(banned)) for n in names), names
    modules = {b.replace("-", "_") for b in banned} | {"sentry_sdk", "segment", "kafka"}
    for path in (root / "src").rglob("*.py"):
        imported = set(
            re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w]*)", path.read_text(), re.M)
        )
        assert not imported & modules, (path.name, imported & modules)


# -------------------------------------------------------------- tracing never changes the maths
def test_a_sink_that_raises_does_not_change_the_reply(ctx: ToolContext) -> None:
    def boom(_: dict[str, Any]) -> None:
        raise RuntimeError("log shipper down")

    replies = [call("simplify", SIMPLIFY), say("It is $x + 1$.")]
    quiet = run(ReplayLLM(replies), ctx)
    loud = run(ReplayLLM(replies), ctx, info=TurnInfo(sink=boom))
    assert loud["text"] == quiet["text"] and loud["stop_reason"] == "completed"
    assert [
        c["data"]["result_text"]
        for c in map(lambda b: b["card"], filter(lambda b: b["kind"] == "card", loud["blocks"]))
    ] == ["x + 1"]
    assert loud["debug"]["events_dropped"] > 0  # the failures are counted, not raised
    assert loud["trace"][0]["status"] == "ok"


def test_a_failing_event_model_is_swallowed() -> None:
    tracer = Tracer("t")
    tracer.emit(AgentStarted, None, max_steps="not a number")  # invalid: dropped, not raised
    assert tracer.events == [] and tracer.dropped == 1
    tracer.emit(AgentStarted, None, max_steps=3)
    assert len(tracer.events) == 1


def test_events_beyond_the_cap_are_counted_not_kept() -> None:
    tracer = Tracer("t", max_events=3)
    for _ in range(5):
        tracer.emit(AgentStarted, None)
    assert len(tracer.events) == 3 and tracer.dropped == 2 and MAX_EVENTS >= 100
    assert [e["seq"] for e in tracer.events] == [1, 2, 3]


def test_the_sink_sees_every_event_as_it_happens() -> None:
    seen: list[dict[str, Any]] = []
    tracer = Tracer("abc", seen.append)
    tracer.emit(AgentStarted, None, model="m")
    assert (
        seen == tracer.events
        and seen[0]["event"] == "agent.started"
        and seen[0]["trace_id"] == "abc"
    )


def test_sandbox_calls_reach_only_the_tracer_with_their_trace_id() -> None:
    engine = Engine(Limits(timeout_seconds=30))
    a, b = Tracer("trace-a"), Tracer("trace-b")
    activate(a, engine)
    activate(b, engine)
    try:
        engine.observer(SandboxCall("simplify", "ok", 5, "trace-a"))  # type: ignore[misc]
        engine.observer(SandboxCall("simplify", "timeout", 9, "trace-b"))  # type: ignore[misc]
        engine.observer(SandboxCall("simplify", "ok", 1, None))  # type: ignore[misc]
        engine.observer(SandboxCall("simplify", "ok", 1, "unregistered"))  # type: ignore[misc]
    finally:
        deactivate(a)
        deactivate(b)
    assert [(c.outcome, c.ms) for c in a.sandbox] == [("ok", 5)]
    assert [(c.outcome, c.ms) for c in b.sandbox] == [("timeout", 9)]
    assert a.sandbox_mark() == 1 and a.sandbox_since(1) == []


def test_two_concurrent_turns_keep_their_own_sandbox_calls(ctx: ToolContext) -> None:
    async def both() -> list[dict[str, Any]]:
        async def one(latex: str) -> dict[str, Any]:
            llm = ReplayLLM([call("simplify", {"latex": latex}), say("ok")])
            async for e in run_agent(llm, [{"role": "user", "content": "q"}], ctx):
                if e["event"] == "message":
                    return e["data"]  # type: ignore[no-any-return]
            raise AssertionError

        return list(await asyncio.gather(one(r"\frac{x^2-1}{x-1}"), one(r"\sin^2 x + \cos^2 x")))

    first, second = asyncio.run(both())
    assert first["debug"]["trace_id"] != second["debug"]["trace_id"]
    for msg in (first, second):
        done = next(e for e in msg["debug"]["events"] if e["event"] == "tool.completed")
        assert done["sandbox_calls"] >= 1 and done["engine_ms"] is not None


# ------------------------------------------------------------------- the API: ids and the log
@pytest.fixture
def chat() -> typing.Iterator[tuple[TestClient, str]]:
    with TestClient(build_app(llm_mode="scripted", agent_events="all")) as client:
        register(client)
        yield client, client.post("/v1/chat/conversations", json={}).json()["id"]


def json_lines(text: str) -> list[dict[str, Any]]:
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def test_one_trace_id_joins_the_header_the_stream_the_stored_message_and_the_log(
    capsys: pytest.CaptureFixture[str], chat: tuple[TestClient, str]
) -> None:
    client, cid = chat
    api_logging.configure_logging()  # bind the logger to the stream capsys captures now
    capsys.readouterr()
    with client.stream(
        "POST",
        f"/v1/chat/conversations/{cid}/messages",
        json={"content": r"simplify $\frac{x^2-1}{x-1}$"},
    ) as response:
        header = response.headers["x-trace-id"]
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    streamed = [json.loads(line[6:]) for line in lines]
    assert header and all(e.get("trace_id") == header for e in streamed)
    messages = client.get(f"/v1/chat/conversations/{cid}").json()["messages"]
    stored = next(m for m in messages if m["role"] == "assistant")["content"]["debug"]
    assert stored["trace_id"] == header
    assert {e["trace_id"] for e in stored["events"]} == {header}
    logged = [r for r in json_lines(capsys.readouterr().out) if r.get("trace_id") == header]
    assert {r["event"] for r in logged} >= EVENTS - {"agent.retry"}
    assert any(r["event"] == "chat_turn" for r in logged)
    assert (
        header != response.headers["x-request-id"]
    )  # a request id and a trace id are different things


def test_summary_mode_logs_one_agent_event_per_request(capsys: pytest.CaptureFixture[str]) -> None:
    with TestClient(build_app(llm_mode="scripted")) as client:  # the default is "summary"
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        capsys.readouterr()
        sse(client, cid, r"simplify $\frac{x^2-1}{x-1}$")
    records = json_lines(capsys.readouterr().out)
    agent = [
        r["event"]
        for r in records
        if r["event"].startswith(("agent.", "llm.", "tool.", "verification."))
    ]
    assert agent == ["agent.completed"]
    done = next(r for r in records if r["event"] == "agent.completed")
    assert done["tool_calls"] == 1 and done["status"] == "ok" and done["stop_reason"] == "completed"


def test_events_can_be_switched_off(capsys: pytest.CaptureFixture[str]) -> None:
    with TestClient(build_app(llm_mode="scripted", agent_events="off")) as client:
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        capsys.readouterr()
        events = sse(client, cid, r"simplify $\frac{x^2-1}{x-1}$")
    out = capsys.readouterr().out
    assert "agent.completed" not in out and "chat_turn" in out  # the request-level line stays
    final = next(d for e, d in events if e == "message")
    assert final["debug"]["events"]  # ... and the stored trace still has its events


def test_no_log_line_contains_content_at_the_most_verbose_setting(
    capsys: pytest.CaptureFixture[str], chat: tuple[TestClient, str]
) -> None:
    client, cid = chat
    api_logging.configure_logging()  # bind the logger to the stream capsys captures now
    marker = "zzTRACEMARKERzz"
    prompts = [
        f"simplify ${marker} + {marker}$",
        f"differentiate ${marker}^2$",
        f"integrate ${marker}$",
        f"series ${marker}$",
        f"solve ${marker}^2 = 4$",
        f"matrix ${marker}$",
        f"lagrangian $\\frac12 m\\dot{{{marker}}}^2$",
        "just tell me without tools: $1+1=3$",
    ]
    capsys.readouterr()
    for prompt in prompts:
        sse(client, cid, prompt)
    out = capsys.readouterr().out
    assert marker not in out and PASSWORD not in out and "ada@example.com" not in out
    assert "agent.completed" in out and "tool.completed" in out
    banned_values = ("x**2", r"\frac", "result_latex", "for_model")
    assert not any(b in out for b in banned_values)
    for record in json_lines(out):
        assert set(record) <= api_logging.ALLOWED_FIELDS


def test_the_log_never_holds_the_arguments_or_results_of_a_tool(
    capsys: pytest.CaptureFixture[str], chat: tuple[TestClient, str]
) -> None:
    client, cid = chat
    api_logging.configure_logging()  # bind the logger to the stream capsys captures now
    capsys.readouterr()
    sse(client, cid, r"simplify $\frac{x^2-1}{x-1}$")
    out = capsys.readouterr().out
    assert "agent.completed" in out  # the log was captured: an empty log would pass vacuously
    assert "x + 1" not in out and "x^2-1" not in out and "x^{2}" not in out


# ------------------------------------------- failed turns, direct tools, and across-turn marks
def test_a_failed_turn_is_stored_with_its_trace(chat: tuple[TestClient, str]) -> None:
    client, cid = chat
    llm = ReplayLLM([call("simplify", SIMPLIFY)])
    llm.fail_on = {1: LLMError("The model host is not reachable."), 2: LLMError("again")}
    client.app.state.local_llm = llm  # type: ignore[attr-defined]
    events = sse(client, cid, "simplify something")
    kinds = [e for e, _ in events]
    assert kinds[-3:] == ["error", "message", "done"]
    stored = client.get(f"/v1/chat/conversations/{cid}").json()["messages"]
    reply = [m for m in stored if m["role"] == "assistant"]
    assert len(reply) == 1  # the failed turn is in the conversation: its trace survives a reload
    content = reply[0]["content"]
    assert content["stop_reason"] == "model_error" and content["debug"]["error"]
    assert len(content["trace"]) == 1 and content["debug"]["trace_id"]


def test_failed_turns_are_logged_as_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with TestClient(build_app(llm_mode="scripted")) as client:
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        llm = ReplayLLM()
        llm.error = LLMError("The model host is not reachable.")
        client.app.state.local_llm = llm  # type: ignore[attr-defined]
        capsys.readouterr()
        sse(client, cid, "hello")
    turn = next(r for r in json_lines(capsys.readouterr().out) if r["event"] == "chat_turn")
    assert turn["status"] == "error" and turn["stop_reason"] == "model_error" and turn["trace_id"]


def test_verified_marks_and_references_carry_into_the_next_turn(
    chat: tuple[TestClient, str],
) -> None:
    client, cid = chat
    client.app.state.local_llm = ReplayLLM([call("simplify", SIMPLIFY), say("ok")])  # type: ignore[attr-defined]
    sse(client, cid, "first")
    client.app.state.local_llm = ReplayLLM(  # type: ignore[attr-defined]
        [
            call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}),
            say("Before: $x + 1$."),
        ]
    )
    events = sse(client, cid, "second")
    msg = next(d for e, d in events if e == "message")
    assert msg["trace"][0]["arguments"]["latex"] == "x + 1"
    assert msg["trace"][0]["status"] == "ok" and msg["trace"][0]["result_id"] == "r2"
    mark = next(
        s
        for b in msg["blocks"]
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] == "math"
    )
    assert mark["status"] == "from_tool"  # verified in the first turn, still verified in the second


def test_direct_tool_runs_carry_a_trace_and_log_no_content(
    capsys: pytest.CaptureFixture[str], chat: tuple[TestClient, str]
) -> None:
    client, cid = chat
    api_logging.configure_logging()  # bind the logger to the stream capsys captures now
    marker = "zzDIRECTzz"
    body = {
        "lagrangian": rf"\frac12 m\dot{{x}}^2 - \frac12 k x^2 + 0 {marker}",
        "coordinates": ["x"],
        "parameters": {"m": 1.0, "k": 4.0},
        "initial_positions": [1.0],
        "initial_velocities": [0.0],
        "t_end": 1.0,
    }
    capsys.readouterr()
    reply = client.post(f"/v1/chat/conversations/{cid}/tools/integrate_motion", json=body)
    assert reply.status_code == 201, reply.text
    debug = reply.json()["content"]["debug"]
    assert debug["mode"] == "direct" and len(debug["trace_id"]) == 32
    assert [e["event"] for e in debug["events"]] == ["tool.completed", "agent.completed"]
    out = capsys.readouterr().out
    assert marker not in out and debug["trace_id"] in out


# ------------------------------------- the trace is conversation data: access, deletion, storage
def second_user(app: Any) -> TestClient:
    other = TestClient(app)
    other.__enter__()
    register(other, "grace@example.com")
    return other


def test_another_account_cannot_read_or_extend_a_trace() -> None:
    app = build_app(llm_mode="scripted")
    with TestClient(app) as ada:
        register(ada)
        cid = ada.post("/v1/chat/conversations", json={}).json()["id"]
        sse(ada, cid, r"simplify $\frac{x^2-1}{x-1}$")
        assert ada.get(f"/v1/chat/conversations/{cid}").json()["messages"][1]["content"]["trace"]
        with TestClient(app) as grace:
            register(grace, "grace@example.com")
            assert grace.get(f"/v1/chat/conversations/{cid}").status_code == 404
            assert grace.delete(f"/v1/chat/conversations/{cid}").status_code == 404
            r = grace.post(f"/v1/chat/conversations/{cid}/messages", json={"content": "hi"})
            assert r.status_code == 404
            assert grace.get("/v1/chat/conversations").json() == []


def test_deleting_a_conversation_deletes_its_traces() -> None:
    app = build_app(llm_mode="scripted", agent_events="all")
    with TestClient(app) as client:
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        sse(client, cid, r"simplify $\frac{x^2-1}{x-1}$")
        client.post(
            f"/v1/chat/conversations/{cid}/tools/integrate_motion",
            json={
                "lagrangian": r"\frac12 m\dot{x}^2 - \frac12 k x^2",
                "coordinates": ["x"],
                "parameters": {"m": 1.0, "k": 4.0},
                "initial_positions": [1.0],
                "initial_velocities": [0.0],
                "t_end": 1.0,
            },
        )
        with app.state.session_factory() as db:
            assert db.scalar(select(func.count()).select_from(ChatMessage)) == 3
        assert client.delete(f"/v1/chat/conversations/{cid}").status_code == 204
        with app.state.session_factory() as db:
            assert db.scalar(select(func.count()).select_from(ChatMessage)) == 0
        assert client.get(f"/v1/chat/conversations/{cid}").status_code == 404


def test_the_trace_has_no_home_but_the_message_row() -> None:
    """No table, file or endpoint of its own: nothing to retain, export or delete separately."""
    from minerva_api import db as models

    tables = set(models.Base.metadata.tables)
    assert not {t for t in tables if "trace" in t or "event" in t or "debug" in t}
    chat_columns = {c.name for c in models.ChatMessage.__table__.columns}
    assert chat_columns == {"id", "conversation_id", "role", "content", "created_at"}
    routes = {getattr(r, "path", "") for r in build_app().routes}
    assert not any("trace" in p or "debug" in p for p in routes)
    src = Path(__file__).resolve().parents[1] / "src" / "minerva_api"
    writers = [
        p.name
        for p in src.rglob("*.py")
        if "evaluation"
        not in p.parts  # the benchmark CLI writes its own report, never a chat trace
        and re.search(r"open\([^)]*[\"']w|write_text|FileHandler|\.dump\(", p.read_text())
    ]
    assert (
        writers == []
    )  # nothing in the server writes files: traces cannot end up on disk elsewhere


def test_the_direct_tool_and_stream_paths_store_traces_only_in_messages() -> None:
    app = build_app(llm_mode="scripted")
    with TestClient(app) as client:
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        sse(client, cid, r"simplify $\frac{x^2-1}{x-1}$")
        with app.state.session_factory() as db:
            rows = list(db.scalars(select(ChatMessage)))
        assert {r.role for r in rows} == {"user", "assistant"}
        assert all("trace_id" not in r.content for r in rows if r.role == "user")
