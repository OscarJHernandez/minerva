"""The server log says which tool failed, what kind of call it was and the coded reason, with no content.

A tool call that did not deliver a verified result gets one line at the default log setting:
`tool.failed` for something to fix (invalid arguments, a parse error, an unverified candidate, a timeout,
a crashed tool) and `tool.declined` for an honest answer (no handbook entry, nothing changed, not
supported, a condition not provable) or a call that was not run. The line carries the tool, the status,
the error category, a fixed reason code, the argument *names* of the tool's own schema and the failed
schema paths. It never carries an argument value, a symbol the user chose or an exception message; the
stored trace under the same `trace_id` has the content.
"""

import asyncio
import dataclasses
import json
from typing import Any

import pytest
from minerva_api import logging as api_logging
from minerva_api.agent.loop import AgentLimits, TurnInfo, run_agent
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import TOOLS, ToolContext, execute
from minerva_api.agent.tracing import (
    OUTCOME_OF_CATEGORY,
    REASON_CODES,
    ErrorCategory,
    ToolDeclined,
    ToolFailed,
    outcome_kind,
)
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

SECRET = "secret@example.com"


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def play(replies: list[Any], ctx: ToolContext) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one turn; return the stored message and every record the log sink was handed."""
    records: list[dict[str, Any]] = []

    async def go() -> list[dict[str, Any]]:
        return [
            e
            async for e in run_agent(
                ReplayLLM(replies),
                [{"role": "user", "content": "q"}],
                ctx,
                AgentLimits(retry_delay_seconds=0),
                info=TurnInfo(sink=records.append),
            )
        ]

    message = next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")
    return message, records


# --------------------------------------------------------------------------------- the mapping
def test_every_error_category_has_an_outcome_kind() -> None:
    assert set(OUTCOME_OF_CATEGORY) == set(ErrorCategory)  # a new category needs a decision


def test_outcome_kind_separates_failures_from_answers() -> None:
    assert outcome_kind("ok", None) is None
    assert outcome_kind("unchanged", None) == "answer"
    assert outcome_kind("no_results", None) == "answer"
    assert outcome_kind("unsupported", ErrorCategory.UNSUPPORTED) == "answer"
    assert outcome_kind("conditions_not_satisfied", "conditions_not_satisfied") == "answer"
    assert outcome_kind("parse_error", ErrorCategory.PARSE_ERROR) == "failure"
    assert outcome_kind("resource_limit", ErrorCategory.TIMEOUT) == "failure"
    assert outcome_kind("error", ErrorCategory.NEEDS_USER_CHOICE) == "answer"
    assert outcome_kind("error", None) == "failure"  # no category: treat as something to look at
    assert outcome_kind("skipped", ErrorCategory.LIMIT_REACHED) == "answer"


# ------------------------------------------------------------- the lines a turn writes
def test_an_equation_sent_to_substitute_is_a_named_failure(ctx: ToolContext) -> None:
    equation = {
        "latex": r"\ddot{r} = - \frac{k}{m r^{2}}",
        "relations": ["r = 1/u"],
        "assumptions": {"r": ["positive"]},
    }
    message, records = play([call("substitute", equation), say("It could not be read.")], ctx)
    failed = [r for r in records if r["event"] == "tool.failed"]
    assert len(failed) == 1
    line = failed[0]
    assert line["tool"] == "substitute" and line["status"] == "parse_error"
    assert line["error_category"] == "parse_error" and line["outcome_kind"] == "failure"
    assert line["arg_fields"] == ["assumptions", "latex", "relations"]
    assert line["trace_id"] == message["debug"]["trace_id"] and line["arguments_valid"] is True
    done = next(r for r in records if r["event"] == "agent.completed")
    assert done["tools_called"] == ["substitute"] and done["failed_tools"] == ["substitute"]
    assert done["declined_tools"] == [] and done["first_failure_code"] == "parse_error"
    assert "r = 1/u" not in json.dumps(records) and "ddot" not in json.dumps(records)


def test_a_handbook_search_with_no_match_is_declined_not_failed(ctx: ToolContext) -> None:
    _, records = play(
        [call("search_handbook", {"query": "zzqx nonsense"}), say("Nothing matched.")], ctx
    )
    assert not [r for r in records if r["event"] == "tool.failed"]
    declined = [r for r in records if r["event"] == "tool.declined"]
    assert [r["tool"] for r in declined] == ["search_handbook"]
    assert declined[0]["outcome_kind"] == "answer" and declined[0]["status"] == "no_results"
    done = next(r for r in records if r["event"] == "agent.completed")
    assert done["declined_tools"] == ["search_handbook"] and done["failed_tools"] == []
    assert done.get("first_failure_code") is None


def test_a_verified_call_writes_no_outcome_line(ctx: ToolContext) -> None:
    _, records = play(
        [call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}), say("It is $x + 1$.")], ctx
    )
    assert not [r for r in records if r["event"] in ("tool.failed", "tool.declined")]
    done = next(r for r in records if r["event"] == "agent.completed")
    assert done["tools_called"] == ["simplify"] and done["failed_tools"] == []


def test_a_repeated_invalid_call_is_logged_each_time_and_the_count_is_bounded(
    ctx: ToolContext,
) -> None:
    bad = {"equation": "F = m a", "quantities": [{"symbol": "k", "unit": "N"}]}
    _, records = play([call("check_units", bad)] * 3 + [say("I could not check it.")], ctx)
    failed = [r for r in records if r["event"] == "tool.failed"]
    assert len(failed) == 3 and {r["tool"] for r in failed} == {"check_units"}
    assert failed[0]["reason_code"] == "missing" and failed[0]["arguments_valid"] is False
    assert failed[0]["invalid_fields"] == ["quantities.*.name"]
    assert [r["repeated"] for r in failed] == [False, True, True]
    assert len(failed) <= AgentLimits().max_tool_calls  # the turn limit is the cap


def test_a_tool_that_crashes_reports_its_class_and_location_not_its_message(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(args: Any, context: ToolContext) -> Any:
        raise ValueError(f"cannot simplify x^2 + {SECRET}")

    monkeypatch.setitem(TOOLS, "simplify", dataclasses.replace(TOOLS["simplify"], run=boom))
    _, records = play([call("simplify", {"latex": "x"}), say("It failed.")], ctx)
    failed = next(r for r in records if r["event"] == "tool.failed")
    assert failed["error_category"] == "tool_failure" and failed["error_type"] == "ValueError"
    assert failed["error_module"].startswith("minerva_api") and failed["error_line"] > 0
    assert SECRET not in json.dumps(records) and "x^2" not in json.dumps(records)


# ----------------------------------------------------------------- nothing chosen by data
def test_invalid_field_paths_never_contain_a_symbol_the_user_chose(ctx: ToolContext) -> None:
    # `assumptions` is a dictionary keyed by the user's symbols: the path must not repeat the key.
    out = execute("simplify", {"latex": "x", "assumptions": {SECRET: [5]}}, ctx)
    assert out.reason_code == "string_type" and out.invalid_fields == ("assumptions.*.*",)
    out = execute("simplify", {"latex": "x", "assumptions": {SECRET: "positive"}}, ctx)
    assert out.reason_code == "list_type" and out.invalid_fields == ("assumptions.*",)
    assert SECRET not in " ".join(out.invalid_fields)
    out = execute("check_units", {"equation": "F = m a", "quantities": [{"unit": "N"}]}, ctx)
    assert out.invalid_fields == ("quantities.*.name",)
    out = execute("series", {"latex": "x"}, ctx)
    assert out.invalid_fields == ("variable",) and out.reason_code == "missing"


def test_event_fields_are_closed_vocabularies_and_drop_anything_else() -> None:
    event = ToolFailed(
        trace_id="t",
        seq=1,
        t_ms=0,
        tool="simplify",
        reason_code=f"bad {SECRET}",
        error_type=f"Error {SECRET}",
        error_module="not_ours.module",
        arg_fields=["latex", SECRET, "x**2"],
        invalid_fields=["assumptions.*", SECRET, "a b"],
    )
    assert event.reason_code is None and event.error_type is None and event.error_module is None
    assert event.arg_fields == ["latex"] and event.invalid_fields == ["assumptions.*"]
    ok = ToolDeclined(trace_id="t", seq=1, t_ms=0, reason_code="missing", error_type="ValueError")
    assert ok.reason_code == "missing" and ok.error_type == "ValueError"
    assert "missing" in REASON_CODES and "value_error" in REASON_CODES


def test_every_tool_argument_name_is_a_plain_name() -> None:
    # `arg_fields` are taken from these, so each one must pass the event's own name check.
    import re

    for tool in TOOLS.values():
        for name in tool.args_model.model_fields:
            assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", name), (tool.name, name)


# ---------------------------------------------------------------------- the real log line
def test_the_default_log_setting_writes_the_failure_line_without_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fastapi.testclient import TestClient

    from .conftest import build_app, register
    from .test_chat import sse

    with TestClient(build_app(llm_mode="scripted")) as client:  # the default is "summary"
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        api_logging.configure_logging()
        capsys.readouterr()
        # an expression the strict reader refuses: the tool fails, and the log says so by name
        sse(client, cid, r"simplify $\sin 2x$")
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    names = [r["event"] for r in lines]
    assert "agent.completed" in names and "tool.completed" not in names
    for r in lines:
        if r["event"] in ("tool.failed", "tool.declined"):
            assert r["trace_id"] and r["tool"] and r["status"] and r["outcome_kind"]
    out = json.dumps(lines)
    assert r"\sin" not in out and "2x" not in out
