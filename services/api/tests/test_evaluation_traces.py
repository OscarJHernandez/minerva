"""Trace export, tool-selection scoring of `load_tools`, multi-turn token totals and the new report metrics."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from minerva_api.agent.evaluation.cases import EvalCase, load_suite
from minerva_api.agent.evaluation.report import build_document, compare, render_markdown, summarise
from minerva_api.agent.evaluation.runner import run_case
from minerva_api.agent.evaluation.score import CaseResult, Observation, combine, score_case
from minerva_api.agent.evaluation.traces import TraceWriter, report_data, trace_path
from minerva_api.agent.loop import AgentLimits
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

FAST = AgentLimits(retry_delay_seconds=0)
CASE_ID = "reciprocal-integral-invents-no-assumption"
IDEAL: list[Any] = [
    call("integrate", {"latex": r"\frac{1}{x}", "variable": "x"}),
    say(r"An antiderivative is $\log{\left(x \right)}$, up to a constant."),
]


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def _replay_module() -> Any:
    path = Path(__file__).resolve().parents[3] / "scripts" / "replay_trace.py"
    spec = importlib.util.spec_from_file_location("replay_trace_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _case() -> EvalCase:
    return {c.id: c for c in load_suite("reliability")}[CASE_ID]


def _observation(calls: list[str]) -> Observation:
    return Observation(
        calls=[(name, {}) for name in calls],
        call_ids=[f"c{i}" for i in range(len(calls))],
        final={"text": "ok", "trace": [], "blocks": [], "debug": {}},
    )


def _simple_case(**extra: Any) -> EvalCase:
    return EvalCase(
        id="t", category="calculus", prompt="p", expected_tool="simplify", source="test", **extra
    )


# ---------------------------------------------------------------------------------------------- scoring
def test_load_tools_is_not_a_tool_selection() -> None:
    """A model that widens its toolset and then calls the right tool chose the right tool."""
    result = score_case(_simple_case(), _observation(["load_tools", "simplify"]))
    assert result.first_tool == "simplify" and result.tool_selected and result.tool_reached
    assert result.calls == 2  # it is still a call, only not a choice of mathematical tool


def test_load_tools_alone_is_not_the_expected_tool() -> None:
    result = score_case(_simple_case(), _observation(["load_tools"]))
    assert result.first_tool is None and not result.tool_selected and not result.tool_reached


def test_a_wrong_tool_after_load_tools_is_still_a_miss() -> None:
    result = score_case(_simple_case(), _observation(["load_tools", "integrate"]))
    assert result.first_tool == "integrate" and not result.tool_selected


def test_combine_sums_tokens_over_turns() -> None:
    a = CaseResult(
        case_id="c", category="calculus", passed=True, expected_tool="x",
        tool_selected=True, tool_reached=True, tokens_in=100, tokens_out=10,
    )  # fmt: skip
    b = a.model_copy(update={"tokens_in": 40, "tokens_out": None})
    merged = combine([a, b])
    assert merged.tokens_in == 140 and merged.tokens_out == 10 and merged.turns == 2
    unreported = combine(
        [a.model_copy(update={"tokens_in": None}), b.model_copy(update={"tokens_in": None})]
    )
    assert unreported.tokens_in is None


# ------------------------------------------------------------------------------------------ the metrics
def _result(**kw: Any) -> CaseResult:
    base: dict[str, Any] = {
        "case_id": "c", "category": "calculus", "passed": True, "expected_tool": "simplify",
        "tool_selected": True, "tool_reached": True, "calls": 1, "llm_calls": 2,
    }  # fmt: skip
    return CaseResult(**{**base, **kw})


def test_summary_has_per_tool_coverage_model_calls_and_consistency() -> None:
    rows = [
        _result(result_ok=True),
        _result(case_id="d", passed=False, result_ok=False, tool_selected=False, invalid_calls=1),
        _result(case_id="e", expected_tool="integrate", turns=2, llm_calls=5),
        _result(case_id="f", expected_tool="integrate", turns=2, passed=False, llm_calls=4),
    ]
    s = summarise(rows)
    assert s["verified_result_coverage"] == 0.5 and s["verified_result_runs"] == 2
    assert s["model_calls"] == {"total": 13, "median": 3.0, "mean": 3.25}
    assert s["multi_turn_runs"] == 2 and s["multi_turn_consistency"] == 0.5
    assert s["per_tool"]["simplify"]["tool_selection"] == 0.5
    assert (
        s["per_tool"]["simplify"]["invalid_calls"] == 1 and s["per_tool"]["integrate"]["runs"] == 2
    )


def test_markdown_and_comparison_show_the_new_metrics() -> None:
    rows = [_result(result_ok=True), _result(case_id="e", turns=2)]
    doc = build_document({"label": "a"}, {"m": rows})
    text = render_markdown(doc)
    assert "Verified-result coverage: 100.0%" in text and "### By expected tool" in text
    other = build_document({"label": "b"}, {"m": [_result(result_ok=False, llm_calls=3)]})
    table = compare(doc, other)
    assert "model calls per run (median)" in table and "tokens in (total)" in table
    assert "verified-result coverage" in table


# ------------------------------------------------------------------------------------------ the traces
def _run(ctx: ToolContext, script: list[Any], writer: TraceWriter | None) -> CaseResult:
    async def go() -> CaseResult:
        return await run_case(ReplayLLM(script), ctx, _case(), FAST, on_turn=writer)

    return asyncio.run(go())


def test_only_failed_turns_are_written_by_default(ctx: ToolContext, tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path)
    assert _run(ctx, IDEAL, writer).passed
    assert writer.written == [] and not list(tmp_path.iterdir())


def test_a_failed_turn_is_written_and_replays_like_the_original(
    ctx: ToolContext, tmp_path: Path
) -> None:
    invented = [
        call("integrate", {"latex": r"\frac{1}{x}", "variable": "x", "assumptions": {"x": ["positive"]}}),
        say(r"An antiderivative is $\log{\left(x \right)}$ for positive x."),
    ]  # fmt: skip
    writer = TraceWriter(tmp_path)
    result = _run(ctx, invented, writer)
    assert not result.passed
    path = trace_path(tmp_path, CASE_ID, 0, 1)
    assert writer.written == [path] and path.exists()

    replay = _replay_module()
    data = replay.extract_data(path.read_text(encoding="utf-8"))
    assert data["format"] == 3 and [m["role"] for m in data["messages"]] == ["user", "assistant"]
    assert data["messages"][0]["text"] == _case().prompt
    replayed = replay.replay(data, ctx)
    assert [r.name for r in replayed] == ["integrate"] and all(r.same for r in replayed)
    assert data["messages"][1]["prose"], "the model's prose is kept for --llm replays"


def test_every_turn_is_written_with_traces_all(ctx: ToolContext, tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path, everything=True)
    assert _run(ctx, IDEAL, writer).passed
    assert len(writer.written) == 1
    replay = _replay_module()
    data = replay.extract_data(writer.written[0].read_text(encoding="utf-8"))
    assert all(r.same for r in replay.replay(data, ctx))


def test_a_turn_without_a_reply_is_exported_as_the_last_error() -> None:
    obs = Observation(error="the model host failed", partial_trace=[{"name": "simplify"}])
    data = report_data("prompt", obs)
    assert [m["role"] for m in data["messages"]] == ["user"]
    assert data["last_error"]["trace"] == [{"name": "simplify"}]
    json.dumps(data)  # serialisable


def test_a_trace_never_reaches_the_report_document(ctx: ToolContext, tmp_path: Path) -> None:
    """The JSON/Markdown reports hold scores (and the existing short reply excerpt) only: the prompt, the tool
    arguments and the trace stay in the trace file."""
    result = _run(ctx, IDEAL, TraceWriter(tmp_path, everything=True))
    doc = build_document({"label": "x"}, {"m": [result]})
    text = json.dumps(doc) + render_markdown(doc)
    assert _case().prompt not in text
    assert (
        "minerva-debug-json" not in text and '"trace"' not in text and "model_arguments" not in text
    )


def _with_formulas(values: list[str]) -> Observation:
    obs = _observation(["simplify"])
    obs.final["blocks"] = [
        {
            "kind": "text",
            "segments": [{"type": "math", "value": v, "status": "unchecked"} for v in values],
        }
    ]
    return obs


def test_a_lone_symbol_is_not_an_unchecked_formula() -> None:
    """`$x$` names a variable and asserts nothing; the loop's own counters leave it out, so does scoring."""
    result = score_case(_simple_case(max_unchecked=0), _with_formulas(["x", r"\theta_1"]))
    assert result.unchecked == 0 and not any("unchecked" in n for n in result.notes)


def test_an_unchecked_relation_still_counts() -> None:
    result = score_case(_simple_case(max_unchecked=0), _with_formulas(["x", "a = b + 1"]))
    assert result.unchecked == 1
    assert any("unchecked" in n for n in result.notes)
