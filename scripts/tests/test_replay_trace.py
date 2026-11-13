import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

SPEC = importlib.util.spec_from_file_location(
    "replay_trace", Path(__file__).resolve().parents[1] / "replay_trace.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["replay_trace"] = MODULE  # dataclasses look their module up here
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def report(trace: list[dict[str, Any]], fence: str = "```") -> str:
    data = {
        "format": 1,
        "messages": [
            {"role": "user", "text": "q", "trace": []},
            {"role": "assistant", "text": "a", "trace": trace},
        ],
        "last_error": None,
    }
    body = f"# report\n\n{MODULE.JSON_START}\n{fence}json\n{json.dumps(data)}\n{fence}\n{MODULE.JSON_END}\n"
    return body


def step(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    out = execute(name, arguments, ctx)
    return {"name": name, "arguments": arguments, "for_model": out.for_model, "status": out.status}


def test_report_of_a_real_call_replays_without_differences(ctx: ToolContext) -> None:
    trace = [step(ctx, "simplify", {"latex": r"\frac{x^2-1}{x-1}"})]
    items = MODULE.replay(MODULE.extract_data(report(trace)), ctx)
    assert len(items) == 1 and items[0].turn == 2 and items[0].same
    assert "DIFFERENT" not in MODULE.render(items)


def test_changed_result_is_reported(ctx: ToolContext) -> None:
    recorded = step(ctx, "simplify", {"latex": r"\frac{x^2-1}{x-1}"})
    recorded["for_model"] = "status=ok; result=$something old$"
    items = MODULE.replay(MODULE.extract_data(report([recorded])), ctx)
    assert not items[0].same
    assert "DIFFERENT" in MODULE.render(items) and "now:" in MODULE.render(items)


def test_bare_function_names_are_visible_in_the_replay(ctx: ToolContext) -> None:
    """The case that motivated the report: `cosx` without a backslash is read as letters."""
    trace = [step(ctx, "simplify", {"latex": "e^{cosx}"})]
    items = MODULE.replay(MODULE.extract_data(report(trace)), ctx)
    assert items[0].same
    assert r"\cos" not in items[0].recorded  # whatever the engine did, the report shows it


def test_longer_fence_and_failed_turn_trace(ctx: ToolContext) -> None:
    data = MODULE.extract_data(report([], fence="````"))
    data["last_error"] = {"trace": [step(ctx, "get_entry", {"id": "nope"})]}
    items = MODULE.replay(data, ctx)
    assert [i.name for i in items] == ["get_entry"] and items[0].recorded_status == "error"


@pytest.mark.parametrize(
    "text",
    ["nothing here", f"{MODULE.JSON_START}\nnot fenced\n{MODULE.JSON_END}"],
)
def test_malformed_reports_are_rejected(text: str) -> None:
    with pytest.raises(MODULE.ReportError):
        MODULE.extract_data(text)


def test_unknown_format_is_rejected() -> None:
    text = f'{MODULE.JSON_START}\n```json\n{{"format": 99}}\n```\n{MODULE.JSON_END}'
    with pytest.raises(MODULE.ReportError, match="format"):
        MODULE.extract_data(text)


def test_controlled_step_calls_replay_to_the_same_text(ctx: ToolContext) -> None:
    """The three step tools are deterministic, so a report of a turn using them replays exactly."""
    trace = [
        step(ctx, "substitute", {"latex": "x^2 + y^2", "relations": ["y^2 = 1 - x^2"]}),
        step(ctx, "isolate_variable", {"equation": r"a x + b = c", "variable": "x"}),
        step(
            ctx,
            "rewrite_part",
            {"latex": r"(x+1)^2 (y+1)^2", "action": "expand", "part": r"(x+1)^2"},
        ),
        step(ctx, "isolate_variable", {"equation": r"\sin x = a", "variable": "x"}),
    ]
    items = MODULE.replay(MODULE.extract_data(report(trace)), ctx)
    assert len(items) == 4 and all(i.same for i in items), MODULE.render(items)


# ------------------------------------------------------------------------------------ format 2
import asyncio  # noqa: E402

from minerva_api.agent.llm import LLMError  # noqa: E402
from minerva_api.agent.loop import AgentLimits, run_agent  # noqa: E402
from minerva_api.agent.replay import ReplayLLM, call, say  # noqa: E402

SIMPLIFY = {"latex": r"\frac{x^2-1}{x-1}"}


def recorded_turn(ctx: ToolContext, llm: ReplayLLM, text: str = "q") -> dict[str, Any]:
    """An assistant turn as the web app's report holds it (see reportData in debug-report.ts)."""

    async def go() -> dict[str, Any]:
        events = [
            e
            async for e in run_agent(
                llm, [{"role": "user", "content": text}], ctx, AgentLimits(retry_delay_seconds=0)
            )
        ]
        return next(e["data"] for e in events if e["event"] == "message")

    final = asyncio.run(go())
    texts = [b for b in final["blocks"] if b["kind"] == "text"]
    segments = [s for b in texts for s in b["segments"] if s["type"] != "text"]
    return {
        "role": "assistant",
        "text": final["text"],
        "trace": final["trace"],
        "debug": final["debug"],
        "truncated": final["truncated"],
        "stop_reason": final["stop_reason"],
        "prose": [MODULE.prose_text(b["segments"]) for b in texts],
        "formulas": [
            {"value": s["value"], "status": s["status"], "note": s.get("note")} for s in segments
        ],
    }


def report2(*turns: dict[str, Any], user: str = "q") -> str:
    data = {
        "format": 2,
        "messages": [{"role": "user", "text": user, "trace": []}, *turns],
        "last_error": None,
    }
    return f"# report\n\n{MODULE.JSON_START}\n```json\n{json.dumps(data)}\n```\n{MODULE.JSON_END}\n"


def test_both_report_formats_are_accepted(ctx: ToolContext) -> None:
    assert MODULE.SUPPORTED_FORMATS == (1, 2)
    turn = recorded_turn(ctx, ReplayLLM([call("simplify", SIMPLIFY), say("It is $x + 1$.")]))
    items = MODULE.replay(MODULE.extract_data(report2(turn)), ctx)
    assert len(items) == 1 and items[0].same and items[0].compare_status
    old = MODULE.replay(MODULE.extract_data(report([step(ctx, "simplify", SIMPLIFY)])), ctx)
    assert old[0].same and not old[0].compare_status


def test_a_changed_status_or_evidence_is_a_difference_even_if_the_text_is_unchanged(
    ctx: ToolContext,
) -> None:
    turn = recorded_turn(ctx, ReplayLLM([call("simplify", SIMPLIFY), say("ok")]))
    tampered = json.loads(json.dumps(turn))
    tampered["trace"][0]["status"] = "unchanged"
    items = MODULE.replay(MODULE.extract_data(report2(tampered)), ctx)
    assert items[0].same_text and not items[0].same_status and not items[0].same
    assert "status:    recorded unchanged, now ok" in MODULE.render(items)

    tampered = json.loads(json.dumps(turn))
    tampered["trace"][0]["evidence"]["check"] = "numeric"
    items = MODULE.replay(MODULE.extract_data(report2(tampered)), ctx)
    assert items[0].same_text and not items[0].same_evidence
    assert "evidence:  recorded" in MODULE.render(items)


def test_the_old_status_quirk_does_not_make_old_reports_differ(ctx: ToolContext) -> None:
    recorded = step(ctx, "simplify", SIMPLIFY)
    recorded["status"] = "something-the-old-server-recorded"
    assert MODULE.replay(MODULE.extract_data(report([recorded])), ctx)[0].same


def test_calls_that_never_ran_are_skipped_not_replayed(ctx: ToolContext) -> None:
    same = call("simplify", SIMPLIFY)
    turn = recorded_turn(ctx, ReplayLLM([same, same, say("ok")]))
    assert [t["repeated"] for t in turn["trace"]] == [False, True]
    items = MODULE.replay(MODULE.extract_data(report2(turn)), ctx)
    assert [i.skipped is not None for i in items] == [False, True]
    assert all(i.same for i in items)
    text = MODULE.render(items)
    assert "skipped" in text and "1 call(s) replayed, 0 different, 1 skipped." in text


# ----------------------------------------------------------------------------- whole-turn replay
def replayed(ctx: ToolContext, turn: dict[str, Any], **kw: Any) -> Any:
    return MODULE.replay_turn(MODULE.extract_data(report2(turn)), ctx, **kw)


def test_a_whole_turn_replays_to_the_same_marks_and_stop_reason(ctx: ToolContext) -> None:
    turn = recorded_turn(
        ctx,
        ReplayLLM(
            [
                call("simplify", SIMPLIFY),
                say(r"The result is $x + 1$, and $(x+1)^2 = x^2 + 1$ is not an identity."),
            ]
        ),
    )
    assert turn["formulas"][0]["status"] == "from_tool"
    outcome = replayed(ctx, turn)
    assert outcome.same, outcome.differences
    assert outcome.message["stop_reason"] == "completed"
    assert "whole-turn replay: same" in MODULE.render_turn(outcome)


def test_references_and_repeats_replay_through_the_real_loop(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("integrate", {"latex": r"x \cos x", "variable": "x"}),
            call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}),
            call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}),
            call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}),
            say("It checks out."),
        ]
    )
    turn = recorded_turn(ctx, llm)
    assert turn["stop_reason"] == "repeated_calls"
    assert turn["trace"][1]["model_arguments"] == {"latex": "@r1", "variables": [{"name": "x"}]}
    outcome = replayed(ctx, turn)
    assert outcome.same, outcome.differences
    assert (
        outcome.message["trace"][1]["arguments"]["latex"] == turn["trace"][1]["arguments"]["latex"]
    )


def test_a_failed_model_call_replays_as_a_failure(ctx: ToolContext) -> None:
    llm = ReplayLLM([call("simplify", SIMPLIFY)])
    llm.fail_on = {1: LLMError("The model host is not reachable."), 2: LLMError("again")}
    turn = recorded_turn(ctx, llm)
    assert turn["stop_reason"] == "model_error"
    outcome = replayed(ctx, turn)
    assert outcome.same, outcome.differences


def test_a_changed_outcome_is_reported_with_what_changed(ctx: ToolContext) -> None:
    turn = recorded_turn(ctx, ReplayLLM([call("simplify", SIMPLIFY), say(r"It is $x + 1$.")]))
    turn["formulas"][0]["status"] = "checked"  # as if an older build had marked it differently
    outcome = replayed(ctx, turn)
    assert not outcome.same
    assert any("formula marks" in d for d in outcome.differences)
    turn["stop_reason"] = "step_limit"
    assert any("stop reason" in d for d in replayed(ctx, turn).differences)


def test_replaying_a_turn_needs_events_and_a_real_assistant_turn(ctx: ToolContext) -> None:
    turn = recorded_turn(ctx, ReplayLLM([say("hi")]))
    no_events = json.loads(json.dumps(turn))
    no_events["debug"]["events"] = []
    with pytest.raises(MODULE.ReportError, match="no assistant turn with trace events"):
        MODULE.replay_turn(MODULE.extract_data(report2(no_events)), ctx)
    with pytest.raises(MODULE.ReportError, match="no trace events"):
        MODULE.replay_turn(MODULE.extract_data(report2(no_events)), ctx, turn=2)
    with pytest.raises(MODULE.ReportError, match="not an assistant turn"):
        MODULE.replay_turn(MODULE.extract_data(report2(turn)), ctx, turn=1)
    with pytest.raises(MODULE.ReportError, match="not an assistant turn"):
        MODULE.replay_turn(MODULE.extract_data(report2(turn)), ctx, turn=9)


def test_a_timeout_is_reported_as_not_replayable(ctx: ToolContext) -> None:
    turn = recorded_turn(ctx, ReplayLLM([say("hi")]))
    for event in turn["debug"]["events"]:
        if event["event"] == "llm.response":
            event["finish_reason"] = "timeout"
    outcome = replayed(ctx, turn)
    assert any("cannot be replayed" in n for n in outcome.notes)


def test_command_line_exit_codes(
    ctx: ToolContext, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    turn = recorded_turn(ctx, ReplayLLM([call("simplify", SIMPLIFY), say("It is $x + 1$.")]))
    good = tmp_path / "good.md"
    good.write_text(report2(turn))
    assert MODULE.main([str(good)]) == 0
    assert MODULE.main([str(good), "--llm"]) == 0
    assert "whole-turn replay: same" in capsys.readouterr().out

    turn["stop_reason"] = "time_limit"
    bad = tmp_path / "bad.md"
    bad.write_text(report2(turn))
    assert MODULE.main([str(bad), "--llm"]) == 1
    assert "difference: stop reason" in capsys.readouterr().out

    junk = tmp_path / "junk.md"
    junk.write_text("nothing")
    assert MODULE.main([str(junk)]) == 2
    assert MODULE.main([str(good), "--llm", "--turn", "1"]) == 2


GEOMETRY_ARGS = {
    "coordinates": ["theta", "phi"],
    "metric": [["a^2", "0"], ["0", "a^2 \\sin^2\\theta"]],
    "assumptions": {"a": ["positive"]},
    "quantities": ["christoffel", "ricci_scalar"],
}


def test_a_geometry_call_replays_exactly_from_its_arguments(ctx: ToolContext) -> None:
    # The geometry travels in the arguments (nothing is stored server-side), and the text the model read
    # has no timings in it, so a recorded call reproduces byte for byte.
    trace = [step(ctx, "geometry", GEOMETRY_ARGS)]
    assert "status=ok" in trace[0]["for_model"] and "ms" not in trace[0]["for_model"]
    items = MODULE.replay(MODULE.extract_data(report(trace)), ctx)
    assert len(items) == 1 and items[0].name == "geometry" and items[0].same


def test_a_geometry_call_whose_result_changed_is_reported(ctx: ToolContext) -> None:
    recorded = step(ctx, "geometry", GEOMETRY_ARGS)
    recorded["for_model"] = recorded["for_model"].replace("2}{a^{2}}", "3}{a^{2}}")
    items = MODULE.replay(MODULE.extract_data(report([recorded])), ctx)
    assert not items[0].same and "DIFFERENT" in MODULE.render(items)
