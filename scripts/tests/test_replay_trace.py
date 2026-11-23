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
    assert MODULE.SUPPORTED_FORMATS == (1, 2, 3, 4)
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


ODE_CALLS = [
    ("ode_analyze", {"equations": [r"y'' + 0.2 y' + y + y^3 = 0"], "independent": "t"}),
    (
        "ode_solve_symbolic",
        {
            "equations": ["y'' + 3y' + 2y = 0"],
            "independent": "t",
            "conditions": ["y(0) = 1", "y'(0) = 0"],
        },
    ),
    (
        "ode_solve_numerical",
        {
            "equations": [r"y'' + c y' + y = 0"],
            "independent": "t",
            "conditions": ["y(0) = 1", "y'(0) = 0"],
            "parameters": {"c": 0.5},
            "t_end": 10,
        },
    ),
]


@pytest.mark.parametrize(("name", "arguments"), ODE_CALLS, ids=[c[0] for c in ODE_CALLS])
def test_a_differential_equation_call_replays_exactly_from_its_arguments(
    ctx: ToolContext, name: str, arguments: dict[str, Any]
) -> None:
    # Every number travels in the arguments and the text the model read has no timings in it, so a
    # recorded call reproduces byte for byte (sampling is seeded, the integrators are deterministic).
    recorded = step(ctx, name, arguments)
    assert "status=ok" in recorded["for_model"]
    items = MODULE.replay(MODULE.extract_data(report([recorded])), ctx)
    assert len(items) == 1 and items[0].name == name and items[0].same


def test_a_differential_equation_call_whose_result_changed_is_reported(ctx: ToolContext) -> None:
    recorded = step(ctx, *ODE_CALLS[2])
    recorded["for_model"] = recorded["for_model"].replace(
        "independent method: passed", "independent method: failed"
    )
    items = MODULE.replay(MODULE.extract_data(report([recorded])), ctx)
    assert not items[0].same and "DIFFERENT" in MODULE.render(items)


def test_a_turn_becomes_a_fixture_without_prose() -> None:
    data = {
        "format": 3,
        "messages": [
            {"role": "user", "text": "q"},
            {
                "role": "assistant",
                "text": "PRIVATE PROSE",
                "debug": {
                    "prompt_sha": "abc",
                    "stop_reason": "completed",
                    "diagnosis": {"stage": "none"},
                },
                "trace": [
                    {
                        "name": "simplify",
                        "arguments": {"latex": "x+x"},
                        "status": "ok",
                        "for_model": "PRIVATE RESULT",
                        "evidence": {
                            "check": "symbolic_zero",
                            "accepted": True,
                            "reason": "verified",
                            "points_checked": 9,
                        },
                    },
                    {"name": "simplify", "arguments": {}, "status": "skipped", "for_model": "x"},
                ],
            },
        ],
    }
    fixture = MODULE.to_fixture(data)
    assert fixture["contains_user_mathematics"] is True  # until a developer minimises it
    assert fixture["calls"] == [
        {
            "name": "simplify",
            "arguments": {"latex": "x+x"},
            "expect": {
                "status": "ok",
                "evidence": {"check": "symbolic_zero", "accepted": True, "reason": "verified"},
            },
        }
    ]
    assert "PRIVATE" not in json.dumps(fixture)
    assert fixture["provenance"]["report_format"] == 3


def test_a_format_3_report_is_read() -> None:
    assert 3 in MODULE.SUPPORTED_FORMATS


def test_a_different_argument_or_verdict_is_a_difference_even_when_names_agree() -> None:
    recorded = {
        "trace": [
            {
                "name": "simplify",
                "status": "ok",
                "arguments": {"latex": "x"},
                "evidence": {"check": "symbolic_zero", "accepted": True, "reason": "verified"},
            }
        ],
        "stop_reason": "completed",
    }
    now = {
        "trace": [
            {
                "name": "simplify",
                "status": "ok",
                "arguments": {"latex": "y"},
                "evidence": {"check": "numeric", "accepted": True, "reason": "verified"},
            }
        ],
        "stop_reason": "completed",
    }
    found = MODULE.compare_turn(recorded, now)
    assert any("arguments executed differ" in d for d in found)
    assert any("evidence recorded" in d for d in found)
    assert MODULE.compare_turn(recorded, recorded) == []


# ------------------------------------------------------------- follow-up turns keep their stored context (#150)
from dataclasses import replace  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from minerva_api.agent.events import AssistantMessage, Block, Card, TraceStep  # noqa: E402
from minerva_api.agent.turn_context import turn_context  # noqa: E402

POLAR = r"\frac{\mu}{2}(\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{G M \mu}{r}"
C1 = r"\mu r^{2} \dot{\theta}"
C2 = r"- \frac{G M \mu}{r} + \frac{\mu r^{2} \dot{\theta}^{2}}{2} + \frac{\mu \dot{r}^{2}}{2}"
RELATION = r"C_{1}^{2} + C_{2}"
TIME_DERIVATIVE = {"latex": RELATION, "coordinates": ["r", "theta"]}


def discovery_message(
    expressions: dict[str, str] | None = None,
    verified: bool = True,
    card_id: str = "d1",
    first: int = 1,
) -> SimpleNamespace:
    """A stored assistant message holding a discovery card with named constants, numbered from `first`."""
    items = expressions or {"C_{1}": C1, "C_{2}": C2}
    step = TraceStep(
        id=card_id,
        name="find_conserved_quantities",
        arguments={"lagrangian": POLAR, "coordinates": ["r", "theta"]},
        for_model="found constants of motion",
        status="ok",
        call_id=f"call_{card_id}",
    )
    card = Card(
        id=card_id,
        tool="find_conserved_quantities",
        type="discovery",
        title="Constants of motion",
        data={
            "report": {
                "system": "lagrangian",
                "function_latex": POLAR,
                "coordinates": ["r", "theta"],
                "constants": [
                    {"expression_latex": f"{name} = {text}", "verified": verified}
                    for name, text in items.items()
                ],
            }
        },
        numbers={f"const:{i}": first + i for i in range(len(items))},
    )
    message = AssistantMessage(
        text="Found the constants.", blocks=[Block(kind="card", card=card)], trace=[step]
    )
    return SimpleNamespace(role="assistant", content=message.model_dump(mode="json"))


def user_message(text: str, counter: int = 0, previous: list[Any] | None = None) -> SimpleNamespace:
    """A stored user message with the numbers and references the chat recorded on it."""
    context = turn_context(previous or [], text, counter=counter)
    content: dict[str, Any] = {"text": text}
    if context.user_equations:
        content["equations"] = context.user_equations
    if context.shown_references:
        content["references"] = context.shown_references
    return SimpleNamespace(role="user", content=content)


def live_turn(
    ctx: ToolContext, previous: list[Any], text: str, llm: ReplayLLM, counter: int = 0
) -> SimpleNamespace:
    """The assistant message the chat route would store for `text`: the real loop on the route's context."""
    context = turn_context(previous, text, counter=counter)
    turn_ctx = replace(ctx)
    context.apply_to(turn_ctx)

    async def go() -> dict[str, Any]:
        events = [
            e
            async for e in run_agent(
                llm,
                context.history,
                turn_ctx,
                AgentLimits(retry_delay_seconds=0),
                **context.agent_kwargs(),
            )
        ]
        return next(e["data"] for e in events if e["event"] == "message")

    return SimpleNamespace(role="assistant", content=asyncio.run(go()))


def report_of(
    stored: list[Any],
    fmt: int = 4,
    counter: int = 0,
    window: dict[str, Any] | None = None,
) -> str:
    """A debug report of stored messages as the web app writes it (`reportData` in debug-report.ts).

    Format 3 is the same without cards, numbered user formulas, references and the `conversation` block.
    """
    messages: list[dict[str, Any]] = []
    for m in stored:
        c = m.content
        row: dict[str, Any] = {"role": m.role, "text": c["text"], "trace": c.get("trace", [])}
        if m.role == "assistant":
            texts = [b for b in c["blocks"] if b["kind"] == "text"]
            segments = [s for b in texts for s in b["segments"] if s["type"] != "text"]
            row.update(
                debug=c.get("debug"),
                truncated=c.get("truncated", False),
                stop_reason=c.get("stop_reason"),
                prose=[MODULE.prose_text(b["segments"]) for b in texts],
                formulas=[
                    {"value": s["value"], "status": s["status"], "note": s.get("note")}
                    for s in segments
                ],
            )
            if fmt >= 4:
                row["blocks"] = c["blocks"]
        elif fmt >= 4:
            row.update(equations=c.get("equations", []), references=c.get("references", []))
        messages.append(row)
    data: dict[str, Any] = {"format": fmt, "messages": messages, "last_error": None}
    if fmt >= 4:
        shown = turn_context(stored, "", counter=0).equations.counter
        data["conversation"] = {
            "equation_counter": max(
                counter, shown
            ),  # as the web app writes it: never below what is shown
            "equation_counter_reported": counter,
            "messages_total": len(stored),
            "first_message_index": 0,
            "messages_included": len(stored),
            **(window or {}),
        }
    return f"# report\n\n{MODULE.JSON_START}\n```json\n{json.dumps(data)}\n```\n{MODULE.JSON_END}\n"


def conversation(
    ctx: ToolContext, opening: list[Any], text: str, llm: ReplayLLM
) -> tuple[list[Any], SimpleNamespace]:
    """The stored conversation after one more exchange, and the new reply."""
    asked = user_message(text, previous=opening)
    answer = live_turn(ctx, opening, text, llm)
    return [*opening, asked, answer], answer


def constants_conversation(ctx: ToolContext, *earlier: SimpleNamespace) -> list[Any]:
    opening = [user_message("Find the constants of motion."), *earlier]
    stored, _ = conversation(
        ctx,
        opening,
        r"Differentiate $C_{1}^{2} + C_{2}$ with respect to time.",
        ReplayLLM([call("time_derivative", TIME_DERIVATIVE), say("It is zero on the motion.")]),
    )
    return stored


def test_a_follow_up_that_needs_verified_constants_replays_the_same_with_the_cards(
    ctx: ToolContext,
) -> None:
    stored = constants_conversation(ctx, discovery_message())
    recorded = stored[-1].content["trace"][0]
    assert (
        recorded["status"] == "ok"
    )  # the live chat expanded the names from the stored discovery card
    assert "constants replaced by their verified expressions" in recorded["for_model"]
    outcome = MODULE.replay_turn(MODULE.extract_data(report_of(stored, counter=2)), ctx)
    assert outcome.faithful and outcome.same, MODULE.render_turn(outcome)
    assert outcome.message is not None and outcome.message["trace"][0]["status"] == "ok"
    assert "whole-turn replay: same" in MODULE.render_turn(outcome)


def test_without_the_cards_the_same_follow_up_is_not_faithful_and_says_what_is_missing(
    ctx: ToolContext,
) -> None:
    stored = constants_conversation(ctx, discovery_message())
    for fmt in (4, 3):
        data = MODULE.extract_data(report_of(stored, fmt=fmt, counter=2))
        if fmt == 4:  # a format 4 report whose cards were left out
            for m in data["messages"]:
                m["blocks"] = []
        outcome = MODULE.replay_turn(data, ctx)
        assert not outcome.faithful and not outcome.same
        assert "constants" in outcome.missing
        # The follow-up now fails on the bare names, which is the missing context, not a verdict: kept apart.
        assert outcome.message is not None and outcome.message["trace"][0]["status"] != "ok"
        assert outcome.differences == [] and outcome.context_differences
        text = MODULE.render_turn(outcome)
        assert "NOT COMPARABLE" in text and "missing constants" in text
        assert "cannot be faithfully replayed" in text and "whole-turn replay: same" not in text


def test_an_ambiguous_definition_is_refused_in_the_replay_exactly_as_live(ctx: ToolContext) -> None:
    changed = {"C_{1}": r"\mu r \dot{\theta}", "C_{2}": C2}
    stored = constants_conversation(
        ctx, discovery_message(), discovery_message(changed, card_id="d2", first=3)
    )
    recorded = stored[-1].content["trace"][0]
    assert recorded["status"] != "ok" and "more than one definition" in recorded["for_model"]
    outcome = MODULE.replay_turn(MODULE.extract_data(report_of(stored, counter=4)), ctx)
    assert outcome.same, MODULE.render_turn(outcome)
    assert outcome.message is not None
    assert outcome.message["trace"][0]["for_model"] == recorded["for_model"]


def test_an_unverified_definition_is_never_inserted_in_the_replay(ctx: ToolContext) -> None:
    stored = constants_conversation(ctx, discovery_message(verified=False))
    recorded = stored[-1].content["trace"][0]
    assert recorded["status"] != "ok" and "not verified" in recorded["for_model"]
    outcome = MODULE.replay_turn(MODULE.extract_data(report_of(stored, counter=2)), ctx)
    assert outcome.same, MODULE.render_turn(outcome)
    assert outcome.message is not None
    assert outcome.message["trace"][0]["for_model"] == recorded["for_model"]


def test_a_reused_result_keeps_its_restrictions_in_the_replay(ctx: ToolContext) -> None:
    opening, _ = conversation(
        ctx,
        [],
        r"Simplify $\frac{x^3-y^3}{x-y}$.",
        ReplayLLM([call("simplify", {"latex": r"\frac{x^3-y^3}{x-y}"}), say("Done.")]),
    )
    stored, answer = conversation(
        ctx,
        opening,
        "Differentiate that result with respect to x.",
        ReplayLLM(
            [call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}), say("Done.")]
        ),
    )
    live_card = next(b["card"] for b in answer.content["blocks"] if b["kind"] == "card")
    assert live_card["data"]["inherited_restrictions_latex"], (
        "the live chat carried the restriction"
    )

    outcome = MODULE.replay_turn(MODULE.extract_data(report_of(stored, counter=3)), ctx)
    assert outcome.same, MODULE.render_turn(outcome)
    assert outcome.message is not None
    replayed_card = next(b["card"] for b in outcome.message["blocks"] if b["kind"] == "card")
    assert (
        replayed_card["data"]["inherited_restrictions_latex"]
        == live_card["data"]["inherited_restrictions_latex"]
    )

    # Format 3 holds the trace but no cards: `@r1`, its LaTeX and its restrictions cannot be rebuilt.
    old = MODULE.replay_turn(MODULE.extract_data(report_of(stored, fmt=3)), ctx)
    assert not old.faithful
    assert {"result handles", "restrictions", "equation table"} <= set(old.missing)
    assert "constants" not in old.missing
    assert old.differences == [] and old.context_differences, MODULE.render_turn(old)


def test_prior_formula_evidence_marks_the_follow_up_as_the_live_chat_did(ctx: ToolContext) -> None:
    lagrangian = (
        r"\frac12 m R^2 (\dot{\theta}^2 + \sin^2(\theta) \dot{\phi}^2) - m g R \cos(\theta)"
    )
    opening, _ = conversation(
        ctx,
        [],
        f"Find the equations of motion for $L = {lagrangian}$",
        ReplayLLM(
            [
                call(
                    "analyze_lagrangian",
                    {"lagrangian": lagrangian, "coordinates": ["theta", "phi"]},
                ),
                say("The equations of motion are on the card."),
            ]
        ),
    )
    answer = r"Then $$\ddot\phi = -2\cot\theta\,\dot\theta\dot\phi$$ holds."
    stored, reply = conversation(
        ctx, opening, "Say the phi equation again.", ReplayLLM([say(answer)])
    )
    live = [
        s
        for b in reply.content["blocks"]
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] == "display_math"
    ]
    assert [s["status"] for s in live] == ["from_tool"], (
        "linked to the earlier turn's verified result"
    )
    outcome = MODULE.replay_turn(MODULE.extract_data(report_of(stored, counter=9)), ctx)
    assert outcome.same, MODULE.render_turn(outcome)
    # The evidence is read from the stored trace, so it survives in formats 2 and 3 too ...
    old = MODULE.replay_turn(MODULE.extract_data(report_of(stored, fmt=3)), ctx)
    assert not any("formula marks" in d for d in old.differences + old.context_differences)
    # ... and the replay really depends on it: without the earlier turn's trace the reply is unchecked.
    stripped = MODULE.extract_data(report_of(stored, counter=9))
    stripped["messages"][1]["trace"] = []
    changed = MODULE.replay_turn(stripped, ctx)
    assert any("formula marks" in d for d in changed.differences + changed.context_differences)


def test_a_first_assistant_turn_misses_nothing_whatever_its_message_index(ctx: ToolContext) -> None:
    plain = [user_message("a note"), live_turn(ctx, [], "a note", ReplayLLM([say("ok")]))]
    for earlier in ([], plain):
        stored, _ = conversation(
            ctx, earlier, "q", ReplayLLM([call("simplify", SIMPLIFY), say("ok")])
        )
        for fmt in (3, 4):
            outcome = MODULE.replay_turn(MODULE.extract_data(report_of(stored, fmt=fmt)), ctx)
            assert outcome.missing == {} and outcome.same, MODULE.render_turn(outcome)
            assert not any("plain text" in n for n in outcome.notes)


def test_a_report_of_one_turn_says_the_earlier_messages_are_missing(ctx: ToolContext) -> None:
    stored = constants_conversation(ctx, discovery_message())
    window = {"messages_total": len(stored), "first_message_index": len(stored) - 2}
    report = report_of(stored[-2:], counter=2, window=window)
    outcome = MODULE.replay_turn(MODULE.extract_data(report), ctx)
    assert "earlier messages" in outcome.missing and not outcome.faithful
    assert "starts at message 3 of 4" in outcome.missing["earlier messages"]


def test_numbering_keeps_the_high_water_mark_and_never_recreates_a_vanished_reference(
    ctx: ToolContext,
) -> None:
    # Equations (1) and (2) are stored. (3) and (4) were given out and their message was replaced, so the
    # mark is 4 and the user's new formula was numbered (5), not (3).
    opening = [user_message("Find the constants of motion."), discovery_message()]
    text = "Now use $$w = x + 1$$ please."
    asked = user_message(text, counter=4, previous=opening)
    assert asked.content["equations"] == [{"number": 5, "latex": "w = x + 1"}]
    stored = [*opening, asked, live_turn(ctx, opening, text, ReplayLLM([say("Noted.")]), counter=4)]
    outcome = MODULE.replay_turn(MODULE.extract_data(report_of(stored, counter=5)), ctx)
    assert outcome.missing == {} and outcome.same, MODULE.render_turn(outcome)
    assert outcome.counter == 4  # not the floor (2): a vanished number is never handed out again

    # A reference to a vanished number (3): the chat told the model it no longer exists; the user's own new
    # formula pins the mark, so the replay says the same. It is never given an equation to point at.
    text2 = "What was equation 3? Take $$w = x + 1$$."
    asked2 = user_message(text2, counter=4, previous=opening)
    assert asked2.content["references"] == [{"number": 3, "latex": None, "source": "unknown"}]
    stored2 = [
        *opening,
        asked2,
        live_turn(ctx, opening, text2, ReplayLLM([say("Gone.")]), counter=4),
    ]
    outcome2 = MODULE.replay_turn(MODULE.extract_data(report_of(stored2, counter=5)), ctx)
    assert outcome2.missing == {} and outcome2.same and outcome2.counter == 4

    # Without a formula to pin the mark the report cannot say whether (3) had vanished: it says so.
    text3 = "What was equation 3?"
    asked3 = user_message(text3, counter=4, previous=opening)
    stored3 = [
        *opening,
        asked3,
        live_turn(ctx, opening, text3, ReplayLLM([say("Gone.")]), counter=4),
    ]
    outcome3 = MODULE.replay_turn(MODULE.extract_data(report_of(stored3, counter=5)), ctx)
    assert "equation table" in outcome3.missing and not outcome3.faithful

    # A mark that cannot produce the numbers the chat recorded is reported, never silently approximated.
    data4 = MODULE.extract_data(report_of(stored, counter=5))
    data4["conversation"]["equation_counter"] = 2
    outcome4 = MODULE.replay_turn(data4, ctx)
    assert outcome4.counter == 2 and "equation table" in outcome4.missing
    assert not outcome4.faithful and not outcome4.same


def test_command_line_exit_codes_for_a_replay_that_cannot_be_faithful(
    ctx: ToolContext, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stored = constants_conversation(ctx, discovery_message())
    old = tmp_path / "old.md"
    old.write_text(report_of(stored, fmt=3))
    assert (
        MODULE.main([str(old), "--llm"]) == 3
    )  # nothing the engine did differently, and not a pass
    out = capsys.readouterr().out
    assert "NOT COMPARABLE" in out and "whole-turn replay: same" not in out
    new = tmp_path / "new.md"
    new.write_text(report_of(stored, counter=2))
    assert MODULE.main([str(new), "--llm"]) == 0
    assert "whole-turn replay: same" in capsys.readouterr().out
