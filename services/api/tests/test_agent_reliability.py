"""Reliability of the agent loop: stopping, repeated calls, retries, malformed output, references, evidence.

A `ReplayLLM` plays the model; the tools, the engine and the claim checker are the real ones.
"""

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from minerva_api.agent import loop as loop_module
from minerva_api.agent.llm import Chunk, LLMError, ToolCall, Usage
from minerva_api.agent.loop import (
    REPEAT_MESSAGE,
    AgentLimits,
    looks_like_tool_call,
    run_agent,
)
from minerva_api.agent.refs import ResultStore
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.segments import normalize
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

SIMPLIFY = {"latex": r"\frac{x^2-1}{x-1}"}
FAST = AgentLimits(retry_delay_seconds=0)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def play(
    llm: Any, ctx: ToolContext, text: str = "q", limits: AgentLimits = FAST, **kw: Any
) -> list[dict[str, Any]]:
    async def go() -> list[dict[str, Any]]:
        return [
            e async for e in run_agent(llm, [{"role": "user", "content": text}], ctx, limits, **kw)
        ]

    return asyncio.run(go())


def final(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(e["data"] for e in events if e["event"] == "message")


def cards(msg: dict[str, Any]) -> list[dict[str, Any]]:
    return [b["card"] for b in msg["blocks"] if b["kind"] == "card"]


def segments(msg: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for b in msg["blocks"] if b["kind"] == "text" for s in b["segments"]]


def event_names(msg: dict[str, Any]) -> list[str]:
    return [e["event"] for e in msg["debug"]["events"]]


# ------------------------------------------------------------------------------- stop reasons
def test_a_finished_answer_is_completed(ctx: ToolContext) -> None:
    msg = final(play(ReplayLLM([call("simplify", SIMPLIFY), say("It is $x + 1$.")]), ctx))
    assert msg["stop_reason"] == "completed" and msg["truncated"] is False
    assert msg["debug"]["stop_reason"] == "completed" and msg["debug"]["error"] is None
    assert "stopped here" not in msg["text"]


def test_the_closing_sentence_names_the_limit_and_is_shown_to_the_user(ctx: ToolContext) -> None:
    msg = final(play(ReplayLLM([call("simplify", SIMPLIFY)]), ctx, limits=AgentLimits(max_steps=1)))
    assert msg["stop_reason"] == "step_limit"
    shown = " ".join(
        s["value"] for b in msg["blocks"] if b["kind"] == "text" for s in b["segments"]
    )
    assert "step limit" in msg["text"] and "step limit" in shown  # in the blocks, not only the text


# --------------------------------------------------------------------------- repeated calls
def test_an_identical_call_is_not_run_again(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[str] = []
    real = loop_module.execute

    def counting(name: str, arguments: dict[str, Any], c: ToolContext) -> Any:
        runs.append(name)
        return real(name, arguments, c)

    monkeypatch.setattr(loop_module, "execute", counting)
    llm = ReplayLLM([call("simplify", SIMPLIFY), call("simplify", SIMPLIFY), say("It is $x + 1$.")])
    msg = final(play(llm, ctx))
    assert runs == ["simplify"] and len(cards(msg)) == 1
    first, second = msg["trace"]
    assert not first["repeated"] and second["repeated"] is True and second["status"] == "skipped"
    assert second["error_category"] == "repeated_call"
    told = [m["content"] for m in llm.seen[-1] if m["role"] == "tool"]
    assert told[-1] == REPEAT_MESSAGE
    assert msg["stop_reason"] == "completed"  # one repeat is tolerated


def test_the_same_call_with_different_arguments_is_not_a_repeat(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("simplify", SIMPLIFY),
            call("simplify", {"latex": r"\sin^2 x + \cos^2 x"}),
            say("Done."),
        ]
    )
    msg = final(play(llm, ctx))
    assert len(cards(msg)) == 2 and not any(t["repeated"] for t in msg["trace"])


def test_a_model_that_keeps_repeating_is_stopped_and_asked_to_summarise(ctx: ToolContext) -> None:
    replies = [
        call("simplify", SIMPLIFY),
        call("simplify", SIMPLIFY),
        call("simplify", SIMPLIFY),
        say("Verified: the result is $x + 1$. Nothing else could be established."),
    ]
    llm = ReplayLLM(replies)
    msg = final(play(llm, ctx))
    assert msg["stop_reason"] == "repeated_calls" and msg["truncated"] is True
    assert len(llm.seen) == 4 and llm.tools_seen[-1] == []  # the last call offers no tools
    assert "Stop calling tools" in llm.seen[-1][-1]["content"]
    assert "Nothing else could be established" in msg["text"]
    assert "not producing new results" in msg["text"]
    assert msg["debug"]["events"][-1]["repeated_calls"] == 2


def test_calls_that_establish_nothing_end_the_turn(ctx: ToolContext) -> None:
    # Three different calls in a row, none of which produced a verified result.
    bad = [call("simplify", {"latex": rf"\foo{{{n}}}"}) for n in "abc"]
    llm = ReplayLLM([*bad, say("I could not read these.")])
    msg = final(play(llm, ctx))
    assert msg["stop_reason"] == "repeated_calls"
    assert [t["status"] for t in msg["trace"]] == ["parse_error"] * 3
    assert [t["error_category"] for t in msg["trace"]] == ["parse_error"] * 3


def test_a_verified_result_resets_the_streak(ctx: ToolContext) -> None:
    replies = [
        call("simplify", {"latex": r"\foo{a}"}),
        call("simplify", {"latex": r"\foo{b}"}),
        call("simplify", SIMPLIFY),
        call("simplify", {"latex": r"\foo{c}"}),
        say("Done."),
    ]
    msg = final(play(ReplayLLM(replies), ctx))
    assert msg["stop_reason"] == "completed"


# ---------------------------------------------------------------------------------- retries
def test_a_model_call_that_fails_outright_is_retried_once(ctx: ToolContext) -> None:
    llm = ReplayLLM([[], say("Hello.")])
    llm.fail_on = {0: LLMError("The model host is not reachable.")}
    events = play(llm, ctx)
    msg = final(events)
    assert [e["event"] for e in events if e["event"] != "text"] == [
        "message"
    ]  # no error: it recovered
    assert msg["text"] == "Hello." and msg["stop_reason"] == "completed"
    retry = [e for e in msg["debug"]["events"] if e["event"] == "agent.retry"]
    assert len(retry) == 1 and retry[0]["reason"] == "model_host" and retry[0]["attempt"] == 1
    assert [c["finish_reason"] for c in msg["debug"]["llm_calls"]] == ["error", "stop"]
    assert len(llm.seen) == 2


def test_a_second_failure_ends_the_turn_and_is_not_retried_again(ctx: ToolContext) -> None:
    llm = ReplayLLM()
    llm.error = LLMError("The model host is not reachable.")
    events = play(llm, ctx)
    assert [e["event"] for e in events] == ["error", "message"]
    msg = final(events)
    assert (
        msg["stop_reason"] == "model_error"
        and msg["debug"]["error"] == "The model host is not reachable."
    )
    assert len(llm.seen) == 2
    assert "stopped answering" in msg["text"]


def test_a_failure_after_work_keeps_the_work(ctx: ToolContext) -> None:
    llm = ReplayLLM([call("simplify", SIMPLIFY)])
    llm.fail_on = {1: LLMError("The model host answered with status 500."), 2: LLMError("again")}
    events = play(llm, ctx)
    kinds = [e["event"] for e in events]
    assert kinds[-2:] == ["error", "message"]
    msg = final(events)
    assert len(cards(msg)) == 1 and len(msg["trace"]) == 1  # the tool result survives the failure
    assert msg["stop_reason"] == "model_error"
    assert len(llm.seen) == 3  # the model call was tried twice; the tool was not run again


def test_a_retry_after_work_recovers_without_repeating_the_computation(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[str] = []
    real = loop_module.execute

    def counting(name: str, arguments: dict[str, Any], c: ToolContext) -> Any:
        runs.append(name)
        return real(name, arguments, c)

    monkeypatch.setattr(loop_module, "execute", counting)
    llm = ReplayLLM([call("simplify", SIMPLIFY), [], say("It is $x + 1$.")])
    llm.fail_on = {1: LLMError("The model host is not reachable.")}
    msg = final(play(llm, ctx))
    assert runs == ["simplify"] and msg["stop_reason"] == "completed"
    assert mark_of(msg, "x + 1") == "from_tool"


def test_text_that_arrived_before_a_failure_is_kept(ctx: ToolContext) -> None:
    class Dies:
        name = "dies"

        async def chat(
            self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
        ) -> AsyncIterator[Chunk]:
            yield Chunk(text="The first part of an answer. ")
            raise LLMError("The model host sent an unreadable reply.")

    msg = final(play(Dies(), ctx))
    assert "first part of an answer" in msg["text"] and msg["stop_reason"] == "model_error"
    assert [c["finish_reason"] for c in msg["debug"]["llm_calls"]] == ["error"]


# --------------------------------------------------------------------------- malformed output
def test_a_call_with_unusable_arguments_is_reported_to_the_model(ctx: ToolContext) -> None:
    bad = [Chunk(tool_calls=[ToolCall("simplify", {}, malformed="arguments_not_json")])]
    llm = ReplayLLM([bad, call("simplify", SIMPLIFY), say("It is $x + 1$.")])
    msg = final(play(llm, ctx))
    first = msg["trace"][0]
    assert first["status"] == "error" and first["error_category"] == "malformed_model_output"
    assert first["arguments_valid"] is False and "not valid JSON" in first["for_model"]
    assert msg["stop_reason"] == "completed" and len(msg["trace"]) == 2
    retry = [e for e in msg["debug"]["events"] if e["event"] == "agent.retry"]
    assert [r["reason"] for r in retry] == ["malformed_model_output"]


def test_a_call_without_a_name_does_not_crash_the_turn(ctx: ToolContext) -> None:
    nameless = [Chunk(tool_calls=[ToolCall("", {}, malformed="missing_name")])]
    msg = final(play(ReplayLLM([nameless, say("Sorry.")]), ctx))
    assert msg["trace"][0]["error_category"] == "malformed_model_output"
    assert msg["trace"][0]["name"] == "unknown" or msg["trace"][0]["name"] == ""


def test_repeated_malformed_calls_end_the_turn(ctx: ToolContext) -> None:
    bad = [Chunk(tool_calls=[ToolCall("simplify", {}, malformed="arguments_not_object")])]
    msg = final(play(ReplayLLM([bad], repeat_last=True), ctx))
    assert msg["stop_reason"] == "malformed_output" and "could not be used" in msg["text"]
    assert len(msg["trace"]) == 2


def test_a_tool_call_written_as_prose_is_detected_and_not_shown(ctx: ToolContext) -> None:
    prose = say('{"name": "simplify", "arguments": {"latex": "x"}}')
    llm = ReplayLLM([prose, call("simplify", SIMPLIFY), say("It is $x + 1$.")])
    msg = final(play(llm, ctx))
    assert '"name"' not in msg["text"] and msg["stop_reason"] == "completed"
    assert any("tool call as text" in str(m["content"]) for m in llm.seen[1])
    assert [e["reason"] for e in msg["debug"]["events"] if e["event"] == "agent.retry"] == [
        "malformed_model_output"
    ]
    twice = final(play(ReplayLLM([prose], repeat_last=True), ctx))
    assert twice["stop_reason"] == "malformed_output"


def test_looks_like_tool_call() -> None:
    assert looks_like_tool_call('{"name": "simplify", "arguments": {}}')
    assert looks_like_tool_call('```json\n{"name": "integrate", "arguments": {}}\n```')
    assert looks_like_tool_call('{"function": {"name": "series"}}')
    assert not looks_like_tool_call('{"name": "simplify"} is the tool I would use.')
    assert not looks_like_tool_call('{"name": "not_a_tool", "arguments": {}}')
    assert not looks_like_tool_call("Use simplify.")
    assert not looks_like_tool_call("[1, 2]")


# ------------------------------------------------------------------------ result references
def test_a_reference_is_replaced_by_the_engines_own_formula(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("integrate", {"latex": r"x \cos x", "variable": "x"}),
            call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}),
            say("Done."),
        ]
    )
    msg = final(play(llm, ctx))
    first, second = msg["trace"]
    assert first["result_id"] == "r1" and "result_id=r1" in llm.seen[1][-1]["content"]
    assert "result_id" not in first["for_model"]  # the stored text stays what the tool returned
    engine_latex = cards(msg)[0]["data"]["result_latex"]
    assert second["arguments"]["latex"] == engine_latex  # exactly what the first tool printed
    assert second["model_arguments"]["latex"] == "@r1"
    assert second["status"] == "ok" and cards(msg)[1]["data"]["result_text"] == "x*cos(x)"


def test_a_reference_inside_an_expression_keeps_precedence(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}),
            call("differentiate", {"latex": "2 @r1", "variables": [{"name": "x"}]}),
            say("Done."),
        ]
    )
    msg = final(play(llm, ctx))
    assert msg["trace"][1]["arguments"]["latex"] == r"2 \left(x + 1\right)"
    assert cards(msg)[1]["data"]["result_text"] == "2"


def test_an_unknown_reference_is_an_error_naming_the_valid_ones(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("simplify", SIMPLIFY),
            call("differentiate", {"latex": "@r9", "variables": [{"name": "x"}]}),
            say("Sorry."),
        ]
    )
    msg = final(play(llm, ctx))
    bad = msg["trace"][1]
    assert bad["status"] == "error" and bad["arguments_valid"] is False
    assert "@r9" in bad["for_model"] and "@r1" in bad["for_model"]
    assert bad["error_category"] == "invalid_arguments"


def test_a_result_that_was_not_verified_has_no_id(ctx: ToolContext) -> None:
    llm = ReplayLLM([call("simplify", {"latex": r"\sqrt{x^2}"}), say("It needs an assumption.")])
    msg = final(play(llm, ctx))
    assert msg["trace"][0]["status"] == "conditions_not_satisfied"
    assert msg["trace"][0]["result_id"] is None
    store = ResultStore()
    assert store.register("simplify", execute("simplify", {"latex": r"\sqrt{x^2}"}, ctx)) is None
    assert (
        store.register("search_handbook", execute("search_handbook", {"query": "gauss"}, ctx))
        is None
    )


def test_references_continue_across_turns(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}), say("Done.")]
    )
    msg = final(play(llm, ctx, prior_results={"r1": "x^{2}"}))
    assert msg["trace"][0]["arguments"]["latex"] == "x^{2}"
    assert cards(msg)[0]["data"]["result_text"] == "2*x"
    again = ReplayLLM([call("simplify", SIMPLIFY), say("ok")])
    msg2 = final(play(again, ctx, prior_results={"r1": "x", "r4": "y"}))
    assert msg2["trace"][0]["result_id"] == "r5"  # ids continue where the conversation left off


def test_reference_substitution_is_strict() -> None:
    store = ResultStore(results={"r1": "x + 1"})
    assert store.resolve({"latex": "@r1"}).arguments["latex"] == "x + 1"
    assert (
        store.resolve({"latex": "@r1 + @r1"}).arguments["latex"]
        == r"\left(x + 1\right) + \left(x + 1\right)"
    )
    assert store.resolve({"query": "@r1"}).arguments["query"] == "@r1"  # not an expression argument
    assert store.resolve({"latex": "a@rx"}).arguments["latex"] == "a@rx"
    assert store.resolve({"latex": "@r2"}).error
    assert store.resolve({"latex": "x", "variable": "@r1"}).count == 0


# ------------------------------------------------------------------- evidence for the marks
def mark_of(msg: dict[str, Any], latex: str) -> str | None:
    key = normalize(latex)
    return next((s["status"] for s in segments(msg) if normalize(s["value"]) == key), None)


def test_a_result_is_marked_verified_but_the_users_own_input_is_only_quoted(
    ctx: ToolContext,
) -> None:
    llm = ReplayLLM(
        [
            call("simplify", SIMPLIFY),
            say(r"You gave $\frac{x^{2} - 1}{x - 1}$ and the result is $x + 1$."),
        ]
    )
    msg = final(play(llm, ctx))
    assert mark_of(msg, "x + 1") == "from_tool"
    assert mark_of(msg, r"\frac{x^{2} - 1}{x - 1}") == "quoted"
    quoted = next(s for s in segments(msg) if s["status"] == "quoted")
    assert "your input" in quoted["note"]


def test_nothing_from_a_result_that_was_not_verified_is_marked_verified(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("simplify", {"latex": r"\log x + \log y"}),
            say(
                r"The identity $\log{\left(a b \right)} = \log{\left(a \right)} + \log{\left(b \right)}$ would apply."
            ),
        ]
    )
    msg = final(play(llm, ctx))
    assert msg["trace"][0]["status"] == "conditions_not_satisfied"
    assert msg["trace"][0]["verified_latex"] == []
    assert (
        mark_of(msg, r"\log{\left(a b \right)} = \log{\left(a \right)} + \log{\left(b \right)}")
        == "quoted"
    )
    quoted = [s for s in segments(msg) if s["status"] == "quoted"]
    assert quoted and "not applied" in quoted[0]["note"]


def test_a_stated_handbook_entry_is_quoted_not_verified(ctx: ToolContext) -> None:
    statement = execute("get_entry", {"id": "const.c"}, ctx).card.data["statement_latex"]
    llm = ReplayLLM(
        [call("get_entry", {"id": "const.c"}), say(f"The handbook states ${statement}$.")]
    )
    msg = final(play(llm, ctx))
    assert mark_of(msg, statement) == "quoted"
    assert "only stated" in next(s for s in segments(msg) if s["status"] == "quoted")["note"]
    checked = execute("get_entry", {"id": "log.product"}, ctx).card.data["statement_latex"]
    msg2 = final(
        play(ReplayLLM([call("get_entry", {"id": "log.product"}), say(f"${checked}$ holds.")]), ctx)
    )
    assert mark_of(msg2, checked) == "from_tool"  # a machine-checked entry is vouched for


def test_a_matrix_equation_is_a_result_not_an_input(ctx: ToolContext) -> None:
    args = {
        "latex": r"\int_0^1 \exp(tA)\,dt",
        "matrices": [{"name": "A", "rows": 2, "cols": 2}],
        "action": "integrate",
        "variable": "t",
        "lower": "0",
        "upper": "1",
    }
    out = execute("matrix_calculus", {**args, "latex": r"\exp(tA)"}, ctx)
    equation = out.card.data["result"].get("equation_latex")
    if equation and out.evidence and out.evidence.accepted:
        assert equation in out.known_latex  # computed by the engine, so verified when the result is


def test_verified_marks_survive_into_later_turns(ctx: ToolContext) -> None:
    first = final(play(ReplayLLM([call("simplify", SIMPLIFY), say("ok")]), ctx))
    step = first["trace"][0]
    known = {normalize(x) for x in step["verified_latex"]}
    quoted = dict(step["quoted"])
    assert normalize("x + 1") in known and quoted
    msg = final(
        play(ReplayLLM([say("Earlier: $x + 1$.")]), ctx, prior_known=known, prior_quoted=quoted)
    )
    assert mark_of(msg, "x + 1") == "from_tool"


def test_a_hallucinated_formula_is_never_verified(ctx: ToolContext) -> None:
    msg = final(play(ReplayLLM([say(r"Surely $\int_0^1 x\,dx = \frac{1}{3}$.")]), ctx))
    assert all(
        s["status"] in ("unchecked", "contradicted") for s in segments(msg) if s["type"] != "text"
    )
    assert msg["debug"]["formulas"] == 1 and msg["debug"]["formulas_with_evidence"] == 0
    assert msg["debug"]["evidence_pct"] == 0.0


def test_evidence_percentage_counts_only_formulas_with_engine_evidence(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [call("simplify", SIMPLIFY), say(r"The result is $x + 1$, not $\frac{x^2}{7} + 1$.")]
    )
    d = final(play(llm, ctx))["debug"]
    assert d["formulas"] == 2 and d["formulas_with_evidence"] == 1 and d["evidence_pct"] == 50.0


def test_evidence_records_what_the_engine_established(ctx: ToolContext) -> None:
    simplify = execute("simplify", SIMPLIFY, ctx).evidence
    assert simplify and simplify.accepted and simplify.check == "symbolic_zero"
    assert simplify.symbolic_zero is True and simplify.restrictions == 1
    assert simplify.points_checked and simplify.points_agreed == simplify.points_checked
    definite = execute(
        "integrate", {"latex": r"\sin x", "variable": "x", "lower": "0", "upper": r"\pi"}, ctx
    ).evidence
    assert definite and definite.check == "quadrature" and definite.points_checked
    blocked = execute("simplify", {"latex": r"\sqrt{x^2}"}, ctx).evidence
    assert blocked and not blocked.accepted and blocked.reason == "conditions_not_satisfied"
    lookup = execute("search_handbook", {"query": "gaussian"}, ctx).evidence
    assert lookup and lookup.check == "lookup" and lookup.accepted and lookup.reason == "lookup"
    none = execute("search_handbook", {"query": "zzzqqqxxx"}, ctx)
    assert none.status == "no_results" and none.evidence and not none.evidence.accepted
    refused = execute("simplify", {}, ctx)
    assert refused.arguments_valid is False and refused.evidence and not refused.evidence.accepted
    assert refused.error_category == "invalid_arguments"
    assert execute("nope", {}, ctx).error_category == "unknown_tool"


def test_an_unsupported_derivative_is_reported_as_such_in_the_trace(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("differentiate", {"latex": "|z|", "variables": [{"name": "z"}]}),
            say("It has no complex derivative."),
        ]
    )
    step = final(play(llm, ctx))["trace"][0]
    assert step["status"] == "unsupported" and step["error_category"] == "unsupported"
    assert step["evidence"]["accepted"] is False and step["evidence"]["check"] == "none"


# ---------------------------------------------------------------------------- assumptions
def test_a_call_that_drops_a_declared_assumption_is_flagged(ctx: ToolContext) -> None:
    llm = ReplayLLM([call("simplify", {"latex": r"\sqrt{x^2}"}), say("Done.")])
    msg = final(play(llm, ctx, prior_assumptions={"x": ["positive"], "y": ["real"]}))
    assert msg["trace"][0]["assumptions_dropped"] == ["x"]  # y is not in this formula
    kept = ReplayLLM(
        [
            call("simplify", {"latex": r"\sqrt{x^2}", "assumptions": {"x": ["positive"]}}),
            say("Done."),
        ]
    )
    ok = final(play(kept, ctx, prior_assumptions={"x": ["positive"]}))
    assert ok["trace"][0]["assumptions_dropped"] == []


def test_assumptions_declared_earlier_in_the_turn_count(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("simplify", {"latex": r"\sqrt{x^2}", "assumptions": {"x": ["positive"]}}),
            call("simplify", {"latex": r"\sqrt{x^4}"}),
            say("Done."),
        ]
    )
    msg = final(play(llm, ctx))
    assert msg["trace"][0]["assumptions_dropped"] == [] and msg["trace"][1][
        "assumptions_dropped"
    ] == ["x"]


# ----------------------------------------------------------------------- claim check status
def test_a_failed_claim_check_is_recorded_not_hidden(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: Any) -> Any:
        raise RuntimeError("secret formula text")

    monkeypatch.setattr(ctx.engine, "check_claims", boom)
    msg = final(play(ReplayLLM([say(r"Surely $(x+1)^2 = x^2 + 1$.")]), ctx))
    assert msg["debug"]["claim_check"] == "error" and msg["debug"]["claims"] == []
    assert "secret" not in str(msg["debug"])
    assert segments(msg)[1]["status"] == "unchecked"


def test_nothing_to_check_is_not_a_failure(ctx: ToolContext) -> None:
    msg = final(play(ReplayLLM([say("No formulas here.")]), ctx))
    assert msg["debug"]["claim_check"] == "none"
    ok = final(play(ReplayLLM([say(r"Surely $(x+1)^2 = x^2 + 1$.")]), ctx))
    assert ok["debug"]["claim_check"] == "ok" and ok["debug"]["claims"]


# ------------------------------------------------------------------------ model usage
def test_token_usage_and_finish_reason_are_recorded_when_the_host_reports_them(
    ctx: ToolContext,
) -> None:
    reply = [Chunk(text="Hello.", usage=Usage(tokens_in=120, tokens_out=7, finish_reason="stop"))]
    msg = final(play(ReplayLLM([reply]), ctx))
    c = msg["debug"]["llm_calls"][0]
    assert (c["tokens_in"], c["tokens_out"], c["finish_reason"]) == (120, 7, "stop")
    assert c["first_token_ms"] is not None
    done = msg["debug"]["events"][-1]
    assert done["tokens_in"] == 120 and done["tokens_out"] == 7


def test_missing_usage_stays_missing(ctx: ToolContext) -> None:
    msg = final(play(ReplayLLM([say("Hello.")]), ctx))
    c = msg["debug"]["llm_calls"][0]
    assert c["tokens_in"] is None and c["tokens_out"] is None
    assert msg["debug"]["events"][-1]["tokens_in"] is None


# ------------------------------------------------------------------------ resource limits
def test_a_resource_limit_in_the_sandbox_gets_a_specific_category() -> None:
    tiny = ToolContext(
        engine=Engine(Limits(timeout_seconds=0.001, max_concurrency=1)), kb=KnowledgeBase.load()
    )
    msg = final(play(ReplayLLM([call("simplify", SIMPLIFY), say("It timed out.")]), tiny))
    step = msg["trace"][0]
    assert step["status"] == "resource_limit"
    assert step["error_category"] in {"timeout", "sandbox_killed", "busy"}
    assert step["engine_ms"] is not None
    verification = [e for e in msg["debug"]["events"] if e["event"] == "verification.completed"]
    assert verification and verification[0]["accepted"] is False
    assert verification[0]["reason"] == "resource_limit"
