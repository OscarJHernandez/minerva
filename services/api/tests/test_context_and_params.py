"""Context window, tool groups, history budget and model parameters.

Regression for the "the chat forgets the conversation" report: the system prompt plus every tool schema
was about 22 000 tokens, a model host cut the start of anything longer than its window, and the model
answered as if it had never seen the earlier turns.
"""

import asyncio
import json
from typing import Any

import httpx
import pytest
from minerva_api.agent import toolsets
from minerva_api.agent.budget import estimate_tokens, fit_history, request_chars, turns
from minerva_api.agent.events import AssistantMessage
from minerva_api.agent.llm import Chunk, OllamaClient, ToolCall
from minerva_api.agent.loop import _schemas_for, run_agent
from minerva_api.agent.params import (
    PARAM_FIELDS,
    EffectiveParams,
    ParamError,
    ParamOverrides,
    defaults,
    resolve,
)
from minerva_api.agent.prompts import SYSTEM_PROMPT
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import TOOLS, ToolContext
from minerva_api.routers_chat import _history
from minerva_api.settings import Settings
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def play(llm: ReplayLLM, ctx: ToolContext, history: list[dict[str, Any]], **kw: Any) -> list[dict]:
    async def go() -> list[dict[str, Any]]:
        return [e async for e in run_agent(llm, history, ctx, **kw)]

    return asyncio.run(go())


def names(schemas: list[dict[str, Any]]) -> list[str]:
    return [s["function"]["name"] for s in schemas]


# --- the size of a request ----------------------------------------------------------------------
def test_the_smallest_request_stays_far_below_a_small_window() -> None:
    """Core prompt + core tools. The full set was ~22k tokens and overflowed a 16k window."""
    prompt = toolsets.prompt_for([], "auto")
    tools = _schemas_for(toolsets.tool_names([], "auto"))
    assert (
        estimate_tokens(request_chars(prompt, tools, [])) < 9000
    )  # was 8000 before creative_telescoping joined the core
    full = estimate_tokens(
        request_chars(SYSTEM_PROMPT, _schemas_for(toolsets.tool_names([], "all")), [])
    )
    assert full > 16000  # the reason for groups: everything at once does not fit a 16k window


def test_every_tool_is_core_or_in_exactly_one_group() -> None:
    placed = [*toolsets.CORE_TOOLS, *(t for g in toolsets.GROUPS for t in toolsets.GROUP_TOOLS[g])]
    assert len(placed) == len(set(placed))
    assert set(placed) | {toolsets.LOAD_TOOLS} == set(TOOLS)


def test_the_prompt_only_grows_when_a_group_is_added() -> None:
    """A model host reuses the processed start of a prompt only while it is identical."""
    base = toolsets.prompt_for([], "auto")
    one = toolsets.prompt_for(["matrix"], "auto")
    two = toolsets.prompt_for(["matrix", "vector"], "auto")
    assert one.startswith(base.rstrip("\n")) and two.startswith(one)
    assert "matrix_calculus" in one and "vector_calculus" not in one
    tools_one = toolsets.tool_names(["matrix"], "auto")
    tools_two = toolsets.tool_names(["matrix", "vector"], "auto")
    assert tools_two[: len(tools_one)] == tools_one


def test_modes_core_auto_and_all() -> None:
    assert toolsets.LOAD_TOOLS not in toolsets.tool_names([], "core")
    assert toolsets.LOAD_TOOLS in toolsets.tool_names([], "auto")
    assert set(toolsets.tool_names([], "all")) == set(TOOLS) - {toolsets.LOAD_TOOLS}
    assert toolsets.prompt_for([], "all") == SYSTEM_PROMPT


def test_keywords_and_use_load_groups_in_order_of_need() -> None:
    assert toolsets.groups_in_text("Differentiate x^2 sin x") == []
    assert "matrix" in toolsets.groups_in_text("Let A and B be 2x2 matrices")
    # the user's actual question: an action functional is a calculus of variations problem
    assert "variation" in toolsets.groups_in_text("Show me the integral we're supposed to maximize")
    events = [
        ("user", "let A be a matrix"),
        ("tool", ("matrix_calculus", {})),
        ("user", "now the gradient and curl of a field"),
        ("tool", (toolsets.LOAD_TOOLS, {"groups": ["geometry", "nonsense"]})),
    ]
    assert toolsets.conversation_groups(events) == ["matrix", "vector", "geometry"]


# --- the loop sends what the conversation needs ---------------------------------------------------
def test_a_plain_question_sends_only_the_core_tools(ctx: ToolContext) -> None:
    llm = ReplayLLM([say("ok")])
    play(llm, ctx, [{"role": "user", "content": "differentiate x^2"}], params=defaults(Settings()))
    sent = names(llm.tools_seen[0])
    assert sent == toolsets.tool_names([], "auto")
    assert "matrix_calculus" not in sent and "statistical_mechanics" not in sent


def test_calling_a_tool_of_an_unloaded_group_loads_the_group(ctx: ToolContext) -> None:
    llm = ReplayLLM([call("matrix_calculus", {"operation": "nope"}), say("done")])
    events = play(llm, ctx, [{"role": "user", "content": "hello"}], params=defaults(Settings()))
    assert "matrix_calculus" not in names(llm.tools_seen[0])
    assert "matrix_calculus" in names(llm.tools_seen[1])
    assert "`matrix_calculus`" in llm.seen[1][0]["content"]  # its rules arrived with it
    message = next(e["data"] for e in events if e["event"] == "message")
    assert message["debug"]["groups"] == ["matrix"]


def test_load_tools_adds_the_group_for_the_next_step(ctx: ToolContext) -> None:
    llm = ReplayLLM([call("load_tools", {"groups": ["geometry"]}), say("loaded")])
    events = play(llm, ctx, [{"role": "user", "content": "hi"}], params=defaults(Settings()))
    assert "geometry" not in names(llm.tools_seen[0])
    assert {"geometry", "tensor_canonicalize"} <= set(names(llm.tools_seen[1]))
    message = next(e["data"] for e in events if e["event"] == "message")
    card = next(b["card"] for b in message["blocks"] if b["kind"] == "card")
    assert card["type"] == "note" and card["data"]["groups"] == ["geometry"]
    bad = ReplayLLM([call("load_tools", {"groups": ["nope"]}), say("x")])
    out = play(bad, ctx, [{"role": "user", "content": "hi"}], params=defaults(Settings()))
    assert next(e["data"] for e in out if e["event"] == "message")["trace"][0]["status"] == "error"


def test_groups_from_earlier_turns_stay_loaded(ctx: ToolContext) -> None:
    llm = ReplayLLM([say("ok")])
    play(
        llm,
        ctx,
        [{"role": "user", "content": "thanks"}],
        prior_groups=["statmech", "matrix"],
        params=defaults(Settings()),
    )
    sent = names(llm.tools_seen[0])
    assert sent.index("statistical_mechanics") < sent.index("matrix_calculus")


def test_without_parameters_everything_is_sent_as_before(ctx: ToolContext) -> None:
    llm = ReplayLLM([say("ok")])
    play(llm, ctx, [{"role": "user", "content": "hi"}])
    assert len(llm.tools_seen[0]) == len(TOOLS) - 1  # all but `load_tools`


# --- history that does not fit is left out, visibly ------------------------------------------------
def convo(n: int, size: int = 400) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(n):
        out.append({"role": "user", "content": f"q{i} " + "x" * size})
        out.append({"role": "assistant", "content": f"a{i} " + "y" * size})
    return out


def test_fit_history_keeps_the_newest_whole_turns() -> None:
    history = [*convo(10), {"role": "user", "content": "now"}]
    kept, dropped = fit_history(history, fixed_chars=0, limit_tokens=1000)
    assert kept[-1]["content"] == "now" and dropped == len(history) - len(kept) > 0
    assert kept[0]["role"] == "user"  # never starts in the middle of an exchange
    assert [m["content"][:2] for m in kept[:2]] == [
        f"q{10 - len(kept) // 2}",
        f"a{10 - len(kept) // 2}",
    ]


def test_the_last_message_is_always_kept() -> None:
    kept, dropped = fit_history(
        [{"role": "user", "content": "z" * 50000}], fixed_chars=100000, limit_tokens=100
    )
    assert len(kept) == 1 and dropped == 0
    assert turns([]) == []


def test_a_turn_that_overflows_trims_old_turns_and_says_so(ctx: ToolContext) -> None:
    params = defaults(Settings()).model_copy(update={"num_ctx": 4096})
    llm = ReplayLLM([say("ok")])
    history = [*convo(12, 600), {"role": "user", "content": "and now?"}]
    events = play(llm, ctx, history, params=params)
    sent = llm.seen[0]
    assert sent[-1]["content"] == "and now?"
    assert len(sent) - 1 < len(history)  # some earlier turns did not go
    message = next(e["data"] for e in events if e["event"] == "message")
    assert message["context_trimmed"] > 0
    assert message["debug"]["history_dropped"] == message["context_trimmed"]
    assert message["debug"]["context_window"] == 4096
    request = next(e for e in message["debug"]["events"] if e["event"] == "llm.request")
    assert request["dropped_messages"] > 0 and request["window"] == 4096
    assert AssistantMessage.model_validate(message).context_trimmed > 0


def test_a_conversation_that_fits_loses_nothing(ctx: ToolContext) -> None:
    llm = ReplayLLM([say("ok")])
    history = [*convo(3, 50), {"role": "user", "content": "again"}]
    events = play(llm, ctx, history, params=defaults(Settings()))
    assert len(llm.seen[0]) == 1 + len(history)
    assert next(e["data"] for e in events if e["event"] == "message")["context_trimmed"] == 0


# --- history rebuilt from storage -------------------------------------------------------------------
class Row:
    def __init__(self, role: str, content: dict[str, Any]) -> None:
        self.role, self.content = role, content


def stored_turn(i: int, result_id: str | None, latex: str) -> list[Row]:
    step = {
        "id": f"s{i}",
        "call_id": f"c{i}",
        "name": "simplify",
        "arguments": {"latex": "x"},
        "for_model": f"status=ok\nresult=${latex}$\n" + "detail " * 300,
        "result_id": result_id,
        "verified_latex": [latex],
        "quoted": {},
    }
    card = {"id": f"s{i}", "data": {"result_latex": latex}}
    return [
        Row("user", {"text": f"question {i}"}),
        Row(
            "assistant",
            {"text": f"answer {i}", "trace": [step], "blocks": [{"kind": "card", "card": card}]},
        ),
    ]


def test_result_ids_survive_beyond_the_window_so_they_are_never_reused() -> None:
    rows = [r for i in range(15) for r in stored_turn(i, f"r{i + 1}", f"x+{i}")]
    history, known, _, results, _ = _history(rows)  # type: ignore[arg-type]
    assert (
        results["r1"] == "x+0" and results["r15"] == "x+14"
    )  # r1 is outside the 20-message window
    assert "x+0" in known
    texts = [m["content"] for m in history if m["role"] == "user"]
    assert (
        "question 0" not in texts and "question 14" in texts
    )  # but the model sees the newest only
    assert history[0]["role"] == "user"


def test_old_tool_results_are_shortened_and_recent_ones_are_not() -> None:
    rows = [r for i in range(4) for r in stored_turn(i, f"r{i + 1}", f"x+{i}")]
    history, *_ = _history(rows)  # type: ignore[arg-type]
    tool = [m["content"] for m in history if m["role"] == "tool"]
    assert "[shortened" in tool[0] and "result=$x+0$" in tool[0] and "result_id=r1" in tool[0]
    assert "[shortened" not in tool[-1] and len(tool[-1]) > len(tool[0])


def test_a_user_message_without_a_stored_reply_is_merged_with_the_next() -> None:
    rows = [Row("user", {"text": "first"}), Row("user", {"text": "second"})]
    history, *_ = _history(rows)  # type: ignore[arg-type]
    assert history == [{"role": "user", "content": "first\n\nsecond"}]


def test_history_budget_in_tokens_applies_to_the_stored_history() -> None:
    rows = [r for i in range(10) for r in stored_turn(i, None, f"x+{i}")]
    small, *_ = _history(rows, max_tokens=800)  # type: ignore[arg-type]
    big, *_ = _history(rows)  # type: ignore[arg-type]
    assert 0 < len(small) < len(big) and small[0]["role"] == "user"


# --- parameters --------------------------------------------------------------------------------------
def test_only_these_fields_can_be_parameters_and_none_is_mathematical() -> None:
    """A verification tolerance, a limit or the claim check must never become a parameter."""
    assert set(ParamOverrides.model_fields) == set(PARAM_FIELDS)
    assert set(EffectiveParams.model_fields) == set(PARAM_FIELDS)
    assert not {"tolerance", "max_tool_calls", "max_steps", "max_seconds", "claims"} & set(
        PARAM_FIELDS
    )
    with pytest.raises(ValueError):  # unknown fields are refused, not ignored
        ParamOverrides(tolerance=1e-3)  # type: ignore[call-arg]


def test_defaults_come_from_settings_and_unset_values_are_not_sent() -> None:
    d = defaults(Settings())
    assert d.ollama_options() == {"num_ctx": 32768, "temperature": 0.2}
    full = defaults(Settings(llm_num_predict=2000, llm_top_p=0.9, llm_top_k=40, llm_temperature=0))
    assert full.ollama_options() == {
        "num_ctx": 32768,
        "temperature": 0.0,
        "num_predict": 2000,
        "top_p": 0.9,
        "top_k": 40,
    }


def test_overrides_need_permission_and_the_context_size_needs_the_allow_list() -> None:
    closed = Settings()
    assert resolve(closed, None, "a@example.com") == defaults(closed)
    assert resolve(closed, ParamOverrides(), "a@example.com") == defaults(closed)
    with pytest.raises(ParamError):
        resolve(closed, ParamOverrides(temperature=0.5), "a@example.com")
    tuning = Settings(llm_tuning=True)
    assert (
        resolve(tuning, ParamOverrides(temperature=0.5, toolset="all"), "a@example.com").toolset
        == "all"
    )
    with pytest.raises(ParamError, match="context size"):
        resolve(tuning, ParamOverrides(num_ctx=16384), "a@example.com")
    allowed = Settings(llm_tuning_emails="Op@Example.com", llm_num_ctx_max=40000)
    assert resolve(allowed, ParamOverrides(num_ctx=40000), "op@example.com").num_ctx == 40000
    with pytest.raises(ParamError, match="limited"):
        resolve(allowed, ParamOverrides(num_ctx=50000), "op@example.com")
    with pytest.raises(ParamError):
        resolve(allowed, ParamOverrides(num_ctx=16384), "someone@example.com")
    assert resolve(tuning, ParamOverrides(num_predict=0), "a@x.io").num_predict is None


def test_ranges_are_enforced() -> None:
    for bad in ({"temperature": 3}, {"top_p": 0}, {"num_ctx": 100}, {"history_messages": 1}):
        with pytest.raises(ValueError):
            ParamOverrides(**bad)


def sent_payload(client: OllamaClient) -> dict[str, Any]:
    async def go() -> None:
        async for _ in client.chat([{"role": "user", "content": "hi"}], []):
            pass

    asyncio.run(go())
    return json.loads(client.seen[0].content)  # type: ignore[attr-defined]


def make_client(**kw: Any) -> OllamaClient:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b'{"message": {"content": "ok"}, "done": true}\n')

    client = OllamaClient("http://h.test", "m", transport=httpx.MockTransport(handler), **kw)
    client.seen = seen  # type: ignore[attr-defined]
    return client


def test_a_turns_parameters_reach_the_model_host_and_the_shared_client_is_untouched() -> None:
    shared = make_client()
    params = defaults(Settings(llm_num_predict=1500, llm_think=True)).model_copy(
        update={"temperature": 0.7, "top_k": 20}
    )
    turn = shared.with_params(params)
    turn.seen = shared.seen  # type: ignore[attr-defined]
    payload = sent_payload(turn)
    assert payload["options"] == {
        "num_ctx": 32768,
        "temperature": 0.7,
        "num_predict": 1500,
        "top_k": 20,
    }
    assert payload["think"] is True
    assert shared.extra_options == {} and shared.temperature == 0.2 and shared.think is False


def test_the_parameters_a_turn_used_are_stored_with_the_reply(ctx: ToolContext) -> None:
    params = defaults(Settings()).model_copy(update={"temperature": 0.4})
    events = play(ReplayLLM([say("ok")]), ctx, [{"role": "user", "content": "hi"}], params=params)
    stored = next(e["data"] for e in events if e["event"] == "message")["debug"]["params"]
    assert stored["temperature"] == 0.4 and stored["num_ctx"] == 32768
    assert all(isinstance(v, int | float | bool | str | type(None)) for v in stored.values())


def test_openrouter_takes_only_what_it_supports() -> None:
    from minerva_api.agent.openrouter import OpenRouterClient

    client = OpenRouterClient("key", "vendor/model")
    params = defaults(Settings(llm_top_p=0.8, llm_num_predict=900))
    turn = client.with_params(params)
    assert turn.model == "vendor/model" and repr(turn) == "OpenRouterClient(model='vendor/model')"
    assert Chunk and ToolCall  # imported for the replay helpers above


# --- the evaluation script takes the same parameters and records them --------------------------------
def test_eval_flags_become_recorded_parameters_and_comparisons_show_the_difference() -> None:
    from minerva_api.agent.evaluation.cli import _parser, build_params
    from minerva_api.agent.evaluation.report import build_document, compare, render_markdown

    from .test_evaluation import META, result

    args = _parser().parse_args(
        [
            "--num-ctx",
            "16384",
            "--no-think",
            "--toolset",
            "core",
            "--top-p",
            "0.9",
            "--num-predict",
            "0",
        ]
    )
    params = build_params(args)
    assert (params.num_ctx, params.toolset, params.top_p, params.think) == (
        16384,
        "core",
        0.9,
        False,
    )
    assert params.num_predict is None  # 0 means the model's own limit
    assert build_params(_parser().parse_args([])) == defaults(Settings())
    with pytest.raises(ValueError):
        build_params(_parser().parse_args(["--temperature", "5"]))

    base_meta = {**META, "params": defaults(Settings()).model_dump()}
    new_meta = {**META, "label": "new", "params": params.model_dump()}
    base = build_document(base_meta, {"m": [result("a")]})
    new = build_document(new_meta, {"m": [result("a")]})
    assert "num_ctx 16384" in render_markdown(new)
    text = compare(base, new)
    assert "The model parameters differ" in text and "| num_ctx | 32768 | 16384 |" in text
    assert "The model parameters differ" not in compare(base, base)
    old = build_document(META, {"m": [result("a")]})  # saved before parameters were recorded
    assert "not recorded" in compare(old, new)


def test_a_prompt_that_reaches_the_window_is_reported_as_an_invalid_measurement() -> None:
    from minerva_api.agent.evaluation.report import build_document, render_markdown, summarise

    from .test_evaluation import META, result

    fine = result("a", max_prompt_tokens=4000, context_window=32768)
    cut = result("b", max_prompt_tokens=32768, context_window=32768, history_dropped=2)
    context = summarise([fine, cut])["context"]
    assert context == {
        "window": 32768,
        "max_prompt_tokens": 32768,
        "runs_at_window": 1,
        "runs_with_history_dropped": 1,
    }
    text = render_markdown(build_document(META, {"m": [fine, cut]}))
    assert "reached the window" in text and "left out in 1 run" in text
    assert "window not recorded" in render_markdown(build_document(META, {"m": [result("a")]}))


def test_a_prompt_cut_by_the_host_is_recognised_by_its_size() -> None:
    """The host keeps about half the window when it cuts: 8 195 tokens of a 16 384 window."""
    from minerva_api.agent.evaluation.report import summarise

    from .test_evaluation import result

    cut = result("a", max_prompt_tokens=8195, context_window=16384)
    assert summarise([cut])["context"]["runs_at_window"] == 1
    fine = result("b", max_prompt_tokens=8000, context_window=32768)
    assert summarise([fine])["context"]["runs_at_window"] == 0


# --- the live view never shows a tool call that was written as prose --------------------------------------------
def streamed_text(events: list[dict[str, Any]]) -> list[str]:
    return [e["data"]["delta"] for e in events if e["event"] == "text"]


def test_a_tool_call_written_as_prose_is_never_streamed(ctx: ToolContext) -> None:
    call_text = '{"name": "simplify", "arguments": {"latex": "x"}}'
    llm = ReplayLLM(
        [[Chunk(text=call_text[:20]), Chunk(text=call_text[20:])], say("Done, it is $x$.")],
        repeat_last=False,
    )
    events = play(llm, ctx, [{"role": "user", "content": "simplify x"}])
    shown = "".join(streamed_text(events))
    assert "arguments" not in shown and "Done, it is" in shown


def test_prose_that_starts_with_a_brace_is_shown_once_it_is_known_to_be_prose(
    ctx: ToolContext,
) -> None:
    llm = ReplayLLM([[Chunk(text="{1, 2, 3}"), Chunk(text=" is a set of three numbers.")]])
    events = play(llm, ctx, [{"role": "user", "content": "what is this"}])
    assert "".join(streamed_text(events)) == "{1, 2, 3} is a set of three numbers."


def test_ordinary_prose_still_streams_piece_by_piece(ctx: ToolContext) -> None:
    llm = ReplayLLM([[Chunk(text="Hello"), Chunk(text=" there"), Chunk(text=", friend.")]])
    events = play(llm, ctx, [{"role": "user", "content": "hi"}])
    assert streamed_text(events) == ["Hello", " there", ", friend."]
