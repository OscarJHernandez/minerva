import asyncio
import json
from typing import Any

import httpx
from minerva_api.agent.llm import LLMError, OllamaClient, ThinkFilter


def lines(*events: dict[str, Any]) -> bytes:
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode()


def client_for(body: bytes, status: int = 200, **kw: Any) -> OllamaClient:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, content=body)

    c = OllamaClient("http://model.test:11434/", "m", transport=httpx.MockTransport(handler), **kw)
    c.seen = seen  # type: ignore[attr-defined]
    return c


async def collect(c: OllamaClient) -> list[Any]:
    return [chunk async for chunk in c.chat([{"role": "user", "content": "hi"}], [])]


def test_streams_text_and_tool_calls_and_ignores_thinking() -> None:
    body = lines(
        {"message": {"role": "assistant", "thinking": "hidden thoughts"}, "done": False},
        {"message": {"role": "assistant", "content": "Hel"}, "done": False},
        {"message": {"role": "assistant", "content": "lo"}, "done": False},
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "simplify", "arguments": {"latex": "x"}}}],
            },
            "done": True,
        },
    )
    chunks = asyncio.run(collect(client_for(body)))
    assert "".join(c.text for c in chunks) == "Hello"
    assert chunks[-1].tool_calls[0].name == "simplify"
    assert chunks[-1].tool_calls[0].arguments == {"latex": "x"}


def test_request_disables_thinking_by_default_and_sets_the_model() -> None:
    c = client_for(lines({"message": {"content": "ok"}, "done": True}))
    asyncio.run(collect(c))
    sent = json.loads(c.seen[0].content)  # type: ignore[attr-defined]
    assert sent["think"] is False and sent["model"] == "m" and sent["stream"] is True
    assert str(c.seen[0].url) == "http://model.test:11434/api/chat"  # type: ignore[attr-defined]


def test_arguments_given_as_a_json_string_are_accepted() -> None:
    body = lines(
        {
            "message": {
                "tool_calls": [{"function": {"name": "t", "arguments": '{"a": 1}'}}],
            },
            "done": True,
        }
    )
    chunks = asyncio.run(collect(client_for(body)))
    assert chunks[0].tool_calls[0].arguments == {"a": 1}


def test_errors_become_safe_llm_errors() -> None:
    for c in (client_for(b"nope", status=500), client_for(b"not json\n")):
        try:
            asyncio.run(collect(c))
        except LLMError as exc:
            assert "model host" in str(exc)
        else:
            raise AssertionError("expected LLMError")


def test_think_filter_drops_reasoning_before_the_closing_tag() -> None:
    f = ThinkFilter()
    out = [
        f.feed(t)
        for t in ["Okay, let me think. ", "The user wants", "...</thi", "nk>The answer is $x+1$."]
    ]
    assert "".join(out) == "The answer is $x+1$." and f.flush() == ""


def test_think_filter_releases_long_untagged_replies_and_drops_unclosed_think() -> None:
    f = ThinkFilter(limit=20)
    assert f.feed("short ") == ""
    assert f.feed("a normal reply that is long enough") != ""
    g = ThinkFilter()
    g.feed("<think>never closed")
    assert g.flush() == ""
    h = ThinkFilter()
    h.feed("plain short reply")
    assert h.flush() == "plain short reply"


def test_strip_thinking_option_filters_the_stream() -> None:
    body = lines(
        {"message": {"content": "Okay, reasoning...</think>"}, "done": False},
        {"message": {"content": "Final."}, "done": True},
    )
    chunks = asyncio.run(collect(client_for(body, strip_thinking=True)))
    assert "".join(c.text for c in chunks) == "Final."


def sent_options(c: OllamaClient) -> dict[str, Any]:
    asyncio.run(collect(c))
    return json.loads(c.seen[0].content)["options"]  # type: ignore[attr-defined]


def test_sampling_defaults_are_unchanged_and_a_seed_is_opt_in() -> None:
    body = lines({"message": {"content": "ok"}, "done": True})
    assert sent_options(client_for(body)) == {"temperature": 0.2}
    assert sent_options(client_for(body, temperature=0.0, seed=7)) == {
        "temperature": 0.0,
        "seed": 7,
    }
    assert sent_options(client_for(body, seed=0)) == {"temperature": 0.2, "seed": 0}


def test_token_usage_and_finish_reason_come_from_the_last_line() -> None:
    body = lines(
        {"message": {"content": "Hel"}, "done": False},
        {
            "message": {"content": "lo"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 120,
            "eval_count": 7,
            "total_duration": 123456,
        },
    )
    chunks = asyncio.run(collect(client_for(body)))
    assert "".join(c.text for c in chunks) == "Hello"
    usage = chunks[-1].usage
    assert usage is not None and (usage.prompt_tokens, usage.completion_tokens) == (120, 7)
    assert usage.finish_reason == "stop"
    assert all(c.usage is None for c in chunks[:-1])


def test_usage_arrives_on_its_own_chunk_when_the_last_line_has_no_content() -> None:
    body = lines(
        {"message": {"content": "ok"}, "done": False},
        {"message": {"content": ""}, "done": True, "done_reason": "length", "eval_count": 3},
    )
    chunks = asyncio.run(collect(client_for(body)))
    assert chunks[-1].text == "" and chunks[-1].usage is not None
    assert chunks[-1].usage.finish_reason == "length"
    assert (chunks[-1].usage.prompt_tokens, chunks[-1].usage.completion_tokens) == (None, 3)


def test_usage_is_absent_when_the_host_reports_none() -> None:
    body = lines({"message": {"content": "ok"}, "done": False}, {"message": {}, "done": True})
    chunks = asyncio.run(collect(client_for(body)))
    assert all(c.usage is None for c in chunks)  # nothing is estimated


def test_a_non_integer_count_is_not_trusted() -> None:
    body = lines(
        {"message": {"content": "ok"}, "done": True, "eval_count": "many", "done_reason": "weird"}
    )
    usage = asyncio.run(collect(client_for(body)))[-1].usage
    assert usage is not None and usage.completion_tokens is None and usage.finish_reason == "other"


def test_unusable_tool_calls_are_marked_not_guessed() -> None:
    def call_of(function: Any) -> list[Any]:
        body = lines({"message": {"tool_calls": [{"function": function}]}, "done": True})
        return asyncio.run(collect(client_for(body)))[-1].tool_calls

    assert call_of({"name": "simplify", "arguments": {"latex": "x"}})[0].malformed is None
    assert call_of({"name": "simplify", "arguments": '{"latex": "x"}'})[0].arguments == {
        "latex": "x"
    }
    bad_json = call_of({"name": "simplify", "arguments": "{not json"})[0]
    assert bad_json.malformed == "arguments_not_json" and bad_json.arguments == {}
    assert (
        call_of({"name": "simplify", "arguments": "[1, 2]"})[0].malformed == "arguments_not_object"
    )
    assert call_of({"name": "simplify", "arguments": 7})[0].malformed == "arguments_not_object"
    assert call_of({"name": "simplify"})[0].malformed is None  # no arguments is allowed
    assert call_of({"arguments": {}})[0].malformed == "missing_name"
    assert call_of({"name": "", "arguments": {}})[0].malformed == "missing_name"
    assert call_of(None)[0].malformed == "missing_name"  # no function at all: no crash


def test_a_tool_call_entry_that_is_not_an_object_does_not_crash_the_parser() -> None:
    body = lines({"message": {"tool_calls": ["simplify", 3, None]}, "done": True})
    calls = asyncio.run(collect(client_for(body)))[-1].tool_calls
    assert [c.malformed for c in calls] == ["missing_name"] * 3
