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
