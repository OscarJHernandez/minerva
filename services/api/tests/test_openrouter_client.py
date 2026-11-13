import asyncio
import json
from typing import Any

import httpx
import pytest
from minerva_api.agent.llm import Chunk, LLMError, ollama_messages
from minerva_api.agent.openrouter import (
    OpenRouterClient,
    check_connection,
    openrouter_messages,
)

KEY = "sk-or-v1-supersecretkey"


def sse(*events: Any, done: bool = True) -> bytes:
    lines = [
        f"{e}\n\n" if isinstance(e, str) and e.startswith(":") else f"data: {json.dumps(e)}\n\n"
        for e in events
    ]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def delta(**d: Any) -> dict[str, Any]:
    return {"choices": [{"delta": d, "finish_reason": None}]}


def finish(reason: str = "stop") -> dict[str, Any]:
    return {"choices": [{"delta": {}, "finish_reason": reason}]}


def client_for(body: bytes, status: int = 200, **kw: Any) -> OpenRouterClient:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, content=body)

    c = OpenRouterClient(KEY, "openai/gpt-4.1-mini", transport=httpx.MockTransport(handler), **kw)
    c.seen = seen  # type: ignore[attr-defined]
    return c


async def collect(c: OpenRouterClient) -> list[Chunk]:
    return [chunk async for chunk in c.chat([{"role": "user", "content": "hi"}], [])]


def test_streams_text_skips_keepalives_and_reads_usage() -> None:
    body = sse(
        ": OPENROUTER PROCESSING",
        delta(content="Hel"),
        delta(content="lo"),
        finish(),
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3, "cost": 0.0004}},
    )
    c = client_for(body)
    chunks = asyncio.run(collect(c))
    assert "".join(x.text for x in chunks) == "Hello"
    usage = next(x.usage for x in chunks if x.usage)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.cost_usd) == (12, 3, 0.0004)
    req = c.seen[0]  # type: ignore[attr-defined]
    sent = json.loads(req.content)
    assert str(req.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert sent["stream"] is True and sent["provider"]["data_collection"] == "deny"
    assert KEY not in req.content.decode()  # the key is a header only


def test_tool_call_deltas_are_accumulated_with_ids_and_order() -> None:
    body = sse(
        delta(
            tool_calls=[
                {"index": 0, "id": "call_a", "function": {"name": "simplify", "arguments": ""}}
            ]
        ),
        delta(tool_calls=[{"index": 0, "function": {"arguments": '{"latex": "x^'}}]),
        delta(
            tool_calls=[
                {
                    "index": 1,
                    "id": "call_b",
                    "function": {"name": "search_handbook", "arguments": "{}"},
                }
            ]
        ),
        delta(tool_calls=[{"index": 0, "function": {"arguments": '2"}'}}]),
        finish("tool_calls"),
    )
    chunks = asyncio.run(collect(client_for(body)))
    calls = [c for ch in chunks for c in ch.tool_calls]
    assert [(c.id, c.name, c.arguments) for c in calls] == [
        ("call_a", "simplify", {"latex": "x^2"}),
        ("call_b", "search_handbook", {}),
    ]


def test_invalid_argument_json_becomes_empty_arguments() -> None:
    body = sse(
        delta(
            tool_calls=[{"index": 0, "id": "c", "function": {"name": "t", "arguments": "{oops"}}]
        ),
        finish(),
    )
    calls = [c for ch in asyncio.run(collect(client_for(body))) for c in ch.tool_calls]
    assert calls[0].arguments == {}


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, "auth"),
        (403, "auth"),
        (402, "credits"),
        (429, "rate_limit"),
        (503, "unavailable"),
        (400, "bad_model"),
        (404, "bad_model"),
        (408, "timeout"),
    ],
)
def test_http_errors_map_to_kinds_and_never_echo_the_key(status: int, kind: str) -> None:
    body = json.dumps({"error": {"message": f"bad key {KEY}"}}).encode()
    with pytest.raises(LLMError) as err:
        asyncio.run(collect(client_for(body, status)))
    assert err.value.kind == kind
    assert KEY not in str(err.value) and "bad key" not in str(err.value)


def test_error_inside_the_stream_is_mapped() -> None:
    body = sse(delta(content="a"), {"error": {"code": 429, "message": KEY}})
    with pytest.raises(LLMError) as err:
        asyncio.run(collect(client_for(body)))
    assert err.value.kind == "rate_limit" and KEY not in str(err.value)


def test_stream_cut_before_the_end_is_an_error() -> None:
    with pytest.raises(LLMError) as err:
        asyncio.run(collect(client_for(sse(delta(content="partial"), done=False))))
    assert err.value.kind == "unavailable"


def test_timeouts_and_disconnects() -> None:
    def boom(kind: type[Exception]) -> OpenRouterClient:
        def handler(request: httpx.Request) -> httpx.Response:
            raise kind("x", request=request)

        return OpenRouterClient(KEY, "a/b", transport=httpx.MockTransport(handler))

    with pytest.raises(LLMError) as err:
        asyncio.run(collect(boom(httpx.ReadTimeout)))
    assert err.value.kind == "timeout"
    with pytest.raises(LLMError) as err2:
        asyncio.run(collect(boom(httpx.ConnectError)))
    assert err2.value.kind == "unavailable"


def test_model_ids_are_validated_and_repr_hides_the_key() -> None:
    for bad in ["", "a b", "x;rm", "../x", "a" * 200]:
        with pytest.raises(LLMError):
            OpenRouterClient(KEY, bad)
    assert KEY not in repr(OpenRouterClient(KEY, "a/b"))


HISTORY = [
    {"role": "user", "content": "q"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "name": "simplify", "arguments": {"latex": "x"}},
            {"id": "call_2", "name": "search_handbook", "arguments": {"query": "q"}},
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "name": "simplify", "content": "r1"},
    {"role": "tool", "tool_call_id": "call_2", "name": "search_handbook", "content": "r2"},
]


def test_history_conversion_for_both_providers() -> None:
    o = openrouter_messages(HISTORY)
    assert o[1]["content"] is None and o[1]["tool_calls"][0]["type"] == "function"
    assert json.loads(o[1]["tool_calls"][1]["function"]["arguments"]) == {"query": "q"}
    assert [m["tool_call_id"] for m in o[2:]] == ["call_1", "call_2"]
    ol = ollama_messages(HISTORY)
    assert ol[1]["tool_calls"][0] == {"function": {"name": "simplify", "arguments": {"latex": "x"}}}
    assert ol[2] == {"role": "tool", "tool_name": "simplify", "content": "r1"}


def check_transport(
    key_status: int = 200, models: Any = None, limit: Any = 5.0
) -> httpx.MockTransport:
    models = (
        models
        if models is not None
        else [
            {"id": "a/tools", "supported_parameters": ["tools", "temperature"]},
            {"id": "a/notools", "supported_parameters": ["temperature"]},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/key"):
            return httpx.Response(key_status, json={"data": {"limit_remaining": limit}})
        return httpx.Response(200, json={"data": models})

    return httpx.MockTransport(handler)


def run_check(model: str, **kw: Any):  # type: ignore[no-untyped-def]
    return asyncio.run(
        check_connection(KEY, model, transport=check_transport(**kw), use_cache=False)
    )


def test_check_connection_outcomes() -> None:
    assert run_check("a/tools").ok
    assert not run_check("a/tools", key_status=401).ok
    r = run_check("a/notools")
    assert not r.ok and "tool" in r.message
    assert not run_check("a/missing").ok
    assert "credits" in run_check("a/tools", limit=0).message
    assert not run_check("bad id").ok
    for r2 in (run_check("a/tools", key_status=401), run_check("a/tools")):
        assert KEY not in r2.message
