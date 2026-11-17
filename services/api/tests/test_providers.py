"""Provider switching: Ollama (local) and OpenRouter (cloud), with mocked hosts only."""

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from minerva_api.agent.llm import OllamaClient
from minerva_api.agent.openrouter import reset_model_cache
from minerva_api.settings import Settings

from .conftest import build_app, make_settings, register
from .test_chat import sse as plain_sse

KEY = "sk-or-v1-supersecretkey-1234567890"
OR = {"kind": "openrouter", "model": "openai/gpt-4.1-mini", "cloud_consent": True}


class FakeOpenRouter:
    """Answers a turn with one simplify call, then with prose once the tool result is back."""

    def __init__(self, status: int = 200) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []
        self.status = status
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            self.requests.append(body)
            self.headers.append(request.headers)
            if self.status != 200:
                return httpx.Response(self.status, json={"error": {"message": KEY}})
            if body["messages"][-1]["role"] == "tool":
                events = [{"choices": [{"delta": {"content": "Done, see the card."}}]}]
            else:
                events = [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_xyz",
                                            "function": {
                                                "name": "simplify",
                                                "arguments": '{"latex": "\\\\frac{x^2-1}{x-1}"}',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
                ]
            text = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
            return httpx.Response(200, content=text.encode())
        if request.url.path.endswith("/key"):
            ok = request.headers.get("authorization") == f"Bearer {KEY}"
            return httpx.Response(200 if ok else 401, json={"data": {"limit_remaining": 3}})
        return httpx.Response(
            200, json={"data": [{"id": "openai/gpt-4.1-mini", "supported_parameters": ["tools"]}]}
        )


def make_client(fake: FakeOpenRouter, **over: object) -> TestClient:
    opts: dict[str, object] = {
        "llm_mode": "scripted",
        "openrouter_byok": True,
        "openrouter_model": "openai/gpt-4.1-mini",
    }
    opts.update(over)
    app = build_app(**opts)
    app.state.openrouter_transport = fake.transport
    client = TestClient(app)
    client.__enter__()
    register(client)
    return client


def turn(client: TestClient, cid: str, text: str, provider: dict | None = None):  # type: ignore[type-arg,no-untyped-def]
    body: dict[str, Any] = {"content": text}
    if provider is not None:
        body["provider"] = provider
    events: list[tuple[str, dict]] = []  # type: ignore[type-arg]
    with client.stream("POST", f"/v1/chat/conversations/{cid}/messages", json=body) as r:
        if r.status_code != 200:
            r.read()
            return r.status_code, r.text, events
        event = ""
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                events.append((event, json.loads(line[6:])))
    return 200, "", events


def new_conv(client: TestClient) -> str:
    return str(client.post("/v1/chat/conversations", json={}).json()["id"])


def test_byok_turn_runs_the_real_tools_and_keeps_tool_call_ids(capsys) -> None:  # type: ignore[no-untyped-def]
    fake = FakeOpenRouter()
    c = make_client(fake)
    cid = new_conv(c)
    code, _, events = turn(c, cid, r"Simplify $\frac{x^2-1}{x-1}$", {**OR, "api_key": KEY})
    assert code == 200
    message = next(d for e, d in events if e == "message")
    assert [b["card"]["tool"] for b in message["blocks"] if b["kind"] == "card"] == ["simplify"]
    assert message["debug"]["provider"] == "openrouter"
    assert message["trace"][0]["call_id"] == "call_xyz"
    assert message["trace"][0]["status"] == "ok"  # verified by the engine, not by the model
    second = fake.requests[1]["messages"]
    assert second[-2]["tool_calls"][0]["id"] == "call_xyz"
    assert second[-1]["tool_call_id"] == "call_xyz"
    assert fake.headers[0]["authorization"] == f"Bearer {KEY}"
    # The key is nowhere else: not in the stream, the stored conversation, or the logs.
    stored = c.get(f"/v1/chat/conversations/{cid}").text
    streamed = json.dumps(events)
    out = capsys.readouterr().out
    for blob in (stored, streamed, out):
        assert KEY not in blob
    assert '"provider": "openrouter"' in out


def test_cloud_needs_consent_and_byok_must_be_enabled() -> None:
    fake = FakeOpenRouter()
    c = make_client(fake)
    cid = new_conv(c)
    code, _, _ = turn(c, cid, "hi", {"kind": "openrouter", "model": "a/b", "api_key": KEY})
    assert code == 422
    assert fake.requests == []
    off = make_client(FakeOpenRouter(), openrouter_byok=False)
    code, _, _ = turn(off, new_conv(off), "hi", {**OR, "api_key": KEY})
    assert code == 503  # nothing enabled: OpenRouter is not offered at all


def test_byok_is_refused_when_only_a_server_key_exists() -> None:
    fake = FakeOpenRouter()
    c = make_client(
        fake,
        openrouter_byok=False,
        openrouter_api_key="server-key",
        openrouter_key_users="ada@example.com",
    )
    code, _, _ = turn(c, new_conv(c), "hi", {**OR, "api_key": KEY})
    assert code == 403 and fake.requests == []


def test_invalid_model_ids_are_refused() -> None:
    c = make_client(FakeOpenRouter())
    code, _, _ = turn(c, new_conv(c), "hi", {**OR, "model": "x; drop", "api_key": KEY})
    assert code == 422


def test_switching_providers_inside_one_conversation_keeps_a_valid_history() -> None:
    fake = FakeOpenRouter()
    c = make_client(fake)
    cid = new_conv(c)
    # Local turn first (scripted stand-in), then the same conversation on OpenRouter, then local again.
    plain_sse(c, cid, r"Simplify $\frac{x^2-1}{x-1}$")
    code, _, events = turn(c, cid, "And again?", {**OR, "api_key": KEY})
    assert code == 200
    sent = fake.requests[0]["messages"]
    calls = {t["id"] for m in sent if m.get("tool_calls") for t in m["tool_calls"]}
    replies = [m["tool_call_id"] for m in sent if m["role"] == "tool"]
    assert calls and set(replies) <= calls  # every tool reply answers a call that was made
    assert all(
        isinstance(t["function"]["arguments"], str)
        for m in sent
        if m.get("tool_calls")
        for t in m["tool_calls"]
    )
    assert any(e == "message" for e, _ in events)
    code, _, back = turn(c, cid, r"Simplify $\sin^2 x+\cos^2 x$")
    assert code == 200
    assert next(d for e, d in back if e == "message")["debug"]["provider"] == "scripted"


def test_a_failing_cloud_provider_is_reported_and_never_falls_back_to_local() -> None:
    fake = FakeOpenRouter(status=402)
    c = make_client(fake)
    code, _, events = turn(c, new_conv(c), "hi", {**OR, "api_key": KEY})
    assert code == 200
    err = next(d for e, d in events if e == "error")
    assert "credits" in err["message"] and KEY not in json.dumps(events)
    assert err["debug"]["error_kind"] == "credits"
    # The failed turn is delivered (so its trace is stored), but it holds no answer, and it was the
    # cloud provider that failed: nothing fell back to the local model.
    reply = next(d for e, d in events if e == "message")
    assert reply["stop_reason"] == "model_error" and reply["debug"]["provider"] == "openrouter"
    assert reply["trace"] == [] and reply["text"].startswith("I stopped here")


def test_a_failing_local_model_never_falls_back_to_the_cloud() -> None:
    fake = FakeOpenRouter()
    c = make_client(fake, llm_mode="ollama", llm_base_url="http://ollama.test:1", llm_model="m")

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    c.app.state.local_llm = OllamaClient(  # type: ignore[attr-defined]
        "http://ollama.test:1", "m", transport=httpx.MockTransport(refuse)
    )
    code, _, events = turn(c, new_conv(c), "hi")
    assert code == 200 and any(e == "error" for e, _ in events)
    assert fake.requests == []


def test_server_key_only_for_listed_accounts_and_under_the_daily_cap() -> None:
    fake = FakeOpenRouter()
    over = {"openrouter_api_key": "server-key-value", "openrouter_daily_turns": 2}
    stranger = make_client(fake, openrouter_key_users="other@example.com", **over)
    code, _, _ = turn(stranger, new_conv(stranger), "hi", OR)
    assert code == 403 and fake.requests == []
    listed = make_client(fake, openrouter_key_users="ADA@example.com", **over)
    assert listed.get("/v1/chat/status").json()["providers"]["openrouter"]["server_key_for_you"]
    for _ in range(2):
        assert turn(listed, new_conv(listed), "hi", OR)[0] == 200
    assert fake.headers[0]["authorization"] == "Bearer server-key-value"
    assert turn(listed, new_conv(listed), "hi", OR)[0] == 429
    star = make_client(fake, openrouter_key_users="*", **over)
    assert turn(star, new_conv(star), "hi", OR)[0] == 200


def test_openrouter_as_deployment_default_needs_a_key_for_users_without_one() -> None:
    fake = FakeOpenRouter()
    c = make_client(fake, llm_mode="openrouter")
    code, _, _ = turn(c, new_conv(c), "hi")  # old-style request, no provider chosen
    assert code == 403 and fake.requests == []
    server = make_client(
        fake, llm_mode="openrouter", openrouter_api_key="server-key", openrouter_key_users="*"
    )
    assert turn(server, new_conv(server), "hi")[0] == 200  # operator chose the cloud default


def test_status_lists_providers_without_secrets_and_stays_public() -> None:
    c = make_client(FakeOpenRouter(), openrouter_api_key="server-key-value")
    body = c.get("/v1/chat/status").text
    assert "server-key-value" not in body
    data = json.loads(body)
    assert data["providers"]["openrouter"]["byok"] is True
    assert data["providers"]["default"] == "ollama"
    anon = TestClient(c.app)
    assert anon.get("/v1/chat/status").status_code == 200


def test_connection_test_endpoint() -> None:
    reset_model_cache()
    fake = FakeOpenRouter()
    c = make_client(fake)
    r = c.post(
        "/v1/chat/providers/openrouter/test", json={"api_key": KEY, "model": "openai/gpt-4.1-mini"}
    )
    assert r.status_code == 200 and r.json()["ok"] is True and KEY not in r.text
    bad = c.post(
        "/v1/chat/providers/openrouter/test",
        json={"api_key": "nope", "model": "openai/gpt-4.1-mini"},
    )
    assert bad.json()["ok"] is False
    assert (
        TestClient(c.app)
        .post("/v1/chat/providers/openrouter/test", json={"api_key": KEY, "model": "a/b"})
        .status_code
        == 401
    )


def test_settings_validation_and_backward_compatibility() -> None:
    with pytest.raises(RuntimeError):
        make_settings(llm_mode="openrouter").validate_for_runtime()
    with pytest.raises(RuntimeError):
        make_settings(llm_mode="openrouter", openrouter_model="a/b").validate_for_runtime()
    make_settings(
        llm_mode="openrouter", openrouter_model="a/b", openrouter_byok=True
    ).validate_for_runtime()
    with pytest.raises(RuntimeError):
        make_settings(openrouter_model="bad model").validate_for_runtime()
    old: Settings = make_settings(llm_mode="ollama", llm_base_url="http://h:1", llm_model="m")
    old.validate_for_runtime()
    assert old.ollama_configured and not old.openrouter_available


def _ollama_client(models: list[str] | None) -> TestClient:
    """An app whose Ollama host (a mock) lists `models`, or is down when `models` is None."""

    def tags(request: httpx.Request) -> httpx.Response:
        if models is None:
            raise httpx.ConnectError("down", request=request)
        return httpx.Response(200, json={"models": [{"name": m} for m in models]})

    app = build_app(llm_mode="ollama", llm_base_url="http://ollama.test:1", llm_model="mine:1b")
    app.state.ollama_probe_transport = httpx.MockTransport(tags)
    client = TestClient(app)
    client.__enter__()
    register(client)
    return client


def test_status_lists_installed_local_models_with_recommended_ones_first() -> None:
    c = _ollama_client(["zzz:1b", "qwen3:8b", "mine:1b", "library/llama3.1:8b"])
    ollama = c.get("/v1/chat/status").json()["providers"]["ollama"]
    names = [m["name"] for m in ollama["models"]]
    assert names == ["library/llama3.1:8b", "qwen3:8b", "mine:1b", "zzz:1b"]
    assert [m["recommended"] for m in ollama["models"]] == [True, True, False, False]
    assert ollama["reachable"] is True
    # Families that are not installed are guidance only; installed ones are not repeated.
    suggested = [s["family"] for s in ollama["suggested"]]
    assert "qwen3" not in suggested and "llama3.1" not in suggested and "mistral-nemo" in suggested
    assert "ollama.test" not in json.dumps(ollama)  # the host address never leaves the server


def test_status_when_the_local_host_is_down_lists_no_models() -> None:
    ollama = _ollama_client(None).get("/v1/chat/status").json()["providers"]["ollama"]
    assert ollama["reachable"] is False and ollama["models"] == []
    assert ollama["suggested"] == []  # nothing to compare with when the host did not answer


def test_a_turn_can_pick_another_installed_model_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minerva_api import providers

    made: list[str] = []

    class Spy(OllamaClient):
        def __init__(self, base_url: str, model: str, *args: Any, **kw: Any) -> None:
            made.append(model)
            super().__init__(
                base_url, model, *args, transport=httpx.MockTransport(_ollama_chat), **kw
            )

    monkeypatch.setattr(providers, "OllamaClient", Spy)
    c = _ollama_client(["mine:1b", "qwen3:8b"])
    c.get("/v1/chat/status")  # fills the installed-model cache, as the settings panel does
    ok, _, _ = turn(c, new_conv(c), "hi", {"kind": "ollama", "model": "qwen3:8b"})
    assert ok == 200 and made == ["qwen3:8b"]
    missing, _, _ = turn(c, new_conv(c), "hi", {"kind": "ollama", "model": "nope:7b"})
    assert missing == 422
    bad, _, _ = turn(c, new_conv(c), "hi", {"kind": "ollama", "model": "x; rm -rf /"})
    assert bad == 422 and made == ["qwen3:8b"]


def _ollama_chat(request: httpx.Request) -> httpx.Response:
    line = {"message": {"role": "assistant", "content": "Hello."}, "done": True}
    return httpx.Response(200, content=(json.dumps(line) + "\n").encode())


def test_scripted_mode_refuses_a_model_choice() -> None:
    c = make_client(FakeOpenRouter())
    code, _, _ = turn(c, new_conv(c), "hi", {"kind": "ollama", "model": "qwen3:8b"})
    assert code == 422


def _memory_client(*, ps: dict[str, Any] | None, tags_rows: list[dict[str, Any]]) -> TestClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            if ps is None:
                return httpx.Response(404)
            return httpx.Response(200, json=ps)
        return httpx.Response(200, json={"models": tags_rows})

    app = build_app(llm_mode="ollama", llm_base_url="http://ollama.test:1", llm_model="mine:1b")
    app.state.ollama_probe_transport = httpx.MockTransport(handler)
    client = TestClient(app)
    client.__enter__()
    register(client)
    return client


def test_status_reports_model_sizes_the_host_gave_and_nothing_else() -> None:
    c = _memory_client(
        ps={
            "models": [
                {
                    "name": "mine:1b",
                    "size": 9_000_000_000,
                    "size_vram": 6_000_000_000,
                    "digest": "x",
                }
            ]
        },
        tags_rows=[
            {"name": "mine:1b", "size": 4_000_000_000, "digest": "abc"},
            {"name": "other:3b", "size": 2_000_000_000},
            {"name": "bare:1b"},  # no size reported
        ],
    )
    body = c.get("/v1/chat/status").json()
    rows = {m["name"]: m for m in body["providers"]["ollama"]["models"]}
    # Loaded: the size in memory (what it uses now) and how much of it is in GPU memory.
    assert rows["mine:1b"]["size_bytes"] == 9_000_000_000
    assert rows["mine:1b"]["size_vram_bytes"] == 6_000_000_000
    assert rows["mine:1b"]["loaded"] is True
    # Installed only: the file size, no GPU figure, not loaded.
    assert rows["other:3b"]["size_bytes"] == 2_000_000_000
    assert "size_vram_bytes" not in rows["other:3b"] and "loaded" not in rows["other:3b"]
    # Not reported: absent, never invented.
    assert "size_bytes" not in rows["bare:1b"]
    assert "ollama.test" not in json.dumps(body) and "digest" not in json.dumps(body)


def test_status_survives_a_host_without_ps_and_ignores_nonsense_sizes() -> None:
    c = _memory_client(
        ps=None,
        tags_rows=[
            {"name": "mine:1b", "size": -5},
            {"name": "other:3b", "size": "big"},
            {"name": "true:1b", "size": True},
        ],
    )
    models = c.get("/v1/chat/status").json()["providers"]["ollama"]["models"]
    assert all("size_bytes" not in m for m in models)
