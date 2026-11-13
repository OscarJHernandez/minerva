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
    assert not any(e == "message" for e, _ in events)


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
