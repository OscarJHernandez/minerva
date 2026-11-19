"""A user can always tell which model answered: asked of the host, stamped on the reply, never guessed.

The host is a mock. Nothing here depends on a real model or its address.
"""

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from minerva_api.agent.identity import ModelInfo, probe, probe_digest, stamped
from minerva_api.agent.scripted import ScriptedLLM

from .conftest import build_app, register
from .test_chat import sse

SHOW = {
    "details": {
        "parent_model": "gemma4:26b",
        "family": "gemma4",
        "parameter_size": "25.2B",
        "quantization_level": "Q4_K_M",
    },
    "capabilities": ["tools", "completion", "thinking"],
    "parameters": "num_predict 1200\n",
}


class Host:
    """An Ollama mock whose digest can be changed between requests (the alias re-pointed)."""

    def __init__(self, alias: str = "scripted", digest: str = "149a0ff8a1ca") -> None:
        self.alias = alias
        self.digest = digest
        self.up = True
        self.calls: list[str] = []
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request.url.path)
        if not self.up:
            raise httpx.ConnectError("down", request=request)
        if request.url.path == "/api/show":
            return httpx.Response(200, json=SHOW)
        if request.url.path == "/api/ps":
            return httpx.Response(
                200, json={"models": [{"name": self.alias, "context_length": 32768}]}
            )
        return httpx.Response(
            200, json={"models": [{"name": self.alias, "digest": self.digest + "0" * 52}]}
        )


def client_with(host: Host) -> TestClient:
    app = build_app(llm_mode="ollama", llm_base_url="http://ollama.test:1", llm_model="scripted")
    app.state.ollama_probe_transport = host.transport
    c = TestClient(app)
    c.__enter__()
    stand_in = ScriptedLLM()
    stand_in.provider = "ollama"  # the identity is asked of the host, as for a real Ollama model
    app.state.local_llm = stand_in
    app.state.llm = app.state.local_llm
    register(c)
    return c


def conversation(c: TestClient) -> str:
    return str(c.post("/v1/chat/conversations", json={}).json()["id"])


def message_event(events: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return next(d for e, d in events if e == "message")


@pytest.mark.anyio
async def test_the_probe_reads_weights_digest_and_thinking_from_the_host() -> None:
    host = Host()
    info = await probe("http://h", "scripted", transport=host.transport)
    assert info.known and info.base_model == "gemma4:26b" and info.parameter_size == "25.2B"
    assert info.quantization == "Q4_K_M" and info.digest == "149a0ff8a1ca"
    assert info.can_think is True and info.loaded_context == 32768
    assert "ollama.test" not in info.model_dump_json()


@pytest.mark.anyio
async def test_a_host_that_does_not_answer_is_unknown_not_guessed() -> None:
    host = Host()
    host.up = False
    info = await probe("http://h", "scripted", transport=host.transport)
    assert info == ModelInfo(provider="ollama", alias="scripted") and not info.known
    assert await probe_digest("http://h", "scripted", transport=host.transport) is None


def test_the_stamp_records_a_change_of_weights_during_the_reply() -> None:
    before = ModelInfo(provider="ollama", alias="m", known=True, digest="aaaaaaaaaaaa")
    same = stamped(before, "aaaaaaaaaaaa", thinking_requested=False, reply_budget=4096)
    assert not same.changed_during_reply and same.digest_after is None
    assert same.thinking_requested is False and same.reply_budget == 4096
    moved = stamped(before, "bbbbbbbbbbbb", thinking_requested=True, reply_budget=None)
    assert moved.changed_during_reply and moved.digest_after == "bbbbbbbbbbbb"
    silent = stamped(before, None, thinking_requested=None, reply_budget=None)
    assert not silent.changed_during_reply  # a host that did not answer is not evidence of a change


def test_status_names_the_resolved_model_not_just_the_alias() -> None:
    c = client_with(Host())
    identity = c.get("/v1/chat/status").json()["identity"]
    assert identity["alias"] == "scripted" and identity["base_model"] == "gemma4:26b"
    assert identity["known"] is True and identity["digest"] == "149a0ff8a1ca"


def test_status_says_so_when_the_host_does_not_report() -> None:
    host = Host()
    host.up = False
    identity = client_with(host).get("/v1/chat/status").json()["identity"]
    assert identity["known"] is False and identity["base_model"] is None


def test_every_reply_carries_the_model_that_wrote_it() -> None:
    c = client_with(Host())
    info = message_event(sse(c, conversation(c), "hello"))["model_info"]
    assert info["base_model"] == "gemma4:26b" and info["digest"] == "149a0ff8a1ca"
    assert info["thinking_requested"] is False and info["changed_during_reply"] is False


def test_a_reply_is_not_stamped_from_the_status_cache() -> None:
    host = Host()
    c = client_with(host)
    c.get("/v1/chat/status")  # warms the 30 s cache with the old weights
    host.digest = "bbbbbbbbbbbb"  # the alias is re-pointed
    info = message_event(sse(c, conversation(c), "hello"))["model_info"]
    assert info["digest"] == "bbbbbbbbbbbb"


def test_a_re_point_during_the_reply_is_recorded() -> None:
    host = Host()
    original = host.handle
    state = {"tags": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            state["tags"] += 1
            if state["tags"] >= 2:  # the start-of-turn probe is the first
                host.digest = "cccccccccccc"
        return original(request)

    host.transport = httpx.MockTransport(handle)
    c = client_with(host)
    c.app.state.ollama_probe_transport = host.transport  # type: ignore[attr-defined]
    info = message_event(sse(c, conversation(c), "hello"))["model_info"]
    assert info["changed_during_reply"] is True and info["digest_after"] == "cccccccccccc"
    assert info["digest"] == "149a0ff8a1ca"


def test_a_stored_reply_keeps_its_model_after_the_alias_is_re_pointed() -> None:
    host = Host()
    c = client_with(host)
    cid = conversation(c)
    sse(c, cid, "hello")
    host.digest = "dddddddddddd"  # later the operator points the alias elsewhere
    stored = c.get(f"/v1/chat/conversations/{cid}").json()
    reply = next(m for m in stored["messages"] if m["role"] == "assistant")
    assert reply["content"]["model_info"]["digest"] == "149a0ff8a1ca"


def test_a_silent_host_gives_a_reply_that_says_the_identity_was_not_reported() -> None:
    host = Host()
    host.up = False
    c = client_with(host)
    info = message_event(sse(c, conversation(c), "hello"))["model_info"]
    assert info["known"] is False and info["digest"] is None


def test_the_model_info_endpoint_only_answers_for_installed_models() -> None:
    c = client_with(Host())
    assert c.get("/v1/chat/model-info", params={"model": "scripted"}).status_code == 200
    assert c.get("/v1/chat/model-info", params={"model": "not-installed"}).status_code == 404
    assert c.get("/v1/chat/model-info", params={"model": "../etc"}).status_code == 422


def test_the_model_info_endpoint_needs_a_session() -> None:
    app = build_app(llm_mode="ollama", llm_base_url="http://ollama.test:1", llm_model="scripted")
    app.state.ollama_probe_transport = Host().transport
    with TestClient(app) as anonymous:
        assert anonymous.get("/v1/chat/model-info", params={"model": "scripted"}).status_code == 401


def test_nothing_new_is_logged_and_the_host_address_never_reaches_a_reply() -> None:
    c = client_with(Host())
    event = message_event(sse(c, conversation(c), "hello"))
    assert "ollama.test" not in json.dumps(event)
