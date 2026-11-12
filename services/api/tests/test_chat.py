import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from .conftest import PASSWORD, build_app, register


def cards_of(message: dict) -> list[dict]:  # type: ignore[type-arg]
    return [b["card"] for b in message["blocks"] if b["kind"] == "card"]


def segments_of(message: dict) -> list[dict]:  # type: ignore[type-arg]
    return [s for b in message["blocks"] if b["kind"] == "text" for s in b["segments"]]


def sse(client: TestClient, conversation_id: str, content: str) -> list[tuple[str, dict]]:  # type: ignore[type-arg]
    events: list[tuple[str, dict]] = []  # type: ignore[type-arg]
    with client.stream(
        "POST", f"/v1/chat/conversations/{conversation_id}/messages", json={"content": content}
    ) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("text/event-stream")
        event = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                events.append((event, json.loads(line[6:])))
    return events


@pytest.fixture
def chat_client() -> Iterator[tuple[TestClient, str]]:
    app = build_app(llm_mode="scripted")
    with TestClient(app) as client:
        register(client)
        conv = client.post("/v1/chat/conversations", json={}).json()
        yield client, conv["id"]


def test_chat_is_unavailable_without_a_model() -> None:
    with TestClient(build_app()) as c:
        register(c)
        assert c.get("/v1/chat/status").json()["enabled"] is False
        conv = c.post("/v1/chat/conversations", json={}).json()
        r = c.post(f"/v1/chat/conversations/{conv['id']}/messages", json={"content": "hi"})
        assert r.status_code == 503


def test_chat_requires_login() -> None:
    with TestClient(build_app(llm_mode="scripted")) as c:
        assert c.get("/v1/chat/conversations").status_code == 401
        assert c.get("/v1/chat/status").status_code == 200  # which mode is not secret


def test_streamed_turn_with_a_tool_card_and_verified_formula(
    chat_client: tuple[TestClient, str],
) -> None:
    client, cid = chat_client
    events = sse(client, cid, r"simplify $\frac{x^2-1}{x-1}$")
    names = [e for e, _ in events]
    assert names[0] == "text" and "tool_call" in names and "card" in names
    assert names[-2:] == ["message", "done"]
    message = next(d for e, d in events if e == "message")
    assert cards_of(message)[0]["type"] == "simplify"
    assert cards_of(message)[0]["data"]["result_text"] == "x + 1"
    formulas = [s for s in segments_of(message) if s["type"] == "display_math"]
    assert formulas and formulas[0]["value"] == "x + 1" and formulas[0]["status"] == "from_tool"
    detail = client.get(f"/v1/chat/conversations/{cid}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["title"].startswith("simplify")


def test_differentiate_turn_streams_a_verified_card(chat_client: tuple[TestClient, str]) -> None:
    client, cid = chat_client
    events = sse(client, cid, r"Give me the second derivative of $x^5$")
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    assert card["type"] == "differentiate"
    assert card["data"]["status"] == "ok"
    assert card["data"]["result_text"] == "20*x**3"
    assert card["data"]["verification"]["passed"] is True


def test_model_written_formula_is_marked_unchecked(chat_client: tuple[TestClient, str]) -> None:
    client, cid = chat_client
    events = sse(client, cid, "just tell me the answer without tools")
    message = next(d for e, d in events if e == "message")
    assert cards_of(message) == []
    formulas = [s for s in segments_of(message) if s["type"] == "math"]
    assert formulas and all(s["status"] == "unchecked" for s in formulas)


def test_identities_conversation_and_followup_history(chat_client: tuple[TestClient, str]) -> None:
    client, cid = chat_client
    first = sse(client, cid, r"which identities split $\log(xy)$ if x, y positive")
    assert next(d for e, d in first if e == "card")["type"] == "identities"
    sse(client, cid, "thanks, what else could I do?")
    detail = client.get(f"/v1/chat/conversations/{cid}").json()
    assert len(detail["messages"]) == 4


def test_conversations_are_private_to_their_owner() -> None:
    app = build_app(llm_mode="scripted")
    with TestClient(app) as a, TestClient(app) as b:
        register(a, "a@example.com")
        register(b, "b@example.com")
        cid = a.post("/v1/chat/conversations", json={}).json()["id"]
        sse(a, cid, "hello")
        assert [c["id"] for c in a.get("/v1/chat/conversations").json()] == [cid]
        assert b.get("/v1/chat/conversations").json() == []
        assert b.get(f"/v1/chat/conversations/{cid}").status_code == 404
        assert b.delete(f"/v1/chat/conversations/{cid}").status_code == 404
        assert (
            b.post(f"/v1/chat/conversations/{cid}/messages", json={"content": "x"}).status_code
            == 404
        )
        assert a.delete(f"/v1/chat/conversations/{cid}").status_code == 204
        assert a.get("/v1/chat/conversations").json() == []


def test_message_validation_and_rate_limit() -> None:
    app = build_app(llm_mode="scripted", chat_rate_per_minute=2)
    with TestClient(app) as c:
        register(c)
        cid = c.post("/v1/chat/conversations", json={}).json()["id"]
        url = f"/v1/chat/conversations/{cid}/messages"
        assert c.post(url, json={"content": ""}).status_code == 422
        assert c.post(url, json={"content": "x" * 5000}).status_code == 422
        codes = [c.post(url, json={"content": "hello"}).status_code for _ in range(4)]
        assert 429 in codes


def test_chat_logs_contain_no_message_content(capsys) -> None:  # type: ignore[no-untyped-def]
    marker = "zzCHATMARKERzz"
    app = build_app(llm_mode="scripted")
    with TestClient(app) as c:
        register(c)
        cid = c.post("/v1/chat/conversations", json={}).json()["id"]
        sse(c, cid, f"simplify ${marker}$ + {marker}")
    out = capsys.readouterr().out
    assert "chat_turn" in out and marker not in out and PASSWORD not in out


def test_scripted_model_is_refused_outside_development() -> None:
    import pytest
    from minerva_api.settings import Settings
    from pydantic import SecretStr

    with pytest.raises(RuntimeError, match="development only"):
        Settings(
            _env_file=None,
            env="production",
            cookie_secure=True,
            session_secret=SecretStr("x" * 40),
            llm_mode="scripted",
        ).validate_for_runtime()
    with pytest.raises(RuntimeError, match="needs MINERVA_LLM_BASE_URL"):
        Settings(_env_file=None, llm_mode="ollama").validate_for_runtime()
