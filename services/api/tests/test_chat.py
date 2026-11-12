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


def test_series_turn_streams_a_verified_card(chat_client: tuple[TestClient, str]) -> None:
    client, cid = chat_client
    events = sse(client, cid, r"Taylor series of $\exponentialE^x$ about 0 to order 3")
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    assert card["type"] == "series"
    assert card["data"]["status"] == "ok"
    assert card["data"]["verification"]["coefficients_passed"] is True
    assert card["data"]["verification"]["remainder_passed"] is True
    assert card["data"]["order"] == 3


def test_expansion_turn_streams_a_verified_card(chat_client: tuple[TestClient, str]) -> None:
    client, cid = chat_client
    events = sse(
        client,
        cid,
        r"Expand about the equilibrium to order 3: $\frac12 \dot{x}^2 - \frac12 x^2 - \frac14 b x^4$",
    )
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    assert card["type"] == "expansion"
    report = card["data"]["report"]
    assert report["status"] == "ok" and report["order"] == 3
    assert report["equations"][0]["verification"]["passed"] is True


def test_expansion_turn_about_a_non_equilibrium_is_refused(
    chat_client: tuple[TestClient, str],
) -> None:
    client, cid = chat_client
    events = sse(
        client,
        cid,
        r"Expand about the equilibrium at x = 1: $\frac12 \dot{x}^2 - \frac12 x^2$",
    )
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    assert card["data"]["report"]["status"] == "unsupported"
    assert card["data"]["report"]["equations"] == []


def test_perturbation_turn_streams_a_verified_card_and_does_not_invent_numbers(
    chat_client: tuple[TestClient, str],
) -> None:
    client, cid = chat_client
    events = sse(
        client,
        cid,
        r"Perturbation theory to order 2, small parameter b: $\frac12 \dot{x}^2 - \frac12 x^2 - \frac14 b x^4$",
    )
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    assert card["type"] == "perturbation"
    report = card["data"]["report"]
    assert report["verification"]["passed"] is True
    assert report["comparison"] is None  # no numbers from the user, so no comparison was run


def test_perturbation_turn_compares_only_with_numbers_taken_from_the_message(
    chat_client: tuple[TestClient, str],
) -> None:
    client, cid = chat_client
    events = sse(
        client,
        cid,
        "Perturbation theory to order 2, small parameter b, compare epsilon=0.1 amplitude=1 t_end=20: "
        r"$\frac12 \dot{x}^2 - \frac12 x^2 - \frac14 b x^4$",
    )
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    comparison = card["data"]["report"]["comparison"]
    assert comparison["status"] == "ok" and comparison["epsilon"] == 0.1
    assert len(comparison["t"]) == len(comparison["numerical"]) == len(comparison["perturbative"])


def test_series_turn_about_a_pole_shows_no_formula(chat_client: tuple[TestClient, str]) -> None:
    client, cid = chat_client
    events = sse(client, cid, r"Taylor series of $\frac{1}{x}$ about 0")
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    assert card["type"] == "series"
    assert card["data"]["status"] == "unsupported"
    assert card["data"]["result_latex"] is None


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


def test_assistant_message_stores_a_debug_record(chat_client: tuple[TestClient, str]) -> None:
    client, cid = chat_client
    with client.stream(
        "POST",
        f"/v1/chat/conversations/{cid}/messages",
        json={"content": r"simplify $\frac{x^2-1}{x-1}$"},
    ) as response:
        request_id = response.headers["x-request-id"]
        response.read()
    stored = client.get(f"/v1/chat/conversations/{cid}").json()["messages"][-1]["content"]
    debug = stored["debug"]
    assert debug["request_id"] == request_id
    assert debug["model"] == "scripted" and debug["mode"] == "scripted"
    assert debug["api_version"] and len(debug["prompt_sha"]) == 12
    assert len(debug["llm_calls"]) >= 2
    assert stored["trace"][0]["name"] == "simplify" and stored["trace"][0]["status"] == "ok"


def test_messages_stored_before_debug_existed_still_work() -> None:
    from minerva_api.routers_chat import _history

    class Row:
        role = "assistant"
        content = {  # noqa: RUF012
            "text": "ok",
            "trace": [{"id": "a", "name": "simplify", "arguments": {}, "for_model": "r=$x$"}],
        }

    history, known = _history([Row()])  # type: ignore[list-item]
    assert history[1]["content"] == "r=$x$" and "x" in known


def test_chat_logs_contain_no_message_content(capsys) -> None:  # type: ignore[no-untyped-def]
    marker = "zzCHATMARKERzz"
    app = build_app(llm_mode="scripted")
    with TestClient(app) as c:
        register(c)
        cid = c.post("/v1/chat/conversations", json={}).json()["id"]
        sse(c, cid, f"simplify ${marker}$ + {marker}")
    out = capsys.readouterr().out
    assert "chat_turn" in out and marker not in out and PASSWORD not in out
    assert '"tool_calls": 1' in out and '"steps"' in out


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


MOTION_BODY = {
    "lagrangian": r"\frac12 m\dot{x}^2 - \frac12 k x^2",
    "coordinates": ["x"],
    "parameters": {"m": 1.0, "k": 4.0},
    "initial_positions": [1.0],
    "initial_velocities": [0.0],
    "t_end": 5.0,
}


def test_motion_form_endpoint_stores_a_card_message_that_later_turns_can_read(
    chat_client: tuple[TestClient, str],
) -> None:
    client, cid = chat_client
    url = f"/v1/chat/conversations/{cid}/tools/integrate_motion"
    r = client.post(url, json=MOTION_BODY)
    assert r.status_code == 201
    message = r.json()["content"]
    card = cards_of(message)[0]
    assert card["type"] == "motion" and card["data"]["report"]["runs"][0]["status"] == "ok"
    assert message["trace"][0]["name"] == "integrate_motion"
    assert message["trace"][0]["arguments"]["parameters"] == {"m": 1.0, "k": 4.0}
    detail = client.get(f"/v1/chat/conversations/{cid}").json()
    assert [m["role"] for m in detail["messages"]] == ["assistant"]
    # a following turn replays the tool call and its compact result as history
    sse(client, cid, "thanks")
    assert len(client.get(f"/v1/chat/conversations/{cid}").json()["messages"]) == 3


def test_motion_form_endpoint_works_without_a_chat_model() -> None:
    with TestClient(build_app()) as c:  # llm_mode off
        register(c)
        cid = c.post("/v1/chat/conversations", json={}).json()["id"]
        r = c.post(f"/v1/chat/conversations/{cid}/tools/integrate_motion", json=MOTION_BODY)
        assert r.status_code == 201


def test_motion_form_endpoint_validation_privacy_and_rate_limit() -> None:
    app = build_app(llm_mode="scripted", compute_rate_per_minute=2)
    with TestClient(app) as a, TestClient(app) as b:
        register(a, "a@example.com")
        register(b, "b@example.com")
        cid = a.post("/v1/chat/conversations", json={}).json()["id"]
        url = f"/v1/chat/conversations/{cid}/tools/integrate_motion"
        assert b.post(url, json=MOTION_BODY).status_code == 404
        bad = {**MOTION_BODY, "initial_positions": []}
        assert a.post(url, json=bad).status_code == 422
        missing = {**MOTION_BODY, "parameters": {"m": 1.0}}
        r = a.post(url, json=missing)  # a bad value is a stored card, not a server error
        assert r.status_code == 201
        assert cards_of(r.json()["content"])[0]["data"]["report"]["status"] == "parse_error"
        codes = [a.post(url, json=MOTION_BODY).status_code for _ in range(3)]
        assert 429 in codes
        assert TestClient(app).post(url, json=MOTION_BODY).status_code == 401  # not logged in


def test_scripted_model_runs_a_numerical_solution_and_a_sweep(
    chat_client: tuple[TestClient, str],
) -> None:
    client, cid = chat_client
    events = sse(client, cid, r"solve $\frac12 m\dot{x}^2 - \frac12 k x^2$ numerically, sweep k")
    message = next(d for e, d in events if e == "message")
    report = cards_of(message)[0]["data"]["report"]
    assert cards_of(message)[0]["type"] == "motion" and len(report["runs"]) == 3


def test_scripted_model_runs_dimensional_analysis_and_a_unit_check(
    chat_client: tuple[TestClient, str],
) -> None:
    client, cid = chat_client
    events = sse(
        client,
        cid,
        "Dimensional analysis: E [J], m [kg], unknown E",
    )
    message = next(d for e, d in events if e == "message")
    card = cards_of(message)[0]
    assert card["type"] == "dimensional" and card["data"]["report"]["closure"] == "missing"
    events = sse(client, cid, r"Check the units of $F = m a$: F [N], m [kg], a [m/s^2]")
    message = next(d for e, d in events if e == "message")
    card = cards_of(message)[0]
    assert card["type"] == "units" and card["data"]["report"]["consistent"] is True


def test_integrate_turn_streams_a_verified_card(chat_client: tuple[TestClient, str]) -> None:
    client, cid = chat_client
    events = sse(client, cid, r"Integrate $x\cos x$")
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    assert card["type"] == "integrate"
    assert card["data"]["status"] == "ok"
    assert card["data"]["verification"]["passed"] is True


def test_definite_integral_turn_streams_a_verified_card(
    chat_client: tuple[TestClient, str],
) -> None:
    client, cid = chat_client
    events = sse(client, cid, r"Integrate $x^2$ from 0 to 3")
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    assert card["type"] == "integrate"
    assert card["data"]["definite"] is True
    assert card["data"]["result_text"] == "9"
    assert card["data"]["quadrature"]["passed"] is True
