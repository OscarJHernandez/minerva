"""The variation operation through the API: compute endpoint, history, the chat tool and its card form."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from .conftest import build_app, register
from .test_chat import cards_of, sse

HIGHER = {
    "integrand": r"\frac{m}{2}\dot{x}^2+\frac{\alpha}{2}\ddot{x}^2-V(x)",
    "functions": ["x"],
    "given_functions": [{"name": "V", "args": ["x"]}],
    "lower": "a",
    "upper": "b",
}


def test_first_variation_endpoint_returns_the_specified_equation_and_saves_the_request(
    authed: TestClient,
) -> None:
    r = authed.post("/v1/compute/variation", json={**HIGHER, "action": "first"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    eq = body["first"]["equations"][0]
    assert eq["equation_latex"] == r"\alpha x^{(4)} - m \ddot{x} - V'\left(x\right) = 0"
    assert body["first"]["boundary_form_latex"].endswith("_{a}^{b}")
    assert body["first"]["checks"]["deciding"] == "passed"
    (item,) = authed.get("/v1/history").json()
    assert item["operation"] == "variation" and item["id"] == body["calculation_id"]
    assert item["result"]["request"]["integrand"] == HIGHER["integrand"]
    assert "save" not in item["result"]["request"]


def test_numerical_actions_ask_for_what_they_need(authed: TestClient) -> None:
    body = {"integrand": r"\frac{m}{2}\dot{x}^2", "functions": ["x"], "lower": "0", "upper": "1"}
    r = authed.post(
        "/v1/compute/variation",
        json={**body, "action": "evaluate", "paths": [{"analytic": {"x": "t"}}]},
    )
    assert r.json()["status"] == "conditions_not_satisfied" and "m" in r.json()["message"]
    ok = authed.post(
        "/v1/compute/variation",
        json={
            **body,
            "action": "evaluate",
            "values": {"m": "2"},
            "paths": [{"analytic": {"x": "t"}}],
        },
    )
    assert ok.json()["status"] == "ok"
    assert ok.json()["evaluation"]["trials"][0]["action"].startswith("1")


def test_variation_validation_and_authentication(authed: TestClient) -> None:
    assert authed.post("/v1/compute/variation", json={"integrand": "x"}).status_code == 422
    too_many = {"integrand": "x", "functions": ["a", "b", "c", "d", "e"]}
    assert authed.post("/v1/compute/variation", json=too_many).status_code == 422
    long_integrand = {"integrand": "x" * 2500, "functions": ["x"]}
    assert authed.post("/v1/compute/variation", json=long_integrand).status_code == 422
    parse = authed.post(
        "/v1/compute/variation", json={"integrand": r"\frac12\dot{x}^2-V(x)", "functions": ["x"]}
    )
    assert parse.status_code == 200 and parse.json()["status"] == "parse_error"
    with TestClient(build_app()) as anonymous:
        response = anonymous.post(
            "/v1/compute/variation", json={"integrand": "x", "functions": ["x"]}
        )
        assert response.status_code == 401


def test_variation_in_a_derivation_continues_the_sequence(authed: TestClient) -> None:
    derivation = authed.post("/v1/derivations").json()["id"]
    first = authed.post(
        "/v1/compute/variation",
        json={"integrand": r"\frac12\dot{x}^2", "functions": ["x"], "derivation_id": derivation},
    ).json()
    second = authed.post(
        "/v1/compute/variation",
        json={
            "integrand": r"\frac12\dot{x}^2",
            "functions": ["x"],
            "action": "second",
            "derivation_id": derivation,
            "parent_id": first["calculation_id"],
        },
    ).json()
    steps = authed.get(f"/v1/derivations/{derivation}").json()["steps"]
    assert [s["operation"] for s in steps] == ["variation", "variation"]
    assert steps[1]["parent_id"] == first["calculation_id"] and second["status"] == "ok"


def test_variation_logs_no_user_input(capsys) -> None:  # type: ignore[no-untyped-def]
    marker = "zzVARMARKERzz"
    with TestClient(build_app()) as c:
        register(c)
        c.post(
            "/v1/compute/variation",
            json={
                "integrand": f"{marker}*x_dot**2",
                "format": "text",
                "functions": ["x"],
                "values": {marker: "1"},
            },
        )
    out = capsys.readouterr().out
    assert '"variation"' in out and marker not in out


# -- the chat tool ------------------------------------------------------------------------------------
def test_chat_turn_derives_the_variation_and_the_card_form_runs_numbers_without_the_model() -> None:
    app = build_app(llm_mode="scripted")
    with TestClient(app) as client:
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        events = sse(client, cid, r"Find the variation of $\frac12\dot{x}^2-\frac12 x^2$")
        message = next(d for e, d in events if e == "message")
        (card,) = cards_of(message)
        assert card["type"] == "variation" and card["data"]["report"]["status"] == "ok"
        first = card["data"]["report"]["first"]
        assert first["equations"][0]["equation_latex"] == r"- x - \ddot{x} = 0"
        # the model made no numeric answer: the trace holds none
        assert "numeric" not in message["trace"][0]["arguments"]
        request: dict[str, Any] = {
            **card["data"]["request"],
            "lower": "0",
            "upper": "1",
            "action": "evaluate",
            "paths": [{"analytic": {"x": r"\sin(t)/\sin(1)"}}],
        }
        r = client.post(f"/v1/chat/conversations/{cid}/tools/variation", json=request)
        assert r.status_code == 201
        (follow,) = cards_of(r.json()["content"])
        trial = follow["data"]["report"]["evaluation"]["trials"][0]
        assert 0.32 < float(trial["action"]) < 0.322  # (1/2) cot 1
        arguments = r.json()["content"]["trace"][0]["arguments"]
        assert arguments["numeric"]["action"] == "evaluate"


def test_the_model_cannot_run_numerical_actions() -> None:
    from minerva_api.agent.tools import TOOLS, VariationArgs

    properties = VariationArgs.model_json_schema()["properties"]
    for field in ("values", "definitions", "paths", "sweep", "perturbations", "with_second"):
        assert field not in properties
    assert properties["action"]["enum"] == ["analyze", "first", "second"]
    assert "variation" in TOOLS


def test_a_smuggled_numeric_argument_is_ignored_by_the_tool() -> None:
    from minerva_api.agent.tools import ToolContext, execute
    from minerva_engine import Engine, Limits
    from minerva_knowledge import KnowledgeBase

    ctx = ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )
    # `action` outside the model's allowed set is an invalid argument, not a numerical run
    refused = execute(
        "variation",
        {"integrand": r"\frac12\dot{x}^2", "functions": ["x"], "action": "evaluate"},
        ctx,
    )
    assert refused.status == "error" and refused.arguments_valid is False
    smuggled = execute(
        "variation",
        {
            "integrand": r"\frac12\dot{x}^2",
            "functions": ["x"],
            "paths": [{"analytic": {"x": "t"}}],
            "values": {"m": "1"},
            "with_second": True,
        },
        ctx,
    )
    assert smuggled.status == "ok"
    assert smuggled.card.data["report"]["evaluation"] is None
    assert smuggled.card.data["report"]["action"] == "first"


def test_variation_form_endpoint_validation_privacy_and_no_model() -> None:
    body = {"integrand": r"\frac12\dot{x}^2", "functions": ["x"], "lower": "0", "upper": "1"}
    app = build_app(llm_mode="scripted")
    with TestClient(app) as a, TestClient(app) as b:
        register(a, "a@example.com")
        register(b, "b@example.com")
        cid = a.post("/v1/chat/conversations", json={}).json()["id"]
        url = f"/v1/chat/conversations/{cid}/tools/variation"
        assert b.post(url, json=body).status_code == 404
        assert a.post(url, json={**body, "functions": []}).status_code == 422
        assert a.post(url, json=body).status_code == 201
        assert TestClient(app).post(url, json=body).status_code == 401
    with TestClient(build_app()) as c:  # llm_mode off: the form route still works
        register(c)
        cid = c.post("/v1/chat/conversations", json={}).json()["id"]
        assert c.post(f"/v1/chat/conversations/{cid}/tools/variation", json=body).status_code == 201
