"""Creative telescoping through the API: compute endpoint, logging, the chat tool and its card."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from minerva_api.agent.tools import TOOLS

from .conftest import build_app, register
from .test_chat import cards_of, sse

GAUSS: dict[str, Any] = {
    "input": r"\exp(-a x^2)",
    "variable": "x",
    "parameter": "a",
    "lower": "-oo",
    "upper": "oo",
    "assumptions": {"a": ["positive"], "x": ["real"]},
}


def test_endpoint_derives_the_equation_and_the_value(authed: TestClient) -> None:
    r = authed.post("/v1/compute/telescoping", json=GAUSS)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["outcome"] == "evaluated"
    assert body["equation"]["homogeneous"] is True
    assert body["certificate_check"]["symbolic_zero"] is True
    assert body["result_latex"] == r"\frac{\sqrt{\pi}}{\sqrt{a}}"
    assert body["value_verification"]["passed"] is True


def test_without_declared_signs_only_the_certificate_is_returned(authed: TestClient) -> None:
    body = {**GAUSS, "assumptions": {"x": ["real"]}}
    r = authed.post("/v1/compute/telescoping", json=body).json()
    assert r["status"] == "conditions_not_satisfied" and r["outcome"] == "certificate"
    assert r["certificate_check"]["passed"] and r["result_latex"] is None


def test_validation_and_authentication(authed: TestClient) -> None:
    assert authed.post("/v1/compute/telescoping", json={"input": "x"}).status_code == 422
    assert (
        authed.post("/v1/compute/telescoping", json={**GAUSS, "max_order": 99}).status_code == 422
    )
    assert (
        authed.post("/v1/compute/telescoping", json={**GAUSS, "input": "x" * 2500}).status_code
        == 422
    )
    one_bound = authed.post("/v1/compute/telescoping", json={**GAUSS, "upper": None})
    assert one_bound.json()["status"] == "parse_error"
    with TestClient(build_app()) as anonymous:
        assert anonymous.post("/v1/compute/telescoping", json=GAUSS).status_code == 401


def test_telescoping_logs_no_user_input(capsys) -> None:  # type: ignore[no-untyped-def]
    marker = "zzTELMARKERzz"
    with TestClient(build_app()) as c:
        register(c)
        c.post(
            "/v1/compute/telescoping",
            json={
                "input": f"exp(-{marker}*x**2)",
                "format": "text",
                "variable": "x",
                "parameter": marker,
                "lower": "0",
                "upper": "oo",
                "assumptions": {marker: ["positive"], "x": ["real"]},
            },
        )
    out = capsys.readouterr().out
    assert '"telescoping"' in out and marker not in out


def test_tool_schema_has_no_field_for_what_only_the_engine_may_produce() -> None:
    fields = set(TOOLS["creative_telescoping"].args_model.model_fields)
    assert fields == {"latex", "assumptions", "variable", "parameter", "lower", "upper"}


def test_chat_turn_uses_the_tool_and_a_partial_result_is_not_a_value() -> None:
    app = build_app(llm_mode="scripted")
    with TestClient(app) as client:
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        text = r"Use creative telescoping on $\exp(-a x^2)$ with parameter a from -oo to oo"
        events = sse(client, cid, text)
        message = next(d for e, d in events if e == "message")
        (card,) = cards_of(message)
        assert card["type"] == "telescoping"
        assert card["data"]["outcome"] == "evaluated"
        assert card["data"]["result_latex"] == r"\frac{\sqrt{\pi}}{\sqrt{a}}"
        assert message["trace"][0]["name"] == "creative_telescoping"
