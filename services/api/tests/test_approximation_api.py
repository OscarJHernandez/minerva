"""The approximation operation through the API: compute endpoint, history, the chat tool and its card form."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from minerva_api.agent.tools import TOOLS, ApproximationArgs, ToolContext, execute

from .conftest import build_app, register
from .test_chat import cards_of, sse

PADE = {
    "input": "exp(x)",
    "format": "text",
    "method": "pade",
    "point": "0",
    "degree": 1,
    "denominator_degree": 1,
}


def test_pade_endpoint_returns_the_approximant_with_its_warning_and_saves_the_request(
    authed: TestClient,
) -> None:
    r = authed.post("/v1/compute/approximation", json=PADE)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["method"] == "pade"
    assert body["result_latex"] == r"\frac{1 + \frac{x}{2}}{1 - \frac{x}{2}}"
    assert "spurious_pole" in body["warnings"]
    assert all(c["passed"] for c in body["checks"] if c["passed"] is not None)
    (item,) = authed.get("/v1/history").json()
    assert item["operation"] == "approximation" and item["id"] == body["calculation_id"]
    assert item["result"]["request"]["method"] == "pade"
    assert "save" not in item["result"]["request"]


def test_a_missing_regime_is_a_question_not_a_guess(authed: TestClient) -> None:
    r = authed.post(
        "/v1/compute/approximation",
        json={"input": "sin(x)", "format": "text", "method": "chebyshev"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "conditions_not_satisfied" and body["reason"] == "domain_unspecified"
    assert body["questions"] and body["result_latex"] is None


def test_a_list_of_variables_for_a_method_of_one_variable_is_refused(authed: TestClient) -> None:
    r = authed.post(
        "/v1/compute/approximation",
        json={
            "input": "cos(a - b)",
            "format": "text",
            "method": "taylor",
            "variables": ["a", "b"],
            "point": "0",
            "order": 2,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "conditions_not_satisfied"
    assert body["reason"] == "fields_not_for_method" and body["result_latex"] is None


def test_interval_methods_report_measurements_not_bounds(authed: TestClient) -> None:
    r = authed.post(
        "/v1/compute/approximation",
        json={
            "input": "sin(x)",
            "format": "text",
            "method": "chebyshev",
            "lower": "0",
            "upper": "3",
            "tolerance": "1e-6",
            "tolerance_kind": "absolute",
        },
    )
    body = r.json()
    assert body["status"] == "ok" and body["tolerance_met"] is True
    assert body["error"]["samples"] > 1000 and "not a bound" in body["message"]
    assert body["evidence"] == "numeric_estimate"
    assert any(c["chosen"] for c in body["candidates"])


def test_approximation_validation_and_authentication(authed: TestClient) -> None:
    assert (
        authed.post("/v1/compute/approximation", json={"input": "x", "method": "magic"}).status_code
        == 422
    )
    assert authed.post("/v1/compute/approximation", json={"method": "taylor"}).status_code == 422
    assert (
        authed.post("/v1/compute/approximation", json={"input": "x", "order": 999}).status_code
        == 422
    )
    assert authed.post("/v1/compute/approximation", json={"input": "x" * 25000}).status_code == 422
    bad = authed.post(
        "/v1/compute/approximation",
        json={"input": r"\frac{1", "method": "taylor", "point": "0", "order": 2},
    )
    assert bad.status_code == 200 and bad.json()["status"] == "parse_error"
    with TestClient(build_app()) as anonymous:
        assert anonymous.post("/v1/compute/approximation", json=PADE).status_code == 401


def test_approximation_in_a_derivation_continues_the_sequence(authed: TestClient) -> None:
    derivation = authed.post("/v1/derivations", json={}).json()["id"]
    first = authed.post(
        "/v1/compute/approximation", json={**PADE, "derivation_id": derivation}
    ).json()
    second = authed.post(
        "/v1/compute/approximation",
        json={
            **PADE,
            "degree": 2,
            "denominator_degree": 2,
            "derivation_id": derivation,
            "parent_id": first["calculation_id"],
        },
    ).json()
    assert second["status"] == "ok"
    steps = authed.get(f"/v1/derivations/{derivation}").json()["steps"]
    assert [s["operation"] for s in steps] == ["approximation", "approximation"]


def test_approximation_logs_no_user_input(capsys) -> None:  # type: ignore[no-untyped-def]
    marker = "zzAPPROXzz"
    with TestClient(build_app()) as c:
        register(c)
        c.post(
            "/v1/compute/approximation",
            json={
                "input": f"{marker}*exp(x)",
                "format": "text",
                "method": "taylor",
                "point": "0",
                "order": 3,
                "assumptions": {marker: ["positive"]},
            },
        )
        c.post(
            "/v1/compute/approximation",
            json={**PADE, "tolerance": "1e-9", "tolerance_kind": "absolute"},
        )
    out = capsys.readouterr().out
    assert '"approximation"' in out and marker not in out and "1e-9" not in out
    assert '"approx_method"' in out


# -- the chat tool ------------------------------------------------------------------------------------
def test_chat_turn_finds_the_pade_approximant_and_the_card_form_adds_the_users_numbers() -> None:
    app = build_app(llm_mode="scripted")
    with TestClient(app) as client:
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        events = sse(client, cid, r"Approximate $\exp(x)$ with a [1/1] Pade approximant about 0")
        message = next(d for e, d in events if e == "message")
        (card,) = cards_of(message)
        assert card["type"] == "approximation" and card["tool"] == "approximation"
        report = card["data"]["report"]
        assert (
            report["status"] == "ok"
            and report["result_latex"] == r"\frac{1 + \frac{x}{2}}{1 - \frac{x}{2}}"
        )
        # the model passed only what the user wrote: no interval, no tolerance
        arguments = message["trace"][0]["arguments"]
        assert arguments.get("lower") is None and arguments.get("tolerance") is None
        request: dict[str, Any] = {**card["data"]["request"], "lower": "-1", "upper": "3"}
        r = client.post(f"/v1/chat/conversations/{cid}/tools/approximation", json=request)
        assert r.status_code == 201
        (follow,) = cards_of(r.json()["content"])
        assert "pole_in_interval" in follow["data"]["report"]["warnings"]
        assert follow["data"]["request"]["upper"] == "3"


def test_chat_turn_without_the_regime_gets_the_tools_question() -> None:
    app = build_app(llm_mode="scripted")
    with TestClient(app) as client:
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        events = sse(client, cid, r"Give a Chebyshev approximation of $\sin x$")
        message = next(d for e, d in events if e == "message")
        (card,) = cards_of(message)
        assert card["data"]["report"]["status"] == "conditions_not_satisfied"
        assert card["data"]["report"]["questions"]
        assert card["data"]["report"]["result_latex"] is None


def test_the_tool_group_loads_on_approximation_words_only() -> None:
    from minerva_api.agent import toolsets

    assert toolsets.groups_in_text("find a Pade approximant of exp") == ["approximation"]
    assert "approximation" in toolsets.groups_in_text(
        "what is the large distance behaviour of 1/(r+h)"
    )
    assert toolsets.groups_in_text("differentiate x^2") == []
    assert "approximation" in toolsets.GROUP_TOOLS["approximation"]


def test_every_user_owned_number_tells_the_model_not_to_invent_it() -> None:
    schema = ApproximationArgs.model_json_schema()["properties"]
    for name in (
        "point",
        "order",
        "small",
        "lower",
        "upper",
        "degree",
        "denominator_degree",
        "period",
        "modes",
        "segments",
        "tolerance",
        "tolerance_kind",
        "max_terms",
    ):
        assert "never invent" in schema[name]["description"], name
    assert "latex" in ApproximationArgs.model_json_schema()["required"]
    assert TOOLS["approximation"].schema()["function"]["parameters"]["type"] == "object"


def test_the_tool_refuses_an_expression_about_a_declared_matrix(
    ctx: ToolContext | None = None,
) -> None:
    from minerva_engine import Engine, Limits
    from minerva_knowledge import KnowledgeBase  # noqa: F401

    context = ToolContext(
        engine=Engine(Limits(timeout_seconds=20, max_concurrency=2)),
        kb=None,
        matrices=frozenset({"A"}),
    )  # type: ignore[arg-type]
    out = execute(
        "approximation",
        {"latex": "A^2", "method": "taylor", "variable": "A", "point": "0", "order": 2},
        context,
    )
    assert out.card.type == "error" and out.status != "ok"


ISSUE_137_CALL = {
    "latex": r"m_2 l_1 l_2 \dot{\theta}_1 \dot{\theta}_2 \cos(\theta_1-\theta_2) + m_2 g l_2 \cos\theta_2",
    "method": "taylor",
    "variables": ["theta_1", "theta_2"],
    "point": "0",
    "order": 2,
}


def _context() -> ToolContext:
    from minerva_engine import Engine, Limits

    return ToolContext(engine=Engine(Limits(timeout_seconds=20, max_concurrency=2)), kb=None)  # type: ignore[arg-type]


def test_the_tool_tells_the_model_why_taylor_with_a_list_of_variables_was_not_run() -> None:
    """Issue 137: this call was `ok` with the expression unchanged, in the default variable x."""
    out = execute("approximation", dict(ISSUE_137_CALL), _context())
    assert out.status == "conditions_not_satisfied"
    assert "reason=fields_not_for_method" in out.for_model
    assert "multivariable" in out.for_model and "approximation=" not in out.for_model
    assert out.known_latex == []
    assert out.card.data["report"]["result_latex"] is None


def test_the_tool_refuses_a_variable_the_expression_does_not_contain() -> None:
    call = {k: v for k, v in ISSUE_137_CALL.items() if k != "variables"}  # variable defaults to x
    out = execute("approximation", call, _context())
    assert out.status == "unsupported" and "reason=variable_missing" in out.for_model
    assert out.known_latex == []


def test_the_tool_expands_in_both_angles_when_asked_with_one_point_each() -> None:
    call = {**ISSUE_137_CALL, "method": "multivariable", "points": ["0", "0"]}
    del call["point"]
    out = execute("approximation", call, _context())
    assert out.status == "ok" and "approximation=" in out.for_model
    assert all(c["passed"] for c in out.card.data["report"]["checks"])


def test_the_card_form_gets_the_same_refusal() -> None:
    app = build_app(llm_mode="scripted")
    with TestClient(app) as client:
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        r = client.post(f"/v1/chat/conversations/{cid}/tools/approximation", json=ISSUE_137_CALL)
        assert r.status_code == 201
        (card,) = cards_of(r.json()["content"])
        assert card["data"]["report"]["reason"] == "fields_not_for_method"
        assert card["data"]["report"]["result_latex"] is None
