"""Vector calculus through the API: the compute route, the agent tool, claims, logs and history."""

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from minerva_api.agent.tools import SCALAR_TOOLS, TOOLS, ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

from .conftest import build_app, register
from .test_chat import cards_of, segments_of, sse

CURL_A = {
    "expression": r"\nabla\times\mathbf{A}",
    "coordinates": "cylindrical",
    "fields": [{"name": "A", "kind": "vector"}],
}


@pytest.fixture(scope="module")
def shared() -> tuple[Engine, KnowledgeBase]:
    return Engine(Limits(timeout_seconds=30, max_concurrency=2)), KnowledgeBase.load()


@pytest.fixture
def ctx(shared: tuple[Engine, KnowledgeBase]) -> ToolContext:
    return ToolContext(engine=shared[0], kb=shared[1])


@pytest.fixture
def chat_client() -> Iterator[tuple[TestClient, str]]:
    app = build_app(llm_mode="scripted")
    with TestClient(app) as client:
        register(client)
        yield client, client.post("/v1/chat/conversations", json={}).json()["id"]


def test_route_returns_a_checked_result_and_keeps_the_problem(authed: TestClient) -> None:
    r = authed.post("/v1/compute/vector", json={**CURL_A, "save": True})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["result_kind"] == "vector"
    assert body["verification"]["numeric_passed"] and body["verification"]["symbolic_zero"] is None
    assert len(body["components_latex"]) == 3 and body["basis_latex"][0] == r"\hat{\rho}"
    assert r"\rho \neq 0" in body["restrictions_latex"]
    (item,) = authed.get("/v1/history").json()
    assert item["operation"] == "vector" and item["id"] == body["calculation_id"]
    assert item["result"]["request"]["fields"][0]["name"] == "A"
    assert "save" not in item["result"]["request"]


def test_route_refuses_what_needs_a_declaration_and_validates_shapes(authed: TestClient) -> None:
    undeclared = authed.post(
        "/v1/compute/vector", json={"expression": r"\nabla k", "coordinates": "cartesian"}
    ).json()
    assert undeclared["status"] == "parse_error" and undeclared["undeclared"] == ["k"]
    blocked = authed.post(
        "/v1/compute/vector",
        json={
            "expression": r"\nabla\times(\nabla f)",
            "fields": [{"name": "f", "kind": "scalar"}],
        },
    ).json()
    assert (
        blocked["status"] == "conditions_not_satisfied" and blocked["blocked"][0]["needs"] == "c2"
    )
    assert blocked["result_latex"] is None
    bad = {"expression": "x", "fields": [{"name": "A", "kind": "vector", "components": ["1"]}]}
    assert authed.post("/v1/compute/vector", json=bad).status_code == 422
    assert authed.post("/v1/compute/vector", json={"expression": "x" * 3000}).status_code == 422


def test_route_needs_a_login(client: TestClient) -> None:
    assert client.post("/v1/compute/vector", json=CURL_A).status_code == 401


def test_route_logs_enums_and_counters_only(authed: TestClient, capsys) -> None:  # type: ignore[no-untyped-def]
    authed.post(
        "/v1/compute/vector",
        json={
            "expression": r"\nabla\cdot\mathbf{secretfield}",
            "fields": [{"name": "secretfield", "kind": "vector"}],
        },
    )
    out = capsys.readouterr().out
    assert '"vector_calculus"' in out and "secretfield" not in out and "nabla" not in out
    event = next(json.loads(x) for x in out.splitlines() if '"event": "vector_calculus"' in x)
    assert set(event) <= {
        "event", "level", "timestamp", "operation", "status", "reason", "coordinate_system",
        "vector_action", "operator_depth", "points", "duration_ms", "request_id", "logger",
    }  # fmt: skip


# -- the agent tool -----------------------------------------------------------------------------
def test_tool_result_is_a_card_with_evidence_and_known_formulas(ctx: ToolContext) -> None:
    out = execute("vector_calculus", CURL_A, ctx)
    assert out.status == "ok" and out.card.type == "vector"
    assert out.evidence and out.evidence.accepted and out.evidence.check == "numeric"
    assert out.evidence.symbolic_zero is None
    assert "numerically checked" in out.for_model and "never 'proved'" in out.for_model
    assert out.known_latex  # the result's formulas may be repeated as verified
    assert "request" in out.card.data and "result" in out.card.data


def test_tool_withheld_result_has_no_verified_formulas(ctx: ToolContext) -> None:
    out = execute(
        "vector_calculus",
        {
            "expression": r"\nabla\cdot(\nabla\times\mathbf{A})",
            "fields": [{"name": "A", "kind": "vector"}],
        },
        ctx,
    )
    assert out.status == "conditions_not_satisfied"
    assert out.evidence and not out.evidence.accepted
    assert "do not declare it yourself" in out.for_model
    assert out.known_latex == []


def test_tool_input_as_read_is_quoted_not_verified(ctx: ToolContext) -> None:
    out = execute("vector_calculus", CURL_A, ctx)
    quoted = [why for _, why in out.quoted_latex]
    assert "your input, as read by Minerva" in quoted


def test_scalar_tools_refuse_declared_vector_fields(ctx: ToolContext) -> None:
    ctx.matrices = frozenset({"E"})
    assert "differentiate" in SCALAR_TOOLS
    out = execute("differentiate", {"latex": r"\mathbf{E} x", "variables": [{"name": "x"}]}, ctx)
    assert out.status == "error" and out.error_category == "unsupported"
    assert "vector field" in out.for_model and "vector_calculus" in out.for_model


def test_tool_arguments_are_validated(ctx: ToolContext) -> None:
    assert "vector_calculus" in TOOLS
    bad = execute("vector_calculus", {"fields": [{"name": "A", "kind": "tensor"}]}, ctx)
    assert bad.status == "error" and not bad.arguments_valid
    smoothness = execute(
        "vector_calculus",
        {"expression": "A", "fields": [{"name": "A", "kind": "vector", "smoothness": "c9"}]},
        ctx,
    )
    assert smoothness.status == "error"


# -- chat -----------------------------------------------------------------------------------------
def test_chat_turn_streams_a_vector_card_and_the_claim_stays_unchecked(
    chat_client: tuple[TestClient, str],
) -> None:
    client, cid = chat_client
    events = sse(client, cid, r"Compute $\nabla\cdot\mathbf{A}$ in spherical coordinates")
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    assert card["type"] == "vector" and card["data"]["result"]["status"] == "ok"
    assert card["data"]["request"]["coordinates"] == "spherical"
    assert message["trace"][0]["name"] == "vector_calculus"
    # the formula the model repeats comes from the tool, so it is marked as such
    assert any(s["status"] == "from_tool" for s in segments_of(message) if s["type"] != "text")


def test_chat_withheld_result_is_explained_not_shown(chat_client: tuple[TestClient, str]) -> None:
    client, cid = chat_client
    events = sse(client, cid, r"Compute $\nabla\times(\nabla f)$")
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    assert card["data"]["result"]["status"] == "conditions_not_satisfied"
    assert "no smoothness was declared" in message["text"]


def test_model_prose_with_a_cross_product_is_never_marked_checked(
    chat_client: tuple[TestClient, str],
) -> None:
    from minerva_api.agent.declarations import declared_vectors  # the words declare the names

    assert declared_vectors([r"Let $a$ and $b$ be vectors."]) == {"a", "b"}
    client, cid = chat_client
    events = sse(client, cid, r"Simplify $x + x$")
    message = next(d for e, d in events if e == "message")
    assert all(s["status"] != "contradicted" for s in segments_of(message) if s["type"] != "text")


# -- identities and controlled derivations ---------------------------------------------------------
def test_listing_identities_applies_none_and_marks_blocked_ones(ctx: ToolContext) -> None:
    out = execute(
        "vector_calculus",
        {
            "expression": r"\nabla\times(\nabla\times\mathbf{A})",
            "fields": [{"name": "A", "kind": "vector"}],
            "action": "identities",
        },
        ctx,
    )
    assert out.status == "ok" and "vector.curl_curl" in out.for_model
    assert "BLOCKED" in out.for_model  # A was not declared twice differentiable
    assert out.card.data["result"]["steps"] == [] and out.known_latex == []
    assert "rules" not in out.card.data["request"]  # the handbook rules are not stored on the card


def test_applying_curl_curl_gives_a_verified_identity_step(ctx: ToolContext) -> None:
    arguments = {
        "expression": r"\nabla\times(\nabla\times\mathbf{A})",
        "fields": [{"name": "A", "kind": "vector", "smoothness": "c2"}],
        "action": "apply_identity",
        "rule_id": "vector.curl_curl",
        "path": [],
    }
    out = execute("vector_calculus", arguments, ctx)
    assert out.status == "ok", out.for_model
    assert out.evidence and out.evidence.accepted
    assert "(identity)" in out.for_model and "CONDITIONAL" not in out.for_model
    assert out.known_latex  # an identity result may be repeated as verified
    result = out.card.data["result"]
    assert result["evidence"] == "identity" and result["steps"][0]["rule_id"] == "vector.curl_curl"
    blocked = execute(
        "vector_calculus",
        {**arguments, "fields": [{"name": "A", "kind": "vector"}]},
        ctx,
    )
    assert blocked.status == "conditions_not_satisfied" and blocked.known_latex == []
    assert "do not declare it yourself" in blocked.for_model


def test_a_result_that_uses_a_premise_is_conditional_and_never_marked_verified(
    ctx: ToolContext,
) -> None:
    out = execute(
        "vector_calculus",
        {
            "expression": r"\nabla\times(\nabla\times\mathbf{E})",
            "fields": [
                {"name": "E", "kind": "vector", "depends_on": ["x", "y", "z", "t"]},
                {"name": "B", "kind": "vector", "depends_on": ["x", "y", "z", "t"]},
            ],
            "premises": [r"\nabla\times\mathbf{E} = -\frac{\partial\mathbf{B}}{\partial t}"],
            "action": "substitute",
        },
        ctx,
    )
    assert out.status == "ok", out.for_model
    assert "CONDITIONAL" in out.for_model and "never call the result an identity" in out.for_model
    assert out.known_latex == []  # shown, but not as a verified formula
    assert any("premises you supplied" in why for _, why in out.quoted_latex)
    assert out.card.data["result"]["evidence"] == "conditional"


def test_derive_through_the_route_with_premises(authed: TestClient) -> None:
    body = {
        "expression": "curl(curl(E))",
        "format": "text",
        "fields": [
            {
                "name": "E",
                "kind": "vector",
                "smoothness": "smooth",
                "depends_on": ["x", "y", "z", "t"],
            },
            {
                "name": "B",
                "kind": "vector",
                "smoothness": "smooth",
                "depends_on": ["x", "y", "z", "t"],
            },
        ],
        "constants": ["mu0", "eps0"],
        "premises": [
            {"equation": "curl(E) = -dt(B)"},
            {"equation": "curl(B) = mu0*eps0*dt(E)"},
        ],
        "action": "derive",
        "target": "-mu0*eps0*dt(dt(E))",
        "save": True,
    }
    r = authed.post("/v1/compute/vector", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "ok", out["message"]
    assert out["evidence"] == "conditional"
    assert set(out["given"]) == {"curl(E) = -dt(B)", "curl(B) = mu0*eps0*dt(E)"}
    assert [s["evidence"] for s in out["steps"]].count("identity") >= 1
    (item,) = authed.get("/v1/history").json()
    assert item["result"]["request"]["premises"][0]["equation"] == "curl(E) = -dt(B)"
    assert "rules" not in item["result"]["request"]


def test_a_client_cannot_supply_its_own_identity(authed: TestClient) -> None:
    own = {
        "id": "vector.mine",
        "lhs": "curl(curl(A))",
        "rhs": "grad(div(A)) + lap(A)",
        "variables": {"A": "vector"},
        "requires": ["c2"],
    }
    body = {
        "expression": "curl(curl(A))",
        "format": "text",
        "fields": [{"name": "A", "kind": "vector", "smoothness": "c2"}],
        "action": "apply_identity",
        "rule_id": "vector.mine",
        "rules": [own],
    }
    out = authed.post("/v1/compute/vector", json=body).json()
    assert out["status"] == "unsupported" and out["diagnostics"]["reason"] == "no_such_rule"
