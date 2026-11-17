"""Vector calculus through the API: the compute route, the agent tool, claims, logs and history."""

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from minerva_api.agent.tools import SCALAR_TOOLS, TOOLS, ToolContext, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
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
    return Engine(budget_limits(max_concurrency=2)), KnowledgeBase.load()


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


# -- electromagnetism ------------------------------------------------------------------------------
VACUUM = ["em.maxwell.gauss_vacuum", "em.maxwell.faraday", "em.maxwell.ampere_vacuum"]


def test_setup_lists_the_derivations_and_the_equations_the_user_may_adopt(ctx: ToolContext) -> None:
    out = execute("electromagnetism", {"action": "setup"}, ctx)
    assert out.status == "ok" and out.card.type == "entries"
    assert "wave_equation_e" in out.for_model and "gauge_invariance" in out.for_model
    assert "em.maxwell.faraday" in out.for_model and "never adopt one for them" in out.for_model
    assert "not available yet" in out.for_model and "pillbox" in out.for_model
    assert out.known_latex == []  # a list of options, not a verified result
    ids = {e["id"] for e in out.card.data["entries"]}
    assert {"em.maxwell.faraday", "em.maxwell.ampere", "em.continuity"} <= ids


def test_the_wave_equation_is_conditional_on_the_premises_and_never_verified(
    ctx: ToolContext,
) -> None:
    out = execute(
        "electromagnetism",
        {
            "action": "derive",
            "derivation": "wave_equation_e",
            "premise_ids": VACUUM,
            "smoothness": "c2",
        },
        ctx,
    )
    assert out.status == "ok", out.for_model
    assert "CONDITIONAL" in out.for_model and "never call the result an identity" in out.for_model
    assert "check on exact solutions of the premises: passed" in out.for_model
    assert "consistency evidence, not a verification" in out.for_model
    assert out.known_latex == []
    assert any("premises you supplied" in why for _, why in out.quoted_latex)
    result = out.card.data["result"]
    assert result["evidence"] == "conditional" and result["derivation"] == "wave_equation_e"
    assert len(result["equations_latex"]) == 1 and result["model_check"]["status"] == "passed"


def test_gauge_invariance_is_a_verified_identity_with_no_premise(ctx: ToolContext) -> None:
    out = execute(
        "electromagnetism",
        {"action": "derive", "derivation": "gauge_invariance", "smoothness": "c2"},
        ctx,
    )
    assert out.status == "ok" and out.card.data["result"]["evidence"] == "identity"
    assert "CONDITIONAL" not in out.for_model and out.known_latex  # an identity may be repeated


def test_a_missing_premise_is_asked_for_and_nothing_is_derived(ctx: ToolContext) -> None:
    out = execute(
        "electromagnetism",
        {
            "action": "derive",
            "derivation": "wave_equation_e",
            "premise_ids": VACUUM[:2],
            "smoothness": "c2",
        },
        ctx,
    )
    assert out.status == "conditions_not_satisfied"
    assert "the user adopts them; do not assume them" in out.for_model
    assert "curl(B) = mu0*epsilon0*dt(E)" in out.for_model and out.known_latex == []


def test_blocked_until_the_user_states_the_smoothness(ctx: ToolContext) -> None:
    out = execute(
        "electromagnetism",
        {"action": "derive", "derivation": "continuity", "premise_ids": [
            "em.maxwell.ampere", "em.maxwell.gauss"]},
        ctx,
    )  # fmt: skip
    assert out.status != "ok" and out.known_latex == []


def test_unknown_derivations_and_premises_are_refused(ctx: ToolContext) -> None:
    assert execute("electromagnetism", {"action": "derive"}, ctx).status == "error"
    bad = execute(
        "electromagnetism",
        {"action": "derive", "derivation": "poynting", "premise_ids": ["vector.curl_curl"]},
        ctx,
    )
    assert bad.status == "error" and not bad.arguments_valid
    assert (
        execute("vector_calculus", {"premise_ids": ["nope"], "action": "derive"}, ctx).status
        == "error"
    )


def test_the_route_resolves_premise_ids_and_only_from_the_handbook(authed: TestClient) -> None:
    fields = [
        {"name": n, "kind": "vector", "smoothness": "c2", "depends_on": ["x", "y", "z", "t"]}
        for n in ("E", "B")
    ]
    body = {
        "expression": "",
        "format": "text",
        "fields": fields,
        "constants": ["mu0", "epsilon0"],
        "premise_ids": [
            "em.maxwell.gauss_magnetic",
            "em.maxwell.faraday",
            "em.maxwell.ampere_vacuum",
        ],
        "action": "em_derive",
        "template": "wave_equation_b",
    }
    out = authed.post("/v1/compute/vector", json=body).json()
    assert out["status"] == "ok", out["message"]
    assert out["evidence"] == "conditional" and out["model_check"]["status"] == "passed"
    assert (
        authed.post(
            "/v1/compute/vector", json={**body, "premise_ids": ["vector.curl_curl"]}
        ).status_code
        == 422
    )


# -- integrals --------------------------------------------------------------------------------------
INVERSE_SQUARE = {"name": "F", "kind": "vector", "components": ["1/r**2", "0", "0"]}
SPHERE = {"q": ["1", "u", "v"], "u": {"name": "u", "lower": "0", "upper": "pi"},
          "v": {"name": "v", "lower": "0", "upper": "2*pi"}}  # fmt: skip


def test_flux_through_a_sphere_is_a_checked_exact_value(ctx: ToolContext) -> None:
    out = execute(
        "vector_calculus",
        {
            "action": "integrate",
            "integral": "flux",
            "expression": "F",
            "coordinates": "spherical",
            "fields": [INVERSE_SQUARE],
            "surface": SPHERE,
        },
        ctx,
    )
    assert out.status == "ok", out.for_model
    assert "exact" in out.for_model and "d_u r x d_v r" in out.for_model
    assert out.known_latex  # an exact, verified value may be repeated
    assert out.card.data["result"]["integral"]["values"][0]["text"] == "4*pi"


def test_a_theorem_lists_its_hypotheses_and_never_says_proved(ctx: ToolContext) -> None:
    out = execute(
        "vector_calculus",
        {
            "action": "integrate",
            "integral": "divergence_theorem",
            "expression": "F",
            "fields": [{"name": "F", "kind": "vector", "components": ["x**2", "y*z", "z**3"]}],
            "region": {"bounds": [["0", "1"], ["0", "2"], ["0", "1"]]},
        },
        ctx,
    )
    assert out.status == "ok", out.for_model
    assert (
        "the divergence theorem: the two sides were computed separately and agree" in out.for_model
    )
    assert "checked" in out.for_model and "never a proof of smoothness" in out.for_model


def test_the_point_charge_ball_is_refused_by_the_theorem_but_not_by_the_flux(
    ctx: ToolContext,
) -> None:
    refused = execute(
        "vector_calculus",
        {
            "action": "integrate",
            "integral": "divergence_theorem",
            "expression": "F",
            "coordinates": "spherical",
            "fields": [INVERSE_SQUARE],
            "region": {"bounds": [["0", "2"], ["0", "pi"], ["0", "2*pi"]]},
        },
        ctx,
    )
    assert refused.status == "unsupported" and refused.known_latex == []
    assert "not defined and finite" in refused.for_model


def test_the_route_integrates_and_stores_the_geometry(authed: TestClient) -> None:
    body = {
        "expression": "F",
        "format": "text",
        "coordinates": "spherical",
        "fields": [INVERSE_SQUARE],
        "action": "integrate",
        "integral": "flux",
        "surface": SPHERE,
        "save": True,
    }
    out = authed.post("/v1/compute/vector", json=body).json()
    assert out["status"] == "ok" and out["integral"]["values"][0]["exact"] is True
    (item,) = authed.get("/v1/history").json()
    assert item["result"]["request"]["surface"]["u"]["name"] == "u"


# -- electrostatics with the user's numbers ---------------------------------------------------------
def test_the_dipole_field_is_computed_with_two_methods_and_the_numbers_are_the_users(
    ctx: ToolContext,
) -> None:
    out = execute(
        "electromagnetism",
        {
            "action": "field",
            "charges": [["q", "0", "0", "0"], ["-q", "0", "0", "d"]],
            "point": ["1", "2", "3"],
            "constants": ["q", "d"],
        },
        ctx,
    )
    assert out.status == "ok", out.for_model
    assert "central difference" in out.for_model and "k_e" in out.for_model
    assert out.card.data["result"]["em"]["kind"] == "field"
    assert out.known_latex  # an exact closed form from the engine


def test_a_boundary_value_result_is_called_a_numerical_approximation(ctx: ToolContext) -> None:
    out = execute(
        "electromagnetism",
        {"action": "boundary_value", "side_a": "1", "side_b": "1", "top": "sin(pi*x)"},
        ctx,
    )
    assert out.status == "ok", out.for_model
    assert "NUMERICAL APPROXIMATION" in out.for_model
    assert out.card.data["result"]["em"]["approximate"] is True


def test_electrostatics_arguments_are_checked_and_a_point_in_the_source_is_refused(
    ctx: ToolContext,
) -> None:
    assert execute("electromagnetism", {"action": "field"}, ctx).status == "error"
    assert execute("electromagnetism", {"action": "multipole"}, ctx).status == "error"
    assert execute("electromagnetism", {"action": "boundary_value"}, ctx).status == "error"
    refused = execute(
        "electromagnetism",
        {"action": "field", "charges": [["1", "0", "0", "0"]], "point": ["0", "0", "0"]},
        ctx,
    )
    assert refused.status == "unsupported" and refused.known_latex == []


def test_the_route_runs_an_electrostatic_problem(authed: TestClient) -> None:
    body = {
        "expression": "",
        "format": "text",
        "action": "em_compute",
        "em": {"kind": "boundary_value", "a": "1", "b": "1", "top": "sin(pi*x)"},
    }
    out = authed.post("/v1/compute/vector", json=body).json()
    assert out["status"] == "ok" and out["em"]["approximate"] is True


def test_the_geometry_tool_computes_the_field_invariant(ctx: ToolContext) -> None:
    metric = [
        ["-1", "0", "0", "0"],
        ["0", "1", "0", "0"],
        ["0", "0", "1", "0"],
        ["0", "0", "0", "1"],
    ]
    out = execute(
        "geometry",
        {
            "coordinates": ["t", "x", "y", "z"],
            "metric": metric,
            "format": "text",
            "assumptions": {"a": ["real"], "b": ["real"]},
            "quantities": ["field_invariant"],
            "tensor": {
                "variance": ["down", "down"],
                "components": [
                    "0",
                    "-a*t",
                    "0",
                    "0",
                    "a*t",
                    "0",
                    "b*x",
                    "0",
                    "0",
                    "-b*x",
                    *["0"] * 6,
                ],
            },
        },
        ctx,
    )
    assert out.status == "ok", out.for_model
    assert out.card.data
