"""The statistical-mechanics agent tool: arguments, honesty, the user's conventions, the direct route."""

import pytest
from fastapi.testclient import TestClient
from minerva_api.agent.tools import TOOLS, ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_chat import cards_of, chat_client, sse  # noqa: F401

TWO_LEVEL = {
    "action": "derive",
    "system": {"kind": "levels", "levels": [{"energy": "0"}, {"energy": "epsilon"}]},
    "assumptions": {"epsilon": ["positive"]},
    "quantities": ["Z", "U"],
}
CLASSICAL = {
    "action": "derive",
    "system": {
        "kind": "classical",
        "hamiltonian": "p**2/(2*m) + k*x**2/2",
        "variables": [{"name": "x"}, {"name": "p"}],
    },
    "assumptions": {"m": ["positive"], "k": ["positive"], "h": ["positive"]},
    "quantities": ["U"],
}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def _property_names(node: object) -> set[str]:
    """Every field name anywhere in a JSON schema (not words in descriptions)."""
    found: set[str] = set()
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict):
            found |= set(props)
        for value in node.values():
            found |= _property_names(value)
    elif isinstance(node, list):
        for value in node:
            found |= _property_names(value)
    return found


def test_the_users_choices_are_not_arguments_of_the_tool() -> None:
    fields = _property_names(TOOLS["statistical_mechanics"].schema())
    assert {"system", "action", "partials"} <= fields
    for name in ("phase_space_cell", "identical", "values", "sweep"):
        assert name not in fields, f"{name} must not be something the model can set"
    refused = execute(
        "statistical_mechanics",
        {**CLASSICAL, "phase_space_cell": "h"},
        ToolContext(engine=Engine(budget_limits()), kb=KnowledgeBase.load()),
    )
    assert refused.status != "ok"  # unknown fields inside the system are rejected, not honoured


def test_a_verified_result_marks_its_formulas_and_quotes_the_input(ctx: ToolContext) -> None:
    out = execute("statistical_mechanics", TWO_LEVEL, ctx)
    assert out.status == "ok" and out.evidence is not None and out.evidence.accepted
    assert out.card.type == "statmech" and out.evidence.points_agreed
    result = out.card.data["result"]
    assert [q["name"] for q in result["quantities"]] == ["Z", "U"]
    assert out.known_latex, "the derived formulas count as verified"
    reasons = {why for _, why in out.quoted_latex}
    assert "your input, as read by Minerva" in reasons  # the user's levels are quoted, not verified
    assert "enumeration" in out.for_model
    assert "not a proof" in out.for_model


def test_a_missing_convention_becomes_a_question_not_a_guess(ctx: ToolContext) -> None:
    out = execute("statistical_mechanics", CLASSICAL, ctx)
    assert out.status == "conditions_not_satisfied"
    assert out.card.data["result"]["reason"] == "convention_unconfirmed"
    assert "ASK THE USER" in out.for_model
    assert out.known_latex == []  # nothing from a refused result counts as verified
    answered = execute(
        "statistical_mechanics",
        CLASSICAL,
        ToolContext(engine=ctx.engine, kb=ctx.kb, statmech_answers={"phase_space_cell": "h"}),
    )
    assert answered.status == "ok"
    assert answered.card.data["answers"] == {"phase_space_cell": "h"}


def test_copies_ask_whether_they_are_indistinguishable(ctx: ToolContext) -> None:
    body = {**TWO_LEVEL, "copies_count": "3"}
    asked = execute("statistical_mechanics", body, ctx)
    assert asked.card.data["result"]["reason"] == "convention_unconfirmed"
    chosen = execute(
        "statistical_mechanics",
        body,
        ToolContext(engine=ctx.engine, kb=ctx.kb, statmech_answers={"identical": False}),
    )
    assert chosen.status == "ok"


def test_invalid_arguments_are_reported_to_the_model(ctx: ToolContext) -> None:
    out = execute("statistical_mechanics", {"system": {"kind": "levels"}}, ctx)
    assert out.status == "error" and out.arguments_valid is False


def test_chat_turn_streams_a_statmech_card(chat_client: tuple[TestClient, str]) -> None:  # noqa: F811
    client, cid = chat_client
    events = sse(client, cid, "What is the free energy of a two-level system?")
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    assert card["type"] == "statmech" and card["data"]["result"]["status"] == "ok"
    assert message["trace"][0]["name"] == "statistical_mechanics"
    assert message["trace"][0]["status"] == "ok"
    events = sse(client, cid, "And the heat capacity of a two-level system?")
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    names = [q["name"] for q in card["data"]["result"]["quantities"]]
    assert "C" in names


def test_direct_route_carries_the_users_choices(chat_client: tuple[TestClient, str]) -> None:  # noqa: F811
    client, cid = chat_client
    url = f"/v1/chat/conversations/{cid}/tools/statistical_mechanics"
    asked = client.post(url, json=CLASSICAL)
    assert asked.status_code == 201
    card = cards_of(asked.json()["content"])[0]
    assert card["data"]["result"]["reason"] == "convention_unconfirmed"
    answered = client.post(url, json={**CLASSICAL, "phase_space_cell": "h"})
    assert answered.status_code == 201
    message = answered.json()["content"]
    card = cards_of(message)[0]
    assert card["data"]["result"]["status"] == "ok"
    assert card["data"]["answers"] == {"phase_space_cell": "h"}
    # the choice is recorded next to the call, so a replay is faithful
    assert message["trace"][0]["arguments"]["answers"] == {"phase_space_cell": "h"}
    evaluated = client.post(
        url,
        json={
            **TWO_LEVEL,
            "action": "evaluate",
            "quantities": ["U"],
            "values": {"beta": "1", "epsilon": "1", "k_B": "1"},
        },
    )
    card = cards_of(evaluated.json()["content"])[0]
    assert card["data"]["result"]["status"] == "ok"
    assert card["data"]["result"]["quantities"][0]["result_kind"] == "numeric"


def test_direct_route_validation_and_ownership(chat_client: tuple[TestClient, str]) -> None:  # noqa: F811
    client, cid = chat_client
    url = f"/v1/chat/conversations/{cid}/tools/statistical_mechanics"
    assert client.post(url, json={"system": {"kind": "nope"}}).status_code == 422
    assert (
        client.post(
            url, json={**TWO_LEVEL, "values": {"a" * 10: "1"} | {str(i): "1" for i in range(30)}}
        ).status_code
        == 422
    )
    missing = client.post(
        "/v1/chat/conversations/00000000-0000-0000-0000-000000000000/tools/statistical_mechanics",
        json=TWO_LEVEL,
    )
    assert missing.status_code == 404
