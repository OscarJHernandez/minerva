"""An "Explore in Minerva" prompt, sent through the real chat endpoint, runs the geometry engine and the user can
carry on: the numbers come from the engine (not from the registry or the page), the evidence kinds stay apart
(Ricci: symbolically proved; Kretschmann: numerically checked, no symbolic proof), the coordinate restrictions
are reported, and a follow-up in the same conversation continues from the same metric.

The model here is the deterministic scripted stand-in (development only): it chooses the tool, the engine does
everything else. The metric the stand-in uses is the one in the prompt, which is pinned against the landing
example by test_exploration_registry.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from .test_chat import cards_of, chat_client, sse  # noqa: F401  (chat_client is a fixture)

ROOT = Path(__file__).resolve().parents[3]
PROMPTS = {
    e["id"]: e["prompt"]
    for e in json.loads((ROOT / "packages" / "ui" / "fixtures" / "explorations.json").read_text())[
        "explorations"
    ]
}


def kinds(card: dict) -> dict[str, dict]:  # type: ignore[type-arg]
    return {q["kind"]: q for q in card["data"]["result"]["quantities"]}


def test_the_full_exploration_runs_the_engine_and_keeps_proof_and_numerics_apart(
    chat_client: tuple[TestClient, str],  # noqa: F811
) -> None:
    client, cid = chat_client
    events = sse(client, cid, PROMPTS["schwarzschild-curvature"])
    assert "error" not in [e for e, _ in events]
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    assert card["type"] == "geometry"
    result = card["data"]["result"]
    assert result["status"] == "ok"
    by_kind = kinds(card)
    assert set(by_kind) == {"christoffel", "ricci", "kretschmann"}

    # The Ricci tensor vanishes: every component zero, and a symbolic zero test established it.
    ricci = by_kind["ricci"]
    assert ricci["all_zero"] is True
    assert ricci["verification"]["numeric_passed"] is True
    assert ricci["verification"]["symbolic_zero"] is True

    # The Kretschmann scalar does not, and it has numerical checks only: never a symbolic proof.
    kretschmann = by_kind["kretschmann"]
    assert kretschmann["all_zero"] is False
    assert kretschmann["scalar_text"] == "48*M**2/r**6"
    assert kretschmann["verification"]["numeric_passed"] is True
    assert kretschmann["verification"]["symbolic_zero"] is not True

    # The restrictions are coordinate breakdowns, and the signature is not assumed.
    restrictions = " ".join(result["restrictions_latex"])
    assert "2 M" in restrictions and "\\sin" in restrictions
    assert result["signature"] is None
    # what the model saw says the same, and the trace records one engine call
    assert message["trace"][0]["name"] == "geometry"


@pytest.mark.parametrize(
    ("exploration", "quantity", "text"),
    [
        ("schwarzschild-ricci", "ricci", None),
        ("schwarzschild-kretschmann", "kretschmann", "48*M**2/r**6"),
    ],
)
def test_the_ricci_and_kretschmann_explorations_are_independent(
    chat_client: tuple[TestClient, str],  # noqa: F811
    exploration: str,
    quantity: str,
    text: str | None,
) -> None:
    client, cid = chat_client
    events = sse(client, cid, PROMPTS[exploration])
    message = next(d for e, d in events if e == "message")
    (card,) = cards_of(message)
    by_kind = kinds(card)
    assert set(by_kind) == {quantity}
    assert by_kind[quantity]["scalar_text"] == text
    assert by_kind[quantity]["status"] == "ok"


def test_the_user_can_continue_the_exploration_without_losing_the_metric(
    chat_client: tuple[TestClient, str],  # noqa: F811
) -> None:
    client, cid = chat_client
    events = sse(client, cid, PROMPTS["schwarzschild-curvature"])
    first = cards_of(next(d for e, d in events if e == "message"))[0]
    # 1. Ask for another quantity of the same metric from the card (no model involved; the request is the card's).
    request = {**first["data"]["request"], "quantities": ["einstein"]}
    r = client.post(f"/v1/chat/conversations/{cid}/tools/geometry", json=request)
    assert r.status_code == 201
    einstein = cards_of(r.json()["content"])[0]
    assert einstein["data"]["result"]["metric_latex"] == first["data"]["result"]["metric_latex"]
    assert kinds(einstein)["einstein"]["all_zero"] is True
    # 2. Keep talking in the same conversation: the earlier exchange is stored in order and the new turn runs.
    again = sse(client, cid, "Now compute the Ricci scalar of the Schwarzschild metric")
    assert "error" not in [e for e, _ in again]
    stored = client.get(f"/v1/chat/conversations/{cid}").json()["messages"]
    roles = [m["role"] for m in stored]
    assert roles[0] == "user" and roles.count("assistant") >= 3
    assert stored[0]["content"]["text"] == PROMPTS["schwarzschild-curvature"]
