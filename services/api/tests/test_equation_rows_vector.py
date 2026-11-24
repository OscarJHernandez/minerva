"""Row numbers for the `vector` card, the last card that listed formulas without numbering them (issue #166).

The card mixes results of different evidence, so each action is decided: a row is numbered only when the formula is
drawn as a formula, the card's own check accepted it and no premise was used. `evaluate` and `convert` (the scalar
result, or every component as the row shows it), the steps of an identity (`apply_identity`, `derive` with no
premise) and the derived point-source (`distributional`) statement are numbered. Anything that holds given a
premise, an integral or theorem, electromagnetism, the Jacobian, a list of identities that could be applied and
the classical value of a field with a point source get no number. A number is a name, never a verification: `@eN`
of a row substitutes its LaTeX exactly and the result says it came from one component of an earlier result.

The keys and rows are pinned against `packages/ui/fixtures/equation-rows.json`, which the browser test renders. The
entries are real engine output; `MINERVA_REGEN_ROW_FIXTURE=1 pytest -k regenerate` adds the missing ones.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

import pytest
from minerva_api.agent.card_equations import (
    EXCLUDED_WITH_REASON,
    LIST_NOT_YET_NUMBERED,
    LIST_READERS,
    SINGLE_RESULT,
    VECTOR_NOT_NUMBERED,
    listed_equations,
    vector,
)
from minerva_api.agent.equations import EquationTable, build_table, number_stored
from minerva_api.agent.refs import COMPONENT_SUFFIX, ResultStore
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_equation_rows_contract import FIXTURE, card_from, number
from .test_total_derivative_handles import Row

POINT = {"name": "r", "kind": "vector", "components": ["x", "y", "z"]}
REQUESTS: dict[str, dict[str, Any]] = {
    "vector_scalar": {
        "expression": r"\nabla\cdot\mathbf{A}",
        "coordinates": "cylindrical",
        "fields": [{"name": "A", "kind": "vector"}],
    },
    "vector_components": {
        "expression": r"\nabla\times\mathbf{A}",
        "coordinates": "cylindrical",
        "fields": [{"name": "A", "kind": "vector"}],
    },
    "vector_convert": {
        "expression": r"\mathbf{F}",
        "coordinates": "cartesian",
        "to": "spherical",
        "action": "convert",
        "fields": [{"name": "F", "kind": "vector", "components": ["x", "y", "z"]}],
    },
    "vector_identity": {
        "expression": r"\nabla\times(\nabla\times\mathbf{A})",
        "fields": [{"name": "A", "kind": "vector", "smoothness": "c2"}],
        "action": "apply_identity",
        "rule_id": "vector.curl_curl",
        "path": [],
    },
    "vector_singular": {
        "expression": r"\nabla\cdot\frac{\mathbf{r}}{|\mathbf{r}|^3}",
        "action": "distributional",
        "fields": [POINT],
    },
}
# the classical value of the same field: 0 away from the source, never numbered
HINT = {
    "expression": r"\nabla\cdot\frac{\mathbf{r}}{|\mathbf{r}|^3}",
    "fields": [POINT],
}
KEYS = {
    "vector_scalar": ["result"],
    "vector_components": ["component:0", "component:1", "component:2"],
    "vector_convert": ["component:0", "component:1", "component:2"],
    "vector_identity": ["step:0"],
    "vector_singular": ["distributional"],
}
# actions whose engine output is a result the reader must never number (real requests where one is cheap)
NOT_NUMBERED = {
    "jacobian": {"action": "jacobian", "coordinates": "spherical"},
    "integrate": {
        "expression": r"\mathbf{F}",
        "action": "integrate",
        "integral": "line",
        "fields": [{"name": "F", "kind": "vector", "components": ["y", "x", "0"]}],
        "curve": {
            "q": ["s", "s**2", "0"],
            "parameter": {"name": "s", "lower": "0", "upper": "1"},
        },
    },
}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def run(ctx: ToolContext, request: dict[str, Any]) -> dict[str, Any]:
    out = execute("vector_calculus", request, ctx)
    assert out.card.type == "vector", out.for_model
    return {k: v for k, v in out.card.data.items() if k != "methods"}


def test_regenerate_the_missing_fixture_entries(ctx: ToolContext) -> None:
    """A helper, not a check: it runs only with `MINERVA_REGEN_ROW_FIXTURE=1` and only adds what is missing."""
    if os.environ.get("MINERVA_REGEN_ROW_FIXTURE") != "1":
        pytest.skip("set MINERVA_REGEN_ROW_FIXTURE=1 to add the missing fixture entries")
    data = json.loads(FIXTURE.read_text())
    for name, request in REQUESTS.items():
        if name in data:
            continue
        card = run(ctx, request)
        assert card["result"]["status"] == "ok", name
        rows = dict(vector(card))
        data[name] = {"data": card, "keys": list(rows), "rows": rows}
    FIXTURE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


# --- the reader -----------------------------------------------------------------------------------------------
def test_vector_is_a_list_reader_and_no_card_is_left_unnumbered() -> None:
    assert "vector" in LIST_READERS and "vector" not in SINGLE_RESULT
    assert not LIST_NOT_YET_NUMBERED
    assert "vector" not in EXCLUDED_WITH_REASON


def test_every_exclusion_of_an_action_states_its_reason() -> None:
    assert set(VECTOR_NOT_NUMBERED) == {
        "jacobian",
        "substitute",
        "apply",
        "em_derive",
        "em_compute",
        "integrate",
        "identities",
    }
    assert all(len(reason) > 20 for reason in VECTOR_NOT_NUMBERED.values())


@pytest.mark.parametrize("name", list(REQUESTS))
def test_the_fixture_keys_and_rows_are_the_readers_and_a_live_card_agrees(
    ctx: ToolContext, fixture: dict[str, Any], name: str
) -> None:
    entry = fixture[name]
    rows = vector(entry["data"])
    assert [k for k, _ in rows] == entry["keys"] == KEYS[name]
    assert dict(rows) == entry["rows"]
    assert listed_equations("vector", entry["data"]) == rows
    live = run(ctx, REQUESTS[name])
    assert [k for k, _ in vector(live)] == KEYS[name]


def test_the_rows_are_the_formulas_the_panel_draws(fixture: dict[str, Any]) -> None:
    result = fixture["vector_scalar"]["data"]["result"]
    assert dict(vector(fixture["vector_scalar"]["data"]))["result"] == result["result_latex"]
    result = fixture["vector_components"]["data"]["result"]
    rows = dict(vector(fixture["vector_components"]["data"]))
    for i, basis in enumerate(result["basis_latex"]):
        assert rows[f"component:{i}"] == f"{basis}:\\quad {result['components_latex'][i]}"
    # the whole vector (`result_latex`) is not drawn as a formula, so it is not numbered
    assert result["result_latex"] not in rows.values()
    step = fixture["vector_identity"]["data"]["result"]["steps"][0]
    assert dict(vector(fixture["vector_identity"]["data"]))["step:0"] == step["after_latex"]
    singular = fixture["vector_singular"]["data"]["result"]["singular"]
    assert dict(vector(fixture["vector_singular"]["data"])) == {
        "distributional": singular["distributional_latex"]
    }


def test_components_follow_the_order_of_the_coordinate_system(fixture: dict[str, Any]) -> None:
    result = fixture["vector_convert"]["data"]["result"]
    assert result["basis_latex"] == [r"\hat{r}", r"\hat{\theta}", r"\hat{\phi}"]
    assert [k for k, _ in vector(fixture["vector_convert"]["data"])] == KEYS["vector_convert"]


# --- what is never numbered ----------------------------------------------------------------------------------
def test_the_classical_value_of_a_point_source_is_not_numbered(ctx: ToolContext) -> None:
    card = run(ctx, HINT)
    assert (
        card["result"]["singular_hint"] == "point_source" and card["result"]["result_latex"] == "0"
    )
    assert vector(card) == []


def test_a_real_result_that_holds_given_a_premise_gets_no_number(ctx: ToolContext) -> None:
    card = run(
        ctx,
        {
            "expression": r"\nabla\times(\nabla\times\mathbf{E})",
            "fields": [
                {"name": "E", "kind": "vector", "depends_on": ["x", "y", "z", "t"]},
                {"name": "B", "kind": "vector", "depends_on": ["x", "y", "z", "t"]},
            ],
            "premises": [r"\nabla\times\mathbf{E} = -\frac{\partial\mathbf{B}}{\partial t}"],
            "action": "substitute",
        },
    )
    assert card["result"]["status"] == "ok" and card["result"]["evidence"] == "conditional"
    assert card["result"]["steps"] and vector(card) == []


@pytest.mark.parametrize("name", list(NOT_NUMBERED))
def test_a_jacobian_and_an_integral_get_no_number(ctx: ToolContext, name: str) -> None:
    card = run(ctx, NOT_NUMBERED[name])
    assert card["result"]["status"] == "ok", card["result"]["message"]
    assert vector(card) == []
    stored = card_from("vector", card)
    number(stored)
    assert stored.number is None and stored.numbers == {}


@pytest.mark.parametrize(
    "action",
    ["jacobian", "substitute", "apply", "em_derive", "em_compute", "integrate", "identities"],
)
def test_an_excluded_action_lists_nothing_even_with_a_numbered_looking_result(
    fixture: dict[str, Any], action: str
) -> None:
    for name in ("vector_scalar", "vector_components", "vector_identity"):
        data = copy.deepcopy(fixture[name]["data"])
        data["result"]["action"] = action
        assert vector(data) == []


@pytest.mark.parametrize("name", ["vector_scalar", "vector_components", "vector_identity"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "not_verified"),
        ("status", "conditions_not_satisfied"),
        ("status", "unsupported"),
        ("evidence", "conditional"),
        ("given", ["\\nabla \\cdot \\mathbf{E} = \\rho"]),
        ("needs_premises", ["em.maxwell.faraday"]),
        ("blocked", [{"rule_id": "x"}]),
        ("integral", {"values": []}),
        ("em", {"items": []}),
        ("verification", None),
    ],
)
def test_a_result_that_was_not_accepted_or_rests_on_a_premise_lists_no_row(
    fixture: dict[str, Any], name: str, field: str, value: Any
) -> None:
    data = copy.deepcopy(fixture[name]["data"])
    data["result"][field] = value
    assert vector(data) == []


@pytest.mark.parametrize(
    "leg",
    [
        {"numeric_passed": False},
        {"symbolic_zero": False},
        {"numeric_passed": False, "symbolic_zero": True},
    ],
)
def test_a_failed_leg_of_the_card_check_lists_no_row(
    fixture: dict[str, Any], leg: dict[str, Any]
) -> None:
    data = copy.deepcopy(fixture["vector_scalar"]["data"])
    data["result"]["verification"].update(leg)
    assert vector(data) == []


def test_a_conditional_or_failed_step_is_not_numbered(fixture: dict[str, Any]) -> None:
    data = copy.deepcopy(fixture["vector_identity"]["data"])
    step = data["result"]["steps"][0]
    for change in ({"evidence": "conditional"}, {"given": ["E = B"]}):
        broken = copy.deepcopy(data)
        broken["result"]["steps"][0].update(change)
        assert vector(broken) == []
    broken = copy.deepcopy(data)
    broken["result"]["steps"][0]["verification"] = {"numeric_passed": False, "symbolic_zero": None}
    assert vector(broken) == []
    # a second, premise-free step keeps its index in the card's own list
    data["result"]["steps"] = [{**step, "evidence": "conditional", "given": ["E = B"]}, step]
    assert [k for k, _ in vector(data)] == ["step:1"]


def test_a_distribution_is_numbered_only_when_derived_and_every_pairing_agreed(
    fixture: dict[str, Any],
) -> None:
    data = fixture["vector_singular"]["data"]
    assert [k for k, _ in vector(data)] == ["distributional"]
    for change in (
        lambda s: s.update(proof_status="not_established"),
        lambda s: s["weak_checks"][0].update(passed=False),
        lambda s: s["flux_checks"][0].update(passed=False),
        lambda s: s.update(weak_checks=[], flux_checks=[]),
        lambda s: s.update(distributional_latex="  "),
    ):
        broken = copy.deepcopy(data)
        change(broken["result"]["singular"])
        assert vector(broken) == []
    assert not any(k == "result" for k, _ in vector(data))  # the classical value is not a row


def test_missing_blank_and_malformed_parts_list_nothing_and_never_fail(
    fixture: dict[str, Any],
) -> None:
    assert vector({}) == [] and vector({"result": None}) == [] and vector({"result": "x"}) == []
    data = copy.deepcopy(fixture["vector_components"]["data"])
    data["result"]["components_latex"] = ["", None, 3]
    data["result"]["basis_latex"] = "x"
    assert vector(data) == []
    data = copy.deepcopy(fixture["vector_identity"]["data"])
    data["result"]["steps"] = [None, "x", {"after_latex": " "}, {"evidence": "identity"}]
    assert vector(data) == []
    data["result"]["steps"] = "nothing"
    assert vector(data) == []


# --- numbering, persistence and reuse ------------------------------------------------------------------------
def test_rows_are_numbered_persist_and_are_reused_as_components(fixture: dict[str, Any]) -> None:
    data = fixture["vector_components"]["data"]
    card = card_from("vector", data)
    table = number(card)
    assert list(card.numbers) == KEYS["vector_components"] and card.number is None
    n = card.numbers["component:1"]
    assert table.items[n].part_of == "vector"
    stored = Row("assistant", {"blocks": [{"kind": "card", "card": card.model_dump(mode="json")}]})
    rebuilt = build_table([stored], counter=table.counter + 3)
    assert {k: e.latex for k, e in rebuilt.items.items()} == {
        k: e.latex for k, e in table.items.items()
    }
    resolved = ResultStore.from_prior({}, {}, table.items, unverified={}).resolve(
        {"latex": f"@e{n}"}
    )
    assert resolved.error is None and resolved.arguments["latex"] == table.items[n].latex
    assert resolved.unverified == [f"equation ({n}){COMPONENT_SUFFIX}"]
    # a conversation numbered before rows existed gets them when it is read
    old = card_from("vector", data, "c2")
    content = {"blocks": [{"kind": "card", "card": old.model_dump(mode="json")}]}
    assert number_stored([("assistant", content)], EquationTable()) is True
    assert set(content["blocks"][0]["card"]["numbers"]) == set(KEYS["vector_components"])
