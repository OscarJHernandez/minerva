"""Row numbers for the steps of the `simplify` card (issue #166).

Every step of a chain carries its own check (`TransformStep.verification`, the sampler and symbolic zero test
run on the step's before and after forms), and the chain is re-verified end to end. A step is numbered as the
equation the panel draws for it (`before = after`) only when that check ran and passed. A number is a name,
never a verification: `@eN` of a row substitutes its LaTeX exactly and the result says it came from one component
of an earlier result. The card's other forms (`alternatives`) and every card that is not `ok` list nothing.

Fixture entry `simplify` is real engine output; `MINERVA_REGEN_ROW_FIXTURE=1 pytest -k regenerate` adds it if
it is missing (it never rewrites an existing one).
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

import pytest
from minerva_api.agent.card_equations import (
    LIST_NOT_YET_NUMBERED,
    LIST_READERS,
    SINGLE_RESULT,
    listed_equations,
    simplify,
)
from minerva_api.agent.equations import EquationTable, build_table, number_stored, relation_kind
from minerva_api.agent.refs import COMPONENT_SUFFIX, ResultStore
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_equation_rows_contract import FIXTURE, card_from, number
from .test_total_derivative_handles import Row

REQUEST = (
    "simplify",
    {"latex": r"\sin^2 x + \cos^2 x + \cosh^2 y - \sinh^2 y"},
)
KEYS = ["step:0", "step:1"]


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def test_regenerate_the_missing_fixture_entry(ctx: ToolContext) -> None:
    """A helper, not a check: it runs only with `MINERVA_REGEN_ROW_FIXTURE=1` and only adds what is missing."""
    if os.environ.get("MINERVA_REGEN_ROW_FIXTURE") != "1":
        pytest.skip("set MINERVA_REGEN_ROW_FIXTURE=1 to add the missing fixture entry")
    data = json.loads(FIXTURE.read_text())
    if "simplify" not in data:
        out = execute(*REQUEST, ctx)
        assert out.status == "ok" and out.card.type == "simplify"
        card = {k: v for k, v in out.card.data.items() if k not in ("methods", "original_request")}
        rows = dict(simplify(card))
        data["simplify"] = {"data": card, "keys": list(rows), "rows": rows}
        FIXTURE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


# --- the reader -----------------------------------------------------------------------------------------------
def test_simplify_is_a_list_reader_and_the_remaining_sets_are_unchanged() -> None:
    assert "simplify" in LIST_READERS and "simplify" not in LIST_NOT_YET_NUMBERED
    assert "simplify" not in SINGLE_RESULT
    assert set() == LIST_NOT_YET_NUMBERED


def test_the_fixture_keys_and_rows(fixture: dict[str, Any]) -> None:
    entry = fixture["simplify"]
    rows = simplify(entry["data"])
    assert [k for k, _ in rows] == entry["keys"] == KEYS
    assert dict(rows) == entry["rows"]
    assert listed_equations("simplify", entry["data"]) == rows


def test_a_live_engine_card_is_read_like_the_fixture(
    ctx: ToolContext, fixture: dict[str, Any]
) -> None:
    out = execute(*REQUEST, ctx)
    assert out.status == "ok" and out.card.type == "simplify"
    assert dict(simplify(out.card.data)) == fixture["simplify"]["rows"]


def test_a_row_is_the_equation_the_panel_draws_and_each_step_was_checked(
    fixture: dict[str, Any],
) -> None:
    data = fixture["simplify"]["data"]
    rows = dict(simplify(data))
    for i, step in enumerate(data["steps"]):
        assert step["verification"]["passed"] is True
        assert rows[f"step:{i}"] == f"{step['before_latex']} = {step['after_latex']}"
        assert relation_kind(rows[f"step:{i}"]) == "equation"
    # the card's own result and its input are different formulas and keep their own handling
    assert data["result_latex"] not in rows.values()
    assert data["input_latex"] not in rows.values()


# --- what is never numbered ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "status",
    ["unchanged", "not_verified", "unsupported", "conditions_not_satisfied", "parse_error"],
)
def test_a_result_that_was_not_accepted_lists_no_row(fixture: dict[str, Any], status: str) -> None:
    data = copy.deepcopy(fixture["simplify"]["data"])
    data["status"] = status
    assert simplify(data) == []


def test_a_step_whose_check_did_not_pass_or_did_not_run_is_not_numbered(
    fixture: dict[str, Any],
) -> None:
    data = copy.deepcopy(fixture["simplify"]["data"])
    data["steps"][0]["verification"]["passed"] = False
    assert [k for k, _ in simplify(data)] == [
        "step:1"
    ]  # the key is the index in the card's own list
    data["steps"][1]["verification"] = None
    assert simplify(data) == []
    del data["steps"][1]["verification"]
    assert simplify(data) == []


def test_the_other_forms_are_not_numbered(fixture: dict[str, Any]) -> None:
    data = copy.deepcopy(fixture["simplify"]["data"])
    data["alternatives"] = [
        {"result_latex": "x", "result_text": "x", "steps": [], "verification": {"passed": True}}
    ]
    assert dict(simplify(data)) == fixture["simplify"]["rows"]


def test_missing_blank_and_malformed_parts_list_nothing_and_never_fail(
    fixture: dict[str, Any],
) -> None:
    assert simplify({}) == [] and simplify({"status": "ok"}) == []
    data = copy.deepcopy(fixture["simplify"]["data"])
    ok = {"passed": True}
    data["steps"] = [
        None,
        "x",
        {"verification": ok},
        {"verification": ok, "before_latex": "x", "after_latex": "  "},
        {"verification": ok, "before_latex": 3, "after_latex": "x"},
        {"verification": "yes", "before_latex": "x", "after_latex": "y"},
    ]
    assert simplify(data) == []
    data["steps"] = "nothing"
    assert simplify(data) == []


# --- numbering, persistence and reuse ------------------------------------------------------------------------
def test_rows_are_numbered_persist_and_are_reused_as_components(fixture: dict[str, Any]) -> None:
    data = fixture["simplify"]["data"]
    card = card_from("simplify", data)
    table = number(card)
    assert list(card.numbers) == KEYS and card.number is not None
    assert card.number not in card.numbers.values()  # the whole result is its own formula
    assert table.items[card.numbers["step:0"]].part_of == "simplify"
    stored = Row("assistant", {"blocks": [{"kind": "card", "card": card.model_dump(mode="json")}]})
    rebuilt = build_table([stored], counter=table.counter + 3)
    assert {n: e.latex for n, e in rebuilt.items.items()} == {
        n: e.latex for n, e in table.items.items()
    }
    n = card.numbers["step:1"]
    resolved = ResultStore.from_prior({}, {}, table.items, unverified={}).resolve(
        {"latex": f"@e{n}"}
    )
    assert resolved.error is None and resolved.arguments["latex"] == table.items[n].latex
    assert resolved.unverified == [f"equation ({n}){COMPONENT_SUFFIX}"]
    # a conversation numbered before rows existed gets them when it is read, and keeps the result's number
    old = card_from("simplify", data, "c2").model_dump(mode="json")
    old["number"], old["numbers"] = 1, {}
    content = {"blocks": [{"kind": "card", "card": old}]}
    assert number_stored([("assistant", content)], EquationTable()) is True
    stored_card = content["blocks"][0]["card"]
    assert set(stored_card["numbers"]) == set(KEYS) and stored_card["number"] == 1
