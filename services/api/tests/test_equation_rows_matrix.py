"""Row numbers for the list rows of the `matrix` card: its steps and its alternative conventions (issue #166).

The evaluated primary result keeps its whole-card number (`result_formula`, `equations.number_blocks`). A row is
numbered only where it is a formula of an accepted, evaluated result: an equation, a representation, a numerical
approximation and a result that was not accepted list nothing. Rows are keyed by position in the card's own
lists, which the engine fills in a fixed order (a derivation is a sequence; the alternatives are by convention).
A number is a name, never a verification: `@eN` of a row substitutes its LaTeX exactly and the result says it
came from one component of an earlier result.

Fixture entries (`matrix_gradient`, `matrix_integral`) are real engine output; `MINERVA_REGEN_ROW_FIXTURE=1
pytest -k regenerate` adds the ones that are missing (it never rewrites an existing one).
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
    matrix,
    result_formula,
)
from minerva_api.agent.equations import EquationTable, build_table, number_stored
from minerva_api.agent.refs import COMPONENT_SUFFIX, ResultStore
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_equation_rows_contract import FIXTURE, card_from, number
from .test_total_derivative_handles import Row

GRADIENT = (
    "matrix_calculus",
    {
        "latex": "-1/2*logdet(S) - 1/2*transpose(x - mu)*inv(S)*(x - mu)",
        "matrices": [
            {"name": "x", "rows": 3, "cols": 1},
            {"name": "mu", "rows": 3, "cols": 1},
            {"name": "S", "rows": 3, "positive_definite": True},
        ],
        "action": "gradient",
        "variable": "S",
    },
)
INTEGRAL = (
    "matrix_calculus",
    {
        "latex": r"\exp(wA)\exp(-wB)",
        "matrices": [
            {"name": "A", "entries": [["1", "0"], ["0", "2"]]},
            {"name": "B", "entries": [["0", "1"], ["0", "0"]]},
        ],
        "action": "integrate",
        "variable": "w",
        "lower": "0",
        "upper": "1",
    },
)
REPRESENTATION = (
    "matrix_calculus",
    {
        "latex": r"\exp(tP)\exp(tQ)",
        "matrices": [{"name": "P", "rows": 2}, {"name": "Q", "rows": 2}],
        "action": "integrate",
        "variable": "t",
        "lower": "0",
        "upper": "1",
    },
)
GRADIENT_KEYS = ["step:0", "alt:0"]
INTEGRAL_KEYS = ["step:2"]  # step 0 has no formula; step 1 is the Sylvester equation (a relation)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def test_regenerate_the_missing_fixture_entries(ctx: ToolContext) -> None:
    """A helper, not a check: it runs only with `MINERVA_REGEN_ROW_FIXTURE=1` and only adds what is missing."""
    if os.environ.get("MINERVA_REGEN_ROW_FIXTURE") != "1":
        pytest.skip("set MINERVA_REGEN_ROW_FIXTURE=1 to add the missing fixture entries")
    data = json.loads(FIXTURE.read_text())
    for name, request in (("matrix_gradient", GRADIENT), ("matrix_integral", INTEGRAL)):
        if name in data:
            continue
        out = execute(*request, ctx)
        assert out.status == "ok" and out.card.type == "matrix"
        card = {k: v for k, v in out.card.data.items() if k != "methods"}
        rows = dict(matrix(card))
        data[name] = {"data": card, "keys": list(rows), "rows": rows}
    FIXTURE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


# --- the reader -----------------------------------------------------------------------------------------------
def test_matrix_is_a_list_reader_and_the_remaining_sets_are_unchanged() -> None:
    assert "matrix" in LIST_READERS and "matrix" not in LIST_NOT_YET_NUMBERED
    assert "matrix" not in SINGLE_RESULT
    assert {"vector"} == LIST_NOT_YET_NUMBERED


def test_the_fixture_keys_and_rows_are_the_readers_and_the_ones_written_by_hand(
    fixture: dict[str, Any],
) -> None:
    for name, keys in (("matrix_gradient", GRADIENT_KEYS), ("matrix_integral", INTEGRAL_KEYS)):
        entry = fixture[name]
        rows = matrix(entry["data"])
        assert [k for k, _ in rows] == entry["keys"] == keys
        assert dict(rows) == entry["rows"]
        assert listed_equations("matrix", entry["data"]) == rows


@pytest.mark.parametrize(
    ("name", "request_"), [("matrix_gradient", GRADIENT), ("matrix_integral", INTEGRAL)]
)
def test_a_live_engine_card_is_read_like_the_fixture(
    ctx: ToolContext, fixture: dict[str, Any], name: str, request_: tuple[str, dict[str, Any]]
) -> None:
    out = execute(*request_, ctx)
    assert out.status == "ok" and out.card.type == "matrix"
    assert [k for k, _ in matrix(out.card.data)] == fixture[name]["keys"]


def test_the_rows_are_the_formulas_the_panel_draws(fixture: dict[str, Any]) -> None:
    data = fixture["matrix_gradient"]["data"]
    result = data["result"]
    rows = dict(matrix(data))
    assert rows["step:0"] == result["steps"][0]["after_latex"]
    assert rows["alt:0"] == result["derivative"]["alternatives"][0]["latex"]
    # the alternative is a different formula from the result, so it is a different equation
    assert rows["alt:0"] != result["result_latex"]
    # the formulas the card shows but does not number: the input as read, the formula the step started from
    assert result["input_latex"] not in rows.values()
    assert result["steps"][0]["before_latex"] not in rows.values()


def test_a_step_that_is_a_relation_or_has_no_formula_is_not_numbered(
    fixture: dict[str, Any],
) -> None:
    data = fixture["matrix_integral"]["data"]
    steps = data["result"]["steps"]
    assert steps[0]["after_latex"] is None and "=" in steps[1]["after_latex"]
    assert "=" not in steps[2]["after_latex"]
    assert [k for k, _ in matrix(data)] == ["step:2"]  # the key is the index in the card's own list


def test_a_row_equal_to_the_whole_result_has_the_number_of_the_result(
    fixture: dict[str, Any],
) -> None:
    card = card_from("matrix", fixture["matrix_gradient"]["data"])
    number(card)
    assert card.number is not None
    assert card.numbers["step:0"] == card.number
    assert card.numbers["alt:0"] != card.number


# --- what is never numbered ----------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["equation", "representation", "numeric"])
def test_a_result_that_is_not_an_evaluated_formula_lists_no_row(
    fixture: dict[str, Any], kind: str
) -> None:
    data = copy.deepcopy(fixture["matrix_gradient"]["data"])
    data["result"]["result_kind"] = kind
    assert matrix(data) == [] and result_formula("matrix", data) is None
    unevaluated = copy.deepcopy(fixture["matrix_gradient"]["data"])
    unevaluated["result"]["evaluated"] = False
    assert matrix(unevaluated) == []


@pytest.mark.parametrize(
    "status", ["not_verified", "unsupported", "conditions_not_satisfied", "parse_error"]
)
def test_a_result_that_was_not_accepted_lists_no_row(fixture: dict[str, Any], status: str) -> None:
    data = copy.deepcopy(fixture["matrix_gradient"]["data"])
    data["result"]["status"] = status
    assert matrix(data) == []


def test_a_real_representation_gets_no_number_at_all(ctx: ToolContext) -> None:
    out = execute(*REPRESENTATION, ctx)
    assert out.status == "ok" and out.card.data["result"]["evaluated"] is False
    assert matrix(out.card.data) == []
    card = card_from("matrix", out.card.data)
    number(card)
    assert card.number is None and card.numbers == {}


def test_missing_blank_and_malformed_parts_list_nothing_and_never_fail(
    fixture: dict[str, Any],
) -> None:
    assert matrix({}) == [] and matrix({"result": None}) == [] and matrix({"result": "x"}) == []
    data = copy.deepcopy(fixture["matrix_gradient"]["data"])
    data["result"]["steps"] = [None, "x", {"after_latex": "  "}, {"after_latex": 3}]
    data["result"]["derivative"] = {"alternatives": [{"latex": ""}, 4]}
    assert matrix(data) == []
    data["result"]["steps"] = "nothing"
    data["result"]["derivative"] = None
    assert matrix(data) == []


def test_a_derivative_without_an_alternative_lists_its_step_only(fixture: dict[str, Any]) -> None:
    data = copy.deepcopy(fixture["matrix_gradient"]["data"])
    data["result"]["derivative"]["alternatives"] = []
    assert [k for k, _ in matrix(data)] == ["step:0"]


# --- numbering, persistence and reuse ------------------------------------------------------------------------
def test_rows_are_numbered_persist_and_are_reused_as_components(fixture: dict[str, Any]) -> None:
    data = fixture["matrix_gradient"]["data"]
    card = card_from("matrix", data)
    table = number(card)
    assert list(card.numbers) == GRADIENT_KEYS and card.number == card.numbers["step:0"]
    assert table.items[card.numbers["alt:0"]].part_of == "matrix"
    stored = Row("assistant", {"blocks": [{"kind": "card", "card": card.model_dump(mode="json")}]})
    rebuilt = build_table([stored], counter=table.counter + 3)
    assert {n: e.latex for n, e in rebuilt.items.items()} == {
        n: e.latex for n, e in table.items.items()
    }
    n = card.numbers["alt:0"]
    resolved = ResultStore.from_prior({}, {}, table.items, unverified={}).resolve(
        {"latex": f"@e{n}"}
    )
    assert resolved.error is None and resolved.arguments["latex"] == table.items[n].latex
    assert resolved.unverified == [f"equation ({n}){COMPONENT_SUFFIX}"]
    # a conversation numbered before rows existed gets them when it is read
    old = card_from("matrix", data, "c2")
    content = {"blocks": [{"kind": "card", "card": old.model_dump(mode="json")}]}
    assert number_stored([("assistant", content)], EquationTable()) is True
    assert set(content["blocks"][0]["card"]["numbers"]) == set(GRADIENT_KEYS)
    assert content["blocks"][0]["card"]["number"] == 1


def test_a_card_stored_with_only_its_whole_result_number_gets_its_rows_when_read(
    fixture: dict[str, Any],
) -> None:
    """A stored card from before this change has `number` and no `numbers`; reading it adds the rows."""
    old = card_from("matrix", fixture["matrix_gradient"]["data"], "c3").model_dump(mode="json")
    old["number"], old["numbers"] = 1, {}
    content = {"blocks": [{"kind": "card", "card": old}]}
    assert number_stored([("assistant", content)], EquationTable()) is True
    card = content["blocks"][0]["card"]
    assert set(card["numbers"]) == set(GRADIENT_KEYS)
    assert (
        card["numbers"]["step:0"] == card["number"]
    )  # the result's own formula keeps the result's number
