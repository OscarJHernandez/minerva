"""A whole-card number for `approximation` and row numbers for the equilibria of `ode_analysis` (issue #166).

`approximation` keeps its result under `data["report"]`; it is numbered as a whole only when the report was
accepted (status ok, an evidence kind, no failed check). The equilibria of an ODE analysis come from the solver
in an order that is not canonical, so they are sorted first and keyed by position in that order. A number is a
name, never a verification: `@eN` of either substitutes the LaTeX exactly and the result says it came from a
tool result with no result id, or from one component of an earlier result.

Fixture entries (`approximation`, `ode_equilibria`) are real engine output; `MINERVA_REGEN_ROW_FIXTURE=1 pytest
-k regenerate` adds the ones that are missing (it never rewrites an existing one).
"""

from __future__ import annotations

import copy
import json
import os
import random
from typing import Any

import pytest
from minerva_api.agent.card_equations import (
    LIST_NOT_YET_NUMBERED,
    LIST_READERS,
    SINGLE_RESULT,
    canonical_eigenvalues,
    canonical_equilibria,
    ode_analysis,
    result_formula,
)
from minerva_api.agent.equations import EquationTable, build_table, number_stored
from minerva_api.agent.refs import COMPONENT_SUFFIX, NO_ID_SUFFIX, ResultStore
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_equation_rows_contract import FIXTURE, card_from, number
from .test_total_derivative_handles import Row

APPROXIMATION = (
    "approximation",
    {"latex": r"\sin x", "method": "taylor", "point": "0", "order": 5},
)
EQUILIBRIA = (
    "ode_analyze",
    {
        "equations": ["y'' - y + y^3 + y'/5 = 0"],
        "unknowns": ["y"],
        "independent": "t",
        "format": "text",
        "detail": "qualitative",
    },
)
EQUILIBRIA_KEYS = [
    "solved:0",
    "field:0",
    "field:1",
    *(
        f"equil:{i}:{suffix}"
        for i in range(3)
        for suffix in ("point:0", "point:1", "jacobian", "eig:0", "eig:1", "char")
    ),
]


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
    if "approximation" not in data:
        out = execute(*APPROXIMATION, ctx)
        assert out.status == "ok" and out.card.type == "approximation"
        data["approximation"] = {
            "data": out.card.data,
            "result_latex": out.card.data["report"]["result_latex"],
        }
    if "ode_equilibria" not in data:
        out = execute(*EQUILIBRIA, ctx)
        assert out.status == "ok" and out.card.type == "ode_analysis"
        card = {k: v for k, v in out.card.data.items() if k != "methods"}
        rows = dict(ode_analysis(card))
        data["ode_equilibria"] = {"data": card, "keys": list(rows), "rows": rows}
    FIXTURE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


# --- approximation ----------------------------------------------------------------------------------------------
def test_an_accepted_approximation_is_numbered_as_a_whole(
    ctx: ToolContext, fixture: dict[str, Any]
) -> None:
    data = fixture["approximation"]["data"]
    assert data["report"]["status"] == "ok" and data["report"]["evidence"] != "none"
    assert result_formula("approximation", data) == data["report"]["result_latex"]
    card = card_from("approximation", data)
    table = number(card)
    assert card.number is not None and card.numbers == {}
    assert table.items[card.number].latex == data["report"]["result_latex"]
    # a live engine card gives the same formula
    out = execute(*APPROXIMATION, ctx)
    assert (
        result_formula("approximation", out.card.data) == fixture["approximation"]["result_latex"]
    )


def test_approximation_is_a_single_result_card_and_not_a_list() -> None:
    assert "approximation" in SINGLE_RESULT
    assert "approximation" not in LIST_READERS and "approximation" not in LIST_NOT_YET_NUMBERED


@pytest.mark.parametrize(
    "status", ["not_verified", "unsupported", "parse_error", "resource_limit", "unchanged"]
)
def test_an_approximation_that_was_not_accepted_gets_no_number(
    fixture: dict[str, Any], status: str
) -> None:
    data = copy.deepcopy(fixture["approximation"]["data"])
    data["report"]["status"] = status
    assert result_formula("approximation", data) is None
    card = card_from("approximation", data)
    number(card)
    assert card.number is None


def test_no_evidence_a_failed_check_or_no_formula_gets_no_number(fixture: dict[str, Any]) -> None:
    base = fixture["approximation"]["data"]
    none = copy.deepcopy(base)
    none["report"]["evidence"] = "none"
    assert result_formula("approximation", none) is None
    failed = copy.deepcopy(base)
    failed["report"]["checks"] = [
        {"name": "independent", "passed": False},
        *failed["report"]["checks"],
    ]
    assert result_formula("approximation", failed) is None
    undecided = copy.deepcopy(base)
    undecided["report"]["checks"] = [{"name": "symbolic", "passed": None}]
    assert result_formula("approximation", undecided) == base["report"]["result_latex"]
    blank = copy.deepcopy(base)
    blank["report"]["result_latex"] = "  "
    assert result_formula("approximation", blank) is None
    assert result_formula("approximation", {"report": None}) is None
    assert result_formula("approximation", {}) is None


def test_the_request_and_the_other_formulas_of_the_report_are_never_the_number(
    fixture: dict[str, Any],
) -> None:
    data = fixture["approximation"]["data"]
    report = data["report"]
    card = card_from("approximation", data)
    table = number(card)
    assert len(table.items) == 1
    assert report["input_latex"] != report["result_latex"]


def test_an_approximation_number_survives_a_rebuild_and_an_old_conversation_gets_it(
    fixture: dict[str, Any],
) -> None:
    card = card_from("approximation", fixture["approximation"]["data"])
    table = number(card)
    stored = Row("assistant", {"blocks": [{"kind": "card", "card": card.model_dump(mode="json")}]})
    rebuilt = build_table([stored], counter=table.counter + 2)
    assert rebuilt.items[card.number or 0].latex == table.items[card.number or 0].latex
    old = card_from("approximation", fixture["approximation"]["data"], "c2").model_dump(mode="json")
    old["number"] = None
    content = {"blocks": [{"kind": "card", "card": old}]}
    assert number_stored([("assistant", content)], EquationTable()) is True
    assert content["blocks"][0]["card"]["number"] == 1


def test_an_approximation_reused_by_number_is_exact_and_says_it_has_no_result_id(
    fixture: dict[str, Any],
) -> None:
    card = card_from("approximation", fixture["approximation"]["data"])
    table = number(card)
    n = card.number or 0
    resolved = ResultStore.from_prior({}, {}, table.items, unverified={}).resolve(
        {"latex": f"@e{n}"}
    )
    assert resolved.error is None
    assert resolved.arguments["latex"] == fixture["approximation"]["data"]["report"]["result_latex"]
    assert resolved.unverified == [f"equation ({n}){NO_ID_SUFFIX}"]


# --- ode_analysis equilibria --------------------------------------------------------------------------------
def test_the_fixture_keys_and_rows_are_the_readers_and_the_ones_written_by_hand(
    fixture: dict[str, Any],
) -> None:
    entry = fixture["ode_equilibria"]
    rows = ode_analysis(entry["data"])
    assert [k for k, _ in rows] == entry["keys"] == EQUILIBRIA_KEYS
    assert dict(rows) == entry["rows"]


def test_a_live_engine_card_is_read_like_the_fixture(
    ctx: ToolContext, fixture: dict[str, Any]
) -> None:
    out = execute(*EQUILIBRIA, ctx)
    assert out.status == "ok"
    assert [k for k, _ in ode_analysis(out.card.data)] == fixture["ode_equilibria"]["keys"]


def test_the_rows_are_the_formulas_the_panel_draws(fixture: dict[str, Any]) -> None:
    data = fixture["ode_equilibria"]["data"]
    q = data["report"]["qualitative"]
    rows = dict(ode_analysis(data))
    ordered = canonical_equilibria(q["equilibria"])
    assert len(ordered) == 3
    for i, e in enumerate(ordered):
        assert rows[f"equil:{i}:point:0"] == e["point_latex"][0]
        assert rows[f"equil:{i}:point:1"] == e["point_latex"][1]
        assert rows[f"equil:{i}:jacobian"] == e["jacobian_latex"]
        assert rows[f"equil:{i}:char"] == e["characteristic_latex"]
        for j, eig in enumerate(canonical_eigenvalues(e["eigenvalues_latex"])):
            assert rows[f"equil:{i}:eig:{j}"] == eig
    texts = set(rows.values())
    assert not texts & set(data["report"]["interpretation"]["equations_latex"])
    for e in ordered:
        assert not texts & set(e["eigenvalues_numeric"])


def test_the_order_of_the_equilibria_does_not_change_the_keys_or_the_rows(
    fixture: dict[str, Any],
) -> None:
    base = fixture["ode_equilibria"]["data"]
    expected = ode_analysis(base)
    rng = random.Random(166)
    equilibria = base["report"]["qualitative"]["equilibria"]
    for _ in range(12):
        shuffled = copy.deepcopy(base)
        q = shuffled["report"]["qualitative"]
        rng.shuffle(q["equilibria"])
        for e in q["equilibria"]:
            rng.shuffle(e["eigenvalues_latex"])
        assert ode_analysis(shuffled) == expected
    reverse = copy.deepcopy(base)
    reverse["report"]["qualitative"]["equilibria"] = list(reversed(equilibria))
    assert ode_analysis(reverse) == expected


def test_the_sort_is_by_point_and_stable_for_equal_points() -> None:
    a = {"point_latex": ["y = 1", "y' = 0"], "jacobian_latex": "J1"}
    b = {"point_latex": ["y = -1", "y' = 0"], "jacobian_latex": "J2"}
    c = {"point_latex": ["y = 0", "y' = 0"], "jacobian_latex": "J3"}
    twin = {"point_latex": ["y = 0", "y' = 0"], "jacobian_latex": "J0"}
    assert canonical_equilibria([a, c, b]) == [b, c, a]
    assert canonical_equilibria([c, twin, b, a]) == [b, twin, c, a]  # by Jacobian within a point
    assert canonical_equilibria("nothing") == [] and canonical_equilibria([1, a]) == [a]
    assert canonical_eigenvalues(["b", "a", "", None]) == ["a", "b"]


def test_a_jacobian_that_was_not_checked_numbers_the_points_only(fixture: dict[str, Any]) -> None:
    data = copy.deepcopy(fixture["ode_equilibria"]["data"])
    data["report"]["qualitative"]["jacobian_checked"] = False
    keys = [k for k, _ in ode_analysis(data)]
    assert keys == [
        "solved:0",
        "field:0",
        "field:1",
        *(f"equil:{i}:point:{j}" for i in range(3) for j in range(2)),
    ]


def test_a_refused_qualitative_part_or_a_card_not_accepted_lists_no_equilibrium(
    fixture: dict[str, Any],
) -> None:
    data = copy.deepcopy(fixture["ode_equilibria"]["data"])
    data["report"]["qualitative"]["status"] = "unsupported"
    assert [k for k, _ in ode_analysis(data)] == ["solved:0"]
    data["report"]["status"] = "not_verified"
    assert ode_analysis(data) == []
    none = copy.deepcopy(fixture["ode_equilibria"]["data"])
    none["report"]["qualitative"]["equilibria"] = []
    assert [k for k, _ in ode_analysis(none)] == ["solved:0", "field:0", "field:1"]


def test_an_equilibrium_without_exact_eigenvalues_or_polynomial_keeps_its_other_rows(
    fixture: dict[str, Any],
) -> None:
    data = copy.deepcopy(fixture["ode_equilibria"]["data"])
    only = data["report"]["qualitative"]["equilibria"][0]
    only["eigenvalues_latex"] = []
    only["characteristic_latex"] = ""
    data["report"]["qualitative"]["equilibria"] = [only]
    keys = [k for k, _ in ode_analysis(data)][3:]
    assert keys == ["equil:0:point:0", "equil:0:point:1", "equil:0:jacobian"]


def test_equilibrium_rows_are_numbered_persist_and_are_reused_as_components(
    fixture: dict[str, Any],
) -> None:
    card = card_from("ode_analysis", fixture["ode_equilibria"]["data"])
    table = number(card)
    assert list(card.numbers) == EQUILIBRIA_KEYS and card.number is None
    assert all(table.items[n].part_of == "ode_analysis" for n in card.numbers.values())
    stored = Row("assistant", {"blocks": [{"kind": "card", "card": card.model_dump(mode="json")}]})
    rebuilt = build_table([stored], counter=table.counter + 3)
    assert {n: e.latex for n, e in rebuilt.items.items()} == {
        n: e.latex for n, e in table.items.items()
    }
    n = card.numbers["equil:1:jacobian"]
    resolved = ResultStore.from_prior({}, {}, table.items, unverified={}).resolve(
        {"latex": f"@e{n}"}
    )
    assert resolved.error is None and resolved.arguments["latex"] == table.items[n].latex
    assert resolved.unverified == [f"equation ({n}){COMPONENT_SUFFIX}"]
    old = card_from("ode_analysis", fixture["ode_equilibria"]["data"], "c2").model_dump(mode="json")
    content = {"blocks": [{"kind": "card", "card": old}]}
    assert number_stored([("assistant", content)], EquationTable()) is True
    assert set(content["blocks"][0]["card"]["numbers"]) == set(EQUILIBRIA_KEYS)
