"""Row numbers for `statmech`, `perturbation` and `dimensional` (issue #133, parts 2 and 3, PR 3 of 3).

Only formulas the card really displays, only for a card that was accepted, and only when the row itself was
checked. Numerical values, unevaluated representations, asymptotic results, the user's own input and the
conditional formulas (the dimensional relation) get no number; the perturbation solution is numbered by its first relation only. A number
is a name, never a verification: `@eN` of a row substitutes its LaTeX exactly and the result says it came from
one component of an earlier result.

The keys and the rows are pinned against `packages/ui/fixtures/equation-rows.json`, which the browser test
renders (`equation-rows.test.tsx`). The entries are real engine output; `MINERVA_REGEN_ROW_FIXTURE=1 pytest -k
regenerate` adds the entries that are missing (it never rewrites an existing one).
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
from minerva_api.agent.card_equations import (
    LIST_NOT_YET_NUMBERED,
    LIST_READERS,
    STATMECH_SYMBOL,
    dimensional,
    listed_equations,
    perturbation,
    split_solution,
    statmech,
    statmech_key,
)
from minerva_api.agent.equations import (
    MAX_EQUATION_CHARS,
    EquationTable,
    build_table,
    number_stored,
)
from minerva_api.agent.refs import COMPONENT_SUFFIX, ResultStore
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_equation_rows_contract import FIXTURE, card_from, number
from .test_total_derivative_handles import Row

NEW = ("statmech", "perturbation", "dimensional")
CARD_TYPE = {"statmech": "statmech", "perturbation": "perturbation", "dimensional": "dimensional"}
REQUESTS: dict[str, tuple[str, dict[str, Any]]] = {
    "statmech": (
        "statistical_mechanics",
        {
            "action": "derive",
            "system": {"kind": "levels", "levels": [{"energy": "0"}, {"energy": "epsilon"}]},
            "assumptions": {"epsilon": ["positive"]},
            "quantities": ["Z", "U", "S", "Var_E", "F", "lnZ"],
        },
    ),
    "perturbation": (
        "perturbation_solve",
        {
            "lagrangian": r"\frac12 \dot{x}^2 - \frac12 x^2 - \frac14 b x^4",
            "coordinates": ["x"],
            "assumptions": {"b": ["positive"]},
            "small_parameter": "b",
            "order": 2,
        },
    ),
    "dimensional": (
        "dimensional_analysis",
        {
            "quantities": [
                {"name": "T", "kind": "period"},
                {"name": "L", "kind": "length"},
                {"name": "g", "kind": "acceleration"},
                {"name": "m", "unit": "kg"},
            ],
            "unknown": "T",
        },
    ),
}
EXPECTED_KEYS = {
    "statmech": [
        "quantity:Z",
        "quantity:U",
        "quantity:S",
        "quantity:Var_E",
        "quantity:F",
        "quantity:lnZ",
    ],
    "perturbation": ["omega0", "omega", "ratio", "solution"],
    "dimensional": ["group:0"],
}


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
    for name, (tool, args) in REQUESTS.items():
        if name in data:
            continue
        out = execute(tool, args, ctx)
        assert out.status == "ok" and out.card.type == CARD_TYPE[name], name
        card = {k: v for k, v in out.card.data.items() if k != "methods"}
        rows = dict(listed_equations(out.card.type, card))
        data[name] = {"data": card, "keys": list(rows), "rows": rows}
    FIXTURE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


# --- the keys and rows the browser marks ------------------------------------------------------------------------
@pytest.mark.parametrize("name", NEW)
def test_the_fixture_keys_and_rows_are_the_readers_and_the_ones_written_by_hand(
    fixture: dict[str, Any], name: str
) -> None:
    entry = fixture[name]
    rows = listed_equations(CARD_TYPE[name], entry["data"])
    assert [k for k, _ in rows] == entry["keys"] == EXPECTED_KEYS[name]
    assert dict(rows) == entry["rows"]


@pytest.mark.parametrize("name", NEW)
def test_a_live_engine_card_is_read_like_the_fixture(
    ctx: ToolContext, fixture: dict[str, Any], name: str
) -> None:
    tool, args = REQUESTS[name]
    out = execute(tool, args, ctx)
    assert out.status == "ok" and out.card.type == CARD_TYPE[name]
    assert [k for k, _ in listed_equations(out.card.type, out.card.data)] == fixture[name]["keys"]


def test_the_three_types_are_list_readers_and_each_remaining_type_is_in_exactly_one_set() -> None:
    for kind in NEW:
        assert kind in LIST_READERS and kind not in LIST_NOT_YET_NUMBERED
    assert {"vector"} == LIST_NOT_YET_NUMBERED


# --- statmech -----------------------------------------------------------------------------------------------
def test_the_statmech_symbol_table_is_the_one_of_the_panel() -> None:
    source = Path(FIXTURE.parents[3], "apps/web/src/components/statmech-panel.tsx").read_text()
    block = re.search(r"const SYMBOL: Record<string, string> = \{(.*?)\};", source, re.S)
    assert block is not None
    found = dict(re.findall(r'(\w+): "((?:[^"\\]|\\.)*)"', block.group(1)))
    assert {k: v.replace("\\\\", "\\") for k, v in found.items()} == STATMECH_SYMBOL
    # the panel builds the row text and the key the same way
    assert "`${SYMBOL[q.name] ?? q.name} = ${q.latex}`" in source
    assert "`quantity:${name}:${k}`" in source


def test_a_quantity_row_is_the_symbol_equals_the_formula_the_panel_draws(
    fixture: dict[str, Any],
) -> None:
    data = fixture["statmech"]["data"]
    rows = dict(statmech(data))
    for q in data["result"]["quantities"]:
        assert rows[f"quantity:{q['name']}"] == f"{STATMECH_SYMBOL[q['name']]} = {q['latex']}"
    card = card_from("statmech", data)
    table = number(card)
    assert list(card.numbers) == EXPECTED_KEYS["statmech"]
    assert card.number is None
    assert all(table.items[n].part_of == "statmech" for n in card.numbers.values())


def test_the_levels_step_and_every_other_listing_is_never_numbered(fixture: dict[str, Any]) -> None:
    data = copy.deepcopy(fixture["statmech"]["data"])
    result = data["result"]
    levels = [s["latex"] for s in result["steps"] if s["name"] == "levels"]
    assert levels, "the fixture must carry the user's levels step"
    result["table"] = {"columns": ["beta", "U"], "rows": [["1", "0.5"]], "variable": "beta"}
    result["notes"] = ["a note with x = 1"]
    texts = {latex for _, latex in statmech(data)}
    assert not texts & set(levels)
    assert not any("0.5" in t or "note" in t for t in texts)
    assert [k for k, _ in statmech(data)] == EXPECTED_KEYS["statmech"]


@pytest.mark.parametrize("kind", ["representation", "numeric", "asymptotic"])
def test_a_quantity_that_is_not_an_exact_result_gets_no_row(
    fixture: dict[str, Any], kind: str
) -> None:
    data = copy.deepcopy(fixture["statmech"]["data"])
    data["result"]["quantities"][1]["result_kind"] = kind
    keys = [k for k, _ in statmech(data)]
    assert "quantity:U" not in keys and len(keys) == len(EXPECTED_KEYS["statmech"]) - 1


def test_a_quantity_with_a_name_outside_the_panels_table_or_a_failed_check_gets_no_row(
    fixture: dict[str, Any],
) -> None:
    data = copy.deepcopy(fixture["statmech"]["data"])
    for name in ("Xi", "Omega", "occupations", "mean_E", "dU/dbeta"):
        extra = {**data["result"]["quantities"][0], "name": name}
        data["result"]["quantities"].append(extra)
    assert [k for k, _ in statmech(data)] == EXPECTED_KEYS["statmech"]
    failed = copy.deepcopy(fixture["statmech"]["data"])
    failed["result"]["quantities"][0]["checks"] = [
        {"name": "enumeration", "subject": "Z", "passed": False, "points": 3, "agreed": 2}
    ]
    assert "quantity:Z" not in [k for k, _ in statmech(failed)]
    undecided = copy.deepcopy(fixture["statmech"]["data"])
    undecided["result"]["quantities"][0]["checks"] = [
        {"name": "symbolic_zero", "subject": "Z", "passed": None, "points": 0, "agreed": 0}
    ]
    assert "quantity:Z" in [k for k, _ in statmech(undecided)]  # undecided is not a disagreement


def test_a_result_that_is_not_ok_lists_nothing(ctx: ToolContext, fixture: dict[str, Any]) -> None:
    for status in ("not_verified", "unsupported", "parse_error", "resource_limit", "unchanged"):
        data = copy.deepcopy(fixture["statmech"]["data"])
        data["result"]["status"] = status
        assert statmech(data) == [], status
    out = execute(
        "statistical_mechanics",
        {
            "action": "derive",
            "system": {
                "kind": "classical",
                "hamiltonian": "p**2/(2*m) + k*x**2/2",
                "variables": [{"name": "x"}, {"name": "p"}],
            },
            "assumptions": {"m": ["positive"], "k": ["positive"], "h": ["positive"]},
            "quantities": ["U"],
        },
        ctx,
    )
    assert out.card.data["result"]["reason"] == "convention_unconfirmed"
    assert statmech(out.card.data) == []


def test_a_repeated_name_keeps_a_unique_and_stable_key(fixture: dict[str, Any]) -> None:
    assert [statmech_key(["Z", "U", "Z", "Z"], i) for i in range(4)] == [
        "quantity:Z",
        "quantity:U",
        "quantity:Z:2",
        "quantity:Z:3",
    ]
    data = copy.deepcopy(fixture["statmech"]["data"])
    data["result"]["quantities"].insert(1, copy.deepcopy(data["result"]["quantities"][0]))
    keys = [k for k, _ in statmech(data)]
    assert len(keys) == len(set(keys)) == len(EXPECTED_KEYS["statmech"]) + 1
    assert keys[:3] == ["quantity:Z", "quantity:Z:2", "quantity:U"]


def test_a_statmech_row_over_500_characters_is_not_numbered(fixture: dict[str, Any]) -> None:
    data = copy.deepcopy(fixture["statmech"]["data"])
    data["result"]["quantities"][0]["latex"] = "x + " * 200 + "1"
    card = card_from("statmech", data)
    number(card)
    assert len(dict(statmech(data))["quantity:Z"]) > MAX_EQUATION_CHARS
    assert "quantity:Z" not in card.numbers and "quantity:U" in card.numbers


# --- perturbation -------------------------------------------------------------------------------------------
def test_the_frequencies_the_period_ratio_and_the_first_relation_of_the_solution_are_numbered(
    fixture: dict[str, Any],
) -> None:
    data = fixture["perturbation"]["data"]
    report = data["report"]
    assert report["solution_latex"] and report["orders"]
    rows = dict(perturbation(data))
    head, tail = report["solution_latex"].split(r",\quad ")
    assert rows == {
        "omega0": report["linear_frequency_latex"],
        "omega": report["frequency_latex"],
        "ratio": report["period_ratio_latex"],
        "solution": head,
    }
    # one number never names both relations, and the notation `\omega t = \tau` is not numbered
    assert report["solution_latex"] not in rows.values() and tail not in rows.values()
    assert rows["solution"].endswith(r"O\left(b^{3}\right)")
    card = card_from("perturbation", data)
    table = number(card)
    assert list(card.numbers) == EXPECTED_KEYS["perturbation"] and card.number is None
    assert all(table.items[n].part_of == "perturbation" for n in card.numbers.values())


def test_a_perturbation_that_was_not_accepted_or_not_verified_lists_nothing(
    fixture: dict[str, Any],
) -> None:
    base = fixture["perturbation"]["data"]
    for status in ("not_verified", "unsupported", "parse_error", "resource_limit", "unchanged"):
        data = copy.deepcopy(base)
        data["report"]["status"] = status
        assert perturbation(data) == [], status
    data = copy.deepcopy(base)
    data["report"]["verification"]["passed"] = False
    assert perturbation(data) == []
    data["report"]["verification"] = None
    assert perturbation(data) == []
    absent = copy.deepcopy(base)
    absent["report"]["frequency_latex"] = None
    assert [k for k, _ in perturbation(absent)] == ["omega0", "ratio", "solution"]


SPLIT = json.loads(
    (FIXTURE.parent / "perturbation-solution-split.json").read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", SPLIT, ids=[c["name"] for c in SPLIT])
def test_the_solution_split_is_closed_and_shared_with_the_panel(case: dict[str, Any]) -> None:
    got = split_solution(case["latex"])
    expected = None if case["relation"] is None else (case["relation"], case["definition"])
    assert got == expected


def test_a_solution_that_does_not_split_is_not_numbered(fixture: dict[str, Any]) -> None:
    for bad in (
        r"u(t) = A + O\left(b^{3}\right)",  # one relation only
        r"u(t) = A + O\left(b^{3}\right),\quad \omega t = \sigma",  # another second relation
        r"u(t) = A,\quad \omega t = \tau",  # no remainder
    ):
        data = copy.deepcopy(fixture["perturbation"]["data"])
        data["report"]["solution_latex"] = bad
        assert "solution" not in dict(perturbation(data)), bad
    data = copy.deepcopy(fixture["perturbation"]["data"])
    data["report"]["solution_latex"] = None
    assert [k for k, _ in perturbation(data)] == ["omega0", "omega", "ratio"]
    data["report"]["status"] = "not_verified"
    assert perturbation(data) == []


def test_a_solution_over_500_characters_is_not_numbered(fixture: dict[str, Any]) -> None:
    data = copy.deepcopy(fixture["perturbation"]["data"])
    data["report"]["solution_latex"] = (
        "u(t) = " + "x + " * 200 + r"1 + O\left(b^{3}\right),\quad \omega t = \tau"
    )
    card = card_from("perturbation", data)
    number(card)
    assert "solution" not in card.numbers and "omega" in card.numbers


def test_a_perturbation_row_over_500_characters_is_not_numbered(fixture: dict[str, Any]) -> None:
    data = copy.deepcopy(fixture["perturbation"]["data"])
    data["report"]["frequency_latex"] = "x + " * 200 + "1"
    card = card_from("perturbation", data)
    number(card)
    assert "omega" not in card.numbers and {"omega0", "ratio"} <= set(card.numbers)


# --- dimensional --------------------------------------------------------------------------------------------
def test_a_verified_group_is_numbered_as_the_card_draws_it_and_the_relation_is_not(
    fixture: dict[str, Any],
) -> None:
    data = fixture["dimensional"]["data"]
    report = data["report"]
    group = report["groups"][0]
    assert dict(dimensional(data)) == {"group:0": f"{group['label']} = {group['latex']}"}
    assert report["relation_latex"] and report["closure"] == "determined"
    assert report["relation_latex"] not in {latex for _, latex in dimensional(data)}
    card = card_from("dimensional", data)
    table = number(card)
    assert list(card.numbers) == ["group:0"] and card.number is None
    assert table.items[card.numbers["group:0"]].part_of == "dimensional"


def test_groups_are_keyed_by_position_and_an_unverified_group_is_skipped(
    fixture: dict[str, Any],
) -> None:
    data = copy.deepcopy(fixture["dimensional"]["data"])
    first = data["report"]["groups"][0]
    second = {**first, "label": r"\Pi_{2}", "latex": r"\frac{g}{L}"}
    third = {**first, "label": r"\Pi_{3}", "latex": "m", "dimensionless_verified": False}
    data["report"]["groups"] = [first, third, second]
    assert [k for k, _ in dimensional(data)] == ["group:0", "group:2"]


def test_a_dimensional_card_that_was_not_accepted_lists_nothing(fixture: dict[str, Any]) -> None:
    for status in ("not_verified", "unsupported", "parse_error", "resource_limit", "unchanged"):
        data = copy.deepcopy(fixture["dimensional"]["data"])
        data["report"]["status"] = status
        assert dimensional(data) == [], status
    data = copy.deepcopy(fixture["dimensional"]["data"])
    data["report"]["groups"][0]["dimensionless_verified"] = False
    assert dimensional(data) == []


def test_a_missing_quantity_card_numbers_no_suggestion_and_no_dimension(ctx: ToolContext) -> None:
    out = execute(
        "dimensional_analysis",
        {
            "quantities": [
                {"name": "E", "kind": "energy"},
                {"name": "m", "kind": "mass"},
                {"name": "p", "kind": "momentum"},
            ],
            "unknown": "E",
        },
        ctx,
    )
    report = out.card.data["report"]
    if report.get("missing"):
        assert all(not k.startswith("sugg") for k, _ in dimensional(out.card.data))
        texts = {latex for _, latex in dimensional(out.card.data)}
        assert report["missing"]["missing_dimension_latex"] not in texts


# --- numbers persist, are never reused, and a row is a name only --------------------------------------------
@pytest.mark.parametrize("name", NEW)
def test_numbers_survive_a_rebuild_and_a_retry_never_reuses_them(
    fixture: dict[str, Any], name: str
) -> None:
    card = card_from(CARD_TYPE[name], fixture[name]["data"])
    table = number(card)
    stored = Row("assistant", {"blocks": [{"kind": "card", "card": card.model_dump(mode="json")}]})
    rebuilt = build_table([stored], counter=table.counter + 3)
    assert {n: e.latex for n, e in rebuilt.items.items()} == {
        n: e.latex for n, e in table.items.items()
    }
    assert all(e.part_of == CARD_TYPE[name] for e in rebuilt.items.values())
    other = "perturbation" if name != "perturbation" else "statmech"
    again = card_from(CARD_TYPE[other], fixture[other]["data"], "c2")
    number(again, rebuilt)
    assert min(again.numbers.values()) == table.counter + 4


@pytest.mark.parametrize("name", NEW)
def test_a_conversation_from_before_gets_the_rows(fixture: dict[str, Any], name: str) -> None:
    old = card_from(CARD_TYPE[name], fixture[name]["data"], "c2").model_dump(mode="json")
    content = {"blocks": [{"kind": "card", "card": old}]}
    assert number_stored([("assistant", content)], EquationTable()) is True
    assert set(content["blocks"][0]["card"]["numbers"]) == set(EXPECTED_KEYS[name])


@pytest.mark.parametrize(
    ("name", "key"),
    [("statmech", "quantity:U"), ("perturbation", "omega"), ("dimensional", "group:0")],
)
def test_a_row_reused_by_number_is_exact_latex_quoted_and_says_it_is_one_component(
    fixture: dict[str, Any], name: str, key: str
) -> None:
    card = card_from(CARD_TYPE[name], fixture[name]["data"])
    table = number(card)
    n = card.numbers[key]
    store = ResultStore.from_prior({}, {}, table.items, unverified={})
    resolved = store.resolve({"latex": f"@e{n}"})
    assert resolved.error is None and resolved.arguments["latex"] == table.items[n].latex
    assert resolved.unverified == [f"equation ({n}){COMPONENT_SUFFIX}"]
    assert table.items[n].part_of == CARD_TYPE[name]


def test_a_relation_row_is_refused_where_an_expression_is_read(fixture: dict[str, Any]) -> None:
    card = card_from("statmech", fixture["statmech"]["data"])
    table = number(card)
    store = ResultStore.from_prior({}, {}, table.items, unverified={})
    n = card.numbers["quantity:Z"]
    assert table.items[n].kind == "equation"
    refused = store.resolve({"latex": f"@e{n}"}, "simplify")
    assert refused.diagnostic is not None and refused.diagnostic.field == "latex"
    pert = card_from("perturbation", fixture["perturbation"]["data"], "c2")
    number(pert, table)
    m = pert.numbers["omega"]
    assert table.items[m].kind == "equation"
    assert (
        ResultStore.from_prior({}, {}, table.items, unverified={})
        .resolve({"latex": f"@e{m}"}, "differentiate")
        .diagnostic
        is not None
    )


# --- order is deterministic -----------------------------------------------------------------------------------
def test_the_rows_come_in_the_same_order_whatever_the_hash_seed() -> None:
    import subprocess
    import sys

    code = (
        "import json\n"
        "from minerva_api.agent.card_equations import listed_equations\n"
        "from minerva_api.agent.tools import ToolContext, execute\n"
        "from minerva_engine import Engine\n"
        "from minerva_engine.testing import budget_limits\n"
        "from minerva_knowledge import KnowledgeBase\n"
        "def main():\n"
        "    ctx = ToolContext(engine=Engine(budget_limits(max_concurrency=1)), kb=KnowledgeBase.load())\n"
        "    out = execute('dimensional_analysis', {'quantities': [{'name': 'T', 'kind': 'period'},"
        " {'name': 'L', 'kind': 'length'}, {'name': 'g', 'kind': 'acceleration'},"
        " {'name': 'm', 'unit': 'kg'}, {'name': 'v', 'kind': 'velocity'}], 'unknown': 'T'}, ctx)\n"
        "    print(json.dumps(listed_equations(out.card.type, out.card.data)))\n"
        "if __name__ == '__main__':\n    main()\n"
    )
    outputs = set()
    for seed in ("1", "7", "13"):
        done = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            timeout=240,
        )
        outputs.add(done.stdout.strip().splitlines()[-1])
    assert len(outputs) == 1
    assert len(json.loads(next(iter(outputs)))) >= 1
