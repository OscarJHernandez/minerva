"""Row numbers for `solve`, `ode_solution` and `ode_analysis`; `tensor` as a single result (issue #133, parts 2 and 3).

Each reader lists only formulas the card really displays and only for a card that was accepted; a representation
(an unevaluated integral, a series, a perturbation expansion), a numerical value, the user's own input and the
equilibria (their order is not canonical) get no number. A number is a name, never a verification: `@eN` of a
row substitutes its LaTeX exactly and the result says it came from one component of an earlier result.

The keys are pinned against `packages/ui/fixtures/equation-rows.json`, which the browser test renders: the keys
the server numbers and the rows the panels mark cannot drift apart.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from minerva_api.agent.card_equations import (
    EXCLUDED_WITH_REASON,
    LIST_READERS,
    SINGLE_RESULT,
    listed_equations,
    ode_analysis,
    ode_solution,
    solve,
)
from minerva_api.agent.equations import (
    MAX_EQUATION_CHARS,
    EquationTable,
    build_table,
    number_blocks,
    number_stored,
)
from minerva_api.agent.events import Block, Card
from minerva_api.agent.refs import COMPONENT_SUFFIX, ResultStore
from minerva_api.agent.replay import call, say
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_total_derivative_handles import Row, card_of, play

FIXTURE = Path(__file__).resolve().parents[3] / "packages/ui/fixtures/equation-rows.json"


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def card_from(kind: str, data: dict[str, Any], card_id: str = "c1") -> Card:
    return Card(id=card_id, tool="t", type=kind, title="t", data=data)  # type: ignore[arg-type]


def number(card: Card, table: EquationTable | None = None) -> EquationTable:
    table = table or EquationTable()
    number_blocks([Block(kind="card", card=card)], table)
    return table


# --- the keys the browser marks ------------------------------------------------------------------------------
EXPECTED_KEYS = {
    "solve": [
        "factor:0",
        "root:0:0",
        "factor:1",
        "root:1:0",
        "factor:2",
        "root:2:0",
        "root:2:1",
        "rootrow:0",
        "rootrow:1",
        "rootrow:2",
        "rootrow:3",
    ],
    "ode_solution": ["solution:0:0"],
    "ode_analysis": ["solved:0", "field:0", "field:1"],
}


@pytest.mark.parametrize("kind", ["solve", "ode_solution", "ode_analysis"])
def test_the_fixture_keys_are_the_ones_the_reader_gives_and_the_ones_written_by_hand(
    fixture: dict[str, Any], kind: str
) -> None:
    entry = fixture[kind]
    assert [k for k, _ in listed_equations(kind, entry["data"])] == entry["keys"]
    assert entry["keys"] == EXPECTED_KEYS[kind]


@pytest.mark.parametrize("kind", ["solve", "ode_solution", "ode_analysis"])
def test_a_live_engine_card_is_read_like_the_fixture(
    ctx: ToolContext, fixture: dict[str, Any], kind: str
) -> None:
    request = {
        "solve": ("solve_polynomial", {"equation": r"(x^2-1)(x^2+x+1) = 0"}),
        "ode_solution": (
            "ode_solve_symbolic",
            {"equations": ["y''(x) = y(x)"], "independent": "x", "method": "closed_form"},
        ),
        "ode_analysis": (
            "ode_analyze",
            {
                "equations": ["y'' + y + y^3 + y'/5 = 0"],
                "unknowns": ["y"],
                "independent": "t",
                "format": "text",
                "detail": "qualitative",
            },
        ),
    }[kind]
    out = execute(request[0], request[1], ctx)
    assert out.status == "ok" and out.card.type == kind
    assert [k for k, _ in listed_equations(kind, out.card.data)] == fixture[kind]["keys"]


# --- classification -----------------------------------------------------------------------------------------
def test_tensor_is_a_single_result_with_no_reader_and_a_real_card_is_numbered_as_a_whole(
    ctx: ToolContext,
) -> None:
    assert "tensor" in SINGLE_RESULT and "tensor" not in LIST_READERS
    out = execute(
        "tensor_canonicalize",
        {
            "expression": "B(a,b) + B(b,a)",
            "tensors": {"B": {"rank": 2, "symmetry": "symmetric"}},
            "dimension": 4,
        },
        ctx,
    )
    assert out.status == "ok", out.for_model
    assert out.card.type == "tensor" and out.card.data.get("result_latex")
    table = number(out.card)
    assert out.card.number == 1 and out.card.numbers == {} and table.counter == 1


def test_the_two_exclusions_are_in_neither_list_nor_single_result_and_say_why() -> None:
    for kind, reason in EXCLUDED_WITH_REASON.items():
        assert kind not in LIST_READERS and kind not in SINGLE_RESULT
        assert listed_equations(kind, {"report": {"status": "ok"}}) == []
        assert reason


# --- solve --------------------------------------------------------------------------------------------------
def test_solve_numbers_factors_and_exact_roots_but_not_the_echo_or_the_decimals(
    fixture: dict[str, Any],
) -> None:
    data = fixture["solve"]["data"]
    rows = dict(solve(data))
    report = data["report"]
    echoed = {
        report["equation_latex"],
        *(r["numeric"] for r in report["roots"] if "." in r["numeric"]),
    }
    assert not echoed & set(rows.values())
    card = card_from("solve", data)
    table = number(card)
    assert set(card.numbers) == set(EXPECTED_KEYS["solve"])
    assert card.number is None
    # the same exact root shown twice (a factor's list and the table of roots) is one number
    assert card.numbers["root:0:0"] == card.numbers["rootrow:1"]
    assert all(table.items[n].part_of == "solve" for n in card.numbers.values())


def test_a_solve_card_that_was_not_accepted_lists_nothing(fixture: dict[str, Any]) -> None:
    for status in ("not_verified", "unsupported", "parse_error", "resource_limit"):
        data = copy.deepcopy(fixture["solve"]["data"])
        data["report"]["status"] = status
        assert solve(data) == []


def test_a_solve_row_over_500_characters_is_not_numbered(fixture: dict[str, Any]) -> None:
    data = copy.deepcopy(fixture["solve"]["data"])
    data["report"]["factors"][0]["factor_latex"] = "x + " * 200 + "1"
    assert len(data["report"]["factors"][0]["factor_latex"]) > MAX_EQUATION_CHARS
    card = card_from("solve", data)
    number(card)
    assert "factor:0" not in card.numbers and "factor:1" in card.numbers


def test_solve_factors_and_roots_come_in_the_same_order_whatever_the_hash_seed() -> None:
    code = (
        "import json\n"
        "from minerva_engine import Engine\n"
        "from minerva_engine.testing import budget_limits\n"
        "def main():\n"
        "    r = Engine(budget_limits(max_concurrency=1)).solve("
        "r'(x^2-1)^2 (x^3-2) (x^2+x+1) (x^2-3) = 0', None, None)\n"
        "    print(json.dumps([[f.factor_latex, f.exact_roots_latex] for f in r.factors]))\n"
        "if __name__ == '__main__':\n    main()\n"
    )
    outputs = set()
    for seed in ("1", "7"):
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


# --- ode_solution -------------------------------------------------------------------------------------------
def test_only_an_evaluated_closed_form_is_numbered(fixture: dict[str, Any]) -> None:
    data = copy.deepcopy(fixture["ode_solution"]["data"])
    item = data["report"]["solutions"][0]
    assert dict(ode_solution(data)) == {"solution:0:0": item["equations_latex"][0]}
    implicit = copy.deepcopy(data)
    implicit["report"]["solutions"][0]["kind"] = "implicit"
    assert [k for k, _ in ode_solution(implicit)] == ["solution:0:0"]
    for kind in ("series", "perturbation", "unevaluated"):
        other = copy.deepcopy(data)
        other["report"]["solutions"][0]["kind"] = kind
        assert ode_solution(other) == [], kind
    unevaluated = copy.deepcopy(data)
    unevaluated["report"]["solutions"][0]["evaluated"] = False
    assert ode_solution(unevaluated) == []


def test_ode_solution_numbers_every_equation_of_every_solution_by_position(
    fixture: dict[str, Any],
) -> None:
    data = copy.deepcopy(fixture["ode_solution"]["data"])
    first = data["report"]["solutions"][0]
    second = copy.deepcopy(first)
    first["equations_latex"] = ["x(t) = a", "y(t) = b"]
    second["equations_latex"] = ["x(t) = -a", "y(t) = -b"]
    data["report"]["solutions"] = [first, second]
    assert [k for k, _ in ode_solution(data)] == [
        "solution:0:0",
        "solution:0:1",
        "solution:1:0",
        "solution:1:1",
    ]


def test_a_solution_card_that_was_not_accepted_lists_nothing_and_notes_are_never_numbered(
    fixture: dict[str, Any],
) -> None:
    data = copy.deepcopy(fixture["ode_solution"]["data"])
    data["report"]["solutions"][0]["restrictions_latex"] = [r"x \neq 0"]
    data["report"]["solutions"][0]["conditions"] = [{"latex": "y(0) = 1", "satisfied": True}]
    texts = {latex for _, latex in ode_solution(data)}
    assert r"x \neq 0" not in texts and "y(0) = 1" not in texts
    data["report"]["status"] = "not_verified"
    assert ode_solution(data) == []


# --- ode_analysis -------------------------------------------------------------------------------------------
def test_an_analysis_numbers_the_solved_form_and_the_field_only(fixture: dict[str, Any]) -> None:
    data = fixture["ode_analysis"]["data"]
    report = data["report"]
    rows = ode_analysis(data)
    assert [k for k, _ in rows] == EXPECTED_KEYS["ode_analysis"]
    texts = {latex for _, latex in rows}
    assert not texts & set(
        report["interpretation"]["equations_latex"]
    )  # the user's own input, quoted
    assert report["qualitative"]["equilibria"] and not any(
        p in texts for e in report["qualitative"]["equilibria"] for p in e["point_latex"]
    )


def test_an_analysis_that_was_not_accepted_or_a_failed_qualitative_part_lists_nothing(
    fixture: dict[str, Any],
) -> None:
    data = copy.deepcopy(fixture["ode_analysis"]["data"])
    data["report"]["qualitative"]["status"] = "unsupported"
    assert [k for k, _ in ode_analysis(data)] == ["solved:0"]
    data["report"]["status"] = "unsupported"
    assert ode_analysis(data) == []


# --- persistence and provenance -----------------------------------------------------------------------------
def test_numbers_survive_a_rebuild_and_are_never_reused(fixture: dict[str, Any]) -> None:
    card = card_from("solve", fixture["solve"]["data"])
    table = number(card)
    stored = Row("assistant", {"blocks": [{"kind": "card", "card": card.model_dump(mode="json")}]})
    rebuilt = build_table([stored], counter=table.counter + 4)
    assert {n: e.latex for n, e in rebuilt.items.items()} == {
        n: e.latex for n, e in table.items.items()
    }
    assert all(e.part_of == "solve" for e in rebuilt.items.values())
    again = card_from("ode_analysis", fixture["ode_analysis"]["data"], "c2")
    number(again, rebuilt)
    assert min(again.numbers.values()) == table.counter + 5


def test_a_conversation_from_before_these_numbers_is_numbered_when_read(
    fixture: dict[str, Any],
) -> None:
    card = card_from("ode_solution", fixture["ode_solution"]["data"]).model_dump(mode="json")
    content = {"blocks": [{"kind": "card", "card": card}]}
    table = EquationTable()
    assert number_stored([("assistant", content)], table) is True
    assert content["blocks"][0]["card"]["numbers"] == {"solution:0:0": 1}


def test_a_row_reused_by_number_is_exact_latex_and_says_it_is_one_component(
    fixture: dict[str, Any],
) -> None:
    card = card_from("solve", fixture["solve"]["data"])
    table = number(card)
    n = card.numbers["factor:2"]
    store = ResultStore.from_prior({}, {}, table.items, unverified={})
    resolved = store.resolve({"latex": f"@e{n}"})
    assert resolved.error is None and resolved.arguments["latex"] == table.items[n].latex
    assert resolved.unverified == [f"equation ({n}){COMPONENT_SUFFIX}"]


def test_a_result_built_on_a_solve_factor_is_quoted_never_verified(
    ctx: ToolContext, fixture: dict[str, Any]
) -> None:
    card = card_from("solve", fixture["solve"]["data"])
    table = number(card)
    n = card.numbers["factor:2"]
    message, _ = play(
        ctx,
        [
            call("differentiate", {"latex": f"@e{n}", "variables": [{"name": "x"}]}),
            say("done"),
        ],
        equations=table,
    )
    step = message["trace"][0]
    assert step["status"] == "ok" and step["verified_latex"] == []
    assert "Input taken from a list" in step["for_model"]
    assert card_of(message)["data"]["unverified_inputs"] == [f"equation ({n}){COMPONENT_SUFFIX}"]


# --- the whole result is numbered next to a list ---------------------------------------------------------------
def _matrix_data() -> dict[str, Any]:
    return {
        "result": {
            "status": "ok",
            "evaluated": True,
            "result_kind": "expression",
            "result_latex": "A + B",
        }
    }


def test_a_list_card_with_a_primary_result_gets_both_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(LIST_READERS, "matrix", lambda data: [("row:0", "C = D")])
    card = card_from("matrix", _matrix_data())
    table = number(card)
    assert card.number == 1 and card.numbers == {"row:0": 2}
    # a conversation numbered before this: the same two numbers when it is read
    stored = card_from("matrix", _matrix_data(), "c2").model_dump(mode="json")
    content = {"blocks": [{"kind": "card", "card": stored}]}
    fresh = EquationTable()
    assert number_stored([("assistant", content)], fresh) is True
    assert content["blocks"][0]["card"]["number"] == 1
    assert content["blocks"][0]["card"]["numbers"] == {"row:0": 2}
    # and a table rebuilt from the stored message knows both
    done = Row("assistant", {"blocks": [{"kind": "card", "card": card.model_dump(mode="json")}]})
    rebuilt = build_table([done])
    assert {n: e.latex for n, e in rebuilt.items.items()} == {
        n: e.latex for n, e in table.items.items()
    }
