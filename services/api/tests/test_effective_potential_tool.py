"""The `effective_potential` tool (issue #207): its schema, its card, its numbers and the handle of the result.

The normalisation `g(v, v) = kappa` is the user's number: it is a required argument with no default and the schema
says to ask. The momenta keep the verified names (`p_{t}`, `p_{\\phi}`); there is no argument for a value or another
name. The radial equation is the card's single result (one number, a result id that carries the restrictions);
`V_eff` and the held momenta are numbered list rows (names only).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from minerva_api.agent.card_equations import (
    LIST_READERS,
    effective_potential,
    listed_equations,
    result_formula,
)
from minerva_api.agent.equations import EquationTable, number_blocks
from minerva_api.agent.events import Block
from minerva_api.agent.refs import REFERENCEABLE_TOOLS, ResultStore
from minerva_api.agent.tools import TOOLS, ToolContext, ToolOutput, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_equation_rows_contract import FIXTURE

F = r"\left(1-\frac{2M}{r}\right)"
ARGS: dict[str, Any] = {
    "coordinates": ["t", "r", "theta", "phi"],
    "metric": [
        [f"-{F}", "0", "0", "0"],
        ["0", f"{F}^{{-1}}", "0", "0"],
        ["0", "0", "r^2", "0"],
        ["0", "0", "0", r"r^2\sin^2\theta"],
    ],
    "assumptions": {"M": ["positive"], "r": ["positive"]},
    "cyclic": ["t", "phi"],
    "radial": "r",
    "fixed": {"theta": r"\frac{\pi}{2}"},
    "normalisation": "-1",
}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


@pytest.fixture(scope="module")
def result(ctx: ToolContext) -> ToolOutput:
    out = execute("effective_potential", ARGS, ctx)
    assert out.status == "ok", out.for_model
    return out


def test_the_normalisation_is_required_and_the_schema_says_to_ask() -> None:
    schema = TOOLS["effective_potential"].schema()["function"]["parameters"]
    assert "normalisation" in schema["required"]
    assert "default" not in schema["properties"]["normalisation"]
    description = schema["properties"]["normalisation"]["description"]
    assert "USER" in description and "ask" in description and "never choose" in description
    # no field for a momentum's value or name
    assert set(schema["properties"]) == {
        "coordinates",
        "metric",
        "format",
        "assumptions",
        "cyclic",
        "radial",
        "fixed",
        "normalisation",
    }


def test_a_call_without_the_normalisation_is_refused_before_any_computation(
    ctx: ToolContext,
) -> None:
    out = execute(
        "effective_potential", {k: v for k, v in ARGS.items() if k != "normalisation"}, ctx
    )
    assert out.arguments_valid is False
    assert out.known_latex == []


def test_the_card_holds_the_radial_equation_the_potential_and_the_held_momenta(
    result: ToolOutput,
) -> None:
    data = result.card.data
    assert result.card.type == "effective_potential" and result.card.tool == "effective_potential"
    assert data["status"] == "ok" and data["result_latex"]
    assert [h["symbol_latex"] for h in data["held_fixed"]] == ["p_{t}", "p_{\\phi}"]
    assert data["fixed"] == [r"\theta = \frac{\pi}{2}"]
    assert data["consistency"][0]["symbolic"] is True
    assert data["restrictions_latex"]
    assert data["request"]["normalisation"] == "-1"
    text = result.for_model
    assert f"${data['result_latex']}$" in text
    assert "never rename it yourself" in text and "substitute" in text
    assert "not statements about singularities" in text
    assert data["result_latex"] in result.known_latex
    assert result.evidence is not None and result.evidence.accepted


def test_the_radial_equation_is_the_single_result_and_the_rest_are_rows(
    result: ToolOutput,
) -> None:
    data = result.card.data
    assert result_formula("effective_potential", data) == data["result_latex"]
    rows = listed_equations("effective_potential", data)
    assert [k for k, _ in rows] == ["veff", "held:t", "held:phi"]
    card = result.card.model_copy(deep=True)
    table = EquationTable()
    number_blocks([Block(kind="card", card=card)], table)
    assert card.number == 1 and card.numbers == {"veff": 2, "held:t": 3, "held:phi": 4}


def test_a_card_that_was_not_accepted_lists_nothing() -> None:
    data = {"status": "not_verified", "veff_latex": "V = 1", "held_fixed": []}
    assert effective_potential(data) == []
    assert result_formula("effective_potential", {"status": "unsupported"}) is None


def test_the_result_id_names_the_radial_equation_with_its_restrictions(result: ToolOutput) -> None:
    assert "effective_potential" in REFERENCEABLE_TOOLS
    store = ResultStore()
    assert store.register("effective_potential", result) == "r1"
    assert store.results["r1"] == result.card.data["result_latex"]
    assert store.restrictions["r1"] == result.card.data["restrictions_latex"]


def test_an_equatorial_restriction_that_is_not_consistent_is_a_typed_refusal(
    ctx: ToolContext,
) -> None:
    out = execute("effective_potential", {**ARGS, "fixed": {"theta": r"\frac{\pi}{3}"}}, ctx)
    assert out.status == "unsupported"
    assert out.reason_code == "restriction_not_consistent"
    assert '"code":"restriction_not_consistent"' in out.for_model
    assert "reason: equatorial_not_consistent" in out.for_model
    assert out.known_latex == []
    assert ResultStore().register("effective_potential", out) is None


def test_an_unreadable_normalisation_is_a_typed_refusal(ctx: ToolContext) -> None:
    out = execute("effective_potential", {**ARGS, "normalisation": "timelike"}, ctx)
    assert out.status == "parse_error"
    assert out.reason_code == "normalisation_unreadable"


def test_text_syntax_is_chosen_for_the_whole_call(ctx: ToolContext) -> None:
    text = {
        **ARGS,
        "metric": [
            ["-(1-2*M/r)", "0", "0", "0"],
            ["0", "1/(1-2*M/r)", "0", "0"],
            ["0", "0", "r^2", "0"],
            ["0", "0", "0", "r^2*sin(theta)^2"],
        ],
        "fixed": {"theta": "pi/2"},
    }
    out = execute("effective_potential", text, ctx)
    assert out.status == "ok", out.for_model
    assert out.card.data["request"]["format"] == "text"


def test_the_fixture_the_browser_renders_is_the_reader_and_a_live_card(result: ToolOutput) -> None:
    entry = json.loads(FIXTURE.read_text())["effective_potential"]
    rows = listed_equations("effective_potential", entry["data"])
    assert [k for k, _ in rows] == entry["keys"] == ["veff", "held:t", "held:phi"]
    assert dict(rows) == entry["rows"]
    assert result_formula("effective_potential", entry["data"]) == entry["data"]["result_latex"]
    assert [k for k, _ in listed_equations("effective_potential", result.card.data)] == entry[
        "keys"
    ]
    assert "effective_potential" in LIST_READERS


# --- the tool group: with Routh reduction, not in the core and not in mechanics ----------------------------
def test_the_tool_is_in_the_reduction_group() -> None:
    from minerva_api.agent.prompts import GROUP_TOOLS
    from minerva_api.agent.toolsets import group_of

    assert group_of("effective_potential") == "reduction"
    assert "effective_potential" in GROUP_TOOLS["reduction"]
    assert "effective_potential" not in GROUP_TOOLS["mechanics"]


@pytest.mark.parametrize(
    "text",
    [
        "Give the radial effective potential for timelike geodesics of the Schwarzschild metric.",
        "What is the effective-potential for light rays here?",
    ],
)
def test_a_keyword_loads_the_group(text: str) -> None:
    from minerva_api.agent.toolsets import groups_in_text

    assert "reduction" in groups_in_text(text)
