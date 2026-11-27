"""A card that lists equations (Christoffel symbols, equations of motion, constants found) numbers each one.

A number is a name, never a verification: a component reused by number is exact LaTeX but is not linked to
the verification or the restrictions of the card it came from, and a result built on one says so.
"""

import asyncio
import typing
from typing import Any

import pytest
from minerva_api.agent.card_equations import (
    EXCLUDED_WITH_REASON,
    LIST_NOT_YET_NUMBERED,
    LIST_READERS,
    SINGLE_RESULT,
    geometry,
    invariants,
    lagrangian,
    listed_equations,
)
from minerva_api.agent.equations import (
    EquationTable,
    build_table,
    find_references,
    number_blocks,
    number_stored,
    resolve_references,
)
from minerva_api.agent.events import Block, Card
from minerva_api.agent.loop import run_agent
from minerva_api.agent.refs import ResultStore
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import GeometryArgs, LagrangianArgs, ToolContext, run_geometry
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

SPHERE = {
    "coordinates": ["theta", "phi"],
    "metric": [["a^2", "0"], ["0", r"a^2 \sin^2\theta"]],
    "assumptions": {"a": ["positive"]},
}


class Row:
    def __init__(self, role: str, content: dict[str, Any]) -> None:
        self.role, self.content = role, content


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def geometry_card(ctx: ToolContext, *quantities: str, card_id: str = "g1") -> Card:
    output = run_geometry(GeometryArgs(**SPHERE, quantities=list(quantities)), ctx)  # type: ignore[arg-type]
    output.card.id = card_id
    return output.card


def number(card: Card, table: EquationTable) -> None:
    number_blocks([Block(kind="card", card=card)], table)


# --- numbering ---------------------------------------------------------------------------------------------
def test_every_card_type_is_classified_for_equation_numbers() -> None:
    kinds = set(typing.get_args(Card.model_fields["type"].annotation))
    sets = [
        set(LIST_READERS),
        set(SINGLE_RESULT),
        set(LIST_NOT_YET_NUMBERED),
        set(EXCLUDED_WITH_REASON),
    ]
    assert set().union(*sets) == kinds, kinds.symmetric_difference(set().union(*sets))
    assert sum(len(s) for s in sets) == len(kinds)  # in exactly one


def test_every_exclusion_states_its_reason() -> None:
    assert set(EXCLUDED_WITH_REASON) == {"ode_numeric", "motion", "multivariate_normal"}
    assert all(len(reason) > 30 for reason in EXCLUDED_WITH_REASON.values())


def test_each_christoffel_symbol_gets_its_own_whole_number_in_display_order(
    ctx: ToolContext,
) -> None:
    card = geometry_card(ctx, "christoffel")
    rows = geometry(card.data)
    assert len(rows) == 2 and all(
        " = " in latex for _, latex in rows
    )  # `label = value`, as the row shows
    table = EquationTable()
    number(card, table)
    assert [card.numbers[k] for k, _ in rows] == list(range(1, len(rows) + 1))
    assert table.counter == len(rows)
    assert all(table.items[n].part_of == "geometry" for n in card.numbers.values())
    assert card.number is None  # a list has no single number of its own


def test_a_scalar_quantity_is_numbered_too_and_the_next_card_continues_the_count(
    ctx: ToolContext,
) -> None:
    table = EquationTable()
    first = geometry_card(ctx, "christoffel", card_id="g1")
    number(first, table)
    after_first = table.counter
    second = geometry_card(ctx, "kretschmann", card_id="g2")
    number(second, table)
    assert second.numbers == {"kretschmann": after_first + 1}
    assert (
        table.items[after_first + 1].latex == second.data["result"]["quantities"][0]["scalar_latex"]
    )


def test_numbers_are_stable_when_the_table_is_rebuilt_and_never_reused(ctx: ToolContext) -> None:
    table = EquationTable()
    card = geometry_card(ctx, "christoffel")
    number(card, table)
    stored = Row("assistant", {"blocks": [{"kind": "card", "card": card.model_dump(mode="json")}]})
    rebuilt = build_table(
        [stored], counter=table.counter + 5
    )  # numbers were given and later removed
    assert {n: e.latex for n, e in rebuilt.items.items()} == {
        n: e.latex for n, e in table.items.items()
    }
    assert all(e.part_of == "geometry" for e in rebuilt.items.values())
    again = geometry_card(ctx, "ricci_scalar", card_id="g2")
    number(again, rebuilt)
    assert min(again.numbers.values()) == table.counter + 6  # the high-water mark is never lowered


def test_a_conversation_from_before_list_numbers_is_numbered_when_it_is_read(
    ctx: ToolContext,
) -> None:
    card = geometry_card(ctx, "christoffel").model_dump(mode="json")
    assert card["numbers"] == {}
    content = {"blocks": [{"kind": "card", "card": card}]}
    table = EquationTable()
    assert number_stored([("assistant", content)], table) is True
    assert sorted(content["blocks"][0]["card"]["numbers"].values()) == list(
        range(1, table.counter + 1)
    )


def test_a_quantity_that_was_not_accepted_lists_no_equations() -> None:
    data = {
        "result": {
            "quantities": [
                {"kind": "ricci", "status": "not_verified", "scalar_latex": "R", "components": []},
                {
                    "kind": "christoffel",
                    "status": "not_verified",
                    "scalar_latex": None,
                    "components": [{"indices": [0, 0, 1], "label_latex": "G", "latex": "x"}],
                },
            ]
        }
    }
    assert geometry(data) == []


def test_lagrangian_and_search_cards_list_their_equations_in_the_order_shown() -> None:
    report = {
        "equations": [{"latex": "e0"}, {"latex": "e1"}],
        "momenta": [{"latex": "p0"}],
        "accelerations": [{"latex": "a0"}],
        "conserved": [{"expression_latex": "h = H"}],
    }
    assert lagrangian({"report": report}) == [
        ("eq:0", "e0"),
        ("eq:1", "e1"),
        ("mom:0", "p0"),
        ("acc:0", "a0"),
        ("cons:0", "h = H"),
    ]
    assert invariants({"report": {"found": [{"expression_latex": "C_1 = x"}, {}]}}) == [
        ("found:0", "C_1 = x")
    ]
    assert listed_equations("simplify", {"report": report}) == []  # not a list card


def test_a_real_kepler_analysis_numbers_its_equations_and_conserved_quantities(
    ctx: ToolContext,
) -> None:
    args = LagrangianArgs(
        lagrangian=r"\frac12 m(\dot{r}^2 + r^2\dot{\theta}^2) + \frac{k}{r}",
        coordinates=["r", "theta"],
    )
    llm = ReplayLLM([call("analyze_lagrangian", args.model_dump()), say("Done.")])

    async def go() -> list[dict[str, Any]]:
        history = [{"role": "user", "content": "analyse it"}]
        return [e async for e in run_agent(llm, history, ctx)]

    message = next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")
    card = next(b["card"] for b in message["blocks"] if b["kind"] == "card")
    keys = [k for k, _ in lagrangian(card["data"])]
    assert {"eq:0", "eq:1", "cons:0"} <= set(keys)
    assert card["numbers"] == {}  # the server numbers after the turn, never the loop or the model
    numbered = Card.model_validate(card)
    number(numbered, EquationTable())
    assert set(numbered.numbers) == set(keys)
    # a formula shown twice (the momentum and the conserved quantity are the same equation) keeps one number
    distinct = {latex for _, latex in lagrangian(card["data"])}
    assert sorted(set(numbered.numbers.values())) == list(range(1, len(distinct) + 1))


# --- references ---------------------------------------------------------------------------------------------
def test_a_reference_to_a_component_resolves_to_its_latex_and_says_what_it_is(
    ctx: ToolContext,
) -> None:
    card = geometry_card(ctx, "christoffel")
    table = EquationTable()
    number(card, table)
    shown, note = resolve_references(find_references("What is equation 2?"), table)
    assert shown[0]["latex"] == table.items[2].latex and shown[0]["source"] == "from_tool"
    assert "one row of a geometry tool result" in note and "NOT verified by Minerva" not in note
    unknown, _ = resolve_references(find_references(f"equation {table.counter + 1}"), table)
    assert unknown[0]["source"] == "unknown"


def test_a_component_reused_by_number_is_marked_as_not_linked_to_its_card(ctx: ToolContext) -> None:
    card = geometry_card(ctx, "ricci_scalar")
    table = EquationTable()
    number(card, table)
    store = ResultStore.from_prior({}, {}, table.items, unverified={})
    r = store.resolve({"latex": "@e1"})
    assert r.error is None and r.arguments["latex"] == table.items[1].latex
    assert r.unverified == ["equation (1), a component of an earlier result"]
    embedded = store.resolve({"latex": "@e1 + 1"})
    assert embedded.arguments["latex"] == rf"\left({table.items[1].latex}\right) + 1"


def test_a_result_built_on_a_component_says_so_and_is_not_marked_verified(
    ctx: ToolContext,
) -> None:
    card = geometry_card(ctx, "ricci_scalar")
    table = EquationTable()
    number(card, table)
    llm = ReplayLLM(
        [
            call("differentiate", {"latex": "@e1", "variables": [{"name": "a"}]}),
            call("differentiate", {"latex": "@r1", "variables": [{"name": "a"}]}),
            say("done"),
        ]
    )

    async def go() -> list[dict[str, Any]]:
        history = [{"role": "user", "content": "differentiate equation 1"}]
        return [e async for e in run_agent(llm, history, ctx, equations=table)]

    message = next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")
    first, second = message["trace"][0], message["trace"][1]
    assert first["status"] == "ok" and first["verified_latex"] == []
    assert "Input taken from a list" in first["for_model"]
    assert "Input not verified" not in first["for_model"]  # it was not written by the model
    cards = [b["card"] for b in message["blocks"] if b["kind"] == "card"]
    label = "equation (1), a component of an earlier result"
    assert cards[0]["data"]["unverified_inputs"] == [label]
    assert "Input taken from a list" in second["for_model"]  # the mark travels through @r1
    assert second["verified_latex"] == []
