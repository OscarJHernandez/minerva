"""Issue #132: a relation among constants of motion found earlier ("the derivative of Equation 14 with
respect to time"). The server, never the model, inserts the verified definitions of the names `C_{1}`...
that an earlier discovery result listed; everything else is a typed refusal."""

from __future__ import annotations

from typing import Any

import pytest
from minerva_api.agent.arguments import repair_arguments
from minerva_api.agent.constants import (
    Expansion,
    KnownConstant,
    Refusal,
    constants_of_blocks,
    expand_constants,
)
from minerva_api.agent.replay import call, say
from minerva_api.agent.tools import TimeDerivativeArgs, ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

POLAR = r"\frac{\mu}{2}(\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{G M \mu}{r}"
RELATION = r"2 C_{1}^{2} C_{2} - C_{3}^{2} \mu^{3} - C_{4}^{2} \mu^{3} + G^{2} M^{2} \mu^{3}"
EXPRESSIONS = {
    "C_{1}": r"\mu r^{2} \dot{\theta}",
    "C_{2}": r"- \frac{G M \mu}{r} + \frac{\mu r^{2} \dot{\theta}^{2}}{2} + \frac{\mu \dot{r}^{2}}{2}",
    "C_{3}": r"- G M \cos{\theta} + r^{3} \dot{\theta}^{2} \cos{\theta} + r^{2} \dot{r} \dot{\theta} \sin{\theta}",
    "C_{4}": r"G M \sin{\theta} - r^{3} \dot{\theta}^{2} \sin{\theta} + r^{2} \dot{r} \dot{\theta} \cos{\theta}",
}


def card(
    card_id: str = "d1",
    *,
    system: str = "lagrangian",
    function: str = POLAR,
    verified: bool = True,
    expressions: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "id": card_id,
        "type": "discovery",
        "data": {
            "report": {
                "system": system,
                "function_latex": function,
                "coordinates": ["r", "theta"],
                "constants": [
                    {"expression_latex": f"{name} = {text}", "verified": verified}
                    for name, text in (expressions or EXPRESSIONS).items()
                ],
            }
        },
    }


def known(*cards: dict[str, Any]) -> tuple[KnownConstant, ...]:
    return tuple(constants_of_blocks([[{"kind": "card", "card": c} for c in cards]]))


@pytest.fixture(scope="module")
def engine_ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=120, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def with_constants(base: ToolContext, constants: tuple[KnownConstant, ...]) -> ToolContext:
    return ToolContext(engine=base.engine, kb=base.kb, constants=constants)


def test_constants_are_read_from_the_stored_discovery_cards_only() -> None:
    found = known(card())
    assert [c.name for c in found] == ["C_1", "C_2", "C_3", "C_4"]
    assert found[0].expression == r"\mu r^{2} \dot{\theta}" and found[0].function == POLAR
    other = {"id": "x", "type": "differentiate", "data": {"report": card()["data"]["report"]}}
    assert known(other) == ()
    assert constants_of_blocks([[{"kind": "text"}, {"kind": "card", "card": "not a card"}]]) == []


def test_only_the_names_the_expression_uses_are_inserted() -> None:
    result = expand_constants(r"C_{1}^2 + C_2 + x", known(card()))
    assert isinstance(result, Expansion) and set(result.definitions) == {"C_1", "C_2"}
    assert result.function == POLAR
    assert expand_constants(r"C_{12} + D_1", known(card())) is None  # no prefix match, no stranger
    assert expand_constants(RELATION, ()) is None


def test_a_name_with_two_definitions_or_an_unverified_one_is_refused_never_guessed() -> None:
    changed = {**EXPRESSIONS, "C_{1}": r"\mu r \dot{\theta}"}
    two = expand_constants(RELATION, known(card("d1"), card("d2", expressions=changed)))
    assert (
        isinstance(two, Refusal) and two.code.value == "ambiguous_constant" and "C_1" in two.symbols
    )
    again = expand_constants(
        RELATION, known(card("d1"), card("d2"))
    )  # the same search twice: one meaning
    assert isinstance(again, Expansion)
    unverified = expand_constants(RELATION, known(card(verified=False)))
    assert isinstance(unverified, Refusal) and unverified.code.value == "constant_not_verified"
    hamiltonian = expand_constants(RELATION, known(card(system="hamiltonian")))
    assert isinstance(hamiltonian, Refusal) and hamiltonian.code.value == "unsupported_definition"
    systems = expand_constants(
        r"C_1 + D_1",
        known(
            card("d1"), card("d2", function=r"\frac12 \dot{r}^2", expressions={"D_{1}": r"\dot{r}"})
        ),
    )
    assert isinstance(systems, Refusal) and systems.code.value == "ambiguous_constant"


def test_time_derivative_of_the_relation_is_zero_on_the_motion(engine_ctx: ToolContext) -> None:
    ctx = with_constants(engine_ctx, known(card()))
    out = execute(
        "time_derivative",
        {"latex": RELATION, "coordinates": ["r", "theta"], "lagrangian": POLAR},
        ctx,
    )
    assert out.status == "ok", out.for_model
    assert out.card.data["zero_on_motion"] is True and len(out.card.data["definitions_used"]) == 4
    assert "constants replaced by their verified expressions" in out.for_model
    assert "zero on the equations of motion" in out.for_model
    assert len(out.for_model) < 3000  # the long unsimplified derivative stays on the card


def test_without_a_lagrangian_the_one_the_constants_were_found_for_is_used(
    engine_ctx: ToolContext,
) -> None:
    ctx = with_constants(engine_ctx, known(card()))
    out = execute("time_derivative", {"latex": RELATION, "coordinates": ["r", "theta"]}, ctx)
    assert out.status == "ok" and out.card.data["zero_on_motion"] is True


def test_bare_names_with_no_earlier_result_are_a_typed_refusal(engine_ctx: ToolContext) -> None:
    out = execute(
        "time_derivative",
        {"latex": RELATION, "coordinates": ["r", "theta"]},
        with_constants(engine_ctx, ()),
    )
    assert out.status == "unsupported" and out.reason_code == "no_dynamic_dependence"
    assert "diagnostic" in out.for_model


def test_a_refused_constant_is_reported_with_its_code(engine_ctx: ToolContext) -> None:
    ctx = with_constants(engine_ctx, known(card(verified=False)))
    out = execute("time_derivative", {"latex": RELATION, "coordinates": ["r", "theta"]}, ctx)
    assert out.status == "error" and out.reason_code == "constant_not_verified"
    assert "diagnostic" in out.for_model and "not verified" in out.for_model


def test_the_model_has_no_way_to_supply_a_definition(engine_ctx: ToolContext) -> None:
    assert "definitions" not in TimeDerivativeArgs.model_json_schema()["properties"]
    # a stray key is dropped by validation and never reaches the engine: the expression stays a bare name
    ctx = with_constants(engine_ctx, ())
    out = execute(
        "time_derivative",
        {"latex": "C_1", "coordinates": ["r"], "definitions": {"C_1": r"r \dot{r}"}},
        ctx,
    )
    assert out.status == "unsupported" and out.reason_code == "no_dynamic_dependence"


def test_an_equation_equal_to_zero_is_repaired_to_the_quantity_and_says_so() -> None:
    fixed, notes = repair_arguments(
        TimeDerivativeArgs, {"latex": RELATION + " = 0", "coordinates": ["r", "theta"]}
    )
    assert fixed["latex"] == RELATION and len(notes) == 1 and "quantity" in notes[0].note
    for untouched in (RELATION + " = 5", RELATION + r" = \mu", "a = b = 0", r"x \leq 0"):
        same, none = repair_arguments(
            TimeDerivativeArgs, {"latex": untouched, "coordinates": ["r"]}
        )
        assert same["latex"] == untouched and not none


def test_the_relation_of_a_real_discovery_is_differentiated_through_the_chat_route() -> None:
    """The five turns of the report, with the real engine: Lagrangian, constants, derivative of the relation."""
    from .test_chat_context import send, start

    for client, cid, llm in start():
        llm.replies = [
            call(
                "find_conserved_quantities",
                {"lagrangian": POLAR, "coordinates": ["r", "theta"], "budget": "thorough"},
            ),
            say("Four constants were found."),
            call(
                "time_derivative",
                {"latex": RELATION + " = 0", "coordinates": ["r", "theta"], "lagrangian": POLAR},
            ),
            say("It is zero on the equations of motion."),
        ]
        send(client, cid, "Find the hidden conserved quantities of the Kepler problem.")
        events = send(client, cid, "Take the derivative of the relation with respect to time.")
        cards = [d for e, d in events if e == "card"]
        assert cards and cards[-1]["type"] == "differentiate", cards
        data = cards[-1]["data"]
        assert data["status"] == "ok" and data["zero_on_motion"] is True
        assert len(data["definitions_used"]) >= 3
