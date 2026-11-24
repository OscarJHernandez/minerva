"""`check_units` with `unknown` through the agent layer (issue #184): the input repair, the typed refusal of a
kind that is not in the table, and the label on units the user did not state.

Expected dimensions are worked out by hand (`m r^3 thetadot^2` is kg m^3 s^-2), never read from the code.
"""

from __future__ import annotations

from typing import Any

import pytest
from minerva_api.agent.arguments import repair_arguments
from minerva_api.agent.card_equations import result_formula
from minerva_api.agent.tools import ToolContext, UnitsCheckArgs, execute
from minerva_api.agent.unit_statements import assumed_quantities, stated
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

C3 = (
    r"C_{3} = - k \cos{\left(\theta \right)} + m r^{3} \dot{\theta}^{2} \cos{\left(\theta \right)}"
    r" + m r^{2} \dot{r} \dot{\theta} \sin{\left(\theta \right)}"
)
KNOWN = [
    {"name": "m", "unit": "kg"},
    {"name": "r", "unit": "m"},
    {"name": "theta", "unit": "rad"},
]


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def with_messages(ctx: ToolContext, *messages: str) -> ToolContext:
    out = ToolContext(engine=ctx.engine, kb=ctx.kb)
    out.user_messages = list(messages)
    return out


# --- the input layer -------------------------------------------------------------------------------
def test_a_unit_null_entry_is_the_quantity_being_asked_about() -> None:
    sent = {"equation": C3, "quantities": [*KNOWN, {"name": "C_3", "unit": None}]}
    fixed, notes = repair_arguments(UnitsCheckArgs, sent)
    assert fixed["unknown"] == ["C_3"]
    assert [q["name"] for q in fixed["quantities"]] == ["m", "r", "theta"]
    assert len(notes) == 1 and "unknown: C_3" in notes[0].target
    assert sent["quantities"][-1] == {
        "name": "C_3",
        "unit": None,
    }  # the model's own call is untouched


def test_an_entry_with_a_kind_or_a_unit_is_not_moved() -> None:
    sent = {
        "equation": "F = m a",
        "quantities": [{"name": "m", "kind": "mass"}, {"name": "a", "unit": "m/s^2"}],
    }
    fixed, notes = repair_arguments(UnitsCheckArgs, sent)
    assert fixed is sent and notes == ()


def test_the_call_of_the_report_is_answered(ctx: ToolContext) -> None:
    """`unit: null` for C_3 is repaired; k is also asked for, so both get the dimension."""
    sent = {
        "equation": C3,
        "quantities": [*KNOWN, {"name": "C_3", "unit": None}, {"name": "k", "unit": None}],
    }
    fixed, _ = repair_arguments(UnitsCheckArgs, sent)  # the loop repairs a call before it runs
    out = execute("check_units", fixed, ctx)
    assert out.status == "ok" and out.arguments_valid
    assert "dimension of C_3, given these inputs: [kg m^3 s^-2]" in out.for_model
    assert "dimension of k, given these inputs: [kg m^3 s^-2]" in out.for_model
    assert "read theta_dot as" in out.for_model


def test_a_kind_outside_the_table_is_refused_with_a_typed_code_naming_both_ways_out(
    ctx: ToolContext,
) -> None:
    out = execute(
        "check_units",
        {"equation": C3, "quantities": [*KNOWN, {"name": "k", "kind": "Kepler parameter"}]},
        ctx,
    )
    assert out.status == "parse_error" and out.reason_code == "unknown_quantity_kind"
    assert '"code":"unknown_quantity_kind"' in out.for_model
    assert "give the unit instead" in out.for_model.lower() and "`unknown`" in out.for_model


def test_the_exactly_one_of_unit_or_kind_text_names_unknown(ctx: ToolContext) -> None:
    out = execute(
        "check_units",
        {"equation": "F = m a", "quantities": [{"name": "m", "unit": "kg", "kind": "mass"}]},
        ctx,
    )
    assert not out.arguments_valid and "`unknown`" in out.for_model


def test_a_dimension_is_not_an_equation_with_a_number(ctx: ToolContext) -> None:
    out = execute(
        "check_units",
        {
            "equation": "F = m a",
            "quantities": [{"name": "m", "unit": "kg"}, {"name": "a", "unit": "m/s^2"}],
            "unknown": ["F"],
        },
        ctx,
    )
    assert result_formula("units", out.card.data) is None


# --- which units the user stated -------------------------------------------------------------------
LAGRANGIAN = r"Find the conserved quantities of $L = \frac12 m (\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{k}{r}$."


@pytest.mark.parametrize(
    ("name", "unit", "message"),
    [
        ("m", "kg", "The mass m of the particle is 2 kg."),
        ("m", "kg", "mass m"),
        ("m", "kg", "m is a mass"),
        ("m", "kg", "Let m be in kg."),
        ("m", "kg", "m = 2 kg"),
        ("m", "kg", "m (kg)"),
        ("r", "m", "r is the radius"),
        ("r", "m", "the radius $r$"),
        ("theta", "1", r"the angle \theta"),
        ("C_3", "kg m^3 s^-2", "C_3 in kg*m^3/s^2"),
    ],
)
def test_a_unit_the_user_wrote_next_to_the_name_is_stated(
    name: str, unit: str, message: str
) -> None:
    assert stated(name, unit, [message])


@pytest.mark.parametrize(
    ("name", "unit", "message"),
    [
        ("m", "kg", "kg"),  # a unit with no name
        ("m", "kg", "m is a length"),  # the user said another dimension
        ("m", "kg", "Mass M of the wedge."),  # another symbol (capital M)
        ("r", "m", LAGRANGIAN),  # holds the name r and the unit letter m, says nothing about either
        ("m", "kg", LAGRANGIAN),
        ("r", "m", "r m"),
        ("r", "m", "r = m"),  # an equation, not a unit
        ("k", "kg m^3 s^-2", "What are the units of C_3? in Equation (3)"),
        ("m", "kg", "The mass of the block is 3 and r is the radius."),
    ],
)
def test_everything_else_is_assumed(name: str, unit: str, message: str) -> None:
    assert not stated(name, unit, [message])


def test_assumed_quantities_lists_only_those_not_stated() -> None:
    messages = ["The mass m is 2 kg and the radius is r."]
    pairs = [("m", "kg"), ("r", "m"), ("theta", "1")]
    assert assumed_quantities(pairs, messages) == ["r", "theta"]


def test_the_card_and_the_text_say_which_units_the_assistant_assumed(ctx: ToolContext) -> None:
    sentence = "Take the mass m in kg."
    chat = with_messages(ctx, LAGRANGIAN, sentence, "What are the units of C_3? in Equation (3)")
    out = execute(
        "check_units", {"equation": C3, "quantities": KNOWN, "unknown": ["C_3", "k"]}, chat
    )
    assert out.card.data["assumed_units"] == ["r", "theta"]  # m was stated, r and theta were not
    assert "assumed by the assistant): r, theta" in out.for_model
    assert "given these inputs" in out.for_model


def test_the_label_never_changes_the_computation(ctx: ToolContext) -> None:
    call: dict[str, Any] = {"equation": C3, "quantities": KNOWN, "unknown": ["C_3", "k"]}
    unlabelled = execute("check_units", call, ctx)  # not a chat turn: nothing is labelled
    stated_all = execute(
        "check_units", call, with_messages(ctx, "mass m in kg, radius r, angle theta")
    )
    none_stated = execute("check_units", call, with_messages(ctx, "hello"))
    assert unlabelled.card.data["assumed_units"] == []
    assert none_stated.card.data["assumed_units"] == ["m", "r", "theta"]
    assert (
        unlabelled.card.data["report"]["derived"]
        == stated_all.card.data["report"]["derived"]
        == none_stated.card.data["report"]["derived"]
    )
