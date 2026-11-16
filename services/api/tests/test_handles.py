"""The model is told the handles of the formulas the user displayed, so it references them instead of copying.

Regression: a recorded turn gave the model the user's Lagrangian and the model re-typed it in a tool argument,
breaking the velocities. `@eN` already replaced a handle by the user's exact text, but the model was never told
the numbers of the message it was answering.
"""

import pytest
from minerva_api.agent.equations import EquationTable, handles_note, number_user_text
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

from .test_agent_reliability import final, play

OSCILLATOR = r"L = \frac12 m \dot{x}^2 - \frac12 k x^2"


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def test_no_displayed_formula_no_note() -> None:
    table = EquationTable()
    assert handles_note(number_user_text("Find the equations of motion for $L = x$.", table)) == ""


def test_the_note_lists_the_handles_in_the_order_written() -> None:
    table = EquationTable()
    numbered = number_user_text(f"Use\n\n$${OSCILLATOR}$$\n\nand\n\n$$E = \\frac12 m v^2$$", table)
    note = handles_note(numbered)
    assert "@e1, @e2" in note and "instead of copying" in note
    assert (
        "not from the user" in note and OSCILLATOR not in note
    )  # handles only, the formula is above


def test_a_long_list_is_cut_and_says_so() -> None:
    table = EquationTable()
    text = "\n".join(f"$${n}x + 1$$" for n in range(1, 13))
    note = handles_note(number_user_text(text, table))
    assert "@e8" in note and "@e9" not in note and "(and 4 more)" in note


def _oscillator_turn(ctx: ToolContext, handle: str, table: EquationTable) -> dict:  # type: ignore[type-arg]
    llm = ReplayLLM(
        [call("analyze_lagrangian", {"lagrangian": handle, "coordinates": ["x"]}), say("Done.")]
    )
    return final(play(llm, ctx, equations=table))


def test_a_handle_in_a_tool_call_is_the_users_formula_character_for_character(
    ctx: ToolContext,
) -> None:
    table = EquationTable()
    number_user_text(f"$${OSCILLATOR}$$", table)
    (step,) = _oscillator_turn(ctx, "@e1", table)["trace"]
    assert step["status"] == "ok", step["for_model"][:300]
    assert step["model_arguments"]["lagrangian"] == "@e1"
    # The user's text, with the leading `L =` removed by the cleaning that every argument gets afterwards.
    assert step["arguments"]["lagrangian"] == r"\frac12 m \dot{x}^2 - \frac12 k x^2"
    assert "input cleaned" in step["for_model"]


def test_the_users_own_formula_is_never_marked_verified_by_being_used(ctx: ToolContext) -> None:
    table = EquationTable()
    number_user_text(f"$${OSCILLATOR}$$", table)
    (step,) = _oscillator_turn(ctx, "@e1", table)["trace"]
    assert step["verified_latex"]  # what the engine derived is verified ...
    assert OSCILLATOR not in step["verified_latex"]  # ... the user's own formula is not


def test_an_unknown_handle_is_refused_not_guessed(ctx: ToolContext) -> None:
    (step,) = _oscillator_turn(ctx, "@e7", EquationTable())["trace"]
    assert step["status"] == "error" and "unknown equation reference @e7" in step["for_model"]
