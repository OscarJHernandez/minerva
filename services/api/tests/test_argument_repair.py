"""Argument shapes a model gets wrong are repaired from a closed list, and the repair is reported.

Regression: `coordinates: "X, Y, s"` and an `expression` field on a tool whose formula is `lagrangian` cost a model
call each. The schema decides the target in every case here; anything it does not decide stays an
`invalid_arguments` refusal.
"""

import pytest
from minerva_api.agent.arguments import repair_arguments
from minerva_api.agent.loop import AgentLimits
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import TOOLS, ToolContext, check_arguments
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

from .test_agent_reliability import final, play

LAGRANGIAN = TOOLS["analyze_lagrangian"].args_model
ODE = TOOLS["ode_solve_symbolic"].args_model


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def test_valid_arguments_are_returned_as_they_came() -> None:
    arguments = {"lagrangian": "1/2*m*x_dot^2", "coordinates": ["x"]}
    repaired, notes = repair_arguments(LAGRANGIAN, arguments)
    assert repaired is arguments and notes == ()


@pytest.mark.parametrize("given", ["X, Y, s", "[X, Y, s]", '["X", "Y", "s"]', "X,Y,s"])
def test_a_list_of_names_given_as_one_string_is_split(given: str) -> None:
    repaired, notes = repair_arguments(LAGRANGIAN, {"lagrangian": "x", "coordinates": given})
    assert repaired["coordinates"] == ["X", "Y", "s"]
    assert notes and notes[0].kind.value == "shape"


def test_a_string_that_is_not_a_list_of_names_is_not_split() -> None:
    repaired, notes = repair_arguments(LAGRANGIAN, {"lagrangian": "x", "coordinates": "x + y"})
    assert repaired["coordinates"] == "x + y" and notes == ()


def test_one_equation_where_a_list_is_wanted_becomes_a_list_of_one() -> None:
    repaired, _ = repair_arguments(ODE, {"equations": "y' = -2 y", "independent": "x"})
    assert repaired["equations"] == ["y' = -2 y"]


def test_a_json_object_given_as_text_is_parsed() -> None:
    repaired, notes = repair_arguments(
        LAGRANGIAN, {"lagrangian": "x", "coordinates": ["x"], "assumptions": '{"m": ["positive"]}'}
    )
    assert repaired["assumptions"] == {"m": ["positive"]} and len(notes) == 1


def test_broken_json_is_left_for_the_validator() -> None:
    arguments = {"lagrangian": "x", "coordinates": ["x"], "assumptions": '{"m": ["positive"'}
    repaired, notes = repair_arguments(LAGRANGIAN, arguments)
    assert repaired is arguments and notes == ()


def test_a_formula_under_another_name_is_moved_to_the_tools_formula_field() -> None:
    repaired, notes = repair_arguments(
        LAGRANGIAN, {"expression": "1/2*m*x_dot^2", "coordinates": ["x"]}
    )
    assert repaired == {"lagrangian": "1/2*m*x_dot^2", "coordinates": ["x"]}
    assert "lagrangian" in notes[0].target


def test_an_alias_is_not_used_when_the_tool_has_that_field_or_two_formula_fields() -> None:
    simplify = TOOLS["simplify"].args_model
    arguments = {"latex": "x", "expression": "y"}
    repaired, _ = repair_arguments(simplify, arguments)
    assert repaired["latex"] == "x"  # the real field wins; the stray one is left for the validator
    conserved = TOOLS["find_conserved_quantities"].args_model
    ambiguous, notes = repair_arguments(conserved, {"expression": "x", "coordinates": ["x"]})
    assert ambiguous["expression"] == "x" and notes == ()


def test_a_number_where_text_is_wanted_is_text() -> None:
    differentiate = TOOLS["series"].args_model
    repaired, _ = repair_arguments(differentiate, {"latex": "\\sin x", "point": 0})
    assert repaired["point"] == "0"
    assert check_arguments("series", {"latex": "\\sin x", "point": 0, "order": 5, "variable": "x"})


def test_a_repaired_call_runs_and_the_repair_is_in_the_trace(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call(
                "analyze_lagrangian",
                {"expression": "1/2*m*x_dot^2 - 1/2*k*x^2", "coordinates": "x"},
            ),
            say("It oscillates."),
        ]
    )
    message = final(play(llm, ctx, limits=AgentLimits(retry_delay_seconds=0)))
    (step,) = message["trace"]
    assert step["status"] == "ok" and step["arguments_valid"]
    assert step["arguments"]["lagrangian"] == "1/2*m*x_dot^2 - 1/2*k*x^2"
    assert step["arguments"]["coordinates"] == ["x"]
    assert "expression" in step["model_arguments"]
    assert "input cleaned:" in step["for_model"] or "shape" in step["for_model"]
