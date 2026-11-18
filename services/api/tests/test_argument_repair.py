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


VECTOR = TOOLS["vector_calculus"].args_model
SURFACE = {
    "q": ["u", "v", "u*v"],
    "u": {"name": "u", "lower": "0", "upper": "1"},
    "v": {"name": "v", "lower": "0", "upper": "1"},
}


def stokes(**over):  # type: ignore[no-untyped-def]
    base = {
        "action": "integrate",
        "integral": "stokes_theorem",
        "fields": [{"name": "F", "kind": "vector", "components": ["-y*z", "x*z", "x*y"]}],
        "surface": SURFACE,
    }
    return {**base, **over}


def test_a_theorem_without_an_expression_is_about_the_one_declared_field_of_its_kind() -> None:
    repaired, notes = repair_arguments(VECTOR, stokes())
    assert repaired["expression"] == "F"
    assert len(notes) == 1 and notes[0].kind.value == "shape" and "F" in notes[0].note


@pytest.mark.parametrize(
    "arguments",
    [
        stokes(expression="G"),  # an expression the model gave is never replaced
        stokes(
            fields=[
                {"name": "F", "kind": "vector", "components": ["x", "y", "z"]},
                {"name": "G", "kind": "vector", "components": ["y", "z", "x"]},
            ]
        ),  # two candidates: no single reading
        stokes(fields=[{"name": "f", "kind": "scalar", "components": ["x"]}]),  # wrong kind
        stokes(integral="flux"),  # not a theorem
        stokes(fields=[]),
    ],
)
def test_the_theorem_repair_never_guesses_between_fields_or_replaces_a_given_expression(
    arguments: dict,
) -> None:
    repaired, _ = repair_arguments(VECTOR, arguments)
    assert repaired.get("expression") == arguments.get("expression")


def test_a_blank_expression_counts_as_missing_and_the_gradient_theorem_takes_a_scalar() -> None:
    assert repair_arguments(VECTOR, stokes(expression="  "))[0]["expression"] == "F"
    grad = {
        "action": "integrate",
        "integral": "gradient_theorem",
        "fields": [{"name": "f", "kind": "scalar", "components": ["x*y"]}],
    }
    assert repair_arguments(VECTOR, grad)[0]["expression"] == "f"


# -- shapes seen in the 2026-11-18 live traces ---------------------------------------------------------------
MATRIX = TOOLS["matrix_calculus"].args_model
NUMERIC = TOOLS["ode_solve_numerical"].args_model
SIMPLIFY = TOOLS["simplify"].args_model


def test_a_matrix_keyed_by_symbol_is_named() -> None:
    given = {
        "latex": "(A+B)^2",
        "action": "simplify",
        "matrices": [{"symbol": "A", "rows": 2, "cols": 2}, {"name": "B", "rows": 2}],
    }
    repaired, notes = repair_arguments(MATRIX, given)
    assert repaired["matrices"] == [{"name": "A", "rows": 2, "cols": 2}, {"name": "B", "rows": 2}]
    assert notes and check_arguments("matrix_calculus", repaired)
    both = {"latex": "A", "matrices": [{"symbol": "A", "name": "Z", "rows": 2}]}
    assert repair_arguments(MATRIX, both)[0] is both  # never overwrites a name the model gave


def test_a_condition_without_its_unknown_takes_the_only_unknown() -> None:
    given = {
        "equations": ["x'' + x = 0"],
        "independent": "t",
        "unknowns": ["x"],
        "conditions": [{"at": "0", "order": 0, "value": 1}, {"at": "0", "order": 1, "value": 0}],
        "t_end": 10,
    }
    repaired, notes = repair_arguments(NUMERIC, given)
    assert [c["unknown"] for c in repaired["conditions"]] == ["x", "x"]
    assert notes


def test_a_condition_is_never_assigned_to_one_of_several_unknowns() -> None:
    given = {
        "equations": ["x' = y", "y' = -x"],
        "independent": "t",
        "unknowns": ["x", "y"],
        "conditions": [{"at": "0", "order": 0, "value": 1}],
    }
    assert repair_arguments(NUMERIC, given)[0] is given  # which unknown? Not decided here


def test_the_default_assumption_is_dropped_and_the_others_kept() -> None:
    given = {"latex": r"\sqrt{x^2}", "assumptions": {"x": ["complex"], "y": ["complex", "real"]}}
    repaired, notes = repair_arguments(SIMPLIFY, given)
    assert repaired["assumptions"] == {"y": ["real"]}
    assert notes and check_arguments("simplify", repaired)
    only = repair_arguments(SIMPLIFY, {"latex": "x", "assumptions": {"x": ["complex"]}})[0]
    assert only["assumptions"] == {}
    untouched = {"latex": "x", "assumptions": {"x": ["positive"]}}
    assert repair_arguments(SIMPLIFY, untouched)[0] is untouched


# -- shapes seen in the 2026-11-18 core-suite traces ---------------------------------------------------------
TELESCOPING = TOOLS["creative_telescoping"].args_model
VECTOR = TOOLS["vector_calculus"].args_model
DIFFERENTIATE = TOOLS["differentiate"].args_model


def test_an_integrand_sent_to_a_tool_whose_formula_is_latex_is_moved() -> None:
    given = {
        "integrand": r"\exp(-a x^2)",
        "variable": "x",
        "parameter": "a",
        "lower": "-oo",
        "upper": "oo",
        "assumptions": {"a": ["positive"]},
    }
    repaired, notes = repair_arguments(TELESCOPING, given)
    assert repaired["latex"] == r"\exp(-a x^2)" and "integrand" not in repaired
    assert notes and check_arguments("creative_telescoping", repaired)


def test_the_integrand_alias_never_overrides_a_formula_or_a_real_field() -> None:
    both = {"latex": "x", "integrand": "y", "variable": "x", "parameter": "a"}
    assert repair_arguments(TELESCOPING, both)[0] is both  # the formula was given
    variation = TOOLS["variation"].args_model  # `integrand` is its own field there
    given = {"integrand": r"\dot{x}^2", "functions": ["x"]}
    assert repair_arguments(variation, given)[0] is given


def test_numbers_in_a_list_of_objects_become_text_where_the_tool_takes_text() -> None:
    given = {
        "action": "evaluate",
        "expression": r"\nabla\cdot F",
        "coordinates": "spherical",
        "fields": [{"name": "F", "kind": "vector", "components": [1, 0, 0.5]}],
    }
    repaired, notes = repair_arguments(VECTOR, given)
    assert repaired["fields"][0]["components"] == ["1", "0", "0.5"]
    assert notes and notes[-1].target == "text"
    assert given["fields"][0]["components"] == [1, 0, 0.5]  # the model's own call is not mutated


def test_numbers_stay_numbers_where_the_tool_takes_a_number() -> None:
    given = {"latex": "x^3", "variables": [{"name": "x", "order": 2}]}
    assert repair_arguments(DIFFERENTIATE, given)[0] is given
    flags = {"latex": "x", "variables": [{"name": "x", "order": True}]}
    assert (
        repair_arguments(DIFFERENTIATE, flags)[0] is flags
    )  # a boolean is not a number to convert
