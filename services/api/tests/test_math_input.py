"""Transport damage in a tool call's arguments is removed before the call is keyed, validated and run.

Regression: a retry of the same Lagrangian arrived with markdown-escaped operators (`\\*`), failed with a parse
error, and a repeat of the first call followed. The cleaned retry is now the same call as the first (refused as
a repeat, not executed twice), the cleaning is shown to the model, and the raw arguments stay in the trace.
"""

from typing import Any

import pytest
from minerva_api.agent.loop import AgentLimits
from minerva_api.agent.mathinput import MATH_FIELDS, prepare_arguments, transport_line
from minerva_api.agent.refs import EXPRESSION_ARGUMENTS
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import _EXPRESSION_FIELDS, TOOLS, ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

from .test_agent_reliability import cards, final, play

WEDGE = (
    "1/2*(M+m)*(Xdot^2 + Ydot^2) + 1/2*m*sdot^2 + m*sdot*(Xdot*cos(alpha) + Ydot*sin(alpha))"
    " - (M+m)*g*Y - m*g*s*sin(alpha)"
)
ESCAPED = WEDGE.replace("*", "\\*")
ARGS = {"lagrangian": WEDGE, "coordinates": ["X", "Y", "s"]}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def test_one_list_says_which_arguments_are_mathematics() -> None:
    assert EXPRESSION_ARGUMENTS is MATH_FIELDS and _EXPRESSION_FIELDS is MATH_FIELDS


def test_every_math_field_of_every_tool_is_registered_or_not_a_string() -> None:
    """A new tool's expression field must be on the list, or its input is never cleaned or referenced."""
    candidates = {"latex", "expression", "equation", "lagrangian", "hamiltonian", "integrand"}
    for name, tool in TOOLS.items():
        for field, info in tool.args_model.model_fields.items():
            if field in candidates:
                assert field in MATH_FIELDS, (name, field)
            elif field.endswith(("latex", "expression")) and info.annotation is str:
                assert field in MATH_FIELDS or field in _OTHER_EXPRESSIONS, (name, field)


# Expression-like fields of a tool that carry something other than the main formula; they are read by their
# own tool and are deliberately not cleaned centrally (listed so a new one is a decision, not an accident).
_OTHER_EXPRESSIONS = {"result_latex", "expected_latex", "relation_latex", "claim_latex"}


def test_arguments_without_damage_are_returned_as_they_came() -> None:
    arguments = {"lagrangian": "1/2*m*x_dot^2", "coordinates": ["x"]}
    cleaned, notes = prepare_arguments("analyze_lagrangian", arguments)
    assert cleaned is arguments and notes == ()


def test_damage_is_removed_in_every_math_field_and_reported() -> None:
    cleaned, notes = prepare_arguments(
        "analyze_lagrangian", {"lagrangian": "$" + ESCAPED + "$", "coordinates": ["X"]}
    )
    assert cleaned["lagrangian"] == WEDGE and cleaned["coordinates"] == ["X"]
    line = transport_line(notes)
    assert line is not None and line.startswith("input cleaned: ")


def test_a_leading_assignment_is_removed_and_said() -> None:
    cleaned, notes = prepare_arguments(
        "find_conserved_quantities", {"hamiltonian": "H = p^2/2 + q^2/2", "coordinates": ["q"]}
    )
    assert cleaned["hamiltonian"] == "p^2/2 + q^2/2"
    assert [n.source for n in notes] == ["H ="]


def test_lists_of_equations_are_cleaned_item_by_item() -> None:
    cleaned, _ = prepare_arguments("ode_solve_symbolic", {"equations": ["$y' = -2y$", "y(0) = 1"]})
    assert cleaned["equations"] == ["y' = -2y", "y(0) = 1"]


def test_greek_letters_follow_the_syntax_of_the_tool() -> None:
    theta = chr(0x3B8)
    cleaned, _ = prepare_arguments("simplify", {"latex": f"\\sin({theta})^2"})
    assert cleaned["latex"] == "\\sin(\\theta )^2"
    cleaned, _ = prepare_arguments(
        "analyze_lagrangian", {"lagrangian": f"1/2*m*{theta}̇^2", "coordinates": ["theta"]}
    )
    assert cleaned["lagrangian"] == "1/2*m*theta_dot^2"
    cleaned, _ = prepare_arguments("tensor_canonicalize", {"expression": f"S({theta},b)"})
    assert cleaned["expression"] == "S(theta,b)"


def test_escaped_input_is_analysed_and_the_model_is_told(ctx: ToolContext) -> None:
    cleaned, notes = prepare_arguments("analyze_lagrangian", {**ARGS, "lagrangian": ESCAPED})
    ctx.input_notes = notes
    try:
        out = execute("analyze_lagrangian", cleaned, ctx)
    finally:
        ctx.input_notes = ()
    assert out.status == "ok", out.for_model[:300]
    assert "input cleaned: " in out.for_model and "`\\*` as `*`" in out.for_model
    assert out.card.data["input_cleaned"][0]["kind"] == "transport"


def test_a_retry_that_differs_only_by_escaping_is_the_same_call(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("analyze_lagrangian", ARGS),
            call("analyze_lagrangian", {**ARGS, "lagrangian": ESCAPED}),
            say("The block and the wedge decouple."),
        ]
    )
    message = final(play(llm, ctx))
    first, second = message["trace"]
    assert first["status"] == "ok" and not first["repeated"]
    assert second["repeated"] is True and second["status"] == "skipped"
    assert second["arguments"]["lagrangian"] == WEDGE  # the cleaned call is what was keyed
    assert len(cards(message)) == 1


def test_the_raw_arguments_stay_in_the_trace(ctx: ToolContext) -> None:
    llm = ReplayLLM([call("analyze_lagrangian", {**ARGS, "lagrangian": ESCAPED}), say("Done.")])
    (step,) = final(play(llm, ctx))["trace"]
    assert step["status"] == "ok"
    assert step["arguments"]["lagrangian"] == WEDGE
    assert step["model_arguments"]["lagrangian"] == ESCAPED
    assert "input cleaned" in step["for_model"]


def test_cleaning_does_not_hide_a_genuine_error(ctx: ToolContext) -> None:
    arguments: dict[str, Any] = {"lagrangian": "$1/2*m*x_dot^2 +$", "coordinates": ["x"]}
    llm = ReplayLLM([call("analyze_lagrangian", arguments), say("It could not be read.")])
    (step,) = final(play(llm, ctx, limits=AgentLimits(retry_delay_seconds=0)))["trace"]
    assert step["status"] == "parse_error"


def test_a_braced_exponent_in_a_text_formula_is_cleaned_and_said() -> None:
    cleaned, notes = prepare_arguments(
        "analyze_lagrangian",
        {"lagrangian": "1/2*m*x_dot^{2} - k*x^{-1}", "coordinates": ["x"]},
    )
    assert cleaned["lagrangian"] == "1/2*m*x_dot^(2) - k*x^(-1)"
    assert notes and "^(...)" in (transport_line(notes) or "")


def test_braces_of_a_latex_formula_are_never_touched() -> None:
    arguments = {"lagrangian": r"\frac{1}{2} m \dot{x}^{2} - k x^{-1}", "coordinates": ["x"]}
    cleaned, notes = prepare_arguments("analyze_lagrangian", arguments)
    assert cleaned is arguments and notes == ()
    explicit = {"lagrangian": "m*x^{12}", "coordinates": ["x"], "format": "latex"}
    assert prepare_arguments("analyze_lagrangian", explicit)[0] is explicit
