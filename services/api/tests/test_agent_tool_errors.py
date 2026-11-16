"""Failures the benchmark showed models repeating: what the tool says back, and what counts as a stuck loop."""

import asyncio
import json
import re
from typing import Any

import pytest
from minerva_api.agent.loop import ANSWER_STATUSES, AgentLimits, run_agent
from minerva_api.agent.prompts import SYSTEM_PROMPT
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

FAST = AgentLimits(retry_delay_seconds=0)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def play(llm: ReplayLLM, ctx: ToolContext, **kw: Any) -> dict[str, Any]:
    async def go() -> list[dict[str, Any]]:
        return [
            e async for e in run_agent(llm, [{"role": "user", "content": "q"}], ctx, FAST, **kw)
        ]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


def test_a_number_for_a_text_argument_is_accepted(ctx: ToolContext) -> None:
    out = execute("series", {"latex": r"\sin x", "variable": "x", "point": 0, "order": 3}, ctx)
    assert out.status == "ok" and out.arguments_valid
    # a boolean is not a number in disguise, and a non-numeric type is still refused
    assert (
        execute("series", {"latex": r"\sin x", "variable": "x", "point": True}, ctx).status
        == "error"
    )


def test_wrong_shaped_arguments_get_every_error_and_the_right_shape(ctx: ToolContext) -> None:
    out = execute("simplify", {"latex": r"x+x", "assumptions": ["x", "positive"]}, ctx)
    assert out.status == "error" and not out.arguments_valid
    assert 'e.g. {"x": ["positive"]}' in out.for_model
    out = execute(
        "matrix_calculus", {"latex": "A", "matrices": ["A", "B"], "action": "evaluate"}, ctx
    )
    assert '[{"name": "A", "rows": 2}]' in out.for_model
    # several problems are reported together, not one per call
    out = execute("series", {"latex": ["x"], "variable": {"a": 1}, "order": "many"}, ctx)
    assert out.for_model.count(":") >= 3


SYSTEM = {
    "equations": ["dy1/dt = y2", "dy2/dt = -y1"],
    "format": "text",
    "independent": "t",
    "unknowns": ["y1", "y2"],
    "kind": "ivp",
    "t_end": 1,
}


def test_ode_conditions_may_give_the_point_and_value_as_numbers(ctx: ToolContext) -> None:
    conditions = [
        {"unknown": "y1", "at": 0, "value": 1},
        {"unknown": "y2", "at": 0, "value": 0},
    ]
    out = execute("ode_solve_numerical", {**SYSTEM, "conditions": conditions}, ctx)
    assert out.status == "ok" and out.arguments_valid
    # a boolean or a non-finite number is still not a number in disguise
    bad = [{"unknown": "y1", "at": True, "value": 1}, conditions[1]]
    assert execute("ode_solve_numerical", {**SYSTEM, "conditions": bad}, ctx).status == "error"


def test_a_wrong_shaped_condition_is_one_clear_problem_not_sixteen(ctx: ToolContext) -> None:
    conditions = [{"type": f"y{i}(0)=1", "value": 1} for i in range(1, 5)]
    out = execute("ode_solve_numerical", {**SYSTEM, "conditions": conditions}, ctx)
    assert out.status == "error" and not out.arguments_valid
    assert "and 13 more" not in out.for_model and "more)" not in out.for_model
    assert "conditions" in out.for_model and "unknown" in out.for_model
    assert '"unknown": "y", "order": 0, "at": "0", "value": "1"' in out.for_model
    assert out.for_model.count("Field required") <= 3


def test_a_bare_e_power_gets_a_hint_but_is_never_reinterpreted(ctx: ToolContext) -> None:
    out = execute("integrate", {"latex": r"e^{-x^2}", "variable": "x"}, ctx)
    assert out.status == "unsupported" and "ordinary symbol" in out.for_model
    assert "\\exp" in out.for_model
    out = execute("integrate", {"latex": r"\exp(-x^2)", "variable": "x"}, ctx)
    assert "ordinary symbol" not in out.for_model


def test_reserved_names_and_tensor_syntax_get_a_fix(ctx: ToolContext) -> None:
    tensor = execute("tensor_canonicalize", {"expression": "S_{ab}A^{ab}"}, ctx)
    if tensor.status == "parse_error":
        assert "S(a,b)*A(-a,-b)" in tensor.for_model


def test_an_unchanged_result_says_it_is_the_answer(ctx: ToolContext) -> None:
    out = execute("simplify", {"latex": "x"}, ctx)
    assert out.status == "unchanged" and "this is the answer" in out.for_model


def test_definite_answers_are_not_a_stuck_loop(ctx: ToolContext) -> None:
    assert {"unchanged", "unsupported", "conditions_not_satisfied"} == ANSWER_STATUSES
    # Four different inputs that all come back `unchanged`: the model may explain what it found.
    inputs = ["x", "y", "z", "w"]
    llm = ReplayLLM(
        [*[call("simplify", {"latex": s}) for s in inputs], say("Nothing simpler was found.")]
    )
    msg = play(llm, ctx)
    assert msg["stop_reason"] == "completed"
    assert [t["status"] for t in msg["trace"]] == ["unchanged"] * 4


def test_errors_still_end_the_turn(ctx: ToolContext) -> None:
    bad = [call("simplify", {"latex": rf"\foo{{{n}}}"}) for n in "abc"]
    msg = play(ReplayLLM([*bad, say("Could not read these.")]), ctx)
    assert msg["stop_reason"] == "repeated_calls"


def test_a_dropped_assumption_is_told_to_the_model(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("simplify", {"latex": r"\sqrt{x^2}", "assumptions": {"x": ["positive"]}}),
            call("simplify", {"latex": r"\sqrt{x^2} + 1"}),
            say("Done."),
        ]
    )
    play(llm, ctx)
    tool_messages = [m["content"] for m in llm.seen[-1] if m.get("role") == "tool"]
    assert "x was declared earlier" in tool_messages[-1]
    assert "x was declared earlier" not in tool_messages[0]


def test_the_prompt_numbers_its_rules_once_and_names_the_tools() -> None:
    import re

    numbers = [int(n) for n in re.findall(r"^(\d+)\. ", SYSTEM_PROMPT, re.MULTILINE)]
    assert numbers == list(range(1, len(numbers) + 1))
    assert "A stated size is enough" in SYSTEM_PROMPT
    for tool in ("simplify", "isolate_variable", "tensor_canonicalize", "check_units"):
        assert f"`{tool}`" in SYSTEM_PROMPT


def test_a_failed_read_never_prints_none_as_the_input() -> None:
    from minerva_api.agent.tools import ToolContext, execute
    from minerva_engine import Engine, Limits
    from minerva_knowledge import KnowledgeBase

    ctx = ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )
    for tool, args in (
        ("differentiate", {"latex": r"x'", "variables": [{"name": "x"}]}),
        ("simplify", {"latex": r"x'"}),
        ("integrate", {"latex": r"x'", "variable": "x"}),
        ("series", {"latex": r"x'", "variable": "x"}),
    ):
        out = execute(tool, args, ctx)
        assert out.status == "parse_error", tool
        assert "$None$" not in out.for_model, tool


def test_differentiating_an_integral_over_an_unknown_curve_says_what_is_not_supported(
    ctx: ToolContext,
) -> None:
    latex = r"T(y_0) = \int_0^{y_0} \frac{\sqrt{1+(x')^2}}{\sqrt{2 g (y_0 - y)}} dy"
    out = execute("differentiate", {"latex": latex, "variables": [{"name": "y_0"}]}, ctx)
    assert out.status == "parse_error" and out.known_latex == []
    assert "prime" in out.for_model and "variation" in out.for_model
    assert "evaluate it with `integrate`" in out.for_model
    assert "do not carry out the differentiation yourself" in out.for_model
    # no hint for an input that has no integral
    plain = execute(
        "differentiate", {"latex": r"\frac{dx}{dy} + 1", "variables": [{"name": "y"}]}, ctx
    )
    assert "evaluate it with `integrate`" not in plain.for_model


# --- argument-shape hints must teach the schema's own names -----------------------------------------
# A hint once said `{"symbol": "F", "unit": "N"}` while the schema requires `name`: the model obeyed
# it three times in a row and the turn ended on a repeated failing call (dimension check, eq. 7).
def test_the_quantity_hint_names_the_key_the_schema_requires(ctx: ToolContext) -> None:
    wrong = {"symbol": "k", "unit": "N"}
    for tool, extra in (
        ("check_units", {"equation": "F = m a"}),
        ("dimensional_analysis", {"target": "F"}),
    ):
        out = execute(tool, {**extra, "quantities": [wrong]}, ctx)
        assert out.status == "error" and not out.arguments_valid
        assert '"symbol"' not in out.for_model
        assert '[{"name": "F", "unit": "N"}]' in out.for_model
        assert "name (text, required)" in out.for_model
    # and a call written the way the message says is read
    ok = execute(
        "check_units",
        {"equation": "F = m a", "quantities": [{"name": "F", "unit": "N"}]},
        ctx,
    )
    assert ok.arguments_valid


def test_the_same_field_name_gets_the_hint_of_its_own_tool(ctx: ToolContext) -> None:
    # `quantities` of `geometry` is a list of names, not of objects, and `variables` of `differentiate`
    # a list of objects, not of names: neither may be told the other's shape.
    out = execute(
        "geometry",
        {"coordinates": ["r"], "metric": [["1"]], "quantities": [{"name": "x"}]},
        ctx,
    )
    assert out.status == "error"
    assert '"name": "F"' not in out.for_model and "'determinant'" in out.for_model
    out = execute("differentiate", {"latex": "x^2", "variables": [5]}, ctx)
    assert out.status == "error"
    assert '["x", "y"]' not in out.for_model and "order (an integer)" in out.for_model


def test_every_hint_example_validates_against_the_field_it_is_shown_for() -> None:
    from minerva_api.agent.tools import _SHAPE_EXAMPLES, TOOLS, _shape_hint
    from pydantic import TypeAdapter

    shown = 0
    for tool in TOOLS.values():
        for field, info in tool.args_model.model_fields.items():
            hint = _shape_hint(tool.args_model, field)
            if hint is None:
                continue
            shown += 1
            example = _SHAPE_EXAMPLES.get(field)
            if example is not None and f"e.g. {json.dumps(example)}" in hint:
                TypeAdapter(info.annotation).validate_python(example)  # never raises
            # every required key of an object the hint describes is named in it
            schema = tool.args_model.model_json_schema()
            defs = schema.get("$defs", {})
            for ref in set(
                re.findall(r'"\$ref": "#/\$defs/(\w+)"', json.dumps(schema["properties"][field]))
            ):
                for key in defs[ref].get("required", [])[:8]:
                    assert f"{key} (" in hint, (tool.name, field, key)
    assert shown > 20


def test_a_derivative_symbol_without_a_unit_is_refused_by_name(ctx: ToolContext) -> None:
    # `\dot{u}` and `\ddot{u}` are read as the symbols `u_dot` and `u_ddot`; the unit check must refuse
    # them with that name (so the model can declare them), never guess a dimension for a derivative.
    base = [
        {"name": "u", "unit": "1/m"},
        {"name": "k", "unit": "N m^2"},
        {"name": "m", "unit": "kg"},
    ]
    out = execute("check_units", {"equation": r"\ddot{u} = - k u^2 / m", "quantities": base}, ctx)
    assert out.status == "parse_error" and "u_ddot" in out.for_model
