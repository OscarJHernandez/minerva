"""What the loop does about a failed tool call before the model hears of it: one deterministic correction,
no equivalent call twice, no tool that keeps failing, and a new diagnostic is progress, the same one is not.

Regression: the recorded turn made four model calls because each failure went back to the model, and a
call that differed from an earlier one only by spelling was run (or refused) as if it were new.
"""

import pytest
from minerva_api.agent.loop import AgentLimits
from minerva_api.agent.recovery import (
    FailureClass,
    canonical_key,
    classify,
    rerun_arguments,
)
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext, ToolOutput, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

from .test_agent_reliability import cards, final, play

BAD = r"\foo{%s}"
SIMPLIFY = {"latex": r"\frac{x^2-1}{x-1}"}
WEDGE = "1/2*m*(Xdot^2 + Ydot^2) - m*g*Y"
ARGS = {"lagrangian": WEDGE, "coordinates": ["X", "Y"]}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


# -- the call key ------------------------------------------------------------------------------------


def test_calls_that_differ_only_by_defaults_or_number_types_have_one_key() -> None:
    a = canonical_key("series", {"latex": r"\sin x", "point": 0, "order": 5, "variable": "x"})
    b = canonical_key("series", {"variable": "x", "order": 5, "latex": r"\sin x", "point": "0"})
    assert a == b


def test_format_auto_is_keyed_by_the_syntax_it_resolves_to() -> None:
    auto = canonical_key("analyze_lagrangian", ARGS)
    text = canonical_key("analyze_lagrangian", {**ARGS, "format": "text"})
    latex = canonical_key("analyze_lagrangian", {**ARGS, "format": "latex"})
    assert auto == text != latex


def test_different_mathematics_has_a_different_key() -> None:
    assert canonical_key("simplify", {"latex": "x+1"}) != canonical_key(
        "simplify", {"latex": "x+2"}
    )


def test_arguments_that_do_not_validate_are_keyed_as_they_are() -> None:
    assert canonical_key("simplify", {"nonsense": 1}) == canonical_key("simplify", {"nonsense": 1})


# -- classification ----------------------------------------------------------------------------------


def test_failures_are_classified_by_kind(ctx: ToolContext) -> None:
    bad = execute("simplify", {"latex": r"\foo{a}"}, ctx)
    assert classify(bad) is FailureClass.PARSE
    assert classify(execute("simplify", {}, ctx)) is FailureClass.INVALID_ARGUMENTS
    ok = execute("simplify", {"latex": "x+x"}, ctx)
    assert classify(ok) is None
    ambiguous = execute(
        "analyze_lagrangian", {"lagrangian": "1/2*xddot^2 - xdot", "coordinates": ["x", "xd"]}, ctx
    )
    assert ambiguous.reason_code in {"ambiguous_derivative", None}


def test_an_answer_is_not_a_failure(ctx: ToolContext) -> None:
    unchanged = execute("simplify", {"latex": "x"}, ctx)
    assert unchanged.status == "unchanged" and classify(unchanged) is None


# -- the one deterministic correction ----------------------------------------------------------------


def _parse_failure(code: str) -> ToolOutput:
    out = ToolOutput.__new__(ToolOutput)
    out.status, out.reason_code, out.arguments_valid, out.error_category = (
        "parse_error",
        code,
        True,
        "parse_error",
    )
    return out


def test_an_explicit_wrong_format_is_switched_once() -> None:
    latex = {**ARGS, "format": "latex"}
    assert rerun_arguments("analyze_lagrangian", latex, _parse_failure("text_syntax_in_latex")) == (
        {**ARGS, "format": "text"},
        "format_switch",
    )
    text = {**ARGS, "format": "text"}
    assert rerun_arguments("analyze_lagrangian", text, _parse_failure("latex_in_text")) == (
        {**ARGS, "format": "latex"},
        "format_switch",
    )


# The metric the deployed model sent in the landing-page trial (2026-11-19), with an explicit `format: latex`: the
# first cell parsed (`^(-1)`) used to fail with an uncoded error, so nothing told the loop to read it as text.
TRIAL_METRIC = {
    "coordinates": ["t", "r", "theta", "phi"],
    "metric": [
        ["-(1-2*M/r)", "0", "0", "0"],
        ["0", "(1-2*M/r)^(-1)", "0", "0"],
        ["0", "0", "r^2", "0"],
        ["0", "0", "0", "r^2*sin(theta)^2"],
    ],
    "quantities": ["ricci"],
    "format": "latex",
}


def test_a_text_metric_sent_as_latex_is_read_as_text_once(ctx: ToolContext) -> None:
    llm = ReplayLLM([call("geometry", TRIAL_METRIC), say("The Ricci tensor vanishes.")])
    message = final(play(llm, ctx))
    (step,) = message["trace"]
    assert step["status"] == "ok" and step["policy"] == "format_switch"
    assert len(message["debug"]["llm_calls"]) == 2  # no retry round trip through the model
    (card,) = cards(message)
    assert card["data"]["request"]["format"] == "text"
    ricci = card["data"]["result"]["quantities"][0]
    assert ricci["kind"] == "ricci" and ricci["all_zero"] is True


def test_a_latex_metric_with_a_real_error_is_not_switched(ctx: ToolContext) -> None:
    # `\frac{1}{` is broken LaTeX, not text syntax: refused once, never re-read as text
    broken = {
        **TRIAL_METRIC,
        "metric": [["\\frac{1}{", "0"], ["0", "1"]],
        "coordinates": ["x", "y"],
    }
    out = execute("geometry", broken, ctx)
    assert out.status == "parse_error"
    assert rerun_arguments("geometry", broken, out) is None


def test_no_switch_without_a_matching_code_or_a_format_field() -> None:
    latex = {**ARGS, "format": "latex"}
    assert rerun_arguments("analyze_lagrangian", latex, _parse_failure("syntax")) is None
    assert rerun_arguments("analyze_lagrangian", latex, _parse_failure("latex_in_text")) is None
    # `substitute` has relations in LaTeX too, so it takes no format: nothing to switch.
    assert (
        rerun_arguments("substitute", {"latex": "x"}, _parse_failure("text_syntax_in_latex"))
        is None
    )


def test_a_wrong_explicit_format_is_recovered_without_a_model_call(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [call("analyze_lagrangian", {**ARGS, "format": "latex"}), say("They fall freely.")]
    )
    message = final(play(llm, ctx))
    (step,) = message["trace"]
    assert step["status"] == "ok" and step["policy"] == "format_switch"
    assert "recovered:" in step["for_model"] and "format text" in step["for_model"]
    assert len(message["debug"]["llm_calls"]) == 2  # the call and the answer: no retry round trip
    assert len(cards(message)) == 1


# -- equivalent failed calls, budgets, progress ---------------------------------------------------------


def test_an_equivalent_call_that_failed_is_not_run_again(ctx: ToolContext) -> None:
    first = {"latex": r"\foo{a}"}
    second = {"latex": "$" + r"\foo{a}" + "$"}  # the same call once the transport is cleaned
    llm = ReplayLLM([call("simplify", first), call("simplify", second), say("It cannot be read.")])
    message = final(play(llm, ctx))
    one, two = message["trace"]
    assert one["status"] == "parse_error"
    assert two["repeated"] is True and two["policy"] == "refused_equivalent"
    assert "did not succeed" in two["for_model"] and one["for_model"] in two["for_model"]


def test_a_tool_that_keeps_failing_is_not_run_again_in_the_turn(ctx: ToolContext) -> None:
    replies = [call("simplify", {"latex": BAD % n}) for n in "abcd"]
    llm = ReplayLLM([*replies, say("Done.")])
    limits = AgentLimits(retry_delay_seconds=0, max_failures_per_tool=2)
    message = final(play(llm, ctx, limits=limits))
    statuses = [t["status"] for t in message["trace"]]
    assert statuses[:2] == ["parse_error", "parse_error"] and statuses[2] == "error"
    assert message["trace"][2]["policy"] == "budget_exhausted"
    assert message["trace"][2]["error_category"] == "limit_reached"


def test_a_result_starts_the_count_afresh(ctx: ToolContext) -> None:
    replies = [
        call("simplify", {"latex": BAD % "a"}),
        call("simplify", SIMPLIFY),
        call("simplify", {"latex": BAD % "b"}),
        call("simplify", {"latex": BAD % "c"}),
        say("Done."),
    ]
    limits = AgentLimits(retry_delay_seconds=0, max_failures_per_tool=2)
    message = final(play(ReplayLLM(replies), ctx, limits=limits))
    assert [t["status"] for t in message["trace"]] == [
        "parse_error",
        "ok",
        "parse_error",
        "parse_error",
    ]


def test_a_new_diagnostic_is_progress_and_the_same_one_is_not(ctx: ToolContext) -> None:
    """Three failures with three different codes do not end the turn; the unreadable-twice rule does."""
    replies = [
        call("analyze_lagrangian", {"lagrangian": "L = 1/2*m*x_dot^2 = 3", "coordinates": ["x"]}),
        call(
            "analyze_lagrangian", {"lagrangian": "1/2*m*(x_dot^2 + y_dot^2)", "coordinates": ["x"]}
        ),
        call(
            "analyze_lagrangian", {"lagrangian": "1/2*m*x_dot^2 - 1/2*k*x^2", "coordinates": ["x"]}
        ),
        say("Done."),
    ]
    message = final(play(ReplayLLM(replies), ctx))
    assert [t["status"] for t in message["trace"]] == ["parse_error", "parse_error", "ok"]
    assert message["stop_reason"] == "completed"


# -- the formula tools read text syntax when LaTeX refuses it ------------------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments", "fragment"),
    [
        ("simplify", {"latex": "sin(x)^2 + cos(x)^2"}, "1"),
        ("differentiate", {"latex": "sin(x)*exp(x)", "variables": [{"name": "x"}]}, "\\sin"),
        ("series", {"latex": "sin(x)", "variable": "x", "point": "0", "order": 5}, "x^{5}"),
        ("integrate", {"latex": "x*cos(x)", "variable": "x"}, "\\sin"),
    ],
)
def test_text_in_a_latex_field_is_read_as_text_without_a_model_call(
    ctx: ToolContext, tool: str, arguments: dict[str, object], fragment: str
) -> None:
    message = final(play(ReplayLLM([call(tool, arguments), say("Done.")]), ctx))
    (step,) = message["trace"]
    assert step["status"] == "ok", step["for_model"][:300]
    assert step["policy"] == "format_switch" and fragment in step["for_model"]
    assert len(message["debug"]["llm_calls"]) == 2


def test_ordinary_latex_is_never_switched(ctx: ToolContext) -> None:
    arguments = {"latex": r"\frac{x^2-1}{x-1}"}
    message = final(play(ReplayLLM([call("simplify", arguments), say("Done.")]), ctx))
    (step,) = message["trace"]
    assert step["status"] == "ok" and step["policy"] is None


def test_a_product_written_like_a_call_stays_latex(ctx: ToolContext) -> None:
    """`x(x+1)` is a product in LaTeX; no marker makes it text, so it is read as before."""
    message = final(
        play(ReplayLLM([call("simplify", {"latex": "x(x+1) - x^2"}), say("Done.")]), ctx)
    )
    (step,) = message["trace"]
    assert step["status"] == "ok" and step["policy"] is None and "$x$" in step["for_model"]
