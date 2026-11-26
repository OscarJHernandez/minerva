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
    failure_signature,
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
    # A grid is re-read under `auto`: one syntax (text) reads every entry here, so nothing is split.
    assert card["data"]["request"]["format"] == "auto"
    assert card["data"]["result"]["cell_syntax"] is None
    ricci = card["data"]["result"]["quantities"][0]
    assert ricci["kind"] == "ricci" and ricci["all_zero"] is True


# The Schwarzschild chat of 2026-11-26: `-(1-2M/r)` is LaTeX only and `r^2*sin(theta)^2` text only, so neither
# syntax reads the whole grid. Explicit `latex` fails, the re-run under `auto` reads it entry by entry.
CHAT_METRIC = {
    **TRIAL_METRIC,
    "metric": [
        ["-(1-2M/r)", "0", "0", "0"],
        ["0", "(1-2M/r)^{-1}", "0", "0"],
        ["0", "0", "r^2", "0"],
        ["0", "0", "0", "r^2*sin(theta)^2"],
    ],
    "quantities": ["christoffel"],
}


def test_a_mixed_metric_sent_as_latex_is_read_entry_by_entry_once(ctx: ToolContext) -> None:
    llm = ReplayLLM([call("geometry", CHAT_METRIC), say("Here are the Christoffel symbols.")])
    message = final(play(llm, ctx))
    (step,) = message["trace"]
    assert step["status"] == "ok" and step["policy"] == "format_switch"
    assert "read entry by entry" in step["for_model"] and "g[3][3]" in step["for_model"]
    (card,) = cards(message)
    assert card["data"]["result"]["cell_syntax"][3][3] == "text"


def test_a_mixed_metric_under_auto_succeeds_on_the_first_call(ctx: ToolContext) -> None:
    out = execute("geometry", {**CHAT_METRIC, "format": "auto"}, ctx)
    assert out.status == "ok", out.for_model
    assert rerun_arguments("geometry", {**CHAT_METRIC, "format": "auto"}, out) is None


def test_a_failed_re_read_is_reported_to_the_model(ctx: ToolContext) -> None:
    # `r0^2` reads as 0 in LaTeX and as r0**2 in text: the grid cannot be split, so the re-run fails too, and
    # the model is told what it said instead of being left with "use format text".
    metric = [row[:] for row in CHAT_METRIC["metric"]]
    metric[2][2] = "r0^2"
    llm = ReplayLLM(
        [call("geometry", {**CHAT_METRIC, "metric": metric}), say("It could not be read.")]
    )
    message = final(play(llm, ctx))
    (step,) = message["trace"]
    assert step["status"] == "parse_error" and step["policy"] is None
    assert "re-read with format auto, it failed too" in step["for_model"], step["for_model"]


def test_geometry_auto_and_latex_are_different_calls() -> None:
    from minerva_api.agent.recovery import canonical_key

    auto = canonical_key("geometry", {**CHAT_METRIC, "format": "auto"})
    latex = canonical_key("geometry", {**CHAT_METRIC, "format": "latex"})
    assert auto != latex


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


# --- the failure budget counts lack of progress, not attempts -----------------------------------------------
# The Schwarzschild chat of 2026-11-26: `sin(` fixed, then told about `theta`, then `2M`: three failures, each about
# something new, and the fourth call (correct) was refused unrun. A failure whose (code, symbols) was not reported
# before in the turn does not use up the budget; the same one again does.

SPRING = (
    "1/2*m*v^2 - k*x^2"  # no velocity of an undeclared name: the only problem is the declared one
)


def _missing(name: str, variant: int = 0) -> dict[str, object]:
    # the same mistake (a declared coordinate the Lagrangian does not use), spelt so that each call is distinct
    return {"lagrangian": SPRING + " " * variant, "coordinates": [name], "format": "text"}


def test_a_failure_signature_is_the_code_and_the_symbols(ctx: ToolContext) -> None:
    out = execute("analyze_lagrangian", _missing("q"), ctx)
    assert failure_signature(out) == ("declared_name_absent", ("q",))
    assert failure_signature(execute("simplify", SIMPLIFY, ctx)) is None


def test_a_model_converging_on_its_input_is_not_cut_off(ctx: ToolContext) -> None:
    replies = [
        call("analyze_lagrangian", _missing("q")),
        call("analyze_lagrangian", _missing("w")),
        call("analyze_lagrangian", _missing("z")),
        call(
            "analyze_lagrangian",
            {"lagrangian": "1/2*m*x_dot^2 - k*x^2", "coordinates": ["x"], "format": "text"},
        ),
        say("Done."),
    ]
    limits = AgentLimits(retry_delay_seconds=0, max_failures_per_tool=2)
    message = final(play(ReplayLLM(replies), ctx, limits=limits))
    statuses = [t["status"] for t in message["trace"]]
    assert statuses == ["parse_error", "parse_error", "parse_error", "ok"]
    assert all(t["policy"] != "budget_exhausted" for t in message["trace"])


def test_a_model_alternating_two_mistakes_still_runs_out(ctx: ToolContext) -> None:
    replies = [call("analyze_lagrangian", _missing("qw"[i % 2], i)) for i in range(8)]
    limits = AgentLimits(retry_delay_seconds=0, max_failures_per_tool=2)
    message = final(play(ReplayLLM([*replies, say("Done.")]), ctx, limits=limits))
    trace = message["trace"]
    # two new mistakes are free, then each repeat counts: refused at the fifth call, as with two strikes before
    assert [t["status"] for t in trace[:4]] == ["parse_error"] * 4
    assert trace[4]["policy"] == "budget_exhausted"
    assert len(trace) <= 6


def test_a_new_mistake_every_call_is_still_bounded_by_the_turn_limits(ctx: ToolContext) -> None:
    replies = [call("analyze_lagrangian", _missing(f"q{i}")) for i in range(20)]
    message = final(play(ReplayLLM([*replies, say("Done.")]), ctx))
    assert len(message["trace"]) <= 12
    assert message["stop_reason"] in ("step_limit", "tool_call_limit", "repeated_calls")
