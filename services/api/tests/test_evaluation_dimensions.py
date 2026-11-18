"""Scoring that separates what a run got right: accepted alternative tools, six dimensions, noise in comparisons.

Before this, a correct answer reached with a tool that was not the expected one failed as a tool-selection error
(or, if listed in `also_ok`, was never checked at all), and `--compare` gave no way to tell a change from noise.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from minerva_api.agent.evaluation.cases import (
    Alternative,
    ArgCheck,
    EvalCase,
    ExpectCard,
    Reference,
    cases_digest,
    load_cases,
)
from minerva_api.agent.evaluation.reference import check_alternatives
from minerva_api.agent.evaluation.report import (
    DIMENSIONS,
    _noise_lines,
    summarise,
)
from minerva_api.agent.evaluation.runner import run_case
from minerva_api.agent.evaluation.score import CaseResult, FailureClass, _dimensions, combine
from minerva_api.agent.loop import AgentLimits
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

LIMITS = AgentLimits(max_steps=6, max_seconds=60)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def solve_case(**changes: Any) -> EvalCase:
    """`x^2 = 4` can be answered with isolate_variable or with solve_polynomial: both are right."""
    base: dict[str, Any] = {
        "id": "dim-solve",
        "category": "symbolic",
        "prompt": "Solve $x^2 = 4$ for $x$.",
        "expected_tool": "isolate_variable",
        "args": ArgCheck(present=["equation", "variable"]),
        "expect_card": ExpectCard(status="ok"),
        "reference": Reference(
            tool="isolate_variable", arguments={"equation": "x^2 = 4", "variable": "x"}
        ),
        "accept": [
            Alternative(
                tool="solve_polynomial",
                args=ArgCheck(equals={"equation": "x^2-4=0"}),
                expect_card=ExpectCard(status="ok"),
                reference=Reference(tool="solve_polynomial", arguments={"equation": "x^2-4=0"}),
            )
        ],
        "source": "hand calculation: x = 2 or x = -2",
    }
    return EvalCase(**{**base, **changes})


def play(case: EvalCase, replies: list[Any], ctx: ToolContext) -> CaseResult:
    return asyncio.run(run_case(ReplayLLM(replies), ctx, case, LIMITS))


# ----------------------------------------------------------------------------- alternative valid tools
def test_a_correct_answer_through_an_accepted_alternative_passes(ctx: ToolContext) -> None:
    r = play(
        solve_case(),
        [call("solve_polynomial", {"equation": "x^2-4=0"}), say("The solutions are 2 and -2.")],
        ctx,
    )
    assert r.passed and r.failure_class is None, r.notes
    assert r.tool_selected and r.tool_reached
    assert r.args_ok is True and r.result_ok is True
    assert (r.operation_ok, r.math_ok, r.evidence_ok, r.efficient) == (True, True, True, True)


def test_the_same_run_fails_when_the_alternative_is_not_accepted(ctx: ToolContext) -> None:
    r = play(
        solve_case(accept=[]),
        [call("solve_polynomial", {"equation": "x^2-4=0"}), say("The solutions are 2 and -2.")],
        ctx,
    )
    assert not r.passed and r.failure_class == FailureClass.TOOL_SELECTION
    assert r.operation_ok is False


def test_an_alternative_is_checked_not_just_listed(ctx: ToolContext) -> None:
    """`also_ok` never looked at the arguments of the other tool; an accepted alternative must meet its own checks."""
    r = play(
        solve_case(),
        [call("solve_polynomial", {"equation": "x^2-9=0"}), say("The solutions are 3 and -3.")],
        ctx,
    )
    assert not r.passed and r.failure_class == FailureClass.ARGUMENTS
    assert any("solve_polynomial" in n for n in r.notes)
    assert r.operation_ok is False


def test_the_expected_tool_still_passes_and_an_alternative_does_not_excuse_a_wrong_one(
    ctx: ToolContext,
) -> None:
    ok = play(
        solve_case(),
        [call("isolate_variable", {"equation": "x^2 = 4", "variable": "x"}), say("x = 2 or -2.")],
        ctx,
    )
    assert ok.passed
    wrong_tool = play(
        solve_case(), [call("search_handbook", {"query": "quadratic"}), say("See above.")], ctx
    )
    assert not wrong_tool.passed and wrong_tool.failure_class == FailureClass.TOOL_SELECTION


def test_a_failed_expected_call_followed_by_a_correct_alternative_passes(ctx: ToolContext) -> None:
    r = play(
        solve_case(),
        [
            call("isolate_variable", {"equation": "x^2 = 4", "variable": "y"}),
            call("solve_polynomial", {"equation": "x^2-4=0"}),
            say("The solutions are 2 and -2."),
        ],
        ctx,
    )
    # one run, two ways: the first was not what the user asked (wrong variable, invalid or an error), the second is
    assert r.args_ok is not False
    assert r.operation_ok is True


# ----------------------------------------------------------------------------- the dimensions
def test_a_correct_but_wasteful_run_is_right_on_every_dimension_except_efficiency(
    ctx: ToolContext,
) -> None:
    r = play(
        solve_case(),
        [
            call("isolate_variable", {"equation": "x^2 = 4", "variable": "x"}),
            call("isolate_variable", {"equation": "x^2 = 4", "variable": "x"}),
            say("x = 2 or x = -2."),
        ],
        ctx,
    )
    assert r.operation_ok and r.math_ok and r.evidence_ok and r.completed
    assert r.efficient is False  # a repeated call is wasted work


def test_a_wrong_tool_is_an_operation_failure_without_being_a_false_statement(
    ctx: ToolContext,
) -> None:
    r = play(solve_case(), [say("It is 2 or -2.")], ctx)
    assert r.operation_ok is False and r.math_ok is True and r.evidence_ok is True


def test_a_false_statement_shown_as_verified_fails_mathematics_and_evidence() -> None:
    flags = {
        "tool_selection": False,
        "arguments": False,
        "parse_error": False,
        "ignored_failure": False,
        "misreported_restriction": False,
        "result_reuse": False,
    }

    def result(**changes: Any) -> CaseResult:
        r = CaseResult(
            case_id="c",
            category="false_claim",
            passed=False,
            expected_tool="check_identity",
            tool_selected=True,
            tool_reached=True,
            **changes,
        )
        _dimensions(r, flags)
        return r

    bad = result(false_verified=1)
    assert bad.math_ok is False and bad.evidence_ok is False and bad.operation_ok is True
    contradicted = result(contradicted=1)
    assert contradicted.math_ok is False and contradicted.evidence_ok is True
    wrong_result = result(result_ok=False)
    assert wrong_result.math_ok is False
    ignored = {**flags, "ignored_failure": True}
    r = result()
    _dimensions(r, ignored)
    assert r.evidence_ok is False and r.math_ok is True


def test_multi_turn_consistency_needs_every_turn_to_pass() -> None:
    def turn(passed: bool) -> CaseResult:
        return CaseResult(
            case_id="c",
            category="followup",
            passed=passed,
            expected_tool="simplify",
            tool_selected=passed,
            tool_reached=passed,
            operation_ok=passed,
            math_ok=True,
            evidence_ok=True,
            efficient=True,
        )

    assert combine([turn(True), turn(True)]).consistent is True
    both = combine([turn(True), turn(False)])
    assert both.consistent is False and both.operation_ok is False and both.math_ok is True
    assert combine([turn(True)]).consistent is None  # a single turn has no consistency to speak of


def test_summaries_report_each_dimension_over_the_runs_scored_on_it() -> None:
    rows = [
        CaseResult(
            case_id=f"c{i}",
            category="symbolic",
            passed=i % 2 == 0,
            expected_tool="simplify",
            tool_selected=True,
            tool_reached=True,
            operation_ok=i % 2 == 0,
            math_ok=True,
            evidence_ok=True,
            efficient=i != 3,
        )
        for i in range(4)
    ]
    dims = summarise(rows)["dimensions"]
    assert dims["operation_ok"] == {"runs": 4, "ok": 2, "rate": 0.5}
    assert dims["math_ok"]["rate"] == 1.0 and dims["efficient"]["rate"] == 0.75
    assert "consistent" not in dims  # no multi-turn run was scored on it
    assert [k for k, _ in DIMENSIONS] == [
        "operation_ok",
        "math_ok",
        "evidence_ok",
        "efficient",
        "consistent",
    ]


# ----------------------------------------------------------------------------- the schema
def test_alternatives_are_validated() -> None:
    with pytest.raises(ValueError, match="different tool"):
        solve_case(accept=[Alternative(tool="isolate_variable")])
    with pytest.raises(ValueError, match="different tool"):
        solve_case(accept=[Alternative(tool="series"), Alternative(tool="series")])
    with pytest.raises(ValueError, match="own tool"):
        solve_case(
            accept=[Alternative(tool="series", reference=Reference(tool="integrate", arguments={}))]
        )
    with pytest.raises(ValueError, match="need an expected tool"):
        EvalCase(
            id="x",
            category="ambiguous",
            prompt="p",
            expected_tool="ask",
            reply_matches="which",
            accept=[Alternative(tool="series")],
            source="s",
        )


def test_the_digest_is_unchanged_without_alternatives_and_changes_with_them() -> None:
    assert cases_digest(load_cases()) == "b7d2202c3d87"
    assert cases_digest([solve_case(accept=[])]) != cases_digest([solve_case()])


def test_an_alternatives_reference_is_run_against_the_engine(ctx: ToolContext) -> None:
    assert check_alternatives(solve_case(), ctx) == []
    broken = solve_case(
        accept=[
            Alternative(
                tool="solve_polynomial",
                expect_card=ExpectCard(status="unsupported"),
                reference=Reference(tool="solve_polynomial", arguments={"equation": "x^2-4=0"}),
            )
        ]
    )
    problems = check_alternatives(broken, ctx)
    assert problems and "solve_polynomial" in problems[0]


# ----------------------------------------------------------------------------- noise in comparisons
def test_a_change_inside_the_noise_is_said_to_be_inside_it() -> None:
    a = {"pass_counts_per_repeat": [87, 90, 90], "pass_count_stdev": 1.41}
    b = {"pass_counts_per_repeat": [88, 91, 91], "pass_count_stdev": 1.41}
    text = "\n".join(_noise_lines(a, b))
    assert "within the noise" in text and "larger than the noise" not in text


def test_a_change_beyond_the_noise_is_said_to_be_beyond_it() -> None:
    a = {"pass_counts_per_repeat": [87, 90, 90], "pass_count_stdev": 1.41}
    b = {"pass_counts_per_repeat": [99, 100, 101], "pass_count_stdev": 0.82}
    assert "larger than the noise" in "\n".join(_noise_lines(a, b))


def test_with_one_repeat_the_noise_is_not_measurable_and_nothing_is_called_larger() -> None:
    a = {"pass_counts_per_repeat": [90], "pass_count_stdev": 0.0}
    b = {"pass_counts_per_repeat": [120], "pass_count_stdev": 0.0}
    text = "\n".join(_noise_lines(a, b))
    assert "not measurable" in text and "larger than the noise" not in text
