"""The agent benchmark: case integrity, ground truth against the engine, scoring, reports, CLI.

Everything here is deterministic: a `ReplayLLM` plays the model, the tools and the claim checker are the
real ones. Real-model evaluations are run separately with scripts/eval_models.py.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from minerva_api.agent.evaluation import cli
from minerva_api.agent.evaluation.cases import (
    ASK,
    CATEGORIES,
    NO_TOOL,
    ArgCheck,
    EvalCase,
    ExpectCard,
    cases_digest,
    load_cases,
)
from minerva_api.agent.evaluation.reference import check_reference, same_expression
from minerva_api.agent.evaluation.report import (
    build_document,
    compare,
    percentile,
    render_markdown,
    summarise,
)
from minerva_api.agent.evaluation.runner import filter_cases, run_case, run_suite
from minerva_api.agent.evaluation.score import (
    CaseResult,
    FailureClass,
    Observation,
    score_case,
)
from minerva_api.agent.llm import Chunk, LLMError
from minerva_api.agent.loop import AgentLimits
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import TOOLS, ToolContext
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

CASES = load_cases()
BY_ID = {c.id: c for c in CASES}
LIMITS = AgentLimits(max_steps=6, max_seconds=60)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def run(case_id: str, replies: list[list[Chunk]], ctx: ToolContext, **kw: Any) -> CaseResult:
    llm = ReplayLLM(replies, repeat_last=kw.pop("repeat_last", False))
    return asyncio.run(run_case(llm, ctx, BY_ID[case_id], kw.pop("limits", LIMITS)))


# ---------------------------------------------------------------------------- the cases themselves
def test_case_file_is_well_formed() -> None:
    assert len(CASES) >= 70
    assert len({c.id for c in CASES}) == len(CASES)
    assert {c.category for c in CASES} == set(CATEGORIES)  # every requested area is covered
    for category in CATEGORIES:
        assert sum(c.category == category for c in CASES) >= 3, category


def test_every_tool_named_by_a_case_exists() -> None:
    names = set(TOOLS)
    for case in CASES:
        named = {case.expected_tool, *case.also_ok, *case.forbid, *case.sequence}
        named -= {NO_TOOL, ASK}
        assert named <= names, (case.id, named - names)


def test_hard_negatives_are_present() -> None:
    """The cases that catch unjustified assumptions and false verification claims must not be dropped."""
    wrong = [c for c in CASES if c.wrong_formulas]
    assert len(wrong) >= 10
    assert {
        "sqrt-no-assumptions",
        "false-log-exp",
        "false-matrix-exp",
        "false-divergent-integral",
    } <= {c.id for c in wrong}
    refusing = [c for c in CASES if c.expect_card and c.expect_card.status != "ok"]
    assert len(refusing) >= 8


def test_case_digest_changes_with_the_cases() -> None:
    other = [
        c.model_copy(update={"prompt": c.prompt + " "}) if i == 0 else c
        for i, c in enumerate(CASES)
    ]
    assert cases_digest(CASES) == cases_digest(load_cases())
    assert cases_digest(CASES) != cases_digest(other)


def test_invalid_case_definitions_are_rejected() -> None:
    base: dict[str, Any] = {
        "id": "x",
        "category": "symbolic",
        "prompt": "p",
        "expected_tool": "simplify",
        "source": "s",
    }
    EvalCase(**base)
    with pytest.raises(ValueError, match="where the expected outcome"):
        EvalCase(**{**base, "source": ""})
    with pytest.raises(ValueError, match="how to recognise"):
        EvalCase(**{**base, "expected_tool": ASK})
    with pytest.raises(ValueError, match="no tool is expected"):
        EvalCase(**{**base, "expected_tool": NO_TOOL, "args": ArgCheck(present=["latex"])})
    with pytest.raises(ValueError, match="expected tool"):
        EvalCase(**{**base, "reference": {"tool": "series", "arguments": {}}})
    with pytest.raises(ValueError):
        EvalCase(**{**base, "reply_matches": "("})


@pytest.mark.parametrize("case", [c for c in CASES if c.reference], ids=lambda c: c.id)
def test_ground_truth_agrees_with_the_engine(case: EvalCase, ctx: ToolContext) -> None:
    """Each expectation was written by hand; the real engine must reproduce it from the reference call."""
    assert case.expect_card is not None, (
        "a reference call needs an expectation to be checked against"
    )
    assert check_reference(case, ctx) == []


def test_a_wrong_expectation_is_caught(ctx: ToolContext) -> None:
    case = BY_ID["simplify-rational"].model_copy(
        update={"expect_card": ExpectCard(result_text="x + 2")}
    )
    assert check_reference(case, ctx)
    case = BY_ID["simplify-rational"].model_copy(
        update={"expect_card": ExpectCard(status="unsupported")}
    )
    assert check_reference(case, ctx)


def test_the_oracle_run_selects_the_tool_and_meets_the_expectation(ctx: ToolContext) -> None:
    """A model that makes the reference call picks the expected tool with the expected result, every time."""
    for case in (c for c in CASES if c.reference):
        assert case.reference is not None
        result = run(case.id, [call(case.reference.tool, case.reference.arguments)], ctx)
        assert result.tool_selected and result.tool_reached, case.id
        assert result.invalid_calls == 0, case.id
        if case.expect_card:
            assert result.result_ok, case.id
        if case.args:
            assert result.args_ok, case.id


def test_same_expression() -> None:
    assert same_expression("x + 1", "1 + x")
    assert same_expression("(x + 1)**2", "x**2 + 2*x + 1")
    assert same_expression("x**x*((log(x)+1)**2+1/x)", "(x**x + x**(x+1)*(log(x)+1)**2)/x")
    assert same_expression("sqrt(pi)*erf(x)/2", "sqrt(pi)*erf(x)/2")  # not parseable: equal text
    assert not same_expression("x + 1", "x + 2")
    assert not same_expression("x", "(")


def test_arg_checks() -> None:
    check = ArgCheck(
        equals={"variable": "x"},
        present=["lower"],
        absent=["assumptions"],
        includes={"variables": [{"name": "x", "order": 2}], "coordinates": ["r"]},
    )
    ok = {
        "variable": "x",
        "lower": "0",
        "variables": [{"name": "x", "order": 2, "extra": 1}],
        "coordinates": ["r", "theta"],
    }
    assert check.holds(ok)
    assert not check.holds({**ok, "variable": "y"})
    assert not check.holds({**ok, "lower": None})
    assert not check.holds({**ok, "assumptions": {"x": ["positive"]}})
    assert not check.holds({**ok, "variables": [{"name": "x", "order": 1}]})
    assert not check.holds({**ok, "variables": "x"})  # wrong shape: a failed check, not a crash
    assert ArgCheck(assumes={"x": "positive"}).holds({"assumptions": {"x": ["positive", "real"]}})
    assert not ArgCheck(assumes={"x": "positive"}).holds({"assumptions": {"x": "positive"}})
    assert not ArgCheck(assumes={"x": "positive"}).holds({"assumptions": None})


# ------------------------------------------------------------------------------------ scoring
def test_correct_tool_and_reply_pass(ctx: ToolContext) -> None:
    r = run(
        "simplify-rational",
        [call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}), say("The result is $x + 1$.")],
        ctx,
    )
    assert r.passed and r.failure_class is None
    assert r.tool_selected and r.args_ok is None and r.result_ok
    assert r.unchecked == 0 and r.false_verified == 0 and r.completed
    assert r.calls == 1 and r.llm_calls == 2 and r.tool_ms >= 0 and r.llm_ms >= 0


def test_wrong_tool_is_a_selection_failure(ctx: ToolContext) -> None:
    r = run(
        "simplify-rational",
        [call("search_handbook", {"query": "rational"}), say("See above.")],
        ctx,
    )
    assert not r.passed and r.failure_class == FailureClass.TOOL_SELECTION
    assert not r.tool_selected and r.first_tool == "search_handbook"


def test_no_tool_where_one_is_needed_is_a_selection_failure(ctx: ToolContext) -> None:
    r = run("simplify-rational", [say("It is $x+1$.")], ctx)
    assert r.failure_class == FailureClass.TOOL_SELECTION and r.calls == 0


def test_invalid_arguments_are_counted_and_classified(ctx: ToolContext) -> None:
    r = run("simplify-rational", [call("simplify", {}), say("Sorry, that failed.")], ctx)
    assert r.invalid_calls == 1 and r.failure_class == FailureClass.ARGUMENTS and not r.passed


def test_an_unknown_tool_is_an_invalid_call(ctx: ToolContext) -> None:
    r = run("simplify-rational", [call("simplfy", {"latex": "x"}), say("Done.")], ctx)
    assert r.invalid_calls == 1 and r.failure_class == FailureClass.TOOL_SELECTION


def test_a_retry_after_invalid_arguments_can_still_pass(ctx: ToolContext) -> None:
    r = run(
        "simplify-rational",
        [
            call("simplify", {}),
            call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}),
            say("It is $x + 1$."),
        ],
        ctx,
    )
    assert r.passed and r.invalid_calls == 1 and r.unnecessary_calls == 1


def test_dropped_assumptions_are_their_own_class(ctx: ToolContext) -> None:
    r = run(
        "assumptions-carried",
        [call("simplify", {"latex": r"\sqrt{k/m}\sqrt{m/k}"}), say("It is 1.")],
        ctx,
    )
    assert r.args_ok is False and r.failure_class == FailureClass.MISSING_ASSUMPTIONS


def test_invented_assumptions_are_caught(ctx: ToolContext) -> None:
    r = run(
        "sqrt-no-assumptions",
        [
            call("simplify", {"latex": r"\sqrt{x^2}", "assumptions": {"x": ["positive"]}}),
            say("It is $x$."),
        ],
        ctx,
    )
    assert r.args_ok is False and r.failure_class == FailureClass.MISSING_ASSUMPTIONS


def test_unchanged_with_an_honest_reply_passes(ctx: ToolContext) -> None:
    r = run(
        "sqrt-no-assumptions",
        [
            call("simplify", {"latex": r"\sqrt{x^2}"}),
            say("It is unchanged: it depends on assumptions about x."),
        ],
        ctx,
    )
    assert r.passed, r.notes


def test_a_wrong_expression_passed_to_the_tool_is_not_the_expected_result(ctx: ToolContext) -> None:
    wrong = call("differentiate", {"latex": "x^2", "variables": [{"name": "x"}]})
    r = run("differentiate-first", [wrong, say("It is $2x$.")], ctx)
    assert r.result_ok is False and r.failure_class == FailureClass.ARGUMENTS


def test_an_unreadable_expression_is_a_parsing_failure(ctx: ToolContext) -> None:
    bad = call("differentiate", {"latex": r"\foo{x}", "variables": [{"name": "x"}]})
    r = run("differentiate-first", [bad, say("I could not read that.")], ctx)
    assert r.failure_class == FailureClass.PARSING


def test_no_tools_expected_and_one_called(ctx: ToolContext) -> None:
    ok = run("chitchat", [say("You are welcome.")], ctx)
    assert ok.passed and ok.tool_selected
    bad = run("chitchat", [call("search_handbook", {"query": "thanks"}), say("Hi.")], ctx)
    assert bad.failure_class == FailureClass.TOOL_SELECTION


def test_a_question_is_the_right_answer_when_numbers_are_missing(ctx: ToolContext) -> None:
    ok = run(
        "motion-values-missing",
        [say("Which values do you want for m and k, and the initial position?")],
        ctx,
    )
    assert ok.passed and ok.tool_selected
    invented = {
        "lagrangian": r"\frac12 m\dot{x}^2 - \frac12 k x^2",
        "coordinates": ["x"],
        "parameters": {"m": 1, "k": 1},
        "initial_positions": [1],
        "initial_velocities": [0],
        "t_end": 10,
    }
    bad = run("motion-values-missing", [call("integrate_motion", invented), say("Done.")], ctx)
    assert bad.failure_class == FailureClass.TOOL_SELECTION and not bad.tool_selected


def test_a_multi_step_case_needs_every_step_in_order(ctx: ToolContext) -> None:
    integrate = call("integrate", {"latex": r"x \cos x", "variable": "x"})
    derive = call("differentiate", {"latex": r"x \sin x + \cos x", "variables": [{"name": "x"}]})
    ok = run("integrate-then-check", [integrate, derive, say("It checks out.")], ctx)
    assert ok.passed, ok.notes
    missing = run("integrate-then-check", [integrate, say("Done.")], ctx)
    assert missing.failure_class == FailureClass.TOOL_SELECTION
    swapped = run("integrate-then-check", [derive, integrate, say("Done.")], ctx)
    assert not swapped.passed and not swapped.tool_selected


def test_a_false_claim_confirmed_without_tools_is_not_passed(ctx: ToolContext) -> None:
    r = run("false-pythagoras", [say("Yes, confirmed: $\\sin^2 x + \\cos^2 x = 2$.")], ctx)
    assert (
        not r.passed and r.false_verified == 0
    )  # the claim checker contradicts it; it is not verified
    assert r.contradicted == 1


def test_the_false_claim_is_refuted_with_the_engine_result(ctx: ToolContext) -> None:
    r = run(
        "false-pythagoras",
        [
            call("simplify", {"latex": r"\sin^2 x + \cos^2 x"}),
            say("That is not true: the sum is $1$, not 2."),
        ],
        ctx,
    )
    assert r.passed, r.notes


def obs_with(segments: list[dict[str, Any]], text: str = "ok", **kw: Any) -> Observation:
    final = {
        "text": text,
        "blocks": [{"kind": "text", "segments": segments}],
        "trace": kw.pop("trace", []),
        "truncated": kw.pop("truncated", False),
        "debug": {"llm_calls": [{"ms": 5, "text_chars": 2, "tool_calls": 0}]},
    }
    return Observation(final=final, **kw)


def test_verified_marks_on_false_statements_are_the_worst_failure() -> None:
    case = BY_ID["false-pythagoras"]
    wrong = {"type": "display_math", "value": r"\sin^2 x + \cos^2 x = 2", "status": "checked"}
    r = score_case(case, obs_with([wrong], text="It is not true, it is 1."))
    assert r.false_verified == 1 and r.failure_class == FailureClass.VERIFICATION
    from_tool = {**wrong, "status": "from_tool"}
    assert score_case(case, obs_with([from_tool])).false_verified == 1
    for status in ("unchecked", "contradicted", "conditional"):
        shown = score_case(case, obs_with([{**wrong, "status": status}]))
        assert shown.false_verified == 0


def test_a_contradicted_correct_statement_is_a_checker_false_positive_not_a_model_error() -> None:
    case = EvalCase(
        id="fp",
        category="symbolic",
        prompt="p",
        expected_tool=NO_TOOL,
        right_formulas=[r"\sin^2 x + \cos^2 x = 1"],
        source="test",
    )
    right = {"type": "math", "value": r"\sin^2 x + \cos^2 x = 1", "status": "contradicted"}
    r = score_case(case, obs_with([right]))
    assert r.checker_false_positive == 1 and r.contradicted == 0
    assert r.failure_class == FailureClass.CLAIM_CHECKER
    other = {"type": "math", "value": "1 = 2", "status": "contradicted"}
    r2 = score_case(case, obs_with([other]))
    assert r2.contradicted == 1 and r2.failure_class == FailureClass.MODEL_HONESTY


def test_a_looping_model_is_stopped_and_the_repeats_are_counted(ctx: ToolContext) -> None:
    same = call("simplify", {"latex": r"\frac{x^2-1}{x-1}"})
    r = run("simplify-rational", [same], ctx, repeat_last=True)
    assert r.failure_class == FailureClass.LOOP and not r.completed
    assert r.stop_reason == "repeated_calls"
    assert r.calls == 1 and r.repeated_calls == 2 and r.unnecessary_calls == 2


def test_a_model_that_never_stops_is_a_limit_failure(ctx: ToolContext) -> None:
    replies = [call("search_handbook", {"query": q}) for q in ("a", "b", "c", "d", "e", "f", "g")]
    r = run("simplify-rational", replies, ctx)
    assert r.failure_class == FailureClass.RESOURCE_LIMIT and r.stop_reason == "step_limit"
    assert r.repeated_calls == 0 and r.calls == 6


def test_a_slow_model_is_a_resource_limit(ctx: ToolContext) -> None:
    class Slow:
        name = "slow"

        async def chat(
            self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
        ) -> AsyncIterator[Chunk]:
            await asyncio.sleep(5)
            yield Chunk(text="late")

    result = asyncio.run(
        run_case(Slow(), ctx, BY_ID["simplify-rational"], AgentLimits(max_steps=2, max_seconds=0.3))
    )
    assert result.failure_class == FailureClass.RESOURCE_LIMIT and not result.completed


def test_an_unreachable_model_host_is_its_own_class(ctx: ToolContext) -> None:
    llm = ReplayLLM()
    llm.error = LLMError("The model host is not reachable.")
    result = asyncio.run(run_case(llm, ctx, BY_ID["simplify-rational"], LIMITS))
    assert result.failure_class == FailureClass.MODEL_HOST and result.stop_reason == "model_error"
    assert not result.completed


def test_a_crash_in_the_loop_is_an_observation_not_an_exception(ctx: ToolContext) -> None:
    class Crash:
        name = "crash"

        async def chat(
            self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
        ) -> AsyncIterator[Chunk]:
            raise RuntimeError("boom")
            yield Chunk()

    result = asyncio.run(run_case(Crash(), ctx, BY_ID["simplify-rational"], LIMITS))
    assert not result.passed and result.failure_class is not None


def test_every_failure_is_classified(ctx: ToolContext) -> None:
    transcripts = [
        [say("I do not know.")],
        [call("simplify", {}), say("x")],
        [call("search_handbook", {"query": "q"}), say("x")],
    ]
    for replies in transcripts:
        r = run("simplify-rational", replies, ctx)
        assert not r.passed and r.failure_class is not None


def test_reasoning_and_long_replies_are_failures(ctx: ToolContext) -> None:
    leaked = run("chitchat", [say("Okay, let's think. You are welcome.")], ctx)
    assert not leaked.passed and "reasoning leaked into the reply" in leaked.notes
    long = run("chitchat", [say("word " * 300)], ctx)
    assert any("too long" in n for n in long.notes)


# ------------------------------------------------------------------------------------ the suite
def test_suite_repeats_and_selection(ctx: ToolContext) -> None:
    chosen = filter_cases(CASES, ids=["chitchat", "ask-when-ambiguous"])
    seen: list[CaseResult] = []
    llm = ReplayLLM([say("Which integral do you mean?")], repeat_last=True)
    results = asyncio.run(
        run_suite(
            llm, ctx, chosen, LIMITS, repeats=3, first_repeat=2, parallel=2, on_result=seen.append
        )
    )
    assert len(results) == 6 and len(seen) == 6
    assert sorted({r.repeat for r in results}) == [2, 3, 4]
    assert [r.case_id for r in results][:3] == [chosen[0].id] * 3
    with pytest.raises(ValueError, match="unknown case ids"):
        filter_cases(CASES, ids=["nope"])
    with pytest.raises(ValueError, match="unknown categories"):
        filter_cases(CASES, categories=["nope"])
    assert {c.category for c in filter_cases(CASES, categories=["followup"])} == {"followup"}


# ----------------------------------------------------------------------------------- reporting
def result(case_id: str, category: str = "symbolic", **kw: Any) -> CaseResult:
    base: dict[str, Any] = {
        "case_id": case_id,
        "category": category,
        "passed": True,
        "expected_tool": "simplify",
        "tool_selected": True,
        "tool_reached": True,
        "calls": 1,
        "completed": True,
        "seconds": 1.0,
        "llm_ms": 100,
        "tool_ms": 10,
        "llm_calls": 2,
        "first_tool": "simplify",
    }
    return CaseResult(**{**base, **kw})


def test_percentiles() -> None:
    assert percentile([], 95) is None
    assert percentile([5], 95) == 5
    assert percentile(list(range(1, 101)), 95) == 95
    assert percentile([3, 1, 2], 50) == 2


def test_summary_metrics() -> None:
    results = [
        result("a", seconds=1.0),
        result(
            "b",
            "calculus",
            seconds=3.0,
            expected_tool="integrate",
            first_tool="differentiate",
            tool_selected=False,
            tool_reached=False,
            passed=False,
            failure_class="tool_selection",
        ),
        result(
            "c",
            "calculus",
            invalid_calls=1,
            calls=2,
            passed=False,
            failure_class="arguments",
            seconds=2.0,
        ),
        result("d", false_verified=1, passed=False, failure_class="verification", completed=False),
        result("e", informational=True, passed=False, failure_class="model_honesty"),
    ]
    s = summarise(results)
    assert s["runs"] == 4 and s["informational_runs"] == 1  # informational runs are not scored
    assert s["passed"] == 1 and s["pass_rate"] == 0.25
    assert s["tool_selection_accuracy"] == 0.75
    assert s["valid_arguments_rate"] == pytest.approx(1 - 1 / 5)
    assert s["completion_rate"] == 0.75
    assert s["false_verified"] == 1
    assert s["failure_classes"] == {"tool_selection": 1, "arguments": 1, "verification": 1}
    assert s["unclassified_failures"] == 0
    assert s["per_category"]["calculus"]["passed"] == 0
    assert s["confusion"]["integrate"] == {"differentiate": 1}
    assert s["latency_seconds"]["median"] == 1.5 and s["latency_seconds"]["max"] == 3.0
    assert s["tokens_in"] is None  # not reported means not invented


def test_summary_finds_unstable_cases() -> None:
    results = [
        result("a", repeat=0),
        result("a", repeat=1, passed=False, failure_class="arguments"),
        result("b", repeat=0),
        result("b", repeat=1),
    ]
    s = summarise(results)
    assert s["flaky_cases"] == ["a"] and s["pass_counts_per_repeat"] == [2, 1]
    assert s["pass_count_stdev"] == 0.5


def test_unclassified_failures_are_visible() -> None:
    assert summarise([result("a", passed=False)])["unclassified_failures"] == 1


META = {
    "label": "base",
    "date": "2026-11-13",
    "git_sha": "abc",
    "cases_digest": "d1",
    "case_count": 4,
    "repeats": 1,
    "seed": 7,
    "temperature": 0.2,
    "prompt_sha": "p",
}


def test_document_and_markdown() -> None:
    doc = build_document(
        META,
        {
            "m1": [
                result("a"),
                result(
                    "b",
                    "calculus",
                    passed=False,
                    tool_selected=False,
                    failure_class="arguments",
                    notes=["bad"],
                ),
            ],
            "m2": None,
        },
    )
    assert doc["schema"] == 1 and doc["models"]["m2"] == {
        "usable": False,
        "summary": None,
        "results": [],
    }
    json.dumps(doc)  # serialisable
    text = render_markdown(doc)
    for needle in (
        "## Headline",
        "By category",
        "Tool selection",
        "Failures by class",
        "m2 | not usable",
        "NOT met",
        "Targets:",
        "`b` (arguments)",
    ):
        assert needle in text, needle
    assert "http" not in text and "11434" not in text  # no host in a report


def test_comparison_lists_changes() -> None:
    base = build_document(
        META, {"m": [result("a", passed=False, failure_class="arguments"), result("b")]}
    )
    new = build_document(
        {**META, "label": "new", "git_sha": "def"},
        {"m": [result("a"), result("b", passed=False, failure_class="tool_selection")]},
    )
    text = compare(base, new)
    assert "Cases that improved (1): `a`" in text and "Cases that got worse (1): `b`" in text
    assert "pass rate | 50.0% | 50.0%" in text
    other = build_document({**META, "cases_digest": "d2"}, {"m": [result("a")]})
    assert "case sets differ" in compare(base, other)
    renamed = build_document(META, {"other-name": [result("a")]})
    assert "m against other-name" in compare(base, renamed)
    two = build_document(META, {"x": [result("a")], "y": [result("a")]})
    assert "No model appears in both" in compare(
        two, build_document(META, {"z": [result("a")], "w": [result("b")]})
    )


# ---------------------------------------------------------------------------------------- the CLI
def test_cli_lists_and_compares(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--list", "--categories", "followup"]) == 0
    out = capsys.readouterr().out
    assert "followup-uses-context" in out and "simplify-rational" not in out

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(build_document(META, {"m": [result("a")]})))
    b.write_text(json.dumps(build_document({**META, "label": "new"}, {"m": [result("a")]})))
    report = tmp_path / "out" / "cmp.md"
    assert cli.main(["--compare-only", str(a), str(b), "--out", str(report)]) == 0
    assert "Baseline against new" in report.read_text()

    junk = tmp_path / "junk.json"
    junk.write_text("{}")
    with pytest.raises(SystemExit):
        cli.main(["--compare-only", str(a), str(junk)])


def test_cli_rejects_bad_selections_and_missing_host(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--cases", "nope"])
    with pytest.raises(SystemExit):
        cli.main(["--cases", "chitchat"])  # no host and no models
    capsys.readouterr()


def test_the_script_is_a_thin_wrapper() -> None:
    text = (Path(__file__).resolve().parents[3] / "scripts" / "eval_models.py").read_text()
    assert "minerva_api.agent.evaluation.cli" in text and len(text.splitlines()) < 30


# -------------------------------------------------------------------- providers and provider metrics
def _args(**over: Any) -> Any:
    import argparse

    base: dict[str, Any] = {
        "base_url": "http://ollama.test:11434",
        "timeout": 5.0,
        "think": False,
        "strip_thinking": False,
        "temperature": 0.2,
        "seed": 7,
    }
    return argparse.Namespace(**{**base, **over})


def test_a_bare_model_name_is_an_ollama_model_and_each_repeat_has_its_own_seed() -> None:
    provider, model, client = cli.make_client("qwen3.5:35b", _args())
    assert (provider, model) == ("ollama", "qwen3.5:35b") and client.seed == 7  # type: ignore[attr-defined]
    assert cli.make_client("qwen3.5:35b", _args(), repeat=2)[2].seed == 9  # type: ignore[attr-defined]
    assert cli.make_client("ollama:gemma4:26b", _args())[:2] == ("ollama", "gemma4:26b")
    assert cli.model_label("ollama", "gemma4:26b") == "gemma4:26b"  # reports keep the bare name
    assert cli.model_label("openrouter", "openai/gpt-4.1-mini") == "openrouter:openai/gpt-4.1-mini"


def test_an_ollama_model_needs_a_host_and_an_openrouter_model_needs_a_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit, match="--base-url"):
        cli.make_client("qwen3.5:35b", _args(base_url=""))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        cli.make_client("openrouter:openai/gpt-4.1-mini", _args(base_url=""))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-0000")
    provider, model, client = cli.make_client("openrouter:openai/gpt-4.1-mini", _args(base_url=""))
    assert (provider, model, client.name) == (
        "openrouter",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1-mini",
    )


def test_the_cli_accepts_openrouter_models_without_a_host(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit):  # stops at the missing key, not at the missing host
        cli.main(["--models", "openrouter:openai/gpt-4.1-mini", "--cases", "chitchat"])
    err = capsys.readouterr().err
    assert "--base-url" not in err
    with pytest.raises(SystemExit):
        cli.main(["--models", "qwen3.5:35b", "--cases", "chitchat"])
    assert "--base-url" in capsys.readouterr().err


def test_provider_metrics_are_scored_and_summarised(ctx: ToolContext) -> None:
    def calls(*calls: dict[str, Any]) -> Observation:
        return obs_with(
            [],
            trace=[
                {"id": "a", "name": "simplify", "status": s}
                for s in ("ok", "error", "unchanged", "skipped")
            ],
        )

    observation = calls()
    observation.final["debug"]["llm_calls"] = [  # type: ignore[index]
        {
            "ms": 9,
            "text_chars": 1,
            "tool_calls": 0,
            "first_token_ms": 120,
            "cost_usd": 0.002,
            "prompt_tokens": 50,
            "completion_tokens": 7,
        },
        {
            "ms": 5,
            "text_chars": 1,
            "tool_calls": 0,
            "first_token_ms": 80,
            "cost_usd": 0.001,
            "prompt_tokens": 60,
            "completion_tokens": 3,
        },
    ]
    observation.final["debug"]["error_kind"] = "rate_limit"  # type: ignore[index]
    r = score_case(BY_ID["chitchat"], observation)
    assert (r.first_token_ms, r.tokens_in, r.tokens_out) == (120, 110, 10)
    assert r.cost_usd == pytest.approx(0.003) and r.error_kind == "rate_limit"
    assert r.tool_failures == 1 and r.unverified_results == 1  # `skipped` is neither
    s = summarise([r, result("b", cost_usd=None)])
    assert s["cost_usd"] == pytest.approx(0.003) and s["rate_limited"] == 1 and s["timeouts"] == 0
    assert s["tool_failures"] == 1 and s["first_token_ms"]["median"] == 120
    doc = build_document(META, {"m": [r]})
    assert "provider rate limits / timeouts: 1 / 0" in render_markdown(doc)
    assert "cost: 0.0030 USD" in render_markdown(doc)


def test_reports_saved_before_these_metrics_still_render() -> None:
    doc = build_document(META, {"m": [result("a")]})
    for key in (
        "tool_failures",
        "unverified_results",
        "cost_usd",
        "first_token_ms",
        "rate_limited",
        "timeouts",
    ):
        del doc["models"]["m"]["summary"][key]
    text = render_markdown(doc)
    assert "provider rate limits / timeouts: n/a / n/a" in text and "cost: n/a" in text
    assert "Baseline against new" in compare(doc, doc)


def test_the_geometry_and_distribution_cases_are_part_of_the_benchmark() -> None:
    ported = {
        "geometry-schwarzschild-ricci",
        "geometry-does-not-invent-a-metric",
        "geometry-follow-up-resends-the-same-metric",
        "delta-integral-uses-the-tool",
        "delta-on-a-limit-asks-for-a-convention",
    }
    assert ported <= set(BY_ID)
    assert all(BY_ID[i].informational for i in ported)  # measured and reported, not scored
    history = BY_ID["geometry-follow-up-resends-the-same-metric"].history
    assert (
        history[1]["tool_calls"][0]["id"] == history[2]["tool_call_id"]
    )  # the neutral message shape
