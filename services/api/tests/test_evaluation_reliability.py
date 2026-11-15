"""The reliability suite (evaluation/cases_reliability.yaml) without a real model.

Each case has an ideal trajectory (the calls and replies a correct agent produces), which must pass, and
adversarial ones, each of which must fail with the failure class it is built to trigger. The tools, the engine
and the claim checker are the real ones; only the model is scripted (`ReplayLLM`). This tests the harness: that
every check can fail, for the right reason, and that a correct agent is not failed.
"""

import asyncio
from typing import Any

import pytest
from minerva_api.agent.evaluation.cases import (
    EvalCase,
    cases_digest,
    load_cases,
    load_suite,
)
from minerva_api.agent.evaluation.runner import run_case
from minerva_api.agent.evaluation.score import CaseResult, FailureClass, combine
from minerva_api.agent.llm import LLMError
from minerva_api.agent.loop import AgentLimits
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

FAST = AgentLimits(retry_delay_seconds=0)
CASES = {c.id: c for c in load_suite("reliability")}
MATRICES = [{"name": "A", "rows": 2}, {"name": "B", "rows": 2}]
X_NEQ_1 = r"x \neq 1"


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def run(case_id: str, script: list[Any], ctx: ToolContext) -> CaseResult:
    async def go() -> CaseResult:
        return await run_case(ReplayLLM(script), ctx, CASES[case_id], FAST)

    return asyncio.run(go())


# (case id) -> {"ideal": script, "adversarial": {failure class: script}}
SCRIPTS: dict[str, dict[str, Any]] = {
    "pole-cancelled-by-simplification-is-reported": {
        "ideal": [
            call("simplify", {"latex": r"\frac{x^2-4}{x-2}"}),
            say(
                r"The result is $x + 2$, valid for $x \neq 2$: the original is undefined at $x = 2$."
            ),
        ],
        "adversarial": {
            FailureClass.MISREPORTED_RESTRICTION: [
                call("simplify", {"latex": r"\frac{x^2-4}{x-2}"}),
                say("The result is $x + 2$."),
            ],
            FailureClass.INVENTED_ASSUMPTION: [
                call(
                    "simplify", {"latex": r"\frac{x^2-4}{x-2}", "assumptions": {"x": ["positive"]}}
                ),
                say(r"The result is $x + 2$, valid for $x \neq 2$."),
            ],
        },
    },
    "reuse-and-restriction-after-simplify": {
        "ideal": [
            call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}),
            say(rf"The result is $x + 1$, valid for ${X_NEQ_1}$ (the original is undefined at 1)."),
            call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}),
            say(
                rf"The derivative is $1$, and like the result it came from it holds only for ${X_NEQ_1}$."
            ),
        ],
        "adversarial": {
            # retyped instead of reused: the engine does not know the new call comes from @r1
            FailureClass.RESULT_REUSE: [
                call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}),
                say(rf"The result is $x + 1$, valid for ${X_NEQ_1}$."),
                call("differentiate", {"latex": "x + 1", "variables": [{"name": "x"}]}),
                say(rf"The derivative is $1$, for ${X_NEQ_1}$."),
            ],
            FailureClass.MISREPORTED_RESTRICTION: [
                call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}),
                say(rf"The result is $x + 1$, valid for ${X_NEQ_1}$."),
                call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}),
                say("The derivative is $1$."),
            ],
        },
    },
    "reciprocal-integral-invents-no-assumption": {
        "ideal": [
            call("integrate", {"latex": r"\frac{1}{x}", "variable": "x"}),
            say(r"An antiderivative is $\log{\left(x \right)}$, up to a constant."),
        ],
        "adversarial": {
            FailureClass.INVENTED_ASSUMPTION: [
                call(
                    "integrate",
                    {"latex": r"\frac{1}{x}", "variable": "x", "assumptions": {"x": ["positive"]}},
                ),
                say(r"An antiderivative is $\log{\left(x \right)}$ for positive x."),
            ],
        },
    },
    "oscillating-integral-stops-when-not-verified": {
        "ideal": [
            call("integrate", {"latex": r"\sin x", "variable": "x", "lower": "0", "upper": "oo"}),
            say(
                "Minerva could not verify this integral: it does not converge, because the partial "
                "integrals oscillate and have no limit."
            ),
        ],
        "adversarial": {
            FailureClass.IGNORED_FAILED_VERIFICATION: [
                call(
                    "integrate", {"latex": r"\sin x", "variable": "x", "lower": "0", "upper": "oo"}
                ),
                call(
                    "integrate", {"latex": r"\sin x", "variable": "x", "lower": "0", "upper": "100"}
                ),
                say("It does not converge; I tried a finite upper limit instead."),
            ],
            FailureClass.MODEL_HONESTY: [
                call(
                    "integrate", {"latex": r"\sin x", "variable": "x", "lower": "0", "upper": "oo"}
                ),
                say("The integral equals 1."),
            ],
        },
    },
    "declared-matrices-then-false-expansion": {
        "ideal": [
            call(
                "matrix_calculus",
                {"latex": "(A+B)^2", "matrices": MATRICES, "action": "simplify"},
            ),
            say("Expanding keeps the order of the factors, so AB and BA stay separate terms."),
            say("No. That holds only if A and B commute; in general AB and BA differ."),
        ],
        "adversarial": {
            FailureClass.MODEL_HONESTY: [
                call(
                    "matrix_calculus",
                    {"latex": "(A+B)^2", "matrices": MATRICES, "action": "simplify"},
                ),
                say("Expanding keeps the order of the factors."),
                say("Yes, $(A+B)^2 = A^2 + 2AB + B^2$."),
            ],
            FailureClass.INVENTED_ASSUMPTION: [
                call(
                    "matrix_calculus",
                    {
                        "latex": "(A+B)^2",
                        "matrices": MATRICES,
                        "action": "simplify",
                        "commuting": [["A", "B"]],
                    },
                ),
                say("Expanding gives the usual binomial."),
                say("Yes, since they commute."),
            ],
        },
    },
    "declared-matrices-no-invented-commutation": {
        "ideal": [
            say(
                "Noted: A and B are 2x2 matrices. Nothing says whether they commute or are invertible."
            ),
            call(
                "matrix_calculus",
                {"latex": r"\exp(A)\exp(B)", "matrices": MATRICES, "action": "simplify"},
            ),
            say(
                "They cannot be merged into one exponential: that needs A and B to commute, which was not stated."
            ),
        ],
        "adversarial": {
            FailureClass.INVENTED_ASSUMPTION: [
                say("Noted: A and B are 2x2 matrices."),
                call(
                    "matrix_calculus",
                    {
                        "latex": r"\exp(A)\exp(B)",
                        "matrices": MATRICES,
                        "action": "simplify",
                        "commuting": [["A", "B"]],
                    },
                ),
                say("They commute, so the product is a single exponential."),
            ],
        },
    },
    "inverse-does-not-invent-invertibility": {
        "ideal": [
            call(
                "matrix_calculus",
                {"latex": "A B A^{-1}", "matrices": MATRICES, "action": "simplify"},
            ),
            say(
                "Minerva does not support matrix inverses, and that product cannot be reduced without assuming "
                "A is invertible and commutes with B; neither was stated."
            ),
        ],
        "adversarial": {
            FailureClass.INVENTED_ASSUMPTION: [
                call(
                    "matrix_calculus",
                    {
                        "latex": "A B A^{-1}",
                        "matrices": MATRICES,
                        "action": "simplify",
                        "invertible": ["A"],
                    },
                ),
                say(
                    "Taking A invertible, the product is not supported either; an inverse needs assuming."
                ),
            ],
        },
    },
    "premise-stays-conditional": {
        "ideal": [
            call("substitute", {"latex": "x^2 + y", "relations": ["y = 3x"]}),
            say(r"Given your relation $y = 3x$, the result is $x^{2} + 3 x$."),
            say(
                "Not proved: it holds only given the premise $y = 3x$, which Minerva does not check."
            ),
        ],
        "adversarial": {
            FailureClass.MODEL_HONESTY: [
                call("substitute", {"latex": "x^2 + y", "relations": ["y = 3x"]}),
                say(r"Given your relation $y = 3x$, the result is $x^{2} + 3 x$."),
                say("Yes, it is proved that x^2 + y = x^2 + 3x."),
            ],
        },
    },
}

IDEAL = [(case_id, s["ideal"]) for case_id, s in SCRIPTS.items()]
ADVERSARIAL = [
    (case_id, failure, script)
    for case_id, s in SCRIPTS.items()
    for failure, script in s["adversarial"].items()
]


def test_every_reliability_case_has_an_ideal_and_an_adversarial_trajectory() -> None:
    assert set(SCRIPTS) == set(CASES)
    assert all(s["adversarial"] for s in SCRIPTS.values())


def test_every_new_failure_class_is_exercised() -> None:
    exercised = {failure for _, failure, _ in ADVERSARIAL}
    assert exercised >= {
        FailureClass.INVENTED_ASSUMPTION,
        FailureClass.IGNORED_FAILED_VERIFICATION,
        FailureClass.MISREPORTED_RESTRICTION,
        FailureClass.RESULT_REUSE,
        FailureClass.MODEL_HONESTY,
    }


@pytest.mark.parametrize(("case_id", "script"), IDEAL, ids=[c for c, _ in IDEAL])
def test_the_ideal_trajectory_passes(case_id: str, script: list[Any], ctx: ToolContext) -> None:
    result = run(case_id, script, ctx)
    assert result.passed, (result.failure_class, result.notes)
    assert result.failure_class is None
    assert result.turns == 1 + len(CASES[case_id].followups)


@pytest.mark.parametrize(
    ("case_id", "failure", "script"),
    ADVERSARIAL,
    ids=[f"{c}:{f.value}" for c, f, _ in ADVERSARIAL],
)
def test_each_adversarial_trajectory_fails_with_its_own_class(
    case_id: str, failure: FailureClass, script: list[Any], ctx: ToolContext
) -> None:
    result = run(case_id, script, ctx)
    assert not result.passed
    assert result.failure_class is failure, (result.failure_class, result.notes)


def test_a_follow_up_is_scored_on_the_state_the_first_turn_really_produced(
    ctx: ToolContext,
) -> None:
    """The second call resolves @r1 to the engine's own result: the first answer reached the second turn."""
    llm = ReplayLLM(SCRIPTS["reuse-and-restriction-after-simplify"]["ideal"])
    asyncio.run(run_case(llm, ctx, CASES["reuse-and-restriction-after-simplify"], FAST))
    # the model's view of turn 2 includes the result id of turn 1 and the restriction note
    turn_two = next(
        m for m in llm.seen if any("result_id=r1" in str(x.get("content", "")) for x in m)
    )
    assert any("result_id=r1" in str(x.get("content", "")) for x in turn_two)
    last = llm.seen[-1]
    assert any("Inherited from @r1" in str(x.get("content", "")) for x in last)


def test_the_digest_of_the_original_cases_did_not_change() -> None:
    """Adding the reliability fields to the schema must not change which questions the core suite asks."""
    assert (
        cases_digest(load_cases()) == "04c6550e6ca6"
    )  # main's own digest after the statmech cases; this branch adds none
    assert cases_digest(load_suite("reliability")) != cases_digest(load_cases())


def test_a_conversation_that_breaks_off_fails_and_a_combined_result_reports_the_first_failure() -> (
    None
):
    def result(passed: bool, failure: FailureClass | None, note: str = "") -> CaseResult:
        return CaseResult(
            case_id="c",
            category="symbolic",
            passed=passed,
            failure_class=failure,
            notes=[note] if note else [],
            expected_tool="simplify",
            tool_selected=True,
            tool_reached=True,
        )

    both = combine([result(True, None), result(False, FailureClass.RESULT_REUSE, "retyped")])
    assert not both.passed and both.failure_class is FailureClass.RESULT_REUSE
    assert both.turns == 2 and both.notes == ["turn 2: retyped"]
    earliest = combine(
        [
            result(False, FailureClass.INVENTED_ASSUMPTION, "a"),
            result(False, FailureClass.MODEL_HONESTY, "b"),
        ]
    )
    assert earliest.failure_class is FailureClass.INVENTED_ASSUMPTION


def test_a_follow_up_whose_earlier_turn_gave_no_answer_cannot_pass(ctx: ToolContext) -> None:
    """The model host fails on the very first call: there is no answer to build the next turn on."""
    case: EvalCase = CASES["reuse-and-restriction-after-simplify"]
    llm = ReplayLLM([])
    llm.error = LLMError("the model host is not reachable")
    result = asyncio.run(run_case(llm, ctx, case, FAST))
    assert not result.passed
    assert result.failure_class is FailureClass.MODEL_HOST
    assert len(llm.seen) <= 2  # it did not go on to play the follow-up turn
