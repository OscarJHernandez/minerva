"""`explicit_operation`: one imperative sentence about one formula, every argument written in it.

Baseline 2026-11-18: "Differentiate |z| with respect to the complex variable z" and "Give the Taylor series of 1/x
about 0 up to order 4" were answered in prose in 3 of 3 runs, without the tool that would have said, with its own
words, that Minerva does not differentiate |z| or expand a pole. A route only proposes the call: the engine decides,
and an unsupported result is an answer the model then reports.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from minerva_api.agent.loop import AgentLimits, TurnInfo, run_agent
from minerva_api.agent.replay import ReplayLLM, say
from minerva_api.agent.routing import RULES, explicit_operation, plan_route
from minerva_api.agent.tools import ToolContext
from minerva_api.agent.tracing import RouteReason
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


@pytest.mark.parametrize(
    ("message", "tool", "arguments"),
    [
        (
            r"Differentiate $|z|$ with respect to the complex variable z.",
            "differentiate",
            {"latex": "|z|", "variables": [{"name": "z"}]},
        ),
        (
            r"Differentiate $x^3 \sin x$ with respect to $x$.",
            "differentiate",
            {"latex": r"x^3 \sin x", "variables": [{"name": "x"}]},
        ),
        (
            r"Please differentiate $\exp(x)\, x^2$ with respect to x",
            "differentiate",
            {"latex": r"\exp(x)\, x^2", "variables": [{"name": "x"}]},
        ),
        (r"Simplify $x^2 + y^2$.", "simplify", {"latex": "x^2 + y^2"}),
        (r"Please simplify $\frac{x^2-1}{x-1}$", "simplify", {"latex": r"\frac{x^2-1}{x-1}"}),
        (
            r"Find the derivative of $t^4$.",  # one plain letter: the variable
            "differentiate",
            {"latex": "t^4", "variables": [{"name": "t"}]},
        ),
        (
            r"Find an antiderivative of $x^5$.",
            "integrate",
            {"latex": "x^5", "variable": "x"},
        ),
        (
            r"Give the Taylor series of $1/x$ about 0 up to order 4.",
            "series",
            {"latex": "1/x", "variable": "x", "point": "0", "order": 4},
        ),
        (
            r"Could you give me the Taylor polynomial of $\cos x$ around the point 0 to order 6?",
            "series",
            {"latex": r"\cos x", "variable": "x", "point": "0", "order": 6},
        ),
    ],
)
def test_a_plain_sentence_with_every_argument_in_it_routes(
    message: str, tool: str, arguments: dict[str, Any]
) -> None:
    route = explicit_operation(message)
    assert route is not None and route.tool == tool and route.arguments == arguments
    assert route.reason == "explicit_operation" and route.answers in (
        {"ok", "unsupported"},
        {"ok", "unchanged", "conditions_not_satisfied"},
    )
    assert plan_route(message) is not None


@pytest.mark.parametrize(
    "message",
    [
        # an argument the user did not give: the model asks
        r"Differentiate $a x^2 + b y$.",  # two letters, no variable named
        r"Give the Taylor series of $\cos x$ about 0.",  # no order
        r"Give the Taylor series of $\cos x$ up to order 4.",  # no point
        r"Differentiate $x^2$.\nDifferentiate $y^2$.",  # two formulas
        # the request is more than the plain operation
        r"Differentiate $\sin x\, e^{x}$ twice with respect to x.",
        r"Differentiate $x^2$ numerically.",
        r"Differentiate $x^2$ assuming x is positive.",
        r"Differentiate $x^2$ with respect to x and y.",
        r"Differentiate $x^2$ with respect to x and explain each step.",
        r"Find an antiderivative of $x^2$ from 0 to 1.",
        r"Compute $\int_0^\pi \sin x \, dx$.",
        r"What is the rate of change of $x^2$ with respect to $x$?",
        r"Differentiate with respect to x.",  # no formula
        r"Simplify $\sqrt{b^2}$ for positive b.",  # an assumption in words: the model passes it, never a rule
        r"Let b be positive. Simplify $\sqrt{b^2}$.",
        r"Simplify $x^2$ with respect to x.",
        r"Simplify $x + 1$ and then differentiate it.",
        r"Simplify $\log(p q)$ and $\log p$.",  # two formulas
        r"Simplify $L = \frac12 m \dot{x}^2$.",  # an equation, not an expression
        # a bare e or i is a symbol here, and the user meant a constant: the model or the user says which
        r"Find an antiderivative of $e^{x^x}$.",
        r"Differentiate $e^{x}$ with respect to x.",
        r"Differentiate $i x$ with respect to x.",
        # two different variables named
        r"Differentiate $x y$ with respect to x with respect to y.",
        # an earlier result is not a formula in this message
        r"Differentiate $@r1$ with respect to x.",
    ],
)
def test_a_request_that_needs_a_decision_the_user_did_not_make_does_not_route(
    message: str,
) -> None:
    assert explicit_operation(message) is None


def test_every_rule_has_a_reason_the_tracer_knows() -> None:
    """A rule whose reason is missing from RouteReason would be silently recorded as declined."""
    from inspect import signature

    reasons = {r.value for r in RouteReason}
    for rule in RULES:
        assert rule.__name__ in reasons, rule.__name__
    assert "explicit_operation" in reasons
    assert len(signature(explicit_operation).parameters) == 2


def run(llm: ReplayLLM, ctx: ToolContext, message: str, routing: bool = True) -> dict[str, Any]:
    async def go() -> list[dict[str, Any]]:
        return [
            e
            async for e in run_agent(
                llm,
                [{"role": "user", "content": message}],
                ctx,
                AgentLimits(retry_delay_seconds=0),
                info=TurnInfo(routing=routing),
            )
        ]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


def test_an_unsupported_result_is_an_answer_the_model_reports(ctx: ToolContext) -> None:
    llm = ReplayLLM([say("Minerva does not differentiate $|z|$ in the complex variable.")])
    message = run(llm, ctx, r"Differentiate $|z|$ with respect to the complex variable z.")
    (step,) = message["trace"]
    assert step["name"] == "differentiate" and step["routed"] is True
    assert step["status"] == "unsupported"  # kept as the answer: the model explains it
    assert message["debug"]["routing"]["reason"] == "explicit_operation"
    assert len(message["debug"]["llm_calls"]) == 1
    assert "unsupported" in llm.seen[0][-1]["content"]


def test_a_supported_result_is_verified_and_the_model_only_writes_the_answer(
    ctx: ToolContext,
) -> None:
    llm = ReplayLLM([say(r"The derivative is $3 x^{2} \sin x + x^{3} \cos x$.")])
    message = run(llm, ctx, r"Differentiate $x^3 \sin x$ with respect to $x$.")
    (step,) = message["trace"]
    assert step["routed"] is True and step["status"] == "ok"
    assert step["evidence"]["accepted"] is True
    assert step[
        "verified_latex"
    ]  # the engine's own result, checked by an independent numerical derivative


def test_a_taylor_series_of_a_pole_is_reported_unsupported_never_expanded(ctx: ToolContext) -> None:
    llm = ReplayLLM([say("There is no Taylor series about a pole; Minerva does not give one.")])
    message = run(llm, ctx, r"Give the Taylor series of $1/x$ about 0 up to order 4.")
    (step,) = message["trace"]
    assert step["name"] == "series" and step["status"] == "unsupported"
    assert step["verified_latex"] == []


def test_routing_off_leaves_the_model_to_plan(ctx: ToolContext) -> None:
    llm = ReplayLLM([say("I would look at that.")])
    message = run(llm, ctx, r"Find an antiderivative of $x^5$.", routing=False)
    assert message["trace"] == [] and message["debug"]["routing"] == {}


def test_a_declined_route_leaves_the_turn_as_if_there_were_none(ctx: ToolContext) -> None:
    """A formula the parser refuses is not a result: the route is dropped and the model plans."""
    llm = ReplayLLM([say("I could not read that formula.")])
    message = run(llm, ctx, r"Differentiate $\frac{d}{dx} x$ with respect to x.")
    assert message["trace"] == []
    assert message["debug"]["routing"]["decision"] == "declined"


def test_simplify_reports_an_unchanged_input_as_the_answer_and_never_invents_an_assumption(
    ctx: ToolContext,
) -> None:
    llm = ReplayLLM([say("It is already as simple as it gets.")])
    message = run(llm, ctx, r"Simplify $x^2 + y^2$.")
    (step,) = message["trace"]
    assert step["name"] == "simplify" and step["routed"] is True and step["status"] == "unchanged"
    assert step["arguments"] == {"latex": "x^2 + y^2"}  # no assumptions of any kind
    assert len(message["debug"]["llm_calls"]) == 1


def test_simplify_that_needs_an_assumption_is_reported_as_such(ctx: ToolContext) -> None:
    llm = ReplayLLM([say("That depends on an assumption about x.")])
    message = run(llm, ctx, r"Simplify $\sqrt{x^2}$.")
    (step,) = message["trace"]
    assert step["status"] == "conditions_not_satisfied" and step["routed"] is True


def test_a_conversation_about_matrices_vectors_or_tensors_is_left_to_their_tools() -> None:
    from minerva_api.agent.routing import RouteContext

    context = RouteContext(
        user_messages=("A and B are 2x2 matrices.", r"Simplify $\exp(A)\exp(B)$.")
    )
    assert explicit_operation(r"Simplify $\exp(A)\exp(B)$.", context) is None
    assert explicit_operation(r"Differentiate $x^2$ with respect to x for the vector r.") is None
    assert (
        explicit_operation(r"Simplify $x^2 + y^2$.") is not None
    )  # the same sentence without them
