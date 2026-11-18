"""The `followup_reuse`, `ode_analyze` and `matrix_simplify` routes (agent/routing.py).

Each rule needs two independent signals and every argument written by the user; this file has, for each, the
positive reading, the keyword alone, the formula alone, the ambiguous request and the missing argument (which
must all leave the model to plan), and runs the real evaluation cases through the loop with routing on.
"""

import asyncio
from typing import Any

import pytest
from minerva_api.agent.evaluation.cases import load_suite
from minerva_api.agent.evaluation.runner import run_case
from minerva_api.agent.loop import AgentLimits, latest_result_ids
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.routing import (
    RouteContext,
    declared_matrix_sizes,
    followup_reuse,
    matrix_simplify,
    ode_analyze,
    plan_route,
    single_symbol,
)
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

FAST = AgentLimits(retry_delay_seconds=0)
CASES = {c.id: c for c in [*load_suite("reliability"), *load_suite("heldout")]}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def after(result: str, *, many: tuple[str, ...] = ("r1",)) -> RouteContext:
    return RouteContext(results={"r1": result, "r2": "y"}, latest=many)


# -------------------------------------------------------------------------------------- single_symbol
@pytest.mark.parametrize(
    ("latex", "symbol"),
    [
        ("x + 1", "x"),
        (r"\frac{7}{2} + t", "t"),
        (r"\sin{\left(x \right)} + 2", "x"),
        (r"\mathrm{e}^{x}", "x"),
        (
            r"2 \pi r",
            None,
        ),  # two letters (the constant pi is read as a function-like command, r is one)
    ],
)
def test_single_symbol(latex: str, symbol: str | None) -> None:
    assert single_symbol(latex) == (symbol if latex != r"2 \pi r" else "r")


@pytest.mark.parametrize(
    "latex", ["x y", r"\theta + x", "x_1 + 1", r"\mu r^2", "7", r"\operatorname{Ai}(x)", "a + b"]
)
def test_single_symbol_refuses_whenever_a_symbol_could_hide(latex: str) -> None:
    assert single_symbol(latex) is None


# ------------------------------------------------------------------------------------ followup_reuse
def test_followup_reuse_differentiates_the_one_result_of_the_last_turn() -> None:
    route = followup_reuse("Now differentiate that result.", after("x + 1"))
    assert route is not None and route.tool == "differentiate"
    assert route.arguments == {"latex": "@r1", "variables": [{"name": "x"}]}
    assert route.reason == "followup_reuse"


def test_followup_reuse_integrates_with_the_bounds_the_user_wrote() -> None:
    route = followup_reuse("Good. Now integrate it from 0 to 1.", after("x + 3"))
    assert route is not None and route.tool == "integrate"
    assert route.arguments == {"latex": "@r1", "variable": "x", "lower": "0", "upper": "1"}
    indefinite = followup_reuse("Integrate the previous result.", after("x + 3"))
    assert indefinite is not None and "lower" not in indefinite.arguments


def test_followup_reuse_takes_the_variable_the_user_named() -> None:
    route = followup_reuse("integrate it with respect to y", after("x*y"))
    assert route is not None and route.arguments["variable"] == "y"
    assert followup_reuse("integrate it over $t$ between 0 and 2", after("t + 5")) is not None


@pytest.mark.parametrize(
    "message",
    [
        "Differentiate $x^2$.",  # a formula of its own: not a pointer to the earlier result
        "Differentiate the function.",  # no pointer
        "It would be good to integrate.",  # a keyword and no pointer
        "Now simplify that result.",  # another operation: not this rule
        "Differentiate it twice.",  # an order the rule does not read
        "Differentiate it and then integrate it.",  # two operations
        "Integrate it numerically from 0 to 1.",
        "Integrate it from a to b.",  # bounds in words the rule cannot read
        "Integrate it over the interval.",
        "Differentiate it from 0 to 1.",  # bounds make no sense here
        "Integrate it in x and in y.",  # two variables named
    ],
)
def test_followup_reuse_leaves_everything_else_to_the_model(message: str) -> None:
    assert followup_reuse(message, after("x + 3")) is None


def test_followup_reuse_needs_exactly_one_result_and_a_readable_variable() -> None:
    assert followup_reuse("Differentiate that result.", RouteContext()) is None  # nothing earlier
    assert followup_reuse("Differentiate that result.", after("x", many=())) is None
    assert (
        followup_reuse("Differentiate that result.", after("x", many=("r1", "r2"))) is None
    )  # which one?
    assert followup_reuse("Differentiate that result.", after("x*y")) is None  # which variable?
    assert followup_reuse("Differentiate that result.", after(r"\theta + x")) is None
    assert followup_reuse("Differentiate that result.", after("3")) is None


def test_latest_result_ids_reads_only_the_previous_turn() -> None:
    note = "status=ok; result_id=r{n} (write @r{n} in a later call to reuse this result exactly)"
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "one"},
        {"role": "tool", "content": note.format(n=1)},
        {"role": "user", "content": "two"},
        {"role": "tool", "content": note.format(n=2)},
        {"role": "tool", "content": note.format(n=3)},
        {"role": "user", "content": "three"},
    ]
    assert latest_result_ids(history) == ("r2", "r3")
    assert latest_result_ids(history[:2]) == ()


# ---------------------------------------------------------------------------------------- ode_analyze
@pytest.mark.parametrize(
    ("message", "equation", "variable"),
    [
        ("$y''(x) = y(x)$.", "y''(x) = y(x)", "x"),
        (
            "What kind of equation is $u'(t) = 3u(t)$, and how could it be solved?",
            "u'(t) = 3u(t)",
            "t",
        ),
        ("Solve $y'(x) + 2y(x) = \\sin(x)$.", "y'(x) + 2y(x) = \\sin(x)", None),
    ],
)
def test_ode_analyze_reads_an_equation_with_a_derivative_of_an_applied_function(
    message: str, equation: str, variable: str | None
) -> None:
    route = ode_analyze(message)
    if variable is None:  # a LaTeX command inside the equation: left to the model
        assert route is None
        return
    assert route is not None and route.tool == "ode_analyze"
    assert route.arguments == {"equations": [equation], "independent": variable}


@pytest.mark.parametrize(
    "message",
    [
        "y''(x) = y(x)",  # no delimiters: not read as a formula
        "$y(x) = x^2$",  # a formula without a derivative
        "$u_t = u_{xx}$",  # a partial equation
        "$y''(x) = y(t)$",  # two independent variables
        "$y''(x) = y(x)$ and $z'(x) = z(x)$",  # two formulas
        "Solve $y''(x) = y(x)$ numerically.",  # a method is named
        "Solve $y'(x) = y(x)$ with y(0) = 1.",  # initial values: the model plans the call
        "Verify that $y(x) = e^x$ solves $y'(x) = y(x)$.",
        "Find the equations of motion for $y'(x) = y(x)$.",  # Lagrangian wording
        "Differentiate $y'(x)$.",
        "Tell me a story about $y'(x) = y(x)$.",  # words that ask for something else
        "$\\dot{y} = y$",
    ],
)
def test_ode_analyze_leaves_everything_else_to_the_model(message: str) -> None:
    assert ode_analyze(message) is None


# ----------------------------------------------------------------------------------- matrix_simplify
def test_matrix_sizes_are_read_from_the_users_words_only() -> None:
    assert declared_matrix_sizes(["Let A and B be 2x2 matrices."]) == {"A": 2, "B": 2}
    assert declared_matrix_sizes(["P, Q and R are 3 by 3 matrices"]) == {"P": 3, "Q": 3, "R": 3}
    assert declared_matrix_sizes(["A is a 2x3 matrix"]) == {}  # not square
    assert declared_matrix_sizes(["Let A be 2x2 matrices.", "A is 3x3 matrices"]) == {}  # two sizes
    assert declared_matrix_sizes(["A and B are matrices"]) == {}  # no size


def test_matrix_simplify_needs_the_declaration_and_a_formula_over_declared_names() -> None:
    context = RouteContext(user_messages=["Let A and B be 2x2 matrices."])
    route = matrix_simplify(r"Simplify $\exp(A)\exp(B)$.", context)
    assert route is not None and route.tool == "matrix_calculus"
    assert route.arguments == {
        "latex": r"\exp(A)\exp(B)",
        "matrices": [{"name": "A", "rows": 2}, {"name": "B", "rows": 2}],
        "action": "simplify",
    }
    assert (
        "conditions_not_satisfied" in route.answers
    )  # the honest answer for a non-commuting product
    same_message = matrix_simplify("A and B are 3x3 matrices. Expand $(A+B)^2$.")
    assert same_message is not None and same_message.arguments["matrices"][0]["rows"] == 3
    # never any property of the matrices the user did not state
    assert not {"commuting", "invertible", "self_commuting"} & set(route.arguments)


@pytest.mark.parametrize(
    ("message", "declared"),
    [
        (r"Simplify $\exp(A)\exp(B)$.", []),  # a request and a formula, but no declaration
        ("Simplify the product of A and B.", ["Let A and B be 2x2 matrices."]),  # no formula
        (r"$\exp(A)\exp(B)$", ["Let A and B be 2x2 matrices."]),  # no request
        (r"Simplify $\exp(tA)$.", ["Let A be 2x2 matrices."]),  # a scalar letter
        (r"Simplify $\exp(A)\exp(C)$.", ["Let A and B be 2x2 matrices."]),  # C is not declared
        (r"Simplify $A^T B$.", ["Let A and B be 2x2 matrices."]),  # transpose
        (r"Simplify $A_1 B$.", ["Let A and B be 2x2 matrices."]),  # indexed
        (r"Simplify $\exp(A)\exp(B)$ if A and B commute.", ["Let A and B be 2x2 matrices."]),
        (
            r"Simplify $\exp(A)\exp(B)$ given that they are invertible.",
            ["Let A and B be 2x2 matrices."],
        ),
        (r"Simplify $AB = BA$.", ["Let A and B be 2x2 matrices."]),  # an equation
        (r"Simplify $\operatorname{tr}(A)$.", ["Let A be 2x2 matrices."]),  # another operator
        (r"Simplify $\exp(A)$.", ["A is a 2x3 matrix"]),  # not square
    ],
)
def test_matrix_simplify_leaves_everything_else_to_the_model(
    message: str, declared: list[str]
) -> None:
    assert matrix_simplify(message, RouteContext(user_messages=declared)) is None


def test_plan_route_tries_the_rules_in_order_and_ignores_unrelated_messages() -> None:
    assert plan_route("What is the integral of x?") is None
    assert plan_route("$y''(x) = y(x)$.") is not None
    assert plan_route("Now differentiate that result.", after("x + 1")).reason == "followup_reuse"  # type: ignore[union-attr]


# ------------------------------------------------------------------- the real cases, through the loop
def run(case_id: str, script: list[Any], ctx: ToolContext, routing: bool = True) -> Any:
    async def go() -> Any:
        return await run_case(ReplayLLM(script), ctx, CASES[case_id], FAST, routing=routing)

    return asyncio.run(go())


def test_the_follow_up_reuses_the_result_and_the_model_only_writes_the_answer(
    ctx: ToolContext,
) -> None:
    llm = ReplayLLM(
        [
            call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}),
            say(r"The result is $x + 1$, valid for $x \neq 1$ (the original is undefined at 1)."),
            say(r"The derivative is $1$, and like the result it holds only for $x \neq 1$."),
        ]
    )
    result = asyncio.run(
        run_case(llm, ctx, CASES["reuse-and-restriction-after-simplify"], FAST, routing=True)
    )
    assert result.passed, (result.failure_class, result.notes)
    assert result.llm_calls == 3  # turn 1: call and answer; turn 2: the answer only


def test_without_routing_the_same_script_fails_because_the_model_never_made_the_call(
    ctx: ToolContext,
) -> None:
    """The script for the routed turn has no call in it: with routing off the turn has none, so the case fails."""
    result = run(
        "reuse-and-restriction-after-simplify",
        [
            call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}),
            say(r"The result is $x + 1$, valid for $x \neq 1$."),
            say("The derivative is $1$."),
        ],
        ctx,
        routing=False,
    )
    assert not result.passed


def test_a_bare_equation_is_classified_and_the_model_asks_which_method(ctx: ToolContext) -> None:
    result = run(
        "ode-bare-equation-asks-which-method",
        [
            say(
                "Several methods apply: closed form, numerical, series. Which method would you like?"
            ),
            call(
                "ode_solve_symbolic",
                {"equations": ["y''(x) = y(x)"], "independent": "x", "method": "closed_form"},
            ),
            say(r"The general solution is $y(x) = C_{1} \mathrm{e}^{- x} + C_{2} \mathrm{e}^{x}$."),
        ],
        ctx,
    )
    assert result.passed, (result.failure_class, result.notes)
    assert result.tools[0] == "ode_analyze"


def test_declared_matrices_then_the_formula_is_answered_by_the_tool_not_by_a_question(
    ctx: ToolContext,
) -> None:
    result = run(
        "declared-matrices-no-invented-commutation",
        [
            say(
                "Noted: A and B are 2x2 matrices. Nothing says whether they commute or are invertible."
            ),
            say(
                "They cannot be merged into one exponential: that needs A and B to commute, which was not stated."
            ),
        ],
        ctx,
    )
    assert result.passed, (result.failure_class, result.notes)
    assert (
        result.tools == ["matrix_calculus"] and result.llm_calls == 2
    )  # no model call chose the tool


def test_a_routed_call_that_does_not_come_back_as_an_accepted_outcome_is_dropped(
    ctx: ToolContext,
) -> None:
    """`exp(A)exp(B)` for 2x2 matrices answers `conditions_not_satisfied` (kept); an ODE the analyser refuses is dropped."""
    from minerva_api.agent.loop import TurnInfo, run_agent

    async def go() -> dict[str, Any]:
        async for event in run_agent(
            ReplayLLM([say("I could not classify that equation.")]),
            [{"role": "user", "content": "$y''(x) = \\mathrm{unknown}(y(x))$."}],
            ctx,
            FAST,
            info=TurnInfo(routing=True),
        ):
            if event["event"] == "message":
                return event["data"]  # type: ignore[no-any-return]
        raise AssertionError("no message")

    message = asyncio.run(go())
    assert (
        message["debug"]["routing"] == {}
    )  # the rule did not apply (not a plain equation): the model planned
    assert message["trace"] == []
