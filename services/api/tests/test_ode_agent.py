"""The differential-equation tools in the agent: what the model reads, what it may not choose, how the
cards and direct routes behave, and the scripted flows."""

from __future__ import annotations

import re
from typing import Any

import pytest
from fastapi.testclient import TestClient
from minerva_api.agent.prompts import SYSTEM_PROMPT
from minerva_api.agent.tools import TOOLS, ToolContext, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .conftest import build_app, register
from .test_chat import cards_of, sse

DUFFING = r"y'' + 0.2 y' + y + y^3 = 0"


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


# --- what the model sees ----------------------------------------------------------------------


def test_the_tools_are_three_and_say_who_chooses_the_numbers() -> None:
    names = [n for n in TOOLS if n.startswith("ode_")]
    assert names == ["ode_analyze", "ode_solve_symbolic", "ode_solve_numerical"]
    numeric = TOOLS["ode_solve_numerical"]
    assert "comes from the USER: ask" in numeric.description
    assert "only if the user explicitly told you to choose" in numeric.description
    schema = numeric.schema()["function"]["parameters"]
    assert "equations" in schema["required"]
    assert "Ask the" in schema["properties"]["t_end"]["description"]
    assert "Ask the user" in schema["properties"]["parameters"]["description"]
    assert (
        "never guess"
        in TOOLS["ode_analyze"].schema()["function"]["parameters"]["properties"]["independent"][
            "description"
        ]
    )


def test_the_prompt_tells_the_model_to_analyse_first_and_not_to_pick_numbers() -> None:
    # Rules are numbered when the prompt is assembled, so find this one by its text.
    rule = SYSTEM_PROMPT[SYSTEM_PROMPT.index("For an ordinary differential equation") :]
    rule = re.split(r"\n\d+\. |\n[A-Z][a-z]+ tools \(call", rule)[0]
    for fragment in (
        "`ode_analyze` first",
        "Minerva's, not yours",
        "never pick them",
        "approximation",
    ):
        assert fragment in rule


# --- ode_analyze ------------------------------------------------------------------------------


def test_analyze_gives_the_model_a_ranked_list_and_quotes_the_input(ctx: ToolContext) -> None:
    out = execute("ode_analyze", {"equations": [DUFFING], "independent": "t"}, ctx)
    assert out.status == "ok" and out.card.type == "ode_analysis"
    assert out.card.data["report"]["structure"]["order"] == 2
    text = out.for_model
    assert "1. numerical_ivp (tool ode_solve_numerical" in text
    assert "not applicable: symbolic_closed_form" in text
    assert "the USER must give every initial value" in text
    assert len(text) < 3500
    assert out.evidence is not None and out.evidence.accepted
    # The equation as read is the user's input: quoted, never marked as verified.
    assert not any("=0" in f.replace(" ", "") for f in out.known_latex)
    assert any("=0" in key for key, _ in out.quoted_latex)  # keys are normalised: no spaces
    # The solved form was confirmed at sample points, so it may be marked as coming from a tool.
    assert any("y''" in f for f in out.known_latex)


def test_an_unreadable_equation_is_an_explicit_status(ctx: ToolContext) -> None:
    out = execute("ode_analyze", {"equations": ["y' = y"]}, ctx)
    assert out.status == "parse_error" and "independent" in out.for_model
    assert out.evidence is not None and not out.evidence.accepted
    assert out.known_latex == []


def test_a_declared_matrix_is_refused(ctx: ToolContext) -> None:
    matrix_ctx = ToolContext(engine=ctx.engine, kb=ctx.kb, matrices=frozenset({"A"}))
    for name, extra in (
        ("ode_analyze", {}),
        ("ode_solve_symbolic", {}),
        ("ode_solve_numerical", {"t_end": 1.0}),
    ):
        out = execute(name, {"equations": ["x' = A x"], "independent": "t", **extra}, matrix_ctx)
        assert out.status == "error" and out.error_category == "unsupported", name
        assert "matrix" in out.for_model, name


# --- ode_solve_symbolic ------------------------------------------------------------------------


def test_a_verified_closed_form_is_marked_as_coming_from_the_tool(ctx: ToolContext) -> None:
    out = execute(
        "ode_solve_symbolic",
        {"equations": ["y' = x y"], "independent": "x", "conditions": ["y(0) = 2"]},
        ctx,
    )
    assert out.status == "ok" and out.card.type == "ode_solution"
    assert (
        out.evidence is not None and out.evidence.accepted and out.evidence.check == "symbolic_zero"
    )
    assert out.evidence.points_checked and out.evidence.points_agreed == out.evidence.points_checked
    assert any("2" in f and "e" in f for f in out.known_latex)
    assert "also proved by a symbolic zero test" in out.for_model
    assert "numerically checked" in out.for_model


def test_a_failed_search_is_not_a_claim_that_nothing_exists(ctx: ToolContext) -> None:
    out = execute("ode_solve_symbolic", {"equations": ["y' = y^2 + x"], "independent": "x"}, ctx)
    assert out.status == "unsupported" and out.known_latex == []
    assert "Nothing is claimed about whether a closed form exists" in out.for_model
    assert out.evidence is not None and not out.evidence.accepted


# --- ode_solve_numerical -----------------------------------------------------------------------

NUMERIC = {
    "equations": [r"y'' + c y' + y = 0"],
    "independent": "t",
    "conditions": ["y(0) = 1", "y'(0) = 0"],
    "parameters": {"c": 0.5},
    "t_end": 10,
}


def test_a_call_without_the_interval_is_refused_so_the_model_must_ask(ctx: ToolContext) -> None:
    arguments = {k: v for k, v in NUMERIC.items() if k != "t_end"}
    out = execute("ode_solve_numerical", arguments, ctx)
    assert out.status == "parse_error" and "t_end" in out.for_model
    assert "No trajectory is shown" in out.for_model


def test_missing_parameter_values_and_conditions_are_refused_by_the_engine(
    ctx: ToolContext,
) -> None:
    out = execute("ode_solve_numerical", {**NUMERIC, "parameters": {}}, ctx)
    assert out.status == "parse_error" and "missing: c" in out.for_model
    assert "No trajectory is shown" in out.for_model
    out = execute("ode_solve_numerical", {**NUMERIC, "conditions": []}, ctx)
    assert out.status == "parse_error" and "does not choose them" in out.for_model


def test_a_verified_trajectory_is_summarised_without_the_series(ctx: ToolContext) -> None:
    out = execute("ode_solve_numerical", NUMERIC, ctx)
    assert out.status == "ok" and out.card.type == "ode_numeric"
    report = out.card.data["report"]
    assert len(report["series"][0]["values"]) == 200
    assert len(out.for_model) < 2500 and "0.5" in out.for_model
    assert "check independent method: passed" in out.for_model
    assert "approximation on the interval given" in out.for_model
    assert out.evidence is not None and out.evidence.accepted
    assert out.card.data["request"]["parameters"] == {"c": 0.5}


def test_a_failed_run_tells_the_model_not_to_describe_a_trajectory(ctx: ToolContext) -> None:
    out = execute(
        "ode_solve_numerical",
        {"equations": ["y' = y^2"], "independent": "t", "conditions": ["y(0) = 1"], "t_end": 2},
        ctx,
    )
    assert out.status in ("not_verified", "resource_limit")
    assert "Do not describe or estimate one" in out.for_model
    assert out.evidence is not None and not out.evidence.accepted
    assert out.card.data["report"]["series"] == []


# --- the chat, with the scripted stand-in ---------------------------------------------------------


@pytest.fixture
def chat_client() -> Any:
    with TestClient(build_app(llm_mode="scripted")) as client:
        register(client)
        yield client, client.post("/v1/chat/conversations", json={}).json()["id"]


def message_of(events: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return next(d for e, d in events if e == "message")


def test_an_analysis_turn_shows_the_ranked_methods(chat_client: Any) -> None:
    client, cid = chat_client
    message = message_of(
        sse(client, cid, rf"Analyse this ODE, independent variable t: ${DUFFING}$")
    )
    (card,) = cards_of(message)
    assert card["type"] == "ode_analysis"
    ranked = [m["id"] for m in card["data"]["report"]["methods"] if m["rank"]]
    assert ranked[0] == "numerical_ivp" and "equilibrium_analysis" in ranked


def test_the_model_asks_for_an_unnamed_independent_variable(chat_client: Any) -> None:
    client, cid = chat_client
    message = message_of(sse(client, cid, r"Solve this ODE: $y' = y$"))
    assert cards_of(message) == []


def test_a_numerical_turn_without_numbers_asks_and_computes_nothing(chat_client: Any) -> None:
    client, cid = chat_client
    message = message_of(
        sse(client, cid, rf"Solve this ODE numerically, independent variable t: ${DUFFING}$")
    )
    assert cards_of(message) == []
    assert not any(b["kind"] == "card" for b in message["blocks"])


def test_a_numerical_turn_uses_only_the_numbers_in_the_message(chat_client: Any) -> None:
    client, cid = chat_client
    message = message_of(
        sse(
            client,
            cid,
            "Solve this ODE numerically, independent variable t, y(0)=1, y'(0)=0, c=0.5, t_end=10: "
            r"$y'' + c y' + y = 0$",
        )
    )
    (card,) = cards_of(message)
    assert card["type"] == "ode_numeric"
    request = card["data"]["request"]
    assert request["parameters"] == {"c": 0.5} and request["t_end"] == 10
    assert card["data"]["report"]["status"] == "ok"


def test_a_symbolic_turn_returns_a_verified_solution(chat_client: Any) -> None:
    client, cid = chat_client
    message = message_of(
        sse(
            client,
            cid,
            "Find the closed form of this ODE, independent variable x, y(0)=2: $y' = x y$",
        )
    )
    (card,) = cards_of(message)
    assert card["type"] == "ode_solution"
    assert card["data"]["report"]["solutions"][0]["verification"]["passed"] is True


def test_a_derivative_statement_in_prose_is_not_marked_checked(chat_client: Any) -> None:
    client, cid = chat_client
    events = sse(client, cid, "Just tell me without tools.")
    # (the scripted answer has a formula that is not from a tool; here only the shape matters)
    assert message_of(events)["blocks"]


# --- the card forms' direct routes ---------------------------------------------------------------


def test_the_numerical_card_form_posts_straight_to_the_tool(chat_client: Any) -> None:
    client, cid = chat_client
    r = client.post(f"/v1/chat/conversations/{cid}/tools/ode_solve_numerical", json=NUMERIC)
    assert r.status_code == 201
    content = r.json()["content"]
    (block,) = content["blocks"]
    assert (
        block["card"]["type"] == "ode_numeric" and block["card"]["data"]["report"]["status"] == "ok"
    )
    (step,) = content["trace"]
    assert step["name"] == "ode_solve_numerical" and step["arguments"]["parameters"] == {"c": 0.5}
    assert content["debug"]["mode"] == "direct"


def test_the_symbolic_card_button_posts_straight_to_the_tool(chat_client: Any) -> None:
    client, cid = chat_client
    r = client.post(
        f"/v1/chat/conversations/{cid}/tools/ode_solve_symbolic",
        json={"equations": ["y' = y"], "independent": "x", "conditions": ["y(0) = 3"]},
    )
    assert r.status_code == 201
    card = r.json()["content"]["blocks"][0]["card"]
    assert card["type"] == "ode_solution" and card["data"]["report"]["status"] == "ok"


def test_the_direct_routes_refuse_bad_bodies_and_other_peoples_conversations(
    chat_client: Any,
) -> None:
    client, cid = chat_client
    bad = {**NUMERIC, "rtol": 0}
    assert (
        client.post(f"/v1/chat/conversations/{cid}/tools/ode_solve_numerical", json=bad).status_code
        == 422
    )
    other = TestClient(client.app)
    register(other, "grace@example.com")
    r = other.post(f"/v1/chat/conversations/{cid}/tools/ode_solve_numerical", json=NUMERIC)
    assert r.status_code == 404


# --- series, perturbation and boundary-value problems ----------------------------------------------


def test_a_series_is_described_as_local_and_checked_not_proved(ctx: ToolContext) -> None:
    out = execute(
        "ode_solve_symbolic",
        {
            "equations": ["y' = y"],
            "independent": "x",
            "conditions": ["y(0) = 1"],
            "method": "series",
            "order": 5,
        },
        ctx,
    )
    assert out.status == "ok" and out.card.type == "ode_solution"
    text = out.for_model
    assert "series" in text and "local approximation" in text
    assert "falls at the claimed order" in text and "not a proof" in text
    assert "also proved" not in text
    assert out.evidence is not None and out.evidence.accepted


def test_a_perturbation_without_a_named_small_parameter_is_refused(ctx: ToolContext) -> None:
    out = execute(
        "ode_solve_symbolic",
        {
            "equations": [r"y'' + y + \epsilon y^3 = 0"],
            "independent": "t",
            "conditions": ["y(0) = 1", "y'(0) = 0"],
            "method": "perturbation",
        },
        ctx,
    )
    assert out.status == "parse_error" and "Name the small parameter" in out.for_model
    assert out.known_latex == []
    schema = TOOLS["ode_solve_symbolic"].schema()["function"]["parameters"]["properties"]
    assert "never choose" in schema["small_parameter"]["description"]


def test_secular_terms_are_reported_to_the_model(ctx: ToolContext) -> None:
    out = execute(
        "ode_solve_symbolic",
        {
            "equations": [r"y'' + y + \epsilon y^3 = 0"],
            "independent": "t",
            "conditions": ["y(0) = 1", "y'(0) = 0"],
            "method": "perturbation",
            "small_parameter": "epsilon",
            "order": 1,
        },
        ctx,
    )
    assert out.status == "ok"
    assert "valid only on bounded intervals" in out.for_model


def test_a_boundary_value_problem_goes_through_the_numerical_tool(ctx: ToolContext) -> None:
    out = execute(
        "ode_solve_numerical",
        {
            "equations": ["y'' = -y"],
            "independent": "x",
            "kind": "bvp",
            "conditions": ["y(0) = 0", "y(1) = 1"],
        },
        ctx,
    )
    assert out.status == "ok" and out.card.type == "ode_numeric"
    assert "solve_bvp" in out.for_model and "check boundary conditions: passed" in out.for_model
    assert out.card.data["report"]["method_used"] == "solve_bvp"


def test_the_scripted_chat_builds_a_series_and_solves_a_boundary_problem(chat_client: Any) -> None:
    client, cid = chat_client
    message = message_of(
        sse(
            client,
            cid,
            "Give the series to order 5 of this ODE, independent variable x, y(0)=1: $y' = y$",
        )
    )
    (card,) = cards_of(message)
    item = card["data"]["report"]["solutions"][0]
    assert card["type"] == "ode_solution" and item["kind"] == "series"
    assert item["expansion"]["order"] == 5
    message = message_of(
        sse(
            client,
            cid,
            "Solve this boundary ODE, independent variable x, y(0)=0, y(1)=1: $y'' = -y$",
        )
    )
    (card,) = cards_of(message)
    assert card["type"] == "ode_numeric" and card["data"]["request"]["kind"] == "bvp"
    assert card["data"]["report"]["status"] == "ok"


# --- equilibria, stability and comparison --------------------------------------------------------


def test_the_qualitative_analysis_is_local_and_asks_for_the_numbers(ctx: ToolContext) -> None:
    out = execute(
        "ode_analyze",
        {
            "equations": [r"y'' + 0.2 y' + y + y^3 = 0"],
            "independent": "t",
            "detail": "qualitative",
            "window": [-2, 2, -2, 2],
        },
        ctx,
    )
    assert out.status == "ok"
    text = out.for_model
    assert "linearly asymptotically stable (a sink: a spiral)" in text
    assert "Local and from the linearisation only" in text
    assert "complete list: True" in text and "a picture of f, not of a solution" in text
    assert out.card.data["report"]["qualitative"]["direction_field"] is not None
    description = TOOLS["ode_analyze"].description
    assert "LOCAL" in description and "never say 'globally stable'" in description
    schema = TOOLS["ode_analyze"].schema()["function"]["parameters"]["properties"]
    assert "never choose" in schema["parameters"]["description"]
    assert "never choose" in schema["window"]["description"]


def test_stability_is_undecided_without_parameter_values(ctx: ToolContext) -> None:
    out = execute(
        "ode_analyze",
        {
            "equations": [r"\dot{x} = a x - b x y", r"\dot{y} = -c y + d x y"],
            "detail": "qualitative",
        },
        ctx,
    )
    assert out.status == "ok" and "undecided" in out.for_model
    assert "classification_text" in out.card.data["report"]["qualitative"]["equilibria"][0]


def test_the_comparison_with_a_closed_form_reaches_the_model(ctx: ToolContext) -> None:
    out = execute(
        "ode_solve_numerical",
        {**NUMERIC, "equations": ["y'' + y = 0"], "parameters": {}, "compare_with": ["cos(t)"]},
        ctx,
    )
    assert out.status == "ok"
    assert "compared with the closed form given: largest difference" in out.for_model
    assert "not a bound on either" in out.for_model


def test_the_scripted_chat_analyses_equilibria(chat_client: Any) -> None:
    client, cid = chat_client
    message = message_of(
        sse(
            client,
            cid,
            "Find the equilibria and their stability of this ODE, independent variable t, window -2,2,-2,2: "
            "$y'' + 0.2 y' + y + y^3 = 0$",
        )
    )
    (card,) = cards_of(message)
    q = card["data"]["report"]["qualitative"]
    assert card["type"] == "ode_analysis" and q["status"] == "ok"
    assert q["equilibria"][0]["classification"] == "stable_spiral"
    assert len(q["direction_field"]["x"]) == 225


def test_the_qualitative_card_form_posts_straight_to_the_tool(chat_client: Any) -> None:
    client, cid = chat_client
    r = client.post(
        f"/v1/chat/conversations/{cid}/tools/ode_analyze",
        json={
            "equations": ["y'' + k*y = 0"],
            "format": "text",
            "independent": "t",
            "detail": "qualitative",
            "parameters": {"k": 4.0},
            "window": [-1, 1, -1, 1],
        },
    )
    assert r.status_code == 201
    card = r.json()["content"]["blocks"][0]["card"]
    assert card["type"] == "ode_analysis"
    assert card["data"]["report"]["qualitative"]["equilibria"][0]["classification"] == "centre"
    bad = client.post(
        f"/v1/chat/conversations/{cid}/tools/ode_analyze",
        json={"equations": ["y' = y"], "independent": "t", "window": [1, 0, 0, 1]},
    )
    assert bad.status_code == 422


# --- ask which method unless the user named one --------------------------------------------------------
def test_the_loop_refuses_a_solver_until_the_user_chooses(ctx: ToolContext) -> None:
    import asyncio

    from minerva_api.agent.llm import Chunk, ToolCall
    from minerva_api.agent.loop import run_agent

    class AnalyseThenSolve:
        """A model that analyses and then goes straight to the closed form, as the report showed."""

        name = provider = "test"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools, **kw):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                yield Chunk(
                    tool_calls=[
                        ToolCall(
                            "ode_analyze", {"equations": ["y''(x) = y(x)"], "independent": "x"}
                        )
                    ]
                )
            elif self.calls == 2:
                yield Chunk(
                    tool_calls=[
                        ToolCall(
                            "ode_solve_symbolic",
                            {
                                "equations": ["y''(x) = y(x)"],
                                "independent": "x",
                                "method": "closed_form",
                            },
                        )
                    ]
                )
            else:
                yield Chunk(text="Which method would you like?")

    async def turn(message: str) -> list[dict[str, Any]]:
        events = []
        history = [{"role": "user", "content": message}]
        async for event in run_agent(AnalyseThenSolve(), history, ctx):  # type: ignore[arg-type]
            events.append(event)
        return events

    def solver_cards(events: list[dict[str, Any]]) -> list[str]:
        return [
            e["data"]["type"]
            for e in events
            if e["event"] == "card" and e["data"]["tool"] == "ode_solve_symbolic"
        ]

    bare = asyncio.run(turn("y''(x) = y(x)."))
    assert solver_cards(bare) == ["error"]  # refused: nothing was solved
    refused = next(e for e in bare if e["event"] == "card" and e["data"]["type"] == "error")
    assert "list the applicable methods" in refused["data"]["data"]["message"]
    named = asyncio.run(turn("Find the closed form of y''(x) = y(x)."))
    assert solver_cards(named) == ["ode_solution"]  # the user named a method: it runs


# --- who chose the numbers ----------------------------------------------------------------------


def _chat_ctx(ctx: ToolContext, messages: list[str], *, delegated: bool = False) -> ToolContext:
    """What the loop sets for a model's own call: the numbers the user wrote, and the hand-over."""
    from minerva_api.agent.choice import numbers_in

    fresh = ToolContext(engine=ctx.engine, kb=ctx.kb)
    fresh.numbers_delegated = delegated
    fresh.user_numbers = frozenset(x for m in messages for x in numbers_in(m))
    return fresh


def test_numbers_the_user_wrote_are_labelled_as_theirs(ctx: ToolContext) -> None:
    chat = _chat_ctx(ctx, ["y(0) = 1 and y'(0) = 0, c = 0.5, run to t = 10"])
    out = execute("ode_solve_numerical", NUMERIC, chat)
    assert out.status == "ok"
    assert out.card.data["values_chosen_by"] == "user" and out.card.data["assistant_values"] == []
    assert "all supplied by the user" in out.for_model


def test_numbers_the_model_supplied_without_being_told_to_are_labelled(ctx: ToolContext) -> None:
    chat = _chat_ctx(ctx, ["solve y'' + c y' + y = 0 numerically, y(0) = 1, y'(0) = 0"])
    out = execute("ode_solve_numerical", NUMERIC, chat)  # c = 0.5 and t_end = 10 are not the user's
    assert out.card.data["values_chosen_by"] == "assistant"
    assert out.card.data["assistant_values"] == ["c = 0.5", "t_end = 10"]
    assert "values the user did not write" in out.for_model
    assert "(all supplied by the user)" not in out.for_model


def test_numbers_chosen_at_the_users_request_are_labelled_and_listed(ctx: ToolContext) -> None:
    chat = _chat_ctx(ctx, ["numerically please, just choose something reasonable"], delegated=True)
    out = execute("ode_solve_numerical", NUMERIC, chat)
    assert out.card.data["values_chosen_by"] == "assistant_delegated"
    assert "y(0) = 1" in out.card.data["assistant_values"]
    assert "because the user asked it to choose" in out.for_model
    assert "(all supplied by the user)" not in out.for_model


def test_the_card_form_and_direct_calls_are_the_users_numbers(ctx: ToolContext) -> None:
    out = execute("ode_solve_numerical", NUMERIC, ctx)  # no chat context: typed on the card
    assert out.card.data["values_chosen_by"] == "user"


def test_integrate_motion_labels_numbers_the_same_way(ctx: ToolContext) -> None:
    arguments = {
        "lagrangian": r"\frac{1}{2} m \dot{x}^2 - \frac{1}{2} k x^2",
        "coordinates": ["x"],
        "parameters": {"m": 1.0, "k": 4.0},
        "initial_positions": [1.0],
        "initial_velocities": [0.0],
        "t_end": 2.0,
    }
    chat = _chat_ctx(ctx, ["pick something reasonable"], delegated=True)
    out = execute("integrate_motion", arguments, chat)
    assert out.card.data["values_chosen_by"] == "assistant_delegated"
    assert "m = 1" in out.card.data["assistant_values"]
    own = _chat_ctx(ctx, ["m = 1, k = 4, x(0) = 1, v(0) = 0, until t = 2"])
    assert execute("integrate_motion", arguments, own).card.data["values_chosen_by"] == "user"
