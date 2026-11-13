"""Agent loop, tools and segment marking, with a deterministic fake model."""

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from minerva_api.agent.llm import Chunk, LLMError, ToolCall
from minerva_api.agent.loop import AgentLimits, TurnInfo, prompt_sha, run_agent
from minerva_api.agent.replay import ReplayLLM
from minerva_api.agent.segments import normalize, split_segments
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

FakeLLM = ReplayLLM  # the shared replay model; the old name is kept for the tests below


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def collect(
    llm: FakeLLM, ctx: ToolContext, history: list[dict[str, Any]], **kw: Any
) -> list[dict[str, Any]]:
    async def go() -> list[dict[str, Any]]:
        return [e async for e in run_agent(llm, history, ctx, **kw)]

    return asyncio.run(go())


def final(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(e["data"] for e in events if e["event"] == "message")


def cards_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [b["card"] for b in message["blocks"] if b["kind"] == "card"]


def segments_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for b in message["blocks"] if b["kind"] == "text" for s in b["segments"]]


def test_text_only_answer_has_no_cards(ctx: ToolContext) -> None:
    llm = FakeLLM([[Chunk(text="Which variables are positive?")]])
    events = collect(llm, ctx, [{"role": "user", "content": "help"}])
    msg = final(events)
    assert cards_of(msg) == [] and msg["text"] == "Which variables are positive?"
    assert [e["event"] for e in events] == ["text", "message"]


def test_tool_result_is_a_card_and_formula_is_verified(ctx: ToolContext) -> None:
    llm = FakeLLM(
        [
            [Chunk(tool_calls=[ToolCall("simplify", {"latex": r"\frac{x^2-1}{x-1}"})])],
            [Chunk(text="The result is $x + 1$.")],
        ]
    )
    events = collect(llm, ctx, [{"role": "user", "content": "simplify"}])
    msg = final(events)
    assert cards_of(msg)[0]["type"] == "simplify"
    assert cards_of(msg)[0]["data"]["status"] == "ok"
    math = [s for s in segments_of(msg) if s["type"] == "math"]
    assert math and math[0]["status"] == "from_tool"
    # the model saw the tool's compact result, not the raw card
    assert any(m["role"] == "tool" and "result=$x + 1$" in m["content"] for m in llm.seen[1])


def test_formula_not_from_a_tool_is_marked_unverified(ctx: ToolContext) -> None:
    llm = FakeLLM([[Chunk(text=r"It equals $\frac{\sqrt{\pi}}{2}$ and also $$x^2+1$$.")]])
    msg = final(collect(llm, ctx, [{"role": "user", "content": "q"}]))
    formulas = [s for s in segments_of(msg) if s["type"] != "text"]
    assert len(formulas) == 2 and all(s["status"] == "unchecked" for s in formulas)


def test_step_cap_stops_a_looping_model(ctx: ToolContext) -> None:
    # Different calls every step (an identical one would be recognised as a repeat, not run again).
    calls = [
        [Chunk(tool_calls=[ToolCall("search_handbook", {"query": q})])]
        for q in ("gaussian", "euler", "planck", "zeta")
    ]
    llm = FakeLLM(calls)
    msg = final(
        collect(llm, ctx, [{"role": "user", "content": "q"}], limits=AgentLimits(max_steps=3))
    )
    assert msg["truncated"] is True and msg["stop_reason"] == "step_limit"
    assert len(cards_of(msg)) == 3
    assert "step limit" in msg["text"]


def test_time_cap_stops_the_turn(ctx: ToolContext) -> None:
    llm = FakeLLM([[Chunk(text="x")]], repeat_last=True)
    msg = final(
        collect(llm, ctx, [{"role": "user", "content": "q"}], limits=AgentLimits(max_seconds=-1))
    )
    assert msg["truncated"] is True and llm.seen == []
    assert msg["stop_reason"] == "time_limit" and "time limit" in msg["text"]


def test_unknown_tool_and_bad_arguments_become_errors_the_model_can_read(ctx: ToolContext) -> None:
    llm = FakeLLM(
        [
            [
                Chunk(
                    tool_calls=[
                        ToolCall("rm_rf", {"path": "/"}),
                        ToolCall("simplify", {"latex": 5}),
                        ToolCall("simplify", {"latex": "x", "assumptions": {"x": ["evil"]}}),
                        ToolCall("simplify", {}),
                    ]
                )
            ],
            [Chunk(text="I could not run that.")],
        ]
    )
    msg = final(collect(llm, ctx, [{"role": "user", "content": "q"}]))
    assert [c["type"] for c in cards_of(msg)] == ["error"] * 4
    tool_msgs = [m["content"] for m in llm.seen[1] if m["role"] == "tool"]
    assert tool_msgs[0].startswith("error: unknown tool")
    assert all(t.startswith("error:") for t in tool_msgs)


def test_tool_arguments_are_bounded(ctx: ToolContext) -> None:
    out = execute("simplify", {"latex": "x" * 5000}, ctx)
    assert out.card.type == "error"
    out = execute("search_handbook", {"query": "a", "limit": 500}, ctx)
    assert out.card.type == "error"


def test_model_host_failure_is_an_error_event_without_a_message(ctx: ToolContext) -> None:
    llm = FakeLLM()
    llm.error = LLMError("The model host is not reachable.")
    events = collect(
        llm, ctx, [{"role": "user", "content": "q"}], limits=AgentLimits(retry_delay_seconds=0)
    )
    # The error is reported first; the turn is still delivered as a message so its trace can be stored.
    assert [e["event"] for e in events] == ["error", "message"]
    assert len(llm.seen) == 2  # the one retry of a call that failed outright


def test_model_host_failure_carries_debug_and_partial_trace(ctx: ToolContext) -> None:
    llm = FakeLLM()
    llm.error = LLMError("The model host is not reachable.")
    info = TurnInfo(request_id="abc", mode="ollama", api_version="9.9")
    events = collect(llm, ctx, [{"role": "user", "content": "q"}], info=info)
    data = events[0]["data"]
    assert data["message"] == "The model host is not reachable."
    assert data["trace"] == []
    debug = data["debug"]
    assert debug["error"] == data["message"]
    assert (debug["request_id"], debug["mode"], debug["model"]) == ("abc", "ollama", "replay")
    assert len(debug["prompt_sha"]) == 12


def test_turn_records_debug_timings_statuses_and_claim_checks(ctx: ToolContext) -> None:
    llm = FakeLLM(
        [
            [Chunk(tool_calls=[ToolCall("simplify", {"latex": r"\frac{x^2-1}{x-1}"})])],
            [Chunk(text="Done: $x+1 = x+1$ and $x^2 = 5x$.")],
        ]
    )
    msg = final(collect(llm, ctx, [{"role": "user", "content": "q"}]))
    step = msg["trace"][0]
    assert step["status"] == "ok" and isinstance(step["duration_ms"], int)
    debug = msg["debug"]
    assert len(debug["llm_calls"]) == 2
    assert debug["llm_calls"][0]["tool_calls"] == 1 and debug["llm_calls"][1]["text_chars"] > 0
    assert debug["max_steps"] == 6 and debug["model"] == "replay"
    by_value = {c["value"]: c["status"] for c in debug["claims"]}
    assert by_value["x^2 = 5x"] in (
        "not_identity",
        "conditional",
    )  # never "fails": may be an equation
    assert debug["prompt_sha"] == prompt_sha()


def test_tool_error_is_recorded_in_the_trace(ctx: ToolContext) -> None:
    llm = FakeLLM([[Chunk(tool_calls=[ToolCall("nope", {})])], [Chunk(text="Could not do that.")]])
    msg = final(collect(llm, ctx, [{"role": "user", "content": "q"}]))
    assert msg["trace"][0]["status"] == "error"


def test_simplify_tool_lists_the_identities_it_used_and_how_each_step_was_checked(
    ctx: ToolContext,
) -> None:
    out = execute("simplify", {"latex": r"\sin^2 x + \cos^2 x + \tan^2 x"}, ctx)
    assert out.status == "ok"
    lines = [line for line in out.for_model.splitlines() if line.startswith("- step")]
    assert len(lines) == 2
    assert "trig.pythagorean" in lines[0] and "trig.tan_secant" in lines[1]
    assert all("symbolic_zero=" in line and "numerically" in line for line in lines)
    assert out.for_model.startswith("status=ok; goal=simplify;")
    # The statements the model was given are known formulas, so quoting them is not "unchecked".
    statements = {e["statement_latex"] for e in out.card.data["entries"].values()}
    assert statements and statements <= set(out.known_latex)
    assert out.card.type == "simplify"


def test_simplify_tool_reports_a_blocked_identity_and_applies_nothing(ctx: ToolContext) -> None:
    out = execute("simplify", {"latex": r"\log x + \log y"}, ctx)
    assert out.status == "conditions_not_satisfied"
    assert "NOT APPLIED" in out.for_model and "log.product" in out.for_model
    assert "result=none" in out.for_model  # nothing to quote as a result
    assert "a positive" in out.for_model and "b positive" in out.for_model
    assert not [line for line in out.for_model.splitlines() if line.startswith("- step")]
    ok = execute(
        "simplify",
        {"latex": r"\log x + \log y", "assumptions": {"x": ["positive"], "y": ["positive"]}},
        ctx,
    )
    assert ok.status == "ok" and "log.product" in ok.for_model


def test_simplify_tool_takes_a_goal_and_the_other_tools_do_not(ctx: ToolContext) -> None:
    from minerva_api.agent.tools import TOOLS

    simplify = TOOLS["simplify"].schema()["function"]["parameters"]["properties"]
    assert set(simplify) >= {"latex", "assumptions", "goal", "target"}
    assert "goal" not in TOOLS["differentiate"].schema()["function"]["parameters"]["properties"]
    out = execute("simplify", {"latex": r"\sin(2x)", "goal": "expand"}, ctx)
    assert out.status == "ok" and "trig.double_sin" in out.for_model
    bad = execute("simplify", {"latex": "x", "goal": "integrate"}, ctx)
    assert bad.status == "error"


def test_tools_have_no_side_effects_surface(ctx: ToolContext) -> None:
    from minerva_api.agent.tools import TOOLS

    assert set(TOOLS) == {
        "search_handbook",
        "get_entry",
        "simplify",
        "differentiate",
        "series",
        "integrate",
        "matrix_calculus",
        "geometry",
        "distribution",
        "find_identities",
        "tensor_canonicalize",
        "solve_polynomial",
        "analyze_lagrangian",
        "expand_motion",
        "perturbation_solve",
        "search_invariants",
        "integrate_motion",
        "dimensional_analysis",
        "check_units",
    }
    for tool in TOOLS.values():
        assert tool.schema()["function"]["parameters"]["type"] == "object"


def test_identities_tool_reports_conditions_honestly(ctx: ToolContext) -> None:
    out = execute("find_identities", {"latex": r"\log(xy)"}, ctx)
    assert "log.product" in out.for_model and "conditions_not_satisfied" in out.for_model
    ok = execute(
        "find_identities",
        {"latex": r"\log(xy)", "assumptions": {"x": ["positive"], "y": ["positive"]}},
        ctx,
    )
    assert "applicable" in ok.for_model


def test_identities_tool_finds_a_generated_identity(ctx: ToolContext) -> None:
    out = execute("find_identities", {"latex": r"\sin(3x)"}, ctx)
    status = {
        line.split(" ", 2)[1]: line.split("): ", 1)[1].split(";", 1)[0]
        for line in out.for_model.splitlines()
        if line.startswith("- ") and "): " in line
    }
    assert status["trig.triple_sin"] == "applicable"
    assert status["trig.double_sin"] == "no_match"  # would need a = 3x/2


def test_segments_parse_all_math_delimiters() -> None:
    known = {normalize("x^{2}")}
    segs = split_segments(r"A $x^2$, B $$y$$, C \(z\), D \[w\], cost \$5.", known)
    kinds = [(s.type, s.value) for s in segs if s.type != "text"]
    assert kinds == [("math", "x^2"), ("display_math", "y"), ("math", "z"), ("display_math", "w")]
    assert segs[1].status == "from_tool"  # x^2 equals x^{2} after normalisation
    assert all(s.status == "unchecked" for s in segs if s.type != "text" and s.value != "x^2")
    assert "".join(s.value for s in segs if s.type == "text").endswith(r"cost \$5.")


def test_normalize_ignores_spacing_and_size_commands() -> None:
    assert normalize(r"\left( a + b \right)^{2}") == normalize("(a+b)^2")


def test_prose_and_cards_keep_the_order_they_happened_in(ctx: ToolContext) -> None:
    llm = FakeLLM(
        [
            [
                Chunk(text="Let me check. "),
                Chunk(tool_calls=[ToolCall("simplify", {"latex": "x+x"})]),
            ],
            [Chunk(text="Done: $2x$.")],
        ]
    )
    msg = final(collect(llm, ctx, [{"role": "user", "content": "q"}]))
    assert [b["kind"] for b in msg["blocks"]] == ["text", "card", "text"]


def test_equations_the_model_writes_are_checked_or_contradicted(ctx: ToolContext) -> None:
    text = (
        r"True: $\frac{x^2-1}{x-1} = x+1$. False: $x^2+2x+1 = x^2+2x$. May be an equation: $(x+1)^2 = x^2 + 1$. "
        r"Not an equation: $x^2$. Unreadable: $\foo = 1$."
    )
    msg = final(collect(FakeLLM([[Chunk(text=text)]]), ctx, [{"role": "user", "content": "q"}]))
    status = {s["value"]: s["status"] for s in segments_of(msg) if s["type"] != "text"}
    assert status[r"\frac{x^2-1}{x-1} = x+1"] == "conditional"  # x = 1 is excluded on one side
    assert status[r"x^2+2x+1 = x^2+2x"] == "contradicted"  # false for every x
    assert status["(x+1)^2 = x^2 + 1"] == "unchecked"  # may be an equation to solve
    assert status["x^2"] == "unchecked"
    assert status[r"\foo = 1"] == "unchecked"


def test_claims_inherit_the_assumptions_the_model_declared(ctx: ToolContext) -> None:
    llm = FakeLLM(
        [
            [
                Chunk(
                    tool_calls=[
                        ToolCall("simplify", {"latex": "x", "assumptions": {"x": ["positive"]}})
                    ]
                )
            ],
            [Chunk(text=r"For positive x, $\sqrt{x^2} = x$.")],
        ]
    )
    msg = final(collect(llm, ctx, [{"role": "user", "content": "q"}]))
    claim = next(s for s in segments_of(msg) if s["type"] == "math")
    assert claim["status"] == "checked"
    # without the declaration it is false for complex x but true for positive x: marked conditional
    llm2 = FakeLLM([[Chunk(text=r"Always $\sqrt{x^2} = x$.")]])
    msg2 = final(collect(llm2, ctx, [{"role": "user", "content": "q"}]))
    claim2 = next(s for s in segments_of(msg2) if s["type"] == "math")
    assert claim2["status"] == "conditional" and "positive" in claim2["note"]


def test_solve_tool_returns_a_report_card_and_retrieves_methods(ctx: ToolContext) -> None:
    out = execute("solve_polynomial", {"equation": "x^5-x-1=0"}, ctx)
    assert out.card.type == "solve"
    report = out.card.data["report"]
    assert report["status"] == "ok" and len(report["roots"]) == 5
    ids = [m["id"] for m in out.card.data["methods"]]
    assert "method.poly.abel_ruffini" in ids and "method.poly.newton" in ids
    assert "radicals=no" in out.for_model and "method.poly.galois_criterion" in out.for_model
    quad = execute("solve_polynomial", {"equation": "ax^2+bx+c=0", "variable": "x"}, ctx)
    assert "method.poly.quadratic_formula" in [m["id"] for m in quad.card.data["methods"]]
    bad = execute("solve_polynomial", {"equation": r"\sin x = x"}, ctx)
    assert bad.card.data["report"]["status"] == "unsupported"
    assert "method.poly.newton" in [m["id"] for m in bad.card.data["methods"]]


def test_lagrangian_tool_returns_equations_constants_and_methods(ctx: ToolContext) -> None:
    out = execute(
        "analyze_lagrangian",
        {
            "lagrangian": r"\frac12 m(\dot{r}^2 + r^2\dot{\theta}^2) + \frac{k}{r}",
            "coordinates": ["r", "theta"],
        },
        ctx,
    )
    assert out.card.type == "lagrangian"
    report = out.card.data["report"]
    assert report["status"] == "ok" and len(report["equations"]) == 2
    assert {c["kind"] for c in report["conserved"]} == {"cyclic", "energy"}
    ids = [m["id"] for m in out.card.data["methods"]]
    assert "method.mech.euler_lagrange" in ids and "method.mech.cyclic_coordinate" in ids
    assert "conserved (cyclic, verified on the equations of motion)" in out.for_model
    bad = execute("analyze_lagrangian", {"lagrangian": "x", "coordinates": []}, ctx)
    assert bad.card.type == "error"


def test_lagrangian_tool_offers_the_numerical_solution_and_echoes_the_request(
    ctx: ToolContext,
) -> None:
    out = execute(
        "analyze_lagrangian",
        {"lagrangian": r"\frac12 m\dot{x}^2 - \frac12 k x^2", "coordinates": ["x"]},
        ctx,
    )
    assert out.card.data["request"]["coordinates"] == ["x"]
    assert "next step: integrate_motion" in out.for_model and "do not choose them" in out.for_model
    report = out.card.data["report"]
    assert all(e["verification"]["passed"] for e in report["equations"])


KEPLER_ARGS = {
    "lagrangian": r"\frac12 m(\dot{r}^2 + r^2\dot{\theta}^2) + \frac{k}{r}",
    "coordinates": ["r", "theta"],
    "angles": ["theta"],
}


def test_invariants_tool_finds_the_runge_lenz_vector_and_says_what_it_searched(
    ctx: ToolContext,
) -> None:
    out = execute("search_invariants", KEPLER_ARGS, ctx)
    assert out.card.type == "invariants" and out.status == "ok"
    report = out.card.data["report"]
    assert len(report["found"]) == 2 and all(c["verified"] for c in report["found"])
    assert "new constant of motion (verified on the equations of motion)" in out.for_model
    assert "searched: " in out.for_model and "proves nothing" in out.for_model
    assert "Energy function" in out.for_model  # already known, factored out
    # what the tool found counts as present in the conversation, so the model may quote it
    assert all(c["expression_latex"] in out.known_latex for c in report["found"])
    assert "not something this result proves" in out.for_model


def test_invariants_tool_refuses_what_it_cannot_search(ctx: ToolContext) -> None:
    timed = execute(
        "search_invariants", {"lagrangian": r"\frac12\dot{x}^2 - t x^2", "coordinates": ["x"]}, ctx
    )
    assert timed.status == "unsupported" and not timed.card.data["report"]["found"]
    too_many = execute("search_invariants", {**KEPLER_ARGS, "coordinates": list("abcd")}, ctx)
    assert too_many.card.type == "error"
    bad_degree = execute("search_invariants", {**KEPLER_ARGS, "degree": 3}, ctx)
    assert bad_degree.card.type == "error"


OSC_ARGS = {
    "lagrangian": r"\frac12 m\dot{x}^2 - \frac12 k x^2",
    "coordinates": ["x"],
    "parameters": {"m": 1.0, "k": 4.0},
    "initial_positions": [1.0],
    "initial_velocities": [0.0],
    "t_end": 5.0,
}


def test_motion_tool_returns_a_card_and_a_compact_summary(ctx: ToolContext) -> None:
    out = execute("integrate_motion", OSC_ARGS, ctx)
    assert out.card.type == "motion" and out.status == "ok"
    report = out.card.data["report"]
    assert report["runs"][0]["status"] == "ok" and len(report["runs"][0]["q"][0]) > 20
    assert out.card.data["request"]["parameters"] == {"m": 1.0, "k": 4.0}
    assert "values as read: m=1, k=4" in out.for_model and "verified:" in out.for_model
    assert len(out.for_model) < 1500  # never the series
    assert any("k" in k for k in out.known_latex)


def test_motion_tool_sweep_and_failures_are_reported(ctx: ToolContext) -> None:
    swept = {**OSC_ARGS, "parameters": {"m": 1.0}, "sweep": {"name": "k", "values": [1, 4]}}
    out = execute("integrate_motion", swept, ctx)
    assert out.for_model.count("- k=") == 2
    missing = execute("integrate_motion", {**OSC_ARGS, "parameters": {"m": 1.0}}, ctx)
    assert missing.status == "parse_error" and "k" in missing.for_model
    bad = execute("integrate_motion", {**OSC_ARGS, "initial_positions": []}, ctx)
    assert bad.card.type == "error"
    nan = execute("integrate_motion", {**OSC_ARGS, "t_end": float("inf")}, ctx)
    assert nan.card.type == "error"


def test_differentiate_tool_returns_a_verified_card(ctx: ToolContext) -> None:
    out = execute("differentiate", {"latex": "x^5", "variables": [{"name": "x", "order": 2}]}, ctx)
    assert out.card.type == "differentiate"
    assert out.card.data["status"] == "ok"
    assert out.card.data["result_text"] == "20*x**3"
    assert "derivative=$20 x^{3}$" in out.for_model
    assert "numerically checked: 2 step(s)" in out.for_model
    assert "20 x^{3}" in out.known_latex and "5 x^{4}" in out.known_latex

    mixed = execute(
        "differentiate", {"latex": "x^2 y^3", "variables": [{"name": "x"}, {"name": "y"}]}, ctx
    )
    assert mixed.card.data["result_text"] == "6*x*y**2"


def test_differentiate_tool_is_honest_about_what_it_cannot_do(ctx: ToolContext) -> None:
    out = execute("differentiate", {"latex": r"\left|x\right|", "variables": [{"name": "x"}]}, ctx)
    assert out.card.type == "differentiate"
    assert out.card.data["status"] == "unsupported"
    assert out.card.data["result_latex"] is None
    assert "derivative=" not in out.for_model and "real" in out.for_model
    real = execute(
        "differentiate",
        {
            "latex": r"\left|x\right|",
            "variables": [{"name": "x"}],
            "assumptions": {"x": ["real"]},
        },
        ctx,
    )
    assert real.card.data["result_text"] == "sign(x)"


PENDULUM_LATEX = r"\frac12 m l^2 \dot{\theta}^2 + m g l \cos\theta"
POSITIVE_MGL = {"m": ["positive"], "g": ["positive"], "l": ["positive"]}


def test_expand_motion_tool_returns_a_verified_card(ctx: ToolContext) -> None:
    out = execute(
        "expand_motion",
        {
            "lagrangian": PENDULUM_LATEX,
            "coordinates": ["theta"],
            "assumptions": POSITIVE_MGL,
            "order": 3,
        },
        ctx,
    )
    assert out.card.type == "expansion"
    report = out.card.data["report"]
    assert report["status"] == "ok"
    assert report["equations"][0]["verification"]["passed"] is True
    assert "expansion (theta, verified)" in out.for_model
    assert "omega^2 provably positive" in out.for_model
    assert "local" in out.for_model  # the model is told the result is local
    assert report["equations"][0]["full_latex"] in out.known_latex
    assert out.card.data["request"]["coordinates"] == ["theta"]


def test_expand_motion_tool_refuses_a_non_equilibrium_and_claims_no_stability(
    ctx: ToolContext,
) -> None:
    refused = execute(
        "expand_motion",
        {
            "lagrangian": PENDULUM_LATEX,
            "coordinates": ["theta"],
            "assumptions": POSITIVE_MGL,
            "equilibrium": {"theta": "1"},
        },
        ctx,
    )
    assert refused.card.data["report"]["status"] == "unsupported"
    assert "expansion (" not in refused.for_model
    top = execute(
        "expand_motion",
        {
            "lagrangian": PENDULUM_LATEX,
            "coordinates": ["theta"],
            "assumptions": POSITIVE_MGL,
            "equilibrium": {"theta": "pi"},
        },
        ctx,
    )
    assert "provably negative (linearised equations only)" in top.for_model
    assert "omega = " not in top.for_model  # no frequency for a negative omega^2


@pytest.mark.parametrize(
    "arguments",
    [
        {"coordinates": ["x"]},
        {"lagrangian": "x", "coordinates": []},
        {"lagrangian": "x", "coordinates": ["x"], "order": 0},
        {"lagrangian": "x", "coordinates": ["x"], "order": 13},
        {"lagrangian": "x" * 3000, "coordinates": ["x"]},
    ],
)
def test_expand_motion_arguments_are_bounded(ctx: ToolContext, arguments: dict[str, Any]) -> None:
    assert execute("expand_motion", arguments, ctx).card.type == "error"


DUFFING_LATEX = r"\frac12 \dot{x}^2 - \frac12 x^2 - \frac14 b x^4"


def test_perturbation_tool_returns_a_verified_card(ctx: ToolContext) -> None:
    out = execute(
        "perturbation_solve",
        {
            "lagrangian": DUFFING_LATEX,
            "coordinates": ["x"],
            "assumptions": {"b": ["positive"]},
            "small_parameter": "b",
            "order": 2,
        },
        ctx,
    )
    assert out.card.type == "perturbation"
    report = out.card.data["report"]
    assert report["status"] == "ok" and report["verification"]["passed"] is True
    assert "residual falls at the claimed order=True" in out.for_model
    assert "never choose them yourself" in out.for_model  # the model is told to ask for numbers
    assert report["frequency_latex"] in out.known_latex
    assert report["period_ratio_latex"] in out.known_latex


def test_perturbation_tool_compares_only_with_numbers_it_was_given(ctx: ToolContext) -> None:
    out = execute(
        "perturbation_solve",
        {
            "lagrangian": DUFFING_LATEX,
            "coordinates": ["x"],
            "assumptions": {"b": ["positive"]},
            "small_parameter": "b",
            "order": 2,
            "compare": {"parameters": {}, "epsilon": 0.1, "amplitude": 1.0, "t_end": 20.0},
        },
        ctx,
    )
    comparison = out.card.data["report"]["comparison"]
    assert comparison["status"] == "ok" and comparison["observed_order"] is not None
    assert "a measurement on this span only" in out.for_model
    assert "never choose them yourself" not in out.for_model  # already compared


def test_perturbation_tool_shows_no_formula_when_it_refuses(ctx: ToolContext) -> None:
    out = execute(
        "perturbation_solve",
        {
            "lagrangian": r"\frac12 \dot{x}^2 - \frac12 x^2 - s(1-\cos x)",
            "coordinates": ["x"],
            "assumptions": {"s": ["positive"]},
            "small_parameter": "s",
        },
        ctx,
    )
    assert out.card.data["report"]["status"] == "unsupported"
    assert out.card.data["report"]["solution_latex"] is None
    assert "frequency:" not in out.for_model and "solution:" not in out.for_model


@pytest.mark.parametrize(
    "arguments",
    [
        {"coordinates": ["x"]},
        {"lagrangian": "x", "coordinates": ["x"], "order": 0},
        {"lagrangian": "x", "coordinates": ["x"], "order": 7},
        {"lagrangian": "x", "coordinates": ["x"], "compare": {"parameters": {}, "epsilon": 0.1}},
        {
            "lagrangian": "x",
            "coordinates": ["x"],
            "compare": {"parameters": {}, "epsilon": -1, "t_end": 1},
        },
        {
            "lagrangian": "x",
            "coordinates": ["x"],
            "compare": {"parameters": {}, "epsilon": 1, "t_end": 1, "extra": 1},
        },
    ],
)
def test_perturbation_arguments_are_bounded(ctx: ToolContext, arguments: dict[str, Any]) -> None:
    assert execute("perturbation_solve", arguments, ctx).card.type == "error"


def test_series_tool_returns_a_verified_card(ctx: ToolContext) -> None:
    out = execute("series", {"latex": r"\sin x", "variable": "x", "order": 5}, ctx)
    assert out.card.type == "series"
    assert out.card.data["status"] == "ok"
    assert out.card.data["result_text"] == "x**5/120 - x**3/6 + x"
    assert "coefficients (Cauchy integral) agree" in out.for_model
    assert "remainder order confirmed" in out.for_model
    assert "Measured truncation error (not a bound)" in out.for_model
    # Both the polynomial and each coefficient are known, so prose quoting them is "checked".
    assert any("x^{5}" in k for k in out.known_latex)
    assert "\\frac{1}{120}" in out.known_latex


def test_series_tool_is_honest_about_what_it_cannot_do(ctx: ToolContext) -> None:
    pole = execute("series", {"latex": r"\frac{1}{x}", "variable": "x"}, ctx)
    assert pole.card.type == "series"
    assert pole.card.data["status"] == "unsupported"
    assert pole.card.data["result_latex"] is None
    assert "series=" not in pole.for_model
    # Nothing was verified, so nothing is vouched for; the input is only quoted.
    assert pole.known_latex == []
    assert [key for key, _ in pole.quoted_latex] == [normalize(pole.card.data["input_latex"])]
    modulus = execute("series", {"latex": r"\left|x\right|", "variable": "x", "point": "1"}, ctx)
    assert modulus.card.data["status"] == "unsupported"


@pytest.mark.parametrize(
    "arguments",
    [
        {"latex": "x"},
        {"latex": "x", "variable": "x", "order": 0},
        {"latex": "x", "variable": "x", "order": 13},
        {"latex": "x", "variable": ""},
        {"latex": "x" * 3000, "variable": "x"},
    ],
)
def test_series_arguments_are_bounded(ctx: ToolContext, arguments: dict[str, Any]) -> None:
    assert execute("series", arguments, ctx).card.type == "error"


@pytest.mark.parametrize(
    "arguments",
    [
        {"latex": "x", "variables": []},
        {"latex": "x", "variables": [{"name": "x", "order": 11}]},
        {"latex": "x", "variables": [{"name": "x", "order": 0}]},
        {"latex": "x"},
        {"latex": "x" * 3000, "variables": [{"name": "x"}]},
    ],
)
def test_differentiate_arguments_are_bounded(ctx: ToolContext, arguments: dict[str, Any]) -> None:
    out = execute("differentiate", arguments, ctx)
    assert out.card.type == "error"


def test_derivative_in_model_prose_is_checked_against_the_tool(ctx: ToolContext) -> None:
    llm = FakeLLM(
        [
            [
                Chunk(
                    tool_calls=[
                        ToolCall("differentiate", {"latex": "x^5", "variables": [{"name": "x"}]})
                    ]
                )
            ],
            [Chunk(text=r"The derivative is $5 x^{4}$, not $6 x^{4}$.")],
        ]
    )
    msg = final(collect(llm, ctx, [{"role": "user", "content": "differentiate"}]))
    assert cards_of(msg)[0]["type"] == "differentiate"
    math = [s for s in segments_of(msg) if s["type"] == "math"]
    assert math[0]["status"] == "from_tool"
    assert math[1]["status"] == "unchecked"  # a formula the tool never produced


def test_bare_equations_become_formula_segments() -> None:
    from minerva_api.agent.segments import split_segments

    segs = split_segments("So x^3 + x^2 + x + 1 = 0 has the root x = -1. Then y = 2.", set())
    math = [s.value for s in segs if s.type == "math"]
    assert math == ["x^3 + x^2 + x + 1 = 0", "x = -1", "y = 2"]
    assert all(s.status == "unchecked" for s in segs if s.type == "math")
    # prose with an equals sign is left alone; markdown emphasis stays outside
    prose = split_segments("The value is equal to the number of items = many words here.", set())
    assert [s.type for s in prose] == ["text"]
    bold = split_segments("**x = 2** is the answer", set())
    assert [s.value for s in bold if s.type == "math"] == ["x = 2"]


def test_dimensional_tool_finds_groups_and_marks_them_known(ctx: ToolContext) -> None:
    out = execute(
        "dimensional_analysis",
        {
            "quantities": [
                {"name": "T", "kind": "period"},
                {"name": "L", "kind": "length"},
                {"name": "g", "kind": "acceleration"},
                {"name": "m", "unit": "kg"},
            ],
            "unknown": "T",
        },
        ctx,
    )
    assert out.card.type == "dimensional" and out.status == "ok"
    report = out.card.data["report"]
    assert report["closure"] == "determined" and report["cannot_enter"] == ["m"]
    assert r"\Pi_{1}" in out.for_model and "closure=determined" in out.for_model
    assert r"T = C\,\sqrt{\frac{L}{g}}" in out.for_model
    assert r"T = C\,\sqrt{\frac{L}{g}}" in out.known_latex
    ids = [m["id"] for m in out.card.data["methods"]]
    assert "method.phys.buckingham_pi" in ids


def test_dimensional_tool_reports_a_missing_quantity_with_handbook_constants(
    ctx: ToolContext,
) -> None:
    out = execute(
        "dimensional_analysis",
        {
            "quantities": [{"name": "m", "unit": "kg"}, {"name": "E", "unit": "J"}],
            "unknown": "E",
        },
        ctx,
    )
    report = out.card.data["report"]
    assert report["closure"] == "missing" and report["missing"]["verified"]
    assert [s["ids"] for s in report["missing"]["suggestions"]] == [["const.c"]]
    assert "MISSING: a quantity with dimension [m s^-1]" in out.for_model
    assert "adding const.c closes it" in out.for_model
    assert "method.phys.dimensional_homogeneity" in [m["id"] for m in out.card.data["methods"]]


def test_dimensional_tool_errors_are_explicit(ctx: ToolContext) -> None:
    bad_unit = execute("dimensional_analysis", {"quantities": [{"name": "x", "unit": "cm"}]}, ctx)
    assert bad_unit.status == "parse_error" and "coherent SI" in bad_unit.for_model
    both = execute(
        "dimensional_analysis", {"quantities": [{"name": "x", "unit": "m", "kind": "length"}]}, ctx
    )
    assert both.card.type == "error" and "exactly one" in both.for_model
    assert execute("dimensional_analysis", {"quantities": []}, ctx).card.type == "error"


def test_units_tool_reports_missing_factor(ctx: ToolContext) -> None:
    out = execute(
        "check_units",
        {
            "equation": "W = m c",
            "quantities": [
                {"name": "W", "unit": "J"},
                {"name": "m", "unit": "kg"},
                {"name": "c", "unit": "m/s"},
            ],
        },
        ctx,
    )
    assert out.card.type == "units" and out.status == "ok"
    assert (
        "consistent=False" in out.for_model
        and "missing factor dimension: [m s^-1]" in out.for_model
    )
    good = execute(
        "check_units",
        {
            "equation": "F = m a",
            "quantities": [
                {"name": "F", "unit": "N"},
                {"name": "m", "unit": "kg"},
                {"name": "a", "unit": "m/s^2"},
            ],
        },
        ctx,
    )
    assert "consistent=True" in good.for_model


def test_integrate_tool_returns_a_verified_card(ctx: ToolContext) -> None:
    out = execute("integrate", {"latex": r"x\cos x", "variable": "x"}, ctx)
    assert out.card.type == "integrate"
    assert out.card.data["status"] == "ok"
    assert out.status == "ok"
    assert "antiderivative=$" in out.for_model
    assert "numerically checked: numeric derivative" in out.for_model
    assert out.card.data["result_latex"] in out.known_latex


def test_integrate_tool_is_honest_when_there_is_no_closed_form(ctx: ToolContext) -> None:
    out = execute("integrate", {"latex": "e^{x^x}", "variable": "x"}, ctx)
    assert out.card.type == "integrate"
    assert out.card.data["status"] == "unsupported"
    assert out.card.data["result_latex"] is None
    assert "antiderivative=" not in out.for_model


def test_integrate_tool_definite_integral_and_conditions(ctx: ToolContext) -> None:
    out = execute(
        "integrate", {"latex": r"\sin x", "variable": "x", "lower": "0", "upper": r"\pi"}, ctx
    )
    assert out.status == "ok"
    assert out.card.data["definite"] is True
    assert "value=$2$" in out.for_model and "numerically checked: quadrature" in out.for_model
    assert out.card.data["quadrature"]["passed"] is True

    blocked = execute(
        "integrate",
        {"latex": r"\exponentialE^{-a x}", "variable": "x", "lower": "0", "upper": "oo"},
        ctx,
    )
    assert blocked.status == "conditions_not_satisfied"
    assert "value=" not in blocked.for_model and "condition: $" in blocked.for_model
    assert blocked.card.data["result_latex"] is None
    ok = execute(
        "integrate",
        {
            "latex": r"\exponentialE^{-a x}",
            "variable": "x",
            "lower": "0",
            "upper": "oo",
            "assumptions": {"a": ["positive"]},
        },
        ctx,
    )
    assert ok.status == "ok"


REGRESSION_MATRICES = [
    {"name": "A", "entries": [["1", "0"], ["0", "2"]]},
    {"name": "B", "entries": [["0", "1"], ["0", "0"]]},
]


def test_matrix_calculus_tool_reports_an_evaluated_integral(ctx: ToolContext) -> None:
    out = execute(
        "matrix_calculus",
        {
            "latex": r"\exp(wA)\exp(-wB)",
            "matrices": REGRESSION_MATRICES,
            "action": "integrate",
            "variable": "w",
            "lower": "0",
            "upper": "1",
        },
        ctx,
    )
    assert out.card.type == "matrix" and out.status == "ok"
    result = out.card.data["result"]
    assert result["result_kind"] == "explicit" and result["evaluated"] is True
    assert result["result_entries"][0][0] in ("-1 + E", "E - 1")
    assert "evaluated=yes" in out.for_model and "verified: numeric" in out.for_model
    assert f"result=${result['result_latex']}$" in out.for_model
    assert result["result_latex"] in out.known_latex
    assert result["equation_latex"] in out.known_latex
    assert any(m["id"].startswith("method.matrix.") for m in out.card.data["methods"])


def test_matrix_calculus_tool_never_calls_a_reduction_an_evaluation(ctx: ToolContext) -> None:
    out = execute(
        "matrix_calculus",
        {
            "latex": r"\exp(tP)\exp(tQ)",
            "matrices": [{"name": "P", "rows": 2}, {"name": "Q", "rows": 2}],
            "action": "integrate",
            "variable": "t",
            "lower": "0",
            "upper": "1",
        },
        ctx,
    )
    assert out.status == "ok"
    assert "evaluated=NO" in out.for_model and "evaluated=yes" not in out.for_model
    assert out.card.data["result"]["evaluated"] is False


def test_matrix_calculus_tool_reports_what_was_not_applied(ctx: ToolContext) -> None:
    out = execute(
        "matrix_calculus",
        {
            "latex": r"\exp(A)\exp(B)",
            "matrices": [{"name": "A", "rows": 2}, {"name": "B", "rows": 2}],
            "action": "simplify",
        },
        ctx,
    )
    assert out.status == "conditions_not_satisfied"
    assert "NOT APPLIED" in out.for_model and "result=" not in out.for_model
    declared = execute(
        "matrix_calculus",
        {
            "latex": r"\exp(A)\exp(B)",
            "matrices": [{"name": "A", "rows": 2}, {"name": "B", "rows": 2}],
            "commuting": [["A", "B"]],
        },
        ctx,
    )
    assert declared.status == "ok" and "AB = BA (declared)" in declared.for_model


def test_matrix_calculus_tool_rejects_bad_arguments_without_raising(ctx: ToolContext) -> None:
    out = execute("matrix_calculus", {"latex": "A", "matrices": []}, ctx)
    assert out.card.type == "error" and out.status == "error"
    out = execute("matrix_calculus", {"latex": "A", "matrices": [{"name": "A"}]}, ctx)
    assert out.card.type == "error"


# --- bounds and failures ------------------------------------------------------------------------
def test_total_tool_calls_are_capped_whatever_the_steps(ctx: ToolContext) -> None:
    def batch(*queries: str) -> list[Chunk]:
        return [Chunk(tool_calls=[ToolCall("search_handbook", {"query": q}) for q in queries])]

    llm = FakeLLM([batch("gaussian", "euler", "planck"), batch("zeta", "bessel", "gamma")])
    limits = AgentLimits(max_steps=6, max_tool_calls=4)
    msg = final(collect(llm, ctx, [{"role": "user", "content": "q"}], limits=limits))
    assert msg["truncated"] is True and "limit" in msg["text"]
    assert msg["stop_reason"] == "tool_call_limit"
    assert len(cards_of(msg)) == 4
    assert len(llm.seen) == 2  # the cap ends the turn: no further model call
    refused = [m["content"] for m in llm.seen[-1] if m["role"] == "tool"][4:]
    assert all("not run" in r for r in refused)
    skipped = [t for t in msg["trace"] if t["status"] == "skipped"]
    assert len(skipped) == 2 and all(t["error_category"] == "limit_reached" for t in skipped)


def test_a_tool_that_raises_becomes_an_error_card_not_a_broken_stream(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import minerva_api.agent.loop as loop_module

    def boom(*_: Any) -> Any:
        raise RuntimeError("secret expression x**2")

    monkeypatch.setattr(loop_module, "execute", boom)
    llm = FakeLLM(
        [[Chunk(tool_calls=[ToolCall("simplify", {"latex": "x"})])], [Chunk(text="Sorry.")]]
    )
    events = collect(llm, ctx, [{"role": "user", "content": "q"}])
    msg = final(events)
    assert [c["type"] for c in cards_of(msg)] == ["error"]
    assert "secret" not in str(msg)  # the cause is never shown or kept


def test_execute_hides_the_cause_of_an_unexpected_tool_failure(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minerva_api.agent import tools as tools_module

    def boom(*_: Any) -> Any:
        raise RuntimeError("secret expression")

    broken = dataclasses.replace(tools_module.TOOLS["simplify"], run=boom)
    monkeypatch.setitem(tools_module.TOOLS, "simplify", broken)
    out = execute("simplify", {"latex": "x"}, ctx)
    assert out.card.type == "error" and "secret" not in out.for_model


def test_a_resource_limit_in_a_tool_is_reported_to_the_model(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minerva_engine import SimplifyResult, Status, TransformResult

    monkeypatch.setattr(
        ctx.engine,
        "transform",
        lambda *_a, **_k: TransformResult(status=Status.RESOURCE_LIMIT, message="took too long"),
    )
    monkeypatch.setattr(
        ctx.engine,
        "simplify",
        lambda _e: SimplifyResult(status=Status.RESOURCE_LIMIT, message="took too long"),
    )
    llm = FakeLLM(
        [[Chunk(tool_calls=[ToolCall("simplify", {"latex": "x"})])], [Chunk(text="It timed out.")]]
    )
    msg = final(collect(llm, ctx, [{"role": "user", "content": "q"}]))
    assert msg["trace"][0]["status"] == "resource_limit"
    assert "resource_limit" in next(m["content"] for m in llm.seen[1] if m["role"] == "tool")


class SlowLLM(FakeLLM):
    """Streams a first chunk and then stalls, like a model host that keeps the connection open."""

    async def chat(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AsyncIterator[Chunk]:
        yield Chunk(text="Let me think. ")
        await asyncio.sleep(30)
        yield Chunk(text="never reached")


def test_a_stalled_model_is_cut_off_at_the_deadline_and_keeps_the_partial_answer(
    ctx: ToolContext,
) -> None:
    started = asyncio.new_event_loop().time()
    msg = final(
        collect(
            SlowLLM(), ctx, [{"role": "user", "content": "q"}], limits=AgentLimits(max_seconds=0.3)
        )
    )
    assert msg["truncated"] is True
    assert "Let me think." in msg["text"] and "never reached" not in msg["text"]
    assert msg["debug"]["duration_ms"] < 10_000
    assert started >= 0


def test_a_slow_tool_does_not_hold_the_turn_past_its_deadline(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    import minerva_api.agent.loop as loop_module

    monkeypatch.setattr(loop_module, "execute", lambda *_: time.sleep(2.5))
    llm = FakeLLM([[Chunk(tool_calls=[ToolCall("simplify", {"latex": "x"})])]], repeat_last=True)
    msg = final(
        collect(llm, ctx, [{"role": "user", "content": "q"}], limits=AgentLimits(max_seconds=0.2))
    )
    assert msg["truncated"] is True
    assert [c["type"] for c in cards_of(msg)] == ["error"]


def test_an_unexpected_model_client_error_is_an_error_event(ctx: ToolContext) -> None:
    class Broken(FakeLLM):
        async def chat(
            self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
        ) -> AsyncIterator[Chunk]:
            raise ValueError("secret prompt text")
            yield Chunk()

    events = collect(Broken(), ctx, [{"role": "user", "content": "q"}])
    assert [e["event"] for e in events] == ["error", "message"]
    assert "secret" not in str(events)


def test_a_hallucinated_formula_is_never_marked_as_checked(ctx: ToolContext) -> None:
    text = r"Known result: $\int_0^1 x\,dx = \frac{1}{3}$."  # false, and no tool produced it
    msg = final(collect(FakeLLM([[Chunk(text=text)]]), ctx, [{"role": "user", "content": "q"}]))
    claim = next(s for s in segments_of(msg) if s["type"] != "text")
    assert claim["status"] in ("unchecked", "contradicted")


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (r"a\;b", "ab"),
        (r"a\quad b\qquad c", "abc"),
        (r"a\:b\>c\!d", "abcd"),
        (r"\pi\,r^2", r"\pi r^2"),
        (r"\exponentialE^{x}", r"\mathrm{e}^{x}"),
        (r"\imaginaryI x", r"\mathrm{i}x"),
        ("{{x}}", "x"),
        (r"\int_0^\infty \mathrm{e}^{-a x}\,dx", r"\int_0^\infty\mathrm{e}^{-ax}\;dx"),
        (r"\left( a \right) \quad b", "(a)b"),
    ],
)
def test_normalize_treats_these_as_the_same_formula(a: str, b: str) -> None:
    assert normalize(a) == normalize(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("a;b", "ab"),  # a literal semicolon is content
        (r"\leftarrow", r"\rightarrow"),
        (r"\leftrightarrow", r"\rightarrow"),
        (r"\in t", r"\int"),
        (r"a \\ b", "ab"),  # a line break is not spacing
        ("e^{x}", r"\mathrm{e}^{x}"),  # a bare e is a variable, not the constant
        ("i x", r"\mathrm{i}x"),
        (r"\mathrm{e}", r"\mathrm{i}"),
    ],
)
def test_normalize_keeps_these_apart(a: str, b: str) -> None:
    assert normalize(a) != normalize(b)


def test_normalize_keeps_tensor_indices() -> None:
    assert normalize(r"R^{\mu}{}_{\nu\rho\sigma}") == r"R^{\mu}_{\nu\rho\sigma}"


def test_prices_are_text_and_multiline_display_math_is_found() -> None:
    segs = split_segments("It costs $5 to $10 in total.", set())
    assert [s.type for s in segs] == ["text"]
    text = "A:\n\\[\n x = 1 \\\\\n y = 2\n\\]\nB: \\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned} C: $z$."
    kinds = [(s.type, s.value) for s in split_segments(text, set()) if s.type != "text"]
    assert kinds[0][0] == "display_math" and r"\\" in kinds[0][1]
    assert kinds[1] == ("display_math", "\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}")
    assert kinds[2] == ("math", "z")


def test_a_handbook_formula_matches_whatever_its_spacing() -> None:
    known = {normalize(r"\int_0^\infty \mathrm{e}^{-a x}\,dx = \frac{1}{a}")}
    segs = split_segments(
        r"$$\int_0^\infty \exponentialE^{-ax} dx = \frac{1}{a}$$ and $\int_0^\infty e^{-ax}dx=\frac1a$.",
        known,
    )
    assert [s.status for s in segs if s.type != "text"] == ["from_tool", "unchecked"]


ROTATING_DISK = (
    r"On the rotating disk $L = T$, with $\theta = \phi + \omega t$ and "
    r"$\dot\theta = \dot\phi + \omega$, so $L = \frac12 m (\dot r^2 + r^2 \dot\theta^2)$."
)


def test_relations_the_model_writes_are_unchecked_with_a_reason_never_contradicted(
    ctx: ToolContext,
) -> None:
    msg = final(
        collect(FakeLLM([[Chunk(text=ROTATING_DISK)]]), ctx, [{"role": "user", "content": "q"}])
    )
    claims = [s for s in segments_of(msg) if s["type"] != "text"]
    assert len(claims) == 4
    assert {s["status"] for s in claims} == {"unchecked"}
    assert all("not tested as an identity" in s["note"] for s in claims)
    assert {c["kind"] for c in msg["debug"]["claims"]} == {"relation"}
    # an equation false for every value is still called so; one only false at generic points is not
    msg = final(
        collect(
            FakeLLM(
                [[Chunk(text=r"False: $x^2 + 2x + 1 = x^2 + 2x$. Wrong: $(x+1)^2 = x^2 + 1$.")]]
            ),
            ctx,
            [{"role": "user", "content": "q"}],
        )
    )
    always, generic = (s for s in segments_of(msg) if s["type"] == "math")
    assert always["status"] == "contradicted"
    assert (generic["status"], "identity") == ("unchecked", msg["debug"]["claims"][1]["kind"])
    assert "not an identity" in generic["note"]


def test_definitions_from_earlier_turns_and_earlier_assumptions_are_used(ctx: ToolContext) -> None:
    llm = FakeLLM(
        [[Chunk(text=r"Then $\cos\theta = \cos\phi\cos(\omega t) - \sin\phi\sin(\omega t)$.")]]
    )
    msg = final(
        collect(
            llm,
            ctx,
            [{"role": "user", "content": "q"}],
            prior_relations=[r"\theta = \phi + \omega t"],
        )
    )
    claim = next(s for s in segments_of(msg) if s["type"] == "math")
    assert claim["status"] == "conditional"
    assert r"\theta = \phi + \omega t" in claim["note"]
    assert msg["debug"]["claims"][0]["given"] == [r"\theta = \phi + \omega t"]
    # a stated definition that the claim contradicts is named
    llm = FakeLLM([[Chunk(text=r"Then $\cos\theta = \cos\phi + \cos(\omega t)$.")]])
    msg = final(
        collect(
            llm,
            ctx,
            [{"role": "user", "content": "q"}],
            prior_relations=[r"\theta = \phi + \omega t"],
        )
    )
    claim = next(s for s in segments_of(msg) if s["type"] == "math")
    assert claim["status"] == "contradicted" and "does not agree with" in claim["note"]
    # assumptions declared in an earlier turn apply
    llm = FakeLLM([[Chunk(text=r"For positive x, $\sqrt{x^2} = x$.")]])
    msg = final(
        collect(
            llm,
            ctx,
            [{"role": "user", "content": "q"}],
            prior_assumptions={"x": ["positive"]},
        )
    )
    assert next(s for s in segments_of(msg) if s["type"] == "math")["status"] == "checked"


SPHERE_ARGS = {
    "coordinates": ["theta", "phi"],
    "metric": [["a^2", "0"], ["0", "a^2 \\sin^2\\theta"]],
    "assumptions": {"a": ["positive"]},
    "quantities": ["ricci_scalar", "christoffel"],
}


def test_geometry_tool_returns_a_card_and_a_checkable_summary(ctx: ToolContext) -> None:
    out = execute("geometry", SPHERE_ARGS, ctx)
    assert out.status == "ok" and out.card.type == "geometry"
    assert out.card.data["request"]["coordinates"] == [
        "theta",
        "phi",
    ]  # the card carries the problem
    text = out.for_model
    assert "status=ok" in text and "convention:" in text and "Levi-Civita" in text
    assert r"value=$\frac{2}{a^{2}}$" in text
    assert "NOT DETERMINED" in text  # sin^2 theta is not known to be strictly positive
    # every formula the model may quote is registered as coming from the tool
    assert normalize(r"\frac{2}{a^{2}}") in {normalize(x) for x in out.known_latex}
    assert any(r"\Gamma^{\theta}{}_{\phi \phi}" in x for x in out.known_latex)
    assert any("sin" in r for r in out.known_latex)  # the restriction sin(theta) != 0


def test_geometry_tool_failures_are_explained_not_decorated(ctx: ToolContext) -> None:
    out = execute("geometry", {**SPHERE_ARGS, "metric": [["1", "0"], ["0", "0"]]}, ctx)
    assert out.status == "unsupported" and "degenerate" in out.for_model
    assert "value=" not in out.for_model
    out = execute("geometry", {**SPHERE_ARGS, "quantities": ["levitation"]}, ctx)
    assert out.status == "error"
    out = execute("geometry", {**SPHERE_ARGS, "coordinates": []}, ctx)
    assert out.status == "error"
    out = execute("geometry", {**SPHERE_ARGS, "metric": [["f(r)", "0"], ["0", "1"]]}, ctx)
    assert out.status == "parse_error" and "declare" in out.for_model


def test_geometry_tool_never_leaks_a_state_between_calls(ctx: ToolContext) -> None:
    # the geometry is resent each time: a second call with other quantities gives the same metric
    first = execute("geometry", SPHERE_ARGS, ctx)
    second = execute("geometry", {**SPHERE_ARGS, "quantities": ["riemann"]}, ctx)
    assert first.card.data["result"]["metric_latex"] == second.card.data["result"]["metric_latex"]
    assert [q["kind"] for q in second.card.data["result"]["quantities"]] == ["riemann"]


def test_geometry_formulas_are_marked_from_the_tool_and_invented_ones_are_not(
    ctx: ToolContext,
) -> None:
    llm = FakeLLM(
        [
            [Chunk(tool_calls=[ToolCall("geometry", SPHERE_ARGS)])],
            [
                Chunk(
                    text=r"The Ricci scalar is $\frac{2}{a^{2}}$, not $\frac{3}{a^{2}}$ "
                    r"and the restriction is $\sin{\left(\theta \right)} \neq 0$."
                )
            ],
        ]
    )
    msg = final(collect(llm, ctx, [{"role": "user", "content": "curvature of the sphere"}]))
    marks = {s["value"]: s["status"] for s in segments_of(msg) if s["type"] != "text"}
    assert marks[r"\frac{2}{a^{2}}"] == "from_tool"
    assert marks[r"\sin{\left(\theta \right)} \neq 0"] == "from_tool"
    assert marks[r"\frac{3}{a^{2}}"] != "from_tool"  # not in any tool result


def test_distribution_tool_reports_steps_conditions_and_evidence(ctx: ToolContext) -> None:
    out = execute(
        "distribution",
        {
            "latex": r"x^2\operatorname{DiracDelta}(x^2-4)",
            "assumptions": {"x": ["real"]},
            "action": "integrate",
            "variable": "x",
            "lower": "-oo",
            "upper": "oo",
        },
        ctx,
    )
    assert out.card.type == "distribution" and out.status == "ok"
    result = out.card.data["result"]
    assert result["result_text"] == "2" and result["equality"] == "value"
    assert "distributions.delta_sifting" in out.for_model
    assert "conditions established exactly: yes" in out.for_model
    assert "supporting evidence, not a proof" in out.for_model
    assert f"result=${result['result_latex']}$" in out.for_model
    assert result["result_latex"] in out.known_latex
    assert any(m["id"] == "distributions.delta_sifting" for m in out.card.data["methods"])
    assert out.card.data["request"]["action"] == "integrate"


def test_distribution_tool_says_the_equality_is_distributional(ctx: ToolContext) -> None:
    out = execute(
        "distribution",
        {
            "latex": r"\operatorname{Heaviside}(x-1)",
            "assumptions": {"x": ["real"]},
            "action": "differentiate",
        },
        ctx,
    )
    assert out.status == "ok"
    assert "equality=AS DISTRIBUTIONS (not pointwise)" in out.for_model
    assert "distributions.heaviside_derivative" in out.for_model


def test_distribution_tool_refusals_name_the_reason_and_show_no_result(ctx: ToolContext) -> None:
    unknown = execute(
        "distribution",
        {"latex": r"\operatorname{DiracDelta}(ax)", "assumptions": {"x": ["real"], "a": ["real"]}},
        ctx,
    )
    assert unknown.status == "conditions_not_satisfied"
    assert "reason=assumptions_insufficient" in unknown.for_model
    assert "result=" not in unknown.for_model
    assert "NOT APPLIED (distributions.delta_scaling)" in unknown.for_model

    boundary = execute(
        "distribution",
        {
            "latex": r"\operatorname{DiracDelta}(x-1)",
            "assumptions": {"x": ["real"]},
            "action": "integrate",
            "lower": "0",
            "upper": "1",
        },
        ctx,
    )
    assert "reason=integration_domain_ambiguous" in boundary.for_model
    assert "endpoint_convention" not in boundary.for_model  # the tool never picks one


def test_distribution_tool_keeps_delta_and_theta_as_symbols(ctx: ToolContext) -> None:
    out = execute("distribution", {"latex": r"\delta(x)", "assumptions": {"x": ["real"]}}, ctx)
    assert out.status == "unsupported"  # no distribution was written: \delta(x) is delta times x


def test_distribution_tool_is_deterministic_for_trace_replay(ctx: ToolContext) -> None:
    args = {
        "latex": r"\operatorname{DiracDelta}(a x)",
        "assumptions": {"x": ["real"], "a": ["real", "nonzero"]},
    }
    first = execute("distribution", args, ctx)
    second = execute("distribution", args, ctx)
    assert first.status == "ok" and first.for_model == second.for_model
