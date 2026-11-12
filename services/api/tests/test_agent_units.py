"""Agent loop, tools and segment marking, with a deterministic fake model."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from minerva_api.agent.llm import Chunk, LLMError, ToolCall
from minerva_api.agent.loop import AgentLimits, TurnInfo, prompt_sha, run_agent
from minerva_api.agent.segments import normalize, split_segments
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase


class FakeLLM:
    """Replays prepared replies, one list of chunks per model call, and records what it was sent."""

    name = "fake"

    def __init__(self, replies: list[list[Chunk]] | None = None, repeat_last: bool = False) -> None:
        self.replies = replies or []
        self.repeat_last = repeat_last
        self.seen: list[list[dict[str, Any]]] = []
        self.error: LLMError | None = None

    async def chat(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AsyncIterator[Chunk]:
        self.seen.append([dict(m) for m in messages])
        if self.error:
            raise self.error
        index = len(self.seen) - 1
        if index >= len(self.replies):
            if not self.repeat_last:
                return
            index = len(self.replies) - 1
        for chunk in self.replies[index]:
            yield chunk


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
    call = [Chunk(tool_calls=[ToolCall("search_handbook", {"query": "gaussian"})])]
    llm = FakeLLM([call], repeat_last=True)
    msg = final(
        collect(llm, ctx, [{"role": "user", "content": "q"}], limits=AgentLimits(max_steps=3))
    )
    assert msg["truncated"] is True
    assert len(cards_of(msg)) == 3
    assert "limit" in msg["text"]


def test_time_cap_stops_the_turn(ctx: ToolContext) -> None:
    llm = FakeLLM([[Chunk(text="x")]], repeat_last=True)
    msg = final(
        collect(llm, ctx, [{"role": "user", "content": "q"}], limits=AgentLimits(max_seconds=-1))
    )
    assert msg["truncated"] is True and llm.seen == []


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
    events = collect(llm, ctx, [{"role": "user", "content": "q"}])
    assert [e["event"] for e in events] == ["error"]


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
    assert (debug["request_id"], debug["mode"], debug["model"]) == ("abc", "ollama", "fake")
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
    assert debug["max_steps"] == 6 and debug["model"] == "fake"
    by_value = {c["value"]: c["status"] for c in debug["claims"]}
    assert by_value["x^2 = 5x"] == "fails" or by_value["x^2 = 5x"] == "conditional"
    assert debug["prompt_sha"] == prompt_sha()


def test_tool_error_is_recorded_in_the_trace(ctx: ToolContext) -> None:
    llm = FakeLLM([[Chunk(tool_calls=[ToolCall("nope", {})])], [Chunk(text="Could not do that.")]])
    msg = final(collect(llm, ctx, [{"role": "user", "content": "q"}]))
    assert msg["trace"][0]["status"] == "error"


def test_tools_have_no_side_effects_surface(ctx: ToolContext) -> None:
    from minerva_api.agent.tools import TOOLS

    assert set(TOOLS) == {
        "search_handbook",
        "get_entry",
        "simplify",
        "find_identities",
        "tensor_canonicalize",
        "solve_polynomial",
        "analyze_lagrangian",
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
        r"True: $\frac{x^2-1}{x-1} = x+1$. False: $(x+1)^2 = x^2 + 1$. "
        r"Not an equation: $x^2$. Unreadable: $\foo = 1$."
    )
    msg = final(collect(FakeLLM([[Chunk(text=text)]]), ctx, [{"role": "user", "content": "q"}]))
    status = {s["value"]: s["status"] for s in segments_of(msg) if s["type"] != "text"}
    assert status[r"\frac{x^2-1}{x-1} = x+1"] == "checked"
    assert status["(x+1)^2 = x^2 + 1"] == "contradicted"
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
