"""A verified Taylor series is not marked contradicted by the post-turn claim check (issue #167).

A series call, then a reply that restates the expansion with Big-O, the centre ("about x = 0") and the places where
the error was measured. The reported build marked the expansion and three sample locations contradicted. Here the
whole turn runs through the real loop with a replayed model: the tool's own result links the restated series, the
centre and the sample points are parameters, and a changed series or a real contradicting definition still behave
as before (never supported / still contradicted).
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from minerva_api.agent.loop import run_agent, series_evidence
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_api.agent.turn_context import _prior_evidence
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

POLY = r"x - \frac{x^3}{6} + \frac{x^5}{120}"
ARGS = {"latex": r"\sin x", "variable": "x", "point": "0", "order": 5}
USER = "Expand $\\sin x$ about $x=0$ to order 5 and show how good it is."


def reply(series: str, extra: str = "") -> str:
    return (
        "The Taylor series of $\\sin x$ about $x=0$ is\n\n"
        f"$$\\sin x = {series}$$\n\n"
        "The measured error is small: at $|x|=0.5$, $|x|=0.1$ and $|x|=0.01$ the truncation error "
        f"falls with the order. {extra}"
    )


RIGHT = reply(f"{POLY} + O(x^6)")


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def run(
    ctx: ToolContext, answer: str, args: dict[str, Any] | None = None, **prior: Any
) -> dict[str, Any]:
    llm = ReplayLLM([call("series", args or ARGS), say(answer)])

    async def go() -> list[dict[str, Any]]:
        history = [{"role": "user", "content": USER}]
        return [e async for e in run_agent(llm, history, ctx, **prior)]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


def marks(message: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        s["value"]: s
        for b in message["blocks"]
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] != "text"
    }


def status_of(message: dict[str, Any], fragment: str) -> str:
    found = [s["status"] for v, s in marks(message).items() if fragment in v]
    assert len(found) == 1, (fragment, list(marks(message)))
    return found[0]


def test_the_restated_series_and_its_sample_points_are_not_contradicted(ctx: ToolContext) -> None:
    message = run(ctx, RIGHT, prior_relations=[r"x = 0"])
    statuses = {v: s["status"] for v, s in marks(message).items()}
    assert "contradicted" not in statuses.values(), statuses
    expansion = next(s for v, s in marks(message).items() if "O(x^6)" in v)
    assert expansion["status"] == "from_tool"
    # worded as the series result: a measurement, never a proof or a bound
    assert "Taylor series the tool computed" in expansion["note"]
    assert "not a proof or a bound" in expansion["note"]
    claims = {c["value"]: c for c in message["debug"]["claims"]}
    linked = next(c for v, c in claims.items() if "O(x^6)" in v)
    assert linked["evidence_rule"] == "series" and linked["evidence_source"] == "r1"
    # the centre and the sample points were checked as parameters, never as definitions
    assert all(c["given"] == [] for c in claims.values())


def test_tool_evidence_and_unchecked_prose_stay_distinguishable(ctx: ToolContext) -> None:
    message = run(ctx, RIGHT)
    by_value = marks(message)
    assert by_value[next(v for v in by_value if "O(x^6)" in v)]["status"] == "from_tool"
    for fragment in ("|x|=0.5", "|x|=0.1"):
        mark = by_value[next(v for v in by_value if fragment in v)]
        assert mark["status"] == "unchecked"  # prose restating a parameter: shown, never "verified"


@pytest.mark.parametrize(
    "series",
    [
        r"x - \frac{x^3}{5} + \frac{x^5}{120} + O(x^6)",
        f"{POLY} + O(x^5)",
        f"{POLY} + O(x^7)",
        f"{POLY} + O((x-1)^6)",
        r"x - \frac{x^3}{6} + O(x^6)",
    ],
    ids=["coefficient", "lower order", "higher order", "centre", "dropped term"],
)
def test_a_changed_series_is_unchecked_and_never_contradicted(
    ctx: ToolContext, series: str
) -> None:
    message = run(ctx, reply(series), prior_relations=[r"x = 0"])
    expansion = status_of(message, "O(")
    assert expansion == "unchecked"
    assert "contradicted" not in {s["status"] for s in marks(message).values()}


def test_another_function_is_not_linked_to_the_series(ctx: ToolContext) -> None:
    message = run(ctx, reply(f"{POLY} + O(x^6)").replace(r"\sin x =", r"\cos x ="))
    assert status_of(message, "O(x^6)") == "unchecked"


def test_a_real_definition_still_contradicts_a_wrong_value_in_the_same_turn(
    ctx: ToolContext,
) -> None:
    horizon = "At the horizon $r = 2M$, so $K = \\frac{48 M^2}{(2M)^6} = \\frac{3}{%sM^4}$."
    relations = [r"x = 0", r"K = \frac{48 M^2}{r^6}"]
    wrong = run(ctx, RIGHT + "\n\n" + horizon % "5", prior_relations=relations)
    assert status_of(wrong, "O(x^6)") == "from_tool"
    assert status_of(wrong, "= \\frac{3}{5M^4}") == "contradicted"
    right = run(ctx, RIGHT + "\n\n" + horizon % "4", prior_relations=relations)
    assert status_of(right, "O(x^6)") == "from_tool"
    assert status_of(right, "= \\frac{3}{4M^4}") != "contradicted"


def test_the_series_record_is_typed_and_comes_from_the_tool(ctx: ToolContext) -> None:
    message = run(ctx, RIGHT)
    step = message["trace"][0]
    assert step["name"] == "series" and step["series"] is not None
    record = step["series"]
    assert record["variable"] == "x" and record["order"] == 5
    assert record["centre_latex"] == "0"
    assert [t["power"] for t in record["terms"]] == [1, 3, 5]
    assert record["distances"] == [0.5, 0.1, 0.01]
    # the model's own text and arguments are not the record: it is built from the engine's result
    assert "series" not in step["arguments"]


def test_a_series_that_was_not_verified_leaves_no_record(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [call("series", {"latex": r"\frac{1}{x}", "variable": "x", "point": "0"}), say("none")]
    )

    async def go() -> list[dict[str, Any]]:
        history = [{"role": "user", "content": "expand 1/x"}]
        return [e async for e in run_agent(llm, history, ctx)]

    message = next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")
    assert "series" not in message["trace"][0]  # absent, so stored traces keep their old shape


def test_two_turns_a_definition_then_the_expansion(ctx: ToolContext) -> None:
    """Turn 1 defines `f` and runs the series; turn 2 restates it as `f(x) = ...`."""
    first = run(ctx, "Let $f(x) = \\sin x$. Its series is computed above.")
    stored = [SimpleNamespace(role="assistant", content={"trace": first["trace"]})]
    prior = _prior_evidence(stored)  # type: ignore[arg-type]
    assert series_evidence([], prior)[0].order == 5
    claim = f"$$f(x) = {POLY} + O(x^6)$$"
    llm = ReplayLLM([say(f"Restated: {claim}")])

    def turn_two(relations: list[str]) -> dict[str, Any]:
        async def go() -> list[dict[str, Any]]:
            history = [{"role": "user", "content": "Restate it for f."}]
            return [
                e
                async for e in run_agent(
                    llm,
                    history,
                    ctx,
                    prior_relations=relations,
                    prior_evidence=prior,
                )
            ]

        return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")

    linked = turn_two([r"f(x) = \sin x", r"x = 0"])
    assert status_of(linked, "O(x^6)") == "from_tool"
    # wrong twin: the earlier turn defined `f` as another function
    llm = ReplayLLM([say(f"Restated: {claim}")])
    twin = turn_two([r"f(x) = \cos x", r"x = 0"])
    assert status_of(twin, "O(x^6)") == "unchecked"
    # wrong twin: `f` was never defined
    llm = ReplayLLM([say(f"Restated: {claim}")])
    assert status_of(turn_two([r"x = 0"]), "O(x^6)") == "unchecked"
