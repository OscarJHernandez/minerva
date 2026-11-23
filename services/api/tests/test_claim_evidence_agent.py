"""The spherical-pendulum turn: the equations the model writes are linked to the verified result.

Regression for "formulas with engine evidence 0/12": lone symbols were counted as formulas and no equation
was compared with what `analyze_lagrangian` had verified.
"""

import asyncio
from typing import Any

import pytest
from minerva_api.agent.loop import is_claim, is_lone_symbol, run_agent
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

LAGRANGIAN = r"\frac12 m R^2 (\dot{\theta}^2 + \sin^2(\theta) \dot{\phi}^2) - m g R \cos(\theta)"
ANSWER_RIGHT = (
    "With coordinates $\\theta$ and $\\phi$, the equations of motion are\n\n"
    "$$\\ddot\\theta = \\sin\\theta\\cos\\theta\\,\\dot\\phi^2 + \\frac{g}{R}\\sin\\theta$$\n\n"
    "$$\\ddot\\phi = -2\\cot\\theta\\,\\dot\\theta\\dot\\phi$$\n\n"
    "$\\phi$ is cyclic, so $p_\\phi = mR^2\\sin^2\\theta\\,\\dot\\phi$ is conserved, and so is "
    "$E=\\frac12 mR^2(\\dot\\theta^2+\\sin^2\\theta\\,\\dot\\phi^2)+mgR\\cos\\theta$."
)
ANSWER_WRONG = ANSWER_RIGHT.replace("+ \\frac{g}{R}", "- \\frac{g}{R}")


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def turn(ctx: ToolContext, answer: str) -> dict[str, Any]:
    llm = ReplayLLM(
        [
            call(
                "analyze_lagrangian",
                {"lagrangian": LAGRANGIAN, "coordinates": ["theta", "phi"]},
            ),
            say(answer),
        ]
    )
    text = f"Find the equations of motion for $L = {LAGRANGIAN}$"

    async def go() -> list[dict[str, Any]]:
        return [e async for e in run_agent(llm, [{"role": "user", "content": text}], ctx)]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


def formulas(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        s
        for b in message["blocks"]
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] != "text"
    ]


def test_the_expected_equations_are_linked_to_the_engine_result(ctx: ToolContext) -> None:
    message = turn(ctx, ANSWER_RIGHT)
    marks = {s["value"]: s["status"] for s in formulas(message)}
    claims = {v: m for v, m in marks.items() if is_claim(v)}
    assert len(claims) == 4
    assert set(claims.values()) == {"from_tool"}, claims
    debug = message["debug"]
    assert debug["formulas_with_evidence"] == 4
    assert {c["evidence_rule"] for c in debug["claims"]} >= {"same_left_side", "named_quantity"}
    # lone symbols assert nothing: they are not counted as formulas that lack evidence
    assert all(not is_claim(s["value"]) for s in formulas(message) if s["status"] == "unchecked")
    assert debug["formulas"] == 4 and debug["evidence_pct"] == 100.0


def test_a_wrong_equation_in_the_same_answer_stays_unchecked(ctx: ToolContext) -> None:
    message = turn(ctx, ANSWER_WRONG)
    marks = {s["value"]: s["status"] for s in formulas(message) if is_claim(s["value"])}
    wrong = [v for v in marks if "ddot\\theta" in v]
    assert len(wrong) == 1 and marks[wrong[0]] == "unchecked"
    assert sum(1 for m in marks.values() if m == "from_tool") == 3
    assert message["debug"]["formulas_with_evidence"] == 3


def test_the_support_note_names_the_result_and_the_restrictions(ctx: ToolContext) -> None:
    message = turn(ctx, ANSWER_RIGHT)
    theta = next(s for s in formulas(message) if "ddot\\theta" in s["value"])
    assert theta["status"] == "from_tool"
    assert "equivalent" in (theta["note"] or "") and "neq 0" in (theta["note"] or "")


@pytest.mark.parametrize(
    ("latex", "lone"),
    [
        (r"\theta", True),
        ("m", True),
        ("x_1", True),
        (r"p_{\theta}", True),
        (r"\dot{\phi}", True),
        (r"\theta + 1", False),
        (r"mR^2", False),
        (r"\theta = 1", False),
        ("2", False),
    ],
)
def test_lone_symbols(latex: str, lone: bool) -> None:
    assert is_lone_symbol(latex) is lone


# -- issue #132: `C_4` re-typed with `\ddot\theta` for `\dot\theta` -------------------------------------------
POLAR = r"\frac{\mu}{2}(\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{G M \mu}{r}"
C4_RIGHT = r"C_4 = G M \sin\theta - r^3\dot\theta^2\sin\theta + r^2\dot r\dot\theta\cos\theta"
C4_RETYPED = C4_RIGHT.replace(r"\dot r\dot\theta\cos", r"\dot r\,\ddot\theta\cos")


def discovery_turn(ctx: ToolContext, answer: str) -> dict[str, Any]:
    llm = ReplayLLM(
        [
            call(
                "find_conserved_quantities",
                {"lagrangian": POLAR, "coordinates": ["r", "theta"], "budget": "thorough"},
            ),
            say(answer),
        ]
    )

    async def go() -> list[dict[str, Any]]:
        text = "Find the hidden conserved quantities of the Kepler problem."
        return [e async for e in run_agent(llm, [{"role": "user", "content": text}], ctx)]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


def test_the_discovered_constant_the_model_repeats_is_linked_to_the_search(
    ctx: ToolContext,
) -> None:
    message = discovery_turn(ctx, f"The fourth constant is $$ {C4_RIGHT} $$")
    claim = next(s for s in formulas(message) if s["value"].strip().startswith("C_4"))
    assert claim["status"] == "from_tool", claim
    assert {c["evidence_rule"] for c in message["debug"]["claims"]} == {"same_left_side"}


def test_a_re_typed_constant_is_unchecked_with_a_note_naming_the_verified_equation(
    ctx: ToolContext,
) -> None:
    message = discovery_turn(ctx, f"The fourth constant is $$ {C4_RETYPED} $$")
    claim = next(s for s in formulas(message) if s["value"].strip().startswith("C_4"))
    assert (
        claim["status"] == "unchecked"
    )  # never contradicted: it is the tool's name, not a false identity
    assert "verified result" in (claim["note"] or "") and "dot{\\theta} \\cos" in (
        claim["note"] or ""
    )
    rules = {c["evidence_rule"] for c in message["debug"]["claims"]}
    assert rules == {"mismatch"}
