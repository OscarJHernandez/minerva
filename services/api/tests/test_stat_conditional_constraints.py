"""Issue #233: a GIVEN parameter whose constraint the declarations cannot establish is a premise.

`v > 0` for a Normal variance left undeclared is part of the parameter space, not something the engine
proved, so the moments hold only given it: conditional, quoted, never marked verified. The constraint of a
parameter being ESTIMATED is the estimator's admissibility and is not a premise about the input.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from minerva_api.agent.evaluation.cases import EvalCase
from minerva_api.agent.evaluation.runner import observe
from minerva_api.agent.loop import AgentLimits
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext, execute
from minerva_api.agent.turn_context import turn_context
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

NORMAL = {
    "family": "normal",
    "parametrization": "variance",
    "parameters": {"mu": "0", "v": "v"},
    "action": "moments",
}
FAST = AgentLimits(retry_delay_seconds=0)
PROMPT = "Find the moments of a normal distribution with variance v."


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def run(ctx: ToolContext, **kw: Any) -> Any:
    return execute("statistical_model", {**NORMAL, **kw}, ctx)


def test_an_undeclared_variance_makes_the_moments_conditional_and_quoted(ctx) -> None:
    out = run(ctx)
    result = out.card.data["result"]
    assert out.status == "ok" and result["evidence"] == "conditional"
    assert any("v > 0" in p for p in result["premises"])
    assert all(s["evidence"] == "conditional" for s in result["steps"])
    assert not out.known_latex and out.quoted_latex
    assert "HOLDS GIVEN" in out.for_model and "never 'proved'" in out.for_model
    (cond,) = [c for c in result["conditions"] if c["parameter"] == "v"]
    assert cond["role"] == "given" and cond["established"] is False


def test_the_declared_twin_stays_a_verified_identity(ctx) -> None:
    out = run(ctx, assumptions={"v": ["positive"]})
    result = out.card.data["result"]
    assert result["evidence"] == "identity" and result["premises"] == []
    assert out.known_latex and not out.quoted_latex


def test_a_provably_invalid_parameter_is_still_refused(ctx) -> None:
    out = run(ctx, parameters={"mu": "0", "v": "-1"})
    assert out.status == "conditions_not_satisfied"
    assert out.card.data["result"]["reason"] == "parameter_invalid"


def test_a_symbolic_given_bernoulli_p_is_conditional_a_number_is_not(ctx) -> None:
    sym = execute(
        "statistical_model",
        {"family": "bernoulli", "parameters": {"p": "p"}, "action": "moments"},
        ctx,
    )
    assert sym.status == "ok" and sym.card.data["result"]["evidence"] == "conditional"
    assert not sym.known_latex and sym.quoted_latex
    num = execute(
        "statistical_model",
        {"family": "bernoulli", "parameters": {"p": "1/3"}, "action": "moments"},
        ctx,
    )
    assert num.card.data["result"]["evidence"] == "identity" and num.known_latex


def test_gamma_with_undeclared_symbolic_shape_is_conditional(ctx) -> None:
    args = {
        "family": "gamma",
        "parameters": {"alpha": "alpha", "beta": "beta"},
        "action": "moments",
    }
    open_ = execute("statistical_model", args, ctx)
    assert open_.card.data["result"]["evidence"] == "conditional" and not open_.known_latex
    declared = execute(
        "statistical_model",
        {**args, "assumptions": {"alpha": ["positive"], "beta": ["positive"]}},
        ctx,
    )
    assert declared.card.data["result"]["evidence"] == "identity" and declared.known_latex


def test_an_estimated_parameters_own_constraint_is_not_a_premise(ctx) -> None:
    """Bernoulli MLE of p: `0 < p < 1` is the estimator's admissibility, not a premise of the input."""
    out = execute(
        "statistical_model",
        {
            "family": "bernoulli",
            "parameters": {},
            "estimate": ["p"],
            "observation": "x",
            "sample_size": "n",
            "iid": True,
            "action": "mle",
            "assumptions": {"n": ["integer", "positive"], "x": ["real"]},
        },
        ctx,
    )
    result = out.card.data["result"]
    assert not any("parameter space" in p for p in result["premises"])
    assert all(c["role"] == "estimated" for c in result["conditions"] if c["parameter"] == "p")


def test_the_tool_does_not_depend_on_the_engine_alone(ctx) -> None:
    """Defence in depth: an ok result with an unestablished given condition and no premise is still quoted."""

    class Stripped:
        def statistical_model(self, request: Any) -> Any:
            result = ctx.engine.statistical_model(request)
            result.premises = []
            result.evidence = "identity"
            return result

    stripped = ToolContext(engine=Stripped(), kb=ctx.kb)  # type: ignore[arg-type]
    out = run(stripped)
    assert out.card.data["result"]["evidence"] == "identity"  # the engine alone said nothing
    assert not out.known_latex and out.quoted_latex
    assert "HOLDS GIVEN" in out.for_model


def test_a_conditional_result_stays_quoted_in_the_next_turn(ctx) -> None:
    case = EvalCase(
        id="stat-conditional",
        category="symbolic",
        prompt=PROMPT,
        expected_tool="statistical_model",
        source="tests/test_stat_conditional_constraints.py",
    )
    llm = ReplayLLM([call("statistical_model", NORMAL), say("The card shows the moments.")])
    obs = asyncio.run(observe(llm, case, ctx, FAST, stored=[]))
    assert not obs.error and obs.final is not None
    (card,) = [c for c in obs.cards.values() if c["type"] == "statistics"]
    assert card["data"]["result"]["evidence"] == "conditional"
    stored = [
        SimpleNamespace(role="user", content={"text": PROMPT}),
        SimpleNamespace(role="assistant", content=obs.final),
    ]
    turn = turn_context(stored, "What is the variance again?")
    moments = list(card["data"]["result"]["moments_latex"].values())
    assert moments
    for m in moments:
        assert m not in turn.prior_known
    assert any(m in turn.prior_quoted for m in moments)


def test_the_scripted_chat_turn_stores_a_conditional_card() -> None:
    from fastapi.testclient import TestClient

    from .conftest import build_app, register
    from .test_chat import cards_of, sse

    with TestClient(build_app(llm_mode="scripted")) as client:
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        sse(client, cid, PROMPT)
        last = client.get(f"/v1/chat/conversations/{cid}").json()["messages"][-1]
        (card,) = cards_of(last["content"])
        result = card["data"]["result"]
        assert result["evidence"] == "conditional" and result["premises"]
