"""The statistics tools: cards, evidence, and what the model may and may not claim."""

from typing import Any

import pytest
from minerva_api.agent.tools import TOOLS, ToolContext, execute, schemas
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

DATA = {"n": ["integer", "positive"], "x": ["real"], "v": ["positive"]}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def mle(**kw: Any) -> dict[str, Any]:
    return {
        "family": "normal",
        "parametrization": "variance",
        "parameters": {"v": "v"},
        "observation": "x",
        "sample_size": "n",
        "estimate": ["mu"],
        "action": "mle",
        "assumptions": DATA,
        **kw,
    }


def test_the_schemas_tell_the_model_never_to_assume_independence_or_invent_inputs() -> None:
    by_name = {s["function"]["name"]: s["function"] for s in schemas()}
    for name in ("sums", "expectation", "statistical_model"):
        assert name in by_name
    stat = by_name["statistical_model"]
    props = stat["parameters"]["properties"]
    assert "never assume" in props["iid"]["description"].lower()
    assert "never choose a prior" in props["prior"]["description"].lower()
    assert (
        "NEVER add a group"
        in by_name["expectation"]["parameters"]["properties"]["independent"]["description"]
    )
    assert "candidate" not in props  # the model derives; it does not check its own answers


def test_a_maximum_likelihood_card_has_evidence_and_holds_given_independence(ctx) -> None:
    out = execute("statistical_model", mle(iid=True), ctx)
    assert out.status == "ok" and out.card.type == "statistics"
    assert out.evidence is not None and out.evidence.accepted and out.evidence.points_checked
    assert "HOLDS GIVEN" in out.for_model and "never 'proved'" in out.for_model
    # a conditional result is quoted, never marked verified
    assert not out.known_latex and out.quoted_latex
    assert out.card.data["request"]["family"] == "normal"


def test_without_a_statement_of_independence_the_model_is_told_to_ask(ctx) -> None:
    out = execute("statistical_model", mle(), ctx)
    assert out.status == "conditions_not_satisfied"
    assert "independent" in out.for_model and "Ask the user" in out.for_model
    assert out.evidence is not None and not out.evidence.accepted
    assert out.known_latex == []


def test_an_inadmissible_estimate_is_not_invented(ctx) -> None:
    out = execute(
        "statistical_model",
        {
            "family": "bernoulli",
            "observation": "x",
            "sample_size": "3",
            "iid": True,
            "estimate": ["p"],
            "action": "mle",
            "data": ["0", "0", "0"],
        },
        ctx,
    )
    assert out.status == "conditions_not_satisfied"
    assert "no maximum-likelihood estimate exists" in out.for_model


def test_moments_are_verified_results_not_conditional(ctx) -> None:
    out = execute(
        "statistical_model",
        {"family": "gamma", "parameters": {"alpha": "alpha", "beta": "beta"}, "action": "moments",
         "assumptions": {"alpha": ["positive"], "beta": ["positive"]}},
        ctx,
    )  # fmt: skip
    assert out.status == "ok" and out.known_latex and not out.quoted_latex
    assert any("\\frac{\\alpha}{\\beta}" in f.replace(" ", "") for f in out.known_latex)


def test_the_expectation_tool_keeps_covariance_unless_independence_is_given(ctx) -> None:
    free = execute("expectation", {"latex": r"\operatorname{Var}(X+Y)", "random": ["X", "Y"]}, ctx)
    assert free.status == "ok" and "no independence was used" in free.for_model
    assert free.known_latex  # an identity: the result is verified
    assert [why for _, why in free.quoted_latex] == ["your input, as read by Minerva"]
    given = execute(
        "expectation",
        {"latex": r"\operatorname{Var}(X+Y)", "random": ["X", "Y"], "independent": [["X", "Y"]]},
        ctx,
    )
    assert given.status == "ok" and "HOLDS GIVEN" in given.for_model
    assert any("independent" in why for _, why in given.quoted_latex)
    assert not any(
        "Cov" in f for f in given.known_latex
    )  # the conditional result is not marked verified


def test_the_sums_tool_says_checked_for_these_sizes_not_proved(ctx) -> None:
    out = execute(
        "sums",
        {"latex": r"\sum_{i=1}^{n}(x_i-\mu)^2", "assumptions": {"n": ["integer", "positive"], "x": ["real"], "mu": ["real"]}},
        ctx,
    )  # fmt: skip
    assert out.status == "ok" and out.card.type == "sums"
    assert "NOT a proof for every n" in out.for_model
    assert (
        out.evidence is not None
        and out.evidence.points_checked
        and out.evidence.symbolic_zero is None
    )
    bad = execute("sums", {"latex": r"\sum_{i=1}^{m} x_i"}, ctx)
    assert bad.status == "conditions_not_satisfied" and "range_not_provable" in bad.for_model


def test_the_matrix_tool_takes_plain_text_for_gradients(ctx) -> None:
    out = execute(
        "matrix_calculus",
        {
            "latex": "-1/2*logdet(S) - 1/2*transpose(x - mu)*inv(S)*(x - mu)",
            "matrices": [
                {"name": "x", "rows": 3, "cols": 1},
                {"name": "mu", "rows": 3, "cols": 1},
                {"name": "S", "rows": 3, "positive_definite": True},
            ],
            "action": "gradient",
            "variable": "S",
        },
        ctx,
    )
    assert out.status == "ok"
    assert (
        "convention=symmetric_frobenius" in out.for_model
        and "alternative (symmetric_entrywise" in out.for_model
    )
    assert "real field" in out.for_model
    undeclared = execute(
        "matrix_calculus",
        {
            "latex": "logdet(X)",
            "matrices": [{"name": "X", "rows": 3}],
            "action": "gradient",
            "variable": "X",
        },
        ctx,
    )
    assert undeclared.status == "conditions_not_satisfied"


def test_the_tools_are_registered_with_the_card_types_the_ui_renders() -> None:
    assert {"sums", "expectation", "statistical_model"} <= set(TOOLS)


# ---------------------------------------------------------------------------------- the scripted stand-in


async def _chunks(user: str) -> list:  # type: ignore[type-arg]
    from minerva_api.agent.scripted import ScriptedLLM

    return [c async for c in ScriptedLLM().chat([{"role": "user", "content": user}], [])]


def test_the_scripted_model_asks_about_independence_and_does_not_assume_it() -> None:
    import asyncio

    asked = asyncio.run(
        _chunks("Find the maximum likelihood estimator of the mean of n Gaussian observations.")
    )
    assert not any(c.tool_calls for c in asked)
    assert "independent" in "".join(c.text or "" for c in asked)
    called = asyncio.run(
        _chunks(
            "Observations are independent and Gaussian with known variance; find the maximum likelihood estimator."
        )
    )
    (call,) = [t for c in called for t in c.tool_calls]
    assert call.name == "statistical_model" and call.arguments["iid"] is True


def test_the_scripted_model_only_declares_independence_when_the_user_says_so() -> None:
    import asyncio

    free = asyncio.run(_chunks(r"Expand $\operatorname{Var}(X+Y)$."))
    (call,) = [t for c in free for t in c.tool_calls]
    assert call.name == "expectation" and "independent" not in call.arguments
    stated = asyncio.run(_chunks(r"Expand $\operatorname{Var}(X+Y)$ for independent X and Y."))
    (call,) = [t for c in stated for t in c.tool_calls]
    assert call.arguments["independent"] == [["X", "Y"]]
