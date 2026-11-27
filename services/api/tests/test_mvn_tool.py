"""The `multivariate_normal` tool (issue #229): schema, evidence, refusals, provenance and a chat of three turns.

The mean, covariance, value conditioned on and map are the user's: the schema has no default for any, a result
is quoted (never marked verified) when it holds given a premise, a refusal carries its reason and no formula,
and a mean or covariance entry that no message of the user wrote is labelled as assumed by the assistant.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient
from minerva_api.agent.mvn_tools import ASSUMED_LABEL, DELEGATED_LABEL, MODEL_LINE
from minerva_api.agent.prompts import GROUP_KEYWORDS, GROUP_TOOLS
from minerva_api.agent.refs import REFERENCEABLE_TOOLS
from minerva_api.agent.tools import TOOLS, ToolContext, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_chat import cards_of, chat_client, sse  # noqa: F401

BASE: dict[str, Any] = {
    "mean": ["1", "-2"],
    "covariance": [["4", "1"], ["1", "2"]],
    "variables": ["x", "y"],
}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def with_messages(ctx: ToolContext, *messages: str) -> ToolContext:
    return ToolContext(engine=ctx.engine, kb=ctx.kb, user_messages=list(messages))


# ---------------------------------------------------------------------------------- schema


def test_nothing_the_user_owns_has_a_default() -> None:
    schema = TOOLS["multivariate_normal"].schema()["function"]["parameters"]
    assert set(schema["required"]) == {"mean", "covariance", "action"}
    for name in ("variables", "keep", "given", "matrix", "offset"):
        assert (
            "default" in schema["properties"][name]
            and schema["properties"][name]["default"] is None
        )
    description = schema["properties"]["covariance"]["description"]
    assert "never infer" in description and "uncorrelated" in description


def test_it_lives_in_the_statistics_group_and_is_not_core() -> None:
    assert "multivariate_normal" in GROUP_TOOLS["statistics"]
    for phrase in (
        "multivariate normal",
        "multivariate gaussian",
        "bivariate normal",
        "bivariate gaussian",
        "joint normal",
    ):
        assert re.search(GROUP_KEYWORDS["statistics"], phrase), phrase


def test_it_has_no_result_handle() -> None:
    assert "multivariate_normal" not in REFERENCEABLE_TOOLS


# ---------------------------------------------------------------------------------- results


def test_a_numeric_model_is_verified_and_its_formulas_are_known(ctx: ToolContext) -> None:
    out = execute("multivariate_normal", {**BASE, "action": "density"}, ctx)
    assert out.status == "ok" and out.card.type == "multivariate_normal"
    assert out.evidence is not None and out.evidence.accepted and out.evidence.check == "numeric"
    assert out.known_latex and not out.quoted_latex
    assert "numerically checked" in out.for_model and "not proved" in out.for_model
    assert "HOLDS GIVEN" not in out.for_model and "assumed" not in out.for_model
    assert (
        out.card.data["provenance"]["who"] == "user"
    )  # outside a chat turn the values are the caller's


def test_the_text_repeats_the_model_it_describes(ctx: ToolContext) -> None:
    out = execute("multivariate_normal", {**BASE, "action": "marginal", "keep": "x"}, ctx)
    assert out.status == "ok"
    line = next(ln for ln in out.for_model.splitlines() if ln.startswith(MODEL_LINE))
    assert 'mean=["1", "-2"]' in line and 'covariance=[["4", "1"], ["1", "2"]]' in line
    assert 'variables=["x", "y"]' in line
    assert (
        out.for_model.splitlines()[1] == line
    )  # right after the head: an older result is cut from the end


def test_an_undecided_symbolic_covariance_is_conditional_and_only_quoted(ctx: ToolContext) -> None:
    out = execute(
        "multivariate_normal",
        {
            "mean": ["m_1", "m_2"],
            "covariance": [["s_1", "c"], ["c", "s_2"]],
            "variables": ["x", "y"],
            "action": "moments",
        },
        ctx,
    )
    assert out.status == "ok"
    assert "HOLDS GIVEN" in out.for_model and "never 'proved'" in out.for_model
    assert out.known_latex == [] and out.quoted_latex  # shown, never marked verified
    assert out.card.data["result"]["evidence"] == "conditional"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"covariance": [["1", "1"], ["1", "1"]], "action": "density"}, "singular_covariance"),
        ({"covariance": [["4", "1"], ["2", "2"]], "action": "density"}, "not_symmetric"),
        ({"covariance": [["1", "2"], ["2", "1"]], "action": "density"}, "not_positive_definite"),
        ({"variables": None, "action": "density"}, "variables_missing"),
    ],
)
def test_a_refusal_carries_its_reason_and_no_formula(
    ctx: ToolContext, changes: dict[str, Any], reason: str
) -> None:
    out = execute("multivariate_normal", {**BASE, **changes}, ctx)
    assert out.status != "ok" and f"reason={reason}" in out.for_model
    assert out.known_latex == [] and out.quoted_latex == []
    assert "$" not in out.for_model.split("model (send")[0]  # no formula before the model line
    assert out.card.data["result"]["reason"] == reason


def test_the_model_is_sent_back_even_when_the_call_was_refused(ctx: ToolContext) -> None:
    out = execute(
        "multivariate_normal",
        {**BASE, "covariance": [["1", "1"], ["1", "1"]], "action": "density"},
        ctx,
    )
    assert 'covariance=[["1", "1"], ["1", "1"]]' in out.for_model


# ---------------------------------------------------------------------------------- provenance


STATED = (
    "A bivariate normal has mean (1, -2) and covariance [[4, 1], [1, 2]] for (x, y). Its density?"
)


def test_values_the_user_wrote_are_not_labelled(ctx: ToolContext) -> None:
    out = execute("multivariate_normal", {**BASE, "action": "density"}, with_messages(ctx, STATED))
    assert ASSUMED_LABEL not in out.for_model
    assert out.card.data["provenance"] == {"who": "user", "values": [], "label": None}


def test_a_value_no_message_wrote_is_labelled_as_assumed(ctx: ToolContext) -> None:
    out = execute(
        "multivariate_normal",
        {**BASE, "covariance": [["4", "0"], ["0", "2"]], "action": "density"},
        with_messages(ctx, STATED),
    )
    assert out.status == "ok"  # a label, not a refusal
    assert ASSUMED_LABEL in out.for_model
    provenance = out.card.data["provenance"]
    assert provenance["who"] == "assistant" and provenance["values"] == ["0"]
    assert provenance["label"] == ASSUMED_LABEL


def test_a_model_with_no_numbers_from_the_user_is_all_assumed(ctx: ToolContext) -> None:
    out = execute(
        "multivariate_normal",
        {**BASE, "action": "density"},
        with_messages(ctx, "Give me the density of some bivariate normal."),
    )
    assert out.card.data["provenance"]["who"] == "assistant"
    assert set(out.card.data["provenance"]["values"]) == {"1", "-2", "4", "2"}


def test_symbolic_entries_match_the_users_spelling(ctx: ToolContext) -> None:
    out = execute(
        "multivariate_normal",
        {
            "mean": ["0", "0"],
            "covariance": [
                ["sigma_x^2", "rho*sigma_x*sigma_y"],
                ["rho*sigma_x*sigma_y", "sigma_y^2"],
            ],
            "variables": ["x", "y"],
            "action": "moments",
            "assumptions": {"sigma_x": ["positive"], "sigma_y": ["positive"], "rho": ["real"]},
        },
        with_messages(
            ctx,
            r"Mean $0, 0$ and covariance $\begin{pmatrix}\sigma_{x}^{2} & \rho\sigma_x\sigma_y\\ "
            r"\rho\sigma_x\sigma_y & \sigma_y^2\end{pmatrix}$ for (x, y).",
        ),
    )
    assert out.card.data["provenance"]["who"] == "user", out.card.data["provenance"]


def test_a_hand_over_in_words_is_labelled_as_chosen_by_the_assistant(ctx: ToolContext) -> None:
    out = execute(
        "multivariate_normal",
        {**BASE, "action": "density"},
        with_messages(
            ctx,
            "You choose the mean and covariance of a bivariate normal for x and y; give its density.",
        ),
    )
    assert out.card.data["provenance"]["who"] == "assistant_delegated"
    assert DELEGATED_LABEL in out.for_model


# ---------------------------------------------------------------------------------- a chat


def test_three_turns_re_send_the_model_the_card_returned(
    chat_client: tuple[TestClient, str],  # noqa: F811
) -> None:
    client, cid = chat_client
    first = next(d for e, d in sse(client, cid, STATED) if e == "message")
    (card1,) = cards_of(first)
    assert card1["type"] == "multivariate_normal" and card1["data"]["result"]["status"] == "ok"
    assert card1["data"]["result"]["density_text"]
    assert card1["data"]["provenance"]["who"] == "user"

    second = next(
        d
        for e, d in sse(client, cid, "Now the marginal of x of that bivariate normal.")
        if e == "message"
    )
    (card2,) = cards_of(second)
    assert (
        card2["data"]["request"]["action"] == "marginal" and card2["data"]["request"]["keep"] == "x"
    )
    assert card2["data"]["request"]["mean"] == ["1", "-2"]  # the model of turn 1, unchanged
    assert card2["data"]["result"]["mean_text"] == ["1"]
    assert card2["data"]["result"]["covariance_text"] == [["4"]]

    third = next(
        d
        for e, d in sse(client, cid, "Now condition that bivariate normal on x = 3.")
        if e == "message"
    )
    (card3,) = cards_of(third)
    request = card3["data"]["request"]
    assert request["covariance"] == [["4", "1"], ["1", "2"]] and request["given"] == {"x": "3"}
    result = card3["data"]["result"]
    assert result["mean_text"] == ["-3/2"] and result["covariance_text"] == [["7/4"]]
    for card in (card2, card3):
        assert (
            card["data"]["provenance"]["who"] == "user"
        )  # the numbers are in the user's first message
    assert json.dumps(third["trace"]).count("multivariate_normal") >= 1
