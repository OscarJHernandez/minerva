"""A number the model writes itself next to a formula is never kept: the server numbers equations."""

import asyncio
from typing import Any

import pytest
from minerva_api.agent.equations import EquationTable, strip_equation_tags
from minerva_api.agent.loop import run_agent
from minerva_api.agent.replay import ReplayLLM, say
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


@pytest.mark.parametrize(
    ("written", "kept"),
    [
        ("so:\n\n$$a = b$$ (3)\n\nnext", "so:\n\n$$a = b$$\n\nnext"),
        ("so $$a$$(12) and \\[b\\]  (4)", "so $$a$$ and \\[b\\]"),
        (
            "as in (3), and $x$ (3) inline",
            "as in (3), and $x$ (3) inline",
        ),  # not after a display formula
        (
            "1. $$a$$\n(2) the second point",
            "1. $$a$$\n(2) the second point",
        ),  # a list marker, next line
        ("$$a$$ (see equation (2))", "$$a$$ (see equation (2))"),
    ],
)
def test_a_number_written_after_a_displayed_formula_is_dropped(written: str, kept: str) -> None:
    assert strip_equation_tags(written) == kept


def test_the_stored_reply_carries_the_servers_number_not_the_models(ctx: ToolContext) -> None:
    table = EquationTable()
    llm = ReplayLLM([say("Here:\n\n$$q = 1$$ (9)\n\nDone.")])

    async def go() -> list[dict[str, Any]]:
        return [
            e
            async for e in run_agent(
                llm, [{"role": "user", "content": "show me"}], ctx, equations=table
            )
        ]

    message = next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")
    segments = [s for b in message["blocks"] if b["kind"] == "text" for s in b["segments"]]
    assert [s["number"] for s in segments if s["type"] == "display_math"] == [1]
    assert "(9)" not in message["text"]
    assert all("(9)" not in s["value"] for s in segments)
