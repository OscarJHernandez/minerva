"""A number the model writes itself next to a formula is never kept: the server numbers equations."""

import asyncio
from typing import Any

import pytest
from minerva_api.agent.equations import EquationTable, strip_equation_tags, strip_label
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
    assert strip_equation_tags(written)[0] == kept


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


# --- a label written inside the formula (issue 134) ----------------------------------------------------


@pytest.mark.parametrize(
    ("body", "mode", "kept", "removed"),
    [
        ("E = mc^2 (1)", "model", "E = mc^2", "(1)"),
        ("F=ma (3)", "model", "F=ma", "(3)"),
        ("F = ma \\tag{3}", "model", "F = ma", "\\tag{3}"),
        ("F = ma \\tag*{3}", "model", "F = ma", "\\tag*{3}"),
        ("F = ma \\quad (3)", "model", "F = ma", "\\quad (3)"),
        ("F = ma \\qquad \\text{(3)}", "model", "F = ma", "\\qquad \\text{(3)}"),
        ("a = b \\quad (1).", "model", "a = b.", "\\quad (1)."),
        ("T = \\pi \\sqrt{\\frac{R}{g}} (2)", "model", "T = \\pi \\sqrt{\\frac{R}{g}}", "(2)"),
        ("y = R(1 - \\cos\\theta) (1)", "model", "y = R(1 - \\cos\\theta)", "(1)"),
        ("y = g (1)", "model", "y = g", "(1)"),  # the accepted risk: new to the conversation
        ("x \\tag{1}", "user", "x", "\\tag{1}"),
        ("x \\quad (1)", "user", "x", "\\quad (1)"),
        (
            "\\begin{equation} F = ma \\tag{3} \\end{equation}",
            "model",
            "\\begin{equation} F = ma \\end{equation}",
            "\\tag{3}",
        ),
    ],
)
def test_a_label_written_inside_the_formula_is_read_off(
    body: str, mode: Any, kept: str, removed: str
) -> None:
    assert strip_label(body, mode) == (kept, removed)


@pytest.mark.parametrize(
    ("body", "mode"),
    [
        ("f(1)", "model"),
        ("(1+x)", "model"),
        ("2(3)", "model"),
        ("x \\cdot (2)", "model"),
        ("\\sin (2)", "model"),
        ("\\Gamma (3)", "model"),
        ("a = (2)", "model"),
        ("f(x) + f (1)", "model"),
        ("\\int_{0}^{(1)} x", "model"),
        ("\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}", "model"),
        ("\\begin{aligned} a &= b \\quad (1) \\\\ c &= d \\end{aligned}", "model"),
        ("x (2)", "user"),  # a factor in the user's own formula
        ("x (2)", "none"),
        ("x \\tag{3}", "none"),
        ("(1)", "model"),
    ],
)
def test_mathematics_is_never_read_as_a_label(body: str, mode: Any) -> None:
    assert strip_label(body, mode) == (body, "")


def test_the_text_pass_reads_every_display_form_and_leaves_the_rest() -> None:
    text = "a $$c (2)$$ b \\[d \\tag{4}\\] c \\begin{equation}F\\end{equation} (5) e $x$ (3)."
    cleaned, removed = strip_equation_tags(text)
    assert cleaned == "a $$c$$ b \\[d\\] c \\begin{equation}F\\end{equation} e $x$ (3)."
    assert removed == ["(2)", "\\tag{4}", "(5)"]


def test_a_formula_numbered_as_written_is_left_alone() -> None:
    cleaned, removed = strip_equation_tags("$$q (2)$$ and $$p (2)$$", lambda body: body == "q (2)")
    assert cleaned == "$$q (2)$$ and $$p$$"
    assert removed == ["(2)"]


def _turn(ctx: ToolContext, table: EquationTable, reply: str, user: str = "go") -> dict[str, Any]:
    async def go() -> list[dict[str, Any]]:
        return [
            e
            async for e in run_agent(
                ReplayLLM([say(reply)]),
                [{"role": "user", "content": user}],
                ctx,
                equations=table,
            )
        ]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


def _numbers(message: dict[str, Any]) -> list[int | None]:
    return [
        s["number"]
        for b in message["blocks"]
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] == "display_math"
    ]


@pytest.mark.parametrize(
    "restated",
    [
        "$$F = ma (3)$$",
        "$$F = ma$$ (3)",
        "$$F = ma \\tag{3}$$",
        "$$F = ma \\quad (3)$$",
        "\\begin{equation} F = ma \\tag{3} \\end{equation}",
        "$$F = ma$$",
    ],
)
def test_one_formula_keeps_one_number_whatever_label_the_model_adds(
    ctx: ToolContext, restated: str
) -> None:
    table = EquationTable()
    first = _turn(ctx, table, "Newton:\n\n$$F = ma$$")
    again = _turn(ctx, table, f"Again:\n\n{restated}")
    assert _numbers(first) == [1]
    assert _numbers(again) == [1] or restated.startswith("\\begin")
    assert table.counter == (2 if restated.startswith("\\begin") else 1)
    assert "(3)" not in again["text"] and "tag" not in again["text"]
    assert all(
        "(3)" not in s["value"]
        for b in again["blocks"]
        if b["kind"] == "text"
        for s in b["segments"]
    )


def test_every_removal_is_recorded_in_the_debug_and_the_trace_counts(ctx: ToolContext) -> None:
    table = EquationTable()
    message = _turn(ctx, table, "$$a = b (1)$$ then $$a = b$$ (2) and $$a = b \\tag{3}$$")
    assert _numbers(message) == [1, 1, 1]
    assert message["debug"]["removed_labels"] == ["(1)", "(2)", "\\tag{3}"]
    completed = next(e for e in message["debug"]["events"] if e["event"] == "agent.completed")
    assert completed["equation_labels_removed"] == 3
    assert completed["equations_reused"] == 2
    assert "a = b" not in str(completed)  # counts only, never a formula


def test_the_reported_conversation_keeps_one_number_per_unchanged_formula(
    ctx: ToolContext,
) -> None:
    """Issue 134: the cycloid was (2), (3) and (7), the tautochrone integral (5) and (8)."""
    brachistochrone = "T = \\int_0^{x_1} \\frac{\\sqrt{1+y'^2}}{\\sqrt{2 g y}}\\, dx"
    cycloid = "x(\\theta) = R(\\theta - \\sin\\theta), \\quad y(\\theta) = R(1 - \\cos\\theta)"
    period = "T = \\pi \\sqrt{\\frac{R}{g}}"
    integral = "T(y_0) = \\int_{0}^{y_0} \\frac{\\sqrt{1 + (dy)^2}}{\\sqrt{2g(y_0 - y)}} \\, dy"
    constant = "T(y_0) = T_0"
    table = EquationTable()
    first = _turn(ctx, table, f"$$ {brachistochrone} $$\n\n$$ {cycloid} $$")
    second = _turn(ctx, table, f"$${cycloid} (1)$$\n\n$${period} (2)$$")
    third = _turn(ctx, table, f"$${integral} (1)$$\n\n$${constant} (2)$$\n\n$${cycloid} (3)$$")
    fourth = _turn(ctx, table, f"$${brachistochrone}$$\n\n$${integral}$$")
    assert _numbers(first) == [1, 2]
    assert _numbers(second) == [2, 3]  # the cycloid is (2) again; only the period is new
    assert _numbers(third) == [4, 5, 2]
    assert _numbers(fourth) == [1, 4]  # the integral without the model's label is still (4)
    assert table.counter == 5
    # A changed formula is a new equation, not a repeat.
    changed = _turn(ctx, table, f"$${integral.replace('dy)^2', 'dx)^2')}$$")
    assert _numbers(changed) == [6]
