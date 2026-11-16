"""Equation numbers: one counter per conversation, references in the user's message, and `@eN` in tool calls."""

import asyncio
import copy
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from minerva_api.agent.equations import (
    EquationTable,
    build_table,
    find_references,
    number_stored,
    number_user_text,
    replay_prose,
    resolve_references,
    unverified_inputs,
)
from minerva_api.agent.events import Block, Card, Segment
from minerva_api.agent.loop import run_agent
from minerva_api.agent.refs import ResultStore
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

from .conftest import build_app, register
from .test_chat_context import send, start


class Row:
    def __init__(self, role: str, content: dict[str, Any]) -> None:
        self.role, self.content = role, content


# --- numbering ------------------------------------------------------------------------------------------
def display(value: str, status: str = "unchecked") -> Segment:
    return Segment(type="display_math", value=value, status=status)  # type: ignore[arg-type]


def test_numbers_follow_reading_order_and_a_formula_keeps_its_number() -> None:
    table = EquationTable()
    card = Card(
        id="c1", tool="simplify", type="simplify", title="t", data={"result_latex": "x + 1"}
    )
    blocks = [
        Block(kind="text", segments=[display("y^2"), Segment(type="math", value="z")]),
        Block(kind="card", card=card),
        Block(kind="text", segments=[display("x + 1", "from_tool"), display("y^2")]),
    ]
    from minerva_api.agent.equations import number_blocks

    number_blocks(blocks, table, {"c1": "r1"})
    assert (
        blocks[0].segments[0].number == 1 and blocks[0].segments[1].number is None
    )  # inline: no number
    assert card.number == 2 and table.items[2].result_id == "r1"
    assert blocks[2].segments[0].number == 2  # the card's own result shown again: the same equation
    assert blocks[2].segments[1].number == 1  # and so is a repeated formula
    assert table.counter == 2


def test_a_long_formula_is_shown_but_not_numbered() -> None:
    assert EquationTable().number_for("x+" * 400, "written_by_model") is None
    assert EquationTable().number_for("   ", "written_by_model") is None


def test_the_users_displayed_formulas_are_numbered_in_order() -> None:
    table = EquationTable()
    found = number_user_text("Take $$x+1$$ and $y$ and then $$y^2$$ and $$x+1$$", table)
    assert [(e["number"], e["latex"]) for e in found] == [(1, "x+1"), (2, "y^2"), (1, "x+1")]
    assert table.items[1].source == "yours"


def test_the_stored_high_water_mark_is_never_lowered() -> None:
    table = build_table([], counter=7)  # messages were removed; the numbers up to 7 were given
    assert table.number_for("a", "written_by_model") == 8


def test_the_table_is_rebuilt_from_stored_messages() -> None:
    rows = [
        Row("user", {"text": "t", "equations": [{"number": 1, "latex": "u"}]}),
        Row(
            "assistant",
            {
                "text": "",
                "trace": [{"id": "s1", "result_id": "r1"}],
                "blocks": [
                    {
                        "kind": "card",
                        "segments": [],
                        "card": {"id": "s1", "number": 2, "data": {"result_latex": "x + 1"}},
                    },
                    {
                        "kind": "text",
                        "segments": [
                            {
                                "type": "display_math",
                                "value": "q",
                                "status": "unchecked",
                                "number": 3,
                            }
                        ],
                    },
                ],
            },
        ),
    ]
    table = build_table(rows)
    assert table.counter == 3
    assert (table.items[1].source, table.items[2].result_id, table.items[3].source) == (
        "yours",
        "r1",
        "written_by_model",
    )


def test_a_conversation_without_numbers_is_numbered_in_reading_order() -> None:
    contents = [
        ("user", {"text": "Start from $$a$$"}),
        (
            "assistant",
            {
                "trace": [{"id": "s1", "result_id": "r1"}],
                "blocks": [
                    {
                        "kind": "text",
                        "segments": [{"type": "display_math", "value": "b", "status": None}],
                    },
                    {
                        "kind": "card",
                        "segments": [],
                        "card": {"id": "s1", "data": {"result_latex": "c"}},
                    },
                ],
            },
        ),
    ]
    original = copy.deepcopy(contents)
    table = EquationTable()
    assert number_stored(contents, table)
    assert contents[0][1]["equations"] == [{"number": 1, "latex": "a"}]
    assert contents[1][1]["blocks"][0]["segments"][0]["number"] == 2
    assert contents[1][1]["blocks"][1]["card"]["number"] == 3 and table.items[3].result_id == "r1"
    again = EquationTable()
    redo = copy.deepcopy(original)
    number_stored(redo, again)
    assert redo == contents  # deterministic: the same numbers every time
    assert not number_stored([("user", {"text": "no formulas"})], EquationTable())


# --- references in the user's message ------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("Integrate equation 3 please", [3]),
        ("integrate eq. (3)", [3]),
        ("Use Eq 2 and eqn. 5", [2, 5]),
        ("Integrate (3)", [3]),
        ("substitute (2) into (5)", [2, 5]),
        ("what is f(3)?", []),  # a function value
        ("evaluate at x_(3)", []),
        ("(1) first thing\n(2) second thing", []),  # list markers
        ("(3)", [3]),  # a lone number is how the equation is displayed
        ("take $x^{(3)}$ and $(4)$", []),  # inside formulas
        ("the value 2(3) is 6", []),
        ("equation 3 and again (3)", [3]),
    ],
)
def test_references_are_found_only_where_they_are_explicit(text: str, numbers: list[int]) -> None:
    assert [r.number for r in find_references(text)] == numbers


def test_resolving_references_tells_what_each_was_taken_to_be() -> None:
    table = EquationTable()
    table.number_for("x + 1", "from_tool", "r1")
    table.number_for("y^2", "written_by_model")
    table.number_for("z", "yours")
    shown, note = resolve_references(
        find_references("add equation 1 to (2) and (3), see eq. 9"), table
    )
    assert [(s["number"], s["source"]) for s in shown] == [
        (1, "from_tool"),
        (2, "written_by_model"),
        (3, "yours"),
        (9, "unknown"),
    ]
    assert "@r1 or @e1" in note and "NOT verified" in note and "(9) does not exist" in note
    assert "not from the user" in note
    # a lone (7) that is no equation is not an error: it may be anything the user meant
    shown, note = resolve_references(find_references("walk to (7)"), table)
    assert shown == [] and note == ""


def test_the_model_sees_the_number_after_each_numbered_formula() -> None:
    content = {
        "text": "Intro\n\n$$x^2 + 1$$\n\nThen \\[y\\] and\n$$z$$ end.",
        "blocks": [
            {
                "kind": "text",
                "segments": [
                    {"type": "display_math", "value": "x^2 + 1", "number": 1},
                    {"type": "display_math", "value": "y", "number": 2},
                    {"type": "display_math", "value": "z", "number": None},
                ],
            }
        ],
    }
    assert replay_prose(content) == "Intro\n\n$$x^2 + 1$$ (1)\n\nThen \\[y\\] (2) and\n$$z$$ end."
    assert replay_prose({"text": "plain", "blocks": []}) == "plain"


# --- @eN in tool arguments -----------------------------------------------------------------------------------
def store_with_equations() -> ResultStore:
    table = EquationTable()
    table.number_for("x + 1", "from_tool", "r1")
    table.number_for("y^2", "written_by_model")
    table.number_for("z", "yours")
    return ResultStore.from_prior(
        {"r1": "x + 1"}, {"r1": [r"x \neq 1"]}, table.items, unverified={}
    )


def test_a_verified_equation_resolves_through_its_result_and_carries_its_restrictions() -> None:
    r = store_with_equations().resolve({"latex": "@e1", "variables": [{"name": "x"}]})
    assert r.error is None and r.arguments["latex"] == "x + 1"
    assert r.inherited == [r"x \neq 1"] and r.unverified == []


def test_an_equation_the_model_wrote_resolves_but_is_marked_unverified() -> None:
    r = store_with_equations().resolve({"latex": "@e2 + 1"})
    assert r.arguments["latex"] == r"\left(y^2\right) + 1"
    assert r.unverified == ["equation (2)"]
    whole = store_with_equations().resolve({"latex": "@e2"})
    assert whole.arguments["latex"] == "y^2"  # the whole argument: exactly the equation
    mine = store_with_equations().resolve({"latex": "@e3"})
    assert (
        mine.arguments["latex"] == "z" and mine.unverified == []
    )  # the user's own input is theirs


def test_an_unknown_equation_is_an_error_not_a_guess() -> None:
    r = store_with_equations().resolve({"latex": "@e9"})
    assert r.error and "unknown equation reference @e9" in r.error and "(1), (2), (3)" in r.error


def test_what_depends_on_an_unverified_equation_stays_marked_when_reused() -> None:
    store = store_with_equations()
    store.unverified["r4"] = ["equation (2)"]
    store.results["r4"] = "q"
    assert store.resolve({"latex": "@r4 + 1"}).unverified == ["equation (2)"]


# --- in the loop -------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def play(llm: ReplayLLM, ctx: ToolContext, text: str, **kw: Any) -> dict[str, Any]:
    async def go() -> list[dict[str, Any]]:
        return [e async for e in run_agent(llm, [{"role": "user", "content": text}], ctx, **kw)]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


def table_with_model_equation() -> EquationTable:
    table = EquationTable()
    table.number_for(r"x^{2} + 2 x", "written_by_model")
    return table


def test_a_result_computed_from_an_unverified_equation_says_so_and_is_not_marked_verified(
    ctx: ToolContext,
) -> None:
    llm = ReplayLLM(
        [
            call("differentiate", {"latex": "@e1", "variables": [{"name": "x"}]}),
            say("It is $2 x + 2$."),
        ]
    )
    message = play(llm, ctx, "differentiate equation 1", equations=table_with_model_equation())
    step = message["trace"][0]
    card = next(b["card"] for b in message["blocks"] if b["kind"] == "card")
    assert step["status"] == "ok" and card["data"]["unverified_inputs"] == ["equation (1)"]
    assert "Input not verified" in step["for_model"] and "equation (1)" in step["for_model"]
    assert step["verified_latex"] == [] and step["quoted"]  # formulas are quoted, never verified
    prose = [
        s
        for b in message["blocks"]
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] != "text"
    ]
    assert prose and all(s["status"] != "from_tool" for s in prose)
    assert step["result_id"] == "r1"
    # the computation was still checked: what is unestablished is the input, and the card carries that
    assert step["evidence"]["accepted"] is True


def test_the_same_chain_from_a_verified_equation_stays_verified(ctx: ToolContext) -> None:
    table = EquationTable()
    table.number_for(r"x^{2} + 2 x", "from_tool", "r1")
    llm = ReplayLLM(
        [
            call("differentiate", {"latex": "@e1", "variables": [{"name": "x"}]}),
            say("It is $2 x + 2$."),
        ]
    )
    message = play(
        llm, ctx, "differentiate equation 1", equations=table, prior_results={"r1": r"x^{2} + 2 x"}
    )
    card = next(b["card"] for b in message["blocks"] if b["kind"] == "card")
    assert "unverified_inputs" not in card["data"]
    assert message["trace"][0]["verified_latex"]
    prose = [
        s
        for b in message["blocks"]
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] == "math"
    ]
    assert prose and prose[0]["status"] == "from_tool"


def test_provenance_travels_with_the_result_into_later_calls(ctx: ToolContext) -> None:
    llm = ReplayLLM(
        [
            call("differentiate", {"latex": "@e1", "variables": [{"name": "x"}]}),
            call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}),
            say("done"),
        ]
    )
    message = play(llm, ctx, "go", equations=table_with_model_equation())
    second = message["trace"][1]
    assert "Input not verified" in second["for_model"]  # inherited through @r1
    assert second["verified_latex"] == []
    assert unverified_inputs([Row("assistant", message)]) == {
        "r1": ["equation (1)"],
        "r2": ["equation (1)"],
    }


def test_numbers_are_given_after_the_turn_and_the_model_cannot_choose_them(
    ctx: ToolContext,
) -> None:
    table = EquationTable()
    llm = ReplayLLM(
        [
            call("differentiate", {"latex": "x^3", "variables": [{"name": "x"}]}),
            say("As in equation (7):\n\n$$3 x^{2}$$\n\nand $$q$$"),
        ]
    )
    message = play(llm, ctx, "differentiate", equations=table)
    cards = [b["card"] for b in message["blocks"] if b["kind"] == "card"]
    segments = [s for b in message["blocks"] if b["kind"] == "text" for s in b["segments"]]
    assert cards[0]["number"] == 1
    shown = [s for s in segments if s["type"] == "display_math"]
    assert [s["number"] for s in shown] == [1, 2]  # the first is the card's own result again
    assert table.counter == 2


# --- over HTTP -------------------------------------------------------------------------------------------------
def test_equations_are_numbered_stored_and_referred_back_to_across_turns() -> None:
    for client, cid, llm in start():
        llm.replies = [say("Here it is:\n\n$$x^2 + 1$$\n\nDone."), say("Reply two.")]
        first = send(client, cid, "show me a quadratic")
        shown = next(d for e, d in first if e == "message")
        number = [
            s["number"]
            for b in shown["blocks"]
            for s in b["segments"]
            if s["type"] == "display_math"
        ]
        assert number == [1]
        second = send(client, cid, "Now take equation 1 and double it, and also $$y = 2$$")
        assert next(d for e, d in second if e == "message")
        messages = client.get(f"/v1/chat/conversations/{cid}").json()["messages"]
        user = messages[2]["content"]
        assert user["references"] == [
            {"number": 1, "latex": "x^2 + 1", "source": "written_by_model"}
        ]
        assert user["equations"] == [{"number": 2, "latex": "y = 2"}]
        assert user["text"].endswith("$$y = 2$$")  # stored as typed, without the note
        seen = llm.seen[1]
        replayed = next(m for m in seen if m["role"] == "assistant")
        assert "$$x^2 + 1$$ (1)" in replayed["content"]
        last_user = seen[-1]["content"]
        assert "(1) = $x^2 + 1$" in last_user and "NOT verified" in last_user
        assert "not from the user" in last_user
        assert "(1) = $x^2" not in user["text"]


def test_the_counter_survives_and_is_never_reused() -> None:
    from minerva_api.db import Conversation

    for client, cid, llm in start():
        llm.replies = [say("$$a$$"), say("$$b$$")]
        send(client, cid, "one")
        send(client, cid, "two")
        with client.app.state.session_factory() as session:  # type: ignore[attr-defined]
            assert session.get(Conversation, uuid.UUID(cid)).equation_counter == 2
        reply = send(client, cid, "three")  # still numbers 3 onwards, whatever was stored before
        assert reply
        messages = client.get(f"/v1/chat/conversations/{cid}").json()["messages"]
        numbers = [
            s["number"]
            for m in messages
            if m["role"] == "assistant"
            for b in m["content"]["blocks"]
            for s in b["segments"]
            if s["type"] == "display_math"
        ]
        assert numbers == [
            1,
            2,
            2,
        ]  # the last reply shows $$b$$ again: the same equation, the same number


def test_an_unknown_equation_is_reported_to_the_user_and_the_model() -> None:
    for client, cid, llm in start():
        llm.replies = [say("ok")]
        send(client, cid, "integrate equation 5")
        user = client.get(f"/v1/chat/conversations/{cid}").json()["messages"][0]["content"]
        assert user["references"] == [{"number": 5, "latex": None, "source": "unknown"}]
        assert "does not exist" in llm.seen[0][-1]["content"]


def test_a_conversation_from_before_equation_numbers_is_numbered_when_it_is_read() -> None:
    from minerva_api.db import ChatMessage, Conversation

    app = build_app(llm_mode="scripted")
    with TestClient(app) as client:
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        with app.state.session_factory() as session:
            for role, content in (
                ("user", {"text": "Start from $$a$$"}),
                (
                    "assistant",
                    {
                        "text": "$$b$$",
                        "trace": [],
                        "blocks": [
                            {
                                "kind": "text",
                                "segments": [
                                    {"type": "display_math", "value": "b", "status": "unchecked"}
                                ],
                            }
                        ],
                    },
                ),
            ):
                session.add(ChatMessage(conversation_id=uuid.UUID(cid), role=role, content=content))
            session.commit()
        first = client.get(f"/v1/chat/conversations/{cid}").json()["messages"]
        again = client.get(f"/v1/chat/conversations/{cid}").json()["messages"]
        assert first[0]["content"]["equations"] == [{"number": 1, "latex": "a"}]
        assert first[1]["content"]["blocks"][0]["segments"][0]["number"] == 2
        assert first == again  # written back: the numbers never change
        with app.state.session_factory() as session:
            assert session.get(Conversation, uuid.UUID(cid)).equation_counter == 2


# --- the server and the client read formulas the same way (AGENTS.md: keep their delimiter rules identical) ---
def delimiter_cases() -> list[dict[str, Any]]:
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "packages"
        / "ui"
        / "fixtures"
        / "math-delimiters.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["cases"]  # type: ignore[no-any-return]


@pytest.mark.parametrize("case", delimiter_cases(), ids=lambda c: c["name"])
def test_the_server_splitter_matches_the_shared_fixture(case: dict[str, Any]) -> None:
    from minerva_api.agent.segments import split_segments

    found = [[s.type, s.value] for s in split_segments(case["text"], set()) if s.type != "text"]
    assert found == case["formulas"]


def test_a_markdown_table_row_is_never_taken_for_an_equation() -> None:
    """`| x = 1 |` is two bars around a cell: a math segment with the bars in it would not split."""
    from minerva_api.agent.segments import split_segments

    text = "Before\n| Feature | Value |\n| :--- | :--- |\n| x = 1 | y = 2 |\n\nThen x = 3 holds."
    parts = split_segments(text, set(), {})
    math = [s.value for b in [parts] for s in b if s.type == "math"]
    assert all("|" not in value for value in math)
    assert "x = 3" in math  # an equation outside the table is still found
    assert "x = 1" not in math and "y = 2" not in math
    # the table text itself arrives unchanged
    joined = "".join(s.value for s in parts if s.type == "text")
    assert "| x = 1 | y = 2 |" in joined
