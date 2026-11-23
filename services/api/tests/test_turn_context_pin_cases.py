"""More pins of what the chat route hands the agent loop (see `test_turn_context_pin.py`): the paths where the
context is cut, merged or numbered on the way in.

* the replay window: a conversation longer than `history_messages` (and one cut by `history_tokens`), where results,
  restrictions, constants, declarations, evidence and unverified inputs of the first messages live only outside the
  window and must still be read from the whole conversation, while the claim context keeps only the last 20 messages
  and the replayed history starts at a user message;
* dropped replies: two user messages in a row, and a conversation that ends in a user message with no reply;
* `_backfill_numbers` on a conversation that predates equation numbers (counter unset) and the same rows with a
  counter that is already set (then nothing is numbered);
* the table's `reused` and `aliases_resolved` counters, and that the stored rows are never edited in place by a turn.

The expected values are `fixtures/turn_context_pin_cases.json`. It was recorded against the route as it was before
the context became one function (base commit 165fcdd, in a scratch checkout of that commit) and is never regenerated
from the branch: a difference here is a change in behaviour of the live chat.
"""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from minerva_api.agent.events import AssistantMessage, Block, Segment
from minerva_api.agent.replay import ReplayLLM, say
from minerva_api.db import ChatMessage, Conversation
from sqlalchemy import select

from .conftest import build_app, register
from .test_chat_context import send
from .test_turn_context_pin import recorded, snapshot, stored_conversation  # noqa: F401

GOLDEN = Path(__file__).parent / "fixtures" / "turn_context_pin_cases.json"
Row = tuple[str, dict[str, Any]]


def snapshot_with_counters(history: Any, ctx: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    out = snapshot(history, ctx, kwargs)
    table = kwargs["equations"]
    out["equations"]["reused"] = table.reused
    out["equations"]["aliases_resolved"] = table.aliases_resolved
    return out


def plain(role: str, text: str) -> Row:
    if role == "user":
        return role, {"text": text}
    message = AssistantMessage(
        text=text,
        blocks=[Block(kind="text", segments=[Segment(type="text", value=text)])],
        trace=[],
    )
    return role, message.model_dump(mode="json")


def filler(pairs: int, width: int = 0) -> list[Row]:
    pad = " and some more words to fill the line" * width
    rows: list[Row] = []
    for i in range(pairs):
        rows.append(plain("user", f"filler question {i} about $q_{i} = {i}${pad}"))
        rows.append(plain("assistant", f"filler answer {i}{pad}"))
    return rows


def unnumbered(rows: list[Row]) -> list[Row]:
    """The same rows as a conversation stored before equation numbers existed: no numbers anywhere."""
    out = copy.deepcopy(rows)
    for _role, content in out:
        content.pop("equations", None)
        for block in content.get("blocks", []):
            if isinstance(block.get("card"), dict):
                block["card"]["number"] = None
                block["card"]["numbers"] = {}
            for segment in block.get("segments", []):
                segment["number"] = None
    return out


def duplicates() -> list[Row]:
    """A displayed formula stored twice under two numbers (before labels were removed): the second is an alias."""
    both = AssistantMessage(
        text="So $$y = x$$ and again $$y = x$$.",
        blocks=[
            Block(
                kind="text",
                segments=[
                    Segment(type="display_math", value="y = x", status="unchecked", number=1),
                    Segment(type="display_math", value="y = x", status="unchecked", number=2),
                ],
            )
        ],
        trace=[],
    )
    return [
        ("user", {"text": "Say it twice."}),
        ("assistant", both.model_dump(mode="json")),
    ]


def seed(client: TestClient, rows: list[Row], counter: int) -> str:
    cid = client.post("/v1/chat/conversations", json={}).json()["id"]
    with client.app.state.session_factory() as session:  # type: ignore[attr-defined]
        for role, content in rows:
            session.add(ChatMessage(conversation_id=uuid.UUID(cid), role=role, content=content))
        session.get(Conversation, uuid.UUID(cid)).equation_counter = counter  # type: ignore[union-attr]
        session.commit()
    return cid


def stored_rows(client: TestClient, cid: str) -> list[dict[str, Any]]:
    """The stored rows straight from the database (the conversation endpoint would backfill numbers itself)."""
    with client.app.state.session_factory() as session:  # type: ignore[attr-defined]
        rows = session.scalars(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == uuid.UUID(cid))
            .order_by(ChatMessage.created_at, ChatMessage.id)
        )
        return [{"role": m.role, "content": copy.deepcopy(m.content)} for m in rows]


def counter_of(client: TestClient, cid: str) -> int:
    with client.app.state.session_factory() as session:  # type: ignore[attr-defined]
        return session.get(Conversation, uuid.UUID(cid)).equation_counter  # type: ignore[union-attr,no-any-return]


def run(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[Row],
    counter: int,
    turns: list[tuple[str, str]],
    **settings: Any,
) -> dict[str, Any]:
    """Seed the rows, send the turns (`mode`, text), and record what the loop was given and what was stored."""
    from minerva_api import routers_chat

    seen: list[dict[str, Any]] = []
    real = routers_chat.run_agent

    async def spy(llm: Any, history: Any, ctx: Any, limits: Any = None, **kwargs: Any) -> Any:
        seen.append(snapshot_with_counters(history, ctx, kwargs))
        async for event in real(llm, history, ctx, limits, **kwargs):
            yield event

    monkeypatch.setattr(routers_chat, "run_agent", spy)
    app = build_app(llm_mode="scripted", **settings)
    out: dict[str, Any] = {}
    with TestClient(app) as client:
        register(client)
        llm = ReplayLLM([say("Understood.")], repeat_last=True)
        client.app.state.local_llm = llm  # type: ignore[attr-defined]
        cid = seed(client, rows, counter)
        before = stored_rows(client, cid)
        out["counter_seeded"] = counter
        for mode, text in turns:
            send(client, cid, text, mode=mode)
        after = stored_rows(client, cid)
        out["rows_before"] = before
        # The stored rows that were there before the turns: edited in place by a turn only when a backfill
        # numbered them. `unchanged` is recorded, and the numbered rows too when they did change.
        out["rows_unchanged"] = after[: len(before)] == before
        if not out["rows_unchanged"]:
            out["rows_after"] = after[: len(before)]
        out["new_user_rows"] = [r["content"] for r in after[len(before) :] if r["role"] == "user"]
        out["counter_after"] = counter_of(client, cid)
        out["recorded"] = seen
        out["tools"] = [[s["function"]["name"] for s in t] for t in llm.tools_seen]
    return json.loads(json.dumps(out))  # type: ignore[no-any-return]


EARLY = stored_conversation  # eight rows: @r1/@r2/@r3, restrictions, constants, declarations, evidence, numbers 1-8
LONG_TURN = "Reuse @r1 and @r2 with equation 7 and equation 3. Let C be a scalar."


def scenario_window_default(mp: pytest.MonkeyPatch) -> dict[str, Any]:
    # 8 early rows + 18 filler = 26 stored messages: the first 6 are beyond the claim context's last 20.
    return run(mp, [*EARLY(), *filler(9)], 8, [("new", LONG_TURN)])


def scenario_window_messages(mp: pytest.MonkeyPatch) -> dict[str, Any]:
    # An odd window: its first message is an assistant message, so the history starts one message later.
    return run(mp, [*EARLY(), *filler(9)], 8, [("new", LONG_TURN)], llm_history_messages=7)


def scenario_window_tokens(mp: pytest.MonkeyPatch) -> dict[str, Any]:
    return run(mp, [*EARLY(), *filler(9, width=6)], 8, [("new", LONG_TURN)], llm_history_tokens=400)


def scenario_window_retry_and_edit(mp: pytest.MonkeyPatch) -> dict[str, Any]:
    turns = [("new", LONG_TURN), ("retry", "again"), ("edit", "Use equation 4. Let C be a scalar.")]
    return run(mp, [*EARLY(), *filler(9)], 8, turns, llm_history_messages=7)


def scenario_dropped_replies(mp: pytest.MonkeyPatch) -> dict[str, Any]:
    # u1 a1 | u2 u3 (u2 never got a reply) a3 | u4 (no reply at all). The new message merges into u4.
    rows = [
        *duplicates(),
        plain("user", "A question whose reply was lost, with $x = 1$."),
        plain("user", "So I ask again, with $y = 2$."),
        plain("assistant", "The reply to the second one."),
        plain("user", "And one more whose reply was lost."),
    ]
    text = "Now say $$y = x$$ in terms of equation 2, and equation 9."
    return run(mp, rows, 2, [("new", text)])


def scenario_dropped_reply_at_the_end_with_retry(mp: pytest.MonkeyPatch) -> dict[str, Any]:
    rows = [*duplicates(), plain("user", "A first question."), plain("user", "A second question.")]
    return run(
        mp, rows, 2, [("new", "Third, with $$z = 1$$."), ("retry", "x"), ("edit", "Fourth.")]
    )


def scenario_backfill_unset_counter(mp: pytest.MonkeyPatch) -> dict[str, Any]:
    rows = unnumbered(EARLY())
    return run(
        mp, rows, 0, [("new", "Compare equation 2, equation 3 and equation 6. Take $$w = 1$$.")]
    )


def scenario_backfill_skipped_when_a_counter_is_set(mp: pytest.MonkeyPatch) -> dict[str, Any]:
    rows = unnumbered(EARLY())
    return run(
        mp, rows, 1, [("new", "Compare equation 2, equation 3 and equation 6. Take $$w = 1$$.")]
    )


SCENARIOS: dict[str, Callable[[pytest.MonkeyPatch], dict[str, Any]]] = {
    f.__name__.removeprefix("scenario_"): f
    for f in (
        scenario_window_default,
        scenario_window_messages,
        scenario_window_tokens,
        scenario_window_retry_and_edit,
        scenario_dropped_replies,
        scenario_dropped_reply_at_the_end_with_retry,
        scenario_backfill_unset_counter,
        scenario_backfill_skipped_when_a_counter_is_set,
    )
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_route_gives_the_loop_exactly_this(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Against a file recorded from the route before the context was shared; it cannot be regenerated here."""
    assert SCENARIOS[name](monkeypatch) == json.loads(GOLDEN.read_text("utf-8"))[name]


def test_the_pinned_cases_say_what_they_should(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same runs read for meaning, so the pins are not only a file."""
    golden = json.loads(GOLDEN.read_text("utf-8"))
    # the window: everything verified in the first messages is still known, though the history left it out
    default = golden["window_default"]["recorded"][0]
    assert default["prior_results"]["r1"] == "x^{2} + x y + y^{2}"
    assert default["prior_restrictions"] == {"r1": [r"x - y \neq 0"], "r2": [r"x - y \neq 0"]}
    assert [c["name"] for c in default["ctx"]["constants"]] == ["C_1", "C_2"]
    assert default["prior_matrices"] == ["A", "B"] and "r3" in default["prior_unverified"]
    assert len(default["prior_evidence"]) == 3
    assert not any("Simplify" in m["content"] for m in default["history"] if m["role"] == "user")
    assert default["prior_assumptions"] == {}  # the claim context keeps only the last 20 messages
    odd = golden["window_messages"]["recorded"][0]["history"]
    assert (
        odd[0]["role"] == "user" and len(odd) == 7
    )  # 7 asked for, one is dropped to start at a user message, plus the new one
    assert len(golden["window_tokens"]["recorded"][0]["history"]) < len(default["history"])
    # dropped replies: two user messages merge into one, and the new text joins a last message with no reply
    merged = golden["dropped_replies"]["recorded"][0]["history"]
    assert [m["role"] for m in merged] == ["user", "assistant", "user", "assistant", "user"]
    assert "reply was lost" in merged[-1]["content"] and "Now say" in merged[-1]["content"]
    assert golden["dropped_replies"]["recorded"][0]["equations"]["aliases_resolved"] == 1
    assert golden["dropped_replies"]["recorded"][0]["equations"]["reused"] == 1
    # backfill numbers rows that predate numbers (once, in place), and leaves them alone when a counter exists
    unset = golden["backfill_unset_counter"]
    assert unset["rows_unchanged"] is False and unset["recorded"][0]["equations"]["counter"] >= 8
    skipped = golden["backfill_skipped_when_a_counter_is_set"]
    assert skipped["rows_unchanged"] is True
    assert (
        skipped["recorded"][0]["equations"]["counter"]
        < unset["recorded"][0]["equations"]["counter"]
    )
    for name in golden:
        if not name.startswith("backfill"):
            assert golden[name]["rows_unchanged"] is True, (
                name
            )  # a turn never edits stored rows in place
