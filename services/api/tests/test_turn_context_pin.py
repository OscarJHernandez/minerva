"""Pins what the chat route hands the agent loop, turn by turn, for a conversation that exercises every piece of
stored context: verified results and their restrictions, a verified discovery card with named constants, an
unverified model equation, a numbered list component, formula evidence, declarations in words, tool groups and
the numbering of retried and edited turns.

The route is run over HTTP with a scripted model; `run_agent` is wrapped so that everything the route gave it (the
rebuilt history, every `prior_*` argument, the equation table and the tool context) is recorded at the moment the
loop starts, and compared with exact values written below. The conversation's stored rows are written by hand
(the shape the route stores), so the expected values do not depend on what the engine prints.

This is a characterisation of the route as it was before its context assembly became one shared function
(issue 150): the same test must pass against that function, unchanged. A difference here is a behaviour change of
the live chat.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from minerva_api import routers_chat
from minerva_api.agent.events import AssistantMessage, Block, Card, Evidence, Segment, TraceStep
from minerva_api.agent.replay import ReplayLLM, say
from minerva_api.db import ChatMessage, Conversation

from .conftest import build_app, register
from .test_chat_context import send

LAGRANGIAN = r"\frac{\mu}{2}(\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{G M \mu}{r}"
C1 = r"\mu r^{2} \dot{\theta}"
C2 = r"- \frac{G M \mu}{r} + \frac{\mu r^{2} \dot{\theta}^{2}}{2} + \frac{\mu \dot{r}^{2}}{2}"
LONG_RESULT = "\n".join(
    f"line {i} of an earlier result, long enough to be shortened" for i in range(30)
)


def _evidence(**over: Any) -> Evidence:
    fields: dict[str, Any] = {"check": "numeric", "accepted": True, "reason": "verified"}
    fields.update(over)
    return Evidence(**fields)


def _assistant(
    text: str, steps: list[TraceStep], cards: list[Card], segments: list[Segment]
) -> dict[str, Any]:
    blocks = [Block(kind="card", card=c) for c in cards]
    if segments:
        blocks.append(Block(kind="text", segments=segments))
    return AssistantMessage(text=text, blocks=blocks, trace=steps).model_dump(mode="json")


def _legacy(content: dict[str, Any]) -> dict[str, Any]:
    """A message stored before verification was recorded per step: no `verified_latex`, no `quoted`."""
    for step in content["trace"]:
        step.pop("verified_latex", None)
        step.pop("quoted", None)
    return content


def stored_conversation() -> list[tuple[str, dict[str, Any]]]:
    """Eight stored messages; the highest equation number they gave out is 8."""
    simplify = TraceStep(
        id="s1",
        name="simplify",
        call_id="call_s1",
        arguments={"expression": r"\frac{x^3-y^3}{x-y}", "assumptions": {"x": ["positive"]}},
        for_model=LONG_RESULT,
        status="ok",
        evidence=_evidence(restrictions=1),
        result_id="r1",
        verified_latex=["x^{2} + x y + y^{2}"],
        quoted={r"\frac{x^3-y^3}{x-y}": "the expression as you wrote it"},
        formula_roles={"x^{2} + x y + y^{2}": "result"},
        variables=["x", "y"],
        restrictions=[r"x - y \neq 0"],
    )
    c_simplify = Card(
        id="s1",
        tool="simplify",
        type="simplify",
        title="Simplify",
        data={"result_latex": "x^{2} + x y + y^{2}", "restrictions_latex": [r"x - y \neq 0"]},
        number=2,
    )
    differentiate = TraceStep(
        id="s2",
        name="differentiate",
        call_id="call_s2",
        arguments={"expression": "x^{2} + x y + y^{2}", "variable": "x"},
        model_arguments={"expression": "@r1", "variable": "x"},
        for_model="derivative: 2 x + y",
        status="ok",
        evidence=_evidence(restrictions=1),
        result_id="r2",
        verified_latex=["2 x + y"],
        formula_roles={"2 x + y": "result"},
        variables=["x", "y"],
        restrictions=[r"x - y \neq 0"],
    )
    c_differentiate = Card(
        id="s2",
        tool="differentiate",
        type="differentiate",
        title="Differentiate",
        data={"result_latex": "2 x + y", "inherited_restrictions_latex": [r"x - y \neq 0"]},
        number=4,
    )
    from_model = TraceStep(
        id="s3",
        name="differentiate",
        call_id="call_s3",
        arguments={"expression": "x^{3} + 1", "variable": "x"},
        model_arguments={"expression": "@e3", "variable": "x"},
        for_model="derivative: $3 x^{2}$",
        status="ok",
        evidence=_evidence(),
        result_id="r3",
        verified_latex=["3 x^{2}"],
        formula_roles={"3 x^{2}": "result"},
        variables=["x"],
    )
    c_from_model = Card(
        id="s3",
        tool="differentiate",
        type="differentiate",
        title="Differentiate",
        data={
            "result_latex": "3 x^{2}",
            "unverified_inputs": ["equation (3), written by the model and not verified by Minerva"],
        },
        number=5,
    )
    discovery = TraceStep(
        id="s4",
        name="find_conserved_quantities",
        call_id="call_s4",
        arguments={"lagrangian": LAGRANGIAN, "coordinates": ["r", "theta"]},
        for_model="found two constants of motion",
        status="ok",
        evidence=_evidence(check="symbolic_zero", symbolic_zero=True),
        verified_latex=[C1, C2],
        formula_roles={C1: "conserved", C2: "conserved"},
        variables=["r", "theta"],
    )
    c_discovery = Card(
        id="s4",
        tool="find_conserved_quantities",
        type="discovery",
        title="Constants of motion",
        data={
            "report": {
                "system": "lagrangian",
                "function_latex": LAGRANGIAN,
                "coordinates": ["r", "theta"],
                "constants": [
                    {"expression_latex": f"C_{{1}} = {C1}", "verified": True},
                    {"expression_latex": f"C_{{2}} = {C2}", "verified": True},
                ],
            }
        },
        numbers={"const:0": 7, "const:1": 8},
    )
    return [
        (
            "user",
            {
                "text": r"Simplify $$\frac{x^3-y^3}{x-y}$$ for positive $x$.",
                "equations": [{"number": 1, "latex": r"\frac{x^3-y^3}{x-y}"}],
            },
        ),
        (
            "assistant",
            _assistant(
                "The result is $$x^{2} + x y + y^{2}$$ and, say, $$y = x^{3} + 1$$.",
                [simplify],
                [c_simplify],
                [
                    Segment(
                        type="display_math",
                        value="x^{2} + x y + y^{2}",
                        status="from_tool",
                        number=2,
                    ),
                    Segment(
                        type="display_math", value="y = x^{3} + 1", status="unchecked", number=3
                    ),
                ],
            ),
        ),
        (
            "user",
            {"text": "Differentiate that result with respect to $x$. Let A and B be matrices."},
        ),
        (
            "assistant",
            _assistant("Done: $$2 x + y$$.", [differentiate], [c_differentiate], []),
        ),
        ("user", {"text": "Now differentiate equation 3 with respect to $x$."}),
        (
            "assistant",
            _legacy(_assistant("That is $$3 x^{2}$$.", [from_model], [c_from_model], [])),
        ),
        (
            "user",
            {
                "text": rf"Find the constants of motion of $$L = {LAGRANGIAN}$$",
                "equations": [{"number": 6, "latex": f"L = {LAGRANGIAN}"}],
            },
        ),
        ("assistant", _assistant("Two constants were found.", [discovery], [c_discovery], [])),
    ]


def snapshot(history: list[dict[str, Any]], ctx: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """What the route gave the loop, as plain comparable data (sets sorted, objects turned into dicts)."""

    def plain(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return plain(dataclasses.asdict(value))
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, set | frozenset):
            return sorted(plain(v) for v in value)
        if isinstance(value, list | tuple):
            return [plain(v) for v in value]
        return value

    table = kwargs["equations"]
    out = {k: plain(v) for k, v in kwargs.items() if k not in ("info", "equations", "params")}
    out["history"] = plain(copy.deepcopy(history))
    out["equations"] = {
        "counter": table.counter,
        "items": {
            n: [e.latex, e.source, e.result_id, e.part_of, e.alias_of]
            for n, e in sorted(table.items.items())
        },
        "index": dict(sorted(table.index.items())),
    }
    out["params"] = kwargs["params"].model_dump(mode="json")
    out["routing"] = kwargs["info"].routing
    out["ctx"] = {
        "user_messages": plain(ctx.user_messages),
        "constants": plain(ctx.constants),
        "matrices": plain(ctx.matrices),
    }
    return out


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    real = routers_chat.run_agent

    async def spy(
        llm: Any, history: Any, ctx: Any, limits: Any = None, **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        seen.append(snapshot(history, ctx, kwargs))
        async for event in real(llm, history, ctx, limits, **kwargs):
            yield event

    monkeypatch.setattr(routers_chat, "run_agent", spy)
    return seen


@pytest.fixture
def chat() -> Iterator[tuple[TestClient, str, ReplayLLM]]:
    app = build_app(llm_mode="scripted")
    with TestClient(app) as client:
        register(client)
        cid = client.post("/v1/chat/conversations", json={}).json()["id"]
        llm = ReplayLLM([], repeat_last=True)
        client.app.state.local_llm = llm  # type: ignore[attr-defined]
        with client.app.state.session_factory() as session:  # type: ignore[attr-defined]
            for role, content in stored_conversation():
                session.add(ChatMessage(conversation_id=uuid.UUID(cid), role=role, content=content))
            session.get(Conversation, uuid.UUID(cid)).equation_counter = 8  # type: ignore[union-attr]
            session.commit()
        yield client, cid, llm


def counter_of(client: TestClient, cid: str) -> int:
    with client.app.state.session_factory() as session:  # type: ignore[attr-defined]
        return session.get(Conversation, uuid.UUID(cid)).equation_counter  # type: ignore[union-attr,no-any-return]


def stored_user(client: TestClient, cid: str) -> dict[str, Any]:
    messages = client.get(f"/v1/chat/conversations/{cid}").json()["messages"]
    return [m for m in messages if m["role"] == "user"][-1]["content"]  # type: ignore[no-any-return]


def run_conversation(
    client: TestClient, cid: str, llm: ReplayLLM, recorded: list[dict[str, Any]]
) -> dict[str, Any]:
    """The four turns, and everything observable about them, as plain data."""
    llm.replies = [say("Understood. $$u = 2$$")]
    out: dict[str, Any] = {}
    # 1. a new turn: references of every kind, a displayed formula of the user's own, a vector in words
    first = (
        "Let v be a vector. Differentiate @r2 again. Compare equation 7 with equation 2, equation 3, "
        "equation 1 and equation 99. Take $$w = x + 1$$ as given."
    )
    send(client, cid, first)
    out["stored1"], out["counter1"] = stored_user(client, cid), counter_of(client, cid)
    # 2. retry: the reply is replaced, its numbers are never reused, the user's formula is numbered again
    send(client, cid, "again", mode="retry")
    out["stored2"], out["counter2"] = stored_user(client, cid), counter_of(client, cid)
    # 3. edit: references to numbers whose messages were replaced
    send(client, cid, "Use equation 10 again, and equation 11. Let C be a scalar.", mode="edit")
    out["stored3"], out["counter3"] = stored_user(client, cid), counter_of(client, cid)
    # 4. a plain follow-up on the edited conversation
    send(client, cid, "Thanks. And now?")
    out["counter4"] = counter_of(client, cid)
    out["recorded"] = recorded
    out["tools"] = [[s["function"]["name"] for s in t] for t in llm.tools_seen]
    return json.loads(json.dumps(out))  # type: ignore[no-any-return]


GOLDEN = Path(__file__).parent / "fixtures" / "turn_context_pin.json"


def test_the_route_gives_the_loop_exactly_this_for_every_kind_of_turn(
    chat: tuple[TestClient, str, ReplayLLM], recorded: list[dict[str, Any]]
) -> None:
    """Everything the route gave the loop in four turns, against a file written from the route before the
    context was shared. There is no way to regenerate it from this test: a difference is a change of behaviour."""
    client, cid, llm = chat
    out = run_conversation(client, cid, llm, recorded)
    assert out == json.loads(GOLDEN.read_text("utf-8"))


def test_the_pinned_context_says_what_it_should(
    chat: tuple[TestClient, str, ReplayLLM], recorded: list[dict[str, Any]]
) -> None:
    """The same run read for meaning, so the pin is not only a file: each category is non-empty and right."""
    client, cid, llm = chat
    out = run_conversation(client, cid, llm, recorded)
    first, retry, edit, last = out["recorded"]
    # verified constants are read from the stored discovery card, never from text
    assert [c["name"] for c in first["ctx"]["constants"]] == ["C_1", "C_2"]
    assert all(c["verified"] and c["system"] == "lagrangian" for c in first["ctx"]["constants"])
    assert first["ctx"]["constants"] == last["ctx"]["constants"]
    # result handles and the restrictions they carry (own and inherited)
    assert first["prior_results"] == {
        "r1": "x^{2} + x y + y^{2}",
        "r2": "2 x + y",
        "r3": "3 x^{2}",
    }
    assert first["prior_restrictions"] == {"r1": [r"x - y \neq 0"], "r2": [r"x - y \neq 0"]}
    # the unverified model equation a result was computed from
    assert first["prior_unverified"] == {
        "r3": ["equation (3), written by the model and not verified by Minerva"]
    }
    # a message stored before verification was recorded is quoted, never verified
    assert first["prior_quoted"]["3x^2"].startswith("from an earlier tool result")
    assert "3x^2" not in first["prior_known"]
    # evidence for the claim check, tool groups, declarations and assumptions
    assert [row["result_id"] for row in first["prior_evidence"]] == ["r1", "r2", None]
    assert first["prior_groups"] == ["matrix", "mechanics"]
    assert first["prior_matrices"] == ["A", "B"] and first["vector_kind_names"] == ["v"]
    assert edit["scalar_names"] == ["C"] and edit["vector_kind_names"] == []
    assert first["prior_assumptions"] == {"x": ["positive"]}
    # the equation table: the list component is named, the numbers of replaced messages stay used
    assert first["equations"]["items"]["7"][3] == "discovery" and first["equations"]["counter"] == 9
    assert retry["equations"]["counter"] == 11 and "9" not in retry["equations"]["items"]
    assert "10" not in edit["equations"]["items"] and edit["equations"]["counter"] == 12
    assert last["equations"]["counter"] == 13
    assert "no longer exists" in edit["history"][-1]["content"]
    assert "does not exist in this conversation" in first["history"][-1]["content"]
    assert out["counter1"] == 10 and out["counter2"] == 12 and out["counter3"] == 13
    # an old tool result is shortened
    assert any("[shortened: an earlier result]" in m["content"] for m in first["history"])
