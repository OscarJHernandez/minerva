"""The follow-up scenario of the bug report, over HTTP: later turns must reach the model with the earlier ones."""

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from minerva_api.agent.llm import Chunk
from minerva_api.agent.replay import ReplayLLM, call, say

from .conftest import build_app, register
from .test_chat import sse


def send(client: TestClient, cid: str, content: str, **body: Any) -> list[tuple[str, dict]]:  # type: ignore[type-arg]
    events: list[tuple[str, dict]] = []  # type: ignore[type-arg]
    with client.stream(
        "POST",
        f"/v1/chat/conversations/{cid}/messages",
        json={"content": content, **body},
    ) as response:
        if response.status_code != 200:
            response.read()
            return [("http", {"status": response.status_code, "body": response.json()})]
        event = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                events.append((event, json.loads(line[6:])))
    return events


def start(**settings: Any) -> Iterator[tuple[TestClient, str, ReplayLLM]]:
    app = build_app(llm_mode="scripted", **settings)
    with TestClient(app) as client:
        register(client)
        llm = ReplayLLM([], repeat_last=True)
        client.app.state.local_llm = llm  # type: ignore[attr-defined]
        yield client, client.post("/v1/chat/conversations", json={}).json()["id"], llm


@pytest.fixture
def chat() -> Iterator[tuple[TestClient, str, ReplayLLM]]:
    yield from start()


def test_a_follow_up_reaches_the_model_with_the_earlier_turns(
    chat: tuple[TestClient, str, ReplayLLM],
) -> None:
    client, cid, llm = chat
    llm.replies = [say("Lagrangian reply."), say("Second reply.")]
    sse(client, cid, "Principle of least action for a falling body")
    sse(client, cid, "Ok, lets assume v_0 = 0 and just give me the general integral")
    second = llm.seen[1]
    roles = [m["role"] for m in second]
    assert roles == ["system", "user", "assistant", "user"]
    assert "least action for a falling body" in second[1]["content"]
    assert second[2]["content"] == "Lagrangian reply."
    assert "v_0 = 0" in second[3]["content"]
    assert "continuing conversation" in second[0]["content"]  # and it is told so


def test_tool_groups_follow_the_conversation_not_only_the_last_message(
    chat: tuple[TestClient, str, ReplayLLM],
) -> None:
    client, cid, llm = chat
    llm.replies = [say("ok")]
    sse(client, cid, "Let A and B be 2x2 matrices")  # loads the matrix group
    sse(client, cid, "thanks, and what next?")  # no keyword of its own
    first = {s["function"]["name"] for s in llm.tools_seen[0]}
    second = {s["function"]["name"] for s in llm.tools_seen[1]}
    assert "matrix_calculus" in first and "matrix_calculus" in second  # only ever grows
    assert "statistical_mechanics" not in second


def test_a_long_conversation_is_trimmed_for_the_window_and_the_reply_says_so() -> None:
    for client, cid, llm in start(llm_num_ctx=2048):
        llm.replies = [say("short")]
        big = "integrate " + "x + " * 900  # about 1000 tokens, within the 4000 character limit
        for i in range(3):
            send(client, cid, f"{big}{i}")
        events = send(client, cid, "and the last one")
        message = next(d for e, d in events if e == "message")
        assert message["context_trimmed"] > 0
        assert message["debug"]["context_window"] == 2048
        assert llm.seen[-1][-1]["content"] == "and the last one"
        stored = client.get(f"/v1/chat/conversations/{cid}").json()["messages"][-1]["content"]
        assert stored["context_trimmed"] == message["context_trimmed"]


def test_status_reports_defaults_and_whether_the_smallest_request_fits() -> None:
    for client, _cid, _llm in start():
        status = client.get("/v1/chat/status").json()
        assert status["context_ok"] is True
        assert status["params"]["defaults"]["num_ctx"] == 32768
        assert status["params"]["editable"] == []
    for client, _cid, _llm in start(llm_num_ctx=2048, llm_toolset="all"):
        assert client.get("/v1/chat/status").json()["context_ok"] is False  # all tools never fit 2k


def test_overrides_are_refused_unless_enabled_and_then_recorded() -> None:
    for client, cid, llm in start():
        llm.replies = [say("ok")]
        denied = send(client, cid, "hi", params={"temperature": 0.9})
        assert denied[0][0] == "http" and denied[0][1]["status"] == 403
        ctx_denied = send(client, cid, "hi", params={"num_ctx": 4096})
        assert ctx_denied[0][1]["status"] == 403
        assert send(client, cid, "hi", params={"temperature": 9})[0][1]["status"] == 422
    for client, cid, llm in start(llm_tuning=True):
        llm.replies = [say("ok")]
        assert client.get("/v1/chat/status").json()["params"]["editable"][0] == "num_predict"
        events = send(client, cid, "hi", params={"temperature": 0.9, "toolset": "core"})
        message = next(d for e, d in events if e == "message")
        assert message["debug"]["params"]["temperature"] == 0.9
        assert message["debug"]["params"]["toolset"] == "core"
        assert "load_tools" not in {s["function"]["name"] for s in llm.tools_seen[0]}
        assert send(client, cid, "hi", params={"num_ctx": 4096})[0][1]["status"] == 403
    for client, cid, llm in start(llm_tuning_emails="ada@example.com", llm_num_ctx_max=8192):
        llm.replies = [say("ok")]
        events = send(client, cid, "hi", params={"num_ctx": 8192})
        assert next(d for e, d in events if e == "message")["debug"]["context_window"] == 8192
        assert send(client, cid, "hi", params={"num_ctx": 16384})[0][1]["status"] == 403


def test_an_interrupted_exchange_does_not_leave_two_user_messages_in_a_row() -> None:
    for client, cid, llm in start():
        llm.replies = [say("ok")]
        send(client, cid, "first question")
        # the reply to this one is lost, as when the browser disconnects mid-turn
        from minerva_api.db import ChatMessage

        with client.app.state.session_factory() as session:  # type: ignore[attr-defined]
            import uuid

            session.add(
                ChatMessage(
                    conversation_id=uuid.UUID(cid), role="user", content={"text": "lost question"}
                )
            )
            session.commit()
        send(client, cid, "next question")
        sent = llm.seen[-1]
        assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
        assert "lost question" in sent[3]["content"] and "next question" in sent[3]["content"]


def test_a_tool_call_in_an_earlier_turn_is_replayed_with_its_result(
    chat: tuple[TestClient, str, ReplayLLM],
) -> None:
    client, cid, llm = chat
    llm.replies = [
        call("differentiate", {"latex": r"x^2 \sin x", "variables": [{"name": "x"}]}),
        say("Derivative above."),
        say("Answer to the follow-up."),
    ]
    llm.repeat_last = False
    sse(client, cid, "differentiate x^2 sin x")
    sse(client, cid, "now simplify that")
    follow_up = llm.seen[2]
    kinds = [m["role"] for m in follow_up]
    assert kinds == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert "result_id=r1" in follow_up[3]["content"]
    assert Chunk  # the replay model builds its replies from chunks
