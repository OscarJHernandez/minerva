"""The follow-up scenario of the bug report, over HTTP: later turns must reach the model with the earlier ones."""

import json
import uuid
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


# --- retry, edit, stop ------------------------------------------------------------------------------------------
def stored(client: TestClient, cid: str) -> list[dict[str, Any]]:
    return client.get(f"/v1/chat/conversations/{cid}").json()["messages"]  # type: ignore[no-any-return]


def test_retry_answers_the_last_message_again_without_storing_it_twice() -> None:
    for client, cid, llm in start():
        llm.replies = [say("first answer $$a = 1$$"), say("second answer")]
        send(client, cid, "tell me about a")
        before = stored(client, cid)
        assert [m["role"] for m in before] == ["user", "assistant"]
        events = send(client, cid, "ignored on retry", mode="retry")
        assert next(d for e, d in events if e == "message")["text"] == "second answer"
        after = stored(client, cid)
        assert [m["role"] for m in after] == ["user", "assistant"]  # one user message, one reply
        assert (
            after[0]["content"]["text"] == "tell me about a"
        )  # the stored text, not the request's
        assert after[0]["id"] == before[0]["id"]  # the same row: not stored again
        sent = llm.seen[-1]
        assert [m["role"] for m in sent] == ["system", "user"] and sent[1][
            "content"
        ] == "tell me about a"


def test_a_replaced_reply_is_gone_and_its_equation_numbers_are_not_reused() -> None:
    for client, cid, llm in start():
        llm.replies = [say("$$a = 1$$"), say("$$b = 2$$"), say("seen")]
        send(client, cid, "one")
        send(client, cid, "one", mode="retry")  # the reply with equation (1) is replaced
        numbers = [
            s["number"]
            for m in stored(client, cid)
            if m["role"] == "assistant"
            for b in m["content"]["blocks"]
            for s in b["segments"]
            if s["type"] == "display_math"
        ]
        assert numbers == [2]  # (1) was given once and is never given again
        send(client, cid, "use equation 1")
        user = stored(client, cid)[-2]["content"]
        assert user["references"] == [{"number": 1, "latex": None, "source": "unknown"}]
        assert "no longer exists" in llm.seen[-1][-1]["content"]


def test_edit_replaces_the_last_message_and_its_reply() -> None:
    for client, cid, llm in start():
        llm.replies = [say("old reply"), say("new reply")]
        send(client, cid, "Integrate $$x^2$$")
        events = send(client, cid, "Integrate $$x^3$$ instead", mode="edit")
        assert next(d for e, d in events if e == "message")["text"] == "new reply"
        messages = stored(client, cid)
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"]["text"] == "Integrate $$x^3$$ instead"
        assert [e["latex"] for e in messages[0]["content"]["equations"]] == ["x^3"]
        sent = llm.seen[-1][-1]["content"]
        # The model gets the user's text as typed, then the handle of the displayed formula (for the model only).
        assert sent.startswith("Integrate $$x^3$$ instead")
        assert "@e2" in sent.removeprefix("Integrate $$x^3$$ instead")


def test_retry_with_nothing_to_answer_is_refused() -> None:
    for client, cid, _llm in start():
        refused = send(client, cid, "x", mode="retry")
        assert refused[0][0] == "http" and refused[0][1]["status"] == 400


def test_a_question_whose_reply_was_lost_can_be_retried() -> None:
    from minerva_api.db import ChatMessage

    for client, cid, llm in start():
        llm.replies = [say("here it is")]
        with client.app.state.session_factory() as session:  # type: ignore[attr-defined]
            session.add(
                ChatMessage(conversation_id=uuid.UUID(cid), role="user", content={"text": "lost"})
            )
            session.commit()
        send(client, cid, "lost", mode="retry")
        assert [m["role"] for m in stored(client, cid)] == ["user", "assistant"]
        assert llm.seen[0][-1]["content"] == "lost"


def test_the_user_is_told_when_formulas_are_being_checked() -> None:
    for client, cid, llm in start():
        llm.replies = [say("It is $x = y + 1$.")]
        events = send(client, cid, "tell me")
        kinds = [e for e, _ in events]
        assert "phase" in kinds and kinds.index("phase") < kinds.index("message")
        assert next(d for e, d in events if e == "phase")["phase"] == "checking"
    for client, cid, llm in start():
        llm.replies = [say("Which variable?")]  # nothing to check: no extra event
        assert "phase" not in [e for e, _ in send(client, cid, "tell me")]


def test_what_arrived_before_the_client_left_is_kept_as_stopped() -> None:
    from minerva_api.agent.events import Card
    from minerva_api.routers_chat import _store_stopped

    for client, cid, _llm in start():
        card = Card(id="c1", tool="simplify", type="error", title="t", data={"message": "m"})
        _store_stopped(
            client.app.state.session_factory,  # type: ignore[attr-defined]
            uuid.UUID(cid),
            ["The answer begins $x", card, "and goes on"],
        )
        (message,) = stored(client, cid)
        content = message["content"]
        assert message["role"] == "assistant" and content["stop_reason"] == "stopped"
        assert content["truncated"] is True
        assert [b["kind"] for b in content["blocks"]] == ["text", "card", "text"]
        assert "$x" in content["text"]
    for client, cid, _llm in start():
        _store_stopped(client.app.state.session_factory, uuid.UUID(cid), [])  # type: ignore[attr-defined]
        assert stored(client, cid) == []  # nothing arrived: nothing is stored


def test_what_the_message_was_found_to_contain_is_sent_before_the_reply() -> None:
    for client, cid, llm in start():
        llm.replies = [say("ok"), say("fine")]
        first = send(client, cid, "Take $$x + 1$$ please")
        kinds = [e for e, _ in first]
        assert kinds.index("user") < kinds.index("message")
        assert next(d for e, d in first if e == "user")["equations"] == [
            {"number": 1, "latex": "x + 1"}
        ]
        second = send(client, cid, "Integrate equation 1")
        user = next(d for e, d in second if e == "user")
        assert user["references"][0]["number"] == 1 and user["references"][0]["latex"] == "x + 1"
    for client, cid, llm in start():
        llm.replies = [say("ok")]
        assert "user" not in [e for e, _ in send(client, cid, "no formulas here")]


def test_the_position_vector_and_its_length_do_not_block_a_derivative_in_r() -> None:
    """Issue #132: the reply wrote `\\mathbf{r}` and `r = |\\mathbf{r}|`; `r` stayed a scalar for the next tool."""
    reply = (
        r"Take the relative position vector $\mathbf{r}$, where $r = |\mathbf{r}|$ is the distance. "
        r"The angular momentum is $\mu r^2 \dot{\theta}$."
    )
    lagrangian = r"\frac12 \mu (\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{G M \mu}{r}"
    for client, cid, llm in start():
        llm.replies = [
            say(reply),
            call(
                "time_derivative",
                {
                    "latex": r"\mu r^2 \dot{\theta}",
                    "coordinates": ["r", "theta"],
                    "lagrangian": lagrangian,
                },
            ),
            say("The result is shown on the card."),
        ]
        send(client, cid, "Write the Kepler Lagrangian in vector and polar form.")
        events = send(client, cid, "Take the time derivative of the angular momentum.")
        cards = [d for e, d in events if e == "card"]
        assert cards and cards[-1]["type"] == "differentiate", cards
        assert cards[-1]["data"]["status"] == "ok"
        refusal = [m for m in llm.seen[-1] if "declared as a matrix" in str(m.get("content", ""))]
        assert not refusal
