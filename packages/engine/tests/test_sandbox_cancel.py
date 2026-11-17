"""A caller that gives up on a sandbox call must stop its child and free its slot (audit defect D3).

Before: the agent's time limit stopped waiting for a worker thread, but the thread (and the child process it
waited for) ran on until the sandbox's own limit, holding one of the few slots and able to start more children.
"""

import threading
import time
from typing import Any

import pytest
from minerva_engine import Limits, sandbox


def _sleeper(conn: Any, op: str, payload: dict[str, Any], cpu: int, mem: int) -> None:
    """Stands in for the child: it never answers within the test."""
    time.sleep(30)


def _answers_now(conn: Any, op: str, payload: dict[str, Any], cpu: int, mem: int) -> None:
    conn.send(("ok", {"done": True}))
    conn.close()


@pytest.fixture
def slow(monkeypatch: pytest.MonkeyPatch) -> sandbox.Sandbox:
    monkeypatch.setattr(sandbox, "_child_main", _sleeper)
    return sandbox.Sandbox(Limits(timeout_seconds=20, max_concurrency=1))


def run_in_thread(
    box: sandbox.Sandbox, event: threading.Event
) -> tuple[threading.Thread, list[Any]]:
    out: list[Any] = []

    def work() -> None:
        sandbox.cancel_event.set(event)
        out.append(box.run("simplify", {}))

    thread = threading.Thread(target=work)
    thread.start()
    return thread, out


def test_a_cancelled_call_kills_its_child_and_frees_its_slot(slow: sandbox.Sandbox) -> None:
    event = threading.Event()
    thread, out = run_in_thread(slow, event)
    time.sleep(0.5)  # the child is running and holds the only slot
    assert thread.is_alive()
    started = time.monotonic()
    event.set()
    thread.join(timeout=5)
    assert not thread.is_alive() and time.monotonic() - started < 10  # not the 20 s limit
    assert out[0].outcome is sandbox.Outcome.KILLED
    # the slot is free again: another caller gets it at once
    assert slow._slots.acquire(blocking=False)
    slow._slots.release()


def test_a_cancelled_caller_starts_no_further_child(slow: sandbox.Sandbox) -> None:
    event = threading.Event()
    event.set()
    sandbox.cancel_event.set(event)
    try:
        result = slow.run("simplify", {})
    finally:
        sandbox.cancel_event.set(None)
    assert result.outcome is sandbox.Outcome.KILLED


def test_a_caller_waiting_for_a_slot_stops_waiting_when_cancelled(slow: sandbox.Sandbox) -> None:
    holder = threading.Event()
    first, _ = run_in_thread(slow, holder)
    time.sleep(0.5)
    waiter = threading.Event()
    second, out = run_in_thread(slow, waiter)  # queued behind the first
    time.sleep(0.3)
    waiter.set()
    second.join(timeout=5)
    assert not second.is_alive() and out[0].outcome is sandbox.Outcome.KILLED
    holder.set()
    first.join(timeout=5)


def test_many_cancellations_leave_every_slot_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_child_main", _sleeper)
    box = sandbox.Sandbox(Limits(timeout_seconds=20, max_concurrency=2))
    for _ in range(4):
        event = threading.Event()
        thread, _ = run_in_thread(box, event)
        time.sleep(0.2)
        event.set()
        thread.join(timeout=5)
    assert box._slots.acquire(blocking=False) and box._slots.acquire(blocking=False)
    box._slots.release()
    box._slots.release()


def test_a_child_that_finishes_before_the_cancel_still_returns_its_whole_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox, "_child_main", _answers_now)
    box = sandbox.Sandbox(Limits(timeout_seconds=20, max_concurrency=1))
    event = threading.Event()
    thread, out = run_in_thread(box, event)
    thread.join(timeout=10)
    assert out[0].outcome is sandbox.Outcome.OK and out[0].value == {"done": True}


def test_without_a_cancel_event_nothing_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_child_main", _answers_now)
    result = sandbox.Sandbox(Limits(timeout_seconds=20)).run("simplify", {})
    assert result.outcome is sandbox.Outcome.OK
    monkeypatch.setattr(sandbox, "_child_main", _sleeper)
    assert (
        sandbox.Sandbox(Limits(timeout_seconds=1)).run("simplify", {}).outcome
        is sandbox.Outcome.TIMEOUT
    )
