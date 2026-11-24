"""The sandbox reports where a call's time went: waiting for a slot (`wait_ms`) and running (`run_ms`).

`ms` is always their sum, for every outcome (ok, busy, timeout, killed). The measurement adds no work: it
reads the clock the sandbox already reads. Children here are stand-ins that wait for a gate file, so a test
decides when a slot is released instead of sleeping for it.
"""

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from minerva_engine import Limits, sandbox
from minerva_engine.observe import SandboxCall


def _gated(conn: Any, op: str, payload: dict[str, Any], cpu: int, mem: int) -> None:
    """Answers once the gate file exists (it signals that it started by creating its own file)."""
    Path(payload["started"]).touch()
    gate = Path(payload["gate"])
    while not gate.exists():
        time.sleep(0.01)
    conn.send(("ok", {"done": True}))
    conn.close()


def _sleeper(conn: Any, op: str, payload: dict[str, Any], cpu: int, mem: int) -> None:
    time.sleep(30)


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 60
    while not path.exists():
        assert time.monotonic() < deadline, "the child never started"
        time.sleep(0.01)


def _call(
    box: sandbox.Sandbox,
    payload: dict[str, Any],
    out: list[Any],
    event: threading.Event | None = None,
    timeout: float | None = None,
) -> threading.Thread:
    def work() -> None:
        sandbox.cancel_event.set(event)
        out.append(box.run("simplify", payload, timeout=timeout))

    thread = threading.Thread(target=work)
    thread.start()
    return thread


def _payload(tmp_path: Path, name: str) -> dict[str, Any]:
    return {"started": str(tmp_path / f"{name}.started"), "gate": str(tmp_path / "gate")}


def _holds_the_only_slot(
    box: sandbox.Sandbox, tmp_path: Path
) -> tuple[threading.Thread, list[Any]]:
    out: list[Any] = []
    holder = _call(box, _payload(tmp_path, "a"), out, timeout=60)
    _wait_for(tmp_path / "a.started")
    return holder, out


def test_a_call_that_waited_for_the_only_slot_reports_the_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sandbox, "_child_main", _gated)
    box = sandbox.Sandbox(Limits(timeout_seconds=60, max_concurrency=1))
    holder, first = _holds_the_only_slot(box, tmp_path)
    second: list[Any] = []
    waiter = _call(box, _payload(tmp_path, "b"), second)
    threading.Event().wait(
        0.3
    )  # the second caller is queued behind the first for a measurable time
    (tmp_path / "gate").touch()
    holder.join(timeout=60)
    waiter.join(timeout=60)
    a, b = first[0], second[0]
    assert a.outcome is b.outcome is sandbox.Outcome.OK
    assert (
        b.wait_ms >= 100
    )  # a lower bound only: a slow machine makes the wait longer, never shorter
    assert a.wait_ms < b.wait_ms
    for result in (a, b):
        assert result.ms == result.wait_ms + result.run_ms
        assert result.run_ms >= 0


def test_a_busy_call_reports_its_whole_wait_and_no_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sandbox, "_child_main", _gated)
    box = sandbox.Sandbox(Limits(timeout_seconds=1, max_concurrency=1))
    holder, _ = _holds_the_only_slot(box, tmp_path)  # a longer per-call limit: it outlives the wait
    busy = box.run("simplify", _payload(tmp_path, "b"))
    (tmp_path / "gate").touch()
    holder.join(timeout=60)
    assert busy.outcome is sandbox.Outcome.BUSY
    assert busy.run_ms == 0 and busy.wait_ms >= 900 and busy.ms == busy.wait_ms


def test_a_timeout_reports_the_time_it_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_child_main", _sleeper)
    box = sandbox.Sandbox(Limits(timeout_seconds=20, max_concurrency=1))
    result = box.run("simplify", {}, timeout=0.4)
    assert result.outcome is sandbox.Outcome.TIMEOUT
    assert result.run_ms >= 350 and result.ms == result.wait_ms + result.run_ms


def test_a_call_cancelled_while_it_waits_reports_the_wait_and_runs_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sandbox, "_child_main", _gated)
    box = sandbox.Sandbox(Limits(timeout_seconds=60, max_concurrency=1))
    holder, _ = _holds_the_only_slot(box, tmp_path)
    cancel = threading.Event()
    out: list[Any] = []
    waiter = _call(box, _payload(tmp_path, "b"), out, event=cancel)
    threading.Event().wait(0.3)
    cancel.set()
    waiter.join(timeout=30)
    assert not waiter.is_alive()
    result = out[0]
    assert result.outcome is sandbox.Outcome.KILLED
    assert result.run_ms == 0 and result.wait_ms >= 100 and result.ms == result.wait_ms
    assert not (tmp_path / "b.started").exists()  # no child was started for it
    (tmp_path / "gate").touch()
    holder.join(timeout=60)


def test_the_observed_call_carries_the_split() -> None:
    call = SandboxCall("simplify", "ok", 7, "t", wait_ms=3, run_ms=4)
    assert call.ms == call.wait_ms + call.run_ms
