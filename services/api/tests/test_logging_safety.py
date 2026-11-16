"""Everything the API writes goes through one allow-list, including what the libraries under it log.

Uvicorn logs an unhandled exception with its traceback, SQLAlchemy a statement with its parameters.
Both go through the standard library's logging, which used to bypass the allow-list: a message with a
user's expression or e-mail address reached stdout. Now a record from a library is reduced to its
logger, its level and, for an exception, the class and where our code raised it.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from minerva_api import logging as api_logging

from .conftest import build_app

SECRET = "secret@example.com"
EXPRESSION = "x^2 + 3*secret"


def lines(out: str) -> list[dict[str, Any]]:
    """Every line the process wrote must be one JSON object: nothing is allowed to bypass the filter."""
    text = [x for x in out.splitlines() if x.strip()]
    return [json.loads(x) for x in text]


def test_a_record_from_a_library_is_reduced_to_a_fixed_event(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api_logging.configure_logging()
    capsys.readouterr()
    try:
        raise ValueError(f"cannot simplify {EXPRESSION} for {SECRET}")
    except ValueError:
        logging.getLogger("uvicorn.error").exception("Exception in ASGI application %s", SECRET)
    logging.getLogger("sqlalchemy.engine").warning("SELECT ... WHERE email = '%s'", SECRET)
    logging.getLogger("some.other.library").warning("token=%s %s", SECRET, EXPRESSION)
    out = capsys.readouterr().out
    records = lines(out)
    assert SECRET not in out and EXPRESSION not in out and "Traceback" not in out
    by_event = {r["event"]: r for r in records}
    asgi = by_event["foreign:uvicorn.error"]
    assert asgi["level"] == "error" and asgi["logger"] == "uvicorn.error"
    assert asgi["error_type"] == "ValueError"
    assert "error_module" not in asgi or asgi["error_module"].startswith("minerva_")
    assert "foreign:sqlalchemy.engine" in by_event
    other = by_event["foreign_log"]
    assert other["logger"] == "other" and other["level"] == "warning"
    for record in records:
        assert set(record) <= api_logging.ALLOWED_FIELDS


def test_the_access_log_is_off_and_uvicorn_writes_through_the_same_filter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api_logging.configure_logging()
    capsys.readouterr()
    logging.getLogger("uvicorn.access").info(
        '%s - "GET /v1/x?q=%s HTTP/1.1" 200', "1.2.3.4", SECRET
    )
    logging.getLogger("uvicorn").info("Started server process")
    out = capsys.readouterr().out
    assert "1.2.3.4" not in out and SECRET not in out and "/v1/x" not in out
    assert [r["event"] for r in lines(out)] == ["foreign:uvicorn"]
    dockerfile = Path(__file__).resolve().parents[3] / "deploy" / "docker" / "api.Dockerfile"
    assert "--no-access-log" in dockerfile.read_text()


def test_configuring_twice_does_not_duplicate_lines(capsys: pytest.CaptureFixture[str]) -> None:
    api_logging.configure_logging()
    api_logging.configure_logging()
    capsys.readouterr()
    logging.getLogger("uvicorn.error").error("boom")
    assert len(lines(capsys.readouterr().out)) == 1


def test_an_unhandled_error_gets_a_fixed_body_and_both_log_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = build_app(llm_mode="off")

    def boom() -> None:
        raise RuntimeError(f"cannot simplify {EXPRESSION} for {SECRET}")

    app.add_api_route("/v1/_boom", boom)
    with TestClient(app, raise_server_exceptions=False) as client:
        api_logging.configure_logging()  # bind the logger to the stream capsys captures now
        capsys.readouterr()
        response = client.get("/v1/_boom")
    out = capsys.readouterr().out
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error"
    assert body["request_id"] == response.headers.get("x-request-id") or body["request_id"]
    assert SECRET not in response.text and EXPRESSION not in response.text
    assert SECRET not in out and EXPRESSION not in out and "Traceback" not in out
    records = lines(out)
    failed = next(r for r in records if r["event"] == "unhandled_error")
    assert failed["error_type"] == "RuntimeError" and failed["route"] == "/v1/_boom"
    assert failed["request_id"] == body["request_id"]
    assert failed["error_module"].startswith("minerva_") or "error_module" not in failed
    request = next(r for r in records if r["event"] == "request" and r["route"] == "/v1/_boom")
    assert request["status_code"] == 500 and request["request_id"] == body["request_id"]
