import base64
import json

from fastapi.testclient import TestClient
from minerva_api.db import ChatMessage, Conversation
from sqlalchemy import func, select

from .conftest import build_app, register

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 32


def post(c: TestClient, data: bytes) -> object:
    return c.post("/v1/chat/formula-image", json={"image": base64.b64encode(data).decode()})


def scripted(**over: object) -> TestClient:
    c = TestClient(build_app(llm_mode="scripted", **over))
    c.__enter__()
    register(c)
    return c


def test_status_reports_whether_images_can_be_read() -> None:
    with TestClient(build_app()) as c:
        register(c)
        assert c.get("/v1/chat/status").json()["images"] is False
    with TestClient(build_app(llm_mode="scripted")) as c:
        assert c.get("/v1/chat/status").json()["images"] is True
    with TestClient(build_app(llm_mode="ollama", llm_base_url="http://h.test", llm_model="m")) as c:
        assert c.get("/v1/chat/status").json()["images"] is False  # no vision model named
    with TestClient(
        build_app(
            llm_mode="ollama", llm_base_url="http://h.test", llm_model="m", llm_vision_model="v"
        )
    ) as c:
        assert c.get("/v1/chat/status").json()["images"] is True


def test_reads_formulas_and_reports_how_the_parser_understood_each() -> None:
    c = scripted()
    for image in (PNG, JPEG, WEBP):
        r = post(c, image)
        assert r.status_code == 200, r.text  # type: ignore[attr-defined]
    formulas = r.json()["formulas"]  # type: ignore[attr-defined]
    good, bad = formulas
    assert good["latex"] == r"\frac{x^2-1}{x-1}" and good["status"] == "ok"
    assert "x^{2} - 1" in good["read_as"]  # shown as read, not simplified
    assert bad["latex"] == r"\sin 2x" and bad["status"] == "parse_error"
    assert bad["read_as"] is None and bad["message"]


def test_requires_login_and_a_configured_reader() -> None:
    with TestClient(build_app(llm_mode="scripted")) as c:
        assert post(c, PNG).status_code == 401  # type: ignore[attr-defined]
    with TestClient(build_app()) as c:
        register(c)
        assert post(c, PNG).status_code == 503  # type: ignore[attr-defined]


def test_rejects_wrong_type_bad_base64_oversize_and_rate() -> None:
    c = scripted(image_max_bytes=10_000, image_rate_per_minute=5)
    assert post(c, b"GIF89a" + b"\x00" * 100).status_code == 415  # type: ignore[attr-defined]
    assert post(c, b"<svg onload=alert(1)>").status_code == 415  # type: ignore[attr-defined]
    assert c.post("/v1/chat/formula-image", json={"image": "!!!"}).status_code == 422
    assert c.post("/v1/chat/formula-image", content=b"not json").status_code == 422
    assert post(c, PNG + b"\x00" * 20_000).status_code == 413  # type: ignore[attr-defined]
    assert post(c, PNG).status_code == 429  # type: ignore[attr-defined]  # sixth call in a minute


def test_nothing_is_stored(capfd) -> None:  # type: ignore[no-untyped-def]
    app = build_app(llm_mode="scripted")
    with TestClient(app) as c:
        register(c)
        assert post(c, PNG).status_code == 200  # type: ignore[attr-defined]
        with app.state.session_factory() as db:
            assert db.scalar(select(func.count()).select_from(ChatMessage)) == 0
            assert db.scalar(select(func.count()).select_from(Conversation)) == 0
    out = capfd.readouterr().out
    lines = [json.loads(ln) for ln in out.splitlines() if ln.startswith("{")]
    assert any(ln.get("event") == "formula_image" for ln in lines)
    assert "frac" not in out and "sin" not in out  # no formula text in the log
