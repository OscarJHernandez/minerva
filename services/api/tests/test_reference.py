from fastapi.testclient import TestClient


def test_reference_is_public(client: TestClient) -> None:
    toc = client.get("/v1/reference/toc").json()
    assert any(c["id"] == "math.trigonometry" for c in toc)
    entries = client.get("/v1/reference/entries", params={"q": "gaussian"}).json()
    assert entries[0]["id"] == "int.gaussian"
    entry = client.get("/v1/reference/entries/log.product").json()
    assert entry["applicable_as_rule"] is True
    assert {c["assumption"] for c in entry["conditions"]} == {"positive"}
    assert entry["verification"] == "symbolic"


def test_missing_entry_404(client: TestClient) -> None:
    assert client.get("/v1/reference/entries/no.such").status_code == 404


def test_search_filters(client: TestClient) -> None:
    rows = client.get("/v1/reference/entries", params={"chapter": "physics.constants"}).json()
    assert rows and all(r["chapter"] == "physics.constants" for r in rows)


def test_health(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200


def test_version_names_the_build_without_a_session() -> None:
    from tests.conftest import build_app

    with TestClient(build_app(build_sha="0123abcd")) as c:
        assert c.get("/v1/version").json() == {"build": "0123abcd"}
        assert c.get("/v1/version").headers["cache-control"] == "no-store"
        assert c.get("/v1/chat/status").json()["build"] == "0123abcd"


def test_version_is_unknown_for_a_local_run(client: TestClient) -> None:
    assert client.get("/v1/version").json() == {"build": "unknown"}
