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
