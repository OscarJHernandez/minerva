from fastapi.testclient import TestClient

from .conftest import PASSWORD, build_app, register


def test_compute_requires_authentication(client: TestClient) -> None:
    r = client.post("/v1/compute/simplify", json={"input": "x"})
    assert r.status_code == 401


def test_simplify_and_history(authed: TestClient) -> None:
    r = authed.post("/v1/compute/simplify", json={"input": r"\sin^2 x + \cos^2 x"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["result_text"] == "1"
    assert body["verification"]["passed"] is True
    history = authed.get("/v1/history").json()
    assert len(history) == 1
    assert history[0]["operation"] == "simplify"
    assert history[0]["input"] == r"\sin^2 x + \cos^2 x"
    assert history[0]["id"] == body["calculation_id"]


def test_save_false_is_not_stored(authed: TestClient) -> None:
    authed.post("/v1/compute/simplify", json={"input": "x+1", "format": "text", "save": False})
    assert authed.get("/v1/history").json() == []


def test_parse_error_is_reported_not_guessed(authed: TestClient) -> None:
    body = authed.post("/v1/compute/simplify", json={"input": r"\nonsense"}).json()
    assert body["status"] == "parse_error"
    assert body["result_latex"] is None


def test_assumption_changes_result(authed: TestClient) -> None:
    plain = authed.post("/v1/compute/simplify", json={"input": r"\sqrt{x^2}"}).json()
    pos = authed.post(
        "/v1/compute/simplify",
        json={"input": r"\sqrt{x^2}", "assumptions": {"x": ["positive"]}},
    ).json()
    assert plain["status"] == "unchanged"
    assert pos["result_text"] == "x"


def test_unknown_assumption_rejected(authed: TestClient) -> None:
    r = authed.post("/v1/compute/simplify", json={"input": "x", "assumptions": {"x": ["evil"]}})
    assert r.status_code == 422


def test_oversized_input_rejected(authed: TestClient) -> None:
    r = authed.post("/v1/compute/simplify", json={"input": "x+" * 3000 + "x"})
    assert r.status_code == 422


def test_identities_endpoint(authed: TestClient) -> None:
    r = authed.post("/v1/compute/identities", json={"input": r"\log(xy)"})
    assert r.status_code == 200
    matches = {m["entry"]["id"]: m["application"] for m in r.json()["matches"]}
    assert matches["log.product"]["applicability"] == "conditions_not_satisfied"
    r2 = authed.post(
        "/v1/compute/identities",
        json={"input": r"\log(xy)", "assumptions": {"x": ["positive"], "y": ["positive"]}},
    )
    matches2 = {m["entry"]["id"]: m["application"] for m in r2.json()["matches"]}
    assert matches2["log.product"]["applicability"] == "applicable"


def test_identities_endpoint_offers_the_triple_angle_not_a_halved_double_angle(
    authed: TestClient,
) -> None:
    """Handbook growth and the matcher fix, seen through the real endpoint."""
    r = authed.post("/v1/compute/identities", json={"input": r"\sin(3x)"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    applicable = {
        m["entry"]["id"]: m
        for m in body["matches"]
        if m["application"]["applicability"] == "applicable"
    }
    assert "trig.triple_sin" in applicable
    assert "trig.double_sin" not in applicable  # would need a = 3x/2
    triple = applicable["trig.triple_sin"]
    assert triple["entry"]["verification"] in {"symbolic", "numeric"}
    assert triple["entry"]["reference"]["source"] == "DLMF"
    assert triple["application"]["verification"]["passed"] is True


def test_tensor_endpoint(authed: TestClient) -> None:
    r = authed.post(
        "/v1/compute/tensor/canonicalize",
        json={
            "input": "S(a,b)*A(-a,-b)",
            "tensors": {
                "S": {"rank": 2, "symmetry": "symmetric"},
                "A": {"rank": 2, "symmetry": "antisymmetric"},
            },
        },
    )
    assert r.status_code == 200
    assert r.json()["result_text"] == "0"


def test_resource_limit_surfaces_as_status() -> None:
    app = build_app(compute_timeout_seconds=2.0)
    with TestClient(app) as c:
        register(c)
        body = c.post(
            "/v1/compute/simplify", json={"input": "(x+1)**9000*(y+1)**9000 - 1", "format": "text"}
        ).json()
        assert body["status"] == "resource_limit"
        # the server is still healthy afterwards
        assert c.post("/v1/compute/simplify", json={"input": "x+y"}).json()["status"] == "unchanged"


def test_history_is_isolated_per_user(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as a, TestClient(app) as b:
        register(a, "a@example.com")
        register(b, "b@example.com")
        created = a.post("/v1/compute/simplify", json={"input": "x*x", "format": "text"}).json()
        cid = created["calculation_id"]
        assert len(a.get("/v1/history").json()) == 1
        assert b.get("/v1/history").json() == []
        assert b.get(f"/v1/history/{cid}").status_code == 404
        assert b.delete(f"/v1/history/{cid}").status_code == 404
        assert a.get(f"/v1/history/{cid}").status_code == 200
        assert a.delete(f"/v1/history/{cid}").status_code == 204
        assert a.get("/v1/history").json() == []


def test_clear_history(authed: TestClient) -> None:
    for expr in ("x+1", "y+1"):
        authed.post("/v1/compute/simplify", json={"input": expr, "format": "text"})
    assert len(authed.get("/v1/history").json()) == 2
    assert authed.delete("/v1/history").status_code == 204
    assert authed.get("/v1/history").json() == []


def test_logs_never_contain_user_input(capsys) -> None:  # type: ignore[no-untyped-def]
    marker = "zzMARKERzz"
    app = build_app()  # created now so logs hit capsys
    with TestClient(app) as c:
        register(c)
        c.post("/v1/compute/simplify", json={"input": f"{marker} + 1", "format": "text"})
        c.post("/v1/auth/login", json={"email": "ada@example.com", "password": PASSWORD})
    out = capsys.readouterr().out
    assert "compute" in out  # something was logged
    assert marker not in out
    assert PASSWORD not in out
    assert "ada@example.com" not in out
