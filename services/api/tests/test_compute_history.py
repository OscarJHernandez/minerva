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


def test_simplify_reports_the_identities_behind_a_chain(authed: TestClient) -> None:
    body = authed.post(
        "/v1/compute/simplify", json={"input": r"\sin^2 x + \cos^2 x + \tan^2 x"}
    ).json()
    assert body["status"] == "ok"
    assert [s["name"] for s in body["steps"]] == ["trig.pythagorean", "trig.tan_secant"]
    assert body["steps"][0]["verification"]["passed"] is True
    assert body["entries"]["trig.tan_secant"]["title"]
    assert body["verification"]["numeric_points_checked"] >= 4
    assert "not a proof" in body["message"]


def test_simplify_goal_and_target(authed: TestClient) -> None:
    expanded = authed.post(
        "/v1/compute/simplify", json={"input": r"(x+y)^3", "goal": "expand"}
    ).json()
    assert expanded["status"] == "ok" and expanded["steps"][0]["name"] == "alg.binomial_cube_sum"
    rewritten = authed.post(
        "/v1/compute/simplify",
        json={"input": r"\tan x\cos x", "goal": "rewrite", "target": ["sin", "cos"]},
    ).json()
    assert rewritten["result_text"] == "sin(x)"
    bad = authed.post("/v1/compute/simplify", json={"input": "x", "goal": "integrate"})
    assert bad.status_code == 422
    needs_target = authed.post(
        "/v1/compute/simplify", json={"input": "x", "goal": "rewrite"}
    ).json()
    assert needs_target["status"] == "unsupported"


def test_blocked_identity_names_its_conditions_and_assumptions_unlock_it(
    authed: TestClient,
) -> None:
    blocked = authed.post("/v1/compute/simplify", json={"input": r"\log x + \log y"}).json()
    assert blocked["status"] == "conditions_not_satisfied"
    assert blocked["steps"] == []
    unmet = {c["variable"] for c in blocked["blocked"][0]["unmet_conditions"]}
    assert unmet == {"a", "b"}
    unlocked = authed.post(
        "/v1/compute/simplify",
        json={"input": r"\log x + \log y", "assumptions": {"x": ["positive"], "y": ["positive"]}},
    ).json()
    assert unlocked["status"] == "ok" and unlocked["steps"][0]["name"] == "log.product"


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
    # Without assumptions the handbook's sqrt(a**2) identities match but are not applied: the
    # response says so (conditions_not_satisfied) instead of a bare "unchanged", and the input comes back.
    assert plain["status"] == "conditions_not_satisfied"
    assert plain["result_latex"] == plain["input_latex"]
    assert {b["rule_id"] for b in plain["blocked"]} == {
        "radical.sqrt_square_abs",
        "radical.sqrt_square_nonneg",
    }
    assert pos["result_text"] == "x"


def test_unknown_assumption_rejected(authed: TestClient) -> None:
    r = authed.post("/v1/compute/simplify", json={"input": "x", "assumptions": {"x": ["evil"]}})
    assert r.status_code == 422


def test_oversized_input_rejected(authed: TestClient) -> None:
    r = authed.post("/v1/compute/simplify", json={"input": "x+" * 3000 + "x"})
    assert r.status_code == 422


def test_differentiate_and_history(authed: TestClient) -> None:
    r = authed.post(
        "/v1/compute/differentiate",
        json={"input": "x**5", "format": "text", "variables": [{"name": "x", "order": 2}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["result_text"] == "20*x**3"
    assert body["verification"]["passed"] is True
    assert [s["latex"] for s in body["steps"]] == ["5 x^{4}", "20 x^{3}"]
    (item,) = authed.get("/v1/history").json()
    assert item["operation"] == "differentiate"
    assert item["id"] == body["calculation_id"]
    assert item["result"]["variables"] == [{"name": "x", "order": 2}]


def test_integrate_and_history(authed: TestClient) -> None:
    r = authed.post(
        "/v1/compute/integrate", json={"input": "x*cos(x)", "format": "text", "variable": "x"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["verification"]["passed"] is True
    (item,) = authed.get("/v1/history").json()
    assert item["operation"] == "integrate"
    assert item["id"] == body["calculation_id"]
    assert item["result"]["variable"] == "x"


def test_integrate_failure_is_saved_with_its_variable(authed: TestClient) -> None:
    body = authed.post(
        "/v1/compute/integrate", json={"input": "exp(x**x)", "format": "text", "variable": "x"}
    ).json()
    assert body["status"] == "unsupported"
    assert body["result_latex"] is None
    assert (
        authed.post("/v1/compute/integrate", json={"input": "x", "variable": ""}).status_code == 422
    )


def test_integrate_requires_authentication(client: TestClient) -> None:
    body = {"input": "x", "format": "text", "variable": "x"}
    assert client.post("/v1/compute/integrate", json=body).status_code == 401


def test_differentiate_mixed_partial_and_assumptions(authed: TestClient) -> None:
    mixed = authed.post(
        "/v1/compute/differentiate",
        json={
            "input": "x**2*y**3",
            "format": "text",
            "variables": [{"name": "x"}, {"name": "y"}],
        },
    ).json()
    assert mixed["result_text"] == "6*x*y**2"
    plain = authed.post(
        "/v1/compute/differentiate",
        json={"input": "abs(x)", "format": "text", "variables": [{"name": "x"}]},
    ).json()
    assert plain["status"] == "unsupported"
    assert plain["result_latex"] is None
    real = authed.post(
        "/v1/compute/differentiate",
        json={
            "input": "abs(x)",
            "format": "text",
            "assumptions": {"x": ["real"]},
            "variables": [{"name": "x"}],
        },
    ).json()
    assert real["result_text"] == "sign(x)"


def test_differentiate_validation(authed: TestClient) -> None:
    url = "/v1/compute/differentiate"
    for variables in ([], [{"name": "x", "order": 11}], [{"name": "x", "order": 0}]):
        r = authed.post(url, json={"input": "x", "variables": variables})
        assert r.status_code == 422
    five = [{"name": n} for n in "abcde"]
    assert authed.post(url, json={"input": "x", "variables": five}).status_code == 422
    too_deep = [{"name": "x", "order": 6}, {"name": "y", "order": 5}]
    body = authed.post(url, json={"input": "x*y", "format": "text", "variables": too_deep}).json()
    assert body["status"] == "unsupported"
    assert (
        authed.post(url, json={"input": "x" * 3000, "variables": [{"name": "x"}]}).status_code
        == 422
    )
    parse = authed.post(url, json={"input": r"\nonsense", "variables": [{"name": "x"}]}).json()
    assert parse["status"] == "parse_error"


def test_differentiate_requires_authentication(client: TestClient) -> None:
    body = {"input": "x", "variables": [{"name": "x"}]}
    assert client.post("/v1/compute/differentiate", json=body).status_code == 401


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
        c.post(
            "/v1/compute/differentiate",
            json={"input": f"{marker} * x", "format": "text", "variables": [{"name": "x"}]},
        )
        c.post("/v1/auth/login", json={"email": "ada@example.com", "password": PASSWORD})
    out = capsys.readouterr().out
    assert "compute" in out  # something was logged
    assert marker not in out
    assert PASSWORD not in out
    assert "ada@example.com" not in out


def test_series_and_history(authed: TestClient) -> None:
    r = authed.post(
        "/v1/compute/series",
        json={"input": "exp(x)", "format": "text", "variable": "x", "order": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["result_text"] == "x**3/6 + x**2/2 + x + 1"
    assert body["verification"]["passed"] is True
    assert [t["power"] for t in body["terms"]] == [0, 1, 2, 3]
    (item,) = authed.get("/v1/history").json()
    assert item["operation"] == "series"
    assert item["id"] == body["calculation_id"]
    assert item["result"]["order"] == 3 and item["result"]["variable"] == "x"


def test_series_about_a_point_and_failures_keep_the_request(authed: TestClient) -> None:
    url = "/v1/compute/series"
    shifted = authed.post(
        url,
        json={
            "input": "log(x)",
            "format": "text",
            "variable": "x",
            "point": "1",
            "order": 2,
            "assumptions": {"x": ["positive"]},
        },
    ).json()
    assert shifted["status"] == "ok"
    pole = authed.post(
        url, json={"input": "1/x", "format": "text", "variable": "x", "order": 3}
    ).json()
    assert pole["status"] == "unsupported" and pole["result_latex"] is None
    items = authed.get("/v1/history").json()
    failed = next(i for i in items if i["status"] == "unsupported")
    assert failed["result"]["variable"] == "x" and failed["result"]["order"] == 3


def test_series_validation(authed: TestClient) -> None:
    url = "/v1/compute/series"
    for extra in ({"order": 0}, {"order": 13}, {"variable": ""}):
        body = {"input": "x", "variable": "x", **extra}
        assert authed.post(url, json=body).status_code == 422
    assert authed.post(url, json={"input": "x" * 3000, "variable": "x"}).status_code == 422
    parse = authed.post(url, json={"input": r"\nonsense", "variable": "x"}).json()
    assert parse["status"] == "parse_error"
    infinity = authed.post(
        url, json={"input": "x", "format": "text", "variable": "x", "point": "oo"}
    ).json()
    assert infinity["status"] == "unsupported"


def test_series_requires_authentication(client: TestClient) -> None:
    body = {"input": "x", "variable": "x"}
    assert client.post("/v1/compute/series", json=body).status_code == 401


def test_integrate_definite_and_history(authed: TestClient) -> None:
    r = authed.post(
        "/v1/compute/integrate",
        json={"input": "sin(x)", "format": "text", "variable": "x", "lower": "0", "upper": "pi"},
    )
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok"
    assert body["definite"] is True and body["result_text"] == "2"
    assert body["quadrature"]["passed"] is True
    (item,) = authed.get("/v1/history").json()
    assert item["result"]["lower_latex"] == "0" and item["result"]["variable"] == "x"
    half = authed.post(
        "/v1/compute/integrate",
        json={"input": "x", "format": "text", "variable": "x", "lower": "0"},
    ).json()
    assert half["status"] == "parse_error"


MATRIX_REGRESSION = {
    "input": r"\exp(wA)\exp(-wB)",
    "matrices": [
        {"name": "A", "entries": [["1", "0"], ["0", "2"]]},
        {"name": "B", "entries": [["0", "1"], ["0", "0"]]},
    ],
    "action": "integrate",
    "variable": "w",
    "lower": "0",
    "upper": "1",
}


def test_matrix_calculus_regression_and_history(authed: TestClient) -> None:
    r = authed.post("/v1/compute/matrix", json=MATRIX_REGRESSION)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["result_kind"] == "explicit" and body["evaluated"] is True
    assert body["result_entries"][0][1] == "-1"
    assert body["verification"]["passed"] is True
    assert body["method"] == "sylvester_reduction"
    (item,) = authed.get("/v1/history").json()
    assert item["operation"] == "matrix_calculus"
    assert item["id"] == body["calculation_id"]
    # the declarations are stored so that the workbench can restore the form
    assert item["result"]["request"]["variable"] == "w"
    assert [m["name"] for m in item["result"]["request"]["matrices"]] == ["A", "B"]


def test_matrix_calculus_reports_a_reduction_as_not_evaluated(authed: TestClient) -> None:
    body = authed.post(
        "/v1/compute/matrix",
        json={
            "input": r"\exp(tP)\exp(tQ)",
            "matrices": [{"name": "P", "rows": 2}, {"name": "Q", "rows": 2}],
            "action": "integrate",
            "variable": "t",
            "lower": "0",
            "upper": "1",
        },
    ).json()
    assert body["status"] == "ok"
    assert body["evaluated"] is False and body["result_kind"] == "representation"
    assert body["equation_latex"]


def test_matrix_calculus_failures_and_validation(authed: TestClient) -> None:
    mats = [{"name": "A", "rows": 2}, {"name": "B", "rows": 2}]
    body = authed.post(
        "/v1/compute/matrix", json={"input": r"\exp(A)\exp(B)", "matrices": mats}
    ).json()
    assert body["status"] == "conditions_not_satisfied" and body["result_latex"] is None
    body = authed.post(
        "/v1/compute/matrix",
        json={"input": "AB", "matrices": [{"name": "A", "rows": 2, "cols": 3}, mats[1]]},
    ).json()
    assert body["status"] == "parse_error"
    assert authed.post("/v1/compute/matrix", json={"input": "A", "matrices": []}).status_code == 422
    assert (
        authed.post(
            "/v1/compute/matrix", json={"input": "A", "matrices": [{"name": "A"}]}
        ).status_code
        == 422
    )
    assert (
        authed.post(
            "/v1/compute/matrix",
            json={"input": "A", "matrices": mats, "action": "invert"},
        ).status_code
        == 422
    )


def test_matrix_calculus_requires_authentication(client: TestClient) -> None:
    assert client.post("/v1/compute/matrix", json=MATRIX_REGRESSION).status_code == 401


def test_matrix_calculus_logs_no_user_input(capsys) -> None:  # type: ignore[no-untyped-def]
    marker = "zzMATRIXMARKERzz"
    app = build_app()
    with TestClient(app) as c:
        register(c)
        c.post(
            "/v1/compute/matrix",
            json={"input": rf"\exp(t{marker})", "matrices": [{"name": marker, "rows": 2}]},
        )
    out = capsys.readouterr().out
    assert "compute" in out and marker not in out
