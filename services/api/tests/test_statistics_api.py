"""POST /v1/compute/{sums,expectation,statistics}: results, refusals, history, ownership and logs."""

import json

from fastapi.testclient import TestClient

from .conftest import PASSWORD, build_app, register

SUMS, EXPECTATION, STATISTICS = (
    "/v1/compute/sums",
    "/v1/compute/expectation",
    "/v1/compute/statistics",
)
A = {"n": ["integer", "positive"], "x": ["real"], "mu": ["real"]}


def post(c: TestClient, url: str, **body: object) -> dict:  # type: ignore[type-arg]
    r = c.post(url, json=body)
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def test_all_three_require_authentication(client: TestClient) -> None:
    for url in (SUMS, EXPECTATION, STATISTICS):
        assert (
            client.post(url, json={"input": "x", "random": ["X"], "family": "normal"}).status_code
            == 401
        )


def test_sums_expand_verify_and_history(authed: TestClient) -> None:
    body = post(
        authed,
        SUMS,
        input="Sum((x[i] - mu)**2, (i, 1, n))",
        format="text",
        assumptions=A,
        action="expand",
    )
    assert body["status"] == "ok" and body["verification"]["passed"]
    assert "not proved for every size" in body["message"]
    (item,) = authed.get("/v1/history").json()
    assert item["operation"] == "sums" and item["id"] == body["calculation_id"]
    assert item["result"]["request"]["action"] == "expand"  # the form can be restored


def test_the_classic_false_identity_is_not_verified(authed: TestClient) -> None:
    body = post(
        authed, SUMS, input="Sum(x[i]**2, (i, 1, n))", format="text", assumptions=A,
        action="check", candidate={"result": "Sum(x[i], (i, 1, n))**2/n"},
    )  # fmt: skip
    assert body["status"] == "not_verified" and body["result_text"] is None


def test_a_sum_is_refused_by_every_other_operation(authed: TestClient) -> None:
    for url, extra in [
        ("/v1/compute/simplify", {}),
        ("/v1/compute/differentiate", {"variables": [{"name": "mu", "order": 1}]}),
        ("/v1/compute/integrate", {"variable": "mu"}),
    ]:
        body = post(
            authed, url, input="Sum(x[i], (i, 1, n))", format="text", assumptions=A, **extra
        )
        assert body["status"] in {"parse_error", "unsupported"}, url


def test_expectation_marks_declared_independence_as_a_premise(authed: TestClient) -> None:
    free = post(authed, EXPECTATION, input="Var(X + Y)", format="text", random=["X", "Y"])
    assert free["status"] == "ok" and free["evidence"] == "identity" and free["given"] == []
    given = post(
        authed,
        EXPECTATION,
        input="Var(X + Y)",
        format="text",
        random=["X", "Y"],
        independent=[["X", "Y"]],
    )
    assert given["evidence"] == "conditional" and given["given"] == ["X and Y are independent"]
    refused = post(authed, EXPECTATION, input="E[Z]", format="text", random=["X"])
    assert refused["status"] == "parse_error"


def test_the_gaussian_mle_is_derived_and_checked(authed: TestClient) -> None:
    body = post(
        authed, STATISTICS, family="normal", parametrization="variance", observation="x",
        sample_size="n", iid=True, estimate=["mu", "v"], action="mle", assumptions=A,
    )  # fmt: skip
    assert body["status"] == "ok" and body["evidence"] == "conditional"
    mu, v = body["estimates"]
    assert mu["text"] == "Sum(x[i], (i, 1, n))/n" and mu["optimum"] == "local"
    assert v["admissible_if"] and all(c["passed"] for c in body["checks"])
    items = [i for i in authed.get("/v1/history").json() if i["operation"] == "statistical_model"]
    assert items and items[0]["result"]["request"]["family"] == "normal"


def test_statistics_refusals_carry_an_enumerated_reason(authed: TestClient) -> None:
    cases = [
        (dict(family="normal", parameters={"mu": "0", "sigma": "-1"}, action="moments"), "parameter_invalid"),
        (dict(family="weibull", action="moments"), "unsupported_family"),
        (dict(family="cauchy", action="moments"), "convergence_not_established"),
        (dict(family="normal", parameters={"sigma": "2"},
                observation="x", sample_size="n", estimate=["mu"], action="mle", assumptions=A), "independence_not_declared"),
        (dict(family="uniform_0_theta", estimate=["theta"], action="fisher_information", assumptions={"theta": ["positive"]}), "regularity_not_satisfied"),
    ]  # fmt: skip
    for body, reason in cases:
        out = post(authed, STATISTICS, **body)
        assert out["reason"] == reason and out["status"] != "ok", reason


def test_validation_rejects_malformed_requests(authed: TestClient) -> None:
    assert (
        authed.post(STATISTICS, json={"family": "normal", "action": "no_such_action"}).status_code
        == 422
    )
    assert authed.post(EXPECTATION, json={"input": "E[X]", "random": []}).status_code == 422
    assert authed.post(SUMS, json={"input": "x[1]", "action": "frobnicate"}).status_code == 422
    huge = "1+" * 1500
    r = authed.post(
        STATISTICS, json={"family": "normal", "parameters": {"mu": huge}, "action": "moments"}
    )
    assert r.status_code == 422


def test_another_users_derivation_is_not_found(client: TestClient) -> None:
    register(client)
    derivation = client.post("/v1/derivations").json()["id"]
    app = build_app()
    with TestClient(app) as other:
        register(other, "bob@example.com")
        for url, body in [
            (SUMS, {"input": "x[1]", "format": "text", "assumptions": A}),
            (EXPECTATION, {"input": "E[X]", "format": "text", "random": ["X"]}),
            (STATISTICS, {"family": "normal", "action": "moments"}),
        ]:
            assert other.post(url, json={**body, "derivation_id": derivation}).status_code == 404, (
                url
            )


def test_logs_hold_enumerated_fields_and_never_the_input(capsys) -> None:  # type: ignore[no-untyped-def]
    marker = "zzmarkerzz"
    app = build_app()
    with TestClient(app) as c:
        register(c)
        post(
            c,
            SUMS,
            input=f"Sum({marker}[i], (i, 1, n))",
            format="text",
            assumptions={**A, marker: ["real"]},
            action="expand",
        )
        post(c, EXPECTATION, input=f"E[{marker}]", format="text", random=[marker])
        post(
            c,
            STATISTICS,
            family="normal",
            parameters={"mu": marker, "sigma": "-1"},
            action="moments",
            data=["7777777"],
        )
        c.post("/v1/auth/login", json={"email": "ada@example.com", "password": PASSWORD})
    out = capsys.readouterr().out
    assert marker not in out and "7777777" not in out
    assert PASSWORD not in out and "ada@example.com" not in out
    compute = [json.loads(line) for line in out.splitlines() if '"event": "compute"' in line]
    assert {c["operation"] for c in compute} >= {"sums", "expectation", "statistical_model"}
    for entry in compute:
        assert set(entry) <= {
            "event", "level", "timestamp", "request_id", "operation", "status",
            "duration_ms", "reason", "user_id",
        }  # fmt: skip


def test_matrix_gradients_and_candidate_checks_over_the_api(authed: TestClient) -> None:
    matrices = [
        {"name": "x", "rows": 3, "cols": 1},
        {"name": "mu", "rows": 3, "cols": 1},
        {"name": "S", "rows": 3, "cols": 3, "positive_definite": True},
    ]
    base = {
        "input": "-1/2*logdet(S) - 1/2*transpose(x - mu)*inv(S)*(x - mu)",
        "format": "text",
        "matrices": matrices,
        "action": "gradient",
        "variable": "S",
    }
    body = post(authed, "/v1/compute/matrix", **base)
    assert body["status"] == "ok" and body["derivative"]["convention"] == "symmetric_frobenius"
    assert body["derivative"]["alternatives"][0]["convention"] == "symmetric_entrywise"
    assert body["result_text"] and body["verification"]["passed"]
    wrong = post(authed, "/v1/compute/matrix", **base, candidate={"result": "-1/2*inv(S)"})
    assert wrong["status"] == "not_verified" and wrong["result_text"] is None
    latex = post(authed, "/v1/compute/matrix", **{**base, "format": "latex"})
    assert latex["status"] == "unsupported"  # the gradient actions read the plain-text syntax
    undeclared = post(
        authed,
        "/v1/compute/matrix",
        **{**base, "input": "logdet(X)", "matrices": [{"name": "X", "rows": 3}], "variable": "X"},
    )
    assert undeclared["reason"] == "invertibility_not_declared"
