"""POST /v1/compute/distribution: results, refusals, history, ownership and logs."""

import json

from fastapi.testclient import TestClient

from .conftest import PASSWORD, build_app, register

URL = "/v1/compute/distribution"
REAL_X = {"x": ["real"]}


def post(c: TestClient, **body: object) -> dict:  # type: ignore[type-arg]
    payload = {"format": "text", "variable": "x", "assumptions": REAL_X, **body}
    r = c.post(URL, json=payload)
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def test_requires_authentication(client: TestClient) -> None:
    assert client.post(URL, json={"input": "x", "variable": "x"}).status_code == 401


def test_integration_result_steps_and_history(authed: TestClient) -> None:
    body = post(
        authed,
        input="x**2*DiracDelta(x**2 - 4)",
        action="integrate",
        lower="-oo",
        upper="oo",
    )
    assert body["status"] == "ok" and body["result_text"] == "2"
    assert body["equality"] == "value"
    assert [s["entry_id"] for s in body["steps"]][-1] == "distributions.delta_sifting"
    assert body["verification"]["passed"] and body["verification"]["conditions_established"]
    assert [e["stage"] for e in body["diagnostics"]["events"]][-1] == "completed"
    (item,) = authed.get("/v1/history").json()
    assert item["operation"] == "distribution" and item["id"] == body["calculation_id"]
    assert item["result"]["request"]["action"] == "integrate"  # the form can be restored


def test_refusals_carry_a_reason_and_no_result(authed: TestClient) -> None:
    repeated = post(authed, input="DiracDelta(x**2)", action="simplify")
    assert repeated["status"] == "unsupported" and repeated["reason"] == "singular_root"
    assert repeated["result_latex"] is None

    ambiguous = post(authed, input="DiracDelta(x - 1)", action="integrate", lower="0", upper="1")
    assert ambiguous["status"] == "conditions_not_satisfied"
    assert ambiguous["reason"] == "integration_domain_ambiguous"

    chosen = post(
        authed,
        input="DiracDelta(x - 1)",
        action="integrate",
        lower="0",
        upper="1",
        endpoint="half",
    )
    assert chosen["status"] == "ok" and chosen["result_text"] == "1/2"

    unknown_sign = post(authed, input="DiracDelta(a*x)", assumptions={**REAL_X, "a": ["real"]})
    assert unknown_sign["reason"] == "assumptions_insufficient"
    assert unknown_sign["blocked"][0]["entry_id"] == "distributions.delta_scaling"


def test_validation_and_latex(authed: TestClient) -> None:
    assert (
        authed.post(URL, json={"input": "x", "variable": "x", "action": "nope"}).status_code == 422
    )
    assert (
        authed.post(URL, json={"input": "x", "variable": "x", "endpoint": "mid"}).status_code == 422
    )
    assert authed.post(URL, json={"input": "x" * 30_000, "variable": "x"}).status_code == 422
    latex = authed.post(
        URL,
        json={
            "input": r"\operatorname{Heaviside}(x - 1)",
            "format": "latex",
            "variable": "x",
            "action": "differentiate",
            "assumptions": REAL_X,
        },
    ).json()
    assert latex["status"] == "ok" and "DiracDelta" in latex["result_latex"]


def test_other_operations_refuse_distributions_over_http(authed: TestClient) -> None:
    r = authed.post(
        "/v1/compute/simplify",
        json={"input": "DiracDelta(x)", "format": "text", "assumptions": REAL_X},
    )
    assert r.json()["status"] == "unsupported"


def test_a_step_cannot_join_another_users_derivation(authed: TestClient, app) -> None:  # type: ignore[no-untyped-def]
    derivation = authed.post("/v1/derivations").json()["id"]
    mine = post(authed, input="DiracDelta(x)", derivation_id=derivation)
    assert mine["calculation_id"]
    with TestClient(app) as other:
        register(other, "bob@example.com")
        r = other.post(
            URL,
            json={
                "input": "DiracDelta(x)",
                "format": "text",
                "variable": "x",
                "assumptions": REAL_X,
                "derivation_id": derivation,
            },
        )
        assert r.status_code == 404


def test_logs_have_events_and_never_the_input(capsys) -> None:  # type: ignore[no-untyped-def]
    marker = "zzmarkerzz"
    app = build_app()
    with TestClient(app) as c:
        register(c)
        post(
            c,
            input=f"{marker}*DiracDelta(x - 1)",
            assumptions={**REAL_X, marker: ["positive"]},
        )
        post(c, input="DiracDelta(x**2)")
        c.post("/v1/auth/login", json={"email": "ada@example.com", "password": PASSWORD})
    out = capsys.readouterr().out
    assert marker not in out and "DiracDelta" not in out
    assert PASSWORD not in out and "ada@example.com" not in out
    events = [
        json.loads(line)
        for line in out.splitlines()
        if line.startswith("{") and '"distribution_event"' in line
    ]
    assert events, "the stage events should be logged"
    for event in events:
        assert event["operation"] == "distribution"
        assert event["request_id"]  # the trace id of the request
        assert event["status"] in {"ok", "failed"}
        assert event["duration_ms"] >= 0
        assert set(event) <= {
            "event",
            "level",
            "timestamp",
            "request_id",
            "operation",
            "stage",
            "status",
            "duration_ms",
            "reason",
            "user_id",
        }
    stages = {e["stage"] for e in events}
    assert {"parsed", "verification_attempted", "completed", "rejected"} <= stages
    assert any(e["reason"] == "singular_root" for e in events)
    compute = [json.loads(line) for line in out.splitlines() if '"event": "compute"' in line]
    assert any(c.get("reason") == "singular_root" for c in compute)


def test_delta_notation_is_asked_then_confirmed(authed: TestClient) -> None:
    kw = {
        "format": "latex",
        "input": r"x^2 \delta(x^2 - 4)",
        "action": "integrate",
        "lower": "-oo",
        "upper": "oo",
    }
    asked = post(authed, **kw)
    assert (
        asked["status"] == "conditions_not_satisfied" and asked["reason"] == "notation_unconfirmed"
    )
    assert [q["name"] for q in asked["questions"]] == ["delta"] and asked["result_latex"] is None

    answered = post(authed, **kw, notation={"delta": "dirac_delta"})
    assert answered["status"] == "ok" and answered["result_text"] == "2"
    assert answered["notation"] == {"delta": "dirac_delta"}

    symbol = post(authed, **kw, notation={"delta": "symbol"})
    assert symbol["status"] == "unsupported"

    items = authed.get("/v1/history").json()
    assert any(i["result"]["request"].get("notation") == {"delta": "dirac_delta"} for i in items)


def test_notation_rejects_names_and_readings_that_do_not_fit(authed: TestClient) -> None:
    for bad in ({"delta": "heaviside"}, {"theta": "dirac_delta"}, {"phi": "symbol"}):
        r = authed.post(URL, json={"input": "x", "variable": "x", "notation": bad})
        assert r.status_code == 422, bad
