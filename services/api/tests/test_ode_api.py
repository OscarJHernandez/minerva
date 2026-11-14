"""The differential-equation endpoints: results, saved steps, validation, ownership, logs."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient
from minerva_api.logging import ALLOWED_FIELDS

from .conftest import build_app, register

GROWTH = {"equations": ["y' = x y"], "independent": "x", "conditions": ["y(0) = 2"]}
OSCILLATOR = {
    "equations": ["y'' + c y' + y = 0"],
    "independent": "t",
    "conditions": ["y(0) = 1", "y'(0) = 0"],
    "parameters": {"c": 0.5},
    "t_end": 10,
}


def test_analyze_describes_the_equation_and_saves_the_step(authed: TestClient) -> None:
    r = authed.post(
        "/v1/compute/ode/analyze",
        json={"equations": [r"y'' + 0.2 y' + y + y^3 = 0"], "independent": "t"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["structure"]["order"] == 2 and body["structure"]["linear"] is False
    methods = {m["id"]: m for m in body["methods"]}
    assert (
        methods["numerical_ivp"]["rank"] == 1
        and methods["symbolic_closed_form"]["applicable"] is False
    )
    assert body["interpretation"]["independent"] == "t"
    (item,) = authed.get("/v1/history").json()
    assert item["operation"] == "ode_analyze" and item["id"] == body["calculation_id"]
    assert item["result"]["request"]["equations"] == [r"y'' + 0.2 y' + y + y^3 = 0"]
    assert "save" not in item["result"]["request"]


def test_solve_returns_a_verified_closed_form(authed: TestClient) -> None:
    r = authed.post("/v1/compute/ode/solve", json=GROWTH)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["method"] == "closed_form"
    (solution,) = body["solutions"]
    assert solution["family"] == "particular" and solution["verification"]["passed"] is True
    assert body["verification"]["points_agreed"] == body["verification"]["points_checked"] >= 4
    assert authed.get("/v1/history").json()[0]["operation"] == "ode_solve_symbolic"


def test_numeric_returns_a_checked_trajectory(authed: TestClient) -> None:
    r = authed.post("/v1/compute/ode/numeric", json=OSCILLATOR)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["method_used"] == "DOP853"
    assert len(body["t"]) == len(body["series"][0]["values"]) == 200
    assert body["verification"]["passed"] is True
    (item,) = authed.get("/v1/history").json()
    assert item["operation"] == "ode_solve_numerical"
    assert item["result"]["request"]["parameters"] == {"c": 0.5}


def test_numeric_does_not_invent_numbers(authed: TestClient) -> None:
    body = authed.post("/v1/compute/ode/numeric", json={**OSCILLATOR, "parameters": {}}).json()
    assert body["status"] == "parse_error" and "missing: c" in body["message"]
    assert body["series"] == [] and body["t"] == []
    body = authed.post("/v1/compute/ode/numeric", json={**OSCILLATOR, "conditions": []}).json()
    assert body["status"] == "parse_error" and "Minerva does not choose them" in body["message"]


def test_refusals_are_statuses_and_saved_for_the_form(authed: TestClient) -> None:
    body = authed.post("/v1/compute/ode/analyze", json={"equations": ["y' = y"]}).json()
    assert body["status"] == "parse_error" and body["structure"] is None
    body = authed.post(
        "/v1/compute/ode/solve", json={"equations": ["y' = y^2 + x"], "independent": "x"}
    ).json()
    assert body["status"] == "unsupported" and "does not show that none exists" in body["message"]
    assert all(h["result"]["request"] for h in authed.get("/v1/history").json())


def test_bad_shapes_are_422(authed: TestClient) -> None:
    for path, payload in (
        ("analyze", {"equations": []}),
        ("analyze", {"equations": ["y' = y"] * 9}),
        ("analyze", {"equations": ["y' = y"], "assumptions": {"y": ["imaginary"]}}),
        ("analyze", {"equations": ["y' = y"], "assumptions": {"y": ["noncommutative"]}}),
        ("solve", {"equations": ["y' = y" + " + y" * 1000]}),
        ("numeric", {**OSCILLATOR, "t_end": "soon"}),
        ("numeric", {**OSCILLATOR, "rtol": 0}),
        ("numeric", {**OSCILLATOR, "samples": 10**6}),
        ("numeric", {**OSCILLATOR, "method": "Euler"}),
    ):
        assert authed.post(f"/v1/compute/ode/{path}", json=payload).status_code == 422, payload


def test_save_can_be_turned_off(authed: TestClient) -> None:
    body = authed.post("/v1/compute/ode/solve", json={**GROWTH, "save": False}).json()
    assert body["calculation_id"] is None and authed.get("/v1/history").json() == []


def test_a_result_joins_only_the_users_own_derivation(
    authed: TestClient, client: TestClient
) -> None:
    mine = authed.post("/v1/derivations", json={}).json()["id"]
    body = authed.post("/v1/compute/ode/solve", json={**GROWTH, "derivation_id": mine}).json()
    assert body["calculation_id"]
    other = TestClient(client.app)
    register(other, "grace@example.com")
    assert (
        other.post("/v1/compute/ode/solve", json={**GROWTH, "derivation_id": mine}).status_code
        == 404
    )


def test_the_endpoints_require_authentication(client: TestClient) -> None:
    for path, payload in (("analyze", GROWTH), ("solve", GROWTH), ("numeric", OSCILLATOR)):
        assert client.post(f"/v1/compute/ode/{path}", json=payload).status_code == 401


def _events(out: str) -> list[dict[str, Any]]:
    events = []
    for line in out.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def test_the_logs_hold_no_equation_name_or_value(capsys) -> None:  # type: ignore[no-untyped-def]
    marker = "zzODEMARKERzz"
    with TestClient(build_app()) as c:
        register(c)
        responses = [
            c.post(
                "/v1/compute/ode/numeric",
                json={
                    "equations": [f"{marker}' = -kappa_{marker}*{marker}"],
                    "format": "text",
                    "independent": "s",
                    "conditions": [f"{marker}(31337) = 2"],
                    "parameters": {f"kappa_{marker}": 0.7},
                    "t_end": 31340,
                },
            ),
            c.post(
                "/v1/compute/ode/solve",
                json={
                    **GROWTH,
                    "equations": [f"{marker}' = x*{marker}"],
                    "format": "text",
                    "conditions": [f"{marker}(0) = 5"],
                },
            ),
            c.post("/v1/compute/ode/analyze", json=GROWTH),
        ]
    out = capsys.readouterr().out
    assert marker not in out and "31337" not in out and "kappa" not in out
    events = _events(out)
    ode = [e for e in events if e.get("event") == "ode"]
    stages = [e for e in events if e.get("event") == "ode_stage"]
    assert {e["operation"] for e in ode} == {
        "ode_analyze",
        "ode_solve_symbolic",
        "ode_solve_numerical",
    }
    for e in [*ode, *stages]:
        assert set(e) <= ALLOWED_FIELDS and e.get("request_id")
    numeric = next(e for e in ode if e["operation"] == "ode_solve_numerical")
    assert numeric["trace_id"] == responses[0].headers["X-Request-ID"]
    assert numeric["ode_method"] == "DOP853" and isinstance(numeric["steps"], int)
    assert {"read", "integrate", "total"} <= {e["stage"] for e in stages}


def test_solve_runs_a_series_and_numeric_runs_a_boundary_problem(authed: TestClient) -> None:
    series = authed.post(
        "/v1/compute/ode/solve", json={**GROWTH, "method": "series", "order": 4}
    ).json()
    assert series["status"] == "ok" and series["method"] == "series"
    assert series["solutions"][0]["expansion"]["kind"] == "taylor"
    bvp = authed.post(
        "/v1/compute/ode/numeric",
        json={
            "equations": ["y'' = -y"],
            "independent": "x",
            "kind": "bvp",
            "conditions": ["y(0) = 0", "y(1) = 1"],
        },
    ).json()
    assert bvp["status"] == "ok" and bvp["method_used"] == "solve_bvp"
    saved = authed.get("/v1/history").json()
    assert {h["operation"] for h in saved} == {"ode_solve_symbolic", "ode_solve_numerical"}
    for payload in ({**GROWTH, "method": "magic"}, {**GROWTH, "order": 99}):
        assert authed.post("/v1/compute/ode/solve", json=payload).status_code == 422


def test_analyze_can_describe_equilibria_and_stability(authed: TestClient) -> None:
    body = authed.post(
        "/v1/compute/ode/analyze",
        json={
            "equations": ["y'' + 0.2 y' + y + y^3 = 0"],
            "independent": "t",
            "detail": "qualitative",
            "window": [-2, 2, -1, 1],
        },
    ).json()
    q = body["qualitative"]
    assert body["status"] == "ok" and q["status"] == "ok" and q["complete"] is True
    assert [e["classification"] for e in q["equilibria"]] == ["stable_spiral"]
    assert "linearisation" in q["local_note"] and len(q["direction_field"]["x"]) == 225
    plain = authed.post(
        "/v1/compute/ode/analyze", json={"equations": ["y' = y"], "independent": "t"}
    ).json()
    assert plain["qualitative"] is None
    for bad in ({"window": [1, 1, 0, 1]}, {"window": [0, 1, 0]}, {"detail": "everything"}):
        payload = {"equations": ["y' = y"], "independent": "t", **bad}
        assert authed.post("/v1/compute/ode/analyze", json=payload).status_code == 422
