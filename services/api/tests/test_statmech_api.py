"""POST /v1/compute/statmech: results, refusals, history, ownership and logs."""

import json

from fastapi.testclient import TestClient

from .conftest import PASSWORD, build_app, register

URL = "/v1/compute/statmech"
TWO_LEVEL = {"kind": "levels", "levels": [{"energy": "0"}, {"energy": "epsilon"}]}
POSITIVE = {"epsilon": ["positive"]}


def post(c: TestClient, **body: object) -> dict:  # type: ignore[type-arg]
    payload = {"system": TWO_LEVEL, "assumptions": POSITIVE, "action": "derive", **body}
    r = c.post(URL, json=payload)
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def test_requires_authentication(client: TestClient) -> None:
    assert client.post(URL, json={"system": TWO_LEVEL}).status_code == 401


def test_derivation_checks_and_history(authed: TestClient) -> None:
    body = post(authed, quantities=["Z", "U", "Var_E"])
    assert body["status"] == "ok"
    names = [q["name"] for q in body["quantities"]]
    assert names == ["Z", "U", "Var_E"]
    u = next(q for q in body["quantities"] if q["name"] == "U")
    assert {c["name"] for c in u["checks"]} >= {"enumeration", "finite_difference"}
    assert body["verification"]["passed"] and body["verification"]["points_agreed"] >= 4
    assert [s["name"] for s in body["steps"]][:2] == ["levels", "partition_function"]
    (item,) = authed.get("/v1/history").json()
    assert item["operation"] == "statmech" and item["id"] == body["calculation_id"]
    assert item["result"]["request"]["system"]["kind"] == "levels"  # the form can be restored


def test_refusals_carry_a_reason_and_no_formula(authed: TestClient) -> None:
    bad = post(authed, system={"kind": "levels", "levels": [{"energy": "0", "degeneracy": "0"}]})
    assert bad["status"] == "conditions_not_satisfied" and bad["reason"] == "invalid_degeneracy"
    assert bad["quantities"] == []
    wrong_fixed = post(
        authed,
        system={"kind": "levels", "levels": [{"energy": "1/V"}, {"energy": "2/V"}]},
        parameters=["V"],
        assumptions={"V": ["positive"]},
        partials=[{"of": "F", "variable": "V", "held_fixed": []}],
    )
    assert wrong_fixed["reason"] == "ensemble_mismatch"
    divergent = post(authed, system={"kind": "level_family", "energy": "-n"}, action="partition")
    assert divergent["reason"] == "divergent"


def test_conventions_come_back_as_questions(authed: TestClient) -> None:
    asked = post(
        authed,
        system={"kind": "classical", "hamiltonian": "p**2/2", "variables": [{"name": "p"}]},
        action="partition",
    )
    assert asked["reason"] == "convention_unconfirmed" and asked["questions"]
    answered = post(
        authed,
        system={
            "kind": "classical",
            "hamiltonian": "p**2/2",
            "variables": [{"name": "p"}],
            "phase_space_cell": "1",
        },
        quantities=["U"],
    )
    assert answered["status"] == "ok"


def test_validation(authed: TestClient) -> None:
    assert authed.post(URL, json={"system": {"kind": "nope"}}).status_code == 422
    assert authed.post(URL, json={"system": TWO_LEVEL, "action": "nope"}).status_code == 422
    too_long = {"kind": "levels", "levels": [{"energy": "1" * 30_000}]}
    assert authed.post(URL, json={"system": too_long}).status_code == 422
    many = {"kind": "levels", "levels": [{"energy": "1"}] * 100}
    assert authed.post(URL, json={"system": many}).status_code == 422


def test_evaluation_with_a_sweep(authed: TestClient) -> None:
    body = post(
        authed,
        action="evaluate",
        quantities=["U"],
        values={"epsilon": "1", "k_B": "1"},
        sweep={"variable": "beta", "start": "1/2", "stop": "2", "points": 4},
    )
    assert body["status"] == "ok" and len(body["table"]["rows"]) == 4


def test_a_step_cannot_join_another_users_derivation(authed: TestClient, app) -> None:  # type: ignore[no-untyped-def]
    derivation = authed.post("/v1/derivations").json()["id"]
    mine = post(authed, quantities=["U"], derivation_id=derivation)
    assert mine["calculation_id"]
    with TestClient(app) as other:
        register(other, "bob@example.com")
        r = other.post(
            URL,
            json={
                "system": TWO_LEVEL,
                "assumptions": POSITIVE,
                "action": "derive",
                "derivation_id": derivation,
            },
        )
        assert r.status_code == 404


def test_logs_have_events_and_never_the_problem(capsys) -> None:  # type: ignore[no-untyped-def]
    marker = "zzmarkerzz"
    app = build_app()
    with TestClient(app) as c:
        register(c)
        post(
            c,
            system={"kind": "levels", "levels": [{"energy": "0"}, {"energy": marker}]},
            assumptions={marker: ["positive"]},
            quantities=["U"],
        )
        post(c, system={"kind": "levels", "levels": [{"energy": "0", "degeneracy": "0"}]})
        c.post("/v1/auth/login", json={"email": "ada@example.com", "password": PASSWORD})
    out = capsys.readouterr().out
    assert marker not in out and "exp(" not in out
    assert PASSWORD not in out and "ada@example.com" not in out
    events = [
        json.loads(line)
        for line in out.splitlines()
        if line.startswith("{") and '"statmech_event"' in line
    ]
    assert events, "the stage events should be logged"
    for event in events:
        assert event["operation"] == "statmech" and event["request_id"]
        assert event["status"] in {"ok", "failed"} and event["duration_ms"] >= 0
        assert event["system_kind"] == "levels"
    assert {"parse", "symbolic", "verify"} <= {e["stage"] for e in events}
    assert any(e.get("reason") == "invalid_degeneracy" for e in events)
    compute = [json.loads(line) for line in out.splitlines() if '"event": "compute"' in line]
    assert any(c.get("reason") == "invalid_degeneracy" for c in compute)
