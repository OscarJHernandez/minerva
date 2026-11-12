from fastapi.testclient import TestClient

from .conftest import register

POSITIVE = {"x": ["positive"], "y": ["positive"]}


def new_derivation(c: TestClient) -> str:
    r = c.post("/v1/derivations")
    assert r.status_code == 201
    return str(r.json()["id"])


def step(c: TestClient, path: str, **body: object) -> dict:  # type: ignore[type-arg]
    r = c.post(f"/v1/compute/{path}", json=body)
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def test_requires_authentication(client: TestClient) -> None:
    assert client.post("/v1/derivations").status_code == 401
    assert client.get("/v1/derivations").status_code == 401
    assert (
        client.post("/v1/compute/apply-identity", json={"input": "x", "rule_id": "a"}).status_code
        == 401
    )


def test_chain_through_text_form(authed: TestClient) -> None:
    d = new_derivation(authed)
    s1 = step(authed, "simplify", input=r"\sin^2 x + \cos^2 x + x", derivation_id=d)
    assert s1["result_text"] == "x + 1"
    s2 = step(
        authed,
        "differentiate",
        input=s1["result_text"],
        format="text",
        variables=[{"name": "x", "order": 1}],
        derivation_id=d,
        parent_id=s1["calculation_id"],
    )
    assert s2["status"] == "ok" and s2["result_text"] == "1"
    got = authed.get(f"/v1/derivations/{d}").json()
    assert [s["operation"] for s in got["steps"]] == ["simplify", "differentiate"]
    assert got["steps"][0]["parent_id"] is None
    assert got["steps"][1]["parent_id"] == s1["calculation_id"]
    assert got["steps"][1]["format"] == "text"
    summary = authed.get("/v1/derivations").json()
    assert summary[0]["steps"] == 2 and summary[0]["first_input"] == r"\sin^2 x + \cos^2 x + x"


def test_apply_identity_with_conditions_proven(authed: TestClient) -> None:
    d = new_derivation(authed)
    r = authed.post(
        "/v1/compute/apply-identity",
        json={
            "input": r"\log(xy)",
            "rule_id": "log.product",
            "assumptions": POSITIVE,
            "derivation_id": d,
        },
    ).json()
    assert r["status"] == "ok"
    assert r["result_text"] == "log(x) + log(y)"
    assert r["verification"]["passed"] is True
    assert r["entry"]["id"] == "log.product"
    saved = authed.get(f"/v1/derivations/{d}").json()["steps"]
    assert saved[0]["operation"] == "apply_identity" and saved[0]["assumptions"] == POSITIVE


def test_apply_identity_blocked_without_assumptions(authed: TestClient) -> None:
    r = authed.post(
        "/v1/compute/apply-identity", json={"input": r"\log(xy)", "rule_id": "log.product"}
    ).json()
    assert r["status"] == "conditions_not_satisfied"
    assert r["result_latex"] is None and r["result_text"] is None
    assert {c["variable"] for c in r["unmet_conditions"]} == {"a", "b"}


def test_apply_identity_that_does_not_occur_is_unchanged(authed: TestClient) -> None:
    r = authed.post(
        "/v1/compute/apply-identity",
        json={"input": "x + 1", "format": "text", "rule_id": "log.product"},
    ).json()
    assert r["status"] == "unchanged" and r["result_text"] is None


def test_unknown_identity_is_404(authed: TestClient) -> None:
    r = authed.post("/v1/compute/apply-identity", json={"input": "x", "rule_id": "no.such.entry"})
    assert r.status_code == 404


def test_stated_entries_are_never_applied(authed: TestClient) -> None:
    from minerva_knowledge import KnowledgeBase

    stated = next(e for e in KnowledgeBase.load().entries if e.rule is None)
    r = authed.post("/v1/compute/apply-identity", json={"input": "x", "rule_id": stated.id})
    assert r.status_code == 404


def test_derivations_are_private(authed: TestClient, app) -> None:  # type: ignore[no-untyped-def]
    d = new_derivation(authed)
    first = step(authed, "simplify", input="x+x", format="text", derivation_id=d)
    with TestClient(app) as other:
        register(other, "bob@example.com")
        assert other.get(f"/v1/derivations/{d}").status_code == 404
        assert other.delete(f"/v1/derivations/{d}").status_code == 404
        r = other.post("/v1/compute/simplify", json={"input": "x", "derivation_id": d})
        assert r.status_code == 404
        d2 = new_derivation(other)
        r = other.post(
            "/v1/compute/simplify",
            json={"input": "x", "derivation_id": d2, "parent_id": first["calculation_id"]},
        )
        assert r.status_code == 422
        assert (
            other.get("/v1/derivations").json() == []
        )  # nothing of the first user's, no steps yet


def test_parent_must_be_in_the_same_derivation(authed: TestClient) -> None:
    d1, d2 = new_derivation(authed), new_derivation(authed)
    s = step(authed, "simplify", input="x+x", format="text", derivation_id=d1)
    r = authed.post(
        "/v1/compute/simplify",
        json={"input": "x", "derivation_id": d2, "parent_id": s["calculation_id"]},
    )
    assert r.status_code == 422
    assert (
        authed.post(
            "/v1/compute/simplify", json={"input": "x", "parent_id": s["calculation_id"]}
        ).status_code
        == 422
    )


def test_unsaved_step_ignores_links(authed: TestClient) -> None:
    r = step(authed, "identities", input=r"\log(xy)", save=False)
    assert r["calculation_id"] is None


def test_delete_derivation_removes_its_steps(authed: TestClient) -> None:
    d = new_derivation(authed)
    step(authed, "simplify", input="x+x", format="text", derivation_id=d)
    step(authed, "simplify", input="y+y", format="text")  # not in the derivation
    assert authed.delete(f"/v1/derivations/{d}").status_code == 204
    assert authed.get(f"/v1/derivations/{d}").status_code == 404
    assert [h["input"] for h in authed.get("/v1/history").json()] == ["y+y"]


def test_deleting_a_step_unlinks_its_children(authed: TestClient) -> None:
    d = new_derivation(authed)
    a = step(authed, "simplify", input="x+x", format="text", derivation_id=d)
    step(
        authed,
        "simplify",
        input="2*x",
        format="text",
        derivation_id=d,
        parent_id=a["calculation_id"],
    )
    assert authed.delete(f"/v1/history/{a['calculation_id']}").status_code == 204
    steps = authed.get(f"/v1/derivations/{d}").json()["steps"]
    assert len(steps) == 1 and steps[0]["parent_id"] is None


def test_clear_history_removes_derivations(authed: TestClient) -> None:
    d = new_derivation(authed)
    step(authed, "simplify", input="x+x", format="text", derivation_id=d)
    assert authed.delete("/v1/history").status_code == 204
    assert authed.get("/v1/derivations").json() == []
