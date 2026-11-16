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


def chain(c: TestClient, d: str) -> dict[str, str]:
    """root -> mid -> leaf, plus a sibling of mid (`other`) and an unrelated root (`lone`)."""
    ids: dict[str, str] = {}
    ids["root"] = step(c, "simplify", input="x+x", format="text", derivation_id=d)["calculation_id"]
    for name, text, parent in (
        ("mid", "2*x", "root"),
        ("leaf", "2*x", "mid"),
        ("other", "2*x", "root"),
    ):
        ids[name] = step(
            c, "simplify", input=text, format="text", derivation_id=d, parent_id=ids[parent]
        )["calculation_id"]
    ids["lone"] = step(c, "simplify", input="y+y", format="text", derivation_id=d)["calculation_id"]
    return ids


def live(c: TestClient, d: str) -> set[str]:
    return {s["id"] for s in c.get(f"/v1/derivations/{d}").json()["steps"]}


def test_deleting_a_step_removes_its_subtree_and_keeps_its_siblings(authed: TestClient) -> None:
    # This replaces "children are kept and unlinked": a continuation of a removed step has nothing
    # to continue from, and an undo brings the whole set back.
    d = new_derivation(authed)
    ids = chain(authed, d)
    r = authed.delete(f"/v1/history/{ids['mid']}")
    assert r.status_code == 200
    assert set(r.json()["ids"]) == {ids["mid"], ids["leaf"]}
    assert live(authed, d) == {ids["root"], ids["other"], ids["lone"]}
    assert authed.get(f"/v1/history/{ids['leaf']}").status_code == 404
    # nothing may continue from a deleted step
    again = authed.post(
        "/v1/compute/simplify",
        json={"input": "x", "format": "text", "derivation_id": d, "parent_id": ids["mid"]},
    )
    assert again.status_code == 422


def test_deleting_a_root_step_removes_everything_below_it(authed: TestClient) -> None:
    d = new_derivation(authed)
    ids = chain(authed, d)
    authed.delete(f"/v1/history/{ids['root']}")
    assert live(authed, d) == {ids["lone"]}
    assert authed.delete(f"/v1/history/{ids['root']}").status_code == 404  # a double click


def test_restore_brings_back_exactly_what_the_delete_hid(authed: TestClient) -> None:
    d = new_derivation(authed)
    ids = chain(authed, d)
    authed.delete(f"/v1/history/{ids['leaf']}")  # an earlier, separate deletion
    authed.delete(f"/v1/history/{ids['mid']}")
    r = authed.post(f"/v1/history/{ids['mid']}/restore")
    assert r.status_code == 200 and set(r.json()["ids"]) == {ids["mid"]}
    assert live(authed, d) == {ids["root"], ids["mid"], ids["other"], ids["lone"]}
    assert authed.post(f"/v1/history/{ids['mid']}/restore").status_code == 404  # not deleted now
    assert authed.post(f"/v1/history/{ids['leaf']}/restore").status_code == 200


def _age(app, model, row_id: str, minutes: int) -> None:  # type: ignore[no-untyped-def]
    import uuid
    from datetime import timedelta

    from minerva_api.db import utcnow

    with app.state.session_factory() as db:
        row = db.get(model, uuid.UUID(row_id))
        row.deleted_at = utcnow() - timedelta(minutes=minutes)
        db.commit()


def test_restore_is_refused_after_the_window_and_the_purge_makes_it_final(app) -> None:  # type: ignore[no-untyped-def]
    from minerva_api.db import Calculation
    from minerva_api.softdelete import purge_expired

    with TestClient(app) as c:
        register(c, "a@example.com")
        d = new_derivation(c)
        ids = chain(c, d)
        c.delete(f"/v1/history/{ids['mid']}")
        _age(app, Calculation, ids["mid"], 11)
        _age(app, Calculation, ids["leaf"], 11)
        assert c.post(f"/v1/history/{ids['mid']}/restore").status_code == 410
        with app.state.session_factory() as db:
            assert purge_expired(db) == 2
            assert db.get(Calculation, __import__("uuid").UUID(ids["leaf"])) is None
        assert c.post(f"/v1/history/{ids['mid']}/restore").status_code == 404


def test_the_purge_does_not_wait_for_the_owner_to_come_back(app) -> None:  # type: ignore[no-untyped-def]
    from minerva_api.db import Derivation
    from minerva_api.softdelete import purge_expired

    with TestClient(app) as c:
        register(c, "gone@example.com")
        d = new_derivation(c)
        step(c, "simplify", input="x+x", format="text", derivation_id=d)
        c.delete(f"/v1/derivations/{d}")
        c.post("/v1/auth/logout")  # the owner is gone; nobody asks for anything of theirs again
        _age(app, Derivation, d, 30)
        with app.state.session_factory() as db:
            assert purge_expired(db) >= 1
            assert db.get(Derivation, __import__("uuid").UUID(d)) is None


def test_derivation_delete_and_restore(authed: TestClient) -> None:
    d = new_derivation(authed)
    ids = chain(authed, d)
    authed.delete(f"/v1/history/{ids['leaf']}")  # deleted earlier: must stay deleted
    assert authed.delete(f"/v1/derivations/{d}").status_code == 204
    assert authed.get(f"/v1/derivations/{d}").status_code == 404
    assert authed.get("/v1/derivations").json() == []
    assert authed.post(f"/v1/history/{ids['mid']}/restore").status_code == 409
    assert authed.post(f"/v1/derivations/{d}/restore").status_code == 200
    assert live(authed, d) == {ids["root"], ids["mid"], ids["other"], ids["lone"]}
    assert authed.post(f"/v1/derivations/{d}/restore").status_code == 404


def test_deleting_the_last_step_hides_the_derivation(authed: TestClient) -> None:
    d = new_derivation(authed)
    a = step(authed, "simplify", input="x+x", format="text", derivation_id=d)
    authed.delete(f"/v1/history/{a['calculation_id']}")
    assert authed.get("/v1/derivations").json() == []


def test_nobody_else_can_delete_or_restore(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as a, TestClient(app) as b:
        register(a, "a@example.com")
        register(b, "b@example.com")
        d = new_derivation(a)
        ids = chain(a, d)
        a.delete(f"/v1/history/{ids['mid']}")
        assert b.post(f"/v1/history/{ids['mid']}/restore").status_code == 404
        assert b.delete(f"/v1/derivations/{d}").status_code == 404
        a.delete(f"/v1/derivations/{d}")
        assert b.post(f"/v1/derivations/{d}/restore").status_code == 404


def test_a_deleted_derivation_cannot_take_new_steps(authed: TestClient) -> None:
    d = new_derivation(authed)
    step(authed, "simplify", input="x+x", format="text", derivation_id=d)
    authed.delete(f"/v1/derivations/{d}")
    r = authed.post(
        "/v1/compute/simplify", json={"input": "x", "format": "text", "derivation_id": d}
    )
    assert r.status_code == 404


def test_clear_history_removes_derivations(authed: TestClient) -> None:
    d = new_derivation(authed)
    step(authed, "simplify", input="x+x", format="text", derivation_id=d)
    assert authed.delete("/v1/history").status_code == 204
    assert authed.get("/v1/derivations").json() == []


def test_apply_identity_reports_the_points_it_excludes(authed: TestClient) -> None:
    r = authed.post(
        "/v1/compute/apply-identity", json={"input": r"\tan(2x)", "rule_id": "trig.double_tan"}
    ).json()
    assert r["status"] == "ok"
    assert r["restrictions_complete"] is True
    assert any(r"\neq" in item for item in r["restrictions_latex"])


def test_simplify_result_can_be_continued_from_its_text(authed: TestClient) -> None:
    d = new_derivation(authed)
    s1 = step(authed, "simplify", input=r"\frac{x^2-1}{x-1}", derivation_id=d)
    assert s1["status"] == "ok" and s1["result_text"]
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
