"""POST /v1/compute/mvn: a result, a refusal, authentication, the input limit and what is logged."""

from fastapi.testclient import TestClient

MVN = "/v1/compute/mvn"
BASE = {"mean": ["1", "-2"], "covariance": [["4", "1"], ["1", "2"]], "variables": ["x", "y"]}


def test_it_requires_authentication(client: TestClient) -> None:
    assert client.post(MVN, json={**BASE, "action": "density"}).status_code == 401


def test_a_conditional_is_the_hand_worked_one(authed: TestClient) -> None:
    r = authed.post(MVN, json={**BASE, "action": "conditional", "given": {"x": "3"}})
    assert r.status_code == 200, r.text
    body = r.json()
    # mean -2 + (1/4)(3 - 1) = -3/2, variance 2 - 1/4 = 7/4
    assert body["status"] == "ok" and body["evidence"] == "identity"
    assert body["mean_text"] == ["-3/2"] and body["covariance_text"] == [["7/4"]]
    assert body["variables"] == ["y"] and all(c["passed"] for c in body["checks"])
    assert "not a proof" in body["message"]


def test_a_singular_covariance_is_refused_with_its_reason(authed: TestClient) -> None:
    body = authed.post(
        MVN,
        json={**BASE, "covariance": [["1", "1"], ["1", "1"]], "action": "density"},
    ).json()
    assert body["status"] == "unsupported" and body["reason"] == "singular_covariance"
    assert not body["density_text"] and not body["mean_text"]


def test_an_action_is_required_and_has_no_default(authed: TestClient) -> None:
    assert authed.post(MVN, json=BASE).status_code == 422


def test_an_enormous_entry_is_refused(authed: TestClient) -> None:
    huge = "1+" * 20000 + "1"
    r = authed.post(MVN, json={**BASE, "mean": [huge, "0"], "action": "moments"})
    assert r.status_code == 422


def test_the_log_holds_no_entry(authed: TestClient, capsys: object) -> None:
    secret = "zq_secret_entry"
    authed.post(
        MVN,
        json={
            "mean": [secret, "0"],
            "covariance": [["s", "0"], ["0", "s"]],
            "variables": ["x", "y"],
            "action": "moments",
            "assumptions": {"s": ["positive"], secret: ["real"]},
        },
    )
    out = capsys.readouterr()  # type: ignore[attr-defined]
    assert secret not in out.out + out.err
