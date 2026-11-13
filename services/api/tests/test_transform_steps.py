"""Controlled steps over HTTP: substitute, rewrite one part, isolate; saved, continued, private."""

from fastapi.testclient import TestClient
from minerva_engine import parse_text

from .conftest import PASSWORD, build_app, register

NC = {"A": ["noncommutative"], "B": ["noncommutative"], "D": ["noncommutative"]}


def new_derivation(c: TestClient) -> str:
    r = c.post("/v1/derivations")
    assert r.status_code == 201
    return str(r.json()["id"])


def post(c: TestClient, path: str, expect: int = 200, **body: object) -> dict:  # type: ignore[type-arg]
    r = c.post(f"/v1/compute/{path}", json=body)
    assert r.status_code == expect, r.text
    return r.json()  # type: ignore[no-any-return]


def test_requires_authentication(client: TestClient) -> None:
    for path in ("substitute", "rewrite", "isolate", "subterms"):
        assert client.post(f"/v1/compute/{path}", json={"input": "x"}).status_code == 401


# --- each endpoint ---------------------------------------------------------------------------------------
def test_substitute_records_the_premise(authed: TestClient) -> None:
    body = post(authed, "substitute", input="x^2 + y^2", relations=["y^2 = 1 - x^2"])
    assert body["status"] == "ok" and body["result_text"] == "1"
    assert body["evidence"] == "conditional"
    assert body["given"][0]["text"] == "y**2 = 1 - x**2"
    assert body["steps"][0]["display_latex"] and body["calculation_id"]
    assert body["diagnostics"]["verification_method"] == "congruence"


def test_substitute_refuses_an_order_dependent_pair_until_the_order_is_given(
    authed: TestClient,
) -> None:
    refused = post(authed, "substitute", input="x + y", relations=["x = y", "y = 1"])
    assert refused["status"] == "unsupported" and refused["reason"] == "order_dependent"
    assert refused["result_text"] is None
    given = post(authed, "substitute", input="x + y", relations=["x = y", "y = 1"], order="given")
    assert given["result_text"] == "2"


def test_rewrite_one_part_and_the_list_of_parts(authed: TestClient) -> None:
    listed = post(authed, "subterms", input=r"\sin(2x) + \sin(2y)", find=r"\sin(2y)")
    assert [s["latex"] for s in listed["subterms"]] == [r"\sin{\left(2 y \right)}"]
    assert "calculation_id" in listed and listed["calculation_id"] is None, (
        "a lookup is never saved"
    )
    pick = listed["subterms"][0]
    body = post(
        authed,
        "rewrite",
        input=r"\sin(2x) + \sin(2y)",
        action="rule",
        rule_id="trig.double_sin",
        selection={"path": pick["path"], "args": pick["args"], "expected_latex": pick["latex"]},
    )
    assert body["status"] == "ok"
    assert parse_text(body["result_text"]) == parse_text("2*sin(y)*cos(y) + sin(2*x)")
    assert body["entry"]["id"] == "trig.double_sin"
    assert body["steps"][0]["matched_latex"] == pick["latex"]


def test_a_stale_selection_is_refused_over_http_too(authed: TestClient) -> None:
    body = post(
        authed,
        "rewrite",
        input=r"\sin(2x) + z",
        action="expand",
        selection={"path": [0], "args": None, "expected_latex": r"\sin{\left(2 y \right)}"},
    )
    assert body["status"] == "parse_error" and body["reason"] == "stale_selection"


def test_an_identity_with_unproven_conditions_is_reported_not_applied(authed: TestClient) -> None:
    body = post(
        authed,
        "rewrite",
        input=r"\log(x y) + z",
        action="rule",
        rule_id="log.product",
        part=r"\log(x y)",
    )
    assert body["status"] == "conditions_not_satisfied" and body["result_text"] is None
    assert body["unmet_conditions"]


def test_collect(authed: TestClient) -> None:
    body = post(authed, "rewrite", input="a x^2 + b x + c x^2 + d x", action="collect", by=["x"])
    assert body["status"] == "ok"
    assert parse_text(body["result_text"]) == parse_text("x**2*(a + c) + x*(b + d)")


def test_isolate_lists_every_solution_and_condition(authed: TestClient) -> None:
    body = post(authed, "isolate", input=r"s = u t + \frac{1}{2} a t^2", variable="t")
    assert body["status"] == "ok" and len(body["solutions"]) == 2
    assert r"a \neq 0" in body["conditions_latex"] and body["exceptional"] == ["a = 0"]
    assert body["complete"] is True
    assert all(s["equation_text"].startswith("t = ") for s in body["solutions"])


def test_isolate_unsupported_and_empty(authed: TestClient) -> None:
    assert post(authed, "isolate", input=r"\sin x = a", variable="x")["status"] == "unsupported"
    none = post(authed, "isolate", input=r"\sqrt{x} = -2", variable="x")
    assert none["status"] == "ok" and none["solutions"] == []


def test_noncommuting_symbols_where_supported_and_refused_elsewhere(authed: TestClient) -> None:
    ok = post(authed, "substitute", input="A^2 B", relations=["A B = D"], assumptions=NC)
    assert ok["status"] == "ok" and ok["result_text"] == "A*D"
    refused = post(authed, "simplify", input="A B - B A", assumptions=NC)
    assert refused["status"] == "unsupported"
    assert (
        post(authed, "isolate", input="A x = B", variable="x", assumptions=NC)["status"]
        == "unsupported"
    )


# --- validation -------------------------------------------------------------------------------------------
def test_malformed_requests_are_rejected(authed: TestClient) -> None:
    post(authed, "substitute", 422, input="x", relations=[])
    post(authed, "substitute", 422, input="x", relations=["x = 1"] * 5)
    post(authed, "substitute", 422, input="x", relations=["x = 1"], order="whatever")
    post(authed, "rewrite", 422, input="x", action="rule")  # no rule_id
    post(authed, "rewrite", 422, input="x", action="expand", rule_id="trig.double_sin")
    post(authed, "rewrite", 422, input="x", action="collect")  # nothing to collect by
    post(
        authed,
        "rewrite",
        422,
        input="x + y",
        action="expand",
        part="x",
        selection={"path": [], "expected_latex": "x"},
    )
    post(
        authed,
        "rewrite",
        422,
        input="x",
        action="expand",
        selection={"path": [-1], "expected_latex": "x"},
    )
    post(authed, "rewrite", 422, input="x", action="nonsense")
    post(authed, "isolate", 422, input="x = 1", variable="")


def test_an_unknown_or_rule_less_identity_is_not_found(authed: TestClient) -> None:
    post(authed, "rewrite", 404, input="x", action="rule", rule_id="no.such.rule")
    post(authed, "subterms", 404, input="x", rule_id="no.such.rule")


def test_the_total_size_of_the_request_is_bounded(authed: TestClient) -> None:
    big = "+".join(["x"] * 400)
    post(authed, "substitute", 422, input=big, relations=["x = " + "y+" * 190 + "y"] * 4)


def test_an_exploding_expansion_is_a_resource_limit_status() -> None:
    with TestClient(build_app(compute_timeout_seconds=2.0)) as c:
        register(c)
        body = post(
            c,
            "rewrite",
            input="(x+1)**9000*(y+1)**9000",
            format="text",
            action="expand",
            part="(x+1)**9000",
        )
        assert body["status"] == "resource_limit" and body["result_text"] is None


# --- derivations: chains, continuation, persistence ---------------------------------------------------------
def test_a_derivation_mixes_old_and_new_steps_and_continues_from_any_of_them(
    authed: TestClient,
) -> None:
    d = new_derivation(authed)
    s1 = post(
        authed, "substitute", input="x^2 + y^2 + z", relations=["y^2 = 1 - x^2"], derivation_id=d
    )
    assert s1["result_text"] == "z + 1"
    s2 = post(
        authed,
        "simplify",
        input=s1["result_text"],
        format="text",
        derivation_id=d,
        parent_id=s1["calculation_id"],
    )
    s3 = post(
        authed,
        "rewrite",
        input="(z + 1)**2",
        format="text",
        action="expand",
        derivation_id=d,
        parent_id=s1["calculation_id"],  # a second branch from step 1
    )
    got = authed.get(f"/v1/derivations/{d}").json()
    assert [s["operation"] for s in got["steps"]] == ["substitute", "simplify", "rewrite"]
    assert got["steps"][2]["parent_id"] == s1["calculation_id"] == got["steps"][1]["parent_id"]
    assert s2["calculation_id"] != s3["calculation_id"]


def test_what_was_asked_is_saved_with_the_result_so_a_reload_shows_it(authed: TestClient) -> None:
    d = new_derivation(authed)
    post(
        authed,
        "substitute",
        input="x + y",
        relations=["x = y", "y = 1"],
        order="given",
        derivation_id=d,
    )
    post(authed, "isolate", input="a x + b = c", variable="x", derivation_id=d)
    post(authed, "rewrite", input="x + 1", action="expand", part="x + 1", derivation_id=d)
    steps = authed.get(f"/v1/derivations/{d}").json()["steps"]
    assert steps[0]["result"]["request"] == {"relations": ["x = y", "y = 1"], "order": "given"}
    assert steps[1]["result"]["request"] == {"variable": "x"}
    assert steps[1]["result"]["solutions"][0]["equation_text"] == "x = (-b + c)/a"
    assert steps[2]["result"]["request"]["part"] == "x + 1"
    assert steps[0]["status"] == "ok"
    # Each step's saved evidence survives the round trip.
    assert steps[0]["result"]["given"][0]["text"] == "x = y"
    assert steps[1]["result"]["complete"] is True


def test_a_solution_can_be_continued_with_its_value(authed: TestClient) -> None:
    d = new_derivation(authed)
    iso = post(authed, "isolate", input="a x + b = c", variable="x", derivation_id=d)
    value = iso["solutions"][0]["text"]
    nxt = post(
        authed,
        "substitute",
        input=value,
        format="text",
        relations=["c = 2*b"],
        derivation_id=d,
        parent_id=iso["calculation_id"],
    )
    assert nxt["status"] == "ok" and nxt["result_text"] == "b/a"


def test_nothing_is_saved_when_not_asked_and_other_users_cannot_join(client: TestClient) -> None:
    register(client)
    d = new_derivation(client)
    post(client, "substitute", input="x", relations=["x = 1"], save=False)
    assert client.get("/v1/history").json() == []
    other = TestClient(client.app)
    register(other, "grace@example.com")
    post(other, "isolate", 404, input="x = 1", variable="x", derivation_id=d)


# --- privacy and debug --------------------------------------------------------------------------------------
def test_logs_carry_counters_and_codes_never_expressions(capsys) -> None:  # type: ignore[no-untyped-def]
    marker = "zzMARKERzz"
    with TestClient(build_app()) as c:
        register(c)
        post(c, "substitute", input=f"{marker} + y", relations=[f"y = {marker}2"], format="text")
        post(
            c,
            "rewrite",
            input=f"{marker}*(x+1)**2",
            format="text",
            action="expand",
            part="(x+1)**2",
        )
        post(c, "rewrite", input=f"{marker} x + {marker} y", action="collect", by=[marker])
        post(c, "isolate", input=f"{marker} x = 3", variable="x")
        post(c, "isolate", input=f"{marker} = 3", variable=marker)
        post(c, "subterms", input=f"{marker} + 1", find=marker)
    out = capsys.readouterr().out
    assert marker not in out and PASSWORD not in out and "ada@example.com" not in out
    for field in ("verify_ms", "verification_method", "verification_outcome", "candidates"):
        assert field in out, field


def test_rejected_candidates_appear_only_with_the_debug_setting_and_are_never_saved() -> None:
    marker = "zzMARKERzz"
    for debug in (False, True):
        with TestClient(build_app(debug_compute=debug)) as c:
            register(c)
            d = new_derivation(c)
            body = post(
                c,
                "substitute",
                input=f"{marker} + y",
                relations=["y = 1", f"{marker} = y"],
                format="text",
                derivation_id=d,
            )
            assert (
                body["status"] == "unsupported"
            )  # order dependent: the alternatives are the detail
            detail = body["diagnostics"]["detail"]
            assert bool(detail) is debug
            assert not c.get("/v1/history").json()[0]["result"]["diagnostics"]["detail"]
