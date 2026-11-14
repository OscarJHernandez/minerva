"""Controlled identity steps: the matcher proposes, the evaluation of the whole expressions decides.

The identities are written here as `VectorRule`s (the engine never reads the handbook); the handbook's
own entries are tested in packages/knowledge."""

import pytest
from minerva_engine.models import Limits, Status
from minerva_engine.vector import FieldDecl, Premise, VectorInput, VectorRule, vector_calculus

L = Limits(timeout_seconds=60)


def rule(id: str, lhs: str, rhs: str, variables: dict[str, str], requires: tuple[str, ...] = ()):
    return VectorRule(id=id, lhs=lhs, rhs=rhs, variables=variables, requires=list(requires))  # type: ignore[arg-type]


CURL_CURL = rule(
    "vector.curl_curl", "curl(curl(A))", "grad(div(A)) - lap(A)", {"A": "vector"}, ("c2",)
)
CURL_GRAD = rule("vector.curl_grad", "curl(grad(f))", "0", {"f": "scalar"}, ("c2",))
TIME_CURL = rule("vector.time_curl", "dt(curl(A))", "curl(dt(A))", {"A": "vector"}, ("c2",))
DIV_CROSS = rule(
    "vector.div_cross",
    "div(cross(A, B))",
    "dot(B, curl(A)) - dot(A, curl(B))",
    {"A": "vector", "B": "vector"},
)
CURL_FA = rule(
    "vector.curl_scalar_vector",
    "curl(f*A)",
    "cross(grad(f), A) + f*curl(A)",
    {"f": "scalar", "A": "vector"},
)
RULES = [CURL_CURL, CURL_GRAD, TIME_CURL, DIV_CROSS, CURL_FA]


def step(expression: str, action: str, *, fields=(), rules=RULES, system="cartesian", **kw):
    return vector_calculus(
        VectorInput(
            expression=expression,
            format="text",
            coordinates=system,
            fields=list(fields),
            rules=list(rules),
            action=action,  # type: ignore[arg-type]
            **kw,
        ),
        L,
    )


def vec(name: str, smoothness: str | None = None, time: bool = False) -> FieldDecl:
    depends = ["x", "y", "z", "t"] if time else None
    return FieldDecl(name=name, kind="vector", smoothness=smoothness, depends_on=depends)  # type: ignore[arg-type]


def sca(name: str, smoothness: str | None = None) -> FieldDecl:
    return FieldDecl(name=name, kind="scalar", smoothness=smoothness)  # type: ignore[arg-type]


# -- listing -------------------------------------------------------------------------------------
def test_listing_shows_matches_and_applies_nothing() -> None:
    result = step("curl(curl(A))", "identities", fields=[vec("A")])
    assert result.status is Status.OK and result.steps == [] and result.result_latex is None
    (match,) = result.applicable
    assert match.rule_id == "vector.curl_curl" and match.path == [] and match.needs == ["c2"]
    assert match.blocked  # A was not declared c2
    assert "none was applied or verified" in result.message


def test_listing_marks_declared_fields_as_applicable() -> None:
    result = step("curl(curl(A))", "identities", fields=[vec("A", "c2")])
    assert [a.blocked for a in result.applicable] == [False]


def test_listing_with_no_match_says_so() -> None:
    result = step("dot(A, B)", "identities", fields=[vec("A"), vec("B")])
    assert result.status is Status.UNCHANGED and result.applicable == []
    assert result.diagnostics.reason == "no_match"


# -- applying an identity ----------------------------------------------------------------------
def test_curl_curl_rewritten_in_terms_of_divergence_and_laplacian() -> None:
    result = step(
        "curl(curl(A))", "apply_identity", fields=[vec("A", "c2")], rule_id="vector.curl_curl"
    )
    assert result.status is Status.OK, result.message
    (s,) = result.steps
    assert s.evidence == "identity" and s.rule_id == "vector.curl_curl" and s.symbolic_zero is True
    assert s.needs == ["c2"] and s.verification and s.verification.numeric_passed
    assert result.result_text is not None and "Lap(A)" in result.result_text
    assert result.evidence == "identity" and result.given == []


def test_an_identity_that_needs_c2_is_blocked_for_undeclared_fields() -> None:
    result = step("curl(curl(A))", "apply_identity", fields=[vec("A")], rule_id="vector.curl_curl")
    assert result.status is Status.CONDITIONS_NOT_SATISFIED
    assert result.blocked and result.blocked[0].needs == "c2" and result.blocked[0].fields == ["A"]
    assert result.result_latex is None and result.steps == []


def test_the_part_is_chosen_by_path_and_nothing_is_picked_for_the_user() -> None:
    expression = "curl(curl(A)) + curl(curl(B))"
    fields = [vec("A", "c2"), vec("B", "c2")]
    ambiguous = step(expression, "apply_identity", fields=fields, rule_id="vector.curl_curl")
    assert ambiguous.status is Status.UNCHANGED
    assert len(ambiguous.applicable) == 2 and ambiguous.result_latex is None
    chosen = step(
        expression,
        "apply_identity",
        fields=fields,
        rule_id="vector.curl_curl",
        path=ambiguous.applicable[0].path,
    )
    assert chosen.status is Status.OK and len(chosen.steps) == 1


def test_a_selected_part_inside_a_larger_expression() -> None:
    result = step(
        "2*curl(curl(A)) + B",
        "apply_identity",
        fields=[vec("A", "c2"), vec("B")],
        rule_id="vector.curl_curl",
    )
    assert result.status is Status.OK
    assert result.steps[0].path != []  # the part is below the sum


def test_identity_applied_right_to_left() -> None:
    result = step(
        "grad(div(A)) - lap(A)",
        "apply_identity",
        fields=[vec("A", "c2")],
        rule_id="vector.curl_curl",
        reverse=True,
    )
    assert result.status is Status.OK, result.message
    assert result.steps[0].reverse and result.result_text == "Curl(Curl(A))"


def test_a_rule_that_cannot_be_read_backwards_is_not_matched_backwards() -> None:
    result = step("0", "apply_identity", fields=[], rule_id="vector.curl_grad", reverse=True)
    assert result.status is Status.UNCHANGED


def test_unknown_rule_and_non_matching_rule() -> None:
    assert step("grad(f)", "apply_identity", fields=[sca("f")], rule_id="nope").status is (
        Status.UNSUPPORTED
    )
    miss = step("grad(f)", "apply_identity", fields=[sca("f")], rule_id="vector.curl_curl")
    assert miss.status is Status.UNCHANGED and miss.diagnostics.reason == "no_match"


def test_product_rule_needs_no_smoothness_and_holds_in_curvilinear_coordinates() -> None:
    result = step(
        "curl(f*A)",
        "apply_identity",
        fields=[sca("f"), vec("A")],
        rule_id="vector.curl_scalar_vector",
        system="spherical",
    )
    assert result.status is Status.OK and result.steps[0].needs == []


# -- wrong identities are refused ------------------------------------------------------------------
@pytest.mark.parametrize(
    ("lhs", "rhs", "variables"),
    [
        ("curl(curl(A))", "grad(div(A)) + lap(A)", {"A": "vector"}),  # sign error
        ("curl(f*A)", "f*curl(A)", {"f": "scalar", "A": "vector"}),  # missing term
        ("div(cross(A, B))", "dot(A, curl(B)) - dot(B, curl(A))", {"A": "vector", "B": "vector"}),
        (
            "curl(cross(A, B))",
            "A*div(B) - B*div(A)",
            {"A": "vector", "B": "vector"},
        ),  # no advection
        ("lap(f*g)", "f*lap(g) + g*lap(f)", {"f": "scalar", "g": "scalar"}),  # no cross term
    ],
)
def test_a_wrong_identity_is_not_verified_and_never_shown(lhs, rhs, variables) -> None:
    wrong = rule("wrong.rule", lhs, rhs, variables, ("c2",))
    fields = [
        FieldDecl(name=n, kind=k, smoothness="c2")  # type: ignore[arg-type]
        for n, k in variables.items()
    ]
    result = step(lhs, "apply_identity", fields=fields, rules=[wrong], rule_id="wrong.rule")
    assert result.status is Status.NOT_VERIFIED, (lhs, result.message)
    assert result.result_latex is None and result.steps == []


def test_an_identity_that_secretly_needs_c2_is_caught_by_the_comparison() -> None:
    # curl grad f = 0 declared as needing nothing: the evaluation of the difference needs c2 itself
    lying = rule("lying.curl_grad", "curl(grad(f))", "0", {"f": "scalar"}, ())
    result = step(
        "curl(grad(f))",
        "apply_identity",
        fields=[sca("f")],
        rules=[lying],
        rule_id="lying.curl_grad",
    )
    assert result.status is Status.CONDITIONS_NOT_SATISFIED
    assert result.blocked and result.result_latex is None


# -- premises ----------------------------------------------------------------------------------------
FARADAY = Premise(equation="curl(E) = -dt(B)", label="Faraday")
AMPERE = Premise(equation="curl(B) = mu0*eps0*dt(E)", label="Ampere-Maxwell")
GAUSS = Premise(equation="div(E) = 0", label="Gauss")
EM_FIELDS = [vec("E", "smooth", True), vec("B", "smooth", True)]


def test_substituting_a_premise_is_conditional_and_names_it() -> None:
    result = step(
        "curl(curl(E))",
        "substitute",
        fields=EM_FIELDS,
        premises=[FARADAY],
        constants=["mu0", "eps0"],
    )
    assert result.status is Status.OK, result.message
    assert result.evidence == "conditional" and result.given == ["curl(E) = -dt(B)"]
    (s,) = result.steps
    assert s.evidence == "conditional" and s.given == ["curl(E) = -dt(B)"]
    assert result.result_text == "-Curl(Pd(B, t))"  # the constant -1 left the curl
    assert "not an identity" in result.message


def test_a_premise_that_does_not_occur_changes_nothing() -> None:
    result = step("curl(B)", "substitute", fields=EM_FIELDS, premises=[FARADAY])
    assert result.status is Status.UNCHANGED and result.diagnostics.reason == "no_match"


def test_several_premises_are_never_applied_in_an_order_minerva_picked() -> None:
    result = step("curl(E) + curl(B)", "substitute", fields=EM_FIELDS, premises=[FARADAY, AMPERE])
    assert result.status is Status.PARSE_ERROR
    ok = step(
        "curl(E) + curl(B)",
        "substitute",
        fields=EM_FIELDS,
        premises=[FARADAY, AMPERE],
        premise=1,
        constants=["mu0", "eps0"],
    )
    assert ok.status is Status.OK and ok.given == ["curl(B) = mu0*eps0*dt(E)"]


def test_a_premise_is_never_checked_and_a_false_one_is_used_as_given() -> None:
    false = Premise(equation="curl(E) = 3*curl(B)", label="not true in general")
    result = step("curl(E)", "substitute", fields=EM_FIELDS, premises=[false])
    assert result.status is Status.OK and result.evidence == "conditional"


def test_an_equation_is_refused_where_an_expression_is_expected() -> None:
    result = step("curl(E) = 0", "identities", fields=EM_FIELDS)
    assert result.status is Status.PARSE_ERROR


# -- an operator on both sides ------------------------------------------------------------------------
def test_apply_curl_to_both_sides_of_faraday() -> None:
    result = step("curl(E) = -dt(B)", "apply", fields=EM_FIELDS, operator="curl", premises=[])
    assert result.status is Status.OK, result.message
    assert result.evidence == "conditional" and result.given == ["curl(E) = -dt(B)"]
    assert result.steps[0].symbolic_zero is True
    assert result.result_text == "Curl(Curl(E)) = -Curl(Pd(B, t))"


def test_apply_needs_the_right_kind_and_an_operator() -> None:
    wrong_kind = step("div(E) = 0", "apply", fields=EM_FIELDS, operator="curl")
    assert wrong_kind.status is Status.PARSE_ERROR
    assert step("curl(E) = 0", "apply", fields=EM_FIELDS).status is Status.PARSE_ERROR


# -- derive -----------------------------------------------------------------------------------------------
def test_derive_one_identity_step_to_a_target() -> None:
    result = step(
        "curl(curl(A))",
        "derive",
        fields=[vec("A", "c2")],
        target="grad(div(A)) - lap(A)",
    )
    assert result.status is Status.OK, result.message
    assert len(result.steps) == 1 and result.evidence == "identity" and result.given == []


def test_derive_the_wave_equation_side_from_maxwell_premises() -> None:
    """curl curl E = -mu0 eps0 d^2E/dt^2 from Faraday and Ampere-Maxwell (the premises, given)."""
    result = step(
        "curl(curl(E))",
        "derive",
        fields=EM_FIELDS,
        constants=["mu0", "eps0"],
        premises=[FARADAY, AMPERE],
        target="-mu0*eps0*dt(dt(E))",
    )
    assert result.status is Status.OK, result.message
    assert result.evidence == "conditional"
    assert set(result.given) == {"curl(E) = -dt(B)", "curl(B) = mu0*eps0*dt(E)"}
    kinds = [s.evidence for s in result.steps]
    assert kinds.count("conditional") == 2 and kinds.count("identity") >= 1
    assert any(s.rule_id == "vector.time_curl" and s.reverse for s in result.steps)


def test_derive_the_other_side_from_the_gauss_premise() -> None:
    result = step(
        "curl(curl(E))",
        "derive",
        fields=EM_FIELDS,
        premises=[GAUSS],
        target="-lap(E)",
    )
    assert result.status is Status.OK, result.message
    assert result.evidence == "conditional" and result.given == ["div(E) = 0"]


def test_derive_without_the_premise_does_not_reach_the_target() -> None:
    result = step("curl(curl(E))", "derive", fields=EM_FIELDS, target="-lap(E)")
    assert result.status is Status.UNCHANGED
    assert result.diagnostics.reason == "not_found" and result.search_complete is not None
    assert "does not say that none exists" in result.message


def test_derive_respects_the_declared_smoothness() -> None:
    result = step("curl(curl(A))", "derive", fields=[vec("A")], target="grad(div(A)) - lap(A)")
    assert result.status is Status.UNCHANGED  # the identity needs c2, which was not declared


def test_derive_a_smaller_form_without_a_target() -> None:
    result = step("curl(grad(f)) + grad(g)", "derive", fields=[sca("f", "c2"), sca("g")])
    assert result.status is Status.UNCHANGED or result.steps  # a smaller form, or none found
    shown = step("curl(grad(f))", "derive", fields=[sca("f", "c2")])
    assert shown.status is Status.OK and shown.result_text == "0"


def test_derive_target_of_another_kind_is_refused() -> None:
    result = step("curl(A)", "derive", fields=[vec("A")], target="div(A)")
    assert result.status is Status.PARSE_ERROR
