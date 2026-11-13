import pytest
from minerva_engine import Engine, Limits, check_claim, check_claims, split_equation


@pytest.mark.parametrize(
    ("latex", "sides"),
    [
        (r"a = b", ["a", "b"]),
        (r"\frac{x^2-1}{x-1} = x+1 \quad \text{for } x \neq 1", [r"\frac{x^2-1}{x-1}", "x+1"]),
        (r"a = b = c", ["a", "b", "c"]),
        (r"f_{x=1}(y) = 2", ["f_{x=1}(y)", "2"]),
        (r"x = 2, \text{ if } y > 0", ["x", "2"]),
    ],
)
def test_split_equation(latex: str, sides: list[str]) -> None:
    assert split_equation(latex) == sides


@pytest.mark.parametrize("latex", ["x+1", "=", "x =", "= y"])
def test_no_equation(latex: str) -> None:
    assert split_equation(latex) is None


L = Limits()


def test_true_equation_holds_and_false_one_fails() -> None:
    assert check_claim(r"\sin^2 x + \cos^2 x = 1", {}, L).status == "holds"
    assert check_claim(r"\sin^2 x + \cos^2 x = 2", {}, L).status == "fails"  # false for every x
    assert check_claim(r"(x+1)^2 = x^2 + 1", {}, L).status == "not_identity"


def test_claims_depend_on_assumptions() -> None:
    claim = r"\sqrt{x^2} = x"
    assert (
        check_claim(claim, {}, L).status == "conditional"
    )  # false for complex x, true for positive x
    assert check_claim(claim, {"x": ["positive"]}, L).status == "holds"
    assert check_claim(claim, {"x": ["negative"]}, L).status == "not_identity"


def test_chains_and_trailing_conditions() -> None:
    assert check_claim(r"2x = x + x = 2x \quad (x \ne 0)", {}, L).status in ("holds", "unparsed")
    assert check_claim(r"x + x = 2x \text{ for all } x", {}, L).status == "holds"


def test_undecidable_inputs_are_reported_not_guessed() -> None:
    assert check_claim("x + 1", {}, L).status == "not_an_equation"
    assert check_claim(r"\foo = 1", {}, L).status == "unparsed"
    assert check_claim(r"\int x\,dx = \frac{x^2}{2}", {}, L).status == "unparsed"


def test_engine_batch_runs_in_the_sandbox() -> None:
    engine = Engine(Limits(timeout_seconds=30))
    results = engine.check_claims([r"x+x = 2x", r"x+x = 3x", "y"], {})
    assert [r.status for r in results] == ["holds", "not_identity", "not_an_equation"]
    assert engine.check_claims([]) == []


def test_claims_true_only_for_real_or_positive_symbols_are_conditional_not_false() -> None:
    real = check_claim(r"\sqrt{x^2} = |x|", {}, L)
    assert real.status == "conditional" and "real" in real.detail
    positive = check_claim(r"\sqrt{x^2} = x", {}, L)
    assert positive.status == "conditional" and "positive" in positive.detail
    assert check_claim(r"(x+1)^2 = x^2 + 1", {}, L).status == "not_identity"
    assert check_claim(r"\sqrt{x^2} = x", {"x": ["positive"]}, L).status == "holds"
    assert check_claim(r"\sqrt{x^2} = -x", {"x": ["positive"]}, L).status == "not_identity"


def test_claims_reading_e_i_as_constants_and_equations_to_solve() -> None:
    assert check_claim(r"e^{i a} = \cos a + i \sin a", {}, L).status == "holds"
    assert check_claim(r"x = -1", {}, L).status == "unverified"
    assert check_claim(r"x^3 + x^2 + x + 1 = 0", {}, L).status == "unverified"
    assert check_claim(r"\sin^2 x + \cos^2 x = 1", {}, L).status == "holds"
    assert check_claim(r"2 + 2 = 5", {}, L).status == "fails"
    assert check_claim(r"\sin x = \cos x", {}, L).status == "not_identity"


def test_removable_singularity_is_conditional_not_holds() -> None:
    out = check_claim(r"\frac{x^2-1}{x-1} = x+1", {}, L)
    assert out.status == "conditional" and r"x \neq 1" in out.detail
    cancelled = check_claim(r"\frac{x-1}{x-1} = 1", {}, L)  # the parser cancels it while reading
    assert cancelled.status == "conditional" and r"x \neq 1" in cancelled.detail
    assert check_claim(r"\frac{\sin x}{x} = \frac{\sin x}{x}", {}, L).status == "holds"


@pytest.mark.parametrize(
    "claim",
    [
        r"F = m a",  # definition
        r"L = T - V",
        r"a = 2t",
        r"m\ddot{x} = -kx",  # equation of motion
        r"\ddot{x} = -\omega^2 x",
        r"m\ddot r = m r\dot\theta^2 - \frac{k}{r^2}",
        r"x^2 = 3x - 2",  # equation to solve: true for some x
    ],
)
def test_equations_definitions_and_motion_are_never_contradicted(claim: str) -> None:
    assert check_claim(claim, {}, L).status in ("unverified", "not_identity")


@pytest.mark.parametrize(
    ("claim", "assumptions"),
    [
        (r"\ln(\exponentialE^{z}) = z", {}),  # only for |Im z| < pi: not 'holds'
        (r"\sqrt{(x-4)^2} = 4-x", {"x": ["real"]}),
        (r"\arctan(\tan(x/3)) = x/3", {"x": ["real"]}),
        (r"|x-10| = 10-x", {"x": ["real"]}),
    ],
)
def test_branch_and_range_errors_are_not_reported_as_holding(
    claim: str, assumptions: dict[str, list[str]]
) -> None:
    assert check_claim(claim, assumptions, L).status != "holds"


def test_declared_real_is_relaxed_to_positive() -> None:
    out = check_claim(r"\sqrt{x^2} = x", {"x": ["real"]}, L)
    assert out.status == "conditional" and "positive" in out.detail


# --- relations: definitions, coordinate changes, constraints, equations of motion -------------
REPORTED = [
    r"L = T",
    r"\theta = \phi + \omega t",
    r"\dot\theta = \dot\phi + \omega",
    r"L = \frac12 m (\dot r^2 + r^2 \dot\theta^2)",
    r"F = m a",
    r"m \ddot x = -k x",
    r"x^2 + y^2 = r^2",
]


@pytest.mark.parametrize("latex", REPORTED)
def test_relations_are_never_called_false_by_independent_sampling(latex: str) -> None:
    result = check_claim(latex, {}, L)
    assert result.status == "unverified"
    assert result.kind == "relation"
    assert "not tested as an identity" in result.detail


def test_identities_are_still_judged_and_say_so() -> None:
    wrong = check_claim(r"(x+1)^2 = x^2 + 1", {}, L)
    assert (wrong.status, wrong.kind) == ("not_identity", "identity")
    assert check_claim(r"\sin x = \cos x", {}, L).status == "not_identity"
    assert check_claim(r"2 + 2 = 5", {}, L).status == "fails"
    assert check_claim(r"x + x = 2x", {}, L).kind == "identity"


def test_a_definition_from_earlier_is_used_by_substitution_and_named() -> None:
    rot = [r"\theta = \phi + \omega t"]
    good = check_claims(
        [r"\cos\theta = \cos\phi\cos(\omega t) - \sin\phi\sin(\omega t)"], {}, rot, L
    )[0]
    assert (good.status, good.kind) == ("conditional", "relation")
    assert good.given == rot and "holds given" in good.detail
    bad = check_claims([r"\cos\theta = \cos\phi + \cos(\omega t)"], {}, rot, L)[0]
    assert (bad.status, bad.kind) == ("fails", "relation")
    assert bad.given == rot and "does not agree with" in bad.detail


def test_definitions_chain_through_the_claims_of_one_message() -> None:
    results = check_claims(
        [
            r"\dot\theta = \dot\phi + \omega",
            r"L = \frac12 m (\dot r^2 + r^2 \dot\theta^2)",
            r"L = \frac12 m (\dot r^2 + r^2 (\dot\phi + \omega)^2)",
        ],
        {},
        [],
        L,
    )
    assert [r.status for r in results] == ["unverified", "unverified", "conditional"]
    assert results[2].given == [
        r"\dot\theta = \dot\phi + \omega",
        r"L = \frac12 m (\dot r^2 + r^2 \dot\theta^2)",
    ]
    assert not any(r.status == "fails" for r in results)


def test_a_new_definition_is_never_a_contradiction() -> None:
    redefined = check_claims([r"\theta = \phi - \omega t"], {}, [r"\theta = \phi + \omega t"], L)[0]
    assert (redefined.status, redefined.kind) == ("unverified", "relation")
    # ...and it replaces the old one for what follows.
    after = check_claims(
        [
            r"\theta = \phi - \omega t",
            r"\cos\theta = \cos\phi\cos(\omega t) + \sin\phi\sin(\omega t)",
        ],
        {},
        [r"\theta = \phi + \omega t"],
        L,
    )
    assert after[1].status == "conditional"
    assert after[1].given == [r"\theta = \phi - \omega t"]


def test_restating_a_definition_is_consistent_with_it() -> None:
    again = check_claims([r"\phi + \omega t = \theta"], {}, [r"\theta = \phi + \omega t"], L)[0]
    assert again.status == "conditional" and again.given == [r"\theta = \phi + \omega t"]


def test_a_stale_definition_does_not_rescue_a_false_identity() -> None:
    result = check_claims([r"(x+1)^2 = x^2 + 1"], {}, [r"x = 0"], L)[0]
    assert (result.status, result.kind, result.given) == ("not_identity", "identity", [])


def test_cyclic_and_unusable_definitions_are_ignored() -> None:
    results = check_claims(
        [r"a = b + 1", r"b = a", r"a + b = 2 a - 1"], {}, [r"\foo = 1", r"x = x + 1", "y"], L
    )
    # `b = a` would define b through itself once a is substituted: skipped, not an error.
    assert results[0].status == "unverified"
    assert results[1].status == "unverified"
    assert results[2].status in ("conditional", "unverified")


def test_unrelated_definitions_leave_relations_unchecked() -> None:
    result = check_claims([r"F = m a"], {}, [r"\theta = \phi + \omega t"], L)[0]
    assert (result.status, result.given) == ("unverified", [])


def test_engine_batch_passes_relations_through_the_sandbox() -> None:
    engine = Engine(Limits(timeout_seconds=30))
    results = engine.check_claims(
        [
            r"\cos\theta = \cos\phi\cos(\omega t) - \sin\phi\sin(\omega t)",
            r"\theta = \phi + \omega t",
        ],
        {},
        [r"\theta = \phi + \omega t"],
    )
    assert [r.status for r in results] == ["conditional", "conditional"]
    assert results[0].given == [r"\theta = \phi + \omega t"]


# -- formulas about matrices declared in the conversation -----------------------------------------
def test_scalar_sampling_would_wrongly_confirm_matrix_equations() -> None:
    """The reason matrices must be named: without them these are tested as commuting scalars."""
    assert check_claim(r"\exp(A)\exp(B) = \exp(A+B)", {}, L).status == "holds"
    assert check_claim(r"AB = BA", {}, L).status == "holds"


def test_equations_about_declared_matrices_are_never_tested_by_sampling() -> None:
    formulas = [
        r"\exp(A)\exp(B) = \exp(A+B)",
        r"AB = BA",
        r"\frac{d}{dt}\exp(tA) = A\exp(tA)",
        r"\sin^2 x + \cos^2 x = 1",
        r"\exp(s)\exp(u) = \exp(s+u)",
    ]
    results = check_claims(formulas, {}, [], L, matrices=["A", "B"])
    assert [r.status for r in results[:3]] == ["unverified"] * 3
    assert all("matrix" in r.detail for r in results[:3])
    assert results[3].status == "holds"  # unrelated scalar claims are checked as before
    assert results[4].status == "holds"  # scalars named differently are not matrices


def test_a_definition_involving_a_matrix_is_not_substituted() -> None:
    results = check_claims([r"X = AB", r"X = BA"], {}, [r"A = 2 B"], L, matrices=["A", "B"])
    assert [r.status for r in results] == ["unverified", "unverified"]


def test_engine_check_claims_passes_the_matrix_names(engine: Engine) -> None:
    [plain] = engine.check_claims([r"AB = BA"])
    [guarded] = engine.check_claims([r"AB = BA"], matrices=["A", "B"])
    assert plain.status == "holds" and guarded.status == "unverified"


def test_distribution_statements_are_never_contradicted() -> None:
    # \theta(x) + \theta(-x) = 1 reads as theta*x - theta*x = 0 != 1: it must not be called false.
    for claim in (
        r"\theta(x) + \theta(-x) = 1",
        r"\delta(-x) = \delta(x)",
        r"\operatorname{Heaviside}(x) + \operatorname{Heaviside}(-x) = 1",
        r"\operatorname{DiracDelta}(2x) = 5 \operatorname{DiracDelta}(x)",
    ):
        result = Engine().check_claims([claim], {"x": ["real"]})[0]
        assert result.status == "unverified", claim
        assert "distribution" in result.detail or "Dirac" in result.detail
