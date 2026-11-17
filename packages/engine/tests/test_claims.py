import pytest
from minerva_engine import Engine, check_claim, check_claims, split_equation
from minerva_engine.testing import budget_limits


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


L = budget_limits()


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
    engine = Engine(budget_limits())
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
    engine = Engine(budget_limits())
    results = engine.check_claims(
        [
            r"\cos\theta = \cos\phi\cos(\omega t) - \sin\phi\sin(\omega t)",
            r"\theta = \phi + \omega t",
        ],
        {},
        [r"\theta = \phi + \omega t"],
    )
    # The first is a consequence of the relation; the second *is* the relation, which cannot justify
    # itself, so it is only a definition stated earlier.
    assert [r.status for r in results] == ["conditional", "unverified"]
    assert results[0].given == [r"\theta = \phi + \omega t"]
    assert "definition stated earlier" in results[1].detail


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


# -- the matrix guard reads the text, not the parsed expression ----------------------------------
@pytest.mark.parametrize(
    "formula",
    [
        r"AB - BA = 0",  # zero as commuting symbols: SymPy cancels the matrices while reading
        r"A B A^{-1} B^{-1} = 1",
        r"\frac{AB}{BA} = 1",
        r"(AB)^{-1} AB = 1",
        r"AB - BA = x - x",
        r"A_1 A_2 = A_2 A_1",  # a declared A covers its subscripted family
        r"(A+B)^2 = A^2 + 2AB + B^2",
        r"(A+B)(A-B) = A^2 - B^2",
        r"(AB)^{-1} = A^{-1} B^{-1}",
    ],
)
def test_false_or_conditional_matrix_identities_are_never_checked(formula: str) -> None:
    [result] = check_claims([formula], {}, [], L, matrices=["A", "B"])
    assert result.status == "unverified"
    assert "matrix" in result.detail


def test_a_matrix_alias_defined_earlier_is_a_matrix_too() -> None:
    results = check_claims([r"CD = DC"], {}, [r"C = AB", r"D = C"], L, matrices=["A", "B"])
    assert [r.status for r in results] == ["unverified"]


def test_noncommutative_declarations_mean_matrix_not_a_broken_check() -> None:
    results = check_claims(
        [r"AB = BA", r"\sin^2 x + \cos^2 x = 1"],
        {"A": ["noncommutative"], "B": ["noncommutative"], "x": ["not_an_assumption"]},
        [],
        L,
    )
    assert [r.status for r in results] == ["unverified", "holds"]


def test_engine_keeps_checking_scalars_when_noncommutative_is_declared(engine: Engine) -> None:
    results = engine.check_claims(
        [r"AB = BA", r"\sin^2 x + \cos^2 x = 1"], {"A": ["noncommutative"], "x": ["bogus"]}
    )
    assert [r.status for r in results] == ["unverified", "holds"]


@pytest.mark.parametrize(
    "formula",
    [r"(AB)^T = A^T B^T", r"(AB)^T = B^T A^T", r"(AB)^{\top} = B^{\top} A^{\top}", r"A^{*} = A"],
)
def test_transpose_notation_is_not_read_as_a_positive_power(formula: str) -> None:
    [result] = check_claims([formula], {}, [], L)
    assert result.status == "unverified"
    assert "transpose" in result.detail


def test_unsupported_matrix_notation_is_not_checked() -> None:
    for formula in (r"\det(AB) = \det(A)\det(B)", r"[A,B] = 0", r"\hat{H} = H"):
        [result] = check_claims([formula], {}, [], L, matrices=["A", "B", "H"])
        assert result.status in {"unverified", "unparsed"}


def test_scalar_names_that_only_look_like_matrices_are_still_checked() -> None:
    [result] = check_claims([r"(a+b)^2 = a^2 + 2ab + b^2"], {}, [], L, matrices=["A", "B"])
    assert result.status == "holds"


@pytest.mark.parametrize(
    "formula",
    [
        r"\frac{dy}{dt} = \frac{y}{t}",  # the general parser cancels the d's: it would call this an identity
        r"\frac{d^2 y}{dt^2} + y = 0",
        r"\frac{d}{dx}\sin x = \cos x",
        r"\frac{\mathrm{d}y}{\mathrm{d}x} = \frac{y}{x}",
        r"\dfrac{dy}{dx} = y",
        r"\frac{\partial f}{\partial x} = 1",
    ],
)
def test_derivative_notation_is_a_relation_never_an_identity(formula: str) -> None:
    [result] = check_claims([formula], {}, [], L)
    assert result.status == "unverified" and result.kind == "relation"
    assert "derivative" in result.detail


def test_ordinary_fractions_with_a_d_are_still_checked() -> None:
    for formula in (r"\frac{d}{2} = \frac{d}{2}", r"\frac{x+1}{d} = \frac{1+x}{d}"):
        [result] = check_claims([formula], {}, [], L)
        assert result.status == "holds"


@pytest.mark.parametrize(
    "latex",
    [r"\frac{dx}{dy} = \frac{x}{y}", r"\frac{d^2x}{dt^2} = \frac{x}{t^2}"],
)
def test_leibniz_equations_are_never_checked_as_ratios_of_letters(latex: str) -> None:
    # d/dy is an operator: reading it as the letters d, y made this "identity" parse and be tested.
    assert check_claim(latex, {}, L).status not in {"holds", "fails", "not_identity"}


# -- statements a correct reply makes that used to be flagged for the wrong reason ------------------
def _check(formula: str, relations: list[str] | None = None) -> object:
    return check_claims([formula], {}, relations or [], L)[0]


def test_a_definition_restated_is_not_justified_by_itself() -> None:
    r0 = r"r_0 = \frac{p_{\theta}^2}{G M m^2}"
    restated = _check(r0, [r0])
    assert restated.status == "unverified" and "definition stated earlier" in restated.detail  # type: ignore[attr-defined]
    assert _check("n=3", ["n=3"]).status == "unverified"  # type: ignore[attr-defined]
    # a real consequence of the definition is still `conditional`, with the definition as its premise
    consequence = _check(r"p_{\theta}^2 = G M m^2 r_0", [r0])
    assert consequence.status == "conditional" and consequence.given == [r0]  # type: ignore[attr-defined]
    # and one that disagrees with it is still `fails`
    wrong = _check(r"p_{\theta}^2 = 2 G M m^2 r_0", [r0])
    assert wrong.status == "fails"  # type: ignore[attr-defined]


def test_delta_of_t_is_not_called_a_dirac_delta_but_is_never_sampled_or_contradicted() -> None:
    perturbation = _check(r"r(t) = r_0 + \delta(t)")
    assert perturbation.status == "unverified"  # type: ignore[attr-defined]
    assert "function of its argument" in perturbation.detail  # type: ignore[attr-defined]
    assert "use the distribution operation" not in perturbation.detail  # type: ignore[attr-defined]
    # the true statement about a delta is still never marked false
    assert _check(r"\delta(-x) = \delta(x)").status == "unverified"  # type: ignore[attr-defined]
    assert _check(r"\theta(0) = 0, \dot{r}(0) = 1").status != "fails"  # type: ignore[attr-defined]


def test_a_definition_of_e_or_i_is_a_definition_not_a_parse_error() -> None:
    energy = _check(r"E = \frac{1}{2} m \dot{r}^2 - \frac{G M m}{r}")
    assert energy.status == "unverified" and energy.kind == "relation"  # type: ignore[attr-defined]
    assert "E_0 or W" in energy.detail  # type: ignore[attr-defined]


def test_an_implication_is_said_to_be_one() -> None:
    result = _check(r"\frac{G M}{r_0^2} = \frac{p^2}{m^2 r_0^3} \implies r_0 = \frac{p^2}{G M m^2}")
    assert result.status == "unverified" and "implication" in result.detail  # type: ignore[attr-defined]
    assert _check(r"x = 1 \Rightarrow x^2 = 1").status == "unverified"  # type: ignore[attr-defined]


def test_ell_is_a_letter_and_a_false_identity_with_it_is_still_not_an_identity() -> None:
    assert _check(r"\ell^2 = \ell").status == "not_identity"  # type: ignore[attr-defined]
    assert _check(r"\ell (\ell + 1) = \ell^2 + \ell").status == "holds"  # type: ignore[attr-defined]
    conic = _check(r"r(\theta) = \frac{\ell}{1 + e \cos \theta}")
    assert conic.status not in ("unparsed", "fails")  # type: ignore[attr-defined]


def test_unreadable_notation_is_named_not_just_refused() -> None:
    prime = _check(r"f'(r) = 2")
    assert prime.status == "unparsed" and "derivative notation" in prime.detail  # type: ignore[attr-defined]
    bar = _check(r"\left. \frac{d^2 V}{dr^2} \right|_{r=r_0} = 1")
    assert bar.status in ("unparsed", "unverified")  # type: ignore[attr-defined]
