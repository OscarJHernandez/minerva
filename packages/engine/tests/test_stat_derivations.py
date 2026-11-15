"""Complete statistical derivations: maximum likelihood, Fisher information, Bayesian updates."""

import pytest
import sympy as sp
from minerva_engine import Engine, Limits, StatModelInput, Status, parse_text
from minerva_engine.indexed.instantiate import verify_equal_indexed

DATA = {
    "n": ["integer", "positive"],
    "x": ["real"],
    "v": ["positive"],
    "sigma": ["positive"],
    "mu": ["real"],
}


@pytest.fixture(scope="module")
def engine() -> Engine:
    # the default limits of the product (10 s sandbox timeout): a derivation must not need more
    return Engine(Limits(max_concurrency=2))


def mle(engine, family, estimate, parameters=None, parametrization=None, assumptions=None, **kw):
    return engine.statistical_model(
        StatModelInput(
            family=family,
            parametrization=parametrization,
            parameters=parameters or {},
            observation="x",
            sample_size=kw.pop("sample_size", "n"),
            iid=kw.pop("iid", True),
            estimate=estimate,
            action=kw.pop("action", "mle"),
            assumptions=assumptions or DATA,
            **kw,
        )
    )


def same(text: str, expected: str, **assumptions: list[str]) -> bool:
    a = {**DATA, **assumptions}
    return verify_equal_indexed(
        parse_text(text, a, allow_sums=True), parse_text(expected, a, allow_sums=True)
    ).passed


# ---------------------------------------------------------------------------------- maximum likelihood


def test_gaussian_mean_with_known_variance(engine) -> None:
    r = mle(engine, "normal", ["mu"], {"v": "v"}, "variance")
    assert r.status is Status.OK and r.evidence == "conditional"
    (mu,) = r.estimates
    assert same(mu.text, "Sum(x[i], (i, 1, n))/n")
    assert (mu.optimum, mu.optimum_evidence) == ("global", "proved")
    assert "the observations are independent and identically distributed" in r.premises
    assert all(c.passed for c in r.checks)


def test_gaussian_mean_and_variance_with_denominator_n(engine) -> None:
    r = mle(engine, "normal", ["mu", "v"], parametrization="variance")
    assert r.status is Status.OK
    mu, v = r.estimates
    assert same(mu.text, "Sum(x[i], (i, 1, n))/n")
    assert same(v.text, "Sum((x[i] - Sum(x[j], (j, 1, n))/n)**2, (i, 1, n))/n")
    assert not same(v.text, "Sum((x[i] - Sum(x[j], (j, 1, n))/n)**2, (i, 1, n))/(n - 1)")
    # the joint log-likelihood is not concave: only a local maximum, never "global"
    assert {e.optimum for e in r.estimates} == {"local"}
    assert v.admissible_if and not v.admissible_if[0].established  # a condition on the data


def test_bernoulli_proportion_is_global_and_names_the_boundary(engine) -> None:
    r = mle(engine, "bernoulli", ["p"], assumptions={"n": ["integer", "positive"]})
    assert r.status is Status.OK
    (p,) = r.estimates
    assert same(p.text, "Sum(x[i], (i, 1, n))/n")
    assert (p.optimum, p.optimum_evidence) == ("global", "proved")
    assert p.admissible_if and p.admissible_if[0].latex.startswith("0 <")


@pytest.mark.parametrize(
    ("data", "reason"),
    [(["0", "0", "0"], "candidate_inadmissible"), (["1", "1"], "candidate_inadmissible")],
)
def test_a_bernoulli_sample_without_both_outcomes_has_no_mle(engine, data, reason) -> None:
    r = engine.statistical_model(
        StatModelInput(
            family="bernoulli",
            observation="x",
            sample_size=str(len(data)),
            iid=True,
            estimate=["p"],
            action="mle",
            data=data,
        )
    )
    assert (r.status, r.reason) == (Status.CONDITIONS_NOT_SATISFIED, reason) and not r.estimates


def test_concrete_data_give_a_number(engine) -> None:
    r = engine.statistical_model(
        StatModelInput(
            family="bernoulli",
            observation="x",
            sample_size="4",
            iid=True,
            estimate=["p"],
            action="mle",
            data=["1", "0", "1", "1"],
        )
    )
    assert r.status is Status.OK and r.estimates[0].text == "3/4"


def test_the_likelihood_of_several_observations_needs_independence(engine) -> None:
    r = mle(engine, "normal", ["mu"], {"v": "v"}, "variance", iid=None)
    assert (r.status, r.reason) == (Status.CONDITIONS_NOT_SATISFIED, "independence_not_declared")
    assert not r.estimates


def test_a_family_whose_support_depends_on_the_parameter_has_no_score(engine) -> None:
    r = mle(
        engine,
        "uniform_0_theta",
        ["theta"],
        assumptions={"n": ["integer", "positive"], "x": ["positive"]},
    )
    assert (r.status, r.reason) == (Status.UNSUPPORTED, "regularity_not_satisfied")


@pytest.mark.parametrize(
    ("estimate", "wrong"),
    [
        (["mu"], {"estimate:mu": "Sum(x[i], (i, 1, n))/(n + 1)"}),
        (["mu"], {"estimate:mu": "x[1]"}),
        (["mu"], {"estimate:mu": "Sum(x[i], (i, 1, n))/n + 1/1000000"}),
        (
            ["mu", "v"],
            {
                "estimate:mu": "Sum(x[i], (i, 1, n))/n",
                "estimate:v": "Sum((x[i] - Sum(x[j], (j, 1, n))/n)**2, (i, 1, n))/(n - 1)",
            },
        ),
        (
            ["mu", "v"],
            {
                "estimate:mu": "Sum(x[i], (i, 1, n))/n",
                "estimate:v": "Sum((x[i] - mu)**2, (i, 1, n))/n",
            },
        ),  # mu unknown
    ],
)
def test_wrong_estimators_are_rejected(engine, estimate, wrong) -> None:
    r = mle(
        engine,
        "normal",
        estimate,
        {"v": "v"} if estimate == ["mu"] else None,
        "variance",
        candidate=wrong,
    )
    assert r.status is Status.NOT_VERIFIED and not r.estimates


def test_the_right_estimator_is_accepted_as_a_candidate(engine) -> None:
    r = mle(
        engine,
        "normal",
        ["mu"],
        {"v": "v"},
        "variance",
        candidate={"estimate:mu": "Sum(x[i], (i, 1, n))/n"},
    )
    assert r.status is Status.OK


def test_the_log_likelihood_and_the_score_are_checked_against_a_direct_sum(engine) -> None:
    ll = mle(engine, "normal", ["mu", "v"], parametrization="variance", action="log_likelihood")
    assert (
        ll.status is Status.OK
        and "loglik" in ll.expressions
        and ll.checks
        and all(c.passed for c in ll.checks)
    )
    sc = mle(engine, "normal", ["mu", "v"], parametrization="variance", action="score")
    assert sc.status is Status.OK and set(sc.expressions) == {"loglik", "score:mu", "score:v"}
    # the score for mu is (sum x - n mu) / v
    assert same(sc.expressions["score:mu"], "(Sum(x[i], (i, 1, n)) - n*mu)/v")


# ---------------------------------------------------------------------------------- Fisher information


def fisher(engine, family, estimate, parameters=None, **kw):
    return engine.statistical_model(
        StatModelInput(
            family=family,
            parameters=parameters or {},
            estimate=estimate,
            action="fisher_information",
            assumptions={
                "mu": ["real"],
                "sigma": ["positive"],
                "p": ["positive"],
                "lam": ["positive"],
                "n": ["integer", "positive"],
            },
            **kw,
        )
    )


def test_normal_information_for_the_mean_and_the_standard_deviation(engine) -> None:
    r = fisher(engine, "normal", ["mu", "sigma"])
    assert r.status is Status.OK
    assert r.matrix_text == [["sigma**(-2)", "0"], ["0", "2/sigma**2"]] or r.matrix_text == [
        ["1/sigma**2", "0"],
        ["0", "2/sigma**2"],
    ]
    assert all(c.passed for c in r.checks)


def test_information_is_additive_over_independent_observations_and_says_so(engine) -> None:
    r = fisher(engine, "normal", ["mu", "sigma"], sample_size="n", iid=True)
    assert r.status is Status.OK and r.evidence == "conditional" and r.premises
    refused = fisher(engine, "normal", ["mu", "sigma"], sample_size="n")
    assert (refused.status, refused.reason) == (
        Status.CONDITIONS_NOT_SATISFIED,
        "independence_not_declared",
    )


def test_non_regular_and_non_polynomial_cases_are_refused_not_guessed(engine) -> None:
    uniform = engine.statistical_model(
        StatModelInput(
            family="uniform_0_theta",
            estimate=["theta"],
            action="fisher_information",
            assumptions={"theta": ["positive"]},
        )
    )
    assert (uniform.status, uniform.reason) == (Status.UNSUPPORTED, "regularity_not_satisfied")
    # the Hessian of the Cauchy location is rational in x: its expectation needs moments that do not exist
    cauchy = engine.statistical_model(
        StatModelInput(
            family="cauchy",
            estimate=["loc"],
            parameters={"scale": "scale"},
            action="fisher_information",
            assumptions={"loc": ["real"], "scale": ["positive"]},
        )
    )
    assert (cauchy.status, cauchy.reason) == (Status.UNSUPPORTED, "no_closed_form")


def test_a_constant_hessian_needs_no_moments_the_gamma_shape_has_the_trigamma_information(
    engine,
) -> None:
    r = engine.statistical_model(
        StatModelInput(
            family="gamma",
            estimate=["alpha"],
            parameters={"beta": "beta"},
            action="fisher_information",
            assumptions={"alpha": ["positive"], "beta": ["positive"]},
        )
    )
    assert r.status is Status.OK and r.matrix_text == [["polygamma(1, alpha)"]]
    assert all(c.passed for c in r.checks)


@pytest.mark.parametrize(
    "wrong",
    [
        [["1/sigma**2", "0"], ["0", "1/sigma**2"]],  # I(sigma) is 2/sigma^2
        [["1/sigma**2", "1/sigma"], ["1/sigma", "2/sigma**2"]],  # mean and scale are orthogonal
        [["1/sigma**2", "0"], ["0", "2/sigma**2 + 1/1000000"]],
        [["1/sigma", "0"], ["0", "2/sigma"]],
    ],
)
def test_wrong_information_matrices_are_rejected(engine, wrong) -> None:
    candidate = {f"matrix[{i}][{j}]": wrong[i][j] for i in range(2) for j in range(2)}
    r = fisher(engine, "normal", ["mu", "sigma"], candidate=candidate)
    assert r.status is Status.NOT_VERIFIED and r.matrix_text is None


def test_a_refuted_matrix_stops_at_the_first_disagreement_and_a_right_one_runs_every_point(
    engine,
) -> None:
    def candidate(m):
        return {f"matrix[{i}][{j}]": m[i][j] for i in range(2) for j in range(2)}

    wrong = fisher(
        engine,
        "normal",
        ["mu", "sigma"],
        candidate=candidate([["1/sigma**2", "0"], ["0", "1/sigma**2"]]),
    )
    form, mean = wrong.checks
    assert wrong.status is Status.NOT_VERIFIED and wrong.matrix_text is None
    assert form.points_checked < 3 and not form.passed and form.points_agreed < form.points_checked
    assert not mean.passed and "not completed" in mean.note  # never reported as a failed mean check

    right = fisher(
        engine,
        "normal",
        ["mu", "sigma"],
        candidate=candidate([["1/sigma**2", "0"], ["0", "2/sigma**2"]]),
    )
    assert right.status is Status.OK
    assert all(c.passed and c.points_checked >= 3 for c in right.checks)


# ---------------------------------------------------------------------------------- posteriors


def posterior(engine, family, estimate, prior, parameters=None, **kw):
    return engine.statistical_model(
        StatModelInput(
            family=family,
            parameters=parameters or {},
            estimate=estimate,
            prior=prior,
            action="posterior",
            assumptions={
                "n": ["integer", "positive"],
                "k": ["integer", "nonnegative"],
                "alpha": ["positive"],
                "beta": ["positive"],
                "x": ["nonnegative"],
            },
            **kw,
        )
    )


def test_beta_binomial_by_matching_the_kernel(engine) -> None:
    r = posterior(
        engine,
        "binomial",
        ["p"],
        {"family": "beta", "parameters": {"alpha": "alpha", "beta": "beta"}},
        {"n": "n"},
        observation="k",
    )
    assert r.status is Status.OK and r.posterior is not None and r.posterior.family == "beta"
    got = {
        k: parse_text(v, {"n": ["integer", "positive"]}) for k, v in r.posterior.parameters.items()
    }
    want = {
        "alpha": parse_text("alpha + k", {"n": ["integer", "positive"]}),
        "beta": parse_text("beta + n - k", {"n": ["integer", "positive"]}),
    }
    assert all(sp.simplify(got[k] - want[k]) == 0 for k in want)
    assert all(c.passed for c in r.checks) and r.evidence == "identity"
    assert any("prior" in p for p in r.premises)  # the prior is an assumption, said so
    assert r.posterior.dropped_factors_latex  # the factor that does not depend on p was recorded


def test_a_conjugate_family_is_found_not_looked_up(engine) -> None:
    r = posterior(
        engine,
        "poisson",
        ["lam"],
        {"family": "gamma", "parameters": {"alpha": "alpha", "beta": "beta"}},
        observation="x",
        sample_size="n",
        iid=True,
    )
    assert r.status is Status.OK and r.posterior is not None and r.posterior.family == "gamma"
    assert r.posterior.parameters["beta"] == "beta + n" and r.evidence == "conditional"
    # a beta prior on a Poisson rate gives a kernel outside the supported families
    bad = posterior(
        engine,
        "poisson",
        ["lam"],
        {"family": "beta", "parameters": {"alpha": "alpha", "beta": "beta"}},
        observation="x",
        sample_size="n",
        iid=True,
    )
    assert (bad.status, bad.reason) == (
        Status.UNSUPPORTED,
        "kernel_not_recognised",
    ) and bad.posterior is None


@pytest.mark.parametrize(
    "wrong",
    [
        {"posterior:alpha": "alpha + k", "posterior:beta": "beta"},  # not updated
        {
            "posterior:alpha": "alpha + k",
            "posterior:beta": "beta + n",
        },  # counts the successes twice
        {"posterior:alpha": "alpha", "posterior:beta": "beta + n - k"},
        {"posterior:alpha": "alpha + k", "posterior:beta": "beta + n - k + 1/1000000"},
    ],
)
def test_wrong_posteriors_are_rejected(engine, wrong) -> None:
    r = posterior(
        engine,
        "binomial",
        ["p"],
        {"family": "beta", "parameters": {"alpha": "alpha", "beta": "beta"}},
        {"n": "n"},
        observation="k",
        candidate=wrong,
    )
    assert r.status is Status.NOT_VERIFIED and r.posterior is None


def test_an_invalid_prior_is_refused(engine) -> None:
    r = posterior(
        engine,
        "binomial",
        ["p"],
        {"family": "beta", "parameters": {"alpha": "-1", "beta": "beta"}},
        {"n": "n"},
        observation="k",
    )
    assert (r.status, r.reason) == (Status.CONDITIONS_NOT_SATISFIED, "parameter_invalid")
