"""Observations and sample sizes are validated once, before any computation (issues #230, #231, #232).

A concrete observation outside the support of the family, a sample size that is not a count and a
Fisher information that ignores the data must be refused or corrected; each case has a valid twin that
still answers.
"""

import mpmath as mp
import pytest
import sympy as sp
from minerva_engine import Engine, StatModelInput, Status
from minerva_engine.testing import budget_limits

A = {
    "mu": ["real"],
    "sigma": ["positive"],
    "n": ["integer", "positive"],
    "N": ["integer", "positive"],
}


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(budget_limits(max_concurrency=2))


def run(engine, family, action="mle", **kw):
    kw.setdefault("observation", "x")
    return engine.statistical_model(
        StatModelInput(family=family, action=action, assumptions=A, **kw)
    )


def refused(r, status, reason) -> None:
    assert (r.status, r.reason) == (status, reason), r.message
    assert not r.estimates and not r.matrix_text and not r.expressions and not r.steps


NOT_MET = Status.CONDITIONS_NOT_SATISFIED
BAD = Status.PARSE_ERROR


# ---------------------------------------------------------------------------------- support (#230)


def test_fractional_bernoulli_data_are_refused_and_zero_one_data_answer(engine) -> None:
    refused(
        run(engine, "bernoulli", estimate=["p"], iid=True, data=["1/5", "4/5"]),
        NOT_MET,
        "observation_outside_support",
    )
    ok = run(engine, "bernoulli", estimate=["p"], iid=True, data=["0", "1", "1", "0", "1"])
    assert ok.status is Status.OK and ok.estimates[0].text == "3/5"


def test_fractional_poisson_data_are_refused_and_counts_answer(engine) -> None:
    refused(
        run(engine, "poisson", estimate=["lam"], iid=True, data=["1/2", "3/2"]),
        NOT_MET,
        "observation_outside_support",
    )
    ok = run(engine, "poisson", estimate=["lam"], iid=True, data=["0", "2", "5"])
    assert ok.status is Status.OK and ok.estimates[0].text == "7/3"


def test_a_bernoulli_log_likelihood_of_two_is_refused(engine) -> None:
    refused(
        run(engine, "bernoulli", "log_likelihood", estimate=["p"], data=["2"]),
        NOT_MET,
        "observation_outside_support",
    )
    assert (
        run(engine, "bernoulli", "log_likelihood", estimate=["p"], data=["1"]).status is Status.OK
    )


@pytest.mark.parametrize("bad", [["-1", "2"], ["1/2", "2"], ["7", "2"]])
def test_binomial_counts_must_be_integers_between_zero_and_n(engine, bad) -> None:
    kw = dict(estimate=["p"], parameters={"n": "5"}, iid=True)
    refused(run(engine, "binomial", data=bad, **kw), NOT_MET, "observation_outside_support")
    assert run(engine, "binomial", data=["3", "2"], **kw).status is Status.OK
    # n itself is a possible count
    assert run(engine, "binomial", data=["5", "2"], **kw).status is Status.OK


def test_a_binomial_count_against_a_symbolic_n_is_an_unestablished_condition(engine) -> None:
    r = run(
        engine,
        "binomial",
        "log_likelihood",
        estimate=["p"],
        parameters={"n": "N"},
        iid=True,
        data=["3", "2"],
    )
    assert r.status is Status.OK
    assert any(not c.established and "every observation" in c.latex for c in r.conditions)
    assert r.evidence == "conditional" and r.premises  # holds only given the symbolic bound


@pytest.mark.parametrize("data", [["0", "1/2"], ["1/4", "1"]])
def test_beta_support_is_the_open_interval(engine, data) -> None:
    kw = dict(estimate=["alpha"], parameters={"beta": "2"}, iid=True)
    refused(run(engine, "beta", data=data, **kw), NOT_MET, "observation_outside_support")
    # (the Beta maximum has no closed form here; the log-likelihood is the valid twin)
    good = run(engine, "beta", "log_likelihood", data=["1/4", "1/2", "3/4"], **kw)
    assert good.status is Status.OK


def test_gamma_support_excludes_zero(engine) -> None:
    kw = dict(estimate=["alpha"], parameters={"beta": "1"}, iid=True)
    refused(run(engine, "gamma", data=["0", "2"], **kw), NOT_MET, "observation_outside_support")


def test_a_single_observation_needs_no_independence(engine) -> None:
    one = run(engine, "poisson", estimate=["lam"], data=["4"])
    assert one.status is Status.OK and one.premises == []
    two = run(engine, "poisson", estimate=["lam"], data=["4", "5"])
    refused(two, NOT_MET, "independence_not_declared")


def test_the_bound_of_an_estimated_parameter_is_left_to_the_estimator(engine) -> None:
    # Binomial with n estimated: the data cannot be compared with an unknown bound here
    r = run(engine, "binomial", estimate=["n", "p"], iid=True, data=["3", "2"])
    assert r.reason != "observation_outside_support"


@pytest.mark.parametrize(("data", "bound"), [(["1"], "1"), (["1", "2"], "2")])
def test_a_uniform_likelihood_holds_only_where_theta_admits_the_data(engine, data, bound) -> None:
    # 1/theta is the density only on (0, theta): for theta at or below an observation the likelihood is 0
    r = engine.statistical_model(
        StatModelInput(
            family="uniform_0_theta",
            action="log_likelihood",
            estimate=["theta"],
            observation="x",
            data=data,
            iid=True,
        )
    )
    assert r.status is Status.OK
    assert r.evidence == "conditional"
    assert any(f"theta > {bound}" in p for p in r.premises), r.premises
    assert all(s.evidence == "conditional" for s in r.steps)


# ---------------------------------------------------------------------------------- counts (#232)


@pytest.mark.parametrize("size", ["-1", "0", "1/2", "2n", "n+1", "x", "mu", " ", "n!"])
def test_a_sample_size_that_is_not_a_count_is_refused_before_any_computation(engine, size) -> None:
    for action in ("fisher_information", "mle", "log_likelihood"):
        r = run(
            engine,
            "normal",
            action,
            estimate=["mu"],
            parameters={"sigma": "1"},
            sample_size=size,
            iid=True,
        )
        refused(r, BAD, "sample_size_invalid")


def test_a_numeric_sample_size_is_a_real_integer_and_n_stays_a_symbol(engine) -> None:
    four = run(engine, "normal", "fisher_information", estimate=["mu"],
               parameters={"sigma": "1"}, sample_size="4", iid=True)  # fmt: skip
    assert four.status is Status.OK and four.matrix_text == [["4"]]
    assert four.evidence == "conditional" and four.premises
    sym = run(engine, "normal", "fisher_information", estimate=["mu"],
              parameters={"sigma": "1"}, sample_size="n", iid=True)  # fmt: skip
    assert sym.status is Status.OK and sym.matrix_text == [["n"]]
    one = run(engine, "normal", "fisher_information", estimate=["mu"],
              parameters={"sigma": "1"}, sample_size="1")  # fmt: skip
    assert one.status is Status.OK and one.matrix_text == [["1"]] and not one.premises


def test_a_numeric_sample_size_without_data_is_refused_for_an_estimator(engine) -> None:
    r = run(engine, "normal", estimate=["mu"], parameters={"sigma": "1"}, sample_size="4", iid=True)
    refused(r, BAD, "sample_size_invalid")


def test_a_sample_size_is_never_a_parameter_name(engine) -> None:
    r = run(
        engine,
        "binomial",
        "fisher_information",
        estimate=["p"],
        parameters={"n": "5"},
        sample_size="n",
        iid=True,
    )
    refused(r, BAD, "sample_size_invalid")


def test_the_count_and_the_data_must_agree(engine) -> None:
    kw = dict(estimate=["mu"], parameters={"sigma": "1"}, iid=True)
    ok = run(
        engine, "normal", "fisher_information", sample_size="4", data=["1", "2", "3", "4"], **kw
    )
    assert ok.status is Status.OK and ok.matrix_text == [["4"]]
    refused(
        run(engine, "normal", "fisher_information", sample_size="4", data=["1", "2"], **kw),
        NOT_MET,
        "sample_size_conflict",
    )
    refused(
        run(engine, "normal", "fisher_information", sample_size="n", data=["1", "2"], **kw),
        NOT_MET,
        "sample_size_conflict",
    )


@pytest.mark.parametrize("action", ["mle", "log_likelihood", "score", "fisher_information"])
def test_empty_data_are_refused(engine, action) -> None:
    r = run(engine, "normal", action, estimate=["mu"], parameters={"sigma": "1"}, data=[], iid=True)
    refused(r, NOT_MET, "no_observations")


# ---------------------------------------------------------------------------------- Fisher (#231)


def _information_of_one(sigma: sp.Rational, which: str) -> mp.mpf:
    """-E[d^2/dtheta^2 log f] by quadrature of a numerically differentiated log density (no Minerva code)."""
    mp.mp.dps = 30

    def logf(x, mu, s):
        return -((x - mu) ** 2) / (2 * s**2) - mp.log(s) - mp.log(mp.sqrt(2 * mp.pi))

    def density(x):
        return mp.npdf(x, 0, float(sigma))

    def second(x):
        if which == "mu":
            return mp.diff(lambda m: logf(x, m, mp.mpf(sigma)), 0, 2)
        return mp.diff(lambda s: logf(x, 0, s), mp.mpf(sigma), 2)

    return -mp.quad(lambda x: second(x) * density(x), [-mp.inf, 0, mp.inf])


def test_fisher_information_counts_the_observations_it_was_given(engine) -> None:
    data = ["1", "2", "3", "4"]
    r = run(engine, "normal", "fisher_information", estimate=["mu"],
            parameters={"sigma": "1"}, data=data, iid=True)  # fmt: skip
    assert r.status is Status.OK and r.matrix_text == [["4"]]
    assert r.evidence == "conditional" and r.premises
    # independent value: the sum over the four observations of -E[second derivative of log f]
    total = sum(_information_of_one(sp.Integer(1), "mu") for _ in data)
    assert abs(total - 4) < mp.mpf(10) ** -12
    assert all(c.passed for c in r.checks)


def test_the_sigma_information_of_four_observations_is_four_times_one(engine) -> None:
    r = run(engine, "normal", "fisher_information", estimate=["sigma"],
            parameters={"mu": "0"}, data=["1", "2", "3", "4"], iid=True)  # fmt: skip
    assert r.status is Status.OK
    expr = sp.sympify(r.matrix_text[0][0], locals={"sigma": sp.Symbol("sigma", positive=True)})
    sigma = sp.Rational(3, 2)
    shown = mp.mpf(str(sp.N(expr.subs(sp.Symbol("sigma", positive=True), sigma), 25)))
    expected = sum(_information_of_one(sigma, "sigma") for _ in range(4))
    assert abs(shown - expected) < mp.mpf(10) ** -10


def test_fisher_information_of_several_observations_needs_independence(engine) -> None:
    kw = dict(estimate=["mu"], parameters={"sigma": "1"})
    refused(
        run(engine, "normal", "fisher_information", data=["1", "2", "3", "4"], **kw),
        NOT_MET,
        "independence_not_declared",
    )
    refused(
        run(engine, "normal", "fisher_information", sample_size="4", **kw),
        NOT_MET,
        "independence_not_declared",
    )
    one = run(engine, "normal", "fisher_information", data=["2"], **kw)
    assert one.status is Status.OK and one.matrix_text == [["1"]] and not one.premises


def test_a_candidate_for_the_information_is_checked_against_the_supplied_count(engine) -> None:
    kw = dict(estimate=["mu"], parameters={"sigma": "1"}, sample_size="4", iid=True)
    good = run(engine, "normal", "fisher_information", candidate={"matrix[0][0]": "4"}, **kw)
    assert good.status is Status.OK and good.matrix_text == [["4"]]
    for wrong in ("1", "-1", "5", "-4", "n"):
        r = run(engine, "normal", "fisher_information", candidate={"matrix[0][0]": wrong}, **kw)
        assert r.status is not Status.OK, wrong
    sym = dict(estimate=["mu"], parameters={"sigma": "1"}, sample_size="n", iid=True)
    ok = run(engine, "normal", "fisher_information", candidate={"matrix[0][0]": "n"}, **sym)
    assert ok.status is Status.OK
    bad = run(engine, "normal", "fisher_information", candidate={"matrix[0][0]": "2*n"}, **sym)
    assert bad.status is not Status.OK


def test_a_stranger_symbol_in_a_candidate_is_not_sampled_into_agreement(engine) -> None:
    r = run(engine, "normal", "fisher_information", estimate=["mu"], parameters={"sigma": "1"},
            sample_size="4", iid=True, candidate={"matrix[0][0]": "q"})  # fmt: skip
    assert r.status is not Status.OK
