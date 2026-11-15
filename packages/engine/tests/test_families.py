"""Probability families: parameters, moments derived and verified, and what is refused."""

import dataclasses
import random

import mpmath as mp
import pytest
import sympy as sp
from minerva_engine import Engine, Limits, StatModelInput, Status
from minerva_engine.stats import moments as mom
from minerva_engine.stats.domain import decide
from minerva_engine.stats.families import FAMILIES, x
from minerva_engine.stats.numeric import Numeric, sample_parameters
from minerva_engine.stats.statmodel import stat_model

A = {
    "mu": ["real"],
    "sigma": ["positive"],
    "p": ["positive"],
    "n": ["integer", "positive"],
    "lam": ["positive"],
    "alpha": ["positive"],
    "beta": ["positive"],
}


def model(family: str, parameters: dict[str, str], **kw) -> StatModelInput:
    return StatModelInput(
        family=family, parameters=parameters, assumptions=A, action="moments", **kw
    )


def test_the_constraint_oracle() -> None:
    assert decide("positive", sp.Integer(3)) is True
    assert decide("positive", sp.Integer(-1)) is False
    assert decide("unit_open", sp.Rational(1, 2)) is True
    assert decide("unit_open", sp.Rational(3, 2)) is False
    assert decide("unit_open", sp.Integer(1)) is False  # the boundary is excluded
    assert decide("positive_integer", sp.Integer(0)) is False
    assert decide("positive_integer", sp.Rational(3, 2)) is False
    assert decide("real", sp.I) is False
    # symbolic: provably satisfied, provably violated, or undecided (assumed, reported)
    assert decide("positive", sp.Symbol("s", positive=True)) is True
    assert decide("positive", sp.Symbol("s", negative=True)) is False
    assert decide("unit_open", sp.Symbol("p", positive=True)) is None
    assert decide("unit_open", sp.Symbol("p", negative=True)) is False


def test_sampled_parameters_respect_the_constraints() -> None:
    family = FAMILIES["bernoulli"]
    p = sp.Symbol("p", positive=True)
    samples = sample_parameters({"p": p}, family.params(None), random.Random(1), 5)
    assert len(samples) == 5 and all(0 < v["p"] < 1 for _, v in samples)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_density_integrates_to_one(name: str) -> None:
    family = FAMILIES[name]
    exprs = {
        p.name: sp.Rational(3, 4)
        if p.constraint == "unit_open"
        else sp.Integer(3)
        if p.constraint == "positive_integer"
        else (sp.Rational(2) if p.constraint == "positive" else sp.Rational(1, 2))
        for p in family.params(None)
    }
    total, _ = Numeric(family, None, exprs).moment(0)
    assert abs(total - 1) < mp.mpf(10) ** -8


@pytest.mark.parametrize(
    ("name", "values", "mean", "variance"),
    [
        ("normal", {"mu": 1, "sigma": 2}, 1, 4),
        ("bernoulli", {"p": sp.Rational(1, 4)}, sp.Rational(1, 4), sp.Rational(3, 16)),
        ("binomial", {"n": 5, "p": sp.Rational(1, 5)}, 1, sp.Rational(4, 5)),
        ("poisson", {"lam": 3}, 3, 3),
        ("beta", {"alpha": 2, "beta": 3}, sp.Rational(2, 5), sp.Rational(1, 25)),
        ("gamma", {"alpha": 3, "beta": 2}, sp.Rational(3, 2), sp.Rational(3, 4)),
        ("uniform_0_theta", {"theta": 6}, 3, 3),
    ],
)
def test_quadrature_reproduces_textbook_moments(name, values, mean, variance) -> None:
    num = Numeric(FAMILIES[name], None, {k: sp.sympify(v) for k, v in values.items()})
    (m1, _), (m2, _) = num.moment(1), num.moment(2)
    assert abs(m1 - mp.mpf(str(sp.N(mean)))) < mp.mpf(10) ** -8
    assert abs((m2 - m1**2) - mp.mpf(str(sp.N(variance)))) < mp.mpf(10) ** -8


def test_convergence_is_established_before_a_moment_is_trusted() -> None:
    cauchy = Numeric(FAMILIES["cauchy"], None, {"loc": sp.Integer(0), "scale": sp.Integer(1)})
    assert cauchy.absolute_moment_exists(1) is False
    assert cauchy.absolute_moment_exists(0) is True  # a density integrates absolutely
    normal = Numeric(FAMILIES["normal"], None, {"mu": sp.Integer(1), "sigma": sp.Integer(2)})
    assert normal.absolute_moment_exists(2) is True


def test_symbolic_moments_come_from_integration_and_summation() -> None:
    deadline = 1e12
    mu, sigma = FAMILIES["normal"].symbols["mu"], FAMILIES["normal"].symbols["sigma"]
    assert mom.raw_moment(FAMILIES["normal"], None, 2, deadline) == mu**2 + sigma**2
    n, p = FAMILIES["binomial"].symbols["n"], FAMILIES["binomial"].symbols["p"]
    assert sp.expand(mom.raw_moment(FAMILIES["binomial"], None, 1, deadline) - n * p) == 0
    assert (
        sp.expand(mom.raw_moment(FAMILIES["binomial"], None, 2, deadline) - n * p * (n * p - p + 1))
        == 0
    )


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(Limits(timeout_seconds=30, max_concurrency=2))


def test_the_normal_family_in_both_parametrisations(engine) -> None:
    r = engine.statistical_model(model("normal", {"mu": "mu", "sigma": "sigma"}))
    assert r.status is Status.OK and r.moments == {
        "mean": "mu",
        "variance": "sigma**2",
        "second": "mu**2 + sigma**2",
    }
    r = engine.statistical_model(
        StatModelInput(
            family="normal",
            parametrization="variance",
            parameters={"mu": "mu", "v": "v"},
            assumptions={"mu": ["real"], "v": ["positive"]},
            action="moments",
        )
    )
    assert r.status is Status.OK and r.moments["variance"] == "v"


def test_numeric_parameter_values_give_numeric_moments(engine) -> None:
    r = engine.statistical_model(model("normal", {"mu": "2", "sigma": "3"}))
    assert r.status is Status.OK and r.moments == {"mean": "2", "variance": "9", "second": "13"}


def test_undecidable_constraints_are_premises_not_refusals(engine) -> None:
    r = engine.statistical_model(model("bernoulli", {"p": "p"}))
    assert r.status is Status.OK
    assert any(c.latex == "0 < p < 1" and not c.established for c in r.conditions)


@pytest.mark.parametrize(
    ("family", "parameters", "status", "reason"),
    [
        (
            "normal",
            {"mu": "mu", "sigma": "-1"},
            Status.CONDITIONS_NOT_SATISFIED,
            "parameter_invalid",
        ),
        (
            "normal",
            {"mu": "mu", "sigma": "0"},
            Status.CONDITIONS_NOT_SATISFIED,
            "parameter_invalid",
        ),
        ("bernoulli", {"p": "3/2"}, Status.CONDITIONS_NOT_SATISFIED, "parameter_invalid"),
        ("bernoulli", {"p": "1"}, Status.CONDITIONS_NOT_SATISFIED, "parameter_invalid"),
        (
            "binomial",
            {"n": "3/2", "p": "1/2"},
            Status.CONDITIONS_NOT_SATISFIED,
            "parameter_invalid",
        ),
        ("poisson", {"lam": "-2"}, Status.CONDITIONS_NOT_SATISFIED, "parameter_invalid"),
        ("poisson", {}, Status.CONDITIONS_NOT_SATISFIED, "parameter_missing"),
        ("normal", {"mu": "mu", "sigma": "sigma", "tau": "1"}, Status.PARSE_ERROR, "parse_error"),
        ("weibull", {"k": "k"}, Status.UNSUPPORTED, "unsupported_family"),
        ("cauchy", {}, Status.CONDITIONS_NOT_SATISFIED, "convergence_not_established"),
    ],
)
def test_refusals_carry_a_status_and_an_enumerated_reason(
    engine, family, parameters, status, reason
) -> None:
    r = engine.statistical_model(model(family, parameters))
    assert (r.status, r.reason) == (status, reason)
    assert not r.moments


@pytest.mark.parametrize(
    ("family", "parameters", "key", "wrong", "right"),
    [
        ("gamma", {"alpha": "alpha", "beta": "beta"}, "mean", "alpha/beta**2", "alpha/beta"),
        ("binomial", {"n": "n", "p": "p"}, "variance", "n*p", "n*p*(1 - p)"),
        ("poisson", {"lam": "lam"}, "second", "lam**2", "lam**2 + lam"),
        (
            "beta",
            {"alpha": "alpha", "beta": "beta"},
            "mean",
            "alpha/(alpha + beta) + 1/1000000",
            "alpha/(alpha + beta)",
        ),
        ("normal", {"mu": "mu", "sigma": "sigma"}, "variance", "sigma", "sigma**2"),
        ("normal", {"mu": "mu", "sigma": "sigma"}, "second", "sigma**2", "mu**2 + sigma**2"),
    ],
)
def test_wrong_moments_are_rejected_and_right_ones_accepted(
    engine, family, parameters, key, wrong, right
) -> None:
    for text, expected in [(wrong, Status.NOT_VERIFIED), (right, Status.OK)]:
        r = engine.statistical_model(model(family, parameters, candidate={f"moment:{key}": text}))
        assert r.status is expected, text


def test_a_wrong_density_is_caught_by_the_normalisation_check(monkeypatch) -> None:
    broken = dataclasses.replace(
        FAMILIES["normal"],
        density=lambda s, t: sp.exp(-((t - s["mu"]) ** 2) / (2 * s["sigma"] ** 2)) / s["sigma"],
    )
    monkeypatch.setitem(FAMILIES, "normal", broken)
    r = stat_model(model("normal", {"mu": "mu", "sigma": "sigma"}), Limits(timeout_seconds=30))
    assert r.status is Status.NOT_VERIFIED and r.reason == "verification_failed"
    assert not r.moments


def test_x_is_the_density_variable() -> None:
    assert x.is_real
