"""The benchmark's reference answers are checked against independent numerics.

A hand-typed reference can be wrong. Here every reference (and every `control` of a wrong-candidate
case) is checked without the engine: sums and expectations by evaluating both sides at concrete data
or over an explicit joint distribution, matrix derivatives by finite differences of the function,
moments, Fisher information and posteriors by quadrature or explicit sums, estimators by the score
and the curvature of the log-likelihood. A wrong candidate must differ from the truth.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mpmath as mp
import pytest
from minerva_knowledge.stat_benchmark import StatCase, flatten, load_cases
from minerva_knowledge.stat_oracle import (
    OracleError,
    close,
    directional_derivative,
    environments,
    equal_texts,
    evaluate,
    numeric_gradient,
    numeric_hessian,
    numeric_jacobian,
)

CASES = load_cases()
LOOSE = mp.mpf(10) ** -14  # for finite-difference derivatives at 50 digits
QUADRATURE = mp.mpf(10) ** -9  # endpoint singularities (shape < 1) limit tanh-sinh accuracy


@pytest.fixture(autouse=True)
def _fifty_digits():
    """mpmath's precision is process-wide and other modules change it: fix it for each test."""
    with mp.workdps(50):
        yield


def truth(case: StatCase) -> dict[str, str]:
    """The known-correct answer: the reference of a supported case, the control of a wrong-answer case."""
    if case.kind == "supported":
        return case.expect.all_outputs()
    assert case.control is not None
    return flatten(case.control)


def of_group(
    *groups: str, kinds: tuple[str, ...] = ("supported", "invalid_candidate")
) -> list[Any]:
    return [
        pytest.param(c, id=c.id)
        for c in CASES
        if c.group in groups and c.kind in kinds and c.kind != "unsupported"
    ]


def envs_of(case: StatCase) -> list[Any]:
    return environments(case.vars, case.matrices, case.independent)


# ------------------------------------------------------------------ every answer is readable


@pytest.mark.parametrize("case", of_group(*{c.group for c in CASES}), ids=lambda c: c.id)
def test_every_reference_is_readable_by_the_oracle(case: StatCase) -> None:
    for key, text in truth(case).items():
        usable = 0
        for env, pmf in envs_of(case):
            try:
                evaluate(text, env, pmf)
            except OracleError:
                continue
            usable += 1
        assert usable >= 3, (case.id, key, text)


@pytest.mark.parametrize("case", of_group(*{c.group for c in CASES}, kinds=("invalid_candidate",)))
def test_a_wrong_candidate_differs_from_the_truth(case: StatCase) -> None:
    assert case.candidate is not None
    wrong, right = flatten(case.candidate), truth(case)
    envs = envs_of(case)
    assert set(wrong) <= set(right), case.id
    differing = [k for k in wrong if not equal_texts(wrong[k], right[k], envs)[0]]
    assert differing, f"{case.id}: the 'wrong' candidate equals the truth"


# ------------------------------------------------------------------ sums and expectations


@pytest.mark.parametrize("case", of_group("sums", "expectation"))
def test_sum_and_expectation_references_equal_the_input(case: StatCase) -> None:
    (text,) = truth(case).values()
    envs = envs_of(case)
    action = case.request["action"]
    assert case.input is not None
    if action == "differentiate":
        variable = case.request["variable"]
        for env, pmf in envs:
            h = mp.mpf(10) ** -12
            slope = (
                evaluate(case.input, {**env, variable: env[variable] + h}, pmf)
                - evaluate(case.input, {**env, variable: env[variable] - h}, pmf)
            ) / (2 * h)
            assert close(slope, evaluate(text, env, pmf), LOOSE), case.id
    else:
        ok, why = equal_texts(text, case.input, envs)
        assert ok, f"{case.id}: {why}"


# ------------------------------------------------------------------ families: moments by quadrature


def _normal(mu: Any, sigma: Any) -> Callable[[Any], Any]:
    return lambda x: mp.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * mp.sqrt(2 * mp.pi))


MOMENT_MODELS: dict[str, Callable[[dict[str, Any]], tuple[Callable[[Any], Any], Any]]] = {
    "normal": lambda e: (_normal(e["mu"], e["sigma"]), ("cont", -mp.inf, mp.inf)),
    "bernoulli": lambda e: (lambda k: e["p"] if k == 1 else 1 - e["p"], ("disc", 0, 1)),
    "binomial": lambda e: (
        lambda k: mp.binomial(e["n"], k) * e["p"] ** k * (1 - e["p"]) ** (e["n"] - k),
        ("disc", 0, int(e["n"])),
    ),
    "poisson": lambda e: (
        lambda k: mp.exp(-e["lam"]) * e["lam"] ** k / mp.factorial(k),
        ("disc", 0, 200),
    ),
    "beta": lambda e: (
        lambda x: (
            x ** (e["alpha"] - 1) * (1 - x) ** (e["beta"] - 1) / mp.beta(e["alpha"], e["beta"])
        ),
        ("cont", 0, 1),
    ),
    "gamma": lambda e: (
        lambda x: (
            e["beta"] ** e["alpha"]
            * x ** (e["alpha"] - 1)
            * mp.exp(-e["beta"] * x)
            / mp.gamma(e["alpha"])
        ),
        ("cont", 0, mp.inf),
    ),
}


def _expect(
    g: Callable[[Any], Any], density: Callable[[Any], Any], support: tuple[Any, ...]
) -> Any:
    kind, lo, hi = support
    if kind == "disc":
        return mp.fsum(g(k) * density(k) for k in range(lo, hi + 1))
    return mp.quad(lambda x: g(x) * density(x), [lo, hi] if lo != -mp.inf else [-mp.inf, 0, mp.inf])


def _numeric_moments(density: Callable[[Any], Any], support: tuple[Any, ...]) -> dict[str, Any]:
    mean = _expect(lambda x: x, density, support)

    def squared_deviation(x: Any) -> Any:
        return (x - mean) ** 2

    return {
        "total": _expect(lambda _x: 1, density, support),
        "mean": mean,
        "variance": _expect(squared_deviation, density, support),
        "second": _expect(lambda x: x**2, density, support),
    }


@pytest.mark.parametrize(
    "case", [p for p in of_group("families") if p.values[0].request["action"] == "moments"]
)
def test_family_moments_agree_with_quadrature(case: StatCase) -> None:
    family = case.request["family"]
    for env, _ in envs_of(case):
        density, support = MOMENT_MODELS[family](env)
        numeric = _numeric_moments(density, support)
        assert close(numeric["total"], mp.mpf(1), QUADRATURE), f"{family} does not normalise"
        for key, text in truth(case).items():
            name = key.split(":", 1)[1]
            assert close(evaluate(text, env), numeric[name], QUADRATURE), (case.id, key)


# ------------------------------------------------------------------ maximum likelihood


def _log_density(case: StatCase) -> Callable[[Any, dict[str, Any]], Any]:
    family, variance = case.request["family"], case.request.get("parametrization") == "variance"
    if family == "normal":
        if variance:
            return lambda x, t: (
                -((x - t["mu"]) ** 2) / (2 * t["v"]) - mp.log(2 * mp.pi * t["v"]) / 2
            )
        return lambda x, t: (
            -((x - t["mu"]) ** 2) / (2 * t["sigma"] ** 2) - mp.log(t["sigma"] * mp.sqrt(2 * mp.pi))
        )
    if family == "bernoulli":
        return lambda x, t: x * mp.log(t["p"]) + (1 - x) * mp.log(1 - t["p"])
    if family == "poisson":
        return lambda x, t: x * mp.log(t["lam"]) - t["lam"] - mp.loggamma(x + 1)
    raise AssertionError(family)


def _loglik(case: StatCase, env: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    logf = _log_density(case)
    known = {k: evaluate(v, env) for k, v in case.request.get("parameters", {}).items()}
    data = env["x"]
    return lambda theta: mp.fsum(logf(x, {**known, **theta}) for x in data)


def _shifted(theta: dict[str, Any], shifts: dict[str, Any]) -> dict[str, Any]:
    out = dict(theta)
    for name, amount in shifts.items():
        out[name] = out[name] + amount
    return out


def _score_and_curvature(
    ll: Callable[[dict[str, Any]], Any], theta: dict[str, Any]
) -> tuple[dict[str, Any], mp.matrix]:
    names, h, k = list(theta), mp.mpf(10) ** -12, mp.mpf(10) ** -8
    score = {
        name: (ll(_shifted(theta, {name: h})) - ll(_shifted(theta, {name: -h}))) / (2 * h)
        for name in names
    }
    hess = mp.zeros(len(names), len(names))
    centre = ll(theta)
    for a, na in enumerate(names):
        for b, nb in enumerate(names):
            if na == nb:
                hess[a, b] = (
                    ll(_shifted(theta, {na: 2 * k}))
                    - 2 * centre
                    + ll(_shifted(theta, {na: -2 * k}))
                ) / (4 * k * k)
            else:
                hess[a, b] = (
                    ll(_shifted(theta, {na: k, nb: k}))
                    - ll(_shifted(theta, {na: k, nb: -k}))
                    - ll(_shifted(theta, {na: -k, nb: k}))
                    + ll(_shifted(theta, {na: -k, nb: -k}))
                ) / (4 * k * k)
    return score, hess


@pytest.mark.parametrize("case", of_group("mle"))
def test_estimators_solve_the_score_equations(case: StatCase) -> None:
    if case.kind == "supported":
        estimates = {
            k.split(":", 1)[1]: t for k, t in truth(case).items() if k.startswith("estimate:")
        }
    else:
        estimates = {k.split(":", 1)[1]: t for k, t in truth(case).items()}
    checked = 0
    for env, _ in envs_of(case):
        if case.request["family"] == "bernoulli" and not 0 < sum(env["x"]) < len(env["x"]):
            continue
        if case.request["family"] == "poisson" and sum(env["x"]) == 0:
            continue
        ll = _loglik(case, env)
        theta = {name: evaluate(text, env) for name, text in estimates.items()}
        wanted = case.request["estimate"]
        full = dict(theta)
        # a parameter that is not estimated here is held at a draw of the case (known-parameter models)
        score, hess = _score_and_curvature(ll, {k: full[k] for k in wanted if k in full})
        scale = max(1, max(abs(v) for v in score.values()), len(env["x"]))
        assert all(abs(s) < mp.mpf(10) ** -7 * scale for s in score.values()), (case.id, score)
        if case.kind == "supported":
            # a strict local maximum: the Hessian is negative definite (leading minors alternate)
            minors = [mp.det(hess[:i, :i]) for i in range(1, hess.rows + 1)]
            assert all((-1) ** (i + 1) * m > 0 for i, m in enumerate(minors)), (case.id, hess)
        checked += 1
    assert checked >= 2, case.id


@pytest.mark.parametrize("case", of_group("mle", kinds=("invalid_candidate",)))
def test_wrong_estimators_do_not_solve_the_score_equations(case: StatCase) -> None:
    assert case.candidate is not None
    wrong = {k.split(":", 1)[1]: t for k, t in flatten(case.candidate).items()}
    off = 0
    for env, _ in envs_of(case):
        ll = _loglik(case, env)
        right = {k.split(":", 1)[1]: evaluate(t, env) for k, t in truth(case).items()}
        theta = {**right}
        for name, text in wrong.items():
            theta[name] = evaluate(text, env)
        score, _ = _score_and_curvature(ll, theta)
        if any(abs(s) > mp.mpf(10) ** -6 for s in score.values()):
            off += 1
    assert off >= 3, case.id


@pytest.mark.parametrize(
    "case_id,data",
    [
        ("mle-bernoulli-all-zeros-has-no-maximiser", 0),
        ("mle-bernoulli-all-ones-has-no-maximiser", 1),
    ],
)
def test_boundary_samples_have_a_monotone_likelihood(case_id: str, data: int) -> None:
    case = next(c for c in CASES if c.id == case_id)
    n = len(case.request["data"])
    assert all(int(x) == data for x in case.request["data"])
    slopes = []
    for p in (mp.mpf(k) / 20 for k in range(1, 20)):
        d = (data * n / p) - ((1 - data) * n / (1 - p))  # derivative of the log-likelihood
        slopes.append(d)
    assert all(s > 0 for s in slopes) or all(s < 0 for s in slopes)


# ------------------------------------------------------------------ Fisher information


def _fisher_numeric(case: StatCase, env: dict[str, Any]) -> mp.matrix:
    logf = _log_density(case)
    known = {k: evaluate(v, env) for k, v in case.request.get("parameters", {}).items()}
    names = case.request["estimate"]
    theta = {
        n: evaluate({"mu": "mu", "sigma": "sigma", "v": "v", "p": "p", "lam": "lam"}[n], env)
        for n in names
    }
    h = mp.mpf(10) ** -15

    def score(x: Any, name: str) -> Any:
        up, down = {**theta, name: theta[name] + h}, {**theta, name: theta[name] - h}
        return (logf(x, {**known, **up}) - logf(x, {**known, **down})) / (2 * h)

    family = case.request["family"]
    density = lambda x: mp.exp(logf(x, {**known, **theta}))  # noqa: E731
    info = mp.zeros(len(names), len(names))
    for a, na in enumerate(names):
        for b, nb in enumerate(names):
            g = lambda x, na=na, nb=nb: score(x, na) * score(x, nb) * density(x)  # noqa: E731
            if family == "normal":
                mu = theta["mu"]
                info[a, b] = mp.quad(g, [-mp.inf, mu - 5, mu, mu + 5, mp.inf])
            elif family == "bernoulli":
                info[a, b] = g(0) + g(1)
            else:
                info[a, b] = mp.fsum(g(k) for k in range(0, 160))
    return info


@pytest.mark.parametrize("case", of_group("fisher"))
def test_fisher_references_match_the_score_variance(case: StatCase) -> None:
    expected = truth(case)
    names = case.request["estimate"]
    size = len(names)
    for env, _ in envs_of(case):
        info = _fisher_numeric(case, env)
        if case.request.get("sample_size"):
            info = info * env["n"]
        for i in range(size):
            for j in range(size):
                value = evaluate(expected[f"matrix[{i}][{j}]"], env)
                assert abs(value - info[i, j]) < mp.mpf(10) ** -6 * max(1, abs(info[i, j])), (
                    case.id,
                    i,
                    j,
                    value,
                    info[i, j],
                )


# ------------------------------------------------------------------ posteriors


def _beta_pdf(a: Any, b: Any) -> Callable[[Any], Any]:
    return lambda t: t ** (a - 1) * (1 - t) ** (b - 1) / mp.beta(a, b)


def _gamma_pdf(a: Any, rate: Any) -> Callable[[Any], Any]:
    return lambda t: rate**a * t ** (a - 1) * mp.exp(-rate * t) / mp.gamma(a)


def _kernel(
    case: StatCase, env: dict[str, Any]
) -> tuple[Callable[[Any], Any], list[Any], list[Any]]:
    """Prior times likelihood as a function of the parameter, its integration range and test points."""
    family, prior = case.request["family"], case.request["prior"]["family"]
    known = {k: evaluate(v, env) for k, v in case.request.get("parameters", {}).items()}
    hyper = {k: evaluate(v, env) for k, v in case.request["prior"]["parameters"].items()}
    data = env.get("x")
    unit = [mp.mpf("0.3"), mp.mpf("0.5"), mp.mpf("0.8")]
    if prior == "beta":
        density = _beta_pdf(hyper["alpha"], hyper["beta"])
        successes, trials = (env["k"], env["n"]) if family == "binomial" else (sum(data), len(data))
        return (
            lambda t: density(t) * t**successes * (1 - t) ** (trials - successes),
            [0, 1],
            unit,
        )
    if prior == "gamma":
        density = _gamma_pdf(hyper["alpha"], hyper["beta"])
        total, count = sum(data), len(data)
        return (
            lambda t: density(t) * t**total * mp.exp(-count * t),
            [0, mp.inf],
            [mp.mpf("0.5"), mp.mpf(1), mp.mpf(3)],
        )
    variance, centre, spread = known["v"], hyper["mu"], hyper["v"]
    return (
        lambda t: mp.exp(
            -((t - centre) ** 2) / (2 * spread)
            - mp.fsum((x - t) ** 2 for x in data) / (2 * variance)
        ),
        [-mp.inf, 0, mp.inf],
        [mp.mpf(-1), mp.mpf(0), mp.mpf(2)],
    )


@pytest.mark.parametrize("case", of_group("posterior"))
def test_posterior_references_are_proportional_to_prior_times_likelihood(case: StatCase) -> None:
    expected, prior = truth(case), case.request["prior"]["family"]
    for env, _ in envs_of(case):
        kernel, breaks, points = _kernel(case, env)
        evidence = mp.quad(kernel, breaks)
        if prior == "normal":
            post = _normal(
                evaluate(expected["posterior:mu"], env),
                mp.sqrt(evaluate(expected["posterior:v"], env)),
            )
        else:
            a, b = (
                evaluate(expected["posterior:alpha"], env),
                evaluate(expected["posterior:beta"], env),
            )
            post = _beta_pdf(a, b) if prior == "beta" else _gamma_pdf(a, b)
        for t in points:
            assert abs(kernel(t) / evidence - post(t)) < mp.mpf(10) ** -8 * max(1, post(t)), (
                case.id,
                t,
            )


# ------------------------------------------------------------------ matrix derivatives


def _decl(case: StatCase, name: str) -> dict[str, Any]:
    return next(m for m in case.matrices if m["name"] == name)


@pytest.mark.parametrize("case", of_group("matrix"))
def test_matrix_references_agree_with_finite_differences(case: StatCase) -> None:
    assert case.input is not None
    action, variable = case.request["action"], case.request["variable"]
    decl = _decl(case, variable)
    constrained = bool(decl.get("symmetric") or decl.get("positive_definite"))
    outputs = truth(case)
    checked = 0
    for env, _ in envs_of(case):
        try:
            reference = evaluate(outputs["result"], env)
            if action == "gradient":
                if constrained:
                    n = env[variable].rows
                    assert close(reference, reference.T, LOOSE), "a symmetric gradient is symmetric"
                    for i in range(n):
                        for j in range(i, n):
                            direction = mp.zeros(n, n)
                            direction[i, j] = direction[j, i] = 1
                            slope = directional_derivative(case.input, env, variable, direction)
                            inner = sum(
                                reference[a, b] * direction[a, b]
                                for a in range(n)
                                for b in range(n)
                            )
                            assert abs(slope - inner) < LOOSE * max(1, abs(slope)), (case.id, i, j)
                    alt = outputs.get("alternative:entrywise")
                    if alt:
                        entrywise = evaluate(alt, env)
                        for i in range(n):
                            for j in range(i, n):
                                direction = mp.zeros(n, n)
                                direction[i, j] = direction[j, i] = 1
                                slope = directional_derivative(case.input, env, variable, direction)
                                assert abs(slope - entrywise[i, j]) < LOOSE * max(1, abs(slope)), (
                                    case.id,
                                    i,
                                    j,
                                )
                else:
                    assert close(reference, numeric_gradient(case.input, env, variable), LOOSE), (
                        case.id
                    )
            elif action == "hessian":
                assert close(reference, numeric_hessian(case.input, env, variable), LOOSE), case.id
            else:
                assert close(reference, numeric_jacobian(case.input, env, variable), LOOSE), case.id
        except OracleError:
            continue
        checked += 1
    assert checked >= 3, case.id
