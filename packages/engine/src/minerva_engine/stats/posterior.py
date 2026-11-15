"""Bayesian updating: the posterior of one parameter, found by matching the kernel, never from a table.

Derivation: the log of prior times likelihood, with every term that does not depend on the parameter
dropped (and recorded), is compared with the log kernel of each family whose support equals the prior's.
The unknown parameters of that family are found by requiring the derivative with respect to the parameter
of the difference to vanish identically (a polynomial identity, solved for the unknowns). If no family
matches the posterior is `kernel_not_recognised`; nothing is invented.

Verification does not use that route: at sampled hyperparameters and data, the posterior density with the
derived parameters is compared with prior x likelihood divided by its integral (quadrature), at several
points, and it must integrate to 1. A prior is not a fact: the result holds for the prior given.
"""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable

import mpmath as mp
import sympy as sp

from ..distributions.models import Condition
from ..errors import ParseError
from ..latexprint import to_latex, to_text
from ..models import ExpressionInput, Limits, Status
from ..ops import parse_expression
from . import likelihood as lh
from .domain import DESCRIPTIONS, decide
from .families import FAMILIES, Family
from .models import PosteriorOut, StatCheck, StatModelInput, StatModelResult, StatStep
from .numeric import DPS, sample_symbol
from .statmodel_types import Reading, Refusal

TOL = mp.mpf(10) ** -8


def _prior(data: StatModelInput, limits: Limits) -> tuple[Family, str | None, dict[str, sp.Expr]]:
    spec = data.prior
    if spec is None:
        raise ParseError("give the prior: its family and hyperparameters")
    family = FAMILIES.get(spec.family)
    if family is None:
        raise Refusal(
            Status.UNSUPPORTED, "unsupported_family", f"{spec.family!r} is not a known prior family"
        )
    try:
        params = family.params(spec.parametrization)
    except KeyError as exc:
        raise ParseError(f"unknown parametrisation {spec.parametrization!r}") from exc
    names = {p.name for p in params}
    if set(spec.parameters) - names:
        raise ParseError(f"the prior has the parameters {', '.join(sorted(names))}")
    hyper: dict[str, sp.Expr] = {}
    for p in params:
        text = spec.parameters.get(p.name)
        if text is None:
            raise Refusal(
                Status.CONDITIONS_NOT_SATISFIED,
                "parameter_missing",
                f"give a value for the prior's {p.name}",
            )
        value = parse_expression(
            ExpressionInput(input=text, format="text", assumptions=data.assumptions), limits
        )
        if decide(p.constraint, value) is False:
            raise Refusal(
                Status.CONDITIONS_NOT_SATISFIED,
                "parameter_invalid",
                f"the prior's {p.name} must satisfy: {DESCRIPTIONS[p.constraint].format(name=to_text(value))}",
            )
        hyper[p.name] = value
    return family, spec.parametrization, hyper


def _drop_free(expr: sp.Expr, theta: sp.Symbol) -> tuple[sp.Expr, sp.Expr]:
    """(terms that do not depend on theta, terms that do) of a sum of logarithmic terms."""
    free, dependent = sp.expand(expr).as_independent(theta, as_Add=True)
    return free, dependent


def match_family(
    log_kernel: sp.Expr, theta: sp.Symbol, prior_family: Family, prior_key: str | None
) -> tuple[Family, str | None, dict[str, sp.Expr]] | None:
    """The family and parameters whose log kernel equals `log_kernel` up to a constant."""
    wanted = prior_family.support(prior_family.symbols)
    candidates = [prior_family, *[f for f in FAMILIES.values() if f is not prior_family]]
    for family in candidates:
        if family.discrete or not family.regular or family.support(family.symbols) != wanted:
            continue
        keys = list(family.parametrization)
        if family is prior_family and prior_key in keys:
            keys = [prior_key, *[k for k in keys if k != prior_key]]
        for key in keys:
            unknowns = {
                p.name: sp.Dummy(
                    p.name,
                    **{
                        k: v
                        for k, v in family.symbols[p.name].assumptions0.items()
                        if v is True and k in {"positive", "real"}
                    },
                )
                for p in family.parametrization[key]
            }
            density = family.density(unknowns, theta)
            _, family_kernel = _drop_free(
                sp.expand(sp.expand_log(sp.log(density), force=True)), theta
            )
            derivative = sp.together(sp.diff(log_kernel - family_kernel, theta))
            numerator = sp.numer(derivative)
            try:
                coefficients = sp.Poly(sp.expand(numerator), theta).all_coeffs()
            except sp.PolynomialError:
                continue
            solutions = sp.solve(coefficients, list(unknowns.values()), dict=True)
            for solution in solutions:
                if set(solution) != set(unknowns.values()):
                    continue
                if sp.simplify(derivative.subs(solution)) != 0:
                    continue
                result = {name: sp.simplify(solution[d]) for name, d in unknowns.items()}
                return family, key, result
    return None


def _log_kernel(
    data: StatModelInput,
    reading: Reading,
    limits: Limits,
    hyper: dict[str, sp.Expr],
    prior_family: Family,
) -> tuple[lh.Likelihood, lh.Abstraction, sp.Symbol, sp.Expr, sp.Expr, sp.Expr]:
    if len(data.estimate) != 1:
        raise ParseError("name the one parameter to update (estimate: [...])")
    lk = lh.build(data, reading, limits)
    ab = lh.abstract_sums(lk.loglik, lk.obs)
    theta = lk.theta[0]
    prior_density = prior_family.density(hyper, theta)
    log_prior = sp.expand(sp.expand_log(sp.log(prior_density), force=True))
    total = sp.expand(ab.expr + log_prior)
    dropped, kernel = _drop_free(total, theta)
    return lk, ab, theta, kernel, dropped, prior_density


@functools.cache
def _density_function(
    family_id: str, key: str | None
) -> tuple[Callable[..., mp.mpf], tuple[str, ...]]:
    family = FAMILIES[family_id]
    names = tuple(p.name for p in family.params(key))
    dummy = sp.Dummy("t")
    dummies = [sp.Dummy(n) for n in names]
    density = family.density({n: d for n, d in zip(names, dummies, strict=True)}, dummy)
    return sp.lambdify([dummy, *dummies], density, "mpmath"), names


def _posterior_density(
    family: Family, key: str | None, values: dict[str, sp.Expr], at: mp.mpf
) -> mp.mpf:
    f, names = _density_function(family.id, key)
    return f(at, *[mp.mpf(str(sp.N(values[n], DPS))) for n in names])


def run_posterior(
    data: StatModelInput, reading: Reading, limits: Limits, deadline: float
) -> StatModelResult:
    prior_family, prior_key, hyper = _prior(data, limits)
    lk, ab, theta, kernel, dropped, _ = _log_kernel(data, reading, limits, hyper, prior_family)
    found = match_family(kernel, theta, prior_family, prior_key)
    if found is None:
        raise Refusal(
            Status.UNSUPPORTED,
            "kernel_not_recognised",
            "prior times likelihood is not the kernel of any family Minerva knows; no posterior is invented "
            "(only the unnormalised kernel is known)",
        )
    family, key, derived = found
    restored = {name: lh_restore(ab, e) for name, e in derived.items()}
    if data.candidate is not None:
        restored = _candidate_parameters(data, family, key, limits)

    check, mass = _verify(
        data, reading, lk, theta, prior_family, hyper, family, key, restored, deadline
    )
    if not (check.passed and mass.passed):
        return StatModelResult(
            status=Status.NOT_VERIFIED,
            action="posterior",
            reason="verification_failed",
            checks=[check, mass],
            premises=lk.premises,
            message="the posterior does not equal prior times likelihood divided by its integral; not shown",
        )
    texts = {n: to_text(e) for n, e in restored.items()}
    latex = {n: to_latex(e) for n, e in restored.items()}
    conditions = _parameter_conditions(family, key, restored, lk)
    steps = [
        StatStep(
            title="Prior times likelihood, up to factors that do not depend on the parameter",
            after_latex=to_latex(sp.exp(kernel)),
            evidence="conditional" if lk.premises else "identity",
        ),
        StatStep(
            title=f"The kernel is that of the {family.title} family",
            after_latex=", ".join(f"{n} = {v}" for n, v in latex.items()),
            after_text=str(texts),
            evidence="conditional" if lk.premises else "identity",
        ),
    ]
    return StatModelResult(
        status=Status.OK,
        action="posterior",
        steps=steps,
        posterior=PosteriorOut(
            family=family.id,
            parameters=texts,
            parameters_latex=latex,
            dropped_factors_latex=[to_latex(sp.exp(dropped))] if dropped != 0 else [],
        ),
        checks=[check, mass],
        premises=[*lk.premises, "the prior is as stated; it is an assumption, not a fact"],
        conditions=[*reading.conditions, *conditions],
        evidence="conditional" if lk.premises else "identity",
        message="equal to prior times likelihood divided by its integral (quadrature) at sampled "
        "values; numerical agreement is evidence, not a proof",
    )


def lh_restore(ab: lh.Abstraction, expr: sp.Expr) -> sp.Expr:
    return ab.restore(expr)


def _candidate_parameters(
    data: StatModelInput, family: Family, key: str | None, limits: Limits
) -> dict[str, sp.Expr]:
    out: dict[str, sp.Expr] = {}
    for p in family.params(key):
        text = (data.candidate or {}).get(f"posterior:{p.name}")
        if text is None:
            raise ParseError(f"give every parameter of the posterior (missing {p.name})")
        out[p.name] = parse_expression(
            ExpressionInput(input=text, format="text", assumptions=data.assumptions),
            limits,
            allow_sums=True,
        )
    return out


def _parameter_conditions(
    family: Family, key: str | None, values: dict[str, sp.Expr], lk: lh.Likelihood
) -> list[Condition]:
    out: list[Condition] = []
    for p in family.params(key):
        expr = values[p.name]
        verdict = decide(p.constraint, _support_facts(expr, lk))
        shown = DESCRIPTIONS[p.constraint].format(name=to_latex(expr))
        out.append(
            Condition(
                latex=shown,
                established=verdict is True,
                how="from the declared assumptions and the support of the data"
                if verdict is True
                else "a condition on the data or the prior: it holds for data in the support",
            )
        )
    return out


def _support_facts(expr: sp.Expr, lk: lh.Likelihood) -> sp.Expr:
    """Replace n - sum(x) by a nonnegative symbol when the observations lie in [0, 1]."""
    if (
        lk.reading.family.unit_interval_data
        and lk.obs.n is not None
        and isinstance(lk.obs.n, sp.Symbol)
    ):
        total = sp.Sum(lk.obs.x[lk.obs.index], (lk.obs.index, 1, lk.obs.n))
        failures = sp.Symbol("F", nonnegative=True)
        return (
            expr.subs(lk.obs.n, total + failures).doit()
            if False
            else expr.subs(lk.obs.n, total + failures)
        )
    return expr


# --------------------------------------------------------------------------- verification


def _verify(
    data: StatModelInput,
    reading: Reading,
    lk: lh.Likelihood,
    theta: sp.Symbol,
    prior_family: Family,
    hyper: dict[str, sp.Expr],
    family: Family,
    key: str | None,
    parameters: dict[str, sp.Expr],
    deadline: float,
) -> tuple[StatCheck, StatCheck]:
    rng = random.Random(20260115)  # noqa: S311 - deterministic sampling, not cryptographic
    proportional = StatCheck(
        kind="posterior",
        method="posterior density against prior x likelihood divided by its integral (quadrature), "
        "at several values of the parameter",
    )
    mass = StatCheck(kind="posterior", method="the posterior density integrates to 1 (quadrature)")
    exprs = {
        **{f"k:{k}": v for k, v in reading.exprs.items() if k not in data.estimate},
        **{f"h:{k}": v for k, v in hyper.items()},
    }
    symbols = sorted(
        {
            s
            for e in [*exprs.values(), *parameters.values()]
            for s in e.free_symbols
            if isinstance(s, sp.Symbol)
        }
        - set(lk.theta),
        key=lambda s: s.name,
    )
    obs = lk.obs
    symbols = [s for s in symbols if s != obs.x and (obs.n is None or s != obs.n)]
    prior_symbol = prior_family.symbols
    support_lo, support_hi = prior_family.support(prior_symbol)
    for _ in range(60):
        if proportional.points_checked >= 4 or time.monotonic() > deadline:
            break
        point = {s: sample_symbol(s, rng) for s in symbols}
        values = _data(lk, reading, point, rng)
        if values is None:
            continue
        fixed, observations = values
        full = {**point, **fixed}
        try:
            outcome = _compare(
                lk,
                theta,
                prior_family,
                hyper,
                family,
                key,
                parameters,
                full,
                observations,
                (support_lo, support_hi),
                prior_symbol,
            )
        except (ValueError, ZeroDivisionError, TypeError):
            continue
        if outcome is None:
            continue
        ok_ratio, ok_mass = outcome
        proportional.points_checked += 1
        proportional.points_agreed += ok_ratio
        mass.points_checked += 1
        mass.points_agreed += ok_mass
    for c in (proportional, mass):
        c.passed = c.points_checked >= 3 and c.points_agreed == c.points_checked
    return proportional, mass


def _data(
    lk: lh.Likelihood, reading: Reading, point: dict[sp.Symbol, sp.Rational], rng: random.Random
) -> tuple[dict[sp.Symbol, sp.Rational], list[sp.Rational]] | None:
    obs = lk.obs
    family = reading.family
    # constraints of the known parameters at this point
    for p in reading.params:
        if p.name in {t.name for t in lk.theta}:
            continue
        if decide(p.constraint, sp.nsimplify(reading.exprs[p.name].subs(point))) is not True:
            return None
    if obs.values is not None:
        return {}, list(obs.values)
    if obs.n is None:  # a single observation (a count of successes in n trials, say)
        if family.id == "binomial":
            trials = int(sp.nsimplify(reading.exprs["n"].subs(point)))
            return {}, [sp.Integer(rng.randint(0, trials))]
        return {}, lh.draw_values(family, 1, rng)
    size = rng.choice([2, 3, 5])
    return {obs.n: sp.Integer(size)}, lh.draw_values(family, size, rng)


def _compare(
    lk: lh.Likelihood,
    theta: sp.Symbol,
    prior_family: Family,
    hyper: dict[str, sp.Expr],
    family: Family,
    key: str | None,
    parameters: dict[str, sp.Expr],
    point: dict[sp.Symbol, sp.Rational],
    observations: list[sp.Rational],
    support: tuple[sp.Expr, sp.Expr],
    prior_symbols: dict[str, sp.Symbol],
) -> tuple[bool, bool] | None:
    density_theta = lk.density
    prior_density = prior_family.density(hyper, theta)

    # one expression in theta alone, lambdified once: the quadrature calls it many times
    joint = prior_density
    for value in observations:
        joint = joint * density_theta.subs(lk.element, value)
    joint = joint.subs(point)
    if joint.free_symbols - {theta}:
        return None
    unnormalised = sp.lambdify(theta, joint, "mpmath")

    posterior_values: dict[str, sp.Expr] = {}
    for name, expr in parameters.items():
        evaluated = lh.at_instance(expr, lk.obs, observations).subs(point)
        if (
            decide(next(p.constraint for p in family.params(key) if p.name == name), evaluated)
            is not True
        ):
            return None
        posterior_values[name] = evaluated
    with mp.workdps(DPS):
        lo, hi = _mp(support[0]), _mp(support[1])
        breaks = (
            [lo, hi]
            if (lo != -mp.inf and hi != mp.inf)
            else ([lo, lo + 1, hi] if hi == mp.inf else [-mp.inf, 0, mp.inf])
        )
        evidence, err = mp.quad(unnormalised, breaks, error=True)
        if evidence <= 0 or err > mp.mpf(10) ** -6 * evidence:
            return None
        total, _ = mp.quad(
            lambda t: _posterior_density(family, key, posterior_values, t), breaks, error=True
        )
        centre = (lo + hi) / 2 if (lo != -mp.inf and hi != mp.inf) else mp.mpf(1)
        probes = (
            [centre / 2, centre, centre * mp.mpf("1.5")] if centre != 0 else [mp.mpf(-1), mp.mpf(1)]
        )
        ok_ratio = all(
            abs(_posterior_density(family, key, posterior_values, t) - unnormalised(t) / evidence)
            <= TOL * max(1, abs(unnormalised(t) / evidence))
            for t in probes
            if lo < t < hi
        )
        return bool(ok_ratio), bool(abs(total - 1) <= TOL)


def _mp(value: sp.Expr) -> mp.mpf:
    if value == sp.oo:
        return mp.inf
    if value == -sp.oo:
        return -mp.inf
    return mp.mpf(str(sp.N(value, DPS)))
