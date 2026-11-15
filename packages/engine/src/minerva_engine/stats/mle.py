"""Log-likelihood, score and maximum-likelihood estimators, derived and checked independently.

Derivation: the log-likelihood of independent observations is expanded, its sums over the data are
abstracted to symbols, the score is differentiated, and the score equations are solved one parameter at a
time (a triangular system; anything else is `system_not_triangular`). Nothing here is trusted:

* the **log-likelihood** is compared with a direct numerical sum of log densities;
* the **score** is compared with central differences of that direct log-likelihood;
* an **estimator** (derived or given) must make the score of the *direct* log-likelihood vanish on explicit
  data, lie in the parameter space, and have a negative definite Hessian there (second differences);
* **global** is claimed only when concavity on the whole parameter space is proved symbolically; otherwise
  the optimum is `local` (proved at the point from the Hessian, or `measured`), and the number of stationary
  points found is stated.

An estimator that lies outside the parameter space for the data (a Bernoulli sample of all zeros, a normal
sample of identical values) is `candidate_inadmissible`: the supremum is not attained, so no MLE is shown.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

import mpmath as mp
import sympy as sp

from ..distributions.models import Condition
from ..errors import ParseError
from ..latexprint import to_latex, to_text
from ..models import ExpressionInput, Limits, Status
from ..ops import parse_expression
from . import likelihood as lh
from .domain import decide
from .models import Estimate, Optimum, StatCheck, StatModelInput, StatModelResult, StatStep
from .numeric import sample_parameters
from .statmodel_types import Reading, Refusal

Point = dict[sp.Symbol, Any]
SIZES = (2, 3, 5)
DPS = 30
STEP = mp.mpf(10) ** -10
STEP2 = mp.mpf(10) ** -6
TOL = mp.mpf(10) ** -7
TOL2 = mp.mpf(10) ** -4


# ---------------------------------------------------------------------------------- direct numerics


class Direct:
    """The log-likelihood computed directly: the sum of log densities at numbers (no symbolic algebra)."""

    def __init__(self, lk: lh.Likelihood) -> None:
        self.lk = lk
        dummy = sp.Dummy("obs")
        density = lk.density.xreplace({lk.element: dummy})
        self.params = sorted(
            (s for s in density.free_symbols if isinstance(s, sp.Symbol) and s != dummy),
            key=lambda s: s.name,
        )
        self.f = sp.lambdify([dummy, *self.params], density, "mpmath")

    def loglik(
        self, values: list[sp.Rational], point: dict[sp.Symbol, sp.Rational | mp.mpf]
    ) -> mp.mpf:
        args = [
            mp.mpf(str(sp.N(point[s], DPS))) if not isinstance(point[s], mp.mpf) else point[s]
            for s in self.params
        ]
        with mp.workdps(DPS):
            return mp.fsum(mp.log(self.f(mp.mpf(str(sp.N(v, DPS))), *args)) for v in values)


def _slope(direct: Direct, values: list[sp.Rational], point: Point, symbol: sp.Symbol) -> mp.mpf:
    base = mp.mpf(str(sp.N(point[symbol], DPS)))
    up, down = {**point, symbol: base + STEP}, {**point, symbol: base - STEP}
    return (direct.loglik(values, up) - direct.loglik(values, down)) / (2 * STEP)


def _loglik_at(
    direct: Direct, values: list[sp.Rational], point: Point, shifts: dict[sp.Symbol, mp.mpf]
) -> mp.mpf:
    return direct.loglik(values, {**point, **shifts})


def _hessian(
    direct: Direct, values: list[sp.Rational], point: Point, theta: list[sp.Symbol]
) -> mp.matrix:
    base = {s: mp.mpf(str(sp.N(point[s], DPS))) for s in theta}
    k = len(theta)
    out = mp.zeros(k, k)
    centre = _loglik_at(direct, values, point, base)
    for a, sa in enumerate(theta):
        for b, sb in enumerate(theta):
            if a == b:
                up = _loglik_at(direct, values, point, {**base, sa: base[sa] + STEP2})
                down = _loglik_at(direct, values, point, {**base, sa: base[sa] - STEP2})
                out[a, a] = (up - 2 * centre + down) / STEP2**2
            else:
                pp = _loglik_at(
                    direct, values, point, {**base, sa: base[sa] + STEP2, sb: base[sb] + STEP2}
                )
                pm = _loglik_at(
                    direct, values, point, {**base, sa: base[sa] + STEP2, sb: base[sb] - STEP2}
                )
                mpp = _loglik_at(
                    direct, values, point, {**base, sa: base[sa] - STEP2, sb: base[sb] + STEP2}
                )
                mm = _loglik_at(
                    direct, values, point, {**base, sa: base[sa] - STEP2, sb: base[sb] - STEP2}
                )
                out[a, b] = (pp - pm - mpp + mm) / (4 * STEP2**2)
    return out


def _negative_definite(h: mp.matrix) -> bool:
    for k in range(1, h.rows + 1):
        minor = mp.det(h[:k, :k])
        if (-1) ** k * minor <= 0:
            return False
    return True


# ---------------------------------------------------------------------------------- preparation


class Prepared:
    def __init__(self, data: StatModelInput, reading: Reading, limits: Limits) -> None:
        self.data = data
        self.reading = reading
        self.lk = lh.build(data, reading, limits)
        self.direct = Direct(self.lk)
        self.ab = lh.abstract_sums(self.lk.loglik, self.lk.obs)
        self.theta = self.lk.theta
        self.score_abs = [sp.diff(self.ab.expr, t) for t in self.theta]

    def restore(self, expr: sp.Expr) -> sp.Expr:
        return self.ab.restore(expr)


def _instances(
    prep: Prepared,
    rng: random.Random,
    accept: Callable[[list[sp.Rational]], bool] | None = None,
    want: int = 4,
) -> list[list[sp.Rational]]:
    """Data in the support of the family at several sizes, optionally only those that `accept`."""
    obs = prep.lk.obs
    if obs.values is not None:
        return [obs.values]
    if obs.n is None:
        return [lh.draw_values(prep.reading.family, 1, rng) for _ in range(want)]
    out: list[list[sp.Rational]] = []
    for size in SIZES * 2:
        for _ in range(40):
            values = lh.draw_values(prep.reading.family, size, rng)
            if accept is None or accept(values):
                out.append(values)
                break
        if len(out) >= want:
            break
    return out


def _points(
    prep: Prepared, rng: random.Random, count: int
) -> list[tuple[Point, dict[str, sp.Expr]]]:
    """Parameter points; with a single concrete data set there are more of them."""
    if prep.lk.obs.values is not None:
        count = max(count, 4)
    return sample_parameters(prep.reading.exprs, prep.reading.params, rng, count)


# ---------------------------------------------------------------------------------- checks


def check_loglik(prep: Prepared, presented: sp.Expr, deadline: float) -> StatCheck:
    rng = random.Random(20260111)  # noqa: S311 - deterministic sampling, not cryptographic
    checked = agreed = 0
    for values in _instances(prep, rng):
        for point, _ in _points(prep, rng, 2):
            if time.monotonic() > deadline:
                break
            expected = prep.direct.loglik(values, point)
            shown = lh.numeric(lh.at_instance(presented, prep.lk.obs, values), point)
            checked += 1
            agreed += abs(shown - expected) <= TOL * max(1, abs(expected))
    return StatCheck(
        kind="score",
        method="the expanded log-likelihood against the direct sum of log densities, on explicit data",
        points_checked=checked,
        points_agreed=agreed,
        passed=checked >= 3 and agreed == checked,
    )


def check_score(prep: Prepared, presented: list[sp.Expr], deadline: float) -> StatCheck:
    rng = random.Random(20260112)  # noqa: S311 - deterministic sampling, not cryptographic
    checked = agreed = 0
    for values in _instances(prep, rng):
        for point, _ in _points(prep, rng, 2):
            if time.monotonic() > deadline:
                break
            ok = True
            for symbol, expr in zip(prep.theta, presented, strict=True):
                slope = _slope(prep.direct, values, point, symbol)
                shown = lh.numeric(lh.at_instance(expr, prep.lk.obs, values), point)
                ok = ok and abs(shown - slope) <= TOL * max(1, abs(slope))
            checked += 1
            agreed += ok
    return StatCheck(
        kind="score",
        method="the score against central differences of the direct log-likelihood, on explicit data",
        points_checked=checked,
        points_agreed=agreed,
        passed=checked >= 3 and agreed == checked,
    )


def _estimate_point(
    prep: Prepared, estimates: dict[sp.Symbol, sp.Expr], values: list[sp.Rational], base: Point
) -> Point | None:
    """The point with the estimators evaluated on the data; None if one lies outside the parameter space."""
    point = dict(base)
    for symbol in prep.theta:
        try:
            value = lh.numeric(lh.at_instance(estimates[symbol], prep.lk.obs, values), base)
        except (ValueError, ZeroDivisionError, TypeError):
            return None
        point[symbol] = value
    for p in prep.reading.params:
        symbol = prep.reading.family.symbols[p.name]
        if (
            symbol in prep.theta
            and decide(p.constraint, sp.nsimplify(sp.Float(str(point[symbol]), DPS))) is not True
        ):
            return None
    return point


def check_estimators(
    prep: Prepared, estimates: dict[sp.Symbol, sp.Expr], deadline: float
) -> tuple[StatCheck, StatCheck, int]:
    """Stationarity on the direct log-likelihood, and a negative definite Hessian (measured)."""
    rng = random.Random(20260113)  # noqa: S311 - deterministic sampling, not cryptographic

    def admissible(values: list[sp.Rational]) -> bool:
        base = {s: p for s, p in _points(prep, rng, 1)[0][0].items()}
        return _estimate_point(prep, estimates, values, base) is not None

    stationary = StatCheck(
        kind="score",
        method="the direct log-likelihood has zero slope at the estimators, on explicit data",
    )
    curvature = StatCheck(
        kind="second_order",
        method="second differences of the direct log-likelihood: negative definite at the estimators",
    )
    admissible_count = 0
    for values in _instances(prep, rng, accept=admissible):
        if time.monotonic() > deadline:
            break
        for point, _ in _points(prep, rng, 1):
            at_hat = _estimate_point(prep, estimates, values, point)
            if at_hat is None:
                continue
            admissible_count += 1
            slopes = [_slope(prep.direct, values, at_hat, s) for s in prep.theta]
            scale = max(
                1, len(values), *[abs(mp.mpf(str(sp.N(at_hat[s], DPS)))) for s in prep.theta]
            )
            stationary.points_checked += 1
            stationary.points_agreed += all(abs(g) <= TOL * scale for g in slopes)
            curvature.points_checked += 1
            curvature.points_agreed += _negative_definite(
                _hessian(prep.direct, values, at_hat, prep.theta)
            )
    for c in (stationary, curvature):
        c.passed = c.points_checked >= 3 and c.points_agreed == c.points_checked
    return stationary, curvature, admissible_count


# ---------------------------------------------------------------------------------- solving


def solve_triangular(prep: Prepared) -> dict[sp.Symbol, sp.Expr]:
    """Solve the score equations one parameter at a time; every step must give exactly one admissible root."""
    equations = dict(zip(prep.theta, prep.score_abs, strict=True))
    constraint = {prep.reading.family.symbols[p.name]: p.constraint for p in prep.reading.params}
    solved: dict[sp.Symbol, sp.Expr] = {}
    pending = list(prep.theta)
    while pending:
        progress = False
        for symbol in list(pending):
            equation = sp.simplify(equations[symbol].subs(solved))
            roots = [
                r
                for r in sp.solve(equation, symbol)
                if not (r.free_symbols & (set(pending) - {symbol}))
            ]
            if len(roots) > 1:  # keep roots that are not provably outside the parameter space
                roots = [r for r in roots if decide(constraint[symbol], r) is not False]
            if len(roots) > 1:
                raise Refusal(
                    Status.UNSUPPORTED,
                    "no_closed_form",
                    "the score equation has several admissible roots",
                )
            if len(roots) == 1:
                solved[symbol] = sp.simplify(roots[0])
                for other in list(solved):
                    solved[other] = sp.simplify(solved[other].subs(symbol, solved[symbol]))
                pending.remove(symbol)
                progress = True
        if not progress:
            raise Refusal(
                Status.UNSUPPORTED,
                "system_not_triangular",
                "the score equations cannot be solved one parameter at a time; Minerva solves only "
                "triangular systems",
            )
    return solved


def _centered(
    prep: Prepared, symbol: sp.Symbol, estimate: sp.Expr, solved: dict[sp.Symbol, sp.Expr]
) -> sp.Expr | None:
    """Present a variance-like estimate as sum (x_i - c)^2 / n when it is exactly that quantity."""
    obs = prep.lk.obs
    if obs.n is None or obs.values is not None or symbol.name not in {"v", "sigma"}:
        return None
    mu = prep.reading.family.symbols["mu"]
    if mu in solved:
        centre = prep.restore(solved[mu])
    elif mu not in prep.theta:
        centre = prep.reading.exprs["mu"]
    else:
        return None
    if obs.index is None:
        return None
    inner = sp.Symbol("j" if obs.index.name != "j" else "k", integer=True, positive=True)
    centre = centre.replace(
        lambda e: isinstance(e, sp.Sum),
        lambda e: sp.Sum(e.function.subs(obs.index, inner), (inner, *e.limits[0][1:])),
    )
    candidate = sp.Sum((obs.x[obs.index] - centre) ** 2, (obs.index, 1, obs.n)) / obs.n
    if symbol.name == "sigma":
        candidate = sp.sqrt(candidate)
    try:
        difference = sp.simplify(
            lh.abstract_sums(candidate, obs).expr - lh.abstract_sums(estimate, obs).expr
        )
    except Exception:
        return None
    return candidate if difference == 0 else None


# ---------------------------------------------------------------------------------- second order


def _facts(prep: Prepared, expr: sp.Expr) -> sp.Expr:
    """Facts used in a sign proof, all consequences of the family: observations in [0, 1] give
    n - sum(x) >= 0, and a parameter in (0, 1) has a positive complement q = 1 - p."""
    if prep.reading.family.unit_interval_data and prep.lk.obs.n is not None:
        s1 = next((s for s in prep.ab.atoms if s.name == "S1"), None)
        if s1 is not None:
            expr = expr.subs(prep.lk.obs.n, s1 + sp.Symbol("F", nonnegative=True))
    for param in prep.reading.params:
        symbol = prep.reading.family.symbols[param.name]
        if param.constraint == "unit_open" and symbol in prep.theta:
            q = sp.Symbol(f"q_{param.name}", positive=True)
            expr = expr.subs(1 - symbol, q).subs(symbol - 1, -q)
    return sp.expand(expr)


def second_order(
    prep: Prepared, solved: dict[sp.Symbol, sp.Expr], measured_ok: bool
) -> tuple[Optimum | None, str | None, str]:
    theta = prep.theta
    hessian = sp.Matrix([[sp.diff(prep.ab.expr, a, b) for b in theta] for a in theta])
    if len(theta) == 1:
        h = _facts(prep, hessian[0, 0])
        if h.is_extended_nonpositive:
            return "global", "proved", "the log-likelihood is concave on the whole parameter space"
    at_hat = hessian.subs(solved)
    proof = _definite_at_point(prep, at_hat, solved)
    if proof is True:
        return (
            "local",
            "proved",
            "the Hessian is negative definite at the estimators (given the admissibility condition)",
        )
    if proof is False:
        return (
            "stationary_only",
            None,
            "the Hessian is not negative definite at the stationary point",
        )
    if measured_ok:
        return (
            "local",
            "measured",
            "negative definite Hessian, measured on explicit data (not proved)",
        )
    return "stationary_only", None, "a stationary point; its type could not be established"


def _definite_at_point(
    prep: Prepared, h: sp.Matrix, solved: dict[sp.Symbol, sp.Expr]
) -> bool | None:
    expr = h.applyfunc(sp.simplify)
    # name each positive estimate: the admissibility condition is exactly that it is positive
    for symbol, estimate in solved.items():
        if symbol.is_positive:
            atoms = [a for a in prep.ab.atoms if estimate.has(a)]
            if not atoms:
                continue
            w = sp.Symbol(f"W_{symbol.name}", positive=True)
            for atom in atoms:
                try:
                    (root,) = sp.solve(estimate - w, atom)
                except (ValueError, TypeError):
                    continue
                expr = expr.subs(atom, root).applyfunc(sp.simplify)
                break
    verdicts = []
    for k in range(1, expr.rows + 1):
        minor = sp.simplify(expr[:k, :k].det())
        sign = minor.is_positive if k % 2 == 0 else minor.is_negative
        verdicts.append(sign)
    if all(v is True for v in verdicts):
        return True
    if any(v is False for v in verdicts):
        return None  # sympy could not decide; the measured check decides
    return None


# ---------------------------------------------------------------------------------- actions


def _expressions(prep: Prepared) -> tuple[sp.Expr, list[sp.Expr]]:
    return prep.restore(prep.ab.expr), [prep.restore(s) for s in prep.score_abs]


def _require_regular(reading: Reading) -> None:
    if not reading.family.regular:
        raise Refusal(
            Status.UNSUPPORTED,
            "regularity_not_satisfied",
            f"the support of the {reading.family.title} family depends on its parameter: the likelihood "
            "is not differentiable there and differentiation under the integral sign is not allowed, "
            "so there is no score",
        )


def run_log_likelihood(
    data: StatModelInput, reading: Reading, limits: Limits, deadline: float
) -> StatModelResult:
    prep = Prepared(data, reading, limits)
    loglik, _ = _expressions(prep)
    check = check_loglik(prep, loglik, deadline)
    if not check.passed:
        raise Refusal(
            Status.NOT_VERIFIED,
            "verification_failed",
            "the log-likelihood did not agree with the direct sum",
        )
    steps = [
        StatStep(
            title="Likelihood",
            after_latex=to_latex(prep.lk.likelihood),
            evidence="conditional" if prep.lk.premises else "definition",
        ),
        StatStep(
            title="Log-likelihood",
            after_latex=to_latex(loglik),
            after_text=to_text(loglik),
            evidence="identity",
        ),
    ]
    return StatModelResult(
        status=Status.OK,
        action="log_likelihood",
        steps=steps,
        expressions={"loglik": to_text(loglik)},
        expressions_latex={"loglik": to_latex(loglik)},
        checks=[check],
        premises=prep.lk.premises,
        conditions=reading.conditions,
        evidence="conditional" if prep.lk.premises else "identity",
        message="the log-likelihood agrees with the direct sum of log densities on explicit data",
    )


def run_score(
    data: StatModelInput, reading: Reading, limits: Limits, deadline: float
) -> StatModelResult:
    if not data.estimate:
        raise ParseError("list the parameters to differentiate with respect to")
    _require_regular(reading)
    prep = Prepared(data, reading, limits)
    loglik, scores = _expressions(prep)
    checks = [check_loglik(prep, loglik, deadline), check_score(prep, scores, deadline)]
    if not all(c.passed for c in checks):
        raise Refusal(
            Status.NOT_VERIFIED,
            "verification_failed",
            "the score did not agree with numerical differentiation",
        )
    names = [t.name for t in prep.theta]
    steps = [
        StatStep(
            title="Log-likelihood",
            after_latex=to_latex(loglik),
            after_text=to_text(loglik),
            evidence="identity",
        ),
        *[
            StatStep(
                title=f"Score for {n}",
                after_latex=to_latex(s),
                after_text=to_text(s),
                evidence="identity",
            )
            for n, s in zip(names, scores, strict=True)
        ],
    ]
    return StatModelResult(
        status=Status.OK,
        action="score",
        steps=steps,
        expressions={
            "loglik": to_text(loglik),
            **{f"score:{n}": to_text(s) for n, s in zip(names, scores, strict=True)},
        },
        expressions_latex={
            "loglik": to_latex(loglik),
            **{f"score:{n}": to_latex(s) for n, s in zip(names, scores, strict=True)},
        },
        checks=checks,
        premises=prep.lk.premises,
        conditions=reading.conditions,
        evidence="conditional" if prep.lk.premises else "identity",
        message="the score agrees with central differences of the direct log-likelihood on explicit data",
    )


def run_mle(
    data: StatModelInput, reading: Reading, limits: Limits, deadline: float
) -> StatModelResult:
    if not data.estimate:
        raise ParseError("list the parameters to estimate")
    _require_regular(reading)
    prep = Prepared(data, reading, limits)
    loglik, scores = _expressions(prep)
    base_checks = [check_loglik(prep, loglik, deadline), check_score(prep, scores, deadline)]
    if not all(c.passed for c in base_checks):
        raise Refusal(
            Status.NOT_VERIFIED,
            "verification_failed",
            "the score did not agree with numerical differentiation",
        )

    if data.candidate is not None:
        solved = _read_candidate(data, reading, prep, limits)
        derived = False
    else:
        solved = solve_triangular(prep)
        derived = True
    estimates = {t: prep.restore(solved[t]) for t in prep.theta}

    stationary, curvature, admissible = check_estimators(prep, estimates, deadline)
    obs = prep.lk.obs
    if obs.values is not None:
        _check_concrete(prep, estimates)
    if admissible < 3 and obs.values is None:
        raise Refusal(
            Status.CONDITIONS_NOT_SATISFIED,
            "candidate_inadmissible",
            "the estimators lie outside the parameter space for the data tried",
        )
    if not stationary.passed:
        return StatModelResult(
            status=Status.NOT_VERIFIED,
            action="mle",
            reason="verification_failed",
            checks=[*base_checks, stationary],
            premises=prep.lk.premises,
            message="the estimators do not make the score of the log-likelihood vanish; not shown",
        )
    if not curvature.passed:
        return StatModelResult(
            status=Status.NOT_VERIFIED,
            action="mle",
            reason="verification_failed",
            checks=[*base_checks, stationary, curvature],
            premises=prep.lk.premises,
            message="a stationary point of the log-likelihood that is not a maximum; not shown",
        )
    optimum, evidence, why = second_order(prep, solved, curvature.passed)
    shown = dict(estimates)
    if derived:
        for symbol in prep.theta:
            centred = _centered(prep, symbol, estimates[symbol], solved)
            if centred is not None:
                shown[symbol] = centred
    if obs.values is not None:  # concrete data: the estimates are numbers (exact, and admissible)
        for symbol in prep.theta:
            shown[symbol] = sp.nsimplify(lh.at_instance(estimates[symbol], obs, obs.values))
    out: list[Estimate] = []
    for name in data.estimate:
        symbol = reading.family.symbols[name]
        param = next(p for p in reading.params if p.name == name)
        expression = shown[symbol]
        out.append(
            Estimate(
                parameter=name,
                latex=to_latex(expression),
                text=to_text(expression),
                optimum=optimum,
                optimum_evidence=evidence,
                admissible_if=[]
                if obs.values is not None
                else _admissibility(param.constraint, expression),
            )
        )
    steps = [
        StatStep(
            title="Likelihood",
            after_latex=to_latex(prep.lk.likelihood),
            evidence="conditional" if prep.lk.premises else "definition",
        ),
        StatStep(
            title="Log-likelihood",
            after_latex=to_latex(loglik),
            after_text=to_text(loglik),
            evidence="identity",
        ),
        *[
            StatStep(
                title=f"Score for {t.name}",
                after_latex=to_latex(s),
                after_text=to_text(s),
                evidence="identity",
            )
            for t, s in zip(prep.theta, scores, strict=True)
        ],
        *[
            StatStep(
                title=f"Estimator of {e.parameter}",
                after_latex=e.latex,
                after_text=e.text,
                evidence="conditional" if prep.lk.premises else "identity",
            )
            for e in out
        ],
    ]
    return StatModelResult(
        status=Status.OK,
        action="mle",
        steps=steps,
        estimates=out,
        checks=[*base_checks, stationary, curvature],
        premises=prep.lk.premises,
        conditions=reading.conditions,
        evidence="conditional" if prep.lk.premises else "identity",
        message=f"{why}; checked on {stationary.points_checked} explicit data sets (numerical agreement is evidence, not a proof)",
    )


def _admissibility(constraint: str, expression: sp.Expr) -> list[Condition]:
    if constraint == "real":
        return []
    shown = to_latex(expression)
    if constraint == "unit_open":
        text, how = (
            rf"0 < {shown} < 1",
            "a condition on the data: the estimator must lie in the parameter space",
        )
    else:
        text, how = (
            rf"{shown} > 0",
            "a condition on the data: the estimator must lie in the parameter space",
        )
    return [Condition(latex=text, established=False, how=how)]


def _check_concrete(prep: Prepared, estimates: dict[sp.Symbol, sp.Expr]) -> None:
    """With concrete data the estimators are numbers and must lie in the parameter space."""
    values = prep.lk.obs.values or []
    for symbol in prep.theta:
        value = sp.nsimplify(lh.at_instance(estimates[symbol], prep.lk.obs, values))
        param = next(p for p in prep.reading.params if p.name == symbol.name)
        if decide(param.constraint, value) is not True:
            raise Refusal(
                Status.CONDITIONS_NOT_SATISFIED,
                "candidate_inadmissible",
                f"for these data the maximum would be at {symbol.name} = {to_text(value)}, which is "
                "outside the parameter space: the supremum of the likelihood is not attained, so there is no "
                "maximum-likelihood estimate",
            )


def _read_candidate(
    data: StatModelInput, reading: Reading, prep: Prepared, limits: Limits
) -> dict[sp.Symbol, sp.Expr]:
    out: dict[sp.Symbol, sp.Expr] = {}
    obs = prep.lk.obs
    allowed = {obs.x, *(obs.n.free_symbols if obs.n is not None else set())}
    for e in reading.exprs.values():
        allowed |= e.free_symbols
    for name in data.estimate:
        text = (data.candidate or {}).get(f"estimate:{name}")
        if text is None:
            raise ParseError(f"give an answer for every parameter to estimate (missing {name})")
        expr = parse_expression(
            ExpressionInput(input=text, format="text", assumptions=data.assumptions),
            limits,
            allow_sums=True,
        )
        used = {s for s in expr.free_symbols if isinstance(s, sp.Symbol)} | {
            b for b in expr.atoms(sp.IndexedBase)
        }
        if any(
            s in prep.theta or s.name in data.estimate for s in used if isinstance(s, sp.Symbol)
        ):
            raise Refusal(
                Status.NOT_VERIFIED,
                "verification_failed",
                "an estimator cannot depend on the unknown parameters it estimates",
            )
        out[reading.family.symbols[name]] = expr
    return out
