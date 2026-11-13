"""The ``statistical_model`` operation: moments, likelihoods, estimators, information and posteriors.

Everything is derived from the density of a family (see `families`) and checked by a route that does
not produce it. Independence is never assumed; the parameter space is part of the model and is decided
by `domain` (a provable violation refuses the model, an undecided constraint is a premise of the result).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import mpmath as mp
import sympy as sp

from ..distributions.models import Condition
from ..errors import LimitError, MinervaMathError, ParseError, RangeNotProvable, UnsupportedInput
from ..latexprint import to_latex, to_text
from ..models import ExpressionInput, Limits, Status
from ..ops import parse_expression
from . import moments as mom
from .domain import DESCRIPTIONS, decide
from .families import FAMILIES, Family, Param
from .models import (
    StatCheck,
    StatModelInput,
    StatModelResult,
    StatReason,
    StatStep,
)
from .numeric import DPS, REL_TOL, Numeric, sample_parameters

MOMENT_KEYS = ("mean", "variance", "second")


class Refusal(MinervaMathError):
    """A refusal with a status and an enumerated reason."""

    def __init__(self, status: Status, reason: StatReason, message: str) -> None:
        super().__init__(message)
        self.status, self.reason = status, reason


@dataclass
class Reading:
    family: Family
    params: tuple[Param, ...]
    exprs: dict[
        str, sp.Expr
    ]  # family parameter name -> the user's expression (or the unknown itself)
    conditions: list[Condition]


def _fail(
    data: StatModelInput, status: Status, reason: StatReason | None, message: str, started: float
) -> StatModelResult:
    return StatModelResult(
        status=status,
        action=data.action,
        reason=reason,
        message=message,
        duration_ms=(time.monotonic() - started) * 1000,
    )


def read_model(data: StatModelInput, limits: Limits) -> Reading:
    family = FAMILIES.get(data.family)
    if family is None:
        raise Refusal(
            Status.UNSUPPORTED,
            "unsupported_family",
            f"{data.family!r} is not a family Minerva knows ({', '.join(sorted(FAMILIES))})",
        )
    try:
        params = family.params(data.parametrization)
    except KeyError as exc:
        raise Refusal(
            Status.PARSE_ERROR,
            "parse_error",
            f"{family.title} has the parametrisations {', '.join(family.parametrization)}",
        ) from exc
    names = {p.name for p in params}
    unknown = set(data.parameters) - names
    if unknown:
        raise Refusal(
            Status.PARSE_ERROR,
            "parse_error",
            f"{sorted(unknown)} are not parameters of this {family.title} model ({', '.join(sorted(names))})",
        )
    if set(data.estimate) - names:
        raise Refusal(
            Status.PARSE_ERROR,
            "parse_error",
            f"cannot estimate {sorted(set(data.estimate) - names)}",
        )
    exprs: dict[str, sp.Expr] = {}
    conditions: list[Condition] = []
    for p in params:
        if p.name in data.estimate:
            exprs[p.name] = family.symbols[p.name]
            continue
        text = data.parameters.get(p.name, p.default)
        if text is None:
            raise Refusal(
                Status.CONDITIONS_NOT_SATISFIED,
                "parameter_missing",
                f"give a value for {p.name} or list it to be estimated",
            )
        exprs[p.name] = parse_expression(
            ExpressionInput(input=text, format="text", assumptions=data.assumptions), limits
        )
    for p in params:
        verdict = decide(p.constraint, exprs[p.name])
        shown = DESCRIPTIONS[p.constraint].format(name=to_latex(exprs[p.name]))
        if verdict is False:
            raise Refusal(
                Status.CONDITIONS_NOT_SATISFIED,
                "parameter_invalid",
                f"{p.name} must satisfy: {shown}",
            )
        conditions.append(
            Condition(
                latex=shown,
                established=verdict is True,
                how="from the declared assumptions"
                if verdict is True
                else "part of the parameter space of the model; the declarations cannot express it",
            )
        )
    return Reading(family, params, exprs, conditions)


def stat_model(data: StatModelInput, limits: Limits) -> StatModelResult:
    started = time.monotonic()
    deadline = started + 0.8 * limits.timeout_seconds
    try:
        reading = read_model(data, limits)
        if data.action == "moments":
            result = _moments(data, reading, deadline)
        else:
            raise UnsupportedInput(f"the action {data.action!r} is not available yet")
    except Refusal as exc:
        return _fail(data, exc.status, exc.reason, str(exc), started)
    except RangeNotProvable as exc:
        return _fail(
            data, Status.CONDITIONS_NOT_SATISFIED, "assumptions_insufficient", str(exc), started
        )
    except LimitError as exc:
        return _fail(data, Status.RESOURCE_LIMIT, "resource_limit", str(exc), started)
    except UnsupportedInput as exc:
        return _fail(data, Status.UNSUPPORTED, "unsupported_form", str(exc), started)
    except ParseError as exc:
        return _fail(data, Status.PARSE_ERROR, "parse_error", str(exc), started)
    except MinervaMathError as exc:
        return _fail(data, Status.PARSE_ERROR, "parse_error", str(exc), started)
    result.duration_ms = (time.monotonic() - started) * 1000
    return result


# ---------------------------------------------------------------------------------- moments


def _numerics(
    data: StatModelInput, reading: Reading
) -> list[tuple[dict[sp.Symbol, sp.Rational], dict[str, sp.Expr], Numeric]]:
    rng = random.Random(20260101)  # noqa: S311 - deterministic sampling, not cryptographic
    samples = sample_parameters(reading.exprs, reading.params, rng, count=5)
    if len(samples) < 3:
        raise Refusal(
            Status.CONDITIONS_NOT_SATISFIED,
            "assumptions_insufficient",
            "no parameter values satisfy the constraints under the declared assumptions",
        )
    return [(pt, vals, Numeric(reading.family, data.parametrization, vals)) for pt, vals in samples]


def _numeric_moments(num: Numeric) -> dict[str, tuple[mp.mpf, mp.mpf]] | None:
    with mp.workdps(DPS):
        (m1, e1), (m2, e2) = num.moment(1), num.moment(2)
        if max(e1, e2) > REL_TOL * max(1, abs(m1), abs(m2)):
            return None  # the quadrature itself is not trustworthy at this point
        return {
            "mean": (m1, e1),
            "second": (m2, e2),
            "variance": (m2 - m1**2, e1 + e2 + 2 * abs(m1) * e1),
        }


def _agree(a: sp.Expr, b: mp.mpf) -> bool:
    with mp.workdps(DPS):
        value = mp.mpf(str(sp.N(a, DPS + 5)))
        return bool(abs(value - b) <= mp.mpf(10) ** -7 * max(1, abs(b)))


def _moments(data: StatModelInput, reading: Reading, deadline: float) -> StatModelResult:
    family = reading.family
    numerics = _numerics(data, reading)
    for _, _, num in numerics:
        # established first: quadrature "agrees" with a divergent integral (a Cauchy mean of 0)
        if not num.absolute_moment_exists(2):
            raise Refusal(
                Status.CONDITIONS_NOT_SATISFIED,
                "convergence_not_established",
                f"the mean and variance of the {family.title} family do not exist (the integral of "
                "|x| f(x) does not converge); a principal value is not the mean",
            )
    checks: list[StatCheck] = []
    with mp.workdps(DPS):
        total = [num.moment(0) for _, _, num in numerics]
    agreed = sum(abs(v - 1) <= mp.mpf(10) ** -7 and e <= mp.mpf(10) ** -6 for v, e in total)
    checks.append(
        StatCheck(
            kind="normalisation",
            method="quadrature or summation of the density at sampled parameter values",
            points_checked=len(total),
            points_agreed=int(agreed),
            passed=agreed == len(total),
        )
    )
    if not checks[0].passed:
        raise Refusal(
            Status.NOT_VERIFIED, "verification_failed", "the density does not integrate to 1"
        )

    wanted = {k.removeprefix("moment:") for k in (data.candidate or {})}
    if data.candidate is not None:
        found = _candidate_moments(data, reading, wanted)
    else:
        found = _derive_moments(reading, data.parametrization, deadline)
    pairs = [(point, num, _numeric_moments(num)) for point, _, num in numerics]
    usable = [(point, num, ref) for point, num, ref in pairs if ref is not None]
    steps: list[StatStep] = []
    for name, expr in found.items():
        agreed = sum(_agree(expr.subs(point), ref[name][0]) for point, _, ref in usable)
        checks.append(
            StatCheck(
                kind="moment_quadrature",
                method=f"{name}: quadrature or summation of the density at sampled parameter values",
                points_checked=len(usable),
                points_agreed=int(agreed),
                passed=len(usable) >= 3 and agreed == len(usable),
            )
        )
    if not all(c.passed for c in checks):
        return StatModelResult(
            status=Status.NOT_VERIFIED,
            action="moments",
            reason="verification_failed",
            checks=checks,
            conditions=reading.conditions,
            message="the moments did not agree with quadrature or summation of the density; not shown",
        )
    texts = {k: to_text(v) for k, v in found.items()}
    latex = {k: to_latex(v) for k, v in found.items()}
    for name in found:
        steps.append(
            StatStep(
                title={
                    "mean": "Mean E[X]",
                    "second": "Second moment E[X^2]",
                    "variance": "Variance",
                }[name],
                after_latex=latex[name],
                after_text=texts[name],
                evidence="identity",
            )
        )
    return StatModelResult(
        status=Status.OK,
        action="moments",
        steps=steps,
        moments=texts,
        moments_latex=latex,
        checks=checks,
        conditions=reading.conditions,
        evidence="identity",
        message="each moment agrees with quadrature or summation of the density at "
        f"{len(usable)} parameter points; numerical agreement is evidence, not a proof",
    )


def _user_substitution(reading: Reading) -> dict[sp.Basic, sp.Expr]:
    return {reading.family.symbols[n]: e for n, e in reading.exprs.items()}


def _derive_moments(
    reading: Reading, parametrization: str | None, deadline: float
) -> dict[str, sp.Expr]:
    try:
        m1 = mom.raw_moment(reading.family, parametrization, 1, deadline)
        m2 = mom.raw_moment(reading.family, parametrization, 2, deadline)
        variance = sp.factor(sp.simplify(m2 - m1**2))
    except mom.NoClosedForm as exc:
        raise Refusal(
            Status.UNSUPPORTED, "no_closed_form", "SymPy found no closed form for this moment"
        ) from exc
    swap = _user_substitution(reading)
    return {
        k: v.xreplace(swap) for k, v in {"mean": m1, "variance": variance, "second": m2}.items()
    }


def _candidate_moments(
    data: StatModelInput, reading: Reading, wanted: set[str]
) -> dict[str, sp.Expr]:
    if not wanted <= set(MOMENT_KEYS):
        raise ParseError(f"the moments are {', '.join(MOMENT_KEYS)}")
    out: dict[str, sp.Expr] = {}
    for key in wanted:
        text = (data.candidate or {})[f"moment:{key}"]
        out[key] = parse_expression(
            ExpressionInput(input=text, format="text", assumptions=data.assumptions), Limits()
        )
    return out
