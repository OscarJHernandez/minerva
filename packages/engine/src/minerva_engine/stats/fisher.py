"""Fisher information, derived as -E[d^2 log f] and checked against the variance of the score.

For one observation, the Hessian of the log density is a polynomial in the observation for the families
here, so its expectation follows from the family's raw moments, which are themselves derived (`moments`).
The check does not use that route: the information is the covariance of the score computed by
quadrature or summation with a numerically differentiated log density, and the expected score must be 0.
A family whose support depends on its parameter is `regularity_not_satisfied`: differentiating under the
integral sign is not allowed, and the two forms of the information would disagree (even in sign).
For n independent observations the information is n times that of one (a stated premise).
"""

from __future__ import annotations

import random
import time

import mpmath as mp
import sympy as sp

from ..errors import ParseError
from ..latexprint import to_latex, to_text
from ..models import ExpressionInput, Limits, Status
from ..ops import parse_expression
from . import moments as mom
from .models import StatCheck, StatModelInput, StatModelResult, StatStep
from .numeric import DPS, Numeric, information_matrix, sample_parameters
from .statmodel_types import Reading, Refusal

TOL = mp.mpf(10) ** -9  # the numerics are good to about 1e-12; a 1e-6 error must be rejected


def _observation(reading: Reading) -> sp.Symbol:
    return sp.Symbol("x", **reading.family.observation)


def symbolic_information(
    data: StatModelInput, reading: Reading, deadline: float
) -> tuple[sp.Matrix, sp.Matrix, list[sp.Symbol]]:
    family = reading.family
    theta = [family.symbols[n] for n in data.estimate]
    obs = _observation(reading)
    density = family.density(reading.exprs, obs)
    logf = sp.expand(sp.expand_log(sp.log(density), force=True))
    hessian = sp.Matrix([[sp.diff(logf, a, b) for b in theta] for a in theta])
    degree = 0
    for entry in hessian:
        try:
            poly = sp.Poly(sp.expand(entry), obs)
        except sp.PolynomialError as exc:
            raise Refusal(
                Status.UNSUPPORTED,
                "no_closed_form",
                "the Hessian of the log density is not a polynomial in the observation, so its "
                "expectation needs moments such as E[log X] that are not derived here",
            ) from exc
        degree = max(degree, poly.degree())
    swap = {family.symbols[n]: e for n, e in reading.exprs.items()}
    raw = {0: sp.Integer(1)}
    for k in range(1, degree + 1):
        try:
            raw[k] = mom.raw_moment(family, data.parametrization, k, deadline).xreplace(swap)
        except mom.NoClosedForm as exc:
            raise Refusal(
                Status.UNSUPPORTED, "no_closed_form", "no closed form for a moment"
            ) from exc
    info = sp.zeros(len(theta), len(theta))
    for a in range(len(theta)):
        for b in range(len(theta)):
            poly = sp.Poly(sp.expand(hessian[a, b]), obs)
            expectation = sum((coeff * raw[mon[0]] for mon, coeff in poly.terms()), sp.Integer(0))
            info[a, b] = sp.factor(sp.simplify(-expectation))
    return info, hessian, theta


def run_fisher(
    data: StatModelInput, reading: Reading, limits: Limits, deadline: float
) -> StatModelResult:
    if not data.estimate:
        raise ParseError("list the parameters whose information is wanted")
    if not reading.family.regular:
        raise Refusal(
            Status.UNSUPPORTED,
            "regularity_not_satisfied",
            f"the support of the {reading.family.title} family depends on its parameter: "
            "differentiation under the integral sign is not allowed, so the Fisher information "
            "(as -E[second derivative] or as the variance of the score) is not defined",
        )
    premises: list[str] = []
    factor: sp.Expr = sp.Integer(1)
    if data.sample_size is not None:
        if data.iid is not True:
            raise Refusal(
                Status.CONDITIONS_NOT_SATISFIED,
                "independence_not_declared",
                "the information of several observations is n times that of one only if they are "
                "independent and identically distributed; state that",
            )
        premises.append("the observations are independent and identically distributed")
        n_flags = {"integer": True, "positive": True}
        factor = sp.Symbol(data.sample_size, **n_flags)
    if data.candidate is not None:
        return _check_candidate(data, reading, limits, factor, premises, deadline)

    info, _, _ = symbolic_information(data, reading, deadline)
    info = (factor * info).applyfunc(sp.simplify)
    check, centred = _verify(data, reading, info, factor, deadline)
    if not (check.passed and centred.passed):
        raise Refusal(
            Status.NOT_VERIFIED,
            "verification_failed",
            "the information did not agree with the variance of the score; not shown",
        )
    text = [[to_text(info[a, b]) for b in range(info.cols)] for a in range(info.rows)]
    latex = [[to_latex(info[a, b]) for b in range(info.cols)] for a in range(info.rows)]
    return StatModelResult(
        status=Status.OK,
        action="fisher_information",
        steps=[
            StatStep(
                title="Fisher information, as minus the expected second derivative of the log density",
                after_latex=to_latex(info),
                after_text=str(text),
                evidence="conditional" if premises else "identity",
            )
        ],
        matrix_text=text,
        matrix_latex=latex,
        checks=[check, centred],
        premises=premises,
        conditions=reading.conditions,
        evidence="conditional" if premises else "identity",
        message="equal to the variance of the score by quadrature, and the expected score is 0; "
        "numerical agreement is evidence, not a proof",
    )


def _samples(reading: Reading) -> list:  # type: ignore[type-arg]
    rng = random.Random(20260114)  # noqa: S311 - deterministic sampling, not cryptographic
    samples = sample_parameters(reading.exprs, reading.params, rng, 5)
    if len(samples) < 3:
        raise Refusal(
            Status.CONDITIONS_NOT_SATISFIED,
            "assumptions_insufficient",
            "no parameter values satisfy the constraints under the declared assumptions",
        )
    return samples


def _numeric_information(
    data: StatModelInput, reading: Reading, values: dict[str, sp.Expr], deadline: float
) -> tuple[mp.matrix, list[mp.mpf]] | None:
    names = [p.name for p in reading.params]
    positions = [names.index(n) for n in data.estimate]
    num = Numeric(reading.family, data.parametrization, values)
    if not num.absolute_moment_exists(2) and not reading.family.discrete:
        return None
    return information_matrix(num, positions)


def _verify(
    data: StatModelInput,
    reading: Reading,
    info: sp.Matrix,
    factor: sp.Expr,
    deadline: float,
) -> tuple[StatCheck, StatCheck]:
    form = StatCheck(
        kind="information",
        method="the information against the variance of the score (quadrature or summation, "
        "numerically differentiated log density)",
    )
    mean = StatCheck(
        kind="information",
        method="the expected score is 0 (regularity), by quadrature or summation",
    )
    for k, (point, values) in enumerate(_samples(reading)):
        if time.monotonic() > deadline:
            break
        reference = _numeric_information(data, reading, values, deadline)
        if reference is None:
            continue
        numeric, means = reference
        # the sample size is not a parameter: give it a value of its own at each point
        unbound = (info.free_symbols | factor.free_symbols) - set(point)
        point = {
            **point,
            **{
                s: sp.Integer(1 + (k + i) % 5)
                for i, s in enumerate(sorted(unbound, key=lambda z: z.name))
            },
        }
        scale = factor.subs(point) if factor != 1 else 1
        size = info.rows
        agreed = True
        for a in range(size):
            for b in range(size):
                shown = mp.mpf(str(sp.N(info[a, b].subs(point), DPS)))
                expected = scale * numeric[a, b]
                agreed = agreed and abs(shown - expected) <= TOL * max(1, abs(expected))
        form.points_checked += 1
        form.points_agreed += agreed
        mean.points_checked += 1
        mean.points_agreed += all(abs(m) <= mp.mpf(10) ** -5 for m in means)
    for c in (form, mean):
        c.passed = c.points_checked >= 3 and c.points_agreed == c.points_checked
    return form, mean


def _check_candidate(
    data: StatModelInput,
    reading: Reading,
    limits: Limits,
    factor: sp.Expr,
    premises: list[str],
    deadline: float,
) -> StatModelResult:
    k = len(data.estimate)
    info = sp.zeros(k, k)
    for a in range(k):
        for b in range(k):
            text = (data.candidate or {}).get(f"matrix[{a}][{b}]")
            if text is None:
                raise ParseError(f"give every entry of the {k}x{k} matrix (missing [{a}][{b}])")
            info[a, b] = parse_expression(
                ExpressionInput(input=text, format="text", assumptions=data.assumptions), limits
            )
    form, mean = _verify(data, reading, info, factor, deadline)
    if not (form.passed and mean.passed):
        return StatModelResult(
            status=Status.NOT_VERIFIED,
            action="fisher_information",
            reason="verification_failed",
            checks=[form, mean],
            premises=premises,
            message="the matrix does not equal the variance of the score; not accepted",
        )
    rows = [[to_text(info[a, b]) for b in range(k)] for a in range(k)]
    return StatModelResult(
        status=Status.OK,
        action="fisher_information",
        matrix_text=rows,
        matrix_latex=[[to_latex(info[a, b]) for b in range(k)] for a in range(k)],
        checks=[form, mean],
        premises=premises,
        conditions=reading.conditions,
        evidence="conditional" if premises else "identity",
        message="equal to the variance of the score by quadrature",
    )
