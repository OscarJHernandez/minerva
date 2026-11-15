"""Fourier series of the periodic extension of a function given on one period.

Coefficients come from composite Gauss-Legendre quadrature built over the places the function is not
smooth, with two panel counts that must agree. Closed forms are attempted with SymPy (under a time
limit) and shown only if they reproduce the quadrature coefficients. The reconstruction is measured
against the function on two disjoint grids; at a jump the series converges to the midpoint value and
the overshoot (Gibbs) does not shrink, which is reported, never hidden.
"""

from __future__ import annotations

import itertools
from typing import Any

import mpmath
import sympy as sp

from ..latexprint import to_latex
from ..models import Status
from .chebyshev import candidate_of
from .common import Context, Refused, finish, parse_text
from .interval import Measured, Reference, build_reference, measure, parse_tolerance, tolerance_met
from .models import ApproximationReport, Check, Complexity, Labeled, OrderTerm, Warning
from .quadnodes import DPS, Nodes, breakpoints, composite, time_limit, usable

MAX_MODES = 64
LADDER = (1, 2, 4, 8, 16, 32, 64)
AGREE = mpmath.mpf(
    "1e-10"
)  # the two quadrature rules agree to this share of the largest coefficient
ZERO = mpmath.mpf("1e-12")  # a coefficient below this share of the largest is a symmetry zero
SYMBOLIC_SECONDS = 2.0
JUMP_GRID = 4


def period_of(ctx: Context) -> tuple[Any, Any, tuple[sp.Expr, sp.Expr], list[str]]:
    data = ctx.data
    notes: list[str] = []
    with mpmath.workdps(DPS):
        if data.lower is not None and data.upper is not None:
            vals = [parse_text(t, data, ctx.limits) for t in (data.lower, data.upper)]
            a, b = (mpmath.mpf(str(sp.N(v, DPS))) for v in vals)
            if not a < b:
                raise Refused(
                    Status.UNSUPPORTED,
                    "point_invalid",
                    "The lower end must be below the upper end.",
                )
            if data.period is not None:
                t = mpmath.mpf(str(sp.N(parse_text(data.period, data, ctx.limits), DPS)))
                if abs(t - (b - a)) > mpmath.mpf(10) ** -20:
                    raise Refused(
                        Status.UNSUPPORTED,
                        "point_invalid",
                        "The period must equal the length of the interval given (the interval is one period).",
                    )
            return a, b, (vals[0], vals[1]), notes
        if data.period is not None:
            t_sym = parse_text(data.period, data, ctx.limits)
            t = mpmath.mpf(str(sp.N(t_sym, DPS)))
            if t <= 0:
                raise Refused(Status.UNSUPPORTED, "point_invalid", "The period must be positive.")
            notes.append(
                "No interval was given, so the function is taken on the symmetric interval of one period "
                "centred at 0."
            )
            return -t / 2, t / 2, (-t_sym / 2, t_sym / 2), notes
    raise Refused(
        Status.CONDITIONS_NOT_SATISFIED,
        "domain_unspecified",
        "The period (or the interval that is one period) is yours to choose.",
        ["What is the period, or on which interval is one period of the function given?"],
    )


def jump_points(ref: Reference) -> list[Any]:
    """Approximate positions where a sample step is much larger than its neighbours (refined by bisection)."""
    ys, xs = ref.ys1, ref.xs1
    span = max(ys) - min(ys)
    if span == 0:
        return []
    found = []
    steps = [abs(q - p) for p, q in itertools.pairwise(ys)]
    typical = sorted(steps)[len(steps) // 2]
    for i, step in enumerate(steps):
        if step > mpmath.mpf("0.2") * span and step > JUMP_GRID * 20 * (
            typical + mpmath.mpf(10) ** -30
        ):
            lo, hi = xs[i], xs[i + 1]
            for _ in range(60):
                mid = (lo + hi) / 2
                left = abs(ref.f(mid) - ref.f(lo))
                right = abs(ref.f(hi) - ref.f(mid))
                lo, hi = (mid, hi) if left < right else (lo, mid)
            found.append((lo + hi) / 2)
    return found


class Coefficients:
    def __init__(self, a0: Any, an: list[Any], bn: list[Any], err: Any) -> None:
        self.a0, self.an, self.bn, self.err = a0, an, bn, err


def _coefficients(rule: Nodes, a: Any, omega: Any, period: Any, modes: int) -> Coefficients:
    scale = 2 / period
    a0 = scale * sum((w * y for w, y in zip(rule.ws, rule.ys, strict=True)), mpmath.mpf(0))
    an, bn = [], []
    for n in range(1, modes + 1):
        c = s = mpmath.mpf(0)
        for x, w, y in zip(rule.xs, rule.ws, rule.ys, strict=True):
            angle = n * omega * x
            c += w * y * mpmath.cos(angle)
            s += w * y * mpmath.sin(angle)
        an.append(scale * c)
        bn.append(scale * s)
    return Coefficients(a0, an, bn, mpmath.mpf(0))


def compute(ctx: Context, ref: Reference, modes: int) -> tuple[Coefficients, list[Any]]:
    a, b = ref.a, ref.b
    period = b - a
    with mpmath.workdps(DPS):
        omega = 2 * mpmath.pi / period
        breaks = sorted({*breakpoints(ctx.expr, ctx.var, a, b), *jump_points(ref)}, key=float)
        points = [a, *[p for p in breaks if a < p < b], b]
        cycles = modes / float(period)
        coarse = composite(points, cycles, ref.f, 1)
        fine = composite(points, cycles, ref.f, 2)
        if not (usable(coarse) and usable(fine)):
            raise Refused(
                Status.UNSUPPORTED,
                "domain_invalid",
                "The function could not be evaluated for the integrals.",
            )
        c1 = _coefficients(coarse, a, omega, period, modes)
        c2 = _coefficients(fine, a, omega, period, modes)
        gap = max(
            [
                abs(c1.a0 - c2.a0),
                *[abs(p - q) for p, q in zip(c1.an + c1.bn, c2.an + c2.bn, strict=True)],
            ]
        )
        c2.err = gap
        return c2, breaks


def _tidy(coeffs: Coefficients) -> tuple[Coefficients, str]:
    largest = max([abs(coeffs.a0), *map(abs, coeffs.an), *map(abs, coeffs.bn)]) or mpmath.mpf(1)
    cut = ZERO * largest
    an = [c if abs(c) > cut else mpmath.mpf(0) for c in coeffs.an]
    bn = [c if abs(c) > cut else mpmath.mpf(0) for c in coeffs.bn]
    a0 = coeffs.a0 if abs(coeffs.a0) > cut else mpmath.mpf(0)
    parity = ""
    if not any(bn) and any(an):
        parity = "cosine-only: the function is even about the centre of the period"
    elif not any(an) and a0 == 0 and any(bn):
        parity = "sine-only: the function is odd about the centre of the period"
    rounded = Coefficients(_r(a0), [_r(c) for c in an], [_r(c) for c in bn], coeffs.err)
    return rounded, parity


def _r(value: Any) -> Any:
    return mpmath.mpf(mpmath.nstr(value, 12, min_fixed=-mpmath.inf, max_fixed=mpmath.inf))


def sigma_factors(modes: int) -> list[Any]:
    """Lanczos sigma factors sinc(n pi / (N + 1))."""
    return [
        mpmath.sinc(n * mpmath.pi / (modes + 1)) if n else mpmath.mpf(1) for n in range(modes + 1)
    ]


def evaluator(ref: Reference, coeffs: Coefficients, sigma: bool) -> Any:
    modes = len(coeffs.an)
    omega = 2 * mpmath.pi / (ref.b - ref.a)
    factors = sigma_factors(modes) if sigma else [mpmath.mpf(1)] * (modes + 1)

    def at(x: Any) -> Any:
        total = coeffs.a0 / 2
        for n in range(1, modes + 1):
            angle = n * omega * x
            total += factors[n] * (
                coeffs.an[n - 1] * mpmath.cos(angle) + coeffs.bn[n - 1] * mpmath.sin(angle)
            )
        return total

    return at


def _ends(ref: Reference) -> tuple[sp.Expr, sp.Expr]:
    if ref.exact_ends is None:
        raise Refused(
            Status.UNSUPPORTED, "point_invalid", "The interval ends are not exact numbers."
        )
    return ref.exact_ends


def _omega(ref: Reference) -> sp.Expr:
    lo, hi = _ends(ref)
    return sp.simplify(2 * sp.pi / (hi - lo))


def _latex(coeffs: Coefficients, variable: str, ref: Reference) -> str:
    x = sp.Symbol(variable)
    omega = _omega(ref)
    parts = []
    if coeffs.a0 != 0:
        parts.append(rf"\frac{{{mpmath.nstr(coeffs.a0, 12)}}}{{2}}")
    for n, (c, s) in enumerate(zip(coeffs.an, coeffs.bn, strict=True), start=1):
        for value, trig in ((c, "cos"), (s, "sin")):
            if value != 0:
                fn = sp.cos if trig == "cos" else sp.sin
                parts.append(rf"{mpmath.nstr(value, 12)}\,{to_latex(fn(n * omega * x))}")
    text = ""
    for piece in parts:
        text = (
            piece
            if not text
            else (f"{text} {piece}" if piece.startswith("-") else f"{text} + {piece}")
        )
    return text or "0"


def _symbolic(ctx: Context, ref: Reference, coeffs: Coefficients) -> list[Labeled]:
    """Closed forms of a_n, b_n, shown only if they reproduce the quadrature coefficients."""
    var = ctx.var
    n = sp.Symbol("n", integer=True, positive=True)
    lo, hi = _ends(ref)
    period = hi - lo
    omega = 2 * sp.pi / period
    out: list[Labeled] = []
    with time_limit(SYMBOLIC_SECONDS):
        forms: dict[str, sp.Expr] = {}
        for name, trig in (("a_n", sp.cos), ("b_n", sp.sin)):
            value = sp.integrate(ctx.expr * trig(n * omega * var), (var, lo, hi))
            if value.has(sp.Integral):
                return []
            forms[name] = sp.simplify(2 * value / period)
        check = min(6, len(coeffs.an))
        for name, series in (("a_n", coeffs.an), ("b_n", coeffs.bn)):
            for k in range(1, check + 1):
                got = forms[name].subs(n, k)
                if got.has(sp.nan, sp.zoo):
                    got = sp.limit(forms[name], n, k)
                if abs(mpmath.mpf(str(sp.N(got, 30))) - series[k - 1]) > mpmath.mpf("1e-9") * (
                    1 + abs(series[k - 1])
                ):
                    return []
        out = [
            Labeled(label=f"{k} (valid for n = 1, 2, ...)", latex=to_latex(v))
            for k, v in forms.items()
        ]
    return out


def _bessel(rule: Nodes, coeffs: Coefficients, period: Any) -> tuple[Any, Any]:
    energy = (2 / period) * sum(
        (w * y * y for w, y in zip(rule.ws, rule.ys, strict=True)), mpmath.mpf(0)
    )
    kept = coeffs.a0**2 / 2 + sum(
        (c * c + s * s for c, s in zip(coeffs.an, coeffs.bn, strict=True)), mpmath.mpf(0)
    )
    return kept, energy


def build(
    ctx: Context,
    ref: Reference,
    modes: int,
    tolerance: Any | None,
    kind: str | None,
    notes: list[str],
    symbolic: bool = True,
) -> tuple[ApproximationReport, Measured | None]:
    sigma = ctx.data.kind == "sigma"
    raw, breaks = compute(ctx, ref, modes)
    coeffs, parity = _tidy(raw)
    largest = max([abs(coeffs.a0), *map(abs, coeffs.an), *map(abs, coeffs.bn)]) or mpmath.mpf(1)
    agree = raw.err <= AGREE * largest
    approx = evaluator(ref, coeffs, sigma)
    measured = measure(ref, approx)
    with mpmath.workdps(DPS):
        rule = composite(
            [ref.a, *breaks, ref.b] if breaks else [ref.a, ref.b],
            modes / float(ref.b - ref.a),
            ref.f,
            2,
        )
        kept, energy = _bessel(rule, coeffs, ref.b - ref.a)
    bessel_ok = kept <= energy * (1 + mpmath.mpf("1e-9")) + mpmath.mpf(10) ** -25
    checks = [
        Check(
            name="coefficients: two quadrature rules agree",
            passed=bool(agree),
            points=2 * modes + 1,
            detail=f"largest difference {mpmath.nstr(raw.err, 3)} (panel count doubled)",
        ),
        Check(
            name="Bessel inequality (energy of the series <= energy of the function)",
            passed=bool(bessel_ok),
            points=1,
            detail=f"the series carries {mpmath.nstr(100 * kept / energy, 6) if energy else '0'}% of the energy",
        ),
        Check(
            name="reconstruction measured on two disjoint grids",
            passed=not measured.unstable,
            points=measured.summary.samples,
        ),
    ]
    report = ApproximationReport(
        status=Status.OK,
        method="fourier",
        variable=ctx.var.name,
        domain_latex=ref.domain_latex,
        result_latex=_latex(coeffs, ctx.var.name, ref),
        checks=checks,
        error=measured.summary,
        error_curve=measured.curve,
        complexity=Complexity(
            modes=modes, coefficients=1 + sum(1 for c in [*coeffs.an, *coeffs.bn] if c != 0)
        ),
        warnings=list(ref.warnings),
        notes=[*notes, *ref.notes],
    )
    for n in range(1, modes + 1):
        for value, trig in ((coeffs.an[n - 1], "a"), (coeffs.bn[n - 1], "b")):
            if value != 0:
                report.orders.append(
                    OrderTerm(
                        order=n,
                        coefficient_latex=mpmath.nstr(value, 12),
                        term_latex=f"{trig}_{{{n}}} = {mpmath.nstr(value, 12)}",
                    )
                )
    if parity:
        report.notes.append(f"Symmetry: {parity}; the vanishing coefficients are omitted.")
    # discontinuities, inside the period and at its ends (the periodic extension jumps if f(a) != f(b))
    edge = abs(ref.f(ref.a) - ref.f(ref.b))
    jumps = "jump_discontinuity" in ref.warnings or bool(jump_points(ref))
    if jumps or edge > mpmath.mpf(10) ** -12 * (1 + ref.scale):
        gibbs: list[Warning] = ["jump_discontinuity", "gibbs"]
        for w in gibbs:
            if w not in report.warnings:
                report.warnings.append(w)
        report.notes.append(
            "The periodic extension jumps: at a jump the series converges to the midpoint value, and the "
            "overshoot near it (about 9% of the jump) does not shrink with more terms. Uniform convergence "
            "is not claimed near the jump."
            + (
                " Lanczos sigma factors were applied, which smooth the overshoot at the cost of resolution."
                if sigma
                else ""
            )
        )
    if not agree or not bessel_ok or measured.unstable:
        report.status = Status.NOT_VERIFIED
        report.reason = "verification_failed"
        report.result_latex = None
        report.message = "Fourier coefficients were computed but their checks did not agree, so they are not shown."
        return report, measured
    if symbolic and not ctx.expired(0.5):
        closed = _symbolic(ctx, ref, coeffs)
        report.extras += closed
        if not closed:
            report.notes.append(
                "Coefficients are numerical: no closed form was found that reproduced them."
            )
    report.tolerance_met = tolerance_met(measured, tolerance, kind)
    report.evidence = "numeric_estimate"
    report.message = (
        f"Fourier series with {modes} modes of the periodic extension. The largest error on "
        f"{measured.summary.samples} sampled points is {measured.summary.max_abs} (a sampled estimate, not a bound)."
    )
    return report, measured


def fourier(ctx: Context) -> ApproximationReport:
    a, b, exact, notes = period_of(ctx)
    ref = build_reference(ctx, (a, b))
    ref.exact_ends = exact
    tolerance, kind = parse_tolerance(ctx)
    data = ctx.data
    if data.modes is not None:
        if not 1 <= data.modes <= MAX_MODES:
            raise Refused(
                Status.UNSUPPORTED,
                "order_out_of_range",
                f"Give a number of modes between 1 and {MAX_MODES}.",
            )
        report, _ = build(ctx, ref, data.modes, tolerance, kind, notes)
        return finish(report, evidence=report.evidence)
    if tolerance is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "Give the number of modes, or an accuracy target.",
            ["How many Fourier modes, or which accuracy?"],
        )
    cap = min(MAX_MODES, data.max_terms or MAX_MODES)
    history = []
    last: ApproximationReport | None = None
    for modes in (m for m in LADDER if m <= cap):
        if ctx.expired():
            break
        report, measured = build(ctx, ref, modes, tolerance, kind, notes, symbolic=False)
        history.append(candidate_of(report, measured, f"{modes} modes"))
        last = report
        if report.status is Status.OK and report.tolerance_met:
            history[-1].chosen = True
            report, _ = build(
                ctx, ref, modes, tolerance, kind, notes
            )  # again, with the closed forms
            report.candidates = history
            report.message += (
                " It is the fewest modes tried that meet the tolerance on the sampled points."
            )
            return report
    if last is None:
        raise Refused(
            Status.RESOURCE_LIMIT,
            "resource_limit",
            "The time budget ended before any number of modes was tried.",
        )
    last.candidates = history
    last.tolerance_met = False
    last.message += (
        f" The tolerance was NOT met by any number of modes up to {history[-1].complexity.modes}"
        " (on the sampled points); the last is shown."
    )
    return last
