"""A function on a finite interval: the reference every interval method is measured against.

The reference is the ORIGINAL expression evaluated with mpmath at 40 digits on two grids that share
no point. The domain scan refuses what cannot be approximated (undefined or non-real values, poles)
and warns about what can be measured but not trusted (jumps). Errors are measurements on these
points: nothing here is a bound.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mpmath
import sympy as sp

from ..latexprint import to_latex
from ..models import Status
from .checks import compile_mp
from .common import Context, Refused, parse_text
from .models import ErrorSummary, PlotPoint, Warning

DPS_REF = 40
GRID = 1001  # grid 1 includes both ends; grid 2 is the midpoints of grid 1
CURVE_POINTS = 200
MASK_FRACTION = mpmath.mpf(
    "1e-3"
)  # relative error is not formed where |f| is below this share of max |f|
UNSTABLE_FACTOR = 1.5  # the two grids' maximum errors may differ by at most this factor
POLE_RATIO = 30  # a sampled value this much above both neighbours (and large) is read as a pole

Approx = Callable[[Any], Any]


@dataclass
class Reference:
    f: Callable[[Any], Any]
    a: Any
    b: Any
    xs1: list[Any]
    ys1: list[Any]
    xs2: list[Any]
    ys2: list[Any]
    scale: Any
    zero_crossing: bool
    warnings: list[Warning] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def domain_latex(self) -> str:
        return rf"\left[{mpmath.nstr(self.a, 12)},\, {mpmath.nstr(self.b, 12)}\right]"

    def t(self, x: Any) -> Any:
        """The affine map of [a, b] onto [-1, 1]."""
        return (2 * x - (self.a + self.b)) / (self.b - self.a)


def _real(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, mpmath.mpc):
        if abs(value.imag) > mpmath.mpf(10) ** -25 * (1 + abs(value.real)):
            return None
        return value.real
    return mpmath.mpf(value)


def parse_interval(ctx: Context) -> tuple[Any, Any]:
    data = ctx.data
    if data.lower is None or data.upper is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "The interval is yours to choose.",
            ["On which interval should the approximation hold?"],
        )
    values = []
    for text in (data.lower, data.upper):
        v = parse_text(text, data, ctx.limits)
        if v.free_symbols or v.is_finite is not True or v.is_real is not True:
            raise Refused(
                Status.UNSUPPORTED,
                "point_invalid",
                "The interval ends must be finite real numbers.",
            )
        values.append(v)
    with mpmath.workdps(DPS_REF):
        a, b = (mpmath.mpf(str(sp.N(v, DPS_REF))) for v in values)
    if not a < b:
        raise Refused(
            Status.UNSUPPORTED, "point_invalid", "The lower end must be below the upper end."
        )
    return a, b


def single_variable(ctx: Context) -> None:
    others = ctx.expr.free_symbols - {ctx.var}
    if others:
        names = ", ".join(sorted(to_latex(s) for s in others))
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "parameter_missing",
            f"An approximation on an interval needs a function of {ctx.var.name} alone; give a value for {names}.",
            [f"What values should {names} take?"],
        )


def build_reference(ctx: Context) -> Reference:
    single_variable(ctx)
    a, b = parse_interval(ctx)
    with mpmath.workdps(DPS_REF):
        f = compile_mp(ctx.expr, ctx.var)
        n = GRID
        xs1 = [a + (b - a) * i / (n - 1) for i in range(n)]
        xs2 = [a + (b - a) * (2 * i + 1) / (2 * (n - 1)) for i in range(n - 1)]
        ys1: list[Any] = []
        ys2: list[Any] = []
        for xs, ys in ((xs1, ys1), (xs2, ys2)):
            for x in xs:
                y = _real(f(x))
                if y is None:
                    raise Refused(
                        Status.UNSUPPORTED,
                        "domain_invalid",
                        f"The function is undefined or not real-valued at x = {mpmath.nstr(x, 8)} in the interval, "
                        "so it cannot be approximated there.",
                    )
                ys.append(y)
        scale = max(abs(y) for y in ys1 + ys2)
        ref = Reference(f, a, b, xs1, ys1, xs2, ys2, scale, _crosses_zero(ys1))
        _scan(ref)
    return ref


def _crosses_zero(ys: list[Any]) -> bool:
    return any(p * q <= 0 for p, q in itertools.pairwise(ys))


def _scan(ref: Reference) -> None:
    """Refuse a pole; warn about a jump."""
    ys = ref.ys1
    typical = sorted(abs(y) for y in ys)[len(ys) // 2] + mpmath.mpf(10) ** -30
    for i in range(1, len(ys) - 1):
        around = max(abs(ys[i - 1]), abs(ys[i + 1]))
        if abs(ys[i]) > POLE_RATIO * around and abs(ys[i]) > 1000 * typical:
            raise Refused(
                Status.UNSUPPORTED,
                "domain_invalid",
                "The function has a pole in the interval (it grows without bound near "
                f"x = {mpmath.nstr(ref.xs1[i], 8)}), so a global approximation is not defined.",
            )
    peak = max(abs(y) for y in ys)
    for p, q in itertools.pairwise(ys):
        # a sign change between two large values is a pole between the samples; a jump or a zero
        # crossing has a small value on at least one side, or values of ordinary size
        if p * q < 0 and peak > 100 * typical and min(abs(p), abs(q)) > mpmath.mpf("0.01") * peak:
            raise Refused(
                Status.UNSUPPORTED,
                "domain_invalid",
                "The function has a pole in the interval (its sign flips across very large values), "
                "so a global approximation is not defined.",
            )
    # A jump does not shrink when the grid is refined; a smooth function's largest step halves.
    coarse = max(abs(q - p) for p, q in itertools.pairwise(ys[::2]))
    fine = max(abs(q - p) for p, q in itertools.pairwise(ys))
    span = max(ys) - min(ys)
    if span > 0 and coarse > mpmath.mpf("1e-3") * span and fine > mpmath.mpf("0.7") * coarse:
        ref.warnings.append("jump_discontinuity")
        ref.notes.append(
            "The function appears to jump (or change abruptly) in the interval: a smooth approximation "
            "will oscillate near it, and the measured error there is not representative."
        )


@dataclass
class Measured:
    summary: ErrorSummary
    max_abs: Any
    max_rel: Any | None
    curve: list[PlotPoint]
    unstable: bool
    undefined_at: Any | None = None


def _fmt(value: Any) -> str:
    if mpmath.isinf(value):
        return "unbounded (the approximant is undefined or has a pole in the interval)"
    return str(mpmath.nstr(value, 4))


def measure(ref: Reference, approx: Approx) -> Measured:
    """Absolute, relative (away from zeros) and RMS error of `approx` against the reference."""
    errors1: list[Any] = []
    errors2: list[Any] = []
    bad: Any | None = None
    with mpmath.workdps(DPS_REF):
        for xs, ys, out in ((ref.xs1, ref.ys1, errors1), (ref.xs2, ref.ys2, errors2)):
            for x, y in zip(xs, ys, strict=True):
                v = _real(approx(x))
                if v is None:
                    bad = x if bad is None else bad
                    out.append(mpmath.inf)
                else:
                    out.append(abs(y - v))
        max1, max2 = max(errors1), max(errors2)
        worst = max(range(len(errors1)), key=errors1.__getitem__)
        cutoff = MASK_FRACTION * ref.scale
        rel = [
            e / abs(y)
            for e, y in zip(errors1 + errors2, ref.ys1 + ref.ys2, strict=True)
            if abs(y) >= cutoff
        ]
        max_rel = max(rel) if rel else None
        rms = (
            mpmath.sqrt(sum(e * e for e in errors1) / len(errors1))
            if max1 != mpmath.inf
            else mpmath.inf
        )
        floor = mpmath.mpf(10) ** -(DPS_REF - 15)
        lo, hi = sorted((max1, max2))
        unstable = max1 != mpmath.inf and hi > floor and hi / max(lo, floor) > UNSTABLE_FACTOR
        step = max(1, len(errors1) // CURVE_POINTS)
        curve = [
            PlotPoint(x=float(ref.xs1[i]), error=float(errors1[i]))
            for i in range(0, len(errors1), step)
        ]
        summary = ErrorSummary(
            samples=len(errors1) + len(errors2),
            max_abs=_fmt(max(max1, max2)),
            rms=_fmt(rms),
            max_rel=_fmt(max_rel) if max_rel is not None else None,
            relative_note=(
                "relative error is formed only where |f| is at least 0.1% of its largest sampled value"
                + ("; f has a zero in the interval" if ref.zero_crossing else "")
            ),
            at=mpmath.nstr(ref.xs1[worst], 6),
        )
        return Measured(summary, max(max1, max2), max_rel, curve, unstable, bad)


def tolerance_met(measured: Measured, tolerance: Any | None, kind: str | None) -> bool | None:
    """Does the sampled error meet the tolerance? None if no tolerance, or it cannot be assessed."""
    if tolerance is None or kind is None:
        return None
    if kind == "absolute":
        return bool(measured.max_abs <= tolerance)
    if measured.max_rel is None:
        return None
    return bool(measured.max_rel <= tolerance)


def parse_tolerance(ctx: Context) -> tuple[Any | None, str | None]:
    """The user's accuracy target as (number, kind); both or neither."""
    data = ctx.data
    if data.tolerance is None and data.tolerance_kind is None:
        return None, None
    if data.tolerance is None or data.tolerance_kind is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "A tolerance needs both a number and whether it is absolute or relative.",
            ["Which tolerance, and is it an absolute or a relative error?"],
        )
    try:
        value = mpmath.mpf(data.tolerance)
    except (ValueError, TypeError):
        raise Refused(
            Status.PARSE_ERROR, "unsupported_input", "The tolerance must be a number."
        ) from None
    if value <= 0:
        raise Refused(Status.UNSUPPORTED, "unsupported_input", "The tolerance must be positive.")
    return value, data.tolerance_kind
