"""Convergence, differentiation under the integral sign, and boundary terms.

Every conclusion is a *sufficient* condition decided from the declared assumptions, never from
sampling, and is reported as a `Condition(latex, established, how)`. What cannot be decided is
reported as not established; nothing is assumed.

Local analysis. At each end of the interval a local variable `t -> 0+` is used (`x = e + t`,
`x = e - t`, `x = 1/t`, `x = -1/t`). With `f = K exp(g) prod b_i^e_i`:

* `|f|` behaves like `exp(Re g)` times `t^kappa`, with `kappa = sum Re(e_i) * ord_t(b_i)`.
  Re g is a rational function of t; if its order is negative its leading coefficient decides
  decay or blow-up, provided that sign is provable.
* `f` and every `d^k f / da^k = R_k f` (k <= N) are absolutely integrable at the end if `|.|`
  decays exponentially, or if `kappa + ord(R_k)` clears the threshold (`> -1` at a finite end,
  `> 1` at infinity, because `dx = dt / t^2` there). All of that is uniform on compact sets of
  the parameter because the deciding quantities are continuous in it and the inequalities are
  strict.
* An end where the phase `Im g` oscillates without bound is only accepted through absolute
  convergence; conditional convergence is not analysed.
* "Divergent" is claimed only when the phase does not oscillate, so that `f ~ C t^kappa` with
  `C != 0` and a constant phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from ..distributions.models import Condition
from ..latexprint import to_latex
from .integrand import Hyperexponential
from .models import Reason

ORDER_CAP = 60


@dataclass
class Problem:
    reason: Reason
    message: str


@dataclass
class End:
    """One end of the interval, seen through the local variable t -> 0+."""

    name: str  # "lower" / "upper"
    point: sp.Expr
    t: sp.Dummy
    x_of_t: sp.Expr


@dataclass
class Analysis:
    conditions: list[Condition] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)
    divergent: bool = False
    values: dict[str, sp.Expr | None] = field(default_factory=dict)  # G at "lower" / "upper"

    @property
    def sound(self) -> bool:
        return not self.problems and not self.divergent


# --- parameter domain -------------------------------------------------------------------------
def parameter_domain(a: sp.Symbol) -> tuple[sp.Expr, sp.Expr] | None:
    """The open interval the parameter ranges over, from its declared assumptions."""
    if a.is_positive or (a.is_nonnegative and a.is_real):
        return sp.Integer(0), sp.oo
    if a.is_negative or (a.is_nonpositive and a.is_real):
        return -sp.oo, sp.Integer(0)
    if a.is_real:
        return -sp.oo, sp.oo
    return None


# --- signs on an interval ---------------------------------------------------------------------
def _sign(value: sp.Expr) -> int | None:
    if value.is_positive:
        return 1
    if value.is_negative:
        return -1
    return None


def _sign_poly_open(p: sp.Expr, var: sp.Symbol, lo: sp.Expr, hi: sp.Expr) -> int | None:
    """Constant sign of the polynomial p(var) on (lo, hi), at most one end infinite."""
    if var not in p.free_symbols:
        return _sign(p)
    s = sp.Dummy("s", positive=True)
    if lo.is_finite and hi.is_finite:
        expr = p.subs(var, lo + (hi - lo) * s / (1 + s))
    elif lo.is_finite:
        expr = p.subs(var, lo + s)
    elif hi.is_finite:
        expr = p.subs(var, hi - s)
    else:
        return None
    numer = sp.numer(sp.together(expr))
    coefficients = sp.Poly(sp.expand(numer), s).all_coeffs()
    if all(c.is_nonnegative for c in coefficients) and any(c.is_positive for c in coefficients):
        return 1
    if all(c.is_nonpositive for c in coefficients) and any(c.is_negative for c in coefficients):
        return -1
    return None


def sign_on_interval(p: sp.Expr, var: sp.Symbol, lo: sp.Expr, hi: sp.Expr) -> int | None:
    """Constant nonzero sign of the polynomial p on the open interval, if provable."""
    if lo.is_finite or hi.is_finite:
        return _sign_poly_open(p, var, lo, hi)
    left = _sign_poly_open(p, var, -sp.oo, sp.Integer(0))
    right = _sign_poly_open(p, var, sp.Integer(0), sp.oo)
    middle = _sign(p.subs(var, 0))
    return left if left is not None and left == right == middle else None


def rational_sign(h: sp.Expr, var: sp.Symbol, lo: sp.Expr, hi: sp.Expr) -> int | None:
    """Constant nonzero sign of a rational function on the interval, factor by factor."""
    total = 1
    for part in (sp.numer(h), sp.denom(h)):
        const, factors = sp.factor_list(part)
        sign = _sign(const)
        if sign is None:
            return None
        total *= sign
        for factor, multiplicity in factors:
            if var not in factor.free_symbols:
                s = _sign(factor)
            else:
                s = sign_on_interval(factor, var, lo, hi)
            if s is None:
                return None
            total *= s if multiplicity % 2 else 1
    return total


# --- local behaviour --------------------------------------------------------------------------
def leading(h: sp.Expr, t: sp.Symbol) -> tuple[int, sp.Expr] | None:
    """(order, coefficient) of the leading term of a rational function of t at t = 0."""
    h = sp.cancel(sp.together(h))
    if h == 0:
        return None
    parts = []
    for part in (sp.numer(h), sp.denom(h)):
        poly = sp.Poly(sp.expand(part), t)
        monoms = sorted(m[0] for m in poly.monoms())
        low = monoms[0]
        parts.append((low, poly.coeff_monomial(t**low)))
    (n_ord, n_lc), (d_ord, d_lc) = parts
    return n_ord - d_ord, sp.cancel(n_lc / d_lc)


def _ends(lower: sp.Expr, upper: sp.Expr) -> list[End]:
    ends = []
    for name, point in (("lower", lower), ("upper", upper)):
        t = sp.Dummy("t", positive=True)
        if point is sp.oo:
            expr = 1 / t
        elif point is -sp.oo:
            expr = -1 / t
        elif name == "lower":
            expr = point + t
        else:
            expr = point - t
        ends.append(End(name, point, t, expr))
    return ends


@dataclass
class _Local:
    status: str  # "none" (no exponential factor), "decay", "blowup", "unknown"
    oscillates: bool
    kappa: sp.Expr  # algebraic order of |f|
    g0: sp.Expr | None  # limit of g when it is finite
    threshold: sp.Expr  # integrable iff kappa + order > threshold


def _local(h: Hyperexponential, end: End) -> _Local | Problem:
    t, xt = end.t, end.x_of_t
    infinite = end.point in (sp.oo, -sp.oo)
    threshold = sp.Integer(1) if infinite else sp.Integer(-1)
    g = sp.cancel(sp.together(h.exponent.subs(h.x, xt)))
    status = "none"
    oscillates = False
    g0: sp.Expr | None = None
    if g != 0:
        real = sp.cancel(sp.together(sp.re(g)))
        imag = sp.cancel(sp.together(sp.im(g)))
        if not (real.is_rational_function(t) and imag.is_rational_function(t)):
            return Problem(
                "convergence_not_established",
                "The real and imaginary parts of the exponent could not be separated; declare "
                "the symbols real.",
            )
        lead = leading(real, t) if real != 0 else None
        if lead is not None and lead[0] < 0:
            sign = _sign(lead[1])
            status = "decay" if sign == -1 else "blowup" if sign == 1 else "unknown"
        phase = leading(imag, t) if imag != 0 else None
        oscillates = phase is not None and phase[0] < 0
        if status == "none" and not oscillates:
            g0 = sp.cancel(g).subs(t, 0)
    kappa: sp.Expr = sp.Integer(0)
    for base, power in h.powers:
        lead_b = leading(base.subs(h.x, xt), t)
        if lead_b is None:
            return Problem("singularity_not_excluded", "A base vanishes identically.")
        kappa += sp.re(power) * lead_b[0]
    return _Local(status, oscillates, sp.simplify(kappa), g0, threshold)


def _order(expr: sp.Expr, end: End) -> int | None:
    lead = leading(expr.subs(sp.Symbol("x"), end.x_of_t), end.t)
    return None if lead is None else lead[0]


def _decide_positive(expr: sp.Expr) -> bool | None:
    """True if provably positive, False if provably not, None if undecided."""
    expr = sp.simplify(expr)
    if expr.is_positive:
        return True
    if expr.is_nonpositive:
        return False
    return None


def analyse(
    h: Hyperexponential,
    ratios: list[sp.Expr],  # R_0 ... R_N
    y: sp.Expr,
    lower: sp.Expr,
    upper: sp.Expr,
) -> Analysis:
    """Everything Phase 2 needs about the integral over (lower, upper)."""
    x = h.x
    out = Analysis()
    add = out.conditions.append
    interval = to_latex(sp.Interval.open(lower, upper)) if lower != upper else ""

    # 1. No singularity inside, and a definite branch for every non-integer power.
    pieces = [sp.denom(r) for r in (h.r1, h.r2, y, *ratios)]
    g_den = sp.denom(sp.cancel(sp.together(h.exponent)))
    pieces.append(g_den)
    for base, _power in h.powers:
        pieces.extend([sp.numer(sp.cancel(base)), sp.denom(sp.cancel(base))])
    seen: set[sp.Expr] = set()
    for piece in pieces:
        _, factors = sp.factor_list(piece)
        for factor, _m in factors:
            if x in factor.free_symbols and factor not in seen:
                seen.add(factor)
                sign = sign_on_interval(factor, x, lower, upper)
                add(
                    Condition(
                        latex=f"{to_latex(factor)} \\neq 0 \\text{{ on }} {interval}",
                        established=sign is not None,
                        how="constant sign shown by a coefficient certificate"
                        if sign is not None
                        else "not provable from the declared assumptions",
                    )
                )
                if sign is None:
                    out.problems.append(
                        Problem(
                            "singularity_not_excluded",
                            "The integrand or one of its parameter derivatives may have a pole "
                            "inside the interval; declare the symbols' signs so that it can be "
                            "excluded.",
                        )
                    )
    for base, power in h.powers:
        if power.is_integer:
            continue
        sign = rational_sign(sp.cancel(base), x, lower, upper)
        add(
            Condition(
                latex=f"{to_latex(base)} > 0 \\text{{ on }} {interval}",
                established=sign == 1,
                how="the base is provably positive, so the power has one real branch"
                if sign == 1
                else "not provable: the power might leave the principal branch",
            )
        )
        if sign != 1:
            out.problems.append(
                Problem(
                    "branch_not_established",
                    "A non-integer power of an expression that is not provably positive.",
                )
            )
    if out.problems:
        return out

    # 2./3. Each end: convergence of f and of every R_k f; 4. the value of G there.
    for end in _ends(lower, upper):
        local = _local(h, end)
        if isinstance(local, Problem):
            out.problems.append(local)
            return out
        label = "the lower end" if end.name == "lower" else "the upper end"
        where = to_latex(end.point)
        if local.status == "decay":
            add(
                Condition(
                    latex=f"|f| \\text{{ decays exponentially at }} {where}",
                    established=True,
                    how=f"the real part of the exponent tends to minus infinity at {label}; the same "
                    "bound holds uniformly for every parameter derivative",
                )
            )
            out.values[end.name] = sp.Integer(0)
            continue
        if local.status == "blowup":
            add(
                Condition(
                    latex=f"|f| \\text{{ grows exponentially at }} {where}",
                    established=False,
                    how="the real part of the exponent tends to plus infinity",
                )
            )
            out.divergent = True
            return out
        if local.status == "unknown":
            add(
                Condition(
                    latex=f"\\text{{behaviour at }} {where}",
                    established=False,
                    how="the sign of the leading exponent coefficient is not provable; declare "
                    "the symbols' signs",
                )
            )
            out.problems.append(
                Problem(
                    "convergence_not_established",
                    "The sign of the leading coefficient of the exponent at an end of the "
                    "interval is not provable from the declared assumptions.",
                )
            )
            return out
        # No exponential factor: power-law behaviour |f| ~ t^kappa.
        for k, ratio in enumerate(ratios):
            order = _order_in(ratio, h.x, end)
            if order is None:
                continue
            kappa_k = local.kappa + order
            verdict = _decide_positive(kappa_k - local.threshold)
            what = "f" if k == 0 else f"\\partial_{{{to_latex(h.a)}}}^{{{k}}} f"
            add(
                Condition(
                    latex=f"{what} \\text{{ integrable at }} {where}",
                    established=verdict is True,
                    how="absolute convergence from the exact local exponent"
                    if verdict is True
                    else "the local exponent does not clear the threshold or is undecided",
                )
            )
            if verdict is not True:
                if k == 0 and verdict is False and not local.oscillates:
                    out.divergent = True
                    return out
                out.problems.append(
                    Problem(
                        "convergence_not_established"
                        if k == 0
                        else "differentiation_not_justified",
                        "Absolute convergence at an end of the interval is not established"
                        + ("." if k == 0 else " for a parameter derivative of the integrand."),
                    )
                )
                return out
        out.values[end.name] = _boundary_value(h, y, end, local)
        if out.values[end.name] is None:
            out.problems.append(
                Problem(
                    "boundary_not_established",
                    "The value of the certificate at an end of the interval could not be "
                    "established.",
                )
            )
            add(
                Condition(
                    latex=f"G \\text{{ at }} {where}",
                    established=False,
                    how="neither a decay nor an exact finite limit could be shown",
                )
            )
    return out


def _order_in(expr: sp.Expr, x: sp.Symbol, end: End) -> int | None:
    lead = leading(expr.subs(x, end.x_of_t), end.t)
    return None if lead is None else lead[0]


def _boundary_value(h: Hyperexponential, y: sp.Expr, end: End, local: _Local) -> sp.Expr | None:
    """lim G = lim y f at the end, when the exponent part is tame; otherwise None."""
    t = end.t
    y_lead = leading(y.subs(h.x, end.x_of_t), t)
    if y_lead is None:
        return sp.Integer(0)
    kappa_g = sp.simplify(local.kappa + y_lead[0])
    if kappa_g.is_positive:
        return sp.Integer(0)
    if kappa_g != 0 or local.g0 is None:
        return None
    value = y_lead[1] * h.constants * sp.exp(local.g0)
    for base, power in h.powers:
        lead_b = leading(base.subs(h.x, end.x_of_t), t)
        if lead_b is None or not lead_b[1].is_positive:
            return None
        value *= lead_b[1] ** power
    return sp.simplify(value)
