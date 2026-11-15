"""The techniques. Each returns a `Found` candidate or None; none of them is trusted.

A candidate is only a *way to find* an antiderivative or a value. `explore` verifies it against the
original integrand with the independent checkers (`check_antiderivative`, `check_definite`), so a
transformation that is wrong, or whose conditions do not hold, is discarded and counted.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Literal

import sympy as sp
from sympy.integrals import manualintegrate as mi

from ..creative_telescoping import boundary as bd
from ..integrate import check_antiderivative, definite_value
from ..latexprint import to_latex, to_text
from ..models import Limits
from ..verify import verify_equal
from . import detect
from .models import Step, Technique

fu = importlib.import_module("sympy.simplify.fu")  # the module, not the function of the same name
MAX_CANDIDATES = 8
NODE_CAP = 600


@dataclass
class Ctx:
    x: sp.Symbol
    lower: sp.Expr | None
    upper: sp.Expr | None
    limits: Limits
    deadline: float
    assumptions: dict[str, list[str]]
    depth: int = 0
    seen: set[sp.Expr] = field(default_factory=set)

    @property
    def definite(self) -> bool:
        return self.lower is not None and self.upper is not None

    def expired(self) -> bool:
        return time.monotonic() > self.deadline


@dataclass
class Found:
    technique: Technique
    kind: Literal["antiderivative", "value"]
    expr: sp.Expr
    steps: list[Step]
    conditions: list[str] = field(default_factory=list)
    blocked: str | None = None  # a fixed reason when a condition was not established


def _size(expr: sp.Expr) -> int:
    return sum(1 for _ in sp.preorder_traversal(expr))


def _int(g: sp.Expr, w: sp.Symbol, ctx: Ctx) -> sp.Expr | None:
    """An antiderivative of g(w), accepted only if an independent numerical derivative agrees."""
    if ctx.expired() or g.has(sp.Integral):
        return None
    try:
        F = sp.integrate(g, w)
    except Exception:
        return None
    if F.has(sp.Integral) or _size(F) > NODE_CAP:
        return None
    return F if check_antiderivative(g, w, F).passed else None


def _step(what: str, expr: sp.Expr | str) -> Step:
    return Step(what=what, latex=expr if isinstance(expr, str) else to_latex(expr))


def _rules(f: sp.Expr, x: sp.Symbol) -> list[object]:
    """SymPy's own rule tree for f (manualintegrate), flattened; empty if it cannot be built."""
    try:
        root = mi.integral_steps(f, x)
    except Exception:
        return []
    out: list[object] = []
    stack: list[object] = [root]
    while stack and len(out) < 200:
        rule = stack.pop()
        out.append(rule)
        for name in ("substep", "second_step", "v_step"):
            child = getattr(rule, name, None)
            if child is not None:
                stack.append(child)
        for name in ("alternatives", "substeps"):
            stack.extend(getattr(rule, name, None) or [])
    return out


# -- substitution ------------------------------------------------------------------------------
def substitution(f: sp.Expr, ctx: Ctx) -> Found | None:
    x = ctx.x
    candidates: list[sp.Expr] = []
    for rule in _rules(f, x):
        if isinstance(rule, mi.URule) and rule.variable == x and rule.u_func not in candidates:
            candidates.append(rule.u_func)
    for node in sorted(sp.preorder_traversal(f), key=_size):
        if (
            node != x
            and x in node.free_symbols
            and not node.is_Number
            and not isinstance(node, sp.Mul)
            and node not in candidates
        ):
            candidates.append(node)
    w = sp.Dummy("u")
    best: tuple[int, Found] | None = None
    for u in candidates[:MAX_CANDIDATES]:
        if ctx.expired():
            break
        du = sp.diff(u, x)
        if du == 0:
            continue
        try:
            g = sp.simplify(f / du).subs(u, w)
            if x in g.free_symbols:
                g = sp.simplify(sp.simplify(f / du).subs(sp.solve(sp.Eq(w, u), x)[0], w))
        except Exception:  # noqa: S112 - a candidate that fails is simply not used
            continue
        if x in g.free_symbols:
            continue
        G = _int(g, w, ctx)
        if G is None:
            continue
        # the substitution that leaves the simplest transformed integral is the one worth showing
        cost = _size(g) + _size(G)
        if best is not None and cost >= best[0]:
            continue
        v = sp.Symbol("u")
        steps = [
            _step("substitution", sp.Eq(v, u)),
            _step("differential", f"\\mathrm{{d}}u = \\left({to_latex(du)}\\right)\\,\\mathrm{{d}}x"),
            _step("transformed integral", f"\\int {to_latex(g.subs(w, v))}\\,\\mathrm{{d}}u"),
            _step("antiderivative in u", G.subs(w, v)),
        ]  # fmt: skip
        best = (cost, Found("substitution", "antiderivative", G.subs(w, u), steps))
    return None if best is None else best[1]


# -- integration by parts ----------------------------------------------------------------------
def parts(f: sp.Expr, ctx: Ctx) -> Found | None:
    x = ctx.x
    choices: list[tuple[sp.Expr, sp.Expr]] = []
    for rule in _rules(f, x):
        if isinstance(rule, mi.PartsRule | mi.CyclicPartsRule):
            pieces = getattr(rule, "u", None), getattr(rule, "dv", None)
            if pieces[0] is None and getattr(rule, "parts_rules", None):
                first = rule.parts_rules[0]
                pieces = first.u, first.dv
            if pieces[0] is not None and pieces[1] is not None and pieces not in choices:
                choices.append((pieces[0], pieces[1]))
    for u, dv in choices[:MAX_CANDIDATES]:
        if ctx.expired() or ctx.depth > 2:
            return None
        v = _int(dv, x, ctx)
        if v is None:
            continue
        rest = sp.simplify(v * sp.diff(u, x))
        if rest in ctx.seen or rest == f:  # a loop: the same integral again
            continue
        ctx.seen.add(rest)
        tail = _int(rest, x, ctx)
        if tail is None:
            continue
        steps = [
            _step("choice of u", u),
            _step("choice of dv", dv),
            _step("v = integral of dv", v),
            _step("remaining integral", sp.Integral(rest, x)),
        ]
        return Found("parts", "antiderivative", sp.expand(u * v - tail), steps)
    return None


# -- rewriting: trigonometric identities and partial fractions ---------------------------------
TRIG_RULES: tuple[tuple[str, object], ...] = (
    ("power reduction", fu.TR7),
    ("product to sum", fu.TR8),
    ("double angle", fu.TR11),
    ("Pythagorean identity", fu.TR5),
    ("Pythagorean identity", fu.TR6),
)


def trigonometric(f: sp.Expr, ctx: Ctx) -> Found | None:
    """Rewrite with a standard identity (or two in a row), keep the simplest integrable form."""
    x = ctx.x
    expanded = sp.expand(f)
    candidates: list[tuple[str, sp.Expr]] = []
    for name, rule in TRIG_RULES:
        try:
            candidates.append((name, rule(expanded)))  # type: ignore[operator]
        except Exception:  # noqa: S112 - a rewrite that fails is simply not used
            continue
    for (n1, r1), (n2, r2) in ((TRIG_RULES[0], TRIG_RULES[1]), (TRIG_RULES[0], TRIG_RULES[2])):
        try:
            candidates.append((f"{n1}, then {n2}", sp.expand(r2(r1(expanded)))))  # type: ignore[operator]
        except Exception:  # noqa: S112
            continue
    best: tuple[int, Found] | None = None
    tried: list[sp.Expr] = []
    for name, g in candidates:
        if ctx.expired():
            break
        if g == f or g in tried or _size(g) > max(2 * _size(f), 40):
            continue
        tried.append(g)
        if not verify_equal(f, g, symbolic=False).passed:
            continue
        G = _int(g, x, ctx)
        if G is None:
            continue
        steps = [
            _step(f"identity: {name}", sp.Eq(f, g, evaluate=False)),
            _step("antiderivative", G),
        ]
        if best is None or _size(G) < best[0]:
            best = (_size(G), Found("trigonometric", "antiderivative", G, steps))
    return None if best is None else best[1]


def partial_fractions(f: sp.Expr, ctx: Ctx) -> Found | None:
    x = ctx.x
    if not f.is_rational_function(x):
        return None
    try:
        g = sp.apart(f, x)
    except Exception:
        return None
    if g == f or len(sp.Add.make_args(g)) < 2 or not verify_equal(f, g, symbolic=False).passed:
        return None
    pieces = []
    total: sp.Expr = sp.Integer(0)
    for term in sp.Add.make_args(g):
        T = _int(term, x, ctx)
        if T is None:
            return None
        pieces.append(sp.Integral(term, x))
        total += T
    steps = [_step("partial fractions", sp.Eq(f, g, evaluate=False))]
    steps += [_step("term", p) for p in pieces[:6]]
    return Found("partial_fractions", "antiderivative", total, steps)


# -- symmetry ------------------------------------------------------------------------------------
def _definite(g: sp.Expr, lo: sp.Expr, hi: sp.Expr, ctx: Ctx) -> sp.Expr | None:
    """A verified definite integral by the direct route (value only if quadrature agreed)."""
    value, _ = definite_value(
        g, ctx.x, lo, hi, notation="", base={"input_latex": to_latex(g), "variable": ctx.x.name},
        deadline=ctx.deadline,
    )  # fmt: skip
    return value


def parity(f: sp.Expr, ctx: Ctx) -> Found | None:
    if not ctx.definite or ctx.lower is None or ctx.upper is None:
        return None
    x, lo, hi = ctx.x, ctx.lower, ctx.upper
    sign = detect.parity(f, x, lo, hi)
    if sign is None:
        return None
    half = _definite(f, sp.Integer(0), hi, ctx)
    if half is None:
        return Found("parity", "value", sp.Integer(0), [], blocked="convergence_not_established")
    conditions = [
        f"\\int_{{0}}^{{{to_latex(hi)}}} f \\text{{ converges (verified)}}",
        f"f(-{to_latex(x)}) = {'+' if sign == 1 else '-'}f({to_latex(x)})",
    ]
    if sign == -1:
        steps = [
            _step("symmetry", f"f(-x) = -f(x) \\text{{ on }} [-{to_latex(hi)}, {to_latex(hi)}]"),
            _step("half-range integral (converges)", half),
        ]
        return Found("parity", "value", sp.Integer(0), steps, conditions)
    steps = [
        _step("symmetry", f"f(-x) = f(x) \\text{{ on }} [-{to_latex(hi)}, {to_latex(hi)}]"),
        _step("half-range integral (converges)", half),
        _step("whole integral", sp.Eq(sp.Symbol("I"), 2 * half, evaluate=False)),
    ]
    return Found("parity", "value", sp.simplify(2 * half), steps, conditions)


def reflection(f: sp.Expr, ctx: Ctx) -> Found | None:
    if not ctx.definite or ctx.lower is None or ctx.upper is None:
        return None
    x, lo, hi = ctx.x, ctx.lower, ctx.upper
    h = detect.reflection(f, x, lo, hi)
    if h is None:
        return None
    value = _definite(h, lo, hi, ctx) if x in h.free_symbols else h * (hi - lo)
    if value is None:
        return None
    steps = [
        _step("reflection", f"x \\mapsto {to_latex(lo + hi)} - x"),
        _step("average of the two forms", h),
        _step("integral of the average", sp.Integral(h, (x, lo, hi))),
    ]
    return Found("reflection", "value", sp.simplify(value), steps)


# -- Frullani -------------------------------------------------------------------------------------
CONTINUOUS = (sp.exp, sp.sin, sp.cos, sp.atan, sp.sinh, sp.cosh, sp.tanh, sp.erf)


def _continuous_on_positive(g: sp.Expr, x: sp.Symbol) -> bool:
    for node in sp.preorder_traversal(g):
        if isinstance(node, sp.Function) and not isinstance(node, CONTINUOUS):
            return False
        if isinstance(node, sp.Pow):
            base, power = node.as_base_exp()
            if x in base.free_symbols and not power.is_integer:
                return False
    _, denominator = sp.fraction(sp.together(g))
    for factor, _m in sp.factor_list(denominator)[1]:
        if x not in factor.free_symbols or isinstance(factor, sp.exp):  # exp never vanishes
            continue
        if (
            not factor.is_polynomial(x)
            or bd.sign_on_interval(factor, x, sp.Integer(0), sp.oo) is None
        ):
            return False
    return True


def _numeric_limit(g: sp.Expr, x: sp.Symbol, at_zero: bool) -> sp.Expr | None:
    import mpmath

    mpmath.mp.dps = 60
    fn = sp.lambdify(x, g, "mpmath")
    try:
        points = (
            [mpmath.mpf(10) ** (-k) for k in (30, 40)]
            if at_zero
            else [mpmath.mpf(10) ** k for k in (30, 40)]
        )
        a, b = fn(points[0]), fn(points[1])
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None
    if abs(a - b) > mpmath.mpf("1e-12") * max(1, abs(a)):
        return None
    return sp.Float(str(mpmath.re(b)), 30)


def frullani(f: sp.Expr, ctx: Ctx) -> Found | None:
    if not ctx.definite or ctx.lower != 0 or ctx.upper is not sp.oo:
        return None
    x = ctx.x
    pair = detect.frullani_pair(f, x)
    if pair is None:
        return None
    g, c1, c2 = pair
    blocked = None
    conditions = []
    for c in (c1, c2):
        ok = bool(c.is_positive)
        conditions.append(f"{to_latex(c)} > 0")
        blocked = blocked or (None if ok else "scales_not_provably_positive")
    if not _continuous_on_positive(g, x):
        blocked = blocked or "continuity_not_established"
    if blocked:
        return Found("frullani", "value", sp.Integer(0), [], conditions, blocked)
    try:
        l0 = sp.limit(g, x, 0, "+")
        linf = sp.limit(g, x, sp.oo)
    except Exception:
        return Found("frullani", "value", sp.Integer(0), [], conditions, "limits_not_established")
    finite = all(
        v.is_finite is True
        and not v.has(sp.nan, sp.zoo, sp.oo, sp.S.NegativeInfinity, sp.AccumBounds, sp.Limit)
        for v in (l0, linf)
    )
    n0, ninf = _numeric_limit(g, x, True), _numeric_limit(g, x, False)
    if not finite or n0 is None or ninf is None:
        return Found("frullani", "value", sp.Integer(0), [], conditions, "limits_not_established")
    if abs(sp.N(l0) - n0) > 1e-8 * max(1, abs(n0)) or abs(sp.N(linf) - ninf) > 1e-8 * max(
        1, abs(ninf)
    ):
        return Found("frullani", "value", sp.Integer(0), [], conditions, "limits_not_established")
    conditions += [f"g(0^+) = {to_latex(l0)}", f"g(\\infty) = {to_latex(linf)}"]
    value = sp.expand_log(sp.simplify((l0 - linf) * sp.log(c2 / c1)), force=True)  # c1, c2 > 0
    steps = [
        _step("two scalings of one function", f"g(x) = {to_latex(g)}, \\quad a = {to_latex(c1)}, \\; b = {to_latex(c2)}"),
        _step("limits of g (both finite)", f"g(0^+) = {to_latex(l0)}, \\; g(\\infty) = {to_latex(linf)}"),
        _step("Frullani", sp.Eq(sp.Symbol("I"), value, evaluate=False)),
    ]  # fmt: skip
    return Found("frullani", "value", value, steps, conditions)


# -- Schwinger parametrisation ----------------------------------------------------------------------
def _positive_on(expr: sp.Expr, x: sp.Symbol, lo: sp.Expr, hi: sp.Expr, strict: bool) -> bool:
    if (expr.is_positive if strict else expr.is_nonnegative) is True:
        return True
    if expr.is_rational_function(x):
        sign = bd.rational_sign(sp.cancel(expr), x, lo, hi)
        return sign == 1
    return False


def schwinger(f: sp.Expr, ctx: Ctx) -> Found | None:
    if not ctx.definite or ctx.lower is None or ctx.upper is None:
        return None
    x, lo, hi = ctx.x, ctx.lower, ctx.upper
    factor = detect.schwinger_factor(f, x)
    if factor is None:
        return None
    rest, base, n = factor
    conditions = [f"{to_latex(base)} > 0 \\text{{ on the interval}}", f"{to_latex(n)} > 0"]
    if not _positive_on(base, x, lo, hi, True) or n.is_positive is not True:
        return Found(
            "schwinger",
            "value",
            sp.Integer(0),
            [],
            conditions,
            "base_or_power_not_provably_positive",
        )
    if not _positive_on(rest, x, lo, hi, False):
        return Found(
            "schwinger", "value", sp.Integer(0), [], conditions, "interchange_not_justified"
        )
    t = sp.Symbol("t", positive=True)
    inner = _definite(rest * sp.exp(-t * base), lo, hi, ctx)
    if inner is None:
        return None
    s = sp.Symbol("t", positive=True)
    outer_integrand = s ** (n - 1) * inner.subs(t, s) / sp.gamma(n)
    value, _ = definite_value(
        outer_integrand, s, sp.Integer(0), sp.oo, notation="",
        base={"input_latex": to_latex(outer_integrand), "variable": "t"}, deadline=ctx.deadline,
    )  # fmt: skip
    if value is None:
        return None
    conditions.append(
        "the remaining factor is nonnegative, so the two integrals may be swapped (Tonelli)"
    )
    steps = [
        _step("Schwinger identity", f"A^{{-n}} = \\frac{{1}}{{\\Gamma(n)}}\\int_0^\\infty t^{{n-1}} \\mathrm{{e}}^{{-tA}}\\,\\mathrm{{d}}t, \\quad A = {to_latex(base)}, \\; n = {to_latex(n)}"),
        _step("inner integral over x", sp.Eq(sp.Symbol("J(t)"), inner, evaluate=False)),
        _step("outer integral over t", sp.Integral(outer_integrand, (s, 0, sp.oo))),
        _step("value", value),
    ]  # fmt: skip
    return Found("schwinger", "value", sp.simplify(value), steps, conditions)


# -- differentiation under the integral sign ---------------------------------------------------------
def parameter_differentiation(f: sp.Expr, ctx: Ctx) -> Found | None:
    if not ctx.definite or ctx.lower is None or ctx.upper is None:
        return None
    from ..creative_telescoping import TelescopingInput, creative_telescoping

    for p in detect.telescopable(f, ctx.x):
        if ctx.expired():
            return None
        request = TelescopingInput(
            input=to_text(f),
            format="text",
            assumptions={
                k: v for k, v in ctx.assumptions.items() if k in {s.name for s in f.free_symbols}
            },
            variable=ctx.x.name,
            parameter=p.name,
            lower=to_text(ctx.lower),
            upper=to_text(ctx.upper),
        )
        result = creative_telescoping(request, ctx.limits)
        value = result.value_expr
        if result.outcome == "evaluated" and value is not None:
            steps = []
            if result.operator:
                steps.append(_step("operator in the parameter", result.operator.latex))
            if result.certificate:
                steps.append(_step("certificate (L f = dG/dx, proved)", result.certificate.latex))
            if result.equation:
                steps.append(_step("differential equation of the integral", result.equation.latex))
            steps.append(_step("value", value))
            return Found("parameter_differentiation", "value", value, steps)
    return None
