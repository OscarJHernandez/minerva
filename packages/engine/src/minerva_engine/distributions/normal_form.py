"""The exact part of the distribution operation: identities applied under proved conditions.

Nothing here samples. Every condition (the variable is real, a scale factor is nonzero, a root is
real and simple, a point lies inside the integration domain) is decided symbolically from the
declared assumptions; when it cannot be, a `Refusal` carries the reason. The numerical evidence for
the result lives in `pairing.py` and is independent of this module.

Conventions fixed here (they are stated in the results, never inherited from SymPy):

- ``DiracDelta(g(x))`` for a polynomial ``g`` with simple real roots ``r_i`` is
  ``sum_i DiracDelta(x - r_i) / |g'(r_i)|``; complex roots do not contribute (``x`` is real).
- ``f(x) DiracDelta(x - r) = f(r) DiracDelta(x - r)`` only if ``f`` is defined and continuous at ``r``.
- ``Heaviside`` is a step with no value at the jump (the value there never matters to a distribution);
  a product of a delta and a step at the same point is undefined and refused.
- A delta exactly on an integration limit counts only under an explicit ``endpoint`` convention.
"""

from __future__ import annotations

import dataclasses

import sympy as sp

from ..latexprint import to_latex
from .models import (
    ENTRY_COMPOSITE,
    ENTRY_HEAVISIDE_DERIVATIVE,
    ENTRY_SCALING,
    ENTRY_SIFTING,
    ENTRY_TIMES_FUNCTION,
    Condition,
    DistributionStep,
    Endpoint,
    Reason,
)

MAX_DEGREE = 6
MAX_OPS = 600
MAX_ROOTS = 12

# Functions that are smooth (infinitely differentiable) on the whole real line.
_SMOOTH_FUNCTIONS = (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh, sp.tanh, sp.atan, sp.asinh)


class Refusal(Exception):
    """The operation cannot give a verified result; `reason` says why."""

    def __init__(self, reason: Reason, message: str, entry_id: str | None = None) -> None:
        super().__init__(message)
        self.reason: Reason = reason
        self.message = message
        self.entry_id = entry_id


@dataclasses.dataclass
class Term:
    """``coeff * DiracDelta(arg)`` (``arg`` is None for a term without a delta)."""

    coeff: sp.Expr
    arg: sp.Expr | None


@dataclasses.dataclass
class Root:
    position: sp.Expr
    slope: sp.Expr  # g'(position), proved nonzero


@dataclasses.dataclass
class Outcome:
    result: sp.Expr
    steps: list[DistributionStep]
    notes: list[str]
    conditions: list[Condition]


# --- reading the expression ---------------------------------------------------------------
def check_size(expr: sp.Expr) -> None:
    if sp.count_ops(expr) > MAX_OPS:
        raise Refusal("resource_limit", "The expression is too large for this operation.")


def require_real(expr: sp.Expr, what: str) -> None:
    for symbol in sorted(expr.free_symbols, key=lambda s: s.name):
        if symbol.is_real is not True:
            raise Refusal(
                "assumptions_insufficient",
                f"{symbol.name} appears in the argument of a {what} but is not declared real. "
                "Declare it real: distributions of a complex argument are not defined here.",
            )


def split_terms(expr: sp.Expr) -> list[Term]:
    """Expand and split into ``coeff * DiracDelta(arg)`` terms; refuse what is not of that form."""
    expanded = sp.expand(expr)
    check_size(expanded)
    terms: list[Term] = []
    for summand in sp.Add.make_args(expanded):
        deltas: list[sp.Basic] = []
        rest: list[sp.Expr] = []
        for factor in sp.Mul.make_args(summand):
            if isinstance(factor, sp.DiracDelta):
                if len(factor.args) > 1:
                    raise Refusal(
                        "unsupported_distribution",
                        "Derivatives of the delta function are not supported in this version.",
                    )
                deltas.append(factor)
            elif factor.has(sp.DiracDelta):
                raise Refusal(
                    "unsupported_distribution",
                    "A delta inside a power or another function (for example a square of a "
                    "delta) is not defined as a distribution.",
                )
            else:
                rest.append(factor)
        if len(deltas) > 1:
            raise Refusal(
                "unsupported_distribution",
                "A product of two delta functions is not defined as a distribution.",
            )
        coeff = sp.Mul(*rest)
        terms.append(Term(coeff, deltas[0].args[0] if deltas else None))
    return terms


def is_smooth(expr: sp.Expr, x: sp.Symbol) -> bool:
    """True if ``expr`` is built only from pieces known to be smooth for every real ``x``."""
    if not expr.has(x):
        return not expr.has(sp.zoo, sp.nan)
    if expr == x:
        return True
    if isinstance(expr, sp.Add | sp.Mul):
        return all(is_smooth(a, x) for a in expr.args)
    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        if exponent.has(x):
            return bool(base.is_positive) and is_smooth(exponent, x) and not base.has(x)
        return bool(exponent.is_Integer and exponent >= 0) and is_smooth(base, x)
    if isinstance(expr, _SMOOTH_FUNCTIONS):
        return all(is_smooth(a, x) for a in expr.args)
    return False


# --- exact roots ---------------------------------------------------------------------------
def simple_real_roots(g: sp.Expr, x: sp.Symbol) -> tuple[list[Root], list[Condition]]:
    """Real roots of the polynomial ``g`` in ``x`` with a proof that each is simple."""
    require_real(g, "delta")
    if not g.has(x):
        raise Refusal(
            "unsupported_distribution",
            "The argument of the delta does not depend on the variable.",
        )
    try:
        poly = sp.Poly(g, x)
    except sp.PolynomialError as exc:
        raise Refusal(
            "unsupported_distribution",
            "Only deltas of polynomials in the variable are supported (for example "
            "DiracDelta(x**2 - 4)); the argument is not a polynomial.",
        ) from exc
    degree = poly.degree()
    if degree > MAX_DEGREE:
        raise Refusal("resource_limit", "The polynomial inside the delta has too high a degree.")
    found = sp.roots(poly, x)
    if sum(found.values()) != degree or len(found) > MAX_ROOTS:
        raise Refusal(
            "unsupported_distribution",
            "The roots of the argument could not all be written in closed form.",
        )
    slope_expr = sp.diff(g, x)
    roots: list[Root] = []
    conditions: list[Condition] = []
    for position, multiplicity in found.items():
        realness = position.is_real
        if realness is None and not position.free_symbols:
            realness = bool(abs(sp.im(sp.N(position, 30))) < sp.Float("1e-25"))
        if realness is None:
            raise Refusal(
                "assumptions_insufficient",
                f"Whether the root {_tex(position)} is real is not decided by the declared "
                "assumptions (for example a sign or a nonzero condition is missing).",
                ENTRY_COMPOSITE if degree > 1 else ENTRY_SCALING,
            )
        if not realness:
            conditions.append(
                Condition(
                    latex=rf"{_tex(position)} \notin \mathbb{{R}}",
                    established=True,
                    how="not a real root, so it carries no delta",
                )
            )
            continue
        if multiplicity > 1:
            raise Refusal(
                "singular_root",
                f"The root {_tex(position)} is a repeated root, where the delta of a function is "
                "not a combination of simple deltas.",
                ENTRY_COMPOSITE,
            )
        slope = sp.simplify(slope_expr.subs(x, position))
        if slope.is_zero is not False and not (slope.is_number and slope != 0):
            raise Refusal(
                "assumptions_insufficient" if slope.is_zero is None else "singular_root",
                f"The slope {_tex(slope)} of the argument at the root {_tex(position)} is not "
                "proved nonzero from the declared assumptions (declare the parameter nonzero, "
                "positive or negative).",
                ENTRY_COMPOSITE if degree > 1 else ENTRY_SCALING,
            )
        roots.append(Root(position=position, slope=slope))
        conditions.append(
            Condition(
                latex=rf"g'({_tex(position)}) = {_tex(slope)} \neq 0",
                established=True,
                how="exact: the slope is nonzero under the declared assumptions, so the root is simple",
            )
        )
    return roots, conditions


def _tex(expr: sp.Basic) -> str:
    return to_latex(expr)


def _same(a: sp.Expr, b: sp.Expr) -> bool:
    return bool(a == b or sp.simplify(a - b) == 0)


# --- multiplier ----------------------------------------------------------------------------
def value_at(coeff: sp.Expr, x: sp.Symbol, point: sp.Expr) -> sp.Expr:
    """``coeff(point)`` if ``coeff`` is defined and continuous there, else a refusal."""
    for atom in coeff.atoms(sp.Heaviside, sp.sign):
        argument = atom.args[0]
        if not argument.has(x):
            continue
        at_point = sp.simplify(argument.subs(x, point))
        if at_point.is_zero is True:
            raise Refusal(
                "unsupported_distribution",
                "A delta multiplied by a step function at its jump point is not defined.",
                ENTRY_TIMES_FUNCTION,
            )
        if at_point.is_zero is None:
            raise Refusal(
                "assumptions_insufficient",
                "Whether the delta sits on the jump of a step function in its coefficient is not "
                "decided by the declared assumptions.",
                ENTRY_TIMES_FUNCTION,
            )
    with sp.evaluate(True):
        value = sp.simplify(coeff.subs(x, point))
    if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or value.is_finite is False:
        raise Refusal(
            "unsupported_distribution",
            "The coefficient of the delta is not defined at the point where the delta is "
            "supported, so the product is not defined.",
            ENTRY_TIMES_FUNCTION,
        )
    return value


# --- simplify and integrate ----------------------------------------------------------------
@dataclasses.dataclass
class Expanded:
    regular: sp.Expr  # terms without a delta
    deltas: list[tuple[sp.Expr, sp.Expr, sp.Expr]]  # (position, multiplier(x), slope)
    max_degree: int
    conditions: list[Condition]


def expand_deltas(expr: sp.Expr, x: sp.Symbol) -> Expanded:
    terms = split_terms(expr)
    regular: list[sp.Expr] = []
    deltas: list[tuple[sp.Expr, sp.Expr, sp.Expr]] = []
    conditions: list[Condition] = []
    max_degree = 0
    cache: dict[sp.Expr, tuple[list[Root], list[Condition]]] = {}
    for term in terms:
        if term.arg is None:
            regular.append(term.coeff)
            continue
        if term.arg not in cache:
            cache[term.arg] = simple_real_roots(term.arg, x)
            max_degree = max(max_degree, sp.Poly(term.arg, x).degree())
            conditions.extend(cache[term.arg][1])
        for root in cache[term.arg][0]:
            deltas.append((root.position, term.coeff, root.slope))
    return Expanded(sp.Add(*regular), deltas, max_degree, conditions)


def _delta_at(position: sp.Expr, x: sp.Symbol) -> sp.Expr:
    return sp.DiracDelta(sp.expand(x - position))


def _with_deltas(
    regular: sp.Expr,
    deltas: list[tuple[sp.Expr, sp.Expr, sp.Expr]],
    x: sp.Symbol,
    *,
    evaluate: bool,
) -> sp.Expr:
    """The expression with every delta expanded; ``evaluate`` also substitutes the multiplier."""
    total: list[sp.Expr] = [regular]
    for position, multiplier, slope in deltas:
        factor = value_at(multiplier, x, position) if evaluate else multiplier
        total.append(factor / sp.Abs(slope) * _delta_at(position, x))
    return sp.Add(*total)


def _collect(expr: sp.Expr, x: sp.Symbol) -> sp.Expr:
    """Merge deltas at provably equal points and drop zero coefficients."""
    merged: list[tuple[sp.Expr, sp.Expr]] = []
    regular: list[sp.Expr] = []
    for term in split_terms(expr):
        if term.arg is None:
            regular.append(term.coeff)
            continue
        position = sp.solve(term.arg, x)
        point = position[0] if len(position) == 1 else None
        if point is None:
            regular.append(term.coeff * sp.DiracDelta(term.arg))
            continue
        for index, (known, total) in enumerate(merged):
            if _same(known, point):
                merged[index] = (known, total + term.coeff)
                break
        else:
            merged.append((point, term.coeff))
    out: list[sp.Expr] = list(regular)
    for point, coefficient in merged:
        coefficient = sp.simplify(coefficient)
        if coefficient != 0:
            out.append(coefficient * _delta_at(point, x))
    return sp.Add(*out)


def simplify_deltas(expr: sp.Expr, x: sp.Symbol) -> Outcome:
    expanded = expand_deltas(expr, x)
    if not expanded.deltas and not any(t.arg is not None for t in split_terms(expr)):
        raise Refusal(
            "unsupported_distribution",
            "There is no Dirac delta to simplify; only deltas of polynomials are simplified.",
        )
    steps: list[DistributionStep] = []
    entry = ENTRY_SCALING if expanded.max_degree <= 1 else ENTRY_COMPOSITE
    title = (
        "Scaling and translation of the delta argument"
        if entry == ENTRY_SCALING
        else "Delta of a polynomial with simple real roots"
    )
    stage1 = _with_deltas(expanded.regular, expanded.deltas, x, evaluate=False)
    steps.append(
        DistributionStep(
            entry_id=entry,
            title=title,
            before_latex=_tex(expr),
            after_latex=_tex(stage1),
            conditions=expanded.conditions,
        )
    )
    stage2 = _with_deltas(expanded.regular, expanded.deltas, x, evaluate=True)
    if any(mult.has(x) for _, mult, _ in expanded.deltas):
        steps.append(
            DistributionStep(
                entry_id=ENTRY_TIMES_FUNCTION,
                title="A continuous factor takes its value at the delta",
                before_latex=_tex(stage1),
                after_latex=_tex(stage2),
                conditions=[
                    Condition(
                        latex=rf"{_tex(mult)}\ \text{{defined and continuous at }}{_tex(pos)}",
                        established=True,
                        how="exact: substitution is finite and no step function jumps there",
                    )
                    for pos, mult, _ in expanded.deltas
                    if mult.has(x)
                ],
            )
        )
    result = _collect(stage2, x)
    notes = ["Equality is as distributions: both sides act identically on every test function."]
    return Outcome(result, steps, notes, expanded.conditions)


def _compare_sign(difference: sp.Expr) -> int:
    value = sp.simplify(difference)
    if value.is_positive:
        return 1
    if value.is_negative:
        return -1
    if value.is_zero:
        return 0
    raise Refusal(
        "assumptions_insufficient",
        "Whether a delta lies inside the integration limits is not decided by the declared "
        "assumptions.",
        ENTRY_SIFTING,
    )


def order_bounds(lower: sp.Expr, upper: sp.Expr) -> tuple[sp.Expr, sp.Expr, int]:
    """The limits in increasing order and the sign the swap contributes to the integral."""
    reversed_limits = (
        lower == sp.oo
        or upper == -sp.oo
        or (
            lower.is_number
            and upper.is_number
            and upper != sp.oo
            and lower != -sp.oo
            and bool(lower > upper)
        )
    )
    return (upper, lower, -1) if reversed_limits else (lower, upper, 1)


def integrate_deltas(
    expr: sp.Expr,
    x: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    endpoint: Endpoint | None,
) -> Outcome:
    terms = split_terms(expr)
    if any(t.arg is None for t in terms):
        raise Refusal(
            "unsupported_distribution",
            "The integrand mixes delta terms with ordinary terms; integrate the ordinary part "
            "with the integrate operation.",
        )
    lower, upper, sign = order_bounds(lower, upper)
    expanded = expand_deltas(expr, x)
    notes: list[str] = []
    steps: list[DistributionStep] = []
    entry = ENTRY_SCALING if expanded.max_degree <= 1 else ENTRY_COMPOSITE
    stage1 = _with_deltas(expanded.regular, expanded.deltas, x, evaluate=False)
    steps.append(
        DistributionStep(
            entry_id=entry,
            title=(
                "Scaling and translation of the delta argument"
                if entry == ENTRY_SCALING
                else "Delta of a polynomial with simple real roots"
            ),
            before_latex=_tex(expr),
            after_latex=_tex(stage1),
            conditions=expanded.conditions,
        )
    )
    stage2 = _with_deltas(expanded.regular, expanded.deltas, x, evaluate=True)
    if any(mult.has(x) for _, mult, _ in expanded.deltas):
        steps.append(
            DistributionStep(
                entry_id=ENTRY_TIMES_FUNCTION,
                title="A continuous factor takes its value at the delta",
                before_latex=_tex(stage1),
                after_latex=_tex(stage2),
                conditions=[
                    Condition(
                        latex=rf"{_tex(mult)}\ \text{{defined and continuous at }}{_tex(pos)}",
                        established=True,
                        how="exact: substitution is finite and no step function jumps there",
                    )
                    for pos, mult, _ in expanded.deltas
                    if mult.has(x)
                ],
            )
        )
    collected = _collect(stage2, x)
    position_conditions: list[Condition] = []
    total: list[sp.Expr] = []
    for term in split_terms(collected):
        points = sp.solve(term.arg, x) if term.arg is not None else []
        if len(points) != 1:
            raise Refusal("unsupported_distribution", "A delta could not be located exactly.")
        point = points[0]
        below = 1 if lower == -sp.oo else _compare_sign(point - lower)
        above = 1 if upper == sp.oo else _compare_sign(upper - point)
        label = _tex(point)
        if below < 0 or above < 0:
            weight = sp.Integer(0)
            how = "outside the limits: contributes 0"
        elif below == 0 or above == 0:
            if endpoint is None:
                raise Refusal(
                    "integration_domain_ambiguous",
                    f"A delta sits exactly on an integration limit (at {label}). Its contribution "
                    "depends on a convention that you must choose: half, include or exclude.",
                    ENTRY_SIFTING,
                )
            weight = {
                "half": sp.Rational(1, 2),
                "include": sp.Integer(1),
                "exclude": sp.Integer(0),
            }[endpoint]
            how = f"on a limit: counted with weight {weight} by the {endpoint} convention you chose"
        else:
            weight = sp.Integer(1)
            how = "strictly inside the limits: contributes fully"
        position_conditions.append(Condition(latex=rf"x = {label}", established=True, how=how))
        total.append(weight * term.coeff)
    value = sp.simplify(sign * sp.Add(*total))
    steps.append(
        DistributionStep(
            entry_id=ENTRY_SIFTING,
            title="Integrating against a delta picks out the value",
            before_latex=_tex(collected),
            after_latex=_tex(value),
            conditions=position_conditions,
        )
    )
    if sign == -1:
        notes.append("The limits were reversed, so the sign of the integral is flipped.")
    if endpoint is not None and any(c.how.startswith("on a limit") for c in position_conditions):
        notes.append(f"Endpoint convention used: {endpoint}.")
    return Outcome(value, steps, notes, expanded.conditions + position_conditions)


# --- differentiate -------------------------------------------------------------------------
@dataclasses.dataclass
class _Jump:
    multiplier: sp.Expr  # f(x), smooth everywhere
    step: sp.Expr  # Heaviside(a x + b)
    slope: sp.Expr  # a, proved nonzero
    position: sp.Expr  # -b / a


def _read_jump(summand: sp.Expr, x: sp.Symbol) -> tuple[_Jump, Condition]:
    factors = sp.Mul.make_args(summand)
    steps = [f for f in factors if isinstance(f, sp.Heaviside)]
    others = [f for f in factors if not isinstance(f, sp.Heaviside)]
    if len(steps) != 1 or any(f.has(sp.Heaviside) for f in others):
        raise Refusal(
            "unsupported_distribution",
            "Each term may contain at most one step function, to the first power.",
        )
    step = steps[0]
    multiplier = sp.Mul(*others)
    argument = step.args[0]
    require_real(argument, "step function")
    try:
        poly = sp.Poly(argument, x)
    except sp.PolynomialError as exc:
        raise Refusal(
            "unsupported_distribution",
            "The argument of a step function must be a polynomial in the variable.",
            ENTRY_HEAVISIDE_DERIVATIVE,
        ) from exc
    if poly.degree() != 1:
        raise Refusal(
            "unsupported_distribution",
            "Only step functions of an affine argument a x + b are supported.",
            ENTRY_HEAVISIDE_DERIVATIVE,
        )
    if not is_smooth(multiplier, x):
        raise Refusal(
            "unsupported_distribution",
            "The factor multiplying a step function must be smooth everywhere.",
            ENTRY_HEAVISIDE_DERIVATIVE,
        )
    slope, offset = poly.all_coeffs()
    if slope.is_zero is not False:
        raise Refusal(
            "assumptions_insufficient",
            f"The slope {_tex(slope)} of the step's argument is not proved nonzero from the "
            "declared assumptions.",
            ENTRY_HEAVISIDE_DERIVATIVE,
        )
    condition = Condition(
        latex=rf"{_tex(slope)} \neq 0",
        established=True,
        how="exact: the slope of the step's argument is nonzero under the declared assumptions",
    )
    return _Jump(multiplier, step, slope, sp.simplify(-offset / slope)), condition


def differentiate_steps(expr: sp.Expr, x: sp.Symbol) -> Outcome:
    """Distributional derivative of ``sum f_k(x) H(a_k x + b_k) + smooth``."""
    expanded = sp.expand(expr)
    check_size(expanded)
    smooth_derivative: list[sp.Expr] = []
    pieces: list[sp.Expr] = []  # f'(x) H(u)
    raw_deltas: list[sp.Expr] = []  # f(x) a DiracDelta(a x + b)
    final_deltas: list[sp.Expr] = []  # f(r) a/|a| DiracDelta(x - r)
    conditions: list[Condition] = []
    sift_conditions: list[Condition] = []
    for summand in sp.Add.make_args(expanded):
        if summand.has(sp.DiracDelta):
            raise Refusal(
                "unsupported_distribution",
                "The derivative of a delta function is not supported in this version.",
            )
        if not summand.has(sp.Heaviside):
            if not is_smooth(summand, x):
                raise Refusal(
                    "unsupported_distribution",
                    "Only terms that are smooth everywhere (polynomials, exponentials, sines, "
                    "cosines and similar) can be differentiated together with a step function.",
                )
            smooth_derivative.append(sp.diff(summand, x))
            continue
        jump, condition = _read_jump(summand, x)
        conditions.append(condition)
        pieces.append(sp.diff(jump.multiplier, x) * jump.step)
        raw_deltas.append(jump.multiplier * jump.slope * sp.DiracDelta(jump.step.args[0]))
        value = value_at(jump.multiplier, x, jump.position)
        final_deltas.append(value * jump.slope / sp.Abs(jump.slope) * _delta_at(jump.position, x))
        sift_conditions.append(
            Condition(
                latex=(rf"{_tex(jump.multiplier)}\ \text{{defined at }}{_tex(jump.position)}"),
                established=True,
                how="exact: smooth everywhere, so its value at the jump exists",
            )
        )
    if not raw_deltas:
        raise Refusal(
            "unsupported_distribution",
            "There is no step function to differentiate; use the differentiate operation.",
        )
    stage1 = sp.Add(*smooth_derivative, *pieces, *raw_deltas)
    steps = [
        DistributionStep(
            entry_id=ENTRY_HEAVISIDE_DERIVATIVE,
            title="The derivative of a step function is a delta at its jump",
            before_latex=_tex(expr),
            after_latex=_tex(stage1),
            conditions=conditions,
        )
    ]
    stage2 = sp.Add(*smooth_derivative, *pieces, *final_deltas)
    others = sp.Add(*[p for p in sp.Add.make_args(stage2) if not p.has(sp.DiracDelta)])
    delta_part = sp.Add(*[p for p in sp.Add.make_args(stage2) if p.has(sp.DiracDelta)])
    result = sp.Add(others, _collect(delta_part, x))
    steps.append(
        DistributionStep(
            entry_id=ENTRY_SCALING,
            title="Scaling of the delta and the value of its coefficient at the jump",
            before_latex=_tex(stage1),
            after_latex=_tex(result),
            conditions=sift_conditions,
        )
    )
    notes = [
        "This is the distributional derivative: each step function keeps its jump and every "
        "jump contributes a delta of the size of the jump."
    ]
    return Outcome(result, steps, notes, conditions + sift_conditions)
