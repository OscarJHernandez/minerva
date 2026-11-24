"""Link an equation the model wrote to a result a tool already verified.

The claim checker used to compare the model's formulas only with each other: a correct equation of motion
written as `\\ddot\\theta = \\sin\\theta\\cos\\theta\\,\\dot\\phi^2 + \\frac{g}{R}\\sin\\theta` was never compared with what
`analyze_lagrangian` had verified, because the two are printed differently. Here the comparison is by
mathematics, never by spelling, and only three kinds of match exist:

1. **Same left side, equivalent right side.** `\\ddot\\theta` equals `\\ddot\\theta` and the right sides agree under
   `verify_equal` (a symbolic zero where SymPy finds one, otherwise sampled points).
2. **Proportional residuals.** `(a - b) = k (c - d)` where `c = d` is verified and `k` does not depend on the
   coordinates, velocities or accelerations of the system (`R m \\ddot\\theta ...` against `m R^2 \\ddot\\theta ...`).
   The zeros of `k` are reported as conditions: multiplying by them loses information there.
3. **A named conserved quantity.** `E = ...` against a verified `h = ...` whose role is a conserved quantity or a
   momentum, with equivalent right sides. Only for those roles: `y = 2` is not supported by a verified `x = 2`.

One more outcome is not a match: a claim that reuses the tool's own name for a verified conserved quantity or
momentum (`C_{4} = ...` against the verified `C_{4} = ...`) with a right side that provably differs (a gap
confirmed at 60 and 120 digits by `counterexample.find_counterexample`). It is reported as a difference from
that result, never as a contradiction: the mark stays "unchecked" and the note says which verified equation it
departs from. A mismatch needs the same name; a different name on the left is the model's own quantity.

A fourth kind exists for Taylor series and only for them (`SeriesEvidence`, `SeriesIndex`): the model's
`f(x) = P(x) + O((x-a)^{n+1})` is linked to a verified series result only if the function, the variable, the
centre, the order, every coefficient of the polynomial and the order of the remainder all agree with that
result (see "Taylor series" at the end). It is never a rule for other asymptotic statements.

A match says "the same statement a verified result made". It does not verify the claim by itself and never
widens what the tool established: the restrictions of the result travel with the match. Anything else (a
rearranged form, terms moved across the equals sign, solved for another symbol) is not matched and stays
exactly as unchecked as before.
"""

from __future__ import annotations

import re
from typing import Literal

import sympy as sp
from pydantic import BaseModel

from .counterexample import Budget, find_counterexample
from .errors import MinervaMathError
from .latexparse import parse_latex
from .models import Limits
from .restrictions import determinant_conditions
from .verify import verify_equal

_MAX_OPS = 400
# Roles whose left side is a name for a quantity (rule 3).
NAMED_ROLES = frozenset({"conserved", "momentum"})


class ClaimEvidence(BaseModel):
    """A formula a tool verified, with what is needed to compare against it."""

    latex: str  # an equation `lhs = rhs`
    source: str = ""  # which result it came from (`r1`), shown to the user
    basis: Literal["symbolic", "numeric"] = "numeric"  # how the tool established it
    role: str = ""  # `equation_of_motion`, `acceleration`, `momentum`, `conserved` or empty
    variables: list[str] = []  # coordinates of the system: their `_dot` and `_ddot` are derived
    restrictions: list[str] = []  # where the result is valid


class SeriesTermEvidence(BaseModel):
    power: int
    coefficient_latex: str


class SeriesEvidence(BaseModel):
    """A Taylor series a tool verified, typed: what was expanded, about what, to which order, with what
    result. Built by the tool from its own result, never from the model's prose (`ToolOutput.series`).

    `distances` are the distances from the centre at which the tool measured its truncation error.
    `assumptions` are the ones declared in the call; a claim is read under exactly those."""

    source: str = ""
    expression_latex: str
    variable: str
    centre_latex: str
    order: int
    terms: list[SeriesTermEvidence] = []
    distances: list[float] = []
    assumptions: dict[str, list[str]] = {}


class EvidenceMatch(BaseModel):
    source: str
    rule: Literal["same_left_side", "proportional_residual", "named_quantity", "series"]
    basis: Literal["symbolic", "numeric"]
    restrictions: list[str] = []
    detail: str = ""


class EvidenceMismatch(BaseModel):
    """The claim uses the name of a verified result and a different expression for it."""

    source: str
    latex: str  # the verified equation, as the tool printed it
    role: str


def _dynamic(names: list[str]) -> set[str]:
    out = {"t"}
    for n in names:
        out |= {n, f"{n}_dot", f"{n}_ddot"}
    return out


def _agree(a: sp.Expr, b: sp.Expr, symbolic: bool = False) -> tuple[bool, bool]:
    """(equivalent, symbolic zero) under the verified-equality check; undecided is not equivalent.

    The symbolic zero test (`sp.simplify`) is run only when it can change the answer given to the user:
    a link is called symbolic only if the tool's own result was a symbolic zero too."""
    try:
        v = verify_equal(a, b, symbolic=symbolic)
    except Exception:
        return False, False
    return bool(v.passed and v.numeric_points_checked >= 4), bool(v.symbolic_zero)


def _split(latex: str) -> list[str] | None:
    from .claims import split_equation

    sides = split_equation(latex)
    return sides if sides is not None and len(sides) == 2 else None


class EvidenceIndex:
    """The parsed evidence of a conversation, built once per check."""

    def __init__(
        self, evidence: list[ClaimEvidence], assumptions: dict[str, list[str]], limits: Limits
    ) -> None:
        self.items: list[tuple[ClaimEvidence, sp.Expr, sp.Expr]] = []
        for item in evidence:
            sides = _split(item.latex)
            if sides is None:
                continue
            try:
                left, right = (parse_latex(s, assumptions, limits) for s in sides)
            except MinervaMathError:
                continue
            self.items.append((item, left, right))

    def match(self, claim: list[sp.Expr]) -> EvidenceMatch | None:
        if len(claim) != 2 or not self.items:
            return None
        for a, b in (claim, claim[::-1]):
            for item, c, d in self.items:
                found = self._same_left_side(a, b, item, c, d) or self._named(a, b, item, c, d)
                if found is not None:
                    return found
        for a, b in (claim, claim[::-1]):
            for item, c, d in self.items:
                found = self._proportional(a, b, item, c, d)
                if found is not None:
                    return found
        return None

    def mismatch(self, claim: list[sp.Expr]) -> EvidenceMismatch | None:
        """The verified result whose name `claim` reuses with an expression that provably differs, if any.
        Only after `match` found nothing. Cancellation noise, a point defined on one side only and an
        exception are not a difference (`find_counterexample` confirms at 60 and 120 digits)."""
        if len(claim) != 2 or not self.items:
            return None
        for a, b in (claim, claim[::-1]):
            if not a.is_Symbol or a.name.endswith(("_dot", "_ddot")) or a in b.free_symbols:
                continue
            for item, c, d in reversed(self.items):
                if item.role not in NAMED_ROLES or not c.is_Symbol or c != a:
                    continue
                found = find_counterexample(
                    b, d, budget=Budget(points=40, seconds=1.0), kinds=("random", "special")
                )
                if found.counterexample is not None:
                    return EvidenceMismatch(source=item.source, latex=item.latex, role=item.role)
        return None

    @staticmethod
    def _basis(item: ClaimEvidence, symbolic: bool) -> Literal["symbolic", "numeric"]:
        return "symbolic" if symbolic and item.basis == "symbolic" else "numeric"

    def _same_left_side(
        self, a: sp.Expr, b: sp.Expr, item: ClaimEvidence, c: sp.Expr, d: sp.Expr
    ) -> EvidenceMatch | None:
        if not (a == c or sp.expand(a - c) == 0) or not a.free_symbols:
            return None
        ok, symbolic = _agree(b, d, item.basis == "symbolic")
        if not ok:
            return None
        return EvidenceMatch(
            source=item.source,
            rule="same_left_side",
            basis=self._basis(item, symbolic),
            restrictions=item.restrictions,
            detail="same statement as the verified result, in an equivalent form",
        )

    def _named(
        self, a: sp.Expr, b: sp.Expr, item: ClaimEvidence, c: sp.Expr, d: sp.Expr
    ) -> EvidenceMatch | None:
        if item.role not in NAMED_ROLES or not a.is_Symbol or not c.is_Symbol:
            return None
        if a.name.endswith(("_dot", "_ddot")) or a in b.free_symbols:
            return None
        ok, symbolic = _agree(b, d, item.basis == "symbolic")
        if not ok:
            return None
        return EvidenceMatch(
            source=item.source,
            rule="named_quantity",
            basis=self._basis(item, symbolic),
            restrictions=item.restrictions,
            detail=f"the same expression as the verified {c.name} ({item.role.replace('_', ' ')}); "
            "the name is the model's",
        )

    def _proportional(
        self, a: sp.Expr, b: sp.Expr, item: ClaimEvidence, c: sp.Expr, d: sp.Expr
    ) -> EvidenceMatch | None:
        # Only equations of motion and accelerations are rescaled; a momentum or constant is a definition.
        if item.role not in ("equation_of_motion", "acceleration"):
            return None
        ra, rc = a - b, c - d
        if (
            ra == 0
            or rc == 0
            or sp.count_ops(ra) > _MAX_OPS
            or sp.count_ops(rc) > _MAX_OPS
            or not (ra.free_symbols & rc.free_symbols)
        ):
            return None
        dynamic = _dynamic(item.variables)
        try:
            ratio = sp.cancel(sp.together(ra / rc))
            if ratio.free_symbols & {s for s in ratio.free_symbols if s.name in dynamic}:
                ratio = sp.simplify(ratio)
        except Exception:
            return None
        if ratio == 0 or ratio.free_symbols & {s for s in ratio.free_symbols if s.name in dynamic}:
            return None
        if ratio.has(sp.zoo, sp.oo, sp.nan):
            return None
        ok, symbolic = _agree(ra, ratio * rc, item.basis == "symbolic")
        if not ok:
            return None
        numerator, denominator = sp.fraction(ratio)
        conditions, complete = determinant_conditions(numerator * denominator)
        if not complete:
            return None
        return EvidenceMatch(
            source=item.source,
            rule="proportional_residual",
            basis=self._basis(item, symbolic),
            restrictions=[
                *item.restrictions,
                *(c for c in conditions if c not in item.restrictions),
            ],
            detail="the same equation as the verified result up to a nonzero factor of the parameters",
        )


# --- Taylor series ----------------------------------------------------------------------------------------
#
# The claim `f(x) = P(x) + O((x-a)^{n+1})` says three things: P is the Taylor polynomial of f of order n about
# a, the remainder is of the next order, and `f` is the function that was expanded. A series tool result
# establishes exactly that (coefficients against Cauchy-integral estimates of the original function, the
# remainder order measured), so the claim is linked to it only when every part is the same:
#
# * the function: the left side is the expanded expression itself (an exact symbolic zero of the difference) or
#   a label `g(x)` that an earlier formula of the conversation defined as that expression (`g(x) = \sin x`). An
#   undefined label does not say which function is meant and is never linked;
# * the variable and the centre: the polynomial is compared with `sum c_k (x - a)^k` built from the result's
#   coefficients and the remainder must be `(x - a)^{n+1}`, so a changed centre changes both;
# * the order and every coefficient: an exact symbolic zero of the difference, never a sampled agreement;
# * the assumptions: the claim is read under the ones declared in the tool call.
#
# Anything else (another remainder order, a changed coefficient, another function, a label nobody defined) is not
# linked, and a claim with Big-O notation that is not linked stays unchecked: pointwise sampling is never used
# for an asymptotic statement. A match is worded as the tool's series result (numerical evidence, the remainder
# order a measurement), never "proved" and never a bound.
#
# Locations. The centre and the places where the tool measured its error (`x = 0.5`, `|x| = 0.1`, `x = a + d`)
# are parameters of that tool call, not definitions: `x = 0` in "about x = 0" must not make a later `|x| = 0.5`
# "disagree" with it. `SeriesIndex.location` recognises exactly those restatements; the claim checker neither
# learns them nor tests anything against them.

_MAX_SERIES_OPS = 800
_LABEL = re.compile(
    r"([A-Za-z](?:_\{?[A-Za-z0-9,]+\}?)?)\s*(?:\\left)?\(\s*([^(),]+?)\s*(?:\\right)?\)"
)
_LOCATION_TOLERANCE = 1e-9


def _plain(name: str) -> str:
    return name.replace("{", "").replace("}", "").replace(",", "").replace(" ", "")


def function_label(side: str) -> tuple[str, str] | None:
    """`g(x)` -> ("g", "x"): a one-letter name applied to one argument, normalised for comparison."""
    found = _LABEL.fullmatch(side.strip())
    if found is None:
        return None
    return _plain(found.group(1)), _plain(found.group(2))


def _parse(text: str, assumptions: dict[str, list[str]], limits: Limits) -> sp.Expr | None:
    try:
        return parse_latex(text, assumptions, limits)
    except MinervaMathError:
        return None


def _same(a: sp.Expr, b: sp.Expr) -> bool:
    """An exact symbolic zero of the difference (bounded); undecided is not the same."""
    try:
        if a == b or sp.expand(a - b) == 0:
            return True
        if sp.count_ops(a) + sp.count_ops(b) > _MAX_SERIES_OPS:
            return False
        return bool(sp.simplify(a - b) == 0)
    except Exception:
        return False


class _Series:
    """A series record, parsed once."""

    def __init__(
        self,
        record: SeriesEvidence,
        assumptions: dict[str, list[str]],
        expression: sp.Expr,
        variable: sp.Symbol,
        centre: sp.Expr,
        polynomial: sp.Expr,
    ) -> None:
        self.record = record
        self.assumptions = assumptions
        self.expression = expression
        self.variable = variable
        self.centre = centre
        self.polynomial = polynomial


class SeriesIndex:
    """The verified series of a conversation, parsed once per check."""

    def __init__(self, records: list[SeriesEvidence], limits: Limits) -> None:
        from .claims import clean_assumptions

        self.limits = limits
        self.items: list[_Series] = []
        for record in records:
            clean, noncommutative = clean_assumptions(record.assumptions)
            if noncommutative:
                continue
            expression = _parse(record.expression_latex, clean, limits)
            centre = _parse(record.centre_latex, clean, limits)
            if expression is None or centre is None or not centre.is_number:
                continue
            variable = next((s for s in expression.free_symbols if s.name == record.variable), None)
            if not isinstance(variable, sp.Symbol):
                continue
            polynomial: sp.Expr = sp.Integer(0)
            coefficients = [_parse(t.coefficient_latex, clean, limits) for t in record.terms]
            if any(c is None for c in coefficients):
                continue
            for term, coefficient in zip(record.terms, coefficients, strict=True):
                if coefficient is not None:
                    polynomial += coefficient * (variable - centre) ** term.power
            self.items.append(_Series(record, clean, expression, variable, centre, polynomial))

    def location(self, sides: list[str]) -> SeriesEvidence | None:
        """The series whose centre or measurement point this equation restates, if any: `x = a`,
        `x = a + d`, `x = a - d` or `|x - a| = d` (`|x| = d` for a = 0) for a distance d the tool
        measured, and nothing else."""
        if len(sides) != 2 or not self.items:
            return None
        for item in self.items:
            parsed = [_parse(side, item.assumptions, self.limits) for side in sides]
            if parsed[0] is None or parsed[1] is None:
                continue
            for one, other in (parsed, parsed[::-1]):
                if self._is_location(item, one, other):
                    return item.record
        return None

    @staticmethod
    def _is_location(item: _Series, one: sp.Expr, other: sp.Expr) -> bool:
        if not other.is_number or other.is_real is not True:
            return False
        value = float(other)
        centre = float(item.centre)
        distances = item.record.distances
        if one == item.variable:
            return abs(value - centre) <= _LOCATION_TOLERANCE or any(
                abs(value - (centre + sign * d)) <= _LOCATION_TOLERANCE
                for d in distances
                for sign in (1, -1)
            )
        if one == sp.Abs(item.variable - item.centre):
            return any(abs(value - d) <= _LOCATION_TOLERANCE for d in distances)
        return False

    def match(
        self,
        function_side: str,
        polynomial_text: str,
        remainder_text: str,
        definitions: dict[tuple[str, str], str],
    ) -> EvidenceMatch | None:
        """`function_side = polynomial_text + O(remainder_text)` against every verified series, newest first."""
        for item in reversed(self.items):
            if not self._same_function(item, function_side, definitions):
                continue
            polynomial = _parse(polynomial_text, item.assumptions, self.limits)
            remainder = _parse(remainder_text, item.assumptions, self.limits)
            if polynomial is None or remainder is None:
                continue
            order = item.record.order
            if not _same(remainder, (item.variable - item.centre) ** (order + 1)):
                continue
            if not _same(polynomial, item.polynomial):
                continue
            declared = "; ".join(
                f"{name} {', '.join(items)}" for name, items in sorted(item.assumptions.items())
            )
            return EvidenceMatch(
                source=item.record.source,
                rule="series",
                basis="numeric",
                detail=(
                    f"the Taylor series the tool computed: order {order} about "
                    f"{item.record.variable} = {item.record.centre_latex}, remainder of order {order + 1}; "
                    "its coefficients were checked against Cauchy-integral estimates of the function and "
                    "the remainder order was measured (numerical evidence, not a proof or a bound)"
                    + (f"; assumptions of the call: {declared}" if declared else "")
                ),
            )
        return None

    def _same_function(
        self, item: _Series, side: str, definitions: dict[tuple[str, str], str]
    ) -> bool:
        label = function_label(side)
        if label is not None:
            body = definitions.get(label)
            if label[1] != _plain(item.record.variable) or body is None:
                return False
            expression = _parse(body, item.assumptions, self.limits)
        else:
            expression = _parse(side, item.assumptions, self.limits)
        return expression is not None and _same(expression, item.expression)
