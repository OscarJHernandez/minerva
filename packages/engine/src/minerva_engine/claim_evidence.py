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

A match says "the same statement a verified result made". It does not verify the claim by itself and never
widens what the tool established: the restrictions of the result travel with the match. Anything else (a
rearranged form, terms moved across the equals sign, solved for another symbol) is not matched and stays
exactly as unchecked as before.
"""

from __future__ import annotations

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


class EvidenceMatch(BaseModel):
    source: str
    rule: Literal["same_left_side", "proportional_residual", "named_quantity"]
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
