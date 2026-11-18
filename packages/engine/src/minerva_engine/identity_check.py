"""Is this stated identity true? A verdict the user asked for, with the witness when it is not.

`claims.py` checks equations a model wrote in its prose and never calls one false unless the sides differ by a
nonzero constant: it cannot tell a false identity from an equation to solve, a definition or an equation of motion,
so a formula that fails at generic points stays "unchecked". Here the USER asked, explicitly, whether one stated
equation holds as an identity. That changes what an honest answer is, not how it is established:

* `proved_equal`: the sampled points agree **and** the symbolic zero test found zero (`Verification.symbolic_zero`).
* `numerically_equal`: no disagreement was found at the random points or at the targeted probes. Evidence, never
  proof, and worded that way.
* `conditional`: false for the declared assumptions but true once symbols are assumed real or positive (`claims._relaxed`);
  the condition is stated, and the counterexample under the declared assumptions is attached when one is found.
* `not_identity`: a counterexample was found by `counterexample.find_counterexample`: an exact rational point where the
  sides differ by an amount that survives 60 and 120 digits (cancellation is counted, never reported). The point is
  the evidence; the equation may still have solutions, and the result says so.

What it never does: call a *relation* false (a symbol alone on one side, a time derivative, a definition), test
matrices, vectors, distributions or sums by sampling numbers (they are `unsupported` with the reason from the claim
checker), or accept a disagreement it could not confirm (`not_verified`). The statuses are the ordinary ones; the
verdict only says what an `ok` result established. A `Counterexample` holds the user's expressions: it is returned to the
caller and never logged (`test_counterexample_privacy.py`).
"""

from __future__ import annotations

from typing import Literal

import sympy as sp
from pydantic import BaseModel

from .claims import ClaimChecker, clean_assumptions, split_equation
from .counterexample import Budget, Counterexample, find_counterexample
from .errors import MinervaMathError
from .latexparse import parse_latex
from .latexprint import to_latex
from .models import Limits, Status, Verification
from .verify import verify_with_witness

Verdict = Literal["proved_equal", "numerically_equal", "conditional", "not_identity"]


class IdentityCounterexample(BaseModel):
    """A point where the two sides differ, with exact values, as LaTeX for the user's own card."""

    substitution: dict[str, str]  # symbol -> exact value (a rational, or a complex rational)
    lhs_value: str
    rhs_value: str
    residual: str
    probe: str  # random | boundary | branch | special | scale


class IdentityResult(BaseModel):
    status: Status
    verdict: Verdict | None = None
    input_latex: str | None = None
    lhs_latex: str | None = None
    rhs_latex: str | None = None
    verification: Verification | None = None
    condition: str | None = None
    counterexample: IdentityCounterexample | None = None
    points_probed: int = 0
    message: str = ""
    duration_ms: int = 0


def _value_latex(expr: sp.Expr, values: dict[sp.Symbol, sp.Expr], fallback: str) -> str:
    """The side at the point, to 10 digits, in LaTeX (our own arithmetic on the exact point)."""
    try:
        return str(to_latex(sp.N(expr.xreplace(values), 10)))
    except (ArithmeticError, TypeError, ValueError, AttributeError):
        return fallback


def _witness(found: Counterexample, lhs: sp.Expr, rhs: sp.Expr) -> IdentityCounterexample:
    return IdentityCounterexample(
        substitution={str(sym.name): str(to_latex(value)) for sym, value in found.values.items()},
        lhs_value=_value_latex(lhs, found.values, found.value_a),
        rhs_value=_value_latex(rhs, found.values, found.value_b),
        residual=found.residual,
        probe=str(found.probe),
    )


def check_identity(
    latex: str,
    assumptions: dict[str, list[str]],
    limits: Limits,
    matrices: list[str] | None = None,
) -> IdentityResult:
    """One equation `lhs = rhs`, read by the strict parser and judged as an identity (see the module docstring)."""
    if len(latex) > limits.max_input_chars:
        return IdentityResult(status=Status.RESOURCE_LIMIT, message="The equation is too long.")
    clean, noncommutative = clean_assumptions(assumptions)
    claim = ClaimChecker(clean, limits, frozenset(matrices or []) | noncommutative).check(latex)
    if claim.status == "not_an_equation":
        return IdentityResult(
            status=Status.PARSE_ERROR,
            input_latex=latex,
            message="Send one equation with a single `=`, for example `\\sin^2 x + \\cos^2 x = 1`.",
        )
    if claim.status == "unparsed":
        return IdentityResult(status=Status.PARSE_ERROR, input_latex=latex, message=claim.detail)
    if claim.status == "unverified":
        # A relation, a matrix or vector statement, a distribution, a sum, a reserved symbol: not decided by
        # sampling numbers, and never called false. The reason is the claim checker's own.
        return IdentityResult(
            status=Status.UNSUPPORTED,
            input_latex=latex,
            message=(
                f"Not decided: {claim.detail}. This is not called true or false: sampling numbers "
                "cannot settle it."
            ),
        )
    sides = split_equation(latex) or []
    try:
        parsed = [parse_latex(side, clean, limits) for side in sides]
    except MinervaMathError as exc:
        return IdentityResult(status=Status.PARSE_ERROR, input_latex=latex, message=str(exc))
    lhs_latex, rhs_latex = (
        str(to_latex(parsed[0])),
        " = ".join(str(to_latex(p)) for p in parsed[1:]),
    )
    base = {"input_latex": latex, "lhs_latex": lhs_latex, "rhs_latex": rhs_latex}

    if claim.status in ("holds", "supported", "conditional"):
        verifications = []
        agreed = True
        for other in parsed[1:]:
            verification, found = verify_with_witness(parsed[0], other, symbolic=True)
            if found is not None:  # the random points agreed and a targeted probe did not
                return IdentityResult(
                    status=Status.OK,
                    verdict="not_identity",
                    verification=verification,
                    counterexample=_witness(found, parsed[0], other),
                    message=_not_identity_message(),
                    **base,
                )
            if not verification.passed:
                agreed = False
                break
            verifications.append(verification)
        if agreed:
            # `conditional` here means equal as functions but not defined at the same points (a cancelled
            # factor): the equality is real and the points where it does not apply are stated.
            proved = all(v.symbolic_zero for v in verifications)
            note = (
                "The symbolic zero test found zero and the sampled points agree."
                if proved
                else "No disagreement was found at random points or at targeted probes (boundaries, "
                "branch cuts, large integers). That is numerical evidence, not a proof."
            )
            return IdentityResult(
                status=Status.OK,
                verdict="proved_equal" if proved else "numerically_equal",
                verification=verifications[0],
                condition=claim.detail if claim.status == "conditional" else None,
                message=note + (f" {claim.detail}." if claim.status == "conditional" else ""),
                **base,
            )
        if claim.status != "conditional":
            return IdentityResult(
                status=Status.NOT_VERIFIED,
                verification=verification,
                message="The sides could not be shown equal or different. " + verification.note,
                **base,
            )

    # fails / not_identity / conditional: a disagreement at the sampled points. Find a point to show.
    searched = find_counterexample(parsed[0], parsed[1], budget=Budget(points=160, seconds=4.0))
    found = searched.counterexample
    if found is None:
        return IdentityResult(
            status=Status.NOT_VERIFIED,
            points_probed=searched.probes_tried,
            message=(
                "The sides disagree at some sampled points, but no counterexample was confirmed at 60 "
                "and 120 digits (cancellation or a domain difference is possible), so it is not shown."
            ),
            **base,
        )
    if claim.status == "conditional":
        return IdentityResult(
            status=Status.OK,
            verdict="conditional",
            condition=claim.detail,
            counterexample=_witness(found, parsed[0], parsed[1]),
            points_probed=searched.probes_tried,
            message=(
                "Not an identity for the assumptions given; it holds "
                f"{_if_clause(claim.detail)}. The counterexample is a point the stated assumptions allow."
            ),
            **base,
        )
    return IdentityResult(
        status=Status.OK,
        verdict="not_identity",
        counterexample=_witness(found, parsed[0], parsed[1]),
        points_probed=searched.probes_tried,
        message=_not_identity_message(),
        **base,
    )


def _if_clause(detail: str) -> str:
    """`holds if x is positive` from the claim checker's own sentence ("false in general, but holds if ...")."""
    _, sep, tail = detail.partition("holds if")
    return f"if{tail}" if sep else "under weaker assumptions"


def _not_identity_message() -> str:
    return (
        "Not an identity: the two sides differ at the point shown (exact values, checked at 60 and 120 "
        "digits). The equation may still hold for some values; this says it does not hold for all."
    )
