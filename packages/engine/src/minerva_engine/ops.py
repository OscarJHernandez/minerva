"""Operations that run *inside* the sandbox child process.

Each operation takes and returns plain JSON-compatible dicts so it can cross the process
boundary. `service.Engine` is the typed entry point; call these directly only in tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import sympy as sp

from .claims import ClaimCheck, check_claim
from .errors import LimitError, MinervaMathError, ParseError
from .features import expression_features
from .latexparse import parse_latex
from .models import (
    Applicability,
    ExpressionInput,
    IdentitiesResult,
    Limits,
    Rule,
    RuleApplication,
    RuleCondition,
    SimplifyResult,
    Status,
    TensorInput,
    TensorResult,
    Verification,
)
from .tensor import canonicalize
from .textparse import parse_text
from .verify import verify_equal

_REST = "__minerva_rest__"


def parse_expression(data: ExpressionInput, limits: Limits) -> sp.Expr:
    if len(data.input) > limits.max_input_chars:
        raise LimitError("input is too long")
    if data.format == "latex":
        return parse_latex(data.input, data.assumptions, limits)
    return parse_text(data.input, data.assumptions, limits)


def _failure(exc: MinervaMathError) -> tuple[Status, str]:
    if isinstance(exc, LimitError):
        return Status.RESOURCE_LIMIT, str(exc)
    return Status.PARSE_ERROR, str(exc)


# --- simplify ---------------------------------------------------------------------------
def op_simplify(payload: dict[str, Any]) -> dict[str, Any]:
    limits = Limits(**payload["limits"])
    data = ExpressionInput(**payload["expression"])
    try:
        expr = parse_expression(data, limits)
    except MinervaMathError as exc:
        status, message = _failure(exc)
        return SimplifyResult(status=status, message=message).model_dump(mode="json")

    input_latex = sp.latex(expr)
    result = sp.simplify(expr)
    if result == expr:
        return SimplifyResult(
            status=Status.UNCHANGED,
            input_latex=input_latex,
            result_latex=input_latex,
            result_text=str(expr),
            message="No simpler form was found. The expression is returned unchanged.",
        ).model_dump(mode="json")

    verification = verify_equal(expr, result)
    if not verification.passed:
        return SimplifyResult(
            status=Status.NOT_VERIFIED,
            input_latex=input_latex,
            verification=verification,
            message=(
                "A different form was produced but it could not be verified, "
                "so it is not shown. " + verification.note
            ),
        ).model_dump(mode="json")

    return SimplifyResult(
        status=Status.OK,
        input_latex=input_latex,
        result_latex=sp.latex(result),
        result_text=str(result),
        verification=verification,
        message="Simplified with SymPy and verified.",
    ).model_dump(mode="json")


# --- solve ----------------------------------------------------------------------------
def op_solve(payload: dict[str, Any]) -> dict[str, Any]:
    from .solve import solve_polynomial

    limits = Limits(**payload["limits"])
    report = solve_polynomial(
        str(payload["equation"]),
        payload.get("variable"),
        payload.get("assumptions", {}),
        limits,
    )
    return report.model_dump(mode="json")


# --- differentiate ---------------------------------------------------------------------
def op_differentiate(payload: dict[str, Any]) -> dict[str, Any]:
    from .differentiate import DiffVariable, differentiate

    result = differentiate(
        ExpressionInput(**payload["expression"]),
        [DiffVariable(**v) for v in payload["variables"]],
        Limits(**payload["limits"]),
    )
    return result.model_dump(mode="json")


# --- lagrangian ------------------------------------------------------------------------
def op_lagrangian(payload: dict[str, Any]) -> dict[str, Any]:
    from .lagrangian import analyze_lagrangian

    report = analyze_lagrangian(
        str(payload["lagrangian"]),
        [str(c) for c in payload["coordinates"]],
        str(payload.get("time", "t")),
        payload.get("assumptions", {}),
        Limits(**payload["limits"]),
    )
    return report.model_dump(mode="json")


# --- dimensional analysis ------------------------------------------------------------------
def op_dimensional_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    from .pi_groups import ConstantCandidate, QuantitySpec, analyze_dimensions

    report = analyze_dimensions(
        [QuantitySpec(**q) for q in payload["quantities"]],
        payload.get("target"),
        [str(r) for r in payload.get("repeating", [])],
        [ConstantCandidate(**c) for c in payload.get("candidates", [])],
        Limits(**payload["limits"]),
    )
    return report.model_dump(mode="json")


def op_check_units(payload: dict[str, Any]) -> dict[str, Any]:
    from .pi_groups import QuantitySpec, check_units

    report = check_units(
        str(payload["equation"]),
        [QuantitySpec(**q) for q in payload["quantities"]],
        Limits(**payload["limits"]),
    )
    return report.model_dump(mode="json")


# --- numerical motion ------------------------------------------------------------------
def op_integrate_motion(payload: dict[str, Any]) -> dict[str, Any]:
    from .motion import MotionReport, integrate_motion, parse_request

    limits = Limits(**payload["limits"])
    parsed = parse_request({k: v for k, v in payload.items() if k != "limits"})
    report = parsed if isinstance(parsed, MotionReport) else integrate_motion(parsed, limits)
    return report.model_dump(mode="json")


# --- claims written by the language model --------------------------------------------
def op_check_claims(payload: dict[str, Any]) -> dict[str, Any]:
    limits = Limits(**payload["limits"])
    assumptions: dict[str, list[str]] = payload.get("assumptions", {})
    results: list[dict[str, Any]] = []
    for latex in payload["formulas"]:
        try:
            results.append(check_claim(str(latex), assumptions, limits).model_dump())
        except Exception:
            results.append(
                ClaimCheck(status="unparsed", detail="could not be checked").model_dump()
            )
    return {"results": results}


# --- features --------------------------------------------------------------------------
def op_features(payload: dict[str, Any]) -> dict[str, Any]:
    limits = Limits(**payload["limits"])
    data = ExpressionInput(**payload["expression"])
    try:
        expr = parse_expression(data, limits)
    except MinervaMathError as exc:
        status, message = _failure(exc)
        return {"status": status.value, "message": message, "features": [], "input_latex": None}
    return {
        "status": Status.OK.value,
        "message": "",
        "features": sorted(expression_features(expr)),
        "input_latex": sp.latex(expr),
    }


# --- identities -------------------------------------------------------------------------
def _pattern_parts(rule: Rule) -> tuple[sp.Expr, sp.Expr, dict[str, sp.Wild]]:
    wilds = {name: sp.Wild(name) for name in rule.variables}
    lhs = parse_text(rule.lhs)
    rhs = parse_text(rule.rhs)
    for expr in (lhs, rhs):
        unknown = {s.name for s in expr.free_symbols} - set(rule.variables)
        if unknown:
            raise ParseError(f"rule {rule.id} uses undeclared variable(s) {sorted(unknown)}")
    mapping = {sp.Symbol(name): wild for name, wild in wilds.items()}
    return lhs.xreplace(mapping), rhs.xreplace(mapping), wilds


def _unmet(
    rule: Rule, bindings: dict[sp.Basic, sp.Basic], wilds: dict[str, sp.Wild]
) -> list[RuleCondition]:
    unmet: list[RuleCondition] = []
    for cond in rule.conditions:
        value = bindings.get(wilds[cond.variable])
        if value is None or getattr(value, f"is_{cond.assumption}", None) is not True:
            unmet.append(cond)
    return unmet


def _invents_structure(bindings: dict[sp.Basic, sp.Basic], expr: sp.Expr) -> bool:
    """True if a match only works by binding variables to structure absent from `expr`.

    SymPy's matcher can solve `a**2 - b**2 = x + y` with `a = sqrt(y)`, `b = sqrt(-x)`, or
    `log(a/b) = log(x*y)` with `b = 1/y`, or `a**2 - b**2 = x**2 + y**2` with `b = I*y`.
    Those rewrites can be valid but are not what a reader means by "this identity applies",
    so matches that invent roots, reciprocals or the imaginary unit are discarded. A
    reciprocal or root that is already in the expression is of course allowed.
    """
    for value in bindings.values():
        for node in sp.preorder_traversal(value):
            if node == sp.I and not expr.has(sp.I):
                return True
            if (
                isinstance(node, sp.Pow)
                and not (node.exp.is_Integer and node.exp.is_positive)
                and not expr.has(node)
            ):
                return True
    return False


def _binds_fraction_of_multiple(
    lhs: sp.Expr, wilds: dict[str, sp.Wild], bindings: dict[sp.Basic, sp.Basic]
) -> bool:
    """True if a variable that the rule only uses as `n*w` (integer n >= 2) is bound to a fraction.

    `sin(2*a)` matches `sin(x)` with `a = x/2` and `sin(3*x)` with `a = 3*x/2`. Those rewrites
    are true, but nobody reading "double angle" means them, and every multiple-angle rule added to
    the handbook would multiply such matches. Variables that also occur bare in the pattern (the
    `a` in `sin(a + b)`) are not restricted, and integer multiples (`a = x`, `a = 3*x`,
    `a = x + y`) still match. Fractional arguments such as `sin(2*x/3)` are conservatively not
    offered as double angles.
    """
    nodes = list(sp.preorder_traversal(lhs))
    for wild in wilds.values():
        occurrences = sum(1 for node in nodes if node == wild)
        multiples = sum(
            1
            for node in nodes
            if node.is_Mul
            and len(node.args) == 2
            and node.args[1] == wild
            and node.args[0].is_Integer
            and node.args[0] >= 2
        )
        value = bindings.get(wild)
        if value is None or occurrences == 0 or occurrences != multiples:
            continue
        for term in sp.Add.make_args(value):
            coefficient = term.as_coeff_Mul()[0]
            if coefficient.is_Rational and not coefficient.is_Integer:
                return True
    return False


def apply_rule(expr: sp.Expr, rule: Rule) -> RuleApplication:
    """Find where `rule` matches inside `expr`, check its conditions, apply, verify."""
    lhs, rhs, wilds = _pattern_parts(rule)
    rest = sp.Wild(_REST)
    if lhs.is_Add:
        pattern, replacement = lhs + rest, rhs + rest
    elif lhs.is_Mul:
        pattern, replacement = lhs * rest, rhs * rest
    else:
        pattern, replacement = lhs, rhs

    first_blocked: RuleApplication | None = None
    for node in sp.preorder_traversal(expr):
        if not isinstance(node, sp.Expr):
            continue
        bindings = node.match(pattern)
        if (
            bindings is None
            or _invents_structure(bindings, expr)
            or _binds_fraction_of_multiple(lhs, wilds, bindings)
        ):
            continue
        instantiated = replacement.xreplace(bindings)
        if instantiated == node:
            continue  # the rewrite would be a no-op (e.g. a**1), not a real application
        unmet = _unmet(rule, bindings, wilds)
        matched_latex = sp.latex(node)
        if unmet:
            first_blocked = first_blocked or RuleApplication(
                rule_id=rule.id,
                applicability=Applicability.CONDITIONS_NOT_SATISFIED,
                matched_subexpression_latex=matched_latex,
                unmet_conditions=unmet,
                message="The pattern matches, but the conditions are not provable "
                "from the declared assumptions, so the rule is not applied.",
            )
            continue
        rewritten = expr.xreplace({node: instantiated})
        if rewritten == expr:
            continue
        verification = verify_equal(expr, rewritten, symbolic=False)
        if not verification.passed:
            return RuleApplication(
                rule_id=rule.id,
                applicability=Applicability.NOT_VERIFIED,
                matched_subexpression_latex=matched_latex,
                verification=verification,
                message="The rule matched but the rewrite could not be verified; not shown.",
            )
        return RuleApplication(
            rule_id=rule.id,
            applicability=Applicability.APPLICABLE,
            matched_subexpression_latex=matched_latex,
            rewritten_latex=sp.latex(rewritten),
            verification=verification,
            message="Matched, conditions hold, and the rewrite was verified.",
        )
    if first_blocked is not None:
        return first_blocked
    return RuleApplication(
        rule_id=rule.id,
        applicability=Applicability.NO_MATCH,
        message="Related, but the pattern does not occur in this expression.",
    )


def op_identities(payload: dict[str, Any]) -> dict[str, Any]:
    limits = Limits(**payload["limits"])
    data = ExpressionInput(**payload["expression"])
    rules = [Rule(**r) for r in payload["rules"]]
    try:
        expr = parse_expression(data, limits)
    except MinervaMathError as exc:
        status, message = _failure(exc)
        return IdentitiesResult(status=status, message=message).model_dump(mode="json")

    applications: list[RuleApplication] = []
    # One sandbox timeout covers every rule. Stop early and say so, instead of letting one slow
    # rule turn the whole request into `resource_limit` with no partial results.
    deadline = time.monotonic() + 0.7 * limits.timeout_seconds
    for rule in rules:
        if time.monotonic() > deadline:
            applications.append(
                RuleApplication(
                    rule_id=rule.id,
                    applicability=Applicability.NO_MATCH,
                    message="Skipped: the time budget for this request was used up.",
                )
            )
            continue
        try:
            applications.append(apply_rule(expr, rule))
        except MinervaMathError as exc:
            applications.append(
                RuleApplication(
                    rule_id=rule.id,
                    applicability=Applicability.NO_MATCH,
                    message=f"Rule could not be evaluated: {exc}",
                )
            )
    return IdentitiesResult(
        status=Status.OK,
        input_latex=sp.latex(expr),
        applications=applications,
        message="",
    ).model_dump(mode="json")


# --- tensor canonicalization -------------------------------------------------------------
def op_tensor_canonicalize(payload: dict[str, Any]) -> dict[str, Any]:
    limits = Limits(**payload["limits"])
    try:
        data = TensorInput(**payload["tensor"])
        parsed, canonical = canonicalize(data, limits)
    except MinervaMathError as exc:
        status, message = _failure(exc)
        return TensorResult(status=status, message=message).model_dump(mode="json")
    changed = canonical != parsed
    return TensorResult(
        status=Status.OK if changed else Status.UNCHANGED,
        input_text=str(parsed),
        result_text=str(canonical),
        result_latex=sp.latex(canonical),
        independent_verification=False,
        message=(
            "Canonical form from SymPy (Butler-Portugal), using only the declared monoterm "
            "symmetries. Multi-term identities (e.g. Bianchi) are not applied. "
            "Not independently verified."
        ),
    ).model_dump(mode="json")


# --- rule self-verification (used by the knowledge-base test suite and build) ----------
def verify_rule(rule: Rule) -> Verification:
    """Check `lhs == rhs` with variables carrying the rule's declared conditions."""
    assumptions: dict[str, list[str]] = {}
    for cond in rule.conditions:
        assumptions.setdefault(cond.variable, []).append(cond.assumption)
    lhs = parse_text(rule.lhs, assumptions)
    rhs = parse_text(rule.rhs, assumptions)
    return verify_equal(lhs, rhs)


OPERATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "simplify": op_simplify,
    "identities": op_identities,
    "features": op_features,
    "check_claims": op_check_claims,
    "solve": op_solve,
    "differentiate": op_differentiate,
    "lagrangian": op_lagrangian,
    "integrate_motion": op_integrate_motion,
    "dimensional_analysis": op_dimensional_analysis,
    "check_units": op_check_units,
    "tensor_canonicalize": op_tensor_canonicalize,
}
