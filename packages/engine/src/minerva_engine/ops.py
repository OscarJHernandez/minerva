"""Operations that run *inside* the sandbox child process.

Each operation takes and returns plain JSON-compatible dicts so it can cross the process
boundary. `service.Engine` is the typed entry point; call these directly only in tests.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy as sp

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


def _introduces_roots(bindings: dict[sp.Basic, sp.Basic], expr: sp.Expr) -> bool:
    """True if a match only works by binding variables to roots absent from `expr`.

    SymPy's matcher can solve `a**2 - b**2 = x + y` with `a = sqrt(y)`, `b = sqrt(-x)`.
    That rewrite is valid but is not what a reader means by "this identity applies", so
    matches that invent fractional powers are discarded.
    """
    for value in bindings.values():
        for node in sp.preorder_traversal(value):
            if (
                isinstance(node, sp.Pow)
                and node.exp.is_Rational
                and not node.exp.is_Integer
                and not expr.has(node)
            ):
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
        if bindings is None or _introduces_roots(bindings, expr):
            continue
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
        rewritten = expr.xreplace({node: replacement.xreplace(bindings)})
        if rewritten == expr:
            continue
        verification = verify_equal(expr, rewritten)
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
    for rule in rules:
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
    "tensor_canonicalize": op_tensor_canonicalize,
}
