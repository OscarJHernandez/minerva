"""Operations that run *inside* the sandbox child process.

Each operation takes and returns plain JSON-compatible dicts so it can cross the process
boundary. `service.Engine` is the typed entry point; call these directly only in tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import sympy as sp

from .claims import check_claims, split_equation
from .distributions.guard import DISTRIBUTION_REFUSAL, contains_distribution
from .errors import LimitError, MinervaMathError, ParseError, UnsupportedInput
from .features import expression_features
from .latexparse import parse_latex
from .latexprint import to_latex
from .models import (
    Applicability,
    ExpressionInput,
    FormulaReading,
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
from .restrictions import domain_restrictions
from .tensor import canonicalize
from .textparse import parse_text
from .verify import verify_equal

_REST = "__minerva_rest__"


def parse_expression(
    data: ExpressionInput,
    limits: Limits,
    denominators: list[sp.Expr] | None = None,
    noncommutative: Iterable[str] = (),
    *,
    allow_distributions: bool = False,
) -> sp.Expr:
    if len(data.input) > limits.max_input_chars:
        raise LimitError("input is too long")
    if data.format == "latex":
        expr = parse_latex(data.input, data.assumptions, limits, denominators, noncommutative)
    else:
        expr = parse_text(data.input, data.assumptions, limits, denominators, noncommutative)
    if not allow_distributions and contains_distribution(expr):
        # Pointwise sampling cannot tell distributions apart (a delta is 0 at every sample point),
        # so every other operation refuses them rather than "verify" a wrong identity.
        raise UnsupportedInput(DISTRIBUTION_REFUSAL)
    return expr


def _failure(exc: MinervaMathError) -> tuple[Status, str]:
    if isinstance(exc, LimitError):
        return Status.RESOURCE_LIMIT, str(exc)
    if isinstance(exc, UnsupportedInput):
        return Status.UNSUPPORTED, str(exc)
    return Status.PARSE_ERROR, str(exc)


def _restriction_note(restrictions: list[str], complete: bool) -> str:
    if not complete:
        return (
            " Where the result is valid could not be determined; do not assume it holds at poles."
        )
    if restrictions:
        return " The result is valid only where " + ", ".join(restrictions) + "."
    return ""


# --- simplify ---------------------------------------------------------------------------
def op_simplify(payload: dict[str, Any]) -> dict[str, Any]:
    limits = Limits(**payload["limits"])
    data = ExpressionInput(**payload["expression"])
    denominators: list[sp.Expr] = []
    try:
        expr = parse_expression(data, limits, denominators)
    except MinervaMathError as exc:
        status, message = _failure(exc)
        return SimplifyResult(status=status, message=message).model_dump(mode="json")

    input_latex = to_latex(expr)
    result = sp.simplify(expr)
    restrictions, complete = domain_restrictions(expr, result, denominators)
    note = _restriction_note(restrictions, complete)
    if result == expr:
        return SimplifyResult(
            status=Status.UNCHANGED,
            input_latex=input_latex,
            result_latex=input_latex,
            result_text=str(expr),
            restrictions_latex=restrictions,
            restrictions_complete=complete,
            message="No simpler form was found. The expression is returned unchanged." + note,
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
        result_latex=to_latex(result),
        result_text=str(result),
        verification=verification,
        restrictions_latex=restrictions,
        restrictions_complete=complete,
        message=(
            "Simplified with SymPy and verified"
            + (
                " (symbolic zero and numeric check)."
                if verification.symbolic_zero
                else " numerically."
            )
            + note
        ),
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


# --- series ----------------------------------------------------------------------------
def op_series(payload: dict[str, Any]) -> dict[str, Any]:
    from .series import series

    result = series(
        ExpressionInput(**payload["expression"]),
        str(payload["variable"]),
        str(payload.get("point", "0")),
        int(payload["order"]),
        Limits(**payload["limits"]),
    )
    return result.model_dump(mode="json")


# --- integrate -------------------------------------------------------------------------
def op_matrix_calculus(payload: dict[str, Any]) -> dict[str, Any]:
    from .matrix import MatrixCalculusInput, matrix_calculus

    result = matrix_calculus(MatrixCalculusInput(**payload["input"]), Limits(**payload["limits"]))
    return result.model_dump(mode="json")


def op_distribution(payload: dict[str, Any]) -> dict[str, Any]:
    from .distributions.models import DistributionInput
    from .distributions.operation import distribution

    result = distribution(DistributionInput(**payload["input"]), Limits(**payload["limits"]))
    return result.model_dump(mode="json")


def op_integrate(payload: dict[str, Any]) -> dict[str, Any]:
    from .integrate import integrate

    result = integrate(
        ExpressionInput(**payload["expression"]),
        str(payload["variable"]),
        Limits(**payload["limits"]),
        lower=payload.get("lower"),
        upper=payload.get("upper"),
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


# --- expansion about an equilibrium -------------------------------------------------------
def op_expand_motion(payload: dict[str, Any]) -> dict[str, Any]:
    from .expansion import expand_motion

    report = expand_motion(
        str(payload["lagrangian"]),
        [str(c) for c in payload["coordinates"]],
        str(payload.get("time", "t")),
        payload.get("assumptions", {}),
        {str(k): str(v) for k, v in payload.get("equilibrium", {}).items()},
        int(payload["order"]),
        Limits(**payload["limits"]),
    )
    return report.model_dump(mode="json")


# --- perturbation solution -------------------------------------------------------------------
def op_perturbation_solve(payload: dict[str, Any]) -> dict[str, Any]:
    from .perturbation import CompareRequest, perturbation_solve

    compare = payload.get("compare")
    small = payload.get("small_parameter")
    report = perturbation_solve(
        str(payload["lagrangian"]),
        [str(c) for c in payload["coordinates"]],
        str(payload.get("time", "t")),
        payload.get("assumptions", {}),
        str(small) if small else None,
        str(payload.get("equilibrium", "0")),
        int(payload["order"]),
        CompareRequest(**compare) if compare else None,
        Limits(**payload["limits"]),
    )
    return report.model_dump(mode="json")


# --- search for constants of motion ---------------------------------------------------
def op_search_invariants(payload: dict[str, Any]) -> dict[str, Any]:
    from .invariants import search_invariants

    report = search_invariants(
        str(payload["lagrangian"]),
        [str(c) for c in payload["coordinates"]],
        [str(a) for a in payload.get("angles", [])],
        int(payload.get("degree", 2)),
        int(payload.get("power_bound", 3)),
        int(payload.get("harmonics", 1)),
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
    relations: list[str] = [str(r) for r in payload.get("relations", [])]
    formulas = [str(f) for f in payload["formulas"]]
    results = check_claims(formulas, assumptions, relations, limits)
    return {"results": [r.model_dump() for r in results]}


# --- formulas transcribed from an image ----------------------------------------------
def op_read_formulas(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse each transcribed formula strictly and show how it was read. Never simplifies."""
    limits = Limits(**payload["limits"])
    results: list[dict[str, Any]] = []
    for latex in payload["formulas"]:
        text = str(latex)
        try:
            if len(text) > limits.max_input_chars:
                raise LimitError("input is too long")
            sides = split_equation(text) or [text]
            parsed = [parse_latex(side, {}, limits) for side in sides]
            read_as = " = ".join(to_latex(p) for p in parsed)
            reading = FormulaReading(status=Status.OK, read_as=read_as)
        except MinervaMathError as exc:
            status, message = _failure(exc)
            reading = FormulaReading(status=status, message=message)
        except Exception:
            reading = FormulaReading(status=Status.PARSE_ERROR, message="could not be read")
        results.append(reading.model_dump(mode="json"))
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
        "input_latex": to_latex(expr),
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


@dataclass(frozen=True)
class Rewrite:
    """One place where a rule matches: the matched node, the rewritten expression, unmet conditions."""

    node: sp.Expr
    rewritten: sp.Expr
    unmet: list[RuleCondition]
    # Operations in what the pattern variables were bound to: when two rules give the same result,
    # the one that needed no negation or other work in its bindings is the one to cite.
    binding_cost: int = 0


def can_reverse(rule: Rule) -> bool:
    """A rule may also be read right to left only if its right side determines every variable.

    A right side that is a bare number or symbol (`1`, `a`) would match anything, and one that
    omits a variable (`sin(a)**2 + cos(a)**2 -> 1` read backwards) cannot bind it.
    """
    rhs = parse_text(rule.rhs)
    if rhs.is_Number or rhs.is_Symbol:
        return False
    return {s.name for s in rhs.free_symbols} >= set(rule.variables)


def rule_rewrites(
    expr: sp.Expr, rule: Rule, *, reverse: bool = False, strict: bool = False
) -> Iterator[Rewrite]:
    """Every place `rule` matches inside `expr`, after the match filters, in tree order.

    Conditions are checked here (`unmet` is empty when they are provable from the declared
    assumptions) but the rewrite is NOT verified: callers must call `verify_equal` before
    accepting one. With `reverse=True` the right side is the pattern. With `strict=True` a sum or
    product pattern matches only a sum or product node: without it `cos(y)**2` "matches"
    `sin(a)**2 + cos(a)**2` with the leftover term `-sin(y)**2` absorbed by `rest`, which is true but
    not what a chain of identities should be built from. `find_identities` keeps the lenient default.
    """
    lhs, rhs, wilds = _pattern_parts(rule)
    if reverse:
        lhs, rhs = rhs, lhs
    rest = sp.Wild(_REST)
    if lhs.is_Add:
        pattern, replacement = lhs + rest, rhs + rest
    elif lhs.is_Mul:
        pattern, replacement = lhs * rest, rhs * rest
    else:
        pattern, replacement = lhs, rhs

    seen: set[sp.Basic] = set()
    for node in sp.preorder_traversal(expr):
        if not isinstance(node, sp.Expr) or node in seen:
            continue
        seen.add(node)
        if strict and ((lhs.is_Add and not node.is_Add) or (lhs.is_Mul and not node.is_Mul)):
            continue
        bindings = node.match(pattern)
        if (
            bindings is None
            or _invents_structure(bindings, expr)
            or _binds_fraction_of_multiple(lhs, wilds, bindings)
        ):
            continue
        if reverse and any(key != rest and value == 0 for key, value in bindings.items()):
            continue  # `2 sin x cos x` is not "sin a + sin b" with b = 0: that variable is invented
        instantiated = replacement.xreplace(bindings)
        if instantiated == node:
            continue  # the rewrite would be a no-op (e.g. a**1), not a real application
        unmet = _unmet(rule, bindings, wilds)
        rewritten = expr.xreplace({node: instantiated})
        if rewritten == expr and not unmet:
            continue
        binding_cost = sum(
            int(sp.count_ops(value)) for key, value in bindings.items() if key != rest
        )
        yield Rewrite(node=node, rewritten=rewritten, unmet=unmet, binding_cost=binding_cost)


def apply_rule(expr: sp.Expr, rule: Rule) -> RuleApplication:
    """Find where `rule` matches inside `expr`, check its conditions, apply, verify."""
    first_blocked: RuleApplication | None = None
    for rewrite in rule_rewrites(expr, rule):
        matched_latex = to_latex(rewrite.node)
        if rewrite.unmet:
            first_blocked = first_blocked or RuleApplication(
                rule_id=rule.id,
                applicability=Applicability.CONDITIONS_NOT_SATISFIED,
                matched_subexpression_latex=matched_latex,
                unmet_conditions=rewrite.unmet,
                message="The pattern matches, but the conditions are not provable "
                "from the declared assumptions, so the rule is not applied.",
            )
            continue
        verification = verify_equal(expr, rewrite.rewritten, symbolic=False)
        restrictions, complete = domain_restrictions(expr, rewrite.rewritten)
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
            rewritten_latex=to_latex(rewrite.rewritten),
            rewritten_text=str(rewrite.rewritten),
            verification=verification,
            restrictions_latex=restrictions,
            restrictions_complete=complete,
            message="Matched, conditions hold, and the rewrite was verified numerically."
            + _restriction_note(restrictions, complete),
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
        input_latex=to_latex(expr),
        applications=applications,
        message="",
    ).model_dump(mode="json")


# --- verified chains of rewrites ----------------------------------------------------------
def op_transform(payload: dict[str, Any]) -> dict[str, Any]:
    from .transform import transform

    result = transform(
        ExpressionInput(**payload["expression"]),
        [Rule(**r) for r in payload["rules"]],
        str(payload.get("goal", "simplify")),
        [str(t) for t in payload.get("target", [])],
        Limits(**payload["limits"]),
        width=int(payload.get("width", 4)),
        depth=int(payload.get("depth", 4)),
    )
    if result.status in (Status.OK, Status.UNCHANGED) and result.result_text:
        denominators: list[sp.Expr] = []
        try:
            limits = Limits(**payload["limits"])
            before = parse_expression(
                ExpressionInput(**payload["expression"]), limits, denominators
            )
            after = parse_text(
                result.result_text, ExpressionInput(**payload["expression"]).assumptions, limits
            )
            result.restrictions_latex, result.restrictions_complete = domain_restrictions(
                before, after, denominators
            )
        except Exception:  # undetermined is reported, never dropped
            result.restrictions_complete = False
        if not result.restrictions_complete:
            result.message += _restriction_note([], False)
        elif result.restrictions_latex:
            result.message += _restriction_note(result.restrictions_latex, True)
    return result.model_dump(mode="json")


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
        result_latex=to_latex(canonical),
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
    "transform": op_transform,
    "features": op_features,
    "check_claims": op_check_claims,
    "read_formulas": op_read_formulas,
    "solve": op_solve,
    "differentiate": op_differentiate,
    "series": op_series,
    "integrate": op_integrate,
    "distribution": op_distribution,
    "matrix_calculus": op_matrix_calculus,
    "lagrangian": op_lagrangian,
    "expand_motion": op_expand_motion,
    "perturbation_solve": op_perturbation_solve,
    "search_invariants": op_search_invariants,
    "integrate_motion": op_integrate_motion,
    "dimensional_analysis": op_dimensional_analysis,
    "check_units": op_check_units,
    "tensor_canonicalize": op_tensor_canonicalize,
}
