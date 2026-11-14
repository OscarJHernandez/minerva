"""Controlled steps on vector expressions: list, apply an identity, substitute a premise, apply an
operator to both sides of an equation, and a bounded search toward a target.

Every move is a *proposal* that is then checked by something other than the move itself:

* An identity (from the handbook, `rules.py`) is applied at a chosen part. The whole expression before
  and after is compared by evaluating their difference with both legs of the vector verifier
  (`Context.residual`): the lowered components must be exactly zero and the independent Cartesian reading
  must agree. An identity that needs `c2` is `blocked` unless every field in the expression was declared
  that smooth by the user. A rewrite whose difference is not zero is `not_verified` and never shown.
* A premise (a field equation the USER supplied) replaces one side of itself by the other. The result is
  `conditional` and names the premise: Minerva never checks that a premise is true and never calls the
  result an identity.
* An operator applied to both sides of an equation is an implication from the given equation, verified by
  the linearity residual `Op(L) - Op(R) - Op(L - R)`.
* `derive` searches, within fixed bounds, for a chain of identity and premise steps from the expression to a
  target (or to the smallest verified form). "Not found" means nothing within the bounds, not that no
  derivation exists.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import sympy as sp

from ..errors import ParseError
from ..models import Status
from .context import Context, Evaluation, Refused
from .expressions import Curl, Div, Grad, Lap, Pd, depth_of_operators, kind_of
from .models import (
    Applicable,
    Blocked,
    Reason,
    VectorDiagnostics,
    VectorResult,
    VectorRule,
    VectorStep,
)
from .rules import ParsedRule, Rewrite, canonical, node_at, parse_rule, rewrites, vector_features

MAX_LISTED = 40
DERIVE_DEPTH = 4
DERIVE_WIDTH = 8
DERIVE_STATES = 400


def split_equation(text: str) -> tuple[str, str] | None:
    """Split on the one top-level `=`; None if there is none (more than one is an error)."""
    depth, cuts = 0, []
    for i, ch in enumerate(text):
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif ch == "=" and depth == 0:
            cuts.append(i)
    if not cuts:
        return None
    if len(cuts) > 1:
        raise ParseError("an equation has exactly one =")
    left, right = text[: cuts[0]].strip(), text[cuts[0] + 1 :].strip()
    if not left or not right:
        raise ParseError("an equation needs both sides")
    return left, right


@dataclass(frozen=True)
class ParsedPremise:
    text: str
    lhs: sp.Expr
    rhs: sp.Expr
    label: str


def _premises(ctx: Context) -> list[ParsedPremise]:
    out = []
    for i, p in enumerate(ctx.inp.premises):
        sides = split_equation(p.equation)
        if sides is None:
            raise Refused(Status.PARSE_ERROR, f"premise {i + 1} is not an equation", "type_error")
        lhs, rhs = ctx.parse(sides[0]), ctx.parse(sides[1])
        if kind_of(lhs) != kind_of(rhs):
            raise Refused(
                Status.PARSE_ERROR, f"premise {i + 1}: the two sides differ in kind", "type_error"
            )
        out.append(ParsedPremise(p.equation, lhs, rhs, p.label or f"premise {i + 1}"))
    return out


def _rules(ctx: Context) -> list[ParsedRule]:
    return [parse_rule(r, ctx.limits) for r in ctx.inp.rules]


def _blocked(ctx: Context, expr: sp.Expr, rule: VectorRule) -> list[Blocked]:
    out = []
    for need in rule.requires:
        lacking = ctx.lacking_smoothness(expr, need)
        if lacking:
            out.append(
                Blocked(
                    needs=need,
                    fields=lacking,
                    note=f"{rule.id} holds for twice continuously differentiable fields; the "
                    "smoothness of these fields was not declared",
                )
            )
    return out


def _fail(
    result: VectorResult,
    status: Status,
    message: str,
    diag: VectorDiagnostics,
    reason: Reason,
) -> None:
    result.status, result.message, diag.reason = status, message, reason


def _show(ctx: Context, expr: sp.Expr, result: VectorResult) -> None:
    result.result_kind = "vector" if kind_of(expr) == "vector" else "scalar"
    result.result_latex = ctx.latex(expr)
    result.result_text = str(expr)


def _applicable(
    ctx: Context, expr: sp.Expr, rules: list[ParsedRule]
) -> list[tuple[Rewrite, list[Blocked]]]:
    found: list[tuple[Rewrite, list[Blocked]]] = []
    for rule in rules:
        for reverse in (False, True):
            for rw in rewrites(expr, rule, reverse=reverse):
                found.append((rw, _blocked(ctx, rw.before, rule.rule)))
    return found


# -- list --------------------------------------------------------------------------------------
def _identities(ctx: Context, expr: sp.Expr, result: VectorResult, diag: VectorDiagnostics) -> None:
    rules = _rules(ctx)
    found = _applicable(ctx, expr, rules)
    result.applicable = [
        Applicable(
            rule_id=rw.rule.rule.id,
            reverse=rw.reverse,
            path=list(rw.path),
            part_latex=ctx.latex(rw.before),
            result_latex=ctx.latex(rw.result),
            needs=list(rw.rule.rule.requires),
            blocked=bool(blocked),
            note=blocked[0].note if blocked else "",
        )
        for rw, blocked in found[:MAX_LISTED]
    ]
    result.result_kind = "vector" if kind_of(expr) == "vector" else "scalar"
    result.search_complete = len(found) <= MAX_LISTED
    if not found:
        result.status = Status.UNCHANGED
        result.message = (
            "no identity of the handbook matches any part of the expression "
            f"(operators found: {', '.join(sorted(vector_features(expr))) or 'none'}); "
            "this lists matches only and applies nothing"
        )
        diag.reason = "no_match"
        return
    result.status = Status.OK
    result.message = (
        f"{len(found)} way(s) to apply an identity; none was applied or verified. Choose one with "
        "`rule_id` and `path` (an identity that needs c2 is blocked unless the fields are declared)"
    )


# -- apply an identity -------------------------------------------------------------------------
def _choose(
    ctx: Context,
    expr: sp.Expr,
    rules: list[ParsedRule],
    result: VectorResult,
    diag: VectorDiagnostics,
) -> Rewrite | None:
    inp = ctx.inp
    rule = next((r for r in rules if r.rule.id == inp.rule_id), None)
    if rule is None:
        _fail(
            result,
            Status.UNSUPPORTED,
            f"the identity {inp.rule_id!r} is not among the identities given",
            diag,
            "no_such_rule",
        )
        return None
    only = tuple(inp.path) if inp.path is not None else None
    options = rewrites(expr, rule, reverse=inp.reverse, only=only)
    if not options:
        _fail(
            result,
            Status.UNCHANGED,
            f"{rule.rule.id} does not match "
            + ("that part" if only is not None else "any part of the expression")
            + (" read right to left" if inp.reverse else ""),
            diag,
            "no_match",
        )
        return None
    if len(options) > 1:
        result.applicable = [
            Applicable(
                rule_id=o.rule.rule.id,
                reverse=o.reverse,
                path=list(o.path),
                part_latex=ctx.latex(o.before),
                result_latex=ctx.latex(o.result),
                needs=list(o.rule.rule.requires),
            )
            for o in options[:MAX_LISTED]
        ]
        _fail(
            result,
            Status.UNCHANGED,
            f"{len(options)} places match: choose one with `path` (nothing was picked for you)",
            diag,
            "no_match",
        )
        return None
    return options[0]


def _verified_step(
    ctx: Context, expr: sp.Expr, rw: Rewrite
) -> tuple[VectorStep | None, Status, str, list[Blocked], Evaluation | None]:
    """Check one rewrite by evaluating the difference of the two whole expressions."""
    rule = rw.rule.rule
    blocked = _blocked(ctx, rw.before, rule)
    if blocked:
        return (
            None,
            Status.CONDITIONS_NOT_SATISFIED,
            (
                f"{rule.id} needs twice continuously differentiable "
                + ", ".join(blocked[0].fields)
                + "; that was not declared, so the step was not applied"
            ),
            blocked,
            None,
        )
    ev = ctx.residual(expr, rw.result)
    if ev.status == "blocked":
        return (
            None,
            Status.CONDITIONS_NOT_SATISFIED,
            (
                "the comparison needs mixed partial derivatives to commute for a field whose smoothness "
                "was not declared"
            ),
            ev.blocked,
            ev,
        )
    if ev.status == "not_verified" or not ev.zero:
        return (
            None,
            Status.NOT_VERIFIED,
            (
                "the rewritten expression does not equal the original (or the independent check did not "
                "confirm it), so it is not shown"
            ),
            [],
            ev,
        )
    step = VectorStep(
        action="apply_identity",
        title=f"{rule.id}" + (" (right to left)" if rw.reverse else ""),
        before_latex=ctx.latex(expr),
        after_latex=ctx.latex(rw.result),
        rule_id=rule.id,
        reverse=rw.reverse,
        path=list(rw.path),
        part_latex=ctx.latex(rw.before),
        evidence="identity",
        needs=list(rule.requires),
        verification=ev.verification,
        symbolic_zero=True,
    )
    return step, Status.OK, "", [], ev


def _apply_identity(
    ctx: Context, expr: sp.Expr, result: VectorResult, diag: VectorDiagnostics
) -> None:
    rw = _choose(ctx, expr, _rules(ctx), result, diag)
    if rw is None:
        return
    step, status, message, blocked, ev = _verified_step(ctx, expr, rw)
    if ev is not None:
        result.verification = ev.verification
        diag.points = ev.verification.points_checked if ev.verification else 0
    if step is None:
        result.blocked = blocked
        _fail(
            result, status, message, diag, "needs_smoothness" if blocked else "verification_failed"
        )
        return
    result.steps = [step]
    result.evidence = "identity"
    _show(ctx, rw.result, result)
    result.status = Status.OK
    result.message = (
        f"{rw.rule.rule.id} applied; the two expressions were compared by the exact components "
        "(difference identically zero) and by an independent Cartesian evaluation. It holds for "
        "fields of the smoothness you declared."
    )


# -- substitute a premise --------------------------------------------------------------------
def _replace(expr: sp.Expr, old: sp.Expr, new: sp.Expr, only: tuple[int, ...] | None) -> sp.Expr:
    if only is not None:
        return expr if node_at(expr, only) != old else _splice(expr, only, new)
    return expr.xreplace({old: new})


def _splice(expr: sp.Expr, path: tuple[int, ...], new: sp.Expr) -> sp.Expr:
    from .rules import splice

    return splice(expr, path, new)


def _substitute(ctx: Context, expr: sp.Expr, result: VectorResult, diag: VectorDiagnostics) -> None:
    premises = _premises(ctx)
    index = ctx.inp.premise
    if index is None and len(premises) == 1:
        index = 0
    if index is None or index >= len(premises):
        raise Refused(
            Status.PARSE_ERROR,
            "name the premise to use (`premise`: its number, from 0); several relations are never "
            "applied in an order Minerva picked",
            "no_such_rule",
        )
    p = premises[index]
    old, new = (p.rhs, p.lhs) if ctx.inp.reverse else (p.lhs, p.rhs)
    only = tuple(ctx.inp.path) if ctx.inp.path is not None else None
    raw = _replace(expr, old, new, only)
    after = canonical(raw, frozenset(ctx.low.constants))
    if raw == expr:
        _fail(
            result,
            Status.UNCHANGED,
            f"the {'right' if ctx.inp.reverse else 'left'} side of {p.label} does not occur "
            + ("at that part" if only is not None else "in the expression")
            + " (it must match exactly, as written)",
            diag,
            "no_match",
        )
        return
    if after != raw:  # constants pulled out of operators: an identity, checked on its own
        check = ctx.residual(raw, after)
        if not check.zero:
            _fail(
                result,
                Status.NOT_VERIFIED,
                "tidying the substituted expression was not confirmed",
                diag,
                "verification_failed",
            )
            return
    result.steps = [
        VectorStep(
            action="substitute",
            title=f"Substitute {p.label}" + (" (right to left)" if ctx.inp.reverse else ""),
            before_latex=ctx.latex(expr),
            after_latex=ctx.latex(after),
            path=list(only or ()),
            part_latex=ctx.latex(old),
            evidence="conditional",
            given=[p.text],
        )
    ]
    result.evidence = "conditional"
    result.given = [p.text]
    _show(ctx, after, result)
    result.status = Status.OK
    result.message = (
        f"holds given {p.label}, which you supplied; Minerva does not check that it is true. "
        "This is a replacement of a part, not an identity."
    )


# -- apply an operator to both sides ------------------------------------------------------------
def _operator(name: str, expr: sp.Expr) -> sp.Expr:
    kind = kind_of(expr)
    if name == "grad":
        if kind != "scalar":
            raise Refused(Status.PARSE_ERROR, "the gradient needs scalar sides", "type_error")
        return Grad(expr)
    if name == "div":
        if kind != "vector":
            raise Refused(Status.PARSE_ERROR, "the divergence needs vector sides", "type_error")
        return Div(expr)
    if name == "curl":
        if kind != "vector":
            raise Refused(Status.PARSE_ERROR, "the curl needs vector sides", "type_error")
        return Curl(expr)
    if name == "lap":
        return Lap(expr)
    return Pd(expr, sp.Symbol("t", real=True))


def _apply_operator(ctx: Context, text: str, result: VectorResult, diag: VectorDiagnostics) -> None:
    if ctx.inp.operator is None:
        raise Refused(Status.PARSE_ERROR, "name the operator to apply", "unsupported_expression")
    sides = split_equation(text)
    if sides is None:
        raise Refused(Status.PARSE_ERROR, "apply needs an equation lhs = rhs", "type_error")
    lhs, rhs = ctx.parse(sides[0]), ctx.parse(sides[1])
    if kind_of(lhs) != kind_of(rhs):
        raise Refused(Status.PARSE_ERROR, "the two sides differ in kind", "type_error")
    raw_l, raw_r = _operator(ctx.inp.operator, lhs), _operator(ctx.inp.operator, rhs)
    # The new equation follows from the given one if Op is linear on the difference: check it.
    ev = ctx.residual(
        sp.Add(raw_l, sp.Mul(-1, raw_r)),
        _operator(ctx.inp.operator, sp.Add(lhs, sp.Mul(-1, rhs))),
    )
    result.verification = ev.verification
    if ev.status == "blocked":
        result.blocked = ev.blocked
        _fail(
            result,
            Status.CONDITIONS_NOT_SATISFIED,
            "needs smoothness that was not declared",
            diag,
            "needs_smoothness",
        )
        return
    if not ev.zero:
        _fail(
            result,
            Status.NOT_VERIFIED,
            "the operator was not confirmed to act the same on both sides",
            diag,
            "verification_failed",
        )
        return
    constants = frozenset(ctx.low.constants)
    new_l, new_r = canonical(raw_l, constants), canonical(raw_r, constants)
    for raw, tidy_form in ((raw_l, new_l), (raw_r, new_r)):
        if raw != tidy_form and not ctx.residual(raw, tidy_form).zero:
            _fail(
                result,
                Status.NOT_VERIFIED,
                "tidying the result was not confirmed",
                diag,
                "verification_failed",
            )
            return
    equation = f"{ctx.latex(new_l)} = {ctx.latex(new_r)}"
    result.steps = [
        VectorStep(
            action="apply",
            title=f"Apply {ctx.inp.operator} to both sides",
            before_latex=f"{ctx.latex(lhs)} = {ctx.latex(rhs)}",
            after_latex=equation,
            evidence="conditional",
            given=[text],
            verification=ev.verification,
            symbolic_zero=True,
        )
    ]
    result.evidence = "conditional"
    result.given = [text]
    result.result_kind = "scalar" if kind_of(new_l) == "scalar" else "vector"
    result.result_latex = equation
    result.result_text = f"{new_l} = {new_r}"
    result.status = Status.OK
    result.message = (
        "follows from the equation you gave (an implication, not an identity); the operator was "
        "confirmed to act linearly on the difference of the sides"
    )


# -- derive ------------------------------------------------------------------------------------
@dataclass
class _Move:
    kind: str  # "identity" | "premise"
    rewrite: Rewrite | None
    premise: ParsedPremise | None
    reverse: bool
    after: sp.Expr
    raw: sp.Expr | None = None  # before the constants were pulled out of the operators


def _moves(
    ctx: Context, expr: sp.Expr, rules: list[ParsedRule], premises: list[ParsedPremise]
) -> list[_Move]:
    out: list[_Move] = []
    for rw, blocked in _applicable(ctx, expr, rules):
        if not blocked:
            out.append(
                _Move(
                    "identity",
                    rw,
                    None,
                    rw.reverse,
                    canonical(rw.result, frozenset(ctx.low.constants)),
                )
            )
    for p in premises:
        for reverse in (False, True):
            old, new = (p.rhs, p.lhs) if reverse else (p.lhs, p.rhs)
            raw = expr.xreplace({old: new})
            after = canonical(raw, frozenset(ctx.low.constants))
            if raw != expr:
                out.append(_Move("premise", None, p, reverse, after, raw))
    return out


def _key(e: sp.Expr) -> sp.Expr:
    return sp.expand(e)


def _derive(ctx: Context, expr: sp.Expr, result: VectorResult, diag: VectorDiagnostics) -> None:
    rules, premises = _rules(ctx), _premises(ctx)
    target = ctx.parse(ctx.inp.target) if ctx.inp.target else None
    if target is not None and kind_of(target) != kind_of(expr):
        raise Refused(Status.PARSE_ERROR, "the target is not of the same kind", "type_error")
    goal = _key(target) if target is not None else None
    start = _key(expr)
    seen: dict[sp.Expr, tuple[sp.Expr | None, _Move | None, sp.Expr]] = {start: (None, None, expr)}
    frontier = deque([(expr, 0)])
    complete = True
    best = (expr.count_ops(), start)
    reached: sp.Expr | None = start if goal is not None and start == goal else None
    while frontier and reached is None:
        current, depth = frontier.popleft()
        if depth >= DERIVE_DEPTH:
            complete = complete and not _moves(ctx, current, rules, premises)
            continue
        moves = _moves(ctx, current, rules, premises)
        if len(moves) > DERIVE_WIDTH:
            complete = False
            moves.sort(key=lambda m: m.after.count_ops())
            moves = moves[:DERIVE_WIDTH]
        for move in moves:
            key = _key(move.after)
            if key in seen:
                continue
            if len(seen) >= DERIVE_STATES:
                complete = False
                break
            seen[key] = (_key(current), move, move.after)
            if goal is not None and key == goal:
                reached = key
                break
            if goal is None:
                size = move.after.count_ops()
                if size < best[0]:
                    best = (size, key)
            frontier.append((move.after, depth + 1))
    result.search_complete = complete
    final = reached if goal is not None else (best[1] if best[1] != start else None)
    if final is None:
        result.status = Status.UNCHANGED
        result.message = (
            "no chain of identities"
            + (" and premises" if premises else "")
            + " reached "
            + ("the target" if goal is not None else "a smaller form")
            + f" within the bounds (depth {DERIVE_DEPTH}, width {DERIVE_WIDTH}); this does not say "
            "that none exists"
        )
        diag.reason = "not_found"
        return
    chain: list[_Move] = []
    expressions: list[sp.Expr] = []
    cursor: sp.Expr | None = final
    while cursor is not None:
        parent, step_move, expression = seen[cursor]
        if step_move is None:
            break
        chain.append(step_move)
        expressions.append(expression)
        cursor = parent
    chain.reverse()
    expressions.reverse()
    steps: list[VectorStep] = []
    given: list[str] = []
    previous = expr
    for move, after in zip(chain, expressions, strict=True):
        if move.kind == "identity":
            assert move.rewrite is not None  # noqa: S101
            rw = Rewrite(
                move.rewrite.rule,
                move.rewrite.reverse,
                move.rewrite.path,
                move.rewrite.before,
                move.rewrite.after,
                after,
            )
            step, status, message, blocked, ev = _verified_step(ctx, previous, rw)
            if step is None:
                result.blocked = blocked
                _fail(
                    result,
                    status,
                    "a step of the chain failed its check: " + message,
                    diag,
                    "verification_failed",
                )
                return
            result.verification = ev.verification if ev else result.verification
            step.action = "derive"
            steps.append(step)
        else:
            p = move.premise
            assert p is not None  # noqa: S101
            old = p.rhs if move.reverse else p.lhs
            if move.raw is not None and move.raw != after:
                tidy_check = ctx.residual(
                    move.raw, after
                )  # constants out of operators: an identity
                if not tidy_check.zero:
                    _fail(
                        result,
                        Status.NOT_VERIFIED,
                        "tidying a substituted expression was not confirmed",
                        diag,
                        "verification_failed",
                    )
                    return
            steps.append(
                VectorStep(
                    action="derive",
                    title=f"Substitute {p.label}" + (" (right to left)" if move.reverse else ""),
                    before_latex=ctx.latex(previous),
                    after_latex=ctx.latex(after),
                    part_latex=ctx.latex(old),
                    evidence="conditional",
                    given=[p.text],
                )
            )
            if p.text not in given:
                given.append(p.text)
        previous = after
    result.steps = steps
    result.given = given
    result.evidence = "conditional" if given else "identity"
    _show(ctx, previous, result)
    result.status = Status.OK
    result.message = (
        f"a chain of {len(steps)} step(s); each identity step was checked by comparing the whole "
        "expressions before and after"
        + (
            "; the result holds only given: " + "; ".join(given) + " (supplied by you, not checked)"
            if given
            else ""
        )
    )
    if not complete:
        result.message += (
            ". The search was cut at its bounds, so this chain may not be the shortest"
        )


# -- entry point -------------------------------------------------------------------------------
def run_step(ctx: Context, result: VectorResult, diag: VectorDiagnostics) -> None:
    inp = ctx.inp
    text = inp.expression
    if inp.action == "apply":
        _apply_operator(ctx, text, result, diag)
        return
    if split_equation(text) is not None:
        raise Refused(
            Status.PARSE_ERROR,
            "this action takes an expression, not an equation (use `apply` for an equation)",
            "type_error",
        )
    expr = ctx.parse(text)
    diag.result_kind = "vector" if kind_of(expr) == "vector" else "scalar"
    diag.operator_depth = depth_of_operators(expr)
    result.input_latex = ctx.latex(expr)
    if inp.action == "identities":
        _identities(ctx, expr, result, diag)
    elif inp.action == "apply_identity":
        _apply_identity(ctx, expr, result, diag)
    elif inp.action == "substitute":
        _substitute(ctx, expr, result, diag)
    else:
        _derive(ctx, expr, result, diag)
