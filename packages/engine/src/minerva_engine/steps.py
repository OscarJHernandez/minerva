"""Controlled single steps: choose a part of an expression, apply one move to exactly that part.

A step never searches. The user chooses the part and the move (a handbook identity, or expand,
factor, cancel, simplify, collect); the move is applied to that part alone and the *whole* expression
before and after is verified (`verify_equal`, or random matrices for noncommuting symbols). A rewrite
whose conditions are not provable from the declared assumptions is reported and never applied.

Parts are addressed by a path through the expression tree (indices into `args`), optionally with a
subset of the terms of a sum or the factors of a product, and carry the LaTeX they were chosen as: if
the expression has changed since, the selection is refused as stale instead of acting on the wrong
place. Replacing rebuilds the ancestors of the chosen node, so exactly one occurrence changes (unlike
`xreplace`, which changes every equal one).

Everything here is deterministic and independent of the language model. Reasons and counts go to
`Diagnostics`; expressions never do.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import sympy as sp

from .claims import split_equation
from .errors import LimitError, MinervaMathError, ParseError, UnsupportedError
from .latexprint import to_latex
from .models import (
    Diagnostics,
    ExpressionInput,
    Limits,
    Reason,
    Rule,
    Status,
    Step,
    StepResult,
    Subterm,
    Verification,
)
from .ncverify import nc_degree, noncommuting_symbols, verify_equal_nc
from .ops import (
    _failure,
    _restriction_note,
    can_reverse,
    parse_expression,
    rule_rewrites,
)
from .restrictions import domain_restrictions
from .verify import verify_equal

MAX_SUBTERMS = 40
MAX_TARGETS = 4
BUILTIN_ACTIONS = ("simplify", "expand", "factor", "cancel", "collect")
_NC_ACTIONS = ("expand", "collect")  # the moves that keep the order of noncommuting factors


# --- addressing ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Chosen:
    """A part of an expression: node at `path`, or the terms/factors `args` of that node."""

    path: tuple[int, ...]
    args: tuple[int, ...] | None
    node: sp.Expr
    piece: sp.Expr


def walk(expr: sp.Basic, path: tuple[int, ...] = ()) -> Iterator[tuple[tuple[int, ...], sp.Basic]]:
    """Every non-atomic node in preorder with its path."""
    if not expr.args:
        return
    yield path, expr
    for index, arg in enumerate(expr.args):
        yield from walk(arg, (*path, index))


def node_at(expr: sp.Expr, path: tuple[int, ...]) -> sp.Expr:
    node = expr
    for index in path:
        node = node.args[index]
    return node


def _piece(node: sp.Expr, args: tuple[int, ...] | None) -> sp.Expr:
    if args is None:
        return node
    return node.func(*[node.args[i] for i in args])


def _subset(node: sp.Expr, target: sp.Expr) -> tuple[int, ...] | None:
    """Indices of the terms/factors of `node` that make up `target` (a smaller sum or product)."""
    if node.func != target.func or len(target.args) >= len(node.args):
        return None
    if node.is_Mul and node.is_commutative is False:  # order matters: a contiguous run only
        k = len(target.args)
        for start in range(len(node.args) - k + 1):
            if tuple(node.args[start : start + k]) == tuple(target.args):
                return tuple(range(start, start + k))
        return None
    unused = list(range(len(node.args)))
    picked: list[int] = []
    for wanted in target.args:
        match = next((i for i in unused if node.args[i] == wanted), None)
        if match is None:
            return None
        unused.remove(match)
        picked.append(match)
    return tuple(sorted(picked))


def splice(expr: sp.Expr, chosen: Chosen, new: sp.Expr) -> sp.Expr:
    """`expr` with the chosen part replaced by `new` and nothing else changed."""
    if chosen.args is None:
        current = new
    else:
        node = chosen.node
        picked = set(chosen.args)
        if node.is_Mul and node.is_commutative is False:
            first = min(chosen.args)
            args = [
                *node.args[:first],
                new,
                *(a for i, a in enumerate(node.args) if i > first and i not in picked),
            ]
        else:
            args = [new, *(a for i, a in enumerate(node.args) if i not in picked)]
        current = node.func(*args)
    chain = [expr]
    for index in chosen.path:
        chain.append(chain[-1].args[index])
    for parent, index in reversed(list(zip(chain[:-1], chosen.path, strict=True))):
        rebuilt = list(parent.args)
        rebuilt[index] = current
        current = parent.func(*rebuilt)
    return current


def find_matches(expr: sp.Expr, target: sp.Expr) -> list[Chosen]:
    """Every place `target` occurs: as a node, or as some of the terms of a sum / factors of a product."""
    found: list[Chosen] = []

    def visit(path: tuple[int, ...], node: sp.Expr) -> None:
        if node == target:
            found.append(Chosen(path, None, node, node))
        elif target.is_Add or target.is_Mul:
            picked = _subset(node, target)
            if picked is not None:
                found.append(Chosen(path, picked, node, _piece(node, picked)))
        for index, arg in enumerate(node.args):
            visit((*path, index), arg)

    visit((), expr)
    return found


def highlight_latex(expr: sp.Expr, chosen: Chosen) -> str:
    """The whole expression with the chosen part boxed, for a picker."""
    marker = sp.Dummy("part", commutative=chosen.piece.is_commutative)
    shown = to_latex(chosen.piece)
    return to_latex(splice(expr, chosen, marker), symbol_names={marker: rf"\boxed{{{shown}}}"})


def _subterms(
    expr: sp.Expr, matches: list[Chosen], blocked: set[int] | None = None
) -> list[Subterm]:
    blocked = blocked or set()
    counts: dict[sp.Expr, int] = {}
    for chosen in matches:
        counts[chosen.piece] = counts.get(chosen.piece, 0) + 1
    seen: dict[sp.Expr, int] = {}
    out: list[Subterm] = []
    for index, chosen in enumerate(matches[:MAX_SUBTERMS], start=1):
        seen[chosen.piece] = seen.get(chosen.piece, 0) + 1
        out.append(
            Subterm(
                index=index,
                path=list(chosen.path),
                args=list(chosen.args) if chosen.args is not None else None,
                latex=to_latex(chosen.piece),
                text=str(chosen.piece),
                highlight_latex=highlight_latex(expr, chosen),
                occurrence=seen[chosen.piece],
                occurrences=counts[chosen.piece],
                blocked=index - 1 in blocked,
            )
        )
    return out


# --- reading ---------------------------------------------------------------------------------------
def read_expression(
    data: ExpressionInput,
    limits: Limits,
    diag: Diagnostics,
    denominators: list[sp.Expr] | None = None,
) -> sp.Expr:
    """Parse an expression (noncommuting symbols allowed). Equations belong to `isolate`."""
    diag.assumptions_declared = len(data.assumptions)
    if data.format == "latex" and split_equation(data.input) is not None:
        raise ParseError("this operation takes an expression, not an equation")
    with diag.timed("parse_ms"):
        expr = parse_expression(data, limits, denominators, allow_noncommutative=True)
        if noncommuting_symbols(expr):
            nc_degree(expr)  # outside the polynomial subset: UnsupportedError
    return expr


def _read_part(text: str, data: ExpressionInput, limits: Limits) -> sp.Expr:
    if not text.strip():
        raise ParseError("name the part")
    if len(text) > limits.max_input_chars:
        raise LimitError("input is too long")
    return parse_expression(
        ExpressionInput(input=text, format=data.format, assumptions=data.assumptions),
        limits,
        allow_noncommutative=True,
    )


def _error(
    operation: str, exc: MinervaMathError, diag: Diagnostics, started: float
) -> dict[str, Any]:
    status, message = _failure(exc)
    return StepResult(
        status=status,
        operation=operation,
        message=message,
        diagnostics=diag,
        duration_ms=int((time.monotonic() - started) * 1000),
    ).model_dump(mode="json")


# --- listing the parts ------------------------------------------------------------------------------
def op_subterms(payload: dict[str, Any]) -> dict[str, Any]:
    """The parts of an expression the user can choose: every node, the places a typed part occurs, or
    the places a handbook rule matches (blocked ones flagged)."""
    started = time.monotonic()
    limits = Limits(**payload["limits"])
    data = ExpressionInput(**payload["expression"])
    diag = Diagnostics(collect_detail=bool(payload.get("debug")))
    try:
        expr = read_expression(data, limits, diag)
        deadline = started + 0.7 * limits.timeout_seconds
        blocked: set[int] = set()
        with diag.timed("candidates_ms"):
            if payload.get("find"):
                matches = find_matches(expr, _read_part(str(payload["find"]), data, limits))
            elif payload.get("rule"):
                rule = Rule(**payload["rule"])
                reverse = bool(payload.get("reverse", False))
                if reverse and not can_reverse(rule):
                    raise UnsupportedError("this identity cannot be applied right to left")
                matches = []
                for path, node in walk(expr):
                    if time.monotonic() > deadline:
                        diag.reject(Reason.DEADLINE)
                        break
                    hits = [
                        rw
                        for rw in rule_rewrites(node, rule, reverse=reverse, strict=True)
                        if rw.node == node
                    ]
                    if hits:
                        if hits[0].unmet:
                            blocked.add(len(matches))
                        matches.append(Chosen(path, None, node, node))
            else:
                matches = [Chosen(path, None, node, node) for path, node in walk(expr)]
        diag.candidates = len(matches)
        shown = _subterms(expr, matches, blocked)
        truncated = len(matches) > MAX_SUBTERMS
        return StepResult(
            status=Status.OK,
            operation="subterms",
            input_latex=to_latex(expr),
            subterms=shown,
            truncated=truncated,
            diagnostics=diag,
            message=(
                f"{len(shown)} part(s) to choose from"
                + (
                    f"; {len(matches)} found, type the part to narrow the choice"
                    if truncated
                    else ""
                )
                + "."
                if shown
                else "No place matches."
            ),
            duration_ms=int((time.monotonic() - started) * 1000),
        ).model_dump(mode="json")
    except MinervaMathError as exc:
        return _error("subterms", exc, diag, started)


# --- rewriting one part --------------------------------------------------------------------------------
@dataclass
class _Move:
    new: sp.Expr | None
    action: str  # Step.action
    name: str
    direction: str | None = None
    reason: Reason | None = None
    unmet: list[Any] | None = None
    message: str = ""


def _collect_targets(texts: list[str], data: ExpressionInput, limits: Limits) -> list[sp.Expr]:
    if not texts or len(texts) > MAX_TARGETS:
        raise ParseError(f"give between 1 and {MAX_TARGETS} things to collect by")
    targets = [_read_part(t, data, limits) for t in texts]
    for target in targets:
        if not target.free_symbols:
            raise ParseError("collect by a variable or a part of the expression, not a number")
        if noncommuting_symbols(target):
            raise UnsupportedError("collecting by a noncommuting symbol is not supported")
    return targets


def _builtin(
    piece: sp.Expr, action: str, texts: list[str], data: ExpressionInput, limits: Limits
) -> _Move:
    if noncommuting_symbols(piece) and action not in _NC_ACTIONS:
        raise UnsupportedError(
            f"only expand and collect keep the order of noncommuting factors; {action} is not supported"
        )
    if action == "collect":
        targets = _collect_targets(texts, data, limits)
        if not any(piece.has(t) for t in targets):
            return _Move(None, "collect", "collect", reason=Reason.PATTERN_NOT_FOUND)
        return _Move(sp.collect(piece, targets), "collect", "sympy.collect")
    function = {
        "simplify": sp.simplify,
        "expand": sp.expand,
        "factor": sp.factor,
        "cancel": sp.cancel,
    }
    if action not in function:
        raise ParseError(f"unknown move {action!r}")
    return _Move(function[action](piece), "builtin", f"sympy.{action}")


def _rule_move(piece: sp.Expr, rule: Rule, reverse: bool) -> _Move:
    if reverse and not can_reverse(rule):
        raise UnsupportedError("this identity cannot be applied right to left")
    # The rule is matched on the chosen part alone, so it cannot reach outside it. `strict` makes a sum
    # or product pattern match only a sum or product, not a term with the rest absorbed.
    hits = [
        rw for rw in rule_rewrites(piece, rule, reverse=reverse, strict=True) if rw.node == piece
    ]
    direction = "reverse" if reverse else "forward"
    if not hits:
        return _Move(None, "rule", rule.id, direction, Reason.NO_MATCH)
    if hits[0].unmet:
        return _Move(None, "rule", rule.id, direction, Reason.CONDITIONS_UNMET, list(hits[0].unmet))
    return _Move(hits[0].rewritten, "rule", rule.id, direction)


def _resolve(
    expr: sp.Expr, payload: dict[str, Any], data: ExpressionInput, limits: Limits, diag: Diagnostics
) -> Chosen | StepResult:
    """The chosen part, or a refusal that tells the user what to do."""
    selection = payload.get("selection")
    part = payload.get("part")
    if selection is not None:
        path = tuple(int(i) for i in selection.get("path", []))
        args = selection.get("args")
        try:
            node = node_at(expr, path)
            picked = tuple(int(i) for i in args) if args is not None else None
            if picked is not None and not all(0 <= i < len(node.args) for i in picked):
                raise IndexError
            chosen = Chosen(path, picked, node, _piece(node, picked))
        except (IndexError, TypeError):
            chosen = None
        if chosen is None or to_latex(chosen.piece) != selection.get("expected_latex"):
            diag.reject(Reason.STALE_SELECTION)
            return StepResult(
                status=Status.PARSE_ERROR,
                operation="rewrite",
                reason=Reason.STALE_SELECTION,
                diagnostics=diag,
                message="The chosen part is no longer in this expression; choose it again.",
            )
        return chosen
    if part is not None:
        matches = find_matches(expr, _read_part(str(part), data, limits))
        diag.candidates = len(matches)
        occurrence = payload.get("occurrence")
        if not matches:
            diag.reject(Reason.STALE_SELECTION)
            return StepResult(
                status=Status.PARSE_ERROR,
                operation="rewrite",
                reason=Reason.STALE_SELECTION,
                diagnostics=diag,
                message="That part does not occur in the expression.",
            )
        if occurrence is None and len(matches) > 1:
            diag.reject(Reason.AMBIGUOUS_SELECTION)
            return StepResult(
                status=Status.PARSE_ERROR,
                operation="rewrite",
                reason=Reason.AMBIGUOUS_SELECTION,
                subterms=_subterms(expr, matches),
                diagnostics=diag,
                message=f"That part occurs {len(matches)} times; say which occurrence to change.",
            )
        index = int(occurrence or 1) - 1
        if not 0 <= index < len(matches):
            diag.reject(Reason.STALE_SELECTION)
            return StepResult(
                status=Status.PARSE_ERROR,
                operation="rewrite",
                reason=Reason.STALE_SELECTION,
                diagnostics=diag,
                message=f"That part occurs {len(matches)} time(s); there is no occurrence {index + 1}.",
            )
        return matches[index]
    return Chosen((), None, expr, expr)


def op_rewrite_at(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply one move to one part and verify the whole expression."""
    started = time.monotonic()
    limits = Limits(**payload["limits"])
    data = ExpressionInput(**payload["expression"])
    action = str(payload["action"])
    diag = Diagnostics(collect_detail=bool(payload.get("debug")))
    denominators: list[sp.Expr] = []

    def finish(result: StepResult) -> dict[str, Any]:
        result.diagnostics = diag
        result.duration_ms = int((time.monotonic() - started) * 1000)
        return result.model_dump(mode="json")

    try:
        expr = read_expression(data, limits, diag, denominators)
        input_latex = to_latex(expr)
        chosen = _resolve(expr, payload, data, limits, diag)
        if isinstance(chosen, StepResult):
            chosen.input_latex = input_latex
            return finish(chosen)

        with diag.timed("transform_ms"):
            if action == "rule":
                if not payload.get("rule"):
                    raise ParseError("name the identity to apply")
                move = _rule_move(
                    chosen.piece, Rule(**payload["rule"]), bool(payload.get("reverse"))
                )
            else:
                move = _builtin(chosen.piece, action, list(payload.get("by") or []), data, limits)
        diag.candidates = max(diag.candidates, 1)

        if move.new is None or move.new == chosen.piece:
            reason = move.reason or Reason.NO_MATCH
            diag.reject(reason)
            if reason is Reason.CONDITIONS_UNMET:
                return finish(
                    StepResult(
                        status=Status.CONDITIONS_NOT_SATISFIED,
                        operation="rewrite",
                        input_latex=input_latex,
                        reason=reason,
                        unmet_conditions=move.unmet or [],
                        message=(
                            "The identity matches this part, but its conditions cannot be shown from "
                            "the declared assumptions, so it is not applied."
                        ),
                    )
                )
            return finish(
                StepResult(
                    status=Status.UNCHANGED,
                    operation="rewrite",
                    input_latex=input_latex,
                    result_latex=input_latex,
                    result_text=str(expr),
                    reason=reason,
                    message=(
                        "None of the things to collect by occurs in this part."
                        if reason is Reason.PATTERN_NOT_FOUND
                        else "The move does not change this part."
                        if move.new is not None
                        else "The identity does not match this part."
                    ),
                )
            )

        result = splice(expr, chosen, move.new)
        if result == expr:
            diag.reject(Reason.NO_MATCH)
            return finish(
                StepResult(
                    status=Status.UNCHANGED,
                    operation="rewrite",
                    input_latex=input_latex,
                    result_latex=input_latex,
                    result_text=str(expr),
                    reason=Reason.NO_MATCH,
                    message="The move does not change the expression.",
                )
            )

        verification = _verify(expr, result, diag)
        if not verification.passed:
            diag.reject(Reason.NOT_VERIFIED)
            diag.note(f"candidate that failed verification: {to_latex(result)}")
            return finish(
                StepResult(
                    status=Status.NOT_VERIFIED,
                    operation="rewrite",
                    input_latex=input_latex,
                    verification=verification,
                    reason=Reason.NOT_VERIFIED,
                    message="The part was rewritten but the whole expression could not be verified, "
                    "so the result is not shown. " + verification.note,
                )
            )
        restrictions, complete = domain_restrictions(expr, result, denominators)
        diag.restrictions_outcome = (
            "incomplete" if not complete else "some" if restrictions else "none"
        )
        step = Step(
            action=move.action,
            name=move.name,
            direction=move.direction,
            matched_latex=to_latex(chosen.piece),
            before_latex=input_latex,
            after_latex=to_latex(result),
            verification=verification,
        )
        return finish(
            StepResult(
                status=Status.OK,
                operation="rewrite",
                input_latex=input_latex,
                result_latex=to_latex(result),
                result_text=str(result),
                steps=[step],
                verification=verification,
                restrictions_latex=restrictions,
                restrictions_complete=complete,
                message=(
                    f"Applied {move.name} to the chosen part; the whole expression was verified "
                    + (
                        "symbolically and numerically."
                        if verification.symbolic_zero
                        else "numerically."
                    )
                    + _restriction_note(restrictions, complete)
                ),
            )
        )
    except MinervaMathError as exc:
        return _error("rewrite", exc, diag, started)


def _verify(before: sp.Expr, after: sp.Expr, diag: Diagnostics) -> Verification:
    with diag.timed("verify_ms"):
        if noncommuting_symbols(before, after):
            diag.verification_method = "matrix"
            verification = verify_equal_nc(before, after)
        else:
            diag.verification_method = "numeric"
            verification = verify_equal(before, after)
    diag.verification_outcome = "passed" if verification.passed else "failed"
    diag.symbolic_zero = verification.symbolic_zero
    return verification
