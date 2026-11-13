"""Substitution of user-supplied relations: replace a left side by a right side, step by step.

A relation `l = r` is a *premise*: the user states it and Minerva does not verify it. The result is
therefore conditional ("holds given l = r"), unless it also holds without the premise, which is
checked separately and reported as such.

How a step is checked. The occurrences of `l` that SymPy recognises are replaced by a placeholder `P`
to give `T(P)`. Two facts make the step sound:

- `T(l)` is the original expression. This is verified (`verify_equal`, or random matrices for
  noncommuting symbols): it shows that the places SymPy chose really are occurrences of `l`.
- `R = T(r)` is built by exact substitution into the tree, and `l = r` gives `T(l) = T(r)`.

So `E = R` whenever the relation holds. Nothing here samples points of the relation, so it works for
relations that no number satisfies and for noncommuting symbols.

Several relations are applied one at a time in the stated order, once each (never repeated until
nothing changes, so `x = y`, `y = x` cannot loop). If another order would give another result the
request is refused until the order is stated: Minerva does not guess which reading was meant.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any

import sympy as sp

from .claims import split_equation
from .errors import MinervaMathError, ParseError, UnsupportedInput
from .latexprint import to_latex
from .models import (
    ExpressionInput,
    Limits,
    Premise,
    Reason,
    Status,
    Step,
    StepDiagnostics,
    StepResult,
    Verification,
)
from .ops import _restriction_note, parse_expression
from .restrictions import domain_restrictions
from .steps import _error, _verify, read_expression

MAX_RELATIONS = 4


@dataclass(frozen=True)
class Relation:
    pattern: sp.Expr
    replacement: sp.Expr

    @property
    def premise(self) -> Premise:
        return Premise(
            latex=f"{to_latex(self.pattern)} = {to_latex(self.replacement)}",
            text=f"{self.pattern} = {self.replacement}",
        )


def read_relation(
    text: str, data: ExpressionInput, limits: Limits, denominators: list[sp.Expr]
) -> Relation:
    sides = split_equation(text)
    if sides is None or len(sides) != 2:
        raise ParseError("write each relation as `left = right` (one equals sign)")
    left, right = (_read_part_with(side, data, limits, denominators) for side in sides)
    if left.is_Number and right.is_Number:
        raise UnsupportedInput("a relation between numbers only cannot be used to substitute")
    # The side with symbols is replaced; `2 = x` reads as "replace x by 2".
    pattern, replacement = (right, left) if left.is_Number else (left, right)
    difference = sp.expand(left - right)
    if difference.is_Number and difference != 0:
        raise UnsupportedInput("this relation can never hold, so nothing follows from it")
    return Relation(pattern, replacement)


def _read_part_with(
    text: str, data: ExpressionInput, limits: Limits, denominators: list[sp.Expr]
) -> sp.Expr:
    if not text.strip():
        raise ParseError("a relation has an empty side")
    return parse_expression(
        ExpressionInput(input=text, format=data.format, assumptions=data.assumptions),
        limits,
        denominators,
        allow_noncommutative=True,
    )


def _marked(expr: sp.Expr, pattern: sp.Expr) -> tuple[sp.Expr, sp.Dummy] | None:
    """`expr` with the occurrences of `pattern` that SymPy recognises replaced by a placeholder."""
    placeholder = sp.Dummy("p", commutative=pattern.is_commutative)
    marked = expr.subs(pattern, placeholder)
    return None if marked == expr else (marked, placeholder)


def _parenthesised(expr: sp.Expr) -> str:
    text = to_latex(expr)
    needs = (
        expr.is_Add
        or (expr.is_Number and bool(expr < 0))
        or (expr.is_Mul and expr.could_extract_minus_sign())
    )
    return rf"\left({text}\right)" if needs else text


def _display(marked: sp.Expr, placeholder: sp.Dummy, replacement: sp.Expr) -> str:
    """The substitution before the arithmetic SymPy does on construction: `x^2 + (1 - x^2)`."""
    return to_latex(marked, symbol_names={placeholder: _parenthesised(replacement)})


def _sequence(expr: sp.Expr, relations: list[Relation], order: tuple[int, ...]) -> sp.Expr:
    for index in order:
        found = _marked(expr, relations[index].pattern)
        if found is not None:
            marked, placeholder = found
            expr = marked.xreplace({placeholder: relations[index].replacement})
    return expr


def _simultaneous(expr: sp.Expr, relations: list[Relation]) -> sp.Expr:
    placeholders: dict[sp.Dummy, sp.Expr] = {}
    for relation in relations:
        found = _marked(expr, relation.pattern)
        if found is not None:
            expr, placeholder = found
            placeholders[placeholder] = relation.replacement
    return expr.xreplace(placeholders)


def _feeds(relations: list[Relation]) -> bool:
    """Is there a cycle: relation i produces symbols that relation j rewrites, and so on back to i?"""
    n = len(relations)
    edge = [
        [
            i != j
            and bool(relations[i].replacement.free_symbols & relations[j].pattern.free_symbols)
            for j in range(n)
        ]
        for i in range(n)
    ]
    for start in range(n):
        seen, stack = set(), [start]
        while stack:
            node = stack.pop()
            for nxt in range(n):
                if edge[node][nxt]:
                    if nxt == start:
                        return True
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
    return False


def _order_dependent(
    expr: sp.Expr, relations: list[Relation], stated: sp.Expr, deadline: float
) -> bool | None:
    """Does any other order, or applying all at once, give another expression? None: not examined."""
    for order in itertools.permutations(range(len(relations))):
        if time.monotonic() > deadline:
            return None
        if _sequence(expr, relations, order) != stated:
            return True
    return bool(_simultaneous(expr, relations) != stated)


def op_substitute(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    limits = Limits(**payload["limits"])
    data = ExpressionInput(**payload["expression"])
    texts = [str(t) for t in payload.get("relations", [])]
    order_mode = payload.get("order")
    diag = StepDiagnostics(collect_detail=bool(payload.get("debug")))
    denominators: list[sp.Expr] = []

    def finish(result: StepResult) -> dict[str, Any]:
        result.diagnostics = diag
        result.duration_ms = int((time.monotonic() - started) * 1000)
        return result.model_dump(mode="json")

    try:
        if not 1 <= len(texts) <= MAX_RELATIONS:
            raise ParseError(f"give between 1 and {MAX_RELATIONS} relations")
        if order_mode not in (None, "given", "simultaneous"):
            raise ParseError("order is `given` or `simultaneous`")
        expr = read_expression(data, limits, diag, denominators)
        with diag.timed("parse_ms"):
            relations = [read_relation(t, data, limits, denominators) for t in texts]
        diag.premises = len(relations)
        input_latex = to_latex(expr)
        deadline = started + 0.7 * limits.timeout_seconds

        # Is the order of several relations part of the answer?
        order_dependent: bool | None = False
        if len(relations) > 1:
            with diag.timed("candidates_ms"):
                stated = _sequence(expr, relations, tuple(range(len(relations))))
                order_dependent = _order_dependent(expr, relations, stated, deadline)
            if diag.collect_detail and len(relations) > 1:
                for order in itertools.permutations(range(len(relations))):
                    diag.note(
                        f"order {[i + 1 for i in order]}: {_sequence(expr, relations, order)}"
                    )
                diag.note(f"all at once: {_simultaneous(expr, relations)}")
            if order_mode is None and order_dependent is not False:
                reason = Reason.CYCLIC if _feeds(relations) else Reason.ORDER_DEPENDENT
                if order_dependent is None:
                    reason = Reason.DEADLINE
                diag.reject(reason)
                return finish(
                    StepResult(
                        status=Status.UNSUPPORTED,
                        operation="substitute",
                        input_latex=input_latex,
                        reason=reason,
                        order_dependent=order_dependent,
                        message=(
                            "The result depends on the order in which the relations are applied "
                            "(or they rewrite each other), so Minerva will not choose one. Say "
                            "whether to apply them in the order given, or all at once."
                            if reason is not Reason.DEADLINE
                            else "Whether the order matters could not be examined in time; state the order."
                        ),
                    )
                )

        steps: list[Step] = []
        used: list[Relation] = []
        current = expr
        failed: Verification | None = None
        with diag.timed("transform_ms"):
            if order_mode == "simultaneous" and len(relations) > 1:
                groups = [list(range(len(relations)))]
            else:
                groups = [[i] for i in range(len(relations))]
            for group in groups:
                before = current
                placeholders: dict[sp.Dummy, Relation] = {}
                marked = current
                for index in group:
                    found = _marked(marked, relations[index].pattern)
                    if found is None:
                        diag.reject(Reason.PATTERN_NOT_FOUND)
                        continue
                    marked, placeholder = found
                    placeholders[placeholder] = relations[index]
                if not placeholders:
                    continue
                diag.candidates += len(placeholders)
                back = {ph: rel.pattern for ph, rel in placeholders.items()}
                verification = _verify_back(before, marked, back, diag)
                if not verification.passed:
                    failed = verification
                    break
                current = marked.xreplace({ph: rel.replacement for ph, rel in placeholders.items()})
                used.extend(placeholders.values())
                shown = [r.premise.latex for r in placeholders.values()]
                last = list(placeholders.items())[-1]
                display = (
                    _display(marked, last[0], last[1].replacement)
                    if len(placeholders) == 1
                    else None
                )
                steps.append(
                    Step(
                        action="substitute",
                        name="; ".join(shown),
                        matched_latex=", ".join(to_latex(r.pattern) for r in placeholders.values()),
                        before_latex=to_latex(before),
                        after_latex=to_latex(current),
                        display_latex=display if display and display != to_latex(current) else None,
                        evidence="conditional",
                        given=shown,
                        verification=verification,
                    )
                )

        if failed is not None:
            diag.reject(Reason.NOT_VERIFIED)
            diag.note(f"verification of a substitution failed: {failed.note}")
            return finish(
                StepResult(
                    status=Status.NOT_VERIFIED,
                    operation="substitute",
                    input_latex=input_latex,
                    verification=failed,
                    reason=Reason.NOT_VERIFIED,
                    order_dependent=order_dependent,
                    message="A substitution could not be verified, so no result is shown. "
                    + failed.note,
                )
            )
        if not steps or current == expr:
            applied = bool(steps)
            return finish(
                StepResult(
                    status=Status.UNCHANGED,
                    operation="substitute",
                    input_latex=input_latex,
                    result_latex=input_latex,
                    result_text=str(expr),
                    steps=steps,
                    reason=None if applied else Reason.PATTERN_NOT_FOUND,
                    order_dependent=order_dependent,
                    message=(
                        "The relations were applied one after another and the result is the "
                        "original expression."
                        if applied
                        else "The left side of no relation occurs in the expression (as SymPy "
                        "reads it), so nothing was replaced."
                    ),
                )
            )

        # Does the result also hold without the relations?
        unconditional = _verify(expr, current, diag)
        diag.verification_method = "congruence"
        passed_steps = [s.verification for s in steps if s.verification is not None]
        evidence = "identity" if unconditional.passed else "conditional"
        verification = (
            unconditional
            if unconditional.passed
            else Verification(
                numeric_points_checked=sum(v.numeric_points_checked for v in passed_steps),
                numeric_points_agreed=sum(v.numeric_points_agreed for v in passed_steps),
                numeric_points_skipped=sum(v.numeric_points_skipped for v in passed_steps),
                passed=True,
                note="each substitution reproduces the original expression when the relation's "
                "left side is put back",
            )
        )
        diag.verification_outcome = "passed"
        restrictions, complete = domain_restrictions(expr, current, denominators)
        diag.restrictions_outcome = (
            "incomplete" if not complete else "some" if restrictions else "none"
        )
        premises = [r.premise for r in used]
        given = "; ".join(p.text for p in premises)
        return finish(
            StepResult(
                status=Status.OK,
                operation="substitute",
                input_latex=input_latex,
                result_latex=to_latex(current),
                result_text=str(current),
                steps=steps,
                evidence=evidence,
                given=premises,
                verification=verification,
                restrictions_latex=restrictions,
                restrictions_complete=complete,
                order_dependent=order_dependent,
                message=(
                    (
                        "The result also holds without the relation(s). "
                        if evidence == "identity"
                        else ""
                    )
                    + (
                        f"Holds given {given}. Those are your premises: Minerva did not check that "
                        "they are true, only that the replacement reproduces the original expression."
                        if evidence == "conditional"
                        else ""
                    )
                    + _restriction_note(restrictions, complete)
                    + (
                        ""
                        if order_dependent is not True
                        else " Another order of the relations would give a different expression."
                    )
                ),
            )
        )
    except MinervaMathError as exc:
        return _error("substitute", exc, diag, started)


def _verify_back(
    before: sp.Expr, marked: sp.Expr, back: dict[sp.Dummy, sp.Expr], diag: StepDiagnostics
) -> Verification:
    verification = _verify(before, marked.xreplace(back), diag)
    diag.verification_method = "congruence"
    return verification
