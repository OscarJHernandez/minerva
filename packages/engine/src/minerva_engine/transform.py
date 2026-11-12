"""Bounded search for a verified chain of rewrites toward a stated goal.

Moves are handbook rules (each at every place it matches, read left to right and, where the right
side determines every variable, right to left) and four SymPy moves (`simplify`, `expand`,
`factor`, `cancel`), which are labelled as SymPy and never as handbook identities. A move enters the
search only after `verify_equal` agrees at assumption-respecting sample points; a move whose
conditions are not provable from the declared assumptions is never applied, only reported.

The search is a small beam search: at most `MAX_WIDTH` states per level, `MAX_DEPTH` levels, a cap on
verifications per state, a visited set (so loops cannot happen), a size cap on intermediate results
and the same wall-clock budget rule as `op_identities`. Numerical agreement is evidence, not proof:
the symbolic zero test is run separately on the reported chain and reported separately.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import sympy as sp

from .errors import MinervaMathError
from .features import expression_features
from .models import (
    BlockedMove,
    ExpressionInput,
    Limits,
    Rule,
    RuleCondition,
    Status,
    TransformChain,
    TransformResult,
    TransformStep,
    Verification,
)
from .ops import _failure, can_reverse, parse_expression, rule_rewrites
from .textparse import parse_text
from .verify import verify_equal

MAX_WIDTH = 6
MAX_DEPTH = 6
MAX_VERIFY_PER_STATE = 8
SCREEN_POINTS = 6  # sample points for screening a move; the reported chain gets the full check
STALE_LEVELS = 2  # stop after this many levels without a better cost
MAX_BLOCKED = 5
MAX_ALTERNATIVES = 2
SIZE_FACTOR = 2
SIZE_SLACK = 20

_GOALS = ("simplify", "expand", "factor", "rewrite")
_EXPANDABLE_FUNCTIONS = frozenset(
    {"sin", "cos", "tan", "sec", "csc", "cot", "sinh", "cosh", "tanh"}
)

CAVEAT = (
    "Each step was checked numerically at assumption-respecting sample points (numerical evidence, "
    "not a proof); the symbolic zero test is reported separately. Steps are equalities where both "
    "sides are defined: removable singularities and branch choices outside the declared "
    "assumptions are not tracked."
)

Cost = tuple[int, int]


def _nodes(expr: sp.Basic) -> int:
    return sum(1 for _ in sp.preorder_traversal(expr))


def _ops(expr: sp.Basic) -> int:
    return int(sp.count_ops(expr))


def _unexpanded(expr: sp.Basic) -> int:
    """Count places `expand` would still change: products with a sum, powers of sums, sums or
    integer multiples inside a trigonometric or hyperbolic argument."""
    count = 0
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow):
            if node.base.is_Add and node.exp.is_Integer and node.exp >= 2:
                count += 1
        elif node.is_Mul:
            if len(node.args) > 1 and any(a.is_Add for a in node.args):
                count += 1
        elif isinstance(node, sp.Function) and node.func.__name__.lower() in _EXPANDABLE_FUNCTIONS:
            arg = node.args[0]
            if arg.is_Add or (arg.is_Mul and arg.args[0].is_Integer and abs(arg.args[0]) >= 2):
                count += 1
    return count


def _foreign_functions(expr: sp.Basic, target: frozenset[str]) -> int:
    return sum(
        1
        for node in sp.preorder_traversal(expr)
        if isinstance(node, sp.Function)
        and not isinstance(node, sp.Derivative)
        and node.func.__name__.lower() not in target
    )


def _cost_function(goal: str, target: frozenset[str]) -> Callable[[sp.Expr], Cost]:
    if goal == "expand":
        return lambda e: (_unexpanded(e), _nodes(e))
    if goal == "factor":
        return lambda e: (len(sp.Add.make_args(e)), _ops(e))
    if goal == "rewrite":
        return lambda e: (_foreign_functions(e, target), _ops(e))
    return lambda e: (_ops(e), _nodes(e))


@dataclass(frozen=True)
class _Move:
    kind: str  # "rule" | "builtin"
    name: str
    direction: str | None
    matched: str | None
    result: sp.Expr
    unmet: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class _Step:
    move: _Move
    before: sp.Expr
    verification: Verification


@dataclass
class _Candidate:
    expr: sp.Expr
    steps: tuple[_Step, ...] = ()
    cost: Cost = (0, 0)

    def rank(self) -> tuple[Cost, int, int]:
        # Equal cost and length: a chain that cites a handbook identity beats a bare SymPy move.
        builtin_last = 1 if self.steps and self.steps[-1].move.kind == "builtin" else 0
        return self.cost, len(self.steps), builtin_last


@dataclass
class _Search:
    rules: list[Rule]
    reversible: dict[str, bool]
    needs: dict[tuple[str, str], frozenset[str]]
    cost: Callable[[sp.Expr], Cost]
    max_nodes: int
    deadline: float
    start_cost: Cost = (0, 0)
    tried: int = 0
    discarded: int = 0
    truncated: bool = False
    blocked: dict[tuple[str, str, str], BlockedMove] = field(default_factory=dict)

    def out_of_time(self) -> bool:
        if time.monotonic() > self.deadline:
            self.truncated = True
        return self.truncated

    def moves(self, expr: sp.Expr, *, with_simplify: bool) -> Iterator[_Move]:
        present = {f for f in expression_features(expr) if f.startswith("f:")}
        for rule in self.rules:
            if self.out_of_time():
                return
            for direction in ("forward", "reverse") if self.reversible[rule.id] else ("forward",):
                if not self.needs[rule.id, direction] <= present:
                    continue  # a pattern that names a function cannot match where it does not occur
                try:
                    for rw in rule_rewrites(
                        expr, rule, reverse=direction == "reverse", strict=True
                    ):
                        unmet = tuple((c.variable, c.assumption) for c in rw.unmet)
                        yield _Move(
                            "rule", rule.id, direction, sp.latex(rw.node), rw.rewritten, unmet
                        )
                except MinervaMathError:
                    break  # a rule the engine cannot read is skipped, never guessed at
        builtins: list[tuple[str, Callable[[sp.Expr], sp.Expr]]] = [
            ("sympy.expand", sp.expand),
            ("sympy.factor", sp.factor),
            ("sympy.cancel", sp.cancel),
        ]
        if with_simplify:
            builtins.insert(0, ("sympy.simplify", sp.simplify))
        for name, fn in builtins:
            if self.out_of_time():
                return
            try:
                result = fn(expr)
            except Exception:  # noqa: S112 - SymPy raises many kinds; a failed move is not a move
                continue
            yield _Move("builtin", name, None, None, result)

    def note_blocked(self, move: _Move) -> None:  # only called for moves from the input itself
        assert move.direction is not None  # noqa: S101 - only rules can be blocked
        if len(self.blocked) >= MAX_BLOCKED or self.cost(move.result) >= self.start_cost:
            return
        key = (move.name, move.direction, move.matched or "")
        self.blocked.setdefault(
            key,
            BlockedMove(
                rule_id=move.name,
                direction=move.direction,
                matched_latex=move.matched or "",
                unmet_conditions=[RuleCondition(variable=v, assumption=a) for v, a in move.unmet],
            ),
        )

    def expand_state(
        self, cand: _Candidate, seen: set[sp.Expr], *, with_simplify: bool
    ) -> list[_Candidate]:
        proposals: dict[sp.Expr, _Move] = {}
        for move in self.moves(cand.expr, with_simplify=with_simplify):
            if move.unmet:
                if not cand.steps:
                    self.note_blocked(move)
                continue
            result = move.result
            if result == cand.expr or result in seen or result in proposals:
                continue
            if _nodes(result) > self.max_nodes:
                continue
            proposals[result] = move
        ordered = sorted(proposals.items(), key=lambda item: self.cost(item[0]))
        accepted: list[_Candidate] = []
        for result, move in ordered[:MAX_VERIFY_PER_STATE]:
            if self.out_of_time():
                break
            self.tried += 1
            verification = verify_equal(cand.expr, result, symbolic=False, max_points=SCREEN_POINTS)
            if not verification.passed:
                self.discarded += 1
                continue
            seen.add(result)
            accepted.append(
                _Candidate(
                    result,
                    (*cand.steps, _Step(move, cand.expr, verification)),
                    self.cost(result),
                )
            )
        return accepted


def _finalize(
    start: sp.Expr, cand: _Candidate, allow_symbolic: bool
) -> tuple[TransformChain, list[TransformStep]] | None:
    """Re-verify a chain with the symbolic zero test added; None if anything does not hold."""
    steps: list[TransformStep] = []
    for step in cand.steps:
        v = verify_equal(step.before, _after(cand, step), symbolic=allow_symbolic)
        if not v.passed:
            return None
        steps.append(
            TransformStep(
                kind=step.move.kind,
                name=step.move.name,
                direction=step.move.direction,
                matched_latex=step.move.matched,
                before_latex=sp.latex(step.before),
                after_latex=sp.latex(_after(cand, step)),
                verification=v,
            )
        )
    end = verify_equal(start, cand.expr, symbolic=allow_symbolic)
    if not end.passed:
        return None
    chain = TransformChain(
        result_latex=sp.latex(cand.expr),
        result_text=str(cand.expr),
        steps=steps,
        verification=end,
    )
    return chain, steps


def _after(cand: _Candidate, step: _Step) -> sp.Expr:
    index = cand.steps.index(step)
    return cand.steps[index + 1].before if index + 1 < len(cand.steps) else cand.expr


def transform(
    data: ExpressionInput,
    rules: list[Rule],
    goal: str,
    target: list[str],
    limits: Limits,
    *,
    width: int = 4,
    depth: int = 4,
) -> TransformResult:
    started = time.monotonic()
    if goal not in _GOALS:
        return TransformResult(
            status=Status.UNSUPPORTED, message=f"unknown goal {goal!r}; use one of {list(_GOALS)}"
        )
    names = frozenset(t.lower() for t in target if t.strip())
    if goal == "rewrite" and not names:
        return TransformResult(
            status=Status.UNSUPPORTED,
            message="the goal 'rewrite' needs the function names to rewrite in terms of",
        )
    try:
        expr = parse_expression(data, limits)
    except MinervaMathError as exc:
        status, message = _failure(exc)
        return TransformResult(status=status, message=message)

    width = max(1, min(width, MAX_WIDTH))
    depth = max(1, min(depth, MAX_DEPTH))
    cost = _cost_function(goal, names)
    input_latex = sp.latex(expr)
    reversible: dict[str, bool] = {}
    needs: dict[tuple[str, str], frozenset[str]] = {}
    usable: list[Rule] = []
    for rule in rules:
        try:
            reversible[rule.id] = can_reverse(rule)
            for direction, side in (("forward", rule.lhs), ("reverse", rule.rhs)):
                features = expression_features(parse_text(side))
                needs[rule.id, direction] = frozenset(f for f in features if f.startswith("f:"))
        except MinervaMathError:
            continue
        usable.append(rule)
    # The more specific rule (fewer pattern variables) is tried first, so when two rules give the
    # same result the simpler identity is the one cited (sin 2a, not a sum-to-product with b = 0).
    usable.sort(key=lambda r: len(r.variables))

    nodes = _nodes(expr)
    search = _Search(
        rules=usable,
        reversible=reversible,
        needs=needs,
        cost=cost,
        max_nodes=nodes * SIZE_FACTOR + SIZE_SLACK if goal != "expand" else nodes * 8 + 100,
        deadline=started + 0.7 * limits.timeout_seconds,
        start_cost=cost(expr),
    )
    start = _Candidate(expr, (), search.start_cost)
    seen: set[sp.Expr] = {expr}
    pool: list[_Candidate] = []
    beam = [start]
    best_cost = search.start_cost
    stale = 0
    for level in range(depth):
        level_found: list[_Candidate] = []
        for cand in beam:
            if search.out_of_time():
                break
            level_found += search.expand_state(cand, seen, with_simplify=level == 0)
        if not level_found:
            break
        pool += level_found
        beam = sorted(level_found, key=_Candidate.rank)[:width]
        # A chain may have to get worse before it gets better (expand, then cancel), but not for long.
        level_best = min(c.cost for c in level_found)
        stale = 0 if level_best < best_cost else stale + 1
        best_cost = min(best_cost, level_best)
        if stale >= STALE_LEVELS or search.out_of_time():
            break

    def improvers() -> list[_Candidate]:
        return sorted((c for c in pool if c.cost < search.start_cost), key=_Candidate.rank)

    if goal == "simplify" and not search.out_of_time():
        top = (improvers() or [start])[0]
        pool += search.expand_state(top, seen, with_simplify=True)

    allow_symbolic = time.monotonic() < started + 0.8 * limits.timeout_seconds
    finals: list[tuple[_Candidate, TransformChain, list[TransformStep]]] = []
    seen_results: set[sp.Expr] = set()
    for cand in improvers():
        if cand.expr in seen_results:
            continue
        done = _finalize(expr, cand, allow_symbolic)
        if done is None:
            search.discarded += 1
            continue
        seen_results.add(cand.expr)
        finals.append((cand, done[0], done[1]))
        if len(finals) > MAX_ALTERNATIVES:
            break

    blocked = list(search.blocked.values())
    common = {
        "goal": goal,
        "input_latex": input_latex,
        "candidates_tried": search.tried,
        "discarded_not_verified": search.discarded,
        "search_complete": not search.truncated,
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
    cut = " The search was cut short by its time budget." if search.truncated else ""
    if finals:
        _, best, steps = finals[0]
        return TransformResult(
            status=Status.OK,
            result_latex=best.result_latex,
            result_text=best.result_text,
            steps=steps,
            verification=best.verification,
            alternatives=[chain for _, chain, _ in finals[1:]],
            blocked=blocked,
            message=f"{len(steps)} verified step(s). {CAVEAT}{cut}",
            **common,
        )
    if blocked:
        return TransformResult(
            status=Status.CONDITIONS_NOT_SATISFIED,
            result_latex=input_latex,
            result_text=str(expr),
            blocked=blocked,
            message="Identities that would improve this expression match, but their conditions "
            "are not provable from the declared assumptions, so nothing was changed." + cut,
            **common,
        )
    return TransformResult(
        status=Status.UNCHANGED,
        result_latex=input_latex,
        result_text=str(expr),
        message=(
            f"No verified chain of up to {depth} steps improves this expression for the goal "
            f"'{goal}'. This does not show that none exists." + cut
        ),
        **common,
    )
