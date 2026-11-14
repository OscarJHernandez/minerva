"""Vector identities as rewrite rules over the typed tree.

A rule (`VectorRule`) is handed over by the knowledge layer (the engine never reads the handbook). Its
two sides are written in text syntax over typed pattern variables (`f`, `g` scalar, `A`, `B`, `C`
vector). This module only *proposes* rewrites: it matches a side at a chosen part of the expression and
instantiates the other side. Whether the rewritten expression equals the original is decided elsewhere
(`steps.py`), by evaluating the difference of the two whole expressions with both legs of the vector
verifier; the matcher is never trusted.

Matching follows the types: a scalar variable binds any scalar part, a vector variable any vector part;
in a product the scalar factors go to the scalar variable together. Sums are matched term by term,
including a subset of the terms of a longer sum. Nothing here evaluates anything.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import sympy as sp

from ..errors import ParseError
from ..models import Limits
from .expressions import (
    Adv,
    Cross,
    Curl,
    Div,
    Dot,
    Grad,
    Lap,
    Pd,
    kind_of,
)
from .models import VectorRule
from .parse import parse_vector_text

Kind = Literal["scalar", "vector"]
Bindings = dict[sp.Symbol, sp.Expr]
MAX_ADD_TERMS = 8
MAX_REWRITES = 200

_HEAD_FEATURES = {
    Grad: "grad",
    Div: "div",
    Curl: "curl",
    Lap: "lap",
    Dot: "dot",
    Cross: "cross",
    Pd: "pd",
    Adv: "adv",
}


@dataclass(frozen=True)
class ParsedRule:
    rule: VectorRule
    lhs: sp.Expr
    rhs: sp.Expr
    variables: dict[sp.Symbol, Kind]

    @property
    def can_reverse(self) -> bool:
        """The right side can be matched only if it is more than a bare number or symbol and
        contains every variable of the left side (otherwise the rewrite would invent them)."""
        if self.rhs.is_Number or isinstance(self.rhs, sp.Symbol):
            return False
        return bool(set(self.variables) <= self.rhs.free_symbols)


def parse_rule(rule: VectorRule, limits: Limits) -> ParsedRule:
    vectors = [n for n, k in rule.variables.items() if k == "vector"]
    names = list(rule.variables)
    time = {"t": ["real"]}  # `dt(...)` in a rule differentiates with respect to the real symbol t
    lhs = parse_vector_text(
        rule.lhs, vectors=vectors, names=names, directions=(), assumptions=time, limits=limits
    )
    rhs = parse_vector_text(
        rule.rhs, vectors=vectors, names=names, directions=(), assumptions=time, limits=limits
    )
    symbols: dict[sp.Symbol, Kind] = {}
    for s in lhs.free_symbols | rhs.free_symbols:
        if s.name == "t" and "t" not in rule.variables:
            continue
        if s.name not in rule.variables:
            raise ParseError(f"rule {rule.id}: {s.name} is not a declared pattern variable")
        symbols[s] = rule.variables[s.name]
    return ParsedRule(rule, lhs, rhs, symbols)


# -- matching --------------------------------------------------------------------------------
def _is_zero(e: sp.Expr) -> bool:
    return bool(e == 0)


def match(pattern: sp.Expr, target: sp.Expr, variables: dict[sp.Symbol, Kind]) -> list[Bindings]:
    """Every way `pattern` matches `target` exactly (the whole of it)."""
    return list(_match(pattern, target, variables, {}))


def _match(
    pattern: sp.Expr, target: sp.Expr, variables: dict[sp.Symbol, Kind], bound: Bindings
) -> Iterator[Bindings]:
    if isinstance(pattern, sp.Symbol) and pattern in variables:
        try:
            if kind_of(target) != variables[pattern]:
                return
        except ParseError:
            return
        if pattern in bound:
            if bound[pattern] == target:
                yield bound
            return
        yield {**bound, pattern: target}
        return
    if pattern.is_Atom:
        if pattern == target:
            yield bound
        return
    if isinstance(pattern, sp.Mul):
        yield from _match_mul(pattern, target, variables, bound)
        return
    if isinstance(pattern, sp.Add):
        yield from _match_add(pattern, target, variables, bound, exact=True)
        return
    if type(pattern) is not type(target) or len(pattern.args) != len(target.args):
        return
    yield from _match_args(list(pattern.args), list(target.args), variables, bound)


def _match_args(
    patterns: list[sp.Expr],
    targets: list[sp.Expr],
    variables: dict[sp.Symbol, Kind],
    bound: Bindings,
) -> Iterator[Bindings]:
    if not patterns:
        yield bound
        return
    for b in _match(patterns[0], targets[0], variables, bound):
        yield from _match_args(patterns[1:], targets[1:], variables, b)


def _kind(e: sp.Expr) -> Kind:
    return kind_of(e)


def _match_mul(
    pattern: sp.Mul, target: sp.Expr, variables: dict[sp.Symbol, Kind], bound: Bindings
) -> Iterator[Bindings]:
    """A product matches a product: the vector factor with the vector factor, the scalar factors
    of the target all go to the (one) scalar variable of the pattern."""
    if not isinstance(target, sp.Mul):
        return
    p_vec = [a for a in pattern.args if _kind(a) == "vector"]
    p_sca = [a for a in pattern.args if _kind(a) == "scalar"]
    t_vec = [a for a in target.args if _kind(a) == "vector"]
    t_sca = [a for a in target.args if _kind(a) == "scalar"]
    if len(p_vec) != len(t_vec) or len(p_vec) > 1:
        return
    if len(p_sca) == 1 and isinstance(p_sca[0], sp.Symbol) and p_sca[0] in variables:
        if not t_sca:
            return  # a product with no scalar factor is not a product of a scalar and a vector
        scalar = sp.Mul(*t_sca)
        for b in _match(p_sca[0], scalar, variables, bound):
            if p_vec:
                yield from _match(p_vec[0], t_vec[0], variables, b)
            else:
                yield b
        return
    if len(p_sca) != len(t_sca):
        return
    for perm in itertools.permutations(t_sca):
        for b in _match_args(list(p_sca), list(perm), variables, bound):
            if p_vec:
                yield from _match(p_vec[0], t_vec[0], variables, b)
            else:
                yield b


def _match_add(
    pattern: sp.Add,
    target: sp.Expr,
    variables: dict[sp.Symbol, Kind],
    bound: Bindings,
    exact: bool,
) -> Iterator[Bindings]:
    if not isinstance(target, sp.Add):
        return
    if exact and len(pattern.args) != len(target.args):
        return
    if len(target.args) > MAX_ADD_TERMS:
        return
    for chosen in itertools.permutations(target.args, len(pattern.args)):
        yield from _match_args(list(pattern.args), list(chosen), variables, bound)


# -- the tree: parts and splicing --------------------------------------------------------------
Path = tuple[int, ...]


def parts(expr: sp.Expr, prefix: Path = ()) -> Iterator[tuple[Path, sp.Expr]]:
    """Every part of the tree with its path (indexes into `.args`), outermost first."""
    yield prefix, expr
    if expr.is_Atom:
        return
    for i, arg in enumerate(expr.args):
        yield from parts(arg, (*prefix, i))


def node_at(expr: sp.Expr, path: Path) -> sp.Expr:
    for i in path:
        expr = expr.args[i]
    return expr


def splice(expr: sp.Expr, path: Path, new: sp.Expr) -> sp.Expr:
    if not path:
        return new
    args = list(expr.args)
    args[path[0]] = splice(args[path[0]], path[1:], new)
    return expr.func(*args)


# -- rewrites ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Rewrite:
    rule: ParsedRule
    reverse: bool
    path: Path
    before: sp.Expr  # the part that was matched
    after: sp.Expr  # what replaces it
    result: sp.Expr  # the whole expression after the rewrite


def rewrites(
    expr: sp.Expr, rule: ParsedRule, *, reverse: bool = False, only: Path | None = None
) -> list[Rewrite]:
    """Every way `rule` rewrites one part of `expr` (all parts, or just the one at `only`)."""
    if reverse and not rule.can_reverse:
        return []
    source, replacement = (rule.rhs, rule.lhs) if reverse else (rule.lhs, rule.rhs)
    out: list[Rewrite] = []
    for path, node in parts(expr):
        if only is not None and path != only:
            continue
        for bindings in match(source, node, rule.variables):
            new = replacement.xreplace(bindings)
            out.append(Rewrite(rule, reverse, path, node, new, splice(expr, path, new)))
        if (
            isinstance(source, sp.Add)
            and isinstance(node, sp.Add)
            and len(source.args) < len(node.args)
        ):
            out += _partial_sums(expr, rule, reverse, source, replacement, path, node)
        if len(out) >= MAX_REWRITES:
            break
    return _distinct(out)


def _partial_sums(
    expr: sp.Expr,
    rule: ParsedRule,
    reverse: bool,
    source: sp.Expr,
    replacement: sp.Expr,
    path: Path,
    node: sp.Expr,
) -> list[Rewrite]:
    """The pattern is a sum of `k` terms and the part is a longer sum: rewrite those `k` terms."""
    out: list[Rewrite] = []
    if len(node.args) > MAX_ADD_TERMS:
        return out
    k = len(source.args)
    for chosen in itertools.permutations(range(len(node.args)), k):
        picked = [node.args[i] for i in chosen]
        for bindings in _match_args(list(source.args), picked, rule.variables, {}):
            rest = [a for i, a in enumerate(node.args) if i not in chosen]
            new_node = sp.Add(*rest, replacement.xreplace(bindings))
            out.append(
                Rewrite(
                    rule,
                    reverse,
                    path,
                    sp.Add(*picked),
                    replacement.xreplace(bindings),
                    splice(expr, path, new_node),
                )
            )
    return out


def _dot_sorted(expr: sp.Expr) -> sp.Expr:
    """The dot product is symmetric: two rewrites that differ only in the order of its arguments are
    the same rewrite (matching `f` and `g` the other way round in a symmetric rule)."""
    return expr.replace(Dot, lambda a, b: Dot(*sorted((a, b), key=sp.default_sort_key)))


def _distinct(items: list[Rewrite]) -> list[Rewrite]:
    seen: set[sp.Expr] = set()
    out: list[Rewrite] = []
    for r in items:
        key = _dot_sorted(r.result)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# -- canonical form: constants out of linear operators, operators of zero --------------------------
_LINEAR = (Grad, Div, Curl, Lap)


def _is_constant(e: sp.Expr, constants: frozenset[str]) -> bool:
    return all(s.name in constants for s in e.free_symbols)


def canonical(expr: sp.Expr, constants: frozenset[str]) -> sp.Expr:
    """Linear bookkeeping that is true for every field: a constant factor leaves a linear operator
    (`curl(-X) = -curl(X)`), and an operator applied to zero is zero. Each change is an identity; a
    step that uses it is still verified by comparing the whole expressions."""

    def pull(node: sp.Expr) -> sp.Expr:
        arg = node.args[0]
        if arg == 0:
            return sp.Integer(0)
        if isinstance(node, Pd):
            if node.args[0] == 0 or _is_constant(node.args[0], constants):
                return sp.Integer(0)
            arg = node.args[0]
        if isinstance(arg, sp.Mul):
            const = [a for a in arg.args if _is_constant(a, constants)]
            rest = [a for a in arg.args if not _is_constant(a, constants)]
            if const and rest:
                inner = sp.Mul(*rest)
                head = (
                    node.func(inner, *node.args[1:]) if isinstance(node, Pd) else node.func(inner)
                )
                return sp.Mul(*const) * head
        return node

    previous = None
    current = expr
    for _ in range(8):  # a few rounds reach a fixed point on any expression we handle
        if current == previous:
            break
        previous = current
        current = current.replace(lambda e: isinstance(e, (*_LINEAR, Pd)), pull)
    return current


# -- features for retrieval ---------------------------------------------------------------------
def vector_features(expr: sp.Expr) -> set[str]:
    """`vector:curl`, `vector:grad`, ... for the operators in `expr` (retrieval only proposes)."""
    found: set[str] = set()
    for node in sp.preorder_traversal(expr):
        for head, name in _HEAD_FEATURES.items():
            if isinstance(node, head):
                found.add(f"vector:{name}")
    return found
