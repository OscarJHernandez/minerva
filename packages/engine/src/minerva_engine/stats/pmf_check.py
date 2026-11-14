"""Checking identities between moments against explicit finite joint distributions.

Sums are expanded for several sizes (`indexed.instantiate`), every random variable becomes a symbol with
two possible values, and a joint distribution with exact rational weights is built: **dependent** unless
independence was declared, in which case the declared blocks are independent and everything else is not.
E, Var and Cov are then evaluated exactly by weighted sums, so both sides become ordinary expressions in
the non-random symbols and the usual checks apply. An identity that silently needs independence fails
on the dependent distributions; one that holds only for independent variables passes only when that was
declared.
"""

from __future__ import annotations

import itertools
import random as _random
from fractions import Fraction

import sympy as sp

from ..errors import MinervaMathError
from ..indexed.instantiate import InstanceError, instances
from ..indexed.randoms import Covariance, Expectation, Variance
from ..verify import verify_equal
from .models import PmfVerification

MAX_ATOMS = 10  # at most 2**10 outcomes
MODELS_PER_SIZE = 2


class PmfError(MinervaMathError):
    """The independence declaration or the number of random variables cannot be modelled."""


def _base(symbol: sp.Symbol) -> str:
    return str(symbol.name.split("__")[0])


def _blocks(atoms: list[sp.Symbol], independent: list[list[str]]) -> list[list[sp.Symbol]]:
    """Groups of random symbols that may depend on each other; different blocks are independent."""
    multi = [set(g) for g in independent if len(g) > 1]
    singles = {g[0] for g in independent if len(g) == 1}
    if len(multi) > 1:
        raise PmfError("only one group of mutually independent variables is supported")
    group = multi[0] if multi else set()
    blocks: dict[object, list[sp.Symbol]] = {}
    for atom in atoms:
        base = _base(atom)
        if base in singles:
            key: object = ("component", atom.name)
        elif base in group:
            key = ("variable", base)
        else:
            key = "rest"
        blocks.setdefault(key, []).append(atom)
    # a component-independent variable next to other undeclared variables cannot be modelled by
    # independent blocks without claiming more independence than was declared
    if "rest" in blocks and any(k[0] == "component" for k in blocks if isinstance(k, tuple)):
        raise PmfError(
            "component independence is supported only when no other random variable is outside "
            "the declared independence group"
        )
    return list(blocks.values())


def joint_models(
    atoms: list[sp.Symbol], independent: list[list[str]], seed: int
) -> list[tuple[dict[sp.Symbol, sp.Rational], sp.Rational]]:
    if len(atoms) > MAX_ATOMS:
        raise PmfError("too many random variables to model at this size")
    rng = _random.Random(seed)  # noqa: S311 - deterministic test data, not cryptographic
    values = {a: [sp.Rational(rng.randint(-8, 8), 4) for _ in range(2)] for a in atoms}
    for a in atoms:
        if values[a][0] == values[a][1]:
            values[a][1] += 1
    block_outcomes = []
    for block in _blocks(atoms, independent):
        weights = [Fraction(rng.randint(1, 9)) for _ in range(2 ** len(block))]
        total = sum(weights)
        options = []
        for w, picks in zip(weights, itertools.product((0, 1), repeat=len(block)), strict=True):
            options.append(
                (
                    {a: values[a][p] for a, p in zip(block, picks, strict=True)},
                    sp.Rational(w.numerator, w.denominator)
                    / sp.Rational(total.numerator, total.denominator),
                )
            )
        block_outcomes.append(options)
    outcomes = []
    for combo in itertools.product(*block_outcomes):
        assignment: dict[sp.Symbol, sp.Rational] = {}
        weight = sp.Integer(1)
        for part, w in combo:
            assignment.update(part)
            weight *= w
        outcomes.append((assignment, weight))
    return outcomes


def reduce_moments(
    expr: sp.Basic, outcomes: list[tuple[dict[sp.Symbol, sp.Rational], sp.Rational]]
) -> sp.Basic:
    """Replace E, Var and Cov by their exact values over the joint distribution."""

    def mean(values: list[sp.Expr]) -> sp.Expr:
        return sp.expand(
            sum((w * v for (_, w), v in zip(outcomes, values, strict=True)), sp.Integer(0))
        )

    def at(e: sp.Basic) -> list[sp.Expr]:
        return [e.xreplace(assignment) for assignment, _ in outcomes]

    def rec(node: sp.Basic) -> sp.Basic:
        if isinstance(node, Expectation | Variance | Covariance):
            args = [rec(a) for a in node.args]
            values = [at(a) for a in args]
            means = [mean(v) for v in values]
            if isinstance(node, Expectation):
                return means[0]
            if isinstance(node, Variance):
                return mean([(v - means[0]) ** 2 for v in values[0]])
            return mean([(x - means[0]) * (y - means[1]) for x, y in zip(*values, strict=True)])
        if not node.args:
            return node
        return node.func(*[rec(a) for a in node.args])

    return rec(expr)


def verify_moments(
    original: sp.Expr,
    candidate: sp.Expr,
    random: frozenset[str],
    independent: list[list[str]],
) -> PmfVerification:
    """Do the two expressions agree on every explicit joint distribution tried?"""
    sizes: list[dict[str, int]] = []
    agreed = exact = checked = 0
    notes: list[str] = []
    try:
        built = instances(original, candidate)
    except InstanceError as exc:
        return PmfVerification(passed=False, note=str(exc))
    for assignment, (a, b) in built:
        atoms = sorted(
            {
                s
                for e in (a, b)
                for s in e.free_symbols
                if isinstance(s, sp.Symbol) and _base(s) in random
            },
            key=lambda s: s.name,
        )
        for k in range(MODELS_PER_SIZE):
            outcomes = joint_models(atoms, independent, seed=20260101 + 7 * k + len(sizes))
            reduced_a, reduced_b = reduce_moments(a, outcomes), reduce_moments(b, outcomes)
            result = verify_equal(reduced_a, reduced_b)
            checked += 1
            agreed += result.passed
            exact += bool(result.symbolic_zero)
            if not result.passed:
                notes.append(f"differs on a joint distribution at {assignment or 'this instance'}")
        sizes.append({s.name: v for s, v in assignment.items()})
    return PmfVerification(
        sizes=sizes,
        models_checked=checked,
        models_agreed=agreed,
        exact_zero_models=exact,
        passed=checked > 0 and agreed == checked,
        note="; ".join(dict.fromkeys(notes)),
    )
