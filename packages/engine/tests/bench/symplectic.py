"""Poisson brackets and exact linear canonical (symplectic) maps. Plain SymPy: nothing here imports
`minerva_engine`, so the reference side of a benchmark never shares code with the discovery side.

Phase variables are ordered `z = (q_1..q_n, p_1..p_n)`. A map is a matrix `M` with `z = M z'`: the old
variables expressed through the new ones. It is canonical iff `M^T J M = J` with `J = [[0, I], [-I, 0]]`.
Every map is built from elementary symplectic factors with small rational entries, and the product is
checked exactly, never assumed.
"""

from __future__ import annotations

import random
from typing import Any

import sympy as sp


def phase_symbols(n: int) -> tuple[list[sp.Symbol], list[sp.Symbol]]:
    q = [sp.Symbol(f"q_{i}") for i in range(1, n + 1)]
    p = [sp.Symbol(f"p_{i}") for i in range(1, n + 1)]
    return q, p


def bracket(f: sp.Expr, g: sp.Expr, q: list[sp.Symbol], p: list[sp.Symbol]) -> sp.Expr:
    """{f, g} = sum df/dq dg/dp - df/dp dg/dq."""
    return sp.Add(
        *[
            sp.diff(f, qi) * sp.diff(g, pi) - sp.diff(f, pi) * sp.diff(g, qi)
            for qi, pi in zip(q, p, strict=True)
        ]
    )


def symplectic_form(n: int) -> sp.Matrix:
    return sp.Matrix(sp.BlockMatrix([[sp.zeros(n), sp.eye(n)], [-sp.eye(n), sp.zeros(n)]]))


def is_symplectic(m: sp.Matrix, n: int) -> bool:
    j = symplectic_form(n)
    return bool((m.T * j * m - j).is_zero_matrix)


def _blocks(a: sp.Matrix, b: sp.Matrix, c: sp.Matrix, d: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(sp.BlockMatrix([[a, b], [c, d]]))


def _symmetric(n: int, rng: random.Random, choices: list[int]) -> sp.Matrix:
    s = sp.zeros(n)
    for i in range(n):
        for j in range(i, n):
            s[i, j] = s[j, i] = rng.choice(choices)
    return s


def elementary(kind: str, n: int, rng: random.Random) -> sp.Matrix:
    """One elementary symplectic matrix. `diagonal` keeps every coordinate separate."""
    eye, zero = sp.eye(n), sp.zeros(n)
    if kind == "scale":  # q_i -> c q_i, p_i -> p_i / c : keeps a Hamiltonian separable
        a = sp.eye(n)
        i = rng.randrange(n)
        a[i, i] = rng.choice([sp.Rational(1, 2), 2, 3, sp.Rational(1, 3), -1])
        return _blocks(a, zero, zero, a.T.inv())
    if kind == "point":  # q -> A q, p -> A^-T p : mixes the coordinates
        a = sp.eye(n)
        for _ in range(rng.randint(1, 2)):
            if n > 1:
                i, j = rng.sample(range(n), 2)
                e = sp.eye(n)
                e[i, j] = rng.choice([-2, -1, 1, 2])
                a = e * a
        return _blocks(a, zero, zero, a.T.inv())
    if kind == "shear_p":  # p -> p + S q
        return _blocks(eye, zero, _symmetric(n, rng, [0, 0, 1, -1, 2]), eye)
    if kind == "shear_q":  # q -> q + S p
        return _blocks(eye, _symmetric(n, rng, [0, 0, 1, -1, 2]), zero, eye)
    if kind == "swap":  # (q_i, p_i) -> (p_i, -q_i)
        m = sp.eye(2 * n)
        i = rng.randrange(n)
        m[i, i] = m[n + i, n + i] = 0
        m[i, n + i], m[n + i, i] = 1, -1
        return m
    raise ValueError(f"unknown kind {kind}")


SEPARABLE_KINDS = ("scale", "swap")
MIXING_KINDS = ("scale", "point", "shear_p", "shear_q", "swap")


def random_map(
    n: int, rng: random.Random, steps: int, kinds: tuple[str, ...] = MIXING_KINDS
) -> tuple[sp.Matrix, list[dict[str, Any]]]:
    """A product of `steps` elementary factors, exact, with the recipe that reproduces it."""
    m = sp.eye(2 * n)
    recipe: list[dict[str, Any]] = []
    for _ in range(steps):
        kind = rng.choice(kinds)
        factor = elementary(kind, n, rng)
        assert is_symplectic(factor, n), kind
        m = m * factor
        recipe.append({"kind": kind, "matrix": [[str(x) for x in row] for row in factor.tolist()]})
    if not is_symplectic(m, n):
        raise AssertionError("the product of canonical maps is not canonical")
    return m, recipe


def map_from_recipe(recipe: list[dict[str, Any]], n: int) -> sp.Matrix:
    m = sp.eye(2 * n)
    for step in recipe:
        m = m * sp.Matrix([[sp.Rational(x) for x in row] for row in step["matrix"]])
    return m


def transform(expr: sp.Expr, m: sp.Matrix, q: list[sp.Symbol], p: list[sp.Symbol]) -> sp.Expr:
    """Express `expr(z)` in the new variables: substitute z = M z'."""
    z = list(q) + list(p)
    images = m * sp.Matrix(z)
    return sp.expand(expr.subs(dict(zip(z, images, strict=True)), simultaneous=True))


def canonical_brackets_preserved(m: sp.Matrix, q: list[sp.Symbol], p: list[sp.Symbol]) -> bool:
    """The old coordinates, as functions of the new ones, have the canonical brackets in the new
    variables: {z_a, z_b}' = J_ab, checked entry by entry."""
    n = len(q)
    z = list(q) + list(p)
    images = list(m * sp.Matrix(z))
    j = symplectic_form(n)
    for a in range(2 * n):
        for b in range(2 * n):
            if sp.expand(bracket(images[a], images[b], q, p) - j[a, b]) != 0:
                return False
    return True
