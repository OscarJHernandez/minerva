"""Expectation, variance and covariance by linearity and bilinearity.

Rules (nothing else is applied):

* E[a X + b Y + c] = a E[X] + b E[Y] + c for constants a, b, c (no independence needed);
* E over a finite sum is the sum of the E (the range is not random);
* E[X Y] = E[X] E[Y] only for variables in a declared independence group;
* Cov is bilinear, Cov(X, X) = Var(X), Cov(c, X) = 0, and Cov(X, Y) = 0 only for declared independence;
* Var(X + Y) = Var X + Var Y + 2 Cov(X, Y) therefore keeps the covariance unless independence is declared.

Whatever independence a result used is returned as a premise: the result then holds *given* it. The
rewrite is a candidate; `pmf_check` verifies it against explicit joint distributions.
"""

from __future__ import annotations

import sympy as sp

from ..indexed.randoms import RANDOM_TYPES, Covariance, Expectation, Variance


class Moments:
    """Rewrites moments of the declared random variables, remembering the independence it used."""

    def __init__(self, random: frozenset[str], independent: list[list[str]]) -> None:
        self.random = random
        self.groups = [frozenset(g) for g in independent]
        self.premises: list[str] = []

    # -- what is random ----------------------------------------------------------------
    def atoms(self, expr: sp.Basic) -> set[sp.Basic]:
        """Random symbols and indexed random elements in `expr` (moments inside count as numbers)."""
        if isinstance(expr, RANDOM_TYPES):
            return set()
        if isinstance(expr, sp.Indexed):
            return {expr} if expr.base.label.name in self.random else set()
        if isinstance(expr, sp.Symbol):
            return {expr} if expr.name in self.random else set()
        found: set[sp.Basic] = set()
        for arg in expr.args:
            found |= self.atoms(arg)
        return found

    @staticmethod
    def _base(atom: sp.Basic) -> str:
        return str(atom.base.label.name if isinstance(atom, sp.Indexed) else atom.name)

    def independent_atoms(self, a: sp.Basic, b: sp.Basic) -> bool:
        if a == b:
            return False
        base_a, base_b = self._base(a), self._base(b)
        if base_a != base_b:
            for group in self.groups:
                if len(group) > 1 and base_a in group and base_b in group:
                    first, second = sorted((base_a, base_b))
                    self._premise(f"{first} and {second} are independent")
                    return True
            return False
        if any(group == {base_a} for group in self.groups):
            ia, ib = a.indices[0], b.indices[0]
            if ia.is_Integer and ib.is_Integer and ia != ib:
                self._premise(f"the components of {base_a} are mutually independent")
                return True
        return False

    def independent(self, u: sp.Basic, v: sp.Basic) -> bool:
        atoms_u, atoms_v = self.atoms(u), self.atoms(v)
        return (
            bool(atoms_u)
            and bool(atoms_v)
            and all(self.independent_atoms(a, b) for a in atoms_u for b in atoms_v)
        )

    def _premise(self, text: str) -> None:
        if text not in self.premises:
            self.premises.append(text)

    def split(self, term: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
        """(factor without random variables, factor with them)."""
        constant: list[sp.Expr] = []
        random: list[sp.Expr] = []
        for factor in sp.Mul.make_args(term):
            (random if self.atoms(factor) else constant).append(factor)
        return sp.Mul(*constant), sp.Mul(*random)

    # -- expectation -------------------------------------------------------------------
    def expectation(self, arg: sp.Expr) -> sp.Expr:
        total: list[sp.Expr] = []
        for term in sp.Add.make_args(sp.expand(arg)):
            constant, rnd = self.split(term)
            if rnd == 1:
                total.append(constant)
            elif isinstance(rnd, sp.Sum) and len(rnd.limits) == 1:
                total.append(constant * sp.Sum(self.expectation(rnd.function), rnd.limits[0]))
            else:
                total.append(constant * self._product(rnd))
        return sp.Add(*total)

    def _product(self, rnd: sp.Expr) -> sp.Expr:
        factors = sp.Mul.make_args(rnd)
        if len(factors) > 1 and all(
            self.independent(u, v) for k, u in enumerate(factors) for v in factors[k + 1 :]
        ):
            return sp.Mul(*[Expectation(f) for f in factors])
        return Expectation(rnd)

    # -- covariance and variance -------------------------------------------------------
    def covariance(self, a: sp.Expr, b: sp.Expr) -> sp.Expr:
        total: list[sp.Expr] = []
        for ta in sp.Add.make_args(sp.expand(a)):
            for tb in sp.Add.make_args(sp.expand(b)):
                total.append(self._cov_term(ta, tb))
        return sp.Add(*total)

    def _fresh_index(self, *used: sp.Basic) -> sp.Symbol:
        names = {s.name for e in used for s in e.free_symbols if isinstance(s, sp.Symbol)}
        names |= {v.name for e in used for n in e.atoms(sp.Sum, sp.Product) for v in n.variables}
        for name in "jklmpqrst":
            if name not in names:
                return sp.Symbol(name, integer=True)
        raise ValueError("no free index name")

    def _cov_term(self, ta: sp.Expr, tb: sp.Expr) -> sp.Expr:
        ca, ra = self.split(ta)
        cb, rb = self.split(tb)
        if ra == 1 or rb == 1:
            return sp.Integer(0)
        if isinstance(ra, sp.Sum) and len(ra.limits) == 1:
            return ca * cb * self._sum_cov(ra, rb)
        if isinstance(rb, sp.Sum) and len(rb.limits) == 1:
            return ca * cb * self._sum_cov(rb, ra)
        if ra == rb:
            return ca * cb * Variance(ra)
        if self.independent(ra, rb):
            return sp.Integer(0)
        return ca * cb * Covariance(ra, rb)

    def _sum_cov(self, s: sp.Sum, other: sp.Expr) -> sp.Expr:
        index, lower, upper = s.limits[0]
        if isinstance(other, sp.Sum) and len(other.limits) == 1:
            j, lower2, upper2 = other.limits[0]
            clash = j == index or other.has(index) or s.has(j)
            if clash:
                fresh = self._fresh_index(s, other)
                other = sp.Sum(other.function.subs(j, fresh), (fresh, lower2, upper2))
                j = fresh
            g = other.function
            diagonal = self._diagonal(s.function, g, index, j, (lower, upper), (lower2, upper2))
            if diagonal is not None:
                return diagonal
            inner = self.covariance(s.function, g)
            return sp.Sum(sp.Sum(inner, (j, lower2, upper2)), (index, lower, upper))
        return sp.Sum(self.covariance(s.function, other), (index, lower, upper))

    def _diagonal(
        self,
        f: sp.Expr,
        g: sp.Expr,
        i: sp.Symbol,
        j: sp.Symbol,
        range_f: tuple[sp.Expr, sp.Expr],
        range_g: tuple[sp.Expr, sp.Expr],
    ) -> sp.Expr | None:
        """Cov(sum_i a_i X_i, sum_j b_j X_j) = sum_i a_i b_i Var(X_i) for independent components."""
        if range_f != range_g:
            return None
        cf, rf = self.split(f)
        cg, rg = self.split(g)
        if not (isinstance(rf, sp.Indexed) and isinstance(rg, sp.Indexed)):
            return None
        if rf.base != rg.base or rf.indices != (i,) or rg.indices != (j,):
            return None
        if not any(group == {self._base(rf)} for group in self.groups):
            return None
        self._premise(f"the components of {self._base(rf)} are mutually independent")
        return sp.Sum(cf * cg.subs(j, i) * Variance(rf), (i, *range_f))

    def variance(self, a: sp.Expr) -> sp.Expr:
        return self.covariance(a, a)

    # -- whole expressions ---------------------------------------------------------------
    def expand(self, expr: sp.Expr) -> sp.Expr:
        """Rewrite every E, Var, Cov inside `expr`, innermost first."""

        def rec(node: sp.Basic) -> sp.Basic:
            if isinstance(node, Expectation):
                return self.expectation(rec(node.args[0]))
            if isinstance(node, Variance):
                return self.variance(rec(node.args[0]))
            if isinstance(node, Covariance):
                return self.covariance(rec(node.args[0]), rec(node.args[1]))
            if not node.args:
                return node
            return node.func(*[rec(a) for a in node.args])

        return sp.expand(rec(expr))

    def to_moments(self, expr: sp.Expr) -> sp.Expr:
        """Express variances and covariances through raw moments: Var U = E[U^2] - E[U]^2."""
        expanded = self.expand(expr)

        def rec(node: sp.Basic) -> sp.Basic:
            if isinstance(node, Variance):
                u = node.args[0]
                return Expectation(sp.expand(u**2)) - Expectation(u) ** 2
            if isinstance(node, Covariance):
                u, v = node.args
                return Expectation(sp.expand(u * v)) - Expectation(u) * Expectation(v)
            if not node.args:
                return node
            return node.func(*[rec(a) for a in node.args])

        return sp.expand(rec(expanded))
