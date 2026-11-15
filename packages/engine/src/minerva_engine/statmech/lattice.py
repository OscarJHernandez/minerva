"""A periodic one-dimensional chain by the transfer matrix.

The symbolic route is Z = Tr T^N = sum_i m_i lambda_i^N for the transfer matrix
``T_ab = exp(-beta (bond(a, b) + (site(a) + site(b))/2))`` built from the bond and site energies. The
independent oracle never uses a transfer matrix: it enumerates every one of the k^N configurations of a
ring of N sites, adds up the bond and site energies directly, and sums the Boltzmann weights with
log-sum-exp. A transfer matrix that was wrong (a missing half site term, a bond counted once) would not
survive that comparison.

The matrix has strictly positive entries, so by the Perron-Frobenius theorem its largest eigenvalue is
simple and strictly dominant; that is what the thermodynamic limit rests on.
"""

from __future__ import annotations

import itertools
from typing import Any

import mpmath as mp
import sympy as sp

from ..distributions.models import Condition
from ..latexprint import to_latex
from .models import ChainSystem, StatStep
from .oracle import BaseOracle, Point, cumulants_from_raw
from .setup import Problem, Refusal

MAX_CONFIGURATIONS = 200_000
MAX_LENGTH = 14


def fill_chain(problem: Problem, s: ChainSystem) -> set[sp.Symbol]:
    if problem.request.copies is not None:
        raise Refusal("unsupported_system", "copies do not apply to a chain: its length is N")
    for name in ("s", "t"):
        current = set(problem.assumptions.get(name, []))
        problem.assumptions[name] = sorted(current | {"real"})
    sym_s, sym_t = problem.parse("s"), problem.parse("t")
    if not (isinstance(sym_s, sp.Symbol) and isinstance(sym_t, sp.Symbol)):
        raise Refusal("parse_error", "internal symbols s and t")
    clash = {sym_s, sym_t} & set(problem.params)
    if clash:
        raise Refusal("parse_error", "s and t are the names of the neighbouring site values")
    values = [problem.parse(v) for v in s.states]
    if any(v.is_real is not True for v in values) or len(set(values)) != len(values):
        raise Refusal(
            "assumptions_insufficient",
            "the site values must be distinct and provably real (declare their symbols real)",
        )
    bond = problem.parse(s.bond)
    site = problem.parse(s.site)
    if {problem.beta, problem.T, problem.k_B} & (bond.free_symbols | site.free_symbols):
        raise Refusal("unsupported_system", "the energies must not depend on the temperature")
    n = problem.parse(s.length)
    if n.is_integer is not True or n.is_positive is not True:
        raise Refusal(
            "assumptions_insufficient",
            "the length must be a positive integer (declare a symbolic one integer and positive)",
        )
    if n.is_Integer and int(n) > MAX_LENGTH:
        raise Refusal(
            "resource_limit",
            f"at most {MAX_LENGTH} sites: the result is checked against a brute-force enumeration of "
            "every configuration",
        )
    k = len(values)
    bond_matrix = sp.Matrix(k, k, lambda a, b: bond.subs({sym_s: values[a], sym_t: values[b]}))
    if not all(
        sp.simplify(bond_matrix[a, b] - bond_matrix[b, a]) == 0 for a in range(k) for b in range(a)
    ):
        raise Refusal(
            "unsupported_system",
            "the bond must be symmetric, bond(s, t) = bond(t, s), for a symmetric transfer matrix",
        )
    for entry in bond_matrix:
        if entry.is_real is not True:
            raise Refusal(
                "assumptions_insufficient", "the bond energies must be provably real on every pair"
            )
    b = problem.beta
    site_values = [site.subs(sym_s, v) for v in values]
    if any(x.is_real is not True for x in site_values):
        raise Refusal("assumptions_insufficient", "the site energies must be provably real")
    transfer = sp.Matrix(
        k,
        k,
        lambda i, j: sp.exp(-b * (bond_matrix[i, j] + (site_values[i] + site_values[j]) / 2)),
    )
    try:
        eigenvalues = transfer.eigenvals()
    except Exception as exc:
        raise Refusal(
            "unsupported_system", "no closed-form eigenvalues of the transfer matrix"
        ) from exc
    if sum(eigenvalues.values()) != k:
        raise Refusal("unsupported_system", "no closed-form eigenvalues of the transfer matrix")
    z = sp.Add(*[m * lam**n for lam, m in eigenvalues.items()])
    problem.kind = "chain"
    problem.Z1 = z
    problem.extra = {
        "states": values,
        "bond": bond,
        "site": site,
        "s": sym_s,
        "t": sym_t,
        "length": n,
        "transfer": transfer,
        "eigenvalues": eigenvalues,
    }
    conditions = [
        Condition(
            latex=r"Z = \operatorname{Tr}\, T^{N} = \sum_i m_i \lambda_i^{N}",
            established=True,
            how="periodic boundary condition: the trace of the N-th power of the transfer matrix",
        ),
        Condition(
            latex=r"T_{ab} > 0",
            established=True,
            how="every entry is an exponential of a real number, so by the Perron-Frobenius theorem the "
            "largest eigenvalue is simple and strictly dominant",
        ),
    ]
    problem.conditions = conditions
    problem.setup_steps = [
        StatStep(
            name="levels",
            title="Chain",
            latex=rf"E = \sum_{{i}} \mathrm{{bond}}(s_i, s_{{i+1}}) + \sum_i \mathrm{{site}}(s_i),\quad "
            rf"\mathrm{{bond}}(s,t) = {to_latex(bond)},\ \mathrm{{site}}(s) = {to_latex(site)},\ "
            rf"N = {to_latex(n)}",
        ),
        StatStep(
            name="transfer_matrix",
            title="Transfer matrix",
            latex=rf"T = {to_latex(transfer)}",
        ),
        StatStep(
            name="partition_function",
            title="Partition function",
            latex=rf"Z = \operatorname{{Tr}}\,T^{{N}} = {to_latex(z)}",
            conditions=conditions,
        ),
    ]
    return {
        x for e in (bond, site, *values) for x in e.free_symbols if isinstance(x, sp.Symbol)
    } - {
        sym_s,
        sym_t,
    } | {x for x in n.free_symbols if isinstance(x, sp.Symbol)}


class ChainOracle(BaseOracle):
    """Brute-force enumeration of every configuration of the ring."""

    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        x = problem.extra
        self._symbols = sorted(
            {s for e in (x["bond"], x["site"], *x["states"], x["length"]) for s in e.free_symbols}
            - {x["s"], x["t"]},
            key=lambda s: s.name,
        )
        self._bond = sp.lambdify([x["s"], x["t"], *self._symbols], x["bond"], modules="mpmath")
        self._site = sp.lambdify([x["s"], *self._symbols], x["site"], modules="mpmath")
        self._states = [sp.lambdify(self._symbols, v, modules="mpmath") for v in x["states"]]
        self._length = sp.lambdify(self._symbols, x["length"], modules="mpmath")
        self.tol = mp.mpf(10) ** -16
        self.diff_tol = mp.mpf(10) ** -10

    def length(self, point: Point) -> int:
        return int(mp.nint(self._length(*(point[s] for s in self._symbols))))

    def energies(self, point: Point) -> list[Any]:
        args = [point[s] for s in self._symbols]
        values = [f(*args) for f in self._states]
        n = self.length(point)
        k = len(values)
        if k**n > MAX_CONFIGURATIONS:
            raise Refusal(
                "resource_limit", "too many configurations to enumerate at a sample point"
            )
        bond = [[self._bond(a, b, *args) for b in values] for a in values]
        site = [self._site(a, *args) for a in values]
        out: list[Any] = []
        for config in itertools.product(range(k), repeat=n):
            total = sum(site[i] for i in config)
            for i in range(n):
                total += bond[config[i]][config[(i + 1) % n]]
            out.append(total)
        return out

    def lnZ1(self, point: Point, beta: Any) -> Any:
        energies = self.energies(point)
        terms = [-beta * e for e in energies]
        top = max(terms)
        return top + mp.log(sum(mp.exp(t - top) for t in terms))

    def cumulants1(self, point: Point, beta: Any, upto: int) -> list[Any]:
        energies = self.energies(point)
        terms = [-beta * e for e in energies]
        top = max(terms)
        weights = [mp.exp(t - top) for t in terms]
        z = sum(weights)
        raw = [mp.mpf(1)] + [
            sum(w * e**k for w, e in zip(weights, energies, strict=True)) / z for k in range(1, 5)
        ]
        return cumulants_from_raw(raw, upto)
