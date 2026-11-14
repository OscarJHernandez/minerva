"""Microcanonical counting: Omega(E) of N copies of a list of levels, S = k_B ln Omega.

A finite system has a discrete spectrum, so Omega is an exact integer count of microstates and the
temperature 1/T = dS/dE is not a derivative here. Minerva reports a finite-difference estimate over the
neighbouring attainable energies and says so, and compares it with the canonical ensemble as a
*measurement*: ensemble equivalence is never assumed.

The count comes from a generating function over the levels (exact fractions). It is checked against two
different algorithms: brute-force enumeration of every assignment of a level to each labelled copy (when
that is small enough) and an enumeration over occupation vectors with multinomial weights.

Indistinguishable copies divide the count of labelled assignments by N! (the classical-limit counting); the
user chooses, and with that choice Omega need not be an integer.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from fractions import Fraction

import mpmath as mp
import sympy as sp

from ..distributions.models import Condition
from ..latexprint import to_latex
from .models import StatCheck, StatMechInput, StatQuantity, StatStep
from .setup import Problem, Refusal

MAX_COPIES = 40
MAX_LEVELS = 6
BRUTE_FORCE_LIMIT = 200_000


def _fraction(value: sp.Expr) -> Fraction:
    return Fraction(int(value.p), int(value.q))


def _numeric_levels(problem: Problem) -> list[tuple[Fraction, int]]:
    out: list[tuple[Fraction, int]] = []
    for e, g in zip(problem.energies, problem.degeneracies, strict=True):
        if e.free_symbols or g.free_symbols or not e.is_Rational or not g.is_Integer:
            raise Refusal(
                "unsupported_system",
                "microcanonical counting needs numeric (rational) energies and integer degeneracies",
            )
        out.append((_fraction(e), int(g)))
    return out


def count_by_generating_function(
    levels: list[tuple[Fraction, int]], copies: int
) -> dict[Fraction, Fraction]:
    """Omega over labelled copies for every attainable total energy: N! [x^N] prod_k sum_n (g_k x)^n/n!."""
    states: dict[tuple[int, Fraction], Fraction] = {(0, Fraction(0)): Fraction(1)}
    for energy, g in levels:
        nxt: dict[tuple[int, Fraction], Fraction] = {}
        for (used, total), weight in states.items():
            for n in range(copies - used + 1):
                key = (used + n, total + n * energy)
                nxt[key] = nxt.get(key, Fraction(0)) + weight * Fraction(g**n, math.factorial(n))
        states = nxt
    out: dict[Fraction, Fraction] = {}
    for (used, total), weight in states.items():
        if used == copies:
            out[total] = out.get(total, Fraction(0)) + weight * math.factorial(copies)
    return out


def count_by_brute_force(levels: list[tuple[Fraction, int]], copies: int) -> dict[Fraction, int]:
    """Every assignment of a level (and one of its g states) to each labelled copy."""
    out: dict[Fraction, int] = {}
    for assignment in itertools.product(range(len(levels)), repeat=copies):
        energy = sum((levels[k][0] for k in assignment), Fraction(0))
        weight = math.prod(levels[k][1] for k in assignment)
        out[energy] = out.get(energy, 0) + weight
    return out


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def count_by_compositions(levels: list[tuple[Fraction, int]], copies: int) -> dict[Fraction, int]:
    """Sum over occupation vectors (n_k) of N!/prod n_k! prod g_k^n_k with energy sum n_k e_k."""
    out: dict[Fraction, int] = {}
    for vector in _compositions(copies, len(levels)):
        energy = sum((n * lv[0] for n, lv in zip(vector, levels, strict=True)), Fraction(0))
        multinomial = math.factorial(copies) // math.prod(math.factorial(n) for n in vector)
        weight = multinomial * math.prod(lv[1] ** n for n, lv in zip(vector, levels, strict=True))
        out[energy] = out.get(energy, 0) + weight
    return out


def _canonical_beta(levels: list[tuple[Fraction, int]], mean_energy: Fraction) -> mp.mpf | None:
    """The canonical beta whose mean energy per copy is ``mean_energy`` (bisection on exact sums)."""
    lo_e = min(e for e, _ in levels)
    hi_e = max(e for e, _ in levels)
    if not lo_e < mean_energy < hi_e:
        return None
    with mp.workdps(30):
        target = mp.mpf(mean_energy.numerator) / mean_energy.denominator
        es = [mp.mpf(e.numerator) / e.denominator for e, _ in levels]
        gs = [mp.mpf(g) for _, g in levels]

        def mean(beta: mp.mpf) -> mp.mpf:
            shift = min(es)
            weights = [g * mp.exp(-beta * (e - shift)) for e, g in zip(es, gs, strict=True)]
            return sum(w * e for w, e in zip(weights, es, strict=True)) / sum(weights)

        lo, hi = mp.mpf(-200), mp.mpf(200)
        if not mean(hi) < target < mean(lo):
            return None
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mean(mid) > target else (lo, mid)
        return (lo + hi) / 2


def microcanonical(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition]]:
    if request.total_energy is None:
        raise Refusal("parse_error", "give the fixed total energy of the microcanonical ensemble")
    copies = problem.copies_count
    if copies is None or not copies.is_Integer:
        raise Refusal(
            "unsupported_system",
            "give the number of copies as an integer: N copies share the fixed total energy",
        )
    n = int(copies)
    levels = _numeric_levels(problem)
    if n > MAX_COPIES or len(levels) > MAX_LEVELS:
        raise Refusal(
            "resource_limit", f"at most {MAX_COPIES} copies and {MAX_LEVELS} levels are counted"
        )
    target_expr = problem.parse(request.total_energy)
    if target_expr.free_symbols or not target_expr.is_Rational:
        raise Refusal("parse_error", "the total energy must be a rational number")
    target = _fraction(target_expr)

    by_dp = count_by_generating_function(levels, n)
    by_compositions = count_by_compositions(levels, n)
    brute_size = len(levels) ** n
    by_brute = count_by_brute_force(levels, n) if brute_size <= BRUTE_FORCE_LIMIT else None

    dp_here = by_dp.get(target, Fraction(0))
    if dp_here == 0:
        raise Refusal(
            "energy_not_attainable",
            "no microstate has this total energy: it is not a sum of N level energies",
        )
    identical = problem.identical
    labelled = int(dp_here)  # an integer for labelled copies
    omega: Fraction = Fraction(labelled, math.factorial(n)) if identical else Fraction(labelled)

    checks = [
        StatCheck(
            name="enumeration",
            subject="Omega",
            passed=by_compositions.get(target, 0) == labelled,
            points=1,
            agreed=1 if by_compositions.get(target, 0) == labelled else 0,
            detail="occupation vectors with multinomial weights, a different algorithm from the generating function",
        )
    ]
    if by_brute is not None:
        ok = by_brute.get(target, 0) == labelled
        checks.append(
            StatCheck(
                name="enumeration",
                subject="Omega",
                passed=ok,
                points=1,
                agreed=1 if ok else 0,
                detail=f"all {brute_size} assignments of a level to each labelled copy, counted directly",
            )
        )
    total_ok = sum(by_dp.values()) == sum(g for _, g in levels) ** n
    checks.append(
        StatCheck(
            name="normalisation",
            subject="Omega",
            passed=total_ok,
            points=1,
            agreed=1 if total_ok else 0,
            detail="the counts over all energies add up to (sum of degeneracies)^N",
        )
    )

    mp.mp.dps = 30
    omega_text = (
        str(omega.numerator) if omega.denominator == 1 else f"{omega.numerator}/{omega.denominator}"
    )
    kB = to_latex(problem.k_B)
    steps = [
        StatStep(
            name="levels",
            title="Levels, copies and fixed energy",
            latex=r"\qquad ".join(
                rf"\varepsilon_{{{i}}} = {to_latex(e)},\ g_{{{i}}} = {to_latex(g)}"
                for i, (e, g) in enumerate(
                    zip(problem.energies, problem.degeneracies, strict=True), 1
                )
            )
            + rf"\qquad N = {n},\quad E = {to_latex(target_expr)}",
        ),
        StatStep(
            name="multiplicity",
            title="Number of microstates",
            latex=rf"\Omega(E) = {omega_text}",
            text=omega_text,
            conditions=[
                Condition(
                    latex=r"\text{finite count}",
                    established=True,
                    how="an exact integer count of microstates; no thermodynamic limit is assumed",
                )
            ],
        ),
        StatStep(
            name="entropy",
            title="Microcanonical entropy",
            latex=rf"S = {kB} \ln \Omega = {kB}\,\ln\left({omega_text}\right)",
            text=f"k_B*log({omega_text})",
        ),
    ]
    quantities = [
        StatQuantity(
            name="Omega",
            title="Number of microstates",
            latex=rf"{omega_text}",
            text=omega_text,
            route="generating function over the levels, N! [x^N] prod_k sum_n (g_k x)^n / n!"
            + (", divided by N!" if identical else ""),
            checks=checks,
        ),
        StatQuantity(
            name="S",
            title="Microcanonical entropy",
            latex=rf"{kB} \ln \left({omega_text}\right)",
            text=f"k_B*log({omega_text})",
            route="S = k_B ln Omega",
            checks=[],
        ),
    ]
    conditions = [
        Condition(
            latex=r"E \in \{\textstyle\sum_i n_i \varepsilon_i\}",
            established=True,
            how="the fixed energy is attainable: some microstate has it",
        )
    ]
    if identical:
        problem.notes.append(
            "Indistinguishable copies divide the labelled count by N! (classical-limit counting); with "
            "many copies per state that count is not an integer and is not exact quantum statistics."
        )

    # a finite-difference temperature over the neighbouring attainable energies (never a derivative)
    energies = sorted(by_dp)
    index = energies.index(target)
    if 0 < index < len(energies) - 1:
        lo_e, hi_e = energies[index - 1], energies[index + 1]

        def ln_omega(e: Fraction) -> mp.mpf:
            value = by_dp[e] / (math.factorial(n) if identical else 1)
            return mp.log(mp.mpf(value.numerator) / value.denominator)

        beta_micro = (ln_omega(hi_e) - ln_omega(lo_e)) / (
            mp.mpf((hi_e - lo_e).numerator) / (hi_e - lo_e).denominator
        )
        beta_can = _canonical_beta(levels, target / n)
        text = f"beta (finite difference over the neighbouring energies) = {mp.nstr(beta_micro, 8)}"
        if beta_can is not None:
            gap = abs(beta_micro - beta_can)
            if abs(beta_can) > mp.mpf("1e-9"):
                difference = f"relative difference {mp.nstr(gap / abs(beta_can), 3)}"
            else:  # beta is zero (infinite temperature): only an absolute difference means anything
                difference = f"absolute difference {mp.nstr(gap, 3)}"
            text += (
                f"; canonical beta with the same mean energy per copy = "
                f"{mp.nstr(0 if abs(beta_can) < mp.mpf('1e-9') else beta_can, 8)}; {difference}"
            )
        quantities.append(
            StatQuantity(
                name="beta_estimate",
                title="Temperature estimate and canonical comparison (a measurement)",
                latex=r"\beta \approx \frac{\Delta \ln \Omega}{\Delta E}",
                text=text,
                result_kind="numeric",
                route="a finite difference over the neighbouring attainable energies, not a derivative; "
                "the canonical value solves U(beta) = E/N. Their agreement is measured, never assumed",
                checks=[],
            )
        )
    return steps, quantities, conditions
