"""Approximations that say what they assume, what they expand in, how accurate they are and where they hold.

An approximation is never an identity. Each method here reports its assumptions, the expansion parameter,
the order of accuracy it claims and its domain of validity, and measures its error against the independent
numerical oracle at a few values of the parameter; the *order* it claims is a deciding check (a truncation
whose error does not fall at the claimed rate is not shown). The error table is a measurement, never a bound.

- ``high_temperature``: Taylor series of ln Z in beta about 0, verified by ``check_series`` (Cauchy-integral
  coefficients and a remainder that falls at the claimed order) and by the enumerated ln Z.
- ``low_temperature``: the ground level plus the first Boltzmann corrections of the excited levels, valid
  when the order of the levels is decided; the limits U -> E0 and S -> k_B ln g0 are measured.
- ``stirling``: ln N! replaced by its asymptotic form; the neglected term falls like 1/(12 N).
- ``thermodynamic_limit``: the free energy per site of a chain from its dominant transfer-matrix eigenvalue,
  with the finite-size error measured against brute-force enumeration.
"""

from __future__ import annotations

import itertools
from typing import Any

import mpmath as mp
import sympy as sp

from ..distributions.models import Condition
from ..latexprint import to_latex
from ..series import check_series
from .canonical import (
    Canonical,
    make_canonical,
    make_oracle,
    partition_conditions,
    partition_steps,
    tidy,
)
from .models import (
    Approximation,
    ErrorRow,
    StatCheck,
    StatMechInput,
    StatQuantity,
    StatStep,
)
from .oracle import DPS, Point, sample_points
from .setup import Problem, Refusal

SHOWN = 6  # significant digits in the error tables


def _nstr(x: Any) -> str:
    return str(mp.nstr(x, SHOWN))


def _quantity(
    canon: Canonical,
    name: str,
    title: str,
    expr: sp.Expr,
    route: str,
    checks: list[StatCheck],
    kind: str = "asymptotic",
) -> StatQuantity:
    return StatQuantity(
        name=name,
        title=title,
        latex=canon.latex(expr),
        text=canon.text(expr),
        result_kind=kind,
        route=route,
        checks=checks,
    )


def _order_check(
    name: str,
    subject: str,
    errors: list[Any],
    power: int,
    detail: str,
    *,
    slack: float = 3.0,
) -> StatCheck:
    """The errors at beta, beta/2, beta/4 must shrink by about 2^power each time (within a slack)."""
    ok = True
    agreed = 0
    for big, small in itertools.pairwise(errors):
        tiny = mp.mpf(10) ** -(DPS - 12)
        if abs(small) <= tiny:  # the truncation is exact (or below the working precision)
            agreed += 1
            continue
        if abs(big) / abs(small) >= mp.mpf(2) ** power / slack:
            agreed += 1
        else:
            ok = False
    return StatCheck(
        name=name,
        subject=subject,
        passed=ok and agreed == len(errors) - 1,
        points=len(errors) - 1,
        agreed=agreed,
        detail=detail,
    )


# -- high temperature -----------------------------------------------------------------------
def high_temperature(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition], Approximation]:
    order = request.order
    canon = make_canonical(problem)
    b = problem.beta
    if sp.count_ops(canon.lnZ) > 400:
        raise Refusal("resource_limit", "ln Z is too large to expand")
    try:
        z0 = sp.limit(canon.Z, b, 0, "+")
    except Exception as exc:
        raise Refusal(
            "singular_expansion_point", "the value of Z at beta = 0 could not be decided"
        ) from exc
    if not (z0.is_finite is True and z0.is_positive is True):
        raise Refusal(
            "singular_expansion_point",
            "Z is not finite and positive at beta = 0 (it diverges there, as for an oscillator or a "
            "density of states), so ln Z has no Taylor series about beta = 0; Laurent and logarithmic "
            "expansions are not supported",
        )
    series = sp.series(canon.lnZ, b, 0, order + 1).removeO()
    poly = sp.Poly(sp.expand(series), b)
    if poly.degree() < 1 and order >= 1 and series.has(sp.log):
        raise Refusal("singular_expansion_point", "ln Z is not analytic at beta = 0")
    coefficients = [tidy(c) for c in reversed(poly.all_coeffs())]
    coefficients += [sp.Integer(0)] * (order + 1 - len(coefficients))
    candidate = sum(c * b**k for k, c in enumerate(coefficients))
    verification = check_series(canon.lnZ, b, sp.Integer(0), candidate, order)
    series_check = StatCheck(
        name="series_check",
        subject="lnZ",
        passed=verification.passed,
        points=verification.parameter_sets_checked,
        agreed=verification.parameter_sets_checked if verification.passed else 0,
        detail=verification.note,
    )
    oracle = make_oracle(problem)
    pt = sample_points(problem, 1)[0]
    with mp.workdps(DPS):
        spread = mp.sqrt(max(oracle.cumulants1(pt, mp.mpf(10) ** -8, 2)[1], mp.mpf(0)))
        b0 = mp.mpf("0.3") / spread if spread > 0 else mp.mpf("0.3")
        betas = [b0, b0 / 2, b0 / 4]
        rows: list[ErrorRow] = []
        ln_errors: list[Any] = []
        u_errors: list[Any] = []
        u_series = -sp.diff(candidate, b)
        for beta in betas:
            at = {**pt, b: beta}
            exact = oracle.quantity("lnZ", pt, beta=beta)
            approx = oracle.eval_expr(candidate, at)
            ln_errors.append(approx - exact)
            u_exact = oracle.quantity("U", pt, beta=beta)
            u_approx = oracle.eval_expr(u_series, at)
            u_errors.append(u_approx - u_exact)
            rows.append(
                ErrorRow(
                    label=f"beta = {_nstr(beta)}",
                    exact=_nstr(exact),
                    approximate=_nstr(approx),
                    error=_nstr(approx - exact),
                    note="ln Z",
                )
            )
        ln_check = _order_check(
            "residual_order",
            "lnZ",
            ln_errors,
            order + 1,
            f"the error of the order-{order} polynomial shrinks by about 2^{order + 1} when beta is halved",
        )
        u_check = _order_check(
            "residual_order",
            "U",
            u_errors,
            order,
            f"minus the derivative of the polynomial has error O(beta^{order}): it shrinks by about 2^{order}",
        )
    steps = [
        *partition_steps(problem, canon),
        StatStep(
            name="expansion",
            title=f"High-temperature expansion of ln Z to order {order}",
            latex=rf"\ln Z = {canon.latex(candidate)} + O\left({to_latex(b)}^{{{order + 1}}}\right)",
            result_kind="asymptotic",
            conditions=[
                Condition(
                    latex=r"Z(0) \in (0, \infty)",
                    established=True,
                    how=f"the limit of Z as beta -> 0 is {to_latex(z0)}, finite and positive",
                )
            ],
        ),
    ]
    quantities = [
        _quantity(
            canon,
            "lnZ_series",
            f"ln Z, order {order} in beta",
            candidate,
            "Taylor series about beta = 0; coefficients checked by a Cauchy-integral estimate and the "
            "remainder's decay, not by SymPy's series",
            [series_check, ln_check],
        ),
        _quantity(
            canon,
            "U_series",
            f"Internal energy, order {order - 1} in beta",
            u_series,
            "minus the derivative of the polynomial for ln Z",
            [u_check],
        ),
    ]
    radius = f"{verification.radius:.3g}" if verification.radius else "not determined"
    approx = Approximation(
        method="high_temperature",
        assumptions=[
            "beta is small: beta times the spread of the energies is much less than 1",
            "ln Z is analytic at beta = 0 (Z(0) finite and positive)",
            "the levels do not depend on beta and the other state variables are fixed",
        ],
        parameter="beta",
        order=f"truncation error O(beta^{order + 1}) for ln Z, O(beta^{order}) for U",
        validity=(
            "inside the radius of convergence of the series, set by the nearest zero of Z(beta) in the "
            f"complex plane; Cauchy-estimate radius at the checked parameter values: {radius}"
        ),
        error_table=rows,
    )
    return steps, quantities, partition_conditions(problem), approx


# -- low temperature ------------------------------------------------------------------------
def low_temperature(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition], Approximation]:
    if problem.kind not in {"levels", "quantum"} or problem.copies_count is not None:
        raise Refusal(
            "nothing_to_approximate",
            "the low-temperature limit is applied to a list of levels (or the eigenvalues of H) "
            "without copies",
        )
    canon = make_canonical(problem)
    energies, degeneracies = problem.energies, problem.degeneracies
    ground = [
        i
        for i, e in enumerate(energies)
        if all((energies[j] - e).is_nonnegative is True for j in range(len(energies)))
    ]
    if not ground:
        raise Refusal(
            "ordering_not_decided",
            "the lowest level is not decided from the declared assumptions (declare the symbols "
            "so that the order of the levels follows, for example epsilon positive)",
        )
    e0 = energies[ground[0]]
    in_ground = [i for i, e in enumerate(energies) if sp.simplify(e - e0) == 0]
    excited = [i for i in range(len(energies)) if i not in in_ground]
    if not excited:
        raise Refusal(
            "nothing_to_approximate", "a single level has no temperature dependence to expand"
        )
    if any((energies[i] - e0).is_positive is not True for i in excited):
        raise Refusal(
            "ordering_not_decided", "an excited level is not provably above the ground level"
        )
    g0 = sp.Add(*[degeneracies[i] for i in in_ground])
    gaps = [sp.simplify(energies[i] - e0) for i in excited]
    lowest = [
        k
        for k, gap in enumerate(gaps)
        if all((other - gap).is_nonnegative is True for other in gaps)
    ]
    if not lowest:
        raise Refusal(
            "ordering_not_decided",
            "the lowest excitation energy is not decided from the assumptions",
        )
    gap_min = gaps[lowest[0]]
    b, kB = problem.beta, problem.k_B
    corrections = [
        (degeneracies[i] / g0, gap, sp.exp(-b * gap)) for i, gap in zip(excited, gaps, strict=True)
    ]
    ln_z_lt = -b * e0 + sp.log(g0) + sum(w * f for w, _, f in corrections)
    u_lt = e0 + sum(w * gap * f for w, gap, f in corrections)
    s_lt = kB * (sp.log(g0) + sum(w * (1 + b * gap) * f for w, gap, f in corrections))
    oracle = make_oracle(problem)
    pt = sample_points(problem, 1)[0]
    rows: list[ErrorRow] = []
    ok_all = True
    agreed = 0
    with mp.workdps(DPS):
        delta = oracle.eval_expr(gap_min, pt)
        weight = 1 + sum(oracle.eval_expr(w, pt) for w, _, _ in corrections)
        constant = 2 * weight
        e0_value = oracle.eval_expr(e0, pt)
        previous: Any = None
        for c in (3, 6, 12):
            beta = mp.mpf(c) / delta
            at = {**pt, b: beta}
            exact = oracle.quantity("U", pt, beta=beta)
            approx = oracle.eval_expr(u_lt, at)
            relative = abs(exact - approx) / abs(exact - e0_value)
            bound = constant * mp.exp(-beta * delta)
            good = relative <= bound and (previous is None or relative <= previous)
            ok_all &= bool(good)
            agreed += 1 if good else 0
            previous = relative
            rows.append(
                ErrorRow(
                    label=f"beta*gap = {c}",
                    exact=_nstr(exact),
                    approximate=_nstr(approx),
                    error=_nstr(approx - exact),
                    note=f"relative to U - E0: {_nstr(relative)}; first-order scale {_nstr(bound)}",
                )
            )
        s_large = oracle.quantity("S", pt, beta=mp.mpf(12) / delta)
        s_limit = pt[kB] * mp.log(oracle.eval_expr(g0, pt))
    check = StatCheck(
        name="residual_order",
        subject="U",
        passed=ok_all and agreed == 3,
        points=3,
        agreed=agreed,
        detail="the relative error is within 2(1 + sum g_j/g_0) e^(-beta gap) and falls as beta grows",
    )
    limits = Condition(
        latex=r"U \to E_0,\quad S \to k_B \ln g_0",
        established=True,
        how=(
            f"measured: at beta*gap = 12, S = {_nstr(s_large)} against k_B ln g_0 = {_nstr(s_limit)}; "
            "the limit is asymptotic, reached only as beta -> infinity"
        ),
    )
    steps = [
        *partition_steps(problem, canon),
        StatStep(
            name="expansion",
            title="Low-temperature expansion",
            latex=rf"U \simeq {canon.latex(u_lt)}",
            result_kind="asymptotic",
            conditions=[
                Condition(
                    latex=r"\varepsilon_j > E_0",
                    established=True,
                    how="the ordering of the levels is provable from the declared assumptions",
                ),
                limits,
            ],
        ),
    ]
    quantities = [
        _quantity(
            canon,
            "lnZ_low",
            "ln Z, first excited corrections",
            ln_z_lt,
            "ground level plus the first Boltzmann factor of each excited level",
            [],
        ),
        _quantity(
            canon,
            "U_low",
            "Internal energy at low temperature",
            u_lt,
            "ground energy plus the first corrections of the excited levels",
            [check],
        ),
        _quantity(
            canon,
            "S_low",
            "Entropy at low temperature",
            s_lt,
            "k_B ln g_0 plus the first corrections; the residual entropy k_B ln g_0 survives at T = 0",
            [],
        ),
    ]
    approx = Approximation(
        method="low_temperature",
        assumptions=[
            "the order of the levels is decided and the ground level is separated by a gap",
            "beta times the lowest excitation energy is much greater than 1",
            "first order in the Boltzmann factors of the excited levels",
        ],
        parameter="exp(-beta * lowest gap)",
        order="relative error O(exp(-beta * lowest gap)) in U - E_0",
        validity="beta * gap of about 3 or more; it says nothing about the ground state being "
        "non-degenerate (the residual entropy is k_B ln g_0)",
        error_table=rows,
    )
    return steps, quantities, partition_conditions(problem), approx


# -- Stirling -------------------------------------------------------------------------------
def stirling(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition], Approximation]:
    n = problem.copies_count
    if n is None or not problem.identical:
        raise Refusal(
            "nothing_to_approximate",
            "Stirling's approximation applies to ln N!, which appears when N indistinguishable "
            "copies are divided by N!; there is none in this result",
        )
    canon = make_canonical(problem)
    b, kB = problem.beta, problem.k_B
    without = canon.lnZ + sp.log(sp.factorial(n))  # N ln Z_1: ln N! cancels exactly
    leading = n * sp.log(n) - n
    full = leading + sp.log(2 * sp.pi * n) / 2
    ln_z_full = without - full
    ln_z_lead = without - leading
    u = canon.base["U"]

    def thermo(ln_z: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
        return tidy(-ln_z / b), tidy(kB * (ln_z + b * u))

    f_full, s_full = thermo(ln_z_full)
    f_lead, s_lead = thermo(ln_z_lead)
    oracle = make_oracle(problem)
    mp.mp.dps = 30
    rows: list[ErrorRow] = []
    scaled: list[Any] = []
    for big in (10, 100, 1000, 10000):
        exact = mp.loggamma(big + 1)
        approx = big * mp.log(big) - big + mp.log(2 * mp.pi * big) / 2
        err = exact - approx
        scaled.append(big * err)
        rows.append(
            ErrorRow(
                label=f"N = {big}",
                exact=_nstr(exact),
                approximate=_nstr(approx),
                error=_nstr(err),
                note=f"N * error = {_nstr(big * err)} (the next term is 1/(12 N))",
            )
        )
    order_ok = all(abs(s - mp.mpf(1) / 12) < mp.mpf("0.03") / 12 for s in scaled[1:])
    pts = sample_points(problem, 6)
    agreed = 0
    for pt in pts:
        count = int(mp.nint(oracle.eval_expr(n, pt))) if n.free_symbols else int(n)
        with mp.workdps(DPS):
            exact_ln = oracle.quantity("lnZ", pt)
            approx_ln = oracle.eval_expr(ln_z_full, pt)
            expected_gap = mp.loggamma(count + 1) - (
                count * mp.log(count) - count + mp.log(2 * mp.pi * count) / 2
            )
            tolerance = oracle.tol * 10**3 * max(1, abs(exact_ln))
            # ln Z carries -ln N! and the substituted one -Stirling(N): they differ by minus the error
            if abs((exact_ln - approx_ln) + expected_gap) <= tolerance:
                agreed += 1
    order_check = StatCheck(
        name="residual_order",
        subject="lnZ",
        passed=order_ok and agreed == len(pts),
        points=len(pts),
        agreed=agreed,
        detail="the error of ln N! times N tends to 1/12, and the enumerated ln Z differs from the "
        "substituted one by exactly the Stirling error",
    )
    steps = [
        *partition_steps(problem, canon),
        StatStep(
            name="expansion",
            title="Stirling's approximation for ln N!",
            latex=r"\ln N! \sim N\ln N - N + \tfrac{1}{2}\ln(2\pi N)",
            result_kind="asymptotic",
            conditions=[
                Condition(
                    latex=r"N \gg 1",
                    established=False,
                    how="an assumption about the system size, not something the engine can establish; "
                    "the measured error is below",
                )
            ],
        ),
    ]
    quantities = [
        _quantity(
            canon,
            "lnZ_stirling",
            "ln Z with Stirling's ln N!",
            ln_z_full,
            "ln Z with ln N! replaced",
            [order_check],
        ),
        _quantity(
            canon,
            "F_stirling",
            "Free energy with Stirling's ln N!",
            f_full,
            "F = -(1/beta) ln Z",
            [],
        ),
        _quantity(
            canon,
            "S_stirling",
            "Entropy with Stirling's ln N!",
            s_full,
            "S = k_B (ln Z + beta U), U exact",
            [],
        ),
        _quantity(
            canon,
            "S_leading",
            "Entropy at leading order (the Sackur-Tetrode form for an ideal gas)",
            s_lead,
            "the same with only N ln N - N: the neglected 1/2 ln(2 pi N) is O(ln N), small beside N",
            [],
        ),
        _quantity(
            canon,
            "F_leading",
            "Free energy at leading order",
            f_lead,
            "F = -(1/beta) ln Z at leading order",
            [],
        ),
    ]
    approx = Approximation(
        method="stirling",
        assumptions=[
            "N is large: N >> 1",
            "the copies are indistinguishable (the user's choice), so ln N! is part of ln Z",
        ],
        parameter="1/N",
        order="the error of the form shown in ln N! is about 1/(12 N); the leading form drops 1/2 ln(2 pi N)",
        validity="N of about 10 or more for the full form; it is an asymptotic equivalence, not an "
        "identity, and the internal energy is untouched (ln N! does not depend on beta)",
        error_table=rows,
    )
    return steps, quantities, partition_conditions(problem), approx


# -- thermodynamic limit of a chain ------------------------------------------------------------
def thermodynamic_limit(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition], Approximation]:
    if problem.kind != "chain":
        raise Refusal(
            "nothing_to_approximate",
            "the thermodynamic limit is analysed for a chain (transfer matrix)",
        )
    n = problem.extra["length"]
    if not isinstance(n, sp.Symbol):
        raise Refusal(
            "assumptions_insufficient",
            "give the length as a symbol N (integer, positive) so that N -> infinity is the question",
        )
    canon = make_canonical(problem)
    b = problem.beta
    eigenvalues: dict[sp.Expr, int] = problem.extra["eigenvalues"]
    oracle = make_oracle(problem)
    pts = sample_points(problem, 6)
    keys = list(eigenvalues)

    def numeric(expr: sp.Expr, pt: Point) -> Any:
        return oracle.eval_expr(expr, pt)

    with mp.workdps(DPS):
        winners = {max(range(len(keys)), key=lambda i: mp.re(numeric(keys[i], pt))) for pt in pts}
        runner_ups = {
            sorted(range(len(keys)), key=lambda i: -abs(numeric(keys[i], pt)))[1]
            if len(keys) > 1
            else 0
            for pt in pts
        }
    if len(winners) != 1:
        raise Refusal(
            "ordering_not_decided",
            "which transfer-matrix eigenvalue is the largest changes with the parameters; the "
            "dominant one is not decided",
        )
    lam_max = keys[winners.pop()]
    f_inf = tidy(-sp.log(lam_max) / b)
    e_inf = tidy(-sp.diff(sp.log(lam_max), b))
    rows: list[ErrorRow] = []
    agreed = 0
    previous: Any = None
    monotone = True
    pt = pts[0]
    with mp.workdps(DPS):
        beta = pt[b]
        lam1 = numeric(lam_max, pt)
        f_limit = -mp.log(lam1) / beta
        for length in (2, 4, 6, 8, 10, 12):
            at = {**pt, n: mp.mpf(length)}
            exact = oracle.quantity("F", at) / length
            error = exact - f_limit
            others = sum(
                mult * (numeric(lam, pt) / lam1) ** length
                for lam, mult in eigenvalues.items()
                if lam != lam_max
            )
            predicted = -mp.log(1 + others) / (beta * length)
            matches = abs(error - predicted) <= mp.mpf(10) ** -15 * max(1, abs(exact))
            decreasing = previous is None or abs(error) <= abs(previous) + mp.mpf(10) ** -30
            agreed += 1 if matches and decreasing else 0
            monotone &= bool(decreasing)
            previous = error
            rows.append(
                ErrorRow(
                    label=f"N = {length}",
                    exact=_nstr(exact),
                    approximate=_nstr(f_limit),
                    error=_nstr(error),
                    note=f"leading finite-size term {_nstr(predicted)}",
                )
            )
        ratio = [abs(numeric(keys[i], pt) / lam1) for i in runner_ups]
    check = StatCheck(
        name="residual_order",
        subject="F",
        passed=agreed == 6,
        points=6,
        agreed=agreed,
        detail="the brute-force F_N/N differs from the limit by -(1/(beta N)) ln(1 + sum (lambda_i/lambda_max)^N), "
        "exponentially small in N",
    )
    steps = [
        *partition_steps(problem, canon),
        StatStep(
            name="expansion",
            title="Thermodynamic limit from the dominant eigenvalue",
            latex=rf"f_\infty = \lim_{{N\to\infty}} \frac{{F}}{{N}} = -\frac{{1}}{{{to_latex(b)}}}\ln\lambda_{{\max}} "
            rf"= {canon.latex(f_inf)}",
            result_kind="exact",
            conditions=[
                Condition(
                    latex=r"\lambda_{\max}\ \text{simple and dominant}",
                    established=True,
                    how="the transfer matrix has positive entries (Perron-Frobenius); the same eigenvalue "
                    "is the largest at every sampled parameter point",
                )
            ],
        ),
    ]
    quantities = [
        _quantity(
            canon,
            "f_inf",
            "Free energy per site, N -> infinity",
            f_inf,
            "-(1/beta) ln of the dominant eigenvalue",
            [check],
            kind="exact",
        ),
        _quantity(
            canon,
            "e_inf",
            "Energy per site, N -> infinity",
            e_inf,
            "minus d/d beta of ln of the dominant eigenvalue",
            [],
            kind="exact",
        ),
    ]
    rate = f"{_nstr(max(ratio))}" if ratio else "n/a"
    approx = Approximation(
        method="thermodynamic_limit",
        assumptions=[
            "N tends to infinity at a fixed temperature, with periodic boundary conditions",
            "the largest transfer-matrix eigenvalue is simple and strictly dominant (positive matrix)",
        ],
        parameter="N",
        order="finite-size error of F/N is -(1/(beta N)) ln(1 + (lambda_2/lambda_1)^N + ...): "
        f"exponentially small in N (ratio |lambda_2/lambda_1| = {rate} at the first sample point)",
        validity="the free energy per site is analytic in beta for beta > 0 (a simple dominant "
        "eigenvalue), so this chain has no phase transition at finite temperature",
        error_table=rows,
    )
    return steps, quantities, partition_conditions(problem), approx


def approximate(
    problem: Problem, request: StatMechInput, deadline: float
) -> tuple[list[StatStep], list[StatQuantity], list[Condition], Approximation]:
    method = request.method
    if method is None:
        raise Refusal(
            "parse_error",
            "name the method: high_temperature, low_temperature, stirling or thermodynamic_limit",
        )
    runners = {
        "high_temperature": high_temperature,
        "low_temperature": low_temperature,
        "stirling": stirling,
        "thermodynamic_limit": thermodynamic_limit,
    }
    return runners[method](problem, request, deadline)
