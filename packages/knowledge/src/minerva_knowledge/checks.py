"""Verification of knowledge-base entries at their declared level.

These checkers run on curated, trusted data (not user input), in-process. The test suite
and `python -m minerva_knowledge verify` run them for every entry, so an entry cannot
claim a verification level it does not meet.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import mpmath
import sympy as sp
from minerva_engine import ParseError, parse_text, verify_rule
from minerva_engine.dimensions import DimensionError, check_equation_dimensions
from minerva_engine.models import TensorInput
from minerva_engine.tensor import canonicalize
from minerva_engine.verify import verify_equal

from .models import (
    AntiderivativeCheck,
    DerivativeCheck,
    DistributionCheck,
    Entry,
    EquationCheck,
    IntegralCheck,
    ParamRange,
    RootFormulaCheck,
    SeriesCheck,
    TensorCheck,
)

REL_TOL = 1e-8


@dataclass
class CheckReport:
    entry_id: str
    level: str
    passed: bool
    detail: str
    symbolic_proof: bool = False  # True only when a symbolic zero was established


def _param_grid(params: dict[str, ParamRange]) -> list[dict[str, float]]:
    if not params:
        return [{}]
    names = list(params)
    axes = [[p.min + (p.max - p.min) * f for f in (0.15, 0.5, 0.85)] for p in params.values()]
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*axes)]


def _bound(text: str) -> object:
    value = parse_text(text)
    if value == sp.oo:
        return mpmath.inf
    if value == -sp.oo:
        return -mpmath.inf
    return mpmath.mpf(str(sp.N(value, 30)))


def _close(a: complex | mpmath.mpf | mpmath.mpc, b: complex | mpmath.mpf | mpmath.mpc) -> bool:
    scale = max(1.0, float(abs(a)), float(abs(b)))
    return float(abs(a - b)) <= REL_TOL * scale


def _check_rule(entry: Entry) -> CheckReport:
    assert entry.rule is not None  # noqa: S101
    v = verify_rule(entry.rule)
    detail = (
        f"numeric {v.numeric_points_agreed}/{v.numeric_points_checked}; "
        f"symbolic zero: {v.symbolic_zero}"
    )
    return CheckReport(entry.id, "rule", v.passed, detail, symbolic_proof=v.symbolic_zero)


def _check_derivative(entry: Entry, c: DerivativeCheck) -> CheckReport:
    var = sp.Symbol(c.variable)
    f = parse_text(c.function)
    expected = parse_text(c.result)
    v = verify_equal(sp.diff(f, var), expected)
    return CheckReport(entry.id, "derivative", v.passed, v.note, symbolic_proof=v.symbolic_zero)


def _check_antiderivative(entry: Entry, c: AntiderivativeCheck) -> CheckReport:
    var = sp.Symbol(c.variable)
    integrand = parse_text(c.integrand)
    antiderivative = parse_text(c.antiderivative)
    v = verify_equal(sp.diff(antiderivative, var), integrand)
    return CheckReport(entry.id, "antiderivative", v.passed, v.note, symbolic_proof=v.symbolic_zero)


def _check_integral(entry: Entry, c: IntegralCheck) -> CheckReport:
    mpmath.mp.dps = 25
    var = sp.Symbol(c.variable)
    names = [sp.Symbol(n) for n in c.params]
    f = sp.lambdify([var, *names], parse_text(c.integrand), "mpmath")
    g = sp.lambdify(names, parse_text(c.result), "mpmath")
    lower, upper = _bound(c.lower), _bound(c.upper)
    points = _param_grid(c.params)
    for point in points:
        args = [mpmath.mpf(point[n]) for n in c.params]
        got = mpmath.quad(lambda t, args=args: f(t, *args), [lower, upper])
        want = g(*args)
        if not _close(got, want):
            return CheckReport(entry.id, "integral", False, f"mismatch at {point}: {got} vs {want}")
    return CheckReport(entry.id, "integral", True, f"{len(points)} parameter points agree")


def _check_series(entry: Entry, c: SeriesCheck) -> CheckReport:
    mpmath.mp.dps = 25
    n = sp.Symbol(c.index)
    names = [sp.Symbol(p) for p in c.params]
    term = sp.lambdify([n, *names], parse_text(c.term), "mpmath")
    g = sp.lambdify(names, parse_text(c.result), "mpmath")
    upper = _bound(c.upper)
    points = _param_grid(c.params)
    for point in points:
        args = [mpmath.mpf(point[p]) for p in c.params]
        got = mpmath.nsum(lambda k, args=args: term(k, *args), [c.lower, upper])
        want = g(*args)
        if not _close(got, want):
            return CheckReport(entry.id, "series", False, f"mismatch at {point}: {got} vs {want}")
    return CheckReport(entry.id, "series", True, f"{len(points)} parameter points agree")


def _check_root_formula(entry: Entry, c: RootFormulaCheck) -> CheckReport:
    mpmath.mp.dps = 30
    x = sp.Symbol(c.variable)
    names = [sp.Symbol(n) for n in c.params]
    k = sp.Symbol(c.index) if c.index else None
    poly = sp.lambdify([x, *names], parse_text(c.polynomial), "mpmath")
    args = ([k] if k is not None else []) + names
    formulas = [sp.lambdify(args, parse_text(r), "mpmath") for r in c.roots]
    count = 0
    for point in _param_grid(c.params):
        values = [mpmath.mpf(point[n]) for n in c.params]
        for formula in formulas:
            for index in c.index_values if k is not None else [0]:
                root = formula(*([index] if k is not None else []), *values)
                residual = abs(poly(root, *values))
                scale = max(1.0, float(abs(root)) ** 3)
                count += 1
                if float(residual) > 1e-15 * scale:
                    return CheckReport(
                        entry.id, "root_formula", False, f"residual {residual} at {point}"
                    )
    return CheckReport(entry.id, "root_formula", True, f"{count} root evaluations give p(root) = 0")


def _check_tensor(entry: Entry, c: TensorCheck) -> CheckReport:
    data = TensorInput(input=c.input, tensors=c.tensors)
    _, canonical = canonicalize(data)
    ok = str(canonical) == c.expected
    return CheckReport(entry.id, "tensor", ok, f"canonical form: {canonical}")


def _check_equation(entry: Entry, c: EquationCheck) -> CheckReport:
    try:
        check_equation_dimensions(c.lhs, c.rhs, c.units)
    except (DimensionError, ParseError) as exc:
        return CheckReport(entry.id, "equation", False, str(exc))
    return CheckReport(entry.id, "equation", True, "dimensionally consistent")


def _check_constant(entry: Entry) -> CheckReport:
    assert entry.constant is not None  # noqa: S101
    try:
        Decimal(entry.constant.value)
        if entry.constant.uncertainty is not None:
            Decimal(entry.constant.uncertainty)
    except InvalidOperation:
        return CheckReport(entry.id, "constant", False, "value is not a decimal number")
    return CheckReport(entry.id, "constant", True, "well-formed (value is stated, not derived)")


def _check_distribution(entry: Entry, c: DistributionCheck) -> CheckReport:
    """Run the instance through the distribution operation and compare with the stated result."""
    from minerva_engine import DistributionInput, ExpressionInput, Limits
    from minerva_engine.distributions.operation import distribution

    assumptions = {**c.assumptions}
    assumptions[c.variable] = sorted({*assumptions.get(c.variable, []), "real"})
    result = distribution(
        DistributionInput(
            expression=ExpressionInput(input=c.input, format="text", assumptions=assumptions),
            action=c.action,
            variable=c.variable,
            lower=c.lower,
            upper=c.upper,
            endpoint=c.endpoint,
        ),
        Limits(timeout_seconds=60),
    )
    if result.status.value != "ok" or result.result_text is None:
        return CheckReport(
            entry.id, "distributional", False, f"{result.status.value}: {result.message}"
        )
    verification = result.verification
    if verification is None or not (verification.passed and verification.conditions_established):
        return CheckReport(entry.id, "distributional", False, "pairing or conditions failed")
    try:
        got = parse_text(result.result_text, assumptions)
        want = parse_text(c.expected, assumptions)
    except ParseError as exc:
        return CheckReport(entry.id, "distributional", False, f"unreadable result: {exc}")
    # Terms are compared as expressions: DiracDelta/Heaviside are inert symbols to simplify.
    if sp.simplify(sp.expand(got - want)) != 0:
        return CheckReport(
            entry.id, "distributional", False, f"result {result.result_text} != {c.expected}"
        )
    return CheckReport(
        entry.id,
        "distributional",
        True,
        f"{verification.pairings_agreed}/{verification.pairings_checked} test-function pairings "
        "agree; conditions established exactly",
    )


def run_check(entry: Entry) -> CheckReport:
    """Run the checker matching the entry's content; `stated` entries report as such."""
    if entry.verification == "stated":
        if entry.constant is not None:
            report = _check_constant(entry)
            report.level = "stated"
            return report
        return CheckReport(entry.id, "stated", True, "not machine-checked by design")
    if entry.rule is not None:
        return _check_rule(entry)
    c = entry.check
    if isinstance(c, DerivativeCheck):
        return _check_derivative(entry, c)
    if isinstance(c, AntiderivativeCheck):
        return _check_antiderivative(entry, c)
    if isinstance(c, IntegralCheck):
        return _check_integral(entry, c)
    if isinstance(c, SeriesCheck):
        return _check_series(entry, c)
    if isinstance(c, RootFormulaCheck):
        return _check_root_formula(entry, c)
    if isinstance(c, TensorCheck):
        return _check_tensor(entry, c)
    if isinstance(c, EquationCheck):
        return _check_equation(entry, c)
    if isinstance(c, DistributionCheck):
        return _check_distribution(entry, c)
    return CheckReport(entry.id, "none", False, "entry has nothing to verify")
