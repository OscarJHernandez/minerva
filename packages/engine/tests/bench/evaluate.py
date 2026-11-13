"""Compare what Minerva reported with the independent reference, and classify every miss.

Nothing here trusts the report. Each discovered constant is re-checked against the generated Hamiltonian
with the benchmark's own bracket (exact), and compared with the references by exact linear algebra on
the invariant *spaces* (the span of the constants, 1 and their pairwise products), never by expression
text. Functional relations are measured with the benchmark's own exact Jacobian rank: evidence for
dependence, since it cannot prove it.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

import sympy as sp
from minerva_engine.discovery import DiscoveredConstant, DiscoveryReport
from minerva_engine.latexparse import parse_latex
from minerva_engine.models import Limits, Status

from .cases import Case, functional_rank
from .descriptors import dominated
from .symplectic import bracket

TIER_CODE = re.compile(r"d(\d+)p(\d+)h\d+t\d+(?:c(\d+))?")
FAILURES = (
    "insufficient_ansatz",
    "missing_function_family",
    "solver_failure",
    "verification_failure",
    "parameter_degeneracy",
    "resource_exhaustion",
    "unclassified",
)


@dataclass
class RefOutcome:
    name: str
    recovered: str  # linear | functional | no
    first_tier: int | None
    dominated: bool
    failure: str | None = None


@dataclass
class Evaluation:
    case_id: str
    status: str
    constants: int = 0
    new_constants: int = 0
    false_positives: list[str] = field(default_factory=list)
    refs: list[RefOutcome] = field(default_factory=list)
    independent_found: int = 0
    reference_rank: int = 0
    rank_lower: int = 0
    rank_upper: int = 0
    rank_proved: bool = False
    trivial_reported: int = 0
    tiers: int = 0
    unknowns: int = 0
    max_unknowns: int = 0
    stop_reason: str = ""
    cause: str = ""
    first_discovery_ms: int | None = None
    runtime_ms: int = 0
    peak_mb: float = 0.0
    rejected: int = 0
    unexplored: int = 0

    @property
    def coverage(self) -> float:
        return sum(r.recovered != "no" for r in self.refs) / max(1, len(self.refs))

    @property
    def linear_coverage(self) -> float:
        return sum(r.recovered == "linear" for r in self.refs) / max(1, len(self.refs))


def searched_tiers(report: DiscoveryReport) -> list[tuple[int, int, int]]:
    out = []
    for t in report.tiers:
        m = TIER_CODE.fullmatch(t.code)
        if m and t.outcome == "searched":
            out.append((int(m[1]), int(m[2]), int(m[3] or 0)))
    return out


def parse_constants(
    report: DiscoveryReport,
) -> list[tuple[DiscoveredConstant, sp.Expr]]:
    out = []
    for c in report.constants:
        rhs = c.expression_latex.split("=", 1)[1]
        out.append((c, parse_latex(rhs, {}, Limits())))
    return out


# --- exact linear algebra on invariant spaces ------------------------------------------------------------
def _matrix(exprs: list[sp.Expr], variables: list[sp.Symbol]) -> sp.Matrix:
    den = sp.Integer(1)
    for e in exprs:
        den = sp.lcm(den, sp.fraction(sp.together(e))[1])
    polys = [sp.Poly(sp.cancel(sp.together(e * den)), *variables) for e in exprs]
    monomials = sorted({m for poly in polys for m in poly.monoms()})
    row = {m: i for i, m in enumerate(monomials)}
    out = sp.zeros(len(monomials), len(exprs))
    for j, poly in enumerate(polys):
        for m, c in poly.terms():
            out[row[m], j] = c
    return out


def algebra(exprs: list[sp.Expr]) -> list[sp.Expr]:
    """1, the expressions and their pairwise products."""
    out: list[sp.Expr] = [sp.Integer(1), *exprs]
    out += [a * b for a, b in itertools.combinations_with_replacement(exprs, 2)]
    return out


def in_span(basis: list[sp.Expr], target: sp.Expr, variables: list[sp.Symbol]) -> bool:
    try:
        m = _matrix([*basis, target], variables)
    except (sp.PolynomialError, TypeError):
        return False
    return int(m[:, :-1].rank()) == int(m.rank())


def bracket_is_zero(expr: sp.Expr, case: Case) -> bool:
    return sp.simplify(bracket(expr, case.H, case.q, case.p)) == 0


# --- evaluation -------------------------------------------------------------------------------------------
def evaluate(case: Case, report: DiscoveryReport) -> Evaluation:
    q, p = case.q, case.p
    variables = [*q, *p]
    ev = Evaluation(
        case_id=case.id,
        status=report.status.value,
        reference_rank=case.ref_rank,
        tiers=len([t for t in report.tiers if t.outcome == "searched"]),
        unknowns=sum(t.unknowns for t in report.tiers),
        max_unknowns=max((t.unknowns for t in report.tiers), default=0),
        stop_reason=report.stop_reason,
        cause=report.cause,
        runtime_ms=report.duration_ms,
        peak_mb=report.peak_memory_mb,
        rejected=sum(t.rejected for t in report.tiers),
        unexplored=len(report.unexplored),
    )
    for event in report.events:
        if event.name.endswith("candidate_found"):
            ev.first_discovery_ms = event.at_ms
            break
    constants = parse_constants(report) if report.status is Status.OK else []
    ev.constants = len(constants)
    ev.new_constants = sum(c.kind == "new" for c, _ in constants)

    # Independent conservation check against the generated Hamiltonian.
    for c, expr in constants:
        if not bracket_is_zero(expr, case):
            ev.false_positives.append(c.id)
    exprs = [e for _, e in constants]
    ev.independent_found = functional_rank(exprs, q, p) if exprs else 0
    if report.independence is not None:
        ev.rank_lower = report.independence.rank_lower
        ev.rank_upper = report.independence.rank_upper
        ev.rank_proved = report.independence.rank_proved

    # Trivial combinations: a new constant inside the algebra of the earlier ones.
    for k, (c, expr) in enumerate(constants):
        if c.kind == "new" and in_span(algebra(exprs[:k]), expr, variables):
            ev.trivial_reported += 1

    tiers = searched_tiers(report)
    tier_indices = sorted({c.tier for c, _ in constants if c.tier is not None})
    for ref in case.refs:
        in_algebra = bool(exprs) and in_span(algebra(exprs), ref.expr, variables)
        first = None
        if in_algebra:
            for t in [None, *tier_indices]:
                subset = [
                    e for c, e in constants if c.tier is None or (t is not None and c.tier <= t)
                ]
                if in_span(algebra(subset), ref.expr, variables):
                    first = -1 if t is None else t
                    break
        recovered = "linear" if in_algebra else "no"
        if recovered == "no" and exprs:
            rank_with = functional_rank([*exprs, ref.expr], q, p)
            if rank_with == ev.independent_found:
                recovered = "functional"
        dom = dominated(ref.descriptor.frontier, tiers)
        failure = None
        expected_broken = case.broken.get(ref.name, False)
        if recovered != "no" and expected_broken:
            failure = "false_claim"  # a symmetry-breaking perturbation did not break it, yet it was reported
        elif recovered == "no" and not expected_broken:
            failure = classify_miss(
                ref.descriptor.representable, dom, ref.descriptor.frontier, report
            )
        ev.refs.append(RefOutcome(ref.name, recovered, first, dom, failure))
    return ev


def classify_miss(
    representable: bool,
    is_dominated: bool,
    frontier: list[tuple[int, int, int]] | None,
    report: DiscoveryReport,
) -> str:
    if report.status is Status.UNSUPPORTED:
        return (
            "missing_function_family"
            if report.cause == "unsupported_function"
            else "solver_failure"
        )
    if report.status is not Status.OK:
        return "resource_exhaustion"
    if not representable:
        return "missing_function_family"
    if is_dominated:
        if any(t.rejected for t in report.tiers):
            return "verification_failure"
        if report.parameter_conditions:
            return "parameter_degeneracy"
        return "solver_failure"
    if report.stop_reason in ("unknown_cap", "time_budget"):
        return "resource_exhaustion"
    if frontier is None:
        return "insufficient_ansatz"
    return "insufficient_ansatz"
