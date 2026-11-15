"""Ordinary differential equations: the corpus of `minerva_engine.ode.benchmark`, adapted to this package.

The ODE corpus is written against the simpler `reference_core.Problem` (id, category, source, expected, judge). Here
each problem gets what this package records besides: where its expected answer comes from (`oracle`) and the
strongest evidence the engine reports for an accepted result (`evidence`). All of them are grouped under one
domain, `ode`; the id keeps the sub-area (analytic, numeric, stiff, boundary, approximation, ...).

Oracles are read from the problem's own source line. Evidence is deliberately conservative: a closed form the engine
checks by substituting it back is `structural`, everything else an accepted ODE result rests on is
`numeric_agreement`, and never a symbolic proof, which this module has no basis to claim.
"""

from __future__ import annotations

from ..ode.benchmark import ODE_PROBLEMS
from .core import Problem

_ANALYTIC = "ode-analytic"


def _oracle(source: str) -> str:
    text = source.lower()
    if "mpmath.odefun" in text:
        return "mpmath_odefun_taylor"
    if "mpmath.expm" in text:
        return "mpmath_expm"
    if "mpmath.taylor" in text:
        return "mpmath_taylor"
    if "mpmath." in text:
        return "closed_form"
    return "closed_form"


def _adapt(p: object) -> Problem:
    from ..reference_core import Problem as Simple

    assert isinstance(p, Simple)  # noqa: S101
    refused = p.expected == "refused"
    return Problem(
        p.id,
        "ode",
        p.source,
        p.expected,
        p.judge,
        oracle=_oracle(p.source),
        evidence="refusal"
        if refused
        else ("structural" if p.category == _ANALYTIC else "numeric_agreement"),
    )


PROBLEMS: list[Problem] = [_adapt(p) for p in ODE_PROBLEMS]
