"""Exact Legendre transform of a Hamiltonian that is quadratic in the momenta, verified by the round trip.

Used only to give the older, Lagrangian-only `search_invariants` the same systems as the adaptive search
(the baseline of the comparison). `H = 1/2 p^T A p + b(q).p + c(q)` with `A` constant and invertible has
`L = 1/2 (v - b)^T A^-1 (v - b) - c(q)`; the transform is accepted only if the Hamiltonian rebuilt from
`L` (`p = dL/dv`, `H = p.v - L`) equals the original exactly. Anything else returns None.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass
class LagrangianForm:
    L: sp.Expr
    velocities: list[sp.Symbol]
    momenta_of_velocity: list[sp.Expr]  # p_i as a function of (q, v)
    latex: str


def to_lagrangian(H: sp.Expr, q: list[sp.Symbol], p: list[sp.Symbol]) -> LagrangianForm | None:
    n = len(q)
    hess = sp.Matrix(n, n, lambda i, j: sp.diff(H, p[i], p[j]))
    if any(e.free_symbols for e in hess) or hess.det() == 0:
        return None
    zero = {pi: 0 for pi in p}
    b = sp.Matrix([sp.diff(H, pi).subs(zero) for pi in p])
    c = H.subs(zero)
    half = sp.Rational(1, 2)
    if (
        sp.expand(
            H - (half * (sp.Matrix(p).T * hess * sp.Matrix(p))[0] + (sp.Matrix(p).T * b)[0] + c)
        )
        != 0
    ):
        return None  # not quadratic in the momenta
    v = [sp.Symbol(f"{qi.name}_dot") for qi in q]
    vv = sp.Matrix(v)
    ainv = hess.inv()
    lag = sp.expand(half * ((vv - b).T * ainv * (vv - b))[0] - c)
    p_of_v = list(ainv * (vv - b))
    # Round trip: p = dL/dv, then H' = p.v - L, with v eliminated through p.
    p_check = [sp.diff(lag, vi) for vi in v]
    if any(sp.expand(a - bb) != 0 for a, bb in zip(p_check, p_of_v, strict=True)):
        return None
    v_of_p = list(hess * sp.Matrix(p) + b)
    rebuilt = sum(pi * vi for pi, vi in zip(p, v_of_p, strict=True)) - lag.subs(
        dict(zip(v, v_of_p, strict=True)), simultaneous=True
    )
    if sp.expand(rebuilt - H) != 0:
        return None
    names = {}
    for vi, qi in zip(v, q, strict=True):
        base, _, idx = qi.name.partition("_")
        names[vi] = rf"\dot{{{base}_{{{idx}}}}}"
    return LagrangianForm(lag, v, p_of_v, str(sp.latex(lag, symbol_names=names)))
