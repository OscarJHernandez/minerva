"""Integrable Hamiltonians with analytically known invariants, with seeded random exact parameters.

Two sources are kept apart in every report so a score on the curated physics-flavoured library is never
pooled with a score on systems built by design:

* `library`: textbook integrable systems (free particle, oscillators incl. resonant ones, a separable
  quartic, a superintegrable system with inverse-square terms, a central quartic, Cartesian Kepler).
* `constructed`: separable sums with random polynomial potentials, and functions `F(I_1, .., I_k)` of
  commuting invariants, which have no physical name and a higher momentum degree.

Every system's invariants are verified at construction: `{I, H} = 0` exactly. The invariants are
*reference data*; they are never given to Minerva.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

import sympy as sp

from .symplectic import bracket, phase_symbols

Q = sp.Rational


@dataclass
class Base:
    name: str
    source: str  # library | constructed
    n: int
    H: sp.Expr
    invariants: list[tuple[str, sp.Expr]]
    params: dict[str, str] = field(default_factory=dict)
    note: str = ""


def _pick(rng: random.Random, values: list[sp.Rational | int]) -> sp.Rational:
    return sp.Rational(rng.choice(values))


def free2(rng: random.Random) -> Base:
    (q1, q2), (p1, p2) = phase_symbols(2)
    mass = _pick(rng, [1, 2, Q(3, 2)])
    H = (p1**2 + p2**2) / (2 * mass)
    inv = [("p_1", p1), ("p_2", p2), ("angular", q1 * p2 - q2 * p1)]
    return Base("free2", "library", 2, H, inv, {"m": str(mass)}, "free particle in the plane")


def osc2_iso(rng: random.Random) -> Base:
    (q1, q2), (p1, p2) = phase_symbols(2)
    k = _pick(rng, [1, 2, 3, Q(1, 2), Q(3, 2)])
    H = (p1**2 + p2**2) / 2 + k * (q1**2 + q2**2) / 2
    inv = [
        ("h_1", p1**2 / 2 + k * q1**2 / 2),
        ("h_2", p2**2 / 2 + k * q2**2 / 2),
        ("angular", q1 * p2 - q2 * p1),
        ("cross", p1 * p2 + k * q1 * q2),
    ]
    return Base("osc2_iso", "library", 2, H, inv, {"k": str(k)}, "isotropic oscillator")


def osc2_res(rng: random.Random) -> Base:
    """Oscillator with frequencies 1 : b. Re and Im of a1^b conj(a2) are conserved (degree b+1)."""
    (q1, q2), (p1, p2) = phase_symbols(2)
    b = rng.choice([2, 3])
    H = (p1**2 + q1**2) / 2 + (p2**2 + b**2 * q2**2) / 2
    rq1, rq2, rp1, rp2 = (sp.Symbol(n, real=True) for n in ("q_1", "q_2", "p_1", "p_2"))
    z = sp.expand((rq1 + sp.I * rp1) ** b * (b * rq2 - sp.I * rp2))
    plain = {rq1: q1, rq2: q2, rp1: p1, rp2: p2}
    inv = [
        ("h_1", (p1**2 + q1**2) / 2),
        ("h_2", (p2**2 + b**2 * q2**2) / 2),
        ("re", sp.expand(sp.re(z).subs(plain))),
        ("im", sp.expand(sp.im(z).subs(plain))),
    ]
    return Base("osc2_res", "library", 2, H, inv, {"b": str(b)}, f"resonant oscillator 1:{b}")


def osc2_aniso(rng: random.Random) -> Base:
    (q1, q2), (p1, p2) = phase_symbols(2)
    k1, k2 = rng.choice([(1, 5), (2, 7), (3, 5), (1, 3)])  # non-resonant frequency-squared pair
    H = (p1**2 + k1 * q1**2) / 2 + (p2**2 + k2 * q2**2) / 2
    inv = [("h_1", (p1**2 + k1 * q1**2) / 2), ("h_2", (p2**2 + k2 * q2**2) / 2)]
    return Base("osc2_aniso", "library", 2, H, inv, {"k": f"{k1},{k2}"}, "non-resonant oscillator")


def quartic_sep(rng: random.Random) -> Base:
    (q1, q2), (p1, p2) = phase_symbols(2)
    a = _pick(rng, [1, 2, Q(1, 2), 3])
    k = _pick(rng, [1, 2, 3])
    h1 = p1**2 / 2 + a * q1**4
    h2 = p2**2 / 2 + k * q2**2 / 2
    return Base(
        "quartic_sep", "library", 2, h1 + h2, [("h_1", h1), ("h_2", h2)], {"a": str(a)}, "quartic"
    )


def central_quartic(rng: random.Random) -> Base:
    (q1, q2), (p1, p2) = phase_symbols(2)
    a = _pick(rng, [1, 2, Q(1, 2)])
    H = (p1**2 + p2**2) / 2 + a * (q1**2 + q2**2) ** 2
    return Base(
        "central_quartic",
        "library",
        2,
        sp.expand(H),
        [("angular", q1 * p2 - q2 * p1)],
        {"a": str(a)},
    )


def sw(rng: random.Random) -> Base:
    """Oscillator with inverse-square terms: superintegrable, with a quadratic integral K."""
    (q1, q2), (p1, p2) = phase_symbols(2)
    k = _pick(rng, [1, 2, 3])
    al = _pick(rng, [1, 2, Q(1, 2)])
    be = _pick(rng, [1, 3, Q(1, 3)])
    h1 = p1**2 / 2 + k * q1**2 / 2 + al / q1**2
    h2 = p2**2 / 2 + k * q2**2 / 2 + be / q2**2
    ang = q1 * p2 - q2 * p1
    kk = ang**2 + 2 * al * q2**2 / q1**2 + 2 * be * q1**2 / q2**2
    return Base(
        "sw",
        "library",
        2,
        h1 + h2,
        [("h_1", h1), ("h_2", h2), ("K", kk)],
        {"k": str(k), "alpha": str(al), "beta": str(be)},
        "oscillator with inverse-square terms",
    )


def kepler_cartesian(rng: random.Random) -> Base:
    (q1, q2), (p1, p2) = phase_symbols(2)
    k = _pick(rng, [1, 2])
    r = sp.sqrt(q1**2 + q2**2)
    ang = q1 * p2 - q2 * p1
    H = (p1**2 + p2**2) / 2 - k / r
    inv = [("angular", ang), ("lrl_x", p2 * ang - k * q1 / r), ("lrl_y", -p1 * ang - k * q2 / r)]
    return Base("kepler_cartesian", "library", 2, H, inv, {"k": str(k)}, "algebraic potential")


def cyclic_sep(rng: random.Random) -> Base:
    (_q1, q2), (p1, p2) = phase_symbols(2)
    k = _pick(rng, [1, 2, 3])
    h2 = p2**2 / 2 + k * q2**2 / 2
    return Base(
        "cyclic_sep", "library", 2, p1**2 / 2 + h2, [("p_1", p1), ("h_2", h2)], {"k": str(k)}
    )


def osc3_iso(rng: random.Random) -> Base:
    q, p = phase_symbols(3)
    k = _pick(rng, [1, 2])
    H = sum(pi**2 / 2 + k * qi**2 / 2 for qi, pi in zip(q, p, strict=True))
    inv = [(f"h_{i + 1}", p[i] ** 2 / 2 + k * q[i] ** 2 / 2) for i in range(3)]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        inv.append((f"angular_{i + 1}{j + 1}", q[i] * p[j] - q[j] * p[i]))
    return Base(
        "osc3_iso", "library", 3, sp.expand(H), inv, {"k": str(k)}, "3-d isotropic oscillator"
    )


# --- constructed ----------------------------------------------------------------------------------------
def sep_sum(rng: random.Random) -> Base:
    """Sum of one-degree-of-freedom Hamiltonians with random even polynomial potentials."""
    (q1, q2), (p1, p2) = phase_symbols(2)
    parts = []
    params = []
    for q, p in ((q1, p1), (q2, p2)):
        c2 = _pick(rng, [Q(1, 2), 1, 2, 3])
        c4 = _pick(rng, [0, 0, 1, Q(1, 3), 2])
        params.append(f"{c2},{c4}")
        parts.append(p**2 / 2 + c2 * q**2 + c4 * q**4)
    h1, h2 = parts
    return Base(
        "sep_sum", "constructed", 2, h1 + h2, [("h_1", h1), ("h_2", h2)], {"c": ";".join(params)}
    )


def func_inv(rng: random.Random) -> Base:
    """H = h_1 + c h_2^2: degree 4 in the momenta, with the same two commuting invariants."""
    (q1, q2), (p1, p2) = phase_symbols(2)
    k1 = _pick(rng, [1, 2, 3])
    k2 = _pick(rng, [1, 2, Q(1, 2)])
    c = _pick(rng, [Q(1, 2), 1, 2])
    h1 = p1**2 / 2 + k1 * q1**2 / 2
    h2 = p2**2 / 2 + k2 * q2**2 / 2
    return Base(
        "func_inv",
        "constructed",
        2,
        sp.expand(h1 + c * h2**2),
        [("h_1", h1), ("h_2", h2)],
        {"k": f"{k1},{k2}", "c": str(c)},
        "function of commuting invariants",
    )


def func_inv_rot(rng: random.Random) -> Base:
    """H = F(h, L) with the isotropic oscillator's energy and angular momentum: rank 2 of the family,
    plus the Fradkin-type invariants of the underlying oscillator, which commute with h and L only up
    to a relation, so only h and L are listed."""
    (q1, q2), (p1, p2) = phase_symbols(2)
    ang = q1 * p2 - q2 * p1
    h = (p1**2 + p2**2) / 2 + (q1**2 + q2**2) / 2
    c = _pick(rng, [Q(1, 2), 1, 2])
    return Base(
        "func_inv_rot",
        "constructed",
        2,
        sp.expand(h + c * ang**2),
        [("h", h), ("angular", ang)],
        {"c": str(c)},
        "oscillator plus a squared angular momentum",
    )


BUILDERS: dict[str, Callable[[random.Random], Base]] = {
    f.__name__: f
    for f in (
        free2,
        osc2_iso,
        osc2_res,
        osc2_aniso,
        quartic_sep,
        central_quartic,
        sw,
        kepler_cartesian,
        cyclic_sep,
        osc3_iso,
        sep_sum,
        func_inv,
        func_inv_rot,
    )
}


def build_base(name: str, rng: random.Random) -> Base:
    base = BUILDERS[name](rng)
    q, p = phase_symbols(base.n)
    for label, inv in base.invariants:
        if sp.simplify(bracket(inv, base.H, q, p)) != 0:
            raise AssertionError(f"reference invariant {label} of {name} is not conserved")
    return base
