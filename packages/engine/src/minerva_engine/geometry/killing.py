"""Killing vectors of an explicit metric, and the conserved charges of its geodesics.

The search only *proposes*. Its unknowns are the lowered components `xi_mu`, each an exact linear
combination of a stated family of functions of the coordinates (lowered, because the rotations are then
polynomial in sin and cos; raised they carry cot). The Killing equation

    d_mu xi_nu + d_nu xi_mu - 2 Gamma^l_{mu nu} xi_l = 0,

cleared of denominators and reduced modulo `s^2 + c^2 = 1` for every angle, is a linear system whose
coefficients are rational functions of the metric's parameters, solved by an exact nullspace over
`QQ(parameters)`. The parameters are therefore generic: at special values (M = 0) there may be more.

The family (`Family`), per coordinate:
  * a coordinate that enters the metric rationally: integer powers in a range read off the metric, with
    negative powers only where the coordinate is in a denominator;
  * a coordinate that enters only through its own sin and cos: trigonometric polynomials up to a degree;
  * a coordinate absent from the metric: either a polynomial of total degree at most 1 in the absent
    coordinates, or the first harmonics (cos, sin) of one of them. Translation along an absent coordinate
    commutes with the Killing operator, so each of these *modes* is an invariant subspace and is solved as
    its own linear system; the union is exactly the nullspace over the whole family.
A short ladder of nested families is searched, cheapest first, up to `MAX_UNKNOWNS` unknowns per linear
system and the deadline; what was not searched is listed. A vector is shown only after the independent
numerical leg in `verify_vectors` (the Lie derivative of the metric by central differences, no Christoffel
symbols) passes. Nothing here says a metric has no further symmetry: "none within the searched family",
and "complete" only when the count reaches the bound n(n+1)/2.

The charges (`geodesic_charges`) are `Q_k = xi_(k)^mu g_{mu nu} v^nu`, the quadratic combinations
`c^{ab} Q_a Q_b` invariant under every found vector (structure constants from exact Lie brackets, an exact
nullspace, products of central charges factored out) and `g(v, v)`; each is shown only if it passes
`Flow.verify` on the geodesic Lagrangian. Killing tensors are not searched.
"""

from __future__ import annotations

import functools
import itertools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix

from ..errors import LimitError
from ..flow import Flow
from ..lagrangian import DerivationError, check_equations, derive_expr, verify_constant
from ..models import Status
from ..vector.algebra import exact_zero, regularise
from .compute import Workspace, geodesic_lagrangian
from .setup import Geometry, GeometryError

MAX_UNKNOWNS = 300  # per linear system; a larger one is listed as not searched
# Precision of the evaluation of d/ds on the geodesic equations, never the tolerance: near a horizon (r close to 2M)
# the terms of the derivative of g(v, v) reach 1e20 and cancel exactly; at 30 digits one Schwarzschild point in 20
# was left with noise above the tolerance (measured), at 50 and 80 none.
CHARGE_DIGITS = 80
LEVELS = 3  # nested families: the first one, then two widenings
FIRST_HARMONIC_DEGREE = 2  # degree of the trigonometric polynomials in an angle at the first level

Role = Literal["power", "angle", "absent"]
Origin = Literal[
    "cyclic coordinate",
    "Killing vector",
    "quadratic combination of Killing charges",
    "normalisation",
]
# Why a metric is refused, as a closed vocabulary (the only part that may be logged).
RefusalReason = Literal["declared_function", "non_rational", "mixed_angle", "irrational_constant"]

REFUSALS: dict[str, str] = {
    "declared_function": (
        "The metric contains a declared function such as f(r); the Killing search needs explicit components "
        "(a family of candidate functions cannot be stated for an unspecified function), so nothing was searched."
    ),
    "non_rational": (
        "The metric contains a root, a logarithm, an exponential or another function that is not the sine or "
        "cosine of a coordinate; the Killing search covers metrics rational in the coordinates and in the sines "
        "and cosines of angle coordinates only, so nothing was searched."
    ),
    "mixed_angle": (
        "A coordinate enters the metric both through its sine or cosine and outside them; the Killing search "
        "has no stated family for such a coordinate, so nothing was searched."
    ),
    "irrational_constant": (
        "The metric contains the imaginary unit or a constant such as pi; the Killing search solves an exact "
        "linear system over the rational numbers and the metric's parameters, so nothing was searched."
    ),
}


class KillingRefusal(GeometryError):
    def __init__(self, reason: RefusalReason) -> None:
        super().__init__(Status.UNSUPPORTED, REFUSALS[reason])
        self.reason = reason


# -- trigonometric polynomial variables ----------------------------------------------------------
@dataclass
class Variables:
    """Polynomial variables for the functions of the coordinates that occur, and the circle relations."""

    geo: Geometry
    sin: dict[sp.Symbol, sp.Dummy] = field(default_factory=dict)
    cos: dict[sp.Symbol, sp.Dummy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for c in self.geo.coords:
            self.sin[c] = sp.Dummy(f"s_{c.name}", real=True)
            self.cos[c] = sp.Dummy(f"c_{c.name}", real=True)

    def to_poly_vars(self, e: sp.Expr) -> sp.Expr:
        """`sin(x)`, `cos(x)` of a coordinate become polynomial variables; anything else is refused."""
        e = sp.expand_trig(regularise(sp.sympify(e)))
        mapping: dict[sp.Basic, sp.Basic] = {}
        for node in e.atoms(sp.sin, sp.cos):
            arg = node.args[0]
            if arg not in self.sin:
                raise KillingRefusal("non_rational")
            mapping[node] = (self.sin if isinstance(node, sp.sin) else self.cos)[arg]
        return e.xreplace(mapping)

    def to_functions(self, e: sp.Expr) -> sp.Expr:
        back: dict[sp.Basic, sp.Basic] = {}
        for c in self.geo.coords:
            back[self.sin[c]] = sp.sin(c)
            back[self.cos[c]] = sp.cos(c)
        return e.xreplace(back)

    def circle_pairs(self, gens: Sequence[sp.Symbol]) -> list[tuple[int, int]]:
        index = {g: i for i, g in enumerate(gens)}
        return [
            (index[self.sin[c]], index[self.cos[c]])
            for c in self.geo.coords
            if self.sin[c] in index and self.cos[c] in index
        ]


def reduce_circle(
    terms: dict[tuple[int, ...], Any], pairs: list[tuple[int, int]]
) -> dict[tuple[int, ...], Any]:
    """Reduce a polynomial (exponent tuple -> coefficient) modulo `s^2 + c^2 - 1` for every pair, so that every
    sine has degree at most 1. Those relations are a Groebner basis, so the reduced form is unique: it is zero
    exactly when the function is zero."""
    out = dict(terms)
    for i_s, i_c in pairs:
        pending = [k for k in out if k[i_s] >= 2]
        while pending:
            for key in pending:
                coeff = out.pop(key)
                low = list(key)
                low[i_s] -= 2
                a = tuple(low)
                low[i_c] += 2
                b = tuple(low)
                out[a] = out.get(a, 0) + coeff
                out[b] = out.get(b, 0) - coeff
            out = {k: v for k, v in out.items() if v != 0}
            pending = [k for k in out if k[i_s] >= 2]
    return {k: v for k, v in out.items() if v != 0}


# -- the family ---------------------------------------------------------------------------------
@dataclass
class Roles:
    role: list[Role]
    low: list[int]  # lowest power of a power coordinate in the metric (negative: in a denominator)
    high: list[int]


def classify(geo: Geometry, variables: Variables) -> Roles:
    """How each coordinate enters the metric. Refuses (with a closed reason) a metric outside the search."""
    entries = [e for e in geo.g if e != 0]
    if any(e.atoms(AppliedUndef) for e in entries):
        raise KillingRefusal("declared_function")
    if any(any(not (a.is_Rational) for a in e.atoms(sp.Number)) for e in entries):
        raise KillingRefusal("irrational_constant")
    if any(e.has(sp.I) or e.atoms(sp.NumberSymbol) for e in entries):
        raise KillingRefusal("irrational_constant")
    polys = [variables.to_poly_vars(e) for e in entries]
    gens = [*geo.coords, *variables.sin.values(), *variables.cos.values(), *geo.params]
    for e in polys:
        if e.atoms(sp.Function) or not e.is_rational_function(*gens):
            raise KillingRefusal("non_rational")
    roles: list[Role] = []
    low: list[int] = []
    high: list[int] = []
    for c in geo.coords:
        s, co = variables.sin[c], variables.cos[c]
        in_trig = any(e.has(s) or e.has(co) for e in polys)
        bare = any(e.has(c) for e in polys)
        if in_trig and bare:
            raise KillingRefusal("mixed_angle")
        if in_trig:
            roles.append("angle")
            low.append(0)
            high.append(0)
            continue
        if not bare:
            roles.append("absent")
            low.append(0)
            high.append(0)
            continue
        lo, hi = 0, 0
        for e in polys:
            num, den = sp.fraction(sp.together(e))
            dn = sp.degree(num, c) if num.has(c) else 0
            dd = sp.degree(den, c) if den.has(c) else 0
            hi = max(hi, int(dn) - int(dd))
            lo = min(lo, -int(dd))
        roles.append("power")
        low.append(lo)
        high.append(hi)
    return Roles(roles, low, high)


@dataclass(frozen=True)
class Mode:
    """Which functions of the absent coordinates: `None` the polynomials of total degree <= 1, otherwise the
    first harmonics of that one coordinate."""

    harmonic: int | None


@dataclass
class Family:
    level: int
    powers: dict[int, tuple[int, int]]  # coordinate index -> (lowest, highest) power
    degree: int  # of the trigonometric polynomials in each angle coordinate
    mode: Mode
    unknowns: int = 0

    def describe(self, names: list[str], roles: Roles, with_mode: bool = True) -> str:
        parts = []
        for i, (lo, hi) in sorted(self.powers.items()):
            parts.append(f"{names[i]}^k for k = {lo}..{hi}")
        angles = [names[i] for i, r in enumerate(roles.role) if r == "angle"]
        if angles:
            parts.append(
                f"polynomials of degree at most {self.degree} in the sine and cosine of {', '.join(angles)}"
            )
        absent = [names[i] for i, r in enumerate(roles.role) if r == "absent"]
        if absent:
            if not with_mode:
                parts.append(
                    f"either 1, {', '.join(absent)} or the cosine and sine of one of {', '.join(absent)}"
                )
            elif self.mode.harmonic is None:
                parts.append(f"1, {', '.join(absent)}")
            else:
                x = names[self.mode.harmonic]
                parts.append(f"cos {x}, sin {x}")
        text = "lowered components in the span of products of: " + "; ".join(parts)
        return text + (f" ({self.unknowns} unknowns)" if with_mode else "")


def ladder(geo: Geometry, roles: Roles) -> list[Family]:
    """Every family of the ladder, cheapest first within each level; level k+1 contains level k."""
    absent = [i for i, r in enumerate(roles.role) if r == "absent"]
    modes = [Mode(None), *(Mode(i) for i in absent)]
    out: list[Family] = []
    for level in range(1, LEVELS + 1):
        widen = level - 1
        powers = {
            i: (roles.low[i] - widen if roles.low[i] < 0 else 0, roles.high[i] + widen)
            for i, r in enumerate(roles.role)
            if r == "power"
        }
        degree = FIRST_HARMONIC_DEGREE + widen
        for mode in modes:
            fam = Family(level, powers, degree, mode)
            fam.unknowns = geo.n * len(_monomials(geo, roles, fam))
            out.append(fam)
    return out


Monomial = dict[
    str, Any
]  # "power": {i: k}, "trig": {i: (a, e)} = cos^a sin^e, "absent": ("one"|"x"|"cos"|"sin", i)


def _monomials(geo: Geometry, roles: Roles, fam: Family) -> list[Monomial]:
    power_axes = [range(lo, hi + 1) for _, (lo, hi) in sorted(fam.powers.items())]
    power_idx = sorted(fam.powers)
    angle_idx = [i for i, r in enumerate(roles.role) if r == "angle"]
    trig_basis = [(a, e) for e in (0, 1) for a in range(fam.degree + 1 - e)]
    absent_idx = [i for i, r in enumerate(roles.role) if r == "absent"]
    if fam.mode.harmonic is None:
        absent_basis: list[tuple[str, int]] = [("one", -1), *(("x", i) for i in absent_idx)]
    else:
        absent_basis = [("cos", fam.mode.harmonic), ("sin", fam.mode.harmonic)]
    out: list[Monomial] = []
    for ks in itertools.product(*power_axes):
        for trig in itertools.product(trig_basis, repeat=len(angle_idx)):
            for ab in absent_basis:
                out.append(
                    {
                        "power": dict(zip(power_idx, ks, strict=True)),
                        "trig": dict(zip(angle_idx, trig, strict=True)),
                        "absent": ab,
                    }
                )
    return out


# -- the search ---------------------------------------------------------------------------------
@dataclass
class KillingSearch:
    lowered: list[list[sp.Expr]]  # one list of n lowered components per vector (exact)
    raised: list[list[sp.Expr]]
    searched: list[str]
    not_searched: list[str]
    complete: bool  # the count reached n(n+1)/2
    cyclic: list[int | None]  # the coordinate c when the vector is d/dc
    roles: Roles
    unknowns: int = 0  # the largest linear system solved
    systems: int = 0


def search(ws: Workspace) -> KillingSearch:
    geo = ws.geo
    variables = Variables(geo)
    roles = classify(geo, variables)
    names = [str(c) for c in geo.coords]
    bound = geo.n * (geo.n + 1) // 2
    gamma = ws.christoffel()
    system = _System(ws, variables, roles, gamma)

    found: dict[Mode, list[list[sp.Expr]]] = {}
    searched: list[str] = []
    skipped: list[tuple[Family, str]] = []
    largest = systems = 0
    families = ladder(geo, roles)
    complete = False
    for fam in families:
        if complete:
            break
        if fam.unknowns > MAX_UNKNOWNS:
            skipped.append((fam, f"above the cap of {MAX_UNKNOWNS} unknowns per linear system"))
            continue
        if time.monotonic() > ws.soft_deadline:
            skipped.append((fam, "the time budget ran out"))
            continue
        vectors = system.solve(fam)
        searched.append(fam.describe(names, roles))
        largest = max(largest, fam.unknowns)
        systems += 1
        found[fam.mode] = vectors  # nested: a higher level of the same mode replaces the lower one
        if sum(len(v) for v in found.values()) >= bound:
            complete = True
    if systems == 0:
        # never word an empty result as "none within the searched family" when nothing was searched
        raise LimitError(
            "The time budget ran out before any family of the Killing search was solved; nothing was searched."
        )
    not_searched = [] if complete else _group_skipped(skipped, names, roles)
    lowered = [v for mode in sorted(found, key=_mode_order) for v in found[mode]]
    ginv = ws.inverse()
    raised: list[list[sp.Expr]] = []
    cyclic: list[int | None] = []
    for low in lowered:
        up = [
            ws.tidy(sum((ginv[m, v] * low[v] for v in range(geo.n)), sp.S.Zero))
            for m in range(geo.n)
        ]
        scale = _constant_part(up, geo)
        if scale != 1:
            up = [ws.tidy(x / scale) for x in up]
            low = [x / scale for x in low]
        unit = _unit_axis(up, geo)
        if unit is not None:
            axis, scale = unit
            up = [sp.S.One if m == axis else sp.S.Zero for m in range(geo.n)]
            low = [ws.tidy(x / scale) for x in low]
        cyclic.append(unit[0] if unit is not None else None)
        raised.append(up)
        lowered[len(raised) - 1] = [ws.tidy(x) for x in low]
    return KillingSearch(
        lowered=lowered,
        raised=raised,
        searched=searched,
        not_searched=not_searched,
        complete=complete,
        cyclic=cyclic,
        roles=roles,
        unknowns=largest,
        systems=systems,
    )


def _group_skipped(skipped: list[tuple[Family, str]], names: list[str], roles: Roles) -> list[str]:
    """One line per level and reason: the family without its mode, and the size of each mode's system."""
    groups: dict[tuple[int, str], list[Family]] = {}
    for fam, reason in skipped:
        groups.setdefault((fam.level, reason), []).append(fam)
    out = []
    for (_, reason), fams in groups.items():
        sizes = ", ".join(str(f.unknowns) for f in fams)
        out.append(
            f"{fams[0].describe(names, roles, with_mode=False)} ({sizes} unknowns): {reason}"
        )
    return out


def _mode_order(m: Mode) -> int:
    return -1 if m.harmonic is None else m.harmonic


def _constant_part(up: list[sp.Expr], geo: Geometry) -> sp.Expr:
    """The factor of the first nonzero component that does not depend on the coordinates (a number or a
    parameter): dividing the whole vector by it is a constant rescaling, so the vector stays a Killing vector."""
    first = next((x for x in up if x != 0), None)
    if first is None:
        return sp.S.One
    constant, _ = sp.factor(first).as_independent(*geo.coords, as_Add=False)
    return constant if constant != 0 else sp.S.One


def _unit_axis(up: list[sp.Expr], geo: Geometry) -> tuple[int, sp.Expr] | None:
    """`(c, s)` when the vector is s * d/dc with s free of the coordinates."""
    nonzero = [i for i, x in enumerate(up) if x != 0]
    if len(nonzero) != 1:
        return None
    scale = up[nonzero[0]]
    if scale.free_symbols & set(geo.coords):
        return None
    return nonzero[0], scale


def _primitive(coefficients: list[sp.Expr]) -> list[sp.Expr]:
    """Rescale a coefficient vector (constants: rationals and the metric's parameters) so the basis reads simply:
    parameters' denominators cleared, the common factor removed, the first coefficient positive. Only a
    constant rescaling, so the vector stays a Killing vector."""
    exprs = [sp.together(x) for x in coefficients]
    dens = [sp.fraction(x)[1] for x in exprs if x != 0]
    common = functools.reduce(sp.lcm, dens, sp.S.One) if dens else sp.S.One
    scaled = [sp.expand(sp.cancel(x * common)) for x in exprs]
    nonzero = [x for x in scaled if x != 0]
    if not nonzero:
        return coefficients
    content = functools.reduce(sp.gcd, nonzero)
    if content != 0:
        scaled = [sp.expand(sp.cancel(x / content)) for x in scaled]
    first = next(x for x in scaled if x != 0)
    lead, _ = first.as_ordered_terms()[0].as_coeff_Mul()
    if lead < 0:
        scaled = [-x for x in scaled]
    return scaled


class _System:
    """The Killing equation for one family, as an exact linear system."""

    def __init__(self, ws: Workspace, variables: Variables, roles: Roles, gamma: Any) -> None:
        self.ws = ws
        self.geo = ws.geo
        self.v = variables
        self.roles = roles
        geo = self.geo
        n = geo.n
        absent = [i for i, r in enumerate(roles.role) if r == "absent"]
        self.coord_gens: list[sp.Symbol] = [
            *(geo.coords[i] for i, r in enumerate(roles.role) if r in ("power", "absent")),
            *(variables.sin[geo.coords[i]] for i, r in enumerate(roles.role) if r == "angle"),
            *(variables.cos[geo.coords[i]] for i, r in enumerate(roles.role) if r == "angle"),
            *(variables.sin[geo.coords[i]] for i in absent),
            *(variables.cos[geo.coords[i]] for i in absent),
        ]
        self.gens = [*self.coord_gens, *geo.params]
        self.ncoord = len(self.coord_gens)
        self.pairs = variables.circle_pairs(self.gens)
        self.index = {g: i for i, g in enumerate(self.gens)}
        # B_{mu nu} (common denominator of the Gamma^l_{mu nu}) and G^l_{mu nu} = B Gamma^l_{mu nu}
        self.B: dict[tuple[int, int], sp.Poly] = {}
        self.G: dict[tuple[int, int, int], sp.Poly] = {}
        for mu in range(n):
            for nu in range(mu, n):
                ws.check_deadline("killing_vectors")
                parts = {}
                for lam in range(n):
                    value = gamma.get((lam, mu, nu))
                    if value == 0:
                        continue
                    num, den = sp.fraction(sp.together(variables.to_poly_vars(value)))
                    parts[lam] = (self._poly(num), self._poly(den))
                common = functools.reduce(
                    lambda a, b: a.lcm(b), (d for _, d in parts.values()), self._poly(sp.S.One)
                )
                self.B[(mu, nu)] = common
                for lam, (num, den) in parts.items():
                    self.G[(lam, mu, nu)] = num * common.exquo(den)
        self.shift = {
            i: (1 - roles.low[i]) if roles.low[i] < 0 else 0
            for i, r in enumerate(roles.role)
            if r == "power"
        }

    def _poly(self, e: sp.Expr) -> sp.Poly:
        p = sp.Poly(e, *self.gens)
        if p.get_domain() not in (sp.ZZ, sp.QQ):
            raise KillingRefusal("irrational_constant")
        return p.set_domain(sp.QQ)

    def _zero_exp(self) -> list[int]:
        return [0] * len(self.gens)

    def _term(
        self, mono: Monomial, coeff: Any = 1, extra: dict[int, int] | None = None
    ) -> dict[tuple[int, ...], Any]:
        """R * monomial as {exponents: coefficient}, R = product of x^shift (clears negative powers)."""
        geo = self.geo
        exp = self._zero_exp()
        for i, k in mono["power"].items():
            exp[self.index[geo.coords[i]]] = k + self.shift[i] + (extra or {}).get(i, 0)
        for i, (a, e) in mono["trig"].items():
            exp[self.index[self.v.cos[geo.coords[i]]]] = a
            exp[self.index[self.v.sin[geo.coords[i]]]] = e
        kind, i = mono["absent"]
        if kind == "x":
            exp[self.index[geo.coords[i]]] = 1
        elif kind == "cos":
            exp[self.index[self.v.cos[geo.coords[i]]]] = 1
        elif kind == "sin":
            exp[self.index[self.v.sin[geo.coords[i]]]] = 1
        if any(x < 0 for x in exp):
            return {}
        return {tuple(exp): QQ(coeff) if not isinstance(coeff, type(QQ(1))) else coeff}

    def derivative(self, mono: Monomial, mu: int) -> dict[tuple[int, ...], Any]:
        """R * d_mu(monomial)."""
        role = self.roles.role[mu]
        out: dict[tuple[int, ...], Any] = {}

        def add(terms: dict[tuple[int, ...], Any]) -> None:
            for k, v in terms.items():
                out[k] = out.get(k, 0) + v

        if role == "power":
            k = mono["power"][mu]
            if k != 0:
                add(self._term(mono, k, {mu: -1}))
        elif role == "angle":
            a, e = mono["trig"][mu]
            if e:  # d sin = cos
                add(self._term({**mono, "trig": {**mono["trig"], mu: (a + 1, e - 1)}}, 1))
            if a:  # d cos^a = -a cos^(a-1) sin
                add(self._term({**mono, "trig": {**mono["trig"], mu: (a - 1, e + 1)}}, -a))
        else:
            kind, i = mono["absent"]
            if i == mu:
                if kind == "x":
                    add(self._term({**mono, "absent": ("one", -1)}, 1))
                elif kind == "cos":
                    add(self._term({**mono, "absent": ("sin", i)}, -1))
                elif kind == "sin":
                    add(self._term({**mono, "absent": ("cos", i)}, 1))
        return {k: v for k, v in out.items() if v != 0}

    def solve(self, fam: Family) -> list[list[sp.Expr]]:
        geo, n = self.geo, self.geo.n
        monos = _monomials(geo, self.roles, fam)
        columns = [(c, m) for c in range(n) for m in monos]
        rows: dict[tuple[Any, ...], dict[int, dict[tuple[int, ...], Any]]] = {}
        params = len(geo.params)
        dpoly = {}
        for j, (c, mono) in enumerate(columns):
            if j % 25 == 0:
                self.ws.check_deadline("killing_vectors")
            base = (
                sp.Poly.from_dict(self._term(mono), *self.gens, domain=sp.QQ)
                if self._term(mono)
                else None
            )
            for mu in range(n):
                for nu in range(mu, n):
                    total = sp.Poly(0, *self.gens, domain=sp.QQ)
                    B = self.B[(mu, nu)]
                    for axis, other in ((mu, nu), (nu, mu)):
                        if other != c:
                            continue
                        key = (j, axis)
                        if key not in dpoly:
                            d = self.derivative(mono, axis)
                            dpoly[key] = (
                                sp.Poly.from_dict(d, *self.gens, domain=sp.QQ) if d else None
                            )
                        if dpoly[key] is not None:
                            total += B * dpoly[key]
                    g = self.G.get((c, mu, nu))
                    if g is not None and base is not None:
                        total -= 2 * g * base
                    if total.is_zero:
                        continue
                    reduced = reduce_circle(dict(total.terms()), self.pairs)
                    for exp, coeff in reduced.items():
                        row_key = (mu, nu, exp[: self.ncoord])
                        entry = rows.setdefault(row_key, {}).setdefault(j, {})
                        pexp = exp[self.ncoord :]
                        entry[pexp] = entry.get(pexp, 0) + coeff
        field_ = sp.QQ.frac_field(*geo.params) if params else sp.QQ
        sparse: dict[int, dict[int, Any]] = {}
        for key in rows:
            out_row: dict[int, Any] = {}
            for j, poly in rows[key].items():
                if params:
                    value = field_.from_sympy(
                        sp.Poly.from_dict(poly, *geo.params, domain=sp.QQ).as_expr()
                    )
                else:
                    value = poly.get((), sp.QQ(0))
                if value:
                    out_row[j] = value
            if out_row:
                sparse[len(sparse)] = out_row
        self.ws.check_deadline("killing_vectors")
        if sparse:
            matrix = DomainMatrix(sparse, (len(sparse), len(columns)), field_).to_dense()
            null = matrix.nullspace()
        else:
            null = DomainMatrix.eye(len(columns), field_).to_dense()
        vectors: list[list[sp.Expr]] = []
        for row in null.to_Matrix().tolist():
            comps = [sp.S.Zero] * n
            for j, a in enumerate(_primitive([sp.sympify(x) for x in row])):
                if a == 0:
                    continue
                c, mono = columns[j]
                comps[c] += sp.sympify(a) * self._monomial_expr(mono)
            vectors.append([sp.expand(x) for x in comps])
        return vectors

    def _monomial_expr(self, mono: Monomial) -> sp.Expr:
        geo = self.geo
        e: sp.Expr = sp.S.One
        for i, k in mono["power"].items():
            e *= geo.coords[i] ** k
        for i, (a, ex) in mono["trig"].items():
            e *= sp.cos(geo.coords[i]) ** a * sp.sin(geo.coords[i]) ** ex
        kind, i = mono["absent"]
        if kind == "x":
            e *= geo.coords[i]
        elif kind == "cos":
            e *= sp.cos(geo.coords[i])
        elif kind == "sin":
            e *= sp.sin(geo.coords[i])
        return e


# -- exact linear relations between functions -----------------------------------------------------
def relations(
    groups: list[list[sp.Expr]],
    geo: Geometry,
    variables: Variables,
    extra: Sequence[sp.Symbol] = (),
    deadline: Callable[[], None] | None = None,
) -> list[list[sp.Expr]]:
    """Every exact relation `sum_j a_j col_j = 0` that holds in each group at once, with constant `a_j` (rationals
    and the metric's parameters, generic). The columns are rational in the coordinates, `extra` symbols (the
    velocities) and the sines and cosines of the coordinates; anything else raises `KillingRefusal`."""
    m = len(groups[0])
    gens: list[sp.Symbol] = [
        *geo.coords,
        *variables.sin.values(),
        *variables.cos.values(),
        *extra,
        *geo.params,
    ]
    ncoord = len(gens) - len(geo.params)
    pairs = variables.circle_pairs(gens)
    rows: dict[tuple[Any, ...], dict[int, dict[tuple[int, ...], Any]]] = {}
    for gi, cols in enumerate(groups):
        if deadline is not None:
            deadline()
        fractions = [sp.fraction(sp.together(variables.to_poly_vars(c))) for c in cols]
        polys = [(_qq_poly(num, gens), _qq_poly(den, gens)) for num, den in fractions]
        common = functools.reduce(
            lambda a, b: a.lcm(b), (d for _, d in polys), sp.Poly(1, *gens, domain=sp.QQ)
        )
        for j, (num, den) in enumerate(polys):
            if num.is_zero:
                continue
            total = num * common.exquo(den)
            for exp, coeff in reduce_circle(dict(total.terms()), pairs).items():
                entry = rows.setdefault((gi, exp[:ncoord]), {}).setdefault(j, {})
                pexp = exp[ncoord:]
                entry[pexp] = entry.get(pexp, 0) + coeff
    field_ = sp.QQ.frac_field(*geo.params) if geo.params else sp.QQ
    sparse: dict[int, dict[int, Any]] = {}
    for key in rows:
        out_row = {}
        for j, poly in rows[key].items():
            value = (
                field_.from_sympy(sp.Poly.from_dict(poly, *geo.params, domain=sp.QQ).as_expr())
                if geo.params
                else poly.get((), sp.QQ(0))
            )
            if value:
                out_row[j] = value
        if out_row:
            sparse[len(sparse)] = out_row
    if not sparse:
        null = DomainMatrix.eye(m, field_).to_dense()
    else:
        null = DomainMatrix(sparse, (len(sparse), m), field_).to_dense().nullspace()
    return [[sp.sympify(x) for x in row] for row in null.to_Matrix().tolist()]


def _qq_poly(e: sp.Expr, gens: Sequence[sp.Symbol]) -> sp.Poly:
    p = sp.Poly(e, *gens)
    if p.get_domain() not in (sp.ZZ, sp.QQ):
        raise KillingRefusal("irrational_constant")
    return p.set_domain(sp.QQ)


# -- verification of the vectors -----------------------------------------------------------------
LIE_METHOD = (
    "Lie derivative of the metric, xi^l d_l g_(mu nu) + g_(l nu) d_mu xi^l + g_(mu l) d_nu xi^l, with the metric "
    "and the candidate differentiated by central differences (mpmath, 50 digits) at random points; no "
    "Christoffel symbols and nothing from the search"
)


def lie_derivative_leg(verifier: Any, raised: list[sp.Expr], name: str) -> Any:
    """The deciding numerical leg for one candidate (raised components): every component of the Lie derivative of
    the metric is zero, measured against the size of the terms that cancel."""
    geo = verifier.geo
    n = geo.n
    items = [(m, raised[m]) for m in range(n)]
    key = "killing:" + "|".join(str(x) for x in raised)

    def fn(ev: Any, point: Any, h: Any) -> list[tuple[Any, ...]]:
        x, p = point
        g = ev.metric(x, p)
        dg = ev.partial_metric(x, p, h)
        xi = ev.candidate(key, items, point)
        dxi = []
        for a in range(n):
            plus = ev.candidate(key, items, (ev._shift(x, a, h), p))
            minus = ev.candidate(key, items, (ev._shift(x, a, -h), p))
            dxi.append([(plus[m] - minus[m]) / (2 * h) for m in range(n)])
        out: list[tuple[Any, ...]] = []
        for mu in range(n):
            for nu in range(mu, n):
                total = 0
                size = 0
                for lam in range(n):
                    for term in (
                        xi[lam] * dg[lam][mu][nu],
                        g[lam][nu] * dxi[mu][lam],
                        g[mu][lam] * dxi[nu][lam],
                    ):
                        total += term
                        size += abs(term)
                out.append((total, 0, size))
        return out

    return verifier.numeric(name, fn)


def lie_derivative_symbolic(geo: Geometry, raised: list[sp.Expr]) -> list[sp.Expr]:
    """The Lie derivative of the metric along the candidate, by SymPy differentiation (a separate, symbolic leg)."""
    n, g, x = geo.n, geo.g, geo.coords
    out = []
    for mu in range(n):
        for nu in range(mu, n):
            total = sp.S.Zero
            for lam in range(n):
                total += raised[lam] * sp.diff(g[mu, nu], x[lam])
                total += g[lam, nu] * sp.diff(raised[lam], x[mu])
                total += g[mu, lam] * sp.diff(raised[lam], x[nu])
            if total != 0:
                out.append(total)
    return out


def verify_vectors(verifier: Any, vectors: list[list[sp.Expr]]) -> list[Any]:
    """One numerical (deciding) and one symbolic leg per candidate, in order."""
    from .models import Check

    checks: list[Any] = []
    for k, up in enumerate(vectors, start=1):
        checks.append(
            lie_derivative_leg(verifier, up, f"xi_({k}): the Lie derivative of the metric vanishes")
        )
        checks.append(
            verifier.symbolic(
                f"xi_({k}): the Lie derivative of the metric simplifies to 0",
                lambda up=up: lie_derivative_symbolic(verifier.geo, up),
                zero=verifier._zero_trig,
            )
        )
    if not vectors:
        checks.append(
            Check(
                name="no candidate",
                kind="symbolic",
                passed=None,
                note="not run: the search found no Killing vector within the family",
            )
        )
    return checks


# -- charges -------------------------------------------------------------------------------------
@dataclass
class Charge:
    label: str
    expr: sp.Expr
    origin: Origin
    passed: bool = False
    symbolic_zero: bool = False
    points_checked: int = 0
    points_agreed: int = 0


@dataclass
class Charges:
    items: list[Charge]
    rule: str  # how the quadratic combinations were formed, or why none were
    notes: list[str] = field(default_factory=list)


QUADRATIC_RULE = (
    "quadratic combinations c^(ab) Q_a Q_b with c invariant under every found vector (structure constants from "
    "the exact Lie brackets, an exact nullspace), products of central charges factored out"
)


def geodesic_flow(geo: Geometry) -> Flow:
    """The geodesic equations as a flow (the Euler-Lagrange equations of `g(v, v)/2`, confirmed by numerical
    differentiation of that Lagrangian before anything is checked on them)."""
    parameter = sp.Symbol("affine_parameter", real=True)
    names = [c.name for c in geo.coords]
    try:
        d = derive_expr(
            geodesic_lagrangian(geo), names, "s", q=geo.coords, qd=geo.velocities, t=parameter
        )
    except DerivationError as exc:
        raise GeometryError(Status.UNSUPPORTED, exc.message) from exc
    if not d.solution:
        raise GeometryError(
            Status.UNSUPPORTED, "The geodesic accelerations could not be solved for."
        )
    if not all(c.passed for c in check_equations(d)):
        raise GeometryError(
            Status.NOT_VERIFIED,
            "The geodesic equations could not be confirmed by numerical differentiation of g(v, v)/2, so no "
            "charge was checked.",
        )
    return Flow(
        system="lagrangian",
        function_latex=d.show(d.lagrangian),
        coordinates=names,
        q=d.q,
        v=d.qd,
        t=d.t,
        params=d.params,
        q_rate=list(d.qd),
        v_rate=[d.solution[a] for a in d.qdd],
        time_dependent=False,
        names=d.names,
        _verify=lambda quantity: verify_constant(d, quantity, digits=CHARGE_DIGITS),
    )


def candidate_charges(ws: Workspace, found: KillingSearch) -> Charges:
    """Linear charges, quadratic invariant combinations and the normalisation, before any check."""
    geo, n = ws.geo, ws.geo.n
    v = geo.velocities
    variables = Variables(geo)
    linear = [ws.tidy(sum((low[m] * v[m] for m in range(n)), sp.S.Zero)) for low in found.lowered]
    items: list[Charge] = []
    for k, (q, axis) in enumerate(zip(linear, found.cyclic, strict=True), start=1):
        origin: Origin = "cyclic coordinate" if axis is not None else "Killing vector"
        items.append(Charge(f"Q_{{{k}}}", q, origin))
    # term by term, so the reader sees g_(mu nu) v^mu v^nu rather than one large fraction
    norm = sp.S.Zero
    for i in range(n):
        for j in range(i, n):
            if geo.g[i, j] != 0:
                norm += ws.tidy(geo.g[i, j] * (1 if i == j else 2)) * v[i] * v[j]
    notes: list[str] = []
    rule = QUADRATIC_RULE
    quadratics: list[sp.Expr] = []
    if len(found.raised) >= 2:
        try:
            quadratics, notes = _quadratic(ws, found, linear, norm, variables)
        except KillingRefusal:
            rule = "no quadratic combination formed: the brackets could not be written in the searched form"
    else:
        rule = "no quadratic combination formed: fewer than two Killing vectors"
    for e in quadratics:
        items.append(
            Charge(f"Q_{{{len(items) + 1}}}", e, "quadratic combination of Killing charges")
        )
    items.append(Charge(f"Q_{{{len(items) + 1}}}", norm, "normalisation"))
    return Charges(items, rule, notes)


def _bracket(geo: Geometry, a: list[sp.Expr], b: list[sp.Expr]) -> list[sp.Expr]:
    x = geo.coords
    return [
        sum(
            (a[k] * sp.diff(b[m], x[k]) - b[k] * sp.diff(a[m], x[k]) for k in range(geo.n)),
            sp.S.Zero,
        )
        for m in range(geo.n)
    ]


def structure_constants(
    ws: Workspace, raised: list[list[sp.Expr]], variables: Variables
) -> dict[tuple[int, int], list[sp.Expr]] | None:
    """`[xi_a, xi_b] = f_ab^c xi_c` exactly, or None when a bracket is not in the span of the found vectors."""
    geo, m = ws.geo, len(raised)
    out: dict[tuple[int, int], list[sp.Expr]] = {}
    for a, b in itertools.combinations(range(m), 2):
        ws.check_deadline("geodesic_constants")
        br = _bracket(geo, raised[a], raised[b])
        if all(sp.expand(x) == 0 for x in br):
            out[(a, b)] = [sp.S.Zero] * m
            continue
        groups = [[up[mu] for up in raised] + [br[mu]] for mu in range(geo.n)]
        rel = relations(
            groups, geo, variables, deadline=lambda: ws.check_deadline("geodesic_constants")
        )
        usable = [r for r in rel if r[-1] != 0]
        if len(rel) != 1 or not usable:
            return None
        r = usable[0]
        out[(a, b)] = [sp.cancel(-r[c] / r[-1]) for c in range(m)]
    return out


def _quadratic(
    ws: Workspace,
    found: KillingSearch,
    linear: list[sp.Expr],
    norm: sp.Expr,
    variables: Variables,
) -> tuple[list[sp.Expr], list[str]]:
    geo, m = ws.geo, len(found.raised)
    f = structure_constants(ws, found.raised, variables)
    if f is None:
        return [], [
            "the Lie brackets of the found vectors do not close among them, so no quadratic combination was formed"
        ]

    def fk(k: int, a: int) -> list[sp.Expr]:  # f_{ka}^d over d
        if k == a:
            return [sp.S.Zero] * m
        return f[(k, a)] if k < a else [-x for x in f[(a, k)]]

    pairs = [(a, b) for a in range(m) for b in range(a, m)]
    col = {p: i for i, p in enumerate(pairs)}

    def cidx(a: int, b: int) -> int:
        return col[(min(a, b), max(a, b))]

    field_ = sp.QQ.frac_field(*geo.params) if geo.params else sp.QQ
    rows: list[dict[int, Any]] = []
    # (F_k C + C F_k^T)_{de} = sum_a f_ka^d C_ae + sum_a C_da f_ka^e = 0
    for k in range(m):
        for d in range(m):
            for e in range(d, m):
                row: dict[int, Any] = {}
                for a in range(m):
                    for coeff, idx in ((fk(k, a)[d], cidx(a, e)), (fk(k, a)[e], cidx(d, a))):
                        if coeff != 0:
                            row[idx] = row.get(idx, field_.zero) + field_.from_sympy(coeff)
                row = {i: v for i, v in row.items() if v}
                if row:
                    rows.append(row)
    nunk = len(pairs)
    if rows:
        inv = DomainMatrix(dict(enumerate(rows)), (len(rows), nunk), field_).to_dense().nullspace()
    else:
        inv = DomainMatrix.eye(nunk, field_).to_dense()
    invariant = inv.to_Matrix().tolist()
    # the centre: z with F_k z = 0 for every k
    crow: list[dict[int, Any]] = []
    for k in range(m):
        for d in range(m):
            row = {a: field_.from_sympy(fk(k, a)[d]) for a in range(m) if fk(k, a)[d] != 0}
            if row:
                crow.append(row)
    if crow:
        centre = DomainMatrix(dict(enumerate(crow)), (len(crow), m), field_).to_dense().nullspace()
        zs = centre.to_Matrix().tolist()
    else:
        zs = [[1 if i == j else 0 for i in range(m)] for j in range(m)]
    products: list[list[Any]] = []
    for i, j in itertools.combinations_with_replacement(range(len(zs)), 2):
        vec = [sp.S.Zero] * nunk
        for a in range(m):
            for b in range(m):
                vec[cidx(a, b)] += sp.sympify(zs[i][a]) * sp.sympify(zs[j][b])
        products.append(vec)
    notes: list[str] = []
    kept: list[list[sp.Expr]] = []
    basis = list(products)
    for w in invariant:
        ws.check_deadline("geodesic_constants")
        trial = [*basis, [sp.sympify(x) for x in w]]
        if _rank(trial, field_) > _rank(basis, field_):
            basis.append([sp.sympify(x) for x in w])
            kept.append(_primitive([sp.sympify(x) for x in w]))
    out: list[sp.Expr] = []
    v = geo.velocities
    for c in kept:
        e = sp.S.Zero
        for (a, b), i in col.items():
            if c[i] != 0:
                e += c[i] * linear[a] * linear[b]
        e = sp.expand(e)
        if exact_zero(e):
            notes.append("a quadratic invariant combination vanishes identically and is not listed")
            continue
        rel = relations([[e, norm]], geo, variables, extra=v)
        if any(r[0] != 0 and r[1] != 0 for r in rel):
            notes.append(
                "a quadratic invariant combination is a constant multiple of the normalisation g(v, v) and is "
                "not listed again"
            )
            continue
        out.append(ws.tidy(e))
    return out, notes


def _rank(vectors: list[list[sp.Expr]], field_: Any) -> int:
    if not vectors:
        return 0
    rows = {
        i: {j: field_.from_sympy(x) for j, x in enumerate(vec) if x != 0}
        for i, vec in enumerate(vectors)
    }
    rows = {i: r for i, r in rows.items() if r}
    if not rows:
        return 0
    width = len(vectors[0])
    return int(
        DomainMatrix(dict(enumerate(rows.values())), (len(rows), width), field_).to_dense().rank()
    )


def check_charge(flow: Flow, charge: Charge) -> Charge:
    """`Flow.verify` decides: symbolic zero and random numerical points of d/ds on the geodesic equations."""
    ok, ver = flow.verify(charge.expr)
    charge.passed = bool(ok)
    if ver is not None:
        charge.symbolic_zero = bool(ver.symbolic_zero)
        charge.points_checked = ver.numeric_points_checked
        charge.points_agreed = ver.numeric_points_agreed
    return charge


# -- the two quantities, as the geometry entry point uses them --------------------------------------
def found(ws: Workspace) -> KillingSearch:
    result: KillingSearch = ws.stage("killing_vectors", lambda: search(ws))
    return result


def vector_checks(verifier: Any) -> list[Any]:
    ws = verifier.ws
    checks: list[Any] = ws.stage(
        "killing_vectors:legs", lambda: verify_vectors(verifier, found(ws).raised)
    )
    return checks


def vector_verification(verifier: Any) -> Any:
    from .models import GeometryVerification
    from .verify import _combine

    checks = vector_checks(verifier)
    if not found(verifier.ws).raised:
        return GeometryVerification(
            method=LIE_METHOD,
            numeric_passed=True,
            symbolic_zero=None,
            checks=checks,
            note="no candidate was found within the family, so nothing is shown and nothing needed checking",
        )
    return _combine(LIE_METHOD, checks)


def charges(ws: Workspace) -> Charges:
    result: Charges = ws.stage("geodesic_constants", lambda: candidate_charges(ws, found(ws)))
    return result


def charge_verification(verifier: Any) -> Any:
    """The vectors' legs, then `Flow.verify` for each charge. A charge that fails is withheld (listed in the notes);
    the quantity is shown only if the vectors passed and at least one charge did."""
    from .models import Check, GeometryVerification

    ws = verifier.ws
    vectors = vector_verification(verifier)
    if not vectors.numeric_passed:
        return vectors
    items = charges(ws)
    flow = ws.stage("geodesic_flow", lambda: geodesic_flow(ws.geo))
    checks: list[Any] = list(vectors.checks)
    for charge in items.items:
        ws.check_deadline("geodesic_constants:verify")
        check_charge(flow, charge)
        if not charge.passed:
            continue
        checks.append(
            Check(
                name=f"{charge.label}: d/ds along the geodesic equations is zero",
                kind="numeric",
                passed=True,
                points_checked=charge.points_checked,
                points_agreed=charge.points_agreed,
                note="agreed at every sampled point (Flow.verify on the Euler-Lagrange equations of g(v, v)/2)",
            )
        )
        checks.append(
            Check(
                name=f"{charge.label}: d/ds along the geodesic equations simplifies to 0",
                kind="symbolic",
                passed=bool(charge.symbolic_zero),
                note="difference simplifies to 0"
                if charge.symbolic_zero
                else "no symbolic proof found (this alone does not mean the result is wrong)",
            )
        )
    shown = [c for c in items.items if c.passed]
    symbolic = [c for c in checks if c.kind == "symbolic" and c.passed is not None]
    return GeometryVerification(
        method="Flow.verify on the geodesic equations (Euler-Lagrange equations of g(v, v)/2, confirmed by "
        "numerical differentiation): d/ds of each charge is zero symbolically and at random points (80 digits)",
        numeric_passed=bool(shown),
        symbolic_zero=(all(c.passed is True for c in symbolic) if symbolic else None),
        checks=checks,
    )


def describe_search(result: KillingSearch, geo: Geometry) -> list[str]:
    """The wording every result carries: the family, what was not searched, and what the count means."""
    n = geo.n
    bound = n * (n + 1) // 2
    count = len(result.raised)
    lines = [f"searched: {s}" for s in result.searched]
    if result.complete:
        lines.append(
            f"{count} Killing vectors, which is n(n+1)/2 = {bound}, the most a {n}-dimensional metric can have on a "
            "connected region: the list is complete."
        )
    else:
        if count == 0:
            lines.append(
                "No Killing vector within the searched family. This does not say the metric has no symmetry "
                f"(at most {bound} are possible in {n} dimensions)."
            )
        else:
            lines.append(
                f"{count} Killing vector{'s' if count != 1 else ''} within the searched family; no further Killing "
                f"vector within it. This does not say the metric has no other symmetry (at most {bound} are "
                f"possible in {n} dimensions)."
            )
        lines.extend(f"not searched: {s}" for s in result.not_searched)
    if geo.params and not result.complete:
        names = ", ".join(str(p) for p in geo.params)
        lines.append(
            f"The parameters ({names}) are treated as generic: at special values there may be more Killing vectors."
        )
    lines.append("Killing tensors (hidden quadratic symmetries) are not searched.")
    return lines
