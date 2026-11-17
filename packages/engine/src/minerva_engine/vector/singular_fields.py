"""The distributional divergence and Laplacian of a point-source Coulomb kernel (action `distributional`).

The classical divergence of `F = C (r - a)/|r - a|^3` is 0 at every point where it is defined, and a numerical
sampler never sees the origin, so `evaluate` answers 0. As a distribution on R^3 it is `4 pi C delta^(3)(r - a)`.
This module establishes that for one family and says so: the Coulomb kernel with a constant strength `C` and a
constant source point `a` (a Laplacian of `C/|r - a|` through its gradient). Everything else is a typed refusal.

How a result is reached (none of it is the language model's, and none of it is a lookup):

1. The field is lowered exactly and written in Cartesian components (`cartesian_field`: shapes that need no
   inverse trigonometry, radial fields in spherical coordinates and fields of `rho`, `z` in cylindrical ones).
2. `match_kernel` reads the source `a` and the strength `C` **from the field itself** (the base of its half-integer
   power is `|r - a|^2`, the numerator is `C (r - a)`), and the match is an exact algebraic identity.
3. The shrinking-sphere argument is applied; its hypotheses are checked exactly for this input:
   H1 the divergence is an exact symbolic zero away from `a`; H2 `F . n = C/eps^2` on the sphere `|r - a| = eps`
   (so the outward flux is `4 pi C` for every `eps`); H3 `(F . F)|r - a|^4 = C^2`, so `|F|` is locally
   integrable and the excised ball contributes nothing as `eps -> 0`. With them,
   `<div F, phi> = -int F . grad(phi) = C oint phi(a + eps n) dOmega -> 4 pi C phi(a)`.
4. Independent numerical legs (`weak.py`) read the *input* field, not the kernel found in step 2: pairings with four
   test functions (values 1, 2, ~0.68 and 0 at the source, one centred away), the flux through two spheres, and the
   regularised field `r/(r^2 + eps^2)^(3/2)`. The pairings must agree to 1e-7 or nothing is shown; the regularisation
   table is reported and never decides.

`proof_status` is `derived` only when H1-H3 are exact and every pairing agreed. It is not called proved, and
numerical agreement is supporting evidence. A user's claim about the result (`target`) is compared exactly.
"""

from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import mpmath as mp
import numpy as np
import sympy as sp

from ..distributions.point_delta import PointDelta
from ..errors import MinervaMathError
from ..models import Status
from .algebra import exact_zero
from .context import SEED, Context, Refused, tidy
from .coordinates import System, get_system
from .expressions import Div, Grad, Lap, kind_of
from .lower import Lowering
from .models import (
    Regularisation,
    SingularClaim,
    SingularResult,
    SingularStep,
    VectorCheck,
    VectorDiagnostics,
    VectorResult,
    VectorVerification,
    WeakPairing,
)
from .numeric import Evaluator, Setup, Skip, make_setup
from .printing import vector_latex
from .weak import (
    EPS_LEVELS,
    PAIRING_TOLERANCE,
    Array,
    Point,
    ScalarField,
    VectorField,
    flux_through_sphere,
    pair_divergence,
    pair_laplacian,
    regularised_pairing,
    standard_test_functions,
)

SCOPE = (
    "Derived for one point source of the Coulomb kernel C (r - a)/|r - a|^3 (or the Laplacian of C/|r - a|) "
    "with a constant strength C and a constant real source a, against smooth rapidly decreasing test "
    "functions. Not covered: any other singular field, several sources, products, derivatives or "
    "compositions of the delta."
)
FLUX_RADII = (0.7, 1.9)
CROSS_CHECK_POINTS = 8
DOMAIN_NOTE = "valid at every point except the source"


@dataclass(frozen=True)
class Kernel:
    """`F = strength (r - source)/|r - source|^3`, read from a field."""

    strength: sp.Expr
    source: tuple[sp.Expr, sp.Expr, sp.Expr]
    base: sp.Expr  # |r - source|^2 as the polynomial found in the field


def _unsupported(message: str) -> Refused:
    return Refused(Status.UNSUPPORTED, message, "unsupported_singular_field")


# -- the field in Cartesian components ---------------------------------------------------------------
def cartesian_symbols() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    return get_system("cartesian").symbols()


def cartesian_vector(system: System, comps: list[sp.Expr]) -> list[sp.Expr] | None:
    """The field with physical components `comps` of `system` as Cartesian components, for the shapes that
    can hold a radial kernel; None for anything else (never a guess)."""
    x, y, z = cartesian_symbols()
    if system.name == "cartesian":
        return [sp.sympify(c) for c in comps]
    q = system.symbols()
    if system.name == "cylindrical":
        rho, phi, zc = q
        f_rho, f_phi, f_z = (sp.sympify(c) for c in comps)
        if sp.simplify(f_phi) != 0 or f_rho.has(phi) or f_z.has(phi):
            return None
        g = sp.cancel(sp.together(f_rho / rho))
        sub = {rho: sp.sqrt(x**2 + y**2), zc: z}
        return [
            sp.simplify(c) for c in (g.xreplace(sub) * x, g.xreplace(sub) * y, f_z.xreplace(sub))
        ]
    r, theta, phi = q
    f_r, f_theta, f_phi = (sp.sympify(c) for c in comps)
    if sp.simplify(f_theta) != 0 or sp.simplify(f_phi) != 0 or f_r.has(theta) or f_r.has(phi):
        return None
    radius = sp.sqrt(x**2 + y**2 + z**2)
    g = f_r.xreplace({r: radius}) / radius
    return [sp.simplify(c) for c in (g * x, g * y, g * z)]


def cartesian_scalar(system: System, f: sp.Expr) -> sp.Expr | None:
    x, y, z = cartesian_symbols()
    f = sp.sympify(f)
    if system.name == "cartesian":
        return f
    q = system.symbols()
    if system.name == "cylindrical":
        rho, phi, zc = q
        if f.has(phi):
            return None
        return f.xreplace({rho: sp.sqrt(x**2 + y**2), zc: z})
    r, theta, phi = q
    if f.has(theta) or f.has(phi):
        return None
    return f.xreplace({r: sp.sqrt(x**2 + y**2 + z**2)})


# -- recognising the kernel --------------------------------------------------------------------------
def match_kernel(
    field: list[sp.Expr], coords: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
) -> Kernel | None:
    """Reads `(C, a)` from `field` if it is exactly `C (r - a)/|r - a|^3`, else None.

    The only half-integer power in the field must have one base `B`; with `s = sqrt(B)` the three components
    times `s^3` must be polynomials `K (x_i - a_i)`. `B` must be `lam |r - a|^2` for a positive constant `lam`
    (a rescaled base, as in `(4 x^2 + 4 y^2 + 4 z^2)^(3/2)`): its quadratic part is `lam (x^2 + y^2 + z^2)`,
    its linear part `-2 lam a . r`, its constant `lam |a|^2`. Then `C = K/lam^(3/2)`. `C` and `a` must not
    depend on the coordinates. An identically zero field is the kernel with `C = 0`.
    """
    comps = [sp.sympify(c) for c in field]
    if all(sp.simplify(c) == 0 for c in comps):
        return Kernel(sp.Integer(0), (sp.Integer(0),) * 3, sum(c**2 for c in coords))
    bases: list[sp.Expr] = []
    for c in comps:
        for p in c.atoms(sp.Pow):
            if p.exp.is_Rational and p.exp.q == 2:
                b = sp.expand(p.base)
                if b not in bases:
                    bases.append(b)
    if len(bases) != 1:
        return None
    base = bases[0]
    s = sp.Dummy("s", positive=True)

    def swap(c: sp.Expr) -> sp.Expr:
        rep = {
            p: s ** (2 * p.exp)
            for p in c.atoms(sp.Pow)
            if p.exp.is_Rational and p.exp.q == 2 and sp.expand(p.base) == base
        }
        return c.xreplace(rep)

    try:
        poly = sp.Poly(base, *coords)
    except sp.PolynomialError:
        return None
    if poly.total_degree() != 2:
        return None
    lam = poly.coeff_monomial(coords[0] ** 2)
    if lam.has(*coords) or lam.is_positive is not True:
        return None
    for i in range(3):
        if sp.simplify(poly.coeff_monomial(coords[i] ** 2) - lam) != 0:
            return None
        for j in range(i + 1, 3):
            if sp.simplify(poly.coeff_monomial(coords[i] * coords[j])) != 0:
                return None
    source = tuple(-poly.coeff_monomial(c) / (2 * lam) for c in coords)
    scaled = lam * sum((c - a) ** 2 for c, a in zip(coords, source, strict=True))
    if sp.simplify(sp.expand(base - scaled)) != 0:
        return None
    numerators: list[sp.Expr] = []
    for c in comps:
        t = sp.cancel(sp.together(swap(c) * s**3))
        if t.has(s) or not t.is_polynomial(*coords):
            return None
        numerators.append(sp.expand(t))
    scale: sp.Expr | None = None
    for i, t in enumerate(numerators):
        coefficient = sp.Poly(t, *coords).coeff_monomial(coords[i]) if t != 0 else sp.Integer(0)
        if coefficient != 0:
            scale = coefficient
            break
    if scale is None:
        scale = sp.Integer(0)
    if scale.has(*coords):
        return None
    for i, t in enumerate(numerators):
        if sp.simplify(t - scale * (coords[i] - source[i])) != 0:
            return None
    strength = sp.simplify(scale / lam ** sp.Rational(3, 2))
    return Kernel(strength, source, base)


def point_source_kernel(low: Lowering, system: System, expr: sp.Expr) -> Kernel | None:
    """The kernel if `expr` (a divergence or Laplacian read by `evaluate`) is of the supported point-source
    form, else None. Structural only (no numerics): `evaluate` uses it to say that its classical value is
    not the whole story and where the source is."""
    try:
        split = _split(expr)
        if split is None:
            return None
        _, _, family, inner = split
        lowered = low.lower(inner)
        if family == "divergence":
            if not isinstance(lowered, tuple):
                return None
            comps = cartesian_vector(system, [tidy(c) for c in lowered])
        else:
            f = cartesian_scalar(
                system, lowered if not isinstance(lowered, tuple) else sp.Integer(0)
            )
            if f is None:
                return None
            comps = [sp.diff(f, c) for c in cartesian_symbols()]
        if comps is None:
            return None
        kernel = match_kernel(comps, cartesian_symbols())
        if kernel is None or kernel.strength == 0:
            return None
        return kernel
    except (MinervaMathError, Refused, sp.SympifyError, ValueError, TypeError, NotImplementedError):
        return None


def tuple_latex(items: tuple[sp.Expr, ...]) -> str:
    return r"\left(" + ", ".join(vector_latex(i) for i in items) + r"\right)"


def source_domain_latex(source: tuple[sp.Expr, sp.Expr, sp.Expr]) -> str:
    """`(x, y, z) != a`: where the classical result holds."""
    return tuple_latex(cartesian_symbols()) + r" \neq " + tuple_latex(source)


def _split(expr: sp.Expr) -> tuple[sp.Expr, sp.Expr, str, sp.Expr] | None:
    """`(prefactor, operator, family, inner)` of `k * div(F)`, `k * lap(f)` or `k * div(grad f)`."""
    prefactor: sp.Expr = sp.Integer(1)
    operator = expr
    if isinstance(expr, sp.Mul):
        heads = [a for a in expr.args if isinstance(a, Div | Lap)]
        if len(heads) != 1:
            return None
        operator = heads[0]
        prefactor = sp.Mul(*[a for a in expr.args if a is not operator])
    if isinstance(operator, Div):
        inner = operator.args[0]
        if isinstance(inner, Grad):
            return prefactor, operator, "laplacian", inner.args[0]
        return prefactor, operator, "divergence", inner
    if isinstance(operator, Lap) and kind_of(operator.args[0]) == "scalar":
        return prefactor, operator, "laplacian", operator.args[0]
    return None


# -- exact hypotheses --------------------------------------------------------------------------------
def _is_zero(e: sp.Expr) -> bool:
    e = sp.simplify(e)
    return e == 0 or exact_zero(e)


def exact_steps(
    field: list[sp.Expr], kernel: Kernel, coords: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
) -> list[SingularStep]:
    """The hypotheses of the shrinking-sphere argument, computed exactly on `field` (the input, not the kernel)."""
    c_tex = vector_latex(kernel.strength)
    divergence = sum((sp.diff(f, c) for f, c in zip(field, coords, strict=True)), sp.Integer(0))
    h1 = _is_zero(divergence)

    w = sp.symbols("w1 w2 w3", real=True)
    shifted = {c: a + wi for c, a, wi in zip(coords, kernel.source, w, strict=True)}
    radius = sp.sqrt(sum(wi**2 for wi in w))
    f_dot_w = sum((f.xreplace(shifted) * wi for f, wi in zip(field, w, strict=True)), sp.Integer(0))
    # F . w = C/|w|, so F . n = C/eps^2 on |w| = eps: the outward flux is C/eps^2 * 4 pi eps^2.
    h2 = _is_zero(f_dot_w * radius - kernel.strength)
    f_squared = sum((f.xreplace(shifted) ** 2 for f in field), sp.Integer(0))
    h3 = _is_zero(f_squared * sum(wi**2 for wi in w) ** 2 - kernel.strength**2)
    steps = [
        SingularStep(
            title="Classical divergence away from the source",
            latex=r"\nabla\cdot\mathbf F = 0 \quad (\mathbf r\neq\mathbf a)",
            status="established" if h1 else "not_established",
            note="exact symbolic zero of the sum of the partial derivatives of the input field"
            if h1
            else "the divergence of the input field is not identically zero",
        ),
        SingularStep(
            title="Flux density on a small sphere",
            latex=rf"\mathbf F\cdot\hat{{\mathbf n}}=\frac{{{c_tex}}}{{\varepsilon^{{2}}}}\ \text{{on }}|\mathbf r-\mathbf a|=\varepsilon",
            status="established" if h2 else "not_established",
            note="exact: F . (r - a) = C/|r - a|, computed from the input field",
        ),
        SingularStep(
            title="Local integrability",
            latex=r"(\mathbf F\cdot\mathbf F)\,|\mathbf r-\mathbf a|^{4}=C^{2}",
            status="established" if h3 else "not_established",
            note="exact; so |F| = |C|/|r - a|^2 is integrable near the source and the excised ball "
            "contributes at most 4 pi |C| sup|grad phi| eps, which tends to 0",
        ),
    ]
    return steps


# -- numerical legs ----------------------------------------------------------------------------------
def _numpy_vector(
    comps: list[sp.Expr],
    coords: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
    params: dict[sp.Symbol, sp.Expr],
) -> VectorField:
    fn = sp.lambdify(coords, [c.subs(params) for c in comps], modules="numpy")

    def call(x: Array, y: Array, z: Array) -> tuple[Array, Array, Array]:
        a, b, c = (np.broadcast_to(np.asarray(v, dtype=np.float64), x.shape) for v in fn(x, y, z))
        return a, b, c

    return call


def _numpy_scalar(
    expr: sp.Expr, coords: tuple[sp.Symbol, sp.Symbol, sp.Symbol], params: dict[sp.Symbol, sp.Expr]
) -> ScalarField:
    fn = sp.lambdify(coords, expr.subs(params), modules="numpy")

    def call(x: Array, y: Array, z: Array) -> Array:
        return np.broadcast_to(np.asarray(fn(x, y, z), dtype=np.float64), x.shape)

    return call


def _agrees(measured: float, expected: float) -> bool:
    return abs(measured - expected) <= PAIRING_TOLERANCE * max(1.0, abs(expected))


def _read_input(evaluator: Evaluator, concrete: sp.Expr, point: Any) -> Any:
    """The input expression at a Cartesian point, or None where the evaluator cannot read it (a pole, a
    branch cut): such a point is skipped, never counted as agreement."""
    try:
        return evaluator.value(concrete, point)
    except (Skip, ValueError, ZeroDivisionError, TypeError, OverflowError):
        return None


def _reading_cross_check(
    setup: Setup,
    ctx: Context,
    inner: sp.Expr,
    family: str,
    numpy_field: VectorField,
    numpy_scalar: ScalarField | None,
    source: Point,
    rng: random.Random,
) -> tuple[int, int]:
    """Compares the Cartesian reading of the field (used by the quadrature) with the independent reading of
    the *input expression* by the finite-difference evaluator at random points. (checked, agreed)."""
    mp.mp.dps = 40
    evaluator = Evaluator(setup, ctx.system, mp.mpf(0))
    concrete = inner.subs(setup.params)
    checked = agreed = 0
    for _ in range(CROSS_CHECK_POINTS):
        direction = np.array([rng.gauss(0, 1) for _ in range(3)])
        direction /= np.linalg.norm(direction)
        radius = rng.uniform(0.5, 2.0)
        p = [source[i] + radius * float(direction[i]) for i in range(3)]
        point = mp.matrix([mp.mpf(v) for v in p])
        value = _read_input(evaluator, concrete, point)
        if value is None:
            continue
        arrays = [np.array([v]) for v in p]
        if family == "divergence":
            ours = [float(c[0]) for c in numpy_field(*arrays)]
            theirs = [float(value[i]) for i in range(3)]
            ok = all(
                abs(o - t) <= 1e-9 * max(1.0, abs(t)) for o, t in zip(ours, theirs, strict=True)
            )
        elif numpy_scalar is not None:
            ours_s = float(numpy_scalar(*arrays)[0])
            ok = abs(ours_s - float(value)) <= 1e-9 * max(1.0, abs(float(value)))
        else:
            continue
        checked += 1
        agreed += int(ok)
    return checked, agreed


# -- reading the user's claim ------------------------------------------------------------------------
_DELTA_HEAD = re.compile(
    r"(?:\\delta\s*\^\s*(?:\{\s*\(\s*3\s*\)\s*\}|\{\s*\\left\(\s*3\s*\\right\)\s*\}|\{\s*3\s*\}|3|\(\s*3\s*\))"
    r"|\b(?:Delta3|delta3|PointDelta))\s*(?:\\left)?\s*\("
)
_R = r"(?:\\mathbf\s*\{\s*r\s*\}|\\mathbf\s+r|\\vec\s*\{\s*r\s*\}|\\vec\s+r|\\boldsymbol\s*\{\s*r\s*\}|\\bm\s*\{\s*r\s*\}|pos\(\))"


def _balanced(text: str, start: int) -> int | None:
    """The index just after the `)` that closes the `(` before `start` (`start` is after the `(`)."""
    depth = 1
    i = start
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _split_top(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    last = 0
    for i, ch in enumerate(text):
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[last:i])
            last = i + 1
    parts.append(text[last:])
    return parts


def _read_source(ctx: Context, arg: str) -> tuple[sp.Expr, sp.Expr, sp.Expr] | None:
    text = re.sub(r"\\left|\\right", "", arg).strip()
    if re.fullmatch(_R + r"|r", text) or text == "0":
        return (sp.Integer(0),) * 3
    m = re.fullmatch(rf"(?:{_R}|r)\s*-\s*(.+)", text)
    if m is None or ctx.system.name != "cartesian":
        return None
    rest = m.group(1).strip()
    named = re.fullmatch(r"\\mathbf\s*\{\s*(\w+)\s*\}|\\mathbf\s+(\w+)|(\w+)", rest)
    if named is not None:
        name = next(g for g in named.groups() if g)
        declared = ctx.low.declared.get(name)
        if declared is not None and declared.explicit is not None and len(declared.explicit) == 3:
            return tuple(sp.sympify(c) for c in declared.explicit)
        return None
    if rest.startswith("(") and rest.endswith(")"):
        items = _split_top(rest[1:-1])
        if len(items) == 3:
            try:
                return tuple(ctx.parse(i.strip()) for i in items)
            except (MinervaMathError, Refused, ValueError):
                return None
    return None


def read_claim(
    ctx: Context, target: str, strength: sp.Expr, source: tuple[sp.Expr, sp.Expr, sp.Expr]
) -> SingularClaim:
    """Compares the user's statement about the distribution with the derived `strength * delta^(3)(r - source)`."""
    text = target.strip()
    shown = text
    match = _DELTA_HEAD.search(text)
    if match is None:
        try:
            value = ctx.parse(text) if text else sp.Integer(0)
        except (MinervaMathError, Refused, ValueError):
            return SingularClaim(
                latex=shown,
                holds=False,
                reason="not_readable",
                note="the statement could not be read",
            )
        del value
        return SingularClaim(
            latex=shown,
            holds=False,
            reason="no_delta_term",
            note="a function without a delta term cannot equal this distribution: it vanishes at every point "
            "away from the source, but its pairing with a test function is nonzero",
        )
    end = _balanced(text, match.end())
    if end is None or _DELTA_HEAD.search(text, end) is not None:
        return SingularClaim(
            latex=shown, holds=False, reason="not_readable", note="one delta term is supported"
        )
    arg = text[match.end() : end - 1]
    claimed_source = _read_source(ctx, arg)
    coefficient_text = (text[: match.start()] + " " + text[end:]).strip()
    coefficient_text = re.sub(r"(\\,|\\;|\\cdot|\\!|\*)\s*$", "", coefficient_text).strip()
    coefficient_text = re.sub(r"^(\\,|\\;|\\cdot|\*)\s*", "", coefficient_text).strip()
    if claimed_source is None:
        return SingularClaim(
            latex=shown,
            holds=False,
            reason="not_readable",
            note="the position of the delta could not be read",
        )
    try:
        if coefficient_text in ("", "+"):
            claimed = sp.Integer(1)
        elif coefficient_text == "-":
            claimed = sp.Integer(-1)
        else:
            claimed = ctx.parse(coefficient_text)
    except (MinervaMathError, Refused, ValueError):
        return SingularClaim(
            latex=shown,
            holds=False,
            reason="not_readable",
            note="the coefficient could not be read",
        )
    if kind_of(claimed) != "scalar":
        return SingularClaim(
            latex=shown, holds=False, reason="not_readable", note="the coefficient is not a scalar"
        )
    if sp.simplify(claimed - strength) == 0:
        same_source = all(
            sp.simplify(c - s) == 0 for c, s in zip(claimed_source, source, strict=True)
        )
        if same_source or sp.simplify(strength) == 0:
            return SingularClaim(
                latex=shown, holds=True, reason="agrees", note="equal as distributions"
            )
        return SingularClaim(
            latex=shown,
            holds=False,
            reason="different_source",
            note="the delta is supported at a different point",
        )
    if (
        not all(sp.simplify(c - s) == 0 for c, s in zip(claimed_source, source, strict=True))
        and sp.simplify(claimed) != 0
    ):
        return SingularClaim(
            latex=shown,
            holds=False,
            reason="different_source",
            note="the delta is supported at a different point",
        )
    return SingularClaim(
        latex=shown,
        holds=False,
        reason="different_strength",
        note="the coefficient (including its sign) differs from the derived one",
    )


# -- the action --------------------------------------------------------------------------------------
def _sample_point(
    source: tuple[sp.Expr, sp.Expr, sp.Expr], params: dict[sp.Symbol, sp.Expr]
) -> Point:
    values = [sp.sympify(a).subs(params) for a in source]
    if any(not v.is_real for v in values):
        raise Refused(
            Status.UNSUPPORTED, "the source position is not a real point", "source_not_real"
        )
    return (float(values[0]), float(values[1]), float(values[2]))


def run_distributional(ctx: Context, result: VectorResult, diag: VectorDiagnostics) -> None:
    started = time.monotonic()
    expr = ctx.parse(ctx.inp.expression)
    result.input_latex = vector_latex(expr)
    split = _split(expr)
    if split is None:
        raise _unsupported(
            "the distributional action takes the divergence of a vector field or the Laplacian of a scalar "
            "field (optionally times a constant)"
        )
    prefactor, _operator, family, inner = split
    if (
        prefactor.has(*ctx.system.symbols())
        or prefactor.has(ctx.low.t)
        or kind_of(prefactor) != "scalar"
    ):
        raise _unsupported(
            "the constant in front of the operator must not depend on the coordinates"
        )
    if inner.has(ctx.low.t):
        raise _unsupported("a field that depends on time is not supported")
    for symbol in inner.free_symbols:
        declared = ctx.low.declared.get(symbol.name)
        if declared is not None and declared.explicit is None:
            raise _unsupported(
                "the field must be explicit (give its components): an abstract field has no source"
            )
    diag.result_kind = "scalar"

    # The classical result, by the ordinary route (lowering plus the independent finite-difference reading).
    classical = ctx.evaluate(expr)
    if classical.status != "ok" or not classical.zero:
        raise _unsupported(
            "the classical divergence of this field is not identically zero away from a point"
        )

    coords = cartesian_symbols()
    lowered = ctx.low.lower(inner)
    if family == "divergence":
        if not isinstance(lowered, tuple):
            raise _unsupported("a vector field is expected inside the divergence")
        field = cartesian_vector(ctx.system, [tidy(c) for c in lowered])
        scalar_f: sp.Expr | None = None
    else:
        scalar_f = cartesian_scalar(
            ctx.system, tidy(lowered) if not isinstance(lowered, tuple) else sp.Integer(0)
        )
        field = None if scalar_f is None else [sp.diff(scalar_f, c) for c in coords]
    if field is None:
        raise _unsupported("this field is not of the form C (r - a)/|r - a|^3 in a supported shape")
    kernel = match_kernel(field, coords)
    if kernel is None:
        raise _unsupported(
            "only the Coulomb kernel C (r - a)/|r - a|^3 with a constant strength and a constant source "
            "is supported; this field is not of that form"
        )
    if any(a.is_real is not True for a in kernel.source):
        raise Refused(
            Status.UNSUPPORTED,
            "the source position must be real: declare its constants real",
            "source_not_real",
        )
    if time.monotonic() > ctx.deadline:
        raise Refused(Status.RESOURCE_LIMIT, "the time budget was used up", "resource_limit")

    strength_total = sp.simplify(4 * sp.pi * prefactor * kernel.strength)
    source = kernel.source
    delta = PointDelta(*source)
    rhs = strength_total * delta
    steps = exact_steps(field, kernel, coords)

    rng = random.Random(SEED)  # noqa: S311 - deterministic sampling, not cryptographic
    setup = make_setup(ctx.low, rng)
    point = _sample_point(source, setup.params)
    strength_num = float((prefactor * kernel.strength).subs(setup.params))
    expected_scale = 4.0 * math.pi * strength_num
    numpy_field = _numpy_vector(field, coords, setup.params)
    numpy_f = _numpy_scalar(scalar_f, coords, setup.params) if scalar_f is not None else None

    # (a) the Cartesian reading used by the quadrature against the independent reading of the input expression
    seen, agreed = _reading_cross_check(setup, ctx, inner, family, numpy_field, numpy_f, point, rng)
    reading_ok = seen >= CROSS_CHECK_POINTS // 2 and agreed == seen

    weak: list[WeakPairing] = []
    for phi in standard_test_functions(point):
        expected = expected_scale * phi.value_at(point)
        if family == "divergence":
            measured = pair_divergence(numpy_field, phi, point)
        else:
            if numpy_f is None:
                raise _unsupported("a scalar field is expected inside the Laplacian")
            measured = pair_laplacian(numpy_f, phi, point)
        # `prefactor` multiplies the operator, so the input's pairing is prefactor times the kernel's.
        measured *= float(prefactor.subs(setup.params)) if prefactor != 1 else 1.0
        weak.append(
            WeakPairing(
                label="pairing with a test function",
                test_function_latex=phi.label_latex,
                expected=expected,
                measured=measured,
                error=abs(measured - expected),
                passed=_agrees(measured, expected),
            )
        )
    flux: list[WeakPairing] = []
    for radius in FLUX_RADII:
        measured_flux = flux_through_sphere(numpy_field, point, radius) * (
            float(prefactor.subs(setup.params)) if prefactor != 1 else 1.0
        )
        flux.append(
            WeakPairing(
                label=f"outward flux through the sphere of radius {radius}",
                expected=expected_scale,
                measured=measured_flux,
                error=abs(measured_flux - expected_scale),
                passed=_agrees(measured_flux, expected_scale),
            )
        )
    skew = standard_test_functions(point)[1]
    regularised: list[Regularisation] = []
    for eps in EPS_LEVELS:
        value, _ = regularised_pairing(strength_num, point, skew, eps)
        target_value = 4.0 * math.pi * strength_num * skew.value_at(point)
        regularised.append(
            Regularisation(
                eps=eps, value=value, expected=target_value, error=abs(value - target_value)
            )
        )

    exact_ok = all(s.status == "established" for s in steps)
    numeric_ok = reading_ok and all(w.passed for w in weak) and all(f.passed for f in flux)
    if not exact_ok or not numeric_ok:
        result.status = Status.NOT_VERIFIED
        result.message = (
            "a distributional result was derived but its checks did not all pass, so it is not shown "
            "(exact hypotheses: "
            + ("all established" if exact_ok else "not all established")
            + "; independent pairings: "
            + ("agreed" if numeric_ok else "did not agree")
            + ")"
        )
        diag.reason = "verification_failed"
        return

    c_tex = vector_latex(kernel.strength)
    steps += [
        SingularStep(
            title="Weak form",
            latex=r"\langle\nabla\cdot\mathbf F,\varphi\rangle=-\int_{\mathbb R^{3}}\mathbf F\cdot\nabla\varphi\,d^{3}r"
            if family == "divergence"
            else r"\langle\nabla^{2}f,\varphi\rangle=\int_{\mathbb R^{3}}f\,\nabla^{2}\varphi\,d^{3}r",
            status="established",
            note="the definition of the distributional derivative (a test function phi is smooth and decays fast)",
        ),
        SingularStep(
            title="Punctured domain and the divergence theorem",
            latex=r"-\int_{|\mathbf r-\mathbf a|>\varepsilon}\mathbf F\cdot\nabla\varphi\,d^{3}r"
            r"=\oint_{|\mathbf r-\mathbf a|=\varepsilon}\varphi\,\mathbf F\cdot\hat{\mathbf n}\,dS",
            status="established",
            note="the outward normal of the punctured domain points toward the source, so the inner boundary "
            "enters with the opposite sign and the volume term vanishes by the first step",
        ),
        SingularStep(
            title="Shrinking the sphere",
            latex=rf"\oint\varphi\,\mathbf F\cdot\hat{{\mathbf n}}\,dS={c_tex}\oint\varphi(\mathbf a+\varepsilon\hat{{\mathbf n}})\,d\Omega"
            rf"\ \xrightarrow{{\varepsilon\to0}}\ 4\pi\,{c_tex}\,\varphi(\mathbf a)",
            status="established",
            note="phi is continuous at the source; the excised ball contributes nothing (local integrability)",
        ),
    ]
    lhs_tex = vector_latex(expr)
    rhs_tex = vector_latex(rhs)
    equation = rf"{lhs_tex} = {rhs_tex}"
    claim = read_claim(ctx, ctx.inp.target, strength_total, source) if ctx.inp.target else None
    domain = source_domain_latex(source)
    result.singular = SingularResult(
        family="divergence" if family == "divergence" else "laplacian",
        classical_latex="0",
        classical_domain_latex=domain,
        distributional_latex=equation,
        strength_latex=vector_latex(strength_total),
        source_latex=tuple_latex(source),
        derivation=steps,
        flux_latex=rf"\oint_{{|\mathbf r-\mathbf a|=\varepsilon}}\mathbf F\cdot\hat{{\mathbf n}}\,dS={vector_latex(sp.simplify(4 * sp.pi * kernel.strength))}",
        weak_checks=weak,
        flux_checks=flux,
        regularisation=regularised,
        proof_status="derived",
        scope=SCOPE,
        unresolved=[
            "behaviour at infinity is assumed rapidly decreasing through the test functions",
        ],
        claim=claim,
    )
    checks = [
        VectorCheck(
            name="classical divergence is zero away from the source (finite differences)",
            passed=True,
        ),
        VectorCheck(name="hypotheses of the shrinking-sphere argument (exact)", passed=True),
        VectorCheck(
            name="Cartesian reading of the field against the input expression",
            passed=True,
            note=f"{agreed} points",
        ),
        VectorCheck(
            name="pairings with test functions (independent quadrature)",
            passed=True,
            note=f"{len(weak)} test functions, tolerance 1e-7",
        ),
        VectorCheck(
            name="flux through spheres (independent quadrature)",
            passed=True,
            note=f"{len(flux)} radii",
        ),
        VectorCheck(
            name="regularised field (reported, supporting evidence only)",
            passed=None,
            note="not part of the decision",
        ),
    ]
    result.verification = VectorVerification(
        method="distributional_weak_form",
        numeric_passed=True,
        points_checked=len(weak) + len(flux) + seen,
        points_agreed=len(weak) + len(flux) + agreed,
        max_residual=max([w.error for w in weak] + [f.error for f in flux]),
        symbolic_zero=None,
        checks=checks,
        note="derived by the shrinking-sphere argument; independent pairings are supporting evidence",
    )
    result.result_latex = rhs_tex
    result.result_text = str(rhs)
    result.equations_latex = [equation]
    result.restrictions_latex = [domain]
    result.restrictions_complete = True
    result.result_kind = "scalar"
    result.assumptions_used = [
        "the coordinates are real and the source position is a real point",
        "test functions are smooth and rapidly decreasing; constants are sampled as real numbers",
    ]
    result.status = Status.OK
    verdict = ""
    if claim is not None:
        verdict = " Your statement " + (
            "agrees with it." if claim.holds else f"does not hold ({claim.reason})."
        )
    result.message = (
        "equal as distributions, not pointwise: the classical value is 0 away from the source and the "
        "distributional one is a point mass. Derived by the shrinking-sphere argument for this kernel (each "
        f"hypothesis checked exactly) and confirmed by {len(weak)} independent pairings and {len(flux)} flux "
        "checks; the numerical pairings are supporting evidence, not a proof." + verdict
    )
    diag.points = len(weak) + len(flux) + seen
    diag.stages.append(
        __import__("minerva_engine.vector.models", fromlist=["VectorStage"]).VectorStage(
            name="singular", duration_ms=int((time.monotonic() - started) * 1000)
        )
    )
