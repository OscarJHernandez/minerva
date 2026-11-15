"""Gradients, Jacobians and Hessians of functions of matrices, over the reals.

The derivative is produced by differentials (not by SymPy's matrix-expression `diff`, which was measured
to return 0 for `log det X` and for the Gaussian log-density):

    d(AB) = dA B + A dB      d(X^-1) = -X^-1 dX X^-1      d(X^T) = (dX)^T
    d tr(M) = tr(dM)         d log|det M| = tr(M^-1 dM)   d det M = det M tr(M^-1 dM)

Every term is brought to the form `c * tr(P dV)` (or `tr(P dV^T)`) by cyclicity of the trace and the
gradient is read off with the shape of the variable (denominator layout: G has the shape of V and
`df = tr(G^T dV)`). A Jacobian of a vector function of a vector uses `df = J dv` (rows = outputs).

The result is shown only if an independent **numerical** derivative of the original function (central
differences at random real matrices of the declared sizes) agrees with it; for a symmetric variable the
candidate must also be symmetric, because `tr(G^T H) = tr(sym(G)^T H)` for symmetric H and a
directional check alone cannot tell the unconstrained gradient from the one within the symmetric
matrices. The symbolic leg is not run (`symbolic_zero` is None); nothing here says "proved".
"""

from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass

import mpmath as mp
import sympy as sp

from ..errors import LimitError, MinervaMathError, UnsupportedInput
from ..latexparse import parse_latex
from ..models import Limits, Status
from . import mexpr as mx
from .models import (
    MatrixAlternative,
    MatrixCalculusInput,
    MatrixCalculusResult,
    MatrixDecl,
    MatrixDerivative,
    MatrixStep,
    MatrixVerification,
)

DPS = 40
SAMPLES = 4  # usable random points required
ATTEMPTS = 24
STEP = mp.mpf(10) ** -10
STEP2 = mp.mpf(10) ** -6
TOL = mp.mpf(10) ** -8
TOL2 = mp.mpf(10) ** -5
MIN_DET = mp.mpf(10) ** -3


class Singular(Exception):
    """The function is not defined (a singular matrix) at a sampled point: the point is skipped."""


# --------------------------------------------------------------------------- differentials


@dataclass
class Direct:
    """c * left * dV^(T if transposed) * right  (left/right None: the identity)."""

    c: mx.SNode
    left: mx.MNode | None
    transposed: bool
    right: mx.MNode | None


@dataclass
class STerm:
    """c * tr(p dV)  or  c * tr(p dV^T)."""

    c: mx.SNode
    p: mx.MNode
    transposed: bool


@dataclass
class Via:
    """(sum of STerm) * n : the differential of a scalar function times a constant-shaped matrix."""

    terms: list[STerm]
    n: mx.MNode


Term = Direct | Via


def _mul(*parts: mx.MNode | None) -> mx.MNode | None:
    present = [p for p in parts if p is not None]
    return mx.m_prod(*present) if present else None


def _scaled(terms: list[STerm], c: mx.SNode) -> list[STerm]:
    return [STerm(mx.s_mul(c, t.c), t.p, t.transposed) for t in terms]


class Differ:
    def __init__(self, ctx: mx.Context, var: str) -> None:
        self.ctx = ctx
        self.var = var
        self.rows = ctx.shapes[var][0]

    # -- matrix valued
    def dm(self, m: mx.MNode) -> list[Term]:
        match m:
            case mx.Var(name):
                return [Direct(mx.ONE, None, False, None)] if name == self.var else []
            case mx.Ident() | mx.DiagPart():
                return []
            case mx.T(a):
                out: list[Term] = []
                for t in self.dm(a):
                    if isinstance(t, Direct):
                        left = mx.T(t.right) if t.right is not None else None
                        right = mx.T(t.left) if t.left is not None else None
                        out.append(Direct(t.c, left, not t.transposed, right))
                    else:
                        out.append(Via(t.terms, mx.T(t.n)))
                return out
            case mx.Inv(a):
                inv = mx.Inv(a)
                out = []
                for t in self.dm(a):
                    if isinstance(t, Direct):
                        out.append(
                            Direct(
                                mx.s_mul(mx.MINUS_ONE, t.c),
                                _mul(inv, t.left),
                                t.transposed,
                                _mul(t.right, inv),
                            )
                        )
                    else:
                        out.append(Via(t.terms, mx.m_scale(mx.MINUS_ONE, mx.m_prod(inv, t.n, inv))))
                return out
            case mx.Prod(fs):
                out = []
                for i, f in enumerate(fs):
                    prefix = mx.m_prod(*fs[:i]) if i else None
                    suffix = mx.m_prod(*fs[i + 1 :]) if i + 1 < len(fs) else None
                    for t in self.dm(f):
                        if isinstance(t, Direct):
                            out.append(
                                Direct(
                                    t.c, _mul(prefix, t.left), t.transposed, _mul(t.right, suffix)
                                )
                            )
                        else:
                            out.append(Via(t.terms, _mul(prefix, t.n, suffix) or t.n))
                return out
            case mx.Add(ts):
                return [t for term in ts for t in self.dm(term)]
            case mx.Scale(c, a):
                out = []
                for t in self.dm(a):
                    if isinstance(t, Direct):
                        out.append(Direct(mx.s_mul(c, t.c), t.left, t.transposed, t.right))
                    else:
                        out.append(Via(_scaled(t.terms, c), t.n))
                dc = self.ds(c)
                if dc:
                    out.append(Via(dc, a))
                return out
        raise mx.MatrixExprError("unsupported matrix expression")

    # -- scalar valued
    def ds(self, s: mx.SNode) -> list[STerm]:
        match s:
            case mx.Num() | mx.Sym():
                return []
            case mx.Tr(a) | mx.AsScalar(a):
                return self._through(a, None, mx.ONE)
            case mx.LogDet(a):
                return self._through(a, mx.Inv(a), mx.ONE)
            case mx.Det(a):
                return self._through(a, mx.Inv(a), mx.Det(a))
            case mx.SAdd(ts):
                return [t for term in ts for t in self.ds(term)]
            case mx.SMul(fs):
                out: list[STerm] = []
                for i, f in enumerate(fs):
                    others = [g for j, g in enumerate(fs) if j != i]
                    out += _scaled(self.ds(f), mx.s_mul(*others) if others else mx.ONE)
                return out
            case mx.SPow(b, k):
                factor = mx.s_mul(mx.Num(sp.Integer(k)), mx.s_pow(b, k - 1))
                return _scaled(self.ds(b), factor)
        raise mx.MatrixExprError("unsupported scalar expression")

    def _through(self, a: mx.MNode, middle: mx.MNode | None, factor: mx.SNode) -> list[STerm]:
        """tr(a), log|det a| (middle a^-1) and det a (middle a^-1, factor det a): the differential of
        a is a sum of terms L dV R, and tr(M L dV R) = tr(R M L dV) by cyclicity."""
        out: list[STerm] = []
        for t in self.dm(a):
            if isinstance(t, Direct):
                p = _mul(t.right, middle, t.left)
                out.append(
                    STerm(
                        mx.s_mul(t.c, factor),
                        p if p is not None else mx.Ident(self.rows),
                        t.transposed,
                    )
                )
            else:  # tr(M n) times the differential of a scalar
                weight = mx.Tr(mx.m_prod(middle, t.n)) if middle is not None else mx.Tr(t.n)
                out += _scaled(t.terms, mx.s_mul(weight, factor))
        return out


def read_gradient(terms: list[STerm], var: str) -> mx.MNode:
    pieces = [mx.m_scale(t.c, t.p if t.transposed else mx.T(t.p)) for t in terms]
    if not pieces:
        return mx.Scale(mx.ZERO, mx.Var(var))
    return mx.m_add(*pieces)


def read_jacobian(terms: list[Term], ctx: mx.Context, var: str, p: int) -> mx.MNode:
    n = ctx.shapes[var][0]
    pieces: list[mx.MNode] = []
    for t in terms:
        if isinstance(t, Direct):
            if not t.transposed:
                coeff = t.c
                if t.right is not None:
                    if mx.shape_of(t.right, ctx) != (1, 1):
                        raise mx.MatrixExprError(
                            "not a vector-valued function of the vector", "non_scalar_gradient"
                        )
                    coeff = mx.s_mul(coeff, mx.AsScalar(t.right))
                pieces.append(mx.m_scale(coeff, t.left if t.left is not None else mx.Ident(n)))
            else:
                if t.right is None:
                    raise mx.MatrixExprError(
                        "not a vector-valued function of the vector", "non_scalar_gradient"
                    )
                right = mx.T(t.right)
                pieces.append(
                    mx.m_scale(t.c, mx.m_prod(t.left, right) if t.left is not None else right)
                )
        else:
            for st in t.terms:
                row = st.p if not st.transposed else mx.T(st.p)
                pieces.append(mx.m_scale(st.c, mx.m_prod(t.n, row)))
    if not pieces:
        raise mx.MatrixExprError("the derivative is identically zero", "identically_zero")
    return mx.m_add(*pieces)


# --------------------------------------------------------------------------- numeric evaluation


Env = tuple[dict[str, mp.matrix], dict[str, mp.mpf]]


def ev_m(m: mx.MNode, env: Env) -> mp.matrix:
    matrices, _scalars = env
    match m:
        case mx.Var(n):
            return matrices[n]
        case mx.T(a):
            return ev_m(a, env).T
        case mx.Inv(a):
            value = ev_m(a, env)
            if abs(mp.det(value)) < MIN_DET * mp.mpf(10) ** -4:
                raise Singular
            return value**-1
        case mx.Prod(fs):
            out = ev_m(fs[0], env)
            for f in fs[1:]:
                out = out * ev_m(f, env)
            return out
        case mx.Add(ts):
            out = ev_m(ts[0], env)
            for t in ts[1:]:
                out = out + ev_m(t, env)
            return out
        case mx.Scale(c, a):
            return ev_s(c, env) * ev_m(a, env)
        case mx.Ident(n):
            return mp.eye(n)
        case mx.DiagPart(a):
            value = ev_m(a, env)
            return mp.diag([value[i, i] for i in range(value.rows)])
    raise mx.MatrixExprError("cannot evaluate this matrix")


def ev_s(s: mx.SNode, env: Env) -> mp.mpf:
    _matrices, scalars = env
    match s:
        case mx.Num(v):
            return mp.mpf(int(v.p)) / mp.mpf(int(v.q))
        case mx.Sym(n):
            return scalars[n]
        case mx.Tr(a):
            m = ev_m(a, env)
            return sum((m[i, i] for i in range(m.rows)), mp.mpf(0))
        case mx.AsScalar(a):
            return ev_m(a, env)[0, 0]
        case mx.Det(a):
            return mp.det(ev_m(a, env))
        case mx.LogDet(a):
            d = abs(mp.det(ev_m(a, env)))
            if d < MIN_DET * mp.mpf(10) ** -4:
                raise Singular
            return mp.log(d)
        case mx.SAdd(ts):
            return sum((ev_s(t, env) for t in ts), mp.mpf(0))
        case mx.SMul(fs):
            out = mp.mpf(1)
            for f in fs:
                out = out * ev_s(f, env)
            return out
        case mx.SPow(b, k):
            base = ev_s(b, env)
            if base == 0 and k < 0:
                raise Singular
            return base**k
    raise mx.MatrixExprError("cannot evaluate this scalar")


# --------------------------------------------------------------------------- setup and sampling


@dataclass
class Setup:
    ctx: mx.Context
    decls: dict[str, MatrixDecl]
    explicit: dict[str, mp.matrix]
    scalars: list[str]
    pd: frozenset[str]


def _draw(decl: MatrixDecl, setup: Setup, rng: random.Random) -> mp.matrix:
    if decl.name in setup.explicit:
        return setup.explicit[decl.name]
    rows, cols = decl.shape
    need_invertible = decl.name in setup.ctx.invertible
    for _ in range(60):
        m = mp.matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                m[i, j] = mp.mpf(rng.randint(-1000, 1000)) / 1000
        if decl.name in setup.pd:
            m = m * m.T + mp.eye(rows) * mp.mpf("0.3")
        elif decl.name in setup.ctx.symmetric:
            m = (m + m.T) / 2
        if not need_invertible or abs(mp.det(m)) > mp.mpf("0.15"):
            return m
    raise Singular


def draw_env(setup: Setup, rng: random.Random) -> Env:
    matrices = {name: _draw(decl, setup, rng) for name, decl in setup.decls.items()}
    scalars = {name: mp.mpf(rng.randint(500, 3000)) / 1000 for name in setup.scalars}
    return matrices, scalars


def _with(env: Env, name: str, value: mp.matrix) -> Env:
    return {**env[0], name: value}, env[1]


def _unit(rows: int, cols: int, i: int, j: int) -> mp.matrix:
    e = mp.zeros(rows, cols)
    e[i, j] = 1
    return e


def _close(a: mp.mpf, b: mp.mpf, tol: mp.mpf) -> bool:
    return bool(abs(a - b) <= tol * max(1, abs(a), abs(b)))


def _matrix_close(a: mp.matrix, b: mp.matrix, tol: mp.mpf) -> bool:
    if (a.rows, a.cols) != (b.rows, b.cols):
        return False
    return all(_close(a[i, j], b[i, j], tol) for i in range(a.rows) for j in range(a.cols))


def _slope(f: mx.SNode, env: Env, var: str, direction: mp.matrix, h: mp.mpf) -> mp.mpf:
    base = env[0][var]
    return (
        ev_s(f, _with(env, var, base + h * direction))
        - ev_s(f, _with(env, var, base - h * direction))
    ) / (2 * h)


@dataclass
class Check:
    checked: int = 0
    agreed: int = 0
    skipped: int = 0
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.checked >= 3 and self.agreed == self.checked


def check_gradient(
    f: mx.SNode,
    var: str,
    g: mx.MNode,
    setup: Setup,
    *,
    constrained: bool,
    alternative: mx.MNode | None = None,
    deadline: float,
    seed: int = 20260101,
) -> tuple[Check, Check | None]:
    """Central differences of the function against the candidate gradient (and an alternative form)."""
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    main, alt = Check(), (Check() if alternative is not None else None)
    rows, cols = setup.ctx.shapes[var]
    with mp.workdps(DPS):
        for _ in range(ATTEMPTS):
            if main.checked >= SAMPLES or time.monotonic() > deadline:
                break
            try:
                env = draw_env(setup, rng)
                candidate = ev_m(g, env)
                other = ev_m(alternative, env) if alternative is not None else None
                ok_main = ok_alt = True
                if (candidate.rows, candidate.cols) != (rows, cols):
                    raise mx.MatrixExprError("the candidate has the wrong shape")
                if constrained:
                    ok_main = _matrix_close(candidate, candidate.T, TOL)  # symmetry is required
                    for i in range(rows):
                        for j in range(i, rows):
                            direction = _unit(rows, cols, i, j)
                            if i != j:
                                direction = direction + _unit(rows, cols, j, i)
                            slope = _slope(f, env, var, direction, STEP)
                            frob = sum(
                                (
                                    candidate[a, b] * direction[a, b]
                                    for a in range(rows)
                                    for b in range(cols)
                                ),
                                mp.mpf(0),
                            )
                            ok_main = ok_main and _close(slope, frob, TOL)
                            if other is not None:
                                ok_alt = ok_alt and _close(slope, other[i, j], TOL)
                    if other is not None:
                        ok_alt = ok_alt and _matrix_close(other, other.T, TOL)
                else:
                    for i in range(rows):
                        for j in range(cols):
                            slope = _slope(f, env, var, _unit(rows, cols, i, j), STEP)
                            ok_main = ok_main and _close(slope, candidate[i, j], TOL)
            except (Singular, ZeroDivisionError, ValueError):
                main.skipped += 1
                continue
            main.checked += 1
            main.agreed += ok_main
            if alt is not None:
                alt.checked += 1
                alt.agreed += ok_alt
    return main, alt


def check_hessian(
    f: mx.SNode, var: str, h_node: mx.MNode, setup: Setup, *, deadline: float, seed: int = 20260102
) -> Check:
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    out = Check()
    n = setup.ctx.shapes[var][0]
    with mp.workdps(DPS):
        for _ in range(ATTEMPTS):
            if out.checked >= SAMPLES or time.monotonic() > deadline:
                break
            try:
                env = draw_env(setup, rng)
                candidate = ev_m(h_node, env)
                base = env[0][var]
                ok = (candidate.rows, candidate.cols) == (n, n)
                for i, j in itertools.product(range(n), repeat=2):
                    if not ok:
                        break
                    ei, ej = _unit(n, 1, i, 0), _unit(n, 1, j, 0)

                    def at(a: mp.matrix, _env: Env = env, _base: mp.matrix = base) -> mp.mpf:
                        return ev_s(f, _with(_env, var, _base + a))

                    second = (
                        at(STEP2 * (ei + ej))
                        - at(STEP2 * (ei - ej))
                        - at(STEP2 * (ej - ei))
                        + at(-STEP2 * (ei + ej))
                    ) / (4 * STEP2 * STEP2)
                    ok = ok and _close(second, candidate[i, j], TOL2)
            except (Singular, ZeroDivisionError, ValueError):
                out.skipped += 1
                continue
            out.checked += 1
            out.agreed += ok
    return out


def check_jacobian(
    f: mx.MNode, var: str, j_node: mx.MNode, setup: Setup, *, deadline: float, seed: int = 20260103
) -> Check:
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    out = Check()
    n = setup.ctx.shapes[var][0]
    with mp.workdps(DPS):
        for _ in range(ATTEMPTS):
            if out.checked >= SAMPLES or time.monotonic() > deadline:
                break
            try:
                env = draw_env(setup, rng)
                candidate = ev_m(j_node, env)
                base = env[0][var]
                ok = candidate.cols == n
                for k in range(n):
                    e = _unit(n, 1, k, 0)
                    plus = ev_m(f, _with(env, var, base + STEP * e))
                    minus = ev_m(f, _with(env, var, base - STEP * e))
                    column = (plus - minus) / (2 * STEP)
                    ok = ok and column.rows == candidate.rows
                    ok = ok and all(
                        _close(column[r, 0], candidate[r, k], TOL) for r in range(column.rows)
                    )
            except (Singular, ZeroDivisionError, ValueError):
                out.skipped += 1
                continue
            out.checked += 1
            out.agreed += ok
    return out


# --------------------------------------------------------------------------- the operation


def _explicit_matrix(decl: MatrixDecl, limits: Limits) -> mp.matrix:
    assert decl.entries is not None  # noqa: S101 - callers check
    rows = []
    for row in decl.entries:
        values = []
        for text in row:
            expr = parse_latex(text, limits=limits)
            if expr.free_symbols or not expr.is_real:
                raise UnsupportedInput(
                    "gradients need numeric, real entries for an explicit matrix"
                )
            values.append(mp.mpf(str(sp.N(expr, 30))))
        rows.append(values)
    return mp.matrix(rows)


def build_setup(data: MatrixCalculusInput, limits: Limits) -> tuple[Setup, list[str], list[str]]:
    """Declared matrices with the properties stated or visible in the entries (and contradictions)."""
    shapes: dict[str, tuple[int, int]] = {}
    explicit: dict[str, mp.matrix] = {}
    invertible = set(data.invertible)
    symmetric: set[str] = set()
    pd: set[str] = set()
    notes: list[str] = []
    names = [d.name for d in data.matrices]
    if len(set(names)) != len(names):
        raise mx.MatrixExprError("a matrix is declared twice")
    unknown = set(data.invertible) - set(names)
    if unknown:
        raise mx.MatrixExprError(
            f"invertible names a matrix that is not declared: {sorted(unknown)}"
        )
    with mp.workdps(DPS):
        for d in data.matrices:
            shapes[d.name] = d.shape
            if d.entries is not None:
                explicit[d.name] = _explicit_matrix(d, limits)
            if d.positive_definite:
                pd.add(d.name)
                symmetric.add(d.name)
                invertible.add(d.name)
            elif d.symmetric:
                symmetric.add(d.name)
        for name, m in explicit.items():
            is_symmetric = m.rows == m.cols and all(
                m[i, j] == m[j, i] for i in range(m.rows) for j in range(m.cols)
            )
            nonsingular = m.rows == m.cols and mp.det(m) != 0
            if name in symmetric and not is_symmetric:
                raise mx.MatrixExprError(
                    f"{name} is declared symmetric but its entries are not",
                    "declaration_contradicted",
                )
            if name in invertible and not nonsingular:
                raise mx.MatrixExprError(
                    f"{name} is declared invertible but its determinant is 0",
                    "declaration_contradicted",
                )
            if name in pd and not all(ev > 0 for ev in mp.eigsy(m)[0]):
                raise mx.MatrixExprError(
                    f"{name} is declared positive definite but is not", "declaration_contradicted"
                )
            if is_symmetric and m.rows == m.cols and name not in symmetric:
                symmetric.add(name)
                notes.append(f"{name} is symmetric (from the entries)")
            if nonsingular and name not in invertible:
                invertible.add(name)
                notes.append(f"{name} is invertible (from the entries)")
    ctx = mx.Context(shapes, frozenset(invertible), frozenset(symmetric))
    decls = {d.name: d for d in data.matrices}
    return Setup(ctx, decls, explicit, [], frozenset(pd)), notes, sorted(pd)


def _walk(node: object):  # type: ignore[no-untyped-def]
    yield node
    match node:
        case (
            mx.T(a)
            | mx.Inv(a)
            | mx.DiagPart(a)
            | mx.Tr(a)
            | mx.Det(a)
            | mx.LogDet(a)
            | mx.AsScalar(a)
        ):
            yield from _walk(a)
        case mx.Prod(fs) | mx.SMul(fs) | mx.Add(fs) | mx.SAdd(fs):
            for f in fs:
                yield from _walk(f)
        case mx.Scale(c, a):
            yield from _walk(c)
            yield from _walk(a)
        case mx.SPow(b, _):
            yield from _walk(b)


def _check_invertibility(f: mx.MNode | mx.SNode, setup: Setup, var: str) -> list[str]:
    """Atoms whose inverse is taken must be declared invertible; composites are accepted as written."""
    used: list[str] = []
    for node in _walk(f):
        arg = None
        if isinstance(node, mx.Inv | mx.LogDet) or (
            isinstance(node, mx.Det) and mx.contains(node.arg, var)
        ):
            arg = node.arg
        if arg is None:
            continue
        missing = mx.atoms_needing_declaration(arg, setup.ctx) if isinstance(arg, mx.Var) else []
        if missing:
            raise mx.MatrixExprError(
                f"{missing[0]} must be declared invertible: the derivative needs its inverse",
                "invertibility_not_declared",
            )
        if not isinstance(arg, mx.Var) and not mx.known_invertible(arg, setup.ctx):
            used.append(
                f"{mx.text_m(arg)} is invertible where it is evaluated (implied by the expression)"
            )
    return sorted(set(used))


def _fail(
    data: MatrixCalculusInput, status: Status, reason: str, message: str
) -> MatrixCalculusResult:
    return MatrixCalculusResult(status=status, action=data.action, reason=reason, message=message)


def run_gradient(
    data: MatrixCalculusInput, limits: Limits, deadline: float
) -> MatrixCalculusResult:
    try:
        return _run(data, limits, deadline)
    except mx.MatrixExprError as exc:
        status = (
            Status.CONDITIONS_NOT_SATISFIED
            if exc.reason == "invertibility_not_declared"
            else (
                Status.UNSUPPORTED
                if exc.reason in {"non_scalar_gradient", "identically_zero"}
                else (
                    Status.RESOURCE_LIMIT if exc.reason == "resource_limit" else Status.PARSE_ERROR
                )
            )
        )
        return _fail(data, status, exc.reason, str(exc))
    except LimitError as exc:
        return _fail(data, Status.RESOURCE_LIMIT, "resource_limit", str(exc))
    except UnsupportedInput as exc:
        return _fail(data, Status.UNSUPPORTED, "unsupported_form", str(exc))
    except MinervaMathError as exc:
        return _fail(data, Status.PARSE_ERROR, "parse_error", str(exc))


MATRIX_NODES = (mx.Var, mx.T, mx.Inv, mx.Prod, mx.Add, mx.Scale, mx.Ident, mx.DiagPart)
TITLES = {
    "gradient": "Gradient by differentials",
    "hessian": "Hessian by differentials",
    "jacobian": "Jacobian by differentials",
}


def _run(data: MatrixCalculusInput, limits: Limits, deadline: float) -> MatrixCalculusResult:
    if data.expression.format != "text":
        raise UnsupportedInput(
            "gradient, jacobian and hessian read the plain-text syntax "
            "(transpose(X), inv(S), tr(A*X), logdet(S), x - mu): write the expression in that form"
        )
    var = data.variable
    if var is None:
        raise mx.MatrixExprError("name the matrix to differentiate with respect to")
    setup, notes, _ = build_setup(data, limits)
    ctx = setup.ctx
    if var not in ctx.shapes:
        raise mx.MatrixExprError(f"{var!r} is not a declared matrix")
    if var in setup.explicit:
        raise UnsupportedInput(
            "the variable of differentiation must be an abstract matrix, not explicit entries"
        )
    parsed = mx.parse(data.expression.input, ctx)
    setup.scalars = sorted(mx.scalar_symbols(parsed))
    vshape = ctx.shapes[var]
    action = data.action

    scalar: mx.SNode | None = None
    vector: mx.MNode | None = None
    value_shape = (1, 1)
    if isinstance(parsed, MATRIX_NODES):
        value_shape = mx.shape_of(parsed, ctx)
        if action == "jacobian":
            vector = parsed
        elif value_shape == (1, 1):
            scalar = mx.AsScalar(parsed)
        else:
            raise mx.MatrixExprError(
                "a matrix-valued function has no gradient here: the derivative of a matrix by a "
                "matrix is a fourth-order tensor",
                "non_scalar_gradient",
            )
    elif action == "jacobian":
        raise mx.MatrixExprError(
            "a scalar function has a gradient, not a Jacobian", "non_scalar_gradient"
        )
    else:
        scalar = parsed
    if action == "jacobian" and (value_shape[1] != 1 or vshape[1] != 1):
        raise mx.MatrixExprError(
            "the Jacobian is for a column vector of a column vector", "non_scalar_gradient"
        )
    if action == "hessian" and vshape[1] != 1:
        raise mx.MatrixExprError("the Hessian is for a function of a vector", "non_scalar_gradient")
    function: mx.MNode | mx.SNode = scalar if scalar is not None else vector  # type: ignore[assignment]

    assumptions = notes + _check_invertibility(function, setup, var)
    constrained = var in ctx.symmetric
    constraint = (
        "positive_definite" if var in setup.pd else ("symmetric" if constrained else "none")
    )
    if constrained and action != "gradient":
        raise UnsupportedInput("Jacobians and Hessians are for unconstrained vectors")

    # -- the candidate: derived, or the user's
    differ = Differ(ctx, var)
    alternative: mx.MNode | None = None
    candidate_text = (data.candidate or {}).get("result")
    if candidate_text is not None:
        given = mx.parse(candidate_text, ctx)
        if not isinstance(given, MATRIX_NODES):
            raise mx.MatrixExprError("the answer to check is a scalar, not a matrix")
        result_node = given
    elif scalar is not None and action == "gradient":
        try:
            result_node = mx.tidy(read_gradient(differ.ds(scalar), var), ctx)
            if constrained:
                half = mx.Num(sp.Rational(1, 2))
                result_node = mx.tidy(
                    mx.m_scale(half, mx.m_add(result_node, mx.T(result_node))), ctx
                )
                alternative = mx.tidy(
                    mx.m_add(
                        mx.m_scale(mx.Num(sp.Integer(2)), result_node),
                        mx.m_scale(mx.MINUS_ONE, mx.DiagPart(result_node)),
                    ),
                    ctx,
                )
        except mx.MatrixExprError as exc:
            if exc.reason != "identically_zero":
                raise
            result_node = mx.Scale(mx.ZERO, mx.Var(var))
    elif scalar is not None:  # hessian: the Jacobian of the gradient
        gradient = mx.tidy(read_gradient(differ.ds(scalar), var), ctx)
        result_node = mx.tidy(read_jacobian(differ.dm(gradient), ctx, var, vshape[0]), ctx)
    else:
        assert vector is not None  # noqa: S101 - one of scalar / vector is set
        result_node = mx.tidy(read_jacobian(differ.dm(vector), ctx, var, value_shape[0]), ctx)

    if (
        candidate_text is None
    ):  # print what is verified: prettify only rearranges, it never changes the value
        result_node = mx.prettify(result_node, ctx)
        alternative = mx.prettify(alternative, ctx) if alternative is not None else None

    # -- verification by an independent numerical derivative
    expected_shape = {
        "gradient": vshape,
        "hessian": (vshape[0], vshape[0]),
        "jacobian": (value_shape[0], vshape[0]),
    }[action]
    if mx.shape_of(result_node, ctx) != expected_shape:
        if candidate_text is not None:
            return MatrixCalculusResult(
                status=Status.NOT_VERIFIED,
                action=action,
                reason="shape_mismatch",
                message=f"the answer has shape {mx.shape_of(result_node, ctx)}, "
                f"the derivative has {expected_shape}",
            )
        raise mx.MatrixExprError("internal: the derivative has the wrong shape")
    alt: Check | None = None
    if action == "jacobian":
        assert vector is not None  # noqa: S101
        main = check_jacobian(vector, var, result_node, setup, deadline=deadline)
    elif action == "hessian":
        assert scalar is not None  # noqa: S101
        main = check_hessian(scalar, var, result_node, setup, deadline=deadline)
    else:
        assert scalar is not None  # noqa: S101
        main, alt = check_gradient(
            scalar,
            var,
            result_node,
            setup,
            constrained=constrained,
            alternative=alternative,
            deadline=deadline,
        )

    if main.checked == 0:
        return _fail(
            data,
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_empty",
            "the function is not defined at any sampled point of the declared sizes (a determinant "
            "or inverse is singular there)",
        )
    verification = MatrixVerification(
        method=(
            "central finite differences of the function at random real matrices of the declared "
            "sizes" + ("; symmetry of the candidate is required" if constrained else "")
        ),
        points_checked=main.checked,
        points_agreed=main.agreed,
        passed=main.passed,
        symbolic_zero=None,
        symbolic_method="not run: the numerical derivative is the only leg here",
        note="numerical agreement is evidence, not a proof",
    )
    input_latex = _input_latex(function)
    if not main.passed:
        return MatrixCalculusResult(
            status=Status.NOT_VERIFIED,
            action=action,
            reason="numeric_disagreement",
            input_latex=input_latex,
            verification=verification,
            assumptions_used=assumptions,
            message="the derivative did not agree with the numerical derivative of the function; "
            "not shown",
        )
    alternatives = (
        [
            MatrixAlternative(
                convention="symmetric_entrywise",
                latex=mx.latex_m(alternative),
                text=mx.text_m(alternative),
                points_checked=alt.checked,
                points_agreed=alt.agreed,
                note="the partial derivatives with respect to the independent entries (i <= j)",
            )
        ]
        if alternative is not None and alt is not None and alt.passed
        else []
    )
    derivative = MatrixDerivative(
        kind=action,
        variable_shape=vshape,
        result_shape=expected_shape,
        layout="numerator" if action == "jacobian" else "shape_of_variable",
        constraint=constraint,
        convention="symmetric_frobenius" if constrained else "unconstrained",
        alternatives=alternatives,
    )
    return MatrixCalculusResult(
        status=Status.OK,
        action=action,
        input_latex=input_latex,
        structure=_structure(ctx, var, action, constraint),
        method=f"{action}_differential",
        method_title=TITLES[action],
        result_kind=action,
        result_latex=mx.latex_m(result_node),
        result_text=mx.text_m(result_node),
        evaluated=True,
        assumptions_used=assumptions,
        steps=[
            MatrixStep(
                method=f"{action}_differential",
                title="Differential, brought to the form tr(G^T dV)"
                if action == "gradient"
                else "Differential of the gradient"
                if action == "hessian"
                else "Differential, brought to the form J dv",
                before_latex=input_latex,
                after_latex=mx.latex_m(result_node),
            )
        ],
        verification=verification,
        derivative=derivative,
        notes=[
            "over the real numbers: a transpose, not a conjugate transpose",
            "logdet is log |det|; its derivative is the same",
            *(
                [
                    "for a symmetric variable this is the gradient within the symmetric matrices "
                    "(Frobenius inner product); the entrywise form is given separately"
                ]
                if constrained
                else []
            ),
        ],
        message="verified against the numerical derivative; numerical agreement is evidence, not a proof",
    )


def _input_latex(node: mx.MNode | mx.SNode) -> str:
    return mx.latex_m(node) if isinstance(node, MATRIX_NODES) else mx.latex_s(node)


def _structure(ctx: mx.Context, var: str, action: str, constraint: str) -> str:
    rows, cols = ctx.shapes[var]
    kind = "vector function" if action == "jacobian" else "scalar function"
    suffix = "" if constraint == "none" else f", {constraint.replace('_', ' ')}"
    return f"{kind} of the {rows}x{cols} matrix {var}{suffix}"
