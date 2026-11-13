"""Matrix expressions with transposes, inverses, traces and determinants (the `gradient` actions).

The existing `expressions` module represents products of declared matrices as noncommutative SymPy
terms, which cannot hold `transpose`, `inv`, `tr`, `det` or `logdet` (SymPy rewrites `X * X**-1` and
`(X**T)**-1` on noncommutative symbols). Gradients therefore use a small tree of their own, read with
Python's `ast` module (never executed) from a plain-text syntax:

    +  -  *  /  **   numbers, names   tr(M) det(M) logdet(M) transpose(M) inv(M)

`*` between two matrices is the matrix product (shapes are checked), `*` with a scalar scales. A product
of matrices whose shape is 1 x 1 (a quadratic form, a trace-like block) is a scalar. Matrices are never
reordered. `logdet` is log |det| (the derivative is the same); everything is over the reals.
"""

from __future__ import annotations

import ast
import itertools
import re
from dataclasses import dataclass

import sympy as sp

from ..errors import MinervaMathError

MAX_NODES = 400
MAX_POWER = 6
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
FUNCTIONS = {"tr", "det", "logdet", "transpose", "inv"}


class MatrixExprError(MinervaMathError):
    """The expression cannot be read, or breaks a shape or declaration rule (`reason` is a code)."""

    def __init__(self, message: str, reason: str = "parse_error") -> None:
        super().__init__(message)
        self.reason = reason


# --------------------------------------------------------------------------- nodes


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class T:
    arg: MNode


@dataclass(frozen=True)
class Inv:
    arg: MNode


@dataclass(frozen=True)
class Prod:
    factors: tuple[MNode, ...]


@dataclass(frozen=True)
class Add:
    terms: tuple[MNode, ...]


@dataclass(frozen=True)
class Scale:
    coeff: SNode
    arg: MNode


@dataclass(frozen=True)
class Ident:
    size: int


@dataclass(frozen=True)
class DiagPart:
    arg: MNode


@dataclass(frozen=True)
class Num:
    value: sp.Rational


@dataclass(frozen=True)
class Sym:
    name: str


@dataclass(frozen=True)
class Tr:
    arg: MNode


@dataclass(frozen=True)
class Det:
    arg: MNode


@dataclass(frozen=True)
class LogDet:
    arg: MNode


@dataclass(frozen=True)
class AsScalar:
    """A matrix product whose shape is 1 x 1, used as a scalar."""

    arg: MNode


@dataclass(frozen=True)
class SAdd:
    terms: tuple[SNode, ...]


@dataclass(frozen=True)
class SMul:
    factors: tuple[SNode, ...]


@dataclass(frozen=True)
class SPow:
    base: SNode
    exponent: int


MNode = Var | T | Inv | Prod | Add | Scale | Ident | DiagPart
SNode = Num | Sym | Tr | Det | LogDet | AsScalar | SAdd | SMul | SPow
ONE = Num(sp.Integer(1))
ZERO = Num(sp.Integer(0))
MINUS_ONE = Num(sp.Integer(-1))


@dataclass(frozen=True)
class Context:
    """Declared matrices: shapes and the properties the user stated or the entries show."""

    shapes: dict[str, tuple[int, int]]
    invertible: frozenset[str] = frozenset()
    symmetric: frozenset[str] = frozenset()


# --------------------------------------------------------------------------- constructors


def s_mul(*factors: SNode) -> SNode:
    flat: list[SNode] = []
    value = sp.Integer(1)
    for f in factors:
        for g in f.factors if isinstance(f, SMul) else (f,):
            if isinstance(g, Num):
                value = value * g.value
            else:
                flat.append(g)
    if value == 0:
        return ZERO
    if value != 1 or not flat:
        flat.insert(0, Num(value))
    return flat[0] if len(flat) == 1 else SMul(tuple(flat))


def s_add(*terms: SNode) -> SNode:
    flat = [g for t in terms for g in (t.terms if isinstance(t, SAdd) else (t,))]
    return flat[0] if len(flat) == 1 else SAdd(tuple(flat))


def s_pow(base: SNode, exponent: int) -> SNode:
    if exponent == 0:
        return ONE
    if exponent == 1:
        return base
    if isinstance(base, Num):
        if base.value == 0 and exponent < 0:
            raise MatrixExprError("division by zero")
        return Num(sp.Rational(base.value) ** exponent)
    return SPow(base, exponent)


def m_prod(*factors: MNode) -> MNode:
    flat = [g for f in factors for g in (f.factors if isinstance(f, Prod) else (f,))]
    return flat[0] if len(flat) == 1 else Prod(tuple(flat))


def m_add(*terms: MNode) -> MNode:
    flat = [g for t in terms for g in (t.terms if isinstance(t, Add) else (t,))]
    return flat[0] if len(flat) == 1 else Add(tuple(flat))


def m_scale(coeff: SNode, arg: MNode) -> MNode:
    if coeff == ONE:
        return arg
    if isinstance(arg, Scale):
        return Scale(s_mul(coeff, arg.coeff), arg.arg)
    return Scale(coeff, arg)


# --------------------------------------------------------------------------- shapes


def shape_of(m: MNode, ctx: Context) -> tuple[int, int]:
    match m:
        case Var(name):
            return ctx.shapes[name]
        case T(arg):
            r, c = shape_of(arg, ctx)
            return c, r
        case Inv(arg) | DiagPart(arg):
            return shape_of(arg, ctx)
        case Ident(size):
            return size, size
        case Scale(_, arg):
            return shape_of(arg, ctx)
        case Prod(factors):
            shapes = [shape_of(f, ctx) for f in factors]
            for (_, c), (r, _) in itertools.pairwise(shapes):
                if c != r:
                    raise MatrixExprError(f"shapes do not match in a product: {shapes}")
            return shapes[0][0], shapes[-1][1]
        case Add(terms):
            shapes = [shape_of(t, ctx) for t in terms]
            if len(set(shapes)) != 1:
                raise MatrixExprError(f"shapes do not match in a sum: {shapes}")
            return shapes[0]
    raise MatrixExprError("unsupported matrix expression")


def _is_scalar_block(shape: tuple[int, int]) -> bool:
    return shape == (1, 1)


def group_chain(factors: list[MNode], ctx: Context) -> tuple[list[SNode], list[MNode]]:
    """Split a flat chain of matrices into scalar blocks (1 x 1 sub-products) and a conformable chain."""
    shapes = [shape_of(f, ctx) for f in factors]
    k = len(factors)

    def conformable(lo: int, hi: int) -> bool:
        return all(shapes[i][1] == shapes[i + 1][0] for i in range(lo, hi - 1))

    if conformable(0, k):
        return [], factors

    def removals(pos: int) -> list[list[tuple[int, int]]]:
        if pos >= k:
            return [[]]
        out = removals(pos + 1)
        for hi in range(pos + 2, k + 1):
            if conformable(pos, hi) and (shapes[pos][0], shapes[hi - 1][1]) == (1, 1):
                out += [[(pos, hi), *rest] for rest in removals(hi)]
        return out

    valid: list[tuple[list[tuple[int, int]], list[int]]] = []
    for removed in removals(0):
        kept = [i for i in range(k) if not any(a <= i < b for a, b in removed)]
        if all(shapes[kept[j]][1] == shapes[kept[j + 1]][0] for j in range(len(kept) - 1)):
            valid.append((removed, kept))
    if not valid:
        raise MatrixExprError(
            "the shapes in this product do not match: " + " * ".join(f"{r}x{c}" for r, c in shapes)
        )
    fewest = min(len(r) for r, _ in valid)
    best = [v for v in valid if len(v[0]) == fewest]
    if len(best) > 1:
        raise MatrixExprError(
            "this product can be grouped in more than one way (a 1x1 block could be a scalar "
            "factor in several places); add parentheses or split it"
        )
    removed, kept = best[0]
    return [AsScalar(m_prod(*factors[a:b])) for a, b in removed], [factors[i] for i in kept]


# --------------------------------------------------------------------------- parsing


def parse(text: str, ctx: Context) -> MNode | SNode:
    """Read an expression. Names that are not declared matrices are scalar symbols."""
    if len(text) > 2000:
        raise MatrixExprError("the expression is too long", "resource_limit")
    try:
        tree = ast.parse(text.strip().replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise MatrixExprError("could not read the expression") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise MatrixExprError("the expression is too large", "resource_limit")
    return _convert(tree.body, ctx)


def _operands(node: ast.expr, out: list[ast.expr]) -> None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        _operands(node.left, out)
        _operands(node.right, out)
    else:
        out.append(node)


def _convert(node: ast.expr, ctx: Context) -> MNode | SNode:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise MatrixExprError("only numbers are allowed as literals")
        return Num(sp.Rational(str(node.value)))
    if isinstance(node, ast.Name):
        if node.id in FUNCTIONS or not _NAME.match(node.id):
            raise MatrixExprError(f"{node.id!r} is a function name here; write {node.id}(…)")
        return Var(node.id) if node.id in ctx.shapes else Sym(node.id)
    if isinstance(node, ast.UnaryOp):
        operand = _convert(node.operand, ctx)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return _scale(MINUS_ONE, operand)
        raise MatrixExprError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        return _binary(node, ctx)
    if isinstance(node, ast.Call):
        return _call(node, ctx)
    raise MatrixExprError(f"unsupported syntax: {type(node).__name__}")


def _scale(coeff: SNode, operand: MNode | SNode) -> MNode | SNode:
    if _is_s(operand):
        return s_mul(coeff, operand)  # type: ignore[arg-type]
    return m_scale(coeff, operand)  # type: ignore[arg-type]


def _is_s(node: object) -> bool:
    return isinstance(node, Num | Sym | Tr | Det | LogDet | AsScalar | SAdd | SMul | SPow)


def _binary(node: ast.BinOp, ctx: Context) -> MNode | SNode:
    op = node.op
    if isinstance(op, ast.Mult):
        parts: list[ast.expr] = []
        _operands(node, parts)
        return _product([_convert(p, ctx) for p in parts], ctx)
    left, right = _convert(node.left, ctx), _convert(node.right, ctx)
    if isinstance(op, ast.Add | ast.Sub):
        sign = ONE if isinstance(op, ast.Add) else MINUS_ONE
        right = _scale(sign, right) if sign != ONE else right
        if _is_s(left) and _is_s(right):
            return s_add(left, right)  # type: ignore[arg-type]
        if _is_s(left) or _is_s(right):
            raise MatrixExprError("a scalar and a matrix cannot be added")
        if shape_of(left, ctx) != shape_of(right, ctx):  # type: ignore[arg-type]
            raise MatrixExprError("the matrices in a sum have different shapes")
        return m_add(left, right)  # type: ignore[arg-type]
    if isinstance(op, ast.Div):
        if not _is_s(right):
            raise MatrixExprError("division by a matrix: write inv(M) and mind the order")
        return _scale(s_pow(right, -1), left)  # type: ignore[arg-type]
    if isinstance(op, ast.Pow):
        if not isinstance(right, Num) or right.value.q != 1:
            raise MatrixExprError("only integer powers are supported")
        k = int(right.value.p)
        if _is_s(left):
            return s_pow(left, k)  # type: ignore[arg-type]
        if k == -1:
            return _inverse(left, ctx)  # type: ignore[arg-type]
        if not 1 <= k <= MAX_POWER:
            raise MatrixExprError(f"a matrix power is between 1 and {MAX_POWER}, or -1")
        return _product([left] * k, ctx)
    raise MatrixExprError("unsupported operator")


def _inverse(m: MNode, ctx: Context) -> MNode:
    rows, cols = shape_of(m, ctx)
    if rows != cols:
        raise MatrixExprError("the inverse of a non-square matrix")
    return Inv(m)


def _call(node: ast.Call, ctx: Context) -> MNode | SNode:
    if not isinstance(node.func, ast.Name) or node.keywords or len(node.args) != 1:
        raise MatrixExprError("unsupported function call")
    name = node.func.id
    if name not in FUNCTIONS:
        raise MatrixExprError(f"unsupported function {name!r}")
    arg = _convert(node.args[0], ctx)
    if _is_s(arg):
        raise MatrixExprError(f"{name} needs a matrix")
    m: MNode = arg  # type: ignore[assignment]
    rows, cols = shape_of(m, ctx)
    if name == "transpose":
        return T(m)
    if name == "inv":
        return _inverse(m, ctx)
    if rows != cols:
        raise MatrixExprError(f"{name} needs a square matrix, not {rows}x{cols}")
    if name == "tr":
        return Tr(m)
    return Det(m) if name == "det" else LogDet(m)


def _product(parts: list[MNode | SNode], ctx: Context) -> MNode | SNode:
    scalars = [p for p in parts if _is_s(p)]
    matrices = [p for p in parts if not _is_s(p)]
    coeffs: list[SNode] = list(scalars)  # type: ignore[arg-type]
    if not matrices:
        return s_mul(*coeffs)
    blocks, chain = group_chain(matrices, ctx)  # type: ignore[arg-type]
    coeffs += blocks
    scalar = s_mul(*coeffs) if coeffs else ONE
    if not chain:
        return scalar
    result = m_prod(*chain)
    if len(matrices) > 1 and shape_of(result, ctx) == (1, 1):
        return s_mul(scalar, AsScalar(result))
    return m_scale(scalar, result)


# --------------------------------------------------------------------------- properties


def contains(node: MNode | SNode, name: str) -> bool:
    match node:
        case Var(n):
            return n == name
        case T(a) | Inv(a) | DiagPart(a) | Tr(a) | Det(a) | LogDet(a) | AsScalar(a):
            return contains(a, name)
        case Prod(fs) | SMul(fs):
            return any(contains(f, name) for f in fs)
        case Add(ts) | SAdd(ts):
            return any(contains(t, name) for t in ts)
        case Scale(c, a):
            return contains(c, name) or contains(a, name)
        case SPow(b, _):
            return contains(b, name)
    return False


def scalar_symbols(node: MNode | SNode) -> set[str]:
    match node:
        case Sym(n):
            return {n}
        case T(a) | Inv(a) | DiagPart(a) | Tr(a) | Det(a) | LogDet(a) | AsScalar(a):
            return scalar_symbols(a)
        case Prod(fs) | SMul(fs) | Add(fs) | SAdd(fs):
            return set().union(*[scalar_symbols(f) for f in fs])
        case Scale(c, a):
            return scalar_symbols(c) | scalar_symbols(a)
        case SPow(b, _):
            return scalar_symbols(b)
    return set()


def known_invertible(m: MNode, ctx: Context) -> bool:
    """Invertibility that follows from the declarations (atoms) and from closure under *, ^T, ^-1."""
    match m:
        case Var(n):
            return n in ctx.invertible
        case T(a) | Inv(a):
            return known_invertible(a, ctx)
        case Prod(fs):
            return all(known_invertible(f, ctx) for f in fs)
        case Scale(Num(v), a):
            return v != 0 and known_invertible(a, ctx)
        case Ident():
            return True
    return False


def atoms_needing_declaration(m: MNode, ctx: Context) -> list[str]:
    """Matrix atoms whose inverse is taken but which are not declared invertible."""
    match m:
        case Var(n):
            return [] if n in ctx.invertible else [n]
        case T(a) | Inv(a):
            return atoms_needing_declaration(a, ctx)
        case Prod(fs):
            return [n for f in fs for n in atoms_needing_declaration(f, ctx)]
        case Scale(_, a):
            return atoms_needing_declaration(a, ctx)
    return []


# --------------------------------------------------------------------------- tidying


def _symmetric(m: MNode, ctx: Context) -> bool:
    return tidy(T(m), ctx) == tidy(m, ctx)


def tidy(m: MNode, ctx: Context) -> MNode:
    """Push transposes inward, drop what is symmetric by declaration, merge like terms."""
    match m:
        case Var() | Ident():
            return m
        case T(a):
            a = tidy(a, ctx)
            return _transpose(a, ctx)
        case Inv(a):
            return Inv(tidy(a, ctx))
        case DiagPart(a):
            return DiagPart(tidy(a, ctx))
        case Scale(c, a):
            inner = tidy(a, ctx)
            if isinstance(c, Num) and isinstance(inner, Add):
                return _merge([m_scale(c, t) for t in inner.terms])
            return m_scale(c, inner)
        case Prod(fs):
            items = [tidy(f, ctx) for f in fs]
            coeff: SNode = ONE
            flat: list[MNode] = []
            for it in items:
                while isinstance(it, Scale):
                    coeff = s_mul(coeff, it.coeff)
                    it = it.arg
                flat.extend(it.factors if isinstance(it, Prod) else [it])
            return m_scale(coeff, m_prod(*flat))
        case Add(ts):
            return _merge([tidy(t, ctx) for t in ts])
    return m


def _transpose(a: MNode, ctx: Context) -> MNode:
    match a:
        case Var(n):
            return a if n in ctx.symmetric else T(a)
        case Ident():
            return a
        case T(b):
            return b
        case Inv(b):
            return a if _symmetric(b, ctx) else T(a)
        case Scale(c, b):
            return m_scale(c, _transpose(b, ctx))
        case Prod(fs):
            return m_prod(*[_transpose(f, ctx) for f in reversed(fs)])
        case Add(ts):
            return _merge([_transpose(t, ctx) for t in ts])
        case DiagPart():
            return a
    return T(a)


def _merge(terms: list[MNode]) -> MNode:
    """Sum like terms: c1 * M + c2 * M = (c1 + c2) * M, with numeric coefficients."""
    order: list[MNode] = []
    numbers: dict[MNode, sp.Rational] = {}
    others: list[MNode] = []
    for t in terms:
        for u in t.terms if isinstance(t, Add) else (t,):
            coeff, body = (u.coeff, u.arg) if isinstance(u, Scale) else (ONE, u)
            if isinstance(coeff, Num):
                if body not in numbers:
                    order.append(body)
                    numbers[body] = sp.Integer(0)
                numbers[body] = numbers[body] + coeff.value
            else:
                others.append(u)
    merged = [m_scale(Num(numbers[b]), b) for b in order if numbers[b] != 0]
    result = merged + others
    if not result:
        raise MatrixExprError("the result is identically zero", "identically_zero")
    return m_add(*result)


# --------------------------------------------------------------------------- rendering (text)


def _num_text(v: sp.Rational) -> str:
    return str(v.p) if v.q == 1 else f"{v.p}/{v.q}"


def text_s(s: SNode, ctx: Context | None = None) -> str:
    match s:
        case Num(v):
            return _num_text(v)
        case Sym(n):
            return n
        case Tr(a):
            return f"tr({text_m(a)})"
        case Det(a):
            return f"det({text_m(a)})"
        case LogDet(a):
            return f"logdet({text_m(a)})"
        case AsScalar(a):
            return f"({text_m(a)})"
        case SAdd(ts):
            return _signed([text_s_signed(t) for t in ts])
        case SMul(fs):
            return "*".join(
                text_s(f) if k == 0 and not isinstance(f, SAdd) else _paren_s(f)
                for k, f in enumerate(fs)
            )
        case SPow(b, k):
            return f"{_paren_s(b)}**{k}" if k >= 0 else f"{_paren_s(b)}**({k})"
    raise MatrixExprError("cannot print this scalar")


def text_s_signed(s: SNode) -> tuple[bool, str]:
    """(negative, text of the absolute term) for printing a sum."""
    if isinstance(s, Num) and s.value < 0:
        return True, text_s(Num(-s.value))
    if isinstance(s, SMul) and isinstance(s.factors[0], Num) and s.factors[0].value < 0:
        return True, text_s(s_mul(Num(-s.factors[0].value), *s.factors[1:]))
    return False, text_s(s)


def _signed(parts: list[tuple[bool, str]]) -> str:
    out = ""
    for k, (negative, body) in enumerate(parts):
        out += (
            (f"-{body}" if negative else body)
            if k == 0
            else (f" - {body}" if negative else f" + {body}")
        )
    return out


def _paren_s(s: SNode) -> str:
    text = text_s(s)
    if isinstance(s, SAdd) or (isinstance(s, Num) and (s.value.q != 1 or s.value < 0)):
        return f"({text})"
    return text


def text_m(m: MNode) -> str:
    match m:
        case Var(n):
            return n
        case T(a):
            return f"transpose({text_m(a)})"
        case Inv(a):
            return f"inv({text_m(a)})"
        case Ident(n):
            return f"eye({n})"
        case DiagPart(a):
            return f"diagpart({text_m(a)})"
        case Prod(fs):
            return "*".join(_paren_m(f) for f in fs)
        case Scale(c, a):
            if c == MINUS_ONE:
                return f"-{_paren_m(a)}"
            lead = _num_text(c.value) if isinstance(c, Num) else _paren_s(c)
            return f"{lead}*{_paren_m(a)}"
        case Add(ts):
            out = ""
            for k, t in enumerate(ts):
                coeff = t.coeff if isinstance(t, Scale) else ONE
                negative = isinstance(coeff, Num) and coeff.value < 0
                body = text_m(m_scale(Num(-coeff.value), t.arg)) if negative else text_m(t)  # type: ignore[union-attr]
                if k == 0:
                    out = f"-{body}" if negative else body
                else:
                    out += f" - {body}" if negative else f" + {body}"
            return out
    raise MatrixExprError("cannot print this matrix")


def _paren_m(m: MNode) -> str:
    text = text_m(m)
    return f"({text})" if isinstance(m, Add | Scale) else text


# --------------------------------------------------------------------------- rendering (LaTeX)

_GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Sigma", "Phi", "Psi", "Omega",
}  # fmt: skip


def _name_latex(name: str) -> str:
    return rf"\{name}" if name in _GREEK else name


def _negate(s: SNode) -> SNode:
    return s_mul(MINUS_ONE, s)


def latex_s(s: SNode) -> str:
    match s:
        case Num(v):
            if v.q == 1:
                return str(v.p)
            sign = "-" if v < 0 else ""
            return rf"{sign}\frac{{{abs(v.p)}}}{{{v.q}}}"
        case Sym(n):
            return _name_latex(n)
        case Tr(a):
            return rf"\operatorname{{tr}}\left({latex_m(a)}\right)"
        case Det(a):
            return rf"\det\left({latex_m(a)}\right)"
        case LogDet(a):
            return rf"\log\left|\det\left({latex_m(a)}\right)\right|"
        case AsScalar(a):
            return latex_m(a)
        case SAdd(ts):
            out = ""
            for k, t in enumerate(ts):
                negative, _ = text_s_signed(t)
                tex = latex_s(t if not negative else _negate(t))
                out += (
                    (f"-{tex}" if negative else tex)
                    if k == 0
                    else (f" - {tex}" if negative else f" + {tex}")
                )
            return out
        case SMul(fs):
            return " ".join(
                rf"\left({latex_s(f)}\right)" if isinstance(f, SAdd) else latex_s(f) for f in fs
            )
        case SPow(b, k):
            return rf"\left({latex_s(b)}\right)^{{{k}}}"
    raise MatrixExprError("cannot print this scalar")


def _latex_base(m: MNode) -> str:
    return latex_m(m) if isinstance(m, Var) else rf"\left({latex_m(m)}\right)"


def latex_m(m: MNode) -> str:
    match m:
        case Var(n):
            return _name_latex(n)
        case T(a):
            return rf"{_latex_base(a)}^{{\top}}"
        case Inv(a):
            return rf"{_latex_base(a)}^{{-1}}"
        case Ident(n):
            return rf"I_{{{n}}}"
        case DiagPart(a):
            return rf"\operatorname{{diag}}\left({latex_m(a)}\right)"
        case Prod(fs):
            return " ".join(
                rf"\left({latex_m(f)}\right)" if isinstance(f, Add | Scale) else latex_m(f)
                for f in fs
            )
        case Scale(c, a):
            if c == MINUS_ONE:
                return rf"-{_latex_factor(a)}"
            return rf"{latex_s(c)} {_latex_factor(a)}"
        case Add(ts):
            out = ""
            for k, t in enumerate(ts):
                coeff = t.coeff if isinstance(t, Scale) else ONE
                negative = isinstance(coeff, Num) and coeff.value < 0
                body = latex_m(m_scale(Num(-coeff.value), t.arg)) if negative else latex_m(t)  # type: ignore[union-attr]
                out += (
                    (f"- {body}" if negative else body)
                    if k == 0
                    else (f" - {body}" if negative else f" + {body}")
                )
            return out
    raise MatrixExprError("cannot print this matrix")


def _latex_factor(m: MNode) -> str:
    return rf"\left({latex_m(m)}\right)" if isinstance(m, Add) else latex_m(m)


def prettify(m: MNode, ctx: Context) -> MNode:
    """Print `(x - mu) * transpose(x - mu)` instead of `(x - mu) * (transpose(x) - transpose(mu))`."""
    match m:
        case Prod(fs):
            out: list[MNode] = []
            for f in map(lambda g: prettify(g, ctx), fs):
                if isinstance(f, Add):
                    for earlier in out:
                        if isinstance(earlier, Add) and tidy(T(earlier), ctx) == f:
                            f = T(earlier)
                            break
                out.append(f)
            return m_prod(*out)
        case Add(ts):
            return m_add(*[prettify(t, ctx) for t in ts])
        case Scale(c, a):
            return m_scale(c, prettify(a, ctx))
        case T(a):
            return T(prettify(a, ctx))
        case Inv(a):
            return Inv(prettify(a, ctx))
        case DiagPart(a):
            return DiagPart(prettify(a, ctx))
    return m
