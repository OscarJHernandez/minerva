"""A small typed normal form for noncommutative matrix expressions.

The parsers build SymPy trees in which declared matrix names are noncommutative symbols, so the
order of every product is preserved. This module converts such a tree into

    Terms = sum of Term;  Term = scalar coefficient * ordered product of Factor
    Factor = Sym(name, power) | Exp(sum of scalar_i * matrix_i)

and rejects everything else as unsupported. The only exact rewriting done here is what holds for
any matrices: powers of one matrix merge, a function of M commutes with M, and exponentials whose
matrices are *declared* to commute merge. Nothing is ever reordered otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from ..errors import LimitError, MinervaMathError, ParseError
from ..latexprint import to_latex

MAX_TERMS = 64
MAX_POWER = 12


class UnsupportedMatrixExpression(MinervaMathError):
    """A valid input outside what this operation does (reported as `unsupported`)."""


MAX_DERIV = 2  # a time-dependent matrix is sampled as a quadratic polynomial in the variable


@dataclass(frozen=True)
class Sym:
    name: str
    power: int = 1
    deriv: int = 0  # number of derivatives with respect to the variable (time-dependent matrices)


@dataclass(frozen=True)
class Exp:
    arg: tuple[tuple[str, sp.Expr], ...]  # (matrix name, scalar coefficient), sorted by name

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(n for n, _ in self.arg)


@dataclass(frozen=True)
class Frechet:
    """The Frechet derivative of the exponential: the integral from 0 to 1 of
    exp((1-s) X) D exp(s X) ds, with X the exponent of `base` and D a direction (a list of
    (matrix, derivative order, scalar coefficient) items). Equals d/dt exp(X(t)) when D = X'(t).
    It is never simplified to D exp(X) here: that needs [X, D] = 0."""

    base: Exp
    direction: tuple[tuple[str, int, sp.Expr], ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([*self.base.names, *(n for n, _, _ in self.direction)]))


Factor = Sym | Exp | Frechet


@dataclass(frozen=True)
class Term:
    coeff: sp.Expr
    factors: tuple[Factor, ...]


Terms = tuple[Term, ...]


@dataclass(frozen=True)
class Commutation:
    """Pairs of matrices known to commute (declared, or established from explicit entries)."""

    pairs: frozenset[frozenset[str]] = frozenset()
    # time-dependent matrices whose values at different times commute (so A and A' commute)
    self_commuting: frozenset[str] = frozenset()

    def commutes(self, a: str, b: str) -> bool:
        return a == b or frozenset((a, b)) in self.pairs

    def sym_commutes(self, a: Sym, b: Sym) -> bool:
        """Whether two plain matrix factors (possibly derivatives) may be swapped."""
        if a.name == b.name:
            return a.deriv == b.deriv or a.name in self.self_commuting
        return self.commutes(a.name, b.name)


@dataclass
class Event:
    """One merge performed while canonicalising."""

    title: str
    before: str
    after: str
    assumption: str | None = None


@dataclass
class Canonical:
    terms: Terms
    events: list[Event] = field(default_factory=list)
    blocked: list[tuple[str, str, str]] = field(default_factory=list)  # (title, needs, latex)
    assumptions: list[str] = field(default_factory=list)


# -- conversion from a SymPy tree ------------------------------------------------------------
def to_terms(expr: sp.Basic, names: frozenset[str]) -> Terms:
    terms = _convert(expr, names)
    if any(not t.factors for t in terms):
        if any(t.factors for t in terms):
            raise UnsupportedMatrixExpression(
                "a scalar is added to a matrix; write the identity matrix explicitly (declare it "
                "as a matrix) or multiply the scalar by a matrix"
            )
        raise UnsupportedMatrixExpression("the expression contains no declared matrix")
    return collect(tuple(terms))


def _convert(e: sp.Basic, names: frozenset[str]) -> list[Term]:
    if e.is_commutative:
        _reject_matrix_names(e, names)
        return [Term(sp.sympify(e), ())]
    if isinstance(e, sp.Symbol):
        if e.name in names:
            return [Term(sp.Integer(1), (Sym(e.name),))]
        raise ParseError(f"{e.name!r} is not a declared matrix")
    if isinstance(e, sp.Add):
        out: list[Term] = []
        for arg in e.args:
            out += _convert(arg, names)
        _cap(out)
        return out
    if isinstance(e, sp.Mul):
        c_part, nc_part = e.args_cnc()
        coeff = sp.Mul(*c_part)
        out = [Term(coeff, ())]
        for factor in nc_part:
            out = _multiply(out, _convert(factor, names))
        return out
    if isinstance(e, sp.Pow):
        base, power = e.args
        if not base.is_commutative:
            if power.is_Integer and 1 <= int(power) <= MAX_POWER:
                inner = _convert(base, names)
                out = inner
                for _ in range(int(power) - 1):
                    out = _multiply(out, inner)
                return out
            raise UnsupportedMatrixExpression(
                "only positive integer powers of a matrix are supported "
                "(no inverses, roots or symbolic powers)"
            )
        raise ParseError(
            "a matrix in an exponent needs the exponential function: write \\exp(tA) or "
            "\\exponentialE^{tA}, because a bare e is an ordinary symbol"
        )
    if isinstance(e, sp.exp):
        return [_exp_factor(e.args[0], names)]
    raise UnsupportedMatrixExpression(
        f"{type(e).__name__.lower()} of a matrix is not supported; only sums, ordered products, "
        "integer powers and the exponential"
    )


def _reject_matrix_names(e: sp.Basic, names: frozenset[str]) -> None:
    clash = {s.name for s in e.free_symbols if s.name in names}
    if clash:  # a declared matrix that reached the scalar parser as a commutative symbol
        raise ParseError(f"{sorted(clash)[0]!r} is a matrix and cannot be used as a scalar here")


def _exp_factor(arg: sp.Basic, names: frozenset[str]) -> Term:
    coeffs: dict[str, sp.Expr] = {}
    for term in _convert(arg, names):
        if (
            len(term.factors) == 1
            and isinstance(term.factors[0], Sym)
            and term.factors[0].power == 1
        ):
            name = term.factors[0].name
            # keep the user's form (for example a factored coefficient) unless terms must be added
            coeffs[name] = sp.expand(coeffs[name] + term.coeff) if name in coeffs else term.coeff
        elif not term.factors:
            raise UnsupportedMatrixExpression("a scalar is added to a matrix inside an exponential")
        else:
            raise UnsupportedMatrixExpression(
                "the exponent must be a sum of scalar multiples of declared matrices "
                "(no products or powers of matrices inside the exponential)"
            )
    arg_items = tuple(sorted((n, c) for n, c in coeffs.items() if c != 0))
    if not arg_items:
        return Term(sp.Integer(1), ())  # exp(0) is the identity: no matrix factor left
    return Term(sp.Integer(1), (Exp(arg_items),))


def _multiply(left: list[Term], right: list[Term]) -> list[Term]:
    if len(left) * len(right) > MAX_TERMS:
        raise LimitError("the expression expands into too many terms")
    return [Term(a.coeff * b.coeff, a.factors + b.factors) for a in left for b in right]


def _cap(terms: list[Term]) -> None:
    if len(terms) > MAX_TERMS:
        raise LimitError("the expression expands into too many terms")


def collect(terms: Terms) -> Terms:
    """Add terms with identical factor lists, drop zero terms, and order the rest deterministically."""
    merged: dict[tuple[Factor, ...], sp.Expr] = {}
    for t in terms:
        key = t.factors
        merged[key] = merged.get(key, sp.Integer(0)) + t.coeff
    out = [Term(sp.expand(c), f) for f, c in merged.items() if sp.expand(c) != 0]
    return tuple(sorted(out, key=lambda term: _sort_key(term.factors)))


def _sort_key(factors: tuple[Factor, ...]) -> str:
    return "|".join(
        f"{f.name}{'p' * f.deriv}^{f.power}"
        if isinstance(f, Sym)
        else ("~" + "+".join(f.names) if isinstance(f, Exp) else "#" + "+".join(f.names))
        for f in factors
    )


# -- shapes ----------------------------------------------------------------------------------
def _dims(shape: tuple[int, int]) -> str:
    return f"{shape[0]} by {shape[1]}"


def shape_of(terms: Terms, shapes: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """The shape of the expression; raises ParseError naming the offending product or sum."""
    result: tuple[int, int] | None = None
    for term in terms:
        current: tuple[int, int] | None = None
        label = ""
        for f in term.factors:
            f_shape, f_label = _factor_shape(f, shapes)
            if current is None:
                current, label = f_shape, f_label
            elif current[1] != f_shape[0]:
                raise ParseError(
                    f"dimension mismatch: {label} is {_dims(current)} and {f_label} is "
                    f"{_dims(f_shape)}, so they cannot be multiplied"
                )
            else:
                current, label = (current[0], f_shape[1]), f"{label}{f_label}"
        if current is None:
            raise UnsupportedMatrixExpression("the expression contains no declared matrix")
        if result is not None and result != current:
            raise ParseError(
                f"dimension mismatch: terms of the sum are {_dims(result)} and {_dims(current)}"
            )
        result = current
    if result is None:
        raise UnsupportedMatrixExpression("the expression is zero")
    return result


def _factor_shape(f: Factor, shapes: dict[str, tuple[int, int]]) -> tuple[tuple[int, int], str]:
    if isinstance(f, Frechet):
        shape, label = _factor_shape(f.base, shapes)
        for name, _, _ in f.direction:
            if shapes[name] != shape:
                raise ParseError(
                    f"dimension mismatch: {name} is {_dims(shapes[name])} but the exponent is "
                    f"{_dims(shape)}"
                )
        return shape, "D" + label
    if isinstance(f, Sym):
        shape = shapes[f.name]
        if f.power != 1 and shape[0] != shape[1]:
            raise ParseError(f"dimension mismatch: {f.name} is {_dims(shape)}, not square")
        return shape, f.name
    first = shapes[f.arg[0][0]]
    for name, _ in f.arg:
        other = shapes[name]
        if other[0] != other[1]:
            raise ParseError(
                f"dimension mismatch: {name} is {_dims(other)}; the exponential needs a "
                "square matrix"
            )
        if other != first:
            raise ParseError(
                f"dimension mismatch: {f.arg[0][0]} is {_dims(first)} and {name} is "
                f"{_dims(other)}, so they cannot be added inside one exponential"
            )
    return first, "e^(" + "+".join(n for n, _ in f.arg) + ")"


# -- canonical form --------------------------------------------------------------------------
def canonicalize(terms: Terms, comm: Commutation) -> Canonical:
    out = Canonical(terms=())
    seen_blocked: set[tuple[str, str]] = set()
    assumed: dict[str, None] = {}
    new_terms: list[Term] = []
    for term in terms:
        factors = list(term.factors)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(factors) - 1:
                a, b = factors[i], factors[i + 1]
                if (
                    isinstance(a, Sym)
                    and isinstance(b, Sym)
                    and a.name == b.name
                    and a.deriv == b.deriv
                ):
                    power = a.power + b.power
                    factors[i : i + 2] = [Sym(a.name, power, a.deriv)] if power != 0 else []
                    changed = True
                elif isinstance(a, Exp) and isinstance(b, Exp):
                    pair_ok, needs = _all_commute(a.names, b.names, comm)
                    if pair_ok:
                        merged = _merge_exp(a, b)
                        before, after = render_factors([a, b]), render_factors(merged)
                        out.events.append(
                            Event(
                                "Merge exponentials of commuting matrices",
                                before,
                                after,
                                _assumption(needs),
                            )
                        )
                        for n in needs:
                            assumed[n] = None
                        factors[i : i + 2] = merged
                        changed = True
                    else:
                        key = (render_factors([a]), render_factors([b]))
                        if key not in seen_blocked:
                            seen_blocked.add(key)
                            out.blocked.append(
                                (
                                    "Merge exponentials into one",
                                    f"{_assumption(needs)} (not declared)",
                                    render_factors([a, b]),
                                )
                            )
                        i += 1
                elif isinstance(a, Exp) and isinstance(b, Sym):
                    ok, needs = _all_commute(a.names, (b.name,), comm)
                    if b.deriv and b.name in a.names and b.name not in comm.self_commuting:
                        ok = False  # A' does not commute with a function of A in general
                    if ok:  # a function of M commutes with M: put the plain matrix first
                        for n in needs:
                            assumed[n] = None
                        factors[i], factors[i + 1] = b, a
                        changed = True
                    else:
                        i += 1
                elif (
                    isinstance(a, Sym)
                    and isinstance(b, Sym)
                    and (a.name, a.deriv) > (b.name, b.deriv)
                    and a.name != b.name
                ):
                    if comm.sym_commutes(a, b):
                        assumed[_commute_text(a.name, b.name)] = None
                        factors[i], factors[i + 1] = b, a
                        changed = True
                    else:
                        i += 1
                else:
                    i += 1
        new_terms.append(Term(term.coeff, tuple(factors)))
    out.terms = collect(tuple(new_terms))
    out.assumptions = list(assumed)
    return out


def _all_commute(
    left: tuple[str, ...], right: tuple[str, ...], comm: Commutation
) -> tuple[bool, list[str]]:
    needs: dict[str, None] = {}
    ok = True
    for x in left:
        for y in right:
            if x == y:
                continue
            if comm.commutes(x, y):
                needs[_commute_text(x, y)] = None
            else:
                ok = False
                needs[_commute_text(x, y)] = None
    return ok, list(needs)


def _commute_text(x: str, y: str) -> str:
    x, y = sorted((x, y))
    return f"{x}{y} = {y}{x}"


def _assumption(needs: list[str]) -> str | None:
    return " and ".join(needs) if needs else None


def _merge_exp(a: Exp, b: Exp) -> list[Factor]:
    coeffs: dict[str, sp.Expr] = {}
    for name, c in (*a.arg, *b.arg):
        coeffs[name] = sp.expand(coeffs.get(name, sp.Integer(0)) + c)
    items = tuple(sorted((n, c) for n, c in coeffs.items() if c != 0))
    return [Exp(items)] if items else []


# -- rendering -------------------------------------------------------------------------------
def name_latex(name: str) -> str:
    return str(to_latex(sp.Symbol(name)))


def _coeff_parts(c: sp.Expr) -> tuple[int, str]:
    """Sign and the latex of |c| ("" when it is 1)."""
    sign = 1
    if c.could_extract_minus_sign():
        sign, c = -1, -c
    if c == 1:
        return sign, ""
    text = str(to_latex(c))
    return sign, f"\\left({text}\\right)" if isinstance(c, sp.Add) else text


def render_factors(factors: list[Factor]) -> str:
    parts: list[str] = []
    for f in factors:
        if isinstance(f, Sym):
            parts.append(_power_latex(_sym_latex(f), f.power, f.deriv))
        elif isinstance(f, Frechet):
            parts.append(render_frechet(f))
        else:
            parts.append(f"\\exp\\left({render_exp_arg(f)}\\right)")
    return " ".join(parts)


def _power_latex(base: str, power: int, deriv: int) -> str:
    if power == 1:
        return base
    return f"{{{base}}}^{{{power}}}" if deriv else f"{base}^{{{power}}}"


def _sym_latex(f: Sym) -> str:
    return name_latex(f.name) + "'" * f.deriv


def render_frechet(f: Frechet) -> str:
    x = render_exp_arg(f.base)
    wrapped = f"\\left({x}\\right)" if len(f.base.arg) > 1 else x
    direction = _join_signed(
        [(*_coeff_parts(c), name_latex(n) + "'" * d) for n, d, c in f.direction]
    )
    if len(f.direction) > 1:
        direction = f"\\left({direction}\\right)"
    return (
        f"\\int_0^1 \\exp\\left((1-\\sigma) {wrapped}\\right) {direction} "
        f"\\exp\\left(\\sigma {wrapped}\\right) d\\sigma"
    )


def render_exp_arg(f: Exp) -> str:
    return _join_signed([(*_coeff_parts(c), name_latex(n)) for n, c in f.arg])


def _join_signed(items: list[tuple[int, str, str]]) -> str:
    """items are (sign, coefficient latex, body latex)."""
    out = ""
    for k, (sign, coeff, body) in enumerate(items):
        piece = f"{coeff} {body}".strip()
        if k == 0:
            out += f"-{piece}" if sign < 0 else piece
        else:
            out += f" {'-' if sign < 0 else '+'} {piece}"
    return out


def render_terms(terms: Terms, names: dict[str, str] | None = None) -> str:
    """LaTeX of a sum of terms. `names` overrides the LaTeX of selected matrix names."""
    if not terms:
        return "0"
    items: list[tuple[int, str, str]] = []
    for t in terms:
        sign, coeff = _coeff_parts(t.coeff)
        body = _render_with_names(list(t.factors), names)
        items.append((sign, coeff, body if body else "I"))
    return _join_signed(items)


def _render_with_names(factors: list[Factor], names: dict[str, str] | None) -> str:
    if not names:
        return render_factors(factors)
    parts = []
    for f in factors:
        if isinstance(f, Sym) and f.name in names:
            parts.append(_power_latex(names[f.name] + "'" * f.deriv, f.power, f.deriv))
        else:
            parts.append(render_factors([f]))
    return " ".join(parts)


def substitute(terms: Terms, var: sp.Symbol, value: sp.Expr) -> Terms:
    """The terms with the scalar variable replaced; an exponential whose argument vanishes is the
    identity and is dropped."""
    out = []
    for t in terms:
        factors: list[Factor] = []
        for f in t.factors:
            if isinstance(f, Exp):
                arg = tuple(
                    (n, c.subs(var, value)) for n, c in f.arg if sp.expand(c.subs(var, value)) != 0
                )
                if arg:
                    factors.append(Exp(arg))
            else:
                factors.append(f)  # a plain matrix or a Frechet factor is left as it is
        out.append(Term(t.coeff.subs(var, value), tuple(factors)))
    return tuple(out)


def depends_on(f: Factor, var: sp.Symbol, dependent: frozenset[str] = frozenset()) -> bool:
    """Whether the factor depends on the variable (a time-dependent matrix counts)."""
    if isinstance(f, Sym):
        return f.name in dependent
    if isinstance(f, Frechet):
        return True
    return any(c.has(var) or n in dependent for n, c in f.arg)
