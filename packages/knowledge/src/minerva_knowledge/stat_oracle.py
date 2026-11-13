"""The statistics benchmark's own oracle: evaluates answer texts at concrete data.

It shares no code with the engine's parsers or checkers. An answer text (the engine's `result_text`
or a hand-written reference) is read with Python's `ast` module and evaluated, never executed
(`eval` is not used), at concrete numbers with mpmath:

* `Sum(f, (i, a, b))` and `Product(...)` are loops; `x[i]` indexes a data vector (1-based);
* matrices are mpmath matrices; `*` between matrices is the matrix product, `inv`, `transpose`,
  `tr`, `det`, `logdet` (= log |det|, which has the same derivative) and `diagpart` are available;
* `E[...]`, `Var(...)` and `Cov(a, b)` are evaluated exactly over a finite joint distribution that
  the oracle builds (a dependent one unless the case declares independence). An answer that
  silently assumes independence therefore disagrees with the reference on generic data.

Two answers are "equal" when they agree, to 25 digits, at every one of several random draws.
"""

from __future__ import annotations

import ast
import itertools
import random
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import mpmath as mp

DIGITS = 50
TOLERANCE = mp.mpf(10) ** -25
MAX_LOOP = 400
MAX_POWER = 10_000
DRAWS = 5
SIZES = (2, 3, 4, 5, 3)


class OracleError(Exception):
    """The text cannot be evaluated by the oracle (unsupported syntax, domain error, ...)."""


Number = Any  # mp.mpf | mp.mpc | mp.matrix | list


def _frac(f: Fraction) -> mp.mpf:
    return mp.mpf(f.numerator) / mp.mpf(f.denominator)


# --------------------------------------------------------------------------- evaluation


class _Pmf:
    """A finite joint distribution: outcomes assign values to the random names."""

    def __init__(self, outcomes: list[dict[str, Number]], weights: list[mp.mpf]) -> None:
        self.outcomes = outcomes
        self.weights = weights


class _Eval:
    def __init__(self, env: Mapping[str, Number], pmf: _Pmf | None) -> None:
        self.env = dict(env)
        self.pmf = pmf
        self.steps = 0

    # -- helpers
    def _tick(self) -> None:
        self.steps += 1
        if self.steps > 2_000_000:
            raise OracleError("evaluation too large")

    def visit(self, node: ast.AST, env: Mapping[str, Number]) -> Number:
        self._tick()
        match node:
            case ast.Expression(body=body):
                return self.visit(body, env)
            case ast.Constant(value=value):
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise OracleError("unsupported constant")
                return mp.mpf(value) if isinstance(value, float) else mp.mpf(int(value))
            case ast.Name(id=name):
                if name in env:
                    return env[name]
                if name == "pi":
                    return mp.pi
                if name == "E":
                    return mp.e
                raise OracleError(f"unbound name {name}")
            case ast.UnaryOp(op=op, operand=operand):
                value = self.visit(operand, env)
                if isinstance(op, ast.USub):
                    return -value
                if isinstance(op, ast.UAdd):
                    return value
                raise OracleError("unsupported unary operator")
            case ast.BinOp(left=left, op=op, right=right):
                return self._binary(self.visit(left, env), op, self.visit(right, env))
            case ast.Subscript(value=ast.Name(id="E"), slice=inner):
                return self._expectation(inner, env)
            case ast.Subscript(value=value, slice=index):
                vector = self.visit(value, env)
                position = self.visit(index, env)
                if not isinstance(vector, list):
                    raise OracleError("subscript of a non-vector")
                k = _as_int(position)
                if not 1 <= k <= len(vector):
                    raise OracleError("index out of range")
                return vector[k - 1]
            case ast.Call(func=ast.Name(id=name), args=args, keywords=[]):
                return self._call(name, args, env)
            case _:
                raise OracleError(f"unsupported syntax {type(node).__name__}")

    def _binary(self, a: Number, op: ast.operator, b: Number) -> Number:
        a_mat, b_mat = isinstance(a, mp.matrix), isinstance(b, mp.matrix)
        if isinstance(op, ast.Add | ast.Sub):
            if a_mat != b_mat:
                raise OracleError("matrix and scalar added")
            return a + b if isinstance(op, ast.Add) else a - b
        if isinstance(op, ast.Mult):
            if a_mat and b_mat and a.cols != b.rows:
                raise OracleError("shape mismatch in a product")
            product = a * b
            # a 1x1 product (a quadratic form, a trace-like block) is a scalar
            if isinstance(product, mp.matrix) and product.rows == product.cols == 1:
                return product[0, 0]
            return product
        if isinstance(op, ast.Div):
            if b_mat:
                raise OracleError("division by a matrix")
            if b == 0:
                raise OracleError("division by zero")
            return a / b
        if isinstance(op, ast.Pow):
            if b_mat:
                raise OracleError("matrix exponent")
            if abs(b) > MAX_POWER:
                raise OracleError("exponent too large")
            if a_mat:
                k = _as_int(b)
                if a.rows != a.cols:
                    raise OracleError("power of a non-square matrix")
                if k < 0:
                    return _inverse(a) ** (-k)
                return a**k
            if a == 0 and b < 0:
                raise OracleError("zero to a negative power")
            return mp.power(a, b)
        raise OracleError("unsupported operator")

    def _loop(self, args: list[ast.expr], env: Mapping[str, Number], product: bool) -> Number:
        """Sum or Product with one or more limits; as in SymPy the first limit is the innermost."""
        limits = args[1:]
        if not limits or not all(isinstance(g, ast.Tuple) and len(g.elts) == 3 for g in limits):
            raise OracleError("malformed Sum/Product")
        outer = limits[-1]
        assert isinstance(outer, ast.Tuple)  # noqa: S101 - checked above
        index, lo, hi = outer.elts
        if not isinstance(index, ast.Name):
            raise OracleError("the index must be a name")
        body = (
            ast.Call(
                func=ast.Name(id="Sum" if not product else "Product"),
                args=[args[0], *limits[:-1]],
                keywords=[],
            )
            if len(limits) > 1
            else args[0]
        )
        a, b = _as_int(self.visit(lo, env)), _as_int(self.visit(hi, env))
        if b - a + 1 > MAX_LOOP:
            raise OracleError("range too long")
        total: Number = mp.mpf(1) if product else mp.mpf(0)
        for k in range(a, b + 1):
            term = self.visit(body, {**env, index.id: mp.mpf(k)})
            total = total * term if product else total + term
        return total

    def _call(self, name: str, args: list[ast.expr], env: Mapping[str, Number]) -> Number:
        if name in ("Sum", "Product"):
            return self._loop(args, env, name == "Product")
        if name == "Var":
            if len(args) != 1:
                raise OracleError("Var takes one argument")
            return self._covariance(args[0], args[0], env)
        if name == "Cov":
            if len(args) != 2:
                raise OracleError("Cov takes two arguments")
            return self._covariance(args[0], args[1], env)
        values = [self.visit(a, env) for a in args]
        if name in ("exp", "log", "sqrt", "Abs", "abs", "gamma", "lgamma", "loggamma", "factorial"):
            if len(values) != 1 or isinstance(values[0], mp.matrix | list):
                raise OracleError(f"{name} takes one scalar")
            (x,) = values
            try:
                match name:
                    case "exp":
                        return mp.exp(x)
                    case "log":
                        if x == 0:
                            raise OracleError("log of zero")
                        return mp.log(x)
                    case "sqrt":
                        return mp.sqrt(x)
                    case "Abs" | "abs":
                        return abs(x)
                    case "gamma":
                        return mp.gamma(x)
                    case "lgamma" | "loggamma":
                        return mp.loggamma(x)
                    case _:
                        return mp.factorial(_as_int(x))
            except (ValueError, ZeroDivisionError) as exc:
                raise OracleError(f"{name}: {exc}") from exc
        if name in ("binomial", "beta"):
            if len(values) != 2 or any(isinstance(v, mp.matrix | list) for v in values):
                raise OracleError(f"{name} takes two scalars")
            return mp.binomial(*values) if name == "binomial" else mp.beta(*values)
        if len(values) == 1 and isinstance(values[0], mp.matrix):
            (m,) = values
            if name == "transpose":
                return m.T
            if m.rows != m.cols and name in ("inv", "tr", "det", "logdet", "diagpart"):
                raise OracleError(f"{name} of a non-square matrix")
            if name == "inv":
                return _inverse(m)
            if name == "tr":
                return sum((m[i, i] for i in range(m.rows)), mp.mpf(0))
            if name == "det":
                return mp.det(m)
            if name == "logdet":
                d = abs(mp.det(m))
                if d < mp.mpf(10) ** -15:
                    raise OracleError("log det of a singular matrix")
                return mp.log(d)
            if name == "diagpart":
                return mp.diag([m[i, i] for i in range(m.rows)])
        raise OracleError(f"unsupported function {name}")

    # -- expectations over the joint distribution
    def _joint(self) -> _Pmf:
        if self.pmf is None:
            raise OracleError("no random variables declared")
        return self.pmf

    def _outcomes(self, expr: ast.expr, env: Mapping[str, Number]) -> list[Number]:
        return [self.visit(expr, {**env, **outcome}) for outcome in self._joint().outcomes]

    def _mean(self, values: list[Number]) -> Number:
        weights = self._joint().weights
        return sum((w * v for w, v in zip(weights, values, strict=True)), mp.mpf(0))

    def _expectation(self, expr: ast.expr, env: Mapping[str, Number]) -> Number:
        return self._mean(self._outcomes(expr, env))

    def _covariance(self, a: ast.expr, b: ast.expr, env: Mapping[str, Number]) -> Number:
        va, vb = self._outcomes(a, env), self._outcomes(b, env)
        ea, eb = self._mean(va), self._mean(vb)
        return self._mean([(x - ea) * (y - eb) for x, y in zip(va, vb, strict=True)])


def _as_int(value: Number) -> int:
    if isinstance(value, mp.matrix | list):
        raise OracleError("an integer was expected")
    rounded = int(mp.nint(value))
    if abs(value - rounded) > mp.mpf(10) ** -20:
        raise OracleError("an integer was expected")
    return rounded


def _inverse(m: mp.matrix) -> mp.matrix:
    if abs(mp.det(m)) < mp.mpf(10) ** -15:
        raise OracleError("singular matrix")
    return m**-1


def evaluate(text: str, env: Mapping[str, Number], pmf: _Pmf | None = None) -> Number:
    """Evaluate an answer text. Raises OracleError if it cannot be read or evaluated."""
    if len(text) > 4000:
        raise OracleError("text too long")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise OracleError(f"cannot read {text!r}") from exc
    with mp.workdps(DIGITS):
        try:
            return _Eval(env, pmf).visit(tree, env)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError, RecursionError) as exc:
            raise OracleError(str(exc)) from exc


def close(a: Number, b: Number, tolerance: mp.mpf = TOLERANCE) -> bool:
    if isinstance(a, mp.matrix) != isinstance(b, mp.matrix):
        return False
    with mp.workdps(DIGITS):
        if isinstance(a, mp.matrix):
            if (a.rows, a.cols) != (b.rows, b.cols):
                return False
            return all(
                abs(a[i, j] - b[i, j]) <= tolerance * max(1, abs(b[i, j]))
                for i in range(a.rows)
                for j in range(a.cols)
            )
        if isinstance(a, list) or isinstance(b, list):
            return False
        return bool(abs(a - b) <= tolerance * max(1, abs(b)))


# --------------------------------------------------------------------------- drawing data

_SCALAR = ("real", "positive", "unit", "bit", "count")


def _scalar(kind: str, rng: random.Random) -> Fraction:
    match kind:
        case "real":
            return Fraction(rng.randint(-20, 20), 4)
        case "positive":
            return Fraction(rng.randint(1, 24), 4)
        case "unit":
            return Fraction(rng.randint(1, 10), 11)
        case "bit":
            return Fraction(rng.randint(0, 1))
        case "count":
            return Fraction(rng.randint(0, 6))
    raise OracleError(f"unknown variable kind {kind}")


def _matrix(rows: int, cols: int, rng: random.Random) -> mp.matrix:
    return mp.matrix(
        [[_frac(Fraction(rng.randint(-8, 8), 4)) for _ in range(cols)] for _ in range(rows)]
    )


def draw_environment(
    spec: Mapping[str, str],
    matrices: Sequence[Mapping[str, Any]],
    independent: Sequence[Sequence[str]],
    rng: random.Random,
    draw: int,
) -> tuple[dict[str, Number], _Pmf | None]:
    """Concrete values for every variable of a case; `draw` selects the size n."""
    env: dict[str, Number] = {}
    n = SIZES[draw % len(SIZES)]
    random_names: list[tuple[str, bool]] = []  # (name, indexed)
    with mp.workdps(DIGITS):
        for name, kind in spec.items():
            if kind == "size":
                env[name] = mp.mpf(n)
            elif kind in _SCALAR:
                env[name] = _frac(_scalar(kind, rng))
            elif kind.startswith("upto:"):
                bound = _as_int(env.get(kind[5:], mp.mpf(n)))
                env[name] = mp.mpf(rng.randint(0, bound))
            elif kind.startswith("vector_") and kind[7:] in _SCALAR:
                length = _as_int(env.get("n", mp.mpf(n)))
                env[name] = [_frac(_scalar(kind[7:], rng)) for _ in range(length)]
            elif kind == "random":
                random_names.append((name, False))
            elif kind == "random_vector":
                random_names.append((name, True))
            else:
                raise OracleError(f"unknown variable kind {kind}")
        for decl in matrices:
            env[decl["name"]] = _draw_declared(decl, rng)
        pmf = _joint(random_names, independent, _as_int(env.get("n", mp.mpf(n))), rng)
    return env, pmf


def _draw_declared(decl: Mapping[str, Any], rng: random.Random) -> mp.matrix:
    rows, cols = int(decl.get("rows", 1)), int(decl.get("cols", 1))
    if "entries" in decl:
        entries = decl["entries"]
        return mp.matrix([[evaluate(str(e), {}) for e in row] for row in entries])
    for _ in range(50):
        m = _matrix(rows, cols, rng)
        if decl.get("positive_definite"):
            m = m * m.T + mp.eye(rows)
        elif decl.get("symmetric"):
            m = (m + m.T) / 2
        if rows != cols or abs(mp.det(m)) > mp.mpf("0.2"):
            return m
    raise OracleError("could not draw a well-conditioned matrix")


def _joint(
    names: list[tuple[str, bool]], independent: Sequence[Sequence[str]], n: int, rng: random.Random
) -> _Pmf | None:
    if not names:
        return None
    atoms: list[tuple[str, int | None]] = []
    for name, indexed in names:
        atoms.extend([(name, k) for k in range(1, n + 1)] if indexed else [(name, None)])
    indexed_names = {name for name, indexed in names if indexed}
    # blocks: each declared group of whole variables is one block of its own; a group naming a
    # single indexed variable makes its components mutually independent; the rest may all depend.
    blocks: list[list[tuple[str, int | None]]] = []
    used: set[tuple[str, int | None]] = set()
    for group in independent:
        if len(group) == 1 and group[0] in indexed_names:
            for atom in atoms:
                if atom[0] == group[0]:
                    blocks.append([atom])
                    used.add(atom)
            continue
        for base in group:
            members = [a for a in atoms if a[0] == base]
            blocks.append(members)
            used.update(members)
    rest = [a for a in atoms if a not in used]
    if rest:
        blocks.append(rest)
    values = [[_frac(Fraction(rng.randint(-8, 8), 4)) for _ in (0, 1)] for _ in atoms]
    value_of = {atom: values[i] for i, atom in enumerate(atoms)}
    block_dists: list[list[tuple[dict[tuple[str, int | None], mp.mpf], mp.mpf]]] = []
    for block in blocks:
        weights = [Fraction(rng.randint(1, 9)) for _ in range(2 ** len(block))]
        total = sum(weights)
        dist = []
        for w, picks in zip(weights, itertools.product((0, 1), repeat=len(block)), strict=True):
            dist.append(
                (
                    {atom: value_of[atom][p] for atom, p in zip(block, picks, strict=True)},
                    _frac(w / total),
                )
            )
        block_dists.append(dist)
    outcomes: list[dict[str, Number]] = []
    weights_out: list[mp.mpf] = []
    for combo in itertools.product(*block_dists):
        assignment: dict[tuple[str, int | None], mp.mpf] = {}
        weight = mp.mpf(1)
        for part, p in combo:
            assignment.update(part)
            weight *= p
        outcome: dict[str, Number] = {}
        for name, indexed in names:
            outcome[name] = (
                [assignment[(name, k)] for k in range(1, n + 1)]
                if indexed
                else assignment[(name, None)]
            )
        outcomes.append(outcome)
        weights_out.append(weight)
    return _Pmf(outcomes, weights_out)


def environments(
    spec: Mapping[str, str],
    matrices: Sequence[Mapping[str, Any]] | None = None,
    independent: Sequence[Sequence[str]] | None = None,
    seed: int = 20260101,
    draws: int = DRAWS,
) -> list[tuple[dict[str, Number], _Pmf | None]]:
    rng = random.Random(seed)  # noqa: S311 - reproducible benchmark data
    return [draw_environment(spec, matrices or [], independent or [], rng, k) for k in range(draws)]


def equal_texts(
    a: str, b: str, envs: list[tuple[dict[str, Number], _Pmf | None]]
) -> tuple[bool, str]:
    """Do two answer texts agree at every draw? (flag, explanation)."""
    compared = 0
    for env, pmf in envs:
        try:
            expected = evaluate(b, env, pmf)
        except OracleError:
            continue  # the reference is undefined at this draw (a pole): skip it
        try:
            actual = evaluate(a, env, pmf)
        except OracleError as exc:
            return False, f"cannot evaluate {a!r}: {exc}"
        compared += 1
        if not close(actual, expected):
            return False, f"{a!r} differs from {b!r}"
    if compared < 3:
        return False, "the reference was defined at too few draws to compare"
    return True, ""


def numeric_gradient(
    text: str, env: Mapping[str, Number], name: str, step: float = 1e-12
) -> mp.matrix:
    """Central differences of a scalar text with respect to every entry of `env[name]`."""
    base = env[name]
    h = mp.mpf(step)
    grad = mp.zeros(base.rows, base.cols)
    with mp.workdps(DIGITS):
        for i in range(base.rows):
            for j in range(base.cols):
                grad[i, j] = _slope(text, env, name, _unit(base, i, j), h)
    return grad


def _unit(base: mp.matrix, i: int, j: int) -> mp.matrix:
    e = mp.zeros(base.rows, base.cols)
    e[i, j] = 1
    return e


def _slope(
    text: str, env: Mapping[str, Number], name: str, direction: mp.matrix, h: mp.mpf
) -> Number:
    plus = evaluate(text, {**env, name: env[name] + h * direction})
    minus = evaluate(text, {**env, name: env[name] - h * direction})
    return (plus - minus) / (2 * h)


def directional_derivative(
    text: str, env: Mapping[str, Number], name: str, direction: mp.matrix, step: float = 1e-12
) -> Number:
    """d/dt text(env[name] + t * direction) at t = 0 (central difference)."""
    with mp.workdps(DIGITS):
        return _slope(text, env, name, direction, mp.mpf(step))


def numeric_hessian(
    text: str, env: Mapping[str, Number], name: str, step: float = 1e-10
) -> mp.matrix:
    """Second differences of a scalar text in a vector variable (never uses a gradient formula)."""
    base = env[name]
    h = mp.mpf(step)
    size = base.rows * base.cols
    hess = mp.zeros(size, size)
    with mp.workdps(DIGITS):

        def at(di: mp.matrix) -> Number:
            return evaluate(text, {**env, name: base + di})

        units = [_unit(base, k, 0) if base.cols == 1 else _unit(base, 0, k) for k in range(size)]
        for i in range(size):
            for j in range(size):
                hess[i, j] = (
                    at(h * (units[i] + units[j]))
                    - at(h * (units[i] - units[j]))
                    - at(h * (units[j] - units[i]))
                    + at(-h * (units[i] + units[j]))
                ) / (4 * h * h)
    return hess


def numeric_jacobian(
    text: str, env: Mapping[str, Number], name: str, step: float = 1e-12
) -> mp.matrix:
    """Jacobian (rows: outputs, columns: inputs) of a vector text in a vector variable."""
    base = env[name]
    h = mp.mpf(step)
    with mp.workdps(DIGITS):
        outputs = evaluate(text, env)
        jac = mp.zeros(outputs.rows, base.rows)
        for k in range(base.rows):
            column = _slope(text, env, name, _unit(base, k, 0), h)
            for r in range(outputs.rows):
                jac[r, k] = column[r, 0]
    return jac


__all__ = [
    "OracleError",
    "close",
    "directional_derivative",
    "draw_environment",
    "environments",
    "equal_texts",
    "evaluate",
    "numeric_gradient",
    "numeric_hessian",
    "numeric_jacobian",
]
