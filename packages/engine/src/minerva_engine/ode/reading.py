"""Reading a differential equation: the only place where ODE notation is interpreted.

Derivative notation is rewritten, before parsing, into placeholder symbols (`Q_a2` is the second
derivative of the first unknown); the existing strict parsers then read everything else, so no other
operation changes. Nothing is guessed: the unknowns and the independent variable come from the
notation or from the user, and an input that leaves them open is a `parse_error` that says what to
declare. Delayed or composed arguments (`y(t-1)`, `y(2t)`), partial derivatives and operators applied
to expressions are refused as unsupported, never reinterpreted as products.

The result (`OdeProblem`) holds each equation as `left - right` over *jet symbols*: one plain symbol
per unknown and derivative order, so the algebra never meets SymPy's `Function`/`Derivative`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

import sympy as sp

from ..errors import LimitError, ParseError, UnsupportedInput
from ..latexparse import _GREEK, _GREEK_NAME
from ..latexprint import to_latex
from ..models import ALLOWED_ASSUMPTIONS, SYMBOL_NAME_MAX, Limits
from .models import MAX_ORDER, OdeCondition, OdeInput

_LETTERS = "abcdefgh"
_RESERVED = re.compile(r"Q_\{?[a-h]\d")
_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_DOTS = {"dot": 1, "ddot": 2, "dddot": 3, "ddddot": 4}

_GREEK_ALT = "|".join(sorted(_GREEK, key=len, reverse=True))
_SUB = r"(?:_\{\s*[A-Za-z0-9,]+\s*\}|_[A-Za-z0-9])"
_NAME = rf"(?:\\(?:{_GREEK_ALT})(?![A-Za-z])|[A-Za-z]){_SUB}?"
_D = r"(?:\\mathrm\{d\}|\\operatorname\{d\}|\\text\{d\}|d)"
_POW = r"\^\s*\{?\s*(\d+)\s*\}?"
_APPLY = r"(?:\s*\(\s*([^()]*?)\s*\))?"

_HEAD = re.compile(
    rf"""
    (?P<frac>\\[dt]?frac\s*\{{\s*{_D}\s*(?:\^\s*\{{?\s*(?P<fn>\d+)\s*\}}?)?\s*(?:\\,\s*)?
        (?P<fname>{_NAME})\s*\}}\s*\{{\s*{_D}\s*(?P<fvar>{_NAME})\s*(?:\^\s*\{{?\s*(?P<fd>\d+)\s*\}}?)?\s*\}})
        (?P<fapply>{_APPLY})
  | (?P<fop>\\[dt]?frac\s*\{{\s*{_D}\s*(?:\^\s*\{{?\s*(?P<on>\d+)\s*\}}?)?\s*\}}\s*\{{\s*{_D}\s*(?P<ovar>{_NAME})\s*
        (?:\^\s*\{{?\s*(?P<od>\d+)\s*\}}?)?\s*\}}\s*(?:\\,\s*)?(?:\(\s*(?P<opname>{_NAME})\s*\)|(?P<opname2>{_NAME})))
        (?P<oapply>{_APPLY})
  | (?P<opbad>\\[dt]?frac\s*\{{\s*{_D}\s*(?:\^\s*\{{?\s*\d+\s*\}}?)?\s*\}}\s*\{{)
  | \\(?P<dotcmd>d?dot|ddot|dddot|ddddot)\s*(?:\{{\s*(?P<dname>{_NAME})\s*\}}(?P<dsub>{_SUB})?|(?P<dname2>{_NAME}))
        (?P<dapply>{_APPLY})
  | (?P<name>{_NAME})
        (?:\s*\^\s*\{{\s*\(\s*(?P<pn>\d+)\s*\)\s*\}}|(?P<primes>(?:\s*'|\^\{{?(?:\\prime)+\}}?|\^\\prime)+))?
        (?P<napply>{_APPLY})
  | \\[A-Za-z]+
    """,
    re.VERBOSE,
)


@dataclass
class Head:
    """A derivative (or the unknown itself) found in the source: where, which, how high, and with
    respect to what (None when the notation does not say)."""

    start: int
    end: int
    name: str
    order: int
    variable: str | None
    argument: str | None  # text inside a following `(...)`, if any
    dotted: bool = False


def canonical(raw: str) -> str:
    """`x_{12}` -> `x_12`, `\\theta_1` -> `theta_1`, `\\varphi` -> `phi`."""
    raw = raw.strip()
    if raw.startswith("\\"):
        command = re.match(r"\\([A-Za-z]+)", raw)
        if command is None:
            raise ParseError("unreadable name")
        rest = raw[command.end() :]
        base = _GREEK_NAME.get(command.group(1), command.group(1))
    else:
        base, rest = raw[:1], raw[1:]
    sub = re.sub(r"[_{}\s]", "", rest)
    return f"{base}_{sub}" if sub else base


def _heads_latex(source: str) -> Iterator[Head]:
    if re.search(r"\\partial|\\nabla", source):
        raise UnsupportedInput(
            "partial derivatives are not supported: only ordinary differential equations"
        )
    for m in _HEAD.finditer(source):
        if m["opbad"]:
            raise UnsupportedInput(
                "d/dt written as an operator is supported only when applied to a single unknown, "
                "as in \\frac{d}{dt} y; write the derivative of the unknown instead"
            )
        if m["frac"]:
            order = int(m["fn"] or 1)
            if int(m["fd"] or 1) != order:
                raise ParseError(
                    "the order in the numerator and denominator of a derivative differ"
                )
            yield Head(
                m.start(),
                m.end(),
                canonical(m["fname"]),
                order,
                canonical(m["fvar"]),
                (m["fapply"] and m["fapply"].strip()[1:-1].strip()) or None,
            )
        elif m["fop"]:
            order = int(m["on"] or 1)
            if int(m["od"] or 1) != order:
                raise ParseError(
                    "the order in the numerator and denominator of a derivative differ"
                )
            name = canonical(m["opname"] or m["opname2"])
            yield Head(m.start(), m.end(), name, order, canonical(m["ovar"]), _arg(m["oapply"]))
        elif m["dotcmd"]:
            name_raw = m["dname"] or m["dname2"]
            name = canonical(name_raw + (m["dsub"] or "")) if m["dname"] else canonical(name_raw)
            yield Head(
                m.start(), m.end(), name, _DOTS[m["dotcmd"]], None, _arg(m["dapply"]), dotted=True
            )
        elif m["name"]:
            if m["pn"]:
                order = int(m["pn"])
            elif m["primes"]:
                order = m["primes"].count("'") + m["primes"].count("\\prime")
            else:
                order = 0
            yield Head(m.start(), m.end(), canonical(m["name"]), order, None, _arg(m["napply"]))


def _arg(group: str | None) -> str | None:
    if not group or not group.strip():
        return None
    inner = group.strip()
    return inner[1:-1].strip() if inner.startswith("(") else inner


_TEXT_NUMBER = re.compile(r"\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?")
_TEXT_CALL = re.compile(r"(?:diff|Derivative)\s*\(")


def _matching(text: str, open_at: int) -> int:
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ParseError("unbalanced parentheses")


def _heads_text(source: str) -> Iterator[Head]:
    i = 0
    while i < len(source):
        number = _TEXT_NUMBER.match(source, i)
        if number:
            i = number.end()
            continue
        call = _TEXT_CALL.match(source, i)
        if call:
            close = _matching(source, call.end() - 1)
            yield _text_diff(source[call.end() : close], i, close + 1)
            i = close + 1
            continue
        ident = _IDENT.match(source, i)
        if not ident:
            i += 1
            continue
        name, end = ident.group(), ident.end()
        order, dotted = 0, False
        for suffix, value in (("_ddot", 2), ("_dot", 1)):
            if name.endswith(suffix) and len(name) > len(suffix):
                name, order, dotted = name[: -len(suffix)], value, True
                break
        primes = re.compile(r"'+").match(source, end)
        if primes:
            order += len(primes.group())
            end = primes.end()
        argument = None
        look = re.compile(r"\s*\(").match(source, end)
        if look:
            close = _matching(source, look.end() - 1)
            argument = source[look.end() : close].strip()
            end = close + 1
        yield Head(i, end, name, order, None, argument, dotted)
        i = end


def _text_diff(inner: str, start: int, end: int) -> Head:
    parts, depth, current = [], 0, ""
    for ch in inner:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    parts.append(current.strip())
    target = re.fullmatch(rf"({_IDENT.pattern})\s*(?:\(\s*([^()]*?)\s*\))?", parts[0])
    if target is None or len(parts) < 2:
        raise UnsupportedInput(
            "diff(...) is supported only as diff(y, t) or diff(y, t, n) for a single unknown y"
        )
    variable = parts[1]
    if not _IDENT.fullmatch(variable):
        raise UnsupportedInput("diff(...) must differentiate with respect to one variable")
    order = 1
    rest = parts[2:]
    if len(rest) == 1 and rest[0].isdigit():
        order = int(rest[0])
    elif rest:
        if not all(item == variable for item in rest):
            raise UnsupportedInput("mixed or repeated-variable diff(...) is not supported")
        order = 1 + len(rest)
    return Head(start, end, target.group(1), order, variable, target.group(2) or None)


# --- reading a whole input ----------------------------------------------------------------


@dataclass
class OdeCondition2:
    unknown: str
    order: int
    at: sp.Expr
    value: sp.Expr


@dataclass
class OdeProblem:
    variable: sp.Symbol
    unknowns: list[str]
    orders: dict[str, int]
    jets: dict[tuple[str, int], sp.Symbol]
    sides: list[tuple[sp.Expr, sp.Expr]]
    parameters: list[sp.Symbol]
    conditions: list[OdeCondition2]
    display: dict[sp.Symbol, str]
    notes: list[str] = field(default_factory=list)

    @property
    def residuals(self) -> list[sp.Expr]:
        return [left - right for left, right in self.sides]

    def latex(self, expr: sp.Expr) -> str:
        return to_latex(expr, symbol_names=self.display)


def _split_once(text: str) -> tuple[str, str]:
    depth, found = 0, []
    for i, ch in enumerate(text):
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif ch == "=" and depth == 0:
            found.append(i)
    if len(found) != 1:
        raise ParseError("each equation needs exactly one equals sign: write left = right")
    left, right = text[: found[0]].strip(), text[found[0] + 1 :].strip()
    if not left or not right:
        raise ParseError("an equation has an empty side")
    return left, right


def _placeholder(fmt: str, index: int, order: int) -> str:
    return (
        f"{{Q_{{{_LETTERS[index]}{order}}}}}" if fmt == "latex" else f"Q_{_LETTERS[index]}{order}"
    )


@dataclass
class _Survey:
    unknowns: dict[str, int]
    variables: set[str]
    arguments: set[str]
    dotted: bool


def _heads(source: str, fmt: str) -> Iterator[Head]:
    return _heads_latex(source) if fmt == "latex" else _heads_text(source)


def _survey(sources: list[str], fmt: str, declared: list[str]) -> _Survey:
    """Pass 1: which names are differentiated (those are the unknowns), and with respect to what."""
    unknowns: dict[str, int] = dict.fromkeys(declared, 0)
    variables: set[str] = set()
    dotted = False
    for source in sources:
        for head in _heads(source, fmt):
            if head.order > 0 or head.dotted:
                unknowns.setdefault(head.name, 0)
                if head.variable:
                    variables.add(head.variable)
                dotted = dotted or head.dotted
    arguments: set[str] = set()
    for source in sources:
        for head in _heads(source, fmt):
            if head.name in unknowns and head.argument is not None:
                arguments.add(canonical(head.argument) if fmt == "latex" else head.argument)
    return _Survey(unknowns, variables, arguments, dotted)


def _rewrite(
    source: str, fmt: str, unknowns: list[str], variable: str, orders: dict[str, int]
) -> str:
    out, last = [], 0
    for head in _heads(source, fmt):
        if head.name not in unknowns:
            continue
        if head.variable is not None and head.variable != variable:
            raise ParseError(
                f"a derivative with respect to {head.variable}, but the independent variable is {variable}"
            )
        if head.argument is not None:
            given = canonical(head.argument) if fmt == "latex" else head.argument
            if given != variable or not re.fullmatch(r"[A-Za-z0-9_\\{} ]+", head.argument):
                raise UnsupportedInput(
                    f"{head.name}({head.argument}): only the unknown itself is supported "
                    f"({head.name}({variable})); delayed or composed arguments are not. "
                    f"If you meant a product, write {head.name}\\cdot({head.argument})"
                )
        orders[head.name] = max(orders.get(head.name, 0), head.order)
        out.append(source[last : head.start])
        out.append(_placeholder(fmt, unknowns.index(head.name), head.order))
        last = head.end
    out.append(source[last:])
    return "".join(out)


def _valid_name(name: str) -> str:
    if (
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
        or len(name) > SYMBOL_NAME_MAX
        or "__" in name
    ):
        raise ParseError(f"{name!r} is not a usable name")
    return name


def _tidy(name: str, fmt: str) -> str:
    return canonical(name) if fmt == "latex" and name.startswith("\\") else name


def _latex_name(name: str) -> str:
    base, _, sub = name.partition("_")
    head = f"\\{base}" if base in _GREEK else base
    return f"{head}_{{{sub}}}" if sub else head


def _display(name: str, order: int) -> str:
    base = _latex_name(name)
    if order == 0:
        return base
    if order <= 3:
        return base + "'" * order
    return f"{base}^{{({order})}}"


def read_ode(data: OdeInput, limits: Limits) -> OdeProblem:
    """Interpret `data`; raises a `MinervaMathError` with a message safe to show."""
    fmt = data.format
    for equation in data.equations:
        if len(equation) > limits.max_input_chars:
            raise LimitError("input is too long")
        if _RESERVED.search(equation):
            raise ParseError(
                "names of the form Q_a1 are reserved by the differential-equation reader"
            )
    declared = [_valid_name(_tidy(name, fmt)) for name in data.unknowns]
    survey = _survey(data.equations, fmt, declared)
    if not survey.unknowns:
        raise ParseError(
            "no unknown function found: differentiate the unknown (y', \\dot y, \\frac{dy}{dt}) "
            "or list it in `unknowns`"
        )
    if len(survey.unknowns) > 8:
        raise UnsupportedInput("at most 8 unknown functions are supported")
    unknowns = list(survey.unknowns)
    for name in unknowns:
        _valid_name(name)

    # The independent variable: declared, else the one the derivatives or the unknowns' arguments name.
    candidates = survey.variables | (survey.arguments if not survey.variables else set())
    if data.independent:
        variable = _valid_name(_tidy(data.independent, fmt))
    elif len(candidates) == 1:
        (variable,) = candidates
    elif len(candidates) > 1:
        raise ParseError("the derivatives name more than one variable: say which is independent")
    elif survey.dotted:
        variable = "t"
    else:
        raise ParseError(
            "say which variable is independent (for example t): the notation y' does not name it"
        )
    _valid_name(variable)
    if variable in unknowns:
        raise ParseError("the independent variable cannot also be an unknown")

    orders: dict[str, int] = dict.fromkeys(unknowns, 0)
    rewritten = [_rewrite(item, fmt, unknowns, variable, orders) for item in data.equations]
    for name in unknowns:
        if orders[name] > MAX_ORDER:
            raise UnsupportedInput(f"derivatives above order {MAX_ORDER} are not supported")

    assumptions = {k: list(v) for k, v in data.assumptions.items() if k not in unknowns}
    if variable not in assumptions:
        assumptions[variable] = ["real"]

    sides: list[tuple[sp.Expr, sp.Expr]] = []
    for item in rewritten:
        left_text, right_text = _split_once(item)
        sides.append(
            (
                _parse(left_text, fmt, assumptions, limits),
                _parse(right_text, fmt, assumptions, limits),
            )
        )

    # Jet symbols: the unknown and each derivative, plain symbols.
    jets: dict[tuple[str, int], sp.Symbol] = {}
    display: dict[sp.Symbol, str] = {}
    for name in unknowns:
        own = [a for a in data.assumptions.get(name, []) if a in ALLOWED_ASSUMPTIONS]
        real = bool(set(own) & {"real", "positive", "negative", "nonnegative", "nonpositive"})
        for order in range(orders[name] + 1):
            kwargs = {a: True for a in own} if order == 0 else ({"real": True} if real else {})
            symbol = sp.Symbol(name if order == 0 else f"{name}_d{order}", **kwargs)
            jets[(name, order)] = symbol
            display[symbol] = _display(name, order)
    mapping = {
        sp.Symbol(f"Q_{_LETTERS[index]}{order}"): jets[(name, order)]
        for index, name in enumerate(unknowns)
        for order in range(orders[name] + 1)
    }
    # A symbol the user wrote that has the name of a jet would silently become that derivative.
    written = {sym.name for pair in sides for e in pair for sym in e.free_symbols}
    clash = sorted(written & {sym.name for sym in jets.values()})
    if clash:
        raise ParseError(f"{clash[0]!r} is reserved for an unknown or one of its derivatives")
    sides = [(a.xreplace(mapping), b.xreplace(mapping)) for a, b in sides]

    var_symbol = next(
        (s for pair in sides for e in pair for s in e.free_symbols if s.name == variable),
        sp.Symbol(variable, real=True),
    )
    free = {s for pair in sides for e in pair for s in e.free_symbols}
    parameters = sorted((free - set(jets.values()) - {var_symbol}), key=lambda s: s.name)
    problem = OdeProblem(
        variable=var_symbol,
        unknowns=unknowns,
        orders=orders,
        jets=jets,
        sides=sides,
        parameters=parameters,
        conditions=[],
        display=display,
    )
    if not data.independent and not survey.variables and survey.arguments:
        problem.notes.append(
            f"the independent variable {variable} was read from the argument of the unknown"
        )
    if survey.dotted and variable == "t" and not data.independent and not survey.variables:
        problem.notes.append("a dot over an unknown is a derivative with respect to t")
    if data.conditions:
        problem.conditions = [
            _read_condition(item, fmt, unknowns, variable, assumptions, limits)
            for item in data.conditions
        ]
        for c in problem.conditions:
            if c.order > MAX_ORDER:
                raise UnsupportedInput("a condition on a derivative above the supported order")
            # A symbol that only a condition mentions (y(0) = y_0) is a parameter too.
            for symbol in c.at.free_symbols | c.value.free_symbols:
                if symbol.name == variable:
                    raise ParseError("a condition's point and value cannot contain the variable")
                if symbol.name in {s.name for s in jets.values()}:
                    raise ParseError(f"{symbol.name!r} is reserved for an unknown")
                if symbol not in problem.parameters:
                    problem.parameters.append(symbol)
        problem.parameters.sort(key=lambda s: s.name)
    return problem


def _parse(text: str, fmt: str, assumptions: dict[str, list[str]], limits: Limits) -> sp.Expr:
    from ..latexparse import parse_latex
    from ..textparse import parse_text

    if fmt == "latex":
        expr = parse_latex(text, assumptions, limits, strict_calls=True)
    else:
        expr = parse_text(text, assumptions, limits, strict_calls=True)
    return expr


def _read_condition(
    item: OdeCondition | str,
    fmt: str,
    unknowns: list[str],
    variable: str,
    assumptions: dict[str, list[str]],
    limits: Limits,
) -> OdeCondition2:
    if isinstance(item, str):
        left, right = _split_once(item)
        heads = [h for h in _heads(left, fmt) if h.name in unknowns]
        if len(heads) != 1 or heads[0].start != 0 or heads[0].end != len(left.strip()):
            raise ParseError(
                "a condition is written like y(0) = 1, y'(0) = 0 or \\dot y(0) = 0: one unknown "
                "or derivative at a point"
            )
        head = heads[0]
        if head.argument is None:
            raise ParseError("a condition needs the point it holds at, as in y(0) = 1")
        unknown, order, at_text, value_text = head.name, head.order, head.argument, right
    else:
        unknown = _tidy(item.unknown, fmt)
        order, at_text, value_text = item.order, item.at, item.value
        if unknown not in unknowns:
            raise ParseError(f"a condition refers to {item.unknown!r}, which is not an unknown")
    if unknown not in unknowns:
        raise ParseError(f"a condition refers to {unknown!r}, which is not an unknown")
    at = _parse(at_text, fmt, assumptions, limits)
    value = _parse(value_text, fmt, assumptions, limits)
    return OdeCondition2(unknown, order, at, value)


__all__ = ["Head", "OdeProblem", "canonical", "read_ode"]
