"""Physical dimensions: reading units, and dimensional-consistency checking of equations.

A dimension is an exponent vector over the SI base units (kg, m, s, A, K, mol, cd) with exact
rational entries. Units are read from a small fixed vocabulary: the base units, the coherent SI
derived units (`DERIVED_UNITS`) and, for whole quantities, named kinds (`QUANTITY_KINDS`, e.g.
"dynamic viscosity"). This is not a unit converter: prefixed or non-coherent units (g, cm, eV, min)
are rejected rather than converted.

`check_equation_dimensions` verifies that both sides of an equation have the same dimensions, that
all terms of every sum agree, and that arguments of transcendental functions are dimensionless. This
catches dimensional mistakes; it does not prove an equation is physically correct.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy as sp

from .errors import MinervaMathError, ParseError
from .textparse import parse_text

BASE_UNITS = ("kg", "m", "s", "A", "K", "mol", "cd")

Dim = tuple[Fraction, ...]
ZERO: Dim = (Fraction(0),) * len(BASE_UNITS)


class DimensionError(MinervaMathError):
    """Dimensions are inconsistent.

    `kind` says which rule was broken (`sum`, `function`, `exponent`, `power`, `sides`). `parts`
    holds the offending expressions with their dimensions, for reports.
    """

    def __init__(
        self,
        message: str,
        kind: str = "sides",
        parts: list[tuple[sp.Expr, Dim]] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.parts = parts or []


class UnsupportedDimensionConstruct(MinervaMathError):
    """The expression uses a construct whose dimension this module does not determine."""


# --- vector arithmetic ---------------------------------------------------------------------
def dim_mul(a: Dim, b: Dim) -> Dim:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def dim_div(a: Dim, b: Dim) -> Dim:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def dim_pow(a: Dim, exponent: Fraction) -> Dim:
    return tuple(x * exponent for x in a)


def _base(unit: str) -> Dim:
    return tuple(Fraction(1 if u == unit else 0) for u in BASE_UNITS)


def _fraction(value: sp.Expr) -> Fraction:
    if isinstance(value, sp.Rational):
        return Fraction(int(value.p), int(value.q))
    raise ParseError("exponents in units and dimensions must be rational numbers")


# --- the unit vocabulary ---------------------------------------------------------------------
# Coherent SI derived units, as base-unit expressions. Original table of definitions.
DERIVED_UNITS: dict[str, str] = {
    "N": "kg*m/s**2",
    "J": "kg*m**2/s**2",
    "W": "kg*m**2/s**3",
    "Pa": "kg/(m*s**2)",
    "Hz": "1/s",
    "C": "A*s",
    "V": "kg*m**2/(A*s**3)",
    "F": "A**2*s**4/(kg*m**2)",
    "ohm": "kg*m**2/(A**2*s**3)",
    "S": "A**2*s**3/(kg*m**2)",
    "Wb": "kg*m**2/(A*s**2)",
    "T": "kg/(A*s**2)",
    "H": "kg*m**2/(A**2*s**2)",
    "rad": "1",
}

# Named quantity kinds, as unit expressions (base or derived units). Lower case, single spaces.
QUANTITY_KINDS: dict[str, str] = {
    "dimensionless": "1",
    "angle": "rad",
    "length": "m",
    "distance": "m",
    "radius": "m",
    "diameter": "m",
    "height": "m",
    "width": "m",
    "thickness": "m",
    "wavelength": "m",
    "area": "m**2",
    "volume": "m**3",
    "mass": "kg",
    "time": "s",
    "period": "s",
    "frequency": "Hz",
    "angular frequency": "rad/s",
    "velocity": "m/s",
    "speed": "m/s",
    "acceleration": "m/s**2",
    "force": "N",
    "weight": "N",
    "momentum": "kg*m/s",
    "angular momentum": "kg*m**2/s",
    "moment of inertia": "kg*m**2",
    "torque": "N*m",
    "energy": "J",
    "work": "J",
    "heat": "J",
    "power": "W",
    "pressure": "Pa",
    "stress": "Pa",
    "density": "kg/m**3",
    "mass flow rate": "kg/s",
    "volumetric flow rate": "m**3/s",
    "dynamic viscosity": "Pa*s",
    "kinematic viscosity": "m**2/s",
    "diffusivity": "m**2/s",
    "surface tension": "N/m",
    "temperature": "K",
    "heat capacity": "J/K",
    "specific heat capacity": "J/(kg*K)",
    "thermal conductivity": "W/(m*K)",
    "heat transfer coefficient": "W/(m**2*K)",
    "amount of substance": "mol",
    "luminous intensity": "cd",
    "current": "A",
    "charge": "C",
    "voltage": "V",
    "electric field": "V/m",
    "magnetic field": "T",
    "resistance": "ohm",
    "capacitance": "F",
    "inductance": "H",
}

_KIND_ALIASES = {"velocities": "velocity", "gravitational acceleration": "acceleration"}
_NOT_COHERENT = frozenset(
    {"g", "cm", "mm", "km", "L", "l", "min", "h", "eV", "bar", "atm", "deg", "degC", "ms", "us"}
)
_MAX_UNIT_LENGTH = 200
_JUXTAPOSITION = re.compile(r"(?<=[\w)])\s+(?=[\w(])")


def _expr_dim(expr: sp.Expr) -> Dim:
    """Dimension of a parsed unit expression: products, rational powers, unit names and 1."""
    if isinstance(expr, sp.Symbol):
        return _unit_symbol(expr.name)
    if isinstance(expr, sp.Number):
        if expr == 1:
            return ZERO
        raise ParseError("numeric factors are not allowed in a unit")
    if isinstance(expr, sp.Mul):
        result = ZERO
        for factor in expr.args:
            result = dim_mul(result, _expr_dim(factor))
        return result
    if isinstance(expr, sp.Pow):
        return dim_pow(_expr_dim(expr.base), _fraction(expr.exp))
    raise ParseError("a unit is a product of unit names with rational powers")


_derived_cache: dict[str, Dim] = {}


def _unit_symbol(name: str) -> Dim:
    if name in BASE_UNITS:
        return _base(name)
    if name in DERIVED_UNITS:
        if name not in _derived_cache:
            _derived_cache[name] = _expr_dim(parse_text(DERIVED_UNITS[name]))
        return _derived_cache[name]
    hint = (
        " (only coherent SI units are accepted; convert to SI first)"
        if name in _NOT_COHERENT
        else ""
    )
    raise ParseError(
        f"unknown unit {name!r}{hint}; use SI base units ({', '.join(BASE_UNITS)}), "
        f"derived units ({', '.join(DERIVED_UNITS)}) or a quantity kind"
    )


def read_unit(text: str) -> Dim:
    """Dimension vector of a unit expression such as `kg*m/s**2`, `J s` or `m^3/(kg s^2)`."""
    if len(text) > _MAX_UNIT_LENGTH:
        raise ParseError("unit text is too long")
    source = _JUXTAPOSITION.sub("*", text.strip())
    if not source:
        raise ParseError("empty unit")
    return _expr_dim(parse_text(source))


def read_kind(kind: str) -> Dim:
    """Dimension vector of a named quantity kind such as `dynamic viscosity`."""
    key = " ".join(kind.lower().replace("-", " ").replace("_", " ").split())
    key = _KIND_ALIASES.get(key, key)
    if key not in QUANTITY_KINDS:
        raise ParseError(
            f"unknown quantity kind {kind!r}; give the unit instead, or use one of: "
            + ", ".join(sorted(QUANTITY_KINDS))
        )
    return read_unit(QUANTITY_KINDS[key])


def dim_text(d: Dim) -> str:
    """Plain text such as `kg m^-1 s^-2`; `1` for a dimensionless quantity."""
    parts = []
    for unit, e in zip(BASE_UNITS, d, strict=True):
        if e == 0:
            continue
        parts.append(unit if e == 1 else f"{unit}^{_exp_text(e)}")
    return " ".join(parts) or "1"


def _exp_text(e: Fraction) -> str:
    return str(e.numerator) if e.denominator == 1 else f"({e.numerator}/{e.denominator})"


def dim_latex(d: Dim) -> str:
    parts = []
    for unit, e in zip(BASE_UNITS, d, strict=True):
        if e == 0:
            continue
        if e == 1:
            parts.append(unit)
        elif e.denominator == 1:
            parts.append(f"{unit}^{{{e.numerator}}}")
        else:
            parts.append(f"{unit}^{{{e.numerator}/{e.denominator}}}")
    return r"\mathrm{" + r"\,".join(parts) + "}" if parts else "1"


# --- dimensions of expressions -----------------------------------------------------------------
def dimension_of(expr: sp.Expr, units: dict[str, Dim]) -> Dim:
    """Dimension of `expr` given the dimension of each symbol (by name).

    Raises `DimensionError` for a sum of unlike terms, a non-dimensionless function argument or
    exponent; `ParseError` for a symbol without a unit; `UnsupportedDimensionConstruct` for anything
    else (derivatives, integrals, ...).
    """
    if expr.is_number:
        return ZERO
    if isinstance(expr, sp.Symbol):
        if expr.name not in units:
            raise ParseError(f"no unit given for symbol {expr.name}")
        return units[expr.name]
    if isinstance(expr, sp.Add):
        dims = [(a, dimension_of(a, units)) for a in expr.args]
        for _, other in dims[1:]:
            if other != dims[0][1]:
                raise DimensionError("terms of a sum have different dimensions", "sum", dims)
        return dims[0][1]
    if isinstance(expr, sp.Mul):
        result = ZERO
        for a in expr.args:
            result = dim_mul(result, dimension_of(a, units))
        return result
    if isinstance(expr, sp.Pow):
        base = dimension_of(expr.base, units)
        if not expr.exp.is_number:
            if dimension_of(expr.exp, units) != ZERO:
                raise DimensionError(
                    "exponent must be dimensionless",
                    "exponent",
                    [(expr.exp, dimension_of(expr.exp, units))],
                )
            if base != ZERO:
                raise DimensionError(
                    "symbolic power of a dimensional quantity", "power", [(expr.base, base)]
                )
            return ZERO
        return dim_pow(base, _fraction(expr.exp))
    if isinstance(expr, sp.Abs):
        return dimension_of(expr.args[0], units)
    if isinstance(expr, sp.Function) and len(expr.args) == 1:
        arg = dimension_of(expr.args[0], units)
        if arg != ZERO:
            raise DimensionError(
                f"argument of {expr.func.__name__} must be dimensionless",
                "function",
                [(expr.args[0], arg)],
            )
        return ZERO
    raise UnsupportedDimensionConstruct(
        f"unsupported construct in dimensional analysis: {type(expr).__name__}"
    )


def check_equation_dimensions(lhs: str, rhs: str, units: dict[str, str]) -> None:
    """Raise `DimensionError` (or `ParseError`) unless `lhs` and `rhs` agree dimensionally."""
    unit_map = {name: read_unit(text) for name, text in units.items()}
    try:
        left = dimension_of(parse_text(lhs), unit_map)
        right = dimension_of(parse_text(rhs), unit_map)
    except UnsupportedDimensionConstruct as exc:
        raise ParseError(str(exc)) from exc
    if left != right:
        raise DimensionError("the two sides have different dimensions")
