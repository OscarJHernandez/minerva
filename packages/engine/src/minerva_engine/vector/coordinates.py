"""The three orthogonal coordinate systems, defined once.

Everything a system contributes is data: its coordinates, the map to Cartesian coordinates, the scale
factors, the orthonormal frame and the points where it degenerates. The exact lowering (`lower.py`) uses
the scale factors and the analytic frame; the independent numerical reading (`numeric.py`) uses only the
map to Cartesian coordinates and its inverse, and finds the frame by differencing that map.

Conventions (returned with every result): coordinates are real; `rho`, `r` are positive; vector fields
are given by physical (orthonormal) components along the unit vectors of the system, which form a
right-handed frame in the order listed; `theta` is the polar angle and `phi` the azimuth.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import mpmath as mp
import sympy as sp

SystemName = Literal["cartesian", "cylindrical", "spherical"]
SYSTEMS: tuple[SystemName, ...] = ("cartesian", "cylindrical", "spherical")
TIME = "t"


@dataclass(frozen=True)
class System:
    name: SystemName
    coords: tuple[str, str, str]
    latex: tuple[str, str, str]
    conventions: tuple[str, ...]
    restriction_latex: tuple[str, ...]

    # -- symbols -----------------------------------------------------------------------
    def assumptions(self) -> dict[str, list[str]]:
        """Assumptions on the coordinate symbols (the parsers build the same symbols)."""
        out = {c: ["real"] for c in self.coords}
        if self.name == "cylindrical":
            out["rho"] = ["positive"]
        if self.name == "spherical":
            out["r"] = ["positive"]
            out["theta"] = ["positive"]
        return out

    def symbols(self) -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
        a = self.assumptions()
        c = self.coords
        return (
            sp.Symbol(c[0], **{flag: True for flag in a[c[0]]}),
            sp.Symbol(c[1], **{flag: True for flag in a[c[1]]}),
            sp.Symbol(c[2], **{flag: True for flag in a[c[2]]}),
        )

    # -- analytic data for the exact route ---------------------------------------------
    def scale_factors(self) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
        q = self.symbols()
        if self.name == "cartesian":
            return sp.Integer(1), sp.Integer(1), sp.Integer(1)
        if self.name == "cylindrical":
            return sp.Integer(1), q[0], sp.Integer(1)
        return sp.Integer(1), q[0], q[0] * sp.sin(q[1])

    def to_cartesian(self) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
        q = self.symbols()
        if self.name == "cartesian":
            return q
        if self.name == "cylindrical":
            return q[0] * sp.cos(q[1]), q[0] * sp.sin(q[1]), q[2]
        r, th, ph = q
        return (
            r * sp.sin(th) * sp.cos(ph),
            r * sp.sin(th) * sp.sin(ph),
            r * sp.cos(th),
        )

    def frame(self) -> sp.Matrix:
        """Columns are the unit vectors, in Cartesian components (the analytic frame)."""
        q = self.symbols()
        if self.name == "cartesian":
            return sp.eye(3)
        if self.name == "cylindrical":
            _, ph, _ = q
            return sp.Matrix([[sp.cos(ph), -sp.sin(ph), 0], [sp.sin(ph), sp.cos(ph), 0], [0, 0, 1]])
        _, th, ph = q
        return sp.Matrix(
            [
                [sp.sin(th) * sp.cos(ph), sp.cos(th) * sp.cos(ph), -sp.sin(ph)],
                [sp.sin(th) * sp.sin(ph), sp.cos(th) * sp.sin(ph), sp.cos(ph)],
                [sp.cos(th), -sp.sin(th), 0],
            ]
        )

    def position_vector(self) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
        """The position vector in physical components of this system."""
        q = self.symbols()
        zero = sp.Integer(0)
        if self.name == "cartesian":
            return q
        if self.name == "cylindrical":
            return q[0], zero, q[2]
        return q[0], zero, zero

    def singular_conditions(self) -> list[sp.Expr]:
        """Expressions that must be nonzero for the frame and the operators to be defined."""
        q = self.symbols()
        if self.name == "cartesian":
            return []
        if self.name == "cylindrical":
            return [q[0]]
        return [q[0], sp.sin(q[1])]

    # -- numerical data for the independent route --------------------------------------
    def x_of_q(self) -> Callable[[Any, Any, Any], tuple[Any, Any, Any]]:
        """Map to Cartesian coordinates in mpmath arithmetic."""
        if self.name == "cartesian":
            return lambda a, b, c: (a, b, c)
        if self.name == "cylindrical":
            return lambda rho, ph, z: (rho * mp.cos(ph), rho * mp.sin(ph), z)
        return lambda r, th, ph: (
            r * mp.sin(th) * mp.cos(ph),
            r * mp.sin(th) * mp.sin(ph),
            r * mp.cos(th),
        )

    def q_of_x(self) -> Callable[[Any, Any, Any], tuple[Any, Any, Any]]:
        """Inverse map (principal values: phi in (-pi, pi], theta in [0, pi])."""
        if self.name == "cartesian":
            return lambda x, y, z: (x, y, z)
        if self.name == "cylindrical":
            return lambda x, y, z: (mp.sqrt(x * x + y * y), mp.atan2(y, x), z)
        return lambda x, y, z: (
            mp.sqrt(x * x + y * y + z * z),
            mp.acos(z / mp.sqrt(x * x + y * y + z * z)),
            mp.atan2(y, x),
        )

    def sample(self, rng: random.Random) -> tuple[Any, Any, Any]:
        """A point away from the axes and the origin, where the system is regular."""

        def u(lo: float, hi: float) -> Any:
            return mp.mpf(rng.uniform(lo, hi))

        if self.name == "cartesian":
            return u(-1.5, 1.5), u(-1.5, 1.5), u(-1.5, 1.5)
        if self.name == "cylindrical":
            return u(0.5, 2.0), u(-2.8, 2.8), u(-1.5, 1.5)
        return u(0.5, 2.0), u(0.4, 2.7), u(-2.8, 2.8)

    # -- names -------------------------------------------------------------------------
    def component_names(self, field: str) -> tuple[str, str, str]:
        return tuple(f"{field}_{c}" for c in self.coords)  # type: ignore[return-value]

    def unit_latex(self, index: int) -> str:
        return rf"\hat{{{self.latex[index]}}}"


_CARTESIAN = System(
    "cartesian",
    ("x", "y", "z"),
    ("x", "y", "z"),
    ("Cartesian coordinates (x, y, z); components along the unit vectors x, y, z.",),
    (),
)
_CYLINDRICAL = System(
    "cylindrical",
    ("rho", "phi", "z"),
    (r"\rho", r"\phi", "z"),
    (
        "Cylindrical coordinates (rho, phi, z), rho > 0, phi the azimuth; physical components "
        "along the right-handed unit vectors (rho, phi, z); scale factors (1, rho, 1).",
    ),
    (r"\rho \neq 0",),
)
_SPHERICAL = System(
    "spherical",
    ("r", "theta", "phi"),
    ("r", r"\theta", r"\phi"),
    (
        "Spherical coordinates (r, theta, phi), r > 0, theta the polar angle, phi the azimuth; "
        "physical components along the right-handed unit vectors (r, theta, phi); scale factors "
        "(1, r, r sin(theta)).",
    ),
    (r"r \neq 0", r"\sin\theta \neq 0"),
)
_BY_NAME: dict[str, System] = {s.name: s for s in (_CARTESIAN, _CYLINDRICAL, _SPHERICAL)}


def get_system(name: str) -> System:
    return _BY_NAME[name]


# Coordinates of one system written in the coordinates of another (the definitions of the change of
# coordinates; the conversion is verified numerically by an independent route, never trusted).
def source_in_target(source: System, target: System) -> dict[str, sp.Expr]:
    """`source` coordinates as expressions in the symbols of `target`."""
    t = target.symbols()
    x, y, z = target.to_cartesian()
    if source.name == "cartesian":
        return dict(zip(source.coords, (x, y, z), strict=True))
    if source.name == "cylindrical":
        if target.name == "cartesian":
            return {"rho": sp.sqrt(t[0] ** 2 + t[1] ** 2), "phi": sp.atan2(t[1], t[0]), "z": t[2]}
        r, th, ph = t  # target spherical
        return {"rho": r * sp.sin(th), "phi": ph, "z": r * sp.cos(th)}
    if target.name == "cartesian":
        rr = sp.sqrt(t[0] ** 2 + t[1] ** 2 + t[2] ** 2)
        return {"r": rr, "theta": sp.acos(t[2] / rr), "phi": sp.atan2(t[1], t[0])}
    rho, ph, zz = t  # target cylindrical
    return {"r": sp.sqrt(rho**2 + zz**2), "theta": sp.atan2(rho, zz), "phi": ph}
