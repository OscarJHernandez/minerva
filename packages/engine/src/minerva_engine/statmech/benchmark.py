"""Statistical-mechanics problems whose answers come from outside Minerva.

Every expected value is computed here with plain arithmetic (`math`, `mpmath`) from the textbook closed
form of the system, never from Minerva's own derivation route or its numerical oracle. A problem expects
either a **correct** result (every quantity it names must agree with the reference to the stated
tolerance) or a **refusal** (the engine must return no result and the enumerated reason the case names).

    python -m minerva_engine.statmech.benchmark [--check]

writes `docs/benchmarks/statmech.json` and `docs/statmech-benchmark.md`; `--check` compares the
deterministic part with the committed JSON and fails on any difference or any wrong answer, which is what
CI does through `tests/test_statmech_benchmark.py`. Latencies are recorded and never gated.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mpmath as mp

from ..models import Limits, Status
from ..service import Engine
from ..textparse import parse_text
from .models import StatMechInput, StatMechResult

Judge = Callable[[StatMechResult], tuple[bool, str]]


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    source: str  # how the reference is obtained, by description only
    request: dict[str, Any]
    expected: str = "correct"  # "correct" or "refused"
    reason: str | None = None  # the enumerated reason a refusal must carry
    judge: Judge | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def value(result: StatMechResult, name: str, **values: float) -> float:
    """A returned formula evaluated at numbers (through Minerva's own safe text parser)."""
    q = next(q for q in result.quantities if q.name == name)
    expr = parse_text(q.text)
    subs = {s: values[s.name] for s in expr.free_symbols if s.name in values}
    return float(expr.subs(subs).evalf(30))


def agrees(pairs: list[tuple[str, float, float]], tol: float = 1e-9) -> tuple[bool, str]:
    bad = [
        f"{label}: got {got:.12g}, expected {want:.12g}"
        for label, got, want in pairs
        if not math.isclose(got, want, rel_tol=tol, abs_tol=1e-12)
    ]
    return (not bad, "; ".join(bad) or f"{len(pairs)} values agree")


# -- references: textbook closed forms, plain arithmetic ---------------------------------------
B, EPS, KB = 0.7, 1.3, 2.0
X = B * EPS


def _two_level(r: StatMechResult) -> tuple[bool, str]:
    z = 1 + math.exp(-X)
    u = EPS * math.exp(-X) / z
    var = EPS**2 * math.exp(-X) / z**2
    s = KB * (math.log(z) + B * u)
    kw = {"beta": B, "epsilon": EPS, "k_B": KB}
    return agrees(
        [
            ("Z", value(r, "Z", **kw), z),
            ("F", value(r, "F", **kw), -math.log(z) / B),
            ("U", value(r, "U", **kw), u),
            ("S", value(r, "S", **kw), s),
            ("Var_E", value(r, "Var_E", **kw), var),
        ]
    )


def _schottky(r: StatMechResult) -> tuple[bool, str]:
    c = KB * X**2 * math.exp(X) / (1 + math.exp(X)) ** 2
    return agrees([("C", value(r, "C", beta=B, epsilon=EPS, k_B=KB), c)])


def _spin(r: StatMechResult) -> tuple[bool, str]:
    b, mu, h = 0.8, 1.1, 0.6
    kw = {"beta": b, "mu": mu, "h": h}
    return agrees(
        [
            ("mean M", value(r, "mean_M", **kw), mu * math.tanh(b * mu * h)),
        ]
    )


def _spin_variance(r: StatMechResult) -> tuple[bool, str]:
    b, mu, h = 0.8, 1.1, 0.6
    return agrees(
        [("Var M", value(r, "variance_M", beta=b, mu=mu, h=h), mu**2 / math.cosh(b * mu * h) ** 2)]
    )


def _finite_states(r: StatMechResult) -> tuple[bool, str]:
    # levels (0, g=1), (1/2, g=3), (2, g=2) at beta = 0.9
    b = 0.9
    levels = [(0.0, 1), (0.5, 3), (2.0, 2)]
    z = sum(g * math.exp(-b * e) for e, g in levels)
    u = sum(g * e * math.exp(-b * e) for e, g in levels) / z
    var = sum(g * e * e * math.exp(-b * e) for e, g in levels) / z - u * u
    return agrees(
        [
            ("Z", value(r, "Z", beta=b), z),
            ("U", value(r, "U", beta=b), u),
            ("Var_E", value(r, "Var_E", beta=b), var),
        ]
    )


def _qho(r: StatMechResult) -> tuple[bool, str]:
    hw = EPS
    kw = {"beta": B, "hbar": hw, "omega": 1.0, "k_B": KB}
    return agrees(
        [
            ("Z", value(r, "Z", **kw), 1 / (2 * math.sinh(X / 2))),
            ("U", value(r, "U", **kw), hw / (2 * math.tanh(X / 2))),
            ("C", value(r, "C", **kw), KB * X**2 / (4 * math.sinh(X / 2) ** 2)),
        ]
    )


def _classical_oscillator(r: StatMechResult) -> tuple[bool, str]:
    b, m, w, h = 0.9, 1.2, 0.7, 2.0
    kw = {"beta": b, "m": m, "omega": w, "h": h}
    return agrees(
        [
            ("Z", value(r, "Z", **kw), 2 * math.pi / (h * w * b)),
            ("U equipartition", value(r, "U", **kw), 1 / b),
            ("Var_E", value(r, "Var_E", **kw), 1 / b**2),
        ]
    )


def _ideal_gas(r: StatMechResult) -> tuple[bool, str]:
    b, m, h, v, n, kb = 0.8, 1.3, 1.1, 2.5, 5, 1.7
    kw = {"beta": b, "m": m, "h": h, "V": v, "N": n, "k_B": kb}
    entropy = kb * (
        n * math.log(v * (2 * math.pi * m / b) ** 1.5 / h**3) - math.lgamma(n + 1) + 1.5 * n
    )
    return agrees(
        [
            ("U = 3NkT/2", value(r, "U", **kw), 1.5 * n / b),
            ("p V = N k T", value(r, "p", **kw) * v * b, float(n)),
            ("S, exact in N!", value(r, "S", **kw), entropy),
        ]
    )


def _dos_3d(r: StatMechResult) -> tuple[bool, str]:
    b = 0.8
    return agrees(
        [
            ("Z", value(r, "Z", beta=b), math.gamma(1.5) / b**1.5),
            ("U", value(r, "U", beta=b), 1.5 / b),
            ("Var_E", value(r, "Var_E", beta=b), 1.5 / b**2),
        ]
    )


def _dos_constant_cumulant(r: StatMechResult) -> tuple[bool, str]:
    b = 0.5
    return agrees([("kappa_3", value(r, "cumulant_E", beta=b), 2 / b**3)])


def _tilted_spin(r: StatMechResult) -> tuple[bool, str]:
    b, h, d, kb = 0.8, 0.6, 0.9, 1.5
    w = math.hypot(h, d)
    x = b * w
    kw = {"beta": b, "h": h, "d": d, "k_B": kb}
    return agrees(
        [
            ("Z", value(r, "Z", **kw), 2 * math.cosh(x)),
            ("U", value(r, "U", **kw), -w * math.tanh(x)),
            ("S", value(r, "S", **kw), kb * (math.log(2 * math.cosh(x)) - x * math.tanh(x))),
        ]
    )


def _tilted_spin_mean(r: StatMechResult) -> tuple[bool, str]:
    b, h, d = 0.8, 0.6, 0.9
    w = math.hypot(h, d)
    return agrees([("<sz>", value(r, "mean_sz", beta=b, h=h, d=d), h * math.tanh(b * w) / w)])


def _grand(statistics: str) -> Judge:
    """N, U, S and Var(N) of non-interacting particles, from the per-state occupation formulas."""
    b, mu, eps, kb = 0.8, -0.3, 1.1, 1.7
    states = [(0.0, 1), (eps, 1), (2 * eps, 2)]

    def judge(r: StatMechResult) -> tuple[bool, str]:
        n = {}
        for e, _ in states:
            x = b * (e - mu)
            n[e] = {
                "fermi": 1 / (math.exp(x) + 1),
                "bose": 1 / (math.exp(x) - 1),
                "boltzmann": math.exp(-x),
            }[statistics]
        total = sum(g * n[e] for e, g in states)
        if statistics == "fermi":
            var = sum(g * n[e] * (1 - n[e]) for e, g in states)
            ent = -sum(
                g * (n[e] * math.log(n[e]) + (1 - n[e]) * math.log(1 - n[e])) for e, g in states
            )
        elif statistics == "bose":
            var = sum(g * n[e] * (1 + n[e]) for e, g in states)
            ent = sum(
                g * ((1 + n[e]) * math.log(1 + n[e]) - n[e] * math.log(n[e])) for e, g in states
            )
        else:
            var = total
            ent = sum(g * n[e] * (1 - math.log(n[e])) for e, g in states)
        kw = {"beta": b, "mu": mu, "eps": eps, "k_B": kb}
        return agrees(
            [
                ("N", value(r, "N", **kw), total),
                ("U", value(r, "U", **kw), sum(g * e * n[e] for e, g in states)),
                ("Var N", value(r, "Var_N", **kw), var),
                ("S", value(r, "S", **kw), kb * ent),
            ]
        )

    return judge


def _binomial(n: int, k: int) -> Judge:
    def judge(r: StatMechResult) -> tuple[bool, str]:
        q = next(q for q in r.quantities if q.name == "Omega")
        s_q = next(q for q in r.quantities if q.name == "S")
        ok = q.text == str(math.comb(n, k)) and s_q.text == f"k_B*log({math.comb(n, k)})"
        return ok, f"Omega = {q.text}, expected C({n}, {k}) = {math.comb(n, k)}"

    return judge


def _ising_ring(n: int, beta: float, j: float, h: float) -> float:
    """Brute-force partition function of a ring of n spins, H = -J sum s s' - h sum s."""
    total = 0.0
    for spins in itertools.product((1, -1), repeat=n):
        energy = -h * sum(spins) - j * sum(spins[i] * spins[(i + 1) % n] for i in range(n))
        total += math.exp(-beta * energy)
    return total


def _ising(r: StatMechResult) -> tuple[bool, str]:
    b, j, h, n = 0.6, 0.9, 0.4, 7
    return agrees(
        [
            (
                "Z (brute force over 2^7 spin configurations)",
                value(r, "Z", beta=b, J=j, h=h, N=n),
                _ising_ring(n, b, j, h),
            )
        ]
    )


def _ising_zero_field(r: StatMechResult) -> tuple[bool, str]:
    b, j, n = 0.7, 1.3, 6
    want = (2 * math.cosh(b * j)) ** n + (2 * math.sinh(b * j)) ** n
    return agrees([("Z = (2 cosh bJ)^N + (2 sinh bJ)^N", value(r, "Z", beta=b, J=j, N=n), want)])


def _ising_limit(r: StatMechResult) -> tuple[bool, str]:
    b, j, h = 0.6, 0.9, 0.4
    lam = math.exp(b * j) * math.cosh(b * h) + math.sqrt(
        math.exp(2 * b * j) * math.sinh(b * h) ** 2 + math.exp(-2 * b * j)
    )
    errors = [
        abs(float(row.error)) for row in (r.approximation.error_table if r.approximation else [])
    ]
    decreasing = (
        bool(errors) and errors == sorted(errors, reverse=True) and errors[-1] < errors[0] / 100
    )
    ok, detail = agrees(
        [
            (
                "f_inf = -(1/beta) ln lambda_max",
                value(r, "f_inf", beta=b, J=j, h=h),
                -math.log(lam) / b,
            )
        ]
    )
    return ok and decreasing, detail + (
        "" if decreasing else "; the finite-size error does not shrink"
    )


def _high_temperature(r: StatMechResult) -> tuple[bool, str]:
    b, e = 0.4, 0.9
    x = b * e
    series = math.log(2) - x / 2 + x**2 / 8 - x**4 / 192  # ln(1 + e^-x) to fourth order
    approx = r.approximation
    scaled = [float(row.error) for row in (approx.error_table if approx else [])]
    falls = len(scaled) == 3 and abs(scaled[0] / scaled[1]) > 2**5 / 3
    ok, detail = agrees([("ln Z series", value(r, "lnZ_series", beta=b, epsilon=e), series)])
    return ok and falls, detail + (
        "" if falls else "; the error does not fall at the claimed order"
    )


def _low_temperature(r: StatMechResult) -> tuple[bool, str]:
    b, e = 4.0, 1.0
    want = e * 0.5 * math.exp(-b * e) + 3 * e * 1.0 * math.exp(-3 * b * e)
    return agrees(
        [("U to first order in the excited levels", value(r, "U_low", beta=b, eps=e), want)]
    )


def _sackur_tetrode(r: StatMechResult) -> tuple[bool, str]:
    b, m, h, v, n, kb = 0.8, 1.3, 1.1, 2.5, 50, 1.7
    kw = {"beta": b, "m": m, "h": h, "V": v, "N": n, "k_B": kb}
    st = kb * n * (math.log(v / n * (2 * math.pi * m / (b * h**2)) ** 1.5) + 2.5)
    return agrees(
        [
            ("Sackur-Tetrode entropy", value(r, "S_leading", **kw), st),
            (
                "with the 1/2 ln(2 pi N) term",
                value(r, "S_stirling", **kw),
                st - kb * 0.5 * math.log(2 * math.pi * n),
            ),
        ]
    )


def _energy_derivative_identities(r: StatMechResult) -> tuple[bool, str]:
    """-d ln Z/d beta = U and d^2 ln Z/d beta^2 = Var(E) on the reference Z, by mpmath differentiation."""
    mp.mp.dps = 30
    levels = [(0, 1), (mp.mpf(1) / 2, 3), (2, 2)]

    def ln_z(b: Any) -> Any:
        return mp.log(sum(g * mp.exp(-b * e) for e, g in levels))

    b = mp.mpf("0.9")
    return agrees(
        [
            ("U = -dlnZ/dbeta", value(r, "U", beta=0.9), float(-mp.diff(ln_z, b))),
            ("Var = d2lnZ/dbeta2", value(r, "Var_E", beta=0.9), float(mp.diff(ln_z, b, 2))),
        ],
        tol=1e-12,
    )


def _pressure_scaling(r: StatMechResult) -> tuple[bool, str]:
    b, v = 0.9, 1.7
    return agrees(
        [("p V = U for E ~ 1/V", value(r, "p", beta=b, V=v) * v, value(r, "U", beta=b, V=v))]
    )


def _copies(r: StatMechResult) -> tuple[bool, str]:
    z1 = 1 + math.exp(-X)
    u1 = EPS * math.exp(-X) / z1
    return agrees(
        [
            ("Z^3", value(r, "Z", beta=B, epsilon=EPS), z1**3),
            ("U x 3", value(r, "U", beta=B, epsilon=EPS), 3 * u1),
        ]
    )


def _huge_energy(r: StatMechResult) -> tuple[bool, str]:
    # E = (0, 1000): the upper level is invisible at beta = 2, so U -> 1000 e^{-2000}, F -> 0
    return agrees([("F", value(r, "F", beta=2.0), -math.log1p(math.exp(-2000.0)) / 2.0)], tol=1e-9)


def _sweep(r: StatMechResult) -> tuple[bool, str]:
    if r.table is None:
        return False, "no table was returned"
    pairs = []
    for row in r.table.rows:
        b = float(row[0])
        pairs.append((f"U(beta={b:.4g})", float(row[1]), 0.5 / math.tanh(b / 2)))
    return agrees(pairs, tol=1e-12)


# -- the corpus -------------------------------------------------------------------------------
TWO = {"kind": "levels", "levels": [{"energy": "0"}, {"energy": "epsilon"}]}
POS_EPS = {"epsilon": ["positive"]}
HW = {"hbar": ["positive"], "omega": ["positive"]}
OSC = {"kind": "level_family", "energy": "hbar*omega*(n + 1/2)"}
CLASSICAL_OSC = {
    "kind": "classical",
    "hamiltonian": "p**2/(2*m) + m*omega**2*x**2/2",
    "variables": [{"name": "x"}, {"name": "p"}],
    "phase_space_cell": "h",
}
BOX = [{"name": c, "domain_lower": "0", "domain_upper": "V**(1/3)"} for c in "xyz"]
GAS = {
    "kind": "classical",
    "hamiltonian": "(px**2 + py**2 + pz**2)/(2*m)",
    "variables": [*BOX, *({"name": p} for p in ("px", "py", "pz"))],
    "phase_space_cell": "h**3",
}
POS_E = {"eps": ["positive"]}
ISING_CHAIN = {
    "kind": "chain",
    "states": ["1", "-1"],
    "bond": "-J*s*t",
    "site": "-h*s",
    "length": "N",
}
ISING_ASSUME = {"J": ["real"], "h": ["real"], "N": ["integer", "positive"]}
GRAND_LEVELS = [
    {"energy": "0"},
    {"energy": "eps"},
    {"energy": "2*eps", "degeneracy": "2"},
]
QH = [["-h", "-d"], ["-d", "h"]]
QSZ = [["1", "0"], ["0", "-1"]]
QSX = [["0", "1"], ["1", "0"]]
QPOS = {"h": ["positive"], "d": ["positive"]}
THREE = {
    "kind": "levels",
    "levels": [
        {"energy": "0"},
        {"energy": "1/2", "degeneracy": "3"},
        {"energy": "2", "degeneracy": "2"},
    ],
}

CASES: list[Case] = [
    Case(
        "two-level-thermodynamics",
        "two-level system",
        "closed form Z = 1 + e^(-beta eps); U, S, Var from direct sums",
        {
            "system": TWO,
            "assumptions": POS_EPS,
            "action": "derive",
            "quantities": ["Z", "F", "U", "S", "Var_E"],
        },
        judge=_two_level,
    ),
    Case(
        "two-level-heat-capacity",
        "two-level system",
        "Schottky heat capacity C = k (beta eps)^2 e^x / (1 + e^x)^2",
        {
            "system": TWO,
            "assumptions": POS_EPS,
            "action": "derive",
            "quantities": [],
            "partials": [{"of": "U", "variable": "T", "held_fixed": [], "label": "C"}],
        },
        judge=_schottky,
    ),
    Case(
        "spin-half-magnetisation",
        "spin-1/2 in a field",
        "mean moment mu tanh(beta mu h)",
        {
            "system": {
                "kind": "levels",
                "levels": [
                    {"energy": "-mu*h", "observables": {"M": "mu"}},
                    {"energy": "mu*h", "observables": {"M": "-mu"}},
                ],
            },
            "parameters": ["h"],
            "assumptions": {"mu": ["positive"], "h": ["real"]},
            "coupling": {"parameter": "h", "observable": "M"},
            "action": "expectation",
            "observable": "M",
            "statistic": "mean",
        },
        judge=_spin,
    ),
    Case(
        "spin-half-fluctuation-response",
        "fluctuation-response relations",
        "variance mu^2 sech^2(beta mu h); equals the second derivative of ln Z in the field",
        {
            "system": {
                "kind": "levels",
                "levels": [
                    {"energy": "-mu*h", "observables": {"M": "mu"}},
                    {"energy": "mu*h", "observables": {"M": "-mu"}},
                ],
            },
            "parameters": ["h"],
            "assumptions": {"mu": ["positive"], "h": ["real"]},
            "coupling": {"parameter": "h", "observable": "M"},
            "action": "expectation",
            "observable": "M",
            "statistic": "variance",
        },
        judge=_spin_variance,
    ),
    Case(
        "finite-state-canonical-ensemble",
        "finite-state canonical ensembles",
        "direct sums over three levels with degeneracies 1, 3, 2",
        {"system": THREE, "action": "derive", "quantities": ["Z", "U", "Var_E"]},
        judge=_finite_states,
    ),
    Case(
        "thermodynamic-derivative-identities",
        "thermodynamic derivative identities",
        "U = -d ln Z/d beta and Var = d^2 ln Z/d beta^2, differentiated numerically on the reference Z",
        {"system": THREE, "action": "derive", "quantities": ["U", "Var_E"]},
        judge=_energy_derivative_identities,
    ),
    Case(
        "quantum-harmonic-oscillator",
        "quantum harmonic oscillator",
        "geometric series: Z = 1/(2 sinh(x/2)), U = (hw/2) coth(x/2), Einstein heat capacity",
        {
            "system": OSC,
            "assumptions": HW,
            "action": "derive",
            "quantities": ["Z", "U"],
            "partials": [{"of": "U", "variable": "T", "held_fixed": [], "label": "C"}],
        },
        judge=_qho,
    ),
    Case(
        "classical-harmonic-oscillator",
        "classical harmonic oscillator",
        "Gaussian phase-space integral Z = 2 pi/(h w beta); equipartition U = kT",
        {
            "system": CLASSICAL_OSC,
            "assumptions": {"m": ["positive"], "omega": ["positive"], "h": ["positive"]},
            "action": "derive",
            "quantities": ["Z", "U", "Var_E"],
        },
        judge=_classical_oscillator,
    ),
    Case(
        "classical-ideal-gas",
        "classical ideal gas",
        "Z = (V (2 pi m/beta)^(3/2)/h^3)^N / N!; U = 3NkT/2, pV = NkT, entropy with the exact ln N!",
        {
            "system": GAS,
            "copies": {"count": "N", "identical": True},
            "parameters": ["V"],
            "assumptions": {
                "m": ["positive"],
                "h": ["positive"],
                "V": ["positive"],
                "N": ["integer", "positive"],
            },
            "action": "derive",
            "quantities": ["U", "S"],
            "partials": [
                {"of": "F", "variable": "V", "held_fixed": ["T"], "scale": "-1", "label": "p"}
            ],
        },
        judge=_ideal_gas,
    ),
    Case(
        "pressure-from-volume-scaling",
        "thermodynamic derivative identities",
        "levels a_i/V scale so that p V = U exactly",
        {
            "system": {
                "kind": "levels",
                "levels": [
                    {"energy": "1/V"},
                    {"energy": "2/V"},
                    {"energy": "5/V", "degeneracy": "2"},
                ],
            },
            "parameters": ["V"],
            "assumptions": {"V": ["positive"]},
            "action": "derive",
            "quantities": ["U"],
            "partials": [
                {"of": "F", "variable": "V", "held_fixed": ["T"], "scale": "-1", "label": "p"}
            ],
        },
        judge=_pressure_scaling,
    ),
    Case(
        "density-of-states-3d",
        "density-of-states integrals",
        "Gamma integral: Z = Gamma(3/2) beta^(-3/2) for rho = sqrt(E)",
        {
            "system": {"kind": "density_of_states", "density": "sqrt(En)", "lower": "0"},
            "action": "derive",
            "quantities": ["Z", "U", "Var_E"],
        },
        judge=_dos_3d,
    ),
    Case(
        "constant-density-third-cumulant",
        "fluctuation-response relations",
        "rho = 1 gives an exponential distribution of E: kappa_3 = 2/beta^3",
        {
            "system": {"kind": "density_of_states", "density": "1", "lower": "0"},
            "action": "expectation",
            "observable": "E",
            "statistic": "cumulant",
            "order": 3,
        },
        judge=_dos_constant_cumulant,
    ),
    Case(
        "spin-in-a-tilted-field",
        "quantum systems",
        "H = -h sz - d sx has eigenvalues +-sqrt(h^2 + d^2): Z = 2 cosh(beta w), von Neumann entropy",
        {
            "system": {"kind": "quantum", "hamiltonian": QH},
            "assumptions": QPOS,
            "action": "derive",
            "quantities": ["Z", "U", "S"],
        },
        judge=_tilted_spin,
    ),
    Case(
        "noncommuting-thermal-average",
        "quantum systems",
        "<sz> = (h/w) tanh(beta w) although [sz, H] != 0; d ln Z/dh = beta <sz> by trace cyclicity",
        {
            "system": {"kind": "quantum", "hamiltonian": QH, "observables": {"sz": QSZ}},
            "assumptions": QPOS,
            "parameters": ["h"],
            "coupling": {"parameter": "h", "observable": "sz"},
            "action": "expectation",
            "observable": "sz",
            "statistic": "mean",
        },
        judge=_tilted_spin_mean,
    ),
    Case(
        "fermi-dirac-thermodynamics",
        "Bose and Fermi occupation numbers",
        "per state n = 1/(e^x + 1), Var n = n(1 - n), entropy -[n ln n + (1 - n) ln(1 - n)]",
        {
            "ensemble": "grand_canonical",
            "system": {"kind": "occupation", "statistics": "fermi", "levels": GRAND_LEVELS},
            "assumptions": POS_E,
            "action": "derive",
            "quantities": ["N", "U", "S", "Var_N"],
        },
        judge=_grand("fermi"),
    ),
    Case(
        "bose-einstein-thermodynamics",
        "Bose and Fermi occupation numbers",
        "per state n = 1/(e^x - 1), Var n = n(1 + n), entropy (1 + n) ln(1 + n) - n ln n",
        {
            "ensemble": "grand_canonical",
            "system": {"kind": "occupation", "statistics": "bose", "levels": GRAND_LEVELS},
            "assumptions": {"eps": ["positive"], "mu": ["negative"]},
            "action": "derive",
            "quantities": ["N", "U", "S", "Var_N"],
        },
        judge=_grand("bose"),
    ),
    Case(
        "classical-counting-limit",
        "Bose and Fermi occupation numbers",
        "Poisson occupation: n = e^(-x), Var N = N, entropy n(1 - ln n) per state",
        {
            "ensemble": "grand_canonical",
            "system": {"kind": "occupation", "statistics": "boltzmann", "levels": GRAND_LEVELS},
            "assumptions": POS_E,
            "action": "derive",
            "quantities": ["N", "U", "S", "Var_N"],
        },
        judge=_grand("boltzmann"),
    ),
    Case(
        "microcanonical-two-level-counting",
        "microcanonical counting",
        "Omega = C(N, k) microstates for N distinguishable two-level units with k excited",
        {
            "ensemble": "microcanonical",
            "system": {"kind": "levels", "levels": [{"energy": "0"}, {"energy": "1"}]},
            "copies": {"count": "40", "identical": False},
            "total_energy": "16",
            "action": "partition",
        },
        judge=_binomial(40, 16),
    ),
    Case(
        "ising-ring-partition-function",
        "Ising chains",
        "Z of a ring of 7 spins in a field, summed over all 2^7 configurations",
        {
            "system": ISING_CHAIN,
            "assumptions": ISING_ASSUME,
            "action": "derive",
            "quantities": ["Z"],
        },
        judge=_ising,
    ),
    Case(
        "ising-zero-field-closed-form",
        "Ising chains",
        "Z = (2 cosh beta J)^N + (2 sinh beta J)^N for a ring without a field",
        {
            "system": {**ISING_CHAIN, "site": "0"},
            "assumptions": {"J": ["positive"], "N": ["integer", "positive"]},
            "action": "derive",
            "quantities": ["Z"],
        },
        judge=_ising_zero_field,
    ),
    Case(
        "ising-thermodynamic-limit",
        "thermodynamic limits",
        "f = -(1/beta) ln lambda_max with lambda_max = e^(bJ) cosh(bh) + sqrt(e^(2bJ) sinh^2(bh) + e^(-2bJ)); "
        "finite-size error from brute force falls exponentially",
        {
            "system": ISING_CHAIN,
            "assumptions": ISING_ASSUME,
            "action": "approximate",
            "method": "thermodynamic_limit",
        },
        judge=_ising_limit,
    ),
    Case(
        "high-temperature-expansion",
        "high- and low-temperature expansions",
        "ln(1 + e^-x) = ln 2 - x/2 + x^2/8 - x^4/192 + ...; the error falls at O(beta^5)",
        {
            "system": TWO,
            "assumptions": POS_EPS,
            "action": "approximate",
            "method": "high_temperature",
            "order": 4,
        },
        judge=_high_temperature,
    ),
    Case(
        "low-temperature-corrections",
        "high- and low-temperature expansions",
        "U = E0 + sum_j (g_j/g_0) eps_j e^(-beta eps_j) to first order, levels 0 (g=2), eps, 3 eps (g=2)",
        {
            "system": {
                "kind": "levels",
                "levels": [
                    {"energy": "0", "degeneracy": "2"},
                    {"energy": "eps"},
                    {"energy": "3*eps", "degeneracy": "2"},
                ],
            },
            "assumptions": POS_E,
            "action": "approximate",
            "method": "low_temperature",
        },
        judge=_low_temperature,
    ),
    Case(
        "stirling-sackur-tetrode",
        "Stirling approximation",
        "S = N k [ln(V/N (2 pi m/(beta h^2))^(3/2)) + 5/2] from the exact ln Z and Stirling's ln N!",
        {
            "system": GAS,
            "copies": {"count": "N", "identical": True},
            "parameters": ["V"],
            "assumptions": {
                "m": ["positive"],
                "h": ["positive"],
                "V": ["positive"],
                "N": ["integer", "positive"],
            },
            "action": "approximate",
            "method": "stirling",
        },
        judge=_sackur_tetrode,
    ),
    Case(
        "independent-copies",
        "independent subsystems",
        "Z_N = Z_1^N and U_N = N U_1 for distinguishable copies",
        {
            "system": TWO,
            "copies": {"count": "3", "identical": False},
            "assumptions": POS_EPS,
            "action": "derive",
            "quantities": ["Z", "U"],
        },
        judge=_copies,
    ),
    Case(
        "numerically-unstable-exponentials",
        "numerical stability",
        "levels 0 and 1000: F = -ln(1 + e^(-1000 beta))/beta without overflow",
        {
            "system": {"kind": "levels", "levels": [{"energy": "0"}, {"energy": "1000"}]},
            "action": "derive",
            "quantities": ["F"],
        },
        judge=_huge_energy,
    ),
    Case(
        "oscillator-sweep",
        "temperature-dependent observables",
        "U(beta) = (1/2) coth(beta/2) for hbar = omega = 1, over a logarithmic sweep",
        {
            "system": OSC,
            "assumptions": HW,
            "action": "evaluate",
            "quantities": ["U"],
            "values": {"hbar": "1", "omega": "1", "k_B": "1"},
            "sweep": {
                "variable": "beta",
                "start": "1/10",
                "stop": "5",
                "points": 7,
                "scale": "log",
            },
        },
        judge=_sweep,
    ),
    # refusals: the reason is part of the case
    Case(
        "divergent-partition-function",
        "invalid and divergent partition functions",
        "E_n = -n: terms grow like e^(beta n)",
        {"system": {"kind": "level_family", "energy": "-n"}, "action": "partition"},
        "refused",
        "divergent",
    ),
    Case(
        "inconclusive-convergence",
        "invalid and divergent partition functions",
        "E_n = ln(n + 1): terms n^(-beta), convergent only for beta > 1",
        {"system": {"kind": "level_family", "energy": "log(n + 1)"}, "action": "partition"},
        "refused",
        "convergence_not_provable",
    ),
    Case(
        "invalid-degeneracy",
        "invalid and divergent partition functions",
        "a degeneracy is a positive integer",
        {
            "system": {"kind": "levels", "levels": [{"energy": "0", "degeneracy": "0"}]},
            "action": "partition",
        },
        "refused",
        "invalid_degeneracy",
    ),
    Case(
        "non-normalisable-density",
        "missing temperature or convergence assumptions",
        "rho = exp(c E) is integrable against e^(-beta E) only for beta > c",
        {
            "system": {"kind": "density_of_states", "density": "exp(c*En)", "lower": "0"},
            "assumptions": {"c": ["positive"]},
            "action": "partition",
        },
        "refused",
        "convergence_not_provable",
    ),
    Case(
        "free-particle-without-a-box",
        "non-normalisable distributions",
        "the position integral of a free particle over the whole line diverges",
        {
            "system": {
                "kind": "classical",
                "hamiltonian": "p**2/2",
                "variables": [{"name": "x"}, {"name": "p"}],
                "phase_space_cell": "1",
            },
            "action": "partition",
        },
        "refused",
        "convergence_not_provable",
    ),
    Case(
        "coupled-oscillators-do-not-factor",
        "invalid ensemble substitutions",
        "a cross term (x - y)^2 couples two variables",
        {
            "system": {
                "kind": "classical",
                "hamiltonian": "(x - y)**2 + p**2",
                "variables": [{"name": "x"}, {"name": "y"}, {"name": "p"}],
                "phase_space_cell": "1",
            },
            "action": "partition",
        },
        "refused",
        "not_separable",
    ),
    Case(
        "phase-space-cell-is-the-users-choice",
        "invalid ensemble substitutions",
        "no default normalisation",
        {
            "system": {"kind": "classical", "hamiltonian": "p**2/2", "variables": [{"name": "p"}]},
            "action": "partition",
        },
        "refused",
        "convention_unconfirmed",
    ),
    Case(
        "indistinguishability-is-the-users-choice",
        "invalid ensemble substitutions",
        "no default N!",
        {"system": TWO, "copies": {"count": "3"}, "assumptions": POS_EPS, "action": "partition"},
        "refused",
        "convention_unconfirmed",
    ),
    Case(
        "derivative-at-the-wrong-fixed-variables",
        "invalid ensemble substitutions",
        "dF/dV in the canonical ensemble holds T fixed, not nothing",
        {
            "system": {"kind": "levels", "levels": [{"energy": "1/V"}, {"energy": "2/V"}]},
            "parameters": ["V"],
            "assumptions": {"V": ["positive"]},
            "action": "derive",
            "quantities": [],
            "partials": [{"of": "F", "variable": "V", "held_fixed": []}],
        },
        "refused",
        "ensemble_mismatch",
    ),
    Case(
        "second-derivative-response-needs-commuting-operators",
        "noncommuting quantum operators",
        "d^2 ln Z/dh^2 is the Kubo-Duhamel covariance, not beta^2 Var(sz), when [sz, H] != 0",
        {
            "system": {"kind": "quantum", "hamiltonian": QH, "observables": {"sz": QSZ}},
            "assumptions": QPOS,
            "parameters": ["h"],
            "coupling": {"parameter": "h", "observable": "sz"},
            "action": "expectation",
            "observable": "sz",
            "statistic": "variance",
        },
        "refused",
        "operators_do_not_commute",
    ),
    Case(
        "covariance-of-noncommuting-operators",
        "noncommuting quantum operators",
        "the covariance of sz and sx depends on their ordering",
        {
            "system": {"kind": "quantum", "hamiltonian": QH, "observables": {"sz": QSZ, "sx": QSX}},
            "assumptions": QPOS,
            "action": "expectation",
            "observable": "sz",
            "other_observable": "sx",
            "statistic": "covariance",
        },
        "refused",
        "operators_do_not_commute",
    ),
    Case(
        "non-hermitian-hamiltonian",
        "invalid and divergent partition functions",
        "a thermal state needs H equal to its conjugate transpose",
        {
            "system": {"kind": "quantum", "hamiltonian": [["0", "a"], ["b", "0"]]},
            "assumptions": {"a": ["real"], "b": ["real"]},
            "action": "partition",
        },
        "refused",
        "not_hermitian",
    ),
    Case(
        "bose-needs-mu-below-the-levels",
        "invalid and divergent partition functions",
        "n = 1/(e^x - 1) diverges for x <= 0: mu must be provably below every level",
        {
            "ensemble": "grand_canonical",
            "system": {"kind": "occupation", "statistics": "bose", "levels": GRAND_LEVELS},
            "assumptions": POS_E,
            "action": "partition",
        },
        "refused",
        "convergence_not_provable",
    ),
    Case(
        "ensemble-is-not-substituted",
        "invalid ensemble substitutions",
        "an occupation system with a chemical potential belongs to the grand canonical ensemble",
        {
            "ensemble": "canonical",
            "system": {"kind": "occupation", "statistics": "fermi", "levels": GRAND_LEVELS},
            "assumptions": POS_E,
            "action": "partition",
        },
        "refused",
        "ensemble_mismatch",
    ),
    Case(
        "heat-capacity-must-hold-mu-fixed",
        "invalid ensemble substitutions",
        "dU/dT in the grand canonical ensemble holds mu fixed; leaving it free is another derivative",
        {
            "ensemble": "grand_canonical",
            "system": {"kind": "occupation", "statistics": "fermi", "levels": GRAND_LEVELS},
            "assumptions": POS_E,
            "action": "derive",
            "quantities": [],
            "partials": [{"of": "U", "variable": "T", "held_fixed": []}],
        },
        "refused",
        "ensemble_mismatch",
    ),
    Case(
        "energy-that-no-microstate-has",
        "microcanonical counting",
        "E = 9/2 is not a sum of four integer level energies",
        {
            "ensemble": "microcanonical",
            "system": {"kind": "levels", "levels": [{"energy": "0"}, {"energy": "1"}]},
            "copies": {"count": "4", "identical": False},
            "total_energy": "9/2",
            "action": "partition",
        },
        "refused",
        "energy_not_attainable",
    ),
    Case(
        "expansion-at-a-singular-point",
        "invalid saddle-point and expansion points",
        "ln Z of an oscillator is not analytic at beta = 0 (Z ~ 1/beta): no Taylor series",
        {
            "system": {"kind": "level_family", "energy": "hbar*omega*(n + 1/2)"},
            "assumptions": {"hbar": ["positive"], "omega": ["positive"]},
            "action": "approximate",
            "method": "high_temperature",
            "order": 2,
        },
        "refused",
        "singular_expansion_point",
    ),
    Case(
        "low-temperature-order-is-not-decided",
        "invalid saddle-point and expansion points",
        "which of two unrelated levels is the ground level cannot be decided",
        {
            "system": {"kind": "levels", "levels": [{"energy": "a"}, {"energy": "b"}]},
            "assumptions": {"a": ["real"], "b": ["real"]},
            "action": "approximate",
            "method": "low_temperature",
        },
        "refused",
        "ordering_not_decided",
    ),
    Case(
        "stirling-needs-a-ln-n-factorial",
        "Stirling approximation",
        "without indistinguishable copies there is no ln N! to approximate",
        {"system": TWO, "assumptions": POS_EPS, "action": "approximate", "method": "stirling"},
        "refused",
        "nothing_to_approximate",
    ),
    Case(
        "thermodynamic-limit-needs-a-symbolic-length",
        "thermodynamic limits",
        "a fixed length has no limit N -> infinity to study",
        {
            "system": {**ISING_CHAIN, "length": "8"},
            "assumptions": {"J": ["real"], "h": ["real"]},
            "action": "approximate",
            "method": "thermodynamic_limit",
        },
        "refused",
        "assumptions_insufficient",
    ),
    Case(
        "complex-energy",
        "invalid and divergent partition functions",
        "an energy must be provably real",
        {"system": {"kind": "levels", "levels": [{"energy": "I"}]}, "action": "partition"},
        "refused",
        "assumptions_insufficient",
    ),
    Case(
        "non-linear-coupling",
        "invalid interchange of derivatives",
        "dE/dh = 2h is not -M, so d^2 ln Z/dh^2 is not beta^2 Var(M)",
        {
            "system": {
                "kind": "levels",
                "levels": [
                    {"energy": "-h**2", "observables": {"M": "1"}},
                    {"energy": "h**2", "observables": {"M": "-1"}},
                ],
            },
            "parameters": ["h"],
            "assumptions": {"h": ["real"]},
            "coupling": {"parameter": "h", "observable": "M"},
            "action": "expectation",
            "observable": "M",
        },
        "refused",
        "coupling_not_linear",
    ),
]


# -- running ----------------------------------------------------------------------------------
def judge_case(case: Case, engine: Engine) -> tuple[str, str]:
    result = engine.statmech(StatMechInput(**case.request))
    if case.expected == "refused":
        refused = result.status is not Status.OK and not result.quantities
        if not refused:
            return (
                "wrong",
                f"returned a result where a refusal was due (status {result.status.value})",
            )
        if result.reason != case.reason:
            return "wrong", f"refused with {result.reason}, expected {case.reason}"
        return "refused_as_expected", f"{result.reason}"
    if result.status is not Status.OK or case.judge is None:
        return "unsupported_regression", f"{result.status.value}: {result.message[:120]}"
    ok, detail = case.judge(result)
    return ("correct", detail) if ok else ("wrong", detail)


def run_cases(engine: Engine | None = None) -> list[dict[str, Any]]:
    engine = engine or Engine(Limits(timeout_seconds=120, max_concurrency=2))
    rows: list[dict[str, Any]] = []
    for case in CASES:
        start = time.monotonic()
        try:
            outcome, detail = judge_case(case, engine)
        except Exception as exc:  # a crash is a result: it must never be read as a pass
            outcome, detail = "wrong", f"{type(exc).__name__} while judging"
        rows.append(
            {
                "id": case.id,
                "category": case.category,
                "source": case.source,
                "expected": case.expected,
                "reason": case.reason,
                "outcome": outcome,
                "detail": detail,
                "latency_ms": int((time.monotonic() - start) * 1000),
            }
        )
    return rows


def deterministic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {k: r[k] for k in ("id", "category", "source", "expected", "reason", "outcome")}
        for r in rows
    ]


def markdown(rows: list[dict[str, Any]]) -> str:
    good = {"correct", "refused_as_expected"}
    lines = [
        "# Statistical mechanics benchmark",
        "",
        "Generated by `python -m minerva_engine.statmech.benchmark`; outcomes are deterministic, "
        "latencies are not gated. Every reference value is plain arithmetic on a textbook closed form, "
        "never Minerva's own route; a refusal case names the enumerated reason it must carry.",
        "",
        f"{sum(r['outcome'] in good for r in rows)} of {len(rows)} as expected; "
        f"wrong: {sum(r['outcome'] == 'wrong' for r in rows)}; "
        f"unsupported where an answer is expected: "
        f"{sum(r['outcome'] == 'unsupported_regression' for r in rows)}.",
        "",
        "| Problem | Category | Expected | Outcome | Reference | ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{r['id']}` | {r['category']} | "
        f"{r['expected']}{' (' + r['reason'] + ')' if r['reason'] else ''} | {r['outcome']} | "
        f"{r['source']} | {r['latency_ms']} |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[5]
    json_path, md_path = root / "docs/benchmarks/statmech.json", root / "docs/statmech-benchmark.md"
    rows = run_cases()
    bad = [r for r in rows if r["outcome"] in ("wrong", "unsupported_regression")]
    if "--check" in argv:
        expected = json.loads(json_path.read_text())["problems"] if json_path.exists() else None
        if expected != deterministic(rows):
            print("outcomes differ from docs/benchmarks/statmech.json", file=sys.stderr)
            return 1
    else:
        json_path.write_text(json.dumps({"problems": deterministic(rows)}, indent=2) + "\n")
        md_path.write_text(markdown(rows))
    for r in bad:
        print(f"{r['id']}: {r['outcome']} ({r['detail']})", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
