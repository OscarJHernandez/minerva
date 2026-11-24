"""Mechanics problems: perturbation theory, conserved quantities, discovery, against exact solutions and brackets."""

from __future__ import annotations

import re

import mpmath as mp
import sympy as sp

from ..discovery import Pins
from ..latexparse import parse_latex
from ..service import Engine
from .core import Problem, verdict


def _duffing(engine: Engine) -> tuple[str, str]:
    """Frequency of x'' + x + eps x^3 = 0 against the exact period from the quarter-period integral.

    The first correction 3 eps A^2 / 8 is Landau and Lifshitz, Mechanics, section 28; the exact period
    comes from energy conservation: T / 4 = integral from 0 to A of dx / sqrt(A^2 - x^2 + eps (A^4 - x^4) / 2).
    """
    report = engine.perturbation_solve(
        r"\frac{1}{2}\dot x^2 - \frac{1}{2}x^2 - \frac{\epsilon}{4}x^4",
        ["x"],
        assumptions={"epsilon": ["positive"]},
        small_parameter="epsilon",
        order=1,
    )
    if report.status.value != "ok" or not report.frequency_latex:
        return verdict("correct", False, None, f"status {report.status.value}")
    eps, amp = mp.mpf("0.01"), mp.mpf(1)
    quarter = mp.quad(lambda s: 1 / mp.sqrt(amp**2 - s**2 + eps * (amp**4 - s**4) / 2), [0, amp])
    exact = 2 * mp.pi / (4 * quarter)
    rhs = report.frequency_latex.split("=", 1)[1]
    value = parse_latex(rhs).subs({sp.Symbol("epsilon"): sp.Rational(1, 100), sp.Symbol("A"): 1})
    # first order: the remaining error is of order eps^2
    return verdict(
        "correct",
        True,
        abs(complex(value.evalf(20)) - complex(exact)) < 1e-4,
        "frequency vs exact period",
    )


def _time_dependent_energy_refused(engine: Engine) -> tuple[str, str]:
    """L = m v^2/2 - k x^2 cos(t)/2: the energy function changes, dE/dt = -(k/2) x^2 sin(t) on the motion."""
    t = sp.Symbol("t")
    m, k = sp.symbols("m k", positive=True)
    x = sp.Function("x")(t)
    energy = m * x.diff(t) ** 2 / 2 + k * x**2 * sp.cos(t) / 2
    on_motion = energy.diff(t).subs(x.diff(t, 2), -k * x * sp.cos(t) / m)
    if sp.simplify(on_motion) == 0:  # the oracle: the energy function is NOT conserved here
        raise RuntimeError("oracle: the energy is conserved, the problem is ill-posed")
    report = engine.analyze_lagrangian(
        r"\frac12 m\dot{x}^2 - \frac12 k x^2 \cos t",
        ["x"],
        assumptions={"m": ["positive"], "k": ["positive"]},
    )
    claimed = [c for c in report.conserved if c.kind == "energy" and c.verified]
    return verdict(
        "refused",
        bool(claimed),
        None,
        f"status {report.status.value}, {len(report.conserved)} constants",
    )


def _rotation_constant_along_the_motion(engine: Engine) -> tuple[str, str]:
    """Isotropic oscillator: the reported angular momentum is constant along the exact solution.

    With m = 3, k = 2 the motion is x = x0 cos(w t) + (vx0 / w) sin(w t) (and the same for y), w = sqrt(k / m).
    The expression is Minerva's; the trajectory and the check are not.
    """
    report = engine.analyze_lagrangian(
        r"\frac12 m(\dot{x}^2+\dot{y}^2) - \frac{k}{2}(x^2+y^2)", ["x", "y"]
    )
    rotation = [c for c in report.conserved if c.kind == "rotation" and c.verified]
    if report.status.value != "ok" or not rotation:
        return verdict("correct", False, None, f"status {report.status.value}")
    source = rotation[0].expression_latex.split("=", 1)[1]
    source = re.sub(r"\\dot\{(\w)\}", r"v_{\1}", source)
    quantity = parse_latex(source)
    names = {str(s).replace("{", "").replace("}", ""): s for s in quantity.free_symbols}
    if not {"m", "x", "y", "v_x", "v_y"} <= set(names):
        return verdict("correct", True, False, f"unexpected symbols {sorted(names)}")
    mass, spring = mp.mpf(3), mp.mpf(2)
    w = mp.sqrt(spring / mass)
    x0, vx0, y0, vy0 = mp.mpf(1), mp.mpf("0.3"), mp.mpf("-0.4"), mp.mpf("0.8")

    def at(t: float) -> mp.mpf:
        c, s = mp.cos(w * t), mp.sin(w * t)
        values = {
            names["m"]: sp.Float(mass, 30),
            names["x"]: sp.Float(x0 * c + vx0 / w * s, 30),
            names["y"]: sp.Float(y0 * c + vy0 / w * s, 30),
            names["v_x"]: sp.Float(-x0 * w * s + vx0 * c, 30),
            names["v_y"]: sp.Float(-y0 * w * s + vy0 * c, 30),
        }
        return mp.mpf(str(sp.N(quantity.subs(values), 25)))

    start = at(0)
    if abs(start) < mp.mpf("1e-6"):
        raise RuntimeError("oracle: the initial condition has no angular momentum")
    drift = max(abs(at(t) - start) for t in (0.7, 1.9, 3.3, 5.1, 8.0))
    return verdict("correct", True, drift < mp.mpf("1e-18"), "constant along the exact solution")


def _isotropic_oscillator_discovery(engine: Engine) -> tuple[str, str]:
    """H = (p1^2 + p2^2)/2 + (q1^2 + q2^2)/2: every reported constant has an exact zero Poisson bracket with H,
    and three of them are functionally independent (rank 3 at a rational point: the maximum for two degrees
    of freedom is 2n - 1 = 3). Brackets and ranks are computed here with SymPy, from the reported expressions."""
    h_source = "(p_1^2+p_2^2)/2+(q_1^2+q_2^2)/2"
    report = engine.discover_invariants(
        h_source, ["q_1", "q_2"], system="hamiltonian", momenta=["p_1", "p_2"]
    )
    if report.status.value != "ok" or not report.constants:
        return verdict("correct", False, None, f"status {report.status.value}")
    hamiltonian = parse_latex(h_source)
    symbols = sorted(hamiltonian.free_symbols, key=str)
    q = [s for s in symbols if str(s).startswith("q")]
    p = [s for s in symbols if str(s).startswith("p")]

    def bracket(f: sp.Expr) -> sp.Expr:
        return sp.expand(
            sum(
                sp.diff(f, qi) * sp.diff(hamiltonian, pi)
                - sp.diff(f, pi) * sp.diff(hamiltonian, qi)
                for qi, pi in zip(q, p, strict=True)
            )
        )

    constants = [parse_latex(c.expression_latex.split("=", 1)[1]) for c in report.constants]
    if any(bracket(c) != 0 for c in constants):
        return verdict("correct", True, False, "a reported constant is not conserved")
    point = dict(zip([*q, *p], [1, 2, 3, 5], strict=True))
    gradients = sp.Matrix([[sp.diff(c, v).subs(point) for v in [*q, *p]] for c in constants])
    return verdict("correct", True, gradients.rank() >= 3, "exact brackets and Jacobian rank")


def _double_pendulum_small_oscillations(engine: Engine) -> tuple[str, str]:
    """Squared frequencies of the double pendulum about the hanging position, against the roots of det(K - w M).

    For small angles the Lagrangian is the quadratic form with mass matrix
    M = [[(m1+m2) l1^2, m2 l1 l2], [m2 l1 l2, m2 l2^2]] and stiffness K = diag((m1+m2) g l1, m2 g l2)
    (Landau and Lifshitz, Mechanics, section 23, problem 1); the squared frequencies are the roots of
    det(K - w M) = 0, found here from its coefficients with `mpmath.polyroots`. The parameters are
    declared nothing: no sign is assumed for them.
    """
    report = engine.expand_motion(
        r"\frac12 (m_1+m_2) l_1^2\dot{\theta}_1^2 + \frac12 m_2 l_2^2\dot{\theta}_2^2 "
        r"+ m_2 l_1 l_2\dot{\theta}_1\dot{\theta}_2\cos(\theta_1-\theta_2) "
        r"+ (m_1+m_2) g l_1\cos\theta_1 + m_2 g l_2\cos\theta_2",
        ["theta_1", "theta_2"],
        "t",
        {},
        {"theta_1": "0", "theta_2": "0"},
        1,
    )
    if report.status.value != "ok" or len(report.modes) != 2:
        return verdict("correct", False, None, f"status {report.status.value}")
    m1, m2, g, l1, l2 = (sp.Rational(x) for x in ("13/10", "7/10", "49/5", "11/10", "9/10"))
    mass = sp.Matrix([[(m1 + m2) * l1**2, m2 * l1 * l2], [m2 * l1 * l2, m2 * l2**2]])
    stiffness = sp.diag((m1 + m2) * g * l1, m2 * g * l2)
    lam = sp.Symbol("lam")
    coefficients = sp.Poly((stiffness - lam * mass).det(), lam).all_coeffs()
    expected = sorted(
        float(mp.re(r)) for r in mp.polyroots([mp.mpf(str(float(c))) for c in coefficients])
    )
    names = dict(zip(("m_1", "m_2", "g", "l_1", "l_2"), (m1, m2, g, l1, l2), strict=True))
    found = []
    for mode in report.modes:
        value = parse_latex(mode.omega_squared_latex.split("=", 1)[1])
        found.append(float(value.subs({x: names[x.name] for x in value.free_symbols}).evalf(20)))
    found.sort()
    return verdict(
        "correct",
        True,
        all(abs(a - b) <= 1e-9 * max(1.0, abs(b)) for a, b in zip(found, expected, strict=True)),
        "squared frequencies vs the roots of det(K - w M)",
    )


def _kepler_routhian(engine: Engine) -> tuple[str, str]:
    """Routh reduction of the planar Kepler problem by the angle.

    Landau and Lifshitz, Mechanics, section 14: with the angular momentum M fixed the radial motion is that
    of a particle in U_eff = U + M^2 / (2 m r^2); section 41 defines the Routhian. With R = L - p c_dot (the
    sign in which R acts as a Lagrangian) that is R = m r_dot^2 / 2 + k / r - p^2 / (2 m r^2), written here
    by hand, with the velocity theta_dot = p / (m r^2).
    """
    report = engine.routh_reduce(
        r"\frac{1}{2} m \left(\dot{r}^2 + r^2\dot{\theta}^2\right) + \frac{k}{r}",
        ["r", "theta"],
        ["theta"],
        assumptions={"m": ["positive"], "k": ["positive"], "r": ["positive"]},
    )
    if report.status.value != "ok" or not report.result_latex or not report.reconstruction:
        return verdict("correct", False, None, f"status {report.status.value}")
    m, k, r, rd, p = sp.symbols("m k r r_dot p_theta")

    def plain(latex: str) -> sp.Expr:
        value = parse_latex(latex)
        return value.xreplace({s: sp.Symbol(s.name) for s in value.free_symbols})

    routhian = plain(report.result_latex)
    velocity = plain(report.reconstruction[0].latex.split("=", 1)[1])
    expected = m * rd**2 / 2 + k / r - p**2 / (2 * m * r**2)
    return verdict(
        "correct",
        True,
        sp.simplify(routhian - expected) == 0 and sp.simplify(velocity - p / (m * r**2)) == 0,
        "Routhian and angular velocity vs the hand-written effective-potential form",
    )


def _schwarzschild_geodesic_constants(engine: Engine) -> tuple[str, str]:
    """Constants of motion of geodesics of the Schwarzschild metric, found by the search after the Routh
    reduction of the cyclic t and phi (issue #152).

    Wald, General Relativity, section 6.3, and Carroll, Spacetime and Geometry, section 5.4: the Killing vectors
    of t and phi give E = (1 - 2M/r) t_dot and L = r^2 sin^2(theta) phi_dot, and the normalisation g(v, v) is
    constant. Written here by hand; each reported constant must be a nonzero constant multiple of exactly one of
    them, each of them must be reported, and nothing else may be (the squared total angular momentum is not a
    polynomial family of the reduced system).
    """
    report = engine.discover_invariants(
        r"\frac{1}{2}\left(-\left(1-\frac{2M}{r}\right)\dot{t}^2+\frac{\dot{r}^2}{1-\frac{2M}{r}}"
        r"+r^2\dot{\theta}^2+r^2\sin^2(\theta)\dot{\phi}^2\right)",
        ["t", "r", "theta", "phi"],
        budget="standard",
        # One family of the reduced system (90 unknowns), so the problem measures the reduction and the
        # mapping back, not the speed of the whole ladder (about 11 s against the sandbox's 30 s otherwise).
        pins=Pins(degree=2, power_bound=2, harmonics=1, time_degree=0),
        time="s",
        assumptions={"M": ["positive"]},
    )
    if report.status.value != "ok" or report.reduction is None:
        return verdict("correct", False, None, f"status {report.status.value}")
    m, r, theta = sp.symbols("M r theta")
    t_dot, r_dot, theta_dot, phi_dot = sp.symbols("t_dot r_dot theta_dot phi_dot")
    f = 1 - 2 * m / r
    expected = [
        f * t_dot,
        r**2 * sp.sin(theta) ** 2 * phi_dot,
        -f * t_dot**2 + r_dot**2 / f + r**2 * theta_dot**2 + r**2 * sp.sin(theta) ** 2 * phi_dot**2,
    ]

    def plain(latex: str) -> sp.Expr:
        value = parse_latex(latex.split("=", 1)[1])
        return value.xreplace({s: sp.Symbol(s.name) for s in value.free_symbols})

    def multiple(found: sp.Expr, target: sp.Expr) -> bool:
        ratio = sp.simplify(found / target)
        return bool(ratio.is_number and ratio != 0)

    found = [plain(c.expression_latex) for c in report.constants]
    matched = [[multiple(c, e) for e in expected] for c in found]
    ok = (
        len(found) == len(expected)
        and all(sum(row) == 1 for row in matched)
        and all(sum(row[j] for row in matched) == 1 for j in range(len(expected)))
    )
    return verdict("correct", True, ok, "E, L and g(v, v) vs the hand-written Killing constants")


PROBLEMS: list[Problem] = [
    Problem(
        "duffing-frequency-shift",
        "mechanics",
        "Landau and Lifshitz, Mechanics, section 28 (anharmonic oscillations)",
        "correct",
        _duffing,
        oracle="mpmath_quad",
    ),
    Problem(
        "lagrangian-time-dependent-energy-refused",
        "mechanics",
        "Landau and Lifshitz, Mechanics, section 6 (energy is conserved only if L has no explicit time)",
        "refused",
        _time_dependent_energy_refused,
        oracle="closed_form",
        evidence="refusal",
    ),
    Problem(
        "lagrangian-rotation-constant-along-the-motion",
        "mechanics",
        "Landau and Lifshitz, Mechanics, section 9 (angular momentum of a central field)",
        "correct",
        _rotation_constant_along_the_motion,
        oracle="exact_solution",
        evidence="symbolic_proof",
    ),
    Problem(
        "expansion-double-pendulum-small-oscillation-frequencies",
        "mechanics",
        "Landau and Lifshitz, Mechanics, section 23, problem 1 (small oscillations of a double pendulum)",
        "correct",
        _double_pendulum_small_oscillations,
        oracle="mpmath_polyroots",
        evidence="numeric_agreement",
    ),
    Problem(
        "routh-kepler-reduced-by-the-angle",
        "mechanics",
        "Landau and Lifshitz, Mechanics, sections 14 and 41 (effective potential; the Routhian); Goldstein, "
        "Classical Mechanics, section 8.3",
        "correct",
        _kepler_routhian,
        oracle="closed_form",
        evidence="symbolic_proof",
    ),
    Problem(
        "discovery-schwarzschild-after-routh-reduction",
        "mechanics",
        "Wald, General Relativity, section 6.3; Carroll, Spacetime and Geometry, section 5.4 (geodesic "
        "constants of the Schwarzschild metric)",
        "correct",
        _schwarzschild_geodesic_constants,
        oracle="closed_form",
        evidence="symbolic_proof",
    ),
    Problem(
        "discovery-isotropic-oscillator",
        "mechanics",
        "Landau and Lifshitz, Mechanics, section 15 (the isotropic oscillator has 2n - 1 independent constants)",
        "correct",
        _isotropic_oscillator_discovery,
        oracle="exact_bracket",
        evidence="symbolic_proof",
        tier="extended",
    ),
]
