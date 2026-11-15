"""Finite-dimensional quantum systems: Z = Tr exp(-beta H), thermal averages Tr(rho A), no commutation assumed.

The symbolic route is the spectral decomposition of a Hermitian matrix: eigenvalues with multiplicities
give Z, and Sylvester's formula exp(-beta H) = sum_i exp(-beta l_i) prod_{j != i} (H - l_j)/(l_i - l_j)
(over the distinct eigenvalues) gives the thermal matrix for a thermal average. The independent route never
uses either: ``mpmath.expm`` of the numerical matrix, its trace, and ``mpmath.eigh``.

Operators are never reordered or assumed to commute. A thermal average is Tr(A exp(-beta H))/Z with A where
it stands; ``[A, H] = 0`` is a condition that is checked and reported, and a fluctuation-response relation
beyond the first derivative is refused unless the coupled operator commutes with H.
"""

from __future__ import annotations

from typing import Any

import mpmath as mp
import sympy as sp

from ..distributions.models import Condition
from ..latexprint import to_latex
from .models import QuantumSystem, StatStep
from .oracle import BaseOracle, Point, cumulants_from_raw
from .setup import Problem, Refusal

MAX_DIM = 4


def _matrix(problem: Problem, rows: list[list[str]], what: str) -> sp.Matrix:
    n = len(rows)
    if n == 0 or any(len(r) != n for r in rows):
        raise Refusal("parse_error", f"{what} must be a square matrix")
    if n > MAX_DIM:
        raise Refusal("resource_limit", f"at most {MAX_DIM}x{MAX_DIM} matrices")
    return sp.Matrix(n, n, lambda i, j: problem.parse(rows[i][j]))


def _is_zero(matrix: sp.Matrix) -> bool:
    return all(sp.simplify(entry) == 0 for entry in matrix)


def is_hermitian(matrix: sp.Matrix) -> bool:
    return _is_zero(matrix - matrix.H)


def commutes(a: sp.Matrix, b: sp.Matrix) -> bool:
    return _is_zero(a * b - b * a)


def fill_quantum(problem: Problem, s: QuantumSystem) -> set[sp.Symbol]:
    H = _matrix(problem, s.hamiltonian, "the Hamiltonian")
    n = H.shape[0]
    if not is_hermitian(H):
        raise Refusal(
            "not_hermitian",
            "the Hamiltonian is not Hermitian under the declared assumptions (declare its "
            "parameters real); a thermal state needs H equal to its conjugate transpose",
        )
    observables: dict[str, sp.Matrix] = {}
    for name, rows in s.observables.items():
        if name == "E":
            raise Refusal("unknown_observable", "E is the energy; it cannot be redefined")
        m = _matrix(problem, rows, f"observable {name}")
        if m.shape != H.shape:
            raise Refusal("parse_error", f"observable {name} must have the size of H")
        observables[name] = m
    try:
        eigenvalues = H.eigenvals()
    except Exception as exc:
        raise Refusal("unsupported_system", "no closed-form eigenvalues were found") from exc
    if sum(eigenvalues.values()) != n or not eigenvalues:
        raise Refusal("unsupported_system", "no closed-form eigenvalues were found")
    energies = list(eigenvalues)
    if any(e.is_real is not True for e in energies):
        raise Refusal(
            "assumptions_insufficient",
            "the eigenvalues are not provably real: declare the parameters of H real",
        )
    if {problem.beta, problem.T, problem.k_B} & H.free_symbols:
        raise Refusal("unsupported_system", "the Hamiltonian must not depend on the temperature")
    degeneracies = [sp.Integer(m) for m in eigenvalues.values()]
    beta = problem.beta
    z1 = sp.Add(*[g * sp.exp(-beta * e) for e, g in zip(energies, degeneracies, strict=True)])
    problem.kind = "quantum"
    problem.energies, problem.degeneracies = energies, degeneracies
    problem.observables = {"E": list(energies)}
    problem.extra = {"H": H, "matrices": observables, "dim": n}
    problem.Z1 = None  # Canonical builds Z from the eigenvalues
    cond_h = Condition(
        latex=r"H = H^{\dagger}",
        established=True,
        how="the matrix equals its conjugate transpose under the declared assumptions",
    )
    cond_z = Condition(
        latex=r"Z = \operatorname{Tr}\, e^{-\beta H} = \sum_i m_i\, e^{-\beta \lambda_i}",
        established=True,
        how="spectral decomposition of a Hermitian matrix; m_i is the multiplicity of the eigenvalue",
    )
    problem.conditions = [cond_h, cond_z]
    problem.setup_steps = [
        StatStep(
            name="levels",
            title="Hamiltonian",
            latex=rf"H = {to_latex(H)}",
        ),
        StatStep(
            name="spectrum",
            title="Eigenvalues and multiplicities",
            latex=r",\quad ".join(
                rf"\lambda_{{{i}}} = {to_latex(e)}\ (\times {to_latex(g)})"
                for i, (e, g) in enumerate(zip(energies, degeneracies, strict=True), 1)
            ),
        ),
        StatStep(
            name="partition_function",
            title="Partition function",
            latex=rf"Z = \operatorname{{Tr}}\,e^{{-{to_latex(beta)} H}} = {to_latex(z1)}",
            conditions=[cond_h, cond_z],
        ),
    ]
    return {s for s in H.free_symbols if isinstance(s, sp.Symbol)} | {
        s for m in observables.values() for s in m.free_symbols if isinstance(s, sp.Symbol)
    }


def thermal_matrix(problem: Problem) -> tuple[sp.Matrix, Condition]:
    """exp(-beta H) by Sylvester's formula over the distinct eigenvalues, and the condition it needs."""
    H: sp.Matrix = problem.extra["H"]
    n = H.shape[0]
    values = problem.energies
    b = problem.beta
    total = sp.zeros(n, n)
    for i, li in enumerate(values):
        projector = sp.eye(n)
        for j, lj in enumerate(values):
            if j != i:
                projector = projector * (H - lj * sp.eye(n)) / (li - lj)
        total += sp.exp(-b * li) * projector
    condition = Condition(
        latex=r"\lambda_i \ne \lambda_j",
        established=False,
        how=(
            "Sylvester's formula divides by eigenvalue differences: at parameter values where two "
            "distinct eigenvalues coincide it is 0/0 and the result holds by continuity only"
        ),
    )
    return total.applyfunc(sp.simplify), condition


class QuantumOracle(BaseOracle):
    """Matrix exponentials, traces and eigendecompositions of the numerical Hamiltonian."""

    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        x = problem.extra
        self._symbols = sorted(
            {s for m in [x["H"], *x["matrices"].values()] for s in m.free_symbols},
            key=lambda s: s.name,
        )
        self._H = sp.lambdify(self._symbols, x["H"], modules="mpmath")
        self._ops = {
            name: sp.lambdify(self._symbols, m, modules="mpmath")
            for name, m in x["matrices"].items()
        }
        self.dim: int = x["dim"]

    def hamiltonian(self, point: Point) -> mp.matrix:
        return mp.matrix(self._H(*(point[s] for s in self._symbols)))

    def operator(self, name: str, point: Point) -> mp.matrix:
        return mp.matrix(self._ops[name](*(point[s] for s in self._symbols)))

    def thermal(self, point: Point, beta: Any) -> tuple[mp.matrix, Any]:
        """The unnormalised thermal matrix exp(-beta H) and its trace Z (real for Hermitian H)."""
        e = mp.expm(-beta * self.hamiltonian(point))
        z = sum(e[i, i] for i in range(self.dim))
        return e, mp.re(z)

    def lnZ1(self, point: Point, beta: Any) -> Any:
        return mp.log(self.thermal(point, beta)[1])

    def lnZ_spectral(self, point: Point, beta: Any) -> Any:
        """ln of the sum of exp(-beta lambda) over mpmath's eigenvalues of H (a second numerical route)."""
        eigenvalues = mp.eigh(self.hamiltonian(point), eigvals_only=True)
        return mp.log(sum(mp.exp(-beta * lam) for lam in eigenvalues))

    def moment(self, point: Point, beta: Any, matrix: mp.matrix, order: int) -> Any:
        """Tr(rho A^k): the operator stays where it is in the product, nothing is reordered."""
        e, z = self.thermal(point, beta)
        power = mp.eye(self.dim)
        for _ in range(order):
            power = power * matrix
        product = e * power / z
        return sum(product[i, i] for i in range(self.dim))

    def cumulants1(self, point: Point, beta: Any, upto: int) -> list[Any]:
        h = self.hamiltonian({**point, self.problem.beta: beta})
        raw = [mp.mpf(1), *[mp.re(self.moment(point, beta, h, k)) for k in range(1, 5)]]
        return cumulants_from_raw(raw, upto)

    def observable_cumulant(self, name: str, point: Point, order: int) -> Any:
        """Cumulant of a matrix observable under the thermal state (real part; A is checked Hermitian)."""
        b = point[self.problem.beta]
        a = self.operator(name, point)
        raw = [mp.mpf(1), *[self.moment(point, b, a, k) for k in range(1, 5)]]
        return cumulants_from_raw(raw, order)[order - 1]

    def quantity(self, name: str, point: Point, *, beta: Any | None = None) -> Any:
        if name == "S" and self.problem.copies_count is None:
            # von Neumann entropy -k_B Tr(rho ln rho) from the eigenvalues of the numerical rho
            p = point if beta is None else {**point, self.problem.beta: beta}
            with mp.workdps(self.precision):
                b = p[self.problem.beta]
                e, z = self.thermal(p, b)
                eigenvalues = mp.eigh(e / z, eigvals_only=True)
                return -p[self.problem.k_B] * sum(w * mp.log(w) for w in eigenvalues if w > 0)
        return super().quantity(name, point, beta=beta)
