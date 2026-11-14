"""Method recommendations derived from the structure of the problem, never from a model's guess.

Each suggestion says whether it applies to this input, whether Minerva implements it, and why. A
method that applies but is not implemented is listed as such, so the user knows it exists without
the tool pretending to run it.
"""

from __future__ import annotations

from .models import MethodSuggestion
from .setup import Problem

# Methods Minerva can run today. Later phases add to this set as they land.
IMPLEMENTED: frozenset[str] = frozenset(
    {
        "ratio_test_sum",
        "dos_integral",
        "phase_space_factorisation",
        "spectral_decomposition",
        "closed_form_sum",
        "enumeration_check",
        "derivative_route",
        "fluctuation_response",
    }
)

NOT_SUPPORTED: tuple[str, ...] = (
    "Grand canonical and microcanonical ensembles",
    "Quantum systems larger than 4x4 or without closed-form eigenvalues",
    "Asymptotic expansions, saddle-point methods and transfer matrices",
    "Interacting classical systems, Monte Carlo sampling and many-body solvers",
)


def _structure(problem: Problem) -> MethodSuggestion | None:
    """The method that follows from the kind of system, with the reason it applies."""
    kind = problem.kind
    if kind == "quantum":
        return MethodSuggestion(
            id="spectral_decomposition",
            title="Spectral decomposition of a Hermitian matrix",
            applies=True,
            implemented=True,
            why="H is an explicit Hermitian matrix, so Z = Tr exp(-beta H) is the sum of exp(-beta "
            "lambda) over its eigenvalues; a thermal average Tr(A exp(-beta H))/Z keeps every operator "
            "in place, and [A, H] = 0 is checked, never assumed",
        )
    if kind == "level_family":
        return MethodSuggestion(
            id="ratio_test_sum",
            title="Infinite level sum with a convergence proof",
            applies=True,
            implemented=True,
            why="the levels form a family E(n): a geometric closed form or SymPy's sum is accepted "
            "only after the ratio test shows convergence for every beta > 0",
        )
    if kind == "density_of_states":
        return MethodSuggestion(
            id="dos_integral",
            title="Density-of-states integral",
            applies=True,
            implemented=True,
            why="Z is the Laplace transform of rho(E): evaluated by the verified definite integrator "
            "for E^k rho e^(-beta E), k = 0, 1, 2, which also justifies differentiating under the integral",
        )
    if kind == "classical":
        return MethodSuggestion(
            id="phase_space_factorisation",
            title="Factorisation of a separable Hamiltonian",
            applies=True,
            implemented=True,
            why="H splits term by term into one-variable parts, so the phase-space integral is a "
            "product of one-variable integrals; the cell and N! are your conventions",
        )
    return None


def suggest(problem: Problem) -> list[MethodSuggestion]:
    n = len(problem.energies)
    has_observables = any(name != "E" for name in problem.observables) or bool(
        problem.extra.get("matrices")
    )
    out = [
        MethodSuggestion(
            id="closed_form_sum",
            title="Closed-form partition function",
            applies=problem.kind in {"levels", "quantum"} or problem.Z1 is not None,
            implemented="closed_form_sum" in IMPLEMENTED,
            why=(
                f"the system is a list of {n} levels, so Z is a finite sum of exponentials that "
                "always converges and can be differentiated term by term"
                if problem.kind == "levels"
                else (
                    "Z has a closed form for this system, so derivatives of ln Z are taken exactly"
                    if problem.Z1 is not None or problem.kind == "quantum"
                    else "no closed form was found: only numerical evaluation is available"
                )
            ),
        ),
        MethodSuggestion(
            id="enumeration_check",
            title="Independent enumeration of the levels",
            applies=True,
            implemented="enumeration_check" in IMPLEMENTED,
            why="every formula is compared with a direct log-sum-exp sum over the levels at sampled "
            "parameter values",
        ),
        MethodSuggestion(
            id="derivative_route",
            title="Thermodynamics from derivatives of ln Z",
            applies=True,
            implemented="derivative_route" in IMPLEMENTED,
            why="U, F, S, the heat capacity and generalised forces follow from derivatives of ln Z "
            "with the other state variables held fixed",
        ),
        MethodSuggestion(
            id="fluctuation_response",
            title="Fluctuation-response relations",
            applies=True,
            implemented="fluctuation_response" in IMPLEMENTED,
            why=(
                "observables are given on the levels; a cumulant of one of them equals a derivative "
                "of ln Z if the energies depend linearly on its conjugate parameter"
                if has_observables
                else "the energy cumulants are derivatives of ln Z with respect to beta; give "
                "observable values on the levels to study other fluctuations"
            ),
        ),
        MethodSuggestion(
            id="high_temperature_expansion",
            title="High-temperature expansion",
            applies=True,
            implemented="high_temperature_expansion" in IMPLEMENTED,
            why="beta -> 0 expansion of ln Z about the infinite-temperature limit",
        ),
        MethodSuggestion(
            id="low_temperature_limit",
            title="Low-temperature limit",
            applies=n > 1,
            implemented="low_temperature_limit" in IMPLEMENTED,
            why="beta -> infinity is dominated by the lowest level(s) when the ordering of the "
            "energies is decided"
            if n > 1
            else "a single level has no temperature dependence beyond the factor e^{-beta E}",
        ),
        MethodSuggestion(
            id="transfer_matrix",
            title="Transfer matrix",
            applies=False,
            implemented="transfer_matrix" in IMPLEMENTED,
            why="needs a lattice or chain with nearest-neighbour couplings, not a list of levels",
        ),
    ]
    first = _structure(problem)
    return [first, *out] if first is not None else out
