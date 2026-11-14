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
        "closed_form_sum",
        "enumeration_check",
        "derivative_route",
        "fluctuation_response",
    }
)

NOT_SUPPORTED: tuple[str, ...] = (
    "Grand canonical and microcanonical ensembles",
    "Infinite level sums and density-of-states integrals",
    "Classical phase-space integrals",
    "Quantum systems given by a Hamiltonian matrix",
    "Asymptotic expansions, saddle-point methods and transfer matrices",
    "Monte Carlo sampling and many-body solvers",
)


def suggest(problem: Problem) -> list[MethodSuggestion]:
    n = len(problem.energies)
    has_observables = any(name != "E" for name in problem.observables)
    out = [
        MethodSuggestion(
            id="closed_form_sum",
            title="Closed-form partition function",
            applies=True,
            implemented="closed_form_sum" in IMPLEMENTED,
            why=f"the system is a list of {n} levels, so Z is a finite sum of exponentials that "
            "always converges and can be differentiated term by term",
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
    return out
