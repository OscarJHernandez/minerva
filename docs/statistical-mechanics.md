# Statistical mechanics

Minerva builds the canonical-ensemble partition function of an explicit system and derives thermodynamic
quantities from it. The language model chooses actions; the engine computes, and a formula is shown only if
an independent numerical evaluation agreed. This page describes what is covered, how each result is checked,
what only the user can decide, and what is not supported.

Where: the workbench mode "Statistical mechanics", the chat tool `statistical_mechanics`, the endpoint
`POST /v1/compute/statmech`, and `Engine.statmech` (`packages/engine/src/minerva_engine/statmech/`).
Examples that CI runs against the real engine are in [capabilities.md](capabilities.md); the benchmark is in
[statmech-benchmark.md](statmech-benchmark.md).

## Systems

| Kind | You give | Z comes from |
| --- | --- | --- |
| `levels` | energies, degeneracies, optional observable values per level | the finite sum `sum_i g_i exp(-beta E_i)` |
| `level_family` | `E(n)`, `g(n)`, an integer range (finite or `oo`) | a geometric closed form, else `sympy.summation`, else an unevaluated sum |
| `density_of_states` | `rho(En)` on `[lower, upper]` | the verified definite integrator, once per moment |
| `classical` | a Hamiltonian, the phase-space variables with their domains | one verified integral per variable, after an exact additive split of H |

Any of them can be taken as `N` independent copies (`copies`). Temperatures are written in `beta` or `T`;
`beta`, `T` and `k_B` are positive by the definition of an ensemble temperature. Energies must be provably
real and degeneracies provably positive integers under the declared assumptions, otherwise the request is
refused (`assumptions_insufficient`, `invalid_degeneracy`).

Text syntax reserves `E` (Euler's number): call a density-of-states energy `En`.

## Actions

- `analyze`: which methods apply to this structure and which of them Minerva implements, each with the reason,
  and a list of what is not supported. Applies-but-not-implemented is shown as such.
- `partition`: the system, `Z`, and the probability of each level (normalisation checked).
- `derive`: `Z`, `ln Z`, `F`, `U`, `S`, `Var(E)` and any number of **partial derivatives** of those with a
  stated held-fixed set. The route is generic: `U = -d ln Z/d beta`, `Var(E) = d^2 ln Z/d beta^2`,
  `S = k_B (ln Z + beta U)`, heat capacity `dU/dT`, pressure `-dF/dV` at fixed temperature, a susceptibility
  a field derivative. No quantity is a stored formula for a particular system.
- `expectation`: mean, raw and central moments, variance, covariance and cumulants of the energy or of an
  observable given per level. For the energy the cumulants are `(-1)^n d^n ln Z/d beta^n`; for an observable
  with a declared linear coupling `E_i(h) = E_i(0) - h A_i` they are `beta^-n d^n ln Z/d h^n`.
- `evaluate`: numbers at values the user supplies, and sweeps over one variable (a table and a plot).

### Held-fixed sets

The state variables of the canonical ensemble are the temperature variable and the declared `parameters`.
A derivative with respect to one of them must hold exactly the others fixed, and the held-fixed set is part
of the request and of the result. Anything else (a derivative at fixed energy or entropy, a missing or extra
variable) is refused (`ensemble_mismatch`): such a derivative needs a change of variables that is not done.

## How a result is checked

Each displayed formula is evaluated at assumption-respecting sample points (exact rationals, then 50-digit
mpmath numbers) and compared with an **independent oracle** that never uses the route that produced it:

- a finite or infinite level sum: direct enumeration of the terms with log-sum-exp and a tail test;
- a density of states or a phase-space factor: adaptive quadrature with an error estimate (a quadrature that
  does not reach its accuracy is a refusal, not a pass);
- derivatives: fourth-order central differences of the **enumerated** quantity, never of the formula, plus,
  where a spectrum exists, the direct moments `U = sum E p`, `Var(E) = sum E^2 p - U^2`;
- the entropy against the Gibbs sum over microstates and against `-dF/dT` taken numerically;
- `F = U - T S` on the enumerated values, normalisation of the probabilities, non-negative variances.

A symbolic-zero leg (the derived formula minus the direct moment simplifies to zero) is reported separately
and only where SymPy finds one; an undecided leg is "undecided", never a refutation, and "proved" is used for
nothing else. Numerical agreement is evidence, not a proof, and the interface says so.

If any deciding check disagrees, nothing is shown (`verification_failed`). The test-suite contains a formula
that is off by one part in 10^9 and must be rejected.

## What is never assumed

- **Convergence.** An infinite sum is accepted only if the ratio test, applied to the terms and to `E^2` times
  the terms, gives a limit below 1 for every `beta > 0`; that also justifies differentiating term by term.
  A limit above 1 is `divergent`; an inconclusive limit (for example `E_n = ln(n+1)`, terms `n^-beta`) is
  `convergence_not_provable`. An integral must be evaluated by the verified integrator under the declared
  assumptions; a value that needs conditions the assumptions do not imply is `convergence_not_provable`, and
  SymPy's report of divergence is passed on as "not verified", never as a proof of divergence. The energy must
  be bounded below, so that differentiating under the integral is justified (dominated convergence).
- **A phase-space cell.** `Z_1 = (1/c) integral exp(-beta H) dGamma` with `c` chosen by the user (`h^3`, `1`).
  Without it the result is `convention_unconfirmed` and carries the question.
- **Indistinguishability.** `Z = Z_1^N / N!` or `Z = Z_1^N` is the user's choice. The first is the
  classical-limit counting, **not** exact Bose or Fermi statistics, and the notes say so.
- **Separability.** A Hamiltonian is split term by term into one-variable parts; a term that couples two
  variables is refused (`not_separable`). No normal-mode change of variables is attempted.
- **Closed forms.** A series or integral without one is evaluated numerically and labelled `numeric`; that is
  never presented as an identity, and symbolic derivation of it is refused.
- **Linear response.** The fluctuation-response relation needs `dE_i/dh = -A_i` on every level with `A_i`
  independent of `h`; checked symbolically, otherwise `coupling_not_linear`.

In the agent, the phase-space cell, the indistinguishability flag, the numbers of an evaluation and a sweep are
not arguments of the tool: the model cannot supply them. They arrive through the card's buttons and form
(`POST /v1/chat/conversations/{id}/tools/statistical_mechanics`) and are stored next to the call.

## Refusal reasons

`divergent`, `convergence_not_provable`, `invalid_degeneracy`, `assumptions_insufficient`, `not_separable`,
`convention_unconfirmed`, `ensemble_mismatch`, `coupling_not_linear`, `unknown_observable`,
`verification_failed`, `unsupported_system`, `resource_limit`, `parse_error`. Each maps to a status; none
produces a formula. Logs carry stage names, counts and these reasons only, never an energy or an expression
(a test enforces it).

## Not supported

- Grand canonical and microcanonical ensembles, and Bose–Einstein or Fermi–Dirac systems of occupation numbers.
- Quantum systems given by a Hamiltonian matrix (`Tr exp(-beta H)`, density matrices, non-commuting operators).
- High- and low-temperature expansions, Stirling's approximation, saddle-point and Laplace methods, Legendre
  transforms, mean field, transfer matrices and thermodynamic limits.
- Interacting classical systems, Monte Carlo and many-body solvers.

`analyze` reports these as not implemented. The handbook already records the relations behind them
(`statmech.*`, `physics.thermodynamics`), as `stated`, `dimensional` or `numeric` entries.

## Where the pieces are

| Piece | Module |
| --- | --- |
| request, result, reasons | `statmech/models.py` |
| parsing and validation, conventions | `statmech/setup.py` |
| level families, densities of states, classical systems and their oracles | `statmech/systems.py` |
| quantities, held-fixed derivatives, expectation values, fluctuation-response | `statmech/canonical.py` |
| the independent numerical oracles and finite differences | `statmech/oracle.py` |
| evaluation and sweeps | `statmech/numeric.py` |
| method recommendations | `statmech/analyze.py` |
| the benchmark | `statmech/benchmark.py` |

The corpus (`make statmech-benchmark`) holds closed-form references (two-level system, Schottky heat
capacity, spin-1/2, quantum and classical oscillators, ideal gas with the exact `ln N!`, densities of states,
derivative identities, fluctuation-response, independent copies, overflow-prone energies, a sweep) and
refusals that must carry a named reason. Five of its problems are also rows of the reference benchmark.
