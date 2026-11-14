# Statistical mechanics

Minerva builds the partition function of an explicit system and derives thermodynamic quantities from it. The
language model chooses actions; the engine computes, and a formula is shown only if an independent numerical
evaluation agreed. This page describes what is covered, how each result is checked, what only the user can
decide, and what is not supported.

Where: the workbench mode "Statistical mechanics", the chat tool `statistical_mechanics`, the endpoint
`POST /v1/compute/statmech`, and `Engine.statmech` (`packages/engine/src/minerva_engine/statmech/`).
Examples that CI runs against the real engine are in [capabilities.md](capabilities.md); the benchmark is in
[statmech-benchmark.md](statmech-benchmark.md).

## Systems and ensembles

| Kind | You give | Ensemble | Z comes from |
| --- | --- | --- | --- |
| `levels` | energies, degeneracies, optional observable values per level | canonical (or microcanonical, below) | the finite sum `sum_i g_i exp(-beta E_i)` |
| `level_family` | `E(n)`, `g(n)`, an integer range (finite or `oo`) | canonical | a geometric closed form, else `sympy.summation`, else an unevaluated sum |
| `density_of_states` | `rho(En)` on `[lower, upper]` | canonical | the verified definite integrator, once per moment |
| `classical` | a Hamiltonian, the phase-space variables with their domains | canonical | one verified integral per variable, after an exact additive split of H |
| `quantum` | an explicit Hermitian matrix H (at most 4x4), optional operator matrices | canonical | the spectral decomposition: eigenvalues with multiplicities |
| `occupation` | single-particle levels, a statistic (`bose`, `fermi`, `boltzmann`) | grand canonical | ln Xi of the statistic, per single-particle state |
| `chain` | a periodic chain: site values, a symmetric bond, a site energy, a length N | canonical | the transfer matrix: `Tr T^N = sum m_i lambda_i^N` |

Any canonical system can be taken as `N` independent copies (`copies`). Temperatures are written in `beta` or
`T`; `beta`, `T` and `k_B` are positive by the definition of an ensemble temperature. Energies must be provably
real and degeneracies provably positive integers under the declared assumptions, otherwise the request is
refused (`assumptions_insufficient`, `invalid_degeneracy`).

An ensemble is never substituted for another: an occupation system with a chemical potential must be given as
grand canonical, the grand canonical ensemble needs an occupation system, microcanonical counting needs a list
of levels, and a total energy belongs to the microcanonical ensemble only (`ensemble_mismatch`).

Text syntax reserves `E` (Euler's number): call a density-of-states energy `En`.

## Actions

- `analyze`: which methods apply to this structure and which of them Minerva implements, each with the reason,
  and a list of what is not supported. Applies-but-not-implemented is shown as such.
- `partition`: the system, `Z` (or ln Xi), the probability of each level and, for particles, the mean occupation
  of each state (normalisation and enumeration checked).
- `derive`: `Z`, `ln Z`, `F`, `U`, `S`, `Var(E)` (grand canonical: `Xi`, `ln Xi`, `Omega`, `N`, `U`, `S`,
  `Var(N)`) and any number of **partial derivatives** of those with a stated held-fixed set. The route is
  generic: `U = -d ln Z/d beta`, `Var(E) = d^2 ln Z/d beta^2`, `S = k_B (ln Z + beta U)`, heat capacity `dU/dT`,
  pressure `-dF/dV` at fixed temperature, a susceptibility a field derivative; for particles
  `N = (1/beta) d ln Xi/d mu`. No quantity is a stored formula for a particular system.
- `expectation`: mean, raw and central moments, variance, covariance and cumulants of the energy, of an
  observable given per level, of a matrix operator, or of the particle number. For the energy the cumulants are
  `(-1)^n d^n ln Z/d beta^n`; for an observable with a declared linear coupling `E_i(h) = E_i(0) - h A_i` they
  are `beta^-n d^n ln Z/d h^n`; for the particle number `beta^-n d^n ln Xi/d mu^n`.
- `evaluate`: numbers at values the user supplies, and sweeps over one variable (a table and a plot).
- `approximate`: an approximation named in `method`, below. Microcanonical counting is requested with
  `ensemble: microcanonical`, `copies` and `total_energy`.

### Held-fixed sets

The state variables of an ensemble are the temperature variable, the chemical potential (grand canonical) and
the declared `parameters`. A derivative with respect to one of them must hold exactly the others fixed, and the
held-fixed set is part of the request and of the result. Anything else (a derivative at fixed energy or
entropy, a missing or extra variable, `dU/dT` that lets mu vary) is refused (`ensemble_mismatch`): such a
derivative needs a change of variables that is not done.

## How a result is checked

Each displayed formula is evaluated at assumption-respecting, generic (non-zero) sample points (exact
rationals, then 50-digit mpmath numbers) and compared with an **independent oracle** that never uses the route
that produced it:

- a finite or infinite level sum: direct enumeration of the terms with log-sum-exp and a tail test;
- a density of states or a phase-space factor: adaptive quadrature with an error estimate (a quadrature that
  does not reach its accuracy is a refusal, not a pass);
- a quantum system: `mpmath.expm` of the numerical matrix, its trace, and `mpmath.eigh`, never the symbolic
  eigenvalues; the entropy is the von Neumann entropy of the numerical density matrix;
- particles: the occupation numbers of every single-particle state are enumerated (0 or 1 for fermions, 0, 1,
  2, ... until negligible for bosons, a Poisson weight for classical counting) and the entropy is the Gibbs
  entropy of that distribution (with the Gibbs correction `-sum p ln(p n!)` for classical counting, where the
  weight carries `1/n!`);
- a chain: brute-force enumeration of every one of the `k^N` configurations of the ring, never a transfer matrix;
- microcanonical counts: the generating function against brute force over all assignments (when small) and an
  enumeration over occupation vectors with multinomial weights;
- derivatives: fourth-order central differences of the **enumerated** quantity, never of the formula, plus,
  where a spectrum exists, the direct moments `U = sum E p`, `Var(E) = sum E^2 p - U^2`;
- the entropy against `-dF/dT` taken numerically; `F = U - T S` (`Omega = U - T S - mu N`) on the enumerated
  values, normalisation of the probabilities, non-negative variances.

A symbolic-zero leg (the derived formula minus the direct moment simplifies to zero) is reported separately and
only where SymPy finds one; an undecided leg is "undecided", never a refutation, and "proved" is used for nothing
else. Numerical agreement is evidence, not a proof, and the interface says so.

If any deciding check disagrees, nothing is shown (`verification_failed`). The test-suite contains a formula that
is off by one part in 10^9 and must be rejected, and a benchmark judge that cannot pass a wrong answer.

## What is never assumed

- **Convergence.** An infinite sum is accepted only if the ratio test, applied to the terms and to `E^2` times
  the terms, gives a limit below 1 for every `beta > 0`; that also justifies differentiating term by term.
  A limit above 1 is `divergent`; an inconclusive limit (for example `E_n = ln(n+1)`, terms `n^-beta`) is
  `convergence_not_provable`. An integral must be evaluated by the verified integrator under the declared
  assumptions; a value that needs conditions the assumptions do not imply is `convergence_not_provable`, and
  SymPy's report of divergence is passed on as "not verified", never as a proof of divergence. The energy must
  be bounded below, so that differentiating under the integral is justified (dominated convergence). Bose
  statistics needs mu provably below every single-particle energy.
- **A phase-space cell.** `Z_1 = (1/c) integral exp(-beta H) dGamma` with `c` chosen by the user (`h^3`, `1`).
  Without it the result is `convention_unconfirmed` and carries the question.
- **Indistinguishability.** `Z = Z_1^N / N!` or `Z = Z_1^N` is the user's choice, for copies and for
  microcanonical counting. The first is the classical-limit counting, **not** exact Bose or Fermi statistics.
- **Separability.** A Hamiltonian is split term by term into one-variable parts; a term that couples two
  variables is refused (`not_separable`). No normal-mode change of variables is attempted.
- **Closed forms.** A series or integral without one is evaluated numerically and labelled `numeric`; that is
  never presented as an identity, and symbolic derivation of it is refused.
- **Commutation.** Matrices are never reordered and never assumed to commute. `Tr(A exp(-beta H))` keeps `A`
  where it stands; Hermiticity and `[A, H] = 0` are checked and reported as conditions. The first derivative of
  ln Z with respect to a coupled parameter is `beta` times the mean by trace cyclicity, but a second derivative
  is the Kubo-Duhamel covariance, not `beta^2 Var(A)`, unless `[A, H] = 0`: otherwise the response relation is
  refused (`operators_do_not_commute`), and so is the covariance of two operators that do not commute. The
  formula for `exp(-beta H)` divides by differences of distinct eigenvalues, and says so.
- **Ensemble equivalence.** A finite microcanonical system has a discrete spectrum, so its `1/T = dS/dE` is a
  finite difference over neighbouring attainable energies, labelled as such; its agreement with the canonical
  ensemble is *measured* (the gap shrinks with N) and never assumed.
- **Linear response.** The fluctuation-response relation needs `dE_i/dh = -A_i` with `A_i` independent of `h`
  (`coupling_not_linear` otherwise).

In the agent, the phase-space cell, the indistinguishability flag, the numbers of an evaluation and a sweep are
not arguments of the tool: the model cannot supply them. They arrive through the card's buttons and form
(`POST /v1/chat/conversations/{id}/tools/statistical_mechanics`) and are stored next to the call.

## Approximations

An approximation is not an identity. Every `approximate` result is labelled `asymptotic` (or an exact limit),
and reports its **assumptions**, **expansion parameter**, the **order of accuracy** it claims, its **validity**
and a **measured error table** against the independent oracle. The error table is a measurement, never a bound,
and the claimed order is a deciding check: a truncation whose error does not fall at the claimed rate is not
shown.

| Method | What it does | Checked by | Refused when |
| --- | --- | --- | --- |
| `high_temperature` | Taylor series of ln Z in beta about 0 to order N, and U from it | `check_series` (Cauchy-integral coefficients, remainder falling at the claimed order) and the enumerated ln Z at beta, beta/2, beta/4 | Z is not finite and positive at beta = 0 (an oscillator, a density of states): `singular_expansion_point` |
| `low_temperature` | ground level plus the first Boltzmann corrections of the excited levels; the limits U to E0 and S to k_B ln g_0 are measured | relative error within `2 (1 + sum g_j/g_0) e^(-beta gap)` and falling, at beta times the gap = 3, 6, 12 | the order of the levels is not decided from the assumptions: `ordering_not_decided` |
| `stirling` | ln N! replaced by `N ln N - N + 1/2 ln(2 pi N)`; F and S re-derived; the leading form of an ideal gas is Sackur-Tetrode | `N times the error` tends to 1/12 at N = 100, 1000, 10000, and the enumerated ln Z differs from the substituted one by exactly the Stirling error | there is no ln N! (no indistinguishable copies): `nothing_to_approximate` |
| `thermodynamic_limit` | free energy and energy per site of a chain from the dominant transfer-matrix eigenvalue | brute-force `F_N/N` for N = 2 ... 12 equals the limit plus `-(1/(beta N)) ln(1 + sum (lambda_i/lambda_max)^N)` | the length is not a symbol N, or the dominant eigenvalue is not decided |

Perron-Frobenius justifies the chain: the transfer matrix has positive entries, so its largest eigenvalue is
simple and strictly dominant, and the free energy per site is analytic in beta (no phase transition at finite
temperature in one dimension).

## Refusal reasons

`divergent`, `convergence_not_provable`, `invalid_degeneracy`, `assumptions_insufficient`, `not_separable`,
`convention_unconfirmed`, `ensemble_mismatch`, `coupling_not_linear`, `unknown_observable`, `not_hermitian`,
`operators_do_not_commute`, `energy_not_attainable`, `singular_expansion_point`, `ordering_not_decided`,
`nothing_to_approximate`, `verification_failed`, `unsupported_system`, `resource_limit`, `parse_error`. Each maps
to a status (a test requires every reason to have one); none produces a formula. Logs carry stage names,
counts and these reasons only, never an energy or an expression (a test enforces it).

## Not supported

- Saddle-point and Laplace methods, Legendre transformations and mean-field approximations (the brief's
  remaining approximation methods); a general thermodynamic-limit analysis beyond the periodic chain.
- Quantum systems larger than 4x4 or without closed-form eigenvalues, density matrices as matrix-valued
  results, and many-body quantum solvers.
- Interacting classical systems, Monte Carlo, open boundary conditions for the chain, transfer matrices of
  more than three states or with non-symmetric bonds.
- Bose-Einstein condensation and other phase-transition statements: nothing here claims one.

`analyze` reports the unsupported methods as not implemented. The handbook already records the relations behind
them (`statmech.*`, `physics.thermodynamics`), as `stated`, `dimensional` or `numeric` entries.

## Where the pieces are

| Piece | Module |
| --- | --- |
| request, result, reasons | `statmech/models.py` |
| parsing and validation, conventions, ensemble matching | `statmech/setup.py` |
| level families, densities of states, classical systems and their oracles | `statmech/systems.py` |
| quantum matrices and their oracle | `statmech/quantum.py` |
| particles in the grand canonical ensemble | `statmech/grand.py` |
| microcanonical counting | `statmech/micro.py` |
| the periodic chain (transfer matrix) and its brute-force oracle | `statmech/lattice.py` |
| quantities, held-fixed derivatives, expectation values, fluctuation-response | `statmech/canonical.py` |
| the independent numerical oracles and finite differences | `statmech/oracle.py` |
| evaluation and sweeps | `statmech/numeric.py` |
| approximations | `statmech/approximate.py` |
| method recommendations | `statmech/analyze.py` |
| the benchmark | `statmech/benchmark.py` |

The corpus (`make statmech-benchmark`) holds closed-form references (two-level system, Schottky heat capacity,
spin-1/2, quantum and classical oscillators, ideal gas with the exact `ln N!`, densities of states, derivative
identities, fluctuation-response, a spin in a tilted field with a non-commuting operator, Bose and Fermi
occupation thermodynamics, binomial microstate counts, Ising rings against brute force, textbook series,
Sackur-Tetrode) and refusals that must carry a named reason. Five of its problems are also rows of the
reference benchmark.
