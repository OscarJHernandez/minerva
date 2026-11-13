# Discovering conserved quantities

`discover_invariants` (tool `find_conserved_quantities`, endpoint `POST /v1/compute/conserved`, workbench page
`/workbench/conserved`) takes a Lagrangian or a Hamiltonian and returns the conserved quantities it can establish, which
of them are functionally independent, and exactly what it searched. It does not know any physical system: the
Kepler problem, oscillators and the rest are regression benchmarks, never special cases.

## Method

1. **Equations of motion.** Lagrangian: Euler–Lagrange solved for the accelerations (`lagrangian.derive`), confirmed
   by numerical differentiation of L. Hamiltonian: `dq/dt = ∂H/∂p`, `dp/dt = −∂H/∂q`, confirmed by numerical
   differentiation of H (`flow.hamilton_check`); H must be regular (invertible Hessian in the momenta). Both become one
   first-order *flow* (`flow.Flow`), and `dI/dt = {I, H} + ∂I/∂t` is the total derivative along it.
2. **A ladder of families.** `I = Σ cᵢ φᵢ`, with φ = (function of the coordinates) × (monomial of degree ≤ d in the
   velocities or momenta) × tʲ. The coordinate functions are powers q^a (negative only for a coordinate that occurs in a
   denominator of the flow), cos/sin up to a harmonic for an angle, and powers of a cyclic coordinate. A coordinate is
   an *angle* if declared, if the equations contain it only through cos and sin, or if it is cyclic: conservation does
   not depend on periodicity, so trigonometric functions of a cyclic coordinate are always a legitimate family. The
   ladder is a fixed schedule (degree, power, power, cyclic power, degree, harmonics, power, degree, power, cyclic power,
   t, t), each step skipped if the family would exceed the unknown cap (a refused degree step is retried with a smaller
   power bound). It is deterministic and does not depend on the model.
3. **Exact determining equations.** The derivative of every basis function is cleared of denominators and expanded once
   (cached across families, with cos²+sin²=1 reduced); the coefficients of each monomial are linear equations in `cᵢ`,
   solved by an exact nullspace over the rational functions of the parameters. No floating point.
4. **No trivial discoveries.** The nullspace is taken modulo the algebra of the constants already known (cyclic momenta,
   energy, rotations) or found: products of up to four of them, including combinations of higher degree that cancel down
   into the family. Directions are taken one at a time, simplest first, so `x − p t/m` is reported and its multiples are
   not.
5. **Verification.** Every direction is verified on the equations of motion, symbolically and at random numerical
   points. A failure is dropped and counted (`cause: verification_failed` when nothing else was found).
6. **Independence** (`independence.py`). The Jacobian of the constants with respect to the phase variables is evaluated
   exactly at rational points (angles through `cos = (1−u²)/(1+u²)`), a *proved lower bound* of the generic rank. Exact
   polynomial relations among the constants are found by a nullspace over their monomials; a relation whose gradient does
   not vanish bounds the rank from above (implicit function theorem). When the bounds meet the rank is proved; otherwise
   the report says which part is. Sampling is never used as proof.

## Completeness

A family's result is complete **within that family**: the nullspace is its whole solution space. Nothing is claimed
outside it. The report lists the families searched, those refused for size (`unexplored`), the stop reason
(`ladder_exhausted`, `unknown_cap`, `time_budget`, `maximal_independent_set`, `unsupported`) and, when nothing new was
found, a `cause` that separates an exhausted ansatz from an unsupported function class, a verification failure and a
resource limit. Parameters are generic symbols: constants that exist only at special values (resonant frequency ratios)
are found only when those values are given, and the report says so.

Budgets: `quick` (8 s, 150 unknowns, stops once the maximum number of independent constants is reached), `standard`
(30 s, 350), `thorough` (90 s, 600, adds powers of t for autonomous systems; time-dependent systems always get them).
Advanced callers can pin one family (`degree`, `power_bound`, `harmonics`, `time_degree`).

## Not supported (reported, never guessed)

Square roots and other non-rational equations of motion (Cartesian Kepler is `unsupported` until an adjoined algebraic
element `ρ² = x²+y²` is implemented), logarithms, non-polynomial time dependence, singular Hessians, more than three
coordinates, arbitrary Killing-tensor PDEs. Naming a constant (Laplace–Runge–Lenz, Fradkin) is left to the model after
the search; nothing in the search uses a name.

## Logging

The sandboxed search returns `events`: `invariant_search.started`, `family_generated`, `determining_equations`,
`nullspace_computed`, `candidate_found`, `verification_completed`, `independence_checked`, `family_expanded`,
`completed`, each with counters and fixed labels only (family code, unknowns, equations, matrix shape, rank, nullity,
stage milliseconds, accepted/rejected, stop reason, peak memory). The API logs them one line each with the request id as
the trace id (`search_log.py`, allow-listed fields). Expressions never reach the server log; they stay in the card kept
with the user's private conversation.

## Benchmarks

`packages/engine/tests/bench` builds Hamiltonians with independently known invariants, entirely in SymPy and sharing no
code with discovery: curated systems (free particle, oscillators incl. resonant 1:2 and 1:3, separable quartic,
inverse-square, central quartic, Cartesian Kepler) and systems constructed by design (separable sums, functions of
commuting invariants), each with seeded random exact parameters. A seeded exact symplectic map built from elementary
factors (point, shear, scaling, swap) is applied to H and to the invariants; at generation the map is checked
against the canonical brackets of the coordinates and `{I, H} = 0` is verified exactly for every transformed invariant.
Each reference invariant carries a descriptor (momentum degree, power range, terms) and its minimal families, found by
exact linear algebra (not assumed). Minerva is given the Hamiltonian and the variable names only
(`EngineInput`).

- `tests/test_discovery_generated.py` (CI, fixed seeds, `fixtures/fixed_seed.json`): fixtures regenerate byte for byte;
  every reported constant has `{I, H} = 0` (no false claims); no trivial combination is reported; every reference that
  lies in a searched family is recovered (invariant spaces compared by exact linear algebra, never by text); symmetry-
  breaking controls reject the invariants whose bracket became nonzero and still find the survivors; unsupported
  families are reported without inventing results; discovery commutes with separable canonical maps; resonant and
  generic parameters differ as they should. `fixtures/pending.json` records references not recovered and why; a case
  that starts passing fails the suite until it is removed.
- `scripts/bench_invariants.py`: `check` (drift), `run`, `compare` (adaptive against the fixed-parameter
  `search_invariants` on the verified Legendre transform, written to [invariant-benchmark.md](invariant-benchmark.md)),
  `sweep --seeds N` (larger randomized evaluation, offline), `promote base:seed:steps:mixing` (a failing case becomes a
  permanent fixed-seed regression, marked pending with its observed misses). Misses are classified as
  `insufficient_ansatz`, `missing_function_family`, `solver_failure`, `verification_failure`, `parameter_degeneracy`,
  `resource_exhaustion` or `unclassified`, so the next technique to improve is visible.

Linear canonical maps keep polynomial Hamiltonians polynomial. Systems whose invariants have denominators that mix
coordinates after a mixing map (inverse-square terms) leave the searched families; they are reported as
`missing_function_family`. Nonlinear canonical maps are an extension of the generator, not implemented.
