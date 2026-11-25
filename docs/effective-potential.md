# Radial effective potential of geodesics (`effective_potential`)

Issue #207, the last piece of the Schwarzschild workflow of #131. One call returns the radial equation

    r_dot^2 + V_eff(r) = alpha p_1^2        (alpha = 1 for Schwarzschild, p_1 = p_t)

as a verified result, from the metric, the cyclic coordinates whose momenta are held, a plane and the
normalisation `g(v, v) = kappa` that **the user** states. `kappa` is a required argument with no default (the tool
schema says to ask: `-1` timelike, `0` null); the model never chooses it. The momenta keep the names the Routh
reduction gives them (`p_{t}`, `p_{\phi}`); `E = -p_t` and `L = p_phi` stay the user's own `substitute`.

## What is established, in order (any failure is a refusal)

1. **The geodesic equations** of `L = g(v, v)/2` are confirmed by numerical differentiation of `L`.
2. **The restriction is consistent.** For every coordinate fixed on a plane (`theta = pi/2`), its acceleration from
   the confirmed equations of the *whole* system, at the fixed value and zero velocity, must be a symbolic zero and a
   numerical zero at random values of everything else. Otherwise the call is refused, never assumed:
   `equatorial_not_consistent` (the acceleration is nonzero: typed diagnostic `restriction_not_consistent`) or
   `consistency_undecided` (it was not reduced to exactly 0).
3. **Routh reduction** of the restricted Lagrangian by the cyclic coordinates, with that reduction's own verification
   (`routh.reduce`: each momentum verified conserved; the Euler-Lagrange residual of R on motions of the restricted
   Lagrangian, `-dR/dp`, and `R = L - sum p c_dot` at sample points).
4. **The normalisation.** `g(v, v) = 2 (R - sum p_c dR/dp_c)` (the velocity of a removed coordinate is `-dR/dp_c`,
   which step 3 verified), solved for `r_dot^2` and split at the momentum of the first cyclic coordinate. A term
   linear in `r_dot` is `radial_not_quadratic`; a coefficient of `p_1^2` that depends on `r`, or a term linear in
   `p_1`, is `not_separable` (list the energy-like coordinate first). Nothing is guessed.

## What decides (independent of the route that built it)

At points of the whole system (the fixed coordinates at their values with zero velocity, random cyclic velocities,
`r_dot` a numerical root of `g(v, v) = kappa` taken from the full Lagrangian, the momenta as central differences of
the full Lagrangian, 40 digits):

- `r_dot^2 + V_eff - alpha p_1^2` vanishes (relative 1e-15, so a 1e-6 error in `V_eff` is far outside it);
- the radial acceleration of the full geodesic equations equals `-V_eff'(r)/2` at fixed momenta (the radial
  equation differentiated along the motion), which sees the shape of `V_eff` and not only its value.

The symbolic leg (the same differentiation along the Routhian's own motion with `r_dot^2` replaced by the radial
equation) and the symbolic consistency of the plane are reported separately. "Proved" is said only when both are
symbolic zeros.

## Restrictions

Zeros of what the formulas divide by (`1 - 2M/r`, the coefficient of `r_dot^2`, the block determinant of the
reduction) are `restrictions_latex`: statements about where the **coordinates** and the formulas are valid, never
about singularities of the geometry. A declared `r > 0` is not repeated as a restriction.

## Handles

The radial equation is the card's single result (one equation number, an `@rN` handle carrying the restrictions).
`V_eff(r) = ...` (`veff`) and each held momentum (`held:<coordinate>`) are numbered list rows (names: `@eN` of a row
carries no verification). The card type is `effective_potential` (`card_equations.effective_potential`); the tool is
in the group `reduction` with `routh_reduce`.

## Tests and benchmarks

- `packages/engine/tests/test_effective_potential.py`: textbook Schwarzschild (timelike `V = (1 - 2M/r)(1 + L^2/r^2)`,
  null, other `kappa`), Reissner-Nordstrom, typed refusals, and wrong twins put between the construction and the
  checks (a dropped `1 - 2M/r`, the sign of the `p_phi^2/r^2` term, a 1e-6 factor), each of which must be `not_verified`.
- `services/api/tests/test_effective_potential_tool.py`: schema (the normalisation is required, no value or name
  fields), card, numbering, result handle, group.
- Reference problems `effective-potential-schwarzschild-timelike` (with the circular orbit at `r = 6M`, `L^2 = 12 M^2`
  as an independent textbook check) and `effective-potential-tilted-plane-refused`
  (`python -m minerva_engine.reference_benchmark`).

## Not covered

Functions declared in the metric, metrics that are not diagonal in the radial coordinate, non-equatorial orbits of
metrics without a plane of symmetry (refused by step 2), more than four coordinates, the turning points and orbit
classification (those are the user's questions about `V_eff`), and the choice of units or of `E` and `L`.
