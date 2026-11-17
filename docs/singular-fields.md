# Point sources and the three-dimensional delta

`vector_calculus` with `action: distributional` derives the distributional divergence of a point-source Coulomb
field and the distributional Laplacian of the Newtonian potential:

```text
∇·( r / r³ )   = 4π δ⁽³⁾(r)          ∇²( 1 / r ) = −4π δ⁽³⁾(r)
∇·( C (r−a) / |r−a|³ ) = 4π C δ⁽³⁾(r−a)
∇·( q/(4πε₀) · r / r³ ) = (q/ε₀) δ⁽³⁾(r)         (Gauss's law for a point charge)
```

This page says exactly what is supported, how a result is reached and checked, what the numerical evidence is
worth, and how the language model reaches the capability. It is not a theory of multidimensional distributions:
**one family is supported and everything else is refused.**

## Pointwise equality versus equality as distributions

The classical divergence of `F = r/r³` is `0` at every point where `F` is defined (`r ≠ 0`). The flux of `F`
through a sphere about the origin is `4π` for every radius. Both are true, and they are not in conflict: the
whole contribution of the singularity is a point mass, which no value of a function at a point can carry.

A distribution is defined by what it does to test functions: `⟨div F, φ⟩ = −∫ F·∇φ d³r`. Two distributions are
equal when they have the same action on every test function. `∇·F` and `0` differ (`⟨∇·F, φ⟩ = 4π φ(0)`), and
`∇·F = 4π δ⁽³⁾` is an equality **as distributions**, never pointwise. Minerva keeps the two statements apart in
every result (`singular.classical_latex` and `singular.distributional_latex`), and the ordinary `evaluate` action
still answers `0` for the classical divergence but now says it is only that (`singular_hint: point_source`) and
reports where the source is (it used to report no restriction at all for a field given in Cartesian components).

## Why sampling cannot verify a delta

A Dirac delta is `0` at every point a sampler can draw. `verify_equal` would "agree" that `0 = 5 δ(x)`, and a
numerical divergence of `r/r³` at any point it can evaluate is `0`. So:

- every pointwise machine (`verify_equal`, the derivative and integral checkers, the vector evaluator, the claim
  checker) refuses a delta (`distributions.guard.contains_distribution`), and so does every vector action except
  `distributional` (`unsupported_distribution_operation`; before this change the vector path would lower and
  "verify" a `DiracDelta` inside a field by sampling);
- the 3D delta is its own typed object, `PointDelta(a1, a2, a3)` (the source point), printed
  `δ⁽³⁾(r − a)`. It is not a product of one-dimensional deltas, it never evaluates and no operation may hand it to
  SymPy (which would treat it as a constant);
- a distributional identity is checked only **against test functions**.

## What is derived, and how

For `F = C (r − a)/|r − a|³` (or the gradient of `f = C/|r − a|`, giving `Δf = −4πC δ⁽³⁾`):

1. **Reading.** The field is lowered exactly and written in Cartesian components (radial fields in spherical
   coordinates, fields of `ρ` and `z` in cylindrical ones). `match_kernel` reads the source `a` and the strength
   `C` **from the field itself** (the base of its half-integer power is `λ|r − a|²`, the numerator is
   `K (r − a)`), as an exact algebraic identity. The model never supplies `4π`, the sign or the source. `a` must be
   provably real (an undeclared constant is not assumed real); `C` may be any constant, including a symbol and 0.
2. **The shrinking-sphere argument**, with its hypotheses checked exactly for this input:
   - H1 `∇·F = 0` away from `a` (an exact symbolic zero, computed from the input field);
   - H2 `F·n̂ = C/ε²` on the sphere `|r − a| = ε` (exact), so the outward flux is `4π C` for every `ε`;
   - H3 `(F·F)|r − a|⁴ = C²` (exact), so `|F| = |C|/|r − a|²` is locally integrable and the excised ball
     contributes nothing as `ε → 0`.

   Then, for a smooth rapidly decreasing `φ`, delete the ball `B_ε(a)`. The divergence theorem on the punctured
   domain (its outward normal points *toward* the source on the inner boundary) gives
   `−∫_{|r−a|>ε} F·∇φ = ∮_{|r−a|=ε} φ F·n̂ dS = C ∮ φ(a + εn̂) dΩ → 4π C φ(a)`. The Laplacian follows through the
   gradient `∇(C/|r − a|) = −C (r − a)/|r − a|³`, so the sign is computed, not looked up.
3. **Independent numerical legs** (`vector/weak.py`, numpy; they read the *input* field, not the kernel found in
   step 1, and share no code with the matcher):
   - the weak pairing `−∫ F·∇φ d³r` (or `∫ f Δφ d³r`) in spherical shells about the source, for four test functions
     with different values at the source — `φ(a) = 1`, `2`, `≈ 0.68` (one centred away from the source) and `0` — each
     compared with `4π C φ(a)` to a relative `1e-7` (the same tolerance as the one-dimensional pairing: a `1e-6`
     error in the coefficient is rejected and tested);
   - the flux through two spheres about the source;
   - a cross-check that the Cartesian reading used by the quadrature equals the finite-difference evaluator's reading
     of the input expression at eight random points;
   - the **regularised field** `r/(r² + ε²)^{3/2}`, whose divergence `3ε²/(r² + ε²)^{5/2}` is paired with a test
     function at three widths. The table is **reported and never decides**: the error falls roughly like `ε`, too
     weak a criterion to gate on.

A result is shown (`status: ok`) only if H1–H3 are exact and every pairing and flux agreed; otherwise it is
`not_verified` and nothing is displayed.

### What "derived" means

`singular.proof_status` is `derived`: every hypothesis of the argument is an exact symbolic fact for this input and
every independent pairing agreed. It is **not called proved** (Minerva says "proved" only for a symbolic zero of the
whole statement), the numerical pairings and the regularisation are supporting evidence, and the step that applies
the divergence theorem and takes the limit is a theorem applied under the stated hypotheses, not a computation. The
scope is stated in every result: one kernel, one source, test functions smooth and rapidly decreasing, constants
sampled as real numbers.

### Claims

`target` takes a statement the user made about the result (`4\pi\delta^{(3)}(\mathbf r)`, `0`,
`-4\pi\delta^{(3)}(\mathbf r - (a_1,a_2,a_3))`). It is compared exactly with the derived result and reported as
`holds`, `no_delta_term` (a function with no delta cannot equal this distribution, which is what rejects
`∇·(r/r³) = 0` as an identity of distributions), `different_strength` (a missing `4π`, the wrong sign:
`∇²(1/r) = +4π δ` is rejected) or `different_source`. A translated delta needs Cartesian input.

## Supported and unsupported

| Supported | Refused (typed reason) |
| --- | --- |
| `∇·F` or `∇²f` (times a constant) of one point-source Coulomb kernel, explicit field | other singular fields (`r/r⁴`, dipoles): `unsupported_singular_field` |
| constant strength `C` (a symbol, a number, negative, zero) | several sources, a sum of kernels: `unsupported_singular_field` |
| constant real source `a` (Cartesian input), the origin in spherical and cylindrical input | an undeclared or non-real source: `source_not_real` |
| `r/r³`, `r̂/r²`, `(x,y,z)/(x²+y²+z²)^{3/2}`, `r/|r|³`, `∇²(1/r)`, `∇·∇(1/|r|)`, `q/(4πε₀)·r/r³` | an abstract field, a time-dependent field |
| a statement to check (`target`) | a `DiracDelta` or `Heaviside` in any other vector action: `unsupported_distribution_operation` |

Not supported, and not claimed: products of the 3D delta with functions, derivatives of it, compositions, other
dimensions, extended sources, Green's functions of other operators, the wave equation's retarded potentials,
general multidimensional distribution theory. Unrestricted operations on `PointDelta` are not implemented.

## How the language model reaches it

The model only calls the tool (`vector_calculus`, action `distributional`; the group's keywords include `delta
function`, `point charge`, `distributional` and `\delta^`) and explains the result. The engine owns the coefficient,
the sign, the source, the hypotheses and the verdict on a claim; the model cannot supply any of them
(`VectorCalculusArgs` has no field for a coefficient). The prompt rule tells it to state both the classical and the
distributional result, say "equal as distributions", quote the status as `derived`, and report the engine's
verdict on a wrong statement. The prose is checked after the turn as for every tool: a formula is marked verified
only if the engine printed it (`\mathbf r` and `\mathbf{r}`, `\delta^3(…)` and `\delta^{(3)}(…)` count as one
spelling), and a `\delta^{(3)}(…)` is never sampled by the claim checker.

## Reproducing the tests

```bash
uv run pytest packages/engine/tests/test_vector_singular.py packages/engine/tests/test_vector_singular_properties.py
uv run pytest packages/knowledge/tests/test_point_source_entries.py
uv run python -m minerva_knowledge verify                       # the five handbook entries
uv run python -m minerva_engine.vector.benchmark --check        # group `distributional`, 19 cases
uv run python -m minerva_engine.reference_benchmark --check     # hand-written mpmath oracles (`dist-*`)
uv run python scripts/eval_models.py --suite core --check-cases # the agent cases' ground truth
uv run python scripts/eval_offline.py --check                   # replayed model: the equation is backed, a wrong one is not
```

The oracles of the reference benchmark are written by hand and independent of the code under test: the regularised
sequence integrated by mpmath, the pairing of `1/r` with the Laplacian of a Gaussian, the flux through a sphere about
a translated source. Handbook entries (`distributions.coulomb_divergence`, `distributions.newtonian_laplacian`,
`distributions.point_source_translation`, `distributions.point_source_scaling`, `em.static.gauss_point_charge`) are
level `distributional`, checked by the same action, and each fails if its stated right-hand side has a wrong
coefficient or sign (tested).
