# Handbook outline

The intended table of contents. The schema, ids and verification harness are designed for all of it;
the UI only shows chapters that contain entries (`planned: true` chapters in `chapters.yaml` are hidden).
Think of it as a book of mathematical tables and physics equations whose entries a program can check and
apply.

Status: **●** has entries, **○** planned.

## Mathematics

| | Chapter | Planned content |
| --- | --- | --- |
| ● | Algebra | Factoring/expansion; later: inequalities, polynomial identities, summation formulas |
| ● | Exponentials and logarithms | Log laws with conditions; branches of the complex logarithm; radicals |
| ● | Trigonometric functions | Pythagorean, angle sum, multiple/half-angle, product-to-sum, exponential forms |
| ● | Hyperbolic functions | Identities and relations to trigonometric functions |
| ● | Derivatives | Elementary table; later: higher derivatives, Leibniz rule, implicit differentiation |
| ● | Integrals | Antiderivative and definite-integral tables; later: Gradshteyn-style families (original statements) |
| ● | Series | Maclaurin series, convergence conditions, classical sums and products |
| ● | Special functions | Gamma now; Beta, Bessel, Legendre, hypergeometric, elliptic, zeta |
| ○ | Integral transforms | Fourier, Laplace, Mellin tables with convergence conditions |
| ○ | Linear algebra | Determinant, trace, eigenvalue identities, decompositions |
| ○ | Vector calculus | Div/grad/curl identities, integral theorems |
| ○ | Differential equations | Standard forms and solution families |
| ○ | Probability and statistics | Distributions, moments, generating functions |
| ○ | Group theory | Lie algebra relations, representation tables |
| ○ | Differential geometry | Connections, curvature identities, coordinate systems |
| ● | Tensor calculus | Index symmetries and contractions; Riemann symmetries; Bianchi (stated) |

## Physics

| | Chapter | Planned content |
| --- | --- | --- |
| ● | Classical mechanics | Dynamics, oscillations; later: Lagrangian/Hamiltonian formalism, rigid bodies |
| ● | Electromagnetism | Force laws; later: Maxwell's equations (SI and Gaussian), potentials, radiation |
| ● | Thermodynamics and statistical mechanics | Gas laws, entropy, radiation; later: partition functions, ensembles |
| ● | Quantum mechanics | Fundamental relations; later: angular momentum algebra, perturbation theory, hydrogen atom |
| ● | Special and general relativity | Kinematics, energy-momentum; field equations (stated); later: metrics, geodesics |
| ● | Quantum field theory | Free field equations (stated); later: propagators, Feynman rules, Ward identities |
| ● | Physical constants | SI defining constants and G; later: full CODATA set with uncertainties and version |

## Entry quality levels

`symbolic` (SymPy proves it) > `numeric` (sampled) > `structural` (tensor canonicaliser) >
`distributional` (equality as distributions: exact conditions plus a test-function pairing, since a
pointwise sample cannot see a Dirac delta) > `dimensional` (units only) > `stated` (literature record, never applied). Promotion towards stronger
levels is a goal as backends improve.
