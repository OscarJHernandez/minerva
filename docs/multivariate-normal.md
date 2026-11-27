# The bivariate Normal distribution

Operation `multivariate_normal` (engine `stats/mvn.py`, models `stats/mvn_models.py`, endpoint
`POST /v1/compute/mvn`, chat tool `multivariate_normal` in the `statistics` group; issue #229).

A model is a mean vector and a covariance matrix of **two** components, in plain-text syntax, plus the names of
the components. One action is asked of it:

| Action | Result | Formula |
| --- | --- | --- |
| `density` | the joint density | `exp(-(d^T Sigma^{-1} d)/2) / (2 pi sqrt(det Sigma))`, the inverse by its adjugate |
| `moments` | mean, covariance and the correlation | `rho = Sigma_12 / sqrt(Sigma_11 Sigma_22)` |
| `marginal` | the distribution of the component kept | `N(mu_i, Sigma_ii)` |
| `conditional` | the distribution of one component given the other's value | mean `mu_i + Sigma_ij/Sigma_jj (v - mu_j)`, variance the Schur complement `Sigma_ii - Sigma_ij^2/Sigma_jj` |
| `affine` | the distribution of `A X + b` (A of one or two rows) | `A mu + b`, `A Sigma A^T` |

Nothing has a default: the mean, the covariance, the component names, the component kept, the value
conditioned on and the map all come from the user. Independence of the components is never assumed: a zero
off-diagonal entry is the user's statement.

## Before anything is computed

* The covariance must be 2 x 2 and **exactly** symmetric. A matrix whose (1, 2) and (2, 1) entries differ is
  `not_symmetric`; Minerva never symmetrises it.
* Positive definiteness is decided by the leading principal minors (`Sigma_11 > 0`, `det Sigma > 0`):
  * provably false (a numeric matrix, or a sign the declared assumptions decide) is `not_positive_definite`;
  * a determinant that is identically zero is `singular_covariance` (`unsupported`): no pseudoinverse and no
    degenerate density are ever produced;
  * undecided (symbolic entries the assumptions do not decide) is an unestablished condition. The result is
    then **conditional**, "holds given $\det\Sigma > 0$", never silently true.
* Symbols are complex unless declared: a model with an undeclared symbol holds given that the entries are real.
* An affine map must have full row rank: a singular 2 x 2 `A` or a zero row is `map_singular` (`unsupported`);
  an undecided rank is a premise.

## How a result is checked

Every result is exact algebra, and it is shown only if an independent numerical leg agrees at sampled admissible
values of the symbols (one point for a fully numeric model, at least three otherwise):

* the printed result is **read back** by the text parser, so the check is of what is shown;
* the reference is a joint density built in floating point from the numeric mean and covariance with NumPy's
  inverse and determinant (no SymPy algebra), integrated by SciPy (`dblquad`, `quad`) over a box of 12 standard
  deviations, with the integrator's error estimate;
* `density`: the printed density integrates to 1 and has the given means and covariances;
  `moments`: the reference density's moments and correlation equal the printed ones;
  `marginal`: the printed marginal equals the joint integrated over the other component at three values;
  `conditional`: the printed conditional equals the joint divided by its integral over the free component;
  `affine`: the mean and covariance of `A X + b` under the reference density equal the printed ones;
* agreement means a relative difference below 1e-9 **and** an error estimate ten times smaller. A quadrature
  that cannot reach that (a correlation of 0.999 does not) counts as no agreement: the result is withheld
  (`not_verified`).

The tests inject wrong twins into the algebra: a density without the off-diagonal term, without its determinant
factor or off by one part in a million, a conditional variance of `Sigma_11` instead of the Schur complement, a
conditional mean off by one part in a million, a marginal with the other variance, and `A Sigma A` instead of
`A Sigma A^T`. Each must give `not_verified` (`packages/engine/tests/test_mvn.py`). The reference problems
(`minerva_engine/reference/mvn.py`) are worked by hand from Anderson, *An Introduction to Multivariate
Statistical Analysis*, Theorems 2.4.3, 2.4.4 and 2.5.1.

Results are worded "numerically checked", never "proved".

## Why only two components

The independent leg is a nested quadrature. Measured with mpmath (with its error estimate), one integral took
about 1 s in two dimensions and 144 to 600 s in three, far over the time a chat tool has. SciPy's `dblquad` takes
about 40 ms per integral in two dimensions. A model of one component is the Normal family of `statistical_model`;
three or more components are `dimension_unsupported`. `statistical_model(family="multivariate_normal")` refuses
and names this operation.

## Not covered

Estimation from data, conjugate priors, singular (degenerate) distributions, more than two components, and a
workbench form.
