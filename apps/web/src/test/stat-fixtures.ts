import type {
  ExpectationResult,
  MatrixCalculusResult,
  StatModelResult,
  SumsResult,
} from "@/lib/types";

const step = (title: string, after: string, evidence: "identity" | "conditional" = "identity") => ({
  title,
  before_latex: "",
  after_latex: after,
  after_text: after,
  evidence,
});

export const sumsOk: SumsResult = {
  status: "ok",
  action: "expand",
  reason: null,
  input_latex: "\\sum_{i=1}^{n} \\left(x_{i}-\\mu\\right)^{2}",
  result_latex: "\\sum_{i=1}^{n} x_{i}^{2} - 2 \\mu \\sum_{i=1}^{n} x_{i} + n \\mu^{2}",
  result_text: "Sum(x[i]**2, (i, 1, n)) - 2*mu*Sum(x[i], (i, 1, n)) + n*mu**2",
  steps: [step("Linearity of finite sums", "\\sum x_i")],
  verification: {
    sizes: [{ n: 1 }, { n: 2 }, { n: 3 }, { n: 5 }],
    instances_checked: 4,
    instances_agreed: 4,
    exact_zero_instances: 4,
    general_symbolic_zero: null,
    passed: true,
    note: "",
  },
  derivative_verification: null,
  conditions: [],
  message: "checked for n = 1; n = 2; n = 3; n = 5 (explicit expansion), not proved for every size",
  duration_ms: 40,
};

export const sumsRefused: SumsResult = {
  ...sumsOk,
  status: "conditions_not_satisfied",
  reason: "range_not_provable",
  result_latex: null,
  result_text: null,
  steps: [],
  verification: null,
  message: "cannot show that the bounds are integers",
};

export const expectationConditional: ExpectationResult = {
  status: "ok",
  action: "expand",
  reason: null,
  input_latex: "\\operatorname{Var}\\left(X + Y\\right)",
  result_latex: "\\operatorname{Var}\\left(X\\right) + \\operatorname{Var}\\left(Y\\right)",
  result_text: "Var(X) + Var(Y)",
  evidence: "conditional",
  given: ["X and Y are independent"],
  steps: [step("Linearity of expectation", "\\operatorname{Var}(X)", "conditional")],
  verification: {
    sizes: [],
    models_checked: 2,
    models_agreed: 2,
    exact_zero_models: 2,
    passed: true,
    note: "",
  },
  message: "",
  duration_ms: 30,
};

export const expectationFree: ExpectationResult = {
  ...expectationConditional,
  result_latex: "2 \\operatorname{Cov}(X, Y) + \\operatorname{Var}(X) + \\operatorname{Var}(Y)",
  result_text: "2*Cov(X, Y) + Var(X) + Var(Y)",
  evidence: "identity",
  given: [],
};

const empty: StatModelResult = {
  status: "ok",
  action: "mle",
  reason: null,
  steps: [],
  moments: {},
  moments_latex: {},
  expressions: {},
  expressions_latex: {},
  estimates: [],
  matrix_text: null,
  matrix_latex: null,
  posterior: null,
  checks: [],
  premises: [],
  conditions: [],
  evidence: null,
  message: "",
  duration_ms: 100,
};

const check = (kind: string, a = 4, n = 4) => ({
  kind,
  method: `${kind} check`,
  points_checked: n,
  points_agreed: a,
  passed: a === n,
  note: "",
});

export const mleGaussian: StatModelResult = {
  ...empty,
  evidence: "conditional",
  premises: ["the observations are independent and identically distributed"],
  estimates: [
    {
      parameter: "mu",
      latex: "\\frac{\\sum_{i=1}^{n} x_{i}}{n}",
      text: "Sum(x[i], (i, 1, n))/n",
      optimum: "local",
      optimum_evidence: "proved",
      admissible_if: [],
    },
    {
      parameter: "v",
      latex: "\\frac{\\sum_{i=1}^{n} \\left(x_{i}-\\bar x\\right)^{2}}{n}",
      text: "Sum((x[i] - Sum(x[j], (j, 1, n))/n)**2, (i, 1, n))/n",
      optimum: "local",
      optimum_evidence: "proved",
      admissible_if: [
        {
          latex: "\\sum_{i=1}^{n} \\left(x_{i}-\\bar x\\right)^{2} > 0",
          established: false,
          how: "a condition on the data: the estimator must lie in the parameter space",
        },
      ],
    },
  ],
  steps: [step("Log-likelihood", "\\ell"), step("Score for mu", "s_\\mu")],
  checks: [check("score"), check("second_order")],
};

export const mleBernoulliGlobal: StatModelResult = {
  ...mleGaussian,
  premises: [],
  evidence: "identity",
  estimates: [
    {
      parameter: "p",
      latex: "\\frac{\\sum x_i}{n}",
      text: "Sum(x[i], (i, 1, n))/n",
      optimum: "global",
      optimum_evidence: "proved",
      admissible_if: [],
    },
  ],
};

export const mleMeasured: StatModelResult = {
  ...mleGaussian,
  estimates: [{ ...mleGaussian.estimates[0]!, optimum_evidence: "measured" }],
};

export const fisher: StatModelResult = {
  ...empty,
  action: "fisher_information",
  matrix_text: [
    ["1/sigma**2", "0"],
    ["0", "2/sigma**2"],
  ],
  matrix_latex: [
    ["\\frac{1}{\\sigma^{2}}", "0"],
    ["0", "\\frac{2}{\\sigma^{2}}"],
  ],
  checks: [check("information"), check("information")],
  evidence: "identity",
};

export const posterior: StatModelResult = {
  ...empty,
  action: "posterior",
  evidence: "identity",
  premises: ["the prior is as stated; it is an assumption, not a fact"],
  posterior: {
    family: "beta",
    parameters: { alpha: "alpha + k", beta: "beta - k + n" },
    parameters_latex: { alpha: "\\alpha + k", beta: "\\beta - k + n" },
    dropped_factors_latex: ["\\binom{n}{k}"],
  },
  checks: [check("posterior")],
};

export const momentsCauchy: StatModelResult = {
  ...empty,
  status: "conditions_not_satisfied",
  action: "moments",
  reason: "convergence_not_established",
  message: "the mean and variance of the Cauchy family do not exist",
  moments: { mean: "0" }, // never shown for a refusal, even if one were present
  moments_latex: { mean: "0" },
};

export const momentsNormal: StatModelResult = {
  ...empty,
  action: "moments",
  evidence: "identity",
  moments: { mean: "mu", variance: "sigma**2" },
  moments_latex: { mean: "\\mu", variance: "\\sigma^{2}" },
  conditions: [
    { latex: "\\sigma > 0", established: true, how: "from the declared assumptions" },
    { latex: "0 < p < 1", established: false, how: "part of the parameter space of the model" },
  ],
  checks: [check("normalisation", 5, 5), check("moment_quadrature", 5, 5)],
};

export const gradientSymmetric: MatrixCalculusResult = {
  status: "ok",
  reason: null,
  action: "gradient",
  input_latex: "-\\frac{1}{2} \\log\\left|\\det\\left(S\\right)\\right|",
  structure: "scalar function of the 3x3 matrix S, positive definite",
  features: [],
  method: "gradient_differential",
  method_title: "Gradient by differentials",
  result_kind: "gradient",
  result_latex: "-\\frac{1}{2} S^{-1}",
  result_text: "-1/2*inv(S)",
  result_entries: null,
  equation_latex: null,
  evaluated: true,
  assumptions_used: [],
  steps: [],
  blocked: [],
  verification: {
    method: "central finite differences",
    points_checked: 4,
    points_agreed: 4,
    passed: true,
    symbolic_zero: null,
    symbolic_method: "not run: the numerical derivative is the only leg here",
    note: "",
  },
  notes: [],
  message: "verified against the numerical derivative",
  duration_ms: 50,
  derivative: {
    kind: "gradient",
    variable_shape: [3, 3],
    result_shape: [3, 3],
    layout: "shape_of_variable",
    field: "real",
    constraint: "positive_definite",
    convention: "symmetric_frobenius",
    alternatives: [
      {
        convention: "symmetric_entrywise",
        latex: "-S^{-1} - \\operatorname{diag}\\left(-\\frac{1}{2} S^{-1}\\right)",
        text: "-inv(S) - diagpart(-1/2*inv(S))",
        points_checked: 4,
        points_agreed: 4,
        note: "the partial derivatives with respect to the independent entries (i <= j)",
      },
    ],
  },
};

export const gradientRefused: MatrixCalculusResult = {
  ...gradientSymmetric,
  status: "conditions_not_satisfied",
  reason: "invertibility_not_declared",
  result_kind: null,
  result_latex: null,
  result_text: null,
  derivative: null,
  verification: null,
  message: "X must be declared invertible: the derivative needs its inverse",
};
