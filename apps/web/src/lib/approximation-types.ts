import type { Status } from "./types";

/** Mirrors `minerva_engine.approx.models` (what the engine returns; never model output). */
export type ApproximationMethod =
  | "auto"
  | "taylor"
  | "perturbation"
  | "laurent"
  | "asymptotic"
  | "multivariable"
  | "leading"
  | "limit"
  | "pade"
  | "chebyshev"
  | "legendre"
  | "fourier"
  | "piecewise"
  | "continued_fraction"
  | "euler_maclaurin"
  | "integral_expansion"
  | "root_expansion"
  | "reversion"
  | "laplace"
  | "accelerate"
  | "compare";

export type ApproximationEvidence =
  "series_checked" | "numeric_estimate" | "bound_established" | "none";

export interface ApproxCheck {
  name: string;
  /** null: could not be decided (never a refutation). */
  passed: boolean | null;
  points: number;
  detail: string;
}

export interface ApproxErrorSummary {
  samples: number;
  max_abs: string | null;
  rms: string | null;
  max_rel: string | null;
  relative_note: string;
  at: string | null;
}

export interface ApproxComplexity {
  degree: number | null;
  numerator_degree: number | null;
  denominator_degree: number | null;
  modes: number | null;
  segments: number | null;
  coefficients: number | null;
}

export interface ApproxPlotPoint {
  x: number;
  error: number;
}

export interface ApproxCandidate {
  method: string;
  label: string;
  status: Status;
  reason: string | null;
  result_latex: string | null;
  complexity: ApproxComplexity;
  error: ApproxErrorSummary | null;
  tolerance_met: boolean | null;
  warnings: string[];
  error_curve: ApproxPlotPoint[];
  chosen: boolean;
}

export interface ApproxSuggestion {
  method: string;
  applies: boolean;
  why: string;
  tried: boolean;
}

export interface ApproximationReport {
  status: Status;
  reason: string | null;
  method: string;
  message: string;
  input_latex: string | null;
  variable: string | null;
  domain_latex: string | null;
  result_latex: string | null;
  result_text: string | null;
  remainder_latex: string | null;
  statement_latex: string | null;
  orders: { order: number; coefficient_latex: string; term_latex: string }[];
  extras: { label: string; latex: string }[];
  evidence: ApproximationEvidence;
  checks: ApproxCheck[];
  complexity: ApproxComplexity;
  error: ApproxErrorSummary | null;
  measured_errors: { distance: number; absolute_error: string; relative_error: string | null }[];
  error_curve: ApproxPlotPoint[];
  tolerance_met: boolean | null;
  warnings: string[];
  candidates: ApproxCandidate[];
  suggestions: ApproxSuggestion[];
  questions: string[];
  notes: string[];
  assumptions_stated: string[];
  duration_ms: number;
  calculation_id?: string | null;
}

/** The numbers and choices of a request that belong to the user. Empty fields are not sent. */
export interface ApproximationNumbers {
  method?: ApproximationMethod;
  variable?: string;
  point?: string;
  direction?: "+" | "-";
  order?: number;
  small?: string;
  lower?: string;
  upper?: string;
  degree?: number;
  denominator_degree?: number;
  period?: string;
  modes?: number;
  segments?: number;
  kind?: string;
  tolerance?: string;
  tolerance_kind?: "absolute" | "relative";
  max_terms?: number;
  variables?: string[];
  points?: string[];
  methods?: string[];
}

/** The arguments of the chat tool (the model's call) and of its direct route. */
export interface ApproximationToolArgs extends ApproximationNumbers {
  latex: string;
  assumptions?: Record<string, string[]>;
}

/** The body of `POST /v1/compute/approximation`. */
export interface ApproximationRequest extends ApproximationNumbers {
  input: string;
  format: "latex" | "text";
  assumptions?: Record<string, string[]>;
  save?: boolean;
}

export const METHOD_LABEL: Record<ApproximationMethod, string> = {
  auto: "Choose automatically",
  taylor: "Taylor series",
  perturbation: "Perturbation expansion",
  laurent: "Laurent series",
  asymptotic: "Asymptotic expansion",
  multivariable: "Several variables",
  leading: "Leading behaviour",
  limit: "Limit",
  pade: "Padé approximant",
  chebyshev: "Chebyshev",
  legendre: "Legendre",
  fourier: "Fourier series",
  piecewise: "Piecewise polynomial",
  continued_fraction: "Continued fraction",
  euler_maclaurin: "Euler–Maclaurin sum",
  integral_expansion: "Expand an integral",
  root_expansion: "Expand a root",
  reversion: "Inverse function",
  laplace: "Laplace's method",
  accelerate: "Series acceleration",
  compare: "Compare methods",
};

/** What a warning means, in words: the engine sends only the fixed word. */
export const WARNING_TEXT: Record<string, string> = {
  removable_singularity:
    "The expression is undefined at the point, but the singularity is removable.",
  pole_in_interval: "The approximation has a pole inside the interval.",
  spurious_pole:
    "The approximant has a pole that the original function does not have: do not trust it near there.",
  jump_discontinuity:
    "The function jumps (or changes abruptly) on this interval; a smooth approximation will oscillate there.",
  gibbs:
    "Near a jump the series overshoots by about 9% of the jump, and this does not shrink with more terms.",
  slow_coefficient_decay:
    "The last coefficients are not small: the function is not well resolved at this degree.",
  asymptotic_not_convergent:
    "An asymptotic expansion, not a convergent series: more terms need not improve it at a fixed value.",
  no_extrapolation: "Undefined outside the interval: it never extrapolates.",
  local_only: "A local approximation: it describes the function near the point only.",
  error_estimate_unstable:
    "The error measured on two different grids disagreed, so the estimate is unreliable.",
  one_sided_limits_differ: "The one-sided limits differ, so there is no two-sided limit.",
  expansion_not_uniform: "The expansion may not be uniform on the interval.",
};
