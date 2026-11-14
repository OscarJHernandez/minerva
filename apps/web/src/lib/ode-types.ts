import type { Assumptions, Status } from "./types";

/** Mirrors `minerva_engine.ode.models`. Equations are strings in the chosen `format`. */

export interface OdeCondition {
  unknown: string;
  order: number;
  at: string;
  value: string;
}

export interface OdeInput {
  equations: string[];
  format: "latex" | "text";
  unknowns: string[];
  independent: string | null;
  assumptions: Assumptions;
  conditions: (OdeCondition | string)[];
}

export type NumericMethod = "auto" | "RK45" | "DOP853" | "Radau" | "BDF" | "LSODA";

export interface OdeNumericInput extends OdeInput {
  parameters: Record<string, number>;
  t_end: number;
  method: NumericMethod;
  rtol: number;
  atol: number;
  samples: number;
}

export interface OdeInterpretation {
  unknowns: string[];
  independent: string;
  parameters: string[];
  orders: Record<string, number>;
  equations_latex: string[];
  conditions_latex: string[];
  notes: string[];
}

export interface OdeStructure {
  kind: "ode" | "system";
  n_unknowns: number;
  n_equations: number;
  orders: Record<string, number>;
  order: number;
  state_dimension: number;
  linear: boolean;
  homogeneous: boolean | null;
  constant_coefficients: boolean | null;
  autonomous: boolean;
  solved_for_highest: boolean;
  position_only: boolean;
}

export type OdeMethodId =
  | "symbolic_closed_form"
  | "numerical_ivp"
  | "numerical_bvp"
  | "power_series"
  | "regular_perturbation"
  | "poincare_lindstedt"
  | "multiple_scales"
  | "equilibrium_analysis"
  | "phase_plane";

export interface MethodSuggestion {
  id: OdeMethodId;
  title: string;
  applicable: boolean;
  available: boolean;
  rank: number | null;
  reason: string;
  requires: string[];
  limitations: string[];
  tool: "ode_solve_symbolic" | "ode_solve_numerical" | "ode_analyze" | null;
  action: string | null;
}

export type ProblemKind =
  "general" | "initial_value" | "boundary_value" | "incomplete" | "overdetermined";

export interface OdeDiagnostics {
  trace_id: string | null;
  stages: { name: string; ms: number }[];
  failure_reason: string | null;
}

export interface OdeAnalysis {
  status: Status;
  message: string;
  interpretation: OdeInterpretation | null;
  structure: OdeStructure | null;
  problem_kind: ProblemKind | null;
  conditions_needed: string[];
  classes: string[];
  solved_latex: string[];
  restrictions_latex: string[];
  methods: MethodSuggestion[];
  notes: string[];
  diagnostics: OdeDiagnostics;
  duration_ms: number;
  calculation_id?: string | null;
}

export interface OdeVerification {
  points_checked: number;
  points_agreed: number;
  points_skipped: number;
  max_residual: number | null;
  symbolic_zero: boolean | null;
  independent_constants: number | null;
  passed: boolean;
  note: string;
}

export interface ConditionCheck {
  latex: string;
  satisfied: boolean;
  residual: number | null;
}

export interface OdeSolutionItem {
  kind: "explicit" | "implicit" | "unevaluated";
  equations_latex: string[];
  rhs_text: string[];
  constants: string[];
  family: "general" | "particular" | "partial";
  evaluated: boolean;
  verification: OdeVerification;
  conditions: ConditionCheck[];
  restrictions_latex: string[];
}

export interface OdeSolution {
  status: Status;
  message: string;
  method: "closed_form" | null;
  interpretation: OdeInterpretation | null;
  solutions: OdeSolutionItem[];
  verification: OdeVerification | null;
  dropped: number;
  notes: string[];
  diagnostics: OdeDiagnostics;
  duration_ms: number;
  calculation_id?: string | null;
}

export interface NumericCheck {
  name:
    | "initial_conditions"
    | "independent_method"
    | "tolerance_refinement"
    | "defect"
    | "first_integral";
  status: "passed" | "failed" | "skipped";
  value: number | null;
  tolerance: number | null;
  note: string;
}

export interface NumericVerification {
  checks: NumericCheck[];
  passed: boolean;
  independent_method: string | null;
  refinement_errors: number[];
  first_integral_latex: string | null;
}

export interface OdeSeries {
  label_latex: string;
  unknown: string;
  order: number;
  values: number[];
}

export type NumericWarning =
  "stiffness_indicated" | "step_collapse" | "not_converged" | "blow_up_suspected" | "domain_left";

export interface OdeNumericReport {
  status: Status;
  message: string;
  interpretation: OdeInterpretation | null;
  method_requested: NumericMethod | null;
  method_used: string | null;
  method_reason: string;
  t_start: number | null;
  t_end: number | null;
  parameters: Record<string, number>;
  initial: { label_latex: string; value: number }[];
  rtol: number | null;
  atol: number | null;
  steps: number | null;
  evaluations: number | null;
  t: number[];
  series: OdeSeries[];
  verification: NumericVerification | null;
  warnings: NumericWarning[];
  notes: string[];
  diagnostics: OdeDiagnostics;
  duration_ms: number;
  calculation_id?: string | null;
}
