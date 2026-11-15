import type { Status } from "./types";

/** Mirrors `minerva_engine.creative_telescoping.models` (what the engine returns; never model output). */
export interface TelescopingCondition {
  latex: string;
  established: boolean;
  how: string;
}

export interface TelescopingResult {
  status: Status;
  /** `evaluated` (a value), `equation` (an ODE only), `certificate` (a proved relation only) or `none`. */
  outcome: "evaluated" | "equation" | "certificate" | "none";
  reason: string | null;
  message: string;
  input_latex: string | null;
  variable: string;
  parameter: string;
  lower_latex: string | null;
  upper_latex: string | null;
  parameter_domain_latex: string | null;
  operator: { order: number; coefficients_latex: string[]; latex: string } | null;
  certificate: { latex: string; latex_ratio: string } | null;
  certificate_check: {
    symbolic_zero: boolean;
    numeric_points_checked: number;
    numeric_points_agreed: number;
    derivative_points_checked: number;
    derivative_points_agreed: number;
    passed: boolean;
    note: string;
  } | null;
  families: { order: number; degree: number; unknowns: number; outcome: string }[];
  conditions: TelescopingCondition[];
  boundary: {
    lower_latex: string;
    upper_latex: string;
    contribution_latex: string;
    vanishes: boolean;
  } | null;
  equation: {
    latex: string;
    homogeneous: boolean;
    rhs_latex: string;
    leading_nonzero: boolean | null;
  } | null;
  solution: {
    general_latex: string;
    initial_point_latex: string;
    initial_values_latex: string[];
    constants_latex: string[];
  } | null;
  result_latex: string | null;
  result_text: string | null;
  value_verification: {
    ode_symbolic_zero: boolean;
    initial_values_hold: boolean;
    quadrature_points_checked: number;
    quadrature_points_agreed: number;
    passed: boolean;
    note: string;
  } | null;
  unresolved: string[];
  notes: string[];
  duration_ms: number;
}

/** What the workbench sends to `POST /v1/compute/telescoping`. */
export interface TelescopingRequest {
  input: string;
  format: "latex" | "text";
  variable: string;
  parameter: string;
  lower?: string | null;
  upper?: string | null;
  assumptions: Record<string, string[]>;
}
