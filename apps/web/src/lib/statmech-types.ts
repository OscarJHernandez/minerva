import type { Status } from "./types";

/** Mirrors `minerva_engine.statmech.models`: the request is the whole problem, results are checked. */

export type StatKind =
  | "levels"
  | "level_family"
  | "density_of_states"
  | "classical"
  | "quantum"
  | "occupation"
  | "chain";
export type StatAction =
  "analyze" | "partition" | "derive" | "expectation" | "evaluate" | "approximate";
export type StatMethodName =
  "high_temperature" | "low_temperature" | "stirling" | "thermodynamic_limit";
export type StatQuantityName =
  "Z" | "lnZ" | "F" | "U" | "S" | "Var_E" | "Xi" | "lnXi" | "Omega" | "N" | "Var_N";
export type StatStatistic =
  "mean" | "moment" | "central_moment" | "cumulant" | "variance" | "covariance";
export type ResultKind = "exact" | "representation" | "numeric" | "asymptotic";

export interface StatLevel {
  energy: string;
  degeneracy: string;
  observables?: Record<string, string>;
}

export type StatSystem =
  | { kind: "levels"; levels: StatLevel[] }
  | {
      kind: "level_family";
      index: string;
      energy: string;
      degeneracy: string;
      lower: string;
      upper: string;
    }
  | {
      kind: "density_of_states";
      variable: string;
      density: string;
      lower: string;
      upper: string;
    }
  | {
      kind: "chain";
      /** Two or three values of a site; the bond must be symmetric in (s, t). */
      states: string[];
      bond: string;
      site?: string;
      /** A symbol (integer, positive) or a number of at most 14. */
      length: string;
    }
  | {
      kind: "occupation";
      /** bose and fermi count each single-particle state; boltzmann is the classical-counting limit. */
      statistics: "bose" | "fermi" | "boltzmann";
      levels: { energy: string; degeneracy: string }[];
      mu: string;
    }
  | {
      kind: "quantum";
      /** Rows of the Hermitian matrix H, entries in text syntax (at most 4x4). */
      hamiltonian: string[][];
      /** Operator matrices of the same size; never assumed to commute with H or each other. */
      observables?: Record<string, string[][]>;
    }
  | {
      kind: "classical";
      hamiltonian: string;
      variables: { name: string; domain_lower: string; domain_upper: string }[];
      /** The user's choice (the divisor of the phase-space measure); never the model's. */
      phase_space_cell?: string | null;
    };

export interface StatPartial {
  of: StatQuantityName;
  variable: string;
  held_fixed: string[];
  order?: number;
  scale?: string;
  label?: string;
}

export interface StatSweep {
  variable: string;
  start: string;
  stop: string;
  points: number;
  scale: "linear" | "log";
}

/** What `POST /v1/compute/statmech` accepts (and what a saved run stores as its `request`). */
export interface StatMechRequest {
  ensemble?: "canonical" | "grand_canonical" | "microcanonical";
  /** Microcanonical only: the fixed total energy (a rational number). */
  total_energy?: string | null;
  method?: StatMethodName | null;
  system: StatSystem;
  copies?: { count: string; identical: boolean | null } | null;
  format?: "text" | "latex";
  temperature?: "beta" | "T";
  parameters?: string[];
  assumptions?: Record<string, string[]>;
  action: StatAction;
  quantities?: StatQuantityName[];
  partials?: StatPartial[];
  observable?: string;
  statistic?: StatStatistic;
  order?: number;
  other_observable?: string | null;
  coupling?: { parameter: string; observable: string } | null;
  values?: Record<string, string>;
  sweep?: StatSweep | null;
}

export interface StatCondition {
  latex: string;
  established: boolean;
  how: string;
}

export interface StatStep {
  name: string;
  title: string;
  latex: string;
  text: string;
  result_kind: ResultKind;
  conditions: StatCondition[];
}

export interface StatCheck {
  name: string;
  subject: string;
  passed: boolean | null;
  points: number;
  agreed: number;
  detail: string;
}

export interface StatQuantity {
  name: string;
  title: string;
  latex: string;
  text: string;
  result_kind: ResultKind;
  held_fixed: string[];
  variable: string | null;
  route: string;
  checks: StatCheck[];
}

export interface StatMethod {
  id: string;
  title: string;
  applies: boolean;
  implemented: boolean;
  why: string;
}

export interface StatMechResult {
  status: Status;
  action: StatAction;
  reason: string | null;
  state_variables: string[];
  steps: StatStep[];
  quantities: StatQuantity[];
  conditions: StatCondition[];
  methods: StatMethod[];
  not_supported: string[];
  questions: string[];
  table: { columns: string[]; rows: string[][]; variable: string } | null;
  verification: {
    method: string;
    points_checked: number;
    points_agreed: number;
    symbolic_zero: boolean | null;
    passed: boolean;
    note: string;
  };
  notes: string[];
  /** For action `approximate`: what it assumes, expands in and costs, measured, never a bound. */
  approximation?: {
    method: StatMethodName;
    assumptions: string[];
    parameter: string;
    order: string;
    validity: string;
    error_table: {
      label: string;
      exact: string;
      approximate: string;
      error: string;
      note: string;
    }[];
  } | null;
  message: string;
  duration_ms: number;
  calculation_id?: string | null;
}

/** The user's choices a card or form can send with a run (never chosen by the model). */
export interface StatAnswers {
  phase_space_cell?: string;
  identical?: boolean;
  values?: Record<string, string>;
  sweep?: StatSweep;
}

/** The model's call as stored on a chat card, so that the card's controls can resend it. */
export type StatToolArgs = Record<string, unknown> & { system: StatSystem };
