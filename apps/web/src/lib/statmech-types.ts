import type { Status } from "./types";

/** Mirrors `minerva_engine.statmech.models`: the request is the whole problem, results are checked. */

export type StatKind = "levels" | "level_family" | "density_of_states" | "classical";
export type StatAction = "analyze" | "partition" | "derive" | "expectation" | "evaluate";
export type StatQuantityName = "Z" | "lnZ" | "F" | "U" | "S" | "Var_E";
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
  ensemble?: "canonical";
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
