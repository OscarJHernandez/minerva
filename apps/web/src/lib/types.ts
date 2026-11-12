/** Types mirroring the Minerva API (see /docs on the API for the authoritative schema). */

export type Status =
  | "ok"
  | "unchanged"
  | "not_verified"
  | "unsupported"
  | "parse_error"
  | "resource_limit"
  | "internal_error";

export type Assumption =
  | "real"
  | "positive"
  | "negative"
  | "nonnegative"
  | "nonpositive"
  | "nonzero"
  | "integer"
  | "rational";

export const ASSUMPTIONS: readonly Assumption[] = [
  "real",
  "positive",
  "negative",
  "nonnegative",
  "nonpositive",
  "nonzero",
  "integer",
  "rational",
];

export type Assumptions = Record<string, Assumption[]>;

export interface Verification {
  symbolic_zero: boolean;
  numeric_points_checked: number;
  numeric_points_agreed: number;
  passed: boolean;
  note: string;
}

export interface SimplifyResult {
  status: Status;
  input_latex: string | null;
  result_latex: string | null;
  result_text: string | null;
  verification: Verification | null;
  message: string;
  duration_ms: number;
  calculation_id?: string | null;
}

export interface DiffVariable {
  name: string;
  order: number;
}

export const MAX_DERIVATIVE_ORDER = 10;

export interface DifferentiateResult {
  status: Status;
  input_latex: string | null;
  variables: DiffVariable[];
  total_order: number;
  notation_latex: string | null;
  result_latex: string | null;
  result_text: string | null;
  steps: { variable: string; order_so_far: number; latex: string }[];
  verification: {
    method: string;
    directions: ("real" | "imaginary")[];
    steps_checked: number;
    points_checked: number;
    points_agreed: number;
    passed: boolean;
    note: string;
  } | null;
  notes: string[];
  message: string;
  duration_ms: number;
  calculation_id?: string | null;
}

export type Applicability = "applicable" | "conditions_not_satisfied" | "not_verified" | "no_match";

export interface RuleCondition {
  variable: string;
  assumption: Assumption;
}

export interface RuleApplication {
  rule_id: string;
  applicability: Applicability;
  matched_subexpression_latex: string | null;
  rewritten_latex: string | null;
  unmet_conditions: RuleCondition[];
  verification: Verification | null;
  message: string;
}

export interface Reference {
  source: string;
  locator?: string | null;
  url?: string | null;
}

export type Verified = "symbolic" | "numeric" | "structural" | "dimensional" | "stated";

export interface EntrySummary {
  id: string;
  title: string;
  kind: string;
  chapter: string;
  statement_latex: string;
  validity: string;
  verification: Verified;
  reference: Reference | null;
}

export interface IdentityMatch {
  entry: EntrySummary;
  application: RuleApplication;
}

export interface IdentitiesResult {
  status: Status;
  input_latex: string | null;
  message: string;
  matches: IdentityMatch[];
  candidates_considered: number;
  duration_ms: number;
  calculation_id?: string | null;
}

export interface HistoryItem {
  id: string;
  operation: "simplify" | "identities" | "differentiate";
  input: string;
  format: "latex" | "text";
  assumptions: Assumptions;
  status: Status;
  result: SimplifyResult | IdentitiesResult;
  created_at: string;
}

export interface User {
  id: string;
  email: string;
}

export interface EntryOut extends EntrySummary {
  section: string | null;
  tags: string[];
  keywords: string[];
  symbols: { name: string; meaning: string; unit: string | null }[];
  constant: { value: string; uncertainty: string | null; unit: string } | null;
  notes: string;
  applicable_as_rule: boolean;
  conditions: RuleCondition[];
}

export interface TocChapter {
  id: string;
  book: "mathematics" | "physics";
  title: string;
  summary: string;
  entry_count: number;
  sections: { id: string; title: string; entry_count: number }[];
}
