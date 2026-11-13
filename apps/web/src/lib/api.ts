import type { DiscoveryReport } from "./chat-types";
import type {
  ApplyIdentityResult,
  Assumptions,
  Derivation,
  DerivationSummary,
  DiffVariable,
  DifferentiateResult,
  HistoryItem,
  IntegrateResult,
  IdentitiesResult,
  MatrixCalculusResult,
  MatrixRequest,
  SeriesResult,
  SimplifyGoal,
  SimplifyResult,
  User,
} from "./types";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Browser-side client. All calls go to same-origin `/api/*`, which the proxy forwards to the API. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    credentials: "same-origin",
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (response.status === 204) return undefined as T;
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(response.status, extractDetail(body, response.status));
  }
  return body as T;
}

export function extractDetail(body: unknown, status: number): string {
  if (typeof body === "object" && body !== null && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      const first = detail[0] as { msg?: unknown };
      if (typeof first?.msg === "string") return first.msg;
    }
  }
  return `Request failed (${status})`;
}

export interface ComputeInput {
  input: string;
  assumptions: Assumptions;
  /** Defaults to LaTeX. The Workbench continues a step from its text form, which re-reads exactly. */
  format?: "latex" | "text";
  save?: boolean;
  /** Save the call as a step of this derivation, optionally continuing from one of its steps. */
  derivation_id?: string;
  parent_id?: string;
}

export interface ConservedInput {
  function: string;
  system: "lagrangian" | "hamiltonian";
  coordinates: string[];
  momenta?: string[] | null;
  angles?: string[];
  budget?: "quick" | "standard" | "thorough";
  /** Advanced: search exactly one family instead of the ladder. */
  pins?: {
    degree?: number | null;
    power_bound?: number | null;
    harmonics?: number | null;
    time_degree?: number | null;
  } | null;
}

export interface SimplifyInput extends ComputeInput {
  goal?: SimplifyGoal;
  target?: string[];
}

export const api = {
  me: () => request<User>("/v1/auth/me"),
  login: (email: string, password: string) =>
    request<User>("/v1/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
  register: (email: string, password: string) =>
    request<User>("/v1/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  logout: () => request<void>("/v1/auth/logout", { method: "POST" }),
  simplify: (c: SimplifyInput) =>
    request<SimplifyResult>("/v1/compute/simplify", {
      method: "POST",
      body: JSON.stringify({ format: "latex", ...c }),
    }),
  differentiate: (c: ComputeInput, variables: DiffVariable[]) =>
    request<DifferentiateResult>("/v1/compute/differentiate", {
      method: "POST",
      body: JSON.stringify({ format: "latex", ...c, variables }),
    }),
  series: (c: ComputeInput, variable: string, point: string, order: number) =>
    request<SeriesResult>("/v1/compute/series", {
      method: "POST",
      body: JSON.stringify({ format: "latex", ...c, variable, point, order }),
    }),
  integrate: (c: ComputeInput, variable: string, bounds?: { lower: string; upper: string }) =>
    request<IntegrateResult>("/v1/compute/integrate", {
      method: "POST",
      body: JSON.stringify({ format: "latex", ...c, variable, ...bounds }),
    }),
  matrix: (c: ComputeInput, req: MatrixRequest) =>
    request<MatrixCalculusResult>("/v1/compute/matrix", {
      method: "POST",
      body: JSON.stringify({ ...c, ...req, format: "latex" }),
    }),
  conserved: (c: ConservedInput) =>
    request<DiscoveryReport>("/v1/compute/conserved", {
      method: "POST",
      body: JSON.stringify(c),
    }),
  identities: (c: ComputeInput) =>
    request<IdentitiesResult>("/v1/compute/identities", {
      method: "POST",
      body: JSON.stringify({ format: "latex", ...c }),
    }),
  applyIdentity: (c: ComputeInput, ruleId: string) =>
    request<ApplyIdentityResult>("/v1/compute/apply-identity", {
      method: "POST",
      body: JSON.stringify({ format: "latex", ...c, rule_id: ruleId }),
    }),
  createDerivation: () => request<Derivation>("/v1/derivations", { method: "POST" }),
  derivations: () => request<DerivationSummary[]>("/v1/derivations"),
  derivation: (id: string) => request<Derivation>(`/v1/derivations/${id}`),
  deleteDerivation: (id: string) => request<void>(`/v1/derivations/${id}`, { method: "DELETE" }),
  history: (limit = 50) => request<HistoryItem[]>(`/v1/history?limit=${limit}`),
  deleteHistoryItem: (id: string) => request<void>(`/v1/history/${id}`, { method: "DELETE" }),
  clearHistory: () => request<void>("/v1/history", { method: "DELETE" }),
};
