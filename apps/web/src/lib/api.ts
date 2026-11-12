import type {
  Assumptions,
  DiffVariable,
  DifferentiateResult,
  HistoryItem,
  IntegrateResult,
  IdentitiesResult,
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
  simplify: (c: ComputeInput) =>
    request<SimplifyResult>("/v1/compute/simplify", {
      method: "POST",
      body: JSON.stringify({ ...c, format: "latex" }),
    }),
  differentiate: (c: ComputeInput, variables: DiffVariable[]) =>
    request<DifferentiateResult>("/v1/compute/differentiate", {
      method: "POST",
      body: JSON.stringify({ ...c, variables, format: "latex" }),
    }),
  integrate: (c: ComputeInput, variable: string, bounds?: { lower: string; upper: string }) =>
    request<IntegrateResult>("/v1/compute/integrate", {
      method: "POST",
      body: JSON.stringify({ ...c, variable, ...bounds, format: "latex" }),
    }),
  identities: (c: ComputeInput) =>
    request<IdentitiesResult>("/v1/compute/identities", {
      method: "POST",
      body: JSON.stringify({ ...c, format: "latex" }),
    }),
  history: (limit = 50) => request<HistoryItem[]>(`/v1/history?limit=${limit}`),
  deleteHistoryItem: (id: string) => request<void>(`/v1/history/${id}`, { method: "DELETE" }),
  clearHistory: () => request<void>("/v1/history", { method: "DELETE" }),
};
