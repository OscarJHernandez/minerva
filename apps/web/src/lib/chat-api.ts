import { ApiError, extractDetail } from "./api";
import type { OdeAnalyzeInput, OdeInput, OdeNumericInput, OdeSolveInput } from "./ode-types";
import type { StatAnswers, StatToolArgs } from "./statmech-types";
import type { DistributionAnswer, GeometryRequest } from "./types";
import type { VariationRequest } from "./variation-types";
import type {
  ChatStatus,
  ConnectionResult,
  ProviderChoice,
  ConversationSummary,
  ExtractedFormula,
  MotionRequest,
  PerturbationRequest,
  StoredMessage,
  StreamEvent,
  ModelParams,
} from "./chat-types";
import { SseParser } from "./sse";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    credentials: "same-origin",
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (response.status === 204) return undefined as T;
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(response.status, extractDetail(body, response.status));
  return body as T;
}

export const chatApi = {
  status: () => json<ChatStatus>("/v1/chat/status"),
  list: () => json<ConversationSummary[]>("/v1/chat/conversations"),
  create: () => json<ConversationSummary>("/v1/chat/conversations", { method: "POST", body: "{}" }),
  get: (id: string) =>
    json<ConversationSummary & { messages: StoredMessage[] }>(`/v1/chat/conversations/${id}`),
  /** Runs the numerical solution with the user's own numbers; the reply is the stored card message. */
  integrateMotion: (id: string, request: MotionRequest) =>
    json<StoredMessage>(`/v1/chat/conversations/${id}/tools/integrate_motion`, {
      method: "POST",
      body: JSON.stringify(request),
    }),
  /** Asks for more about the geometry on a card; its whole request is posted back unchanged. */
  geometry: (id: string, request: GeometryRequest) =>
    json<StoredMessage>(`/v1/chat/conversations/${id}/tools/geometry`, {
      method: "POST",
      body: JSON.stringify({ ...request, format: undefined }),
    }),
  /** Runs a variation card with the numbers the user typed (the model cannot run these steps). */
  variationDirectly: (id: string, request: VariationRequest) =>
    json<StoredMessage>(`/v1/chat/conversations/${id}/tools/variation`, {
      method: "POST",
      body: JSON.stringify(request),
    }),
  /** Runs a delta card again with the user's answer to "is that a Dirac delta?" (no model involved). */
  distribution: (id: string, request: DistributionAnswer) =>
    json<StoredMessage>(`/v1/chat/conversations/${id}/tools/distribution`, {
      method: "POST",
      body: JSON.stringify(request),
    }),
  /** Equilibria and local stability, with the parameter values and window the user typed. */
  odeQualitative: (id: string, request: OdeAnalyzeInput) =>
    json<StoredMessage>(`/v1/chat/conversations/${id}/tools/ode_analyze`, {
      method: "POST",
      body: JSON.stringify(request),
    }),
  /** Finds a closed form when the user presses the button on an analysis card. */
  odeSolveSymbolic: (id: string, request: OdeInput | OdeSolveInput) =>
    json<StoredMessage>(`/v1/chat/conversations/${id}/tools/ode_solve_symbolic`, {
      method: "POST",
      body: JSON.stringify(request),
    }),
  /** Integrates with the initial values, parameters and interval the user typed into the form. */
  odeSolveNumerical: (id: string, request: OdeNumericInput) =>
    json<StoredMessage>(`/v1/chat/conversations/${id}/tools/ode_solve_numerical`, {
      method: "POST",
      body: JSON.stringify(request),
    }),
  /** Runs a statistical-mechanics card again with the user's own choices (no model involved). */
  statmech: (id: string, request: StatToolArgs & StatAnswers) =>
    json<StoredMessage>(`/v1/chat/conversations/${id}/tools/statistical_mechanics`, {
      method: "POST",
      body: JSON.stringify(request),
    }),
  /** Solves again with the comparison numbers the user typed; the reply is the stored card message. */
  perturbationSolve: (id: string, request: PerturbationRequest) =>
    json<StoredMessage>(`/v1/chat/conversations/${id}/tools/perturbation_solve`, {
      method: "POST",
      body: JSON.stringify(request),
    }),
  /** Checks a key and model without a paid completion. The key is sent once and not stored. */
  testOpenRouter: (apiKey: string | undefined, model: string) =>
    json<ConnectionResult>("/v1/chat/providers/openrouter/test", {
      method: "POST",
      body: JSON.stringify({ api_key: apiKey || undefined, model }),
    }),
  remove: (id: string) => json<void>(`/v1/chat/conversations/${id}`, { method: "DELETE" }),
};

/** Asks the server to transcribe the formulas in a picture (nothing is stored). */
export async function readFormulaImage(
  base64: string,
  signal?: AbortSignal,
): Promise<ExtractedFormula[]> {
  const body = await json<{ formulas: ExtractedFormula[] }>("/v1/chat/formula-image", {
    method: "POST",
    body: JSON.stringify({ image: base64 }),
    signal,
  });
  return body.formulas;
}

/** Sends a message and calls `onEvent` for every streamed event until the reply is complete. */
export async function sendMessage(
  conversationId: string,
  content: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
  provider?: ProviderChoice,
  params?: Partial<ModelParams>,
): Promise<void> {
  const response = await fetch(`/api/v1/chat/conversations/${conversationId}/messages`, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content,
      ...(provider ? { provider } : {}),
      ...(params && Object.keys(params).length > 0 ? { params } : {}),
    }),
    signal,
  });
  if (!response.ok || !response.body) {
    const body: unknown = await response.json().catch(() => null);
    throw new ApiError(response.status, extractDetail(body, response.status));
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const event of parser.push(decoder.decode(value, { stream: true }))) onEvent(event);
  }
}
