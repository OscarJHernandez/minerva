import { ApiError, extractDetail } from "./api";
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
): Promise<void> {
  const response = await fetch(`/api/v1/chat/conversations/${conversationId}/messages`, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(provider ? { content, provider } : { content }),
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
