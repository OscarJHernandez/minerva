import { ApiError, extractDetail } from "./api";
import type { ChatStatus, ConversationSummary, StoredMessage, StreamEvent } from "./chat-types";
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
  remove: (id: string) => json<void>(`/v1/chat/conversations/${id}`, { method: "DELETE" }),
};

/** Sends a message and calls `onEvent` for every streamed event until the reply is complete. */
export async function sendMessage(
  conversationId: string,
  content: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`/api/v1/chat/conversations/${conversationId}/messages`, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
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
