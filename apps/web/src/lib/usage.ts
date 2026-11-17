import type { LocalModel, StoredMessage } from "./chat-types";

/** Fraction of the window from which the meter warns. */
export const NEAR_LIMIT = 0.8;

export interface TokenSums {
  prompt: number;
  completion: number;
  /** Model calls the provider reported no count for: the sums then leave them out. */
  unreported: number;
}

export interface ConversationUsage {
  /** The window of the last reply's model in tokens; null for a hosted model (not known here). */
  window: number | null;
  /** What the last request used: the provider's own prompt count, or an estimate (never mixed). */
  used: number | null;
  estimated: boolean;
  /** Earlier messages left out of the last reply to fit the window. */
  trimmed: number;
  /** The last reply's calls, and every reply of the conversation. */
  lastTurn: TokenSums | null;
  total: TokenSums;
  turns: number;
  /** Which model answered last, for the memory figure. */
  model: string;
  provider: string;
}

function sums(content: StoredMessage["content"]): TokenSums | null {
  const calls = content.debug?.llm_calls;
  if (!calls) return null;
  const out: TokenSums = { prompt: 0, completion: 0, unreported: 0 };
  for (const c of calls) {
    if (typeof c.prompt_tokens !== "number" && typeof c.completion_tokens !== "number") {
      out.unreported += 1;
      continue;
    }
    out.prompt += c.prompt_tokens ?? 0;
    out.completion += c.completion_tokens ?? 0;
  }
  return out;
}

/**
 * What the conversation has used, read from the debug data stored with every reply (counts only).
 * Nothing is invented: a count the provider did not report stays missing, and the one estimate
 * (the server's own pessimistic one) is labelled as such. Null before any reply carries data.
 */
export function conversationUsage(messages: StoredMessage[]): ConversationUsage | null {
  const total: TokenSums = { prompt: 0, completion: 0, unreported: 0 };
  let turns = 0;
  let last: StoredMessage | null = null;
  for (const m of messages) {
    if (m.role !== "assistant" || !m.content.debug) continue;
    const s = sums(m.content);
    if (!s) continue;
    turns += 1;
    total.prompt += s.prompt;
    total.completion += s.completion;
    total.unreported += s.unreported;
    last = m;
  }
  if (!last?.content.debug) return null;
  const debug = last.content.debug;
  const calls = debug.llm_calls ?? [];
  const reported = [...calls].reverse().find((c) => typeof c.prompt_tokens === "number");
  const cloud = debug.provider === "openrouter";
  let used: number | null = null;
  let estimated = false;
  if (reported && typeof reported.prompt_tokens === "number") {
    used = reported.prompt_tokens;
  } else if (debug.prompt_estimate_tokens) {
    used = debug.prompt_estimate_tokens;
    estimated = true;
  }
  return {
    // `context_window` is the local window the server sent; it means nothing for a hosted model.
    window: !cloud && debug.context_window ? debug.context_window : null,
    used,
    estimated,
    trimmed: last.content.context_trimmed ?? debug.history_dropped ?? 0,
    lastTurn: sums(last.content),
    total,
    turns,
    model: debug.model ?? "",
    provider: debug.provider ?? "",
  };
}

export type Level = "ok" | "near" | "full";

export function level(used: number | null, window: number | null): Level {
  if (used === null || !window) return "ok";
  const f = used / window;
  return f >= 1 ? "full" : f >= NEAR_LIMIT ? "near" : "ok";
}

/** "6.2 GB": decimal gigabytes as the host reports them; null when it reported nothing. */
export function formatGB(bytes: number | null | undefined): string | null {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes <= 0) return null;
  const gb = bytes / 1e9;
  return `${gb >= 10 ? gb.toFixed(0) : gb.toFixed(1)} GB`;
}

/** The memory line for a local model: total, and the part in GPU memory when the host says so. */
export function memoryLine(model: LocalModel | undefined): string | null {
  const total = formatGB(model?.size_bytes);
  if (!total || !model) return null;
  const vram = formatGB(model.size_vram_bytes);
  if (model.loaded) {
    return vram ? `${total} in memory (${vram} on the GPU)` : `${total} in memory`;
  }
  return `${total} on disk, not loaded`;
}
