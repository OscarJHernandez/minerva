import type { LLMCall, ModelParams, TurnDebug } from "./chat-types";

/** Ranges the server enforces (agent/params.py); the inputs use them as hints, the server decides. */
export const PARAM_LIMITS = {
  num_ctx: { min: 2048, max: 262144, step: 1024 },
  num_predict: { min: 0, max: 65536, step: 100 },
  temperature: { min: 0, max: 2, step: 0.1 },
  top_p: { min: 0.01, max: 1, step: 0.05 },
  top_k: { min: 1, max: 1000, step: 1 },
  history_messages: { min: 2, max: 200, step: 1 },
  history_tokens: { min: 256, max: 262144, step: 1024 },
} as const;

export type Overrides = Partial<ModelParams>;

/** Drops fields the user cleared or set back to the server default. */
export function cleanOverrides(next: Overrides, defaults: ModelParams): Overrides {
  const out: Overrides = {};
  for (const key of Object.keys(next) as (keyof ModelParams)[]) {
    const value = next[key];
    if (value === undefined || value === null || Number.isNaN(value)) continue;
    if (value === defaults[key]) continue;
    (out as Record<string, unknown>)[key] = value;
  }
  return out;
}

/**
 * The prompt filled the window, or shows the signature of a host cutting it: a host handed more than its
 * window keeps about half of it (8 195 tokens of a 16 384 window, measured).
 */
export function reachedWindow(promptTokens: number, window: number): boolean {
  return promptTokens >= window || Math.abs(promptTokens - (Math.floor(window / 2) + 3)) <= 2;
}

function peak(calls: LLMCall[]): number | null {
  const tokens = calls.map((c) => c.prompt_tokens).filter((n): n is number => n != null);
  return tokens.length > 0 ? Math.max(...tokens) : null;
}

/** What produced a reply: the parameters, the tool groups, and the request against the window. */
export function settingsSummary(d: TurnDebug | null | undefined): string[] {
  if (!d) return [];
  const lines: string[] = [];
  const params = d.params ?? {};
  if (Object.keys(params).length > 0) {
    lines.push(
      "settings: " +
        Object.entries(params)
          .map(([k, v]) => `${k} ${v === null ? "model default" : String(v)}`)
          .join(", "),
    );
  }
  if (d.groups && d.groups.length > 0) lines.push(`tool groups loaded: ${d.groups.join(", ")}`);
  if (d.context_window) {
    const seen = peak(d.llm_calls);
    lines.push(
      `context window ${d.context_window} tokens · request about ${d.prompt_estimate_tokens ?? "?"} (estimated)` +
        (seen != null ? ` · host counted ${seen}` : "") +
        (seen != null && reachedWindow(seen, d.context_window)
          ? " · AT THE WINDOW: the host may have cut it"
          : ""),
    );
  }
  if (d.history_dropped) {
    lines.push(`${d.history_dropped} earlier message(s) left out so the request fit the window`);
  }
  return lines;
}
