import raw from "@/data/landing-sessions.json";
import type { FormulaStatus } from "@minerva/ui/marks";

/**
 * Shape of apps/site/src/data/landing-sessions.json, written by scripts/record_landing_sessions.py from live runs
 * of the real chat loop. Nothing here is typed by hand: the text, the marks, the numbers and the results are what
 * the model wrote and the engine returned in the first of five runs that all passed.
 */
export type SessionSegment = {
  type: "text" | "math" | "display_math";
  value: string;
  status?: FormulaStatus;
  number?: number;
};
export type SessionRow = { key: string; latex: string };
export type SessionSection = {
  title: string;
  evidence: string;
  rows: SessionRow[];
  note: string | null;
};
export type SessionCard = {
  kind: "card";
  tool: string;
  title: string;
  status: string;
  numbers: Record<string, number>;
  number: number | null;
  sections: SessionSection[];
  restrictions_latex: string[];
};
export type SessionText = { kind: "text"; segments: SessionSegment[] };
export type SessionTurn = {
  user: { text: string };
  assistant: { blocks: (SessionCard | SessionText)[]; stop_reason: string | null };
};
export type Session = {
  id: string;
  title: string;
  /** Which run is shown (1-based): the first in which every turn passed. */
  shown_run: number;
  /** How many of the runs passed every turn. */
  passed_runs: number;
  turns: SessionTurn[];
};
export type SessionSettings = {
  temperature: number;
  context_window_tokens: number;
  reply_budget_tokens: number | null;
  thinking: boolean;
  top_p: number | null;
  top_k: number | null;
  tool_routing: boolean;
};
export type SessionModel = {
  name: string;
  family?: string;
  parameter_size?: string;
  base_model?: string | null;
  quantization?: string;
  digest?: string;
};
type SessionData = {
  date: string;
  model: SessionModel | null;
  runs_per_conversation: number;
  tool_time_limit_seconds: number;
  thinking: boolean;
  reply_budget_tokens: number;
  seed: number;
  settings: SessionSettings;
  sessions: Session[];
};

const data = raw as unknown as SessionData;

export const RECORDING = {
  date: data.date,
  model: data.model,
  runs: data.runs_per_conversation,
  limit: data.tool_time_limit_seconds,
  seed: data.seed,
  settings: data.settings,
};

/** Every setting that shapes a reply, as the run sent it, in words. */
export function settingsLine(s: SessionSettings, limit: number, seed: number): string {
  const n = (x: number) => x.toLocaleString("en-US");
  return [
    s.thinking ? "thinking on" : "thinking off",
    s.reply_budget_tokens ? `reply budget ${n(s.reply_budget_tokens)} tokens` : "the model's own reply limit",
    `temperature ${s.temperature}`,
    `context window ${n(s.context_window_tokens)} tokens`,
    `tool time limit ${limit} s`,
    `seed ${seed}`,
  ].join(" · ");
}

/** A recorded session by id. Throws at build time if the recording no longer has it. */
export function session(id: string): Session {
  const found = data.sessions.find((s) => s.id === id);
  if (!found) throw new Error(`landing-sessions.json has no session "${id}"; re-record it`);
  return found;
}

/** How the model is named on the page: what the weights are, as the host reported them. */
export function modelLabel(model: SessionModel | null): string {
  if (!model) return "model not reported";
  const parts = [model.family, model.parameter_size, model.quantization].filter(Boolean);
  const digest = model.digest ? ` (digest ${model.digest.slice(0, 12)})` : "";
  return `${model.name}${parts.length ? `: ${parts.join(", ")}` : ""}${digest}`;
}
