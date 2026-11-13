import type {
  AssistantContent,
  Card,
  ChatStatus,
  StoredMessage,
  TraceStep,
  TurnDebug,
} from "./chat-types";

/** Bump when the JSON block changes shape, so `scripts/replay_trace.py` can tell. */
export const REPORT_FORMAT = 1;
export const JSON_START = "<!-- minerva-debug-json -->";
export const JSON_END = "<!-- /minerva-debug-json -->";
const MAX_SHOWN = 4000;

export interface LastError {
  message: string;
  debug?: TurnDebug;
  trace?: TraceStep[];
}

export interface ReportInput {
  title: string;
  messages: StoredMessage[];
  status: ChatStatus | null;
  lastError?: LastError | null;
  now?: Date;
}

/** A code fence longer than any run of backticks inside `text`, so content cannot close it early. */
export function fence(text: string, lang = ""): string {
  const longest = Math.max(0, ...(text.match(/`+/g) ?? []).map((m) => m.length));
  const ticks = "`".repeat(Math.max(3, longest + 1));
  return `${ticks}${lang}\n${text}\n${ticks}`;
}

function clip(text: string): string {
  return text.length <= MAX_SHOWN
    ? text
    : `${text.slice(0, MAX_SHOWN)}\n[... ${text.length - MAX_SHOWN} more characters; see the JSON block]`;
}

function cardSummary(card: Card | null | undefined): string[] {
  if (!card) return [];
  const out = [`- card: ${card.type}`];
  if (card.type === "simplify") {
    const d = card.data;
    out.push(`- status: ${d.status}`);
    out.push(`- input as read: ${d.input_latex ?? "-"}`);
    out.push(`- result: ${d.result_latex ?? "-"}`);
    if (d.verification) {
      out.push(
        `- verification: numeric ${d.verification.numeric_points_agreed}/${d.verification.numeric_points_checked}, symbolic zero ${d.verification.symbolic_zero}`,
      );
    }
    out.push(`- engine message: ${d.message}`);
  } else if (
    card.type === "solve" ||
    card.type === "lagrangian" ||
    card.type === "motion" ||
    card.type === "dimensional" ||
    card.type === "units"
  ) {
    out.push(`- status: ${card.data.report.status}`, `- message: ${card.data.report.message}`);
  } else if (card.type === "error") {
    out.push(`- message: ${card.data.message}`);
  } else if ("status" in card.data) {
    out.push(`- status: ${String(card.data.status)}`);
  }
  return out;
}

function formulas(content: Partial<AssistantContent>) {
  return (content.blocks ?? [])
    .filter((b) => b.kind === "text")
    .flatMap((b) => b.segments)
    .filter((s) => s.type !== "text")
    .map((s) => ({ value: s.value, status: s.status, note: s.note ?? null }));
}

function turnMarkdown(m: StoredMessage, n: number): string[] {
  if (m.role === "user") return [`### ${n}. User`, fence(m.content.text)];
  const c = m.content;
  const lines = [`### ${n}. Assistant`];
  const d = c.debug;
  if (d) {
    lines.push(
      `request ${d.request_id || "-"} · ${d.duration_ms} ms · ${d.llm_calls.length} model call(s) · ` +
        `model ${d.model} · prompt ${d.prompt_sha}`,
    );
  }
  (c.trace ?? []).forEach((t, i) => {
    const card = (c.blocks ?? []).find((b) => b.kind === "card" && b.card.id === t.id)?.card;
    lines.push(
      `#### Tool call ${i + 1}: ${t.name} (${t.status ?? "status unknown"}, ${t.duration_ms ?? "?"} ms)`,
      "Arguments the model sent:",
      fence(clip(JSON.stringify(t.arguments, null, 2)), "json"),
      ...cardSummary(card),
      "What the model read back:",
      fence(clip(t.for_model)),
    );
  });
  lines.push("Model prose, raw:", fence(c.text || "(none)"));
  const marks = formulas(c);
  if (marks.length > 0) {
    lines.push(
      "Formula marks:",
      ...marks.map(
        (f) => `- ${f.status ?? "unmarked"}: \`${f.value}\`${f.note ? ` (${f.note})` : ""}`,
      ),
    );
  }
  for (const claim of d?.claims ?? []) {
    lines.push(
      `- claim check ${claim.status}${claim.kind ? ` [${claim.kind}]` : ""}: \`${claim.value}\`${claim.detail ? ` (${claim.detail})` : ""}${claim.given?.length ? ` given: ${claim.given.join("; ")}` : ""}`,
    );
  }
  if (c.truncated) lines.push("- the step or time limit ended this turn early");
  return lines;
}

/** The machine-readable part: what `scripts/replay_trace.py` re-runs. Cards are summarised above. */
export function reportData(input: ReportInput) {
  return {
    format: REPORT_FORMAT,
    generated_at: (input.now ?? new Date()).toISOString(),
    title: input.title,
    app: { mode: input.status?.mode ?? null, model: input.status?.model ?? null },
    messages: input.messages.map((m) => ({
      role: m.role,
      created_at: m.created_at,
      text: m.content.text,
      trace: m.content.trace ?? [],
      debug: m.content.debug ?? null,
      truncated: m.content.truncated ?? false,
      formulas: m.role === "assistant" ? formulas(m.content) : [],
    })),
    last_error: input.lastError ?? null,
  };
}

/** Markdown for a human or agent to read, ending with a JSON block for scripts. */
export function buildDebugReport(input: ReportInput): string {
  const data = reportData(input);
  const lines = [
    "# Minerva chat debug report",
    `format ${REPORT_FORMAT} · ${data.generated_at} · mode ${data.app.mode ?? "?"} · model ${data.app.model ?? "?"}`,
    "",
    "## Conversation",
  ];
  input.messages.forEach((m, i) => lines.push("", ...turnMarkdown(m, i + 1)));
  if (input.lastError) {
    lines.push("", "## Last turn failed", fence(input.lastError.message));
    if (input.lastError.debug) {
      lines.push(`request ${input.lastError.debug.request_id || "-"}`);
    }
  }
  lines.push(
    "",
    "## Raw data",
    JSON_START,
    fence(JSON.stringify(data, null, 2), "json"),
    JSON_END,
    "",
  );
  return lines.join("\n");
}

/** One assistant turn with the user message before it, for the per-message copy button. */
export function buildTurnReport(
  input: Omit<ReportInput, "messages">,
  messages: StoredMessage[],
  index: number,
): string {
  const from = index > 0 && messages[index - 1]?.role === "user" ? index - 1 : index;
  return buildDebugReport({ ...input, messages: messages.slice(from, index + 1) });
}
