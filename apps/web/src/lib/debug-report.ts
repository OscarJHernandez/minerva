import type {
  AssistantContent,
  Card,
  ChatStatus,
  StoredMessage,
  TraceStep,
  TurnDebug,
} from "./chat-types";
import { buildTimeline, stageTimings, STOP_EXPLANATION } from "./trace-timeline";

/**
 * Bump when the JSON block changes shape, so `scripts/replay_trace.py` can tell.
 * 2: adds trace ids, stop reasons, events, evidence and error categories per step.
 */
export const REPORT_FORMAT = 2;
export const CONTENT_NOTICE =
  "This report contains conversation content (your messages, expressions and tool results). " +
  "Share it only where you would share the conversation. The server log never holds this content.";
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
  } else if (card.type === "step") {
    const r = card.data.result;
    out.push(`- operation: ${card.data.operation}`, `- status: ${r.status}`);
    if (card.data.operation === "isolate") {
      out.push(`- solutions: ${card.data.result.solutions.length}`);
      out.push(`- conditions on all: ${card.data.result.conditions_latex.join("; ") || "-"}`);
    } else {
      out.push(`- input as read: ${card.data.result.input_latex ?? "-"}`);
      out.push(`- result: ${card.data.result.result_latex ?? "-"}`);
      out.push(`- evidence: ${card.data.result.evidence}`);
    }
    const d = r.diagnostics;
    out.push(
      `- diagnostics: read ${d.parse_ms} ms, candidates ${d.candidates_ms} ms, transform ${d.transform_ms} ms, ` +
        `verify ${d.verify_ms} ms; ${d.candidates} candidate(s); rejected ${JSON.stringify(d.rejected)}; ` +
        `check ${d.verification_method}: ${d.verification_outcome}; restrictions ${d.restrictions_outcome}`,
    );
    out.push(`- engine message: ${r.message}`);
  } else if (card.type === "matrix") {
    const r = card.data.result;
    out.push(`- status: ${r.status}`, `- action: ${r.action ?? "-"}`);
    out.push(
      `- method: ${r.method ?? "-"}; kind: ${r.result_kind ?? "-"}; evaluated: ${r.evaluated}`,
    );
    if (r.reason) out.push(`- reason: ${r.reason}`);
    if (r.derivative) {
      out.push(
        `- derivative: ${r.derivative.kind}, constraint ${r.derivative.constraint}, convention ${r.derivative.convention}, result ${r.result_text ?? "-"}`,
      );
    }
    if (r.verification) {
      out.push(
        `- verification: numeric ${r.verification.points_agreed}/${r.verification.points_checked}, exact leg ${r.verification.symbolic_zero ?? "not run"}`,
      );
    }
    const d = r.diagnostics;
    if (d) {
      const tried = d.methods_tried.map((m) => `${m.id}:${m.outcome}`).join(", ");
      const stages = d.stages.map((s) => `${s.name} ${s.ms.toFixed(0)} ms`).join(", ");
      out.push(
        `- diagnostics: trace ${d.trace_id ?? "-"}; tried [${tried}]; commutation ${d.commutation}; ` +
          `digits ${d.digits ?? "-"}, tolerance ${d.tolerance ?? "-"}, residual ${d.max_residual ?? "-"}; ` +
          `fallbacks [${d.fallbacks.join(", ")}]; failure ${d.failure_reason ?? "-"}; ${stages}`,
      );
    }
  } else if (card.type === "sums" || card.type === "expectation") {
    const r = card.data.result;
    out.push(`- status: ${r.status}${r.reason ? ` (${r.reason})` : ""}`, `- action: ${r.action}`);
    out.push(`- input as read: ${r.input_latex ?? "-"}`, `- result: ${r.result_latex ?? "-"}`);
    if ("given" in r)
      out.push(`- evidence: ${r.evidence ?? "-"}; given: ${r.given.join("; ") || "-"}`);
    if (r.verification) {
      const v = r.verification;
      out.push(
        "instances_agreed" in v
          ? `- verification: instances ${v.instances_agreed}/${v.instances_checked}, sizes ${JSON.stringify(v.sizes)}`
          : `- verification: joint distributions ${v.models_agreed}/${v.models_checked}`,
      );
    }
    out.push(`- engine message: ${r.message}`);
  } else if (card.type === "statistics") {
    const r = card.data.result;
    out.push(`- status: ${r.status}${r.reason ? ` (${r.reason})` : ""}`, `- action: ${r.action}`);
    out.push(`- evidence: ${r.evidence ?? "-"}; premises: ${r.premises.join("; ") || "-"}`);
    for (const e of r.estimates) {
      out.push(
        `- estimate ${e.parameter}: ${e.text} [${e.optimum ?? "-"}/${e.optimum_evidence ?? "-"}]`,
      );
    }
    if (r.matrix_text) out.push(`- matrix: ${JSON.stringify(r.matrix_text)}`);
    if (r.posterior)
      out.push(`- posterior ${r.posterior.family}: ${JSON.stringify(r.posterior.parameters)}`);
    out.push(
      `- checks: ${r.checks.map((c) => `${c.kind} ${c.points_agreed}/${c.points_checked}${c.passed ? "" : "!"}`).join(", ") || "-"}`,
    );
    out.push(`- engine message: ${r.message}`);
  } else if (card.type === "geometry") {
    const r = card.data.result;
    out.push(`- status: ${r.status}`, `- dimension: ${r.dimension}`, `- message: ${r.message}`);
    // which quantity failed and how, then the stages: slow, large or failing ones stand out
    for (const q of r.quantities) {
      const checks = q.verification?.checks.map((c) => `${c.kind[0]}:${c.passed}`).join(" ") ?? "-";
      out.push(`- quantity ${q.kind}: ${q.status} (checks ${checks})`);
    }
    out.push(
      `- signature: ${r.signature ?? "not determined"}; restrictions: ${r.restrictions_latex.length}${r.restrictions_complete ? "" : " (incomplete)"}`,
      ...r.diagnostics.map(
        (d) =>
          `- stage ${d.stage}: ${d.duration_ms} ms${d.ops === null ? "" : `, ${d.ops} ops`}, ${d.failure ?? d.status}`,
      ),
    );
  } else if (card.type === "distribution") {
    const d = card.data.result;
    out.push(`- status: ${d.status}${d.reason ? ` (${d.reason})` : ""}`);
    out.push(`- action: ${d.action ?? "-"}`);
    out.push(`- input as read: ${d.input_latex ?? "-"}`);
    out.push(`- result: ${d.result_latex ?? "-"}`);
    if (d.verification) {
      out.push(
        `- verification: ${d.verification.pairings_agreed}/${d.verification.pairings_checked} test-function pairings, conditions established ${d.verification.conditions_established}`,
      );
    }
    out.push(
      `- stages: ${d.diagnostics.events.map((e) => `${e.stage}${e.status === "failed" ? "!" : ""}`).join(" > ")}`,
    );
    out.push(`- engine message: ${d.message}`);
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

/** The model's prose as it wrote it, one string per text block, with the formula delimiters restored. */
function proseBlocks(content: Partial<AssistantContent>): string[] {
  return (content.blocks ?? [])
    .filter((b) => b.kind === "text")
    .map((b) =>
      b.segments
        .map((s) =>
          s.type === "display_math"
            ? `$$${s.value}$$`
            : s.type === "math"
              ? `$${s.value}$`
              : s.value,
        )
        .join(""),
    );
}

function turnMarkdown(m: StoredMessage, n: number): string[] {
  if (m.role === "user") return [`### ${n}. User`, fence(m.content.text)];
  const c = m.content;
  const lines = [`### ${n}. Assistant`];
  const d = c.debug;
  if (d) {
    lines.push(
      `trace ${d.trace_id || "-"} · request ${d.request_id || "-"} · ${d.duration_ms} ms · ` +
        `${d.llm_calls.length} model call(s) · model ${d.model} · prompt ${d.prompt_sha}`,
    );
    const stop = d.stop_reason ?? c.stop_reason ?? null;
    if (stop)
      lines.push(
        `stop reason: ${stop}${STOP_EXPLANATION[stop] ? ` (${STOP_EXPLANATION[stop]})` : ""}`,
      );
    if (d.error) lines.push(`error: ${d.error}`);
    const stages = stageTimings(d);
    if (stages.length > 0) {
      lines.push(
        "",
        "| Stage | ms |",
        "| --- | --- |",
        ...stages.map(([k, ms]) => `| ${k.trim()} | ${ms} |`),
        "",
      );
    }
    const tokens = d.llm_calls.filter(
      (x) => x.prompt_tokens != null || x.completion_tokens != null,
    );
    if (tokens.length > 0) {
      lines.push(
        `tokens: ${tokens.reduce((n, x) => n + (x.prompt_tokens ?? 0), 0)} in, ${tokens.reduce((n, x) => n + (x.completion_tokens ?? 0), 0)} out (as reported by the model host)`,
      );
    }
    if (d.evidence_pct != null) {
      lines.push(
        `formulas with engine evidence: ${d.formulas_with_evidence}/${d.formulas} (${d.evidence_pct}%)`,
      );
    }
    if (d.claim_check && d.claim_check !== "ok") lines.push(`claim check: ${d.claim_check}`);
    const timeline = buildTimeline(d, c.trace);
    if (timeline.length > 0) {
      lines.push("Timeline:", ...timeline.map((r) => `- ${r.t_ms} ms · ${r.title}: ${r.detail}`));
    }
  }
  (c.trace ?? []).forEach((t, i) => {
    const card = (c.blocks ?? []).find((b) => b.kind === "card" && b.card.id === t.id)?.card;
    const ev = t.evidence;
    lines.push(
      `#### Tool call ${i + 1}: ${t.name} (${t.status ?? "status unknown"}, ${t.duration_ms ?? "?"} ms)`,
      `- arguments valid: ${t.arguments_valid ?? "unknown"}${t.error_category ? ` · category: ${t.error_category}` : ""}${t.repeated ? " · repeated call, not run again" : ""}${t.result_id ? ` · result id ${t.result_id}` : ""}${t.engine_ms != null ? ` · ${t.engine_ms} ms in the sandbox` : ""}`,
      ...(ev
        ? [
            `- evidence: ${ev.check}, ${ev.accepted ? "accepted" : "not accepted"} (${ev.reason})` +
              (ev.points_checked != null
                ? `, ${ev.points_agreed ?? "?"}/${ev.points_checked} points`
                : "") +
              (ev.restrictions ? `, ${ev.restrictions} restriction(s)` : ""),
          ]
        : []),
      ...(t.assumptions_dropped?.length
        ? [`- declared earlier but not repeated in this call: ${t.assumptions_dropped.join(", ")}`]
        : []),
      "Arguments the model sent:",
      fence(clip(JSON.stringify(t.model_arguments ?? t.arguments, null, 2)), "json"),
      ...(t.model_arguments
        ? [
            "After replacing references to earlier results:",
            fence(clip(JSON.stringify(t.arguments, null, 2)), "json"),
          ]
        : []),
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
  if (c.truncated) {
    lines.push(
      `- this turn ended early (${c.debug?.stop_reason ?? c.stop_reason ?? "limit or failure"})`,
    );
  }
  return lines;
}

/** The machine-readable part: what `scripts/replay_trace.py` re-runs. Cards are summarised above. */
export function reportData(input: ReportInput) {
  return {
    format: REPORT_FORMAT,
    contains_conversation_content: true,
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
      stop_reason: m.content.stop_reason ?? m.content.debug?.stop_reason ?? null,
      prose: m.role === "assistant" ? proseBlocks(m.content) : [],
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
    `> ${CONTENT_NOTICE}`,
    "",
    "## Conversation",
  ];
  input.messages.forEach((m, i) => lines.push("", ...turnMarkdown(m, i + 1)));
  if (input.lastError) {
    lines.push("", "## Last turn failed", fence(input.lastError.message));
    if (input.lastError.debug) {
      lines.push(
        `trace ${input.lastError.debug.trace_id || "-"} · request ${input.lastError.debug.request_id || "-"}`,
      );
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
