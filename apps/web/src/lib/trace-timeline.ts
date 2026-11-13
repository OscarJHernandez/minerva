import type { TraceEvent, TraceStep, TurnDebug } from "./chat-types";

/**
 * A chronological, human-readable account of one agent turn, built from the content-free trace events
 * and the stored trace. It answers: what the agent chose, whether the arguments were valid, what the tool
 * returned, what was verified, why a result was accepted or not, and why the turn went on, retried or
 * stopped. Used by the debug panel and the copied report, so they cannot disagree.
 *
 * Events hold no content; the only content here comes from `trace` (the tool name and status), and only
 * names and statuses are used, never arguments or results.
 */
export interface TimelineRow {
  id: string;
  /** Milliseconds since the turn started (monotonic clock on the server). */
  t_ms: number;
  step: number | null;
  kind: "start" | "model" | "tool" | "verify" | "retry" | "stop" | "end";
  title: string;
  detail: string;
  tone: "neutral" | "ok" | "warn" | "bad";
}

/** Why a turn ended, in the words a developer needs. */
export const STOP_EXPLANATION: Record<string, string> = {
  completed: "the model answered without asking for more tools",
  step_limit: "the limit on model steps for one answer was reached",
  tool_call_limit: "the limit on tool calls for one answer was reached",
  time_limit: "the time limit for one answer was reached",
  repeated_calls:
    "the model kept repeating calls or making calls that established nothing, so the turn was wrapped up",
  model_error: "the model host failed (a failed call is retried once before this)",
  malformed_output: "the model's tool calls could not be used, even after one correction",
  internal_error: "an internal error; the server log has the trace id and counters only",
};

/** Why a tool result was or was not accepted as an engine result. */
export const REASON_EXPLANATION: Record<string, string> = {
  verified: "the engine verified it",
  lookup: "a handbook record was retrieved (not computed)",
  unchanged:
    "nothing verified was found, so the input stands (this is not a proof that none exists)",
  not_verified: "a candidate was found but did not pass the verification, so it is not shown",
  conditions_not_satisfied: "a step needs a condition the declared assumptions do not establish",
  unsupported: "outside what the engine supports; no fallback was tried",
  parse_error: "the engine could not read the input",
  resource_limit: "the time or memory limit was reached",
  internal_error: "the engine failed unexpectedly",
  no_results: "nothing matched",
  error: "the call was refused or failed",
};

const num = (v: unknown): number | null => (typeof v === "number" ? v : null);
const str = (v: unknown): string => (typeof v === "string" ? v : "");

function points(e: TraceEvent): string {
  const checked = num(e.points_checked);
  const agreed = num(e.points_agreed);
  return checked === null ? "" : ` · ${agreed ?? "?"}/${checked} sample points agreed`;
}

/** Rows in the order they happened; empty for messages stored before tracing existed. */
export function buildTimeline(
  debug: TurnDebug | null | undefined,
  trace: TraceStep[] | undefined,
): TimelineRow[] {
  const events = debug?.events ?? [];
  const byCall = new Map((trace ?? []).map((t) => [t.id, t]));
  const rows: TimelineRow[] = [];
  const startedTool = new Map<string, TraceEvent>();
  for (const e of events) {
    const base = { id: `${e.trace_id}-${e.seq}`, t_ms: e.t_ms, step: e.step ?? null };
    switch (e.event) {
      case "agent.started":
        rows.push({
          ...base,
          kind: "start",
          tone: "neutral",
          title: "Turn started",
          detail:
            `model ${str(e.model) || "?"} · up to ${num(e.max_steps) ?? "?"} steps, ` +
            `${num(e.max_tool_calls) ?? "?"} tool calls, ${num(e.max_seconds) ?? "?"} s`,
        });
        break;
      case "llm.request":
        break; // merged into the answer row
      case "llm.response": {
        const tool = str(e.proposed_tool);
        const calls = num(e.tool_calls) ?? 0;
        const tokens =
          num(e.tokens_in) !== null || num(e.tokens_out) !== null
            ? ` · tokens ${num(e.tokens_in) ?? "?"} in, ${num(e.tokens_out) ?? "?"} out`
            : "";
        const finish = str(e.finish_reason);
        const failed = finish === "error" || finish === "timeout";
        rows.push({
          ...base,
          kind: "model",
          tone: failed ? "bad" : "neutral",
          title: `Model call ${base.step ?? ""}: ${failed ? `failed (${finish})` : calls > 0 ? `asked for ${calls} tool call${calls === 1 ? "" : "s"}` : "answered in words"}`,
          detail:
            `${num(e.latency_ms) ?? "?"} ms` +
            (num(e.first_token_ms) !== null
              ? ` (first output after ${num(e.first_token_ms)} ms)`
              : "") +
            ` · finish ${finish || "?"}${tokens}` +
            (tool ? ` · first proposed tool: ${tool}` : "") +
            (calls > 0 ? " · continues: the tool results go back to the model" : ""),
        });
        break;
      }
      case "tool.started":
        startedTool.set(str(e.call_id), e);
        break;
      case "tool.completed": {
        const started = startedTool.get(str(e.call_id));
        const step = byCall.get(str(e.call_id));
        const valid = started ? started.arguments_valid !== false : step?.arguments_valid !== false;
        const status = str(e.status);
        const category = str(e.error_category);
        const repeated = started?.repeated === true || step?.repeated === true;
        const refs = num(started?.references);
        const bad = status === "error" || status === "skipped" || status === "other";
        rows.push({
          ...base,
          kind: "tool",
          tone:
            repeated || status === "skipped"
              ? "warn"
              : bad
                ? "bad"
                : status === "ok"
                  ? "ok"
                  : "warn",
          title: `Tool ${str(e.tool)}: ${repeated ? "not run again (an identical call already ran)" : status === "skipped" ? "not run (limit reached)" : `status ${status}`}`,
          detail:
            `arguments ${valid ? "valid" : "INVALID"}` +
            (refs
              ? ` · ${refs} reference${refs === 1 ? "" : "s"} to earlier results replaced`
              : "") +
            (num(e.duration_ms) !== null ? ` · ${num(e.duration_ms)} ms` : "") +
            (num(e.engine_ms) !== null
              ? ` (${num(e.engine_ms)} ms in the sandbox, ${num(e.sandbox_calls) ?? 0} call${num(e.sandbox_calls) === 1 ? "" : "s"})`
              : "") +
            (category ? ` · category ${category}` : "") +
            (str(e.result_id) ? ` · result id ${str(e.result_id)}` : "") +
            (step?.assumptions_dropped?.length
              ? ` · declared earlier but not repeated here: ${step.assumptions_dropped.join(", ")}`
              : ""),
        });
        break;
      }
      case "verification.completed": {
        if (e.scope === "claims") {
          const n = num(e.claims) ?? 0;
          rows.push({
            ...base,
            kind: "verify",
            tone: e.accepted === true ? "neutral" : "warn",
            title: `Claim check of the model's own formulas: ${n} tested`,
            detail:
              `${num(e.claims_holding) ?? 0} hold, ${num(e.claims_conditional) ?? 0} conditional, ` +
              `${num(e.claims_contradicted) ?? 0} contradicted, ${num(e.claims_unchecked) ?? 0} unchecked` +
              (num(e.verify_ms) !== null ? ` · ${num(e.verify_ms)} ms` : ""),
          });
          break;
        }
        const accepted = e.accepted === true;
        const reason = str(e.reason);
        const restrictions = num(e.restrictions);
        rows.push({
          ...base,
          kind: "verify",
          tone: accepted ? "ok" : "warn",
          title: `Verification of ${str(e.tool)}: ${accepted ? "accepted" : "rejected"}`,
          detail:
            `check ${str(e.check)}${points(e)}` +
            (e.symbolic_zero === true ? " · symbolic zero test passed" : "") +
            (restrictions
              ? ` · ${restrictions} restriction${restrictions === 1 ? "" : "s"}${e.restrictions_complete === false ? " (list may be incomplete)" : ""}`
              : "") +
            ` · ${REASON_EXPLANATION[reason] ?? reason}`,
        });
        break;
      }
      case "agent.retry":
        rows.push({
          ...base,
          kind: "retry",
          tone: "warn",
          title: `Retry ${num(e.attempt) ?? 1}: ${str(e.reason).replaceAll("_", " ")}`,
          detail:
            e.reason === "model_host"
              ? "the model call failed before any output; it is tried once more (no computation is repeated)"
              : "the model wrote an unusable tool call; it is told and may try once more",
        });
        break;
      case "agent.stopped": {
        const reason = str(e.reason);
        rows.push({
          ...base,
          kind: "stop",
          tone: reason === "completed" ? "ok" : "warn",
          title: `Stopped: ${reason.replaceAll("_", " ")}`,
          detail:
            `${STOP_EXPLANATION[reason] ?? ""} · ${num(e.steps) ?? 0} model call(s), ${num(e.tool_calls) ?? 0} tool call(s)` +
            (str(e.error_category) ? ` · category ${str(e.error_category)}` : ""),
        });
        break;
      }
      case "agent.completed":
        rows.push({
          ...base,
          kind: "end",
          tone: e.status === "ok" ? "neutral" : "bad",
          title: `Finished in ${num(e.total_ms) ?? "?"} ms (${str(e.status)})`,
          detail:
            `model ${num(e.llm_ms) ?? 0} ms · tools ${num(e.tool_ms) ?? 0} ms (sandbox ${num(e.engine_ms) ?? 0} ms) · ` +
            `claim check ${num(e.verify_ms) ?? 0} ms · formulas with engine evidence ` +
            `${num(e.formulas_with_evidence) ?? 0}/${num(e.formulas) ?? 0}` +
            (num(e.evidence_pct) !== null ? ` (${num(e.evidence_pct)}%)` : ""),
        });
        break;
      default:
        break;
    }
  }
  return rows;
}

/** The stage timings of a turn as `[label, milliseconds]`, from the final event. */
export function stageTimings(debug: TurnDebug | null | undefined): [string, number][] {
  const done = [...(debug?.events ?? [])].reverse().find((e) => e.event === "agent.completed");
  if (!done) return [];
  const pick = (label: string, key: string): [string, number][] =>
    typeof done[key] === "number" ? [[label, done[key] as number]] : [];
  return [
    ...pick("total", "total_ms"),
    ...pick("model calls", "llm_ms"),
    ...pick("tool calls", "tool_ms"),
    ...pick("  of which in the sandbox", "engine_ms"),
    ...pick("claim check", "verify_ms"),
  ];
}
