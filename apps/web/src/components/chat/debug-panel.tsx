"use client";

import { useState } from "react";
import type { AssistantContent } from "@/lib/chat-types";
import { buildTimeline, stageTimings, STOP_EXPLANATION } from "@/lib/trace-timeline";

const TONE = {
  neutral: "text-muted",
  ok: "text-ok",
  warn: "text-warn",
  bad: "text-bad",
} as const;

export function CopyButton({ label, getText }: { label: string; getText: () => string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  return (
    <button
      type="button"
      className="text-xs text-accent hover:underline"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(getText());
          setState("copied");
        } catch {
          setState("failed");
        }
        setTimeout(() => setState("idle"), 1500);
      }}
    >
      {state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : label}
    </button>
  );
}

/**
 * What the agent did in one turn, in order: what it chose, whether the arguments were valid, what was
 * verified and why a result was accepted or not, and why it went on, retried or stopped; then the raw
 * arguments and results, which are your conversation content.
 */
export function DebugPanel({
  content,
  getReport,
}: {
  content: Partial<AssistantContent>;
  getReport: () => string;
}) {
  const trace = content.trace ?? [];
  const d = content.debug;
  const timeline = buildTimeline(d, trace);
  const stages = stageTimings(d);
  const stop = d?.stop_reason ?? content.stop_reason ?? null;
  const summary =
    `Trace · ${trace.length} tool call${trace.length === 1 ? "" : "s"}` +
    (d ? ` · ${d.duration_ms} ms · ${d.model}` : " · no timing recorded") +
    (stop && stop !== "completed" ? ` · stopped: ${stop.replaceAll("_", " ")}` : "");
  return (
    <details className="rounded border border-rule bg-sunk px-3 py-2 text-xs">
      <summary className="cursor-pointer text-muted">{summary}</summary>
      <div className="mt-2 space-y-3">
        {d && (
          <p className="break-all text-muted">
            trace {d.trace_id || "-"} · request {d.request_id || "-"} · prompt {d.prompt_sha} ·{" "}
            {d.llm_calls.length} model call{d.llm_calls.length === 1 ? "" : "s"} (
            {d.llm_calls.map((c) => `${c.ms} ms`).join(", ")})
            {d.evidence_pct != null
              ? ` · ${d.formulas_with_evidence}/${d.formulas} formulas with engine evidence`
              : ""}
          </p>
        )}
        {stop && stop !== "completed" && (
          <p className="text-warn">Ended early: {STOP_EXPLANATION[stop] ?? stop}.</p>
        )}
        {timeline.length > 0 && (
          <div>
            <div className="font-medium">Timeline</div>
            <ol className="mt-1 space-y-1.5" data-testid="trace-timeline">
              {timeline.map((row) => (
                <li key={row.id} data-kind={row.kind} className="grid grid-cols-[4.5rem_1fr] gap-2">
                  <span className="text-right tabular-nums text-muted">{row.t_ms} ms</span>
                  <span>
                    <span className={TONE[row.tone]}>{row.title}</span>
                    <span className="block break-words text-muted">{row.detail}</span>
                  </span>
                </li>
              ))}
            </ol>
          </div>
        )}
        {stages.length > 0 && (
          <p className="text-muted">
            {stages.map(([label, ms]) => `${label.trim()} ${ms} ms`).join(" · ")}
          </p>
        )}
        {trace.length > 0 && (
          <p className="text-muted">
            Arguments and results below are your conversation content: they are stored only in this
            conversation and are not written to the server log.
          </p>
        )}
        {trace.map((t, i) => (
          <div key={t.id}>
            <div className="font-medium">
              {i + 1}. {t.name} · {t.status ?? "?"} · {t.duration_ms ?? "?"} ms
              {t.error_category ? ` · ${t.error_category}` : ""}
              {t.result_id ? ` · ${t.result_id}` : ""}
            </div>
            <div className="mt-1 text-muted">Arguments the model sent</div>
            <pre className="overflow-x-auto whitespace-pre-wrap break-all">
              {JSON.stringify(t.model_arguments ?? t.arguments, null, 2)}
            </pre>
            {t.model_arguments && (
              <>
                <div className="mt-1 text-muted">After replacing references to earlier results</div>
                <pre className="overflow-x-auto whitespace-pre-wrap break-all">
                  {JSON.stringify(t.arguments, null, 2)}
                </pre>
              </>
            )}
            <div className="mt-1 text-muted">What the model read back</div>
            <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-all">
              {t.for_model}
            </pre>
          </div>
        ))}
        {(d?.claims ?? []).map((c, i) => (
          <p key={i} className="break-all">
            claim {c.status}
            {c.kind ? ` (${c.kind})` : ""}: <code>{c.value}</code>
            {c.detail ? ` (${c.detail})` : ""}
          </p>
        ))}
        <CopyButton label="Copy this turn" getText={getReport} />
      </div>
    </details>
  );
}
