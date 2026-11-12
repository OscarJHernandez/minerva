"use client";

import { useState } from "react";
import type { AssistantContent } from "@/lib/chat-types";

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

/** What the model sent to each tool and what it read back, shown under an assistant message. */
export function DebugPanel({
  content,
  getReport,
}: {
  content: Partial<AssistantContent>;
  getReport: () => string;
}) {
  const trace = content.trace ?? [];
  const d = content.debug;
  const summary =
    `Trace · ${trace.length} tool call${trace.length === 1 ? "" : "s"}` +
    (d ? ` · ${d.duration_ms} ms · ${d.model}` : " · no timing recorded");
  return (
    <details className="rounded border border-rule bg-sunk px-3 py-2 text-xs">
      <summary className="cursor-pointer text-muted">{summary}</summary>
      <div className="mt-2 space-y-3">
        {d && (
          <p className="break-all text-muted">
            request {d.request_id || "-"} · prompt {d.prompt_sha} · {d.llm_calls.length} model call
            {d.llm_calls.length === 1 ? "" : "s"} ({d.llm_calls.map((c) => `${c.ms} ms`).join(", ")}
            )
          </p>
        )}
        {trace.map((t, i) => (
          <div key={t.id}>
            <div className="font-medium">
              {i + 1}. {t.name} · {t.status ?? "?"} · {t.duration_ms ?? "?"} ms
            </div>
            <div className="mt-1 text-muted">Arguments the model sent</div>
            <pre className="overflow-x-auto whitespace-pre-wrap break-all">
              {JSON.stringify(t.arguments, null, 2)}
            </pre>
            <div className="mt-1 text-muted">What the model read back</div>
            <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-all">
              {t.for_model}
            </pre>
          </div>
        ))}
        {(d?.claims ?? []).map((c, i) => (
          <p key={i} className="break-all">
            claim {c.status}: <code>{c.value}</code>
            {c.detail ? ` (${c.detail})` : ""}
          </p>
        ))}
        <CopyButton label="Copy this turn" getText={getReport} />
      </div>
    </details>
  );
}
