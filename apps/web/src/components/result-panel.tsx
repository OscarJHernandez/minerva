"use client";

import { Math } from "@minerva/ui/math";
import { useState } from "react";
import type { SimplifyResult, Status } from "@/lib/types";

const STATUS: Record<Status, { label: string; tone: string }> = {
  ok: { label: "Simplified and verified", tone: "bg-ok-soft text-ok" },
  unchanged: { label: "No simpler form found", tone: "bg-sunk text-muted" },
  not_verified: { label: "Could not verify a result", tone: "bg-warn-soft text-warn" },
  unsupported: { label: "Not supported", tone: "bg-warn-soft text-warn" },
  parse_error: { label: "Could not read the expression", tone: "bg-bad-soft text-bad" },
  resource_limit: { label: "Resource limit reached", tone: "bg-warn-soft text-warn" },
  internal_error: { label: "Computation failed", tone: "bg-bad-soft text-bad" },
};

export function StatusPill({ status }: { status: Status }) {
  const s = STATUS[status];
  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${s.tone}`}>
      {s.label}
    </span>
  );
}

export function ResultPanel({ result }: { result: SimplifyResult }) {
  const [copied, setCopied] = useState(false);
  const v = result.verification;
  const shown = result.status === "ok" || result.status === "unchanged";

  return (
    <section
      aria-labelledby="result-heading"
      className="rounded-lg border border-rule bg-raised p-5"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="result-heading" className="font-serif text-xl">
          Simplify
        </h2>
        <StatusPill status={result.status} />
      </div>

      {result.input_latex && (
        <div className="mt-4">
          <div className="text-xs uppercase tracking-wide text-muted">Input as read</div>
          <Math latex={result.input_latex} display />
        </div>
      )}

      {shown && result.result_latex && (
        <div className="mt-2">
          <div className="text-xs uppercase tracking-wide text-muted">
            {result.status === "ok" ? "Result" : "Result (unchanged)"}
          </div>
          <Math latex={result.result_latex} display className="text-2xl" label="Result" />
          <button
            type="button"
            className="mt-1 text-xs text-accent hover:underline"
            onClick={async () => {
              await navigator.clipboard?.writeText(result.result_latex ?? "");
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            }}
          >
            {copied ? "Copied" : "Copy LaTeX"}
          </button>
        </div>
      )}

      <p className="mt-3 text-sm text-muted">{result.message}</p>

      {v && (
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          <dt className="text-muted">Numeric check</dt>
          <dd>
            {v.numeric_points_agreed} of {v.numeric_points_checked} sample points agree
          </dd>
          <dt className="text-muted">Symbolic check</dt>
          <dd>{v.symbolic_zero ? "Difference simplifies to 0" : "No symbolic proof found"}</dd>
          <dt className="text-muted">Computed in</dt>
          <dd>{result.duration_ms} ms</dd>
        </dl>
      )}
    </section>
  );
}
