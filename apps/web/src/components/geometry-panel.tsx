"use client";

import { Math } from "@minerva/ui/math";
import { useId, useState } from "react";
import { QUANTITY_OPTIONS } from "@/lib/geometry-form";
import type {
  GeometryQuantity,
  GeometryQuantityResult,
  GeometryResult,
  GeometryVerification,
} from "@/lib/types";
import { StatusPill } from "./result-panel";

/** What the evidence amounts to, in the words the rest of Minerva uses: "proved" only with a symbolic zero. */
function evidenceLabel(v: GeometryVerification): string {
  return v.symbolic_zero === true
    ? "Proved by a symbolic zero test, and numerically checked"
    : "Numerically checked; no symbolic proof";
}

function checkMark(passed: boolean | null): string {
  if (passed === true) return "passed";
  if (passed === false) return "failed";
  return "not run";
}

function Quantity({ q }: { q: GeometryQuantityResult }) {
  const shown = q.status === "ok";
  const v = q.verification;
  return (
    <section
      aria-label={q.title}
      className="rounded border border-rule bg-sunk p-3"
      data-testid={`quantity-${q.kind}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-medium">{q.title}</h3>
        <StatusPill status={q.status} okLabel={v ? evidenceLabel(v) : "Verified"} />
      </div>

      {shown && q.scalar_latex !== null && (
        <div className="overflow-x-auto">
          <Math latex={q.scalar_latex} display label={q.title} />
        </div>
      )}

      {shown && q.scalar_latex === null && q.all_zero && q.kind !== "geodesics" && (
        <p className="mt-2 text-sm">Every component is zero.</p>
      )}

      {shown && q.components.length > 0 && (
        <ul className="mt-2 space-y-1">
          {q.components.map((c) => (
            <li key={c.indices.join(",")} className="min-w-0 overflow-x-auto">
              <Math latex={q.kind === "geodesics" ? c.latex : `${c.label_latex} = ${c.latex}`} />
            </li>
          ))}
        </ul>
      )}

      {shown && q.symmetry_note && q.kind !== "geodesics" && (
        <p className="mt-2 text-xs text-muted">{q.symmetry_note}</p>
      )}
      {shown &&
        q.notes.map((n) => (
          <p key={n} className="mt-1 text-xs text-muted">
            {n}
          </p>
        ))}

      {!shown && q.message && (
        <p role="note" className="mt-2 rounded bg-warn-soft p-2 text-sm text-warn">
          {q.message}
        </p>
      )}

      {v && (
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-muted">
            Checks ({v.checks.length}): {v.checks.filter((c) => c.passed === true).length} passed
          </summary>
          <ul className="mt-2 space-y-1 text-xs">
            {v.checks.map((c) => (
              <li key={`${c.kind}-${c.name}`} className="min-w-0">
                <span className="font-medium">{c.kind === "numeric" ? "Numeric" : "Symbolic"}</span>{" "}
                {c.name}: {checkMark(c.passed)}
                {c.kind === "numeric" && c.points_checked > 0
                  ? ` (${c.points_agreed} of ${c.points_checked} sample points)`
                  : ""}
                {c.passed !== true && c.note ? (
                  <span className="text-muted"> — {c.note}</span>
                ) : null}
              </li>
            ))}
          </ul>
          <p className="mt-1 text-xs text-muted">Numeric method: {v.method}.</p>
        </details>
      )}
    </section>
  );
}

/**
 * Differential geometry: the metric, the conventions used, where the coordinates break down, every
 * requested quantity with the checks it passed, and per-stage timing. A quantity is drawn only when the
 * engine verified it; numerical agreement is labelled as such, never as a proof.
 */
export function GeometryPanel({
  result,
  asked = [],
  onMore,
  busy = false,
}: {
  result: GeometryResult;
  /** Quantities of the same geometry that can be asked for next (chat: posts the card's own request). */
  asked?: GeometryQuantity[];
  onMore?: (quantity: GeometryQuantity) => void;
  busy?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const headingId = useId();
  const have = new Set(result.quantities.map((q) => q.kind));
  const more = QUANTITY_OPTIONS.filter(
    (o) => !o.tensor && !have.has(o.id) && !asked.includes(o.id),
  );
  return (
    <section aria-labelledby={headingId} className="rounded-lg border border-rule bg-raised p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id={headingId} className="font-serif text-xl">
          Differential geometry
        </h2>
        <StatusPill status={result.status} okLabel="Computed and verified" />
      </div>

      {result.metric_latex && (
        <div className="mt-4">
          <div className="text-xs uppercase tracking-wide text-muted">
            Metric g in ({result.coordinates_latex.map((c, i) => (i ? `, ${c}` : c)).join("")})
          </div>
          <div className="overflow-x-auto">
            <Math latex={`g = ${result.metric_latex}`} display label="Metric" />
          </div>
          <button
            type="button"
            className="text-xs text-accent hover:underline"
            onClick={async () => {
              await navigator.clipboard?.writeText(result.metric_latex ?? "");
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            }}
          >
            {copied ? "Copied" : "Copy metric LaTeX"}
          </button>
        </div>
      )}

      {result.dimension > 0 && (
        <p className="mt-2 text-sm" data-testid="signature">
          <span className="text-xs uppercase tracking-wide text-muted">Signature </span>
          {result.signature ?? "not determined"}
          {result.signature === null && result.signature_note ? (
            <span className="text-muted"> — {result.signature_note}</span>
          ) : null}
        </p>
      )}

      {result.restrictions_latex.length > 0 && (
        <div className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn" data-testid="restrictions">
          <div>The coordinate expressions below are valid only where</div>
          <ul className="mt-1">
            {result.restrictions_latex.map((r) => (
              <li key={r} className="overflow-x-auto text-ink">
                <Math latex={r} />
              </li>
            ))}
          </ul>
          <p className="mt-1 text-xs">
            This says where the coordinates break down, not whether the geometry is singular
            {result.restrictions_complete
              ? "."
              : "; root, logarithm and branch domains were not analysed, so there may be more."}
          </p>
        </div>
      )}
      {result.restrictions_latex.length === 0 && !result.restrictions_complete && (
        <p role="note" className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn">
          Where the coordinate expressions are valid could not be fully determined.
        </p>
      )}

      {result.message && <p className="mt-3 text-sm text-muted">{result.message}</p>}

      <div className="mt-4 space-y-3">
        {result.quantities.map((q) => (
          <Quantity key={q.kind} q={q} />
        ))}
      </div>

      {onMore && more.length > 0 && (
        <div className="mt-4">
          <div className="text-xs uppercase tracking-wide text-muted">
            Also compute, for the same metric
          </div>
          <div className="mt-1 flex flex-wrap gap-2">
            {more.map((o) => (
              <button
                key={o.id}
                type="button"
                disabled={busy}
                onClick={() => onMore(o.id)}
                className="rounded border border-rule px-2 py-1 text-xs hover:bg-sunk disabled:opacity-50"
              >
                {o.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {result.assumptions_used.length > 0 && (
        <details className="mt-4">
          <summary className="cursor-pointer text-sm font-medium">Assumptions</summary>
          <ul className="mt-1 list-disc pl-5 text-sm">
            {result.assumptions_used.map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
        </details>
      )}

      {result.conventions.length > 0 && (
        <details className="mt-2">
          <summary className="cursor-pointer text-sm font-medium">Conventions</summary>
          <ul className="mt-1 space-y-1 text-sm">
            {result.conventions.map((c) => (
              <li key={c} className="min-w-0 break-words">
                {c}
              </li>
            ))}
          </ul>
        </details>
      )}

      {result.diagnostics.length > 0 && (
        <details className="mt-2">
          <summary className="cursor-pointer text-sm font-medium">
            Stages <span className="font-normal text-muted">({result.duration_ms} ms in all)</span>
          </summary>
          <div className="mt-1 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-muted">
                <tr>
                  <th className="pr-3 font-normal">Stage</th>
                  <th className="pr-3 font-normal">Time</th>
                  <th className="pr-3 font-normal">Size</th>
                  <th className="font-normal">Outcome</th>
                </tr>
              </thead>
              <tbody>
                {result.diagnostics.map((d, i) => (
                  <tr key={`${d.stage}-${i}`}>
                    <td className="pr-3 font-mono">{d.stage}</td>
                    <td className="pr-3">{d.duration_ms} ms</td>
                    <td className="pr-3">{d.ops === null ? "" : `${d.ops} ops`}</td>
                    <td>{d.status === "ok" ? "ok" : (d.failure ?? d.status)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {result.peak_memory_mb !== null && (
            <p className="mt-1 text-xs text-muted">Peak memory {result.peak_memory_mb} MB.</p>
          )}
        </details>
      )}
    </section>
  );
}
