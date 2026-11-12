"use client";

import { Math } from "@minerva/ui/math";
import { useId, useState } from "react";
import type { SeriesResult } from "@/lib/types";
import { StatusPill } from "./result-panel";

/** A Taylor polynomial, shown only if both independent checks passed. */
export function SeriesPanel({ result }: { result: SeriesResult }) {
  const headingId = useId();
  const [copied, setCopied] = useState(false);
  const v = result.verification;
  const shown = result.status === "ok" && result.result_latex !== null;
  const full =
    shown && result.result_latex
      ? result.remainder_latex
        ? `${result.result_latex} + ${result.remainder_latex}`
        : result.result_latex
      : null;

  return (
    <section aria-labelledby={headingId} className="rounded-lg border border-rule bg-raised p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id={headingId} className="font-serif text-xl">
          Series
        </h2>
        <StatusPill status={result.status} okLabel="Expanded and verified" />
      </div>

      {result.input_latex && (
        <div className="mt-4">
          <div className="text-xs uppercase tracking-wide text-muted">
            Input as read
            {result.variable && result.point_latex !== null && (
              <>
                {" "}
                · in {result.variable} about {result.point_latex}, order {result.order}
              </>
            )}
          </div>
          <Math latex={result.input_latex} display />
        </div>
      )}

      {full && (
        <div className="mt-2">
          <div className="text-xs uppercase tracking-wide text-muted">Result</div>
          <Math latex={full} display className="text-2xl" label="Result" />
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
      {result.notes.map((note) => (
        <p key={note} className="mt-1 text-sm text-warn">
          {note}
        </p>
      ))}

      {v && (
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          <dt className="text-muted">Coefficients</dt>
          <dd>
            {v.coefficients_passed
              ? "agree with a Cauchy-integral estimate (no differentiation)"
              : "do not agree with a Cauchy-integral estimate"}
          </dd>
          <dt className="text-muted">Remainder</dt>
          <dd>
            {v.remainder_passed
              ? `shrinks at order ${result.order + 1}, as claimed`
              : "was not confirmed at the claimed order"}
          </dd>
          <dt className="text-muted">Checked at</dt>
          <dd>
            {v.parameter_sets_checked} parameter set{v.parameter_sets_checked === 1 ? "" : "s"},{" "}
            {v.directions.join(" and ")} direction
            {v.directions.length === 1 ? "" : "s"}
            {v.radius !== null && `, circle radius ${v.radius}`}
          </dd>
          <dt className="text-muted">Computed in</dt>
          <dd>{result.duration_ms} ms</dd>
        </dl>
      )}

      {shown && result.measured_errors.length > 0 && (
        <div className="mt-4 overflow-x-auto">
          <h3 className="text-sm font-medium">Measured error of the truncation</h3>
          <table className="mt-2 w-full text-left text-sm">
            <thead className="text-xs text-muted">
              <tr>
                <th className="pb-1 pr-3 font-normal">Distance from the point</th>
                <th className="pb-1 pr-3 font-normal">Absolute</th>
                <th className="pb-1 font-normal">Relative</th>
              </tr>
            </thead>
            <tbody>
              {result.measured_errors.map((m) => (
                <tr key={m.distance} className="border-t border-rule">
                  <td className="py-1 pr-3">{m.distance}</td>
                  <td className="py-1 pr-3 font-mono text-xs">{m.absolute_error}</td>
                  <td className="py-1 font-mono text-xs">{m.relative_error ?? "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1 text-xs text-muted">
            These are measurements of this expression at these distances, not a bound.
          </p>
        </div>
      )}
    </section>
  );
}
