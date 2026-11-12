"use client";

import { Math } from "@minerva/ui/math";
import { useState } from "react";
import type { IntegrateResult } from "@/lib/types";
import { StatusPill } from "./result-panel";

export function IntegralPanel({ result }: { result: IntegrateResult }) {
  const [copied, setCopied] = useState(false);
  const v = result.verification;
  const shown = result.status === "ok" && result.result_latex;

  return (
    <section
      aria-labelledby="integral-heading"
      className="rounded-lg border border-rule bg-raised p-5"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="integral-heading" className="font-serif text-xl">
          Integrate
        </h2>
        <StatusPill status={result.status} okLabel="Integrated and verified" />
      </div>

      {result.input_latex && (
        <div className="mt-4">
          <div className="text-xs uppercase tracking-wide text-muted">Input as read</div>
          <Math latex={result.input_latex} display />
        </div>
      )}

      {result.notation_latex && (
        <div className="mt-2">
          <div className="text-xs uppercase tracking-wide text-muted">Integral taken</div>
          <Math latex={result.notation_latex} display />
        </div>
      )}

      {shown && result.result_latex && (
        <div className="mt-2">
          <div className="text-xs uppercase tracking-wide text-muted">
            {result.definite ? "Value" : "Antiderivative"}
          </div>
          <Math latex={result.result_latex} display className="text-2xl" label="Result" />
          {result.numeric && <p className="text-sm text-muted">≈ {result.numeric}</p>}
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

      {result.conditions_latex.length > 0 && (
        <div className="mt-2">
          <div className="text-xs uppercase tracking-wide text-muted">Needs</div>
          <ul className="space-y-1">
            {result.conditions_latex.map((c) => (
              <li key={c}>
                <Math latex={c} />
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="mt-3 text-sm text-muted">{result.message}</p>
      {result.notes.map((note) => (
        <p key={note} className="mt-1 text-sm text-warn">
          {note}
        </p>
      ))}

      {result.quadrature && (
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          <dt className="text-muted">Quadrature check</dt>
          <dd>
            {result.quadrature.points_agreed} of {result.quadrature.points_checked} sample
            {result.quadrature.points_checked === 1 ? "" : "s"} agree with independent numerical
            integration
          </dd>
          <dt className="text-muted">Computed in</dt>
          <dd>{result.duration_ms} ms</dd>
        </dl>
      )}

      {v && (
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          <dt className="text-muted">Numeric check</dt>
          <dd>
            {`${v.points_agreed} of ${v.points_checked} sample points agree`}
            {v.branches_checked > 1 && `, over ${v.branches_checked} cases`}
          </dd>
          <dt className="text-muted">Directions</dt>
          <dd>
            {v.directions.join(" and ")}
            {v.directions.includes("imaginary") && " (complex variable: both must agree)"}
          </dd>
          <dt className="text-muted">Symbolic check</dt>
          <dd>
            {v.symbolic_zero === true
              ? "The symbolic derivative minus the integrand simplifies to 0"
              : "Not established (the numeric check decides)"}
          </dd>
          <dt className="text-muted">Computed in</dt>
          <dd>{result.duration_ms} ms</dd>
        </dl>
      )}
    </section>
  );
}
