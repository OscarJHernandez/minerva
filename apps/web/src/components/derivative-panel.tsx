"use client";

import { Math } from "@minerva/ui/math";
import { useState } from "react";
import type { DifferentiateResult } from "@/lib/types";
import { StatusPill } from "./result-panel";

export function DerivativePanel({ result }: { result: DifferentiateResult }) {
  const [copied, setCopied] = useState(false);
  const v = result.verification;
  const shown = result.status === "ok" && result.result_latex;
  const intermediate = result.steps.slice(0, -1);

  return (
    <section
      aria-labelledby="derivative-heading"
      className="rounded-lg border border-rule bg-raised p-5"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="derivative-heading" className="font-serif text-xl">
          Differentiate
        </h2>
        <StatusPill status={result.status} okLabel="Differentiated, numerically checked" />
      </div>

      {result.input_latex && (
        <div className="mt-4">
          <div className="text-xs uppercase tracking-wide text-muted">Input as read</div>
          <Math latex={result.input_latex} display />
        </div>
      )}

      {result.notation_latex && result.status === "ok" && (
        <div className="mt-2">
          <div className="text-xs uppercase tracking-wide text-muted">Derivative taken</div>
          <Math latex={result.notation_latex} display />
        </div>
      )}

      {shown && result.result_latex && (
        <div className="mt-2">
          <div className="text-xs uppercase tracking-wide text-muted">Result</div>
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

      {result.status === "ok" && intermediate.length > 0 && (
        <details className="mt-3 text-sm">
          <summary className="cursor-pointer text-muted">
            Intermediate derivatives ({intermediate.length})
          </summary>
          <ol className="mt-2 space-y-1">
            {intermediate.map((s) => (
              <li key={s.order_so_far} className="flex items-baseline gap-3">
                <span className="w-24 shrink-0 text-xs text-muted">
                  after d{s.variable}, step {s.order_so_far}
                </span>
                <Math latex={s.latex} />
              </li>
            ))}
          </ol>
        </details>
      )}

      <p className="mt-3 text-sm text-muted">{result.message}</p>
      {result.notes.map((note) => (
        <p key={note} className="mt-1 text-sm text-warn">
          {note}
        </p>
      ))}

      {v && (
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          <dt className="text-muted">Numeric check</dt>
          <dd>
            {v.steps_checked === 0
              ? v.note
              : `${v.points_agreed} of ${v.points_checked} sample points agree, over ${v.steps_checked} step${v.steps_checked === 1 ? "" : "s"}`}
          </dd>
          {v.steps_checked > 0 && (
            <>
              <dt className="text-muted">Directions</dt>
              <dd>
                {v.directions.join(" and ")}
                {v.directions.includes("imaginary") && " (complex variable: both must agree)"}
              </dd>
            </>
          )}
          <dt className="text-muted">Symbolic check</dt>
          <dd>None: no independent symbolic test of a derivative exists</dd>
          <dt className="text-muted">Computed in</dt>
          <dd>{result.duration_ms} ms</dd>
        </dl>
      )}
    </section>
  );
}
