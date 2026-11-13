"use client";

import { Math } from "@minerva/ui/math";
import { useState } from "react";
import type { EntrySummary, MatrixCalculusResult, MatrixResultKind } from "@/lib/types";
import { EntryCard } from "./entry-card";
import { StatusPill } from "./result-panel";

const OK_LABEL: Record<MatrixResultKind, string> = {
  explicit: "Evaluated and verified",
  expression: "Transformed and verified",
  equation: "Reduced, not evaluated",
  representation: "Represented exactly, not evaluated",
};

const RESULT_HEADING: Record<MatrixResultKind, string> = {
  explicit: "Result",
  expression: "Result",
  equation: "Equation",
  representation: "Exact representation",
};

/**
 * Matrix calculus: what was recognised, which method was chosen, what it assumed, every step and
 * how the result was checked. A result is drawn only when the engine verified it, and an equation
 * or a representation is labelled as such, never as an evaluated integral.
 */
export function MatrixPanel({
  result,
  methods = [],
}: {
  result: MatrixCalculusResult;
  methods?: EntrySummary[];
}) {
  const [copied, setCopied] = useState(false);
  const v = result.verification;
  const kind = result.result_kind;
  const shown = result.status === "ok" && kind !== null;

  return (
    <section
      aria-labelledby="matrix-heading"
      className="rounded-lg border border-rule bg-raised p-5"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="matrix-heading" className="font-serif text-xl">
          Matrix calculus
        </h2>
        <StatusPill status={result.status} okLabel={kind ? OK_LABEL[kind] : "Verified"} />
      </div>

      {result.input_latex && (
        <div className="mt-4">
          <div className="text-xs uppercase tracking-wide text-muted">Input as read</div>
          <div className="overflow-x-auto">
            <Math latex={result.input_latex} display />
          </div>
        </div>
      )}

      {result.structure && (
        <div className="mt-2">
          <div className="text-xs uppercase tracking-wide text-muted">Recognised structure</div>
          <p className="text-sm">{result.structure}</p>
        </div>
      )}

      {shown && result.method && (
        <p className="mt-2 text-sm">
          <span className="text-xs uppercase tracking-wide text-muted">Method </span>
          {result.method_title ?? result.method} <code className="text-xs">{result.method}</code>
        </p>
      )}

      {shown && kind && (kind === "equation" || kind === "representation") && (
        <p
          role="note"
          className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn"
          data-testid="not-evaluated"
        >
          The integral was not evaluated. This is{" "}
          {kind === "equation"
            ? "an equation that it satisfies"
            : "an exact representation through a block-matrix exponential"}
          , verified, but not entries.
        </p>
      )}

      {shown && kind && result.result_latex && (
        <div className="mt-2">
          <div className="text-xs uppercase tracking-wide text-muted">{RESULT_HEADING[kind]}</div>
          <div className="overflow-x-auto">
            <Math latex={result.result_latex} display label="Result" />
          </div>
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

      {shown && result.equation_latex && (
        <div className="mt-2">
          <div className="text-xs uppercase tracking-wide text-muted">
            Verified equation (X is the integral)
          </div>
          <div className="overflow-x-auto">
            <Math latex={result.equation_latex} display label="Equation" />
          </div>
        </div>
      )}

      {shown && result.steps.length > 0 && (
        <div className="mt-4">
          <h3 className="text-sm font-medium">
            Steps <span className="font-normal text-muted">({result.steps.length})</span>
          </h3>
          <ol className="mt-2 space-y-2">
            {result.steps.map((step, i) => (
              <li
                key={`${i}-${step.method}`}
                className="min-w-0 rounded border border-rule bg-sunk p-3 text-sm"
              >
                <div>
                  {i + 1}. {step.title} <code className="text-xs">{step.method}</code>
                </div>
                {step.note && <div className="text-xs text-muted">{step.note}</div>}
                {step.before_latex && (
                  <div className="overflow-x-auto">
                    <Math latex={step.before_latex} display />
                  </div>
                )}
                {step.after_latex && (
                  <div className="overflow-x-auto">
                    <Math latex={`\\Rightarrow\\; ${step.after_latex}`} display />
                  </div>
                )}
              </li>
            ))}
          </ol>
        </div>
      )}

      {shown && result.assumptions_used.length > 0 && (
        <div className="mt-4">
          <h3 className="text-sm font-medium">Assumptions used</h3>
          <ul className="list-disc pl-5 text-sm">
            {result.assumptions_used.map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
        </div>
      )}

      {result.blocked.length > 0 && (
        <div className="mt-4 rounded bg-warn-soft p-3 text-sm text-warn">
          <div>Not applied: a condition is not established.</div>
          <ul className="mt-2 space-y-2">
            {result.blocked.map((b) => (
              <li key={`${b.method}-${b.needs}-${b.latex}`} className="min-w-0">
                <span className="font-medium">{b.title}</span> needs {b.needs}
                {b.latex && (
                  <div className="overflow-x-auto text-ink">
                    <Math latex={b.latex} />
                  </div>
                )}
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

      {v && (
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          <dt className="text-muted">Numeric check</dt>
          <dd>
            {v.points_agreed} of {v.points_checked} sample points agree ({v.method})
          </dd>
          <dt className="text-muted">Exact check</dt>
          <dd>
            {v.symbolic_zero === true
              ? `Passed: ${v.symbolic_method}`
              : v.symbolic_zero === false
                ? `Failed: ${v.symbolic_method}`
                : "Not run (the numeric check decides)"}
          </dd>
          <dt className="text-muted">Computed in</dt>
          <dd>{result.duration_ms} ms</dd>
        </dl>
      )}

      {methods.length > 0 && (
        <details className="mt-4">
          <summary className="cursor-pointer text-sm font-medium">
            Methods from the handbook{" "}
            <span className="font-normal text-muted">({methods.length}): show</span>
          </summary>
          <div className="mt-2 space-y-3">
            {methods.map((m) => (
              <EntryCard key={m.id} entry={m} />
            ))}
          </div>
        </details>
      )}
    </section>
  );
}
