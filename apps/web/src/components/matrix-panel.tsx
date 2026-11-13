"use client";

import { Math } from "@minerva/ui/math";
import { useState } from "react";
import type {
  EntrySummary,
  MatrixCalculusResult,
  MatrixDiagnostics,
  MatrixResultKind,
} from "@/lib/types";
import { EntryCard } from "./entry-card";
import { StatusPill } from "./result-panel";

/**
 * The status label never says more than the checks did: the numerical legs always ran, and the
 * exact leg is named only when it passed.
 */
function okLabel(kind: MatrixResultKind | null, exact: boolean | null | undefined): string {
  const how = exact ? "checked numerically and exactly" : "checked numerically";
  switch (kind) {
    case "explicit":
      return `Evaluated, ${how}`;
    case "expression":
      return `Transformed, ${how}`;
    case "equation":
      return "Reduced, not evaluated";
    case "representation":
      return "Represented exactly, not evaluated";
    case "numeric":
      return "Numerical approximation, checked independently";
    default:
      return "Checked numerically";
  }
}

const RESULT_HEADING: Record<MatrixResultKind, string> = {
  explicit: "Result",
  expression: "Result",
  equation: "Equation",
  representation: "Exact representation",
  numeric: "Numerical value (approximation)",
};

/** Why a verified result is still not an evaluated closed form. */
function notEvaluatedText(kind: MatrixResultKind, action: MatrixCalculusResult["action"]): string {
  const what = action === "integrate" ? "The integral" : "The result";
  if (kind === "equation") {
    return `${what} was not evaluated. This is an equation that it satisfies, checked, but not entries.`;
  }
  if (kind === "numeric") {
    return `${what} was not evaluated in closed form. This is a numerical approximation on the stated span: a measurement checked by independent methods, not a closed form and not a proof for all times.`;
  }
  return `${what} was not evaluated. This is an exact representation (a block-matrix exponential or an integral), checked, but not entries.`;
}

/**
 * Matrix calculus: what was recognised, which method was chosen, what it assumed, every step and
 * how the result was checked. A result is drawn only when the engine verified it, and an equation
 * or a representation is labelled as such, never as an evaluated integral.
 */
export function MatrixPanel({
  result,
  methods = [],
  headingId = "matrix-heading",
}: {
  result: MatrixCalculusResult;
  methods?: EntrySummary[];
  /** Unique when several panels are on one page (the workbench shows every run). */
  headingId?: string;
}) {
  const [copied, setCopied] = useState(false);
  const v = result.verification;
  const kind = result.result_kind;
  const shown = result.status === "ok" && kind !== null;

  return (
    <section aria-labelledby={headingId} className="rounded-lg border border-rule bg-raised p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id={headingId} className="font-serif text-xl">
          Matrix calculus
        </h2>
        <StatusPill status={result.status} okLabel={okLabel(kind, v?.symbolic_zero)} />
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

      {shown && kind && !result.evaluated && (
        <p
          role="note"
          className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn"
          data-testid="not-evaluated"
        >
          {notEvaluatedText(kind, result.action)}
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

      {result.diagnostics && <Diagnostics d={result.diagnostics} />}

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

const SOURCE_TEXT: Record<MatrixDiagnostics["commutation"], string> = {
  not_used: "not used",
  declared: "declared",
  from_entries: "visible in the entries",
  mixed: "declared and from the entries",
};

function stageMs(d: MatrixDiagnostics, name: string): number | null {
  return d.stages.find((s) => s.name === name)?.ms ?? null;
}

/** How the result was reached: methods tried, checks, precision, residual and timings. No mathematics. */
function Diagnostics({ d }: { d: MatrixDiagnostics }) {
  const stages = d.stages.filter((s) => s.name !== "total");
  return (
    <details className="mt-4" data-testid="matrix-diagnostics">
      <summary className="cursor-pointer text-sm font-medium">
        How this was computed <span className="font-normal text-muted">(diagnostics)</span>
      </summary>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-muted">Methods tried</dt>
        <dd>
          {d.methods_tried.length === 0
            ? "none applied"
            : d.methods_tried.map((m) => `${m.id}: ${m.outcome.replace("_", " ")}`).join("; ")}
        </dd>
        <dt className="text-muted">Matrices</dt>
        <dd>
          {d.matrices} declared ({d.explicit} explicit, {d.time_dependent} time dependent)
          {d.dimensions.length > 0 &&
            `, sizes ${d.dimensions.map(([r, c]) => `${r}×${c}`).join(", ")}`}
        </dd>
        <dt className="text-muted">Commutation</dt>
        <dd>{SOURCE_TEXT[d.commutation]}</dd>
        {d.invertibility !== "not_used" && (
          <>
            <dt className="text-muted">Invertibility</dt>
            <dd>{SOURCE_TEXT[d.invertibility]}</dd>
          </>
        )}
        {d.verification && (
          <>
            <dt className="text-muted">Checks</dt>
            <dd>
              <code className="text-xs">{d.verification}</code>, {d.points_agreed} of{" "}
              {d.points_checked} points
            </dd>
          </>
        )}
        {d.digits !== null && (
          <>
            <dt className="text-muted">Precision</dt>
            <dd>
              {d.digits} digits
              {d.tolerance !== null && `, tolerance ${d.tolerance.toExponential(0)}`}
              {d.max_residual !== null && `, largest residual ${d.max_residual.toExponential(1)}`}
            </dd>
          </>
        )}
        {stages.length > 0 && (
          <>
            <dt className="text-muted">Time per stage</dt>
            <dd>
              {stages.map((s) => `${s.name.replace("_", " ")} ${s.ms.toFixed(0)} ms`).join(", ")}
              {stageMs(d, "total") !== null && `; total ${stageMs(d, "total")?.toFixed(0)} ms`}
            </dd>
          </>
        )}
        {d.fallbacks.length > 0 && (
          <>
            <dt className="text-muted">Fallbacks</dt>
            <dd>{d.fallbacks.join(", ")}</dd>
          </>
        )}
        {d.failure_reason && (
          <>
            <dt className="text-muted">Failure</dt>
            <dd>{d.failure_reason}</dd>
          </>
        )}
        {d.trace_id && (
          <>
            <dt className="text-muted">Trace id</dt>
            <dd>
              <code className="break-all text-xs">{d.trace_id}</code>
            </dd>
          </>
        )}
      </dl>
    </details>
  );
}
