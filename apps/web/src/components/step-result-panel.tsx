"use client";

import { Math } from "@minerva/ui/math";
import Link from "next/link";
import type { Diagnostics, IsolateResult, StepRecord, StepResult } from "@/lib/types";
import { Evidence, Restrictions, StatusPill } from "./result-panel";

const REASON: Record<string, string> = {
  order_dependent: "The result depends on the order of the relations",
  cyclic: "The relations rewrite each other",
  stale_selection: "The chosen part is no longer in the expression",
  ambiguous_selection: "The part occurs more than once",
  premise_false: "The relation can never hold",
  pattern_not_found: "The left side of the relation does not occur",
  no_match: "The move does not apply to this part",
  conditions_unmet: "The identity's conditions are not provable from the assumptions",
  form: "The unknown is in a form that is not supported",
  extraneous: "A candidate did not satisfy the equation",
};

/** Premises first: what the result depends on and that Minerva did not check it. */
export function Premises({ given }: { given: { latex: string; text: string }[] }) {
  if (given.length === 0) return null;
  return (
    <div
      className="mt-3 rounded border border-rule bg-sunk px-3 py-2 text-sm"
      data-testid="premises"
    >
      <div className="text-xs uppercase tracking-wide text-muted">
        Given: your relation{given.length > 1 ? "s" : ""} (stated by you, not checked by Minerva)
      </div>
      <ul className="mt-1 space-y-0.5">
        {given.map((p) => (
          <li key={p.text}>
            <Math latex={p.latex} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function Conditions({ title, latex }: { title: string; latex: string[] }) {
  if (latex.length === 0) return null;
  return (
    <div className="mt-2 text-sm" data-testid="conditions">
      <span className="text-muted">{title} </span>
      {latex.map((c, i) => (
        <span key={c}>
          {i > 0 && ", "}
          <Math latex={c} />
        </span>
      ))}
    </div>
  );
}

function justification(step: StepRecord, entryTitle: string | undefined): React.ReactNode {
  switch (step.action) {
    case "substitute":
      return (
        <>
          Replaced <Math latex={step.matched_latex ?? ""} /> using your relation
        </>
      );
    case "rule":
      return (
        <>
          Applied{" "}
          <Link href={`/reference/${step.name}`} className="hover:text-accent hover:underline">
            {entryTitle ?? step.name}
          </Link>
          {step.direction === "reverse" ? " (read right to left)" : ""}
          {step.matched_latex ? (
            <>
              {" "}
              to <Math latex={step.matched_latex} />
            </>
          ) : null}
        </>
      );
    case "collect":
    case "builtin":
      return (
        <>
          {step.name.replace("sympy.", "SymPy: ")}{" "}
          <span className="text-xs text-muted">(not a handbook identity)</span>
          {step.matched_latex ? (
            <>
              {" "}
              on <Math latex={step.matched_latex} />
            </>
          ) : null}
        </>
      );
    case "peel":
      return <>Undo {step.name}</>;
    default:
      return <>{step.name}</>;
  }
}

function StepList({ steps, entryTitle }: { steps: StepRecord[]; entryTitle?: string }) {
  if (steps.length === 0) return null;
  return (
    <ol className="mt-3 space-y-2" aria-label="Steps">
      {steps.map((step, i) => (
        <li key={`${step.name}-${i}`} className="rounded border border-rule px-3 py-2">
          <div className="text-xs text-muted">{justification(step, entryTitle)}</div>
          <div className="mt-1 overflow-x-auto">
            <Math latex={step.before_latex} />
            {step.display_latex && (
              <>
                {" "}
                <span aria-hidden="true">→</span> <Math latex={step.display_latex} />
              </>
            )}{" "}
            <span aria-hidden="true">→</span> <Math latex={step.after_latex} />
          </div>
          {step.conditions_latex.length > 0 && (
            <Conditions title="needs" latex={step.conditions_latex} />
          )}
          {step.evidence === "conditional" && (
            <p className="mt-1 text-xs text-muted">Valid given your relation.</p>
          )}
        </li>
      ))}
    </ol>
  );
}

export function DiagnosticsView({ d }: { d: Diagnostics }) {
  const rejected = Object.entries(d.rejected);
  return (
    <details className="mt-3 text-xs text-muted">
      <summary className="cursor-pointer">Diagnostics</summary>
      <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
        <dt>Time</dt>
        <dd>
          read {d.parse_ms} ms · candidates {d.candidates_ms} ms · transform {d.transform_ms} ms ·
          verify {d.verify_ms} ms
        </dd>
        <dt>Candidates</dt>
        <dd>{d.candidates}</dd>
        {rejected.length > 0 && (
          <>
            <dt>Rejected</dt>
            <dd>
              {rejected.map(([reason, n]) => `${REASON[reason] ?? reason} (${n})`).join("; ")}
            </dd>
          </>
        )}
        <dt>Check</dt>
        <dd>
          {d.verification_method.replace("_", " ")}: {d.verification_outcome.replace("_", " ")}
        </dd>
        <dt>Restrictions</dt>
        <dd>{d.restrictions_outcome.replace("_", " ")}</dd>
      </dl>
      {d.detail.length > 0 && (
        <pre
          className="mt-2 whitespace-pre-wrap rounded bg-sunk p-2 font-mono"
          data-testid="debug-detail"
        >
          {d.detail.join("\n")}
        </pre>
      )}
    </details>
  );
}

function Outcome({
  status,
  evidence,
}: {
  status: StepResult["status"];
  evidence: StepResult["evidence"];
}) {
  return (
    <StatusPill
      status={status}
      okLabel={evidence === "conditional" ? "Holds given your relation" : "Verified"}
    />
  );
}

/** The result of substituting relations, or of rewriting one part of an expression. */
export function StepResultPanel({ result }: { result: StepResult }) {
  const ok = result.status === "ok" || result.status === "unchanged";
  const why = result.reason ? REASON[result.reason] : null;
  return (
    <div className="rounded-lg border border-rule bg-raised p-4 text-sm" data-testid="step-result">
      <div className="flex flex-wrap items-center gap-2">
        <Outcome status={result.status} evidence={result.evidence} />
        {result.order_dependent === true && result.status === "ok" && (
          <span className="text-xs text-warn">another order would give another result</span>
        )}
      </div>
      {!ok && why && <p className="mt-2 text-warn">{why}.</p>}
      <p className="mt-2 text-muted">{result.message}</p>
      {result.status === "ok" && result.evidence === "conditional" && (
        <Premises given={result.given} />
      )}
      {ok && <StepList steps={result.steps} entryTitle={result.entry?.title} />}
      {result.status === "ok" && result.result_latex && (
        <Math latex={result.result_latex} display className="mt-3 text-xl" label="Result" />
      )}
      {result.unmet_conditions.length > 0 && (
        <p className="mt-2 text-warn">
          Needs:{" "}
          {result.unmet_conditions.map((c) => `${c.variable} must be ${c.assumption}`).join(", ")}{" "}
          (pattern variables of the identity). Declare the assumptions to apply it.
        </p>
      )}
      <Conditions title="Valid when" latex={result.conditions_latex} />
      {ok && (
        <Restrictions latex={result.restrictions_latex} complete={result.restrictions_complete} />
      )}
      {result.verification && (
        <p className="mt-2">
          <Evidence v={result.verification} />
        </p>
      )}
      <DiagnosticsView d={result.diagnostics} />
    </div>
  );
}

/** The solutions of an equation, each with the conditions under which it holds. */
export function IsolatePanel({
  result,
  onContinue,
}: {
  result: IsolateResult;
  /** Continue the derivation with the value of solution `index` (the Workbench only). */
  onContinue?: (index: number) => void;
}) {
  const ok = result.status === "ok";
  const why = result.reason ? REASON[result.reason] : null;
  return (
    <div
      className="rounded-lg border border-rule bg-raised p-4 text-sm"
      data-testid="isolate-result"
    >
      <div className="flex flex-wrap items-center gap-2">
        <StatusPill
          status={result.status}
          okLabel={result.complete ? "Verified, none dropped" : "Verified"}
        />
        {result.method && (
          <span className="text-xs text-muted">
            {result.method === "inversion" ? "undoing each operation" : "polynomial"}
          </span>
        )}
      </div>
      {!ok && why && <p className="mt-2 text-warn">{why}.</p>}
      <p className="mt-2 text-muted">{result.message}</p>
      {ok && (
        <>
          <StepList steps={result.steps} />
          {result.solutions.length === 0 ? (
            <p className="mt-3 font-medium" data-testid="no-solution">
              No solution.
            </p>
          ) : (
            <ul className="mt-3 space-y-2" aria-label="Solutions">
              {result.solutions.map((s, i) => (
                <li key={s.text} className="rounded border border-rule px-3 py-2">
                  <div className="overflow-x-auto">
                    <Math latex={s.latex} display className="text-lg" label={`Solution ${i + 1}`} />
                  </div>
                  {s.branch && <p className="text-xs text-muted">{s.branch}</p>}
                  <Conditions title="needs" latex={s.conditions_latex} />
                  {onContinue && (
                    <button
                      type="button"
                      onClick={() => onContinue(i)}
                      className="mt-1 rounded border border-accent px-2.5 py-1 text-xs font-medium text-accent hover:bg-accent-soft"
                    >
                      Continue with this solution
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}
          <Conditions title="All solutions need" latex={result.conditions_latex} />
          <Conditions title="The equation itself excludes" latex={result.excluded_latex} />
          {result.exceptional.length > 0 && (
            <div className="mt-2 text-warn" data-testid="exceptional">
              Not covered if{" "}
              {result.exceptional.map((e, i) => (
                <span key={e}>
                  {i > 0 && ", "}
                  <Math latex={e} />
                </span>
              ))}
              : the equation then degenerates and must be examined separately.
            </div>
          )}
          {result.assumption_note && (
            <p className="mt-2 text-xs text-muted">{result.assumption_note}</p>
          )}
          {result.completeness_note && (
            <p className="mt-1 text-xs text-muted">Nothing dropped: {result.completeness_note}.</p>
          )}
        </>
      )}
      {result.verification && result.solutions.length > 0 && (
        <p className="mt-2">
          <Evidence v={result.verification} />
        </p>
      )}
      <DiagnosticsView d={result.diagnostics} />
    </div>
  );
}
