"use client";

import { Math } from "@minerva/ui/math";
import Link from "next/link";
import { useState } from "react";
import type {
  BlockedMove,
  EntrySummary,
  SimplifyGoal,
  SimplifyResult,
  Status,
  TransformStep,
  Verification,
} from "@/lib/types";
import { VerificationBadge } from "./verification-badge";

const GOAL: Record<SimplifyGoal, { heading: string; done: string }> = {
  simplify: { heading: "Simplify", done: "Simplified" },
  expand: { heading: "Expand", done: "Expanded" },
  factor: { heading: "Factor", done: "Factored" },
  rewrite: { heading: "Rewrite", done: "Rewritten" },
};

const STATUS: Record<Status, { label: string; tone: string }> = {
  ok: { label: "Simplified", tone: "bg-ok-soft text-ok" },
  unchanged: { label: "No simpler form found", tone: "bg-sunk text-muted" },
  not_verified: { label: "Could not verify a result", tone: "bg-warn-soft text-warn" },
  conditions_not_satisfied: { label: "Conditions not satisfied", tone: "bg-warn-soft text-warn" },
  unsupported: { label: "Not supported", tone: "bg-warn-soft text-warn" },
  parse_error: { label: "Could not read the expression", tone: "bg-bad-soft text-bad" },
  resource_limit: { label: "Resource limit reached", tone: "bg-warn-soft text-warn" },
  internal_error: { label: "Computation failed", tone: "bg-bad-soft text-bad" },
};

export function StatusPill({ status, okLabel }: { status: Status; okLabel?: string }) {
  const s = status === "ok" && okLabel ? { ...STATUS.ok, label: okLabel } : STATUS[status];
  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${s.tone}`}>
      {s.label}
    </span>
  );
}

/** Where the result is valid when it excludes points the input did not (a cancelled singularity). */
export function Restrictions({
  latex,
  complete,
}: {
  latex: string[] | undefined;
  complete: boolean;
}) {
  if (!complete) {
    return (
      <p className="mt-3 text-sm text-warn">
        Where this result is valid could not be determined. Do not assume it holds at poles.
      </p>
    );
  }
  if (!latex || latex.length === 0) return null;
  return (
    <div className="mt-3 text-sm" data-testid="restrictions">
      <span className="text-muted">Valid only for </span>
      {latex.map((item, index) => (
        <span key={item}>
          {index > 0 && ", "}
          <Math latex={item} />
        </span>
      ))}
    </div>
  );
}

function Evidence({ v }: { v: Verification }) {
  return (
    <span
      className="text-xs text-muted"
      title="Agreement at sample points is numerical evidence, not a proof. Only a zero difference found by SymPy is symbolic."
    >
      {v.numeric_points_agreed}/{v.numeric_points_checked} points agree
      {" · "}
      {v.symbolic_zero ? "difference simplifies to 0" : "no symbolic proof found"}
    </span>
  );
}

function StepName({ step, entry }: { step: TransformStep; entry: EntrySummary | undefined }) {
  if (step.kind === "builtin") {
    return (
      <span>
        {step.name.replace("sympy.", "SymPy: ")}{" "}
        <span className="text-xs text-muted">(not a handbook identity)</span>
      </span>
    );
  }
  return (
    <span>
      {entry ? (
        <Link href={`/reference/${step.name}`} className="hover:text-accent hover:underline">
          {entry.title}
        </Link>
      ) : (
        step.name
      )}{" "}
      <code className="text-xs text-muted">{step.name}</code>
      {step.direction === "reverse" && (
        <span className="text-xs text-muted"> (read right to left)</span>
      )}
      {entry && (
        <>
          {" "}
          <VerificationBadge level={entry.verification} />
        </>
      )}
    </span>
  );
}

function Steps({ result }: { result: SimplifyResult }) {
  const steps = result.steps ?? [];
  if (steps.length === 0) return null;
  return (
    <div className="mt-4">
      <h3 className="text-sm font-medium">
        Steps <span className="font-normal text-muted">({steps.length})</span>
      </h3>
      <ol className="mt-2 space-y-2">
        {steps.map((step, i) => (
          <li
            key={`${i}-${step.name}`}
            className="min-w-0 rounded border border-rule bg-sunk p-3 text-sm"
          >
            <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
              <span>
                {i + 1}. <StepName step={step} entry={result.entries?.[step.name]} />
              </span>
              <Evidence v={step.verification} />
            </div>
            <div className="overflow-x-auto">
              <Math latex={`${step.before_latex} = ${step.after_latex}`} display />
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

function OtherForms({ result }: { result: SimplifyResult }) {
  const alternatives = result.alternatives ?? [];
  if (alternatives.length === 0) return null;
  return (
    <details className="mt-4">
      <summary className="cursor-pointer text-sm font-medium">
        Other forms <span className="font-normal text-muted">({alternatives.length}): show</span>
      </summary>
      <ul className="mt-2 space-y-2">
        {alternatives.map((alt) => (
          <li key={alt.result_text} className="rounded border border-rule p-3 text-sm">
            <div className="overflow-x-auto">
              <Math latex={alt.result_latex} display />
            </div>
            <div className="text-xs text-muted">
              {alt.steps.length} {alt.steps.length === 1 ? "step" : "steps"}, verified end to end:{" "}
              {alt.verification.numeric_points_agreed}/{alt.verification.numeric_points_checked}{" "}
              points agree
            </div>
          </li>
        ))}
      </ul>
    </details>
  );
}

function NotApplied({ result }: { result: SimplifyResult }) {
  const blocked: BlockedMove[] = result.blocked ?? [];
  if (blocked.length === 0) return null;
  return (
    <div className="mt-4 rounded bg-warn-soft p-3 text-sm text-warn">
      <div>Matches, but not applied: the conditions cannot be shown from your assumptions.</div>
      <ul className="mt-2 space-y-2">
        {blocked.map((b) => {
          const entry = result.entries?.[b.rule_id];
          return (
            <li key={`${b.rule_id}-${b.direction}-${b.matched_latex}`} className="min-w-0">
              {entry ? (
                <>
                  <span className="font-medium">{entry.title}</span>{" "}
                  <code className="text-xs">{b.rule_id}</code>
                </>
              ) : (
                <code className="text-xs">{b.rule_id}</code>
              )}
              <ul className="list-disc pl-5">
                {b.unmet_conditions.map((c) => (
                  <li key={`${c.variable}-${c.assumption}`}>
                    <code>{c.variable}</code> must be {c.assumption} (the pattern variable bound to
                    the matched part)
                  </li>
                ))}
              </ul>
            </li>
          );
        })}
      </ul>
    </div>
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
          {GOAL[result.goal ?? "simplify"].heading}
        </h2>
        <StatusPill
          status={result.status}
          okLabel={
            v?.symbolic_zero
              ? `${GOAL[result.goal ?? "simplify"].done} and proved equal`
              : `${GOAL[result.goal ?? "simplify"].done}, numerically checked`
          }
        />
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

      <Restrictions
        latex={result.restrictions_latex}
        complete={result.restrictions_complete ?? true}
      />
      <p className="mt-3 text-sm text-muted">{result.message}</p>

      {result.status === "ok" && <Steps result={result} />}
      {result.status === "ok" && <OtherForms result={result} />}
      <NotApplied result={result} />

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
