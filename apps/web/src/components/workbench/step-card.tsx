"use client";

import { Math } from "@minerva/ui/math";
import Link from "next/link";
import { useState } from "react";
import {
  assumptionsSummary,
  continuation,
  resultLatex,
  stepLabel,
  verificationSummary,
} from "@/lib/derivation";
import type {
  ApplyIdentityResult,
  DifferentiateResult,
  HistoryItem,
  IntegrateResult,
  IsolateResult,
  SeriesResult,
  SimplifyResult,
  StepResult,
} from "@/lib/types";
import { DerivativePanel } from "../derivative-panel";
import { IntegralPanel } from "../integral-panel";
import { ResultPanel, StatusPill } from "../result-panel";
import { SeriesPanel } from "../series-panel";
import { DiagnosticsView, IsolatePanel, Premises, StepResultPanel } from "../step-result-panel";

interface Props {
  item: HistoryItem;
  /** 1-based position in the derivation. */
  number: number;
  /** Position of the step this one continued, when it is still in the derivation. */
  parentNumber: number | null;
  current: boolean;
  /** `which`: for an equation, the index of the solution to continue with. */
  onContinue: (item: HistoryItem, edit: boolean, which?: number) => void;
  onDelete: (item: HistoryItem) => void;
}

function FullResult({ item }: { item: HistoryItem }) {
  switch (item.operation) {
    case "simplify":
      return <ResultPanel result={item.result as SimplifyResult} />;
    case "differentiate":
      return <DerivativePanel result={item.result as unknown as DifferentiateResult} />;
    case "integrate":
      return <IntegralPanel result={item.result as unknown as IntegrateResult} />;
    case "series":
      return <SeriesPanel result={item.result as unknown as SeriesResult} />;
    case "substitute":
    case "rewrite":
      return <StepResultPanel result={item.result as unknown as StepResult} />;
    case "isolate":
      return <DiagnosticsView d={(item.result as unknown as IsolateResult).diagnostics} />;
    case "apply_identity": {
      const r = item.result as unknown as ApplyIdentityResult;
      return (
        <div className="rounded-lg border border-rule bg-raised p-4 text-sm">
          {r.matched_subexpression_latex && (
            <div>
              <div className="text-xs uppercase tracking-wide text-muted">
                Applied to this part of the expression
              </div>
              <Math latex={r.matched_subexpression_latex} display />
            </div>
          )}
          <p className="text-muted">{r.message}</p>
          {r.verification && (
            <p className="mt-1">
              {r.verification.numeric_points_agreed} of {r.verification.numeric_points_checked}{" "}
              sample points agree
              {r.verification.symbolic_zero ? "; difference simplifies to 0" : ""}
            </p>
          )}
        </div>
      );
    }
    default:
      return null;
  }
}

function Copy({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="text-xs text-accent hover:underline"
      onClick={async () => {
        await navigator.clipboard?.writeText(value);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
    >
      {copied ? "Copied" : label}
    </button>
  );
}

export function StepCard({ item, number, parentNumber, current, onContinue, onDelete }: Props) {
  const cont = continuation(item);
  const latex = resultLatex(item);
  const failed = item.status !== "ok" && item.status !== "unchanged";
  const assumptions = assumptionsSummary(item.assumptions);
  const verified = verificationSummary(item);
  const message = (item.result as unknown as { message?: string }).message;
  const extra = item.result as unknown as {
    restrictions_latex?: string[];
    restrictions_complete?: boolean;
  };
  const restrictions = failed ? [] : (extra.restrictions_latex ?? []);
  const restrictionsIncomplete = !failed && extra.restrictions_complete === false;
  const continuing = cont !== null;
  const identity =
    item.operation === "apply_identity" ? (item.result as unknown as ApplyIdentityResult) : null;
  const isolated = item.operation === "isolate" ? (item.result as unknown as IsolateResult) : null;
  const stepped =
    item.operation === "substitute" || item.operation === "rewrite"
      ? (item.result as unknown as StepResult)
      : null;
  const okLabel =
    stepped?.evidence === "conditional"
      ? "Holds given your relation"
      : isolated?.complete
        ? "Verified, none dropped"
        : "Verified";

  return (
    <li
      id={`step-${item.id}`}
      aria-label={`Step ${number}: ${stepLabel(item)}`}
      className={`rounded-lg border p-4 ${current ? "border-accent" : "border-rule"} bg-raised`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-medium">
          <span className="mr-2 inline-block min-w-6 rounded bg-sunk px-1.5 text-center font-mono text-xs text-muted">
            {number}
          </span>
          {identity ? (
            <Link
              href={`/reference/${identity.rule_id}`}
              className="hover:text-accent hover:underline"
            >
              {identity.entry.title}
            </Link>
          ) : (
            stepLabel(item)
          )}
          <span className="ml-2 font-normal text-muted">
            {parentNumber !== null ? `from step ${parentNumber}` : "new input"}
          </span>
        </h3>
        <StatusPill status={item.status} okLabel={okLabel} />
      </div>

      <div className="mt-2 text-sm text-muted">
        {item.format === "latex" ? (
          <Math latex={item.input} />
        ) : (
          <code className="font-mono text-xs">{item.input}</code>
        )}
      </div>

      {stepped && item.status === "ok" && stepped.evidence === "conditional" && (
        <Premises given={stepped.given} />
      )}
      {latex && (
        <Math latex={latex} display className="text-xl" label={`Result of step ${number}`} />
      )}
      {isolated && (
        <div className="mt-2">
          <IsolatePanel result={isolated} onContinue={(i) => onContinue(item, false, i)} />
        </div>
      )}
      {failed && message && <p className="mt-2 text-sm text-warn">{message}</p>}

      {restrictions.length > 0 && (
        <p className="mt-1 text-xs text-muted">
          Excludes points the input allowed:{" "}
          {restrictions.map((r, i) => (
            <span key={r}>
              {i > 0 && ", "}
              <Math latex={r} />
            </span>
          ))}
        </p>
      )}
      {restrictions.length > 0 && continuing && (
        <p className="text-xs text-muted">
          A step that continues from this result does not carry these exclusions.
        </p>
      )}
      {restrictionsIncomplete && (
        <p className="mt-1 text-xs text-warn">
          The points this step excludes could not all be determined.
        </p>
      )}
      {(assumptions || verified) && (
        <p className="mt-1 text-xs text-muted">
          {assumptions
            ? `Assumptions: ${assumptions}`
            : "No assumptions declared (complex symbols)"}
          {verified ? ` · ${verified}` : ""}
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1">
        {cont ? (
          <>
            <button
              type="button"
              onClick={() => onContinue(item, false)}
              className="rounded border border-accent px-2.5 py-1 text-xs font-medium text-accent hover:bg-accent-soft"
            >
              Continue from here
            </button>
            <button
              type="button"
              onClick={() => onContinue(item, true)}
              className="text-xs text-accent hover:underline"
            >
              Edit
            </button>
            {cont.latex && <Copy label="Copy LaTeX" value={cont.latex} />}
            <Copy label="Copy text" value={cont.text} />
          </>
        ) : (
          failed && <span className="text-xs text-muted">No result to continue from.</span>
        )}
        <button
          type="button"
          onClick={() => onDelete(item)}
          aria-label={`Delete step ${number}`}
          className="ml-auto text-xs text-muted hover:text-bad"
        >
          Delete
        </button>
      </div>

      <details className="mt-3 text-sm">
        <summary className="cursor-pointer text-muted">Full verification</summary>
        <div className="mt-2">
          <FullResult item={item} />
        </div>
      </details>
    </li>
  );
}
