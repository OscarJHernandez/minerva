"use client";

import { Math } from "@minerva/ui/math";
import { useId } from "react";
import type { VectorRequest, VectorResult } from "@/lib/types";
import { StatusPill } from "./result-panel";

/**
 * Vector calculus: what was read, the result by component along the unit vectors, where the
 * coordinates break down and how the result was checked. A result is drawn only when the engine
 * verified it; numerical agreement is labelled as such, never as a proof.
 */
export function VectorPanel({
  result,
  request,
}: {
  result: VectorResult;
  request?: VectorRequest;
}) {
  const headingId = useId();
  const shown = result.status === "ok";
  const v = result.verification;
  const label =
    result.evidence === "conditional"
      ? "Holds given the premises you supplied"
      : result.evidence === "identity" && result.steps.length > 0
        ? "Each step checked"
        : v
          ? v.symbolic_zero === true
            ? "Proved by a symbolic zero test, and numerically checked"
            : "Numerically checked; no symbolic proof"
          : "Verified";
  return (
    <section
      aria-labelledby={headingId}
      className="rounded-lg border border-rule bg-raised p-5"
      data-testid="vector-panel"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id={headingId} className="font-serif text-xl">
          Vector calculus
        </h2>
        <StatusPill status={result.status} okLabel={label} />
      </div>

      <p className="mt-1 text-xs text-muted">
        {result.action === "convert" && result.to
          ? `Converted from ${request?.coordinates ?? "the given"} to ${result.coordinates} coordinates`
          : `${result.coordinates[0]?.toUpperCase() ?? ""}${result.coordinates.slice(1)} coordinates`}
      </p>

      {result.input_latex && (
        <div className="mt-3">
          <div className="text-xs uppercase tracking-wide text-muted">Input as read</div>
          <div className="overflow-x-auto">
            <Math latex={result.input_latex} display label="Input" />
          </div>
        </div>
      )}

      {shown && result.result_kind === "matrix" && result.result_latex && (
        <div className="mt-3 space-y-2">
          <div className="text-xs uppercase tracking-wide text-muted">
            Jacobian ∂(x, y, z)/∂(q₁, q₂, q₃)
          </div>
          <div className="overflow-x-auto">
            <Math latex={result.result_latex} display label="Jacobian" />
          </div>
          {result.determinant_latex && (
            <div className="overflow-x-auto">
              <Math latex={`\\det J = ${result.determinant_latex}`} display label="Determinant" />
            </div>
          )}
          {result.scale_factors_latex.length > 0 && (
            <div className="overflow-x-auto">
              <Math
                latex={`h = (${result.scale_factors_latex.join(",\\ ")})`}
                label="Scale factors"
              />
            </div>
          )}
        </div>
      )}

      {shown && result.result_kind === "scalar" && result.result_latex && (
        <div className="mt-3">
          <div className="text-xs uppercase tracking-wide text-muted">Result</div>
          <div className="overflow-x-auto">
            <Math latex={result.result_latex} display className="text-xl" label="Result" />
          </div>
        </div>
      )}

      {shown && result.result_kind === "vector" && (
        <div className="mt-3">
          <div className="text-xs uppercase tracking-wide text-muted">
            Result, by component along the unit vectors
          </div>
          <ul className="mt-1 space-y-1" data-testid="components">
            {result.components_latex.map((c, i) => (
              <li key={result.basis_latex[i] ?? i} className="min-w-0 overflow-x-auto">
                <Math latex={`${result.basis_latex[i] ?? ""}:\\quad ${c}`} label="Component" />
              </li>
            ))}
          </ul>
        </div>
      )}

      {shown && result.steps.length > 0 && (
        <div className="mt-3" data-testid="steps">
          <div className="text-xs uppercase tracking-wide text-muted">Steps</div>
          <ol className="mt-1 space-y-3">
            {result.steps.map((step, i) => (
              <li
                key={`${i}-${step.title}`}
                className="rounded border border-rule bg-sunk p-3"
                data-testid={`step-${i}`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-sm font-medium">
                    {i + 1}. {step.title}
                  </span>
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs ${
                      step.evidence === "conditional"
                        ? "bg-warn-soft text-warn"
                        : "bg-ok-soft text-ok"
                    }`}
                  >
                    {step.evidence === "conditional" ? "Conditional" : "Identity"}
                  </span>
                </div>
                {step.part_latex && (
                  <div className="mt-1 overflow-x-auto text-sm">
                    <span className="text-xs text-muted">Part </span>
                    <Math latex={step.part_latex} label="Part" />
                  </div>
                )}
                <div className="overflow-x-auto">
                  <Math latex={step.after_latex} display label={`Step ${i + 1} result`} />
                </div>
                {step.needs.length > 0 && (
                  <p className="text-xs text-muted">
                    Needs {step.needs.join(", ")} fields (declared by you).
                  </p>
                )}
                {step.given.length > 0 && (
                  <p className="text-xs text-warn">Given: {step.given.join("; ")}</p>
                )}
              </li>
            ))}
          </ol>
        </div>
      )}

      {shown && (result.equations_latex?.length ?? 0) > 0 && (
        <div className="mt-3" data-testid="equations">
          <div className="text-xs uppercase tracking-wide text-muted">Derived equation</div>
          {result.equations_latex?.map((eq) => (
            <div key={eq} className="overflow-x-auto">
              <Math latex={eq} display label="Derived equation" />
            </div>
          ))}
        </div>
      )}

      {result.given.length > 0 && (
        <p
          role="note"
          className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn"
          data-testid="given"
        >
          This holds only given {result.given.join("; ")}. You supplied{" "}
          {result.given.length === 1 ? "it" : "them"}: Minerva does not check that{" "}
          {result.given.length === 1 ? "it is" : "they are"} true, so the result is not an identity.
        </p>
      )}

      {result.applicable.length > 0 && (
        <div className="mt-3" data-testid="applicable">
          <div className="text-xs uppercase tracking-wide text-muted">
            Identities that match (none applied)
          </div>
          <ul className="mt-1 space-y-2">
            {result.applicable.map((a, i) => (
              <li
                key={`${a.rule_id}-${a.path.join(".")}-${i}`}
                className={`rounded border border-rule p-2 ${a.blocked ? "opacity-70" : ""}`}
              >
                <div className="text-sm font-medium">
                  {a.rule_id}
                  {a.reverse ? " (right to left)" : ""}
                  {a.path.length > 0 ? ` at part ${a.path.join(".")}` : ""}
                </div>
                <div className="overflow-x-auto">
                  <Math latex={`${a.part_latex} \\;\\to\\; ${a.result_latex}`} label="Rewrite" />
                </div>
                {a.blocked && <p className="text-xs text-warn">Blocked: {a.note}</p>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.search_complete === false && result.steps.length > 0 && (
        <p className="mt-2 text-xs text-muted">
          The search stopped at its bounds, so this chain may not be the shortest.
        </p>
      )}

      {result.blocked.length > 0 && (
        <div role="note" className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn">
          {result.blocked.map((b) => (
            <p key={`${b.needs}-${b.fields.join()}`}>
              Not shown: it needs{" "}
              {b.needs === "c2" ? "twice continuously differentiable" : "smooth"}{" "}
              {b.fields.join(", ")}. {b.note}.
            </p>
          ))}
          <p className="mt-1 text-xs">
            Minerva does not assume smoothness. If the field is that smooth, declare it and ask
            again.
          </p>
        </div>
      )}

      {(result.needs_premises?.length ?? 0) > 0 && (
        <div
          role="note"
          className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn"
          data-testid="needs"
        >
          <p>
            This derivation needs these field equations as premises, and they were not supplied:
          </p>
          <ul className="mt-1 list-disc pl-5">
            {result.needs_premises?.map((n) => (
              <li key={n}>
                <code>{n}</code>
              </li>
            ))}
          </ul>
          <p className="mt-1 text-xs">They are yours to adopt: Minerva does not assume them.</p>
        </div>
      )}

      {result.undeclared.length > 0 && (
        <p role="note" className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn">
          Not declared: {result.undeclared.join(", ")}. Declare each as a field or a constant; an
          undeclared symbol is never treated as a constant.
        </p>
      )}

      {result.message && !shown && <p className="mt-3 text-sm text-muted">{result.message}</p>}

      {shown && result.restrictions_latex.length > 0 && (
        <div className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn" data-testid="restrictions">
          <div>Valid only where</div>
          <ul className="mt-1">
            {result.restrictions_latex.map((r) => (
              <li key={r} className="overflow-x-auto text-ink">
                <Math latex={r} />
              </li>
            ))}
          </ul>
          <p className="mt-1 text-xs">
            This says where the coordinates break down, not whether the field is singular
            {result.restrictions_complete
              ? "."
              : "; there may be more restrictions that were not determined."}
          </p>
        </div>
      )}

      {result.model_check && result.model_check.status !== "failed" && (
        <p className="mt-3 text-xs text-muted" data-testid="model-check">
          {result.model_check.status === "passed"
            ? `Evaluated on exact solutions of the premises: consistent with ${result.model_check.models} of them at ${result.model_check.points} points. Evidence, not a verification.`
            : `Not evaluated on exact solutions: ${result.model_check.note}`}
        </p>
      )}

      {v && (
        <details className="mt-3">
          <summary className="cursor-pointer text-sm font-medium">
            How it was checked ({v.points_agreed} of {v.points_checked} points)
          </summary>
          <ul className="mt-1 space-y-1 text-xs">
            {v.checks.map((c) => (
              <li key={c.name}>
                {c.name}: {c.passed === true ? "passed" : c.passed === false ? "failed" : "not run"}
                {c.note ? <span className="text-muted"> — {c.note}</span> : null}
              </li>
            ))}
          </ul>
          <p className="mt-1 text-xs text-muted">
            An independent reading of the input by finite differences in Cartesian coordinates.
            Numerical agreement is evidence, not a proof.
          </p>
        </details>
      )}

      {result.assumptions_used.length > 0 && (
        <details className="mt-2">
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
    </section>
  );
}
