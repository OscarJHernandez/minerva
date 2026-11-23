"use client";

import { Math } from "@minerva/ui/math";
import { useId, type ReactNode } from "react";
import type { DifferentiateResult } from "@/lib/types";
import { EquationNumber } from "./chat/equation-number";
import { FormulaButtons } from "./chat/formula-actions";
import { StatusPill } from "./result-panel";

// Presentation only: never shorten or algebraically rewrite the engine's LaTeX.
const LONG_FORMULA = 240;

export function DerivativePanel({
  result,
  total = false,
  number,
  actions,
}: {
  result: DifferentiateResult;
  total?: boolean;
  number?: number | null;
  actions?: ReactNode;
}) {
  const headingId = useId();
  const v = result.verification;
  const shown = result.status === "ok" && !!result.result_latex;
  const onMotion = shown && result.zero_on_motion != null;
  const zeroOnMotion = onMotion && result.zero_on_motion === true && !!result.on_motion_latex;
  const long = (result.result_latex?.length ?? 0) > LONG_FORMULA;
  // When the engine decided the on-motion question (zero_on_motion is true or false) it appended that
  // evaluation to `steps`: a substitution of the equations of motion, not another differentiation.
  // The last differentiation step is the result itself, so it is never listed as intermediate.
  const evaluationSteps = result.zero_on_motion != null ? 1 : 0;
  const intermediate = result.steps.slice(0, -(1 + evaluationSteps));
  const definitions = result.definitions_used ?? [];
  const rawResult = shown && result.result_latex && (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between gap-3">
        <div className="text-xs uppercase tracking-wide text-muted">
          {onMotion ? "Before using the equations of motion" : "Result"}
        </div>
        {number != null && <EquationNumber n={number} />}
      </div>
      <Math
        latex={result.result_latex}
        display
        className={long ? "text-base" : "text-2xl"}
        label="Result"
      />
      {long && (
        <p className="text-xs text-muted">Scroll horizontally to read the full expression.</p>
      )}
      {actions || <FormulaButtons latex={result.result_latex} />}
    </div>
  );

  return (
    <section
      aria-labelledby={headingId}
      className="min-w-0 rounded-lg border border-rule bg-raised p-4 sm:p-5"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id={headingId} className="font-serif text-xl">
          {total ? "Total time derivative" : "Differentiate"}
        </h2>
        <StatusPill
          status={result.status}
          okLabel={
            v?.steps_checked === 0 ? "Differentiated" : "Differentiated, numerically checked"
          }
        />
      </div>

      {zeroOnMotion && result.on_motion_latex && (
        <div className="mt-4 rounded border border-rule bg-ok-soft p-3">
          <h3 className="text-sm font-medium">Zero on the supplied equations of motion</h3>
          <Math
            latex={result.on_motion_latex}
            display
            className="text-2xl"
            label="Result on the equations of motion"
          />
          <p className="text-sm">
            Numerically confirmed
            {result.symbolic_zero_on_motion === true ? "; also a symbolic zero." : "."}
          </p>
          <p className="mt-1 text-xs text-muted">
            This conclusion applies to motions of the supplied Lagrangian.
          </p>
        </div>
      )}
      {onMotion && result.zero_on_motion === false && (
        <div className="mt-4 rounded border border-rule p-3">
          <h3 className="text-sm font-medium">Not zero on the supplied equations of motion</h3>
          <p className="mt-1 text-sm text-muted">
            The numerical check found a nonzero time derivative on these motions.
          </p>
        </div>
      )}

      {shown && (
        <div className="mt-4">
          {onMotion || long ? (
            <details>
              <summary className="cursor-pointer text-sm font-medium">
                {onMotion
                  ? "Full derivative before using the equations of motion"
                  : "Full derivative"}
              </summary>
              <div className="mt-3">{rawResult}</div>
            </details>
          ) : (
            rawResult
          )}
          {long && !onMotion && (
            <p className="mt-2 text-sm text-muted">
              The verified derivative is long. Expand it to read or copy the complete expression.
            </p>
          )}
        </div>
      )}

      {result.input_latex && (
        <details className="mt-4">
          <summary className="cursor-pointer text-sm text-muted">
            Input and derivative taken
          </summary>
          <div className="mt-2 text-xs uppercase tracking-wide text-muted">Input as read</div>
          <Math latex={result.input_latex} display />
          {result.notation_latex && result.status === "ok" && (
            <>
              <div className="mt-2 text-xs uppercase tracking-wide text-muted">
                Derivative taken
              </div>
              <Math latex={result.notation_latex} display />
            </>
          )}
        </details>
      )}

      {shown && definitions.length > 0 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-sm text-muted">
            Definitions substituted ({definitions.length})
          </summary>
          <p className="mt-2 text-sm text-muted">
            The derivative and its checks use these expressions in place of the named quantities.
          </p>
          {definitions.map((latex, i) => (
            <Math key={i} latex={latex} display />
          ))}
        </details>
      )}

      {onMotion && result.zero_on_motion === false && result.on_motion_latex && (
        <details className="mt-3">
          <summary className="cursor-pointer text-sm text-muted">
            Expression after substituting the equations of motion
          </summary>
          <p className="mt-2 text-sm text-muted">
            An exact substitution of the confirmed accelerations; this expression was not separately
            sampled.
          </p>
          <Math latex={result.on_motion_latex} display label="Substituted expression" />
        </details>
      )}

      {result.status === "ok" && intermediate.length > 0 && (
        <details className="mt-3 text-sm">
          <summary className="cursor-pointer text-muted">
            Intermediate derivatives ({intermediate.length})
          </summary>
          <ol className="mt-2 space-y-2">
            {intermediate.map((s) => (
              <li key={s.order_so_far} className="min-w-0">
                <div className="text-xs text-muted">
                  after d{s.variable}, step {s.order_so_far}
                </div>
                <Math latex={s.latex} display />
              </li>
            ))}
          </ol>
        </details>
      )}

      {!shown && <p className="mt-3 text-sm text-muted">{result.message}</p>}
      <details className="mt-4" open={!shown}>
        <summary className="cursor-pointer text-sm text-muted">Verification and notes</summary>
        {shown && <p className="mt-2 text-sm text-muted">{result.message}</p>}
        {result.notes.map((note) => (
          <p key={note} className="mt-2 text-sm text-muted">
            {note}
          </p>
        ))}
        {v && (
          <dl className="mt-3 grid grid-cols-1 gap-x-4 gap-y-1 text-sm sm:grid-cols-[auto_minmax(0,1fr)]">
            <dt className="text-muted">Derivative check</dt>
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
            <dt className="text-muted">Symbolic derivative check</dt>
            <dd>None: no independent symbolic test of a derivative exists</dd>
            {zeroOnMotion && (
              <>
                <dt className="text-muted">Zero on motion</dt>
                <dd>
                  {result.symbolic_zero_on_motion === true
                    ? "Numerically confirmed and a symbolic zero on the supplied equations of motion."
                    : "Numerically confirmed; no symbolic zero reported."}
                </dd>
              </>
            )}
            <dt className="text-muted">Computed in</dt>
            <dd>{result.duration_ms} ms</dd>
          </dl>
        )}
      </details>
    </section>
  );
}
