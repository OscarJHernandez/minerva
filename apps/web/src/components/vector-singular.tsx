"use client";

import { Math } from "@minerva/ui/math";
import type { VectorResult, VectorSingular } from "@/lib/types";

const STATUS_LABEL: Record<string, string> = {
  established: "established",
  checked: "checked numerically",
  not_established: "not established",
};

const CLAIM_TEXT: Record<string, string> = {
  agrees: "Your statement agrees with the derived result.",
  no_delta_term:
    "Your statement has no delta term. A function that vanishes away from the source cannot equal this distribution.",
  different_strength: "Your statement has a different coefficient (including its sign).",
  different_source: "Your statement puts the delta at a different point.",
  not_readable: "Your statement could not be read as a multiple of a delta.",
};

/**
 * The classical result and the distributional one, side by side and never merged: the classical divergence of
 * r/r³ is 0 wherever it is defined, and as a distribution it is a point mass. The derivation lists each step
 * with what is known about it; the pairings are independent numerical evidence, not a proof.
 */
export function SingularSection({
  result,
  singular,
}: {
  result: VectorResult;
  singular: VectorSingular;
}) {
  const agreed = singular.weak_checks.filter((w) => w.passed).length;
  return (
    <div className="mt-3 space-y-3" data-testid="singular">
      <div>
        <div className="text-xs uppercase tracking-wide text-muted">
          Classical result (pointwise, away from the source)
        </div>
        <div className="overflow-x-auto">
          <Math
            latex={`${result.input_latex} = ${singular.classical_latex}`}
            display
            label="Classical"
          />
        </div>
        <div className="overflow-x-auto text-xs text-muted">
          valid where <Math latex={singular.classical_domain_latex} />
        </div>
      </div>

      <div>
        <div className="text-xs uppercase tracking-wide text-muted">
          Distributional result (as a distribution on ℝ³)
        </div>
        <div className="overflow-x-auto">
          <Math
            latex={singular.distributional_latex}
            display
            className="text-xl"
            label="Distributional"
          />
        </div>
        <p className="text-xs text-muted" data-testid="singular-status">
          {singular.proof_status === "derived"
            ? "Derived by the shrinking-sphere argument for this kernel and confirmed by independent pairings with test functions. Derived is not proved: the pairings are supporting evidence."
            : "Not established."}
        </p>
      </div>

      {singular.claim && (
        <p
          role="note"
          data-testid="singular-claim"
          className={`rounded p-3 text-sm ${singular.claim.holds ? "bg-ok-soft text-ok" : "bg-warn-soft text-warn"}`}
        >
          <span className="overflow-x-auto">
            <Math latex={singular.claim.latex} />
          </span>{" "}
          {CLAIM_TEXT[singular.claim.reason] ?? singular.claim.note}
        </p>
      )}

      <details>
        <summary className="cursor-pointer text-sm font-medium">How it was derived</summary>
        <ol className="mt-1 space-y-2 text-sm" data-testid="singular-derivation">
          {singular.derivation.map((step) => (
            <li key={step.title}>
              <div className="font-medium">
                {step.title}{" "}
                <span className="text-xs font-normal text-muted">
                  ({STATUS_LABEL[step.status] ?? step.status})
                </span>
              </div>
              <div className="overflow-x-auto">
                <Math latex={step.latex} display />
              </div>
              <div className="text-xs text-muted">{step.note}</div>
            </li>
          ))}
        </ol>
      </details>

      <details>
        <summary className="cursor-pointer text-sm font-medium">
          Independent checks ({agreed} of {singular.weak_checks.length} pairings agree)
        </summary>
        <ul className="mt-1 space-y-1 text-xs" data-testid="singular-pairings">
          {singular.weak_checks.map((w) => (
            <li key={w.test_function_latex} className="overflow-x-auto">
              <Math latex={w.test_function_latex} /> — expected {w.expected.toPrecision(8)},
              measured {w.measured.toPrecision(8)} ({w.passed ? "agrees" : "differs"})
            </li>
          ))}
          {singular.flux_checks.map((f) => (
            <li key={f.label}>
              {f.label}: {f.measured.toPrecision(8)} ({f.passed ? "agrees" : "differs"})
            </li>
          ))}
        </ul>
        {singular.regularisation.length > 0 && (
          <p className="mt-1 text-xs text-muted" data-testid="singular-regularisation">
            Regularised field r/(r²+ε²)^(3/2): the pairing approaches the same value as ε shrinks (
            {singular.regularisation
              .map((r) => `ε=${r.eps}: error ${r.error.toPrecision(3)}`)
              .join("; ")}
            ). Supporting evidence only.
          </p>
        )}
        <p className="mt-1 text-xs text-muted">{singular.scope}</p>
      </details>
    </div>
  );
}
