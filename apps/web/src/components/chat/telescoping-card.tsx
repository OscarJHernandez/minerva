import { CopyMath as Formula } from "@/components/copyable-math";
import type { TelescopingResult } from "@/lib/telescoping-types";
import { StatusPill } from "../result-panel";

const OUTCOME: Record<TelescopingResult["outcome"], string> = {
  evaluated: "Evaluated and checked",
  equation: "Equation only: not evaluated",
  certificate: "Certificate only: not evaluated",
  none: "No result",
};

function integral(r: TelescopingResult): string | null {
  if (!r.input_latex) return null;
  const bounds = r.lower_latex && r.upper_latex ? `_{${r.lower_latex}}^{${r.upper_latex}}` : "";
  return `\\int${bounds} ${r.input_latex}\\,\\mathrm{d}${r.variable}`;
}

/**
 * Creative telescoping: the operator and certificate the engine found, whether the identity was
 * proved, which convergence and boundary conditions were established, the differential equation,
 * and the value only if it was derived and checked. A certificate or an equation is never shown as
 * the value of the integral: the heading says which one this is.
 */
export function TelescopingCard({ result }: { result: TelescopingResult }) {
  const notation = integral(result);
  const check = result.certificate_check;
  const verified = result.value_verification;
  return (
    <section
      className="rounded-lg border border-rule bg-raised p-5"
      aria-label="Creative telescoping"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-xl">Creative telescoping</h3>
        <StatusPill
          status={result.status}
          okLabel={result.outcome === "evaluated" ? "Derived and checked" : "Partial result"}
        />
      </div>
      {notation && <Formula latex={notation} display />}
      {result.parameter_domain_latex && (
        <p className="text-sm text-muted">
          Parameter: <Formula latex={result.parameter_domain_latex} />
        </p>
      )}
      <p className="mt-1 text-sm text-muted">{result.message}</p>
      {result.status === "ok" && (
        <p className="mt-2 text-sm font-medium">{OUTCOME[result.outcome]}</p>
      )}

      {result.operator && (
        <div className="mt-3">
          <h4 className="text-xs uppercase tracking-wide text-muted">
            Operator (order {result.operator.order})
          </h4>
          <Formula latex={result.operator.latex} display />
        </div>
      )}
      {result.certificate && (
        <div className="mt-2">
          <h4 className="text-xs uppercase tracking-wide text-muted">
            Certificate: L f = dG/d{result.variable}
          </h4>
          <Formula latex={`G = ${result.certificate.latex}`} display />
          {check && (
            <p className="text-sm text-muted">
              {check.symbolic_zero ? "Proved exactly" : "Not proved"}: {check.note}. Numerical legs
              (can only veto): {check.numeric_points_agreed}/{check.numeric_points_checked} points,{" "}
              {check.derivative_points_agreed}/{check.derivative_points_checked} derivative points.
            </p>
          )}
        </div>
      )}

      {result.conditions.length > 0 && (
        <div className="mt-3">
          <h4 className="text-xs uppercase tracking-wide text-muted">
            Conditions, from the declared assumptions only
          </h4>
          <ul className="mt-1 space-y-1 text-sm">
            {result.conditions.map((c, i) => (
              <li key={i} className="flex flex-wrap items-baseline gap-x-2">
                <span className={c.established ? "text-ok" : "text-bad"}>
                  {c.established ? "established" : "not established"}
                </span>
                <Formula latex={c.latex} />
                <span className="text-muted">{c.how}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.boundary && (
        <div className="mt-3">
          <h4 className="text-xs uppercase tracking-wide text-muted">
            Boundary contribution G(upper) − G(lower)
          </h4>
          <Formula latex={result.boundary.contribution_latex} display />
        </div>
      )}
      {result.equation && (
        <div className="mt-2">
          <h4 className="text-xs uppercase tracking-wide text-muted">
            Differential equation ({result.equation.homogeneous ? "homogeneous" : "inhomogeneous"})
          </h4>
          <Formula latex={result.equation.latex} display />
        </div>
      )}
      {result.solution && (
        <div className="mt-2">
          <h4 className="text-xs uppercase tracking-wide text-muted">
            Solution and initial values
          </h4>
          <Formula latex={result.solution.general_latex} display />
          <ul className="text-sm text-muted">
            {result.solution.initial_values_latex.map((v, i) => (
              <li key={i}>
                at <Formula latex={result.solution?.initial_point_latex ?? ""} />:{" "}
                <Formula latex={v} />
              </li>
            ))}
          </ul>
        </div>
      )}
      {result.result_latex && (
        <div className="mt-2">
          <h4 className="text-xs uppercase tracking-wide text-muted">Value</h4>
          <Formula latex={`I = ${result.result_latex}`} display />
          {verified && (
            <p className="text-sm text-muted">
              {verified.note}. Quadrature: {verified.quadrature_points_agreed}/
              {verified.quadrature_points_checked} parameter values agree.
            </p>
          )}
        </div>
      )}

      {result.unresolved.length > 0 && (
        <div className="mt-3 rounded bg-warn-soft px-3 py-2 text-sm text-warn">
          <strong>Not established:</strong>
          <ul className="ml-4 list-disc">
            {result.unresolved.map((u, i) => (
              <li key={i}>{u}</li>
            ))}
          </ul>
        </div>
      )}
      {result.notes.map((n, i) => (
        <p key={i} className="mt-2 text-xs text-muted">
          {n}
        </p>
      ))}
      {result.families.length > 0 && (
        <p className="mt-2 text-xs text-muted">
          Searched {result.families.length} famil{result.families.length === 1 ? "y" : "ies"} (
          operator order, certificate degree). A relation found within them says nothing about other
          families; none found proves nothing outside them.
        </p>
      )}
    </section>
  );
}
