import { Math } from "@minerva/ui/math";
import type { PerturbationReport, PerturbationRequest } from "@/lib/chat-types";
import { StatusPill } from "../result-panel";
import { LinePlot } from "./line-plot";
import { PerturbationForm } from "./perturbation-form";

function sci(v: number | null): string {
  return v === null ? "–" : v === 0 ? "0" : v.toExponential(2);
}

/** A perturbation solution; formulas appear only if the residual and initial-condition checks passed. */
export function PerturbationCard({
  report,
  request,
}: {
  report: PerturbationReport;
  request?: PerturbationRequest;
}) {
  const shown = report.status === "ok";
  const v = report.verification;
  const c = report.comparison;
  return (
    <section
      className="rounded-lg border border-rule bg-raised p-5"
      aria-label="Perturbation solution"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-xl">Perturbation solution</h3>
        <StatusPill status={report.status} okLabel="Solved and verified" />
      </div>
      <p className="mt-1 text-sm text-muted">{report.message}</p>

      {shown && (
        <>
          <p className="mt-3 text-xs text-muted">
            {report.mode === "amplitude"
              ? "Small parameter: the amplitude of the initial displacement."
              : "Small parameter: a parameter of the Lagrangian; amplitude A is the initial displacement."}{" "}
            Starts at rest; solved to order {report.order}.
          </p>
          <div className="mt-2 space-y-1 overflow-x-auto">
            {report.linear_frequency_latex && (
              <Math latex={report.linear_frequency_latex} display />
            )}
            {report.frequency_latex && (
              <Math latex={report.frequency_latex} display label="Frequency" />
            )}
            {report.period_ratio_latex && (
              <Math latex={report.period_ratio_latex} display label="Period ratio" />
            )}
          </div>
          {report.solution_latex && (
            <div className="mt-2 overflow-x-auto">
              <h4 className="text-sm font-medium">Solution</h4>
              <Math latex={report.solution_latex} display label="Solution" />
            </div>
          )}
          <details className="mt-3 text-sm">
            <summary className="cursor-pointer text-muted">
              Each order ({report.orders.length})
            </summary>
            <ul className="mt-2 space-y-2">
              {report.orders.map((o) => (
                <li key={o.order} className="overflow-x-auto rounded border border-rule p-3">
                  <Math latex={o.shape_latex} display />
                  {o.frequency_latex && <Math latex={o.frequency_latex} display />}
                </li>
              ))}
            </ul>
          </details>
        </>
      )}

      {v && (
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          <dt className="text-muted">Residual</dt>
          <dd>
            {v.residual_passed
              ? `falls at the claimed order in the unexpanded equation of motion (${v.parameter_sets_checked} parameter set${v.parameter_sets_checked === 1 ? "" : "s"})`
              : "does not fall at the claimed order"}
          </dd>
          <dt className="text-muted">Initial conditions</dt>
          <dd>{v.initial_conditions_passed ? "hold" : "do not hold"}</dd>
          <dt className="text-muted">Secular terms</dt>
          <dd>{v.bounded ? "none: every order is a finite sum of cos(mτ)" : "not excluded"}</dd>
        </dl>
      )}

      {shown && c && (
        <div className="mt-4" aria-label="Comparison with the numerical solution">
          <h4 className="text-sm font-medium">Against the verified numerical solution</h4>
          {c.status === "ok" ? (
            <>
              <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                <dt className="text-muted">Largest error</dt>
                <dd>{sci(c.max_error)}</dd>
                <dt className="text-muted">With half the small parameter</dt>
                <dd>{sci(c.max_error_half)}</dd>
                <dt className="text-muted">Error fell at order</dt>
                <dd>
                  {c.observed_order === null ? "–" : c.observed_order.toFixed(1)} (predicted{" "}
                  {c.predicted_order})
                </dd>
              </dl>
              <p className="mt-1 text-xs text-muted">Error: {c.scaling}.</p>
              <LinePlot
                series={[
                  { label: "numerical", x: c.t, y: c.numerical },
                  { label: "perturbative", x: c.t, y: c.perturbative },
                ]}
                xLabel="t"
                yLabel="u"
                label="Perturbative and numerical solutions against time"
              />
              {c.notes.map((n) => (
                <p key={n} className="mt-1 text-xs text-muted">
                  {n}
                </p>
              ))}
            </>
          ) : (
            <p className="mt-1 text-sm text-warn">No comparison is shown: {c.message}</p>
          )}
        </div>
      )}

      {shown && request && <PerturbationForm base={request} parameters={report.parameters} />}

      {shown && !c && (
        <p className="mt-4 text-xs text-muted">
          This is a local series in the small parameter. To compare it with the numerical solution,
          give parameter values, the small parameter or amplitude and an end time.
        </p>
      )}
      {report.notes.map((n) => (
        <p key={n} className="mt-2 text-sm text-warn">
          {n}
        </p>
      ))}
    </section>
  );
}
