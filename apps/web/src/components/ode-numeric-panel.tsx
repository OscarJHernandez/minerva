import type { ReactNode } from "react";
import type { NumericCheck, NumericWarning, OdeNumericReport } from "@/lib/ode-types";
import { LinePlot, type PlotSeries } from "./chat/line-plot";
import { StatusPill } from "./result-panel";

const CHECK: Record<NumericCheck["name"], string> = {
  initial_conditions: "The initial values are reproduced",
  independent_method: "An independent integrator agrees",
  tolerance_refinement: "Refining the tolerance converges",
  defect: "The integral form of the equation holds over the steps",
  first_integral: "A verified conserved quantity stays constant",
  boundary_conditions: "The conditions hold at both ends",
  residual: "The equation holds on a fine grid",
  mesh_refinement: "A refined mesh gives the same solution",
  shooting: "Integrating from the left end reaches the right end",
};

const WARNING: Record<NumericWarning, string> = {
  stiffness_indicated:
    "The step size was limited by stability or the budget ran out: the problem looks stiff (judged from measured behaviour, not predicted).",
  step_collapse: "The step size collapsed: a singularity, or a very stiff region.",
  not_converged: "Refining the tolerance did not converge.",
  blow_up_suspected: "The solution grew beyond any bound.",
  domain_left: "The right-hand side left its real domain.",
};

const MARK = { passed: "passed", failed: "FAILED", skipped: "not run" } as const;

/** A label for a plot axis or legend: the LaTeX name with its markup removed. */
export function plainLabel(latex: string): string {
  return latex.replace(/\\([A-Za-z]+)/g, "$1").replace(/[{}]/g, "");
}

function format(value: number | null): string {
  if (value === null) return "";
  if (value === 0) return "0";
  return value.toExponential(1).replace("e+", "e");
}

export function trajectories(report: OdeNumericReport): PlotSeries[] {
  return report.series.map((s) => ({ label: plainLabel(s.label_latex), x: report.t, y: s.values }));
}

/** A numerical approximation on a given interval: its plots, and every check that was run on it. */
export function OdeNumericPanel({
  report,
  independent,
  children,
}: {
  report: OdeNumericReport;
  independent?: string;
  children?: ReactNode;
}) {
  const variable = independent ?? report.interpretation?.independent ?? "t";
  const [first, second] = report.series;
  const phase: PlotSeries[] =
    report.series.length === 2 && first && second
      ? [{ label: "trajectory", x: first.values, y: second.values }]
      : [];
  return (
    <section
      className="rounded-lg border border-rule bg-raised p-4"
      aria-label="Numerical solution"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-lg">Numerical solution</h3>
        <StatusPill
          status={report.status}
          okLabel="Numerical approximation, checked independently"
        />
      </div>
      {report.status !== "ok" && <p className="mt-3 text-sm text-warn">{report.message}</p>}
      {report.warnings.length > 0 && (
        <ul className="mt-3 list-disc pl-5 text-sm text-warn">
          {report.warnings.map((w) => (
            <li key={w}>{WARNING[w]}</li>
          ))}
        </ul>
      )}
      {report.status === "ok" && report.t_start !== null && report.t_end !== null && (
        <>
          <p className="mt-3 text-sm">
            On the interval {variable} from {report.t_start} to {report.t_end}, with{" "}
            <strong>{report.method_used}</strong>: {report.method_reason}. {report.steps} steps,{" "}
            {report.evaluations} evaluations; rtol {report.rtol}, atol {report.atol}.
          </p>
          <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-sm">
            <dt className="text-muted">
              {report.method_used === "solve_bvp" ? "Conditions (yours)" : "Initial values (yours)"}
            </dt>
            <dd>
              {report.initial.map((i) => `${plainLabel(i.label_latex)} = ${i.value}`).join(", ")}
            </dd>
            {Object.keys(report.parameters).length > 0 && (
              <>
                <dt className="text-muted">Parameters (yours)</dt>
                <dd>
                  {Object.entries(report.parameters)
                    .map(([k, v]) => `${k} = ${v}`)
                    .join(", ")}
                </dd>
              </>
            )}
          </dl>
          <LinePlot
            series={trajectories(report)}
            xLabel={variable}
            yLabel="value"
            label={`The solution against ${variable}`}
          />
          {phase.length === 1 && first && second && (
            <LinePlot
              series={phase}
              xLabel={plainLabel(first.label_latex)}
              yLabel={plainLabel(second.label_latex)}
              label={`Phase portrait: ${plainLabel(second.label_latex)} against ${plainLabel(first.label_latex)}`}
            />
          )}
        </>
      )}
      {report.verification && (
        <details className="mt-3 text-sm" open={report.status !== "ok"}>
          <summary className="cursor-pointer text-muted">
            How it was checked
            {report.verification.independent_method
              ? ` (compared with ${report.verification.independent_method})`
              : ""}
          </summary>
          <table className="mt-2 w-full text-left text-xs">
            <thead>
              <tr className="text-muted">
                <th className="py-1 pr-2 font-medium">Check</th>
                <th className="py-1 pr-2 font-medium">Result</th>
                <th className="py-1 pr-2 font-medium">Measured</th>
                <th className="py-1 font-medium">Limit</th>
              </tr>
            </thead>
            <tbody>
              {report.verification.checks.map((c) => (
                <tr key={c.name} className="border-t border-rule align-top">
                  <td className="py-1 pr-2">{CHECK[c.name]}</td>
                  <td className={`py-1 pr-2 ${c.status === "failed" ? "text-bad" : ""}`}>
                    {MARK[c.status]}
                  </td>
                  <td className="py-1 pr-2">{format(c.value)}</td>
                  <td className="py-1">{format(c.tolerance)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-muted">
            These are measurements, not a bound on the error.
          </p>
        </details>
      )}
      {report.notes.length > 0 && (
        <ul className="mt-3 list-disc pl-5 text-xs text-muted">
          {report.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
      {children}
    </section>
  );
}
