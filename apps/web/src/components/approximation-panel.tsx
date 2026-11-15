"use client";

import { useId } from "react";
import {
  METHOD_LABEL,
  WARNING_TEXT,
  type ApproxCandidate,
  type ApproxCheck,
  type ApproximationMethod,
  type ApproximationReport,
} from "@/lib/approximation-types";
import { LinePlot, type PlotSeries } from "./chat/line-plot";
import { CopyMath as Formula } from "./copyable-math";
import { StatusPill } from "./result-panel";

const LABEL = "text-xs uppercase tracking-wide text-muted";
const CELL = "px-2 py-1.5 align-top";

const EVIDENCE: Record<ApproximationReport["evidence"], string> = {
  series_checked: "Series checked numerically",
  numeric_estimate: "Checked numerically",
  bound_established: "Bound established",
  none: "Approximated",
};

function methodLabel(method: string): string {
  return METHOD_LABEL[method as ApproximationMethod] ?? method;
}

function mark(c: ApproxCheck): { text: string; tone: string } {
  if (c.passed === true) return { text: "passed", tone: "text-ok" };
  if (c.passed === false) return { text: "failed", tone: "text-bad" };
  return { text: "undecided", tone: "text-warn" };
}

function cost(c: ApproxCandidate["complexity"]): string {
  const parts: string[] = [];
  if (c.numerator_degree !== null && c.denominator_degree !== null) {
    parts.push(`[${c.numerator_degree}/${c.denominator_degree}]`);
  } else if (c.degree !== null) parts.push(`degree ${c.degree}`);
  if (c.modes !== null) parts.push(`${c.modes} modes`);
  if (c.segments !== null) parts.push(`${c.segments} segments`);
  if (c.coefficients !== null) parts.push(`${c.coefficients} coefficients`);
  return parts.join(", ") || "–";
}

/** log10 of an absolute error, floored so that an exact point stays on the chart. */
function log10(v: number): number {
  return Math.log10(Math.max(v, 1e-18));
}

function plotSeries(report: ApproximationReport): PlotSeries[] {
  const curves: { label: string; points: { x: number; error: number }[] }[] =
    report.candidates.length > 1
      ? report.candidates
          .filter((c) => c.error_curve.length > 1)
          .slice(0, 6)
          .map((c) => ({ label: c.label, points: c.error_curve }))
      : [];
  if (curves.length === 0 && report.error_curve.length > 1) {
    curves.push({ label: "absolute error", points: report.error_curve });
  }
  return curves.map((c) => ({
    label: c.label,
    x: c.points.map((p) => p.x),
    y: c.points.map((p) => log10(p.error)),
  }));
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section aria-label={title} className="mt-4">
      <h4 className={LABEL}>{title}</h4>
      {children}
    </section>
  );
}

function Candidates({ items }: { items: ApproxCandidate[] }) {
  return (
    <div className="mt-1 overflow-x-auto">
      <table className="min-w-max text-sm" aria-label="Candidates">
        <thead className="text-left text-xs text-muted">
          <tr>
            <th className={CELL}>Candidate</th>
            <th className={CELL}>Cost</th>
            <th className={CELL}>Largest sampled error</th>
            <th className={CELL}>Tolerance (sampled)</th>
            <th className={CELL}>Notes</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-rule">
          {items.map((c, i) => (
            <tr key={`${c.label}-${i}`} className={c.chosen ? "bg-ok-soft font-medium" : ""}>
              <td className={CELL}>
                {c.label}
                {c.chosen && <span className="ml-1 text-xs text-ok">chosen</span>}
              </td>
              <td className={CELL}>{cost(c.complexity)}</td>
              <td className={`${CELL} font-mono text-xs`}>
                {c.status === "ok" ? (c.error?.max_abs ?? "–") : c.reason ? c.reason : c.status}
              </td>
              <td className={CELL}>
                {c.tolerance_met === null ? "–" : c.tolerance_met ? "met" : "not met"}
              </td>
              <td className={`${CELL} text-xs text-warn`}>
                {c.warnings.map((w) => WARNING_TEXT[w] ?? w).join(" ")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * An approximation: the formula, the order of what was dropped, what it was checked against, the
 * error as a measurement on sampled points (never a bound), and every warning in words. Nothing is
 * shown as a result unless the engine's checks passed; a refusal shows only its reason and questions.
 */
export function ApproximationPanel({ report }: { report: ApproximationReport }) {
  const id = useId();
  const shown = report.status === "ok" || report.status === "unchanged";
  const headline = report.result_latex ?? report.statement_latex;
  const series = shown ? plotSeries(report) : [];
  return (
    <section
      aria-labelledby={id}
      className="rounded-lg border border-rule bg-raised p-5"
      data-testid="approximation"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 id={id} className="font-serif text-xl">
          Approximation <span className="text-sm text-muted">· {methodLabel(report.method)}</span>
        </h3>
        <StatusPill status={report.status} okLabel={EVIDENCE[report.evidence]} />
      </div>
      <p className="mt-1 text-sm text-muted">{report.message}</p>

      {!shown && report.questions.length > 0 && (
        <div className="mt-3 rounded bg-warn-soft px-3 py-2 text-sm text-warn" role="status">
          <strong>Needed from you:</strong>
          <ul className="ml-4 list-disc">
            {report.questions.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>
        </div>
      )}

      {shown && (
        <>
          {report.input_latex && (
            <Block title="Input as read">
              <Formula latex={report.input_latex} display />
            </Block>
          )}
          {report.domain_latex && (
            <p className="mt-2 text-sm text-muted">
              Valid for <Formula latex={report.domain_latex} />
            </p>
          )}
          {headline && (
            <Block title="Result">
              <div data-testid="approximation-result">
                <Formula
                  latex={
                    report.result_latex && report.remainder_latex
                      ? `${report.result_latex} + ${report.remainder_latex}`
                      : headline
                  }
                  display
                  className="text-xl"
                  label="Result"
                />
              </div>
              {report.result_latex && report.statement_latex && (
                <p className="text-sm text-muted">
                  where <Formula latex={report.statement_latex} />
                </p>
              )}
            </Block>
          )}
          {report.tolerance_met !== null && (
            <p
              className={`mt-2 rounded px-3 py-2 text-sm ${report.tolerance_met ? "bg-ok-soft text-ok" : "bg-warn-soft text-warn"}`}
              role="status"
            >
              {report.tolerance_met
                ? "The tolerance is met on the sampled points: numerical support, not a guarantee."
                : "The requested tolerance is NOT met by this approximation (on the sampled points)."}
            </p>
          )}
          {report.warnings.length > 0 && (
            <ul className="mt-2 space-y-1 text-sm text-warn" aria-label="Warnings">
              {report.warnings.map((w) => (
                <li key={w}>⚠ {WARNING_TEXT[w] ?? w}</li>
              ))}
            </ul>
          )}
          {report.extras.length > 0 && (
            <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
              {report.extras.map((e, i) => (
                <div key={i} className="contents">
                  <dt className="text-muted">{e.label}</dt>
                  <dd className="overflow-x-auto">
                    <Formula latex={e.latex} />
                  </dd>
                </div>
              ))}
            </dl>
          )}
          {report.orders.length > 0 && report.orders.length <= 40 && (
            <Block title="Terms">
              <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-sm">
                {report.orders.map((o, i) => (
                  <li key={i}>
                    <Formula latex={o.term_latex} />
                  </li>
                ))}
              </ul>
            </Block>
          )}
          {report.candidates.length > 0 && (
            <Block title="Comparison">
              <Candidates items={report.candidates} />
            </Block>
          )}
          {report.error && report.candidates.length === 0 && (
            <Block title="Measured error">
              <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                <dt className="text-muted">Largest</dt>
                <dd className="font-mono text-xs">{report.error.max_abs ?? "–"}</dd>
                <dt className="text-muted">Root mean square</dt>
                <dd className="font-mono text-xs">{report.error.rms ?? "–"}</dd>
                <dt className="text-muted">Largest relative</dt>
                <dd className="font-mono text-xs">{report.error.max_rel ?? "–"}</dd>
              </dl>
              <p className="mt-1 text-xs text-muted">
                Measured on {report.error.samples} sampled points: a measurement, not a bound.
                {report.error.relative_note ? ` ${report.error.relative_note}.` : ""}
              </p>
            </Block>
          )}
          {report.measured_errors.length > 0 && (
            <Block title="Measured truncation error">
              <table className="mt-1 text-sm" aria-label="Measured truncation error">
                <thead className="text-left text-xs text-muted">
                  <tr>
                    <th className={CELL}>Distance from the point</th>
                    <th className={CELL}>Absolute</th>
                    <th className={CELL}>Relative</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-rule font-mono text-xs">
                  {report.measured_errors.map((m) => (
                    <tr key={m.distance}>
                      <td className={CELL}>{m.distance}</td>
                      <td className={CELL}>{m.absolute_error}</td>
                      <td className={CELL}>{m.relative_error ?? "–"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-1 text-xs text-muted">
                Measurements of this expression at these distances, not a bound.
              </p>
            </Block>
          )}
          {series.length > 0 && (
            <Block title="Error over the interval">
              <LinePlot
                series={series}
                xLabel={report.variable ?? "x"}
                yLabel="log₁₀ |error|"
                label="Absolute error of the approximation on the interval, sampled"
              />
            </Block>
          )}
          {report.checks.length > 0 && (
            <Block title="Checks">
              <ul className="mt-1 space-y-1 text-sm" aria-label="Checks">
                {report.checks.map((c, i) => {
                  const m = mark(c);
                  return (
                    <li key={i}>
                      <span className={m.tone}>{m.text}</span> · {c.name}
                      {c.points > 0 && <span className="text-muted"> ({c.points})</span>}
                      {c.detail && <span className="block text-xs text-muted">{c.detail}</span>}
                    </li>
                  );
                })}
              </ul>
            </Block>
          )}
          {report.assumptions_stated.map((a, i) => (
            <p key={i} className="mt-2 text-xs text-muted">
              {a}
            </p>
          ))}
          {report.notes.map((n, i) => (
            <p key={i} className="mt-2 text-xs text-muted">
              {n}
            </p>
          ))}
          {report.suggestions.length > 0 && (
            <details className="mt-3 text-xs text-muted">
              <summary className="cursor-pointer">Methods considered</summary>
              <ul className="mt-1 space-y-0.5">
                {report.suggestions.map((s) => (
                  <li key={s.method}>
                    {methodLabel(s.method)}:{" "}
                    {s.tried ? "tried" : s.applies ? "applies" : "not used"} – {s.why}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}
    </section>
  );
}
