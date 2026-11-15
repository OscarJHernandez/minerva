"use client";

import { CopyMath as Math } from "@/components/copyable-math";
import { useId } from "react";
import type { Status } from "@/lib/types";
import type {
  BoundaryEffect,
  Comparison,
  Need,
  NextStepId,
  SecondFact,
  SecondNumeric,
  VariationChecks,
  VariationReport,
} from "@/lib/variation-types";
import { LinePlot, type PlotSeries } from "./chat/line-plot";
import { StatusPill } from "./result-panel";

const LABEL = "text-xs uppercase tracking-wide text-muted";
const CELL = "px-2 py-1.5 align-top";

const EFFECT_TEXT: Record<BoundaryEffect, string> = {
  removed_by_fixed: "Removed: the perturbation is zero here (declared fixed)",
  natural_condition: "Free: this coefficient must vanish (natural condition)",
  kept_unspecified: "Kept: no condition was declared, nothing is assumed",
  vanishes_identically: "The coefficient is identically zero",
};

const NEXT_TEXT: Record<NextStepId, string> = {
  first_variation: "Derive the first variation with every integration by parts",
  euler_lagrange: "Read off the Euler–Lagrange equations",
  boundary_terms: "Examine the boundary terms",
  declare_conditions: "Declare the endpoint conditions at each end",
  evaluate_action: "Evaluate the action on a trial path",
  verify_variation: "Check the first variation against finite differences",
  solve_stationary: "Look for a stationary path",
  second_variation: "Derive the second variation",
};

const NEED_TEXT: Record<Need, string> = {
  interval: "the interval",
  endpoint_conditions: "the endpoint conditions",
  parameter_values: "values for the parameters",
  trial_path: "a trial path",
  initial_guess: "a starting guess",
};

const FACT_TEXT: Record<SecondFact, string> = {
  not_a_weak_minimum: "Not a weak minimum: a direction of negative second variation was certified.",
  not_a_weak_maximum: "Not a weak maximum: a direction of positive second variation was certified.",
  legendre_violated_for_minimum:
    "The Legendre condition for a minimum is violated at a sampled point.",
  legendre_violated_for_maximum:
    "The Legendre condition for a maximum is violated at a sampled point.",
  no_negative_direction_found:
    "No direction of negative second variation was found among those tried.",
  no_positive_direction_found:
    "No direction of positive second variation was found among those tried.",
};

const ACTION_TITLE: Record<VariationReport["action"], string> = {
  analyze: "Structure of the functional",
  first: "First variation",
  second: "Second variation",
  evaluate: "Action on trial paths",
  verify: "Finite-difference check",
  solve: "Stationary path",
};

/** The status label never says more than the checks did. */
function okLabel(report: VariationReport): string {
  switch (report.action) {
    case "analyze":
      return "Structure read";
    case "first":
    case "second":
      return "Derived, checked numerically";
    case "evaluate":
      return "Evaluated, two quadrature rules agree";
    case "verify":
      return `Compared with finite differences: ${report.verification?.verdict ?? "done"}`;
    case "solve":
      return "Stationary path found, checked";
  }
}

export function num(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
  if (value === 0) return "0";
  const a = abs(value);
  return a >= 1e4 || a < 1e-3 ? value.toExponential(2) : String(Number(value.toPrecision(4)));
}
const abs = (v: number) => (v < 0 ? -v : v);

function flag(value: boolean | null): string {
  return value === null ? "not run" : value ? "passed" : "failed";
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-5 min-w-0" aria-label={title}>
      <h3 className={LABEL}>{title}</h3>
      <div className="mt-1 min-w-0">{children}</div>
    </section>
  );
}

function Formula({ latex, label }: { latex: string; label?: string }) {
  return (
    <div className="overflow-x-auto">
      <Math latex={latex} display label={label} />
    </div>
  );
}

function Notes({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  return (
    <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-muted">
      {items.map((n) => (
        <li key={n}>{n}</li>
      ))}
    </ul>
  );
}

function Table({
  head,
  children,
  label,
}: {
  head: string[];
  children: React.ReactNode;
  label: string;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-max text-left text-sm" aria-label={label}>
        <thead>
          <tr className="border-b border-rule text-xs uppercase tracking-wide text-muted">
            {head.map((h) => (
              <th key={h} scope="col" className={`${CELL} font-normal`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-rule">{children}</tbody>
      </table>
    </div>
  );
}

function ComparisonView({ c }: { c: Comparison }) {
  return (
    <div className="mt-2 text-sm">
      <p>
        <span className="text-muted">
          {c.kind === "bubble" ? "Localised perturbation" : "Perturbation"}
        </span>
        {c.function ? ` of ${c.function}` : ""}: {c.verdict}. Claimed value{" "}
        <code className="font-mono text-xs">{c.analytic}</code>, scale{" "}
        <code className="font-mono text-xs">{c.scale}</code>
        {c.observed_order !== null ? `, observed order ${num(c.observed_order)}` : ""}, Richardson
        relative error {num(c.richardson_rel_error)}.
      </p>
      <Table
        label="Finite-difference comparison"
        head={["Step", "Difference quotient", "Absolute error", "Relative error"]}
      >
        {c.rows.map((r) => (
          <tr key={r.step}>
            <td className={`${CELL} font-mono text-xs`}>{r.step}</td>
            <td className={`${CELL} font-mono text-xs`}>{r.quotient}</td>
            <td className={`${CELL} font-mono text-xs`}>{num(r.abs_error)}</td>
            <td className={`${CELL} font-mono text-xs`}>{num(r.rel_error)}</td>
          </tr>
        ))}
      </Table>
    </div>
  );
}

function Checks({ checks }: { checks: VariationChecks }) {
  const tone =
    checks.deciding === "passed"
      ? "text-ok"
      : checks.deciding === "failed"
        ? "text-bad"
        : "text-warn";
  return (
    <Block title="Numerical evidence">
      <p className="text-sm" data-testid="checks-statement">
        {checks.statement}
      </p>
      <p className="mt-1 text-sm">
        <span className="text-muted">Deciding comparison: </span>
        <span className={`font-medium ${tone}`}>{checks.deciding}</span>
        <span className="text-muted">
          {" "}
          · pointwise check {flag(checks.pointwise_passed)} · integration-by-parts steps checked{" "}
          {checks.stages_checked} ({flag(checks.stages_passed)}) · SymPy {flag(checks.sympy_agrees)}
        </span>
      </p>
      {checks.comparisons.length > 0 && (
        <details className="mt-2">
          <summary className="cursor-pointer text-sm text-muted">
            Comparison tables ({checks.comparisons.length})
          </summary>
          {checks.comparisons.map((c, i) => (
            <ComparisonView key={i} c={c} />
          ))}
        </details>
      )}
    </Block>
  );
}

function SecondFacts({ numeric }: { numeric: SecondNumeric }) {
  return (
    <div className="mt-2 text-sm" data-testid="second-numeric">
      <p>
        <span className="text-muted">First variation on this path: </span>
        {numeric.stationary.replace(/_/g, " ")} (residual {num(numeric.stationary_residual)})
      </p>
      {numeric.legendre && (
        <p>
          <span className="text-muted">Legendre matrix at {numeric.legendre.points} points: </span>
          eigenvalues from {num(numeric.legendre.smallest_eigenvalue)} to{" "}
          {num(numeric.legendre.largest_eigenvalue)}
        </p>
      )}
      {numeric.ritz && (
        <p>
          <span className="text-muted">
            Restricted to {numeric.ritz.dimension} admissible polynomial perturbations:{" "}
          </span>
          {numeric.ritz.negative} negative, {numeric.ritz.zero} zero, {numeric.ritz.positive}{" "}
          positive eigenvalues
        </p>
      )}
      {numeric.directions.length > 0 && (
        <Table label="Directions" head={["Direction", "Second variation", "Sign", "Certified"]}>
          {numeric.directions.map((d) => (
            <tr key={d.label}>
              <td className={CELL}>{d.label}</td>
              <td className={`${CELL} font-mono text-xs`}>{d.second_variation}</td>
              <td className={CELL}>{d.sign}</td>
              <td className={CELL}>{d.certified ? "yes" : "no"}</td>
            </tr>
          ))}
        </Table>
      )}
      {numeric.facts.length > 0 && (
        <ul className="mt-2 list-disc space-y-1 pl-5">
          {numeric.facts.map((f) => (
            <li key={f}>{FACT_TEXT[f]}</li>
          ))}
        </ul>
      )}
      {numeric.not_established.length > 0 && (
        <div className="mt-2">
          <div className={LABEL}>Not established</div>
          <ul className="list-disc space-y-1 pl-5 text-muted">
            {numeric.not_established.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </div>
      )}
      <p className="mt-2 text-muted">{numeric.statement}</p>
    </div>
  );
}

/**
 * One result of the variational calculus, shown as the derivation it is: the functional, the varied
 * functions, the first variation and each integration by parts, the Euler-Lagrange equations, the
 * boundary contributions and what each declared condition does to them, then the numerical evidence.
 * Only a result that passed the engine's own checks is drawn; every other status shows its message.
 * The wording of what a number establishes is the server's (`statement` fields), shown verbatim.
 */
export function VariationPanel({
  report,
  headingId,
}: {
  report: VariationReport;
  /** Unique when several panels are on one page. */
  headingId?: string;
}) {
  const uid = useId();
  const id = headingId ?? `variation-${uid}`;
  const shown = report.status === "ok" || report.status === "unchanged";
  const a = report.analysis;
  const f = report.first;
  const symbolic =
    report.action === "analyze" || report.action === "first" || report.action === "second";

  return (
    <section aria-labelledby={id} className="rounded-lg border border-rule bg-raised p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id={id} className="font-serif text-xl">
          Variational calculus{" "}
          <span className="text-sm text-muted">· {ACTION_TITLE[report.action]}</span>
        </h2>
        <StatusPill status={report.status as Status} okLabel={okLabel(report)} />
      </div>

      {!shown && (
        <p className="mt-3 text-sm text-warn" data-testid="variation-message">
          {report.message}
        </p>
      )}

      {shown && (
        <>
          {report.message && <p className="mt-2 text-sm text-muted">{report.message}</p>}

          {a && (
            <Block title="Functional">
              <Formula latex={a.functional_latex} label="Functional" />
              <ul
                className="mt-1 list-disc space-y-0.5 pl-5 text-sm text-muted"
                aria-label="How the input was read"
              >
                {a.reading.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
              {symbolic && a.unspecified > 0 && (
                <p className="mt-2 text-sm text-warn">
                  {a.unspecified} endpoint condition{a.unspecified === 1 ? " is" : "s are"} not
                  declared; each one stays undecided in the boundary terms below.
                </p>
              )}
            </Block>
          )}

          {f && (
            <>
              <Block title="Varied functions">
                <div className="flex flex-wrap gap-x-6">
                  {f.varied_latex.map((l) => (
                    <Math key={l} latex={l} />
                  ))}
                </div>
              </Block>
              <Block title="First variation of the integrand">
                <Formula latex={f.delta_integrand_latex} label="Variation of the integrand" />
              </Block>
              <Block title="Integration by parts">
                <ol className="space-y-3">
                  {f.stages.map((s) => (
                    <li key={s.index} className="rounded border border-rule p-3 text-sm">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <span className="font-medium">
                          {s.function === null
                            ? "Before any integration by parts"
                            : `Integration by parts ${s.index}: derivative ${s.moved_order} of the perturbation of ${s.function} moved onto the rest`}
                        </span>
                        <span
                          className={
                            s.checked === false ? "text-bad" : s.checked ? "text-ok" : "text-muted"
                          }
                        >
                          {s.checked === null
                            ? s.function === null
                              ? ""
                              : "not checked"
                            : s.checked
                              ? "✓ checked"
                              : "✗ check failed"}
                        </span>
                      </div>
                      <div className={`${LABEL} mt-2`}>Integrand</div>
                      <Formula latex={s.integrand_latex} />
                      <div className={LABEL}>Boundary function so far</div>
                      <Formula latex={s.boundary_latex} />
                      <p className="text-xs text-muted">
                        Needs the path to have derivatives up to order {s.needs_derivatives}.
                      </p>
                    </li>
                  ))}
                </ol>
              </Block>
              <Block title="Euler–Lagrange equations">
                {f.null_lagrangian && (
                  <p className="text-sm text-muted">
                    The integrand is a null Lagrangian: its Euler–Lagrange expressions are
                    identically zero.
                  </p>
                )}
                {f.equations.map((e) => (
                  <div key={e.function}>
                    <div className="text-xs text-muted">
                      for {e.function} (order {e.order})
                    </div>
                    <Formula latex={e.equation_latex} label={`Equation for ${e.function}`} />
                  </div>
                ))}
              </Block>
              <Block title="Boundary contributions">
                <Formula latex={f.boundary_form_latex} label="Boundary form" />
                {f.endpoint_terms_latex.trim() !== "" && f.endpoint_terms_latex.trim() !== "0" && (
                  <div>
                    <div className="text-xs text-muted">Endpoint terms of the action</div>
                    <Formula latex={f.endpoint_terms_latex} />
                  </div>
                )}
                <Table
                  label="Boundary terms"
                  head={[
                    "End",
                    "Function",
                    "Derivative",
                    "Coefficient",
                    "Declared",
                    "Effect",
                    "Natural condition",
                  ]}
                >
                  {f.boundary.map((b) => (
                    <tr key={`${b.end}-${b.function}-${b.order}`}>
                      <td className={CELL}>{b.end}</td>
                      <td className={CELL}>{b.function}</td>
                      <td className={CELL}>
                        <Math latex={b.perturbation_latex} />
                      </td>
                      <td className={CELL}>
                        <Math latex={b.coefficient_latex} />
                      </td>
                      <td className={CELL}>{b.state}</td>
                      <td className={`${CELL} min-w-48 max-w-64`} data-effect={b.effect}>
                        {EFFECT_TEXT[b.effect]}
                      </td>
                      <td className={CELL}>
                        {b.natural_condition_latex ? (
                          <Math latex={b.natural_condition_latex} />
                        ) : (
                          "–"
                        )}
                      </td>
                    </tr>
                  ))}
                </Table>
                <div className={`${LABEL} mt-3`}>Complete first variation</div>
                <Formula latex={f.first_variation_latex} label="First variation" />
              </Block>
              <Block title="Stationarity conditions">
                {f.stationarity_latex.map((l) => (
                  <Formula key={l} latex={l} />
                ))}
                <Notes items={f.notes} />
              </Block>
              <Checks checks={f.checks} />
            </>
          )}

          {report.second && (
            <Block title="Second variation">
              <div className="text-xs text-muted">Convention</div>
              <Formula latex={report.second.convention_latex} />
              <div className="text-xs text-muted">Quadratic form</div>
              <Formula latex={report.second.quadratic_form_latex} label="Second variation" />
              {report.second.endpoint_form_latex.trim() !== "" &&
                report.second.endpoint_form_latex.trim() !== "0" && (
                  <>
                    <div className="text-xs text-muted">Endpoint terms</div>
                    <Formula latex={report.second.endpoint_form_latex} />
                  </>
                )}
              <div className="text-xs text-muted">Legendre matrix</div>
              <Formula latex={report.second.legendre_matrix_latex} />
              <Checks checks={report.second.checks} />
              {report.second.numeric && <SecondFacts numeric={report.second.numeric} />}
            </Block>
          )}

          {report.evaluation && (
            <Block title="Action on the given paths">
              <p className="text-sm">
                Interval{" "}
                <code className="font-mono text-xs">
                  {report.evaluation.lower} to {report.evaluation.upper}
                </code>
                {Object.keys(report.evaluation.parameters).length > 0 && (
                  <>
                    , with{" "}
                    {Object.entries(report.evaluation.parameters)
                      .map(([k, v]) => `${k} = ${v}`)
                      .join(", ")}
                  </>
                )}
              </p>
              <Table
                label="Trial paths"
                head={["Path", "Action", "Integral", "Endpoint terms", "Quadrature error", "Route"]}
              >
                {report.evaluation.trials.map((t) => (
                  <tr key={t.label}>
                    <td className={CELL}>
                      {t.label}
                      {t.parameter_value ? ` (${t.parameter_value})` : ""}
                    </td>
                    <td className={`${CELL} font-mono text-xs`}>{t.action}</td>
                    <td className={`${CELL} font-mono text-xs`}>{t.integral}</td>
                    <td className={`${CELL} font-mono text-xs`}>{t.endpoint_terms}</td>
                    <td className={`${CELL} font-mono text-xs`}>{num(t.quadrature_error)}</td>
                    <td className={CELL}>
                      {t.route}
                      {t.half_data_difference !== null
                        ? `; half the samples change it by ${num(t.half_data_difference)}`
                        : ""}
                    </td>
                  </tr>
                ))}
              </Table>
              {report.evaluation.trials.some((t) => t.conditions.length > 0) && (
                <div className="mt-3 text-sm">
                  <div className={LABEL}>Declared conditions on each path</div>
                  {report.evaluation.trials.map((t) =>
                    t.conditions.length === 0 ? null : (
                      <div key={t.label} className="mt-1">
                        <span className="font-medium">{t.label}</span>
                        <ul className="list-disc pl-5">
                          {t.conditions.map((c) => (
                            <li
                              key={`${c.function}-${c.order}-${c.end}`}
                              className={c.met ? "" : "text-bad"}
                            >
                              {c.met ? "met" : "not met"}: derivative {c.order} of {c.function} at
                              the {c.end} end must be {c.required}; the path gives {c.found}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ),
                  )}
                </div>
              )}
              {report.evaluation.ordering && (
                <div className="mt-3 text-sm">
                  <div className={LABEL}>Ordering of these values only</div>
                  <ol className="list-decimal pl-5">
                    {report.evaluation.ordering.map((o) => (
                      <li key={o}>{o}</li>
                    ))}
                  </ol>
                </div>
              )}
              <Notes items={report.evaluation.notes} />
              <p className="mt-2 text-sm" data-testid="evaluation-statement">
                {report.evaluation.statement}
              </p>
            </Block>
          )}

          {report.verification && (
            <Block title="Finite-difference check of the first variation">
              <p className="text-sm">
                <span className="text-muted">Route: </span>
                {report.verification.route}
                <span className="text-muted"> · verdict: </span>
                <span className="font-medium">{report.verification.verdict}</span>
              </p>
              <Table
                label="Perturbations"
                head={[
                  "Perturbation",
                  "Kind",
                  "Admissible",
                  "Interior part",
                  "Boundary part",
                  "Claimed",
                  "Comparison",
                ]}
              >
                {report.verification.perturbations.map((p) => (
                  <tr key={p.label}>
                    <td className={CELL}>{p.label}</td>
                    <td className={CELL}>{p.kind.replace(/_/g, " ")}</td>
                    <td className={CELL}>
                      {p.admissible === null ? "n/a" : p.admissible ? "yes" : "no"}
                    </td>
                    <td className={`${CELL} font-mono text-xs`}>{p.interior_part}</td>
                    <td className={`${CELL} font-mono text-xs`}>{p.boundary_part}</td>
                    <td className={`${CELL} font-mono text-xs`}>{p.analytic}</td>
                    <td className={CELL}>{p.comparison.verdict}</td>
                  </tr>
                ))}
              </Table>
              <details className="mt-2">
                <summary className="cursor-pointer text-sm text-muted">Comparison tables</summary>
                {report.verification.perturbations.map((p) => (
                  <div key={p.label} className="mt-2">
                    <div className="text-sm font-medium">{p.label}</div>
                    <ComparisonView c={p.comparison} />
                  </div>
                ))}
              </details>
              <Notes items={report.verification.notes} />
              <p className="mt-2 text-sm" data-testid="verification-statement">
                {report.verification.statement}
              </p>
            </Block>
          )}

          {report.solution && (
            <Block title="Stationary path">
              <Table
                label="Methods"
                head={["Method", "Ran", "Converged", "Levels", "Successive differences", "Note"]}
              >
                {report.solution.methods.map((m) => (
                  <tr key={m.method}>
                    <td className={CELL}>{m.method.replace(/_/g, " ")}</td>
                    <td className={CELL}>{m.ran ? "yes" : "no"}</td>
                    <td className={CELL}>{m.converged ? "yes" : "no"}</td>
                    <td className={`${CELL} font-mono text-xs`}>{m.levels.join(", ")}</td>
                    <td className={`${CELL} font-mono text-xs`}>
                      {m.successive_differences.map(num).join(", ")}
                    </td>
                    <td className={CELL}>{m.note}</td>
                  </tr>
                ))}
              </Table>
              <p className="mt-2 text-sm">
                {report.solution.confirmed_by_both
                  ? "Both methods found the path"
                  : "Not confirmed by both methods"}
                {report.solution.agreement !== null
                  ? `; largest relative difference between them ${num(report.solution.agreement)}`
                  : ""}
                {report.solution.distinct_paths ? "; the paths found are distinct" : ""}.
              </p>
              <SolutionPlot report={report} />
              <Table
                label="Paths found"
                head={["Method", "Action", "Quadrature error", "Residual", "Tolerance", "Verdict"]}
              >
                {report.solution.paths.map((p) => (
                  <tr key={p.method}>
                    <td className={CELL}>{p.method.replace(/_/g, " ")}</td>
                    <td className={`${CELL} font-mono text-xs`}>{p.action}</td>
                    <td className={`${CELL} font-mono text-xs`}>
                      {num(p.action_quadrature_error)}
                    </td>
                    <td className={`${CELL} font-mono text-xs`}>{num(p.stationarity.residual)}</td>
                    <td className={`${CELL} font-mono text-xs`}>{num(p.stationarity.tolerance)}</td>
                    <td className={CELL}>{p.stationarity.verdict.replace(/_/g, " ")}</td>
                  </tr>
                ))}
              </Table>
              {report.solution.second && <SecondFacts numeric={report.solution.second} />}
              <Notes items={report.solution.notes} />
              <p className="mt-2 text-sm" data-testid="solution-statement">
                {report.solution.statement}
              </p>
            </Block>
          )}

          {report.notes.length > 0 && <Notes items={report.notes} />}

          {a &&
            (report.action === "analyze" ||
              report.action === "first" ||
              report.action === "second") && <NextSteps report={report} />}
        </>
      )}
    </section>
  );
}

function SolutionPlot({ report }: { report: VariationReport }) {
  const sol = report.solution;
  if (!sol || sol.paths.length === 0) return null;
  const series: PlotSeries[] = [];
  for (const p of sol.paths) {
    for (const [name, values] of Object.entries(p.samples.values)) {
      series.push({
        label: `${name}(${report.analysis?.variable ?? "t"}), ${p.method.replace(/_/g, " ")}`,
        x: p.samples.t,
        y: values,
      });
    }
  }
  return (
    <LinePlot
      series={series}
      xLabel={report.analysis?.variable ?? "t"}
      yLabel="stationary path"
      label="Stationary path found, from the samples returned by the server"
    />
  );
}

function NextSteps({ report }: { report: VariationReport }) {
  const steps = report.analysis?.next_steps ?? [];
  if (steps.length === 0) return null;
  return (
    <Block title="What you can do next">
      <ul className="list-disc space-y-1 pl-5 text-sm">
        {steps.map((s) => (
          <li key={s.id}>
            {NEXT_TEXT[s.id]}
            {s.needs.length > 0 && (
              <span className="text-muted">
                {" "}
                (needs from you: {s.needs.map((n) => NEED_TEXT[n]).join(", ")})
              </span>
            )}
          </li>
        ))}
      </ul>
    </Block>
  );
}
