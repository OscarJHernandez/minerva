import { Math as Formula } from "@minerva/ui/math";
import type { ReactNode } from "react";
import type { MethodSuggestion, OdeAnalysis, ProblemKind } from "@/lib/ode-types";
import { OdeQualitativePanel } from "./ode-qualitative-panel";
import { StatusPill } from "./result-panel";
import type { PlotSeries } from "./chat/line-plot";

const KIND: Record<ProblemKind, string> = {
  general: "No conditions were given: this asks for the general solution.",
  initial_value: "The conditions make this an initial-value problem.",
  boundary_value: "The conditions are at two points: this is a boundary-value problem.",
  incomplete: "Fewer conditions were given than the order requires.",
  overdetermined: "More conditions were given than the order allows.",
};

const CLASS_TITLE: Record<string, string> = {
  separable: "separable",
  linear_first_order: "linear, first order",
  exact: "exact",
  bernoulli: "Bernoulli",
  homogeneous_degree_zero: "homogeneous of degree zero",
  linear_constant_coefficients: "linear with constant coefficients",
  euler_cauchy: "Euler–Cauchy (equidimensional)",
  second_order_linear: "second-order linear",
};

function Chip({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-full border border-rule bg-sunk px-2.5 py-0.5 text-xs">
      {children}
    </span>
  );
}

function Method({ m, action }: { m: MethodSuggestion; action?: ReactNode }) {
  return (
    <li
      className="rounded border border-rule bg-paper p-3"
      data-method={m.id}
      data-applicable={m.applicable}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm font-medium">
          {m.rank !== null && <span className="mr-2 text-muted">{m.rank}.</span>}
          {m.title}
        </p>
        {action}
      </div>
      <p className="mt-1 text-sm text-muted">{m.reason}</p>
      {m.applicable && (m.requires.length > 0 || m.limitations.length > 0) && (
        <details className="mt-2 text-xs">
          <summary className="cursor-pointer text-muted">What it needs and where it stops</summary>
          {m.requires.length > 0 && (
            <>
              <p className="mt-2 font-medium">Needs</p>
              <ul className="list-disc pl-5 text-muted">
                {m.requires.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
            </>
          )}
          {m.limitations.length > 0 && (
            <>
              <p className="mt-2 font-medium">Limits</p>
              <ul className="list-disc pl-5 text-muted">
                {m.limitations.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
            </>
          )}
        </details>
      )}
    </li>
  );
}

/**
 * How Minerva read a differential equation, its structure, and the approaches that can be tried, in
 * order. "Can be tried" never means "will succeed": each method verifies its own result.
 */
export function OdeAnalysisPanel({
  report,
  actions,
  trajectories,
  children,
}: {
  report: OdeAnalysis;
  /** Numerical trajectories (two states) to draw over a direction field. */
  trajectories?: PlotSeries[];
  /** An element to put next to a method (a button), by method id. */
  actions?: (m: MethodSuggestion) => ReactNode;
  children?: ReactNode;
}) {
  const { interpretation: meaning, structure: s } = report;
  const applicable = report.methods.filter((m) => m.applicable);
  const other = report.methods.filter((m) => !m.applicable);
  return (
    <section
      className="rounded-lg border border-rule bg-raised p-4"
      aria-label="Differential equation analysis"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-lg">Differential equation</h3>
        <StatusPill status={report.status} okLabel="Read and classified" />
      </div>
      {report.status !== "ok" && <p className="mt-3 text-sm text-warn">{report.message}</p>}
      {meaning && (
        <div className="mt-3">
          <p className="text-xs font-medium uppercase tracking-wide text-muted">
            As Minerva read it
          </p>
          <ul className="mt-1 space-y-1 overflow-x-auto">
            {meaning.equations_latex.map((eq) => (
              <li key={eq}>
                <Formula latex={eq} />
              </li>
            ))}
            {meaning.conditions_latex.map((c) => (
              <li key={c}>
                <Formula latex={c} />
              </li>
            ))}
          </ul>
          <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-sm">
            <dt className="text-muted">Unknown{meaning.unknowns.length > 1 ? "s" : ""}</dt>
            <dd>{meaning.unknowns.join(", ")}</dd>
            <dt className="text-muted">Independent variable</dt>
            <dd>{meaning.independent}</dd>
            <dt className="text-muted">Parameters</dt>
            <dd>{meaning.parameters.length ? meaning.parameters.join(", ") : "none"}</dd>
          </dl>
          {meaning.notes.length > 0 && (
            <ul className="mt-2 list-disc pl-5 text-xs text-muted">
              {meaning.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          )}
        </div>
      )}
      {s && (
        <div className="mt-4">
          <p className="text-xs font-medium uppercase tracking-wide text-muted">Structure</p>
          <div className="mt-1 flex flex-wrap gap-2">
            <Chip>{s.kind === "ode" ? "single equation" : `system of ${s.n_equations}`}</Chip>
            <Chip>order {s.order}</Chip>
            <Chip>{s.state_dimension} first-order states</Chip>
            <Chip>{s.linear ? "linear" : "nonlinear"}</Chip>
            {s.linear && <Chip>{s.homogeneous ? "homogeneous" : "forced"}</Chip>}
            {s.constant_coefficients && <Chip>constant coefficients</Chip>}
            <Chip>{s.autonomous ? "autonomous" : "depends on the variable"}</Chip>
            <Chip>
              {s.solved_for_highest ? "solved for the highest derivative" : "not solved for it"}
            </Chip>
          </div>
          {report.classes.length > 0 && (
            <p className="mt-2 text-sm">
              Classes (SymPy&apos;s classification; each solver checks its own result):{" "}
              {report.classes.map((c) => CLASS_TITLE[c] ?? c).join(", ")}.
            </p>
          )}
          {report.solved_latex.length > 0 && (
            <ul className="mt-2 overflow-x-auto">
              {report.solved_latex.map((eq) => (
                <li key={eq}>
                  <Formula latex={eq} />
                </li>
              ))}
            </ul>
          )}
          {report.restrictions_latex.length > 0 && (
            <p className="mt-1 text-sm text-muted">
              Solved form valid where{" "}
              {report.restrictions_latex.map((r, i) => (
                <span key={r}>
                  {i > 0 && ", "}
                  <Formula latex={r} />
                </span>
              ))}
              .
            </p>
          )}
          {report.problem_kind && (
            <p className="mt-2 text-sm">
              {KIND[report.problem_kind]}
              {report.conditions_needed.length > 0 && (
                <span className="text-muted"> Needed: {report.conditions_needed.join(", ")}.</span>
              )}
            </p>
          )}
        </div>
      )}
      {report.qualitative && (
        <OdeQualitativePanel
          qualitative={report.qualitative}
          trajectories={trajectories}
          independent={meaning?.independent ?? "t"}
        />
      )}
      {applicable.length > 0 && (
        <div className="mt-4">
          <p className="text-xs font-medium uppercase tracking-wide text-muted">
            Approaches that can be tried, in order
          </p>
          <ol className="mt-2 space-y-2">
            {applicable.map((m) => (
              <Method key={m.id} m={m} action={actions?.(m)} />
            ))}
          </ol>
          <p className="mt-2 text-xs text-muted">
            “Can be tried” means Minerva will attempt it and check the result, not that it will
            succeed.
          </p>
        </div>
      )}
      {other.length > 0 && (
        <details className="mt-3 text-sm">
          <summary className="cursor-pointer text-muted">
            Not applicable here ({other.length})
          </summary>
          <ul className="mt-2 space-y-2">
            {other.map((m) => (
              <Method key={m.id} m={m} />
            ))}
          </ul>
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
