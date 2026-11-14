import { Math as Formula } from "@minerva/ui/math";
import type { ReactNode } from "react";
import type { OdeSolution, OdeSolutionItem } from "@/lib/ode-types";
import { StatusPill } from "./result-panel";

const KIND = {
  explicit: "explicit",
  implicit: "implicit (a relation between the variables)",
  unevaluated: "contains an integral that could not be evaluated (a representation, not a formula)",
} as const;

function family(item: OdeSolutionItem): string {
  if (item.family === "particular")
    return "A particular solution: every constant is fixed by the conditions.";
  if (item.family === "general") {
    return `The general solution: ${item.constants.length} independent constant${item.constants.length === 1 ? "" : "s"} (${item.constants.join(", ")}), as many as the order.`;
  }
  return "A family of solutions with fewer independent constants than the order: not claimed to be the general solution.";
}

export function formatResidual(value: number | null): string {
  if (value === null) return "n/a";
  if (value === 0) return "0";
  return value.toExponential(1).replace("e+", "e");
}

function Item({
  item,
  onUse,
}: {
  item: OdeSolutionItem;
  onUse?: (text: string, equation: string) => void;
}) {
  const v = item.verification;
  const usable =
    onUse && item.kind === "explicit" && item.rhs_text.length === 1 && item.constants.length === 0;
  return (
    <li className="rounded border border-rule bg-paper p-3" data-kind={item.kind}>
      <ul className="space-y-1 overflow-x-auto">
        {item.equations_latex.map((eq) => (
          <li key={eq}>
            <Formula latex={eq} display />
          </li>
        ))}
      </ul>
      <p className="mt-2 text-sm">
        <span className="text-muted">{KIND[item.kind]}. </span>
        {family(item)}
      </p>
      {item.conditions.length > 0 && (
        <ul className="mt-2 text-sm">
          {item.conditions.map((c) => (
            <li key={c.latex} className={c.satisfied ? "text-ok" : "text-bad"}>
              <Formula latex={c.latex} /> {c.satisfied ? "holds" : "does not hold"}
            </li>
          ))}
        </ul>
      )}
      {item.restrictions_latex.length > 0 && (
        <p className="mt-1 text-sm text-muted">
          Valid only where{" "}
          {item.restrictions_latex.map((r, i) => (
            <span key={r}>
              {i > 0 && ", "}
              <Formula latex={r} />
            </span>
          ))}
          .
        </p>
      )}
      <details className="mt-2 text-xs">
        <summary className="cursor-pointer text-muted">
          How it was checked: {v.points_agreed} of {v.points_checked} points agree
          {v.symbolic_zero ? ", also proved" : ", numerically checked"}
        </summary>
        <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5">
          <dt className="text-muted">Equation as written</dt>
          <dd>
            substituted at {v.points_checked} random points; derivatives taken numerically, not by
            SymPy
          </dd>
          <dt className="text-muted">Largest residual</dt>
          <dd>{formatResidual(v.max_residual)} of the size of the terms</dd>
          <dt className="text-muted">Symbolic zero test</dt>
          <dd>
            {v.symbolic_zero
              ? "the equation simplifies to zero exactly (reported separately)"
              : "not decided: this is a numerical check, not a proof"}
          </dd>
          {v.independent_constants !== null && item.constants.length > 0 && (
            <>
              <dt className="text-muted">Independent constants</dt>
              <dd>{v.independent_constants} (numerical rank)</dd>
            </>
          )}
        </dl>
      </details>
      {usable && (
        <button
          type="button"
          className="mt-2 rounded border border-rule px-2 py-1 text-xs hover:border-accent"
          onClick={() => onUse(item.rhs_text[0] ?? "", item.equations_latex[0] ?? "")}
        >
          Use the right-hand side in Expression mode
        </button>
      )}
    </li>
  );
}

/** Closed-form solutions, each shown only because it was substituted back into the equation. */
export function OdeSolutionPanel({
  report,
  onUse,
  children,
}: {
  report: OdeSolution;
  onUse?: (text: string, equation: string) => void;
  children?: ReactNode;
}) {
  const proved = report.verification?.symbolic_zero === true;
  return (
    <section
      className="rounded-lg border border-rule bg-raised p-4"
      aria-label="Closed-form solution"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-lg">Closed form</h3>
        <StatusPill
          status={report.status}
          okLabel={
            proved ? "Solved, checked numerically and proved" : "Solved, checked numerically"
          }
        />
      </div>
      {report.status !== "ok" && <p className="mt-3 text-sm text-warn">{report.message}</p>}
      {report.solutions.length > 1 && (
        <p className="mt-3 text-sm text-muted">
          {report.solutions.length} solutions were found and all pass the check (for example the two
          signs of a square root).
        </p>
      )}
      {report.solutions.length > 0 && (
        <ul className="mt-3 space-y-3">
          {report.solutions.map((item) => (
            <Item key={item.equations_latex.join("|")} item={item} onUse={onUse} />
          ))}
        </ul>
      )}
      {report.dropped > 0 && (
        <p className="mt-3 text-sm text-warn">
          {report.dropped} candidate{report.dropped === 1 ? "" : "s"} from SymPy failed the check
          and
          {report.dropped === 1 ? " is" : " are"} not shown.
        </p>
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
