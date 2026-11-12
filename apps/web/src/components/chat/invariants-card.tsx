import { Math as Formula } from "@minerva/ui/math";
import type { InvariantReport } from "@/lib/chat-types";
import { StatusPill } from "../result-panel";

/** Constants of motion found by the search, each verified, with the exact family that was searched. */
export function InvariantsCard({ report }: { report: InvariantReport }) {
  return (
    <section
      className="rounded-lg border border-rule bg-raised p-5"
      aria-label="Search for more constants of motion"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-xl">Search for more constants of motion</h3>
        <StatusPill status={report.status} okLabel="Searched and verified" />
      </div>
      {report.lagrangian_latex && <Formula latex={`L = ${report.lagrangian_latex}`} display />}
      <p className="text-sm text-muted">{report.message}</p>

      {report.found.length > 0 && (
        <ul className="mt-3 space-y-2">
          {report.found.map((c, i) => (
            <li key={i} className="rounded border border-rule p-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-medium">{c.name}</span>
                <span
                  className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                    c.verified ? "bg-ok-soft text-ok" : "bg-warn-soft text-warn"
                  }`}
                >
                  {c.verified ? "dC/dt = 0 verified" : "not verified"}
                </span>
              </div>
              <Formula latex={c.expression_latex} display />
              {c.verification && (
                <p className="mt-1 text-xs text-muted">
                  On the equations of motion: {c.verification.numeric_points_agreed} of{" "}
                  {c.verification.numeric_points_checked} sample points give 0
                  {c.verification.symbolic_zero ? "; the derivative simplifies to 0" : ""}.
                </p>
              )}
            </li>
          ))}
        </ul>
      )}

      {report.ansatz && (
        <div className="mt-4">
          <h4 className="text-sm font-medium">What was searched</h4>
          <p className="text-sm text-muted">{report.ansatz}</p>
        </div>
      )}
      {report.known.length > 0 && (
        <p className="mt-2 text-xs text-muted">
          Already known and left out of the count: {report.known.join("; ")}.
        </p>
      )}
      {report.notes.map((n, i) => (
        <p key={i} className="mt-2 text-xs text-muted">
          {n}
        </p>
      ))}
    </section>
  );
}
