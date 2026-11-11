import { Math } from "@minerva/ui/math";
import type { LagrangianReport } from "@/lib/chat-types";
import type { EntrySummary } from "@/lib/types";
import { EntryCard } from "../entry-card";
import { StatusPill } from "../result-panel";

/** Equations of motion and verified constants of motion of a Lagrangian. */
export function LagrangianCard({
  report,
  methods,
}: {
  report: LagrangianReport;
  methods: EntrySummary[];
}) {
  return (
    <section className="rounded-lg border border-rule bg-raised p-5" aria-label="Lagrangian">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-xl">Lagrangian</h3>
        <StatusPill status={report.status} okLabel="Analysed and verified" />
      </div>
      {report.lagrangian_latex && <Math latex={`L = ${report.lagrangian_latex}`} display />}
      <p className="text-sm text-muted">{report.message}</p>

      {report.equations.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-medium">Equations of motion (Euler–Lagrange)</h4>
          {report.equations.map((e, i) => (
            <Math key={i} latex={e.latex} display />
          ))}
        </div>
      )}

      {report.momenta.length > 0 && (
        <div className="mt-3">
          <h4 className="text-sm font-medium">Conjugate momenta</h4>
          <div className="flex flex-wrap gap-x-6">
            {report.momenta.map((m, i) => (
              <Math key={i} latex={m.latex} />
            ))}
          </div>
        </div>
      )}

      {report.accelerations.length > 0 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-sm font-medium">
            Accelerations <span className="font-normal text-muted">(solved form): show</span>
          </summary>
          {report.accelerations.map((a, i) => (
            <Math key={i} latex={a.latex} display />
          ))}
        </details>
      )}

      <div className="mt-4">
        <h4 className="text-sm font-medium">Constants of motion</h4>
        {report.conserved.length === 0 ? (
          <p className="mt-1 text-sm text-muted">
            None found among the symmetries Minerva searches for.
          </p>
        ) : (
          <ul className="mt-2 space-y-2">
            {report.conserved.map((c, i) => (
              <li key={i} className="rounded border border-rule p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium">{c.name}</span>
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                      c.verified ? "bg-ok-soft text-ok" : "bg-warn-soft text-warn"
                    }`}
                  >
                    {c.verified ? "dQ/dt = 0 verified" : "not verified"}
                  </span>
                </div>
                <Math latex={c.expression_latex} display />
                <p className="text-xs text-muted">{c.reason}</p>
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
      </div>

      {report.tested.length > 0 && (
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-muted">
          {report.tested.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ul>
      )}
      {report.notes.map((n, i) => (
        <p key={i} className="mt-2 text-xs text-muted">
          {n}
        </p>
      ))}

      {methods.length > 0 && (
        <details className="mt-4">
          <summary className="cursor-pointer text-sm font-medium">
            Methods from the handbook{" "}
            <span className="font-normal text-muted">({methods.length}): show</span>
          </summary>
          <div className="mt-2 space-y-3">
            {methods.map((m) => (
              <EntryCard key={m.id} entry={m} />
            ))}
          </div>
        </details>
      )}
    </section>
  );
}
