import { Math } from "@minerva/ui/math";
import type { UnitsCheckReport } from "@/lib/chat-types";
import type { EntrySummary } from "@/lib/types";
import { EntryCard } from "../entry-card";
import { StatusPill } from "../result-panel";

/** Dimensional consistency of one equation, and the factor that is missing when it fails. */
export function UnitsCard({
  report,
  methods,
}: {
  report: UnitsCheckReport;
  methods: EntrySummary[];
}) {
  return (
    <section className="rounded-lg border border-rule bg-raised p-5" aria-label="Unit check">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-xl">Unit check</h3>
        {report.consistent === null ? (
          <StatusPill status={report.status} />
        ) : (
          <span
            className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${
              report.consistent ? "bg-ok-soft text-ok" : "bg-warn-soft text-warn"
            }`}
          >
            {report.consistent ? "Dimensionally consistent" : "Not dimensionally consistent"}
          </span>
        )}
      </div>
      {report.equation_latex && <Math latex={report.equation_latex} display />}
      <p className="text-sm text-muted">{report.message}</p>

      {report.sides.length > 0 && (
        <ul className="mt-3 space-y-1 text-sm">
          {report.sides.map((s, i) => (
            <li key={i}>
              <Math latex={s.latex} /> <span className="text-muted">has dimension</span>{" "}
              <Math latex={s.dimension_latex} />
            </li>
          ))}
        </ul>
      )}

      {report.missing_factor_latex && (
        <p className="mt-3 rounded border border-warn p-3 text-sm">
          A factor with dimension <Math latex={report.missing_factor_latex} /> is missing on the
          right-hand side (or the left-hand side has too much of it).
        </p>
      )}

      {report.problem_terms.length > 0 && (
        <ul className="mt-3 space-y-1 text-sm">
          {report.problem_terms.map((t, i) => (
            <li key={i}>
              <Math latex={t.latex} /> <span className="text-muted">is</span>{" "}
              <Math latex={t.dimension_latex} />
            </li>
          ))}
        </ul>
      )}

      {report.quantities.length > 0 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-sm font-medium">
            How the units were read{" "}
            <span className="font-normal text-muted">({report.quantities.length}): show</span>
          </summary>
          <ul className="mt-2 space-y-1 text-sm">
            {report.quantities.map((q) => (
              <li key={q.name}>
                <Math latex={q.latex} /> <span className="text-muted">{q.given} =</span>{" "}
                <Math latex={q.dimension_latex} />
              </li>
            ))}
          </ul>
        </details>
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
