import { Math } from "@minerva/ui/math";
import type { DimensionalReport } from "@/lib/chat-types";
import type { EntrySummary } from "@/lib/types";
import { EntryCard } from "../entry-card";
import { StatusPill } from "../result-panel";

const CLOSURE: Record<string, string> = {
  determined: "Fixed by dimensions, up to a constant",
  up_to_function: "Fixed up to a function of the other groups",
  missing: "Quantity missing",
};

/** Dimensionless groups, what they fix about the unknown, and what is missing when nothing does. */
export function DimensionalCard({
  report,
  methods,
}: {
  report: DimensionalReport;
  methods: EntrySummary[];
}) {
  const missing = report.missing;
  return (
    <section
      className="rounded-lg border border-rule bg-raised p-5"
      aria-label="Dimensional analysis"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-xl">Dimensional analysis</h3>
        <StatusPill status={report.status} okLabel="Analysed and checked" />
      </div>
      <p className="mt-1 text-sm text-muted">{report.message}</p>

      {report.quantities.length > 0 && (
        <div className="mt-4 overflow-x-auto">
          <h4 className="text-sm font-medium">How the units were read</h4>
          <table className="mt-1 w-full text-left text-sm">
            <thead className="text-xs uppercase tracking-wide text-muted">
              <tr>
                <th className="py-1 pr-3 font-medium">Quantity</th>
                <th className="py-1 pr-3 font-medium">Given</th>
                <th className="py-1 font-medium">Base units</th>
              </tr>
            </thead>
            <tbody>
              {report.quantities.map((q) => (
                <tr key={q.name} className="border-t border-rule">
                  <td className="py-1 pr-3">
                    <Math latex={q.latex} />
                    {q.name === report.target && (
                      <span className="ml-2 rounded-full bg-accent-soft px-2 py-0.5 text-xs text-accent">
                        unknown
                      </span>
                    )}
                    {q.meaning && <span className="ml-2 text-xs text-muted">{q.meaning}</span>}
                  </td>
                  <td className="py-1 pr-3 text-muted">{q.given}</td>
                  <td className="py-1">
                    <Math latex={q.dimension_latex} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {report.groups.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-medium">
            Dimensionless groups{" "}
            <span className="font-normal text-muted">
              ({report.groups.length} = {report.quantities.length} quantities − rank {report.rank})
            </span>
          </h4>
          <ul className="mt-2 space-y-2">
            {report.groups.map((g) => (
              <li
                key={g.label}
                className="flex flex-wrap items-center gap-x-3 rounded border border-rule px-3"
              >
                <Math latex={`${g.label} = ${g.latex}`} display />
                {g.dimensionless_verified && (
                  <span className="rounded-full bg-ok-soft px-2 py-0.5 text-xs font-medium text-ok">
                    dimensionless: checked
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {report.closure && report.closure !== "none" && (
        <div className="mt-4">
          <h4 className="text-sm font-medium">{CLOSURE[report.closure]}</h4>
          {report.relation_latex && <Math latex={report.relation_latex} display />}
          {report.cannot_enter.length > 0 && (
            <p className="text-xs text-muted">
              Cannot enter by dimensions alone: {report.cannot_enter.join(", ")}.
            </p>
          )}
        </div>
      )}

      {missing && (
        <div
          className="mt-4 rounded border border-warn p-3"
          role="group"
          aria-label="Missing quantity"
        >
          <h4 className="text-sm font-medium text-warn">
            A quantity is missing: it must carry the dimension{" "}
            <Math latex={missing.missing_dimension_latex} />
          </h4>
          <p className="mt-1 text-sm text-muted">{missing.explanation}</p>
          <p className="mt-1 text-xs text-muted">
            {report.target} has dimension <Math latex={missing.target_dimension_latex} />; base
            units that cannot be balanced: {missing.unbalanced.join(", ")}.
          </p>
          {missing.suggestions.length > 0 ? (
            <div className="mt-2">
              <h5 className="text-sm font-medium">Handbook constants that would close it</h5>
              <ul className="mt-1 space-y-1">
                {missing.suggestions.map((s) => (
                  <li key={s.ids.join("+")} className="text-sm">
                    <span className="text-muted">{s.ids.join(" and ")}</span>
                    {s.relation_latex && <Math latex={s.relation_latex} display />}
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="mt-2 text-sm text-muted">
              No single handbook constant, and no pair, supplies it.
            </p>
          )}
        </div>
      )}

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
