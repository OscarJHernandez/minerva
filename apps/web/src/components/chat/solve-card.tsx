import { Math } from "@minerva/ui/math";
import type { SolveReport } from "@/lib/chat-types";
import type { EntrySummary } from "@/lib/types";
import { EntryCard } from "../entry-card";
import { StatusPill } from "../result-panel";

const RADICALS = {
  yes: { label: "Formula in radicals exists", tone: "bg-ok-soft text-ok" },
  no: { label: "No formula in radicals", tone: "bg-warn-soft text-warn" },
  undetermined: { label: "Solvability not decided", tone: "bg-sunk text-muted" },
} as const;

/** Result of the polynomial solver: factors, what is known about formulas, verified roots, methods. */
export function SolveCard({ report, methods }: { report: SolveReport; methods: EntrySummary[] }) {
  const v = report.verification;
  return (
    <section className="rounded-lg border border-rule bg-raised p-5" aria-label="Solve">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-xl">Solve</h3>
        <StatusPill status={report.status} />
      </div>
      {report.equation_latex && <Math latex={report.equation_latex} display />}
      <p className="text-sm text-muted">{report.message}</p>

      {report.factors.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-medium">Factors over the rationals</h4>
          <ul className="mt-2 space-y-2">
            {report.factors.map((f, i) => (
              <li key={i} className="rounded border border-rule p-3">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <Math latex={f.factor_latex} />
                  {f.multiplicity > 1 && (
                    <span className="text-xs text-muted">multiplicity {f.multiplicity}</span>
                  )}
                  <span className="text-xs text-muted">degree {f.degree}</span>
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs font-medium ${RADICALS[f.radicals].tone}`}
                    title={f.radicals_reason}
                  >
                    {RADICALS[f.radicals].label}
                  </span>
                </div>
                <p className="mt-1 text-xs text-muted">{f.radicals_reason}</p>
                {f.exact_roots_latex.length > 0 && (
                  <p className="mt-1 text-sm">
                    Exact roots:{" "}
                    {f.exact_roots_latex.map((r, j) => (
                      <span key={j} className="mr-3">
                        <Math latex={r} />
                      </span>
                    ))}
                  </p>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {report.roots.length > 0 && (
        <div className="mt-4 overflow-x-auto">
          <h4 className="text-sm font-medium">Roots</h4>
          <table className="mt-2 w-full text-left text-sm">
            <thead className="text-xs text-muted">
              <tr>
                <th className="pb-1 pr-3 font-normal">#</th>
                <th className="pb-1 pr-3 font-normal">Numerical value</th>
                <th className="pb-1 pr-3 font-normal">Type</th>
                <th className="pb-1 font-normal">Exact</th>
              </tr>
            </thead>
            <tbody>
              {report.roots.map((r, i) => (
                <tr key={i} className="border-t border-rule">
                  <td className="py-1 pr-3 text-muted">{i + 1}</td>
                  <td className="py-1 pr-3 font-mono text-xs">{r.numeric}</td>
                  <td className="py-1 pr-3">{r.real ? "real" : "complex"}</td>
                  <td className="py-1">{r.exact_latex ? <Math latex={r.exact_latex} /> : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {v && (
        <p className="mt-3 text-sm">
          <span className="text-muted">Checked: </span>
          roots substituted back {v.roots_substituted_back ? "✓" : "✗"} · root count equals degree{" "}
          {v.count_matches_degree ? "✓" : "✗"} · Vieta&rsquo;s relations {v.vieta_holds ? "✓" : "✗"}
          {v.exact_roots_match_numeric !== null &&
            ` · exact roots agree with numerical ${v.exact_roots_match_numeric ? "✓" : "✗"}`}
        </p>
      )}
      {report.notes.length > 0 && (
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-muted">
          {report.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}

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
