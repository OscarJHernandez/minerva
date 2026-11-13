import { Math as Formula } from "@minerva/ui/math";
import type { DiscoveredConstant, DiscoveryReport } from "@/lib/chat-types";
import { StatusPill } from "../result-panel";

const KIND: Record<DiscoveredConstant["kind"], { label: string; tone: string }> = {
  known: { label: "Already known", tone: "bg-raised text-muted border border-rule" },
  new: { label: "Newly found", tone: "bg-ok-soft text-ok" },
};

const INDEPENDENCE: Record<DiscoveredConstant["independence"], string> = {
  independent: "Functionally independent of the earlier ones (proved)",
  dependent: "Functionally dependent on the earlier ones (proved)",
  unresolved: "Dependence on the earlier ones not established",
};

/**
 * Conserved quantities found by the adaptive search: each verified on the equations of motion, whether it
 * was already known, how many are functionally independent (with what was proved), and the exact families
 * that were and were not searched. Finding nothing is reported as "within the explored families".
 */
export function DiscoveryCard({ report }: { report: DiscoveryReport }) {
  const fresh = report.constants.filter((c) => c.kind === "new");
  const searched = report.tiers.filter((t) => t.outcome === "searched");
  const ind = report.independence;
  const symbol = report.system === "hamiltonian" ? "H" : "L";
  return (
    <section
      className="rounded-lg border border-rule bg-raised p-5"
      aria-label="Conserved quantities"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-xl">Conserved quantities</h3>
        <StatusPill status={report.status} okLabel="Searched and verified" />
      </div>
      {report.function_latex && <Formula latex={`${symbol} = ${report.function_latex}`} display />}
      <p className="text-sm text-muted">{report.message}</p>
      {report.status === "ok" && fresh.length === 0 && (
        <p className="mt-2 text-sm">
          No additional constant was found <strong>within the explored families</strong>. That does
          not mean the system has none.
        </p>
      )}
      {report.angles.length > 0 && (
        <p className="mt-2 text-xs text-muted">
          Searched with cos and sin of: {report.angles.join(", ")} (declared, detected, or cyclic).
        </p>
      )}

      {report.constants.length > 0 && (
        <ul className="mt-3 space-y-2">
          {report.constants.map((c) => (
            <li key={c.id} className="rounded border border-rule p-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-medium">{c.name}</span>
                <span
                  className={`rounded-full px-2 py-0.5 text-xs font-medium ${KIND[c.kind].tone}`}
                >
                  {KIND[c.kind].label}
                </span>
                <span
                  className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                    c.verified ? "bg-ok-soft text-ok" : "bg-warn-soft text-warn"
                  }`}
                >
                  {c.verified ? "dC/dt = 0 verified" : "not verified"}
                </span>
              </div>
              <Formula latex={c.expression_latex} display />
              <p className="text-xs text-muted">
                {c.id === report.constants[0]?.id && c.independence === "independent"
                  ? "Functionally independent (proved)"
                  : INDEPENDENCE[c.independence]}
                {c.tier !== null ? `; found in family ${c.tier + 1}` : ""}.
                {c.verification
                  ? ` On the equations of motion ${c.verification.numeric_points_agreed} of ${c.verification.numeric_points_checked} sample points give 0${
                      c.verification.symbolic_zero ? "; the derivative simplifies to 0" : ""
                    }.`
                  : ""}
              </p>
            </li>
          ))}
        </ul>
      )}

      {ind && ind.constants > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-medium">Functional independence</h4>
          <p className="text-sm text-muted">
            {ind.rank_proved
              ? `${ind.rank_lower} of the ${ind.constants} are functionally independent (proved by an exact Jacobian rank and the relations below).`
              : `At least ${ind.rank_lower} and at most ${ind.rank_upper} are functionally independent; only the lower bound is proved.`}
            {ind.max_possible !== null ? ` No more than ${ind.max_possible} are possible.` : ""}
          </p>
          {ind.relations.map((r, i) => (
            <div key={i} className="mt-1">
              <Formula latex={r.latex} display />
            </div>
          ))}
          {ind.relations.length > 0 && (
            <p className="text-xs text-muted">
              Exact polynomial relations among the constants above (identities in the phase
              variables).
            </p>
          )}
          {ind.notes.map((n, i) => (
            <p key={i} className="mt-1 text-xs text-muted">
              {n}
            </p>
          ))}
        </div>
      )}

      <details className="mt-4">
        <summary className="cursor-pointer text-sm font-medium">
          What was searched ({searched.length} {searched.length === 1 ? "family" : "families"})
        </summary>
        <ol className="mt-2 list-decimal space-y-1 pl-5 text-xs text-muted">
          {searched.map((t) => (
            <li key={t.index}>
              {t.description}. {t.unknowns} unknown coefficients, {t.equations} equations, nullspace{" "}
              {t.nullity}; {t.accepted} new
              {t.rejected ? `, ${t.rejected} rejected` : ""} ({t.duration_ms} ms).
            </li>
          ))}
        </ol>
        {report.unexplored.length > 0 && (
          <>
            <h4 className="mt-3 text-sm font-medium">Not searched</h4>
            <ul className="mt-1 list-disc space-y-1 pl-5 text-xs text-muted">
              {report.unexplored.map((u, i) => (
                <li key={i}>{u}</li>
              ))}
            </ul>
          </>
        )}
      </details>

      {report.notes.map((n, i) => (
        <p key={i} className="mt-2 text-xs text-muted">
          {n}
        </p>
      ))}
      <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-muted">
        {report.limitations.map((l, i) => (
          <li key={i}>{l}</li>
        ))}
      </ul>
    </section>
  );
}
