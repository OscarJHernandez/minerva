import { CopyMath as Formula } from "@/components/copyable-math";
import type { RouthReport } from "@/lib/chat-types";
import { Restrictions, StatusPill } from "../result-panel";
import { Numbered } from "./equation-number";

function points(checks: { numeric_points_agreed: number; numeric_points_checked: number }[]) {
  return checks.map((c) => `${c.numeric_points_agreed}/${c.numeric_points_checked}`).join(", ");
}

/**
 * Routh reduction: the Routhian R (the card's single result), the momenta held fixed as symbols and the
 * velocities of the removed coordinates. A row number is a name for the formula, not a verification.
 */
export function RouthCard({ report }: { report: RouthReport }) {
  const ok = report.status === "ok" && report.result_latex;
  return (
    <section className="rounded-lg border border-rule bg-raised p-5" aria-label="Routh reduction">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-xl">Routh reduction</h3>
        <StatusPill
          status={report.status}
          okLabel={report.checks.symbolic ? "Reduced, proved and checked" : "Reduced and checked"}
        />
      </div>
      {report.lagrangian_latex && <Formula latex={`L = ${report.lagrangian_latex}`} display />}
      <p className="text-sm text-muted">{report.message}</p>
      {report.reading && report.reading.normalizations.length > 0 && (
        <p className="mt-1 text-xs text-muted" aria-label="How the input was read">
          Read as:{" "}
          {report.reading.normalizations.map((n, i) => (
            <span key={i}>
              {i > 0 && "; "}
              <code>{n.source}</code> → <code>{n.target}</code>
              {n.note ? ` (${n.note})` : ""}
            </span>
          ))}
        </p>
      )}

      {ok && (
        <>
          <div className="mt-4">
            <h4 className="text-sm font-medium">
              Routhian: a Lagrangian for {report.remaining.join(", ")}
            </h4>
            <Formula latex={`R = ${report.result_latex}`} display />
            <Restrictions
              latex={report.restrictions_latex}
              complete={report.restrictions_complete}
            />
          </div>

          <div className="mt-4">
            <h4 className="text-sm font-medium">Held fixed</h4>
            {report.held_fixed.map((h) => (
              <div key={h.coordinate}>
                <Numbered k={`held:${h.coordinate}`}>
                  <Formula latex={h.latex} display />
                </Numbered>
                <p className="text-xs text-muted">{h.reason}.</p>
              </div>
            ))}
          </div>

          <div className="mt-4">
            <h4 className="text-sm font-medium">Velocities of the removed coordinates</h4>
            {report.reconstruction.map((r) => (
              <Numbered key={r.coordinate} k={`velocity:${r.coordinate}`}>
                <Formula latex={r.latex} display />
              </Numbered>
            ))}
          </div>

          <div className="mt-4 text-xs text-muted" aria-label="How it was checked">
            <p>
              Checked numerically on motions of the original Lagrangian: the Euler–Lagrange residual
              of R vanished at {points(report.checks.euler_lagrange)} sample points, and −∂R/∂p and
              the velocity formula returned the sampled velocity at{" "}
              {points(report.checks.reconstruction)}
              {report.checks.definition && (
                <>; R equalled L − Σ p ċ at {points([report.checks.definition])}</>
              )}
              .
            </p>
            <p className="mt-1">
              Symbolic check{report.checks.symbolic ? " (proof)" : ""}:{" "}
              {report.checks.symbolic_note || "not run"}.
            </p>
            <p className="mt-1">{report.convention}</p>
          </div>
        </>
      )}
    </section>
  );
}
