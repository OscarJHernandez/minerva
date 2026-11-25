import { CopyMath as Formula } from "@/components/copyable-math";
import type { EffectivePotentialReport } from "@/lib/chat-types";
import { Restrictions, StatusPill } from "../result-panel";
import { Numbered } from "./equation-number";

function points(c: { numeric_points_agreed: number; numeric_points_checked: number } | null) {
  return c ? `${c.numeric_points_agreed}/${c.numeric_points_checked}` : "not run";
}

/**
 * The radial effective potential of geodesics: the radial equation (the card's single result), V_eff and the
 * momenta held fixed as rows. A row number is a name for the formula, not a verification. The normalisation is
 * the user's own number and is shown as such.
 */
export function EffectivePotentialCard({ report }: { report: EffectivePotentialReport }) {
  const ok = report.status === "ok" && report.result_latex;
  const proved = report.verification?.symbolic_zero === true;
  return (
    <section
      className="rounded-lg border border-rule bg-raised p-5"
      aria-label="Effective potential"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-xl">Effective potential</h3>
        <StatusPill
          status={report.status}
          okLabel={proved ? "Derived, proved and checked" : "Derived and checked"}
        />
      </div>
      {report.normalisation_latex && (
        <p className="mt-1 text-sm text-muted">
          Normalisation chosen by you: <Formula latex={report.normalisation_latex} />
        </p>
      )}
      <p className="text-sm text-muted">{report.message}</p>

      {ok && (
        <>
          <div className="mt-4">
            <h4 className="text-sm font-medium">Radial equation</h4>
            <Formula latex={report.result_latex ?? ""} display />
            <Restrictions
              latex={report.restrictions_latex}
              complete={report.restrictions_complete}
            />
          </div>

          {report.veff_latex && (
            <div className="mt-4">
              <h4 className="text-sm font-medium">Effective potential</h4>
              <Numbered k="veff">
                <Formula latex={report.veff_latex} display />
              </Numbered>
            </div>
          )}

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

          {report.fixed.length > 0 && (
            <p className="mt-4 text-xs text-muted" aria-label="The restriction to a plane">
              Restricted to{" "}
              {report.fixed.map((f, i) => (
                <span key={f}>
                  {i > 0 && ", "}
                  <Formula latex={f} />
                </span>
              ))}{" "}
              only after computing that the geodesic equations keep{" "}
              {report.consistency.map((c) => c.coordinate).join(", ")} there (its acceleration is{" "}
              {report.consistency.every((c) => c.symbolic) ? "exactly" : "numerically"} zero).
            </p>
          )}

          <div className="mt-4 text-xs text-muted" aria-label="How it was checked">
            <p>
              Checked numerically against the full geodesic equations: r˙² + V_eff − α p² vanished
              at {points(report.checks.normalisation)} points on g(v, v) = κ, and the radial
              acceleration equalled −V_eff′/2 at {points(report.checks.radial_acceleration)}.
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
