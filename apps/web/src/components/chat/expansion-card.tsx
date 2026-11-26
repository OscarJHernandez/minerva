import { CopyMath as Math } from "@/components/copyable-math";
import type { ExpansionReport } from "@/lib/chat-types";
import { canonicalModes } from "@/lib/expansion-modes";
import { StatusPill } from "../result-panel";
import { Numbered } from "./equation-number";

const SIGN = {
  positive: { label: "ω² > 0 (follows from the declared assumptions)", tone: "bg-ok-soft text-ok" },
  negative: {
    label: "ω² < 0 (follows from the declared assumptions; linearised equations only)",
    tone: "bg-warn-soft text-warn",
  },
  zero: { label: "ω² = 0", tone: "bg-sunk text-muted" },
  unknown: { label: "Sign not decided by the declared assumptions", tone: "bg-sunk text-muted" },
} as const;

/** Equations of motion expanded about an equilibrium; shown only if every expansion was verified. */
export function ExpansionCard({ report }: { report: ExpansionReport }) {
  const shown = report.status === "ok";
  const eq = report.equilibrium_verification;
  return (
    <section
      className="rounded-lg border border-rule bg-raised p-5"
      aria-label="Expansion about an equilibrium"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-xl">Expansion about an equilibrium</h3>
        <StatusPill status={report.status} okLabel="Expanded and verified" />
      </div>
      <p className="mt-1 text-sm text-muted">{report.message}</p>

      {report.equilibrium.length > 0 && (
        <div className="mt-3 text-sm">
          <p>
            Equilibrium:{" "}
            {eq && (
              <span className="text-xs text-muted">
                {eq.passed ? "(accelerations vanish there: checked)" : "(not an equilibrium)"}
              </span>
            )}
          </p>
          {report.equilibrium.map((e) => (
            <Numbered key={e.name} k={`equil:${e.name}`}>
              <Math latex={`${e.name_latex || e.name} = ${e.latex}`} />
            </Numbered>
          ))}
        </div>
      )}

      {!shown && (report.accelerations_at_point?.length ?? 0) > 0 && (
        <div className="mt-3 text-sm">
          <p>
            Accelerations there at zero velocity{" "}
            <span className="text-xs text-muted">(from the checked equations of motion)</span>
          </p>
          <ul className="mt-1 space-y-1">
            {report.accelerations_at_point?.map((a) => (
              <li key={a.name} className="overflow-x-auto">
                <Math latex={`\\ddot{${a.name_latex || a.name}} = ${a.latex}`} />
              </li>
            ))}
          </ul>
        </div>
      )}

      {shown && report.equations.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-medium">Accelerations to degree {report.order}</h4>
          <ul className="mt-2 space-y-2">
            {report.equations.map((e) => (
              <li key={e.coordinate} className="overflow-x-auto rounded border border-rule p-3">
                <Numbered k={`eq:${e.coordinate}`}>
                  <Math latex={e.full_latex} display />
                </Numbered>
                <p className="text-xs text-muted">
                  {e.verification?.passed
                    ? "Coefficients agree with a Cauchy-integral estimate and the remainder falls at the claimed order."
                    : "Not verified."}
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}

      {shown && report.stiffness_latex && (
        <div className="mt-4">
          <h4 className="text-sm font-medium">Linear part</h4>
          <div className="overflow-x-auto">
            <Math
              latex={`\\ddot{\\delta q} = A\\,\\delta q${report.velocity_latex ? " + B\\,\\dot{\\delta q}" : ""},\\quad A = ${report.stiffness_latex}${report.velocity_latex ? `,\\quad B = ${report.velocity_latex}` : ""}`}
              display
            />
          </div>
        </div>
      )}

      {shown && report.characteristic_latex && (
        <div className="mt-2 overflow-x-auto">
          <h4 className="text-sm font-medium">Characteristic polynomial</h4>
          <Numbered k="char">
            <Math latex={report.characteristic_latex} display />
          </Numbered>
        </div>
      )}

      {shown && report.modes.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-medium">Normal modes</h4>
          <ul className="mt-2 space-y-2">
            {canonicalModes(report.modes).map((m, i) => (
              <li key={i} className="rounded border border-rule p-3">
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
                  <Numbered k={`mode:${i}:omega2`}>
                    <Math latex={m.omega_squared_latex} />
                  </Numbered>
                  {m.omega_latex && (
                    <Numbered k={`mode:${i}:omega`}>
                      <Math latex={m.omega_latex} />
                    </Numbered>
                  )}
                  {m.eigenvector_latex && (
                    <Numbered k={`mode:${i}:vector`}>
                      <Math latex={`v = ${m.eigenvector_latex}`} />
                    </Numbered>
                  )}
                </div>
                <span
                  className={`mt-1 inline-block rounded-full px-2 py-0.5 text-xs font-medium ${SIGN[m.sign].tone}`}
                >
                  {SIGN[m.sign].label}
                </span>
                <span className="ml-2 text-xs text-muted">
                  {m.verified ? "root checked by substitution" : "not verified"}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {shown && (
        <p className="mt-4 text-xs text-muted">
          This is a local expansion: it describes motion near the equilibrium and says nothing about
          large motion.
        </p>
      )}
      {report.notes.map((n) => (
        <p key={n} className="mt-2 text-sm text-warn">
          {n}
        </p>
      ))}
    </section>
  );
}
