import { Math as Formula } from "@minerva/ui/math";
import type { Classification, OdeQualitative } from "@/lib/ode-types";
import { Numbered } from "./chat/equation-number";
import { PhasePlot, type Marker } from "./chat/phase-plot";
import type { PlotSeries } from "./chat/line-plot";

const TONE: Record<Classification, "ok" | "warn" | "muted"> = {
  stable_node: "ok",
  stable_spiral: "ok",
  unstable_node: "warn",
  unstable_spiral: "warn",
  saddle: "warn",
  centre: "muted",
  non_hyperbolic: "muted",
  undecided: "muted",
};

const BADGE = {
  ok: "bg-ok-soft text-ok",
  warn: "bg-warn-soft text-warn",
  muted: "bg-sunk text-muted",
};

const NAME: Record<Classification, string> = {
  stable_node: "sink (node)",
  stable_spiral: "sink (spiral)",
  unstable_node: "source (node)",
  unstable_spiral: "source (spiral)",
  saddle: "saddle",
  centre: "linear centre",
  non_hyperbolic: "not decided by the linearisation",
  undecided: "undecided",
};

/**
 * Equilibria with their linearisation. Every statement here is local and from the linearisation: nothing
 * on this panel says that a system is stable or unstable as a whole.
 */
export function OdeQualitativePanel({
  qualitative: q,
  trajectories,
  independent,
}: {
  qualitative: OdeQualitative;
  /** Numerical trajectories (two states) to draw over the field. */
  trajectories?: PlotSeries[];
  independent: string;
}) {
  if (q.status !== "ok") {
    return (
      <div className="mt-4" data-qualitative="refused">
        <p className="text-xs font-medium uppercase tracking-wide text-muted">
          Equilibria and local stability
        </p>
        <p className="mt-1 text-sm text-warn">{q.message}</p>
      </div>
    );
  }
  const markers: Marker[] = q.equilibria
    .filter((e) => e.point_numeric.length === 2)
    .map((e) => ({
      x: e.point_numeric[0] ?? 0,
      y: e.point_numeric[1] ?? 0,
      label: `${e.point_latex.map((p) => p.replace(/\\/g, "")).join(", ")}: ${NAME[e.classification]}`,
      tone: TONE[e.classification],
    }));
  const states = q.states_latex;
  return (
    <div className="mt-4" data-qualitative="ok">
      <p className="text-xs font-medium uppercase tracking-wide text-muted">
        Equilibria and local stability
      </p>
      <ul className="mt-1 overflow-x-auto">
        {q.field_latex.map((f, i) => (
          <li key={f}>
            <Numbered k={`field:${i}`}>
              <Formula latex={f} />
            </Numbered>
          </li>
        ))}
      </ul>
      {q.equilibria.length === 0 ? (
        <p className="mt-2 text-sm text-muted">No equilibrium was found.</p>
      ) : (
        <ul className="mt-2 space-y-2">
          {q.equilibria.map((e) => (
            <li
              key={e.point_latex.join("|")}
              className="rounded border border-rule bg-paper p-3"
              data-classification={e.classification}
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm">
                  {e.point_latex.map((p, i) => (
                    <span key={p}>
                      {i > 0 && ", "}
                      <Formula latex={p} />
                    </span>
                  ))}
                </span>
                <span
                  className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${BADGE[TONE[e.classification]]}`}
                >
                  {NAME[e.classification]}
                </span>
              </div>
              <p className="mt-1 text-sm">{e.classification_text}.</p>
              <details className="mt-2 text-xs">
                <summary className="cursor-pointer text-muted">Jacobian and eigenvalues</summary>
                <div className="mt-2 overflow-x-auto">
                  <Formula latex={e.jacobian_latex} />
                </div>
                {e.eigenvalues_latex.length > 0 && (
                  <p className="mt-1">
                    Exact eigenvalues:{" "}
                    {e.eigenvalues_latex.map((l, i) => (
                      <span key={l}>
                        {i > 0 && ", "}
                        <Formula latex={l} />
                      </span>
                    ))}
                  </p>
                )}
                {e.eigenvalues_numeric.length > 0 && (
                  <p className="mt-1">Numerical eigenvalues: {e.eigenvalues_numeric.join(", ")}</p>
                )}
                {e.characteristic_latex && (
                  <p className="mt-1 overflow-x-auto">
                    Characteristic polynomial: <Formula latex={e.characteristic_latex} />
                  </p>
                )}
              </details>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-2 text-xs text-muted">
        {q.complete
          ? "These are all of the equilibria (the field is polynomial)."
          : "These are the equilibria found; there may be others."}{" "}
        Each was confirmed by substituting it back, and the Jacobian was checked against numerical
        differences of the field.
      </p>
      <p className="mt-1 text-sm text-warn">{q.local_note}</p>
      {q.direction_field && states.length === 2 && (
        <PhasePlot
          field={q.direction_field}
          markers={markers}
          trajectories={trajectories}
          xLabel={states[0]?.replace(/\\/g, "") ?? independent}
          yLabel={states[1]?.replace(/\\/g, "") ?? ""}
          label={`Phase plane: ${states[1]?.replace(/\\/g, "")} against ${states[0]?.replace(/\\/g, "")}`}
        />
      )}
      {q.notes.length > 0 && (
        <ul className="mt-2 list-disc pl-5 text-xs text-muted">
          {q.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
