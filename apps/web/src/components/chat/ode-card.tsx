"use client";

import type { ReactNode } from "react";
import type { OdeAnalysis, OdeInput, OdeNumericReport, OdeSolution } from "@/lib/ode-types";
import { OdeAnalysisPanel } from "../ode-analysis-panel";
import {
  OdeBoundaryForm,
  OdeNumericForm,
  OdeQualitativeForm,
  OdeSolveForm,
  OdeSymbolicForm,
} from "../ode-forms";
import { OdeNumericPanel } from "../ode-numeric-panel";
import { OdeSolutionPanel } from "../ode-solution-panel";
import { useMotionRunner } from "./motion-runner";

/**
 * The analysis of a differential equation, with the two things the user can do next. The model never
 * enters the numbers: the forms post them straight to the tool.
 */
export function OdeAnalysisCard({ report, request }: { report: OdeAnalysis; request: OdeInput }) {
  const runner = useMotionRunner();
  const meaning = report.interpretation;
  const ode = runner?.ode;
  const applicable = (...ids: string[]) =>
    report.methods.some((m) => ids.includes(m.id) && m.applicable);
  const sections: { id: string; show: boolean; title: string; note: string; body: ReactNode }[] =
    ode && meaning && report.status === "ok"
      ? [
          {
            id: "closed",
            show: applicable("symbolic_closed_form"),
            title: "Find a closed form",
            note: "(checked by substituting it back)",
            body: <OdeSymbolicForm base={request} meaning={meaning} onRun={ode.symbolic} />,
          },
          {
            id: "series",
            show: applicable("power_series"),
            title: "Build a series about a point",
            note: "(local; you give the initial values and the order)",
            body: (
              <OdeSolveForm base={request} meaning={meaning} method="series" onRun={ode.symbolic} />
            ),
          },
          {
            id: "perturbation",
            show: applicable("regular_perturbation", "poincare_lindstedt"),
            title: "Expand in a small parameter",
            note: "(you name the parameter)",
            body: (
              <OdeSolveForm
                base={request}
                meaning={meaning}
                method="perturbation"
                onRun={ode.symbolic}
              />
            ),
          },
          {
            id: "ivp",
            show: applicable("numerical_ivp"),
            title: "Solve numerically",
            note: "(you choose the numbers): show",
            body: <OdeNumericForm base={request} meaning={meaning} onRun={ode.numeric} />,
          },
          {
            id: "qualitative",
            show: applicable("equilibrium_analysis", "phase_plane"),
            title: "Equilibria and local stability",
            note: "(local, from the linearisation)",
            body: <OdeQualitativeForm base={request} meaning={meaning} onRun={ode.qualitative} />,
          },
          {
            id: "bvp",
            show: applicable("numerical_bvp"),
            title: "Solve the boundary-value problem",
            note: "(you give the conditions)",
            body: <OdeBoundaryForm base={request} meaning={meaning} onRun={ode.numeric} />,
          },
        ]
      : [];
  return (
    <OdeAnalysisPanel report={report}>
      {sections.some((x) => x.show) && (
        <div className="mt-4 space-y-3">
          {sections
            .filter((x) => x.show)
            .map((x) => (
              <details key={x.id} className="rounded border border-rule p-3">
                <summary className="cursor-pointer text-sm font-medium">
                  {x.title} <span className="font-normal text-muted">{x.note}</span>
                </summary>
                <div className="mt-3">{x.body}</div>
              </details>
            ))}
        </div>
      )}
    </OdeAnalysisPanel>
  );
}

export function OdeSolutionCard({ report }: { report: OdeSolution; request: OdeInput }) {
  return <OdeSolutionPanel report={report} />;
}

export function OdeNumericCard({ report }: { report: OdeNumericReport; request: OdeInput }) {
  return <OdeNumericPanel report={report} />;
}
