"use client";

import type { OdeAnalysis, OdeInput, OdeNumericReport, OdeSolution } from "@/lib/ode-types";
import { OdeAnalysisPanel } from "../ode-analysis-panel";
import { OdeNumericForm, OdeSymbolicForm } from "../ode-forms";
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
  const applicable = (id: string) => report.methods.some((m) => m.id === id && m.applicable);
  return (
    <OdeAnalysisPanel report={report}>
      {ode && meaning && report.status === "ok" && (
        <div className="mt-4 space-y-3">
          {applicable("symbolic_closed_form") && (
            <details className="rounded border border-rule p-3">
              <summary className="cursor-pointer text-sm font-medium">
                Find a closed form{" "}
                <span className="font-normal text-muted">(checked by substituting it back)</span>
              </summary>
              <div className="mt-3">
                <OdeSymbolicForm base={request} meaning={meaning} onRun={ode.symbolic} />
              </div>
            </details>
          )}
          {applicable("numerical_ivp") && (
            <details className="rounded border border-rule p-3">
              <summary className="cursor-pointer text-sm font-medium">
                Solve numerically{" "}
                <span className="font-normal text-muted">(you choose the numbers): show</span>
              </summary>
              <div className="mt-3">
                <OdeNumericForm base={request} meaning={meaning} onRun={ode.numeric} />
              </div>
            </details>
          )}
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
