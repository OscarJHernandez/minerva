"use client";

import { useState } from "react";
import { Math as Formula } from "@minerva/ui/math";
import type { MotionReport, MotionRequest, MotionRun } from "@/lib/chat-types";
import { StatusPill } from "../result-panel";
import { LinePlot, type PlotSeries } from "./line-plot";

function sci(v: number | null): string {
  return v === null ? "-" : v === 0 ? "0" : v.toExponential(1);
}

function runLabel(report: MotionReport, run: MotionRun): string {
  return report.sweep_name && run.parameter_value !== null
    ? `${report.sweep_name} = ${run.parameter_value}`
    : "run";
}

function values(record: Record<string, number>): string {
  const items = Object.entries(record).map(([k, v]) => `${k} = ${v}`);
  return items.length > 0 ? items.join(", ") : "no parameters";
}

/** Verified numerical trajectories, the drift of the constants of motion, and the phase portrait. */
export function MotionCard({ report }: { report: MotionReport; request: MotionRequest }) {
  const verified = report.runs.filter((r) => r.status === "ok");
  const [coordinate, setCoordinate] = useState(0);
  const [runIndex, setRunIndex] = useState(0);
  const sweeping = report.sweep_name !== null;
  const c = clamp(coordinate, report.coordinates.length);
  const driftRun = verified[clamp(runIndex, verified.length)];

  const trajectories: PlotSeries[] = verified.map((r) => ({
    label: runLabel(report, r),
    x: report.t,
    y: r.q[c] ?? [],
  }));
  const phase: PlotSeries[] = verified.map((r) => ({
    label: runLabel(report, r),
    x: r.q[c] ?? [],
    y: r.v[c] ?? [],
  }));
  const drift: PlotSeries[] = (driftRun?.conserved ?? []).map((s) => {
    const top = maximum(s.values.map(Math.abs)) || 1;
    return {
      label: s.name,
      x: report.t,
      y: s.values.map((v) => Math.abs(v - (s.values[0] ?? 0)) / top),
    };
  });
  const name = report.coordinates[c] ?? "q";

  return (
    <section
      className="rounded-lg border border-rule bg-raised p-5"
      aria-label="Numerical solution"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-xl">Numerical solution</h3>
        <StatusPill status={report.status} okLabel="Computed and verified" />
      </div>
      {report.lagrangian_latex && <Formula latex={`L = ${report.lagrangian_latex}`} display />}
      <p className="text-sm text-muted">{report.message}</p>
      <p className="mt-2 text-xs text-muted">
        Values as read: {values(report.parameters)}
        {sweeping && `; ${report.sweep_name} varied`}. Start: {report.coordinates.join(", ")} ={" "}
        {report.initial_positions.join(", ")}; velocities {report.initial_velocities.join(", ")}.
        Time from {report.t_start} to {report.t_end}.
      </p>

      {verified.length > 0 && (
        <>
          {report.coordinates.length > 1 && (
            <label className="mt-3 block text-xs text-muted">
              Coordinate
              <select
                className="ml-2 rounded border border-rule bg-paper px-2 py-1 text-sm"
                value={c}
                onChange={(e) => setCoordinate(Number(e.target.value))}
              >
                {report.coordinates.map((n, i) => (
                  <option key={n} value={i}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
          )}
          <h4 className="mt-4 text-sm font-medium">{name} against time</h4>
          <LinePlot series={trajectories} xLabel="t" yLabel={name} label={`${name} against time`} />
          <h4 className="mt-4 text-sm font-medium">Phase portrait</h4>
          <LinePlot
            series={phase}
            xLabel={name}
            yLabel={`d${name}/dt`}
            label={`Phase portrait of ${name}`}
          />
          <h4 className="mt-4 text-sm font-medium">Drift of the constants of motion</h4>
          {sweeping && verified.length > 1 && (
            <label className="mt-1 block text-xs text-muted">
              Run
              <select
                className="ml-2 rounded border border-rule bg-paper px-2 py-1 text-sm"
                value={clamp(runIndex, verified.length)}
                onChange={(e) => setRunIndex(Number(e.target.value))}
              >
                {verified.map((r, i) => (
                  <option key={i} value={i}>
                    {runLabel(report, r)}
                  </option>
                ))}
              </select>
            </label>
          )}
          {drift.length > 0 ? (
            <>
              <p className="text-xs text-muted">
                |Q(t) − Q(0)| relative to the largest |Q|: how well each verified constant stays
                constant along the computed motion.
              </p>
              <LinePlot
                series={drift}
                xLabel="t"
                yLabel="relative drift"
                yFloor={0}
                label="Drift of the constants of motion"
              />
            </>
          ) : (
            <p className="mt-1 text-sm text-muted">
              No verified constant of motion exists for this Lagrangian (for example, it depends
              explicitly on time), so there is nothing to follow.
            </p>
          )}
        </>
      )}

      <h4 className="mt-4 text-sm font-medium">Verification of each run</h4>
      <ul className="mt-1 space-y-1 text-sm">
        {report.runs.map((r, i) => (
          <li key={i} className="rounded border border-rule px-3 py-2">
            <span className="font-medium">{runLabel(report, r)}</span>{" "}
            <span className={r.status === "ok" ? "text-ok" : "text-warn"}>
              {r.status === "ok" ? "verified" : r.status.replace("_", " ")}
            </span>
            {r.status === "ok" ? (
              <p className="text-xs text-muted">
                Second method (RK4) differs by {sci(r.verification.method_difference)}; integrating
                back misses the start by {sci(r.verification.reversal_error)};{" "}
                {r.verification.drift_checked > 0
                  ? `${r.verification.drift_checked} constant(s) of motion drift by at most ${sci(r.verification.max_drift)}`
                  : "no constant of motion to follow"}
                ; {r.verification.steps} steps (all relative to the size of the motion).
              </p>
            ) : (
              <p className="text-xs text-muted">{r.message} No data is shown for this run.</p>
            )}
          </li>
        ))}
      </ul>
      {report.notes.map((n, i) => (
        <p key={i} className="mt-2 text-xs text-muted">
          {n}
        </p>
      ))}
    </section>
  );
}

function clamp(i: number, length: number): number {
  return Math.max(0, Math.min(i, length - 1));
}
function maximum(values: number[]): number {
  return values.reduce((a, b) => (b > a ? b : a), 0);
}
