import type { MotionReport, MotionRun } from "@/lib/chat-types";

const t = [0, 1, 2, 3, 4];

export function okRun(value: number | null = null): MotionRun {
  return {
    status: "ok",
    parameter_value: value,
    message: "Verified",
    q: [[1, 0.5, -0.4, -1, 0.2]],
    v: [[0, -1, -1.5, 0.3, 1.8]],
    conserved: [
      {
        name: "Energy function",
        kind: "energy",
        expression_latex: "h = x",
        values: [2, 2, 2.000001, 2, 1.999999],
        relative_drift: 5e-7,
      },
    ],
    verification: {
      method_difference: 1.7e-9,
      reversal_error: 4.3e-9,
      max_drift: 5e-7,
      drift_checked: 1,
      steps: 468,
      passed: true,
      notes: [],
    },
  };
}

export function failedRun(value: number | null = null): MotionRun {
  return {
    status: "not_verified",
    parameter_value: value,
    message: "Not verified: the Runge-Kutta cross-check differs by 4.09e+00 (relative).",
    q: [],
    v: [],
    conserved: [],
    verification: {
      method_difference: 4.09,
      reversal_error: 0.55,
      max_drift: null,
      drift_checked: 0,
      steps: 7503,
      passed: false,
      notes: [],
    },
  };
}

export function report(runs: MotionRun[], sweep: string | null = null): MotionReport {
  return {
    status: runs.some((r) => r.status === "ok") ? "ok" : "not_verified",
    lagrangian_latex: "\\frac{m \\dot{x}^{2}}{2}",
    coordinates: ["x"],
    accelerations: ["\\ddot{x} = - \\frac{k x}{m}"],
    parameters: { m: 1 },
    initial_positions: [1],
    initial_velocities: [0],
    t_start: 0,
    t_end: 4,
    sweep_name: sweep,
    t,
    runs,
    notes: [],
    message: `${runs.filter((r) => r.status === "ok").length} of ${runs.length} run(s) verified.`,
    duration_ms: 5,
  };
}
