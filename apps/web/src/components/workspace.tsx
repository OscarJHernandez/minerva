"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ApiError, api } from "@/lib/api";
import { MAX_DERIVATIVE_ORDER, MAX_SERIES_ORDER } from "@/lib/types";
import type {
  Assumptions,
  DifferentiateResult,
  IntegrateResult,
  HistoryItem,
  IdentitiesResult,
  SeriesResult,
  SimplifyGoal,
  SimplifyResult,
} from "@/lib/types";
import { AssumptionsEditor } from "./assumptions-editor";
import { DerivativePanel } from "./derivative-panel";
import { IntegralPanel } from "./integral-panel";
import { HistoryList } from "./history-list";
import { IdentitiesPanel } from "./identities-panel";
import { MathInput } from "./math-input";
import { ResultPanel } from "./result-panel";
import { SeriesPanel } from "./series-panel";

type Output =
  | { kind: "simplify"; data: SimplifyResult }
  | { kind: "identities"; data: IdentitiesResult }
  | { kind: "differentiate"; data: DifferentiateResult }
  | { kind: "series"; data: SeriesResult }
  | { kind: "integrate"; data: IntegrateResult }
  | null;

type Kind = "simplify" | "identities" | "differentiate" | "integrate" | "series";

const EXAMPLES = [
  { label: "sin² + cos²", latex: "\\sin^2 x + \\cos^2 x" },
  { label: "(x²−1)/(x−1)", latex: "\\frac{x^2-1}{x-1}" },
  { label: "log(xy)", latex: "\\log(xy)" },
  { label: "√(x²)", latex: "\\sqrt{x^2}" },
];

const GOALS: { value: SimplifyGoal; label: string }[] = [
  { value: "simplify", label: "simplify" },
  { value: "expand", label: "expand" },
  { value: "factor", label: "factor" },
  { value: "rewrite", label: "rewrite in terms of" },
];
const FUNCTION_NAME = /^[A-Za-z][A-Za-z0-9]{0,31}$/;

// A finite real number as the server reads it in text syntax: 0, -2, 1/2, 0.5, pi.
const POINT_PATTERN = /^-?(\d+(\.\d+)?(\/\d+)?|pi)$/;
const VARIABLE_PATTERN = /^[A-Za-z][A-Za-z0-9_]{0,31}$/;

export function Workspace() {
  const router = useRouter();
  const [latex, setLatex] = useState("");
  const [assumptions, setAssumptions] = useState<Assumptions>({});
  const [output, setOutput] = useState<Output>(null);
  const [variable, setVariable] = useState("x");
  const [order, setOrder] = useState(1);
  const [point, setPoint] = useState("0");
  const [seriesOrder, setSeriesOrder] = useState(6);
  const [lower, setLower] = useState("");
  const [upper, setUpper] = useState("");
  const [goal, setGoal] = useState<SimplifyGoal>("simplify");
  const [target, setTarget] = useState("sin, cos");
  const [busy, setBusy] = useState<Kind | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  const handleError = useCallback(
    (e: unknown) => {
      if (e instanceof ApiError && e.status === 401) {
        router.push("/login");
        return;
      }
      setError(e instanceof ApiError ? e.message : "The request failed. Try again.");
    },
    [router],
  );

  const refreshHistory = useCallback(async () => {
    try {
      setHistory(await api.history());
    } catch (e) {
      handleError(e);
    }
  }, [handleError]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial data load
    void refreshHistory();
  }, [refreshHistory]);

  const variableValid = VARIABLE_PATTERN.test(variable);
  const pointValid = POINT_PATTERN.test(point);
  const targetNames = target
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  const targetValid =
    goal !== "rewrite" ||
    (targetNames.length >= 1 &&
      targetNames.length <= 8 &&
      targetNames.every((t) => FUNCTION_NAME.test(t)));

  async function run(kind: Kind) {
    if (!latex.trim() || busy) return;
    const oneBound = (lower.trim() === "") !== (upper.trim() === "");
    if (kind === "integrate" && oneBound) {
      setError("Give both bounds for a definite integral, or leave both empty.");
      return;
    }
    if (kind === "simplify" && !targetValid) {
      setError("Name the functions to rewrite in terms of, for example sin, cos.");
      return;
    }
    if ((kind === "differentiate" || kind === "integrate") && !variableValid) {
      setError("Enter the variable to differentiate or integrate by, for example x.");
      return;
    }
    if (kind === "series" && (!variableValid || !pointValid)) {
      setError(
        "Enter the variable (for example x) and a finite expansion point such as 0, 1/2 or pi.",
      );
      return;
    }
    setBusy(kind);
    setError(null);
    try {
      const payload = { input: latex, assumptions };
      if (kind === "simplify") {
        const data = await api.simplify({
          ...payload,
          goal,
          target: goal === "rewrite" ? targetNames : [],
        });
        setOutput({ kind, data });
        setActiveId(data.calculation_id ?? null);
      } else if (kind === "series") {
        const data = await api.series(payload, variable, point, seriesOrder);
        setOutput({ kind, data });
        setActiveId(data.calculation_id ?? null);
      } else if (kind === "differentiate") {
        const data = await api.differentiate(payload, [{ name: variable, order }]);
        setOutput({ kind, data });
        setActiveId(data.calculation_id ?? null);
      } else if (kind === "integrate") {
        const data = await api.integrate(
          payload,
          variable,
          lower.trim() ? { lower: lower.trim(), upper: upper.trim() } : undefined,
        );
        setOutput({ kind, data });
        setActiveId(data.calculation_id ?? null);
      } else {
        const data = await api.identities(payload);
        setOutput({ kind, data });
        setActiveId(data.calculation_id ?? null);
      }
      await refreshHistory();
    } catch (e) {
      handleError(e);
    } finally {
      setBusy(null);
    }
  }

  function open(item: HistoryItem) {
    setLatex(item.input);
    setAssumptions(item.assumptions);
    setActiveId(item.id);
    setError(null);
    if (item.operation === "differentiate") {
      const data = item.result as unknown as DifferentiateResult;
      const first = data.variables?.[0];
      if (first) {
        setVariable(first.name);
        setOrder(first.order);
      }
      setOutput({ kind: "differentiate", data });
      return;
    }
    if (item.operation === "series") {
      const data = item.result as unknown as SeriesResult & { point_requested?: string };
      if (data.variable) setVariable(data.variable);
      if (data.point_requested) setPoint(data.point_requested);
      if (data.order) setSeriesOrder(data.order);
      setOutput({ kind: "series", data });
      return;
    }
    if (item.operation === "integrate") {
      const data = item.result as unknown as IntegrateResult;
      if (data.variable) setVariable(data.variable);
      setLower(data.definite ? (data.lower_latex ?? "") : "");
      setUpper(data.definite ? (data.upper_latex ?? "") : "");
      setOutput({ kind: "integrate", data });
      return;
    }
    if (item.operation === "simplify") setGoal((item.result as SimplifyResult).goal ?? "simplify");
    setOutput(
      item.operation === "simplify"
        ? { kind: "simplify", data: item.result as SimplifyResult }
        : { kind: "identities", data: item.result as IdentitiesResult },
    );
  }

  async function remove(item: HistoryItem) {
    try {
      await api.deleteHistoryItem(item.id);
      if (item.id === activeId) setActiveId(null);
      await refreshHistory();
    } catch (e) {
      handleError(e);
    }
  }

  async function clear() {
    if (!window.confirm("Delete all saved calculations? This cannot be undone.")) return;
    try {
      await api.clearHistory();
      setActiveId(null);
      await refreshHistory();
    } catch (e) {
      handleError(e);
    }
  }

  return (
    <div className="grid gap-8 pb-16 pt-8 lg:grid-cols-[1fr_17rem]">
      <div className="min-w-0 space-y-6">
        <MathInput value={latex} onChange={setLatex} onSubmit={() => run("simplify")} />

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => run("simplify")}
            disabled={!latex.trim() || busy !== null || !targetValid}
            className="rounded bg-accent px-4 py-2 text-sm font-medium text-accent-ink disabled:opacity-50"
          >
            {busy === "simplify" ? "Simplifying…" : "Simplify"}
          </button>
          <label className="flex items-center gap-1 text-xs text-muted">
            goal
            <select
              value={goal}
              onChange={(e) => setGoal(e.target.value as SimplifyGoal)}
              aria-label="Goal of the simplification"
              className="rounded border border-rule bg-raised px-2 py-1.5 text-sm text-ink"
            >
              {GOALS.map((g) => (
                <option key={g.value} value={g.value}>
                  {g.label}
                </option>
              ))}
            </select>
          </label>
          {goal === "rewrite" && (
            <label className="flex items-center gap-1 text-xs text-muted">
              functions
              <input
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                aria-label="Functions to rewrite in terms of, separated by commas"
                aria-invalid={!targetValid}
                maxLength={200}
                className="w-28 rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm text-ink"
              />
            </label>
          )}
          <button
            type="button"
            onClick={() => run("identities")}
            disabled={!latex.trim() || busy !== null}
            className="rounded border border-accent px-4 py-2 text-sm font-medium text-accent disabled:opacity-50"
          >
            {busy === "identities" ? "Searching…" : "Find identities"}
          </button>
          <button
            type="button"
            onClick={() => run("integrate")}
            disabled={!latex.trim() || busy !== null || !variableValid}
            className="rounded border border-accent px-4 py-2 text-sm font-medium text-accent disabled:opacity-50"
          >
            {busy === "integrate" ? "Integrating…" : "Integrate"}
          </button>
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              onClick={() => run("differentiate")}
              disabled={!latex.trim() || busy !== null || !variableValid}
              className="rounded border border-accent px-4 py-2 text-sm font-medium text-accent disabled:opacity-50"
            >
              {busy === "differentiate" ? "Differentiating…" : "Differentiate"}
            </button>
            <label className="flex items-center gap-1 text-xs text-muted">
              by
              <input
                value={variable}
                onChange={(e) => setVariable(e.target.value.trim())}
                aria-label="Variable to differentiate or integrate by"
                maxLength={32}
                className="w-14 rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm text-ink"
              />
            </label>
            <label className="flex items-center gap-1 text-xs text-muted">
              from
              <input
                value={lower}
                onChange={(e) => setLower(e.target.value)}
                aria-label="Lower bound of a definite integral (optional)"
                placeholder="lower"
                maxLength={200}
                className="w-16 rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm text-ink"
              />
            </label>
            <label className="flex items-center gap-1 text-xs text-muted">
              to
              <input
                value={upper}
                onChange={(e) => setUpper(e.target.value)}
                aria-label="Upper bound of a definite integral (optional)"
                placeholder="upper"
                maxLength={200}
                className="w-16 rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm text-ink"
              />
            </label>
            <label className="flex items-center gap-1 text-xs text-muted">
              order
              <input
                type="number"
                min={1}
                max={MAX_DERIVATIVE_ORDER}
                value={order}
                onChange={(e) =>
                  setOrder(Math.min(MAX_DERIVATIVE_ORDER, Math.max(1, Number(e.target.value) || 1)))
                }
                aria-label="Order of the derivative"
                className="w-16 rounded border border-rule bg-raised px-2 py-1.5 text-sm text-ink"
              />
            </label>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => run("series")}
              disabled={!latex.trim() || busy !== null || !variableValid || !pointValid}
              className="rounded border border-accent px-4 py-2 text-sm font-medium text-accent disabled:opacity-50"
            >
              {busy === "series" ? "Expanding…" : "Series"}
            </button>
            <label className="flex items-center gap-1 text-xs text-muted">
              about
              <input
                value={point}
                onChange={(e) => setPoint(e.target.value.trim())}
                aria-label="Point to expand about"
                maxLength={64}
                className="w-14 rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm text-ink"
              />
            </label>
            <label className="flex items-center gap-1 text-xs text-muted">
              to order
              <input
                type="number"
                min={1}
                max={MAX_SERIES_ORDER}
                value={seriesOrder}
                onChange={(e) =>
                  setSeriesOrder(
                    Math.min(MAX_SERIES_ORDER, Math.max(1, Number(e.target.value) || 1)),
                  )
                }
                aria-label="Order of the series"
                className="w-16 rounded border border-rule bg-raised px-2 py-1.5 text-sm text-ink"
              />
            </label>
          </div>
          <span className="ml-2 hidden text-xs text-muted sm:inline">Try:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              type="button"
              onClick={() => setLatex(ex.latex)}
              className="rounded-full border border-rule px-2.5 py-1 text-xs text-muted hover:bg-sunk"
            >
              {ex.label}
            </button>
          ))}
        </div>

        <AssumptionsEditor value={assumptions} onChange={setAssumptions} />

        {error && (
          <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
            {error}
          </p>
        )}

        {output?.kind === "simplify" && <ResultPanel result={output.data} />}
        {output?.kind === "identities" && <IdentitiesPanel result={output.data} />}
        {output?.kind === "differentiate" && <DerivativePanel result={output.data} />}
        {output?.kind === "series" && <SeriesPanel result={output.data} />}
        {output?.kind === "integrate" && <IntegralPanel result={output.data} />}
        {!output && (
          <p className="text-sm text-muted">
            Enter an expression, then simplify it, differentiate it, find an antiderivative, expand
            it in a series or look up identities. Results are checked before they are shown; if
            Minerva cannot verify or simplify something, it says so.
          </p>
        )}
      </div>

      <aside className="lg:border-l lg:border-rule lg:pl-6">
        <HistoryList
          items={history}
          activeId={activeId}
          onOpen={open}
          onDelete={remove}
          onClear={clear}
        />
      </aside>
    </div>
  );
}
