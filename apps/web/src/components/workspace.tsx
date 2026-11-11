"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ApiError, api } from "@/lib/api";
import type { Assumptions, HistoryItem, IdentitiesResult, SimplifyResult } from "@/lib/types";
import { AssumptionsEditor } from "./assumptions-editor";
import { HistoryList } from "./history-list";
import { IdentitiesPanel } from "./identities-panel";
import { MathInput } from "./math-input";
import { ResultPanel } from "./result-panel";

type Output =
  | { kind: "simplify"; data: SimplifyResult }
  | { kind: "identities"; data: IdentitiesResult }
  | null;

const EXAMPLES = [
  { label: "sin² + cos²", latex: "\\sin^2 x + \\cos^2 x" },
  { label: "(x²−1)/(x−1)", latex: "\\frac{x^2-1}{x-1}" },
  { label: "log(xy)", latex: "\\log(xy)" },
  { label: "√(x²)", latex: "\\sqrt{x^2}" },
];

export function Workspace() {
  const router = useRouter();
  const [latex, setLatex] = useState("");
  const [assumptions, setAssumptions] = useState<Assumptions>({});
  const [output, setOutput] = useState<Output>(null);
  const [busy, setBusy] = useState<"simplify" | "identities" | null>(null);
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

  async function run(kind: "simplify" | "identities") {
    if (!latex.trim() || busy) return;
    setBusy(kind);
    setError(null);
    try {
      const payload = { input: latex, assumptions };
      if (kind === "simplify") {
        const data = await api.simplify(payload);
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
            disabled={!latex.trim() || busy !== null}
            className="rounded bg-accent px-4 py-2 text-sm font-medium text-accent-ink disabled:opacity-50"
          >
            {busy === "simplify" ? "Simplifying…" : "Simplify"}
          </button>
          <button
            type="button"
            onClick={() => run("identities")}
            disabled={!latex.trim() || busy !== null}
            className="rounded border border-accent px-4 py-2 text-sm font-medium text-accent disabled:opacity-50"
          >
            {busy === "identities" ? "Searching…" : "Find identities"}
          </button>
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
        {!output && (
          <p className="text-sm text-muted">
            Enter an expression, then simplify it or look up identities. Results are verified before
            they are shown; if Minerva cannot verify or simplify something, it says so.
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
