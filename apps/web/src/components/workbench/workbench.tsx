"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ApiError, api, type ComputeInput } from "@/lib/api";
import {
  DEFAULT_SETTINGS,
  continuation,
  validateSettings,
  withRealVariable,
  type Settings,
  type StepOperation,
} from "@/lib/derivation";
import type {
  Assumptions,
  DerivationSummary,
  GeometryRequest,
  HistoryItem,
  IdentitiesResult,
  IdentityMatch,
} from "@/lib/types";
import { AssumptionsEditor } from "../assumptions-editor";
import { DerivationView } from "./derivation-view";
import { ExpressionEditor, type EditorFormat } from "./expression-editor";
import { GeometryWorkbench } from "./geometry-workbench";
import { IdentityPicker } from "./identity-picker";
import { OperationBar } from "./operation-bar";
import { SavedPanel } from "./saved-panel";

type Call = (c: ComputeInput) => Promise<{ calculation_id?: string | null }>;

function Section({
  id,
  title,
  open,
  onToggle,
  children,
}: {
  id: string;
  title: string;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <section
      aria-labelledby={`${id}-heading`}
      className="border-t border-rule pt-3 first:border-t-0 first:pt-0"
    >
      <h2 id={`${id}-heading`} className="text-sm font-medium">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          aria-controls={`${id}-body`}
          className="flex w-full items-center justify-between py-1 text-left"
        >
          {title}
          <span aria-hidden="true" className="text-xs text-muted">
            {open ? "−" : "+"}
          </span>
        </button>
      </h2>
      {open && (
        <div id={`${id}-body`} className="mt-2">
          {children}
        </div>
      )}
    </section>
  );
}

/**
 * A deterministic scratchpad: operations and handbook identities applied to an expression, one step
 * at a time, each step verified by the server and saved to the account. Nothing here calls a model.
 */
export function Workbench() {
  const router = useRouter();
  const [format, setFormat] = useState<EditorFormat>("latex");
  const [latex, setLatex] = useState("");
  const [text, setText] = useState("");
  const [assumptions, setAssumptions] = useState<Assumptions>({});
  const [derivationId, setDerivationId] = useState<string | null>(null);
  const [steps, setSteps] = useState<HistoryItem[]>([]);
  const [fromId, setFromId] = useState<string | null>(null);
  const [selected, setSelected] = useState<StepOperation | null>(null);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lookup, setLookup] = useState<{ key: string; result: IdentitiesResult } | null>(null);
  const [saved, setSaved] = useState<DerivationSummary[]>([]);
  const [earlier, setEarlier] = useState<HistoryItem[]>([]);
  const [focusSignal, setFocusSignal] = useState(0);
  const [open, setOpen] = useState({ identities: true, assumptions: true, saved: false });
  const [mode, setMode] = useState<"expression" | "geometry">("expression");
  const [geometryRestore, setGeometryRestore] = useState<{
    request: GeometryRequest;
    nonce: number;
  } | null>(null);

  const value = format === "latex" ? latex : text;
  const key = JSON.stringify([value, format, assumptions]);
  const fromStep = steps.find((s) => s.id === fromId) ?? null;
  const fromResult = fromStep ? continuation(fromStep) : null;
  const intact = fromResult !== null && format === "text" && text === fromResult.text;
  const parentId = intact && fromStep ? fromStep.id : undefined;
  const lookupFresh = lookup?.key === key ? lookup.result : null;

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

  const refreshSaved = useCallback(async () => {
    try {
      const [derivations, history] = await Promise.all([api.derivations(), api.history()]);
      setSaved(derivations);
      setEarlier(history.filter((h) => !h.derivation_id));
    } catch (e) {
      handleError(e);
    }
  }, [handleError]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial data load
    void refreshSaved();
    // Phones start with the side sections folded so the derivation is not pushed off screen.
    if (window.matchMedia?.("(min-width: 1024px)").matches === false) {
      setOpen({ identities: false, assumptions: false, saved: false });
    }
  }, [refreshSaved]);

  function continueFrom(item: HistoryItem, edit: boolean) {
    const next = continuation(item);
    if (!next) return;
    setFormat("text");
    setText(next.text);
    setFromId(item.id);
    setAssumptions(item.assumptions); // the result is valid under the assumptions it was made with
    setError(null);
    if (edit) setFocusSignal((n) => n + 1);
  }

  /** Run `call` on the current expression and add its result as the next step. */
  async function addStep(busyKey: string, call: Call) {
    if (!value.trim() || busy) return;
    setBusy(busyKey);
    setError(null);
    try {
      let id = derivationId;
      if (!id) {
        id = (await api.createDerivation()).id;
        setDerivationId(id);
      }
      const response = await call({
        input: value,
        format,
        assumptions,
        derivation_id: id,
        parent_id: parentId,
      });
      const derivation = await api.derivation(id);
      setSteps(derivation.steps);
      const added = derivation.steps.find((s) => s.id === response.calculation_id);
      if (added) continueFrom(added, false);
      setLookup(null);
      void refreshSaved();
    } catch (e) {
      handleError(e);
    } finally {
      setBusy(null);
    }
  }

  function run(operation: StepOperation) {
    const problem = validateSettings(operation, settings);
    if (problem) {
      setError(problem);
      return;
    }
    const by = settings.variable;
    if (operation === "simplify") void addStep(operation, (c) => api.simplify(c));
    else if (operation === "differentiate")
      void addStep(operation, (c) => api.differentiate(c, [{ name: by, order: settings.order }]));
    else if (operation === "integrate")
      void addStep(operation, (c) =>
        api.integrate(
          c,
          by,
          settings.lower.trim()
            ? { lower: settings.lower.trim(), upper: settings.upper.trim() }
            : undefined,
        ),
      );
    else if (operation === "distribution")
      void addStep(operation, (c) =>
        api.distribution(
          { ...c, assumptions: withRealVariable(c.assumptions, by) },
          {
            action: settings.distAction,
            variable: by,
            lower: settings.distAction === "integrate" ? settings.lower.trim() : null,
            upper: settings.distAction === "integrate" ? settings.upper.trim() : null,
            endpoint:
              settings.distAction === "integrate" && settings.endpoint ? settings.endpoint : null,
          },
        ),
      );
    else void addStep(operation, (c) => api.series(c, by, settings.point, settings.seriesOrder));
  }

  async function find() {
    if (!value.trim() || busy) return;
    setBusy("identities");
    setError(null);
    try {
      const result = await api.identities({ input: value, format, assumptions, save: false });
      setLookup({ key, result });
      setOpen((o) => ({ ...o, identities: true }));
    } catch (e) {
      handleError(e);
    } finally {
      setBusy(null);
    }
  }

  function apply(match: IdentityMatch) {
    void addStep(match.entry.id, (c) => api.applyIdentity(c, match.entry.id));
  }

  function reset() {
    setDerivationId(null);
    setSteps([]);
    setFromId(null);
    setLookup(null);
    setError(null);
    setFormat("latex");
    setLatex("");
    setText("");
  }

  async function openDerivation(id: string) {
    try {
      const derivation = await api.derivation(id);
      reset();
      setDerivationId(id);
      setSteps(derivation.steps);
      const last = [...derivation.steps].reverse().find((s) => continuation(s) !== null);
      if (last) continueFrom(last, false);
    } catch (e) {
      handleError(e);
    }
  }

  async function deleteDerivation(id: string) {
    if (!window.confirm("Delete this derivation and its steps? This cannot be undone.")) return;
    try {
      await api.deleteDerivation(id);
      if (id === derivationId) reset();
      await refreshSaved();
    } catch (e) {
      handleError(e);
    }
  }

  async function deleteStep(item: HistoryItem) {
    try {
      await api.deleteHistoryItem(item.id);
      if (derivationId) setSteps((await api.derivation(derivationId)).steps);
      if (item.id === fromId) setFromId(null);
      await refreshSaved();
    } catch (e) {
      handleError(e);
    }
  }

  function openEarlier(item: HistoryItem) {
    if (item.operation === "geometry") {
      // the whole problem was stored with the result, so the form can be restored and changed
      const request = (item.result as { request?: GeometryRequest }).request;
      if (request) {
        setMode("geometry");
        setGeometryRestore({ request, nonce: Date.now() });
      }
      return;
    }
    setMode("expression");
    reset();
    setFormat(item.format);
    if (item.format === "latex") setLatex(item.input);
    else setText(item.input);
    setAssumptions(item.assumptions);
  }

  async function deleteEarlier(item: HistoryItem) {
    try {
      await api.deleteHistoryItem(item.id);
      await refreshSaved();
    } catch (e) {
      handleError(e);
    }
  }

  async function clearEarlier() {
    if (!window.confirm("Delete all saved calculations and derivations? This cannot be undone."))
      return;
    try {
      await api.clearHistory();
      reset();
      await refreshSaved();
    } catch (e) {
      handleError(e);
    }
  }

  const toggle = (name: keyof typeof open) => setOpen((o) => ({ ...o, [name]: !o[name] }));

  return (
    <div className="grid gap-x-8 gap-y-6 pb-16 pt-8 lg:grid-cols-[minmax(0,1fr)_20rem] lg:grid-rows-[auto_1fr]">
      <div className="min-w-0 space-y-4 lg:col-start-1 lg:row-start-1">
        <p className="text-xs text-muted">
          Working with a Lagrangian or Hamiltonian?{" "}
          <Link href="/workbench/conserved" className="text-accent hover:underline">
            Find its conserved quantities
          </Link>
        </p>
        <div role="group" aria-label="Workbench mode" className="flex gap-2">
          {(["expression", "geometry"] as const).map((m) => (
            <button
              key={m}
              type="button"
              aria-pressed={mode === m}
              onClick={() => setMode(m)}
              className={`rounded border px-3 py-1.5 text-sm ${
                mode === m
                  ? "border-accent bg-accent text-accent-ink"
                  : "border-rule bg-raised text-ink hover:bg-sunk"
              }`}
            >
              {m === "expression" ? "Expression" : "Geometry"}
            </button>
          ))}
        </div>
        {mode === "expression" && (
          <>
            <ExpressionEditor
              format={format}
              latex={latex}
              text={text}
              onLatex={setLatex}
              onText={setText}
              onFormat={setFormat}
              onSubmit={() => run(selected ?? "simplify")}
              continuing={
                fromStep && fromResult
                  ? {
                      step: steps.findIndex((s) => s.id === fromStep.id) + 1,
                      latex: fromResult.latex,
                      intact,
                    }
                  : null
              }
              onClear={() => {
                setFromId(null);
                if (format === "latex") setLatex("");
                else setText("");
              }}
              focusSignal={focusSignal}
            />
            <OperationBar
              selected={selected}
              onSelect={setSelected}
              settings={settings}
              onSettings={setSettings}
              onRun={run}
              onFind={find}
              empty={!value.trim()}
              busy={busy}
            />
            {error && (
              <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
                {error}
              </p>
            )}
          </>
        )}
      </div>

      <aside
        aria-label="Identities, assumptions and saved work"
        className="space-y-3 lg:col-start-2 lg:row-span-2 lg:row-start-1 lg:border-l lg:border-rule lg:pl-6"
      >
        {mode === "expression" && (
          <>
            <Section
              id="identities"
              title="Identities"
              open={open.identities}
              onToggle={() => toggle("identities")}
            >
              <IdentityPicker
                result={lookupFresh}
                stale={lookup !== null && lookupFresh === null}
                searching={busy === "identities"}
                applying={busy !== null && busy !== "identities" ? busy : null}
                disabled={!value.trim() || busy !== null}
                onFind={find}
                onApply={apply}
              />
            </Section>
            <Section
              id="assumptions"
              title="Assumptions"
              open={open.assumptions}
              onToggle={() => toggle("assumptions")}
            >
              <AssumptionsEditor value={assumptions} onChange={setAssumptions} />
            </Section>
          </>
        )}
        <Section id="saved" title="Saved" open={open.saved} onToggle={() => toggle("saved")}>
          <SavedPanel
            derivations={saved}
            activeId={derivationId}
            earlier={earlier}
            onOpen={openDerivation}
            onNew={reset}
            onDelete={deleteDerivation}
            onOpenEarlier={openEarlier}
            onDeleteEarlier={deleteEarlier}
            onClearEarlier={clearEarlier}
          />
        </Section>
      </aside>

      <div className="min-w-0 lg:col-start-1 lg:row-start-2">
        {mode === "geometry" ? (
          <GeometryWorkbench
            restore={geometryRestore}
            onUnauthorized={() => router.push("/login")}
            onSaved={() => void refreshSaved()}
          />
        ) : (
          <DerivationView
            steps={steps}
            currentId={fromId}
            onContinue={continueFrom}
            onDelete={deleteStep}
          />
        )}
      </div>
    </div>
  );
}
