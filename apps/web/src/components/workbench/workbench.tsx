"use client";

import Link from "next/link";
import { Math } from "@minerva/ui/math";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  api,
  type ChatResult,
  type ChatResultSource,
  type ComputeInput,
} from "@/lib/api";
import {
  DEFAULT_SETTINGS,
  inheritedDomain,
  collectTargets,
  continuation,
  independenceGroups,
  randomNames,
  relationLines,
  validateSettings,
  withRealVariable,
  type Settings,
  type StepOperation,
} from "@/lib/derivation";
import type {
  Assumptions,
  DiffVariable,
  DerivationSummary,
  DistributionNotation,
  DistributionResult,
  GeometryRequest,
  MatrixRequest,
  VectorRequest,
  HistoryItem,
  IdentitiesResult,
  IdentityMatch,
  StatModelRequest,
  StepResult,
  Subterm,
  SimplifyGoal,
} from "@/lib/types";
import { AssumptionsEditor } from "../assumptions-editor";
import { DerivationView } from "./derivation-view";
import { UndoToast } from "./undo-toast";
import { ExpressionEditor, type EditorFormat } from "./expression-editor";
import { restoreMatrixForm, type MatrixForm } from "@/lib/matrix-form";
import type { StatMechRequest } from "@/lib/statmech-types";
import { GeometryWorkbench } from "./geometry-workbench";
import { restoreStatForm, type StatForm } from "@/lib/statistics-form";
import { StatisticsWorkbench } from "./statistics-workbench";
import { MatrixWorkbench } from "./matrix-workbench";
import { VectorWorkbench } from "./vector-workbench";
import { restoreVectorForm, type VectorForm } from "@/lib/vector-form";
import { OdeWorkbench } from "./ode-workbench";
import type { OdeInput } from "@/lib/ode-types";
import { VariationalWorkbench, type VariationRestore } from "./variational-workbench";
import type { VariationRequest } from "@/lib/variation-types";
import { StatMechWorkbench } from "./statmech-workbench";
import { IdentityPicker } from "./identity-picker";
import { OperationBar } from "./operation-bar";
import { PartPicker, type PartAction } from "./part-picker";
import { SavedPanel } from "./saved-panel";

const MODE_GROUPS = [
  { label: "Algebra and calculus", modes: [["expression", "Expression"]] },
  {
    label: "Matrices and vectors",
    modes: [
      ["matrix", "Matrices"],
      ["vector", "Vectors"],
    ],
  },
  { label: "Differential equations", modes: [["ode", "Differential equations"]] },
  {
    label: "Statistics",
    modes: [
      ["statistics", "Statistics"],
      ["statmech", "Statistical mechanics"],
    ],
  },
  {
    label: "Physics and geometry",
    modes: [
      ["geometry", "Geometry"],
      ["variational", "Variational calculus"],
    ],
  },
] as const;

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
  const generation = useRef(0);
  const [imported, setImported] = useState<{ source: ChatResultSource; result: ChatResult } | null>(
    null,
  );
  const [draftKey, setDraftKey] = useState<string | null>(null);
  const [rewriteGoal, setRewriteGoal] = useState<SimplifyGoal | null>(null);
  const [rewriteTarget, setRewriteTarget] = useState<string[]>([]);
  const [needsBounds, setNeedsBounds] = useState(false);
  const [savedVariables, setSavedVariables] = useState<DiffVariable[] | null>(null);
  const [format, setFormat] = useState<EditorFormat>("latex");
  const [latex, setLatex] = useState("");
  const [text, setText] = useState("");
  const [assumptions, setAssumptions] = useState<Assumptions>({});
  const [derivationId, setDerivationId] = useState<string | null>(null);
  const [steps, setSteps] = useState<HistoryItem[]>([]);
  const [fromId, setFromId] = useState<string | null>(null);
  const [fromWhich, setFromWhich] = useState(0); // the solution continued, for an equation
  const [selected, setSelected] = useState<StepOperation | null>(null);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // What was just deleted, with the way back (see `deleteStep`, `deleteDerivation`).
  const [notice, setNotice] = useState<{ message: string; undo: () => Promise<void> } | null>(null);
  const [variationId, setVariationId] = useState<string | null>(null); // the variational mode's open derivation
  const deleting = useRef(new Set<string>());
  const [lookup, setLookup] = useState<{ key: string; result: IdentitiesResult } | null>(null);
  const [partOpen, setPartOpen] = useState(false);
  const [parts, setParts] = useState<{ key: string; result: StepResult } | null>(null);
  const [chosen, setChosen] = useState<{ key: string; part: Subterm } | null>(null);
  const [saved, setSaved] = useState<DerivationSummary[]>([]);
  const [earlier, setEarlier] = useState<HistoryItem[]>([]);
  const [focusSignal, setFocusSignal] = useState(0);
  const [open, setOpen] = useState({ identities: true, assumptions: true, saved: false });
  const [mode, setMode] = useState<
    | "expression"
    | "geometry"
    | "matrix"
    | "vector"
    | "statistics"
    | "statmech"
    | "ode"
    | "variational"
  >("expression");
  const [modeQuery, setModeQuery] = useState("");
  const [visitedModes, setVisitedModes] = useState<Set<typeof mode>>(new Set());
  const [vectorRestore, setVectorRestore] = useState<{ form: VectorForm; nonce: number } | null>(
    null,
  );
  const [statmechRestore, setStatmechRestore] = useState<{
    request: StatMechRequest;
    nonce: number;
  } | null>(null);
  const [matrixRestore, setMatrixRestore] = useState<{
    form: MatrixForm;
    nonce: number;
  } | null>(null);
  const [odeRestore, setOdeRestore] = useState<{ request: OdeInput; nonce: number } | null>(null);
  const [variationRestore, setVariationRestore] = useState<VariationRestore | null>(null);
  const [statRestore, setStatRestore] = useState<{ form: StatForm; nonce: number } | null>(null);
  const [geometryRestore, setGeometryRestore] = useState<{
    request: GeometryRequest;
    nonce: number;
  } | null>(null);

  const value = format === "latex" ? latex : text;
  const key = JSON.stringify([value, format, assumptions]);
  const fromStep = steps.find((s) => s.id === fromId) ?? null;
  const fromResult = fromStep ? continuation(fromStep, fromWhich) : null;
  const intact = fromResult !== null && text === fromResult.text;
  const parentId =
    intact && fromStep && JSON.stringify(assumptions) === JSON.stringify(fromStep.assumptions)
      ? fromStep.id
      : undefined;
  const chatSource =
    imported &&
    format === "text" &&
    value === imported.result.result_text &&
    JSON.stringify(assumptions) === JSON.stringify(imported.result.original_request.assumptions)
      ? imported.source
      : undefined;
  const sourceDomain = imported
    ? {
        exclusions: [
          ...(imported.result.restrictions_latex ?? []),
          ...(imported.result.conditions_latex ?? []),
        ],
        incomplete: imported.result.restrictions_complete === false,
      }
    : fromStep && intact
      ? inheritedDomain(fromStep)
      : null;
  const lookupFresh = lookup?.key === key ? lookup.result : null;
  // A part is chosen for this exact expression: when it changes, the choice is dropped.
  const partsFresh = parts?.key === key ? parts.result : null;
  const chosenPart = chosen?.key === key ? chosen.part : null;

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

  useEffect(
    () => () => {
      ++generation.current;
    },
    [],
  );

  useEffect(() => {
    if (!api.me) return;
    let cancelled = false;
    const request = generation.current;
    void api
      .me()
      .then((account) => {
        if (cancelled) return;
        const storageKey = `minerva:draft:${account.id}:workbench`;
        try {
          const raw = sessionStorage.getItem(storageKey);
          if (
            raw &&
            request === generation.current &&
            !new URLSearchParams(location.search).has("card")
          ) {
            const draft = JSON.parse(raw) as {
              format: EditorFormat;
              latex: string;
              text: string;
              assumptions: Assumptions;
              settings: Settings;
              selected: StepOperation | null;
              derivationId: string | null;
              steps: HistoryItem[];
              fromId: string | null;
              fromWhich: number;
              imported: typeof imported;
              needsBounds?: boolean;
              rewriteGoal?: SimplifyGoal | null;
              rewriteTarget?: string[];
              savedVariables?: DiffVariable[] | null;
            };
            setFormat(draft.format);
            setLatex(draft.latex);
            setText(draft.text);
            setAssumptions(draft.assumptions);
            setSettings(draft.settings);
            setSelected(draft.selected);
            setDerivationId(draft.derivationId);
            setSteps(draft.steps);
            setFromId(draft.fromId);
            setFromWhich(draft.fromWhich);
            setImported(draft.imported ?? null);
            setNeedsBounds(draft.needsBounds ?? false);
            setRewriteGoal(draft.rewriteGoal ?? null);
            setRewriteTarget(draft.rewriteTarget ?? []);
            setSavedVariables(draft.savedVariables ?? null);
            if (draft.derivationId && api.derivation) {
              // The stored steps are a copy: what the server holds now is what counts (a step may
              // have been deleted in another tab, or the whole derivation).
              void api
                .derivation(draft.derivationId)
                .then((fresh) => {
                  if (cancelled || request !== generation.current) return;
                  setSteps(fresh.steps);
                  setFromId((id) => (id && fresh.steps.some((x) => x.id === id) ? id : null));
                })
                .catch((e: unknown) => {
                  if (cancelled || request !== generation.current) return;
                  if (e instanceof ApiError && e.status === 404) {
                    setDerivationId(null);
                    setSteps([]);
                    setFromId(null);
                    setFromWhich(0);
                    setError("That derivation no longer exists; it was deleted.");
                  }
                });
            }
          }
        } catch {
          // The workbench remains usable without browser storage.
        }
        setDraftKey(storageKey);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!draftKey) return;
    try {
      if (!latex && !text && !derivationId && !Object.keys(assumptions).length) {
        sessionStorage.removeItem(draftKey);
      } else {
        sessionStorage.setItem(
          draftKey,
          JSON.stringify({
            format,
            latex,
            text,
            assumptions,
            settings,
            selected,
            derivationId,
            steps,
            fromId,
            fromWhich,
            imported,
            needsBounds,
            rewriteGoal,
            rewriteTarget,
            savedVariables,
          }),
        );
      }
    } catch {
      // Storage quotas do not prevent computation.
    }
  }, [
    draftKey,
    format,
    latex,
    text,
    assumptions,
    settings,
    selected,
    derivationId,
    steps,
    fromId,
    fromWhich,
    imported,
    needsBounds,
    rewriteGoal,
    rewriteTarget,
    savedVariables,
  ]);

  useEffect(() => {
    const query = new URLSearchParams(location.search);
    const conversation_id = query.get("conversation");
    const card_id = query.get("card");
    if (!conversation_id || !card_id) return;
    const source = { conversation_id, card_id };
    const request = ++generation.current;
    void api.chatResult(source).then(
      (result) => {
        if (request !== generation.current) return;
        setImported({ source, result });
        setText(result.result_text);
        setFormat("text");
        setAssumptions(result.original_request.assumptions);
        window.history.replaceState(null, "", "/workbench");
      },
      (error: unknown) => {
        if (request === generation.current) handleError(error);
      },
    );
  }, [handleError]);

  function invalidate() {
    ++generation.current;
    setBusy(null);
    setError(null);
  }

  function continueFrom(item: HistoryItem, edit: boolean, which?: number) {
    const next = continuation(item, which);
    if (!next) return;
    invalidate();
    setImported(null);
    setLatex("");
    setFormat("text");
    setText(next.text);
    setFromId(item.id);
    setFromWhich(which ?? 0);
    setAssumptions(item.assumptions); // the result is valid under the assumptions it was made with
    setError(null);
    if (edit) setFocusSignal((n) => n + 1);
  }

  /** Run `call` on the current expression and add its result as the next step. */
  async function addStep(busyKey: string, call: Call) {
    if (!value.trim() || busy) return;
    if (sourceDomain && (sourceDomain.exclusions.length || sourceDomain.incomplete)) {
      if (
        (!parentId && !chatSource) ||
        !["simplify", "differentiate", "expand", "factor", "substitute"].includes(busyKey)
      ) {
        setError("This operation cannot preserve the source result's domain restrictions.");
        return;
      }
    }
    const request = ++generation.current;
    setBusy(busyKey);
    setError(null);
    try {
      let id = derivationId;
      if (!id) {
        id = (await api.createDerivation()).id;
        if (generation.current === request) setDerivationId(id);
      }
      const response = await call({
        input: value,
        format,
        assumptions,
        derivation_id: id,
        parent_id: parentId,
        ...(chatSource ? { chat_source: chatSource } : {}),
      });
      const derivation = await api.derivation(id);
      if (generation.current !== request) {
        void refreshSaved();
        return;
      }
      setSteps(derivation.steps);
      const added = derivation.steps.find((s) => s.id === response.calculation_id);
      if (added) continueFrom(added, false);
      setLookup(null);
      void refreshSaved();
    } catch (e) {
      if (generation.current === request) handleError(e);
    } finally {
      if (generation.current === request) setBusy(null);
    }
  }

  /** The user answered "is that a Dirac delta?": the same call again, as a new step, with the answer. */
  async function answerNotation(item: HistoryItem, notation: DistributionNotation) {
    const saved = (item.result as unknown as DistributionResult).request;
    if (!saved || busy) return;
    const request = ++generation.current;
    setBusy("distribution");
    setError(null);
    try {
      const id = derivationId ?? item.derivation_id ?? (await api.createDerivation()).id;
      if (generation.current === request) setDerivationId(id);
      const response = await api.distribution(
        {
          input: item.input,
          format: item.format,
          assumptions: item.assumptions,
          derivation_id: id,
          parent_id: item.parent_id ?? undefined,
        },
        { ...saved, notation },
      );
      const derivation = await api.derivation(id);
      if (generation.current !== request) {
        void refreshSaved();
        return;
      }
      setSteps(derivation.steps);
      const added = derivation.steps.find((s) => s.id === response.calculation_id);
      if (added) continueFrom(added, false);
      void refreshSaved();
    } catch (e) {
      if (generation.current === request) handleError(e);
    } finally {
      if (generation.current === request) setBusy(null);
    }
  }

  function run(operation: StepOperation) {
    if (
      operation === "integrate" &&
      needsBounds &&
      (!settings.lower.trim() || !settings.upper.trim())
    ) {
      setError("Enter both bounds to rerun this saved definite integral.");
      return;
    }
    if (operation === "simplify" && rewriteGoal === "rewrite" && !rewriteTarget.length) {
      setError(
        "The saved rewrite does not include its target functions. Choose Simplify to start a new operation.",
      );
      return;
    }
    const problem = validateSettings(operation, settings);
    if (problem) {
      setError(problem);
      return;
    }
    const by = settings.variable;
    const selection = chosenPart
      ? { path: chosenPart.path, args: chosenPart.args, expected_latex: chosenPart.latex }
      : undefined;
    if (operation === "simplify") {
      const goal = rewriteGoal ?? "simplify";
      void addStep(goal, (c) =>
        goal === "simplify" ? api.simplify(c) : api.simplify({ ...c, goal, target: rewriteTarget }),
      );
    } else if (operation === "sums")
      void addStep(operation, (c) =>
        api.sums(c, {
          action: settings.sumsAction,
          variable: settings.sumsAction === "differentiate" ? by : null,
        }),
      );
    else if (operation === "expectation")
      void addStep(operation, (c) =>
        api.expectation(c, {
          action: settings.expectAction,
          random: randomNames(settings.random),
          independent: independenceGroups(settings.independent),
        }),
      );
    else if (operation === "substitute")
      void addStep(operation, (c) =>
        api.substitute(c, relationLines(settings.relations), settings.relationOrder || undefined),
      );
    else if (operation === "collect")
      void addStep(operation, (c) =>
        api.rewrite(c, { action: "collect", by: collectTargets(settings.collectBy), selection }),
      );
    else if (operation === "isolate")
      void addStep(operation, (c) => api.isolate(c, settings.solveFor));
    else if (operation === "differentiate")
      void addStep(operation, (c) =>
        api.differentiate(c, savedVariables ?? [{ name: by, order: settings.order }]),
      );
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
    const request = ++generation.current;
    setBusy("identities");
    setError(null);
    try {
      const result = await api.identities({ input: value, format, assumptions, save: false });
      if (generation.current !== request) return;
      setLookup({ key, result });
      setOpen((o) => ({ ...o, identities: true }));
    } catch (e) {
      if (generation.current === request) handleError(e);
    } finally {
      if (generation.current === request) setBusy(null);
    }
  }

  function apply(match: IdentityMatch) {
    if (chosenPart) {
      // The identity is matched against the chosen part alone.
      const selection = {
        path: chosenPart.path,
        args: chosenPart.args,
        expected_latex: chosenPart.latex,
      };
      void addStep(match.entry.id, (c) =>
        api.rewrite(c, { action: "rule", rule_id: match.entry.id, selection }),
      );
    } else {
      void addStep(match.entry.id, (c) => api.applyIdentity(c, match.entry.id));
    }
  }

  async function listParts(find: string) {
    if (!value.trim() || busy) return;
    const request = ++generation.current;
    setBusy("parts");
    setError(null);
    try {
      const result = await api.subterms(
        { input: value, format, assumptions },
        find ? { find } : {},
      );
      if (generation.current !== request) return;
      setParts({ key, result });
      setChosen(null);
    } catch (e) {
      if (generation.current === request) handleError(e);
    } finally {
      if (generation.current === request) setBusy(null);
    }
  }

  function runPart(action: PartAction) {
    if (!chosenPart) {
      setError("Choose a part first.");
      return;
    }
    const selection = {
      path: chosenPart.path,
      args: chosenPart.args,
      expected_latex: chosenPart.latex,
    };
    void addStep(`part-${action}`, (c) => api.rewrite(c, { action, selection }));
  }

  function reset() {
    invalidate();
    setAssumptions({});
    setImported(null);
    setRewriteGoal(null);
    setRewriteTarget([]);
    setSavedVariables(null);
    setNeedsBounds(false);
    setSelected(null);
    setSettings(DEFAULT_SETTINGS);
    setParts(null);
    setChosen(null);
    setPartOpen(false);
    setDerivationId(null);
    setSteps([]);
    setFromId(null);
    setLookup(null);
    setError(null);
    setFormat("latex");
    setLatex("");
    setText("");
    // the variational form starts over too (a new derivation, or another problem was opened)
    setVariationRestore({ nonce: generation.current, derivationId: null, steps: [] });
  }

  async function openDerivation(id: string) {
    invalidate();
    const request = generation.current;
    try {
      const derivation = await api.derivation(id);
      if (generation.current !== request) return;
      reset();
      if (derivation.steps[0]?.operation === "variation") {
        // a variational derivation lives in its own mode, which keeps its steps and form
        setMode("variational");
        setVariationId(id);
        setVariationRestore({
          nonce: generation.current,
          derivationId: id,
          steps: derivation.steps,
        });
        return;
      }
      setMode("expression");
      setDerivationId(id);
      setSteps(derivation.steps);
      const last = [...derivation.steps].reverse().find((s) => continuation(s) !== null);
      if (last) continueFrom(last, false);
      else {
        const input = derivation.steps.at(-1);
        if (input) restoreExpression(input);
      }
    } catch (e) {
      if (generation.current === request) handleError(e);
    }
  }

  /** Take back a deletion, then show what the server now holds. */
  async function undo(restore: () => Promise<unknown>) {
    setNotice(null);
    try {
      await restore();
      if (derivationId) setSteps((await api.derivation(derivationId)).steps);
      await refreshSaved();
    } catch (e) {
      handleError(e);
    }
  }

  async function deleteDerivation(id: string) {
    if (deleting.current.has(id)) return;
    deleting.current.add(id);
    try {
      await api.deleteDerivation(id);
      if (id === derivationId) reset();
      else if (id === variationId) {
        setVariationRestore({ nonce: generation.current, derivationId: null, steps: [] });
      }
      setNotice({
        message: "Derivation deleted.",
        undo: () => undo(() => api.restoreDerivation(id)),
      });
      await refreshSaved();
    } catch (e) {
      handleError(e);
    } finally {
      deleting.current.delete(id);
    }
  }

  /**
   * Delete a step and the steps that continued it. The screen changes at once and is put back, with
   * the reason, if the server refuses. The change is a filter of the current list, never a copy taken
   * before the request, so a step added or an edit made meanwhile is not undone by it.
   */
  async function deleteStep(item: HistoryItem) {
    if (deleting.current.has(item.id)) return;
    deleting.current.add(item.id);
    const below = new Set([item.id]);
    for (let grew = true; grew;) {
      grew = false;
      for (const s of steps) {
        if (s.parent_id && below.has(s.parent_id) && !below.has(s.id)) {
          below.add(s.id);
          grew = true;
        }
      }
    }
    const removed = steps.filter((s) => below.has(s.id));
    const dropFrom = (ids: Set<string>) => setSteps((all) => all.filter((s) => !ids.has(s.id)));
    dropFrom(below);
    if (fromId && below.has(fromId)) {
      setFromId(null);
      setFromWhich(0);
    }
    try {
      const { ids } = await api.deleteHistoryItem(item.id);
      dropFrom(new Set(ids));
      setNotice({
        message: ids.length > 1 ? `Step and ${ids.length - 1} after it deleted.` : "Step deleted.",
        undo: () => undo(() => api.restoreHistoryItem(item.id)),
      });
      await refreshSaved();
    } catch (e) {
      setSteps((all) =>
        [...all, ...removed.filter((r) => !all.some((s) => s.id === r.id))].sort((a, b) =>
          a.created_at.localeCompare(b.created_at),
        ),
      );
      handleError(e);
    } finally {
      deleting.current.delete(item.id);
    }
  }

  /** A right-hand side from a closed-form solution, to work on as an ordinary expression. */
  function continueWithSolution(expression: string) {
    setMode("expression");
    reset();
    setFormat("text");
    setText(expression);
  }

  function openEarlier(item: HistoryItem) {
    invalidate();
    if (
      item.operation === "ode_analyze" ||
      item.operation === "ode_solve_symbolic" ||
      item.operation === "ode_solve_numerical"
    ) {
      // the whole problem was stored with the result, so the form can be restored and changed
      const request = (item.result as { request?: OdeInput }).request;
      if (request) {
        setMode("ode");
        setOdeRestore({ request, nonce: generation.current });
      }
      return;
    }
    if (item.operation === "statmech") {
      // the whole problem was stored with the result, so the form can be restored and changed
      const request = (item.result as { request?: StatMechRequest }).request;
      if (request) {
        setMode("statmech");
        setStatmechRestore({ request, nonce: generation.current });
      }
      return;
    }
    if (item.operation === "geometry") {
      // the whole problem was stored with the result, so the form can be restored and changed
      const request = (item.result as { request?: GeometryRequest }).request;
      if (request) {
        setMode("geometry");
        setGeometryRestore({ request, nonce: generation.current });
      }
      return;
    }
    if (item.operation === "vector") {
      // the whole problem was stored with the result, so the form can be restored and changed
      const request = (item.result as { request?: VectorRequest }).request;
      if (request) {
        setMode("vector");
        setVectorRestore({
          form: restoreVectorForm(request, item.input, item.format, item.assumptions),
          nonce: generation.current,
        });
        return;
      }
    }
    if (item.operation === "variation") {
      // the whole functional was stored with the result, so the form can be restored and changed
      const request = (item.result as { request?: Partial<VariationRequest> }).request;
      if (request) {
        setMode("variational");
        setVariationRestore({ nonce: generation.current, derivationId: null, steps: [], request });
        return;
      }
    }
    if (item.operation === "statistical_model") {
      // the whole problem was stored with the result, so the form can be restored and changed
      const request = (item.result as { request?: StatModelRequest }).request;
      if (request) {
        setMode("statistics");
        setStatRestore({ form: restoreStatForm(request), nonce: generation.current });
        return;
      }
    }
    if (item.operation === "matrix_calculus") {
      // the declarations were stored with the result, so the form can be restored and changed
      const request = (item.result as { request?: MatrixRequest }).request;
      if (request) {
        setMode("matrix");
        setMatrixRestore({
          form: restoreMatrixForm(request, item.input, item.format, item.assumptions),
          nonce: generation.current,
        });
        return;
      }
    }
    setMode("expression");
    reset();
    restoreExpression(item);
  }

  function restoreExpression(item: HistoryItem) {
    setFormat(item.format);
    if (item.format === "latex") setLatex(item.input);
    else setText(item.input);
    setAssumptions(item.assumptions);
    const result = item.result as unknown as { original_request?: Record<string, unknown> };
    const request = result.original_request ?? (item.result as unknown as Record<string, unknown>);
    const variable =
      typeof request.variable === "string" ? request.variable : DEFAULT_SETTINGS.variable;
    if (item.operation === "simplify") {
      setSelected("simplify");
      const goal = request.goal as SimplifyGoal | undefined;
      setRewriteGoal(goal ?? null);
      setRewriteTarget(Array.isArray(request.target) ? request.target : []);
    }
    if (item.operation === "differentiate" && Array.isArray(request.variables)) {
      const variables = request.variables as DiffVariable[];
      const first = variables[0];
      setSelected("differentiate");
      if (first) setSettings({ ...DEFAULT_SETTINGS, variable: first.name, order: first.order });
      if (variables.length > 1) setSavedVariables(variables);
    }
    if (item.operation === "integrate" || item.operation === "series") {
      setSelected(item.operation);
      setSettings({
        ...DEFAULT_SETTINGS,
        variable,
        lower: typeof request.lower === "string" ? request.lower : "",
        upper: typeof request.upper === "string" ? request.upper : "",
        point: String(request.point ?? request.point_requested ?? DEFAULT_SETTINGS.point),
        seriesOrder:
          typeof request.order === "number" ? request.order : DEFAULT_SETTINGS.seriesOrder,
      });
      if (item.operation === "integrate" && request.definite && !result.original_request) {
        setNeedsBounds(true);
        setError(
          "This saved integral does not include its original bounds. Enter both bounds before rerunning it.",
        );
      }
    }
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
          {" · "}A parameterised integral?{" "}
          <Link href="/workbench/telescoping" className="text-accent hover:underline">
            Creative telescoping
          </Link>
          {" · "}Need a simpler expression?{" "}
          <Link href="/workbench/approximation" className="text-accent hover:underline">
            Approximations
          </Link>
        </p>
        {error && (
          <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
            {error}
          </p>
        )}
        {notice && (
          <UndoToast
            message={notice.message}
            onUndo={() => void notice.undo()}
            onDismiss={() => setNotice(null)}
          />
        )}
        <label className="block text-xs text-muted">
          Search workbench modes
          <input
            type="search"
            value={modeQuery}
            onChange={(event) => setModeQuery(event.target.value)}
            className="ml-2 rounded border border-rule bg-raised px-2 py-1 text-sm"
          />
        </label>
        <div role="group" aria-label="Workbench mode" className="flex flex-wrap gap-2">
          {MODE_GROUPS.map(({ label, modes }) => {
            const matches = modes.filter(([, name]) =>
              (label + " " + name).toLowerCase().includes(modeQuery.trim().toLowerCase()),
            );
            if (!matches.length) return null;
            return (
              <div key={label} className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs text-muted">{label}</span>
                {matches.map(([next, name]) => (
                  <button
                    key={next}
                    type="button"
                    aria-pressed={mode === next}
                    onClick={() => {
                      invalidate();
                      setVisitedModes((old) => new Set([...old, mode, next]));
                      setMode(next);
                    }}
                    className={
                      mode === next
                        ? "rounded border border-accent bg-accent px-3 py-1.5 text-sm text-accent-ink"
                        : "rounded border border-rule bg-raised px-3 py-1.5 text-sm hover:bg-sunk"
                    }
                  >
                    {name}
                  </button>
                ))}
              </div>
            );
          })}
        </div>
        {mode === "expression" && (
          <>
            <ExpressionEditor
              format={format}
              latex={latex}
              text={text}
              onLatex={(next) => {
                invalidate();
                setLatex(next);
                setText("");
              }}
              onText={(next) => {
                invalidate();
                setText(next);
                setLatex("");
              }}
              onFormat={(next) => {
                invalidate();
                setFormat(next);
              }}
              assumptions={assumptions}
              onConvert={(next, converted) => {
                invalidate();
                setFormat(next);
                if (next === "latex") setLatex(converted);
                else setText(converted);
              }}
              onSubmit={() => run(selected ?? "simplify")}
              continuing={
                fromStep && fromResult
                  ? {
                      step: steps.findIndex((s) => s.id === fromStep.id) + 1,
                      latex: fromResult.latex,
                      intact,
                      domain: sourceDomain,
                      taylor: fromStep.operation === "series",
                    }
                  : null
              }
              onClear={() => {
                invalidate();
                setFromId(null);
                setLatex("");
                setText("");
              }}
              focusSignal={focusSignal}
            />
            {imported && (
              <div className="text-xs text-muted" role="status">
                {imported.result.operation === "series"
                  ? "Taylor polynomial from chat"
                  : "Result from chat"}
                {sourceDomain?.exclusions.map((restriction) => (
                  <div key={restriction}>
                    Source restriction: <Math latex={restriction} />
                  </div>
                ))}
                {sourceDomain?.incomplete && (
                  <p>Some source restrictions could not be determined.</p>
                )}
              </div>
            )}
            {rewriteGoal && (
              <button
                type="button"
                className="rounded border border-rule px-3 py-2 text-sm"
                disabled={!value.trim() || busy !== null}
                onClick={() => run("simplify")}
              >
                Run {rewriteGoal}
                {rewriteGoal === "rewrite" && rewriteTarget.length > 0
                  ? " in terms of " + rewriteTarget.join(", ")
                  : ""}
              </button>
            )}
            {savedVariables && (
              <p className="text-xs text-muted">
                Saved derivative variables:{" "}
                {savedVariables
                  .map(({ name, order }) => name + " (order " + order + ")")
                  .join(", ")}
                . Editing the derivative settings replaces this list with one variable.
              </p>
            )}
            <OperationBar
              selected={selected}
              onSelect={(operation) => {
                setRewriteGoal(null);
                setSavedVariables(null);
                setSelected(operation);
              }}
              settings={settings}
              onSettings={(next) => {
                invalidate();
                setSettings(next);
                setSavedVariables(null);
              }}
              onRun={run}
              onFind={find}
              partOpen={partOpen}
              onTogglePart={() => setPartOpen((o) => !o)}
              partChosen={chosenPart !== null}
              empty={!value.trim()}
              busy={busy}
            />
            {partOpen && (
              <PartPicker
                result={partsFresh}
                stale={parts !== null && partsFresh === null}
                searching={busy === "parts"}
                chosen={chosenPart}
                disabled={!value.trim() || busy !== null}
                onList={listParts}
                onChoose={(part) => setChosen(part ? { key, part } : null)}
                onRun={runPart}
              />
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
                assumptions={assumptions}
                onApprove={async (next) => {
                  const request = generation.current;
                  try {
                    const result = await api.identities({
                      input: value,
                      format,
                      assumptions: next,
                      save: false,
                    });
                    if (request !== generation.current) return null;
                    if (result.status !== "ok") return result.message;
                    invalidate();
                    setAssumptions(next);
                    setLookup({ key: JSON.stringify([value, format, next]), result });
                    return null;
                  } catch {
                    return "Could not validate the assumptions. Your declarations are unchanged.";
                  }
                }}
              />
            </Section>
            <Section
              id="assumptions"
              title="Assumptions"
              open={open.assumptions}
              onToggle={() => toggle("assumptions")}
            >
              <AssumptionsEditor
                value={assumptions}
                onChange={(next) => {
                  invalidate();
                  setAssumptions(next);
                }}
              />
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
        {(mode === "vector" || visitedModes.has("vector")) && (
          <div hidden={mode !== "vector"}>
            <VectorWorkbench
              restore={vectorRestore}
              draftKey={draftKey ? draftKey + ":vector" : undefined}
              onUnauthorized={() => router.push("/login")}
              onSaved={() => void refreshSaved()}
            />
          </div>
        )}
        {(mode === "ode" || visitedModes.has("ode")) && (
          <div hidden={mode !== "ode"}>
            <OdeWorkbench
              restore={odeRestore}
              draftKey={draftKey ? draftKey + ":ode" : undefined}
              onUnauthorized={() => router.push("/login")}
              onSaved={() => void refreshSaved()}
              onUseExpression={(expression) => continueWithSolution(expression)}
            />
          </div>
        )}
        {(mode === "statistics" || visitedModes.has("statistics")) && (
          <div hidden={mode !== "statistics"}>
            <StatisticsWorkbench
              restore={statRestore}
              draftKey={draftKey ? draftKey + ":statistics" : undefined}
              onUnauthorized={() => router.push("/login")}
              onSaved={() => void refreshSaved()}
            />
          </div>
        )}
        {(mode === "matrix" || visitedModes.has("matrix")) && (
          <div hidden={mode !== "matrix"}>
            <MatrixWorkbench
              restore={matrixRestore}
              draftKey={draftKey ? draftKey + ":matrix" : undefined}
              onUnauthorized={() => router.push("/login")}
              onSaved={() => void refreshSaved()}
            />
          </div>
        )}
        {(mode === "variational" || visitedModes.has("variational")) && (
          <div hidden={mode !== "variational"}>
            <VariationalWorkbench
              restore={variationRestore}
              draftKey={draftKey ? draftKey + ":variational" : undefined}
              onUnauthorized={() => router.push("/login")}
              onSaved={() => void refreshSaved()}
              onDerivationChange={setVariationId}
            />
          </div>
        )}
        {(mode === "statmech" || visitedModes.has("statmech")) && (
          <div hidden={mode !== "statmech"}>
            <StatMechWorkbench
              restore={statmechRestore}
              draftKey={draftKey ? draftKey + ":statmech" : undefined}
              onUnauthorized={() => router.push("/login")}
              onSaved={() => void refreshSaved()}
            />
          </div>
        )}
        {(mode === "geometry" || visitedModes.has("geometry")) && (
          <div hidden={mode !== "geometry"}>
            <GeometryWorkbench
              restore={geometryRestore}
              draftKey={draftKey ? draftKey + ":geometry" : undefined}
              onUnauthorized={() => router.push("/login")}
              onSaved={() => void refreshSaved()}
            />
          </div>
        )}
        {mode === "expression" && (
          <DerivationView
            steps={steps}
            currentId={fromId}
            onContinue={continueFrom}
            onApprove={async (item, next) => {
              if (busy || !derivationId) return "Wait for the current calculation to finish.";
              const request = ++generation.current;
              const controller = new AbortController();
              const expression = { input: item.input, format: item.format, assumptions: next };
              setBusy("simplify");
              try {
                const reading = await api.inspect(expression, controller.signal);
                if (request !== generation.current) return null;
                if (reading.status !== "ok") return reading.message;
                const original = item.result as {
                  goal?: SimplifyGoal;
                  original_request?: { goal?: SimplifyGoal; target?: string[] };
                };
                const goal = original.original_request?.goal ?? original.goal;
                const target = original.original_request?.target;
                if (goal === "rewrite" && !target?.length)
                  return "The original rewrite target is unavailable.";
                const result = await api.simplify({
                  ...expression,
                  derivation_id: derivationId,
                  goal,
                  target,
                });
                const derivation = await api.derivation(derivationId);
                if (request !== generation.current) return null;
                setSteps(derivation.steps);
                const added = derivation.steps.find((step) => step.id === result.calculation_id);
                if (added && continuation(added)) continueFrom(added, false);
                void refreshSaved();
                return null;
              } catch (error) {
                return request === generation.current && error instanceof ApiError
                  ? error.message
                  : "Could not rerun the calculation.";
              } finally {
                if (request === generation.current) setBusy(null);
              }
            }}
            onChoose={(item, operation) => {
              continueFrom(item, true);
              setSettings(DEFAULT_SETTINGS);
              setSavedVariables(null);
              setNeedsBounds(false);
              setRewriteGoal(operation === "expand" || operation === "factor" ? operation : null);
              setSelected(
                operation === "identities"
                  ? null
                  : operation === "expand" || operation === "factor"
                    ? "simplify"
                    : operation,
              );
              if (operation === "identities") setOpen((old) => ({ ...old, identities: true }));
            }}
            onDelete={deleteStep}
            onAnswerNotation={answerNotation}
          />
        )}
      </div>
    </div>
  );
}
