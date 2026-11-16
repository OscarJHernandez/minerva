"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";
import type { ApproximationToolArgs } from "@/lib/approximation-types";
import { chatApi, sendMessage } from "@/lib/chat-api";
import { buildDebugReport, buildTurnReport, type LastError } from "@/lib/debug-report";
import { useDebugMode } from "@/lib/debug-mode";
import { splitMath } from "@/lib/inline-math";
import type { OdeAnalyzeInput, OdeInput, OdeNumericInput, OdeSolveInput } from "@/lib/ode-types";
import type { StatAnswers, StatToolArgs } from "@/lib/statmech-types";
import type { DistributionAnswer, GeometryRequest } from "@/lib/types";
import type { VariationRequest } from "@/lib/variation-types";
import type {
  AssistantContent,
  Block,
  ChatStatus,
  ConversationSummary,
  MotionRequest,
  PerturbationRequest,
  StoredMessage,
  StreamEvent,
} from "@/lib/chat-types";
import { modelSettings, providerChoice } from "@/lib/model-settings";
import { Composer } from "./composer";
import { ModelPicker } from "./model-picker";
import { ModelSettingsPanel } from "./model-settings-panel";
import { CopyButton } from "./debug-panel";
import { conversationPath } from "@/lib/next-path";
import { retryLabel, untilLastUser, withUserDetails } from "@/lib/retry";
import { InsertReferenceProvider } from "./equation-number";
import { FormulaActionsProvider } from "./formula-actions";
import { ParamsPanel } from "./params-panel";
import type { Overrides } from "@/lib/model-params";
import { ConversationList } from "./conversation-list";
import { MessageView } from "./message-view";
import { PendingView, type Pending } from "./pending-view";
import { MotionRunnerProvider } from "./motion-runner";
import { RichText } from "./rich-text";

const EXAMPLES = [
  { title: "Simplify and verify", text: "Simplify $\\frac{x^2-1}{x-1}$" },
  {
    title: "Which identities apply?",
    text: "Which identities apply to $\\log(xy)$ if x and y are positive?",
  },
  { title: "Look up a formula", text: "What is the Gaussian integral formula?" },
  { title: "Be careful with roots", text: "Simplify $\\sqrt{x^2}$" },
  {
    title: "Equations of motion",
    text: "Find the equations of motion and conserved quantities of the Lagrangian $\\frac12 m(\\dot{r}^2 + r^2\\dot{\\theta}^2) + \\frac{k}{r}$",
  },
  {
    title: "Solve an equation",
    text: "Solve $x^5 - x - 1 = 0$ and tell me if there is a closed formula",
  },
];

export function ChatApp({ initialId }: { initialId?: string } = {}) {
  const router = useRouter();
  const [status, setStatus] = useState<ChatStatus | null>(null);
  const [accountId, setAccountId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<StoredMessage[]>([]);
  const [pending, setPending] = useState<Pending | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [listOpen, setListOpen] = useState(false);
  const [debugOn, setDebugOn] = useDebugMode();
  const [paramOverrides, setParamOverrides] = useState<Overrides>({});
  const [referenceRequest, setReferenceRequest] = useState<{ id: number; text: string } | null>(
    null,
  );
  // "Edit in chat" on a formula in a reply puts it in the message box (a formula, not a reference).
  const [insertRequest, setInsertRequest] = useState<{ id: number; latex: string } | null>(null);
  const requestCounter = useRef(0); // ids for requests to the composer (insert, replace)
  const formulaActions = useMemo(
    () => ({
      editInChat: (latex: string) => setInsertRequest({ id: ++requestCounter.current, latex }),
      openInWorkbench: (cardId: string) => {
        if (activeId)
          router.push(
            `/workbench?conversation=${encodeURIComponent(activeId)}&card=${encodeURIComponent(cardId)}`,
          );
      },
    }),
    [activeId, router],
  );
  const [replaceRequest, setReplaceRequest] = useState<{ id: number; text: string } | null>(null);
  const [editing, setEditing] = useState(false); // the box holds the last message, being changed
  const [lastError, setLastError] = useState<LastError | null>(null);
  const [modelOpen, setModelOpen] = useState(false);
  const cloudConfirmed = useRef<Set<string>>(new Set());
  const abortRef = useRef<AbortController | null>(null);
  // Each navigation owns its asynchronous work. A late response may finish, but cannot paint
  // into a different conversation.
  const viewGeneration = useRef(0);
  const activeIdRef = useRef<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const pendingRef = useRef<Pending | null>(null);
  const stickRef = useRef(true); // the reader is at the end, so new text scrolls into view
  const [atEnd, setAtEnd] = useState(true);

  const fail = useCallback(
    (e: unknown) => {
      if (e instanceof ApiError && e.status === 401) {
        // Log in again and come back to this conversation.
        const here = window.location.pathname;
        return router.push(here === "/" ? "/login" : `/login?next=${encodeURIComponent(here)}`);
      }
      setError(e instanceof ApiError ? e.message : "Something went wrong. Try again.");
    },
    [router],
  );

  const refreshList = useCallback(async () => {
    try {
      setConversations(await chatApi.list());
    } catch (e) {
      fail(e);
    }
  }, [fail]);

  useEffect(() => {
    chatApi.status().then(setStatus).catch(fail);
    api
      .me()
      .then((user) => setAccountId(user.id))
      .catch(fail);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial data load
    void refreshList();
  }, [fail, refreshList]);

  useEffect(() => {
    // Only follow the reply while the reader is at the end: scrolling up to read must not be undone.
    if ((messages.length > 0 || pending) && stickRef.current) {
      bottomRef.current?.scrollIntoView({ block: "end" });
    }
  }, [messages, pending]);

  function onLogScroll() {
    const el = logRef.current;
    if (!el) return;
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    stickRef.current = near;
    setAtEnd(near);
  }

  function jumpToEnd() {
    stickRef.current = true;
    setAtEnd(true);
    bottomRef.current?.scrollIntoView({ block: "end" });
  }

  // The address names the open conversation, so a reload or a new login comes back to it. replaceState:
  // no history entry per conversation, and no navigation (the page is not rebuilt).
  useEffect(() => {
    const target = activeId ? conversationPath(activeId) : "/";
    if (window.location.pathname !== target) window.history.replaceState(null, "", target);
  }, [activeId]);

  useEffect(
    () => () => {
      ++viewGeneration.current;
      abortRef.current?.abort();
    },
    [],
  );

  function clearComposerRequests() {
    setInsertRequest(null);
    setReferenceRequest(null);
    setReplaceRequest(null);
    setEditing(false);
  }

  async function rename(id: string, title: string) {
    try {
      const updated = await chatApi.rename(id, title);
      setConversations((items) => items.map((item) => (item.id === id ? updated : item)));
    } catch (error) {
      fail(error);
    }
  }

  async function open(id: string, options: { keepError?: boolean } = {}) {
    const generation = ++viewGeneration.current;
    if (!options.keepError) clearComposerRequests();
    abortRef.current?.abort();
    activeIdRef.current = id;
    setActiveId(id);
    setMessages([]);
    pendingRef.current = null;
    setPending(null);
    if (!options.keepError) {
      setError(null);
      setLastError(null);
    }
    setListOpen(false);
    try {
      const detail = await chatApi.get(id);
      if (viewGeneration.current !== generation) return;
      setMessages(detail.messages);
      setPending(null);
    } catch (e) {
      if (viewGeneration.current !== generation) return;
      if (e instanceof ApiError && e.status === 404) {
        // An address for a conversation that is gone (deleted, or someone else's): start fresh.
        setActiveId(null);
        activeIdRef.current = null;
        setMessages([]);
        setError("That conversation was not found.");
        return;
      }
      fail(e);
    }
  }

  useEffect(() => {
    if (!initialId) return;
    const timer = setTimeout(() => void open(initialId), 0);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only once, for the address the page was opened at
  }, [initialId]);

  function startNew() {
    clearComposerRequests();
    pendingRef.current = null;
    ++viewGeneration.current;
    abortRef.current?.abort();
    activeIdRef.current = null;
    setActiveId(null);
    setMessages([]);
    setPending(null);
    setError(null);
    setLastError(null);
    setListOpen(false);
  }

  async function remove(id: string) {
    if (!window.confirm("Delete this conversation? This cannot be undone.")) return;
    try {
      await chatApi.remove(id);
      if (id === activeIdRef.current) startNew();
      if (accountId) {
        try {
          sessionStorage.removeItem(`minerva:draft:${accountId}:${id}`);
          sessionStorage.removeItem(`minerva:draft:${accountId}:${id}:formula`);
        } catch {
          // Storage availability does not affect deletion.
        }
      }
      await refreshList();
    } catch (e) {
      fail(e);
    }
  }

  /** Moving a conversation that already has turns to the cloud needs one explicit yes. */
  function confirmCloud(): boolean {
    const choice = providerChoice(modelSettings.get());
    if (choice?.kind !== "openrouter" || !activeId || messages.length === 0) return true;
    if (cloudConfirmed.current.has(activeId)) return true;
    const ok = window.confirm(
      "Send this conversation to OpenRouter? Its earlier messages and Minerva's tool results " +
        "will be transmitted to OpenRouter and the model's provider. Your local model keeps nothing of it.",
    );
    if (ok) cloudConfirmed.current.add(activeId);
    return ok;
  }

  /** Shows one reply as it arrives; the ref lets Stop keep what had arrived. */
  function patchPending(change: (p: Pending) => Pending) {
    const next = pendingRef.current ? change(pendingRef.current) : null;
    pendingRef.current = next;
    setPending(next);
  }

  /**
   * Sends a message. Returns false only when nothing was sent (the request failed before the reply began):
   * the composer then gives the text back, and the message is not left in the conversation.
   */
  async function send(text: string, mode: SendMode = "new"): Promise<boolean> {
    const generation = viewGeneration.current;
    if (pendingRef.current) return false;
    pendingRef.current = { items: [], tools: [], cards: 0, startedAt: Date.now() };
    setPending(pendingRef.current);
    setError(null);
    setLastError(null);
    const controller = new AbortController();
    abortRef.current = controller;
    let id = activeIdRef.current;
    let bubble = false; // the message was added to the screen
    let started = false; // the server answered and the reply began
    let finished = false; // a reply or an error event arrived
    try {
      if (!id) {
        const created = await chatApi.create();
        if (generation !== viewGeneration.current || controller.signal.aborted) {
          void refreshList();
          return false;
        }
        id = created.id;
        activeIdRef.current = id;
        setActiveId(id);
      }
      if (mode === "new") {
        setMessages((m) => [
          ...m,
          {
            id: `local-${m.length}`,
            role: "user",
            content: { text },
            created_at: new Date().toISOString(),
          },
        ]);
        bubble = true;
      } else {
        // The reply is replaced; an edited message changes its text. The server keeps the same message.
        setMessages((m) => untilLastUser(m, mode === "edit" ? text : undefined));
      }
      stickRef.current = true; // sending jumps to the end
      await sendMessage(
        id,
        text,
        (event: StreamEvent) => {
          if (generation !== viewGeneration.current || activeIdRef.current !== id) return;
          started = true;
          if (event.event === "text") {
            patchPending((p) => appendText(p, event.data.delta));
          } else if (event.event === "tool_call") {
            patchPending((p) => ({ ...p, tools: [...p.tools, event.data] }));
          } else if (event.event === "user") {
            // What the server found in the message: equation numbers and what a reference was taken to be.
            setMessages((m) => withUserDetails(m, event.data));
          } else if (event.event === "phase") {
            patchPending((p) => ({ ...p, phase: event.data.phase }));
          } else if (event.event === "card") {
            patchPending((p) => ({
              ...p,
              items: [...p.items, { kind: "card", card: event.data }],
              cards: p.cards + 1,
            }));
          } else if (event.event === "message") {
            const content: AssistantContent = event.data;
            finished = true;
            setMessages((m) => [
              ...m,
              {
                id: `reply-${m.length}`,
                role: "assistant",
                content,
                created_at: new Date().toISOString(),
              },
            ]);
            pendingRef.current = null;
            setPending(null);
          } else if (event.event === "error") {
            finished = true;
            setError(event.data.message);
            setLastError(event.data);
            pendingRef.current = null;
            setPending(null);
          }
        },
        controller.signal,
        providerChoice(modelSettings.get()),
        debugOn && Object.keys(paramOverrides).length > 0 ? paramOverrides : undefined,
        mode === "new" ? undefined : mode,
      );
      if (generation !== viewGeneration.current) return true;
      if (!finished && id) {
        // The stream ended without a reply (a dropped connection): show what the server kept.
        setError("The connection ended before the reply was complete. Showing what was saved.");
        await open(id, { keepError: true });
      }
      await refreshList();
      return true;
    } catch (e) {
      if (generation !== viewGeneration.current) return true;
      if (e instanceof DOMException && e.name === "AbortError") {
        const partial = pendingRef.current;
        if (partial && partial.items.length > 0) {
          setMessages((m) => [...m, stoppedMessage(partial, m.length)]);
        }
        return true;
      }
      if (!started) {
        // Nothing was sent: take the message off the screen and give the text back to the composer.
        if (bubble) setMessages((m) => m.slice(0, -1));
        fail(e);
        if (mode !== "new" && id) await open(id, { keepError: true }); // what the server still has
        if (mode === "new" && id && activeIdRef.current === id) {
          // Creating the conversation remounted the composer. Restore into the new one.
          setReplaceRequest({ id: ++requestCounter.current, text });
          return true;
        }
        return false;
      }
      fail(e);
      if (id) await open(id, { keepError: true });
      return true;
    } finally {
      if (generation === viewGeneration.current) {
        abortRef.current = null;
        pendingRef.current = null;
        setPending(null);
      }
    }
  }

  /** The Lagrangian card's form: runs the numerical solution with the user's own numbers. */
  const runMotion = useCallback(
    async (request: MotionRequest) => {
      if (!activeId) throw new ApiError(400, "Open a conversation first.");
      const generation = viewGeneration.current;
      try {
        const message = await chatApi.integrateMotion(activeId, request);
        if (generation === viewGeneration.current && activeIdRef.current === activeId)
          setMessages((m) => [...m, message]);
        await refreshList();
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) router.push("/login");
        throw e;
      }
    },
    [activeId, refreshList, router],
  );

  /** The perturbation card's form: solves again with the user's own comparison numbers. */
  const runPerturbation = useCallback(
    async (request: PerturbationRequest) => {
      if (!activeId) throw new ApiError(400, "Open a conversation first.");
      const generation = viewGeneration.current;
      try {
        const message = await chatApi.perturbationSolve(activeId, request);
        if (generation === viewGeneration.current && activeIdRef.current === activeId)
          setMessages((m) => [...m, message]);
        await refreshList();
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) router.push("/login");
        throw e;
      }
    },
    [activeId, refreshList, router],
  );

  /** The delta card's question: runs the call again with the user's answer, without the model. */
  const runDistribution = useCallback(
    async (request: DistributionAnswer) => {
      if (!activeId) throw new ApiError(400, "Open a conversation first.");
      const generation = viewGeneration.current;
      try {
        const message = await chatApi.distribution(activeId, request);
        if (generation === viewGeneration.current && activeIdRef.current === activeId)
          setMessages((m) => [...m, message]);
        await refreshList();
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) router.push("/login");
        throw e;
      }
    },
    [activeId, refreshList, router],
  );

  /** The statistical-mechanics card's question: runs the call again with the user's choice. */
  const runStatMech = useCallback(
    async (request: StatToolArgs & StatAnswers) => {
      if (!activeId) throw new ApiError(400, "Open a conversation first.");
      const generation = viewGeneration.current;
      try {
        const message = await chatApi.statmech(activeId, request);
        if (generation === viewGeneration.current && activeIdRef.current === activeId)
          setMessages((m) => [...m, message]);
        await refreshList();
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) router.push("/login");
        throw e;
      }
    },
    [activeId, refreshList, router],
  );

  /** The geometry card's buttons: more quantities for the same metric, without the model. */
  const runGeometry = useCallback(
    async (request: GeometryRequest) => {
      if (!activeId) throw new ApiError(400, "Open a conversation first.");
      const generation = viewGeneration.current;
      try {
        const message = await chatApi.geometry(activeId, request);
        if (generation === viewGeneration.current && activeIdRef.current === activeId)
          setMessages((m) => [...m, message]);
        await refreshList();
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) router.push("/login");
        throw e;
      }
    },
    [activeId, refreshList, router],
  );

  /** An analysis card's forms: a closed form or a numerical solution with the user's own numbers. */
  const runOde = useMemo(
    () => ({
      symbolic: async (request: OdeInput | OdeSolveInput) => {
        if (!activeId) throw new ApiError(400, "Open a conversation first.");
        const generation = viewGeneration.current;
        try {
          const message = await chatApi.odeSolveSymbolic(activeId, request);
          if (generation === viewGeneration.current && activeIdRef.current === activeId)
            setMessages((m) => [...m, message]);
          await refreshList();
        } catch (e) {
          if (e instanceof ApiError && e.status === 401) router.push("/login");
          throw e;
        }
      },
      qualitative: async (request: OdeAnalyzeInput) => {
        if (!activeId) throw new ApiError(400, "Open a conversation first.");
        const generation = viewGeneration.current;
        try {
          const message = await chatApi.odeQualitative(activeId, request);
          if (generation === viewGeneration.current && activeIdRef.current === activeId)
            setMessages((m) => [...m, message]);
          await refreshList();
        } catch (e) {
          if (e instanceof ApiError && e.status === 401) router.push("/login");
          throw e;
        }
      },
      numeric: async (request: OdeNumericInput) => {
        if (!activeId) throw new ApiError(400, "Open a conversation first.");
        const generation = viewGeneration.current;
        try {
          const message = await chatApi.odeSolveNumerical(activeId, request);
          if (generation === viewGeneration.current && activeIdRef.current === activeId)
            setMessages((m) => [...m, message]);
          await refreshList();
        } catch (e) {
          if (e instanceof ApiError && e.status === 401) router.push("/login");
          throw e;
        }
      },
    }),
    [activeId, refreshList, router],
  );

  /** The variation card's form: the user's own numbers, run without the model. */
  const runVariation = useCallback(
    async (request: VariationRequest) => {
      if (!activeId) throw new ApiError(400, "Open a conversation first.");
      const generation = viewGeneration.current;
      try {
        const message = await chatApi.variationDirectly(activeId, request);
        if (generation === viewGeneration.current && activeIdRef.current === activeId)
          setMessages((m) => [...m, message]);
        await refreshList();
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) router.push("/login");
        throw e;
      }
    },
    [activeId, refreshList, router],
  );

  /** The approximation card's form: the user's own numbers, run without the model. */
  const runApproximation = useCallback(
    async (request: ApproximationToolArgs) => {
      if (!activeId) throw new ApiError(400, "Open a conversation first.");
      const generation = viewGeneration.current;
      try {
        const message = await chatApi.approximationDirectly(activeId, request);
        if (generation === viewGeneration.current && activeIdRef.current === activeId)
          setMessages((m) => [...m, message]);
        await refreshList();
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) router.push("/login");
        throw e;
      }
    },
    [activeId, refreshList, router],
  );

  const busy = pending !== null;
  const lastUserIndex = messages.map((m) => m.role).lastIndexOf("user");

  /** Answer the last message again; the reply after it is replaced. */
  async function retryLast() {
    const text = messages[lastUserIndex]?.content.text;
    if (text) await send(text, "retry");
  }

  function startEdit(text: string) {
    setEditing(true);
    setReplaceRequest({ id: ++requestCounter.current, text });
  }

  function cancelEdit() {
    setEditing(false);
    setReplaceRequest({ id: ++requestCounter.current, text: "" });
  }

  async function submit(text: string): Promise<boolean> {
    const mode: SendMode = editing ? "edit" : "new";
    const ok = await send(text, mode);
    if (ok) setEditing(false);
    return ok;
  }
  const title = conversations.find((c) => c.id === activeId)?.title ?? "";

  if (status && !status.enabled) {
    return (
      <div className="mx-auto mt-24 max-w-lg text-center">
        <h1 className="font-serif text-3xl">Chat is not set up on this server</h1>
        <p className="mt-3 text-muted">
          The assistant needs a language model host. Everything else works without one: the
          workbench computes and verifies directly, and the handbook is browsable.
        </p>
        <div className="mt-6 flex justify-center gap-3">
          <Link
            href="/workbench"
            className="rounded bg-accent px-4 py-2 font-medium text-accent-ink"
          >
            Open the workbench
          </Link>
          <Link href="/reference" className="rounded border border-rule px-4 py-2 font-medium">
            Browse the handbook
          </Link>
        </div>
      </div>
    );
  }

  return (
    <InsertReferenceProvider
      value={(n) => setReferenceRequest({ id: ++requestCounter.current, text: `(${n})` })}
    >
      <MotionRunnerProvider
        value={{
          run: runMotion,
          perturb: runPerturbation,
          geometry: runGeometry,
          distribution: runDistribution,
          ode: runOde,
          variation: runVariation,
          approximation: runApproximation,
          statmech: runStatMech,
        }}
      >
        <FormulaActionsProvider value={formulaActions}>
          <div className="flex h-[calc(100dvh-3.5rem-1px)] gap-6">
            <aside className="hidden w-60 shrink-0 py-4 md:block">
              <ConversationList
                items={conversations}
                activeId={activeId}
                onOpen={open}
                onNew={startNew}
                onDelete={remove}
                onRename={rename}
              />
            </aside>

            <section className="flex min-w-0 flex-1 flex-col" aria-label="Conversation">
              <div className="flex items-center justify-between py-2 md:hidden">
                <button
                  type="button"
                  onClick={() => setListOpen((o) => !o)}
                  aria-expanded={listOpen}
                  className="rounded border border-rule px-2.5 py-1 text-sm"
                >
                  Conversations
                </button>
                <button type="button" onClick={startNew} className="text-sm text-accent">
                  New
                </button>
              </div>
              {listOpen && (
                <div className="mb-2 max-h-64 overflow-y-auto rounded-lg border border-rule bg-raised p-3 md:hidden">
                  <ConversationList
                    items={conversations}
                    activeId={activeId}
                    onOpen={open}
                    onNew={startNew}
                    onDelete={remove}
                    onRename={rename}
                  />
                </div>
              )}

              {status?.context_ok === false && (
                <p
                  role="alert"
                  className="mb-2 rounded bg-warn-soft px-3 py-2 text-sm text-warn"
                  data-testid="context-warning"
                >
                  The model&apos;s context window is smaller than Minerva&apos;s instructions, so it
                  will lose the conversation and most of its tools. The operator can raise it
                  (MINERVA_LLM_NUM_CTX) or send fewer tools (MINERVA_LLM_TOOLSET=core).
                </p>
              )}
              <div
                ref={logRef}
                onScroll={onLogScroll}
                className="relative flex-1 overflow-y-auto py-4 pr-1"
                role="log"
                aria-live="off"
              >
                {messages.length === 0 && !pending ? (
                  <EmptyState onPick={(t) => void send(t)} disabled={busy || !status} />
                ) : (
                  <div className="mx-auto max-w-3xl space-y-6">
                    {messages.map((m, i) => (
                      <MessageView
                        key={m.id}
                        message={m}
                        onRetry={
                          !busy &&
                          ((m.role === "assistant" && i === messages.length - 1) ||
                            (m.role === "user" && i === messages.length - 1))
                            ? () => void retryLast()
                            : undefined
                        }
                        retryLabel={retryLabel(m.content.stop_reason)}
                        onEdit={
                          !busy && m.role === "user" && i === lastUserIndex
                            ? () => startEdit(m.content.text)
                            : undefined
                        }
                        getReport={
                          debugOn && m.role === "assistant"
                            ? () => buildTurnReport({ title, status }, messages, i)
                            : undefined
                        }
                      />
                    ))}
                    {pending && <PendingView pending={pending} />}
                    {error && (
                      <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
                        {error}
                      </p>
                    )}
                  </div>
                )}
                <div ref={bottomRef} />
                {!atEnd && messages.length > 0 && (
                  <div className="pointer-events-none sticky bottom-2 flex justify-center">
                    <button
                      type="button"
                      onClick={jumpToEnd}
                      className="pointer-events-auto rounded-full border border-rule bg-raised px-3 py-1 text-xs shadow"
                    >
                      Jump to latest
                    </button>
                  </div>
                )}
                <p className="sr-only" role="status">
                  {pending
                    ? "Minerva is replying."
                    : messages.at(-1)?.role === "assistant"
                      ? "Reply ready."
                      : ""}
                </p>
              </div>

              <div className="mx-auto w-full max-w-3xl">
                {error && messages.length === 0 && (
                  <p role="alert" className="mb-2 rounded bg-bad-soft px-3 py-2 text-sm text-bad">
                    {error}
                  </p>
                )}
                {editing && (
                  <p className="mb-1 flex items-center gap-3 text-xs text-muted" role="status">
                    Editing your last message: sending replaces it and its reply.
                    <button
                      type="button"
                      onClick={cancelEdit}
                      className="text-accent hover:underline"
                    >
                      Cancel
                    </button>
                  </p>
                )}
                <Composer
                  key={`${accountId ?? "loading"}:${activeId ?? "new"}`}
                  draftKey={
                    accountId ? `minerva:draft:${accountId}:${activeId ?? "new"}` : undefined
                  }
                  disabled={!status}
                  imagesEnabled={status?.images === true}
                  beforeSend={confirmCloud}
                  toolbar={
                    status?.providers ? (
                      <ModelPicker
                        status={status}
                        open={modelOpen}
                        onToggle={() => setModelOpen((o) => !o)}
                      />
                    ) : null
                  }
                  panel={
                    modelOpen && status ? (
                      <ModelSettingsPanel status={status} onClose={() => setModelOpen(false)} />
                    ) : null
                  }
                  busy={busy}
                  insertRequest={insertRequest}
                  referenceRequest={referenceRequest}
                  replaceRequest={replaceRequest}
                  onSend={submit}
                  onStop={() => abortRef.current?.abort()}
                />
                <div className="flex items-center gap-4 py-1 text-xs text-muted">
                  <label className="flex items-center gap-1.5">
                    <input
                      type="checkbox"
                      checked={debugOn}
                      onChange={(e) => setDebugOn(e.target.checked)}
                    />
                    Debug mode
                  </label>
                  {debugOn && (
                    <CopyButton
                      label="Copy debug report"
                      getText={() =>
                        buildDebugReport({ title, messages, status, lastError, replying: busy })
                      }
                    />
                  )}
                </div>
                {debugOn && status?.params && status.params.editable.length > 0 && (
                  // Folded by default: on a phone the panel would otherwise fill the screen.
                  <details className="mt-1" data-testid="params-details">
                    <summary className="cursor-pointer text-xs text-muted">
                      Model parameters for the next turn
                      {Object.keys(paramOverrides).length > 0
                        ? ` (${Object.keys(paramOverrides).length} changed)`
                        : ""}
                    </summary>
                    <div className="mt-1">
                      <ParamsPanel
                        info={status.params}
                        value={paramOverrides}
                        onChange={setParamOverrides}
                      />
                    </div>
                  </details>
                )}
              </div>
            </section>
          </div>
        </FormulaActionsProvider>
      </MotionRunnerProvider>
    </InsertReferenceProvider>
  );
}

function appendText(p: Pending, delta: string): Pending {
  const last = p.items[p.items.length - 1];
  const items =
    last?.kind === "text"
      ? [...p.items.slice(0, -1), { kind: "text" as const, text: last.text + delta }]
      : [...p.items, { kind: "text" as const, text: delta }];
  return { ...p, items };
}

type SendMode = "new" | "retry" | "edit";

/** What had arrived when the user pressed Stop, kept on screen and marked as stopped. */
function stoppedMessage(p: Pending, index: number): StoredMessage {
  const blocks: Block[] = p.items.map((item) =>
    item.kind === "card"
      ? { kind: "card", segments: [], card: item.card }
      : { kind: "text", segments: splitMath(item.text), card: null },
  );
  const text = p.items.flatMap((i) => (i.kind === "text" ? [i.text] : [])).join("\n\n");
  return {
    id: `stopped-${index}`,
    role: "assistant",
    created_at: new Date().toISOString(),
    content: { text, blocks, truncated: true, stop_reason: "stopped" },
  };
}

function EmptyState({ onPick, disabled }: { onPick: (t: string) => void; disabled: boolean }) {
  return (
    <div className="mx-auto mt-10 max-w-2xl text-center sm:mt-20">
      <h1 className="font-serif text-4xl tracking-tight">What are we working on?</h1>
      <p className="mx-auto mt-3 max-w-xl text-muted">
        Describe a problem. Minerva works through it with verified computation and a handbook of
        known formulas, and says plainly when it cannot verify something.
      </p>
      <div className="mt-8 grid gap-3 text-left sm:grid-cols-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.title}
            type="button"
            disabled={disabled}
            onClick={() => onPick(ex.text)}
            className="rounded-xl border border-rule bg-raised p-4 text-left hover:border-accent disabled:opacity-50"
          >
            <div className="text-sm font-medium">{ex.title}</div>
            <div className="mt-1 text-sm text-muted">
              <RichText segments={splitMath(ex.text)} />
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
