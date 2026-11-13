"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "@/lib/api";
import { chatApi, sendMessage } from "@/lib/chat-api";
import { buildDebugReport, buildTurnReport, type LastError } from "@/lib/debug-report";
import { useDebugMode } from "@/lib/debug-mode";
import { splitMath } from "@/lib/inline-math";
import type { GeometryRequest } from "@/lib/types";
import type {
  AssistantContent,
  Card,
  ChatStatus,
  ConversationSummary,
  MotionRequest,
  PerturbationRequest,
  StoredMessage,
  StreamEvent,
  ToolRun,
} from "@/lib/chat-types";
import { CardView } from "./card-view";
import { Composer } from "./composer";
import { CopyButton } from "./debug-panel";
import { ConversationList } from "./conversation-list";
import { MessageView } from "./message-view";
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

const TOOL_LABEL: Record<string, string> = {
  simplify: "Simplifying and verifying",
  find_identities: "Checking handbook identities",
  search_handbook: "Searching the handbook",
  get_entry: "Reading a handbook entry",
  tensor_canonicalize: "Canonicalizing the tensor expression",
  matrix_calculus: "Running the matrix calculus and verifying the result",
  geometry: "Computing the geometry and checking it independently",
  solve_polynomial: "Solving and verifying the roots",
  analyze_lagrangian: "Deriving equations of motion and checking symmetries",
  integrate_motion: "Integrating the equations of motion and verifying the run",
  expand_motion: "Expanding the equations of motion about the equilibrium",
  perturbation_solve: "Solving by perturbation theory and checking the residual",
};

type PendingItem = { kind: "text"; text: string } | { kind: "card"; card: Card };

interface Pending {
  items: PendingItem[];
  tools: ToolRun[];
  cards: number;
}

export function ChatApp() {
  const router = useRouter();
  const [status, setStatus] = useState<ChatStatus | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<StoredMessage[]>([]);
  const [pending, setPending] = useState<Pending | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [listOpen, setListOpen] = useState(false);
  const [debugOn, setDebugOn] = useDebugMode();
  const [lastError, setLastError] = useState<LastError | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const fail = useCallback(
    (e: unknown) => {
      if (e instanceof ApiError && e.status === 401) return router.push("/login");
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
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial data load
    void refreshList();
  }, [fail, refreshList]);

  useEffect(() => {
    if (messages.length > 0 || pending) bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages, pending]);

  async function open(id: string) {
    setError(null);
    setLastError(null);
    setListOpen(false);
    try {
      const detail = await chatApi.get(id);
      setActiveId(id);
      setMessages(detail.messages);
      setPending(null);
    } catch (e) {
      fail(e);
    }
  }

  function startNew() {
    abortRef.current?.abort();
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
      if (id === activeId) startNew();
      await refreshList();
    } catch (e) {
      fail(e);
    }
  }

  async function send(text: string) {
    setError(null);
    setLastError(null);
    try {
      let id = activeId;
      if (!id) {
        const created = await chatApi.create();
        id = created.id;
        setActiveId(id);
      }
      setMessages((m) => [
        ...m,
        {
          id: `local-${m.length}`,
          role: "user",
          content: { text },
          created_at: new Date().toISOString(),
        },
      ]);
      setPending({ items: [], tools: [], cards: 0 });
      const controller = new AbortController();
      abortRef.current = controller;
      await sendMessage(
        id,
        text,
        (event: StreamEvent) => {
          if (event.event === "text") {
            setPending((p) => p && appendText(p, event.data.delta));
          } else if (event.event === "tool_call") {
            setPending((p) => p && { ...p, tools: [...p.tools, event.data] });
          } else if (event.event === "card") {
            setPending(
              (p) =>
                p && {
                  ...p,
                  items: [...p.items, { kind: "card", card: event.data }],
                  cards: p.cards + 1,
                },
            );
          } else if (event.event === "message") {
            const content: AssistantContent = event.data;
            setMessages((m) => [
              ...m,
              {
                id: `reply-${m.length}`,
                role: "assistant",
                content,
                created_at: new Date().toISOString(),
              },
            ]);
            setPending(null);
          } else if (event.event === "error") {
            setError(event.data.message);
            setLastError(event.data);
            setPending(null);
          }
        },
        controller.signal,
      );
      await refreshList();
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      fail(e);
    } finally {
      abortRef.current = null;
      setPending(null);
    }
  }

  /** The Lagrangian card's form: runs the numerical solution with the user's own numbers. */
  const runMotion = useCallback(
    async (request: MotionRequest) => {
      if (!activeId) throw new ApiError(400, "Open a conversation first.");
      try {
        const message = await chatApi.integrateMotion(activeId, request);
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
      try {
        const message = await chatApi.perturbationSolve(activeId, request);
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
      try {
        const message = await chatApi.geometry(activeId, request);
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
    <MotionRunnerProvider
      value={{ run: runMotion, perturb: runPerturbation, geometry: runGeometry }}
    >
      <div className="flex h-[calc(100dvh-3.5rem-1px)] gap-6">
        <aside className="hidden w-60 shrink-0 py-4 md:block">
          <ConversationList
            items={conversations}
            activeId={activeId}
            onOpen={open}
            onNew={startNew}
            onDelete={remove}
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
              />
            </div>
          )}

          <div className="flex-1 overflow-y-auto py-4 pr-1" role="log" aria-live="polite">
            {messages.length === 0 && !pending ? (
              <EmptyState onPick={send} disabled={busy || !status} />
            ) : (
              <div className="mx-auto max-w-3xl space-y-6">
                {messages.map((m, i) => (
                  <MessageView
                    key={m.id}
                    message={m}
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
          </div>

          <div className="mx-auto w-full max-w-3xl">
            {error && messages.length === 0 && (
              <p role="alert" className="mb-2 rounded bg-bad-soft px-3 py-2 text-sm text-bad">
                {error}
              </p>
            )}
            <Composer
              disabled={!status}
              imagesEnabled={status?.images === true}
              busy={busy}
              onSend={send}
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
                  getText={() => buildDebugReport({ title, messages, status, lastError })}
                />
              )}
            </div>
          </div>
        </section>
      </div>
    </MotionRunnerProvider>
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

function PendingView({ pending }: { pending: Pending }) {
  const running = pending.tools.length > pending.cards ? pending.tools[pending.cards] : null;
  const hasText = pending.items.some((i) => i.kind === "text");
  return (
    <div className="space-y-3" aria-busy="true">
      {pending.items.map((item, i) =>
        item.kind === "text" ? (
          // Closed formulas render as they arrive; an unfinished one stays text until it closes.
          <div key={i} className="leading-relaxed">
            <RichText segments={splitMath(item.text)} />
          </div>
        ) : (
          <CardView key={item.card.id} card={item.card} />
        ),
      )}
      {(running || !hasText) && (
        <p className="text-sm text-muted">
          {running ? `${TOOL_LABEL[running.name] ?? "Working"}…` : "Thinking…"}
        </p>
      )}
    </div>
  );
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
