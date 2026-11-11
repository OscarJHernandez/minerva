import { Math } from "@minerva/ui/math";

const APP_URL = process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000";
const REPO_URL = process.env.NEXT_PUBLIC_REPO_URL ?? "https://github.com/OscarJHernandez/minerva";

const EXAMPLES: {
  title: string;
  input: string;
  output: string;
  note: string;
  tone: "ok" | "warn";
  badge: string;
}[] = [
  {
    title: "Simplify, then verify",
    input: "\\sin^2 x + \\cos^2 x",
    output: "1",
    note: "SymPy simplifies; an independent numeric check at 12 random complex points confirms the result before it is shown.",
    tone: "ok",
    badge: "Verified",
  },
  {
    title: "Identities state their conditions",
    input: "\\log(xy)",
    output: "\\log x + \\log y",
    note: "Only offered once x and y are declared positive. Without that assumption the rule matches but is not applied, and Minerva says why.",
    tone: "warn",
    badge: "Needs x, y > 0",
  },
  {
    title: "Assumptions change the answer",
    input: "\\sqrt{x^2}",
    output: "|x|\\ \\ (x\\ \\text{real}),\\qquad x\\ \\ (x \\ge 0)",
    note: "For a general complex x the expression is returned unchanged, because “x” would be wrong.",
    tone: "ok",
    badge: "Assumption-aware",
  },
  {
    title: "Tensor symmetries",
    input: "S_{ab}A^{ab}",
    output: "0",
    note: "Symmetric contracted with antisymmetric, found by Butler–Portugal canonicalization. Only monoterm symmetries are applied; multi-term identities like Bianchi are reported as out of scope.",
    tone: "ok",
    badge: "SymPy tensor",
  },
];

const AVAILABLE = [
  [
    "Verified simplification",
    "SymPy results are checked symbolically and numerically; failures are reported, never guessed.",
  ],
  [
    "Identity search with applicability checks",
    "Retrieval proposes candidates; the engine matches, checks conditions and verifies each rewrite.",
  ],
  [
    "A citable handbook",
    "About a hundred entries across algebra, calculus, series, special functions, tensors, mechanics, electromagnetism, thermodynamics, quantum mechanics, relativity and constants.",
  ],
  [
    "Tensor canonicalization",
    "Index symmetries and contractions through the HTTP API (workspace interface planned).",
  ],
  [
    "An API that stands alone",
    "Everything the interface does is available over documented HTTP endpoints.",
  ],
];

const PLANNED = [
  ["Cadabra integration", "Advanced tensor algebra and quantum field theory."],
  ["SageManifolds integration", "Differential geometry on explicit manifolds."],
  ["Derivatives, integrals, limits", "In the workspace, with the same verification discipline."],
  ["Semantic and structural search", "Retrieval beyond keywords, behind the same interface."],
  [
    "Optional local language model",
    "To suggest and explain, never to compute or to decide correctness.",
  ],
];

function Section({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="border-t border-rule py-16 sm:py-20">
      <div className="mx-auto max-w-5xl px-5 sm:px-8">
        <h2 id={`${id}-h`} className="font-serif text-3xl tracking-tight sm:text-4xl">
          {title}
        </h2>
        <div className="mt-8">{children}</div>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <>
      <header className="mx-auto flex h-16 max-w-5xl items-center justify-between px-5 sm:px-8">
        <a href="#top" className="font-serif text-2xl italic tracking-tight">
          Minerva
        </a>
        <nav aria-label="Main" className="flex items-center gap-5 text-sm">
          <a href="#examples" className="hidden text-muted hover:text-ink sm:inline">
            Examples
          </a>
          <a href="#capabilities" className="hidden text-muted hover:text-ink sm:inline">
            Capabilities
          </a>
          <a href={REPO_URL} className="text-muted hover:text-ink">
            GitHub
          </a>
          <a href={`${APP_URL}/login`} className="text-muted hover:text-ink">
            Log in
          </a>
          <a
            href={APP_URL}
            className="rounded bg-accent px-3.5 py-1.5 font-medium text-accent-ink hover:opacity-90"
          >
            Launch app
          </a>
        </nav>
      </header>

      <main id="top">
        <div className="mx-auto max-w-5xl px-5 pb-20 pt-16 sm:px-8 sm:pb-28 sm:pt-24">
          <p className="text-sm uppercase tracking-[0.18em] text-muted">
            Open-source mathematical computing
          </p>
          <h1 className="mt-4 max-w-3xl font-serif text-5xl leading-[1.05] tracking-tight sm:text-7xl">
            A calculator that checks its own work.
          </h1>
          <p className="mt-6 max-w-2xl font-serif text-xl leading-relaxed text-muted sm:text-2xl">
            Minerva combines computer algebra with a handbook of mathematical tables and physical
            equations. It simplifies expressions, finds the identities that apply to them, and tells
            you exactly what each one assumes.
          </p>
          <div className="mt-10 flex flex-wrap gap-3">
            <a
              href={APP_URL}
              className="rounded bg-accent px-5 py-2.5 font-medium text-accent-ink hover:opacity-90"
            >
              Launch the workspace
            </a>
            <a
              href={REPO_URL}
              className="rounded border border-rule px-5 py-2.5 font-medium hover:bg-sunk"
            >
              View source on GitHub
            </a>
          </div>
          <div className="mt-16 rounded-lg border border-rule bg-raised px-6 py-8 text-center text-3xl sm:text-4xl">
            <Math
              latex="\sin^2 x + \cos^2 x \;=\; 1"
              display
              label="sin squared plus cos squared equals one"
            />
          </div>
        </div>

        <Section id="examples" title="What it does, and what it refuses to do">
          <div className="grid gap-5 md:grid-cols-2">
            {EXAMPLES.map((ex) => (
              <article key={ex.title} className="rounded-lg border border-rule bg-raised p-6">
                <div className="flex items-baseline justify-between gap-3">
                  <h3 className="font-serif text-xl">{ex.title}</h3>
                  <span
                    className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                      ex.tone === "ok" ? "bg-ok-soft text-ok" : "bg-warn-soft text-warn"
                    }`}
                  >
                    {ex.badge}
                  </span>
                </div>
                <div className="mt-4 grid grid-cols-[auto_1fr] items-center gap-x-4 text-xl">
                  <Math latex={ex.input} />
                  <Math latex={String.raw`\;\longrightarrow\; ${ex.output}`} />
                </div>
                <p className="mt-4 text-sm text-muted">{ex.note}</p>
              </article>
            ))}
          </div>
        </Section>

        <Section id="principles" title="Correctness first">
          <div className="grid gap-8 md:grid-cols-3">
            {[
              [
                "Computation is symbolic",
                "SymPy does the algebra. Minerva wraps it with strict parsers (no eval), isolated processes and hard time and memory limits.",
              ],
              [
                "Knowledge is checked before it is used",
                "Every handbook entry is verified at a declared level in the test suite. Entries recorded from the literature are labelled “stated” and are never applied.",
              ],
              [
                "Language models never decide",
                "An optional local model may suggest or explain. Whether an identity applies is decided by the engine, from its conditions.",
              ],
            ].map(([t, d]) => (
              <div key={t}>
                <h3 className="font-serif text-xl">{t}</h3>
                <p className="mt-2 text-muted">{d}</p>
              </div>
            ))}
          </div>
        </Section>

        <Section id="capabilities" title="Capabilities">
          <div className="grid gap-12 md:grid-cols-2">
            <div>
              <h3 className="text-sm font-medium uppercase tracking-wide text-ok">Available now</h3>
              <dl className="mt-4 space-y-5">
                {AVAILABLE.map(([t, d]) => (
                  <div key={t}>
                    <dt className="font-medium">{t}</dt>
                    <dd className="text-sm text-muted">{d}</dd>
                  </div>
                ))}
              </dl>
            </div>
            <div>
              <h3 className="text-sm font-medium uppercase tracking-wide text-muted">Planned</h3>
              <dl className="mt-4 space-y-5">
                {PLANNED.map(([t, d]) => (
                  <div key={t}>
                    <dt className="font-medium">{t}</dt>
                    <dd className="text-sm text-muted">{d}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
          <p className="mt-10 max-w-2xl text-sm text-muted">
            Domains on the roadmap: differential geometry, tensor calculus, quantum field theory and
            general symbolic mathematics. Minerva is early; this page lists what exists today and
            keeps the rest labelled as planned.
          </p>
        </Section>

        <Section id="open-source" title="Open source">
          <p className="max-w-2xl text-lg text-muted">
            Minerva is developed in the open under the Apache License 2.0. Contributions to the
            handbook are plain YAML files with a verification test, so adding a formula does not
            require writing code.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <a
              href={REPO_URL}
              className="rounded bg-accent px-5 py-2.5 font-medium text-accent-ink hover:opacity-90"
            >
              Repository
            </a>
            <a
              href={`${APP_URL}/register`}
              className="rounded border border-rule px-5 py-2.5 font-medium hover:bg-sunk"
            >
              Create an account
            </a>
          </div>
        </Section>
      </main>

      <footer className="border-t border-rule py-10 text-sm text-muted">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-5 sm:px-8">
          <span>
            <span className="font-serif italic">Minerva</span> · Apache-2.0
          </span>
          <span>This site stores nothing about you: no cookies, no analytics.</span>
        </div>
      </footer>
    </>
  );
}
