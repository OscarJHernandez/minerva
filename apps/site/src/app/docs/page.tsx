import type { Metadata } from "next";
import { ExampleView } from "@/components/example-view";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import { CAPABILITIES, HANDBOOK_ENTRIES } from "@/lib/capabilities";
import { SELF_HOSTING_URL, repoFile } from "@/lib/links";

export const metadata: Metadata = {
  title: "Docs: what Minerva can do today — Minerva",
  description:
    "Every capability of the open-source Minerva workbench with examples run through the real engine: inputs, results, verification, and what it declines to do.",
};

export default function Docs() {
  const available = CAPABILITIES.filter((c) => c.status === "available");
  const planned = CAPABILITIES.filter((c) => c.status === "planned");
  return (
    <>
      <SiteHeader />
      <main className="mx-auto max-w-5xl px-5 pb-24 pt-12 sm:px-8 sm:pt-16">
        <p className="text-sm uppercase tracking-[0.18em] text-muted">Docs</p>
        <h1 className="mt-4 max-w-3xl font-serif text-4xl leading-tight tracking-tight sm:text-5xl">
          What Minerva can do today
        </h1>
        <p className="mt-5 max-w-2xl font-serif text-lg leading-relaxed text-muted sm:text-xl">
          Each example below was run through the real engine and handbook when this page was built;
          nothing is typed by hand. Statuses are explicit, and what Minerva declines to do is shown
          next to what it does. The handbook holds {HANDBOOK_ENTRIES} entries.
        </p>
        <p className="mt-4 max-w-2xl text-sm text-muted">
          The same record is kept in the repository as{" "}
          <a className="underline hover:text-ink" href={repoFile("docs/capabilities.md")}>
            docs/capabilities.md
          </a>
          , and CI fails if it drifts from the code. To run your own instance and connect your own
          model, see the{" "}
          <a className="underline hover:text-ink" href={SELF_HOSTING_URL}>
            self-hosting guide
          </a>
          .
        </p>

        <nav aria-label="Capabilities" className="mt-10 flex flex-wrap gap-x-5 gap-y-2 text-sm">
          {available.map((c) => (
            <a key={c.id} href={`#${c.id}`} className="text-muted hover:text-ink">
              {c.title}
            </a>
          ))}
          <a href="#planned" className="text-muted hover:text-ink">
            Planned
          </a>
        </nav>

        {available.map((cap) => (
          <section
            key={cap.id}
            id={cap.id}
            aria-labelledby={`${cap.id}-h`}
            className="mt-14 scroll-mt-6 border-t border-rule pt-10"
          >
            <h2 id={`${cap.id}-h`} className="font-serif text-2xl tracking-tight sm:text-3xl">
              {cap.title}
            </h2>
            <p className="mt-2 text-sm text-muted">
              Available in: {cap.where.join(", ")}.
              {cap.needs_model && " Needs a model you host and connect."}
            </p>
            <p className="mt-4 max-w-3xl text-muted">{cap.summary}</p>
            {cap.limits.length > 0 && (
              <div className="mt-4">
                <h3 className="text-sm font-medium">Limits</h3>
                <ul className="mt-1 max-w-3xl list-disc space-y-1 pl-5 text-sm text-muted">
                  {cap.limits.map((l) => (
                    <li key={l}>{l.replaceAll("`", "")}</li>
                  ))}
                </ul>
              </div>
            )}
            {cap.examples.length > 0 && (
              <div className="mt-6 grid gap-5 md:grid-cols-2">
                {cap.examples.map((ex) => (
                  <ExampleView key={ex.id} ex={ex} />
                ))}
              </div>
            )}
          </section>
        ))}

        <section
          id="planned"
          aria-labelledby="planned-h"
          className="mt-14 scroll-mt-6 border-t border-rule pt-10"
        >
          <h2 id="planned-h" className="font-serif text-2xl tracking-tight sm:text-3xl">
            Planned
          </h2>
          <p className="mt-2 text-sm text-muted">
            Not implemented. Listed so that nobody mistakes them for features.
          </p>
          <dl className="mt-5 grid gap-5 md:grid-cols-2">
            {planned.map((c) => (
              <div key={c.id}>
                <dt className="font-medium">{c.title}</dt>
                <dd className="text-sm text-muted">{c.summary}</dd>
              </div>
            ))}
          </dl>
        </section>

        <section className="mt-14 border-t border-rule pt-10">
          <h2 className="font-serif text-2xl tracking-tight sm:text-3xl">More in the repository</h2>
          <ul className="mt-4 space-y-2 text-muted">
            <li>
              <a className="underline hover:text-ink" href={repoFile("docs/model-evaluation.md")}>
                Model evaluation
              </a>
              : how candidate local models score on tool calling and honesty.
            </li>
            <li>
              <a className="underline hover:text-ink" href={repoFile("docs/architecture.md")}>
                Architecture
              </a>
              : how the engine, handbook, API and agent fit together.
            </li>
            <li>
              <a className="underline hover:text-ink" href={repoFile("docs/handbook-benchmark.md")}>
                Handbook benchmark
              </a>
              : identity retrieval quality on hand-written cases.
            </li>
            <li>
              <a className="underline hover:text-ink" href={repoFile("docs/roadmap.md")}>
                Roadmap
              </a>
              : what comes next.
            </li>
          </ul>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
