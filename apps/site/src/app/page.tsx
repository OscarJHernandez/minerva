import Link from "next/link";
import { Math } from "@minerva/ui/math";
import { MathLine, StatusBadge } from "@/components/example-view";
import { Section } from "@/components/section";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import { example, inputLatex, inputSource, outputLatex, type Example } from "@/lib/capabilities";
import {
  APP_URL,
  CONTRIBUTING_URL,
  ISSUES_URL,
  OPENROUTER_URL,
  REPO_URL,
  SELF_HOSTING_URL,
} from "@/lib/links";

const primary = "rounded bg-accent px-5 py-2.5 font-medium text-accent-ink hover:opacity-90";
const secondary = "rounded border border-rule px-5 py-2.5 font-medium hover:bg-sunk";

const BENEFITS = [
  [
    "You lead",
    "Pick the problem, assumptions, and next step. Minerva suggests methods; you decide what to explore.",
  ],
  [
    "Check the mathematics",
    "A symbolic engine computes and checks results. Unverified claims stay labelled as such.",
  ],
  [
    "Run it yourself",
    "Host it with Docker or Kubernetes, bring a local model, or use the tools without AI.",
  ],
  [
    "Make it yours",
    "Apache-2.0 open source. Read it, modify it, report bugs, or contribute what is missing.",
  ],
];

type Row = { label: string; node: React.ReactNode };

function Card({
  title,
  badge,
  tone,
  rows,
  note,
}: {
  title: string;
  badge: string;
  tone: string;
  rows: Row[];
  note: string;
}) {
  return (
    <article className="min-w-0 rounded-lg border border-rule bg-raised p-5 sm:p-6">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="font-serif text-xl">{title}</h3>
        <StatusBadge status={tone} label={badge} />
      </div>
      <dl className="mt-4 space-y-3">
        {rows.map((r) => (
          <div key={r.label}>
            <dt className="text-xs uppercase tracking-wide text-muted">{r.label}</dt>
            <dd className="mt-0.5">{r.node}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-4 text-sm text-muted">{note}</p>
    </article>
  );
}

const tex = (latex: string) => <MathLine latex={latex} />;
const typed = (ex: Example) => {
  const line = ex.input.find((l) => l.source);
  if (line?.syntax !== "text") return tex(inputLatex(ex));
  return (
    <div>
      <code className="block overflow-x-auto whitespace-pre text-sm">{inputSource(ex)}</code>
      {tex(inputLatex(ex))}
    </div>
  );
};
const plain = (text: string) => <span className="text-sm text-muted">{text}</span>;

export default function Home() {
  const kepler = example("lagrangian-kepler-polar");
  const trig = example("simplify-trig-quotients");
  const logBlocked = example("identity-log-blocked");
  const logOk = example("identity-log-applicable");

  return (
    <>
      <SiteHeader />
      <main id="top">
        <div className="mx-auto max-w-5xl px-5 pb-16 pt-14 sm:px-8 sm:pb-20 sm:pt-20">
          <p className="text-sm uppercase tracking-[0.18em] text-muted">
            Open source · Self-hostable · Your exploration
          </p>
          <h1 className="mt-4 max-w-3xl font-serif text-4xl leading-[1.05] tracking-tight sm:text-6xl">
            Explore mathematics. Own the process.
          </h1>
          <p className="mt-6 max-w-2xl font-serif text-xl leading-relaxed text-muted sm:text-2xl">
            Minerva is an open-source mathematical assistant you can run yourself to explore
            ideas, try different approaches, and verify results on your terms.
          </p>
          <p className="mt-4 max-w-2xl text-base text-muted">
            You choose what to investigate next. Minerva suggests tools and checks the math,
            without taking the discovery away from you.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a href={APP_URL} className={primary}>
              Try Minerva
            </a>
            <a href={SELF_HOSTING_URL} className={secondary}>
              Run it yourself
            </a>
          </div>
          <p className="mt-4 text-sm text-muted">
            <a href={REPO_URL} className="underline hover:text-ink">
              View the source on GitHub
            </a>
            {" · "}
            <Link href="/docs" className="underline hover:text-ink">
              Documentation
            </Link>
          </p>
        </div>

        <Section id="why" title="Why Minerva">
          <div className="grid gap-x-12 gap-y-8 md:grid-cols-2">
            {BENEFITS.map(([title, description]) => (
              <div key={title}>
                <h3 className="font-serif text-xl">{title}</h3>
                <p className="mt-1 max-w-md text-muted">{description}</p>
              </div>
            ))}
          </div>
        </Section>

        <Section id="examples" title="Explore a problem. Check the result.">
          <p className="max-w-2xl text-muted">
            Start with a question. Minerva applies mathematical tools and shows what they
            establish, including their limitations.
          </p>
          <figure className="mt-10 rounded-lg border border-rule bg-raised px-5 py-7 sm:px-8 sm:py-9">
            <figcaption className="text-xs uppercase tracking-wide text-muted">
              A real example: conserved quantities in the Kepler problem
            </figcaption>
            <p className="mt-3 text-xs uppercase tracking-wide text-muted">You type</p>
            <code className="mt-1 block overflow-x-auto whitespace-pre text-sm sm:text-base">
              {inputSource(kepler)}
            </code>
            <div className="mt-3 space-y-1 text-xl sm:text-2xl">
              <div className="overflow-x-auto overflow-y-hidden">
                <Math
                  latex={`L = ${inputLatex(kepler)}`}
                  label="Lagrangian of the Kepler problem in polar coordinates"
                />
              </div>
              <div className="overflow-x-auto overflow-y-hidden">
                <Math latex={outputLatex(kepler, "Conserved: cyclic")} />
              </div>
              <div className="overflow-x-auto overflow-y-hidden">
                <Math latex={outputLatex(kepler, "Conserved: energy")} />
              </div>
            </div>
            <p className="mt-5 max-w-2xl text-sm text-muted">
              Minerva checks these conserved quantities against the equations of motion.
              It also tells you what it has not checked, rather than inventing a proof.
            </p>
          </figure>
          <div className="mt-6 grid gap-5 md:grid-cols-2">
            <Card
              title="Simplify, then verify"
              badge="Verified"
              tone="ok"
              rows={[
                { label: "You type", node: typed(trig) },
                { label: "Result", node: tex(outputLatex(trig, "Result")) },
              ]}
              note="SymPy simplifies; an independent numeric check at random complex points and a symbolic zero test confirm the result before it is shown."
            />
            <Card
              title="Identities state their conditions"
              badge="Needs x, y > 0"
              tone="warn"
              rows={[
                { label: "You type", node: typed(logBlocked) },
                { label: "No assumptions", node: plain(logBlocked.outputs[0]?.text ?? "") },
                { label: "With x, y positive", node: tex(outputLatex(logOk, "Logarithm")) },
              ]}
              note="The handbook identity is found either way. It is applied only once its conditions are provable from the assumptions you declare."
            />
          </div>
          <p className="mt-6 text-sm text-muted">
            <Link href="/docs" className="underline hover:text-ink">
              See the full capability list and verified examples
            </Link>
            .
          </p>
        </Section>

        <Section id="get-started" title="Your tools. Your choice.">
          <div className="grid gap-8 md:grid-cols-2">
            <div>
              <h3 className="font-serif text-xl">Run on your infrastructure</h3>
              <p className="mt-2 text-muted">
                Self-host the workbench, data, and optional AI model. Docker and Kubernetes
                deployments are documented. You can also use OpenRouter or work without AI.
              </p>
              <a href={SELF_HOSTING_URL} className="mt-4 inline-block underline hover:text-ink">
                Self-hosting guide
              </a>
            </div>
            <div>
              <h3 className="font-serif text-xl">Open source means you can change it</h3>
              <p className="mt-2 text-muted">
                Inspect the code, extend the math engine, add a verified identity, or submit
                an issue or pull request. Minerva is licensed under Apache-2.0.
              </p>
              <p className="mt-4 text-sm">
                <a href={REPO_URL} className="underline hover:text-ink">
                  GitHub
                </a>
                {" · "}
                <a href={ISSUES_URL} className="underline hover:text-ink">
                  Issues and ideas
                </a>
                {" · "}
                <a href={CONTRIBUTING_URL} className="underline hover:text-ink">
                  Contribute
                </a>
              </p>
            </div>
          </div>
          <p className="mt-8 max-w-3xl text-sm text-muted">
            Your own deployment keeps data in infrastructure you control when connected to
            a locally hosted model. External providers receive the conversations and tool
            results sent to them. The hosted demo also processes input on its operator&apos;s
            server.{" "}
            <a href={OPENROUTER_URL} className="underline hover:text-ink">
              Model choices and privacy
            </a>
            .
          </p>
        </Section>
      </main>
      <SiteFooter />
    </>
  );
}
