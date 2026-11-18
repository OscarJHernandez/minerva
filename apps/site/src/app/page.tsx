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
  const keplerSearch = example("invariants-kepler-polar");
  const schwarzschild = example("geometry-schwarzschild");
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

        <Section id="examples" title="Find what the obvious symmetries miss">
          <p className="max-w-3xl text-lg text-muted">
            Give Minerva the Kepler Lagrangian. Beyond the familiar energy and angular momentum,
            its symbolic search discovers two more conserved expressions: the components of
            the Laplace–Runge–Lenz vector.
          </p>
          <figure className="mt-8 overflow-hidden rounded-lg border border-rule bg-raised">
            <div className="border-b border-rule p-5 sm:p-8">
              <figcaption className="text-xs font-medium uppercase tracking-[0.14em] text-muted">
                The problem
              </figcaption>
              <p className="mt-2 text-sm text-muted">
                Search for conserved quantities in the Kepler system.
              </p>
              <div className="mt-4 overflow-x-auto py-2 text-xl sm:text-2xl">
                <Math
                  latex={`L = ${inputLatex(keplerSearch)}`}
                  label="Kepler Lagrangian in polar coordinates"
                />
              </div>
              <p className="mt-3 text-sm text-muted">
                Coordinates: r, θ · Angular coordinate: θ
              </p>
            </div>

            <div className="p-5 sm:p-8">
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <h3 className="font-serif text-2xl">Two hidden constants of motion</h3>
                <StatusBadge status="ok" label="Engine-verified" />
              </div>
              <p className="mt-2 max-w-2xl text-sm text-muted">
                Discovered by searching candidate expressions, not by looking up the
                Runge–Lenz vector.
              </p>
              <div className="mt-6 space-y-5">
                <div className="rounded border border-rule bg-paper p-4">
                  <p className="text-xs uppercase tracking-wide text-muted">First component</p>
                  <div className="mt-2 overflow-x-auto py-2 text-lg sm:text-xl">
                    <Math latex={outputLatex(keplerSearch, "Constant of motion 1")} />
                  </div>
                </div>
                <div className="rounded border border-rule bg-paper p-4">
                  <p className="text-xs uppercase tracking-wide text-muted">Second component</p>
                  <div className="mt-2 overflow-x-auto py-2 text-lg sm:text-xl">
                    <Math latex={outputLatex(keplerSearch, "Constant of motion 2")} />
                  </div>
                </div>
              </div>
              <p className="mt-5 text-sm font-medium text-ink">
                Verified: each time derivative is symbolically zero on the equations
                of motion, with 20/20 numerical checks.
              </p>
              <p className="mt-2 max-w-2xl text-sm text-muted">
                Energy and angular momentum are excluded as already known.
                The search covers a finite family of candidate expressions;
                it does not establish that no others exist.
              </p>
              <div className="mt-6 flex flex-wrap items-center gap-4">
                <a href={APP_URL} className={primary}>
                  Explore in Minerva
                </a>
                <Link href="/docs" className="text-sm underline hover:text-ink">
                  See the verified computation
                </Link>
              </div>
            </div>
          </figure>
          <figure className="mt-8 rounded-lg border border-rule bg-raised p-5 sm:p-8">
            <figcaption className="text-xs font-medium uppercase tracking-[0.14em] text-muted">
              Another discovery · General relativity
            </figcaption>
            <h3 className="mt-3 font-serif text-2xl sm:text-3xl">
              How can empty spacetime still be curved?
            </h3>
            <p className="mt-2 max-w-2xl text-muted">
              Give Minerva the Schwarzschild metric and ask it to compute the curvature.
              The Ricci tensor vanishes, but a different curvature invariant does not.
            </p>
            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <div className="min-w-0 rounded border border-rule bg-paper p-4">
                <p className="text-xs uppercase tracking-wide text-muted">Ricci tensor</p>
                <div className="mt-3 overflow-x-auto py-3 text-xl sm:text-2xl">
                  <Math
                    latex={String.raw`R_{\\mu\\nu}=0`}
                    label="Every component of the Ricci tensor is zero"
                  />
                </div>
                <p className="mt-2 text-sm text-muted">
                  All components vanish. Symbolically checked.
                </p>
              </div>
              <div className="min-w-0 rounded border border-rule bg-paper p-4">
                <p className="text-xs uppercase tracking-wide text-muted">
                  Kretschmann scalar
                </p>
                <div className="mt-3 overflow-x-auto py-3 text-xl sm:text-2xl">
                  <Math
                    latex={String.raw`R_{\\mu\\nu\\rho\\sigma}R^{\\mu\\nu\\rho\\sigma}=\\frac{48M^2}{r^6}`}
                    label="The Kretschmann scalar equals 48 M squared divided by r to the sixth power"
                  />
                </div>
                <p className="mt-2 text-sm text-muted">
                  Curvature survives. Numerically checked.
                </p>
              </div>
            </div>
            <p className="mt-5 max-w-3xl text-sm text-muted">
              Both results come from the engine&apos;s Schwarzschild metric example.
              The Ricci result has a symbolic check; the full curvature contraction
              has independent numerical checks, not a symbolic proof. Validity
              excludes coordinate singularities.
            </p>
            <div className="mt-4">
              <Link href="/docs" className="text-sm underline hover:text-ink">
                Inspect the curvature calculation and assumptions
              </Link>
            </div>
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
